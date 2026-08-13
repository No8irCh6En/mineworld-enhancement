"""
    Wrap the Huggingface Transformers Llama to PyTorch Lightning Module.
"""
import os
import sys
import inspect 
from tomlkit import key
import torch
import math
import random
from typing import Optional, List, Tuple
from xformers.ops import memory_efficient_attention
import time
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import LlamaConfig
from typing import Optional, List, Dict, Any
from util.helper import get_obj_from_str, instantiate_from_config
from train import IMAGE_TOKEN_LENGTH, ACTION_TOKEN_LENGTH
from diagonal_decoding import decode_one_token, decode_some_token, decode_n_tokens, decode_n_tokens_for_gradio, prefill, img_diagd_decode_n_tokens, sample_n_top_k, sample_n_top_p, video_diagd_decode_n_tokens, img_diagd_decode_n_token_for_gradio, speculative_decoding_step, speculative_img_diagd_decode_n_tokens
from torch.nn.attention import SDPBackend
from speculative_wrapper import get_inference_functions

torch.backends.cuda.matmul.allow_tf32 = False

logger = logging.get_logger(__name__)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

if not (parentdir in sys.path):
    sys.path.insert(0, parentdir) 

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine values.
        sin (torch.Tensor): Sine values.
        position_ids (torch.Tensor): Position IDs.

    Returns:
        torch.Tensor: Query and key tensors with rotary position embeddings applied.
    """
    # print(f"[DEBUG]cos[position_ids]: {cos[position_ids].shape}, sin[position_ids]: {sin[position_ids].shape}")
    cos = cos[position_ids].unsqueeze(0).unsqueeze(2)
    sin = sin[position_ids].unsqueeze(0).unsqueeze(2)
    if cos.dim() == 5:
        cos = cos.squeeze(0).transpose(1,2)
        sin = sin.squeeze(0).transpose(1,2)
    # print(f"[DEBUG] q shape: {q.shape}, k shape: {k.shape}, cos shape: {cos.shape}, sin shape: {sin.shape}")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def weighted_autoregressive_loss(logits, labels, ignore_index=-100, segment_len=336):
    """
    logits: [B, T, V]
    labels: [B, T]
    """
    B, T, V = logits.size()
    loss_total = 0
    segment_count = 0

    # Compute cross entropy per token (before masking)
    ce_loss = F.cross_entropy(
        logits.view(-1, V),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction='none'
    ).view(B, T)

    for b in range(B):
        in_segment = False
        start_idx = None

        for t in range(T + 1):  # include T to handle last segment
            if t < T and labels[b, t] != ignore_index:
                if not in_segment:
                    in_segment = True
                    start_idx = t
            else:
                if in_segment:
                    end_idx = t  # exclusive
                    in_segment = False
                    seg_len = end_idx - start_idx
                    assert seg_len == segment_len, f"Segment length mismatch: {seg_len} != {segment_len}"

                    # Extract losses and apply weights
                    segment_loss = ce_loss[b, start_idx:end_idx]  # [336]
                    weights = torch.tensor([segment_len / (i + 1) for i in range(segment_len)], device=logits.device)  # [336]
                    weighted = segment_loss * weights
                    loss_total += weighted.sum()
                    segment_count += 1

    if segment_count == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return loss_total / segment_count


def transform_mask_to_weight(x: torch.Tensor, max_val: float = 2.0, exp: float = 1.0) -> torch.Tensor:
    """
    检查 x ∈ [0, 1]，并按公式 w = 1 + (max_val - 1) * (1 - x**exp) 计算权重。

    Args:
        x: 任意形状的 tensor（通常为 [B, T]）。
        max_val: “最大权重”，当 x=0 时 w=max_val；当 x=1 时 w=1。
        exp: 指数 e，控制曲线形状。

    Returns:
        与 x 同形状/设备/浮点 dtype 的权重 tensor。
    """
    if not torch.all((x >= 0) & (x <= 1)).item():
        raise ValueError("transform_mask_to_weight: 输入张量包含不在 [0, 1] 的值。")

    x = x.to(dtype=torch.float32)  # 计算用 float（也可用 x.dtype 但一般用 float 更安全）
    w = 1.0 + (max_val - 1.0) * (1.0 - torch.pow(x, exp))
    return w


def transform_nov_mask_to_weight(x: torch.Tensor, max_val: float = 2.0) -> torch.Tensor:
    if not torch.all((x >= 0) & (x <= 1)).item():
        raise ValueError("transform_nov_mask_to_weight: 输入张量包含不在 [0, 1] 的值。")

    x = x.to(dtype=torch.float32)
    w = 1.0 + (max_val - 1.0) * x
    return w


class LlamaLVM(torch.nn.Module):
    def __init__(
        self,
        transformer_config,
        model_class: str,
        tokenizer_config = None,
        
        # ===== 新增：各项 loss 的权重 =====
        ce_loss_weight: float = 1.0,
        # [新增] 推测去噪 Loss 的权重 (建议 0.1 ~ 0.5)
        spec_loss_weight: float = 0.0, 
        # [新增] 替换概率 (多少比例的 token 被替换成上一帧的)
        spec_swap_prob: float = 0.3,
        
        gate_loss_weight: float = 10.0,
        attn_kl_weight: float = 1.0,         # KL(M || A)
        ctx_loss_weight: float = 2.0,        # MSE(AV, MV)
        m_pred_kl_weight: float = 0.5,       # KL(M || M_hat)

        # 行权重上限（把 nov_mask→weight，三项可分别设置）
        ce_row_max: float = 100.0,
        attn_row_max: float = 10.0,
        ctx_row_max: float = 10.0,
        mpred_row_max: float = 10.0,

        # 上下文一致用的度量: "mse" or "cos"
        ctx_metric: str = "mse",
    ):
        super().__init__()
        self.config = instantiate_from_config(transformer_config)
        self.transformer = get_obj_from_str(model_class)(self.config)

        if tokenizer_config is not None:
            self.tokenizer = instantiate_from_config(tokenizer_config)
            # print(f"[DEBUG] LlamaLVM: loaded tokenizer from config: {tokenizer_config}")

        # --- 保存超参 ---
        self.ce_loss_weight = float(ce_loss_weight)
        # [新增] 保存参数
        self.spec_loss_weight = float(spec_loss_weight)
        self.spec_swap_prob = float(spec_swap_prob)
        
        self.gate_loss_weight = float(gate_loss_weight)
        self.attn_kl_weight = float(attn_kl_weight)
        self.ctx_loss_weight = float(ctx_loss_weight)
        self.m_pred_kl_weight = float(m_pred_kl_weight)

        self.ce_row_max = float(ce_row_max)
        self.attn_row_max = float(attn_row_max)
        self.ctx_row_max = float(ctx_row_max)
        self.mpred_row_max = float(mpred_row_max)

        assert ctx_metric in ("mse", "cos")
        self.ctx_metric = ctx_metric


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        nov_mask: Optional[torch.Tensor] = None,
        train_with_noise: bool = False,
        output_hidden_states: bool = False, # 控制是否返回最后一层特征
    ):
        device = input_ids.device
        ignore_index = -100

        # ---------- 安全 helper：在 float32 中计算行权重 ----------
        def safe_row_weight(base_mask_or_weights, max_val: float):
            """
            接受：
              - [B, T-1] 的 shift_nov_mask，或
              - [N] 的 g_gt_rows（行级）
            返回 float32 的权重张量，范围 [0, max_val]，并清理 NaN/Inf。
            """
            if base_mask_or_weights is None:
                return None
            # 允许 base 是已经是权重或是原始 mask；统一调用 transform 函数更简单
            if base_mask_or_weights.dim() == 1:
                w = transform_nov_mask_to_weight(base_mask_or_weights, max_val=max_val)
            else:
                w = transform_nov_mask_to_weight(base_mask_or_weights, max_val=max_val)
            w = w.to(device=device, dtype=torch.float32)
            w = torch.nan_to_num(w, nan=0.0, posinf=max_val, neginf=0.0)
            return w.clamp_min(0.0).clamp_max(float(max_val))


        # ===== Step 2: transformer (主任务 Forward) =====
        res = self.transformer(
            input_ids = input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            nov_mask=nov_mask,
            output_hidden_states=output_hidden_states,
        )
        
        if output_hidden_states:
            logits = res.logits
            last_hidden_state = res.last_hidden_state
        else:
            logits = res

        # ===== Step 3: compute losses =====
        total_loss = None
        info: Dict[str, float] = {}

        # ---------- 3.1 Token CE (主任务) ----------
        ce_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()  # [B, T-1, V]
            shift_labels = labels[..., 1:].contiguous()      # [B, T-1]
            B, Tm1, V = shift_logits.shape

            valid_mask = (shift_labels != ignore_index)      # [B, T-1]

            # 交叉熵逐位；结果无梯度问题，之后转 float32 再乘权重
            per_token = F.cross_entropy(
                shift_logits.reshape(-1, V),
                shift_labels.reshape(-1),
                ignore_index=ignore_index,
                reduction="none",
            ).reshape(B, Tm1)  # [B, T-1]

            if nov_mask is None:
                row_w = torch.ones_like(shift_labels, dtype=torch.float32, device=device) * float(self.ce_row_max)
            else:
                shift_nov_mask = nov_mask[..., 1:].contiguous()     # [B, T-1]
                row_w = safe_row_weight(shift_nov_mask, max_val=self.ce_row_max)  # float32

            # 只在 valid 位置生效（valid 也升到 float32）
            w = (valid_mask.float().to(torch.float32) * row_w)  # float32
            per_token_f32 = per_token.to(torch.float32)

            ce_loss_f32 = (per_token_f32 * w).sum() / (w.sum().clamp_min(1e-8))
            ce_loss = ce_loss_f32.to(logits.dtype)  # 回到原 dtype（通常是 fp16/bf16）

            info["loss/token_ce"] = float(ce_loss_f32.detach().item())
            total_loss = self.ce_loss_weight * ce_loss

        # ============================================================
        # [新增] 3.2 Speculative Denoising Loss (辅助任务)
        # 逻辑：随机把部分 Input 替换为上一帧的 Token，但 Label 不变。
        # 这迫使模型学会：即使看到的是旧像素(Draft)，也要预测出正确的新像素(Target)。
        # ============================================================
        if self.spec_loss_weight > 0.0 and labels is not None and train_with_noise:
            # 1. 参数准备
            # 假设 model 内部有这些常量，或者从 config 获取
            LEN_O = getattr(self.transformer.model, 'LEN_O', 336)
            LEN_A = getattr(self.transformer.model, 'LEN_A', 11)
            PERIOD = LEN_O + LEN_A
            
            B, T = input_ids.shape
            
            # 2. 构造噪声输入 (Noisy Input)
            # 只有当历史长度足够回溯一帧时才替换 (idx >= PERIOD)
            if T > PERIOD:
                # 生成随机掩码: 概率 < prob
                rand_mask = torch.rand((B, T), device=device) < self.spec_swap_prob
                
                # 保护掩码: 前 PERIOD 个 token 不能换 (没得换)
                valid_period_mask = torch.arange(T, device=device).unsqueeze(0) >= PERIOD
                
                # 最终替换掩码
                swap_mask = rand_mask & valid_period_mask
                
                # 构造上一帧的 token (整体右移 PERIOD)
                prev_frame_tokens = torch.zeros_like(input_ids)
                prev_frame_tokens[:, PERIOD:] = input_ids[:, :-PERIOD]
                
                # 执行替换: 如果 mask 为 True，用上一帧的；否则用原来的
                noisy_input_ids = torch.where(swap_mask, prev_frame_tokens, input_ids)
                
                # 3. 辅助 Forward (Noisy Pass)
                # 注意：这里不需要梯度传回 input，只传回模型参数
                # 关键点：position_ids 不变！RoPE 会保证相对位置正确。
                noisy_logits = self.transformer(
                    input_ids=noisy_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    nov_mask=nov_mask,
                )
                
                # 4. 计算辅助 Loss
                # Label 依然是原始的 labels (Ground Truth)
                shift_noisy_logits = noisy_logits[..., :-1, :].contiguous()
                # shift_labels 复用上面的
                
                per_token_spec = F.cross_entropy(
                    shift_noisy_logits.reshape(-1, self.config.vocab_size),
                    shift_labels.reshape(-1),
                    ignore_index=ignore_index,
                    reduction="none",
                ).reshape(B, -1) # [B, T-1]
                
                # 复用主任务计算出的权重 w (valid_mask * row_w)
                # 注意：这里 w 的长度是 T-1，需要确保维度对齐
                # 如果上面 ce_loss 计算中定义了 w，这里直接用
                # 否则需要重新计算一遍 w
                
                # (假设上面 ce_loss 代码块里已经计算了 w)
                spec_loss_f32 = (per_token_spec.to(torch.float32) * w).sum() / (w.sum().clamp_min(1e-8))
                spec_loss = spec_loss_f32.to(total_loss.dtype)
                
                info["loss/spec_denoise"] = float(spec_loss_f32.detach().item())
                total_loss = total_loss + self.spec_loss_weight * spec_loss

        # ===== 汇总 =====
        if total_loss is None:
            total_loss = torch.zeros((), device=device, dtype=logits.dtype)

        info["loss/total"] = float(total_loss.detach().to(torch.float32).item())

        ret = {"loss": total_loss, "logits": logits, "info": info}
        if output_hidden_states:
            ret["last_hidden_state"] = last_hidden_state # 只返回最后一层
        return ret
    


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        device=None,
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        self.rope_type = "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.max_position_embeddings = config.max_position_embeddings
        inv_freq, _ = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
    
    def _set_cos_sin_cache(self, device, dtype):
        """
        Set the cosine and sine cache for positional embeddings.

        Args:
            seq_len (int): The sequence length.
            device (str): The device on which the cache tensors will be stored.
            dtype: The data type of the cache tensors.

        """
        t = torch.arange(
            self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False
        )
    
    def forward(self, x, seq_len=None):
        """
        Forward pass of the LlamaRotaryEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape [bs, num_attention_heads, seq_len, head_size].
            seq_len (int): The sequence length. If greater than the cached length, the cache will be updated.

        Returns:
            tuple: A tuple containing two tensors, the cosine and sine embeddings, both of shape [1, 1, seq_len, dim].
        """
        if seq_len > self.max_position_embeddings:
            raise ValueError("seq length should less than max embedding")

        return (
            self.cos_cached[:seq_len, :].to(dtype=x.dtype),
            self.sin_cached[:seq_len, :].to(dtype=x.dtype),
        )

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
  
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        assert (self.head_dim * self.num_heads) == self.hidden_size, "hidden_size must be divisible by num_heads"
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.max_batch_size = getattr(config, "max_batch_size", 1)
        self.init_kv_cache()



    def init_kv_cache(self, dtype=torch.float16):
        cache_shape = (self.max_batch_size, self.max_position_embeddings, self.num_key_value_heads, self.head_dim)
        self.cache_k = torch.zeros(cache_shape, dtype=dtype).cuda()
        self.cache_v = torch.zeros(cache_shape, dtype=dtype).cuda()
    
    def pre_allocate_kv_cache(self, batch_size: int):
        """
        Explicitly re-allocate KV cache to support a larger batch size (parallelism).
        """
        if batch_size > self.max_batch_size:
            cache_shape = (batch_size, self.max_position_embeddings, self.num_key_value_heads, self.head_dim)
            # 保留旧数据
            old_k = self.cache_k
            old_v = self.cache_v
            
            self.cache_k = torch.zeros(cache_shape, dtype=old_k.dtype, device=old_k.device)
            self.cache_v = torch.zeros(cache_shape, dtype=old_v.dtype, device=old_v.device)
            
            # Copy old data
            current_bsz = old_k.shape[0]
            self.cache_k[:current_bsz] = old_k
            self.cache_v[:current_bsz] = old_v
            
            self.max_batch_size = batch_size

    def copy_kv_cache(self, src_idx: int, indices: torch.Tensor):
        """
        Copy KV cache from a source batch index to destination(s) at specific indices.
        
        Args:
            src_idx: The index of the source batch (the winner).
            indices: Tensor of position indices to sync. [K]
        """
        # Ensure indices is long tensor
        indices = indices.to(dtype=torch.long)
        
        # 取出源数据 (视图)
        # shape: [len(indices), n_head, head_dim]
        current_k = self.cache_k[src_idx, indices]
        current_v = self.cache_v[src_idx, indices]

        # 广播模式：复制给所有人 (最高效)
        # self.cache_k[:, indices] 的 shape 是 [B, len(indices), n_head, head_dim]
        # current_k.unsqueeze(0) 的 shape 是 [1, len(indices), n_head, head_dim]
        self.cache_k[:, indices] = current_k.unsqueeze(0)
        self.cache_v[:, indices] = current_v.unsqueeze(0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            positions_embedding = None,
    ):
        
        # print(f"[DEBUG] LlamaAttention.forward: hidden_states.shape = {hidden_states.shape}, position_ids.shape = {position_ids.shape}")
        # start_time = time.perf_counter()

        bsz, q_len, _ = hidden_states.size() # 1, 7350, 1024
        query_states = self.q_proj(hidden_states) # [1, 7350, 1024]
        key_states = self.k_proj(hidden_states) # [1, 7350, 256]
        value_states = self.v_proj(hidden_states) # [1, 7350, 256]
        # print(f"[DEBUG] q_len: {q_len}, k_len: {k_len}")
        # print(f"[DEBUG] query_states.shape: {query_states.shape}, key_states.shape: {key_states.shape}, value_states.shape: {value_states.shape}")
        
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ) # [1, 7350, 16, 64]
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ) # [1, 7350, 4, 64]
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ) # [1, 7350, 4, 64]

        cos, sin = positions_embedding # [7500, 64]

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        ) # [1, 7350, 16, 64], [1, 7350, 4, 64]
        
        self.cache_k[:bsz, position_ids] = key_states.to(self.cache_k.dtype).detach()
        self.cache_v[:bsz, position_ids] = value_states.to(self.cache_v.dtype).detach()
        key_states, value_states = (
                self.cache_k[:bsz, :, :],
                self.cache_v[:bsz, :, :],
            )
        
        # TODO: we need specify key_states and value_states for each query to speed up
        
        # TODO: idea is given as below
        # if bsz does not represent simple patch token is involved in forward()
        # if we have [1, t, 16, 64] where t represents small token length
        # softmax([q_i * k_j.T for j in img_rel_pos_bias]/ sqrt{d}) * [v_j for j in img_rel_pos_bias] = attn_i
        # attn_output = [attn_i for i in range(t)]
        
        ## Temporarily fix img_rel_pos_bias to check correctness

        # print(f"[DEBUG] key_states.shape before repeat: {key_states.shape}, value_states.shape before repeat: {value_states.shape}, query_states.shape: {query_states.shape}")

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)

        query_states, key_states, value_states = map(lambda x: x.transpose(1, 2), (query_states, key_states, value_states))
        

        # print(f"[DEBUG] key_states.dtype: {key_states.dtype}, value_states.dtype: {value_states.dtype}, query_states.dtype: {query_states.dtype}")
        if key_states.dtype != query_states.dtype:
            key_states = key_states.to(query_states.dtype)
            value_states = value_states.to(query_states.dtype)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).contiguous()
        
            
        # ========== Naive implementation (too slow) ==========
        # for i in range(q_len):
        #     key_states_i = key_states[:, (i + img_rel_pos_bias).clamp(min=0), :, :]
        #     value_states_i = value_states[:, (i + img_rel_pos_bias).clamp(min=0), :, :]
        #     key_states_i = key_states_i.repeat_interleave(self.num_key_value_groups, dim=2) # [1, T, 16, 64]
        #     value_states_i = value_states_i.repeat_interleave(self.num_key_value_groups, dim=2) # [1, T, 16, 64]
        #     query_states_i = query_states[:, i:i+1, :, :] # [1, 1, 16, 64]
        #     # query_states_i, key_states_i, value_states_i = map(lambda x: x.transpose(1, 2), (query_states_i, key_states_i, value_states_i))
        #     query_states_i = query_states_i.repeat_interleave(key_states_i.size(1), dim=1) 
            
        #     attn_output_i = memory_efficient_attention(
        #         query_states_i, key_states_i, value_states_i).transpose(1, 2)
        
        #     attn_output = attn_output_i if i == 0 else torch.cat([attn_output, attn_output_i], dim=1)
            

        # # 1. 转置后 shape: [B, H, Q, K]
        # q = query_states  # [B, H, Q, D]
        # k = key_states    # [B, H, K, D]
        # v = value_states  # [B, H, K, D]

        # # 2. Attention scores（未加 mask）
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, Q, K]

        # # 3. Attention mask（可选）
        # if attention_mask is not None:
        #     attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # # 4. Softmax 得到 attention weights
        # attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, Q, K]
        
        # # print(f"[DEBUG] attn_weights: shape={attn_weights.shape}, dtype={attn_weights.dtype}")
        
        

        # # ✅ 可以在这里保存 attention weights
        # if not hasattr(self, "latest_attn_weights"):
        #     self.latest_attn_weights = []

        # self.latest_attn_weights.append(attn_weights.mean(dim=1).detach().cpu())

        # # 5. attention output
        # attn_output = torch.matmul(attn_weights, v)  # [B, H, Q, D]
        # attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q, H, D]

        # print(attn_output_old.shape, attn_output_old[0][0][0][:10])
        # print(attn_output.shape, attn_output[0][0][0][:10])

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        # print(f"[DEBUG] Attention forward time: {time.perf_counter() - start_time:.4f} seconds")
        return attn_output
    

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            positions_embedding = None,
    ):
        """
        Forward pass for the LlamaDecoderLayer.

        Args:
            hidden_states (torch.FloatTensor): Input tensor of shape `(batch, seq_len, embed_dim)`.
            attention_mask (torch.FloatTensor, optional): Attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (torch.LongTensor, optional): Positional IDs tensor.


        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]: Tuple containing:
                - hidden_states (torch.FloatTensor): Output tensor.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            positions_embedding=positions_embedding,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =========================
# Cross-Attn（加入先验 + 监督信息输出）
# =========================


# =========================
# LlamaModel（加入 \hat M 预测、退火、gate 混合、监督信息上抛）
# =========================
class LlamaModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # === base
        self.padding_idx = config.pad_token_id
        self.vocab_size  = config.vocab_size
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm   = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.max_position_embedding = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.max_position_embedding, self.max_position_embedding, dtype=torch.bool)),
            persistent=False,
        )

        # === layout 常量
        self.LEN_O = IMAGE_TOKEN_LENGTH
        self.LEN_A = ACTION_TOKEN_LENGTH
        self.PERIOD = self.LEN_O + self.LEN_A


        # === \hat M 预测头： [q; bev_cls] → 84 logits
        hidden = config.hidden_size
        self.m_prior_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 84)
        )

        # === 推理缓存
        self._infer_bev_cache = None  # [B, 85, H] 或 None
        self._infer_bev_cls_cache = None

        # === 训练进度（退火）
        self.register_buffer("global_step_tensor", torch.zeros((), dtype=torch.long))
        self.register_buffer("total_steps_tensor", torch.ones((), dtype=torch.long))
        # 可配置退火参数
        self.teacher_frac = getattr(config, "teacher_frac", 0.6)  # 前30%步 teacher→student 混合
        self.beta0 = getattr(config, "beta0", 6.0)
        self.beta_min = getattr(config, "beta_min", 2.0)
        self.gate_teacher_frac = getattr(config, "gate_teacher_frac", 0.8)  # gate 的 teacher 混合更快收敛
                
        self.post_init()


    # -------- mask 构建（保持你的逻辑） --------
    def _create_attention_mask(self, position_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                               only_previous: bool = False):
        attn_mask = self.causal_mask[position_ids]
        # print(f"[DEBUG] position_ids shape: {position_ids.shape}, attn_mask shape: {attn_mask.shape}")
        
        # T = position_ids.size(0)
        # max_pos = self.causal_mask.size(-1)
        # pos_1d = position_ids.view(-1)
        # img_rel_pos_bias = torch.tensor([
        #     -1042, -695, -443, -442, -420, -419, -418, -396, -395, -394,
        #     -372, -371, -370, -349, -348, -347, -346, -345, -325, -324,
        #     -323, -322, -300, -299, -298, -51, -50, -49, -48, -47, -46,
        #     -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17,
        #     -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, 
        #     -3, -2, -1
        # ], device=position_ids.device)
        # # img_rel_pos_bias = self.img_rel_pos_bias.to(position_ids.device)
        # j = (pos_1d.unsqueeze(1) + img_rel_pos_bias).clamp(0, max_pos - 1)
        # img_mask = torch.zeros(T, max_pos, dtype=torch.bool, device=position_ids.device)
        # img_mask.scatter_(1, j, True)
        # # print(f"[DEBUG] img_mask shape: {img_mask.shape}")
        if attention_mask is not None:
            X = attn_mask.shape[0]
            custom_mask = attention_mask.bool().repeat(X, 1)
            causal_mask = attn_mask[:, :X]
            final_attn_mask = attn_mask.clone()
            final_attn_mask[:, :X] = causal_mask & custom_mask  #& img_mask[:, :X]
        else:
            final_attn_mask = attn_mask #& img_mask
        only_previous = False # 20260513 NEWLY ADDED
            
        # print(f"[DEBUG] final_attn_mask shape:{final_attn_mask.shape}")
        
        # NEW HERE
        if only_previous:
            # [Fix] 解决维度不匹配问题
            # 目标：生成形状为 [B, T, MaxPos] 的 mask
            # 规则：Query(行) 只能看到 Key(列) 属于上一帧及之前的 token
            
            # 1. 获取 Key 的最大位置 (对应 final_attn_mask 的最后一维)
            max_key_pos = final_attn_mask.size(-1)
            
            # 2. 生成 Key 的位置索引 [MaxPos]
            # [Fix] 去掉 .view(1, 1, -1)，让它自动广播。
            # 这样如果 position_ids 是 1D [T]，结果就是 2D [T, MaxPos]
            # 如果 position_ids 是 2D [B, T]，结果就是 3D [B, T, MaxPos]
            key_positions = torch.arange(max_key_pos, device=position_ids.device)
            
            # 3. 计算每个 Query 所在的帧起始位置 [B, T] 或 [T]
            # 只要 KeyPos < StartOfCurrentFrame，就说明 Key 在上一帧或更早
            query_frame_start = (position_ids // self.PERIOD) * self.PERIOD
            
            # 4. 生成 Mask
            # 比较：KeyPos < QueryFrameStart
            # unsqueeze(-1) 将 query_frame_start 变为 [..., 1]，以便与 [MaxPos] 广播
            frame_mask = key_positions < query_frame_start.unsqueeze(-1)
            
            final_attn_mask = final_attn_mask & frame_mask
            
        # print(f"[DEBUG] final_attn_mask shape: {final_attn_mask.shape}")
        # with open("check/1.txt", "w") as f:
        #     for i in range(final_attn_mask.shape[0]):
        #         line = ""
        #         for j in range(final_attn_mask.shape[1]):
        #             line += "[{i}][{j}]:{val}".format(i=i, j=j, val=int(final_attn_mask[i][j]))
        #         f.write(line + "\n")
        # print(f"[DEBUG] final_attn_mask: shape={final_attn_mask.shape}, dtype={final_attn_mask.dtype}")
        
        return final_attn_mask

    # -------- 主 forward --------
    def forward(
        self,
        input_ids,                  # [B, T]
        attention_mask=None,
        position_ids=None,          # [B, T]
        nov_mask=None,              # [B, T] —— gate 的 g_gt
        output_hidden_states=False, # 这里的含义改为：是否返回最后一层的 hidden_state
    ):
        B, T = input_ids.shape
        device = input_ids.device
        H = self.config.hidden_size
        
        # print(f"[DEBUG] postion_ids shape: {position_ids.shape}")

        # print(f"[DEBUG] LlamaModel forward: input_ids shape={input_ids.shape}, dtype={input_ids.dtype}")
        # token embedding
        hidden_states = self.embed_tokens(input_ids).clone()
        positions_embedding = self.rotary_emb(hidden_states, seq_len=self.max_position_embedding)
        final_attn_mask = self._create_attention_mask(position_ids, attention_mask, only_previous=True)
        # print(f"[DEBUG] LlamaModel forward: positions_ids {position_ids}")


        for i, layer in enumerate(self.layers):
            # print(f"[DEBUG] LlamaModel forward: layer {i} start, layer type: {type(layer)}")
            # start_time = time.perf_counter()
            hidden_states = layer(
                hidden_states,
                attention_mask=final_attn_mask,
                position_ids=position_ids,
                positions_embedding=positions_embedding,
            )
            # print(f"[DEBUG] LlamaModel forward: layer {i} end, time taken: {time.perf_counter() - start_time:.4f} seconds")

                    
        outputs_norm = self.norm(hidden_states)
        # print(f"[DEBUG] LlamaModel forward: input_ids shape={input_ids.shape}, outputs_norm shape={outputs_norm.shape}")

        
        if output_hidden_states:
            # 只返回最终经过 LayerNorm 的张量
            return outputs_norm, outputs_norm
        
        return outputs_norm

    def get_input_embeddings(self):
        return self.embed_tokens
    

class LlamaForCausalLM(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()      

    # def forward(
    #         self,
    #         input_ids: torch.LongTensor = None,
    #         position_ids: Optional[torch.LongTensor] = None,
    # ):

    #     outputs = self.model(
    #         input_ids=input_ids,
    #         position_ids=position_ids,
    #     )
    #     logits = self.lm_head(outputs[:, :, :])
    #     return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        nov_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False, 
        **kwargs,
    ):
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            nov_mask=nov_mask,
            output_hidden_states=output_hidden_states,
        )
        
        if output_hidden_states:
            hidden_states, last_hidden_state = res
            logits = self.lm_head(hidden_states)
            # 模拟一个带有 last_hidden_state 属性的对象，方便外部通过 .last_hidden_state 访问
            from dataclasses import dataclass
            @dataclass
            class ModelOutput:
                logits: torch.Tensor
                last_hidden_state: torch.Tensor
            return ModelOutput(logits=logits, last_hidden_state=last_hidden_state)
        else:
            logits = self.lm_head(res)
            return logits 
    def refresh_kvcache(self):
        for i in self.model.layers:
            i.self_attn.init_kv_cache()
            
    def prepare_parallel_speculation(self, max_candidates: int):
        """
        Pre-allocate KV cache for parallel speculation.
        Args:
            max_candidates: The maximum number of parallel candidates we expect (e.g., top_k=5).
                            This will resize the KV cache to (max_candidates + 1) batch size.
                            (Slot 0 for Main, Slots 1..K for Candidates)
        """
        required_bsz = max_candidates + 1
        for layer in self.model.layers:
            layer.self_attn.pre_allocate_kv_cache(required_bsz)

    def expand_kv_cache(self, src_idx: int, current_length: int):
        """
        Expand the KV cache by copying the history from one batch index to multiple destination indices.
        Args:
            src_idx: Source batch index (usually 0, Main/Guard).
            dst_indices: List of destination batch indices (e.g., [1, 2, 3, 4, 5] for 5 candidates).
            current_length: The length of history to copy.
        """
        for layer in self.model.layers:
            # 为了性能，这里可以进一步优化为一次性 tensor copy，但基于 layer 的循环简单易读
            layer.self_attn.copy_kv_cache(src_idx, torch.arange(current_length, device=layer.self_attn.cache_k.device))

    def restore_kv_cache(self, src_idx: int, indices: torch.Tensor):
        """
        Restore a set of specific positions in the KV cache from one batch index to all others.
        """
        for layer in self.model.layers:
            layer.self_attn.copy_kv_cache(src_idx, indices)

    def parallel_speculative_generate_step(
        self,
        draft_input_ids,     # [N, K+1] N个候选序列，每个序列长度 K+1 (StartToken + K Drafts)
        draft_position_ids,  # [N, K+1] 对应的 Position IDs
        history_len,         # int KV Cache 中已有的有效历史长度
        draft_batch_indices, # List[int] 长度为 N，指明每一行 draft 对应哪个 Batch Slot (e.g. [1,2,3])
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        """
        批次并行推测解码验证 (Batch Speculative Verification Step).
        同时验证 N 个候选序列。
        
        Args:
            draft_input_ids: [N, K+1]
                Row 0: [x_t, d^0_1, d^0_2, ...] (Candidate 0)
                Row 1: [x_t, d^1_1, d^1_2, ...] (Candidate 1)
                ...
        
        Returns:
            best_tokens (torch.Tensor): [L, 1] 验证后的最佳 Token 序列。
            valid_len (int): 接受的长度。
        """
        N = draft_input_ids.shape[0]
        K_plus_1 = draft_input_ids.shape[1]
        
        max_idx = max(draft_batch_indices)
        current_bsz = self.model.layers[0].self_attn.cache_k.shape[0]
        assert(max_idx < current_bsz), f"main_batch_idx {max_idx} out of bounds (max {current_bsz})"

        # 2. 构造全 Batch 输入
        # 这里我们需要构造一个 [B_model, K+1] 的大 Tensor，只填充感兴趣的行
        model_input_ids = torch.zeros((current_bsz, K_plus_1), dtype=draft_input_ids.dtype, device=draft_input_ids.device)
        model_position_ids = torch.zeros((current_bsz, K_plus_1), dtype=torch.long, device=draft_input_ids.device)
        
        for i, batch_idx in enumerate(draft_batch_indices):
            model_input_ids[batch_idx] = draft_input_ids[i]
            # [Fix] 这里的 position_ids 应该是 2D 的 [B, K+1]
            model_position_ids[batch_idx] = draft_position_ids[i]
            
        # 3. 前向计算 (Parallel Verification across Batches & Time)
        # 这一步会同时计算所有 Candidate 的所有 Step 的 Logits
        with torch.no_grad():
            logits = self.forward(
                input_ids=model_input_ids,
                position_ids=model_position_ids
            )
        
        # 4. 提取感兴趣的 Logits [N, K+1, V]
        relevant_logits = logits[draft_batch_indices] 
        
        # 5. 采样 / Argmax
        if top_p is not None:
            candidates = sample_n_top_p(relevant_logits, temperature, top_p)
        else:
            candidates = sample_n_top_k(relevant_logits, temperature, top_k)
        
        # candidates shape: [N, K+1]
        
        # 6. 验证逻辑 (找出 N 个候选里，那个匹配最长的)
        # 注意：这里我们不仅要 Verify，还要 Pick Best Candidate。
        # 简单策略：看哪个 Candidate 验证通过的长度最长。
        
        best_candidate_idx = -1
        max_valid_len = -1
        best_accepted_seq = None
        
        for i in range(N):
            input_seq = draft_input_ids[i]      # [x_t, d^i_1, d^i_2...]
            target_seq = input_seq[1:]          # [d^i_1, d^i_2, d^i_3...] (Expectation)
            
            verify_preds = candidates[i, :-1]   # [Pred(x_t), Pred(d^i_1)...] (Reality)
            bonus_token = candidates[i, -1]
            
            # Greedy Match Verification
            match_mask = (verify_preds == target_seq)
            # 找到第一个不匹配的位置
            # cumprod 技巧：[1, 1, 0, 1] -> [1, 1, 0, 0]
            valid_mask = match_mask.cumprod(dim=0)
            valid_len = valid_mask.sum().item()
            
            if valid_len > max_valid_len:
                max_valid_len = valid_len
                best_candidate_idx = i
                
                # 构造这一行的结果序列
                # 接受的部分 + 第一个修正的部分(或 Bonus)
                # accepted_part 从 target_seq 取，因为它代表 input_ids 中那些被验证为正确的 draft token
                accepted_part = target_seq[:valid_len]
                
                if valid_len == len(target_seq):
                    # 全对，送一个 Bonus
                    next_token = bonus_token.unsqueeze(0)
                else:
                    # 在 valid_len 处断了，取出该处的修正值
                    next_token = verify_preds[valid_len].unsqueeze(0)
                    
                best_accepted_seq = torch.cat([accepted_part, next_token], dim=0)

        # 7. 回写 (Restore)
        # 将最佳 Candidate 对应的 KV Cache 刷回 Main Batch
        
        if best_candidate_idx != -1:
            best_batch_idx = draft_batch_indices[best_candidate_idx]
            
            tokens_to_restore = len(best_accepted_seq) # valid_len + 1
            
            # Construct indices for restoration
            # Assuming we are just appending to history in a linear fashion for this specific function
            dev = self.model.layers[0].self_attn.cache_k.device
            indices = torch.arange(history_len, history_len + tokens_to_restore, device=dev, dtype=torch.long)

            self.restore_kv_cache(
                src_idx=best_batch_idx,
                indices=indices
            )
            return best_accepted_seq, tokens_to_restore
        
        return None, 0

    def naive_generate(self, input_ids, max_new_tokens, temperature=1.0, action_all=None, top_p=None, top_k=None):

        # print(action_all.shape) # [15, 1, 11]
        # self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        self.prefill = prefill
        # print(f"[DEBUG] naive_generate input_ids: {input_ids}")
        if action_all is not None:
            input_ids = torch.cat([input_ids, action_all[0]], dim=-1) # [1, 336 + 11]
        position_ids = torch.arange(0, input_ids.shape[1], device="cuda") # [0, 1, 2, ..., 345, 346]
        next_token = self.prefill(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            temperature=temperature,
            top_k = top_k, # None
            top_p = top_p, # 0.8
        ) # [1, 1], [1, 1]

        # self.decode_one_token = torch.compile(decode_one_token, mode="max-autotune", fullgraph=True)
        self.decode_one_token = decode_one_token
        position_ids = torch.tensor([input_ids.shape[1]], dtype=torch.long, device="cuda")
        # print(f"[DEBUG] position_ids: {position_ids}")
        
        # print(max_new_tokens) # 5040
        generated_tokens = decode_n_tokens(
            self,
            input_ids = next_token.view(1, -1),
            position_ids = position_ids,
            num_generate_tokens = max_new_tokens - 1,
            temperature = temperature,
            decode_one_token_function=self.decode_one_token,
            action=action_all,
            top_p = top_p,
            top_k = top_k,
        )
        return torch.cat(generated_tokens, dim=1)
    
    def prefill_for_gradio(self, input_ids, temperature=1.0):
        self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        last_pos = input_ids.shape[1]
        position_ids = torch.arange(0, last_pos, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            next_token = self.prefill(
                self,
                input_ids=input_ids,
                position_ids=position_ids,
                temperature=temperature,
            )
        return next_token, last_pos
    
    def decode_img_token_for_gradio(self, input_action, position_id, max_new_tokens, temperature=1.0):
        self.decode_one_token = torch.compile(decode_one_token, mode="max-autotune", fullgraph=True)
        # self.decode_one_token = decode_one_token
        # WARNING
        position_ids = torch.arange(position_id, position_id + input_action.shape[1], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            generated_tokens, position_id = decode_n_tokens_for_gradio(
                self,
                input_ids = input_action,
                position_ids = position_ids,
                num_generate_tokens = max_new_tokens,
                temperature = temperature,
                decode_one_token_function=self.decode_one_token,
            )
        # WARNING
        return generated_tokens, position_id
    
    def diagd_img_token_for_gradio(self, input_action, position_id, max_new_tokens, temperature=1.0, windowsize=2):
        self.decode_some_token = torch.compile(decode_some_token, mode="max-autotune", fullgraph=True)
        position_ids = torch.arange(position_id, position_id + input_action.shape[1], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            generated_tokens, position_id = img_diagd_decode_n_token_for_gradio(
                self,
                input_ids = input_action,
                position_ids = position_ids,
                num_generate_tokens = max_new_tokens,
                temperature = temperature,
                decode_some_token_function=self.decode_some_token,
                windowsize = windowsize,
            )
        return generated_tokens, position_id


    def img_diagd_generate(self, input_ids, max_new_tokens, temperature=1.0, action_all=None, windowsize=2, top_p=None, top_k=None):

        self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        # self.prefill = prefill
        start_time = time.perf_counter()
        input_ids = torch.cat([input_ids, action_all[0]], dim=-1)
        position_ids = torch.arange(0, input_ids.shape[1], device="cuda")
        next_token = self.prefill(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            temperature=temperature,
            top_k = top_k,
            top_p = top_p,
        )
        
        end_time = time.perf_counter()
        print(f"[DEBUG] Prefill time: {end_time - start_time:.4f} seconds")


        self.decode_some_token = torch.compile(decode_some_token, mode="max-autotune", fullgraph=True)
        # self.decode_some_token = decode_some_token
        position_ids = torch.tensor([input_ids.shape[1]], dtype=torch.long, device="cuda")

        generated_tokens = img_diagd_decode_n_tokens(
            self,
            input_ids = next_token.view(1, -1),
            position_ids = position_ids,
            num_generate_tokens = max_new_tokens - 1,
            temperature = temperature,
            decode_some_token_function=self.decode_some_token,
            windowsize = windowsize,
            action=action_all,
            prompt=input_ids,
            top_k = top_k,
            top_p = top_p,
        )
        return torch.cat(generated_tokens, dim=1)
    
    def vid_diagd_generate(self, input_ids, max_new_tokens,windowsize=2, temperature=1.0, action_all=None,**kwargs):

        self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        input_ids = torch.cat([input_ids, action_all[0]], dim=-1)
        position_ids = torch.arange(0, input_ids.shape[1], device="cuda")
        next_token = self.prefill(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            temperature=temperature,
        )

        self.decode_some_token = torch.compile(decode_some_token, mode="max-autotune", fullgraph=True)
        # self.decode_some_token = decode_some_token
        position_ids = torch.tensor([input_ids.shape[1]], dtype=torch.long, device="cuda")

        generated_tokens = video_diagd_decode_n_tokens(
            self,
            input_ids = next_token.view(1, -1),
            position_ids = position_ids,
            num_generate_tokens = max_new_tokens - 1,
            temperature = temperature,
            decode_some_token_function=self.decode_some_token,
            windowsize = windowsize,
            action=action_all,
            prompt=input_ids,
            **kwargs
        )
        return torch.cat(generated_tokens, dim=1)

    
    def speculative_generate_single_pic(
        self, 
        input_ids, 
        max_new_tokens, 
        action_all=None, 
        guess_step=2, 
        top_k=None, 
        top_p=None, 
        temperature=1.0
    ):
        """
        基于上一帧位置猜测的推测采样生成 (Speculative Decoding)
        策略：只要 Guess(t) 正确，就直接接受基于 Guess(t) 预测出的 Truth(t+1)。
        """
        LEN_IMG = getattr(self.model, 'LEN_O', 336) 
        LEN_ACT = getattr(self.model, 'LEN_A', 11)
        # 修改：因为 generated_ids 里不再存 Action，所以回溯上一帧只需要跳过 LEN_IMG
        PERIOD = LEN_IMG 
        
        device = input_ids.device
        
        # 1. 初始 Prefill
        if action_all is not None:
            input_ids = torch.cat([input_ids, action_all[0]], dim=-1)

        # [新增] 独立的位置计数器，初始化为当前 input_ids 的长度 (指向下一个空位)
        pos_counter = input_ids.shape[1]
        
        position_ids = torch.arange(0, input_ids.shape[1], device=device)
        
        self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        self._compiled_spec_step = torch.compile(speculative_decoding_step, mode="max-autotune", fullgraph=True)
        
        next_token = self.prefill(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        generated_ids = torch.cat([input_ids.view(-1), next_token.view(-1)], dim=0)
        
        # 初始状态：curr_input_ids 是刚刚生成的 next_token
        curr_input_ids = next_token.view(-1)
        
        frame_idx = 0
        img_token_idx = 1
        tokens_generated_count = 1
        
        while tokens_generated_count < max_new_tokens:
            
            # --- 1. 构造猜测 (Draft) ---
            remaining_in_frame = LEN_IMG - img_token_idx
            step = min(guess_step, remaining_in_frame)
            if step <= 0: step = 1

            draft_tokens = []
            history_len = generated_ids.shape[0]
            
            valid_guess = True
            for k in range(step):
                # 计算当前猜测的索引（一个 frame 之前）
                src_idx = history_len + k - PERIOD
                if src_idx < 0: 
                    valid_guess = False
                    break
                guess = generated_ids[src_idx:src_idx+1]
                draft_tokens.append(guess)
            # print(f"[DEBUG] Draft tokens: {[t.item() for t in draft_tokens]}, draft_idx: {[(history_len + k - PERIOD) for k in range(step)]}")
            # print(f"[DEBUG] current_input: {[t.item() for t in curr_input_ids]}")
            if not valid_guess:
                draft_tokens = []
                step = 1 

            # --- 2. 构造模型输入 ---
            
            PASS_VALIDATE_GUESS = True
            
            if len(draft_tokens) > 0:
                # 使用所有 draft tokens
                draft_seq = torch.cat(draft_tokens[:], dim=0)
                model_input_1d = torch.cat([curr_input_ids, draft_seq], dim=0)
            else:
                model_input_1d = curr_input_ids

            model_input = model_input_1d.unsqueeze(0)

            # [修正 1] 动态计算 start_pos
            # pos_counter 指向下一个空位。
            # model_input 的结尾对应 pos_counter + step (预测未来)
            # model_input 的开头对应 pos_counter - len(curr_input_ids) (回填历史/修正Cache)
            start_pos = pos_counter - curr_input_ids.shape[0]
            model_pos = torch.arange(start_pos, start_pos + model_input.shape[1], device=device)
            # print(f"[DEBUG] curr_input_ids.shape: {curr_input_ids.shape}, draft_tokens count: {len(draft_tokens)}, model_input.shape: {model_input.shape}, model_pos: {model_pos}")
            # TODO: check position_ids correctness
            
            candidates_1d = self._compiled_spec_step(
                self,
                model_input,
                model_pos,
                temperature,
                top_k,
                top_p,
                len(draft_tokens) # 传入当前的 step 长度
            )
            
            # # --- 3. Forward ---
            # with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
            #     logits = self.forward(input_ids=model_input, position_ids=model_pos)
            
            # # --- 4. Verify & Accept (激进策略) ---
            # ctx_len = curr_input_ids.shape[0]
            

            # relevant_logits = logits[:, ctx_len - 1 : ctx_len + step]
            # # logits [t+1, t+2, ..., t+curr_input_len+step] 
            # # 假设输入 [t,...,t+curr_input_len+step-1]
            # # 其中输入中 [t,...,t+curr_input_len-1] 从 curr_input_ids 来，剩下的是 draft
            # # 现在相当于取出了 [t+curr_input_len, ..., t+curr_input_len+step] 的 logits
            # # 后面是我们真正要算的东西
            
            # if top_p is not None:
            #     candidates = sample_n_top_p(relevant_logits, temperature, top_p)
            # else:
            #     candidates = sample_n_top_k(relevant_logits, temperature, top_k)
            
            # candidates_1d = candidates.view(-1)
            # # 此时取出的是 [t+curr_input_len, ..., t+curr_input_len+step]
            
            # 始终接受第一个 (基于真实历史预测的)
            valid_tokens = [candidates_1d[0:1]]
            accepted_count = 1
            
            if not PASS_VALIDATE_GUESS:
                for k in range(len(draft_tokens)):
                    # draft_tokens: [t+curr_input_len, ..., t+curr_input_len+step-1]
                    guess = draft_tokens[k]
                    truth = candidates_1d[k+1] # 对应 guess 的下一个真实值
                    
                    if guess.item() == truth.item(): # TODO: change validation rule
                        # 猜对了！
                        # 直接接受下一个预测值 (candidates[k+1])
                        valid_tokens.append(candidates_1d[k+1:k+2])
                        accepted_count += 1
                    else:
                        # 猜错了，停止，后面的都不要了
                        break
            else:
                # 激进策略：直接接受后续所有基于猜测生成的 token
                if step >= 1:
                    valid_tokens.append(candidates_1d[1:step+1])
                    accepted_count = step + 1

            # --- 5. 防止溢出当前帧 ---
            remaining_in_frame = LEN_IMG - img_token_idx
            if accepted_count > remaining_in_frame:
                full_new_part = torch.cat(valid_tokens, dim=0)
                new_part = full_new_part[:remaining_in_frame]
                accepted_count = remaining_in_frame
            else:
                new_part = torch.cat(valid_tokens, dim=0)
            
            # 这里 new_part: [t+curr_input_len, ..., t+curr_input_len+accepted_count-1]
            # 大概率 accepted_count = step + 1
            # 为什么要留这么多？为了放到 kv_cache 里
            # 这里需要明晰的是：我们不需要把那些激进的推测重新放回cache
            # 因为已经接受了它们的预测结果

            # --- 6. Update History ---
            generated_ids = torch.cat([generated_ids, new_part], dim=0)
            
            img_token_idx += accepted_count
            tokens_generated_count += accepted_count
            
            # 更新 pos_counter
            pos_counter += accepted_count
            
            
            curr_input_ids = new_part

            # 边界处理：换帧
            if img_token_idx >= LEN_IMG:
                frame_idx += 1
                if frame_idx < action_all.shape[0]:
                    next_action = action_all[frame_idx].view(-1)
                    
                    # [修改 2] 优化 Action 拼接逻辑
                    # curr_input_ids 目前包含了本帧最后的几个像素 (new_part)
                    # 我们直接把 Action 拼在后面，一起喂进去。
                    # 这样：
                    # 1. 最后的像素会被计算 KV (刷新 Cache)
                    # 2. Action 会被计算 KV (建立 Cache)
                    
                    curr_input_ids = torch.cat([curr_input_ids, next_action], dim=0)
                    
                    img_token_idx = 0 
                    # Action 也是新生成的（虽然是强制的），位置也要加
                    pos_counter += next_action.shape[0] 
                else:
                    break 

        return generated_ids[input_ids.shape[1]:].unsqueeze(0)


    def speculative_diag_generate_img_token(self, input_ids, max_new_tokens, temperature=1.0, action_all=None, windowsize=2, top_p=None, top_k=None, draft_func=None, action_pred_func=None, use_baseline=False):
        
        # If baseline mode, just use standard (non-speculative) img_diagd_generate
        if use_baseline:
            return self.img_diagd_generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens,
                action_all=action_all, windowsize=windowsize,
                top_p=top_p, top_k=top_k, temperature=temperature
            )
        
        if draft_func is None or action_pred_func is None:
            draft_func, action_pred_func = get_inference_functions()
        # NOTE: speculative decoding feeds the model with VARYING shapes
        # (Main batch=1 vs Main+Spec batch=1+K, different diagonal lengths).
        # torch.compile with dynamic shapes on a 20-layer transformer has huge
        # compile overhead for every distinct shape. Run uncompiled for now to
        # verify correctness; compile optimizations come later.
        self.draft_func = draft_func
        self.action_pred_func = action_pred_func
        self.prefill = prefill
        print(f"[DEBUG] speculative_diag_generate_img_token input_ids: {input_ids}, shape: {input_ids.shape}, action_all shape: {action_all.shape}")
        # Concatenate the first action to the input, matching img_diagd_generate behavior
        input_ids = torch.cat([input_ids, action_all[0].view(1, -1)], dim=-1)
        position_ids = torch.arange(0, input_ids.shape[1], device="cuda")
        self.decode_some_token = torch.compile(decode_some_token, fullgraph=True, dynamic=True)
        
        generated_tokens = speculative_img_diagd_decode_n_tokens(
            self, input_ids, position_ids, num_generate_tokens=max_new_tokens,
            draft_func=self.draft_func, action_pred_func=self.action_pred_func,
            prefill_func=self.prefill, decode_some_token_function=self.decode_some_token,
            windowsize=windowsize, top_k=top_k, top_p=top_p, temperature=temperature,
            action=action_all, prompt=input_ids
        )
        
        # generated_tokens is list[list[int]] from speculative_img_diagd_decode_n_tokens.
        # Flatten into a single 1D tensor (matching img_diagd_generate's return convention).
        flat_tokens = []
        for seq in generated_tokens:
            if isinstance(seq, list):
                flat_tokens.extend(seq)
            elif isinstance(seq, torch.Tensor):
                flat_tokens.extend(seq.view(-1).tolist())
        
        return torch.tensor(flat_tokens, device=input_ids.device).unsqueeze(0)
