import torch
import torch.nn as nn
from transformers import LlamaConfig
from lvm import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

class EagleDraftModel(nn.Module):
    def __init__(self, config: LlamaConfig, num_hidden_layers: int = 1):
        """
        初始化 Eagle Draft Model。
        通常 Eagle 只使用 1 层 Transformer Layer，这使得它极快。
        """
        super().__init__()
        self.config = config
        
        # 1. 创建一份 Draft Model 的轻量级配置
        draft_config = LlamaConfig(**config.to_dict())
        draft_config.num_hidden_layers = num_hidden_layers
        
        # 2. 定义层
        # Embed Tokens: 将 input_ids 转为 hidden_size，用于与 Base Model 特征相加
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # [新增] 初始化旋转位置编码逻辑
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Transformer Layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(draft_config) for _ in range(num_hidden_layers)]
        )
        
        # Norm & Head
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        base_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values=None,
        use_cache=False,
    ):
        """
        Args:
            input_ids: [B, T]
            base_hidden_states: [B, T, H] -> 来自 Base Model 最后一层的输出 (必须 detach)
        """
        
        # 0. 确保长度对齐 (不使用截断，而是通过输入保证)
        # 假设 base_hidden_states.shape[1] == input_ids.shape[1] == 2418
        
        # 1. 融合特征
        draft_embeds = self.embed_tokens(input_ids)
        
        hidden_states = base_hidden_states + draft_embeds
        # 此时 hidden_states 会保持 fp16，后续生成的 query 自然也是 fp16
        
        # 2. 【核心修复】：构造“可广播”的 RoPE 缓存
        # 获取原始缓存 [Max_S, Dim]
        cos, sin = self.rotary_emb(hidden_states, seq_len=self.config.max_position_embeddings)
        
        pos_emb = (cos, sin)
        # print(f"[DEBUG] pos_emb shapes: cos {pos_emb[0].shape}, sin {pos_emb[1].shape}")

        # 3. 传入层
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                positions_embedding=pos_emb,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits