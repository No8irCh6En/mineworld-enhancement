import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import cv2

import h5py

import numpy as np
import torch
import random

import transformers
import tokenizers

from rich import print
from PIL import Image
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
# --- 修改：移除 EarlyStoppingCallback ---
from transformers import Trainer, TrainerCallback
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime
from einops import rearrange

from util.helper import load_model
from inference.inference import get_args
from mcdataset import MCDataset
# from vis_tools import save_bev_batch_grid


# Constants
IGNORE_INDEX = -100

IMAGE_TOKEN_LENGTH = 336
ACTION_TOKEN_LENGTH = 11

PAD_TOKEN_ID = 8264
MIN_IMAGE_TOKEN_ID = 0
MAX_IMAGE_TOKEN_ID = 8191
MIN_ACTION_TOKEN_ID = 8192
MAX_ACTION_TOKEN_ID = 8261


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class CustomArguments:
    # data_root: str = field(metadata={"help": "Path to dataset root"})
    model_ckpt: str = field(metadata={"help": "Path to model checkpoint"})
    config: str = field(metadata={"help": "Path to model config file"})
    # output_dir: str = field(metadata={"help": "Directory to save outputs"})
    frames: int = field(metadata={"help": "Number of input frames"})
    
    demo_num: int = field(default=1, metadata={"help": "Number of demo samples to show"})
    window_size: int = field(default=2, metadata={"help": "Temporal window size"})
    accelerate_algo: str = field(default="naive", metadata={"help": "Accelerate algorithm option (e.g., naive)"})
    fps: int = field(default=6, metadata={"help": "FPS for video output"})

    top_k: Optional[int] = field(default=None, metadata={"help": "Use top-k sampling"})
    top_p: Optional[float] = field(default=None, metadata={"help": "Use top-p (nucleus) sampling"})

    val_data_num: int = field(default=500, metadata={"help": "Number of validation samples"})


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

    state_encoder: Optional[str] = field(default=None)
    state_size: Optional[int] = field(default=5120)
    token_num: Optional[int] = field(default=16)
    use_transformer: Optional[bool] = field(default=False)
    tune_state_encoder: Optional[bool] = field(default=False)
    pretrain_state_encoder: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    use_states: bool = False
    max_train_length: int = 2
    dataset_dir: str = '/data/cliang/mineworld'
    evalset_dir: str = '/data/cliang/mineworld/validation'
    frame_height: int = 224
    frame_width: int = 384
    # 将其明确视为 Dilated Attention 的偏移模式
    img_rel_token_offset: Optional[List[int]] = field(
        default_factory=lambda: [
            -1042, -695, -443, -442, -420, -419, -418, -396, -395, -394,
            -372, -371, -370, -349, -348, -347, -346, -345, -325, -324,
            -323, -322, -300, -299, -298, -51, -50, -49, -48, -47, -46,
            -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17,
            -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, 
            -3, -2, -1
        ],
        metadata={"help": "The fixed relative offsets for the Dilated Attention pattern."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_state_encoder: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    state_encoder_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    output_dir: str = "outputs"
    train_from_scratch: bool = True

    # --- 新增：设置评估和早停相关的默认参数 ---
    # 1. 开启评估策略，设置为 'steps' (按步数) 或 'epoch' (按轮数)
    evaluation_strategy: str = field(default="epoch", metadata={"help": "Evaluation strategy (steps/epoch)"})
    # 2. 评估间隔步数 (如果策略是 steps)
    eval_steps: int = field(default=500, metadata={"help": "Run evaluation every X steps"})
    # 3. 保存策略必须与评估策略一致 ('steps' 或 'epoch')，以便加载最佳模型
    save_strategy: str = field(default="epoch", metadata={"help": "Save strategy"})
    # 4. 保存间隔步数
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X steps"})
    # 5. 最多保留几个 checkpoint，防止硬盘爆满
    save_total_limit: int = field(default=1, metadata={"help": "Limit total checkpoints"})
    # 6. 训练结束时加载最佳模型
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load best model at end of training"})
    # 7. 用于判断最佳模型的指标 (通常是 eval_loss)
    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "Metric to use for early stopping"})
    # 8. 指标是否越大越好 (loss 是越小越好，所以是 False)
    greater_is_better: bool = field(default=False, metadata={"help": "Whether the metric is better when larger"})
    # 9. 日志记录频率
    logging_steps: int = field(default=50, metadata={"help": "Log every X steps"})


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        dataset_dir: str,
        frame_tokenizer,
        action_tokenizer,
        data_args,
    ):
        super(LazySupervisedDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.frame_tokenizer = frame_tokenizer
        self.action_tokenizer = action_tokenizer
        self.data_args = data_args

        self.index_tuples = []  # (episode_num: str, step_num: int)

        image_root = os.path.join(dataset_dir, "images")
        assert os.path.exists(image_root), f"Image folder not found: {image_root}"

        token_ids_root = os.path.join(dataset_dir, "token_ids")
        os.makedirs(token_ids_root, exist_ok=True)

        print("Indexing and tokenizing all episodes...")

        for episode_name in tqdm(sorted(os.listdir(image_root))):
            episode_path = os.path.join(image_root, episode_name)
            if not os.path.isdir(episode_path):
                continue

            episode_num = episode_name.replace("episode_", "")
            token_ids_path = os.path.join(token_ids_root, episode_name, "token_ids.h5")

            if os.path.exists(token_ids_path):
                with h5py.File(token_ids_path, "r") as f:
                    # --- 修改 1: 读取已存在文件时，同时检查 actions 和 images 的长度 ---
                    num_actions = len(f["actions"])
                    num_images = len(f["images"])
                    # 有效的 step 必须保证有对应的 action 和 下一帧 image
                    total_steps = min(num_actions, num_images - 1)
                
                # 如果 total_steps <= 0，说明数据有问题，跳过
                if total_steps > 0:
                    for step in range(total_steps):
                        self.index_tuples.append((episode_num, step))
                continue

            os.makedirs(os.path.dirname(token_ids_path), exist_ok=True)

            image_filenames = sorted([
                f for f in os.listdir(episode_path)
                if f.endswith(".png")
            ])
            if len(image_filenames) < 2:
                continue

            TARGET_SIZE = (self.data_args.frame_height, self.data_args.frame_width)  # (H, W)
            frames = []
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            # print(f"[DEBUG] image_filenames: {image_filenames} ... total {len(image_filenames)} frames")
            for img_file in image_filenames:
                img_path = os.path.join(episode_path, img_file)
                img = Image.open(img_path).convert("RGB")
                img = img.resize(TARGET_SIZE[::-1])  # PIL resize
                img_np = np.asarray(img)
                frame_tensor = torch.from_numpy(img_np.astype(np.uint8)).to("cuda")
                frame_tensor = frame_tensor.permute(2, 0, 1).float() / 255.0  # [3, H, W]
                frame_tensor = normalize(frame_tensor).unsqueeze(0)  # [1, 3, H, W]
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    frame_tensor = frame_tensor.to(dtype=torch.float16)
                    token_ids = frame_tokenizer.tokenize_images(frame_tensor)
                token_ids = token_ids.view(-1).cpu().numpy()
                frames.append(token_ids)

            image_tokens = np.stack(frames, axis=0)  # [T+1, 336]

            action_path = os.path.join(self.dataset_dir, "actions", episode_name, "action.jsonl")
            if not os.path.exists(action_path):
                print(f"[Warning] Missing action file for {episode_name}, skipping.")
                continue

            with open(action_path, 'r') as f:
                action_lines = f.readlines()

            action_tokens = []
            for line in action_lines:
                line = eval(line.strip(), {"__builtins__": None}, {})
                line['camera'] = np.array(line['camera'])
                token_ids = action_tokenizer.get_action_index_from_actiondict(line, action_vocab_offset=8192)
                action_tokens.append(token_ids)

            action_tokens = np.stack(action_tokens, axis=0)  # [T, 11]

            with h5py.File(token_ids_path, "w") as f:
                f.create_dataset("images", data=image_tokens, dtype='i8')
                f.create_dataset("actions", data=action_tokens, dtype='i8')

            # --- 修改 2: 新创建文件时，同样限制 total_steps ---
            total_steps = min(len(action_tokens), len(image_tokens) - 1)
            
            if total_steps > 0:
                for step in range(total_steps):
                    self.index_tuples.append((episode_num, step))


    def __len__(self):
        return len(self.index_tuples)
    
    @property
    def lengths(self):
        pass

    @property
    def modality_lengths(self):
        pass

    def __getitem__(self, i):
        # --- 恢复 __getitem__ 的原始逻辑，返回完整的序列 ---
        episode_num, step_num = self.index_tuples[i]
        step_num = int(step_num)
        k = self.data_args.max_train_length

        token_ids_path = os.path.join(self.dataset_dir, "token_ids", f"episode_{episode_num}", "token_ids.h5")

        with h5py.File(token_ids_path, "r") as f:
            image_tokens_all = f["images"]
            action_tokens_all = f["actions"]
            
            # --- 修改 3: 在 getitem 中计算 max_k 时也要考虑边界 ---
            num_actions = len(action_tokens_all)
            num_images = len(image_tokens_all)
            
            # 计算剩余可用的步数，受限于动作数量和图像数量
            # 我们需要读取到 step_num + max_k 的图像，所以 step_num + max_k < num_images
            limit_by_images = num_images - 1 - step_num
            limit_by_actions = num_actions - step_num
            
            # 取最小值，确保不越界
            max_steps_available = min(limit_by_images, limit_by_actions)
            max_k = min(k, max_steps_available)
            
            # 如果 max_k < 0 (理论上不应该发生，因为 __init__ 过滤了)，则设为 0 或处理异常
            max_k = max(0, max_k)

            img_range = range(step_num, step_num + max_k + 1)
            act_range = range(step_num, step_num + max_k)
            image_tokens = [image_tokens_all[t][:] for t in img_range]
            action_tokens = [action_tokens_all[t][:] for t in act_range]

        input_ids = []
        labels = []
        token_types = []

        # 初始图像帧 (上下文)
        input_ids.extend(list(image_tokens[0]))
        labels.extend([IGNORE_INDEX] * IMAGE_TOKEN_LENGTH)
        token_types.extend(list(range(IMAGE_TOKEN_LENGTH)))

        # 后续的 动作-图像 对
        for j in range(1, max_k + 1):
            # 动作 (上下文)
            input_ids.extend(list(action_tokens[j-1]))
            labels.extend([IGNORE_INDEX] * ACTION_TOKEN_LENGTH)
            token_types.extend(list(range(IMAGE_TOKEN_LENGTH, IMAGE_TOKEN_LENGTH + ACTION_TOKEN_LENGTH)))

            # 后续图像帧 (预测目标)
            input_ids.extend(list(image_tokens[j]))
            labels.extend(list(image_tokens[j])) # 预测整个图像
            token_types.extend(list(range(IMAGE_TOKEN_LENGTH)))
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_types": torch.tensor(token_types, dtype=torch.long),
        }


@dataclass
class DataCollatorForSupervisedDataset:
    """
    Collator for Dilated Attention.
    Creates gather_indices for sparse attention instead of a large mask.
    """
    pad_token_id: int
    img_rel_token_offset: List[int]

    def random_replace_tokens_by_scale(
        self, 
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        image_replace_prob: float = 0.5,
        action_replace_prob: float = 0.0,
        ):

        replaced = input_ids.clone()
        B, T = input_ids.shape

        prob_bins = [0.30, 0.20, 0.20, 0.20, 0.05, 0.05]
        mask_scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        chosen_scales = random.choices(mask_scales, weights=prob_bins, k=B)

        for b in range(B):
            scale = chosen_scales[b]
            img_prob = image_replace_prob * scale
            act_prob = action_replace_prob * scale

            for i in range(T):
                if token_types[b, i] == -1: # -1 是 padding 的 type
                    continue
                
                if 0 <= token_types[b, i] < IMAGE_TOKEN_LENGTH:
                    if random.random() < img_prob:
                        replaced[b, i] = random.randint(MIN_IMAGE_TOKEN_ID, MAX_IMAGE_TOKEN_ID)
                elif IMAGE_TOKEN_LENGTH <= token_types[b, i] < IMAGE_TOKEN_LENGTH + ACTION_TOKEN_LENGTH:
                    if random.random() < act_prob:
                        replaced[b, i] = random.randint(MIN_ACTION_TOKEN_ID, MAX_ACTION_TOKEN_ID)
        return replaced

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = [instance["input_ids"] for instance in instances]
        labels_list    = [instance["labels"]    for instance in instances]
        token_types_list = [instance["token_types"] for instance in instances]

        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        token_types_padded = pad_sequence(token_types_list, batch_first=True, padding_value=-1)

        # B, T = input_ids_padded.shape
        # device = input_ids_padded.device
        
        # # 1. 创建 gather_indices 用于 Dilated Attention
        # W = len(self.img_rel_token_offset)
        # offsets = torch.tensor(self.img_rel_token_offset, device=device, dtype=torch.long).view(1, 1, W)
        # q_indices = torch.arange(T, device=device).view(1, T, 1)
        # # gather_indices: [B, T, W], 存储每个query要关注的key的绝对索引
        # gather_indices = (q_indices + offsets).expand(B, -1, -1)

        # # 2. 创建 gather_mask 用于屏蔽无效的 gathered keys
        # # gather_mask: [B, T, W], True表示有效
        # # 因果性: k <= q  =>  q + r <= q  =>  r <= 0
        # causal_mask = (offsets <= 0) # Shape: [1, 1, W]
        # # 越界: k >= 0
        # out_of_bounds_mask = (gather_indices >= 0) # Shape: [B, T, W]
        # # 最终的 gather_mask
        # final_gather_mask = causal_mask & out_of_bounds_mask # Shape: [B, T, W]

        # 3. 创建 position_ids
        position_ids_list = [torch.arange(len(seq), dtype=torch.long) for seq in input_ids_list]
        position_ids_padded = pad_sequence(position_ids_list, batch_first=True, padding_value=0).squeeze(0)

        # 应用随机 token 替换
        input_ids_padded = self.random_replace_tokens_by_scale(
            input_ids_padded, token_types_padded, image_replace_prob=0.4, action_replace_prob=0.0
        )

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "position_ids": position_ids_padded,
            # 传递给模型的关键参数
            # "gather_indices": gather_indices,
            # "gather_mask": final_gather_mask,
        }
    

class MineWorldTrainer(Trainer):
    def create_optimizer(
        self,            # LoRA 层学习率
        ### BEV 部分暂时不使用
        # bev_embedder_lr: float = 1e-3,    # BEV embedder 学习率
        # bev_token_lr: float = 5e-4,       # [BEV_START]/[BEV_END] token embedding 学习率
        old_token_lr: float = 5e-5        # 原 token embedding 学习率（图像/动作）
    ):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        config = self.args
        base_lr = config.learning_rate
        lora_lr = config.learning_rate
        
        for name, param in model.named_parameters():
            if "tokenizer" in name:
                param.requires_grad = False

        is_lora = isinstance(model.transformer, PeftModel)

        if is_lora:
            embed_matrix = model.transformer.model.model.embed_tokens.weight
        else:
            embed_matrix = model.transformer.model.embed_tokens.weight
            print(f"[DEBUG]: embed_matrix size: {embed_matrix.size()}")

        embed_matrix.requires_grad = True

        ### BEV 部分暂时不使用
        # bev_token_ids = [BEV_START_TOKEN_ID, BEV_END_TOKEN_ID]
        # old_token_ids = [i for i in range(embed_matrix.size(0)) if i not in bev_token_ids]

        # bev_token_params = [embed_matrix[i] for i in bev_token_ids if embed_matrix[i].requires_grad]
        # old_token_params = [embed_matrix[i] for i in old_token_ids if embed_matrix[i].requires_grad]

        # bev_token_param_ids = {id(p) for p in bev_token_params}

        embed_params = [embed_matrix]

        decay_params = set()
        for name, param in model.named_parameters():
            if "bias" not in name and "layernorm" not in name:
                decay_params.add(name)

        ### BEV 部分暂时不使用
        # bev_decay_params = []
        # bev_nodecay_params = []

        # for name, param in model.bev_embedder.named_parameters():
        #     if not param.requires_grad:
        #         continue
        #     if "bias" in name or "layernorm" in name:
        #         bev_nodecay_params.append(param)
        #     else:
        #         bev_decay_params.append(param)

        # bev_embedder_params = [p for p in model.bev_embedder.parameters() if p.requires_grad]

        base_params = []
        lora_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "tokenizer" in name or "bev_embedder" in name or "embed_tokens" in name:
                continue
            ### BEV 部分暂时不使用
            # if id(param) in bev_token_param_ids:
            #     continue
            if "lora_" in name:
                lora_params.append((name, param))
            else:
                base_params.append((name, param))

        optimizer_grouped_parameters = []

        if not is_lora:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in base_params if n in decay_params],
                    "weight_decay": config.weight_decay,
                    "lr": base_lr,
                },
                {
                    "params": [p for n, p in base_params if n not in decay_params],
                    "weight_decay": 0.0,
                    "lr": base_lr,
                },
            ]

        if is_lora and lora_params:
            optimizer_grouped_parameters.append({
                "params": [p for _, p in lora_params],
                "weight_decay": 0.0,
                "lr": lora_lr,
            })


        optimizer_grouped_parameters.append({
            "params": embed_params,
            "weight_decay": 0.0,
            "lr": base_lr,
        })

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(config)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        param_log = {
            "frozen_parameters": [],
            "trainable_groups": []
        }

        param_id_to_name = {id(p): n for n, p in model.named_parameters()}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                param_log["frozen_parameters"].append(name)

        for group in optimizer_grouped_parameters:
            lr = group.get("lr", base_lr)
            group_names = []
            for p in group["params"]:
                pname = param_id_to_name.get(id(p), "<unnamed>")
                group_names.append(pname)
            param_log["trainable_groups"].append({
                "lr": lr,
                "weight_decay": group.get("weight_decay", 0.0),
                "parameters": group_names
            })

        os.makedirs(config.output_dir, exist_ok=True)
        param_log_path = os.path.join(config.output_dir, "optimizer_param_groups.json")
        with open(param_log_path, "w", encoding="utf-8") as f:
            json.dump(param_log, f, indent=2, ensure_ascii=False)

        print(f"[Info] Optimizer param groups saved to: {param_log_path}")
        return self.optimizer


    def _save_checkpoint(self, model, trial, metrics=None):
        try:
            return super()._save_checkpoint(model=model, trial=trial, metrics=metrics)
        except TypeError:
            # fallback to old interface
            return super()._save_checkpoint(model, trial)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)


class DummyTokenizer:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def save_pretrained(self, output_dir):
        pass 

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Not for actual use")


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['bev_embedder']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def make_supervised_data_module(
        data_args,
        frame_tokenizer,
        action_tokenizer,
        pad_token_id,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = LazySupervisedDataset(
        dataset_dir=data_args.dataset_dir,
        frame_tokenizer=frame_tokenizer,
        action_tokenizer=action_tokenizer,
        data_args=data_args,
    )
    
    ## NEW HERE
    
    eval_dataset = LazySupervisedDataset(
        dataset_dir=data_args.evalset_dir,
        frame_tokenizer=frame_tokenizer,
        action_tokenizer=action_tokenizer,
        data_args=data_args
    )

    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=pad_token_id,
        img_rel_token_offset=data_args.img_rel_token_offset,
    )

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (CustomArguments, ModelArguments, DataArguments, TrainingArguments))
    custom_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = OmegaConf.load(custom_args.config)

    if getattr(training_args, "train_from_scratch", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_output_dir = os.path.join(training_args.output_dir, timestamp)
        pathlib.Path(new_output_dir).mkdir(parents=True, exist_ok=True)
        training_args.output_dir = new_output_dir

    assert training_args.per_device_train_batch_size == 1 

    local_rank = training_args.local_rank # 1
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
        ## QUESTION 好像之后没用到 bnb_model_from_pretrained_args ?

    model = load_model(config, custom_args.model_ckpt, gpu=True, eval_mode=False)
    print(f"[bold magenta][MINEWORLD][TRAIN][/bold magenta] Load Model From {custom_args.model_ckpt}")
    
    if model_args.freeze_backbone:
        model.transformer.model.requires_grad_(False)
        # QUESTION 这个时候 model 的类型是什么？ 任何用了 transformer 的模型都可以这么用吗
        # 这里 transformer 并不是通俗意义上的 transformer 模型

    if training_args.bits in [4, 8]: # 16
        from peft import prepare_model_for_kbit_training
        model.transformer = prepare_model_for_kbit_training(model.transformer, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing: 
        if hasattr(model.transformer.model, "enable_input_require_grads"):
            model.transformer.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.transformer.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # QUESTION 这里开启关闭梯度下降的逻辑看晕了
    # NOT TEMP TRUE

    if training_args.lora_enable: # True
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,                                     # LoRA 的秩（Rank），决定 LoRA 低秩矩阵的维度。r 越大，模型表达能力越强，但训练参数也会增多. 取 128
            lora_alpha=training_args.lora_alpha,                        # 缩放系数，类似于学习率的缩放因子，控制 LoRA 适配器的权重比例。取 256
            target_modules=find_all_linear_names(model.transformer),    # 找到所有 Linear 层，并在这些层应用 LoRA。LoRA 仅对 Linear 层进行参数优化，不会影响 Transformer 结构。
            lora_dropout=training_args.lora_dropout,                    # LoRA dropout 率，用于防止过拟合。取 0.05
            bias=training_args.lora_bias,                               # 是否对 LoRA 适配器的偏置项进行训练。取 none
            task_type="CAUSAL_LM",                                      # 指定任务类型。"CAUSAL_LM" 代表 自回归语言模型（如 LLaVA、GPT-3、Llama）。如果是 seq2seq 任务，应该用 "SEQ_2_SEQ_LM"。
        )

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")

        model.transformer = get_peft_model(model.transformer, lora_config)  # model 现在是 一个带有 LoRA 适配器的 PEFTModel，其中 Linear 层将使用 LoRA 低秩矩阵进行训练。

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
            ## QUESTION 这个模块的逻辑是什么

    data_module = make_supervised_data_module(
        data_args=data_args,
        frame_tokenizer=model.tokenizer,
        action_tokenizer=MCDataset(),
        pad_token_id=PAD_TOKEN_ID,
    )
    
    trainer = MineWorldTrainer(
        model=model,
        tokenizer=DummyTokenizer(pad_token_id=PAD_TOKEN_ID),
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("1")
        trainer.train(resume_from_checkpoint=True)
        
    else:
        print("2")
        trainer.train()
        ## QUESTION HOW?

    trainer.save_state()

    model.config.use_cache = True

    # 保存整个模型（权重 + config）
    # if training_args.local_rank in [-1, 0]:
    #     model.save_pretrained(training_args.output_dir)
    #     model.config.save_pretrained(training_args.output_dir)

    if training_args.local_rank in [-1, 0]:
        # torch.save(model, os.path.join(training_args.output_dir, "full_model.pt"))
        # print("✅ Full model saved including structure and parameters.")

        save_path = os.path.join(training_args.output_dir, "model_weights.pt")

        # 判断是否是 DeepSpeed 封装的模型
        if hasattr(model, "module"):
            # DeepSpeed 模型（或 DDP），要 unwrap
            state_dict = model.module.state_dict()
        else:
            # 普通模型
            state_dict = model.state_dict()

        torch.save(state_dict, save_path)
        print(f"Saved model weights to {save_path}")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")