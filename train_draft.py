import os
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, TrainerCallback
from omegaconf import OmegaConf
import transformers

# 引入现有项目组件
from mcdataset import MCDataset
from train import (
    CustomArguments, ModelArguments, DataArguments, 
    make_supervised_data_module, DummyTokenizer, 
    PAD_TOKEN_ID
)
from util.helper import load_model
from util.draft_model import EagleDraftModel

class DraftModelWrapper(nn.Module):
    """
    包装器：包含冻结的 Base Model 和 可训练的 Draft Model。
    这样 Trainer 可以统一管理设备 (.to(device))。
    """
    def __init__(self, base_model, draft_model):
        super().__init__()
        self.base_model = base_model
        self.draft_model = draft_model
        
        # 冻结 Base Model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval() # 设为 eval 模式 (关闭 Dropout 等)
        
        # 激活 Draft Model
        for param in self.draft_model.parameters():
            param.requires_grad = True
        self.draft_model.train()

    def forward(self, input_ids, labels, token_types, **kwargs):
        # 这个 forward 主要是为了满足 HF Trainer 的检查机制
        # 实际逻辑全部在 EagleDraftTrainer.compute_loss 中
        pass

    def save_pretrained(self, output_dir):
        # 只保存 draft model
        torch.save(self.draft_model.state_dict(), os.path.join(output_dir, "draft_model.bin"))

class EagleDraftTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Eagle 训练逻辑：
        1. 跑 Base Model (Frozen) -> 拿 Last Hidden State
        2. 跑 Draft Model (Trainable) -> 拿 Logits
        3. 计算 CrossEntropy
        """
        # 兼容单卡和多卡模式，提取真正的模型对象
        model_inner = model.module if hasattr(model, "module") else model
        
        base_model = model_inner.base_model
        draft_model = model_inner.draft_model
        
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        
        # 注意: Llama 可能需要生成 position_ids
        # 这里假设 LlamaLVM 内部会自动处理，或者 data collator 没传 position_ids
        # 如果需要显式传递 position_ids，可以在这里生成
        position_ids = inputs.get("position_ids", None)

        # ==========================
        # Step 1: Base Model Forward
        # ==========================
        with torch.no_grad():
            # LlamaLVM 需要支持 output_hidden_states=True
            # 如果 LlamaLVM 对接的是 internal transformer，可能需要在这里调整调用方式
            # 假设 base_model 是 LlamaForCausalLM 或者是对其的封装
            
            # print(f"[DEBUG] input_ids: {input_ids.shape}, position_ids: {position_ids.shape}")
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
            # 获取最后一层的 Hidden State
            # outputs.hidden_states 是一个 tuple，最后一个元素是 last layer output
            base_last_hidden = outputs["last_hidden_state"]
        
        # 这里的 detach 非常重要！我们只训练 Draft，不更新 Base
        base_features = base_last_hidden.detach()

        # ==========================
        # Step 2: Draft Model Forward
        # ==========================
        # 输入：input_ids (用于 embedding) + base_features (用于增强)
        logits = draft_model(
            input_ids=input_ids,
            base_hidden_states=base_features,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        # ==========================
        # Step 3: Compute Loss
        # ==========================
        # 目标：预测下一个 token (Standard Causal LM Loss)
        # Shift logits 和 labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 修改：如果 num_items_in_batch 存在，使用 'sum' 模式进行更精确的分布式对齐
        if num_items_in_batch is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss / num_items_in_batch
        else:
            # 兼容旧版本
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, logits) if return_outputs else loss

    # 重写 save_model 以防 HF Trainer 同时也存了 Base Model 导致文件巨大
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 只保存 draft model 部分
        torch.save(self.model.draft_model.state_dict(), os.path.join(output_dir, "draft_model.pt"))
        print(f"Draft model saved to {output_dir}")


def train_draft_entry():
    parser = transformers.HfArgumentParser(
        (CustomArguments, ModelArguments, DataArguments, TrainingArguments))
    
    # 将启动命令塞进这个列表
    args_list = [
        "--config", "configs/300M_16f.yaml",
        "--model_ckpt", "/data/jjli/workspace/mineworld/checkpoints/300M_16f.ckpt",
        "--dataset_dir", "/data/cliang/mineworld/dataset/",
        "--evalset_dir", "/data/cliang/mineworld/validation",
        "--output_dir", "./checkpoints/draft_v1",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--num_train_epochs", "3",
        "--fp16", "True",
        "--save_steps", "500",
        "--frames", "15",
        "--max_train_length", "15"
    ]
    
    # 传递 args 列表，如果不传则默认从命令行 sys.argv 读取
    custom_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)

    # 1. 加载 Base Model (使用你现有的 helper)
    config = OmegaConf.load(custom_args.config)
    print(f"Loading Base Model from {custom_args.model_ckpt}...")
    
    # 强制 eval_mode=True 以确保 dropout 关闭
    # gpu=False 是因为我们想让 Trainer 来处理设备分配，
    # 或者如果你只有单卡，可以在 load_model 里让它上 GPU，但要小心显存
    base_model = load_model(config, custom_args.model_ckpt, gpu=True, eval_mode=True)
    
    
    # 2. 从 Base Model 获取 Config 来初始化 Draft Model
    # 因为我们需要词表大小、Hidden Size 等参数完全一致
    # 注意：Base Mode 可能是 LlamaLVM，它里面包裹着 transformer
    if hasattr(base_model, 'config'):
        base_config = base_model.config
    else:
        # Fallback: 尝试去 transformer 属性里找
        base_config = base_model.transformer.config

    print("Initializing Eagle Draft Model (1 layer)...")
    draft_model = EagleDraftModel(base_config, num_hidden_layers=1)

    # 3. 包装模型
    model_wrapper = DraftModelWrapper(base_model, draft_model)

    # 4. 准备数据
    # 使用你现有的 tokenizer (在 base_model里)


    data_module = make_supervised_data_module(
        data_args=data_args,
        frame_tokenizer=base_model.tokenizer,
        action_tokenizer=MCDataset(), 
        pad_token_id=PAD_TOKEN_ID
    )
    
    # 5. 设置 Trainer
    trainer = EagleDraftTrainer(
        model=model_wrapper,
        args=training_args,
        tokenizer=DummyTokenizer(pad_token_id=PAD_TOKEN_ID),
        **data_module,
    )

    print("Starting Draft Model Training...")
    trainer.train()

    # 6. 保存最终权重
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train_draft_entry()