import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import cv2
import argparse
import os
import glob
import json
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt
import random
from collections import defaultdict
import csv # Added import for logging
import re # Added import
from torchvision import transforms
import datetime # [新增] 用于生成时间戳

# --- Imports ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from util.dataset_reader import MultiModalDataset, VideoEvalDataset, SequentialMultiModalDataset, evaluate_video_metrics
    from util.attn_model import AttentionTokenPredictor
    from mcdataset import MCDataset
    # 新增: 导入 NeighborLoss
    # from util.neighbor_loss import NeighborConsistencyLoss
    # NeighborConsistencyLoss = None
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback definition if file not found immediately (for safety)


# Import VAE
try:
    from vae import VAE
except ImportError:
    from vae import VAE

# Import DepthAnything
try:
    from util.DepthAnythingWrapper import DepthAnythingWrapper, DEPTH_ANYTHING_TRANSFORM
except ImportError:
    DepthAnythingWrapper = None 
    DEPTH_ANYTHING_TRANSFORM = None


# --- Helper: Get Tokens from Input Image ---
def get_input_tokens(vae, img_tensor):
    """
    将输入图像编码为 Token ID，用于计算 Static Loss (Merge Loss)
    img_tensor: (B, 3, H, W) range [-1, 1]
    Returns: (B, H, W) token indices
    """
    with torch.no_grad():
        # 假设 VAE 接受 [-1, 1] 的输入
        z = vae.model.encoder(img_tensor)
        z = vae.model.quant_conv(z)
        # quantize 返回: z_q, loss, (perplexity, min_encodings, min_encoding_indices)
        _, _, info = vae.model.quantize(z)
        # info[2] 是 indices, shape 通常是 (B*H*W) 或 (B, H*W)
        # 需要 reshape 回 (B, H_down, W_down)
        # 假设下采样倍率是 16 (224/16=14, 384/16=24)
        h_dim = z.shape[2]
        w_dim = z.shape[3]
        input_tokens = info[2].view(z.shape[0], h_dim, w_dim)
    return input_tokens


# --- Visualization Helper ---
def visualize_uncertainty(model, vae, depth_model, loader, device, output_dir, epoch, max_save=4):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    
    # 统计指标容器
    metrics = {
        "raw_acc": 0, "merged_acc": 0,
        "raw_mse": 0, "merged_mse": 0,
        "count": 0
    }

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if saved >= max_save: break
            
            # [数据解包]
            if len(batch_data) >= 8:
                (img_tensor, depth_input, action_vec, _,
                 target_token, _, _, _) = batch_data
            else:
                (img_tensor, depth_input, action_vec, _, target_token) = batch_data

            img_tensor = img_tensor.to(device)
            depth_input = depth_input.to(device)
            action_vec = action_vec.to(device)
            target_token = target_token.to(device)
            
            if target_token.dim() == 4 and target_token.size(1) == 1:
                target_token = target_token.squeeze(1)
            
            # Depth & Input Prep
            depth_map = depth_model(depth_input)
            depth_map = depth_map.unsqueeze(1)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            img_depth = torch.cat([img_tensor, depth_map], dim=1)
            
            # --- Forward ---
            pred_conf_logits, pred_token_logits = model(img_depth, action_vec) # Logs: (B, 8192, H, W)
            confidence = torch.sigmoid(pred_conf_logits) # (B, 1, H, W)
            pred_token_ids = torch.argmax(pred_token_logits, dim=1) # (B, H, W)
            
            # --- Logic for Merged Prediction (Post-Processing) ---
            # 为了得到 Merged Token，我们需要对比模型预测概率和“抄袭上一帧”的概率
            # 但在可视化层面，直接在像素/Latent层面融合更直观
            
            # 1. 获取 Input 的 Token (作为 Static Baseline)
            input_tokens = get_input_tokens(vae, img_tensor) # (B, H, W)
            
            # 2. Merged Token Selection (Hard Selection for Accuracy Calculation)
            # 如果 Conf > 0.5 (或随机采样)，选 Pred，否则选 Input
            # 这里我们用 Hard Threshold 0.5 来演示
            merged_token_ids = torch.where(confidence.squeeze(1) > 0.5, pred_token_ids, input_tokens)

            # --- Metrics Calculation ---
            # Token Accuracy
            raw_correct = (pred_token_ids == target_token).float().mean().item()
            merged_correct = (merged_token_ids == target_token).float().mean().item()
            metrics["raw_acc"] += raw_correct
            metrics["merged_acc"] += merged_correct
            metrics["count"] += 1
            
            # --- Decode Images ---
            codebook = vae.model.quantize.embedding.weight
            
            # Helper to decode
            def safe_decode(ids):
                # Permute logic: (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
                z = F.embedding(ids, codebook).permute(0, 3, 1, 2)
                z = vae.model.post_quant_conv(z)
                return vae.model.decoder(z)

            # 1. GT Image
            img_gt = safe_decode(target_token)
            
            # 2. Raw Pred Image
            img_pred = safe_decode(pred_token_ids)
            
            # 3. Input Image (Static Baseline)
            # img_input = img_tensor # 直接用输入图
            # 或者用重构图以保持 VAE 风格一致:
            img_input_recon = safe_decode(input_tokens)

            # 4. Neural Merged Image (Soft Blending in Image Space)
            # 这种融合视觉上更顺滑，展示了 Conf 的作用
            conf_expanded =  F.interpolate(confidence, size=img_pred.shape[2:], mode='nearest')
            img_merged = conf_expanded * img_pred + (1 - conf_expanded) * img_input_recon
            
            # Stats (MSE in Image Space)
            metrics["raw_mse"] += F.mse_loss(img_pred, img_gt).item()
            metrics["merged_mse"] += F.mse_loss(img_merged, img_gt).item()

            # --- Visualization ---
            for i in range(img_tensor.size(0)):
                if saved >= max_save: break
                
                def to_vis(t):
                    t = t.detach().cpu().numpy().transpose(1, 2, 0)
                    t = (t + 1.0) / 2.0
                    return np.clip(t, 0, 1)
                
                vis_gt = to_vis(img_gt[i])
                vis_input = to_vis(img_input_recon[i])
                vis_raw = to_vis(img_pred[i])
                vis_merged = to_vis(img_merged[i])
                
                # Confidence Heatmap
                vis_conf = confidence[i, 0].detach().cpu().numpy()
                vis_conf = cv2.resize(vis_conf, (384, 224), interpolation=cv2.INTER_NEAREST)
                vis_conf_color = cv2.applyColorMap((vis_conf * 255).astype(np.uint8), cv2.COLORMAP_JET)
                vis_conf_color = vis_conf_color.astype(np.float32) / 255.0

                # Layout: 1x5 [GT, Input(Prev), Raw Pred, Merged Pred, Conf]
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                
                axes[0].imshow(vis_gt)
                axes[0].set_title("GT (Next Frame)")
                
                axes[1].imshow(vis_input)
                axes[1].set_title("Input (Prev Frame)")
                
                axes[2].imshow(vis_raw)
                axes[2].set_title("Raw Model Pred")
                
                axes[3].imshow(vis_merged)
                axes[3].set_title("Merged (Conf Weighted)")
                
                axes[4].imshow(vis_conf_color)
                axes[4].set_title("Confidence (Red=High/Pred)")
                
                for ax in axes: ax.axis('off')
                
                plt.suptitle(f"Epoch {epoch} | Sample {saved} | Raw Acc: {raw_correct:.2%} | Merged Acc: {merged_correct:.2%}")
                plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_sample_{saved}.png"))
                plt.close()
                saved += 1
                
    # Print Metrics Summary
    if metrics["count"] > 0:
        print(f"\n[Vis Stats] Raw Acc: {metrics['raw_acc']/metrics['count']:.2%} | Merged Acc: {metrics['merged_acc']/metrics['count']:.2%}")
        print(f"[Vis Stats] Raw MSE: {metrics['raw_mse']/metrics['count']:.5f} | Merged MSE: {metrics['merged_mse']/metrics['count']:.5f}")

# --- Training Loop ---
def train(args):
    # [新增] 生成本次训练的唯一 ID (时间戳)
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建实验总目录，例如: experiments/20231027_123045
    experiment_dir = os.path.join("experiments", start_time)

    if args.use_neighbor_loss:
        from util.neighbor_loss import NeighborConsistencyLoss
    else: NeighborConsistencyLoss = None
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_master = (local_rank == 0)
    else:
        device = torch.device(args.device)
        is_master = True
        local_rank = 0

    if is_master: 
        print("Loading VAE & DepthAnything...")
        print(f"Target Coverage (Min Confidence): {args.target_coverage * 100}%")
        
        # --- [修改] 初始化 CSV Logger 到时间戳目录 ---
        os.makedirs(experiment_dir, exist_ok=True)
        # 创建 Checkpoints 子目录
        ckpt_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        # 创建 可视化 子目录
        vis_base_dir = os.path.join(experiment_dir, "vis_val")
        
        log_csv_path = os.path.join(experiment_dir, "loss_log.csv")
        
        # 如果不是 resuming 或者文件不存在，写入表头
        if not args.resume_from or not os.path.exists(log_csv_path):
            with open(log_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Epoch", "Total Loss", 
                    "Merge Loss (Weighted)", "Base Loss", "Coverage Loss", "Latent MSE Loss",
                    "Raw Pixel Loss (Prediction)", "Raw Static Loss (Input Copy)", 
                    "Avg Confidence", "Top1 Acc", "Top5 Acc"
                ])
        print(f"Logging loss history to {log_csv_path}")

    vae = VAE(args.vae_config, args.vae_ckpt)
    vae.to(device)
    vae.eval()
    for param in vae.parameters(): param.requires_grad = False
    
    depth_model = DepthAnythingWrapper(device, (384, 224))
    depth_model.eval()
    for param in depth_model.parameters(): param.requires_grad = False
    
    # --- Episode Splitting Logic ---
    all_label_files = sorted(glob.glob(os.path.join(args.label_dir, "*_labels.npy")))
    all_ep_names = [os.path.basename(f).replace("_labels.npy", "") for f in all_label_files]
    
    random.seed(42)
    random.shuffle(all_ep_names)
    
    if args.max_episodes > 0:
        all_ep_names = all_ep_names[:args.max_episodes]
        
    split_idx = int(0.9 * len(all_ep_names))
    train_ep_names = all_ep_names[:split_idx]
    val_ep_names = all_ep_names[split_idx:]
    
    if is_master:
        print(f"Train Episodes: {len(train_ep_names)} | Val Episodes: {len(val_ep_names)}")

    # --- Phase 1: Teacher Forcing Dataset (Efficient) ---
    # [修改] 不再使用两个分开的数据集，而是统一使用 SequentialMultiModalDataset
    # 这样可以在 Batch 内部动态决定用 GT 还是 Pred
    # train_set_tf = MultiModalDataset(...) <--- 删除或注释掉
    
    # 使用 SequentialMultiModalDataset 作为训练集
    print("Initializing Sequential Dataset for Schedule Sampling...")
    train_set = SequentialMultiModalDataset(args.image_dir, args.action_dir, args.label_dir, specific_episodes=train_ep_names)
    
    val_set = VideoEvalDataset(args.image_dir, args.action_dir, args.label_dir, episodes=val_ep_names[:8]) 

    if len(train_set) == 0: return

    # --- Samplers & Loaders ---
    if is_ddp:
        sampler = DistributedSampler(train_set, shuffle=True)
    else:
        sampler = None

    # 统一使用一个 Loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(sampler is None), 
                              num_workers=4, pin_memory=True, sampler=sampler)
    
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = AttentionTokenPredictor(input_channels=4, action_dim=11, num_tokens=8192).to(device)
    
    # [新增] 获取 VAE Codebook 并归一化，用于计算余弦相似度
    # 放在 model 定义之后，循环之前
    with torch.no_grad():
        codebook = vae.model.quantize.embedding.weight.detach().clone() # (8192, D)
        codebook_norm = F.normalize(codebook, p=2, dim=1).to(device)

    # Resume
    if args.resume_from and os.path.exists(args.resume_from):
        if is_master: print(f"Resuming from {args.resume_from}...")
        state_dict = torch.load(args.resume_from, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    if is_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 新增: 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # --- Loss Functions ---
    # 使用 NeighborConsistencyLoss 替代纯 CE
    neighbor_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "util", "all_token_neighbors_cosine.json")
    
    # 定义一个标准的 CE Loss，带有 Label Smoothing，用于 Warmup 和基础指导
    criterion_ce_smooth = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)

    if args.use_neighbor_loss:
        # 建议开启 use_spatial_tolerance 以处理轻微错位
        criterion_soft = NeighborConsistencyLoss(neighbor_json, device, alpha=0.5, use_spatial_tolerance=True)
    else:
        criterion_soft = criterion_ce_smooth # Fallback
    
    accumulation_steps = args.accum_steps
    best_trusted_acc = 0.0
    
    # 加载邻居 Map 用于评估 (以及用于 Static Loss 计算)
    neighbor_map = {}
    if os.path.exists(neighbor_json):
        try:
            with open(neighbor_json, 'r') as f:
                ndata = json.load(f)
                for k, v in ndata.items():
                    neighbor_map[int(k)] = set(v.get('neighbors', []))
        except: pass

    # --- Schedule Sampling Parameters ---
    # 初始 GT 概率 (1.0 = 100% Teacher Forcing)
    ss_prob_gt = 1.0 
    # 每次衰减的幅度
    ss_decay_rate = 0.1 
    # 最终最小 GT 概率 (保留一点 GT 引导通常比较好)
    ss_min_gt = 0.3

    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)
            
        # --- [Schedule Sampling Logic] ---
        # 策略: 每 5 个 Epoch 更新一次概率
        # 如果 epoch > warmup，开始衰减
        if epoch >= args.warmup_epochs and epoch % 5 == 0 and epoch > 0:
            ss_prob_gt = max(ss_min_gt, ss_prob_gt - ss_decay_rate)
            
        if is_master:
            print(f"\n--- Epoch {epoch+1} | GT Prob (p): {ss_prob_gt:.2f} | Pred Prob (1-p): {1-ss_prob_gt:.2f} ---")
        
        model.train()
        
        # Is Warmup?
        is_warmup = epoch < args.warmup_epochs
        current_cov_weight = 0.0 if is_warmup else args.cov_weight
        
        total_loss = 0
        total_conf = 0
        total_task_loss = 0 
        total_cov_loss = 0
        total_base_loss = 0
        total_top1 = 0 
        total_top5 = 0 
        total_raw_pixel_loss = 0
        total_raw_static_loss = 0
        
        # [新增] 统计变量
        total_latent_mse = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_master else train_loader
        
        for i, seq_batch_data in enumerate(iterator):
            # 安全解包逻辑
            (img_tensor, depth_input, action_vec,
                img_prev, depth_prev, action_prev, has_prev_flag,
                target_token, target_img, target_depth) = seq_batch_data

            img_tensor = img_tensor.to(device)
            depth_input = depth_input.to(device)
            action_vec = action_vec.to(device)
            target_token = target_token.to(device)
            has_prev_flag = has_prev_flag.to(device)
            target_img = target_img.to(device)
            target_depth = target_depth.to(device)
            
            
            if target_token.dim() == 4 and target_token.size(1) == 1:
                target_token = target_token.squeeze(1)

            # --- Schedule Sampling Decision ---
            # 决定当前 Batch (或者 Batch 中的每张图) 是用 GT 还是 Pred
            # 这里做 Image-level random: 抛硬币
            # 如果随机数 < p，使用 GT (Teacher Forcing)
            # 否则尝试使用 Pred (Self Forcing)
            use_gt = random.random() < ss_prob_gt
            
            # 如果没有上一帧数据 (img_prev is None)，强制使用 GT
            if img_prev is None: use_gt = True

            # --- Input Preparation ---
            current_img_input = img_tensor 
            current_depth_input = depth_input
            
            # --- Logic Branch ---
            if use_gt:
                # [新增] Data Augmentation: Noise Injection & Blur
                # 这就是你想要的 "数据增强"
                if random.random() < 0.5: # 50% 概率触发
                    aug_type = random.choice(['noise', 'blur'])
                    
                    if aug_type == 'blur':
                        # 模拟 VAE 模糊
                        sigma = random.uniform(0.1, 1.5)
                        current_img_input = transforms.GaussianBlur(kernel_size=5, sigma=sigma)(img_tensor)
                    
                    elif aug_type == 'noise':
                        # 模拟 Sensor Noise / VAE Artifacts
                        # 生成噪声 (B, 3, H, W)
                        noise = torch.randn_like(img_tensor) * 0.1 # 强度 0.1
                        # img_tensor 是 [-1, 1]，加噪后截断
                        current_img_input = torch.clamp(img_tensor + noise, -1.0, 1.0)
                else:
                    current_img_input = img_tensor
                
                current_depth_input = depth_input
            else:
                # [Self-Forcing / Model Prediction]
                
                img_prev = img_prev.to(device)
                depth_prev = depth_prev.to(device)
                action_prev = action_prev.to(device)
                has_prev_flag = has_prev_flag.to(device)
                
                with torch.no_grad(): 
                    # 1. 计算 Prev Depth
                    d_map_prev = depth_model(depth_prev).unsqueeze(1)
                    d_map_prev = (d_map_prev - d_map_prev.min()) / (d_map_prev.max() - d_map_prev.min() + 1e-6)
                    inp_prev = torch.cat([img_prev, d_map_prev], dim=1)
                    
                    # 2. Predict t-1 -> t
                    # [修改] 同时也获取 Confidence Logits
                    pred_conf_logits_prev, pred_token_logits_prev = model(inp_prev, action_prev)
                    pred_tokens_prev = torch.argmax(pred_token_logits_prev, dim=1)
                    
                    # [新增] 计算 Confidence 并 Resize
                    conf_prev = torch.sigmoid(pred_conf_logits_prev) # (B, 1, 14, 24)
                    
                    # Resize Conf 到图像尺寸 (384, 224) 用于像素级融合
                    # 注意 img_prev 是 (B, 3, 224, 384) [H, W]
                    conf_prev_img = F.interpolate(conf_prev, size=(224, 384), mode='bilinear', align_corners=False)

                    # 3. Decode chunks (Safe VAE Decode)
                    decoded_chunks = []
                    decode_bs = 2 
                    for k in range(0, pred_tokens_prev.shape[0], decode_bs):
                        # ...existing code... (解码保持不变)
                        token_chunk = pred_tokens_prev[k : k + decode_bs]
                        latent_shape = (token_chunk.shape[0], 14, 24, 64)
                        with torch.autocast(device_type='cuda', dtype=torch.float32):
                            quant = vae.model.quantize.get_codebook_entry(token_chunk, latent_shape)
                            quant2 = vae.model.post_quant_conv(quant)
                            img_chunk = vae.model.decoder(quant2)
                        decoded_chunks.append(img_chunk.float())
                    
                    pred_img_prev = torch.cat(decoded_chunks, dim=0)
                    pred_img_prev = torch.clamp(pred_img_prev, -1.0, 1.0) # Raw Prediction

                    # [新增] 执行 Merged Logic
                    # Merged = Pred * Conf + Input(t-1) * (1 - Conf)
                    # 这里的 input(t-1) 就是 img_prev
                    pred_img_merged = pred_img_prev * conf_prev_img + img_prev * (1.0 - conf_prev_img)
                    
                    # 4. Replace Input
                    # mask=1 -> 用 Merged Pred; mask=0 -> 用 GT (img_tensor)
                    mask = has_prev_flag.view(-1, 1, 1, 1).bool() 
                    
                    # [修改] 使用 Merged 结果作为 Input
                    current_img_input = torch.where(mask, pred_img_merged, img_tensor)
                    
                    # 5. Re-calc Depth for new input
                    img_01 = (current_img_input * 0.5 + 0.5)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    norm_input = (img_01 - mean) / std
                    
                    current_depth_input = F.interpolate(norm_input, size=(224, 392), mode='bilinear', align_corners=False)

            # --- Forward Pass (Common) ---
            # Depth Calculation
            with torch.no_grad():

                depth_map = depth_model(current_depth_input)
                
                # depth_map 输出可能是 (B, 392, 224) 也就是和输入一样宽
                if depth_map.dim() == 3:
                     depth_map = depth_map.unsqueeze(1) # (B, 1, H, W)
                
                # [关键修复] Resize depth map back to VAE image size (384) for concatenation
                # 之前那个 assert depth_map.shape == (224, 384) 会报错，因为现在出来的是 392
                # 我们必须把它缩放回 384
                if depth_map.shape[-1] != 384:
                    depth_map = F.interpolate(
                        depth_map,
                        size=(224, 384), # <--- 缩放回 384 以匹配 VAE 输入
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 这里的 assert 可以保留了，因为我们上面手动 resize 回来了
                assert depth_map.shape[-2:] == (224, 384)
            
            img_depth = torch.cat([current_img_input, depth_map], dim=1)
            
            # [修改] Model Forward，请求返回特征
            # 注意：请确认 util/attn_model.py 中的 forward 已支持 return_features=True
            model_output = model(img_depth, action_vec, return_features=True)
            
            feats_ar = None
            if isinstance(model_output, tuple) and len(model_output) == 3:
                pred_conf_logits, pred_token_logits, feats_ar = model_output
            else:
                pred_conf_logits, pred_token_logits = model_output
            
            confidence = torch.sigmoid(pred_conf_logits) # (B, 1, 14, 24)

            # [新增] Feature Consistency Loss 定义
            loss_feat = torch.tensor(0.0, device=device)
            
            if (not use_gt) and (feats_ar is not None):
                with torch.no_grad():
                    # 1. 构建 Teacher (GT) 的输入
                    # depth_input 是原始 Batch 中的 GT Depth Input
                    d_map_gt = depth_model(depth_input)
                    if d_map_gt.dim() == 3: d_map_gt = d_map_gt.unsqueeze(1)
                    if d_map_gt.shape[-1] != 384:
                        d_map_gt = F.interpolate(d_map_gt, size=(224, 384), mode='bilinear', align_corners=False)
                    d_map_gt = (d_map_gt - d_map_gt.min()) / (d_map_gt.max() - d_map_gt.min() + 1e-6)
                    input_gt_teacher = torch.cat([img_tensor, d_map_gt], dim=1)
                    
                    # 2. Extract GT Features
                    _, _, feats_gt = model(input_gt_teacher, action_vec, return_features=True)
                    
                # [关键修改] 使用 Cosine Similarity 替代 MSE，且降低权重
                # 这样可以防止 MSE 导致的数值爆炸 (3.7 -> 0.0x)
                
                # Flatten features: (B, C, H, W) -> (B, -1)
                f_pred_ar = feats_ar['f_pred'].reshape(feats_ar['f_pred'].shape[0], -1)
                f_pred_gt = feats_gt['f_pred'].reshape(feats_gt['f_pred'].shape[0], -1)
                
                f_comb_ar = feats_ar['combined'].reshape(feats_ar['combined'].shape[0], -1)
                f_comb_gt = feats_gt['combined'].reshape(feats_gt['combined'].shape[0], -1)
                
                # Cosine Loss = 1 - CosineSimilarity
                loss_f_pred = 1.0 - F.cosine_similarity(f_pred_ar, f_pred_gt, dim=1).mean()
                loss_combined = 1.0 - F.cosine_similarity(f_comb_ar, f_comb_gt, dim=1).mean()
                
                # 现在的 Loss 范围是 [0, 2]，非常稳定，不会爆炸
                loss_feat = loss_f_pred + loss_combined
                
                # with torch.no_grad():
                
                #     d_map_gt = depth_model(target_depth)
                #     if d_map_gt.dim() == 3: d_map_gt = d_map_gt.unsqueeze(1)
                #     if d_map_gt.shape[-1] != 384:
                #         d_map_gt = F.interpolate(d_map_gt, size=(224, 384), mode='bilinear', align_corners=False)
                #     d_map_gt = (d_map_gt - d_map_gt.min()) / (d_map_gt.max() - d_map_gt.min() + 1e-6)
                #     input_gt_teacher = torch.cat([target_img, d_map_gt], dim=1)
                    
                #     _, _, feats_gt = model(input_gt_teacher, torch.zeros_like(action_vec), return_features=True)
                #     # Here no need to use action_vec for GT feature extraction
                    
                # f_pred_ar = feats_ar['f_pred'].reshape(feats_ar['f_pred'].shape[0], -1)
                # f_comb_ar = feats_ar['combined'].reshape(feats_ar['combined'].shape[0], -1)
                
                # f_tgt = feats_gt['f_curr'].reshape(feats_gt['f_curr'].shape[0], -1)
                # loss_f_pred = 1.0 - F.cosine_similarity(f_pred_ar, f_tgt, dim=1).mean()
                # loss_combined = 1.0 - F.cosine_similarity(f_comb_ar, f_tgt, dim=1).mean()
                # loss_feat = loss_f_pred + loss_combined
                    
                


            # --- [修复] 添加 Top-1 和 Top-5 准确率计算 ---
            with torch.no_grad():
                # pred_token_logits: (B, 8192, H, W) -> (B, H, W, 8192) -> (N, 8192)
                # 展平以便计算
                logits_flat = pred_token_logits.permute(0, 2, 3, 1).reshape(-1, 8192)
                target_flat = target_token.reshape(-1)
                
                # Top-1 Accuracy
                _, pred_top1 = logits_flat.max(dim=1)
                acc_top1 = (pred_top1 == target_flat).float().mean().item()
                
                # Top-5 Accuracy
                # topk 返回 (values, indices)
                _, pred_top5 = logits_flat.topk(5, dim=1) # (N, 5)
                # 将 target 扩展为 (N, 5) 以便比较
                target_expanded = target_flat.unsqueeze(1).expand_as(pred_top5)
                # 只要 5 个预测中有一个等于 target，就是 True
                acc_top5 = (pred_top5 == target_expanded).float().sum(dim=1).mean().item()
                
                total_top1 += acc_top1
                total_top5 += acc_top5
            # -------------------------------------------
            
            # 1. 获取真值的 Embedding (B, H, W, D)
            with torch.no_grad():
                target_emb = F.embedding(target_token, codebook) # (B, H, W, D)
                # target_emb = F.normalize(target_emb, p=2, dim=-1) # 注意：算 MSE 时最好不要 Normalize，要逼近真实数值

            # 2. 计算预测的加权 Embedding (Softmax Trick) -> 可微分的 Latent
            # pred_token_logits: (B, 8192, H, W) -> (B, H, W, 8192)
            pred_probs = F.softmax(pred_token_logits, dim=1).permute(0, 2, 3, 1) 
            
            # 矩阵乘法得到 Soft Latent
            b, c, h, w = pred_token_logits.shape
            pred_probs_flat = pred_probs.reshape(-1, c)
            
            # (N, 8192) @ (8192, D) -> (N, D)
            pred_emb_flat = pred_probs_flat @ codebook 
            
            # Reshape 回地图尺寸，方便后续可能的卷积操作
            pred_emb_map = pred_emb_flat.view(b, h, w, -1) # (B, H, W, D)
            
            # --- Constraint 1: Latent MSE (强制数值逼近) ---
            loss_latent_mse = F.mse_loss(pred_emb_map, target_emb)

            # =========================================================================
            # [新增] Spectral Texture Loss (频域纹理损失) - 专治平滑
            # =========================================================================
            # 原理: 平滑意味着高频丢失。我们在频域对比 Latent，强迫预测包含高频细节。
            
            # 1. 准备数据: (B, H, W, D) -> (B, D, H, W)
            # 使用 float32 确保 FFT 精度
            pred_fft_in = pred_emb_map.permute(0, 3, 1, 2).float()
            tgt_fft_in = target_emb.permute(0, 3, 1, 2).float()
            
            # 2. 快速傅里叶变换 (RFFT2)
            # 输出: (B, D, H, W/2 + 1) 的复数 Tensor
            pred_fft = torch.fft.rfft2(pred_fft_in, norm='ortho')
            tgt_fft = torch.fft.rfft2(tgt_fft_in, norm='ortho')
            
            # 3. 计算频谱的 Log 幅度 (Log-Magnitude)
            # 取 Log 是关键！因为低频能量通常巨大，高频(纹理)能量很小。
            #如果不取 Log，Loss 会被低频主导，依然学不到纹理。取 Log 后对高频更敏感。
            pred_mag = torch.log(torch.abs(pred_fft) + 1e-8)
            tgt_mag = torch.log(torch.abs(tgt_fft) + 1e-8)
            
            # 4. 计算 L1 Loss
            loss_spectral = F.l1_loss(pred_mag, tgt_mag)
            # =========================================================================

            # [可选] 之前的梯度损失也能抗平滑，如果有的话保留
            # loss_texture_grad = ...

            if is_warmup:
                loss_pixel = criterion_ce_smooth(pred_token_logits, target_token)
            else:
                loss_pixel = criterion_soft(pred_token_logits, target_token)

            # --- 3. Merge Loss Logic (Optimized) ---
            # 计算 "Static Loss": 如果直接用上一帧(Input)作为预测，Loss 是多少？
            
            # 获取 Input Tokens
            # 注意: 如果用 Self-Forcing，Input Image 变了，所以 Input Tokens 也要从 current_img_input 算
            input_tokens = get_input_tokens(vae, current_img_input) # (B, H, W)
            
            # [优化] 使用 Embedding Cosine Similarity + 3x3 空间容忍计算 Static Loss
            with torch.no_grad():
                # 1. 获取 Embedding
                # input_tokens: (B, H, W) -> (B, H, W, D)
                inp_emb = F.embedding(input_tokens, codebook_norm)
                tgt_emb = F.embedding(target_token, codebook_norm)
                
                
                if args.use_neighbor_loss:
                    # 2. 准备 3x3 邻域搜索 (Spatial Tolerance)
                    # 我们希望：如果 Target 和 Input 的 (x,y) 或其周围 3x3 邻居相似，则 Static Loss 很小
                    # inp_emb: (B, H, W, D) -> (B, D, H, W)
                    inp_emb_perm = inp_emb.permute(0, 3, 1, 2)
                    
                    # 使用 unfold 提取 3x3 滑动窗口
                    # output: (B, D*9, H, W)
                    # padding=1 保证输出尺寸不变
                    inp_unfolded = F.unfold(inp_emb_perm, kernel_size=3, padding=1)
                    inp_unfolded = inp_unfolded.view(inp_emb.shape[0], inp_emb.shape[3], 9, inp_emb.shape[1], inp_emb.shape[2])
                    # shape: (B, D, 9, H, W)
                    
                    # tgt_emb: (B, H, W, D) -> (B, D, 1, H, W)
                    tgt_emb_expanded = tgt_emb.permute(0, 3, 1, 2).unsqueeze(2)
                    
                    # 3. 计算相似度 (Cosine Similarity)
                    # sum(A * B) over dim=1 (D dimension)
                    # sim_matrix: (B, 9, H, W) -> 每个像素对应 Input 3x3 邻域的 9 个相似度
                    sim_matrix = (inp_unfolded * tgt_emb_expanded).sum(dim=1)
                    
                    # 4. 取最大相似度 (Best Match in 3x3)
                    max_sim, _ = sim_matrix.max(dim=1) # (B, H, W)
                else: max_sim = (inp_emb * tgt_emb).sum(dim=-1) 
                
                # 5. 定义 Static Loss
                # 如果相似度是 1.0，Loss = 0
                # 如果相似度是 0.0，Loss = 1.0 (或者更大，可加权重)
                # 这里的 5.0 是一个缩放因子，让不相似的惩罚更重
                loss_static_map = (1.0 - max_sim) * 5.0
                
                # 截断一下，防止负数（虽然理论上 cos sim <= 1）
                loss_static_map = torch.clamp(loss_static_map, min=0)

            # Merge Loss 公式:
            # Loss = Conf * Loss_Pred + (1 - Conf) * Loss_Static
            
            if is_warmup:
                loss_merge = loss_pixel.mean()
            else:
                # 注意：loss_static_map 不需要梯度，它只是一个 Target
                loss_merge = (confidence * loss_pixel + (1.0 - confidence) * loss_static_map).mean()
            
            # [新增] 记录原始的 Pixel Loss 和 Static Loss 均值，用于分析是否过于拟合静态背景
            with torch.no_grad():
                total_raw_pixel_loss += loss_pixel.mean().item()
                total_raw_static_loss += loss_static_map.mean().item() # 这代表如果全盘照抄上一帧，会产生的 Loss
                total_latent_mse += loss_latent_mse.item() # Record

            # 4. Base Loss (Unweighted)
            # 依然保留，用于防止模型在 Conf=0 时彻底停止学习预测头
            loss_base = loss_pixel.mean()
            
            # 5. Coverage Loss
            avg_conf = confidence.mean()
            loss_cov = torch.clamp(args.target_coverage - avg_conf, min=0) ** 2
            
            # 6. Total Loss
            # 修改 Loss 公式
            
            # [关键调整] 权重配置
            # 降低 MSE 权重 (防止为了数值准确而牺牲纹理)
            latent_mse_weight = 5.0 # 原来是 10.0，建议调低
            
            feat_loss_weight = 2.0
            
            # [新增] 频域损失权重
            # 这个 Loss 通常数值在 0.1~0.5 左右，给一个较大的权重强迫模型重视
            spectral_weight = 5.0 
            
            loss = loss_merge + args.base_loss_weight * loss_base + current_cov_weight * loss_cov \
                 + latent_mse_weight * loss_latent_mse \
                 + feat_loss_weight * loss_feat \
                 + spectral_weight * loss_spectral  # <--- 加入这一项！
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            # 记录日志
            # print(f"... LatentMSE={loss_latent_mse.item():.4f} ...")
            total_conf += avg_conf.item()
            total_task_loss += loss_merge.item() # Log merge loss
            total_base_loss += loss_base.item()
            total_cov_loss += loss_cov.item()
            
        # Step Scheduler
        scheduler.step()

        if is_master:
            num_batches = len(train_loader)
            
            # 计算平均值
            avg_loss = total_loss / num_batches
            avg_merge = total_task_loss / num_batches
            avg_base = total_base_loss / num_batches
            avg_cov = total_cov_loss / num_batches
            avg_raw_pix = total_raw_pixel_loss / num_batches
            avg_raw_static = total_raw_static_loss / num_batches
            avg_conf_val = total_conf / num_batches
            avg_top1 = total_top1 / num_batches
            avg_top5 = total_top5 / num_batches
            avg_lat_mse = total_latent_mse / num_batches

            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | "
                  f"Avg Conf={avg_conf_val:.4f} | "
                  f"Acc Top1={avg_top1:.2%} | "
                  f"Acc Top5={avg_top5:.2%} | "
                  f"MergeLoss={avg_merge:.4f} | "
                  f"LatentMSE={avg_lat_mse:.5f}")
            
            # CSV 写入
            with open(log_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, 
                    f"{avg_loss:.6f}", 
                    f"{avg_merge:.6f}", 
                    f"{avg_base:.6f}", 
                    f"{avg_cov:.6f}", 
                    f"{avg_lat_mse:.6f}", 
                    f"{avg_raw_pix:.6f}", 
                    f"{avg_raw_static:.6f}", 
                    f"{avg_conf_val:.6f}", 
                    f"{avg_top1:.6f}", 
                    f"{avg_top5:.6f}"
                ])

            # [修改] 保存最新的模型到新目录
            state_dict = model.module.state_dict() if is_ddp else model.state_dict()
            torch.save(state_dict, os.path.join(ckpt_dir, "latest_model.pth"))
            
            # --- Evaluation ---
            if (epoch + 4) % 5 == 0:
                print("Evaluating...")
                model.eval()
                val_conf_sum = 0
                val_correct_trusted = 0
                val_total_trusted = 0
                val_total_pixels = 0
                val_neighbor_correct = 0 # 新增
                
                with torch.no_grad():
                    # Update Val Loop to unpack 8 items
                    for batch_data in val_loader:
                        (img, depth, act, _, tgt, 
                         _, _, _) = batch_data # Val 暂时只跑 Single Step Teacher Forcing
                         
                        img, depth, act, tgt = img.to(device), depth.to(device), act.to(device), tgt.to(device)
                        
                        d_map = depth_model(depth).unsqueeze(1)
                        d_map = (d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-6)
                        inp = torch.cat([img, d_map], dim=1)
                        
                        p_conf, p_tok = model(inp, act)
                        conf = torch.sigmoid(p_conf).squeeze(1)
                        pred = torch.argmax(p_tok, dim=1)
                        
                        correct = (pred == tgt).float()
                        
                        # Neighbor Accuracy Calculation
                        if neighbor_map:
                            # CPU check for neighbors (slow but accurate)
                            p_np = pred.cpu().numpy().flatten()
                            t_np = tgt.cpu().numpy().flatten()
                            for p_val, t_val in zip(p_np, t_np):
                                if p_val == t_val:
                                    val_neighbor_correct += 1
                                elif t_val in neighbor_map and p_val in neighbor_map[t_val]:
                                    val_neighbor_correct += 1
                        else:
                            val_neighbor_correct += correct.sum().item()

                        val_conf_sum += conf.sum().item()
                        val_total_pixels += tgt.numel()
                        
                        # Trusted Accuracy (Conf > 0.5)
                        trusted_mask = (conf > 0.5)
                        val_correct_trusted += (correct * trusted_mask.float()).sum().item()
                        val_total_trusted += trusted_mask.float().sum().item()
                
                avg_val_conf = val_conf_sum / val_total_pixels
                trusted_acc = val_correct_trusted / (val_total_trusted + 1e-6)
                trusted_prop = val_total_trusted / val_total_pixels
                neighbor_acc = val_neighbor_correct / val_total_pixels
                
                print(f"Val Epoch {epoch+1}: Avg Conf={avg_val_conf:.4f} | "
                      f"Trusted Acc={trusted_acc:.4f} (on {trusted_prop*100:.1f}%) | "
                      f"Neighbor Acc={neighbor_acc:.4f}")
                
                # [修改] 可视化输出路径指向时间戳子文件夹
                visualize_uncertainty(model, vae, depth_model, val_loader, device, 
                                      vis_base_dir, epoch+1)
                
                # --- Run Video Metrics Evaluation ---
                eval_train_eps = train_ep_names[:8]
                eval_val_eps = val_ep_names[:8]
                
                evaluate_video_metrics(model, vae, depth_model, eval_train_eps, args.image_dir, args.action_dir, args.label_dir, device, prefix="Train")
                evaluate_video_metrics(model, vae, depth_model, eval_val_eps, args.image_dir, args.action_dir, args.label_dir, device, prefix="Val")
                # ------------------------------------
                
                if trusted_acc > best_trusted_acc and trusted_prop > 0.4: 
                    best_trusted_acc = trusted_acc
                    # [修改] 保存最佳模型到新目录
                    torch.save(state_dict, os.path.join(ckpt_dir, "best_model.pth"))
                    print(f"New best trusted model saved to {ckpt_dir}!")

    if is_ddp: dist.destroy_process_group()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--action_dir", type=str, default="/data/cliang/mineworld/dataset/actions")
    # 注意：这里默认指向新的 neighbor_labels 目录
    parser.add_argument("--label_dir", type=str, default="/data/cliang/mineworld/neighbor_labels_top5")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--target_coverage", type=float, default=0.5) # 实际上这个参数在新逻辑里没用了，但保留兼容
    parser.add_argument("--cov_weight", type=float, default=100.0)
    parser.add_argument("--base_loss_weight", type=float, default=1.0)
    
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--use_neighbor_loss", action='store_true')
    

    args = parser.parse_args()
    train(args)