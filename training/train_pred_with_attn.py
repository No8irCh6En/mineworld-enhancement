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
from util.dataset_reader import MultiModalDataset, VideoEvalDataset, evaluate_and_visualize, evaluate_video_metrics
from util.attn_model import AttentionTokenPredictor
from util.neighbor_loss import NeighborConsistencyLoss

# Import VAE
try:
    from vae import VAE
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vae import VAE

# Import DepthAnything
try:
    from util.DepthAnythingWrapper import DepthAnythingWrapper, DEPTH_ANYTHING_TRANSFORM
except ImportError:
    print("Warning: Could not import DepthAnythingWrapper. Make sure util/DepthAnythingWrapper.py exists.")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DepthAnythingWrapper = None 
    DEPTH_ANYTHING_TRANSFORM = None

# --- Import MCDataset for Action Processing ---
try:
    from mcdataset import MCDataset
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mcdataset import MCDataset



# --- Training Loop ---
def train(args):
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

    if is_master: print("Loading VAE & DepthAnything...")
    
    vae = VAE(args.vae_config, args.vae_ckpt)
    vae.to(device)
    vae.eval()
    for param in vae.parameters(): param.requires_grad = False
    
    depth_model = DepthAnythingWrapper(device, (384, 224))
    depth_model.eval()
    for param in depth_model.parameters(): param.requires_grad = False
    
    # Load Neighbor Mask
    neighbor_mask = None
    if args.use_neighbor_mask:
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        neighbor_json = os.path.join(workspace_root, "analysis_results", "shift_token_neighbors.json")
        if os.path.exists(neighbor_json):
            neighbor_mask = load_neighbor_mask(neighbor_json, device=device, top_k=30)
        else:
            if is_master: print(f"Warning: {neighbor_json} not found.")

    # --- Episode Splitting Logic ---
    all_label_files = sorted(glob.glob(os.path.join(args.label_dir, "*_labels.npy")))
    all_ep_names = [os.path.basename(f).replace("_labels.npy", "") for f in all_label_files]
    
    # Shuffle and Split
    random.seed(42)
    random.shuffle(all_ep_names)
    
    if args.max_episodes > 0:
        all_ep_names = all_ep_names[:args.max_episodes]
        
    # 90/10 Split
    split_idx = int(0.9 * len(all_ep_names))
    train_ep_names = all_ep_names[:split_idx]
    val_ep_names = all_ep_names[split_idx:]
    
    if is_master:
        print(f"Total Episodes: {len(all_ep_names)}")
        print(f"Train Episodes: {len(train_ep_names)}")
        print(f"Val Episodes: {len(val_ep_names)}")

    # Datasets
    train_set = MultiModalDataset(args.image_dir, args.action_dir, args.label_dir, specific_episodes=train_ep_names)
    val_set = MultiModalDataset(args.image_dir, args.action_dir, args.label_dir, specific_episodes=val_ep_names)
    
    if len(train_set) == 0: return

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if is_ddp else None
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # Model
    model = AttentionTokenPredictor(input_channels=4, action_dim=11, num_tokens=8192).to(device)
    
    # --- Resume from Checkpoint ---
    if args.resume_from and os.path.exists(args.resume_from):
        if is_master:
            print(f"Resuming training from {args.resume_from}...")
        state_dict = torch.load(args.resume_from, map_location=device)
        
        # Handle 'module.' prefix if loading from DDP checkpoint to non-DDP model (or vice versa)
        # Here we load into the base model before wrapping with DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    # ------------------------------

    if is_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion_ce = nn.CrossEntropyLoss(reduction='none') 
    criterion_mse = nn.MSELoss()
    criterion_soft = NeighborConsistencyLoss("analysis_results/all_token_neighbors_cosine.json", device, alpha=0.5)
    
    accumulation_steps = args.accum_steps
    best_token_acc = 0.0
    
    for epoch in range(args.epochs):
        if is_ddp: train_sampler.set_epoch(epoch)
        model.train()
        
        # --- Scheduled Sampling Probability Calculation ---
        ss_prob = 0.0
        if args.scheduled_sampling and epoch >= args.ss_start_epoch:
            # Linear ramp up: 0 -> ss_max_prob over ss_warmup_epochs
            effective_epoch = epoch - args.ss_start_epoch
            progress = min(1.0, effective_epoch / max(1, args.ss_warmup_epochs))
            progress = max(0.0, progress)
            ss_prob = args.ss_max_prob * progress
            
        if is_master:
            print(f"Epoch {epoch+1}: Scheduled Sampling Prob = {ss_prob:.4f}")
        # --------------------------------------------------

        total_loss = 0
        total_loss_cls = 0
        total_loss_ce = 0
        total_loss_mse = 0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_master else train_loader
        
        for i, (img_tensor, depth_input, action_vec, target_cls, target_token) in enumerate(iterator):
            img_tensor = img_tensor.to(device)
            depth_input = depth_input.to(device)
            action_vec = action_vec.to(device)
            target_cls = target_cls.to(device)
            target_token = target_token.to(device)
            
            # --- Scheduled Sampling: Input Reconstruction ---
            if args.scheduled_sampling and ss_prob > 0:
                # Randomly select samples in the batch to corrupt
                mask_ss = torch.rand(img_tensor.shape[0], device=device) < ss_prob
                if mask_ss.any():
                    with torch.no_grad():
                        imgs_to_recon = img_tensor[mask_ss]
                        # 1. Tokenize
                        token_ids = vae.tokenize_images(imgs_to_recon) # (N, 14, 24)
                        
                        # 2. Decode
                        codebook = vae.model.quantize.embedding.weight
                        z_q = F.embedding(token_ids, codebook).permute(0, 3, 1, 2) # (N, C, H, W)
                        z_q = vae.model.post_quant_conv(z_q)
                        recon_imgs = vae.model.decoder(z_q)
                        
                        # 3. Replace (VAE output is typically [-1, 1] like input)
                        img_tensor[mask_ss] = recon_imgs
            # ------------------------------------------------
            
            with torch.no_grad():
                depth_map = depth_model(depth_input)
                depth_map = depth_map.unsqueeze(1)
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            
            img_depth = torch.cat([img_tensor, depth_map], dim=1)
            pred_cls, pred_token_logits = model(img_depth, action_vec) 
            
            if neighbor_mask is not None:
                with torch.no_grad():
                    input_token_ids = vae.tokenize_images(img_tensor)
                    batch_mask = F.embedding(input_token_ids, neighbor_mask)
                    batch_mask.scatter_(-1, target_token.unsqueeze(-1), 1.0)
                    batch_mask = batch_mask.permute(0, 3, 1, 2)
                pred_token_logits = pred_token_logits + (1.0 - batch_mask) * -1e9

            # 1. Classification Loss (Balanced)
            pred_cls_flat = pred_cls.view(-1)
            target_cls_flat = target_cls.view(-1)
            pos_mask = (target_cls_flat > 0.5)
            neg_mask = ~pos_mask
            num_pos = pos_mask.sum()
            
            if num_pos > 0:
                neg_indices = torch.where(neg_mask)[0]
                num_neg_keep = min(len(neg_indices), int(num_pos * 1))
                perm = torch.randperm(len(neg_indices), device=device)[:num_neg_keep]
                neg_indices_keep = neg_indices[perm]
                pos_indices = torch.where(pos_mask)[0]
                keep_indices = torch.cat([pos_indices, neg_indices_keep])
                loss_cls = F.binary_cross_entropy_with_logits(pred_cls_flat[keep_indices], target_cls_flat[keep_indices])
            else:
                loss_cls = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 2. CE Loss
            loss_ce = criterion_soft(pred_token_logits, target_token)

            # 3. MSE Loss
            pred_token_logits = torch.clamp(pred_token_logits, min=-20, max=20)
            z_probs = F.softmax(pred_token_logits, dim=1) 
            codebook = vae.model.quantize.embedding.weight
            z_probs_perm = z_probs.permute(0, 2, 3, 1)
            z_soft = torch.matmul(z_probs_perm, codebook) 
            z_soft = z_soft.permute(0, 3, 1, 2) 
            z_soft = vae.model.post_quant_conv(z_soft) 
            recon_img = vae.model.decoder(z_soft)
            
            with torch.no_grad():
                z_gt = F.embedding(target_token, codebook).permute(0, 3, 1, 2)
                z_gt = vae.model.post_quant_conv(z_gt) 
                target_img_recon = vae.model.decoder(z_gt)
            
            loss_mse = criterion_mse(recon_img, target_img_recon)
            
            loss = loss_cls + loss_ce + args.reg_weight * loss_mse
            loss = loss / accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            total_loss_cls += loss_cls.item()
            total_loss_ce += loss_ce.item()
            total_loss_mse += loss_mse.item()
            
        if is_master:
            num_batches = len(train_loader)
            avg_loss = total_loss / num_batches
            avg_cls = total_loss_cls / num_batches
            avg_ce = total_loss_ce / num_batches
            avg_mse = total_loss_mse / num_batches
            
            print(f"Epoch {epoch+1}: Total={avg_loss:.4f} | Cls={avg_cls:.4f} | CE={avg_ce:.4f} | MSE={avg_mse:.4f}")
            
            os.makedirs("pred_model_attn", exist_ok=True)
            state_dict = model.module.state_dict() if is_ddp else model.state_dict()
            torch.save(state_dict, "pred_model_attn/latest_model.pth")
            
            if (epoch + 1) % 5 == 0:
                print(f"Running evaluation on validation set...")
                val_cls_acc, val_token_acc = evaluate_and_visualize(
                    model, vae, depth_model, val_loader, device, 
                    output_dir="vis_results_attn_val", epoch=epoch+1, 
                    neighbor_mask=neighbor_mask, 
                    save_images=True, max_save_count=4
                )
                
                print(f"Epoch {epoch+1} Validation: Cls Acc={val_cls_acc:.4f} | Token Acc={val_token_acc:.4f}")
                
                # --- Run Video Metrics Evaluation ---
                # Select a small subset of episodes for speed
                eval_train_eps = train_ep_names[:8]
                eval_val_eps = val_ep_names[:8]
                
                evaluate_video_metrics(model, vae, depth_model, eval_train_eps, args.image_dir, args.action_dir, args.label_dir, device, prefix="Train")
                evaluate_video_metrics(model, vae, depth_model, eval_val_eps, args.image_dir, args.action_dir, args.label_dir, device, prefix="Val")
                # ------------------------------------

                if val_token_acc > best_token_acc:
                    best_token_acc = val_token_acc
                    print(f"New best model found! Saving to pred_model_attn/best_model.pth")
                    torch.save(state_dict, "pred_model_attn/best_model.pth")

    if is_master:
        print("Training finished. Loading best model for final visualization...")
        best_model_path = "pred_model_attn/best_model.pth"
        if os.path.exists(best_model_path):
            if is_ddp:
                model.module.load_state_dict(torch.load(best_model_path))
            else:
                model.load_state_dict(torch.load(best_model_path))
            
            final_vis_dir = "/data/cliang/mineworld/final_vis_results_attn"
            print(f"Saving all validation images to {final_vis_dir}...")
            evaluate_and_visualize(
                model, vae, depth_model, val_loader, device, 
                output_dir=final_vis_dir, epoch="best", 
                neighbor_mask=neighbor_mask, 
                save_images=True, max_save_count=None
            )
            print("Done.")

    if is_ddp: dist.destroy_process_group()

# Helper function for neighbor mask
def load_neighbor_mask(json_path, num_tokens=8192, device='cuda', top_k=30):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Loading neighbor mask from {json_path} with top_k={top_k}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    mask = torch.zeros((num_tokens, num_tokens), dtype=torch.float32, device=device)
    for token_str, info in data.items():
        token_id = int(token_str)
        if token_id >= num_tokens: continue
        neighbors = info['neighbors']
        if top_k is not None and top_k > 0: neighbors = neighbors[:top_k]
        valid_neighbors = [n for n in neighbors if n < num_tokens]
        mask[token_id, valid_neighbors] = 1.0
        mask[token_id, token_id] = 1.0 
    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=-1, help="Limit number of episodes to load (randomly selected)")
    parser.add_argument("--val_samples", type=int, default=2000, help="Fixed number of validation samples")

    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--action_dir", type=str, default="/data/cliang/mineworld/dataset/actions")
    parser.add_argument("--label_dir", type=str, default="/data/cliang/mineworld/misalignment_dataset_labels")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_weight", type=float, default=10.0) 
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_neighbor_mask", action='store_true')
    
    # New arguments for Scheduled Sampling
    parser.add_argument("--scheduled_sampling", action='store_true', help="Enable Scheduled Sampling (simulated autoregressive training)")
    parser.add_argument("--ss_max_prob", type=float, default=0.8, help="Max probability of using reconstructed inputs")
    parser.add_argument("--ss_warmup_epochs", type=int, default=50, help="Epochs to reach max probability")
    parser.add_argument("--ss_start_epoch", type=int, default=250, help="Epoch to start scheduled sampling (pure teacher forcing before this)")
    
    # New argument for resuming training
    parser.add_argument("--resume_from", type=str, default="", help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    train(args)