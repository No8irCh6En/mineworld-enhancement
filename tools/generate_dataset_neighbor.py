import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import numpy as np
import cv2
import argparse
import os
import glob
import json
from tqdm import tqdm
import sys
import math
import multiprocessing

# Add workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 关键优化 1: 禁用 OpenCV 多线程，防止与 DataLoader 冲突 ---
cv2.setNumThreads(0)

try:
    from vae import VAE
except ImportError:
    print("Error: Could not import VAE. Run from workspace root.")
    sys.exit(1)

class ImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # 这里的读取是 CPU 瓶颈所在
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return torch.zeros(3, 224, 384), idx
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 直接转 Tensor，不做任何 Resize/Normalize 操作，最快速度传给 GPU
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1) 
        return img_tensor, idx

def load_neighbor_matrix(json_path, topk, device):
    """
    加载邻居 JSON 并转换为 GPU 上的布尔邻接矩阵。
    Matrix[u, v] = True 表示 v 是 u 的 Top-K 邻居之一。
    """
    print(f"Loading neighbor map from {json_path} with Top-K={topk}...")
    with open(json_path, 'r') as f:
        neighbor_dict = json.load(f)
    
    # 8192 个 Token
    num_tokens = 8192
    # 创建布尔矩阵 (8192, 8192)
    # adj_matrix[u, v] = 1 means v is in neighbor_list(u)
    adj_matrix = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    
    for k, v in neighbor_dict.items():
        src_token = int(k)
        if src_token >= num_tokens: continue
        
        # --- 修复开始：兼容不同的 JSON 结构 ---
        if isinstance(v, list):
            # Case 1: {"0": [1, 2, 3]}
            neighbors = v
        elif isinstance(v, dict) and 'neighbors' in v:
            # Case 2: {"0": {"neighbors": [1, 2, 3], "scores": [...]}}
            neighbors = v['neighbors']
        else:
            # 未知结构，跳过或打印警告
            # print(f"Warning: Unknown structure for token {k}: {type(v)}")
            continue
        # --- 修复结束 ---
        
        # 截取 Top-K
        neighbors = neighbors[:topk]
        
        # 过滤掉超出范围的 token (以防万一)
        valid_neighbors = [n for n in neighbors if n < num_tokens]
        
        if valid_neighbors:
            adj_matrix[src_token, valid_neighbors] = True
            
    return adj_matrix.to(device)

def generate_labels(args):
    # --- DDP Setup ---
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_master = (rank == 0)
    else:
        device = torch.device(args.device)
        rank = 0
        world_size = 1
        is_master = True

    if is_master:
        print(f"Loading VAE on {device} (World Size: {world_size})...")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Load Neighbor Matrix ---
    # 所有进程都需要加载这个矩阵
    adj_matrix = load_neighbor_matrix(args.neighbor_file, args.topk, device)
    
    vae = VAE(args.vae_config, args.vae_ckpt)
    vae.to(device)
    vae.eval()
    
    # Get all episodes
    all_episodes = sorted(os.listdir(args.image_dir))
    if args.max_episodes > 0:
        all_episodes = all_episodes[:args.max_episodes]
    
    # --- Distribute Episodes ---
    my_episodes = [ep for i, ep in enumerate(all_episodes) if i % world_size == rank]
    
    if is_master:
        print(f"Total Episodes: {len(all_episodes)}")
    print(f"[Rank {rank}] Processing {len(my_episodes)} episodes...")
    
    # --- 关键优化 2: 智能计算 num_workers ---
    total_cores = multiprocessing.cpu_count()
    workers_per_gpu = max(1, int(total_cores / world_size) - 2)
    workers_per_gpu = min(workers_per_gpu, 8)
    
    if rank == 0:
        print(f"Auto-configured: {workers_per_gpu} workers per GPU")

    iterator = tqdm(my_episodes, desc=f"Rank {rank}", position=rank)
    
    for ep_name in iterator:
        ep_folder = os.path.join(args.image_dir, ep_name)
        if not os.path.isdir(ep_folder): continue
        
        out_path = os.path.join(args.output_dir, f"{ep_name}_labels.npy")
        if os.path.exists(out_path) and not args.overwrite:
            continue
            
        import re
        images = glob.glob(os.path.join(ep_folder, "image_*.png"))
        if not images: images = glob.glob(os.path.join(ep_folder, "*.png"))
        images.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', os.path.basename(x))])
        
        if len(images) < 2: continue
        
        dataset = ImageDataset(images)
        
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=workers_per_gpu, 
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if len(dataset) > args.batch_size else False
        )
        
        # 我们需要收集整个 Episode 的 Token 才能做时序对比
        # 为了速度，我们尽量保持在 GPU 上，除非显存不够
        episode_tokens_list = []
        
        with torch.no_grad():
            for img_batch, _ in loader:
                img_batch = img_batch.to(device, non_blocking=True).float()
                
                # GPU Resize
                if img_batch.shape[2] != 224 or img_batch.shape[3] != 384:
                    img_batch = F.interpolate(img_batch, size=(224, 384), mode='bilinear', align_corners=False)
                
                img_batch = img_batch / 127.5 - 1.0
                
                token_ids = vae.tokenize_images(img_batch) # (B, 14, 24)
                episode_tokens_list.append(token_ids)
        
        # 合并整个 Episode 的 Token (T, 14, 24)
        full_tokens = torch.cat(episode_tokens_list, dim=0)
        num_frames = full_tokens.shape[0]
        
        if num_frames < 2: continue

        # --- Neighbor Check Logic (GPU Accelerated) ---
        # Current Tokens: T=1 to End
        curr_tokens = full_tokens[1:] # (T-1, 14, 24)
        
        # Previous Tokens: T=0 to End-1
        prev_tokens = full_tokens[:-1] # (T-1, 14, 24)
        
        # 1. Unfold Previous Tokens to get 3x3 patches
        # Input: (N, H, W) -> Need (N, 1, H, W) for unfold
        prev_tokens_padded = F.pad(prev_tokens.float().unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        # Unfold: (N, 9, H, W)
        prev_patches = F.unfold(prev_tokens_padded, kernel_size=3).view(num_frames-1, 9, 14, 24).long()
        
        # 2. Check Similarity against all 9 spatial neighbors
        # adj_matrix[u, v] -> Is v a neighbor of u?
        # u comes from prev_patches (the potential source)
        # v comes from curr_tokens (the target we want to explain)
        
        # Expand curr_tokens to match patches: (N, 1, H, W) -> (N, 9, H, W)
        curr_tokens_expanded = curr_tokens.unsqueeze(1).expand(-1, 9, -1, -1)
        
        # Lookup: result shape (N, 9, H, W) boolean
        # adj_matrix is (8192, 8192). 
        # We use advanced indexing. Flattening helps to keep dimensions straight.
        is_neighbor_check = adj_matrix[prev_patches.flatten(), curr_tokens_expanded.flatten()]
        is_neighbor_check = is_neighbor_check.view(num_frames-1, 9, 14, 24)
        
        # 3. Reduce: If ANY of the 9 neighbors is a semantic match, then it's "Similar"
        is_similar = is_neighbor_check.any(dim=1) # (T-1, 14, 24) boolean
        
        # --- Prepare Output ---
        # Channel 0: Label (1 if NOT similar/Changed, 0 if Similar/Unchanged)
        # Channel 1: Token ID
        
        out_labels = torch.zeros((num_frames - 1, 14, 24, 2), dtype=torch.int16, device=device)
        
        # Label = 1 (Changed) if NOT similar
        out_labels[:, :, :, 0] = (~is_similar).int()
        out_labels[:, :, :, 1] = curr_tokens.int()
        
        # Save to CPU numpy
        np.save(out_path, out_labels.cpu().numpy())

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--output_dir", type=str, default="/data/cliang/mineworld/neighbor_labels")
    parser.add_argument("--neighbor_file", type=str, required=True, help="Path to all_neighbors_cosine.json")
    parser.add_argument("--topk", type=int, default=5, help="Top-K neighbors to consider from the json file")
    
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--overwrite", action='store_true')
    
    args = parser.parse_args()
    generate_labels(args)