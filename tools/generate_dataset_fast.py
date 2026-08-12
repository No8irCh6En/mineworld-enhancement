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
    # 总核数 32 / 4 卡 = 8 核/卡。留 2 个核给主进程和系统，给 DataLoader 分配 6 个。
    total_cores = multiprocessing.cpu_count()
    workers_per_gpu = max(1, int(total_cores / world_size) - 2)
    # 限制上限，防止过多
    workers_per_gpu = min(workers_per_gpu, 8)
    
    if rank == 0:
        print(f"Auto-configured: {workers_per_gpu} workers per GPU (Total {workers_per_gpu * world_size} workers)")

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
        
        # --- 关键优化 3: 开启 prefetch_factor ---
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=workers_per_gpu, 
            pin_memory=True,
            prefetch_factor=2, # 每个 worker 提前读 2 个 batch
            persistent_workers=True if len(dataset) > args.batch_size else False # 避免重复创建进程
        )
        
        all_tokens = []
        
        with torch.no_grad():
            for img_batch, _ in loader:
                # non_blocking=True 加速传输
                img_batch = img_batch.to(device, non_blocking=True).float()
                
                # 在 GPU 上做 Resize (A800 做这个是秒杀)
                if img_batch.shape[2] != 224 or img_batch.shape[3] != 384:
                    img_batch = F.interpolate(img_batch, size=(224, 384), mode='bilinear', align_corners=False)
                
                img_batch = img_batch / 127.5 - 1.0
                
                token_ids = vae.tokenize_images(img_batch) 
                all_tokens.append(token_ids.cpu().numpy())
        
        full_tokens = np.concatenate(all_tokens, axis=0) 
        num_frames = full_tokens.shape[0]
        
        labels = np.zeros((num_frames - 1, 14, 24, 2), dtype=np.int16)
        labels[:, :, :, 1] = full_tokens[1:] 
        
        np.save(out_path, labels)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--output_dir", type=str, default="/data/cliang/mineworld/uncertainty_labels")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    # 建议默认 batch_size 增大
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--overwrite", action='store_true')
    
    args = parser.parse_args()
    generate_labels(args)