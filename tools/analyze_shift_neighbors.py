import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vae import VAE
except ImportError:
    print("Could not import VAE. Make sure you are in the project root.")
    sys.exit(1)

def load_vae(config_path, ckpt_path, device='cuda'):
    vae = VAE(config_path, ckpt_path)
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae

def get_image_paths(dataset_dir, num_samples=1000):
    print(f"Scanning images in {dataset_dir}...")
    # Recursive search
    images = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True))
    if not images:
        print("No images found!")
        return []
    
    if len(images) > num_samples:
        # Random sample
        np.random.seed(42)
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = [images[i] for i in indices]
    
    print(f"Selected {len(images)} images for analysis.")
    return images

def preprocess_image(img_path, target_size=(224, 384)):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    
    # Normalize to [-1, 1]
    img = img.astype(np.float32) / 127.5 - 1.0
    return img

def apply_shift(img_np, dx, dy):
    """
    Apply pixel shift.
    img_np: (H, W, 3)
    dx, dy: pixel shift
    """
    h, w = img_np.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # Use BORDER_REFLECT to minimize border artifacts affecting tokens
    shifted = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return shifted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--output_path", type=str, default="analysis_results/shift_token_neighbors.json")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of images to process")
    parser.add_argument("--shift_range", type=int, default=4, help="Max pixel shift (e.g. 4 means [-4, 4])")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    device = args.device
    vae = load_vae(args.vae_config, args.vae_ckpt, device)
    
    image_paths = get_image_paths(args.dataset_dir, args.num_samples)
    
    # Sparse matrix to store counts: count_matrix[token_a][token_b] = count
    # Using dict of dicts for sparsity
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    # Shifts to test: dense grid in [-range, range]
    shifts = []
    for dx in range(-args.shift_range, args.shift_range + 1):
        for dy in range(-args.shift_range, args.shift_range + 1):
            if dx == 0 and dy == 0: continue
            shifts.append((dx, dy))
            
    print(f"Testing {len(shifts)} shift patterns per image.")
    
    # Process in batches
    batch_imgs = []
    
    pbar = tqdm(total=len(image_paths))
    
    for i, img_path in enumerate(image_paths):
        img = preprocess_image(img_path)
        if img is not None:
            batch_imgs.append(img)
            
        if len(batch_imgs) >= args.batch_size or i == len(image_paths) - 1:
            if not batch_imgs: continue
            
            # 1. Process Original Batch
            orig_tensor = torch.from_numpy(np.stack(batch_imgs)).permute(0, 3, 1, 2).to(device) # (B, 3, H, W)
            with torch.no_grad():
                orig_tokens = vae.tokenize_images(orig_tensor) # (B, h, w)
            
            orig_tokens_np = orig_tokens.cpu().numpy()
            
            # 2. Process Shifted Batches
            # To save time, we can process multiple shifts for the same batch
            # But to save VRAM, let's do one shift at a time for the batch
            
            for dx, dy in shifts:
                shifted_batch = [apply_shift(img, dx, dy) for img in batch_imgs]
                shift_tensor = torch.from_numpy(np.stack(shifted_batch)).permute(0, 3, 1, 2).to(device)
                
                with torch.no_grad():
                    shift_tokens = vae.tokenize_images(shift_tensor) # (B, h, w)
                
                shift_tokens_np = shift_tokens.cpu().numpy()
                
                # 3. Accumulate Counts
                # Flatten to (N,)
                flat_orig = orig_tokens_np.flatten()
                flat_shift = shift_tokens_np.flatten()
                
                for t_orig, t_shift in zip(flat_orig, flat_shift):
                    if t_orig != t_shift: # Only record changes? Or keep self-loops?
                        # Keeping self-loops is safer to see stability
                        cooccurrence[t_orig][t_shift] += 1
            
            batch_imgs = []
            pbar.update(args.batch_size)
            
    pbar.close()
    
    print("Aggregating results...")
    results = {}
    
    for token_id in tqdm(range(8192)):
        if token_id in cooccurrence:
            neighbors_dict = cooccurrence[token_id]
            # Sort by count descending
            sorted_neighbors = sorted(neighbors_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Take Top-K
            top_neighbors = [int(n[0]) for n in sorted_neighbors[:args.top_k]]
            # Normalize scores (counts)
            total_count = sum(neighbors_dict.values())
            scores = [float(n[1] / total_count) for n in sorted_neighbors[:args.top_k]]
            
            results[str(token_id)] = {
                "neighbors": top_neighbors,
                "scores": scores
            }
        else:
            # Token never seen or never changed
            results[str(token_id)] = {
                "neighbors": [],
                "scores": []
            }
            
    print(f"Saving to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()