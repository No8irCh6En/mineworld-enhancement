import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
import random
import json
from tqdm import tqdm
import glob
import cv2
from collections import defaultdict
import argparse


# 添加路径以导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common_metrics_on_video_quality")
sys.path.append(METRICS_DIR)

from util.helper import instantiate_from_config

def load_model(config_path, ckpt_path=None):
    print(f"Loading config from {config_path}...")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}...")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    
    model.eval()
    model.cuda()
    return model

try:
    from vae import VAE
    from common_metrics_on_video_quality.calculate_lpips import calculate_lpips
    from common_metrics_on_video_quality.calculate_ssim import calculate_ssim_parallel
    from common_metrics_on_video_quality.calculate_psnr import calculate_psnr
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def compute_neighbor_map(vae, device, model, topk=5):
    """
    计算 VAE Codebook 中每个 Token 的 Top-K 邻居
    """
    print("Computing Codebook Neighbor Map...")
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "embed_tokens"):
        transformer_embed = model.model.model.embed_tokens.weight.data
    elif hasattr(model, "transformer") and hasattr(model.transformer, "model") and hasattr(model.transformer.model, "embed_tokens"):
        transformer_embed = model.transformer.model.embed_tokens.weight.data
    else:
        # Fallback to VAE codebook if transformer model not available or structure differs
        print("Warning: Could not find embed_tokens in model, falling back to VAE codebook + post_quant_conv")
        codebook = vae.model.quantize.embedding.weight.detach()
        z_q = codebook.unsqueeze(-1).unsqueeze(-1)
        z_post = vae.model.post_quant_conv(z_q)
        transformer_embed = z_post.view(z_post.shape[0], -1)

    # 只取前 8192 个 (Image Tokens)
    codebook_features = transformer_embed[:8192] # [8192, 1024]
    
    print(f"Using Feature Embedding shape: {codebook_features.shape}")

    # 归一化以计算余弦相似度
    codebook_norm = F.normalize(codebook_features, p=2, dim=1)
    
    # 计算相似度矩阵 (8192, 8192)
    sim_matrix = torch.matmul(codebook_norm, codebook_norm.T)
    
    # 获取 Top-K (包含自身，所以取 K+1)
    _, indices = torch.topk(sim_matrix, k=topk+1, dim=1)
    
    # 转为 Python 字典: {token_id: [neighbor_id1, neighbor_id2, ...]}
    # 排除自身
    neighbor_map = indices[:, 1:].cpu().numpy() # (8192, K)
    return neighbor_map

def perturb_tokens(tokens, neighbor_map, replace_prob=1.0, use_spatial_neighbor=False):
    """
    将 tokens 中的值替换为其邻居
    tokens: (B, H, W)
    neighbor_map: (8192, K) numpy array
    replace_prob: 替换概率
    use_spatial_neighbor: 如果为 True，则从 3x3 空间邻域内的 Token 的 Top-K 邻居中随机选择
    """
    B, H, W = tokens.shape
    device = tokens.device
    
    # 生成随机掩码
    mask = torch.rand(B, H, W, device=device) < replace_prob
    
    if not use_spatial_neighbor:
        # --- 原始逻辑：只替换为 Codebook 空间上的邻居 ---
        tokens_np = tokens.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        K = neighbor_map.shape[1]
        random_choice = np.random.randint(0, K, size=(B, H, W))
        
        flat_tokens = tokens_np.flatten()
        flat_choices = random_choice.flatten()
        flat_mask = mask_np.flatten()
        
        new_tokens_flat = flat_tokens.copy()
        indices_to_replace = np.where(flat_mask)[0]
        
        if len(indices_to_replace) > 0:
            tokens_to_replace = flat_tokens[indices_to_replace]
            choices_for_replace = flat_choices[indices_to_replace]
            replacements = neighbor_map[tokens_to_replace, choices_for_replace]
            new_tokens_flat[indices_to_replace] = replacements
            
        return torch.from_numpy(new_tokens_flat.reshape(B, H, W)).to(device)
    
    else:
        # --- 新逻辑：替换为 3x3 空间邻域内 Token 的 Codebook 邻居 ---
        # 1. 提取 3x3 邻域
        # tokens: (B, H, W) -> (B, 1, H, W)
        tokens_padded = F.pad(tokens.float().unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        # Unfold to get 3x3 patches: (B, 9, H, W)
        patches = F.unfold(tokens_padded, kernel_size=3).view(B, 9, H, W).long()
        
        # 2. 随机选择一个空间邻居 (0-8)
        spatial_choice = torch.randint(0, 9, (B, H, W), device=device)
        # gather 选中的空间邻居 Token ID
        chosen_spatial_tokens = torch.gather(patches, 1, spatial_choice.unsqueeze(1)).squeeze(1) # (B, H, W)
        
        # 3. 找到这个空间邻居在 Codebook 里的 Top-K 邻居
        # 转回 CPU numpy 处理查表 (因为 neighbor_map 是 numpy)
        chosen_spatial_tokens_np = chosen_spatial_tokens.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        K = neighbor_map.shape[1]
        codebook_choice = np.random.randint(0, K, size=(B, H, W))
        
        flat_spatial_tokens = chosen_spatial_tokens_np.flatten()
        flat_codebook_choices = codebook_choice.flatten()
        flat_mask = mask_np.flatten()
        
        new_tokens_flat = tokens.cpu().numpy().flatten() # 初始为原值
        indices_to_replace = np.where(flat_mask)[0]
        
        if len(indices_to_replace) > 0:
            # 拿到选中的空间邻居 Token ID
            tokens_to_replace = flat_spatial_tokens[indices_to_replace]
            # 拿到随机的 Top-K 索引
            choices_for_replace = flat_codebook_choices[indices_to_replace]
            # 查表得到最终替换值
            replacements = neighbor_map[tokens_to_replace, choices_for_replace]
            new_tokens_flat[indices_to_replace] = replacements
            
        return torch.from_numpy(new_tokens_flat.reshape(B, H, W)).to(device)

def decode_tokens(vae, token_ids):
    codebook = vae.model.quantize.embedding.weight
    z = F.embedding(token_ids, codebook).permute(0, 3, 1, 2) # (B, D, H, W)
    z = vae.model.post_quant_conv(z)
    decoded = vae.model.decoder(z)
    return decoded # [-1, 1]

def main(args):
    device = torch.device(args.device)
    
    CONFIG_PATH = "configs/modify.yaml"
    CKPT_PATH = "/data/jjli/workspace/mineworld/checkpoints/300M_16f.ckpt"
    
    # 只有在需要 transformer embedding 时才加载大模型，否则只加载 VAE 即可
    # 这里为了保持一致性，我们还是加载它
    try:
        model = load_model(CONFIG_PATH, CKPT_PATH)
    except Exception as e:
        print(f"Warning: Failed to load transformer model ({e}). Will use VAE codebook features.")
        model = None
    
    # 1. Load VAE
    print(f"Loading VAE from {args.vae_ckpt}...")
    vae = VAE(args.vae_config, args.vae_ckpt).to(device)
    vae.eval()
    
    # 2. Compute Neighbors
    neighbor_map = compute_neighbor_map(vae, device, model, topk=args.topk)
    
    # 3. Load Images
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "**/*.png"), recursive=True))
    if args.max_images > 0:
        image_files = image_files[:args.max_images]
    
    print(f"Processing {len(image_files)} images...")
    
    orig_rec_frames = []
    perturb_rec_frames = []
    
    batch_size = args.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i+batch_size]
            batch_imgs = []
            
            for img_path in batch_files:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (384, 224)) # VAE input size
                img = img.astype(np.float32) / 127.5 - 1.0
                batch_imgs.append(img)
            
            img_tensor = torch.tensor(np.array(batch_imgs)).permute(0, 3, 1, 2).to(device) # (B, 3, H, W)
            
            # Encode
            z = vae.model.encoder(img_tensor)
            z = vae.model.quant_conv(z)
            _, _, info = vae.model.quantize(z)
            
            # Get Tokens (B, H, W)
            h_dim = z.shape[2]
            w_dim = z.shape[3]
            tokens = info[2].view(img_tensor.shape[0], h_dim, w_dim)
            
            # 1. Original Reconstruction
            rec_orig = decode_tokens(vae, tokens)
            rec_orig = (rec_orig + 1.0) * 0.5 # [0, 1]
            
            # 2. Perturbed Reconstruction
            tokens_perturbed = perturb_tokens(
                tokens, 
                neighbor_map, 
                replace_prob=1.0, 
                use_spatial_neighbor=args.use_neighbor
            )
            rec_perturb = decode_tokens(vae, tokens_perturbed)
            rec_perturb = (rec_perturb + 1.0) * 0.5 # [0, 1]
            
            orig_rec_frames.append(rec_orig.cpu())
            perturb_rec_frames.append(rec_perturb.cpu())
            
            # Save visualization for first batch
            if i == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                for j in range(min(10, len(batch_files))):
                    orig_np = (rec_orig[j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pert_np = (rec_perturb[j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    combined = np.hstack([orig_np, pert_np])
                    cv2.imwrite(os.path.join(args.output_dir, f"vis_{j}.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    # Concatenate all frames
    tensor_orig = torch.cat(orig_rec_frames, dim=0).unsqueeze(1) # (N, 1, 3, H, W)
    tensor_perturb = torch.cat(perturb_rec_frames, dim=0).unsqueeze(1) # (N, 1, 3, H, W)
    
    print(f"Calculating metrics on {tensor_orig.shape[0]} frames...")
    
    results = {}
    
    # LPIPS
    print("Calculating LPIPS...")
    lpips_res = calculate_lpips(tensor_perturb, tensor_orig, device)
    results['lpips'] = np.mean(lpips_res['value'])
    
    # SSIM
    print("Calculating SSIM...")
    ssim_res = calculate_ssim_parallel(tensor_perturb, tensor_orig)
    results['ssim'] = np.mean(ssim_res['value'])
    
    # PSNR
    print("Calculating PSNR...")
    psnr_res = calculate_psnr(tensor_perturb, tensor_orig)
    results['psnr'] = np.mean(psnr_res['value'])
    
    print("\n" + "="*50)
    mode_str = "Spatial 3x3 Neighbor + Codebook Top-K" if args.use_neighbor else "Codebook Top-K Only"
    print(f"  ROBUSTNESS EVALUATION ({mode_str})")
    print("="*50)
    print(f"Comparing [Original VAE Rec] vs [Perturbed VAE Rec]")
    print(f"  LPIPS (Lower is better) : {results['lpips']:.4f}")
    print(f"  SSIM  (Higher is better): {results['ssim']:.4f}")
    print(f"  PSNR  (Higher is better): {results['psnr']:.4f}")
    print("="*50)
    
    # Save results
    with open(os.path.join(args.output_dir, "robustness_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images", help="Path to images")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--output_dir", type=str, default="eval_neighbor_robustness")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=5, help="Number of neighbors to sample from")
    parser.add_argument("--max_images", type=int, default=100, help="Limit number of images for quick test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_neighbor", action='store_true', help="If set, sample from spatial 3x3 neighbor's Top-K")
    
    args = parser.parse_args()
    main(args)