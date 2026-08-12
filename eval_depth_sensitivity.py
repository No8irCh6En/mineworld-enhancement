import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import argparse
import os
import glob
import json
import sys
from tqdm import tqdm
from collections import defaultdict

# --- 1. 路径设置 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_metrics_on_video_quality")
sys.path.append(METRICS_DIR)

try:
    from common_metrics_on_video_quality.calculate_lpips import calculate_lpips
    from vae import VAE
    from util.DepthAnythingWrapper import DepthAnythingWrapper, DEPTH_ANYTHING_TRANSFORM
    from util.attn_model import AttentionTokenPredictor
    from mcdataset import MCDataset
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# --- 2. 修改 Dataset (取三帧: Curr -> Next -> NextNext) ---
# 为了验证 AR Depth 的影响，我们需要至少三帧: 
# t=0 (Input) -> t=1 (Predict AR Image & Depth) -> t=2 (验证用 AR Depth 预测 t+2 的准确率)
class FrameTripletDataset(Dataset):
    def __init__(self, image_dir, action_dir, label_dir, max_samples=1000):
        self.samples = []
        self.mc_helper = MCDataset()
        
        label_files = sorted(glob.glob(os.path.join(label_dir, "*_labels.npy")))
        if max_samples > 0:
            import random
            random.seed(42)
            random.shuffle(label_files)
        
        count = 0
        for lf in label_files:
            if max_samples > 0 and count >= max_samples: break
            
            ep_name = os.path.basename(lf).replace("_labels.npy", "")
            act_file = os.path.join(action_dir, ep_name, "action.jsonl")
            img_folder = os.path.join(image_dir, ep_name)
            labels = np.load(lf)
            
            if not os.path.isdir(img_folder): continue
            
            # Action loading
            ep_actions = []
            if os.path.exists(act_file):
                try:
                    with open(act_file, 'r') as f:
                        for line in f:
                            json_action = json.loads(line)
                            env_action, _ = self.mc_helper.json_action_to_env_action(json_action)
                            vec = [
                                float(env_action['forward']), float(env_action['back']),
                                float(env_action['left']), float(env_action['right']),
                                float(env_action['jump']), float(env_action['sneak']),
                                float(env_action['sprint']),
                                float(env_action['camera'][0]), float(env_action['camera'][1]),
                                float(env_action['attack']), float(env_action['use'])
                            ]
                            ep_actions.append(np.array(vec, dtype=np.float32))
                except: pass

            import re
            images = glob.glob(os.path.join(img_folder, "image_*.png")) 
            if not images: images = glob.glob(os.path.join(img_folder, "*.png"))
            images.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', os.path.basename(x))])

            valid_len = min(len(images), len(labels), len(ep_actions))
            
            # 我们需要连续三帧: t, t+1, t+2
            for i in range(valid_len - 2):
                self.samples.append({
                    "img_t": images[i],
                    "img_t1": images[i+1],
                    "img_t2": images[i+2],
                    "act_t": ep_actions[i],   # Action to produce t+1
                    "act_t1": ep_actions[i+1], # Action to produce t+2
                    "label_t2": labels[i+2]   # Target Token for t+2 (验证目标)
                })
                count += 1
                if max_samples > 0 and count >= max_samples: break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        def load(p):
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (224, 384):
                img = cv2.resize(img, (384, 224))
            return torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2,0,1)

        img_t = load(item['img_t'])
        img_t1 = load(item['img_t1']) # GT intermediate
        
        act_t = torch.from_numpy(item['act_t'])
        act_t1 = torch.from_numpy(item['act_t1'])
        
        label_np = item['label_t2']
        if label_np.ndim == 3:
            token_t2 = torch.from_numpy(label_np[:, :, 1]).long()
        else:
            token_t2 = torch.zeros((14, 24), dtype=torch.long)
            
        return img_t, img_t1, act_t, act_t1, token_t2

# --- 3. 辅助函数 (增强版: 返回 Stats) ---
def get_batch_depth_map(depth_model, img_tensor, device):
    """
    严谨地使用 DepthAnything 的预处理流水线批量计算深度。
    Return: 
        depth_map_norm: (B, 1, H, W) normalized [0, 1]
        stats: dict containing raw and normalized statistics
    """
    B, C, H, W = img_tensor.shape
    
    # 1. 转换为 Numpy HWC [0, 1]
    img_np = (img_tensor.permute(0, 2, 3, 1).cpu().numpy() + 1.0) * 0.5
    
    # 2. Transform
    depth_inputs_list = []
    for i in range(B):
        processed = DEPTH_ANYTHING_TRANSFORM({'image': img_np[i]})['image'] 
        depth_inputs_list.append(torch.from_numpy(processed))
        
    # 3. Stack -> GPU
    depth_inputs = torch.stack(depth_inputs_list).to(device)
    
    # 4. Model Forward
    with torch.no_grad():
        raw_depth_map = depth_model(depth_inputs) 
        if raw_depth_map.dim() == 3:
            raw_depth_map = raw_depth_map.unsqueeze(1) # (B, 1, H, W)
            
    # --- 5. Statistics Collection (Raw) ---
    # 计算 Batch 内平均的指标
    flat_raw = raw_depth_map.view(B, -1)
    raw_stats = {
        "raw_mean": flat_raw.mean().item(),
        "raw_min": flat_raw.min(dim=1)[0].mean().item(), # Average of per-sample mins
        "raw_max": flat_raw.max(dim=1)[0].mean().item(), # Average of per-sample maxs
        "raw_std": flat_raw.std(dim=1).mean().item()
    }

    # 6. Normalize (Per Sample Min-Max)
    d_min = raw_depth_map.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    d_max = raw_depth_map.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    denom = d_max - d_min
    safe_denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
    
    depth_map_norm = (raw_depth_map - d_min) / safe_denom
    depth_map_norm = torch.where(denom < 1e-6, torch.zeros_like(depth_map_norm), depth_map_norm)

    # --- 7. Statistics Collection (Normalized) ---
    flat_norm = depth_map_norm.view(B, -1)
    norm_stats = {
        "norm_mean": flat_norm.mean().item(),
        "norm_std": flat_norm.std(dim=1).mean().item()
        # min is always 0, max is always 1 (unless flat), so no need to log
    }
    
    stats = {**raw_stats, **norm_stats}

    return depth_map_norm, stats

def decode_tokens(vae, tokens):
    codebook = vae.model.quantize.embedding.weight
    z = F.embedding(tokens, codebook).permute(0, 3, 1, 2)
    return vae.model.decoder(vae.model.post_quant_conv(z))

# --- 4. 评估逻辑 (核心修改) ---
def evaluate_sensitivity(args):
    device = torch.device(args.device)
    
    print("[INFO] Loading Models...")
    try:
        vae = VAE(args.vae_config, args.vae_ckpt).to(device).eval()
        depth_model = DepthAnythingWrapper(device, (384, 224)) 
        model = AttentionTokenPredictor(input_channels=4, action_dim=11, num_tokens=8192).to(device)
        
        state_dict = torch.load(args.model_ckpt, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return

    # 使用 Triplet Dataset
    dataset = FrameTripletDataset(args.image_dir, args.action_dir, args.label_dir, max_samples=args.num_samples)
    if len(dataset) == 0:
        print("[ERROR] Dataset empty.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"[INFO] Dataset loaded with {len(dataset)} triplets.")
    
    # 统计器
    stats = {
        "depth_metrics": {
            "mse": [],
            "l1_diff": [],
            "gt_raw_mean": [], "gt_raw_max": [], "gt_raw_std": [],
            "ar_raw_mean": [], "ar_raw_max": [], "ar_raw_std": [],
            "gt_norm_mean": [], "ar_norm_mean": []
        },
        "acc_using_GT_Depth": {"top1": [], "top5": []},   
        "acc_using_AR_Depth": {"top1": [], "top5": []}    
    }

    with torch.no_grad():
        for img_t, img_t1_gt, act_t, act_t1, token_t2 in tqdm(loader, desc="Evaluating"):
            img_t = img_t.to(device)
            img_t1_gt = img_t1_gt.to(device)
            act_t = act_t.to(device)
            act_t1 = act_t1.to(device)
            token_t2 = token_t2.to(device)
            
            # --- Step 1: Generate t+1 (AR Image) ---
            depth_t, _ = get_batch_depth_map(depth_model, img_t, device)
            input_t = torch.cat([img_t, depth_t], dim=1)
            
            # Predict t+1 Tokens
            logits_t1 = model(input_t, act_t)
            if isinstance(logits_t1, tuple): logits_t1 = logits_t1[1]
            pred_tokens_t1 = torch.argmax(logits_t1, dim=1)
            
            # Decode to Image (AR Generated Image t+1)
            img_t1_ar = decode_tokens(vae, pred_tokens_t1)
            # Clip VAE output to safe range (Crucial for DepthAnything stability)
            img_t1_ar = torch.clamp(img_t1_ar, -1.0, 1.0) 
            
            # --- Step 2: Compare Depths ---
            # 1. Depth from GT Image t+1
            depth_t1_gt, s_gt = get_batch_depth_map(depth_model, img_t1_gt, device)
            
            # 2. Depth from Predicted AR Image t+1
            depth_t1_ar, s_ar = get_batch_depth_map(depth_model, img_t1_ar, device)
            
            # 统计 Depth 差异
            mse = F.mse_loss(depth_t1_gt, depth_t1_ar).item()
            l1 = F.l1_loss(depth_t1_gt, depth_t1_ar).item()
            
            stats["depth_metrics"]["mse"].append(mse)
            stats["depth_metrics"]["l1_diff"].append(l1)
            
            # 记录分布统计
            stats["depth_metrics"]["gt_raw_mean"].append(s_gt["raw_mean"])
            stats["depth_metrics"]["gt_raw_max"].append(s_gt["raw_max"])
            stats["depth_metrics"]["gt_raw_std"].append(s_gt["raw_std"])
            stats["depth_metrics"]["gt_norm_mean"].append(s_gt["norm_mean"])
            
            stats["depth_metrics"]["ar_raw_mean"].append(s_ar["raw_mean"])
            stats["depth_metrics"]["ar_raw_max"].append(s_ar["raw_max"])
            stats["depth_metrics"]["ar_raw_std"].append(s_ar["raw_std"])
            stats["depth_metrics"]["ar_norm_mean"].append(s_ar["norm_mean"])
            
            # --- Step 3: Predict t+2 using different Depths ---
            # Case A: Input = [Img_t1_GT, Depth_t1_GT] (上限)
            input_gt = torch.cat([img_t1_gt, depth_t1_gt], dim=1)
            logits_t2_gt = model(input_gt, act_t1)
            if isinstance(logits_t2_gt, tuple): logits_t2_gt = logits_t2_gt[1]
            
            # Case C: Input = [Img_t1_GT, Depth_t1_AR] (混合: 好图 + 烂Depth)
            input_mixed = torch.cat([img_t1_gt, depth_t1_ar], dim=1) 
            logits_t2_mixed = model(input_mixed, act_t1)
            if isinstance(logits_t2_mixed, tuple): logits_t2_mixed = logits_t2_mixed[1]

            # Calculate Accuracies
            target_flat = token_t2.reshape(-1)
            
            def calc_acc(logits):
                flat = logits.permute(0, 2, 3, 1).reshape(-1, 8192)
                _, p1 = flat.max(dim=1)
                a1 = (p1 == target_flat).float().mean().item()
                _, p5 = flat.topk(5, dim=1)
                a5 = (p5 == target_flat.unsqueeze(1)).float().sum(dim=1).mean().item()
                return a1, a5

            a1_gt, a5_gt = calc_acc(logits_t2_gt)
            a1_ar, a5_ar = calc_acc(logits_t2_mixed)
            
            stats["acc_using_GT_Depth"]["top1"].append(a1_gt)
            stats["acc_using_GT_Depth"]["top5"].append(a5_gt)
            stats["acc_using_AR_Depth"]["top1"].append(a1_ar)
            stats["acc_using_AR_Depth"]["top5"].append(a5_ar)

    # --- Report ---
    dm = stats["depth_metrics"]
    
    print("\n" + "="*80)
    print(f"{'DETAILED DEPTH STATISTICS':^80}")
    print("="*80)
    print(f"{'Metric':<25} | {'GT (Ground Truth)':<20} | {'AR (Generated)':<20} | {'Diff'}")
    print("-" * 80)
    
    # helper for mean
    avg = lambda k: np.mean(dm[k])
    
    print(f"{'Raw Mean Value':<25} | {avg('gt_raw_mean'):.4f}               | {avg('ar_raw_mean'):.4f}               | {avg('ar_raw_mean')-avg('gt_raw_mean'):+.4f}")
    print(f"{'Raw Max Value':<25} | {avg('gt_raw_max'):.4f}               | {avg('ar_raw_max'):.4f}               | {avg('ar_raw_max')-avg('gt_raw_max'):+.4f}")
    print(f"{'Raw Std Dev (Contrast)':<25} | {avg('gt_raw_std'):.4f}               | {avg('ar_raw_std'):.4f}               | {avg('ar_raw_std')-avg('gt_raw_std'):+.4f}")
    print("-" * 80)
    print(f"{'Norm Mean Value [0,1]':<25} | {avg('gt_norm_mean'):.4f}               | {avg('ar_norm_mean'):.4f}               | {avg('ar_norm_mean')-avg('gt_norm_mean'):+.4f}")
    print("-" * 80)
    print(f"{'MSE (GT vs AR)':<25} | {avg('mse'):.6f}")
    print(f"{'L1 Diff (GT vs AR)':<25} | {avg('l1_diff'):.6f}")

    print("\n" + "="*80)
    print(f"{'DOWNSTREAM TASK SENSITIVITY':^80}")
    print("="*80)
    print(f"{'Condition':<30} | {'Top-1 Acc':<10} | {'Rel Drop (% of Clean)'}")
    print("-" * 80)
    
    base_acc = np.mean(stats["acc_using_GT_Depth"]["top1"])
    ar_acc = np.mean(stats["acc_using_AR_Depth"]["top1"])
    drop = (base_acc - ar_acc) / base_acc * 100
    
    print(f"{'Using GT Depth (Perfect)':<30} | {base_acc:.4f}     | -")
    print(f"{'Using AR Generated Depth':<30} | {ar_acc:.4f}     | -{drop:.2f}%")
    print("="*80)
    
    if drop > 5.0:
        print("\n[CONCLUSION] Model is SENSITIVE to depth noise (>5% drop).")
        print("Suggestion: Enhance training with depth noise augmentation.")
    else:
        print("\n[CONCLUSION] Model is ROBUST to depth noise (<5% drop).")
        print("Suggestion: Look for other causes (e.g. accumulating image blur).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径请根据你的环境修改
    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--action_dir", type=str, default="/data/cliang/mineworld/dataset/actions")
    parser.add_argument("--label_dir", type=str, default="/data/cliang/mineworld/misalignment_dataset_labels")
    parser.add_argument("--model_ckpt", type=str, default="pred_model_attn/best_model.pth")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=200) # 只测200个样本就够了，验证趋势

    args = parser.parse_args()
    evaluate_sensitivity(args)