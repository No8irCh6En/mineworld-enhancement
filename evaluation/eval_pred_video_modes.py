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
import random # [新增] 导入 random
from tqdm import tqdm
from collections import defaultdict

# --- 1. 路径设置与依赖导入 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common_metrics_on_video_quality")
sys.path.append(METRICS_DIR)

# 导入 common_metrics 中的计算函数
try:
    from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
    from common_metrics_on_video_quality.calculate_lpips import calculate_lpips
    from common_metrics_on_video_quality.calculate_ssim import calculate_ssim_parallel
    from common_metrics_on_video_quality.calculate_psnr import calculate_psnr
    print("[INFO] Successfully imported video metrics functions.")
except ImportError as e:
    print(f"[ERROR] Failed to import metrics: {e}")
    sys.exit(1)

try:
    from vae import VAE
except ImportError:
    print("[ERROR] Could not import VAE.")

try:
    from util.DepthAnythingWrapper import DepthAnythingWrapper, DEPTH_ANYTHING_TRANSFORM
except ImportError:
    print("[WARNING] Could not import DepthAnythingWrapper.")
    DepthAnythingWrapper = None 
    DEPTH_ANYTHING_TRANSFORM = None

try:
    from mcdataset import MCDataset
except ImportError:
    print("[ERROR] Could not import MCDataset.")

# --- 导入模型定义 ---
try:
    from util.attn_model import AttentionTokenPredictor
    print("[INFO] Successfully imported AttentionTokenPredictor from util.attn_model.")
except ImportError:
    try:
        from train_pred_with_attn import AttentionTokenPredictor
        print("[INFO] Successfully imported AttentionTokenPredictor from train_pred_with_attn.")
    except ImportError:
        print("[ERROR] Could not import AttentionTokenPredictor.")
        sys.exit(1)

# --- 2. 数据集定义 ---
class MultiModalDataset(Dataset):
    def __init__(self, image_dir, action_dir, label_dir):
        self.samples = []
        self.mc_helper = MCDataset() 
        
        print(f"[INFO] Scanning dataset...")
        label_files = sorted(glob.glob(os.path.join(label_dir, "*_labels.npy")))
        
        # Group by Episode
        self.episodes = defaultdict(list)

        for lf in label_files:
            ep_name = os.path.basename(lf).replace("_labels.npy", "")
            act_file = os.path.join(action_dir, ep_name, "action.jsonl")
            ep_actions = []
            
            # Load Actions
            if os.path.exists(act_file):
                try:
                    with open(act_file, 'r') as f:
                        for line in f:
                            if not line.strip(): continue # Skip empty lines
                            try:
                                json_action = json.loads(line)
                                
                                # [强壮的判断逻辑]
                                # 1. 已经是处理好的 Env Action ?
                                if 'forward' in json_action and 'camera' in json_action and 'keyboard' not in json_action:
                                    env_action = json_action
                                # 2. 是 Raw Action ?
                                else:
                                    env_action, _ = self.mc_helper.json_action_to_env_action(json_action)

                                vec = [
                                    float(env_action.get('forward', 0)), float(env_action.get('back', 0)),
                                    float(env_action.get('left', 0)), float(env_action.get('right', 0)),
                                    float(env_action.get('jump', 0)), float(env_action.get('sneak', 0)),
                                    float(env_action.get('sprint', 0)),
                                    float(env_action.get('camera', [0, 0])[0]), float(env_action.get('camera', [0, 0])[1]),
                                    float(env_action.get('attack', 0)), float(env_action.get('use', 0))
                                ]
                                ep_actions.append(np.array(vec, dtype=np.float32))
                            except Exception as parse_err:
                                print(f"Warning: Failed to parse action line in {ep_name}: {parse_err}")
                                # 如果解析失败，塞一个全0，保证长度一致性（或者直接break）
                                # 为了鲁棒性，这里 append 0
                                ep_actions.append(np.zeros(11, dtype=np.float32))

                except Exception as e: 
                    print(f"File read error {e}")
                    pass
            
            labels = np.load(lf) 
            img_folder = os.path.join(image_dir, ep_name)
            if not os.path.isdir(img_folder): continue
            
            # 使用自然排序
            import re
            images = glob.glob(os.path.join(img_folder, "image_*.png"))
            if not images: images = glob.glob(os.path.join(img_folder, "*.png"))
            images.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', os.path.basename(x))])
            
            valid_len = min(len(images), len(labels))
            
            for i in range(valid_len):
                act = ep_actions[i] if i < len(ep_actions) else np.zeros(11, dtype=np.float32)
                sample = {
                    "img_path": images[i],
                    "action": act,
                    "label": labels[i],
                    "ep_name": ep_name,
                    "frame_idx": i
                }
                self.samples.append(sample)
                self.episodes[ep_name].append(sample)
        
        # Sort frames within episodes
        for ep in self.episodes:
            self.episodes[ep].sort(key=lambda x: x['frame_idx'])

        print(f"[INFO] Found {len(self.episodes)} episodes, Total frames: {len(self.samples)}")

    def load_image_tensor(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] != 224 or img_rgb.shape[1] != 384:
            img_rgb = cv2.resize(img_rgb, (384, 224))
        img_norm = img_rgb.astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1) # (3, H, W)
        return img_tensor

# --- 3. 核心评估逻辑 ---

# [新增] 辅助函数：计算每帧的平均 PSNR
def calculate_per_frame_psnr_torch(pred_tensor, gt_tensor):
    """
    Inputs:
        pred_tensor: (B, T, C, H, W) range [0, 1]
        gt_tensor:   (B, T, C, H, W) range [0, 1]
    Returns:
        List[float]: Average PSNR for each time step [t0, t1, ...]
    """
    B, T, C, H, W = pred_tensor.shape
    psnrs_per_step = []
    
    # 确保在同一设备
    if pred_tensor.device != gt_tensor.device:
        pred_tensor = pred_tensor.to(gt_tensor.device)

    for t in range(T):
        p_t = pred_tensor[:, t] # (B, C, H, W)
        g_t = gt_tensor[:, t]   # (B, C, H, W)
        
        # MSE per sample: (B,)
        mse = ((p_t - g_t) ** 2).mean(dim=[1, 2, 3])
        
        # Handle 0 mse (perfect match)
        mse = torch.clamp(mse, min=1e-10)
        
        # PSNR per sample: (B,)
        # p_max = 1.0
        psnr = 10 * torch.log10(1.0 / mse)
        
        # Average over batch
        avg_psnr_t = psnr.mean().item()
        psnrs_per_step.append(avg_psnr_t)
        
    return psnrs_per_step

def get_depth_input(img_tensor, transform):
    # img_tensor: (C, H, W) in [-1, 1]
    # Transform expects [0, 1] numpy
    img_01 = (img_tensor.permute(1, 2, 0).cpu().numpy() + 1.0) * 0.5
    depth_input = transform({'image': img_01})['image']
    return torch.from_numpy(depth_input).unsqueeze(0) # (1, H, W)

def decode_tokens(vae, token_ids):
    codebook = vae.model.quantize.embedding.weight
    z = F.embedding(token_ids, codebook).permute(0, 3, 1, 2)
    decoded = vae.model.decoder(vae.model.post_quant_conv(z))
    return decoded # (B, 3, H, W) in [-1, 1]

def evaluate_video_modes(args):
    device = torch.device(args.device)
    
    # --- Load Models ---
    print("[INFO] Loading Models...")
    vae = VAE(args.vae_config, args.vae_ckpt).to(device).eval()
    depth_model = DepthAnythingWrapper(device, (384, 224))
    depth_model.eval()
    
    model = AttentionTokenPredictor(input_channels=4, action_dim=11, num_tokens=8192).to(device)
    
    # Load Checkpoint
    print(f"[INFO] Loading checkpoint from {args.model_ckpt}")
    state_dict = torch.load(args.model_ckpt, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[WARNING] Strict loading failed, trying loose loading: {e}")
        
    model.eval()

    # --- Load Data ---
    dataset = MultiModalDataset(args.image_dir, args.action_dir, args.label_dir)
    
    # 智能筛选 Episode
    required_len = args.seq_len + 1
    valid_ep_names = [ep for ep in dataset.episodes.keys() if len(dataset.episodes[ep]) >= required_len]
    
    print(f"[INFO] Found {len(valid_ep_names)} episodes with length >= {required_len}")
    
    if len(valid_ep_names) == 0:
        max_len = max([len(x) for x in dataset.episodes.values()]) if dataset.episodes else 0
        print(f"[ERROR] No episodes found long enough for seq_len={args.seq_len}. Max episode length is {max_len}.")
        return

    if args.max_episodes > 0:
        # [修改] 根据 seed 随机采样
        print(f"[INFO] Randomly selecting {args.max_episodes} episodes with seed {args.seed}...")
        random.seed(args.seed)
        # 使用 sample 进行无放回采样，且不用担心索引越界（取 min 保证安全）
        num_to_sample = min(len(valid_ep_names), args.max_episodes)
        ep_names = random.sample(valid_ep_names, num_to_sample)
    else:
        ep_names = valid_ep_names
    
    all_gt_videos = []
    all_tf_videos = [] 
    all_ar_videos = [] 
    all_tf_merged_videos = []
    all_ar_merged_videos = []
    
    save_img_dir = os.path.join(args.output_dir, "video_frames")
    os.makedirs(save_img_dir, exist_ok=True)

    print(f"[INFO] Starting Video Inference on {len(ep_names)} episodes...")

    with torch.no_grad():
        for ep_name in tqdm(ep_names, desc="Processing Episodes"):
            frames_data = dataset.episodes[ep_name]
            if len(frames_data) < args.seq_len + 1: continue 
            
            clip_data = frames_data[:args.seq_len + 1]
            
            ep_gt_frames = []
            ep_tf_frames = []
            ep_ar_frames = []
            ep_tf_merged_frames = []
            ep_ar_merged_frames = []
            
            # Initial State for AR
            curr_ar_img = dataset.load_image_tensor(clip_data[0]['img_path']).to(device).unsqueeze(0) 
            
            for t in range(args.seq_len):
                # Current GT (Input for TF)
                curr_gt_img = dataset.load_image_tensor(clip_data[t]['img_path']).to(device).unsqueeze(0)
                next_gt_img = dataset.load_image_tensor(clip_data[t+1]['img_path']).to(device).unsqueeze(0)
                action_vec = torch.from_numpy(clip_data[t]['action']).to(device).unsqueeze(0) 
                
                # --- 1. Teacher Forcing (GT_t -> Pred_t+1) ---
                depth_input_tf = get_depth_input(curr_gt_img.squeeze(0), DEPTH_ANYTHING_TRANSFORM).to(device)
                depth_map_tf = depth_model(depth_input_tf).unsqueeze(1)
                depth_map_tf = (depth_map_tf - depth_map_tf.min()) / (depth_map_tf.max() - depth_map_tf.min() + 1e-6)
                
                input_tf = torch.cat([curr_gt_img, depth_map_tf], dim=1)
                
                output_tf = model(input_tf, action_vec)
                if isinstance(output_tf, tuple):
                    pred_conf_logits_tf, pred_token_logits_tf = output_tf
                    conf_tf = torch.sigmoid(pred_conf_logits_tf) 
                else:
                    pred_token_logits_tf = output_tf
                    conf_tf = None

                pred_token_ids_tf = torch.argmax(pred_token_logits_tf, dim=1)
                pred_img_tf = decode_tokens(vae, pred_token_ids_tf) 
                
                # --- 2. Autoregressive (Pred_t -> Pred_t+1) ---
                
                # [DEBUG EXPERIMENT] 暂时使用 GT Depth 来验证 AR 生成能力
                # 如果这个版本的指标很高，说明模型本身没问题，是 DepthEstimation 拖后腿
                
                # Originally:
                depth_input_ar = get_depth_input(curr_ar_img.squeeze(0), DEPTH_ANYTHING_TRANSFORM).to(device)
                
                # [Modified]: Force use GT Depth (from curr_gt_img) even in AR mode
                # depth_input_ar = get_depth_input(curr_gt_img.squeeze(0), DEPTH_ANYTHING_TRANSFORM).to(device)
                
                depth_map_ar = depth_model(depth_input_ar).unsqueeze(1)
                depth_map_ar = (depth_map_ar - depth_map_ar.min()) / (depth_map_ar.max() - depth_map_ar.min() + 1e-6)
                
                input_ar = torch.cat([curr_ar_img, depth_map_ar], dim=1)
                
                output_ar = model(input_ar, action_vec)
                if isinstance(output_ar, tuple):
                    pred_conf_logits_ar, pred_token_logits_ar = output_ar
                    conf_ar = torch.sigmoid(pred_conf_logits_ar)
                else:
                    pred_token_logits_ar = output_ar
                    conf_ar = None

                pred_token_ids_ar = torch.argmax(pred_token_logits_ar, dim=1)
                
                # --- Modified Autoregressive Logic (Decode -> Encode -> Decode) ---
                # 1. Decode predicted tokens to temporary image
                pred_img_ar = decode_tokens(vae, pred_token_ids_ar)
                
                # # 2. Re-encode the image to get "aligned" tokens
                # # tokenize_images expects [-1, 1] input, which temp_img_ar provides
                # re_encoded_tokens_ar = vae.tokenize_images(pred_img_ar)
                
                # # 3. Ensure shape compatibility (tokenize_images might return flattened indices)
                # if re_encoded_tokens_ar.shape != pred_token_ids_ar.shape:
                #     re_encoded_tokens_ar = re_encoded_tokens_ar.view_as(pred_token_ids_ar)
                
                # # 4. Decode the re-encoded tokens to get the final image for this step
                # pred_img_ar = decode_tokens(vae, re_encoded_tokens_ar) 
                
                # --- Helper: Create Merged (Pred * Conf + Prev_Frame * (1-Conf)) ---
                def create_merged_tensor(pred_tensor, conf_tensor, prev_tensor):
                    # pred_tensor: (1, 3, H, W) in [-1, 1]
                    # prev_tensor: (1, 3, H, W) in [-1, 1] (上一帧)
                    # conf_tensor: (1, 1, H, W) in [0, 1]
                    
                    pred_01 = (pred_tensor.cpu() + 1) * 0.5
                    prev_01 = (prev_tensor.cpu() + 1) * 0.5
                    
                    if conf_tensor is None:
                        return pred_01
                    
                    conf_cpu = conf_tensor.cpu() # (1, 1, 14, 24)
                    # Resize conf to image size (使用 bilinear 插值以获得平滑的混合效果)
                    conf_resized = F.interpolate(conf_cpu, size=(224, 384), mode='bilinear', align_corners=False)
                    
                    # 核心逻辑：自信时用预测，不自信时用上一帧
                    merged = pred_01 * conf_resized + prev_01 * (1 - conf_resized)
                    return merged

                # Generate Merged Tensors
                # TF 的上一帧是 curr_gt_img
                tf_merged_tensor = create_merged_tensor(pred_img_tf, conf_tf, curr_gt_img)
                # AR 的上一帧是 curr_ar_img (注意：这里必须在更新 curr_ar_img 之前调用)
                ar_merged_tensor = create_merged_tensor(pred_img_ar, conf_ar, curr_ar_img)
                
                # Update AR state for next step
                curr_ar_img = pred_img_ar
                
                ep_gt_frames.append((next_gt_img.cpu() + 1) * 0.5)
                ep_tf_frames.append((pred_img_tf.cpu() + 1) * 0.5)
                ep_ar_frames.append((pred_img_ar.cpu() + 1) * 0.5)
                ep_tf_merged_frames.append(tf_merged_tensor)
                ep_ar_merged_frames.append(ar_merged_tensor)

                # --- Visualization ---
                if args.save_images:
                    save_path = os.path.join(save_img_dir, f"{ep_name}_t{t}.png")
                    
                    def to_vis(t): return (t.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    img_gt_vis = to_vis(ep_gt_frames[-1])
                    img_tf_vis = to_vis(ep_tf_frames[-1])
                    img_ar_vis = to_vis(ep_ar_frames[-1])
                    img_tf_merged_vis = to_vis(tf_merged_tensor)
                    img_ar_merged_vis = to_vis(ar_merged_tensor)
                    
                    # Confidence Heatmaps
                    if conf_tf is not None:
                        c_tf_raw = conf_tf.squeeze().cpu().numpy() 
                        c_tf_raw = cv2.resize(c_tf_raw, (384, 224), interpolation=cv2.INTER_NEAREST)
                        c_tf_vis = cv2.applyColorMap((c_tf_raw * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    else:
                        c_tf_vis = np.zeros_like(img_gt_vis)

                    if conf_ar is not None:
                        c_ar_raw = conf_ar.squeeze().cpu().numpy()
                        c_ar_raw = cv2.resize(c_ar_raw, (384, 224), interpolation=cv2.INTER_NEAREST)
                        c_ar_vis = cv2.applyColorMap((c_ar_raw * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    else:
                        c_ar_vis = np.zeros_like(img_gt_vis)

                    # --- Layout Construction ---
                    # Row 1: GT | TF_Pred | AR_Pred | TF_Conf
                    row1 = np.hstack([
                        cv2.cvtColor(img_gt_vis, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(img_tf_vis, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(img_ar_vis, cv2.COLOR_RGB2BGR),
                        c_tf_vis
                    ])
                    
                    # Row 2: GT | TF_Merged | AR_Merged | AR_Conf
                    row2 = np.hstack([
                        cv2.cvtColor(img_gt_vis, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(img_tf_merged_vis, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(img_ar_merged_vis, cv2.COLOR_RGB2BGR),
                        c_ar_vis
                    ])
                    
                    # Stack Rows Vertically
                    combined = np.vstack([row1, row2])
                    
                    # Add Labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.6
                    color = (255, 255, 255)
                    thick = 2
                    w = 384
                    h = 224
                    
                    # Row 1 Labels
                    cv2.putText(combined, f"GT", (10, 30), font, scale, color, thick)
                    cv2.putText(combined, f"TF Pred (Raw)", (w+10, 30), font, scale, color, thick)
                    cv2.putText(combined, f"AR Pred (Raw)", (w*2+10, 30), font, scale, color, thick)
                    cv2.putText(combined, f"TF Conf", (w*3+10, 30), font, scale, color, thick)
                    
                    # Row 2 Labels
                    cv2.putText(combined, f"GT", (10, h+30), font, scale, color, thick)
                    cv2.putText(combined, f"TF Merged", (w+10, h+30), font, scale, color, thick)
                    cv2.putText(combined, f"AR Merged", (w*2+10, h+30), font, scale, color, thick)
                    cv2.putText(combined, f"AR Conf", (w*3+10, h+30), font, scale, color, thick)
                    
                    cv2.imwrite(save_path, combined)

            all_gt_videos.append(torch.cat(ep_gt_frames, dim=0))
            all_tf_videos.append(torch.cat(ep_tf_frames, dim=0))
            all_ar_videos.append(torch.cat(ep_ar_frames, dim=0))
            all_tf_merged_videos.append(torch.cat(ep_tf_merged_frames, dim=0))
            all_ar_merged_videos.append(torch.cat(ep_ar_merged_frames, dim=0))

    if len(all_gt_videos) == 0:
        print("[ERROR] No videos were processed successfully.")
        return

    # --- 4. Calculate Video Metrics ---
    tensor_gt = torch.stack(all_gt_videos, dim=0)
    tensor_tf = torch.stack(all_tf_videos, dim=0)
    tensor_ar = torch.stack(all_ar_videos, dim=0)
    tensor_tf_merged = torch.stack(all_tf_merged_videos, dim=0)
    tensor_ar_merged = torch.stack(all_ar_merged_videos, dim=0)
    
    print(f"\n[INFO] Calculating Metrics on tensor shape: {tensor_gt.shape}")
    
    results = {}
    
    def run_metrics(name, videos_pred, videos_gt):
        print(f"--- Evaluating {name} ---")
        m = {}
        
        # [新增] 计算逐帧 PSNR
        print("Calculating Per-Frame PSNR...")
        per_frame_psnr = calculate_per_frame_psnr_torch(videos_pred, videos_gt)
        
        print("Calculating FVD...")
        m['fvd'] = calculate_fvd(videos_pred, videos_gt, device, method='styleganv', only_final=False)
        print("Calculating LPIPS...")
        m['lpips'] = calculate_lpips(videos_pred, videos_gt, device)
        print("Calculating SSIM...")
        m['ssim'] = calculate_ssim_parallel(videos_pred, videos_gt)
        print("Calculating PSNR (Global)...")
        m['psnr'] = calculate_psnr(videos_pred, videos_gt)
        
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj

        full_result = {
            "summary": {
                "fvd_mean": float(np.mean(list(m['fvd']['value'].values())) if isinstance(m['fvd']['value'], dict) else np.mean(m['fvd']['value'])),
                "lpips_mean": float(np.mean(list(m['lpips']['value']))),
                "ssim_mean": float(np.mean(list(m['ssim']['value']))),
                "psnr_mean": float(np.mean(list(m['psnr']['value']))),
                "psnr_per_frame": per_frame_psnr # [新增] 添加到 summary
            },
            "details": convert_to_serializable(m)
        }
        return full_result

    results['TeacherForcing_Raw'] = run_metrics("Teacher Forcing (Raw)", tensor_tf, tensor_gt)
    results['TeacherForcing_Merged'] = run_metrics("Teacher Forcing (Merged)", tensor_tf_merged, tensor_gt)
    
    results['Autoregressive_Raw'] = run_metrics("Autoregressive (Raw)", tensor_ar, tensor_gt)
    results['Autoregressive_Merged'] = run_metrics("Autoregressive (Merged)", tensor_ar_merged, tensor_gt)
    
    print("\n" + "="*50)
    print("       VIDEO EVALUATION REPORT       ")
    print("="*50)
    
    for mode, res in results.items():
        print(f"[{mode}]")
        print(f"  FVD   : {res['summary']['fvd_mean']:.4f}")
        print(f"  LPIPS : {res['summary']['lpips_mean']:.4f}")
        print(f"  SSIM  : {res['summary']['ssim_mean']:.4f}")
        print(f"  PSNR  : {res['summary']['psnr_mean']:.4f}")
        
        # [新增] 打印逐帧趋势
        print("  Per-Frame PSNR Trend:")
        psnr_trend = res['summary']['psnr_per_frame']
        # 每行打印 5 个
        for i in range(0, len(psnr_trend), 5):
            chunk = psnr_trend[i:i+5]
            chunk_str = " | ".join([f"T{i+j+1}: {val:.2f}" for j, val in enumerate(chunk)])
            print(f"    {chunk_str}")
            
        print("-" * 30)

    out_file = os.path.join(args.output_dir, "video_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--action_dir", type=str, default="/data/cliang/mineworld/dataset/actions")
    parser.add_argument("--label_dir", type=str, default="/data/cliang/mineworld/misalignment_dataset_labels")
    parser.add_argument("--model_ckpt", type=str, default="pred_model_attn/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="eval_results_video")
    
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_episodes", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--save_images", action='store_true', help="Save visualized frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode selection") # [新增] Seed 参数

    args = parser.parse_args()
    evaluate_video_modes(args)