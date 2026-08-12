import os
import sys
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add tools/ directory to sys.path to allow importing from analysis
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Add project root to sys.path
sys.path.append(os.path.dirname(current_dir))

try:
    from vae import VAE
except ImportError:
    print("Could not import VAE. Make sure you are running this script from the project root or tools directory.")
    sys.exit(1)

try:
    from util.DepthAnythingWrapper import DepthAnythingWrapper, DEPTH_ANYTHING_TRANSFORM
except ImportError:
    print("Could not import DepthAnythingWrapper. Check your path.")
    sys.exit(1)

def load_vae(config_path, ckpt_path, device='cuda'):
    print(f"Loading VAE from {config_path} and {ckpt_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"VAE config not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")
        
    vae = VAE(config_path, ckpt_path)
    vae.to(device)
    vae.eval()
    return vae

def load_depth_model(device='cuda', img_dims=(384, 224)):
    print("Loading DepthAnything model...")
    model = DepthAnythingWrapper(device, img_dims)
    return model

def preprocess_image_vae(image, target_size=(224, 384)):
    # image: numpy array (H, W, C) RGB
    if image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [-1, 1]
    image = image.astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    return image

def preprocess_image_depth(image, device):
    # image: numpy array (H, W, C) RGB, uint8
    # Normalize to 0-1
    image = image.astype(np.float32) / 255.0
    # Transform
    sample = DEPTH_ANYTHING_TRANSFORM({'image': image})
    tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(device)
    return tensor

def get_tokens(vae, image_tensor):
    with torch.no_grad():
        indices = vae.tokenize_images(image_tensor)
    return indices

def get_depth(depth_model, image_tensor):
    with torch.no_grad():
        depth = depth_model(image_tensor) # (1, 1, H, W)
    return depth

def apply_depth_aware_transform(image, depth_map, sx, sy, angle):
    """
    Apply a transform where translation is proportional to depth.
    sx, sy: Maximum shift (at depth=1.0)
    angle: Global rotation
    """
    h, w = image.shape[:2]
    
    # 1. Create the displacement map based on depth
    grid_y, grid_x = np.indices((h, w), dtype=np.float32)
    
    shift_x = sx * depth_map
    shift_y = sy * depth_map
    
    map_x = grid_x - shift_x
    map_y = grid_y - shift_y
    
    # Apply depth-based warping
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # 2. Apply Global Rotation
    if angle != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        warped = cv2.warpAffine(warped, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
    return warped

def load_video(video_path):
    print(f"Attempting to load video from {video_path}...")
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    iterator = range(total_frames) if total_frames > 0 else iter(int, 1)
    for _ in tqdm(iterator, desc="Reading video frames", total=total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames

def visualize_comparison(img_t, img_t1, depth_map, matches_baseline, matches_best, best_params, save_path):
    H, W = img_t.shape[:2]
    
    # Normalize depth for vis
    depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Inputs
    axes[0, 0].imshow(img_t)
    axes[0, 0].set_title("Frame T (Source)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_t1)
    axes[0, 1].set_title("Frame T+1 (Target)")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(depth_vis)
    axes[0, 2].set_title("Depth Map (Frame T)")
    axes[0, 2].axis('off')
    
    # Row 2: Results
    # Baseline Matches
    vis_base = np.zeros((14, 24, 3), dtype=np.uint8) + 50
    vis_base[matches_baseline.reshape(14, 24)] = [0, 255, 0]
    vis_base = cv2.resize(vis_base, (W, H), interpolation=cv2.INTER_NEAREST)
    
    axes[1, 0].imshow(vis_base)
    axes[1, 0].set_title(f"Baseline (No Shift)\nMatch Rate: {matches_baseline.mean():.1%}")
    axes[1, 0].axis('off')
    
    # Best Warped Matches
    vis_best = np.zeros((14, 24, 3), dtype=np.uint8) + 50
    vis_best[matches_best.reshape(14, 24)] = [0, 255, 0]
    vis_best = cv2.resize(vis_best, (W, H), interpolation=cv2.INTER_NEAREST)
    
    sx, sy, angle = best_params
    axes[1, 1].imshow(vis_best)
    axes[1, 1].set_title(f"Depth-Aware Warp\nSx={sx}, Sy={sy}, Rot={angle}\nMatch Rate: {matches_best.mean():.1%}")
    axes[1, 1].axis('off')
    
    # Difference / Improvement
    diff = matches_best.astype(int) - matches_baseline.astype(int)
    diff = diff.reshape(14, 24) # Reshape to match grid dimensions (Fix here)
    
    # 1 (Green): Gained match, -1 (Red): Lost match, 0 (Gray): No change
    vis_diff = np.zeros((14, 24, 3), dtype=np.uint8) + 128
    vis_diff[diff == 1] = [0, 255, 0]   # Improved
    vis_diff[diff == -1] = [255, 0, 0]  # Worsened
    vis_diff = cv2.resize(vis_diff, (W, H), interpolation=cv2.INTER_NEAREST)
    
    axes[1, 2].imshow(vis_diff)
    axes[1, 2].set_title("Improvement Map\n(Green=Fixed, Red=Broken)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze misalignment using Depth-Aware Warping.")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    
    parser.add_argument("--dataset_dir", type=str, default="/data/cliang/mineworld/validation/small_validation/")
    parser.add_argument("--episode_name", type=str, default="clip_13")
    parser.add_argument("--output_dir", type=str, default="analysis_results/depth_warp/", help="Directory to save visualizations")
    
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=5)
    
    # Search parameters for Parallax Vector (Sx, Sy)
    # Shift = Sx * normalized_depth
    parser.add_argument("--search_range", type=int, default=12, help="Max pixel shift range for depth=1")
    parser.add_argument("--search_step", type=int, default=2)
    parser.add_argument("--rotation_range", type=int, default=2)
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Models
    try:
        vae = load_vae(args.vae_config, args.vae_ckpt, device)
        depth_model = load_depth_model(device, img_dims=(384, 224))
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # Load Data
    frames = None
    full_path = os.path.join(args.dataset_dir, args.episode_name)
    if not os.path.exists(full_path) and os.path.exists(full_path + ".mp4"):
        full_path += ".mp4"
        
    if os.path.isfile(full_path) and full_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        frames = load_video(full_path)
            
    if not frames:
        print(f"Failed to load frames. Checked path: {full_path}")
        return

    # Slice frames
    end_frame = min(args.start_frame + args.num_frames + 1, len(frames))
    frames_to_process = frames[args.start_frame : end_frame]
    
    if len(frames_to_process) < 2:
        print("Not enough frames.")
        return
    
    # Search Grid
    # sx, sy represent the shift at the closest point (depth=1.0)
    sx_list = list(range(-args.search_range, args.search_range + 1, args.search_step))
    sy_list = list(range(-args.search_range, args.search_range + 1, args.search_step))
    angle_list = list(range(-args.rotation_range, args.rotation_range + 1, 1)) if args.rotation_range > 0 else [0]
    
    params_list = []
    for angle in angle_list:
        for sy in sy_list:
            for sx in sx_list:
                params_list.append((sx, sy, angle))
    
    print(f"Search space size: {len(params_list)}")

    for i in tqdm(range(len(frames_to_process) - 1)):
        img_t = frames_to_process[i]
        img_t1 = frames_to_process[i+1]
        
        # 1. Get Depth for Frame T
        tensor_depth_in = preprocess_image_depth(img_t, device)
        depth_map_full = get_depth(depth_model, tensor_depth_in).squeeze().cpu().numpy() # (H, W)
        
        # Print Depth Stats
        print(f"  Raw Depth Stats - Min: {depth_map_full.min():.4f}, Max: {depth_map_full.max():.4f}, Mean: {depth_map_full.mean():.4f}")

        # Robust Normalization using percentiles
        # This handles outliers (like infinite sky or sensor noise) better than Min-Max
        d_min = np.percentile(depth_map_full, 2)
        d_max = np.percentile(depth_map_full, 98)
        depth_norm = (depth_map_full - d_min) / (d_max - d_min + 1e-6)
        depth_norm = np.clip(depth_norm, 0, 1)
        
        # 2. Get Target Tokens
        tensor_t1 = preprocess_image_vae(img_t1).to(device)
        tokens_t1 = get_tokens(vae, tensor_t1).cpu().numpy().flatten()
        
        # Baseline Tokens (No shift)
        tensor_t = preprocess_image_vae(img_t).to(device)
        tokens_t_baseline = get_tokens(vae, tensor_t).cpu().numpy().flatten()
        matches_baseline = (tokens_t_baseline == tokens_t1)
        
        # 3. Search for Best Depth-Aware Transform
        best_score = -1
        best_params = (0, 0, 0)
        best_matches = None
        
        for sx, sy, angle in params_list:
            if sx == 0 and sy == 0 and angle == 0:
                matches = matches_baseline
            else:
                # Apply warp
                img_warped = apply_depth_aware_transform(img_t, depth_norm, sx, sy, angle)
                
                # Tokenize
                tensor_warped = preprocess_image_vae(img_warped).to(device)
                tokens_warped = get_tokens(vae, tensor_warped).cpu().numpy().flatten()
                matches = (tokens_warped == tokens_t1)
            
            score = matches.sum()
            
            if score > best_score:
                best_score = score
                best_params = (sx, sy, angle)
                best_matches = matches
        
        # Print Stats
        print(f"\nFrame {args.start_frame + i} -> {args.start_frame + i + 1}:")
        print(f"  Baseline Match Rate: {matches_baseline.mean():.2%}")
        print(f"  Best Depth-Aware Match Rate: {best_score/len(tokens_t1):.2%}")
        print(f"  Optimal Params: Sx={best_params[0]}, Sy={best_params[1]}, Angle={best_params[2]}")
        print(f"  (Sx, Sy are shifts at max depth. Pixel shift = Sx * depth_norm)")
        
        if args.visualize:
            vis_filename = f"depth_warp_{args.start_frame + i}.png"
            vis_path = os.path.join(args.output_dir, vis_filename)
            visualize_comparison(img_t, img_t1, depth_norm, matches_baseline, best_matches, best_params, vis_path)

if __name__ == "__main__":
    main()