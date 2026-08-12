import os
import sys
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

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

def preprocess_image(image, target_size=(224, 384)):
    # image: numpy array (H, W, C) RGB
    if image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [-1, 1]
    image = image.astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    return image

def get_tokens(vae, image_tensor):
    with torch.no_grad():
        # image_tensor: (B, C, H, W)
        indices = vae.tokenize_images(image_tensor)
    return indices

def apply_transform(image, dx, dy, angle):
    # image: numpy array (H, W, C)
    # dx, dy: pixels
    # angle: degrees
    
    h, w = image.shape[:2]
    center = (w // 2, h)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Add translation
    M[0, 2] += dx
    M[1, 2] += dy
    
    # Apply affine transform
    # Use borderReplicate to avoid black borders affecting tokens too much
    transformed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return transformed

def load_video(video_path):
    print(f"Attempting to load video from {video_path}...")
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames.")
    
    # Use tqdm if total_frames is reasonable, otherwise just iterate
    iterator = range(total_frames) if total_frames > 0 else iter(int, 1)
    
    for _ in tqdm(iterator, desc="Reading video frames", total=total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames

def load_neighbors(path):
    print(f"Loading neighbors from {path}...")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert to int keys for faster lookup
        # Structure: "0": {"neighbors": [...], "scores": [...]}
        neighbors_dict = {}
        for k, v in data.items():
            neighbors_dict[int(k)] = v['neighbors']
        return neighbors_dict
    except Exception as e:
        print(f"Error loading neighbor file: {e}")
        return None

def check_neighbor_hits(tokens_t, tokens_t1, neighbors_dict, top_k=30):
    # tokens_t, tokens_t1: numpy arrays of shape (N,)
    hits = np.zeros(len(tokens_t), dtype=bool)
    
    for i in range(len(tokens_t)):
        t = int(tokens_t[i])
        t1 = int(tokens_t1[i])
        
        if t == t1:
            hits[i] = True
            continue
            
        if t in neighbors_dict:
            # Check top K
            if t1 in neighbors_dict[t][:top_k]:
                hits[i] = True
                
    return hits

def visualize_neighbor_hits(img_t, img_t1, matches_baseline, matches_neighbor, save_path, top_k, token_shape=(14, 24)):
    H, W = img_t.shape[:2]
    grid_h, grid_w = token_shape
    
    mask_baseline = matches_baseline.reshape(grid_h, grid_w)
    mask_neighbor = matches_neighbor.reshape(grid_h, grid_w)
    
    mask_baseline_img = cv2.resize(mask_baseline.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    mask_neighbor_img = cv2.resize(mask_neighbor.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].imshow(img_t)
    axes[0, 0].set_title("Frame T")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_t1)
    axes[0, 1].set_title("Frame T+1")
    axes[0, 1].axis('off')
    
    # Baseline
    axes[1, 0].imshow(mask_baseline_img, cmap='gray')
    axes[1, 0].set_title(f"Baseline Matches ({matches_baseline.mean():.1%})")
    axes[1, 0].axis('off')
    
    # Neighbor Hits
    # Gray = No match, White = Baseline, Orange = Neighbor Hit (but not baseline)
    vis_map = np.zeros((H, W, 3), dtype=np.uint8)
    vis_map[:] = [50, 50, 50]
    
    neighbor_only = (mask_neighbor_img == 1) & (mask_baseline_img == 0)
    vis_map[neighbor_only] = [255, 165, 0] # Orange
    vis_map[mask_baseline_img == 1] = [255, 255, 255] # White
    
    axes[1, 1].imshow(vis_map)
    axes[1, 1].set_title(f"Neighbor Top-{top_k} Hits ({matches_neighbor.mean():.1%})\nOrange = Recovered by Neighbor")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_matches_detailed(img_t, img_t1, matches_baseline, matches_oracle, token_shift_params, save_path, token_shape=(14, 24)):
    # img_t, img_t1: (H, W, 3) RGB uint8
    # matches_baseline: (N,) boolean
    # matches_oracle: (N,) boolean
    # token_shift_params: (N, 3) [dx, dy, angle] for the best match
    
    H, W = img_t.shape[:2]
    grid_h, grid_w = token_shape
    
    # Reshape masks
    mask_baseline = matches_baseline.reshape(grid_h, grid_w)
    mask_oracle = matches_oracle.reshape(grid_h, grid_w)
    
    # Reshape params
    params_grid = token_shift_params.reshape(grid_h, grid_w, 3)
    dx_grid = params_grid[:, :, 0]
    dy_grid = params_grid[:, :, 1]
    
    # Upscale masks for overlay
    mask_baseline_img = cv2.resize(mask_baseline.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    mask_oracle_img = cv2.resize(mask_oracle.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    
    # --- Statistics for Title ---
    # Filter only recovered tokens (Oracle True, Baseline False)
    recovered_indices = np.where(matches_oracle & ~matches_baseline)[0]
    if len(recovered_indices) > 0:
        recovered_params = token_shift_params[recovered_indices]
        # Count unique shifts
        unique_shifts, counts = np.unique(recovered_params, axis=0, return_counts=True)
        sorted_indices = np.argsort(-counts) # Descending
        top_shifts_str = []
        for i in range(min(3, len(unique_shifts))):
            idx = sorted_indices[i]
            shift = unique_shifts[idx]
            count = counts[idx]
            top_shifts_str.append(f"[dx={shift[0]}, dy={shift[1]}, a={shift[2]}]: {count}")
        stats_title = "Top Recovered Shifts:\n" + "\n".join(top_shifts_str)
    else:
        stats_title = "No additional tokens recovered."

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Frame T
    axes[0, 0].imshow(img_t)
    axes[0, 0].set_title("Frame T (Source)")
    axes[0, 0].axis('off')
    
    # 2. Frame T+1
    axes[0, 1].imshow(img_t1)
    axes[0, 1].set_title("Frame T+1 (Target)")
    axes[0, 1].axis('off')
    
    # 3. Match Map
    # Gray = No match, White = Baseline, Green = Recovered
    vis_map = np.zeros((H, W, 3), dtype=np.uint8)
    vis_map[:] = [50, 50, 50] # Dark Gray
    
    recovered_mask = (mask_oracle_img == 1) & (mask_baseline_img == 0)
    vis_map[recovered_mask] = [0, 255, 0] # Green
    vis_map[mask_baseline_img == 1] = [255, 255, 255] # White
    
    axes[1, 0].imshow(vis_map)
    axes[1, 0].set_title(f"Matches\nBaseline: {matches_baseline.mean():.1%} -> Oracle: {matches_oracle.mean():.1%}\n(Green = Recovered)")
    axes[1, 0].axis('off')
    
    # 4. Shift Field (DX/DY Visualization)
    # Create a color map where color indicates direction and intensity indicates magnitude
    # Normalize dx, dy to 0-255 range for visualization
    # Let's use HSV: Hue = Direction, Saturation = Magnitude
    
    hsv_map = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    
    # Only visualize for Oracle matches (including baseline, as baseline is shift 0,0)
    # But baseline shift is 0,0 so it will be black/white.
    
    mag, ang = cv2.cartToPolar(dx_grid.astype(np.float32), dy_grid.astype(np.float32))
    
    # Hue: Direction (0-180 in OpenCV, but here we map angle 0-360 to 0-1)
    hsv_map[..., 0] = ang * 180 / np.pi / 2 # 0-180
    
    # Saturation: Magnitude. Max shift is usually small (e.g. 4). Normalize to 0-255
    # Let's say max expected shift is 8 pixels.
    hsv_map[..., 1] = np.clip(mag * 30, 0, 255) # Scale up magnitude
    
    # Value: 255 if matched, 0 if not matched
    hsv_map[..., 2] = np.where(mask_oracle, 255, 0)
    
    hsv_map = hsv_map.astype(np.uint8)
    bgr_flow = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2RGB)
    bgr_flow_upscaled = cv2.resize(bgr_flow, (W, H), interpolation=cv2.INTER_NEAREST)
    
    axes[1, 1].imshow(bgr_flow_upscaled)
    axes[1, 1].set_title(f"Best Shift Field (Color=Dir, Bright=Mag)\n{stats_title}")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Verify if spatial misalignment causes token changes.")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    
    # Dataset arguments
    parser.add_argument("--dataset_dir", type=str, default="/data/cliang/mineworld/validation/small_validation/")
    parser.add_argument("--episode_name", type=str, default="clip_13")
    parser.add_argument("--output_dir", type=str, default="analysis_results/misalign/", help="Directory to save visualizations")
    
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--search_range", type=int, default=7, help="Pixel search range +/- (e.g. 4 means -4 to +4)")
    parser.add_argument("--search_step", type=int, default=2, help="Pixel search step")
    parser.add_argument("--rotation_range", type=int, default=10, help="Rotation range +/- degrees")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true", default=True, help="Save visualization image (default True)")
    
    # Neighbor arguments
    parser.add_argument("--neighbor_file", type=str, default='analysis_results/neighbor.json', help="Path to neighbor JSON file")
    parser.add_argument("--neighbor_top_k", type=int, default=30, help="Top K neighbors to check")
    
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VAE
    try:
        vae = load_vae(args.vae_config, args.vae_ckpt, device)
    except Exception as e:
        print(f"Failed to load VAE: {e}")
        return

    # Load Neighbors
    neighbors_dict = None
    if args.neighbor_file:
        neighbors_dict = load_neighbors(args.neighbor_file)

    # Load Data Logic
    frames = None
    full_path = os.path.join(args.dataset_dir, args.episode_name)
    if not os.path.exists(full_path) and os.path.exists(full_path + ".mp4"):
        full_path += ".mp4"
        
    if os.path.isfile(full_path) and full_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        frames = load_video(full_path)
            
    if not frames:
        print(f"Failed to load frames. Checked path: {full_path}")
        return

    print(f"Loaded {len(frames)} frames.")

    if len(frames) < 2:
        print("Need at least 2 frames.")
        return

    # Slice frames
    end_frame = min(args.start_frame + args.num_frames + 1, len(frames))
    frames_to_process = frames[args.start_frame : end_frame]
    
    if len(frames_to_process) < 2:
        print(f"Not enough frames in range {args.start_frame} to {end_frame}")
        return
    
    print(f"Processing {len(frames_to_process)-1} pairs of frames...")

    total_tokens = 0
    total_matches_baseline = 0
    total_matches_best = 0
    total_matches_neighbor = 0
    
    # Generate Search Grid and Sort by Magnitude (Distance from 0,0)
    # This ensures we find the "smallest" shift that works.
    dx_list = list(range(-args.search_range, args.search_range + 1, args.search_step))
    dy_list = list(range(-args.search_range, args.search_range + 1, args.search_step))
    angle_list = list(range(-args.rotation_range, args.rotation_range + 1, 1)) if args.rotation_range > 0 else [0]

    shifts = []
    for angle in angle_list:
        for dy in dy_list:
            for dx in dx_list:
                # Weight angle slightly more to prefer translation over rotation if ambiguous? 
                # Or just treat 1 degree ~ 1 pixel.
                dist = dx**2 + dy**2 + abs(angle)*0.1 
                shifts.append({'dx': dx, 'dy': dy, 'angle': angle, 'dist': dist})
    
    # Sort shifts by distance so argmax picks the smallest shift later
    shifts.sort(key=lambda x: x['dist'])
    
    print(f"Total combinations per frame: {len(shifts)}")

    for i in tqdm(range(len(frames_to_process) - 1)):
        img_t = frames_to_process[i]
        img_t1 = frames_to_process[i+1]
        
        # Target Tokens (Frame T+1)
        tensor_t1 = preprocess_image(img_t1).to(device)
        tokens_t1 = get_tokens(vae, tensor_t1).cpu().numpy().flatten() # (N,)
        
        # Baseline Tokens (Frame T)
        tensor_t = preprocess_image(img_t).to(device)
        tokens_t = get_tokens(vae, tensor_t).cpu().numpy().flatten() # (N,)
        
        matches_baseline = (tokens_t == tokens_t1)
        
        # --- Neighbor Check ---
        matches_neighbor = None
        if neighbors_dict:
            matches_neighbor = check_neighbor_hits(tokens_t, tokens_t1, neighbors_dict, args.neighbor_top_k)
            total_matches_neighbor += matches_neighbor.sum()
            
            if args.visualize:
                vis_filename_nb = f"{os.path.splitext(os.path.basename(args.episode_name))[0]}_frame_{args.start_frame + i}_{args.start_frame + i+1}_neighbor.png"
                vis_path_nb = os.path.join(args.output_dir, vis_filename_nb)
                visualize_neighbor_hits(img_t, img_t1, matches_baseline, matches_neighbor, vis_path_nb, args.neighbor_top_k)
        
        # --- Spatial Search ---
        shifted_tokens_map = [] 
        shift_params_list = []

        for s in shifts:
            dx, dy, angle = s['dx'], s['dy'], s['angle']
            
            if dx == 0 and dy == 0 and angle == 0:
                shifted_tokens = tokens_t
            else:
                img_shifted = apply_transform(img_t, dx, dy, angle)
                tensor_shifted = preprocess_image(img_shifted).to(device)
                shifted_tokens = get_tokens(vae, tensor_shifted).cpu().numpy().flatten()
            
            shifted_tokens_map.append(shifted_tokens)
            shift_params_list.append([dx, dy, angle])
        
        shifted_tokens_map = np.array(shifted_tokens_map) # (NumShifts, N)
        shift_params_arr = np.array(shift_params_list)    # (NumShifts, 3)
        
        # Best Local Shift (Oracle)
        matches_mask = (shifted_tokens_map == tokens_t1) # (NumShifts, N)
        has_match_any_shift = matches_mask.any(axis=0)   # (N,)
        
        # Find index of best match (first True in sorted shifts)
        # argmax returns 0 if no True found, so we must mask it
        best_match_indices = matches_mask.argmax(axis=0) # (N,)
        
        # Extract params for each token
        token_shift_params = np.zeros((len(tokens_t1), 3), dtype=int) # (N, 3)
        token_shift_params[has_match_any_shift] = shift_params_arr[best_match_indices[has_match_any_shift]]
        
        best_local_matches = has_match_any_shift.sum()
        
        # Visualization (Spatial)
        if args.visualize:
            vis_filename = f"{os.path.splitext(os.path.basename(args.episode_name))[0]}_frame_{args.start_frame + i}_{args.start_frame + i+1}.png"
            vis_path = os.path.join(args.output_dir, vis_filename)
            visualize_matches_detailed(img_t, img_t1, matches_baseline, has_match_any_shift, token_shift_params, vis_path)

        # Stats
        n_tokens = len(tokens_t1)
        total_tokens += n_tokens
        total_matches_baseline += matches_baseline.sum()
        total_matches_best += best_local_matches
        
        # Print per-frame stats
        print(f"Frame {args.start_frame + i}: Baseline={matches_baseline.mean():.1%}, Oracle={has_match_any_shift.mean():.1%}", end="")
        if matches_neighbor is not None:
            print(f", Neighbor={matches_neighbor.mean():.1%}")
        else:
            print()
        
    print("="*30)
    print(f"Overall Stats ({len(frames_to_process)-1} pairs):")
    if total_tokens > 0:
        print(f"Baseline Match Rate: {total_matches_baseline/total_tokens:.2%}")
        print(f"Oracle Match Rate:   {total_matches_best/total_tokens:.2%}")
        print(f"Improvement (Oracle):{(total_matches_best - total_matches_baseline)/total_tokens:.2%}")
        if neighbors_dict:
            print(f"Neighbor Top-{args.neighbor_top_k} Match Rate: {total_matches_neighbor/total_tokens:.2%}")
            print(f"Improvement (Neighbor): {(total_matches_neighbor - total_matches_baseline)/total_tokens:.2%}")

if __name__ == "__main__":
    main()