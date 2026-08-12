import os
import sys
import argparse
import torch
import numpy as np
import cv2
import glob
from tqdm import tqdm
import torch.distributed as dist

# Add tools/ directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

try:
    from vae import VAE
except ImportError:
    print("Could not import VAE.")
    sys.exit(1)

def load_vae(config_path, ckpt_path, device='cuda'):
    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError("VAE config or checkpoint not found.")
    vae = VAE(config_path, ckpt_path)
    vae.to(device)
    vae.eval()
    return vae

def preprocess_image(image, target_size=(224, 384)):
    if image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def get_tokens(vae, image_tensor):
    with torch.no_grad():
        indices = vae.tokenize_images(image_tensor)
    return indices

def apply_transform(image, dx, dy, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2) 
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    transformed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return transformed

def load_frames(dataset_dir, episode_name):
    path = os.path.join(dataset_dir, episode_name)
    if os.path.isdir(path):
        image_files = sorted(glob.glob(os.path.join(path, "image_*.png")))
        if not image_files:
            image_files = sorted(glob.glob(os.path.join(path, "*.png")))
        if not image_files: return None
        frames = []
        for img_path in image_files: 
            img = cv2.imread(img_path)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames
    return None

def process_episode_batched(frames, vae, shifts, device, save_params=False, batch_size=8):
    """
    Generate labels for an episode using batch processing.
    """
    shift_params_list = [[s['dx'], s['dy'], s['angle']] for s in shifts]
    # shift_params_arr = np.array(shift_params_list) # (S, 3)

    grid_h, grid_w = 14, 24
    num_pairs = len(frames) - 1
    episode_labels = []

    # Process in batches
    for i in range(0, num_pairs, batch_size):
        # Fix: Calculate exact batch size for this iteration to avoid mismatch at the end of the list
        current_bs = min(batch_size, num_pairs - i)
        
        batch_frames_t = frames[i : i + current_bs]
        batch_frames_t1 = frames[i+1 : i + 1 + current_bs]
        
        # 1. Get Tokens for t1 (Target)
        tensor_t1 = torch.cat([preprocess_image(img) for img in batch_frames_t1]).to(device)
        tokens_t1 = get_tokens(vae, tensor_t1).cpu().numpy() # (B, H, W)
        tokens_t1 = tokens_t1.reshape(current_bs, -1)      # (B, N)
        
        # 2. Get Tokens for t (Baseline)
        tensor_t = torch.cat([preprocess_image(img) for img in batch_frames_t]).to(device)
        tokens_t = get_tokens(vae, tensor_t).cpu().numpy() # (B, H, W)
        tokens_t = tokens_t.reshape(current_bs, -1)        # (B, N)
        
        # 3. Oracle Search
        # Store tokens for all shifts: (S, B, N)
        # Use int16 or int32 to save memory if needed, but tokens are usually int
        shifted_tokens_map = np.zeros((len(shifts), current_bs, tokens_t.shape[1]), dtype=tokens_t.dtype)
        
        for s_idx, s in enumerate(shifts):
            if s['dx'] == 0 and s['dy'] == 0 and s['angle'] == 0:
                shifted_tokens_map[s_idx] = tokens_t
            else:
                # Apply transform to batch (CPU part)
                shifted_imgs = [apply_transform(img, s['dx'], s['dy'], s['angle']) for img in batch_frames_t]
                # VAE Encode (GPU part)
                tensor_shifted = torch.cat([preprocess_image(img) for img in shifted_imgs]).to(device)
                shifted_tokens = get_tokens(vae, tensor_shifted).cpu().numpy()
                shifted_tokens_map[s_idx] = shifted_tokens.reshape(current_bs, -1)
        
        # 4. Generate Labels
        # matches_mask: (S, B, N)
        matches_mask = (shifted_tokens_map == tokens_t1[None, :, :])
        
        # Best match index (first one in sorted shifts list)
        best_indices = matches_mask.argmax(axis=0) # (B, N)
        has_match = matches_mask.any(axis=0)       # (B, N)
        is_baseline_correct = (tokens_t == tokens_t1) # (B, N)
        
        # Prepare output array
        # If save_params: (B, N, 5) -> [recoverable, dx, dy, angle, target_token]
        # Else:           (B, N, 2) -> [recoverable, target_token]
        out_dim = 5 if save_params else 2
        batch_labels = np.zeros((current_bs, tokens_t.shape[1], out_dim), dtype=np.float32)
        
        # Fill target token
        batch_labels[..., -1] = tokens_t1
        
        # Fill recoverable flag
        # Recoverable if (has_match AND NOT baseline_correct)
        # But wait, if baseline is correct, we usually mark it as 0 (no fix needed) or handle it separately.
        # The logic in original code:
        # if baseline_correct: [0, ..., target]
        # elif has_match:      [1, ..., target]
        # else:                [0, ..., target]
        
        recoverable_mask = has_match & (~is_baseline_correct)
        batch_labels[..., 0] = recoverable_mask.astype(np.float32)
        
        if save_params:
            # Fill dx, dy, angle for recoverable tokens
            # best_indices is (B, N), values are indices into shifts
            # We need to map these indices to dx, dy, angle
            
            # Create arrays of shift params
            shifts_dx = np.array([s['dx'] for s in shifts])
            shifts_dy = np.array([s['dy'] for s in shifts])
            shifts_angle = np.array([s['angle'] for s in shifts])
            
            # Use advanced indexing
            # best_indices contains the index of the shift that worked.
            # If multiple worked, argmax gives the first one (which is best because shifts is sorted by dist).
            # If none worked, argmax gives 0. But recoverable_mask filters those out.
            
            batch_dx = shifts_dx[best_indices]
            batch_dy = shifts_dy[best_indices]
            batch_angle = shifts_angle[best_indices]
            
            # Apply mask
            batch_labels[..., 1] = batch_dx * recoverable_mask
            batch_labels[..., 2] = batch_dy * recoverable_mask
            batch_labels[..., 3] = batch_angle * recoverable_mask
            
        # Reshape to grid (B, H, W, C)
        batch_labels = batch_labels.reshape(current_bs, grid_h, grid_w, out_dim)
        episode_labels.append(batch_labels)

    if not episode_labels:
        return np.array([])
        
    return np.concatenate(episode_labels, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=-1, help="Maximum number of episodes to process. -1 for all.")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--dataset_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--output_dir", type=str, default="/data/cliang/mineworld/test", help="Directory to save .npy files")
    parser.add_argument("--episode_name", type=str, default="all")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=20)
    
    # Search params
    parser.add_argument("--search_range", type=int, default=6)
    parser.add_argument("--search_step", type=int, default=2) 
    parser.add_argument("--rotation_range", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_params", action="store_true", help="If set, save dx, dy, angle in the output labels.")
    
    # Batching and DDP
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for VAE encoding")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP") # torchrun might not pass this as arg anymore
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
    parser.add_argument("--rank", type=int, default=0, help="Global rank of this process")

    args = parser.parse_args()
    
    # DDP Setup: Read from environment variables if available (torchrun standard)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

    is_distributed = args.world_size > 1 or args.local_rank != -1
    
    if is_distributed:
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            dist.init_process_group(backend="nccl")
            # args.rank and args.world_size are already set from env
        else:
            # Manual DDP launch without torchrun (fallback)
            device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        vae = load_vae(args.vae_config, args.vae_ckpt, device)
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return
    
    # Get list of episodes
    if args.episode_name == "all":
        all_items = os.listdir(args.dataset_dir)
        # Modified: Accept any directory, not just "episode_"
        episodes = sorted([d for d in all_items if os.path.isdir(os.path.join(args.dataset_dir, d))])
        
        if args.max_episodes > 0:
            episodes = episodes[:args.max_episodes]
            print(f"Limiting to first {args.max_episodes} episodes.")
    else:
        episodes = [args.episode_name]

    # Split work among ranks
    if is_distributed:
        my_episodes = episodes[args.rank::args.world_size]
        print(f"Rank {args.rank}/{args.world_size} processing {len(my_episodes)} episodes.")
    else:
        my_episodes = episodes

    # Setup Search Space
    shifts = []
    dx_list = list(range(-args.search_range, args.search_range + 1, args.search_step))
    dy_list = list(range(-args.search_range, args.search_range + 1, args.search_step))
    angle_list = list(range(-args.rotation_range, args.rotation_range + 1, 1))
    
    for angle in angle_list:
        for dy in dy_list:
            for dx in dx_list:
                dist_val = dx**2 + dy**2 + abs(angle)*0.5
                shifts.append({'dx': dx, 'dy': dy, 'angle': angle, 'dist': dist_val})
    shifts.sort(key=lambda x: x['dist'])

    if args.rank == 0:
        print(f"Search space size: {len(shifts)}")

    # Loop
    iterator = tqdm(my_episodes, desc=f"Rank {args.rank}") if args.rank == 0 else my_episodes
    
    for ep_name in iterator:
        output_file = os.path.join(args.output_dir, f"{ep_name}_labels.npy")
        if os.path.exists(output_file):
            continue

        frames = load_frames(args.dataset_dir, ep_name)
        if not frames: continue
            
        end_frame = min(len(frames), args.start_frame + args.num_frames + 1)
        frames_to_process = frames[args.start_frame : end_frame]
        
        if len(frames_to_process) < 2: continue
            
        labels = process_episode_batched(
            frames_to_process, 
            vae, 
            shifts, 
            device, 
            save_params=args.save_params,
            batch_size=args.batch_size
        )
        
        if len(labels) > 0:
            np.save(output_file, labels)

    print("Done.")

if __name__ == "__main__":
    main()