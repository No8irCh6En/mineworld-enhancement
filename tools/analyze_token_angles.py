import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from omegaconf import OmegaConf

# 添加项目根目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.helper import instantiate_from_config

# 导入新模块
from tools.analysis.data_loader import load_dataset_episode, preprocess_frame, get_tokens_and_embeddings, parse_action
from tools.analysis.metrics import find_best_matches_rect, analyze_neighbor_hit, analyze_combined_multiframe
from tools.analysis.visualization import plot_combined_map, plot_rank_map_with_text
from tools.analysis.experiment_swap import run_swap_experiment

def load_models(config_path, ckpt_path=None):
    print(f"Loading config from {config_path}...")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}...")
        sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    model.eval()
    model.cuda()
    return model

def analyze_data(model, data_source, sem_json_path, mse_json_path, output_dir="analysis_results", episode_name="default", max_frames=5, height=224, width=384):
    # 为每个 episode 创建独立文件夹
    save_dir = os.path.join(output_dir, episode_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving analysis results to {save_dir}")
    
    # === 设置窗口大小 ===
    PAD = 2  # 修改这里: 2=5x5, 3=7x7
    WIN_SIZE = 2 * PAD + 1
    WIN_NAME = f"{WIN_SIZE}x{WIN_SIZE}"
    print(f"Using search window size: {WIN_NAME} (pad={PAD})")
    
    print(f"Loading SEMANTIC neighbor dictionary from {sem_json_path}...")
    with open(sem_json_path, 'r') as f:
        sem_dict = json.load(f)

    print(f"Loading MSE neighbor dictionary from {mse_json_path}...")
    with open(mse_json_path, 'r') as f:
        mse_dict = json.load(f)
    
    frames = data_source["frames"]
    coords = data_source.get("coords") 
    tokens_all = data_source.get("tokens") 
    actions = data_source.get("actions")

    # Determine layout
    if tokens_all is not None:
        total_tokens = tokens_all.size
        tokens_per_frame = (height // 16) * (width // 16) # 14 * 24 = 336
        
        if total_tokens % tokens_per_frame == 0:
            num_frames = total_tokens // tokens_per_frame
            tokens_all = tokens_all.reshape(num_frames, tokens_per_frame)
            print(f"Reshaped tokens to ({num_frames}, {tokens_per_frame})")
            
            H_tok = height // 16
            W_tok = width // 16
            print(f" inferred token layout: {H_tok}x{W_tok}")
        else:
            print(f"Warning: Token count {total_tokens} is not divisible by {tokens_per_frame}. Ignoring precomputed tokens.")
            tokens_all = None
            
    if tokens_all is None:
        dummy_tensor = preprocess_frame(frames[0], height=height, width=width)
        dummy_ids, dummy_emb = get_tokens_and_embeddings(model, dummy_tensor)
        seq_len = dummy_emb.shape[1]
        
        downsample_factor = 16
        expected_h = height // downsample_factor
        expected_w = width // downsample_factor
        
        if seq_len == expected_h * expected_w:
            H_tok, W_tok = expected_h, expected_w
            print(f" inferred token layout: {H_tok}x{W_tok}")
        else:
            side = int(np.sqrt(seq_len))
            H_tok, W_tok = side, side
            print(f" Fallback to square {side}x{side}")

    # Helper to get data for a frame index
    def get_frame_data(idx):
        if tokens_all is not None and idx < len(tokens_all):
            # Use precomputed tokens
            tok = torch.tensor(tokens_all[idx]).cuda()
            
            # Ensure shape (1, Seq)
            if tok.dim() == 1: 
                tok = tok.unsqueeze(0)
            
            tok = tok.long()
            # Need to import get_embeddings_from_ids from data_loader if used here, 
            # but we imported get_tokens_and_embeddings which uses it internally.
            # We need to expose get_embeddings_from_ids in data_loader.py
            from tools.analysis.data_loader import get_embeddings_from_ids
            emb = get_embeddings_from_ids(model, tok)
            return tok, emb
        else:
            # Use VAE inference
            if idx < len(frames):
                tensor = preprocess_frame(frames[idx], height=height, width=width)
                return get_tokens_and_embeddings(model, tensor)
            return None, None

    num_pairs = min(len(frames) - 1, max_frames)
    
    for i in range(num_pairs):
        frame_t = frames[i]
        frame_t1 = frames[i+1]
        
        ids_t_minus_1, _ = get_frame_data(i-1) if i > 0 else (None, None)
        ids_t_plus_2, _ = get_frame_data(i+2)
        
        ids_t, emb_t = get_frame_data(i)
        ids_t1, emb_t1 = get_frame_data(i+1)
        
        if ids_t is None or ids_t1 is None:
            print(f"Skipping frame {i} due to missing data")
            continue

        # === 执行替换实验 ===
        run_swap_experiment(model, ids_t1, ids_t, ids_t_minus_1, mse_dict, H_tok, W_tok, PAD, i, save_dir)

        # 1. 计算位移场
        best_sims, offsets = find_best_matches_rect(emb_t, emb_t1, H_tok, W_tok, pad=PAD)
        offset_y = offsets[:, 0].view(H_tok, W_tok).cpu().numpy()
        offset_x = offsets[:, 1].view(H_tok, W_tok).cpu().numpy()
        
        # 获取 Action 信息
        action_text = "No Action"
        if actions and i < len(actions):
            action_text = parse_action(actions[i])
        
        # 2. SEMANTIC Analysis
        sem_ranks, sem_hit = analyze_neighbor_hit(ids_t, ids_t1, sem_dict)
        sem_rank_map = sem_ranks.reshape(H_tok, W_tok)
        
        sem_cat_fwd, sem_val_fwd, sem_stats_fwd = analyze_combined_multiframe(ids_t, ids_t1, ids_t_plus_2, sem_dict, H_tok, W_tok, pad=PAD)
        sem_cat_bwd, sem_val_bwd, sem_stats_bwd = analyze_combined_multiframe(ids_t1, ids_t, ids_t_minus_1, sem_dict, H_tok, W_tok, pad=PAD)

        # 3. MSE Analysis
        mse_ranks, mse_hit = analyze_neighbor_hit(ids_t, ids_t1, mse_dict)
        mse_rank_map = mse_ranks.reshape(H_tok, W_tok)
        
        mse_cat_fwd, mse_val_fwd, mse_stats_fwd = analyze_combined_multiframe(ids_t, ids_t1, ids_t_plus_2, mse_dict, H_tok, W_tok, pad=PAD)
        mse_cat_bwd, mse_val_bwd, mse_stats_bwd = analyze_combined_multiframe(ids_t1, ids_t, ids_t_minus_1, mse_dict, H_tok, W_tok, pad=PAD)
        
        # --- 可视化 (3行4列) ---
        fig, axes = plt.subplots(3, 4, figsize=(24, 15))
        fig.suptitle(f"Frame {i}->{i+1} | Act: {action_text} | Sem Fwd Hit: {sem_stats_fwd['total_hit']:.1%} | MSE Fwd Hit: {mse_stats_fwd['total_hit']:.1%}", fontsize=16)
        
        # === Row 1: Basic Info & Semantic Direct ===
        axes[0, 0].imshow(frame_t); axes[0, 0].set_title("Frame T")
        axes[0, 1].imshow(frame_t1); axes[0, 1].set_title("Frame T+1")
        
        direct_sim = F.cosine_similarity(emb_t, emb_t1, dim=-1).view(H_tok, W_tok).cpu().numpy()
        im1 = axes[0, 2].imshow(direct_sim, cmap='viridis', vmin=0, vmax=1)
        axes[0, 2].set_title("Direct Cosine Sim")
        plt.colorbar(im1, ax=axes[0, 2])
        
        plot_rank_map_with_text(axes[0, 3], sem_rank_map, "SEMANTIC Neighbor Hit (Direct)")
        
        # === Row 2: Motion & Semantic Forward Combined ===
        sim_map = best_sims.view(H_tok, W_tok).cpu().numpy()
        im2 = axes[1, 0].imshow(sim_map, cmap='viridis', vmin=0, vmax=1)
        axes[1, 0].set_title(f"Best Match Sim ({WIN_NAME} Search)")
        plt.colorbar(im2, ax=axes[1, 0])
        
        axes[1, 1].set_title("Embedding Motion Field")
        axes[1, 1].imshow(cv2.cvtColor(frame_t1, cv2.COLOR_RGB2GRAY), cmap='gray', alpha=0.5, extent=[0, W_tok, H_tok, 0])
        X, Y = np.meshgrid(np.arange(W_tok), np.arange(H_tok)) 
        mask = (offset_x != 0) | (offset_y != 0)
        axes[1, 1].quiver(X[mask]+0.5, Y[mask]+0.5, offset_x[mask], offset_y[mask], color='red', scale=1, scale_units='xy', width=0.003, headwidth=3, headlength=4, headaxislength=3.5)
        
        if coords is not None and i+1 < len(coords):
            depth_t = coords[i][:, 2].reshape(H_tok, W_tok)
            im3 = axes[1, 2].imshow(depth_t, cmap='plasma')
            axes[1, 2].set_title("Depth (Ground Truth)")
            plt.colorbar(im3, ax=axes[1, 2])
        else:
            axes[1, 2].text(0.5, 0.5, "No Coords", ha='center')
            
        sem_title_fwd = f"SEM Fwd (T->T+1/T+2)\nGrn(T+1):{sem_stats_fwd['pri_rate']:.1%} Blu(T+2):{sem_stats_fwd['sec_rate']:.1%}"
        plot_combined_map(axes[1, 3], sem_cat_fwd, sem_val_fwd, sem_title_fwd)

        # === Row 3: MSE Analysis & Backward Combined ===
        sem_title_bwd = f"SEM Bwd (T+1->T/T-1)\nGrn(T):{sem_stats_bwd['pri_rate']:.1%} Blu(T-1):{sem_stats_bwd['sec_rate']:.1%}"
        plot_combined_map(axes[2, 0], sem_cat_bwd, sem_val_bwd, sem_title_bwd)
        
        mse_title_bwd = f"MSE Bwd (T+1->T/T-1)\nGrn(T):{mse_stats_bwd['pri_rate']:.1%} Blu(T-1):{mse_stats_bwd['sec_rate']:.1%}"
        plot_combined_map(axes[2, 1], mse_cat_bwd, mse_val_bwd, mse_title_bwd)
        
        plot_rank_map_with_text(axes[2, 2], mse_rank_map, "MSE (Visual) Neighbor Hit (Direct)")
        
        mse_title_fwd = f"MSE Fwd (T->T+1/T+2)\nGrn(T+1):{mse_stats_fwd['pri_rate']:.1%} Blu(T+2):{mse_stats_fwd['sec_rate']:.1%}"
        plot_combined_map(axes[2, 3], mse_cat_fwd, mse_val_fwd, mse_title_fwd)
        
        plt.tight_layout()
        save_name = os.path.join(save_dir, f"analysis_{i:03d}.png")
        plt.savefig(save_name)
        plt.close()
        print(f"Saved {save_name}")

if __name__ == "__main__":
    # === 配置 ===
    CONFIG_PATH = "configs/modify.yaml"
    CKPT_PATH = None 
    
    DATASET_DIR = "/home/cliang/mineworld/outputs_video/plain_all"
    
    # === 开关 ===
    SINGLE_EPISODE_MODE = True  # True: 只分析 EPISODE_NAME; False: 遍历 DATASET_DIR
    EPISODE_NAME = "episode_13"  # 仅在 SINGLE_EPISODE_MODE=True 时生效
    
    VIDEO_PATH = "assets/demo_video.mp4" 
    
    SEM_JSON = "analysis_results/all_token_neighbors.json" 
    MSE_JSON = "analysis_results/all_token_neighbors_mse.json" 
    
    FRAME_HEIGHT = 224
    FRAME_WIDTH = 384
    MAX_FRAMES = 12

    if not os.path.exists(SEM_JSON) or not os.path.exists(MSE_JSON):
        print(f"Error: JSON files not found. Please run analyze_token_relations.py twice (once for sem, once for mse).")
        if os.path.exists(MSE_JSON) and not os.path.exists(SEM_JSON):
            SEM_JSON = MSE_JSON
        elif os.path.exists(SEM_JSON) and not os.path.exists(MSE_JSON):
            MSE_JSON = SEM_JSON
        else:
            sys.exit(1)

    model = load_models(CONFIG_PATH, CKPT_PATH)
    
    # 确定要处理的 Episode 列表
    episodes_to_process = []
    
    if SINGLE_EPISODE_MODE:
        episodes_to_process.append(EPISODE_NAME)
    else:
        print(f"Scanning {DATASET_DIR} for episodes...")
        
        images_dir = os.path.join(DATASET_DIR, "images")
        if os.path.exists(images_dir) and os.path.isdir(images_dir):
            print(f"Found 'images' directory, scanning inside: {images_dir}")
            for item in sorted(os.listdir(images_dir)):
                if item.startswith("episode_"):
                    episodes_to_process.append(item)

        if os.path.exists(DATASET_DIR):
            for item in sorted(os.listdir(DATASET_DIR)):
                item_path = os.path.join(DATASET_DIR, item)
                if os.path.isdir(item_path):
                    if item == "analysis_results": continue
                    if item == "images": continue 
                    if item in episodes_to_process: continue 
                    
                    has_images = os.path.exists(os.path.join(item_path, "images"))
                    has_tokens = os.path.exists(os.path.join(item_path, "tokens.npy"))
                    is_clip_folder = item.startswith("clip_") or item.startswith("episode_")
                    
                    if has_images or has_tokens or is_clip_folder:
                        episodes_to_process.append(item)
        else:
            print(f"Dataset directory {DATASET_DIR} does not exist.")

    print(f"Found {len(episodes_to_process)} episodes to process: {episodes_to_process}")

    for ep_name in episodes_to_process:
        print(f"\n=== Processing {ep_name} ===")
        try:
            data_source = load_dataset_episode(DATASET_DIR, ep_name)
            
            if data_source is None and SINGLE_EPISODE_MODE:
                print("Dataset not found, falling back to video file...")
                cap = cv2.VideoCapture(VIDEO_PATH)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                data_source = {"frames": frames, "coords": None, "tokens": None, "actions": None}
            
            if data_source:
                analyze_data(model, data_source, SEM_JSON, MSE_JSON, output_dir="analysis_results", episode_name=ep_name, max_frames=MAX_FRAMES, height=FRAME_HEIGHT, width=FRAME_WIDTH)
            else:
                print(f"Skipping {ep_name}: Could not load data.")
                
        except Exception as e:
            print(f"Error processing {ep_name}: {e}")
            import traceback
            traceback.print_exc()