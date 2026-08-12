import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def plot_combined_map(ax, cat_map, val_map, title):
    H, W = cat_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.float32)
    
    # 1: Green (Primary)
    color_img[cat_map == 1] = [0.6, 1.0, 0.6]
    # 2: Blue (Secondary)
    color_img[cat_map == 2] = [0.6, 0.8, 1.0] # 浅蓝色
    # 3: Red (Miss)
    color_img[cat_map == 3] = [1.0, 0.6, 0.6]
    
    ax.imshow(color_img)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    font_size = 6
    if H > 20 or W > 30: font_size = 4
    
    for r in range(H):
        for c in range(W):
            if cat_map[r, c] != 3: # 只有命中的才显示数字
                ax.text(c, r, str(val_map[r, c]), ha='center', va='center', 
                        color='black', fontsize=font_size, fontweight='bold')

def plot_rank_map_with_text(ax, rank_map, title):
    """
    绘制 Rank Map，并在格子里填数字
    """
    H, W = rank_map.shape
    
    # Hit (Green)
    hit_mask = (rank_map <= 30)
    color_img = np.zeros((H, W, 3), dtype=np.float32)
    color_img[hit_mask] = [0.6, 1.0, 0.6] # 浅绿色背景
    
    # Self (Darker Green)
    self_mask = (rank_map == 0)
    color_img[self_mask] = [0.2, 0.8, 0.2] 
    
    # Miss (Red)
    miss_mask = (rank_map == 31)
    color_img[miss_mask] = [1.0, 0.6, 0.6] # 浅红色背景
    
    ax.imshow(color_img)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    font_size = 6
    if H > 20 or W > 30:
        font_size = 4 
        
    for r in range(H):
        for c in range(W):
            val = rank_map[r, c]
            if val <= 30:
                text_color = 'black'
                text_str = str(val)
                ax.text(c, r, text_str, ha='center', va='center', 
                        color=text_color, fontsize=font_size, fontweight='bold')

def plot_swap_experiment(img_recon_orig, img_recon_swap, swap_mask, i, save_dir, stats_text):
    """
    绘制替换实验对比图
    """
    H_tok, W_tok = swap_mask.shape
    
    fig_swap, ax_swap = plt.subplots(1, 3, figsize=(18, 6))
    fig_swap.suptitle(f"Frame {i+1} Reconstruction using History (T, T-1) [MSE Neighbors]\n{stats_text}", fontsize=14)
    
    ax_swap[0].imshow(img_recon_orig)
    ax_swap[0].set_title("Original Frame T+1")
    ax_swap[0].axis('off')
    
    ax_swap[1].imshow(img_recon_swap)
    ax_swap[1].set_title(f"Reconstructed with History Neighbors")
    ax_swap[1].axis('off')
    
    # 差异图 + Mask
    diff = cv2.absdiff(img_recon_orig, img_recon_swap)
    diff = np.clip(diff * 3, 0, 255) # 增强对比
    ax_swap[2].imshow(diff)
    
    # 叠加颜色 Mask
    mask_overlay = np.zeros((H_tok, W_tok, 4))
    mask_overlay[swap_mask == 1] = [0, 1, 0, 0.2] # Green tint
    mask_overlay[swap_mask == 2] = [0, 0, 1, 0.2] # Blue tint
    mask_overlay[swap_mask == 0] = [1, 0, 0, 0.2] # Red tint (Miss)
    
    mask_overlay_resized = cv2.resize(mask_overlay, (img_recon_orig.shape[1], img_recon_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    ax_swap[2].imshow(mask_overlay_resized)
    ax_swap[2].set_title("Diff (Enhanced) + Source Mask (G:T, B:T-1, R:Miss)")
    ax_swap[2].axis('off')
    
    plt.tight_layout()
    swap_save_name = os.path.join(save_dir, f"swap_experiment_{i:03d}.png")
    plt.savefig(swap_save_name)
    plt.close(fig_swap)
    print(f"Saved Swap Experiment to {swap_save_name}")