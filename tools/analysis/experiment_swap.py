import torch
import numpy as np
from .visualization import plot_swap_experiment

def decode_tokens(model, tokens, H, W):
    """
    将 Token IDs 解码为 RGB 图片 (numpy array, 0-255)
    tokens: [1, Seq] or [Seq]
    """
    vae = model.tokenizer
    with torch.no_grad():
        # 确保 tokens 是 LongTensor 且形状正确
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        if hasattr(vae, 'token2image'):
            # 之前的 inference.py 里的接口
            return vae.token2image(tokens.squeeze())
            
        print(f"Decode failed: Unknown VAE interface")
        return np.zeros((H*16, W*16, 3), dtype=np.uint8)

def swap_tokens_multiframe(ids_target, ids_primary, ids_secondary, neighbor_dict, H, W, pad=2):
    """
    尝试用 ids_primary (T) 或 ids_secondary (T-1) 窗口中的 Token 替换 ids_target (T+1)。
    """
    grid_target = ids_target.view(H, W).cpu().numpy()
    grid_pri = ids_primary.view(H, W).cpu().numpy()
    grid_sec = ids_secondary.view(H, W).cpu().numpy() if ids_secondary is not None else None
    
    grid_swapped = grid_target.copy()
    
    # 0: Miss, 1: Primary Hit (Green), 2: Secondary Hit (Blue)
    swap_mask = np.zeros((H, W), dtype=np.int32)
    
    # 动态 padding
    grid_pri_padded = np.pad(grid_pri, ((pad, pad), (pad, pad)), mode='constant', constant_values=-1)
    if grid_sec is not None:
        grid_sec_padded = np.pad(grid_sec, ((pad, pad), (pad, pad)), mode='constant', constant_values=-1)
        
    window_size = 2 * pad + 1
    
    for r in range(H):
        for c in range(W):
            curr_token = grid_target[r, c]
            curr_str = str(curr_token)
            
            # --- 1. Check Primary (T) ---
            window_pri = grid_pri_padded[r : r+window_size, c : c+window_size]
            
            # 1.1 Self Match
            if curr_token in window_pri:
                swap_mask[r, c] = 1 # Green
                continue # 已经是同一个token了，不用换（或者说换了也没变）
            
            # 1.2 Neighbor Match
            found_in_pri = False
            if curr_str in neighbor_dict:
                neighbors = neighbor_dict[curr_str]["neighbors"]
                # [Modified] 限制仅使用 Top 30 邻居
                for n_id in neighbors[:30]:
                    if n_id in window_pri:
                        grid_swapped[r, c] = n_id # 替换为历史中存在的那个邻居
                        swap_mask[r, c] = 1 # Green
                        found_in_pri = True
                        break
            
            if found_in_pri:
                continue
                
            # --- 2. Check Secondary (T-1) ---
            if grid_sec is not None:
                window_sec = grid_sec_padded[r : r+window_size, c : c+window_size]
                
                # 2.1 Self Match
                if curr_token in window_sec:
                    swap_mask[r, c] = 2 # Blue
                    continue
                
                # 2.2 Neighbor Match
                if curr_str in neighbor_dict:
                    neighbors = neighbor_dict[curr_str]["neighbors"]
                    # [Modified] 限制仅使用 Top 30 邻居
                    for n_id in neighbors[:30]:
                        if n_id in window_sec:
                            grid_swapped[r, c] = n_id # 替换为历史中存在的那个邻居
                            swap_mask[r, c] = 2 # Blue
                            break
                                
    return torch.tensor(grid_swapped, device=ids_target.device).flatten().unsqueeze(0), swap_mask

def run_swap_experiment(model, ids_t1, ids_t, ids_t_minus_1, mse_dict, H_tok, W_tok, pad, i, save_dir):
    """
    执行替换实验并保存结果
    """
    try:
        # Target: T+1 (ids_t1)
        # Primary: T (ids_t)
        # Secondary: T-1 (ids_t_minus_1)
        ids_swapped, swap_mask = swap_tokens_multiframe(ids_t1, ids_t, ids_t_minus_1, mse_dict, H_tok, W_tok, pad=pad)
        
        # 解码图片
        # 1. 原始 T+1
        img_recon_orig = decode_tokens(model, ids_t1, H_tok, W_tok)
        # 2. 替换后的 T+1
        img_recon_swap = decode_tokens(model, ids_swapped, H_tok, W_tok)
        
        # 统计
        total_pixels = H_tok * W_tok
        cnt_green = np.sum(swap_mask == 1)
        cnt_blue = np.sum(swap_mask == 2)
        cnt_red = np.sum(swap_mask == 0)
        
        stats_text = f"Green(T): {cnt_green/total_pixels:.1%}, Blue(T-1): {cnt_blue/total_pixels:.1%}, Red(Miss): {cnt_red/total_pixels:.1%}"
        
        plot_swap_experiment(img_recon_orig, img_recon_swap, swap_mask, i, save_dir, stats_text)
        
    except Exception as e:
        print(f"Swap experiment failed for frame {i}: {e}")
        import traceback
        traceback.print_exc()