import torch
import torch.nn.functional as F
import numpy as np

def find_best_matches_rect(emb_t, emb_t1, H, W, pad=2):
    B, Seq, Dim = emb_t.shape
    emb_t_2d = emb_t.view(1, H, W, Dim).permute(0, 3, 1, 2) 
    emb_t1_2d = emb_t1.view(1, H, W, Dim).permute(0, 3, 1, 2)
    
    kernel_size = 2 * pad + 1
    padding = pad
    padded_t1 = F.pad(emb_t1_2d, (padding, padding, padding, padding), mode='replicate')
    patches = padded_t1.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    patches = patches.permute(0, 2, 3, 4, 5, 1).reshape(-1, kernel_size*kernel_size, Dim) 
    
    targets = emb_t.view(Seq, 1, Dim)
    sims = F.cosine_similarity(targets, patches, dim=-1) 
    best_sims, best_local_idx = torch.max(sims, dim=1) 
    
    dy = (best_local_idx // kernel_size) - pad
    dx = (best_local_idx % kernel_size) - pad
    offsets = torch.stack([dy, dx], dim=1) 
    return best_sims, offsets

def analyze_neighbor_hit(ids_t, ids_t1, neighbor_dict):
    """
    Rank 定义:
    0: Self Match
    1-30: Neighbor Match
    31: Miss
    """
    seq_len = ids_t.shape[1]
    ranks = np.full(seq_len, 31, dtype=np.int32) 
    
    ids_t_np = ids_t.cpu().numpy()[0]
    ids_t1_np = ids_t1.cpu().numpy()[0]
    
    hit_count = 0
    for i in range(seq_len):
        curr_token_int = ids_t_np[i]
        next_token_int = ids_t1_np[i]
        curr_token_str = str(curr_token_int)
        
        if curr_token_int == next_token_int:
            ranks[i] = 0
            hit_count += 1
            continue
            
        if curr_token_str in neighbor_dict:
            neighbors = neighbor_dict[curr_token_str]["neighbors"]
            if next_token_int in neighbors:
                rank = neighbors.index(next_token_int) + 1 
                ranks[i] = rank
                hit_count += 1
            
    return ranks, hit_count / seq_len

def analyze_neighbor_hit_window(ids_t, ids_t1, neighbor_dict, H, W, pad=2):
    """
    在 (2*pad+1)x(2*pad+1) 邻域内搜索 Self 或 Neighbor (Top 30)
    pad=2 -> 5x5
    pad=3 -> 7x7
    """
    grid_t = ids_t.view(H, W).cpu().numpy()
    grid_t1 = ids_t1.view(H, W).cpu().numpy()
    
    # 动态 padding
    grid_t1_padded = np.pad(grid_t1, ((pad, pad), (pad, pad)), mode='constant', constant_values=-1)
    
    ranks_map = np.full((H, W), 31, dtype=np.int32) 
    
    window_size = 2 * pad + 1

    for r in range(H):
        for c in range(W):
            curr_token_int = grid_t[r, c]
            curr_token_str = str(curr_token_int)
            
            # 动态切片
            window = grid_t1_padded[r : r+window_size, c : c+window_size]
            
            if curr_token_int in window:
                ranks_map[r, c] = 0
                continue
            
            if curr_token_str in neighbor_dict:
                neighbors = neighbor_dict[curr_token_str]["neighbors"]
                best_rank = 31
                for i, n_id in enumerate(neighbors):
                    if n_id in window:
                        best_rank = i + 1
                        break
                ranks_map[r, c] = best_rank
            
    return ranks_map

def analyze_combined_multiframe(ids_target, ids_primary, ids_secondary, neighbor_dict, H, W, pad):
    """
    检查 ids_target 是否在 ids_primary (Green) 或 ids_secondary (Blue) 中找到
    Forward: Target=T, Primary=T+1, Secondary=T+2
    Backward: Target=T+1, Primary=T, Secondary=T-1
    """
    # 1. Check Primary (Green)
    ranks_pri = analyze_neighbor_hit_window(ids_target, ids_primary, neighbor_dict, H, W, pad)
    
    # 2. Check Secondary (Blue) - 仅当 ids_secondary 存在时
    if ids_secondary is not None:
        ranks_sec = analyze_neighbor_hit_window(ids_target, ids_secondary, neighbor_dict, H, W, pad)
    else:
        ranks_sec = np.full((H, W), 31, dtype=np.int32)
        
    # Combine
    # 1: Green (in Primary)
    # 2: Blue (in Secondary, but NOT in Primary)
    # 3: Red (Miss in both)
    
    cat_map = np.full((H, W), 3, dtype=np.int32)
    val_map = np.full((H, W), 31, dtype=np.int32)
    
    # Mask for Primary Hit (Green)
    mask_pri = (ranks_pri <= 30)
    cat_map[mask_pri] = 1
    val_map[mask_pri] = ranks_pri[mask_pri]
    
    # Mask for Secondary Hit (Blue) - 只有在 Primary 没命中的地方才算
    mask_sec = (ranks_sec <= 30) & (~mask_pri)
    cat_map[mask_sec] = 2
    val_map[mask_sec] = ranks_sec[mask_sec]
    
    # Calculate hit rates
    total = H * W
    hit_pri = np.count_nonzero(mask_pri)
    hit_sec = np.count_nonzero(mask_sec)
    
    stats = {
        "pri_rate": hit_pri / total,
        "sec_rate": hit_sec / total,
        "total_hit": (hit_pri + hit_sec) / total
    }
    
    return cat_map, val_map, stats