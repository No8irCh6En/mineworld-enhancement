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

# 添加项目根目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def decode_single_token(model, token_id):
    """
    将单个 Token ID 解码为图像 Patch (用于可视化)。
    """
    vae = model.tokenizer
    try:
        device = next(vae.parameters()).device
    except StopIteration:
        device = next(model.parameters()).device
    
    tokens = torch.tensor([token_id], device=device).long()
    
    # 获取 embedding_dim
    try:
        embedding_dim = vae.model.quantize.embedding.weight.shape[1]
    except:
        embedding_dim = 64
        
    shape = (1, 1, 1, embedding_dim)
    
    try:
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                # 尝试直接查表，比 get_codebook_entry 更稳健
                z_q = vae.model.quantize.embedding(tokens) # [1, Dim]
                z_q = z_q.view(1, embedding_dim, 1, 1)
                
                quant2 = vae.model.post_quant_conv(z_q)
                dec = vae.model.decoder(quant2) 
        
        decoded = dec[0].detach().cpu() 
        decoded = torch.clamp(decoded, -1., 1.)
        decoded = (decoded + 1.0) / 2.0 * 255.0
        decoded = decoded.permute(1, 2, 0).numpy().astype(np.uint8) 
        return decoded

    except Exception as e:
        # print(f"Error decoding token {token_id}: {e}") 
        return np.zeros((16, 16, 3), dtype=np.uint8)

def analyze_all_relations_by_cosine(model, top_k=60):
    """
    [修改版] 基于 Embedding Cosine Similarity 计算最近邻 (仿照学长的方法)
    """
    print(f"Calculating Cosine Similarity matrix for ALL tokens (Top-{top_k})...")
    
    # === 修改：使用 Transformer 的 Embedding (1024维) ===
    try:
        print("Accessing Transformer Embeddings (expecting 1024 dimensions)...")
        
        # 调试：打印 model 的属性，找到正确的路径
        # print(f"Model type: {type(model)}")
        # print(f"Model attributes: {dir(model)}")
        
        # 尝试路径 1: LlamaLVM -> model (LlamaForCausalLM) -> model (LlamaModel) -> embed_tokens
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "embed_tokens"):
             transformer_embed = model.model.model.embed_tokens.weight.data
        # 尝试路径 2: LlamaLVM -> transformer (LlamaForCausalLM) -> model (LlamaModel) -> embed_tokens
        # (根据 train.py line 715: model.transformer.model.requires_grad_(False))
        elif hasattr(model, "transformer") and hasattr(model.transformer, "model") and hasattr(model.transformer.model, "embed_tokens"):
             transformer_embed = model.transformer.model.embed_tokens.weight.data
        else:
            raise AttributeError("Could not find embed_tokens in model")

        # 只取前 8192 个 (Image Tokens)
        codebook_features = transformer_embed[:8192] # [8192, 1024]
        print(f"Using Transformer Embedding shape: {codebook_features.shape}")
        
    except AttributeError as e:
        print(f"Error accessing Transformer embeddings: {e}")
        print("Falling back to VAE Codebook (64 dim)...")
        
        vae = model.tokenizer
        try:
            device = next(vae.parameters()).device
        except:
            device = torch.device('cuda')
        
        # 既然用户提到了 tokenize_images，我们看看那里发生了什么
        # h = encoder(x) -> quant_conv(h) -> quantize(h)
        # quantize 内部是查表 embedding.weight
        
        codebook = vae.model.quantize.embedding.weight.data # [8192, 64]
        
        # 如果用户坚持要用 VAE 里的东西凑出高维向量，那只能是 post_quant_conv
        # 但那也只是 64 -> 64 (通常 post_quant_conv 不改变通道数，或者改变很少)
        # 除非 VAE 的 decoder 第一层做了升维。
        
        z_q = codebook.unsqueeze(-1).unsqueeze(-1).to(device)
        z_post = vae.model.post_quant_conv(z_q)
        codebook_features = z_post.view(z_post.shape[0], -1)
        print(f"VAE Feature shape: {codebook_features.shape}")

    # 2. Normalize Embeddings (L2 Norm)
    codebook_norm = F.normalize(codebook_features, p=2, dim=-1) # [N, D]

    # 3. Calculate Cosine Similarity Matrix
    print("Calculating pairwise Cosine Similarity matrix...")
    sim_matrix = torch.matmul(codebook_norm, codebook_norm.t()) # [N, N]
    
    # 4. Scale to [0, 1]
    sim_matrix = (sim_matrix + 1.0) / 2.0
    
    # 排除自身
    sim_matrix.fill_diagonal_(-1.0)
    
    # === 新增统计逻辑 Start ===
    threshold = 0.8
    count_high_sim = (sim_matrix > threshold).sum().item()
    avg_high_sim = count_high_sim / 8192
    
    print(f"\n{'='*10} Similarity Statistics {'='*10}")
    print(f"Total pairs with Similarity > {threshold}: {count_high_sim}")
    print(f"Average neighbors > {threshold} per token: {avg_high_sim:.4f}")
    
    print(f"\nDistribution Overview:")
    for t in [0.6, 0.7, 0.8, 0.9, 0.95]:
        c = (sim_matrix > t).sum().item()
        avg = c / 8192
        print(f"  - Sim > {t:.2f}: Total {c:<8} (Avg {avg:.2f} per token)")
    print(f"{'='*40}\n")
    # === 新增统计逻辑 End ===

    print(f"Finding top-{top_k} nearest neighbors...")
    top_vals, top_idxs = torch.topk(sim_matrix, k=top_k, dim=1, largest=True)
    
    # === 修复：detach() ===
    top_vals = top_vals.detach().cpu().numpy()
    top_idxs = top_idxs.detach().cpu().numpy()
    
    all_results = {}
    for i in range(8192):
        all_results[str(i)] = {
            "neighbors": top_idxs[i].tolist(),
            "scores": top_vals[i].tolist() 
        }
        
    return all_results

def analyze_temporal_consistency(model, all_results, video_dir='/data/cliang/mineworld/dataset/images/'):
    print(f"\n{'='*10} Analyzing Temporal Consistency {'='*10}")
    print(f"Scanning images in {video_dir}...")
    
    # 1. Prepare Embeddings for Cosine Sim Calculation
    print("Preparing embeddings for similarity calculation...")
    try:
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "embed_tokens"):
             transformer_embed = model.model.model.embed_tokens.weight.data
        elif hasattr(model, "transformer") and hasattr(model.transformer, "model") and hasattr(model.transformer.model, "embed_tokens"):
             transformer_embed = model.transformer.model.embed_tokens.weight.data
        else:
            raise AttributeError("Could not find embed_tokens")
            
        # Normalize for fast cosine sim
        # [8192, 1024]
        embeddings = transformer_embed[:8192]
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
    except Exception as e:
        print(f"Error preparing embeddings: {e}")
        return

    # 1.5 Prepare Neighbor Lookup Table (Keep existing logic)
    num_tokens = 8192 
    first_key = list(all_results.keys())[0]
    top_k = len(all_results[first_key]["neighbors"])
    
    vae = model.tokenizer
    try:
        device = next(vae.parameters()).device
    except:
        device = torch.device('cuda')

    neighbor_table = torch.zeros((num_tokens, top_k), dtype=torch.long, device=device)
    for i in range(num_tokens):
        if str(i) in all_results:
            neighbor_table[i] = torch.tensor(all_results[str(i)]["neighbors"], device=device)
            
    # 2. Load Images (Recursive Search)
    image_paths = sorted(glob.glob(os.path.join(video_dir, "**", "*.png"), recursive=True))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(video_dir, "**", "*.jpg"), recursive=True))
        
    if not image_paths:
        print(f"No images found in {video_dir}")
        return

    print(f"Found {len(image_paths)} images total.")

    # Group by episode
    episodes = defaultdict(list)
    for p in image_paths:
        parent_dir = os.path.basename(os.path.dirname(p))
        filename = os.path.basename(p)
        if "episode_" in parent_dir:
            ep_id = parent_dir
        elif "episode_" in filename:
            ep_id = filename.split('_')[1]
        else:
            ep_id = parent_dir
            
        try:
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                t = int(numbers[-1])
                episodes[ep_id].append((t, p))
        except ValueError:
            pass
            
    print(f"Grouped into {len(episodes)} episodes.")
    
    MAX_EPISODES = 50
    if len(episodes) > MAX_EPISODES:
        print(f"Limiting analysis to first {MAX_EPISODES} episodes...")
        limited_keys = list(episodes.keys())[:MAX_EPISODES]
        episodes = {k: episodes[k] for k in limited_keys}
            
    # 3. Process Episodes
    total_tokens = 0
    in_top_k_count = 0
    in_top_k_3x3_count = 0 # 新增：3x3 Top-K 命中数
    
    # 统计 1: 直接对齐的相似度
    total_sim = 0.0
    sim_buckets = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
    
    # 统计 2: 3x3 邻域最佳相似度
    total_sim_3x3 = 0.0
    sim_buckets_3x3 = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
    
    # 统计 3: 偏移概率分布 (Offset Distribution)
    # 0: (-1,-1), 1: (-1,0), 2: (-1,1)
    # 3: (0,-1),  4: (0,0),  5: (0,1)
    # 6: (1,-1),  7: (1,0),  8: (1,1)
    offset_counts = torch.zeros(9, dtype=torch.long, device=device)
    
    for ep_id, frames in tqdm(episodes.items(), desc="Processing Episodes"):
        frames.sort(key=lambda x: x[0])
        prev_tokens = None
        
        MAX_FRAMES_PER_EP = 20
        if len(frames) > MAX_FRAMES_PER_EP:
            frames = frames[:MAX_FRAMES_PER_EP]
        
        for t, img_path in frames:
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w, _ = img.shape
            new_h = (h // 16) * 16
            new_w = (w // 16) * 16
            if new_h != h or new_w != w:
                img = cv2.resize(img, (new_w, new_h))
                
            img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if hasattr(vae, 'tokenize_images'):
                    curr_tokens = vae.tokenize_images(img_tensor) # [1, H, W]
                else:
                    enc = vae.model.encoder(img_tensor)
                    quant = vae.model.quant_conv(enc)
                    _, _, (_, _, curr_tokens) = vae.model.quantize(quant)
                    curr_tokens = curr_tokens.view(1, quant.shape[2], quant.shape[3])

            if prev_tokens is not None:
                if prev_tokens.shape == curr_tokens.shape:
                    # prev_tokens: [1, H, W]
                    # curr_tokens: [1, H, W]
                    H, W = prev_tokens.shape[1], prev_tokens.shape[2]
                    
                    p_flat = prev_tokens.view(-1) # [N]
                    c_flat = curr_tokens.view(-1) # [N]
                    
                    # --- 1. Top-K Check (Direct) ---
                    p_neighbors = neighbor_table[p_flat] # [N, K]
                    matches = (p_neighbors == c_flat.unsqueeze(1)).any(dim=1)
                    in_top_k_count += matches.sum().item()
                    
                    # --- 2. Direct Cosine Similarity ---
                    emb_p = embeddings[p_flat] # [N, 1024]
                    emb_c = embeddings[c_flat] # [N, 1024]
                    sims = (emb_p * emb_c).sum(dim=1)
                    sims = (sims + 1.0) / 2.0 # [0, 1]
                    
                    total_sim += sims.sum().item()
                    for thresh in sim_buckets:
                        sim_buckets[thresh] += (sims > thresh).sum().item()
                    
                    # --- 3. 3x3 Neighborhood Analysis ---
                    # Pad curr_tokens 以处理边界
                    curr_padded = F.pad(curr_tokens.float(), (1, 1, 1, 1), mode='replicate').long() # [1, H+2, W+2]
                    
                    # Unfold: [1, H, W, 3, 3] -> [1, H, W, 9]
                    patches = curr_padded.unfold(1, 3, 1).unfold(2, 3, 1) 
                    patches = patches.contiguous().view(1, H, W, 9) 
                    patches_flat = patches.view(-1, 9) # [N, 9]
                    
                    # 3a. 3x3 Top-K Check
                    # 检查 patches_flat 中的任意一个是否在 p_neighbors 中
                    # p_neighbors: [N, K] -> [N, 1, K]
                    # patches_flat: [N, 9] -> [N, 9, 1]
                    # 这种广播比较太大了 [N, 9, K]，可能会 OOM
                    # 优化：对于每个 N，只要 patches_flat[n] 与 p_neighbors[n] 有交集即可
                    # 使用 mask 方法
                    
                    # 简单方法：遍历 9 个位置 (虽然慢一点，但省显存)
                    any_match = torch.zeros_like(matches)
                    for k in range(9):
                        neighbor_k = patches_flat[:, k] # [N]
                        match_k = (p_neighbors == neighbor_k.unsqueeze(1)).any(dim=1)
                        any_match = any_match | match_k
                    
                    in_top_k_3x3_count += any_match.sum().item()

                    # 3b. 3x3 Best Similarity & Offset
                    # 获取这 9 个邻居的 Embedding
                    emb_neighbors = embeddings[patches_flat] # [N, 9, 1024]
                    
                    # 计算 prev_token 与这 9 个邻居的相似度
                    # emb_p: [N, 1024] -> [N, 1, 1024]
                    sims_3x3 = (emb_p.unsqueeze(1) * emb_neighbors).sum(dim=-1) # [N, 9]
                    sims_3x3 = (sims_3x3 + 1.0) / 2.0
                    
                    # 取最大值和对应的索引 (Offset Index)
                    best_sims_3x3, best_indices = sims_3x3.max(dim=1) # [N], [N]
                    
                    total_sim_3x3 += best_sims_3x3.sum().item()
                    for thresh in sim_buckets_3x3:
                        sim_buckets_3x3[thresh] += (best_sims_3x3 > thresh).sum().item()
                        
                    # 统计 Offset 分布
                    # best_indices 是 0~8
                    # 使用 bincount 统计
                    counts = torch.bincount(best_indices, minlength=9)
                    offset_counts += counts
                    
                    total_tokens += matches.numel()
            
            prev_tokens = curr_tokens

    if total_tokens > 0:
        ratio = in_top_k_count / total_tokens
        ratio_3x3 = in_top_k_3x3_count / total_tokens
        
        avg_sim = total_sim / total_tokens
        avg_sim_3x3 = total_sim_3x3 / total_tokens
        
        print(f"\nTemporal Consistency Results:")
        print(f"Total Transitions Checked: {total_tokens}")
        print(f"{'-'*30}")
        print(f"Matches in Top-{top_k} (Direct): {in_top_k_count} ({ratio*100:.2f}%)")
        print(f"Matches in Top-{top_k} (3x3):    {in_top_k_3x3_count} ({ratio_3x3*100:.2f}%)")
        print(f"{'-'*30}")
        
        print(f"1. Direct Alignment (Same Position):")
        print(f"   Average Cosine Similarity: {avg_sim:.4f}")
        print(f"   Distribution:")
        for thresh in sorted(sim_buckets.keys()):
            count = sim_buckets[thresh]
            pct = count / total_tokens * 100
            print(f"     > {thresh:.1f}: {count:<8} ({pct:.2f}%)")
            
        print(f"{'-'*30}")
        print(f"2. 3x3 Neighborhood Best Match (Allow 1-pixel shift):")
        print(f"   Average Best Similarity:   {avg_sim_3x3:.4f}")
        print(f"   Distribution:")
        for thresh in sorted(sim_buckets_3x3.keys()):
            count = sim_buckets_3x3[thresh]
            pct = count / total_tokens * 100
            print(f"     > {thresh:.1f}: {count:<8} ({pct:.2f}%)")
            
        print(f"{'-'*30}")
        print(f"3. Offset Probability Distribution (Where did the token go?):")
        # 0: (-1,-1), 1: (-1,0), 2: (-1,1)
        # 3: (0,-1),  4: (0,0),  5: (0,1)
        # 6: (1,-1),  7: (1,0),  8: (1,1)
        offset_names = [
            "(-1,-1) TL", "(-1, 0) T ", "(-1, 1) TR",
            "( 0,-1) L ", "( 0, 0) C ", "( 0, 1) R ",
            "( 1,-1) BL", "( 1, 0) B ", "( 1, 1) BR"
        ]
        total_offsets = offset_counts.sum().item()
        for i in range(9):
            count = offset_counts[i].item()
            pct = count / total_offsets * 100
            print(f"   {offset_names[i]}: {count:<8} ({pct:.2f}%)")
            
    else:
        print("No valid transitions found.")

def visualize_subset(model, all_results, num_samples=8, output_path="token_relations_cosine.png"):
    """
    可视化 (修改了标签显示，显示 Sim 而不是 MSE)
    """
    valid_keys = list(all_results.keys())
    sample_keys = random.sample(valid_keys, num_samples)
    
    # 限制可视化数量
    MAX_VIS_NEIGHBORS = 10
    
    subset_results = []
    for k in sample_keys:
        item = all_results[k]
        subset_results.append({
            "query": int(k),
            "neighbors": item["neighbors"][:MAX_VIS_NEIGHBORS],
            "scores": item["scores"][:MAX_VIS_NEIGHBORS]
        })
        
    num_rows = len(subset_results)
    num_cols = 1 + len(subset_results[0]["neighbors"]) 
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
    fig.suptitle(f"Visual Nearest Neighbors (Top-{MAX_VIS_NEIGHBORS})\nMetric: Cosine Similarity (Higher is Better)", fontsize=16)
    
    print(f"Decoding images for visualization...")
    
    for row_idx, item in enumerate(subset_results):
        q_id = item["query"]
        neighbors = item["neighbors"]
        scores = item["scores"]
        
        # 1. Query
        img_q = decode_single_token(model, q_id)
        ax_q = axes[row_idx, 0]
        ax_q.imshow(img_q)
        ax_q.set_ylabel(f"ID: {q_id}", fontsize=12, rotation=90)
        ax_q.set_xticks([])
        ax_q.set_yticks([])
        if row_idx == 0: ax_q.set_title("Query", color='blue', fontweight='bold')
        for spine in ax_q.spines.values():
            spine.set_edgecolor('blue'); spine.set_linewidth(2)

        # 2. Neighbors
        for col_idx, (n_id, score) in enumerate(zip(neighbors, scores)):
            img_n = decode_single_token(model, n_id)
            ax_n = axes[row_idx, col_idx + 1]
            ax_n.imshow(img_n)
            ax_n.set_xticks([])
            ax_n.set_yticks([])
            # 修改这里：显示 Sim
            ax_n.set_xlabel(f"ID:{n_id}\nSim:{score:.4f}", fontsize=8)
            if row_idx == 0: ax_n.set_title(f"Rank-{col_idx+1}")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    # === 配置 ===
    CONFIG_PATH = "configs/modify.yaml"
    CKPT_PATH = "/data/jjli/workspace/mineworld/checkpoints/300M_16f.ckpt"
    
    OUTPUT_DIR = "analysis_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 修改文件名以区分
    JSON_PATH = os.path.join(OUTPUT_DIR, "all_token_neighbors_cosine.json")
    IMG_PATH = os.path.join(OUTPUT_DIR, "token_relations_cosine_check.png")

    # === 运行 ===
    model = load_model(CONFIG_PATH, CKPT_PATH)
    
    # 使用新的 Cosine 方法
    all_results = analyze_all_relations_by_cosine(model, top_k=120)
    
    print(f"Saving Cosine-based relations to {JSON_PATH}...")
    with open(JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 抽样可视化
    visualize_subset(model, all_results, num_samples=8, output_path=IMG_PATH)
    
    # 新增：时间一致性分析
    analyze_temporal_consistency(model, all_results)