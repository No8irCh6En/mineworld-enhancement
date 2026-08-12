import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class NeighborConsistencyLoss(nn.Module):
    def __init__(self, json_path, device, num_tokens=8192, alpha=0.5, use_spatial_tolerance=False):
        """
        alpha: 软标签损失的权重 (0.0 - 1.0). 
               Loss = (1 - alpha * similarity) * CE_Loss
        """
        super().__init__()
        self.alpha = alpha
        self.device = device
        self.num_tokens = num_tokens
        self.use_spatial_tolerance = use_spatial_tolerance
        
        print(f"[NeighborLoss] Loading neighbor relations from {json_path}...")
        self.similarity_matrix = self._load_similarity_matrix(json_path)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def _load_similarity_matrix(self, json_path):
        # 默认全 0 矩阵 (无相似度)
        matrix = torch.zeros((self.num_tokens, self.num_tokens), device=self.device, dtype=torch.float16)
        
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Using strict CE loss.")
            return matrix

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 填充矩阵
            # 假设 JSON 格式: {token_id: {"neighbors": [id1, ...], "scores": [s1, ...]}}
            # 如果没有 scores，默认给 0.9
            count = 0
            for token_str, info in data.items():
                idx = int(token_str)
                if idx >= self.num_tokens: continue
                
                neighbors = info.get('neighbors', [])
                # 如果有 scores 字段就用，没有就默认 0.9
                scores = info.get('scores', [0.9] * len(neighbors)) 
                
                if len(neighbors) > 0:
                    valid_indices = [n for n in neighbors if n < self.num_tokens]
                    valid_scores = [s for n, s in zip(neighbors, scores) if n < self.num_tokens]
                    
                    if valid_indices:
                        matrix[idx, valid_indices] = torch.tensor(valid_scores, dtype=torch.float16, device=self.device)
                        count += 1
            print(f"[NeighborLoss] Loaded neighbors for {count} tokens.")
        except Exception as e:
            print(f"[NeighborLoss] Error loading JSON: {e}")
            
        return matrix

    def forward(self, logits, target):
        """
        logits: (B, NumTokens, H, W) or (B, NumTokens)
        target: (B, H, W) or (B,)
        Returns: (B, H, W) or (B,) - per pixel loss
        """
        # 1. 标准 CE Loss (Per Pixel)
        ce = self.ce_loss(logits, target) # (B, H, W)
        
        if self.alpha == 0:
            return ce

        # 2. Soft Target Adjustment
        # 如果预测的 Token 是 GT 的邻居，则降低 Loss
        with torch.no_grad():
            # 获取预测最大概率的 Token ID
            pred_tokens = torch.argmax(logits, dim=1) # (B, H, W)
            
            # 展平以进行索引
            target_flat = target.view(-1)
            pred_flat = pred_tokens.view(-1)
            
            # 查表获取相似度: matrix[target, pred]
            # 注意：matrix 是 (NumTokens, NumTokens)
            sim_scores = self.similarity_matrix[target_flat, pred_flat] # (N,)
            sim_scores = sim_scores.view(target.shape) # (B, H, W)
            
            # --- 3. Spatial Consistency (3x3) ---
            # 检查预测的 token 是否出现在 GT 的 3x3 邻域内
            if target.dim() == 3 and self.use_spatial_tolerance:
                # target: (B, H, W)
                B, H, W = target.shape
                # 使用 replicate padding 防止边界填充 0 造成误判
                target_padded = F.pad(target.unsqueeze(1).float(), (1, 1, 1, 1), mode='replicate')
                # unfold 提取 3x3 patches: (B, 9, H*W) -> (B, 9, H, W)
                patches = F.unfold(target_padded, kernel_size=3, padding=0)
                patches = patches.view(B, 9, H, W)
                
                # 扩展预测值以进行广播比较
                pred_expanded = pred_tokens.unsqueeze(1).float() # (B, 1, H, W)
                
                # 只要匹配 3x3 范围内任意一个 GT，就认为是空间邻居
                spatial_hit = (patches == pred_expanded).any(dim=1).float() # (B, H, W)
                
                # 取语义相似度和空间匹配的最大值
                sim_scores = torch.max(sim_scores.float(), spatial_hit)
            
        # Loss 衰减公式: Loss_new = Loss_old * (1 - alpha * similarity)
        # 如果完全匹配 (sim=0, 因为对角线通常不在 neighbors 里或者我们没设)，Loss 不变 (其实 CE 已经很小了)
        # 如果是高相似邻居 (sim=0.9)，Loss 变为原来的 (1 - 0.5 * 0.9) = 0.55 倍
        # 这样模型预测邻居时受到的惩罚变小
        
        weighted_ce = ce * (1.0 - self.alpha * sim_scores.float())
        
        return weighted_ce