import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import json
import os
import glob
import numpy as np
from tqdm import tqdm
from mcdataset import MCDataset
import torch.nn.functional as F
import random
import time

# =============================================================================
# Constants
# =============================================================================
MIN_ACTION_TOKEN_ID = 8192 
MAX_ACTION_TOKEN_ID = 8261
ACTION_VOCAB_SIZE = MAX_ACTION_TOKEN_ID - MIN_ACTION_TOKEN_ID + 1 # 70
ACTION_TOKEN_LENGTH = 11
SEQ_LEN = 4  # 保持您的设置

# =============================================================================
# New Model: GatedConvActionPredictor (Mathematically close to GRU, but fast)
# =============================================================================
class GatedConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # 1. Causal Padding (关键点！)
        # GRU 是因果的（只看过去），普通卷积看两边。
        # 我们只在左边填充，确保它只看“历史”。
        self.padding = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((self.padding, 0), 0.0)
        
        # 2. 卷积输出双倍通道
        # 一半用于“内容”(Value)，一半用于“门”(Gate)
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size, dilation=dilation)
        
        self.dropout = nn.Dropout(dropout)
        
        # 3. 1x1 Conv 用于混合通道信息
        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        residual = x
        
        # Pad -> Conv
        y = self.pad(x)
        y = self.conv(y) # [B, 2*C, Seq]
        
        # GLU Logic: Split -> Sigmoid * Identity
        # 这一步模拟了 GRU 的非线性门控能力
        val, gate = y.chunk(2, dim=1)
        y = val * F.sigmoid(gate) 
        
        y = self.dropout(y)
        y = self.out_proj(y)
        
        # Residual Connection (模拟 RNN 的状态传递)
        return residual + y

class GatedConvActionPredictor(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=512, num_layers=4):
        super().__init__()
        self.seq_len = SEQ_LEN
        
        # 1. Embedding
        self.embedding = nn.Embedding(ACTION_VOCAB_SIZE, embed_dim)
        
        # Input Channel Calc
        input_channels = ACTION_TOKEN_LENGTH * embed_dim 
        
        # 2. Input Projection (Channel Expansion)
        self.input_proj = nn.Conv1d(input_channels, hidden_dim, 1)
        
        # 3. Stack of Gated Convolutions
        # Dilation 依次增大，扩大感受野，类似堆叠的 RNN 层
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i # 1, 2, 4, 8...
            layers.append(GatedConvBlock(hidden_dim, kernel_size=3, dilation=dilation))
        
        self.backbone = nn.Sequential(*layers)
        
        # 4. Final Head
        self.head = nn.Linear(hidden_dim, ACTION_TOKEN_LENGTH * ACTION_VOCAB_SIZE)

    def forward(self, x):
        B, S, L = x.shape # [B, 4, 11]
        
        x = self.embedding(x)
        x = x.view(B, S, -1).permute(0, 2, 1) # [B, C, S]
        
        # Project to hidden dim
        x = self.input_proj(x)
        
        # Gated Conv Pass
        x = self.backbone(x)
        
        # Take Last Step
        # 经过因果卷积，最后一个时间步汇聚了所有历史信息
        last_feat = x[:, :, -1] # [B, Hidden]
        
        logits = self.head(last_feat)
        return logits.view(B, ACTION_TOKEN_LENGTH, ACTION_VOCAB_SIZE)

    def predict_top_k_vectors(self, x, k=5):
        with torch.no_grad():
            logits = self(x)
            top_k_seqs = beam_search_top_k(logits, k)
        return top_k_seqs

# =============================================================================
# Helper: Beam Search (Copy from existing)
# =============================================================================
def beam_search_top_k(logits, k=5):
    """
    Find top-k sequences of length 11 from independent logits.
    logits: [Batch, 11, Vocab]
    Returns: [Batch, k, 11]
    """
    B, Len, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1) # [B, 11, V]
    
    topk_vals, topk_inds = torch.topk(log_probs[:, 0, :], k, dim=-1)
    
    beam_seqs = topk_inds.unsqueeze(-1) 
    beam_scores = topk_vals             
    
    for t in range(1, Len):
        curr_log_probs = log_probs[:, t, :] 
        candidates_scores = beam_scores.unsqueeze(-1) + curr_log_probs.unsqueeze(1)
        candidates_scores_flat = candidates_scores.view(B, -1)
        best_scores, best_indices_flat = torch.topk(candidates_scores_flat, k, dim=-1)
        
        beam_indices = best_indices_flat // V
        token_indices = best_indices_flat % V
        
        new_beam_seqs = torch.zeros(B, k, t+1, dtype=torch.long, device=logits.device)
        for b in range(B):
            prev_seqs = beam_seqs[b][beam_indices[b]]
            current_tokens = token_indices[b].unsqueeze(-1)
            new_beam_seqs[b] = torch.cat([prev_seqs, current_tokens], dim=-1)
            
        beam_seqs = new_beam_seqs
        beam_scores = best_scores
        
    return beam_seqs

# =============================================================================
# Streaming Dataset (Copy from existing)
# =============================================================================
class IterableActionDataset(IterableDataset):
    def __init__(self, file_list, seq_len=SEQ_LEN):
        self.files = file_list
        self.seq_len = seq_len
    
    def _init_mcdataset(self):
         if not hasattr(self, "mcdataset"):
            self.mcdataset = MCDataset()
            if not hasattr(self.mcdataset, "action_vocab"):
                self.mcdataset.make_action_vocab(action_vocab_offset=MIN_ACTION_TOKEN_ID)

    def process_file(self, fpath):
        self._init_mcdataset()
        episode_tokens = []
        samples = []
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        line_dict = json.loads(line.strip())
                        if 'camera' in line_dict: line_dict['camera'] = np.array(line_dict['camera'])
                        
                        tokens = self.mcdataset.get_action_index_from_actiondict(
                            line_dict, action_vocab_offset=MIN_ACTION_TOKEN_ID
                        )
                        if len(tokens) != ACTION_TOKEN_LENGTH: continue
                        
                        tokens = np.array(tokens) - MIN_ACTION_TOKEN_ID
                        tokens = np.clip(tokens, 0, ACTION_VOCAB_SIZE - 1)
                        episode_tokens.append(tokens)
                    except: continue
            
            if len(episode_tokens) > self.seq_len:
                episode_tokens = np.array(episode_tokens)
                for i in range(len(episode_tokens) - self.seq_len):
                    x = episode_tokens[i : i+self.seq_len]  # Past actions [Seq, 11]
                    y = episode_tokens[i+self.seq_len]      # Next action [11]
                    samples.append((x, y))
        except: pass
        return samples

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None: my_files = self.files
        else: my_files = [f for i, f in enumerate(self.files) if i % worker_info.num_workers == worker_info.id]
        
        random.shuffle(my_files)
        for fpath in my_files:
            file_samples = self.process_file(fpath)
            random.shuffle(file_samples) 
            for x, y in file_samples:
                yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def get_all_files(root_dir, pattern="action.jsonl", cache_path="file_list_cache.txt"):
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f: return [line.strip() for line in f if line.strip()]
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        if pattern in filenames: files.append(os.path.join(root, pattern))
            
    with open(cache_path, 'w') as f:
        for p in files: f.write(p + "\n")
    return files

# =============================================================================
# Training
# =============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- CONFIG ---
    data_root = "/home/bml/storage/mnt/v-665ada01f1324f39/org/users/jjli/datasets/Minecraft/MineWorld/action_clip" # Modify this!
    batch_size = 40960
    lr = 2e-3 # Slightly higher LR for CNN
    epochs = 20
    num_workers = 16 
    
    files = get_all_files(data_root, pattern="action.jsonl", cache_path="/home/bml/storage/mnt/v-665ada01f1324f39/org/users/cliang/Mineworld/dataset/action_files_list.txt")
    if not files: return 
    
    random.shuffle(files)
    train_end = int(0.95 * len(files))
    train_files = files[:train_end]
    val_files = files[train_end:]
    
    train_loader = DataLoader(IterableActionDataset(train_files), batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(IterableActionDataset(val_files), batch_size=batch_size, num_workers=num_workers)
    
    # Init Model
    # 512维度, 4层通常对这种任务足够了，接近 RNN 2层 256的效果
    model = GatedConvActionPredictor(embed_dim=128, hidden_dim=512, num_layers=4).to(device)
    
    # COMPILE HERE!
    print("Compiling model...")
    model = torch.compile(model) 
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training ConvActionPredictor...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(x)
            
            # 使用 reshape
            loss = criterion(logits.reshape(-1, ACTION_VOCAB_SIZE), y.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            steps += 1
            if steps % 10 == 0: pbar.set_postfix({"loss": loss.item()})
            
        # Validation
        model.eval()
        hits, total = 0, 0
        val_limit = 100
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                if i >= val_limit: break
                x_val, y_val = x_val.to(device), y_val.to(device)
                
                # Reshape for validation if needed, but model output is [B, 11, 70]
                logits = model(x_val) 
                
                preds = torch.argmax(logits, dim=-1) # [B, 11]
                matches = (preds == y_val).all(dim=-1)
                
                hits += matches.sum().item()
                total += x_val.size(0)

        acc = hits/total if total > 0 else 0
        print(f"Epoch {epoch+1} | Acc (Exact Match): {acc:.4f}")
        
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(raw_model.state_dict(), "action_predictor_conv_latest.pth")

if __name__ == "__main__":
    train()