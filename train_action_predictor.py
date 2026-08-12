import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader,  get_worker_info
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
# From train.py
IMAGE_TOKEN_LENGTH = 336
ACTION_TOKEN_LENGTH = 11

MIN_ACTION_TOKEN_ID = 8192 
MAX_ACTION_TOKEN_ID = 8261
ACTION_VOCAB_SIZE = MAX_ACTION_TOKEN_ID - MIN_ACTION_TOKEN_ID + 1 # 70

# Hyperparams
PREDICTOR_EMBED_DIM = 64
PREDICTOR_HIDDEN_DIM = 256
PREDICTOR_LAYERS = 2
SEQ_LEN = 8 

safe_globals = {"array": np.array}

# =============================================================================
# Helper: Beam Search for Vector Top-K
# =============================================================================
def beam_search_top_k(logits, k=5):
    """
    Find top-k sequences of length 11 from independent logits.
    logits: [Batch, 11, Vocab]
    Returns: [Batch, k, 11]
    """
    B, Len, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1) # [B, 11, V]
    
    # Start with first position
    topk_vals, topk_inds = torch.topk(log_probs[:, 0, :], k, dim=-1)
    
    beam_seqs = topk_inds.unsqueeze(-1) # [B, K, 1]
    beam_scores = topk_vals             # [B, K]
    
    # Iterate positions 1..10
    for t in range(1, Len):
        curr_log_probs = log_probs[:, t, :] # [B, V]
        
        # [B, K, 1] + [B, 1, V] -> [B, K, V]
        candidates_scores = beam_scores.unsqueeze(-1) + curr_log_probs.unsqueeze(1)
        candidates_scores_flat = candidates_scores.view(B, -1)
        
        best_scores, best_indices_flat = torch.topk(candidates_scores_flat, k, dim=-1)
        
        beam_indices = best_indices_flat // V
        token_indices = best_indices_flat % V
        
        new_beam_seqs = torch.zeros(B, k, t+1, dtype=torch.long, device=logits.device)
        for b in range(B):
            prev_seqs = beam_seqs[b][beam_indices[b]] # [K, t]
            current_tokens = token_indices[b].unsqueeze(-1) # [K, 1]
            new_beam_seqs[b] = torch.cat([prev_seqs, current_tokens], dim=-1)
            
        beam_seqs = new_beam_seqs
        beam_scores = best_scores
        
    return beam_seqs

# =============================================================================
# Streaming Dataset (Handles 700k+ files efficiently)
# =============================================================================
class IterableActionDataset(IterableDataset):
    def __init__(self, file_list, seq_len=SEQ_LEN):
        self.files = file_list
        self.seq_len = seq_len
        # MCDataset initialization might be heavy, be careful in workers
    
    def _init_mcdataset(self):
         # Lazy init to avoid pickling issues or redundant setup
         if not hasattr(self, "mcdataset"):
            self.mcdataset = MCDataset()
            if not hasattr(self.mcdataset, "action_vocab"):
                self.mcdataset.make_action_vocab(action_vocab_offset=MIN_ACTION_TOKEN_ID)

    def process_file(self, fpath):
        """Reads one file and extracts all valid sequences independently."""
        self._init_mcdataset()
        episode_tokens = []
        samples = []
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        # json.loads is usually faster/safer than eval for standard jsonl
                        line_dict = json.loads(line.strip())
                        
                        if 'camera' in line_dict:
                             line_dict['camera'] = np.array(line_dict['camera'])
                        
                        tokens = self.mcdataset.get_action_index_from_actiondict(
                            line_dict, 
                            action_vocab_offset=MIN_ACTION_TOKEN_ID
                        )
                        
                        if len(tokens) != ACTION_TOKEN_LENGTH:
                            continue
                            
                        # Shift to [0, 70) for Embedding layer
                        tokens = np.array(tokens) - MIN_ACTION_TOKEN_ID
                        tokens = np.clip(tokens, 0, ACTION_VOCAB_SIZE - 1)
                        
                        episode_tokens.append(tokens)
                    except Exception:
                        continue
            
            if len(episode_tokens) > self.seq_len:
                episode_tokens = np.array(episode_tokens)
                for i in range(len(episode_tokens) - self.seq_len):
                    x = episode_tokens[i : i+self.seq_len]  # Past actions
                    y = episode_tokens[i+self.seq_len]      # Next action
                    samples.append((x, y))
        except Exception as e:
            # File read error, skip
            pass
            
        return samples

    def __iter__(self):
        # 1. Sharding for Multi-Process DataLoader
        worker_info = get_worker_info()
        if worker_info is None:  # Single-process
            my_files = self.files
        else:  # Multi-process: split files among workers
            my_files = [f for i, f in enumerate(self.files) 
                        if i % worker_info.num_workers == worker_info.id]
        
        # 2. Shuffle file order for randomness
        random.shuffle(my_files)
        
        # 3. Stream data
        for fpath in my_files:
            file_samples = self.process_file(fpath)
            # Add local shuffle for better batch variety
            random.shuffle(file_samples) 
            for x, y in file_samples:
                yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# =============================================================================
# Model
# =============================================================================
class ActionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(ACTION_VOCAB_SIZE, PREDICTOR_EMBED_DIM)
        
        self.input_dim = ACTION_TOKEN_LENGTH * PREDICTOR_EMBED_DIM
        
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=PREDICTOR_HIDDEN_DIM,
            num_layers=PREDICTOR_LAYERS,
            batch_first=True,
            dropout=0.1
        )
        
        self.head = nn.Linear(PREDICTOR_HIDDEN_DIM, ACTION_TOKEN_LENGTH * ACTION_VOCAB_SIZE)
    
    def forward(self, x):
        # x: [B, Seq, 11]
        B, S, L = x.shape
        emb = self.embedding(x) 
        emb_flat = emb.view(B, S, -1)
        
        out, _ = self.rnn(emb_flat)
        last_hidden = out[:, -1, :] 
        
        logits_flat = self.head(last_hidden)
        logits = logits_flat.view(B, ACTION_TOKEN_LENGTH, ACTION_VOCAB_SIZE)
        return logits
    
    def predict_top_k_vectors(self, x, k=5):
        """ Returns: [B, k, 11] (Indices in 0-70 range) """
        with torch.no_grad():
            logits = self(x)
            top_k_seqs = beam_search_top_k(logits, k)
        return top_k_seqs

# =============================================================================
# Fast File Scanning helper
# =============================================================================
def get_all_files(root_dir, pattern="action.jsonl", cache_path="file_list_cache.txt"):
    """
    Optimized scan with caching.
    """
    if os.path.exists(cache_path):
        print(f"Loading files from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(files)} files from cache.")
        return files
        
    print(f"Scanning {root_dir} for {pattern}... (This only happens once)")
    files = []
    # os.walk is usually faster than glob.glob for deep trees
    start_time = time.time()
    for root, dirs, filenames in os.walk(root_dir):
        if pattern in filenames:
             files.append(os.path.join(root, pattern))
             
        if len(files) % 10000 == 0 and len(files) > 0:
            print(f"Found {len(files)} files...", end='\r')
            
    print(f"\nScan complete. Found {len(files)} files in {time.time() - start_time:.1f}s")
    
    # Save cache
    print(f"Saving cache to {cache_path}...")
    with open(cache_path, 'w') as f:
        for p in files:
            f.write(p + "\n")
            
    return files

# =============================================================================
# Training
# =============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---------------------------------------------------------
    # !!!!!!!!!!!!! PLEASE UPDATE THIS PATH !!!!!!!!!!!!!
    # ---------------------------------------------------------
    data_root = "/home/cliang/mineworld/dataset_example" # <--- Change to your real data root
    # ---------------------------------------------------------
    
    # Increase batch size significantly since we are using DataParallel
    # Original 40000 might be too big for one GPU RAM if sequences are long, 
    # but for actions it's small. Adjust as needed.
    batch_size = 4096 * torch.cuda.device_count() if torch.cuda.is_available() else 4096
    lr = 1e-3
    epochs = 20
    num_workers = 16 
    
    # 1. Fast Scan with Cache
    files = get_all_files(data_root, pattern="action.jsonl", cache_path="action_files_list.txt")
    
    if len(files) == 0:
        print("No files found. Please check data_root path.")
        return
        
    print(f"Found {len(files)} files. Splitting train/val.")
    
    # Simple split by files
    random.shuffle(files)
    train_size = int(0.95 * len(files))
    train_files = files[:train_size]
    val_files = files[train_size:]
    
    # Use Iterable datasets
    train_ds = IterableActionDataset(train_files, seq_len=SEQ_LEN)
    val_ds = IterableActionDataset(val_files, seq_len=SEQ_LEN)
    
    # Note: shuffle=False for IterableDataset, shuffling happens inside
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True) 
    
    # 2. Initialize Model
    model = ActionPredictor()
    
    # 3. DataParallel for Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting Training on {len(train_files)} files with effective batch size {batch_size}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        steps = 0
        
        # Note: tqdm won't know total length of IterableDataset
        pbar = tqdm(train_loader, desc=f"Train Ep {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for Parallel output: [Batch, Length, Vocab] -> Flatten
            loss = criterion(logits.view(-1, ACTION_VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = train_loss / steps if steps > 0 else 0
            
        model.eval()
        hits = 0
        total = 0
        # Validation loop
        val_steps = 0
        MAX_VAL_STEPS = 100 
        
        print("Validating...")
        for x, y in val_loader:
             if val_steps > MAX_VAL_STEPS: break
             x, y = x.to(device), y.to(device)
             with torch.no_grad():
                # Handling DataParallel `.module` attribute access
                if isinstance(model, nn.DataParallel):
                    preds = model.module.predict_top_k_vectors(x, k=5)
                else:
                    preds = model.predict_top_k_vectors(x, k=5)
                    
                y_exp = y.unsqueeze(1) 
                matches = (preds == y_exp).all(dim=-1).any(dim=-1)
                hits += matches.sum().item()
                total += x.size(0)
             val_steps += 1

        acc = hits/total if total > 0 else 0
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Top-5 Acc: {acc:.4f}")
        
        # Save model (unwrap DataParallel before saving)
        save_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(save_model.state_dict(), "action_predictor_latest.pth")

if __name__ == "__main__":
    train()