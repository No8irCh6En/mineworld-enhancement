import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import cv2
import argparse
import os
import glob
import json
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Import VAE
try:
    from vae import VAE
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from vae import VAE

# Import DepthAnything
try:
    from util.DepthAnythingWrapper import DepthAnythingWrapper, DEPTH_ANYTHING_TRANSFORM
except ImportError:
    print("Warning: Could not import DepthAnythingWrapper. Make sure util/DepthAnythingWrapper.py exists.")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    DepthAnythingWrapper = None 
    DEPTH_ANYTHING_TRANSFORM = None

# --- Import MCDataset for Action Processing ---
try:
    from mcdataset import MCDataset
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mcdataset import MCDataset

# --- New: Load Neighbor Mask ---
def load_neighbor_mask(json_path, num_tokens=8192, device='cuda', top_k=30):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Loading neighbor mask from {json_path} with top_k={top_k}...")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Neighbor file {json_path} not found.")
        return None
    
    # Default to 0 (masked)
    mask = torch.zeros((num_tokens, num_tokens), dtype=torch.float32, device=device)
    
    count = 0
    for token_str, info in data.items():
        token_id = int(token_str)
        if token_id >= num_tokens: continue
        neighbors = info['neighbors']
        
        # --- 修改：只取 Top K ---
        if top_k is not None and top_k > 0:
            neighbors = neighbors[:top_k]
            
        # Filter valid neighbors
        valid_neighbors = [n for n in neighbors if n < num_tokens]
        
        mask[token_id, valid_neighbors] = 1.0
        mask[token_id, token_id] = 1.0 # Self-loop (Input token is always a candidate)
        count += 1
        
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Loaded neighbors for {count} tokens.")
    return mask

# --- 1. Dataset ---
class MultiModalDataset(Dataset):
    def __init__(self, image_dir, action_dir, label_dir, cache=True):
        self.samples = []
        self.cache = cache
        self.mc_helper = MCDataset() # Use MCDataset helper
        
        self.action_map = {}
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Scanning for actions in {action_dir}...")

        label_files = sorted(glob.glob(os.path.join(label_dir, "*_labels.npy")))
        
        for lf in label_files:
            ep_name = os.path.basename(lf).replace("_labels.npy", "")
            act_file = os.path.join(action_dir, ep_name, "action.jsonl")
            ep_actions = []
            
            if os.path.exists(act_file):
                # Use MCDataset logic to parse actions
                try:
                    with open(act_file, 'r') as f:
                        for line in f:
                            try:
                                json_action = json.loads(line)
                                # Use MCDataset to convert JSON to Env Action (handles conflicts, camera scaling)
                                env_action, _ = self.mc_helper.json_action_to_env_action(json_action)
                                
                                # Construct 11-dim vector matching the model input
                                # [forward, back, left, right, jump, sneak, sprint, cam_x, cam_y, attack, use]
                                vec = [
                                    float(env_action['forward']), float(env_action['back']),
                                    float(env_action['left']), float(env_action['right']),
                                    float(env_action['jump']), float(env_action['sneak']),
                                    float(env_action['sprint']),
                                    float(env_action['camera'][0]), float(env_action['camera'][1]),
                                    float(env_action['attack']), float(env_action['use'])
                                ]
                                ep_actions.append(np.array(vec, dtype=np.float32))
                            except Exception as e:
                                # print(f"Error parsing action line: {e}")
                                ep_actions.append(np.zeros(11, dtype=np.float32))
                except Exception as e:
                    print(f"Error reading action file {act_file}: {e}")
            
            labels = np.load(lf) 
            img_folder = os.path.join(image_dir, ep_name)
            if not os.path.isdir(img_folder): continue
            
            images = sorted(glob.glob(os.path.join(img_folder, "image_*.png")))
            if not images: images = sorted(glob.glob(os.path.join(img_folder, "*.png")))
            
            valid_len = min(len(images), len(labels))
            
            for i in range(valid_len):
                act = ep_actions[i] if i < len(ep_actions) else np.zeros(11, dtype=np.float32)
                self.samples.append({
                    "img_path": images[i],
                    "action": act,
                    "label": labels[i]
                })
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Total samples: {len(self.samples)}")

    def _load_item(self, idx):
        item = self.samples[idx]
        
        img_bgr = cv2.imread(item['img_path'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] != 224 or img_rgb.shape[1] != 384:
            img_rgb = cv2.resize(img_rgb, (384, 224))
            
        # VAE Input: [-1, 1]
        img_norm = img_rgb.astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)
        
        # Depth Input: [0, 1]
        img_01 = (img_norm + 1.0) * 0.5
        depth_input = DEPTH_ANYTHING_TRANSFORM({'image': img_01})['image']
        depth_input_tensor = torch.from_numpy(depth_input)
        
        # --- 修改：不再扩展成 Map，直接返回向量 ---
        action_vec = torch.from_numpy(item['action']) # (11,)
        
        label = torch.from_numpy(item['label'])
        target_cls = label[:, :, 0].unsqueeze(0) 
        target_token = label[:, :, 1].long()     
        
        return img_tensor, depth_input_tensor, action_vec, target_cls, target_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._load_item(idx)

# --- 2. Model (Modified for Late Fusion) ---
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetTokenPredictor(nn.Module):
    def __init__(self, input_channels=4, action_dim=11, num_tokens=8192): 
        super().__init__()
        # 1. Image Branch (RGB + Depth = 4 channels)
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, stride=2), 
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = ResBlock(32, 64, stride=2)   
        self.layer2 = ResBlock(64, 128, stride=2)  
        self.layer3 = ResBlock(128, 256, stride=2) 
        # layer4 输出 512 通道，尺寸 14x24
        self.layer4 = ResBlock(256, 512, stride=1) 
        
        # 2. Action Branch (MLP)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512), # 映射到与 ResNet 输出相同的通道数
            nn.ReLU()
        )
        
        # 3. Fusion & Heads
        # 融合后通道数维持 512 (相加) 或者 1024 (拼接)
        # 这里采用简单的相加融合 (Addition Fusion)
        
        self.head_cls = nn.Conv2d(512, 1, 1)
        self.head_token = nn.Conv2d(512, num_tokens, 1) 
        
        nn.init.normal_(self.head_token.weight, std=0.01)
        nn.init.constant_(self.head_token.bias, 0)

    def forward(self, img_depth, action_vec):
        # img_depth: (B, 4, H, W)
        # action_vec: (B, 11)
        
        # 1. Process Image
        x = self.initial(img_depth)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B, 512, 14, 24)
        
        # 2. Process Action
        act_feat = self.action_mlp(action_vec) # (B, 512)
        act_feat = act_feat.unsqueeze(-1).unsqueeze(-1) # (B, 512, 1, 1)
        
        # 3. Late Fusion (Broadcasting addition)
        # 将动作特征加到每一个 Patch 上
        x = x + act_feat 
        
        return self.head_cls(x), self.head_token(x)

# --- 3. Visualization & Evaluation ---
def evaluate_and_visualize(model, vae, depth_model, loader, device, output_dir, epoch, neighbor_mask=None, save_images=False, max_save_count=None):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    total_cls_acc = 0
    total_token_acc = 0
    total_samples = 0
    saved_count = 0 # Counter for saved images
    
    # Use tqdm for evaluation progress
    iterator = tqdm(loader, desc=f"Evaluating Epoch {epoch}")
    
    with torch.no_grad():
        for batch_idx, (img_tensor, depth_input, action_vec, target_cls, target_token) in enumerate(iterator):
            img_tensor = img_tensor.to(device)
            depth_input = depth_input.to(device)
            action_vec = action_vec.to(device)
            target_cls = target_cls.to(device)
            target_token = target_token.to(device)
            
            depth_map = depth_model(depth_input)
            depth_map = depth_map.unsqueeze(1)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            
            img_depth = torch.cat([img_tensor, depth_map], dim=1)
            pred_cls, pred_token_logits = model(img_depth, action_vec)
            
            # --- Apply Neighbor Mask (Inference) ---
            if neighbor_mask is not None:
                 input_token_ids = vae.tokenize_images(img_tensor)
                 batch_mask = F.embedding(input_token_ids, neighbor_mask)
                 batch_mask = batch_mask.permute(0, 3, 1, 2)
                 pred_token_logits = pred_token_logits + (1.0 - batch_mask) * -1e9
            
            pred_token_ids = torch.argmax(pred_token_logits, dim=1) 
            pred_cls_binary = (torch.sigmoid(pred_cls) > 0.5).float()
            
            # --- Metrics ---
            # 1. Classification Accuracy
            cls_correct = (pred_cls_binary == target_cls).float().sum()
            total_cls_acc += cls_correct.item()
            
            # 2. Token Accuracy (Only on recoverable patches or all patches? Let's do all)
            token_correct_mask = (pred_token_ids == target_token)
            total_token_acc += token_correct_mask.float().sum().item()
            
            total_samples += target_cls.numel() # B * H * W
            
            # --- Visualization (Save images if requested and limit not reached) ---
            if save_images and (max_save_count is None or saved_count < max_save_count):
                codebook = vae.model.quantize.embedding.weight
                
                # Decode Target
                z_q_target = F.embedding(target_token, codebook).permute(0, 3, 1, 2)
                z_q_target = vae.model.post_quant_conv(z_q_target) 
                decoded_target = vae.model.decoder(z_q_target)
                
                # Decode Pred
                z_q_pred = F.embedding(pred_token_ids, codebook).permute(0, 3, 1, 2)
                z_q_pred = vae.model.post_quant_conv(z_q_pred) 
                decoded_pred = vae.model.decoder(z_q_pred)
                
                # Decode Merged
                mask = pred_cls_binary
                z_raw_pred = F.embedding(pred_token_ids, codebook).permute(0, 3, 1, 2)
                z_raw_target = F.embedding(target_token, codebook).permute(0, 3, 1, 2)
                z_merged = mask * z_raw_pred + (1 - mask) * z_raw_target
                z_merged = vae.model.post_quant_conv(z_merged) 
                decoded_merged = vae.model.decoder(z_merged)

                for i in range(img_tensor.size(0)):
                    # Check limit again inside batch loop
                    if max_save_count is not None and saved_count >= max_save_count:
                        break

                    def to_img(t): 
                        t = t.cpu().permute(1, 2, 0).numpy()
                        return np.clip(t * 0.5 + 0.5, 0, 1)
                    
                    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                    
                    img_np = to_img(img_tensor[i])
                    t_cls = target_cls[i, 0].cpu().numpy()
                    p_cls = pred_cls_binary[i, 0].cpu().numpy()
                    t_correct = token_correct_mask[i].cpu().numpy()
                    
                    conf_map = np.zeros((t_cls.shape[0], t_cls.shape[1], 3))
                    
                    hit_mask = (p_cls==1) & (t_cls==1)
                    conf_map[hit_mask & (t_correct==1)] = [0, 1, 0]   # Green
                    conf_map[hit_mask & (t_correct==0)] = [1, 1, 0]   # Yellow
                    conf_map[(p_cls==1) & (t_cls==0)] = [1, 0, 0]     # Red
                    conf_map[(p_cls==0) & (t_cls==1)] = [0, 0, 1]     # Blue
                    
                    conf_map_resized = cv2.resize(conf_map, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    axes[0,0].imshow(img_np)
                    axes[0,0].imshow(conf_map_resized, alpha=0.4)
                    axes[0,0].set_title("G:Perfect, Y:BadFix, R:FP, B:Miss")
                    
                    axes[0,1].imshow(depth_map[i, 0].cpu().numpy(), cmap='inferno')
                    axes[0,1].set_title("Depth")
                    
                    axes[0,2].imshow(to_img(decoded_target[i]))
                    axes[0,2].set_title("Target")
                    
                    axes[1,0].imshow(to_img(decoded_pred[i]))
                    axes[1,0].set_title("Pred")
                    
                    axes[1,1].imshow(to_img(decoded_merged[i]))
                    axes[1,1].set_title("Merged")
                    
                    act = action_vec[i].cpu().numpy()
                    axes[1,2].bar(range(11), act)
                    axes[1,2].set_title("Action")
                    
                    plt.tight_layout()
                    # Save with batch index to avoid overwriting
                    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{i}.png"))
                    plt.close()
                    saved_count += 1

    avg_cls_acc = total_cls_acc / total_samples
    avg_token_acc = total_token_acc / total_samples
    
    return avg_cls_acc, avg_token_acc

# --- 4. Training Loop ---
def train(args):
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_master = (local_rank == 0)
    else:
        device = torch.device(args.device)
        is_master = True
        local_rank = 0

    if is_master: print("Loading VAE & DepthAnything...")
    
    vae = VAE(args.vae_config, args.vae_ckpt)
    vae.to(device)
    vae.eval()
    for param in vae.parameters(): param.requires_grad = False
    
    depth_model = DepthAnythingWrapper(device, (384, 224))
    depth_model.eval()
    for param in depth_model.parameters(): param.requires_grad = False
    
    # --- Load Neighbor Mask ---
    neighbor_mask = None


    if args.use_neighbor_mask:
        workspace_root = os.path.dirname(os.path.abspath(__file__))
        # Use the shift-based neighbors
        neighbor_json = os.path.join(workspace_root, "analysis_results", "shift_token_neighbors.json")
        if os.path.exists(neighbor_json):
            neighbor_mask = load_neighbor_mask(neighbor_json, device=device, top_k=30)
        else:
            if is_master: print(f"Warning: {neighbor_json} not found. Training without neighbor prior.")

    dataset = MultiModalDataset(args.image_dir, args.action_dir, args.label_dir, cache=True)
    if len(dataset) == 0: return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_sampler = DistributedSampler(train_set, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if is_ddp else None
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            sampler=val_sampler, num_workers=4, pin_memory=True)
    
    model = ResNetTokenPredictor(input_channels=4, action_dim=11, num_tokens=8192).to(device)
    
    if is_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion_ce = nn.CrossEntropyLoss(reduction='none') 
    criterion_mse = nn.MSELoss()
    
    accumulation_steps = args.accum_steps
    
    best_token_acc = 0.0
    
    for epoch in range(args.epochs):
        if is_ddp: train_sampler.set_epoch(epoch)
        model.train()
        
        # --- Statistics Accumulators ---
        total_loss = 0
        total_loss_cls = 0
        total_loss_ce = 0
        total_loss_mse = 0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_master else train_loader
        
        for i, (img_tensor, depth_input, action_vec, target_cls, target_token) in enumerate(iterator):
            img_tensor = img_tensor.to(device)
            depth_input = depth_input.to(device)
            action_vec = action_vec.to(device)
            target_cls = target_cls.to(device)
            target_token = target_token.to(device)
            
            with torch.no_grad():
                depth_map = depth_model(depth_input)
                depth_map = depth_map.unsqueeze(1)
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            
            # 修改 Forward 调用
            img_depth = torch.cat([img_tensor, depth_map], dim=1)
            pred_cls, pred_token_logits = model(img_depth, action_vec) 
            
            # --- Apply Neighbor Mask ---
            if neighbor_mask is not None:
                with torch.no_grad():
                    # 1. Get Input Tokens (from current frame)
                    input_token_ids = vae.tokenize_images(img_tensor) # (B, H, W)
                    
                    # 2. Lookup Neighbors
                    batch_mask = F.embedding(input_token_ids, neighbor_mask)
                    
                    # 3. Ensure Ground Truth is included (Crucial for training stability)
                    batch_mask.scatter_(-1, target_token.unsqueeze(-1), 1.0)
                    
                    # 4. Permute to (B, 8192, H, W)
                    batch_mask = batch_mask.permute(0, 3, 1, 2)
                
                # 5. Apply Mask (Large negative penalty for non-neighbors)
                pred_token_logits = pred_token_logits + (1.0 - batch_mask) * -1e9

            # --- Balanced Sampling (1:1 Ratio) to Reduce Blue ---
            pred_cls_flat = pred_cls.view(-1)
            target_cls_flat = target_cls.view(-1)
            
            pos_mask = (target_cls_flat > 0.5)
            neg_mask = ~pos_mask
            
            num_pos = pos_mask.sum()
            
            if num_pos > 0:
                # 1:1 Ratio: Keep all positives, sample equal number of negatives
                neg_indices = torch.where(neg_mask)[0]
                num_neg_keep = min(len(neg_indices), int(num_pos * 1)) # 1:1 Ratio
                
                perm = torch.randperm(len(neg_indices), device=device)[:num_neg_keep]
                neg_indices_keep = neg_indices[perm]
                
                pos_indices = torch.where(pos_mask)[0]
                
                keep_indices = torch.cat([pos_indices, neg_indices_keep])
                
                loss_cls = F.binary_cross_entropy_with_logits(pred_cls_flat[keep_indices], target_cls_flat[keep_indices])
            else:
                # Fallback if no positives in batch
                neg_indices = torch.where(neg_mask)[0]
                if len(neg_indices) > 0:
                    num_keep = min(len(neg_indices), 1000)
                    perm = torch.randperm(len(neg_indices), device=device)[:num_keep]
                    loss_cls = F.binary_cross_entropy_with_logits(pred_cls_flat[perm], target_cls_flat[perm])
                else:
                    loss_cls = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 2. CE Loss
            loss_ce_raw = criterion_ce(pred_token_logits, target_token)
            mask_cls = (target_cls > 0.5).float().squeeze(1) 
            
            # High weight for positives to ensure Green
            weights = torch.ones_like(mask_cls) + mask_cls * 20.0 
            loss_ce = (loss_ce_raw * weights).mean()

            # 3. MSE Loss
            pred_token_logits = torch.clamp(pred_token_logits, min=-20, max=20)
            z_probs = F.softmax(pred_token_logits, dim=1) 
            
            codebook = vae.model.quantize.embedding.weight
            
            z_probs_perm = z_probs.permute(0, 2, 3, 1)
            z_soft = torch.matmul(z_probs_perm, codebook) 
            z_soft = z_soft.permute(0, 3, 1, 2) 
            
            z_soft = vae.model.post_quant_conv(z_soft) 
            recon_img = vae.model.decoder(z_soft)
            
            with torch.no_grad():
                z_gt = F.embedding(target_token, codebook).permute(0, 3, 1, 2)
                z_gt = vae.model.post_quant_conv(z_gt) 
                target_img_recon = vae.model.decoder(z_gt)
            
            loss_mse = criterion_mse(recon_img, target_img_recon)
            
            # Total Loss
            loss = loss_cls + loss_ce + args.reg_weight * loss_mse
            loss = loss / accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # --- Accumulate Statistics (Use .item() to get float value) ---
            total_loss += loss.item() * accumulation_steps
            total_loss_cls += loss_cls.item()
            total_loss_ce += loss_ce.item()
            total_loss_mse += loss_mse.item()
            
        if is_master:
            num_batches = len(train_loader)
            avg_loss = total_loss / num_batches
            avg_cls = total_loss_cls / num_batches
            avg_ce = total_loss_ce / num_batches
            avg_mse = total_loss_mse / num_batches
            
            print(f"Epoch {epoch+1}: Total={avg_loss:.4f} | Cls={avg_cls:.4f} | CE={avg_ce:.4f} | MSE={avg_mse:.4f}")
            
            # Save latest checkpoint
            os.makedirs("pred_model", exist_ok=True)
            state_dict = model.module.state_dict() if is_ddp else model.state_dict()
            torch.save(state_dict, "pred_model/latest_model.pth")
            
            # --- Evaluation every 5 epochs ---
            if (epoch + 1) % 5 == 0:
                print(f"Running evaluation on validation set...")
                val_cls_acc, val_token_acc = evaluate_and_visualize(
                    model, vae, depth_model, val_loader, device, 
                    output_dir="vis_results_val", epoch=epoch+1, 
                    neighbor_mask=neighbor_mask, 
                    save_images=True, max_save_count=4 # Only save 4 images
                )
                
                print(f"Epoch {epoch+1} Validation: Cls Acc={val_cls_acc:.4f} | Token Acc={val_token_acc:.4f}")
                
                if val_token_acc > best_token_acc:
                    best_token_acc = val_token_acc
                    print(f"New best model found! Saving to pred_model/best_model.pth")
                    torch.save(state_dict, "pred_model/best_model.pth")

    # --- Final Step: Load Best Model and Save All Images ---
    if is_master:
        print("Training finished. Loading best model for final visualization...")
        best_model_path = "pred_model/best_model.pth"
        if os.path.exists(best_model_path):
            # Load best weights
            if is_ddp:
                model.module.load_state_dict(torch.load(best_model_path))
            else:
                model.load_state_dict(torch.load(best_model_path))
            
            final_vis_dir = "/data/cliang/mineworld/final_vis_results" # Or args.final_output_dir
            print(f"Saving all validation images to {final_vis_dir}...")
            evaluate_and_visualize(
                model, vae, depth_model, val_loader, device, 
                output_dir=final_vis_dir, epoch="best", 
                neighbor_mask=neighbor_mask, 
                save_images=True, max_save_count=None # Save ALL images
            )
            print("Done.")

    if is_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--action_dir", type=str, default="/data/cliang/mineworld/dataset/actions")
    parser.add_argument("--label_dir", type=str, default="/data/cliang/mineworld/misalignment_dataset_labels")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_weight", type=float, default=10.0) 
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_neighbor_mask", action="store_true", help="Whether to use neighbor mask for training")
    
    args = parser.parse_args()
    train(args)