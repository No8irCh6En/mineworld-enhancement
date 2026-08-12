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
import random
from collections import defaultdict
import re  # 添加这行
import contextlib # [新增] 用于屏蔽输出
import io         # [新增] 用于屏蔽输出


from mcdataset import MCDataset

from util.DepthAnythingWrapper import DepthAnythingWrapper, DEPTH_ANYTHING_TRANSFORM


from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from common_metrics_on_video_quality.calculate_lpips import calculate_lpips
from common_metrics_on_video_quality.calculate_ssim import calculate_ssim_parallel
from common_metrics_on_video_quality.calculate_psnr import calculate_psnr

# --- Dataset ---
class MultiModalDataset(Dataset):
    def __init__(self, image_dir, action_dir, label_dir, max_episodes=-1, cache=True, specific_episodes=None):
        self.samples = []
        self.cache = cache
        self.mc_helper = MCDataset() 
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Scanning for actions in {action_dir}...")

        if specific_episodes is not None:
            label_files = [os.path.join(label_dir, f"{ep}_labels.npy") for ep in specific_episodes]
            # Filter out non-existent files
            label_files = [f for f in label_files if os.path.exists(f)]
        else:
            label_files = sorted(glob.glob(os.path.join(label_dir, "*_labels.npy")))
            
            # Randomly select a subset of episodes if requested
            if max_episodes > 0 and len(label_files) > max_episodes:
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"Selecting {max_episodes} episodes out of {len(label_files)}...")
                # Use a fixed seed so all DDP processes select the SAME subset
                random.seed(42) 
                random.shuffle(label_files)
                label_files = label_files[:max_episodes]
        
        for lf in label_files:
            ep_name = os.path.basename(lf).replace("_labels.npy", "")
            act_file = os.path.join(action_dir, ep_name, "action.jsonl")
            ep_actions = []
            
            if os.path.exists(act_file):
                try:
                    with open(act_file, 'r') as f:
                        for line in f:
                            try:
                                json_action = json.loads(line)
                                env_action, _ = self.mc_helper.json_action_to_env_action(json_action)
                                vec = [
                                    float(env_action['forward']), float(env_action['back']),
                                    float(env_action['left']), float(env_action['right']),
                                    float(env_action['jump']), float(env_action['sneak']),
                                    float(env_action['sprint']),
                                    float(env_action['camera'][0]), float(env_action['camera'][1]),
                                    float(env_action['attack']), float(env_action['use'])
                                ]
                                ep_actions.append(np.array(vec, dtype=np.float32))
                            except:
                                ep_actions.append(np.zeros(11, dtype=np.float32))
                except:
                    pass
            
            labels = np.load(lf) 
            img_folder = os.path.join(image_dir, ep_name)
            if not os.path.isdir(img_folder): continue
            
            images = sorted(glob.glob(os.path.join(img_folder, "image_*.png")))
            if not images: images = sorted(glob.glob(os.path.join(img_folder, "*.png")))
            
            images.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', os.path.basename(x))])
            
            valid_len = min(len(images), len(labels))
            
            for i in range(valid_len):
                act = ep_actions[i] if i < len(ep_actions) else np.zeros(11, dtype=np.float32)
                self.samples.append({
                    "img_path": images[i],
                    "action": act,
                    "label": labels[i]
                })
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Total samples loaded: {len(self.samples)}")

    def _load_item(self, idx):
        item = self.samples[idx]
        img_bgr = cv2.imread(item['img_path'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] != 224 or img_rgb.shape[1] != 384:
            img_rgb = cv2.resize(img_rgb, (384, 224))
            
        img_norm = img_rgb.astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)
        
        img_01 = (img_norm + 1.0) * 0.5
        depth_input = DEPTH_ANYTHING_TRANSFORM({'image': img_01})['image']
        depth_input_tensor = torch.from_numpy(depth_input)
        
        action_vec = torch.from_numpy(item['action'])
        
        label = torch.from_numpy(item['label'])
        target_cls = label[:, :, 0].unsqueeze(0) 
        target_token = label[:, :, 1].long()     
        
        return img_tensor, depth_input_tensor, action_vec, target_cls, target_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._load_item(idx)

# --- Helper Dataset for Video Evaluation ---
class VideoEvalDataset(Dataset): # Inherit from Dataset
    def __init__(self, image_dir, action_dir, label_dir, episodes):
        self.episodes = defaultdict(list)
        self.mc_helper = MCDataset()
        
        # New: Flattened list for __getitem__ access
        self.flattened_samples = []
        
        for ep_name in episodes:
            act_file = os.path.join(action_dir, ep_name, "action.jsonl")
            ep_actions = []
            if os.path.exists(act_file):
                try:
                    with open(act_file, 'r') as f:
                        for line in f:
                            try:
                                json_action = json.loads(line)
                                env_action, _ = self.mc_helper.json_action_to_env_action(json_action)
                                vec = [
                                    float(env_action['forward']), float(env_action['back']),
                                    float(env_action['left']), float(env_action['right']),
                                    float(env_action['jump']), float(env_action['sneak']),
                                    float(env_action['sprint']),
                                    float(env_action['camera'][0]), float(env_action['camera'][1]),
                                    float(env_action['attack']), float(env_action['use'])
                                ]
                                ep_actions.append(np.array(vec, dtype=np.float32))
                            except:
                                ep_actions.append(np.zeros(11, dtype=np.float32))
                except:
                    pass
            
            img_folder = os.path.join(image_dir, ep_name)
            if not os.path.isdir(img_folder): continue
            images = sorted(glob.glob(os.path.join(img_folder, "image_*.png")))
            if not images: images = sorted(glob.glob(os.path.join(img_folder, "*.png")))
            images.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', os.path.basename(x))])

            # We don't strictly need labels for generation, just images and actions
            # But for __getitem__ validation loop, we might want labels if available, OR just return dummy labels.
            # Let's check for labels to support evaluation loop
            label_path = os.path.join(label_dir, f"{ep_name}_labels.npy")
            labels = np.load(label_path) if os.path.exists(label_path) else None

            valid_len = min(len(images), len(ep_actions))
            if labels is not None:
                valid_len = min(valid_len, len(labels))
            
            for i in range(valid_len):
                sample_dict = {
                    "img_path": images[i],
                    "action": ep_actions[i],
                    "frame_idx": i,
                    # [修改 1] 明确指定 dummy label 为 int64，防止 np.zeros 默认为 float64
                    "label": labels[i] if labels is not None else np.zeros((14,24,2), dtype=np.int64), 
                    "ep_name": ep_name 
                }
                self.episodes[ep_name].append(sample_dict)
                self.flattened_samples.append(sample_dict)
                
            self.episodes[ep_name].sort(key=lambda x: x['frame_idx'])

    def load_image_tensor(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] != 224 or img_rgb.shape[1] != 384:
            img_rgb = cv2.resize(img_rgb, (384, 224))
        img_norm = img_rgb.astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1) # (3, H, W)
        return img_tensor

    def __len__(self):
        return len(self.flattened_samples)

    def __getitem__(self, idx):
        item = self.flattened_samples[idx]
        
        # Load Image
        img_tensor = self.load_image_tensor(item['img_path'])
        
        # Load Depth
        # Re-using the manual logic inside load_image_tensor for depth would be cleaner,
        # but let's replicate logic or refactor. Here replicating logic for simplicity of fix.
        # transform expects [0, 1] numpy
        img_np_01 = (img_tensor.permute(1, 2, 0).numpy() + 1.0) * 0.5
        depth_input = DEPTH_ANYTHING_TRANSFORM({'image': img_np_01})['image']
        depth_input_tensor = torch.from_numpy(depth_input)
        
        action_vec = torch.from_numpy(item['action'])
        
        # Label handling
        # Assuming label format matches MultiModalDataset: (H, W, 2)
        label_np = item['label']
        if label_np.ndim == 3:
             target_cls = torch.from_numpy(label_np[:, :, 0]).unsqueeze(0)
             # [修改 2] 再次确认这里使用了 .long()
             target_token = torch.from_numpy(label_np[:, :, 1]).long()
        else:
             target_cls = torch.zeros((1, 14, 24))
             target_token = torch.zeros((14, 24), dtype=torch.long)

        # 为了兼容 train_uncertainty.py 中的 validation loop 解包
        # 解包期望: (img, depth, action, cls, token, [prev_stuff...]_dummy)
        # 实际上 val_loader 使用的是标准 MultiModalDataset 格式或者 VideoEvalDataset
        # 如果是 VideoEvalDataset，我们需要确保它返回的格式能被 loop 正确解包。
        
        # Loop expectes 8 items if sequential logic is present or checking len.
        # But wait, in train_uncertainty.py:
        # for batch_data in val_loader:
        #    (img, depth, act, tgt, _, _, _, _) = batch_data
        
        # So we must return 8 items, with the last 4 being dummy/None.
        
        return (img_tensor, depth_input_tensor, action_vec, target_cls, target_token, 
                torch.empty(0), torch.empty(0), torch.empty(0)) # 3 dummies + 1 dummy flag not needed if unpacked correctly


# --- [修改] Sequential Dataset for Self-Forcing ---
class SequentialMultiModalDataset(Dataset):
    def __init__(self, image_dir, action_dir, label_dir, max_episodes=-1, specific_episodes=None):
        self.samples = []
        self.mc_helper = MCDataset() 
        
        # 收集所有 Label 文件
        if specific_episodes is not None:
            label_files = [os.path.join(label_dir, f"{ep}_labels.npy") for ep in specific_episodes]
            label_files = [f for f in label_files if os.path.exists(f)]
        else:
            label_files = sorted(glob.glob(os.path.join(label_dir, "*_labels.npy")))
            if max_episodes > 0 and len(label_files) > max_episodes:
                random.seed(42) 
                random.shuffle(label_files)
                label_files = label_files[:max_episodes]
        
        # 遍历构建样本列表
        for lf in label_files:
            ep_name = os.path.basename(lf).replace("_labels.npy", "")
            act_file = os.path.join(action_dir, ep_name, "action.jsonl")
            ep_actions = []
            
            # 加载 Action
            if os.path.exists(act_file):
                try:
                    with open(act_file, 'r') as f:
                        for line in f:
                            try:
                                json_action = json.loads(line)
                                env_action, _ = self.mc_helper.json_action_to_env_action(json_action)
                                vec = [
                                    float(env_action['forward']), float(env_action['back']),
                                    float(env_action['left']), float(env_action['right']),
                                    float(env_action['jump']), float(env_action['sneak']),
                                    float(env_action['sprint']),
                                    float(env_action['camera'][0]), float(env_action['camera'][1]),
                                    float(env_action['attack']), float(env_action['use'])
                                ]
                                ep_actions.append(np.array(vec, dtype=np.float32))
                            except:
                                ep_actions.append(np.zeros(11, dtype=np.float32))
                except:
                    pass
            
            labels = np.load(lf) 
            img_folder = os.path.join(image_dir, ep_name)
            if not os.path.isdir(img_folder): continue
            
            # 排序 images
            images = sorted(glob.glob(os.path.join(img_folder, "image_*.png")))
            if not images: images = sorted(glob.glob(os.path.join(img_folder, "*.png")))
            images.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', os.path.basename(x))])
            
            # [核心修改]
            # 我们需要预测下一帧 (Next prediction)，所以有效长度要减 1
            # 样本 i 的定义:
            #   Input: Image[i]
            #   Target: Label[i+1] (Next Frame)
            #   Prev: Image[i-1]
            valid_len = min(len(images), len(labels), len(ep_actions))
            
            # i 是 Current Input 的索引
            # 范围: 从 0 到 valid_len - 2 (保证 i+1 存在)
            for i in range(valid_len - 1):
                # Target 是 i+1
                next_label = labels[i+1] 
                next_img_path = images[i+1]
                
                # Current Input (i)
                curr_img_path = images[i]
                curr_act = ep_actions[i]
                
                # Prev Input (i-1)
                if i > 0:
                    prev_img_path = images[i-1]
                    prev_act = ep_actions[i-1]
                    has_prev = True
                else:
                    # 第一帧没有上一帧，Dummy fallback
                    prev_img_path = curr_img_path 
                    prev_act = curr_act
                    has_prev = False

                self.samples.append({
                    "img_path": curr_img_path,
                    "action": curr_act,
                    "prev_img_path": prev_img_path,
                    "prev_action": prev_act,
                    "has_prev": has_prev,
                    "target_label": next_label, # [修改] 存储下一帧的 Label
                    "target_path": next_img_path
                })

    def _load_image(self, path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # 简单的错误处理
            img_bgr = np.zeros((224, 384, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if img_rgb.shape[0] != 224 or img_rgb.shape[1] != 384:
            img_rgb = cv2.resize(img_rgb, (384, 224))
            
        img_norm = img_rgb.astype(np.float32) / 127.5 - 1.0 # [-1, 1]
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1) # (3, H, W)
        
        # Depth Input [0, 1]
        img_01 = (img_norm + 1.0) * 0.5
        depth_input = DEPTH_ANYTHING_TRANSFORM({'image': img_01})['image'] # numpy (H, W)
        depth_input_tensor = torch.from_numpy(depth_input)
        
        return img_tensor, depth_input_tensor

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Current Step (t) Input [Teacher Forcing Input / Replaced by Pred in Stage 2]
        img_curr, depth_curr = self._load_image(item['img_path'])
        action_curr = torch.from_numpy(item['action'])
        
        target_img, target_depth = self._load_image(item['target_path'])
        
        # 2. Target (t+1)
        label = torch.from_numpy(item['target_label'])
        target_token = label[:, :, 1].long()
        
        # 3. Previous Step (t-1) Input [Source of Self-Forcing Rollout]
        if item['has_prev']:
            img_prev, depth_prev = self._load_image(item['prev_img_path'])
            action_prev = torch.from_numpy(item['prev_action'])
            has_prev = 1.0 
        else:
            # Padding
            img_prev, depth_prev = img_curr.clone(), depth_curr.clone()
            action_prev = action_curr.clone()
            has_prev = 0.0
        

        return (img_curr, depth_curr, action_curr, 
                img_prev, depth_prev, action_prev, has_prev,
                target_token, target_img, target_depth)

    def __len__(self):
        return len(self.samples)




# --- Visualization & Evaluation ---
def evaluate_and_visualize(model, vae, depth_model, loader, device, output_dir, epoch, neighbor_mask=None, save_images=False, max_save_count=None):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    total_cls_acc = 0
    total_token_acc = 0
    total_samples = 0
    saved_count = 0 
    
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
            
            if neighbor_mask is not None:
                 input_token_ids = vae.tokenize_images(img_tensor)
                 batch_mask = F.embedding(input_token_ids, neighbor_mask)
                 batch_mask = batch_mask.permute(0, 3, 1, 2)
                 pred_token_logits = pred_token_logits + (1.0 - batch_mask) * -1e9
            
            pred_token_ids = torch.argmax(pred_token_logits, dim=1) 
            pred_cls_binary = (torch.sigmoid(pred_cls) > 0.5).float()
            
            cls_correct = (pred_cls_binary == target_cls).float().sum()
            total_cls_acc += cls_correct.item()
            
            token_correct_mask = (pred_token_ids == target_token)
            total_token_acc += token_correct_mask.float().sum().item()
            
            total_samples += target_cls.numel()
            
            if save_images and (max_save_count is None or saved_count < max_save_count):
                codebook = vae.model.quantize.embedding.weight
                z_q_target = F.embedding(target_token, codebook).permute(0, 3, 1, 2)
                z_q_target = vae.model.post_quant_conv(z_q_target) 
                decoded_target = vae.model.decoder(z_q_target)
                
                z_q_pred = F.embedding(pred_token_ids, codebook).permute(0, 3, 1, 2)
                z_q_pred = vae.model.post_quant_conv(z_q_pred) 
                decoded_pred = vae.model.decoder(z_q_pred)
                
                mask = pred_cls_binary
                z_raw_pred = F.embedding(pred_token_ids, codebook).permute(0, 3, 1, 2)
                z_raw_target = F.embedding(target_token, codebook).permute(0, 3, 1, 2)
                z_merged = mask * z_raw_pred + (1 - mask) * z_raw_target
                z_merged = vae.model.post_quant_conv(z_merged) 
                decoded_merged = vae.model.decoder(z_merged)

                for i in range(img_tensor.size(0)):
                    if max_save_count is not None and saved_count >= max_save_count: break

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
                    conf_map[hit_mask & (t_correct==1)] = [0, 1, 0]
                    conf_map[hit_mask & (t_correct==0)] = [1, 1, 0]
                    conf_map[(p_cls==1) & (t_cls==0)] = [1, 0, 0]
                    conf_map[(p_cls==0) & (t_cls==1)] = [0, 0, 1]
                    
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
                    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{i}.png"))
                    plt.close()
                    saved_count += 1

    avg_cls_acc = total_cls_acc / total_samples
    avg_token_acc = total_token_acc / total_samples
    return avg_cls_acc, avg_token_acc

# --- Video Metrics Evaluation ---
def evaluate_video_metrics(model, vae, depth_model, episode_names, image_dir, action_dir, label_dir, device, seq_len=14, prefix="Val"):
    print(f"\n[INFO] Starting Video Evaluation on {prefix} set ({len(episode_names)} episodes)...")
    
    # Helper functions
    def get_depth_input(img_tensor):
        # img_tensor: (C, H, W) in [-1, 1]
        img_01 = (img_tensor.permute(1, 2, 0).cpu().numpy() + 1.0) * 0.5
        depth_input = DEPTH_ANYTHING_TRANSFORM({'image': img_01})['image']
        return torch.from_numpy(depth_input).unsqueeze(0) # (1, H, W)

    def decode_tokens(vae, token_ids):
        codebook = vae.model.quantize.embedding.weight
        z = F.embedding(token_ids, codebook).permute(0, 3, 1, 2)
        decoded = vae.model.decoder(vae.model.post_quant_conv(z))
        return decoded # (B, 3, H, W) in [-1, 1]

    dataset = VideoEvalDataset(image_dir, action_dir, label_dir, episode_names)
    
    all_gt_videos = []
    all_tf_videos = []
    all_tf_merged_videos = [] # [新增]
    all_ar_videos = []
    all_ar_merged_videos = [] # [新增]

    # [新增] 特征相似度统计容器
    feature_metrics = defaultdict(list)
    
    with torch.no_grad():
        # for ep_name in tqdm(episode_names, desc=f"Generating Videos ({prefix})"):
        for ep_name in episode_names:
            frames_data = dataset.episodes[ep_name]
            if len(frames_data) < seq_len + 1: continue
            
            clip_data = frames_data[:seq_len + 1]
            ep_gt_frames = []
            ep_tf_frames = []
            ep_tf_merged_frames = [] # [新增]
            ep_ar_frames = []
            ep_ar_merged_frames = [] # [新增]
            
            # Initial State for AR
            curr_ar_img = dataset.load_image_tensor(clip_data[0]['img_path']).to(device).unsqueeze(0)
            
            for t in range(seq_len):
                curr_gt_img = dataset.load_image_tensor(clip_data[t]['img_path']).to(device).unsqueeze(0)
                next_gt_img = dataset.load_image_tensor(clip_data[t+1]['img_path']).to(device).unsqueeze(0) # Next Frame GT
                action_vec = torch.from_numpy(clip_data[t]['action']).to(device).unsqueeze(0)
                
                # --- Pre-calculate GT Feature of Next Frame (Target) ---
                # 我们想知道模型的 f_pred 是否逼近了 f_next_gt
                # 技巧：把 Next GT Image 当作 Input 喂进去，取出的 f_curr 就是它的特征
                depth_input_next = get_depth_input(next_gt_img.squeeze(0)).to(device)
                depth_map_next = depth_model(depth_input_next).unsqueeze(1)
                depth_map_next = (depth_map_next - depth_map_next.min()) / (depth_map_next.max() - depth_map_next.min() + 1e-6)
                input_next_gt_for_feat = torch.cat([next_gt_img, depth_map_next], dim=1)
                
                # Forward Next GT to get "Ideal Features"
                # Action 不重要，因为 f_curr 是在 action 注入之前提取的
                _, _, feats_target = model(input_next_gt_for_feat, action_vec, return_features=True)
                f_target_gt = feats_target['f_curr'].view(1, -1) # Flatten (The "Ground Truth" Feature)

                # 1. Teacher Forcing (Run on GT Input)
                depth_input_tf = get_depth_input(curr_gt_img.squeeze(0)).to(device)
                depth_map_tf = depth_model(depth_input_tf).unsqueeze(1)
                depth_map_tf = (depth_map_tf - depth_map_tf.min()) / (depth_map_tf.max() - depth_map_tf.min() + 1e-6)
                input_tf = torch.cat([curr_gt_img, depth_map_tf], dim=1)
                
                # [修改] 捕获 conf logits
                pred_conf_logits_tf, pred_token_logits_tf, feats_tf = model(input_tf, action_vec, return_features=True)
                
                # Raw Pred
                pred_img_tf_raw = decode_tokens(vae, torch.argmax(pred_token_logits_tf, dim=1))

                # [新增] TF Merged Calculation
                # TF 模式下的 Merged: Pred * Conf + GT(Current) * (1-Conf)
                conf_tf = torch.sigmoid(pred_conf_logits_tf)
                conf_tf_img = F.interpolate(conf_tf, size=(224, 384), mode='bilinear', align_corners=False)
                pred_img_tf_merged = pred_img_tf_raw * conf_tf_img + curr_gt_img * (1.0 - conf_tf_img)
                
                # 2. Autoregressive (Run on AR Input)
                depth_input_ar = get_depth_input(curr_ar_img.squeeze(0)).to(device)
                depth_map_ar = depth_model(depth_input_ar).unsqueeze(1)
                depth_map_ar = (depth_map_ar - depth_map_ar.min()) / (depth_map_ar.max() - depth_map_ar.min() + 1e-6)
                input_ar = torch.cat([curr_ar_img, depth_map_ar], dim=1)
                
                # [修改] 捕获 conf logits
                pred_conf_logits_ar, pred_token_logits_ar, feats_ar = model(input_ar, action_vec, return_features=True)
                
                # Raw Pred
                pred_img_ar_raw = decode_tokens(vae, torch.argmax(pred_token_logits_ar, dim=1))
                
                # [新增] AR Merged Calculation
                # AR 模式下的 Merged: Pred * Conf + AR(Prev_Output) * (1-Conf)
                conf_ar = torch.sigmoid(pred_conf_logits_ar)
                conf_ar_img = F.interpolate(conf_ar, size=(224, 384), mode='bilinear', align_corners=False)
                pred_img_ar_merged = pred_img_ar_raw * conf_ar_img + curr_ar_img * (1.0 - conf_ar_img)

                # [关键修正] AR 的下一步输入使用 Merged 结果
                curr_ar_img = pred_img_ar_merged
                
                # Saving frames
                ep_gt_frames.append((next_gt_img.cpu() + 1) * 0.5)
                ep_tf_frames.append((pred_img_tf_raw.cpu() + 1) * 0.5)
                ep_tf_merged_frames.append((pred_img_tf_merged.cpu() + 1) * 0.5) # [新增]
                ep_ar_frames.append((pred_img_ar_raw.cpu() + 1) * 0.5)
                ep_ar_merged_frames.append((pred_img_ar_merged.cpu() + 1) * 0.5) # [新增]

                # --- [新增] 计算 PREDICTION vs REAL FUTURE 特征差异 ---
                # Q: 模型预测出的 t+1 特征 (f_pred)，和 真实的 t+1 特征 像不像？
                
                # AR 路径产生的预测特征 (Prediction from Noisy Input)
                f_pred_ar = feats_ar['f_pred'].view(1, -1) 
                
                # TF 路径产生的预测特征 (Prediction from Clean Input)
                f_pred_tf = feats_tf['f_pred'].view(1, -1)

                cos_pred_vs_future_ar = F.cosine_similarity(f_pred_ar, f_target_gt).item()
                mse_pred_vs_future_ar = F.mse_loss(f_pred_ar, f_target_gt).item()
                cos_pred_vs_future_tf = F.cosine_similarity(f_pred_tf, f_target_gt).item()
                mse_pred_vs_future_tf = F.mse_loss(f_pred_tf, f_target_gt).item()
                cos_pred_ar_vs_tf_pred = F.cosine_similarity(f_pred_ar, f_pred_tf).item()
                mse_pred_ar_vs_tf_pred = F.mse_loss(f_pred_ar, f_pred_tf).item()

                feature_metrics["pred_vs_future_ar_cos"].append(cos_pred_vs_future_ar)
                feature_metrics["pred_vs_future_ar_mse"].append(mse_pred_vs_future_ar)
                feature_metrics["pred_vs_future_tf_cos"].append(cos_pred_vs_future_tf)
                feature_metrics["pred_vs_future_tf_mse"].append(mse_pred_vs_future_tf)
                feature_metrics["pred_ar_vs_tf_pred_cos"].append(cos_pred_ar_vs_tf_pred)
                feature_metrics["pred_ar_vs_tf_pred_mse"].append(mse_pred_ar_vs_tf_pred)

            
            all_gt_videos.append(torch.cat(ep_gt_frames, dim=0))
            all_tf_videos.append(torch.cat(ep_tf_frames, dim=0))
            all_tf_merged_videos.append(torch.cat(ep_tf_merged_frames, dim=0)) # [新增]
            all_ar_videos.append(torch.cat(ep_ar_frames, dim=0))
            all_ar_merged_videos.append(torch.cat(ep_ar_merged_frames, dim=0)) # [新增]

    if len(all_gt_videos) == 0:
        print("No valid videos generated.")
        return

    tensor_gt = torch.stack(all_gt_videos, dim=0)
    tensor_tf = torch.stack(all_tf_videos, dim=0)
    tensor_tf_merged = torch.stack(all_tf_merged_videos, dim=0) # [新增]
    tensor_ar = torch.stack(all_ar_videos, dim=0)
    tensor_ar_merged = torch.stack(all_ar_merged_videos, dim=0) # [新增]
    
    # [新增] 辅助函数：计算每帧的平均 PSNR
    def calculate_per_frame_psnr_torch(pred_tensor, gt_tensor):
        """
        Inputs:
            pred_tensor: (B, T, C, H, W) range [0, 1]
            gt_tensor:   (B, T, C, H, W) range [0, 1]
        Returns:
            List[float]: Average PSNR for each time step [t0, t1, ...]
        """
        B, T, C, H, W = pred_tensor.shape
        psnrs_per_step = []
        
        # 确保在同一设备
        device = pred_tensor.device
        if gt_tensor.device != device:
            gt_tensor = gt_tensor.to(device)

        for t in range(T):
            p_t = pred_tensor[:, t] # (B, C, H, W)
            g_t = gt_tensor[:, t]   # (B, C, H, W)
            
            # MSE per sample: (B,)
            mse = ((p_t - g_t) ** 2).mean(dim=[1, 2, 3])
            
            # Handle 0 mse (perfect match)
            mse = torch.clamp(mse, min=1e-10)
            
            # PSNR per sample: (B,)
            # p_max = 1.0
            psnr = 10 * torch.log10(1.0 / mse)
            
            # Average over batch
            avg_psnr_t = psnr.mean().item()
            psnrs_per_step.append(avg_psnr_t)
            
        return psnrs_per_step

    def compute_and_print(name, pred, gt):
        # [新增] 计算逐帧 PSNR
        per_frame_psnr = calculate_per_frame_psnr_torch(pred, gt)

        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            try:
                fvd = calculate_fvd(pred, gt, device, method='styleganv', only_final=False)
                fvd_val = np.mean(list(fvd['value'].values())) if isinstance(fvd['value'], dict) else np.mean(fvd['value'])
            except: fvd_val = -1
            
            try: lpips = np.mean(list(calculate_lpips(pred, gt, device)['value']))
            except: lpips = -1
            
            try: ssim = np.mean(list(calculate_ssim_parallel(pred, gt)['value']))
            except: ssim = -1
            
            try: psnr = np.mean(list(calculate_psnr(pred, gt)['value']))
            except: psnr = -1
        
        print(f"[{name}]")
        print(f"  FVD   : {fvd_val:.4f} (Lower is better)")
        print(f"  LPIPS : {lpips:.4f} (Lower is better)")
        print(f"  SSIM  : {ssim:.4f} (Higher is better)")
        print(f"  PSNR  : {psnr:.4f} (Higher is better)")
        
        # [新增] 打印逐帧趋势
        print("  Per-Frame PSNR Trend:")
        # 每行打印 5 个
        for i in range(0, len(per_frame_psnr), 5):
            chunk = per_frame_psnr[i:i+5]
            chunk_str = " | ".join([f"T{i+j+1}: {val:.2f}" for j, val in enumerate(chunk)])
            print(f"    {chunk_str}")
            
        print("-" * 30)

    print(f"\n{'='*50}")
    print(f"       VIDEO EVALUATION REPORT ({prefix})       ")
    print(f"{'='*50}")
    compute_and_print("TeacherForcing (Raw)", tensor_tf, tensor_gt)
    compute_and_print("TeacherForcing (Merged)", tensor_tf_merged, tensor_gt) # [新增]
    print("-" * 30 + " AR " + "-" * 30)
    compute_and_print("Autoregressive (Raw)", tensor_ar, tensor_gt)
    compute_and_print("Autoregressive (Merged)", tensor_ar_merged, tensor_gt) # [新增]
    print("="*50 + "\n")
    
    # [新增] 打印特征分析报告
    print(f"\n{'='*50}")
    print(f"       INTERNAL FEATURE STABILITY REPORT       ")
    print(f"       Comparing (AR Input Feature) vs (GT Input Feature) ")
    print(f"{'='*50}")
    
    # Group by feature type
    keys = ["f_curr", "f_pred", "combined"]
    desc_map = {
        "f_curr": "Current Visual Enc",
        "f_pred": "Predicted Next Enc (Before Attn)",
        "combined": "Final Fusion (After Attn)"
    }
    
    print(f"{'Layer Name':<30} | {'Cos Sim (Higher Better)':<20} | {'MSE Dist (Lower Better)':<20}")
    print("-" * 80)

    # [新增] 打印 "预测 vs 未来" 报告
    print(f"\n{'='*50}")
    print(f"       PREDICTIVE SEMANTIC ACCURACY REPORT       ")
    print(f"       How well does f_pred match the REAL future feature? ")
    print(f"{'='*50}")
    
    avg_pred_ar_cos = np.mean(feature_metrics["pred_vs_future_ar_cos"])
    avg_pred_ar_mse = np.mean(feature_metrics["pred_vs_future_ar_mse"])
    avg_pred_tf_cos = np.mean(feature_metrics["pred_vs_future_tf_cos"])
    avg_pred_tf_mse = np.mean(feature_metrics["pred_vs_future_tf_mse"])
    avg_pred_ar_vs_tf_cos = np.mean(feature_metrics["pred_ar_vs_tf_pred_cos"])
    avg_pred_ar_vs_tf_mse = np.mean(feature_metrics["pred_ar_vs_tf_pred_mse"])
    
    print(f"{'Metric':<35} | {'Value':<10}")
    print("-" * 50)
    print(f"{'AR Pred vs True Future (Cos)':<35} | {avg_pred_ar_cos:.4f}")
    print(f"{'AR Pred vs True Future (MSE)':<35} | {avg_pred_ar_mse:.5f}")
    print(f"{'TF Pred vs True Future (Cos - Baseline)':<35} | {avg_pred_tf_cos:.4f}")
    print(f"{'TF Pred vs True Future (MSE - Baseline)':<35} | {avg_pred_tf_mse:.5f}")
    print(f"{'AR Pred vs TF Pred (Cos)':<35} | {avg_pred_ar_vs_tf_cos:.4f}")
    print(f"{'AR Pred vs TF Pred (MSE)':<35} | {avg_pred_ar_vs_tf_mse:.5f}")
    
    print("-" * 50)
    print(f"Gap (TF - AR): {avg_pred_tf_cos - avg_pred_ar_cos:.4f}")
    print(f"[Interpretation]: If Gap is large, model struggles to predict correct semantics from noisy AR inputs.")
    print(f"" + "="*50)
