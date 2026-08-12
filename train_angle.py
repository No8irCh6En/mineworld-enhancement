import os
import sys
import json
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add tools/ and util/ directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'util'))

try:
    from util.depth_anything.dpt import DepthAnything
    from util.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
except ImportError:
    print("Could not import DepthAnything utils. Make sure 'util' folder is in the path.")
    sys.exit(1)

# --- 1. Data Processing Utils ---

def get_depth_transform(height=224, width=384):
    return Compose([
        Resize(
            width=width,
            height=height,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

def process_action(action_dict):
    """
    Convert action dictionary to a flat vector.
    Keys: ESC, back, drop, forward, hotbar.1-9, inventory, jump, left, right, sneak, sprint, swapHands, camera, attack, use, pickItem
    """
    # Binary keys (14 + 9 = 23)
    binary_keys = [
        "ESC", "back", "drop", "forward", "inventory", "jump", "left", "right", 
        "sneak", "sprint", "swapHands", "attack", "use", "pickItem"
    ]
    hotbar_keys = [f"hotbar.{i}" for i in range(1, 10)]
    
    vec = []
    for k in binary_keys:
        vec.append(float(action_dict.get(k, 0)))
    for k in hotbar_keys:
        vec.append(float(action_dict.get(k, 0)))
        
    # Camera (2 floats)
    cam = action_dict.get("camera", [0.0, 0.0])
    vec.extend(cam)
    
    return np.array(vec, dtype=np.float32) # Size: 23 + 2 = 25

# --- 2. Dataset ---

class MisalignmentDataset(Dataset):
    def __init__(self, dataset_dir, labels_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.samples = [] # List of (image_path, action_dict, label_frame_idx, label_path)

        # Discover episodes
        label_files = sorted(glob.glob(os.path.join(labels_dir, "*_labels.npy")))
        
        print(f"Found {len(label_files)} label files.")
        
        for l_path in label_files:
            ep_name = os.path.basename(l_path).replace("_labels.npy", "")
            ep_dir = os.path.join(dataset_dir, ep_name)
            action_path = os.path.join(ep_dir, "action.jsonl")
            
            if not os.path.exists(ep_dir) or not os.path.exists(action_path):
                continue
                
            # Load Actions
            actions = []
            with open(action_path, 'r') as f:
                for line in f:
                    actions.append(json.loads(line))
            
            # Load Labels to get length
            # We don't load the full numpy array here to save RAM, just map indices
            # But we need to know T. Let's assume we can memmap or load once.
            # For simplicity, let's trust the file existence and image count.
            
            image_files = sorted(glob.glob(os.path.join(ep_dir, "image_*.png")))
            if not image_files:
                image_files = sorted(glob.glob(os.path.join(ep_dir, "*.png")))
            
            # Label shape is (T, 14, 24, 5). T = len(images) - 1 usually.
            # We need to match index i of label to image i and action i.
            
            # Verify length
            # We will load the label file on demand or cache it if memory allows.
            # For training speed, let's pre-load labels if dataset is small (<10GB).
            # If large, use memmap.
            try:
                labels = np.load(l_path, mmap_mode='r')
                T = labels.shape[0]
            except:
                print(f"Error loading {l_path}")
                continue

            # Check consistency
            # We need image_t and action_t for label_t
            if len(image_files) <= T or len(actions) <= T:
                # print(f"Skipping {ep_name}: Imgs={len(image_files)}, Acts={len(actions)}, Labels={T}")
                continue
                
            for t in range(T):
                self.samples.append({
                    'image_path': image_files[t],
                    'action': actions[t],
                    'label_path': l_path,
                    'label_idx': t
                })

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load Image
        img = cv2.imread(item['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        if self.transform:
            img = self.transform({'image': img})['image'] # (3, H, W) tensor
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        # Load Action
        action_vec = process_action(item['action'])
        
        # Load Label
        # Using mmap to read specific slice
        labels_mmap = np.load(item['label_path'], mmap_mode='r')
        label_data = labels_mmap[item['label_idx']] # (14, 24, 5)
        # label_data: [recoverable, dx, dy, angle, target_token_id]
        
        # We only need [recoverable, dx, dy, angle] for regression
        # Shape: (14, 24, 4)
        target = label_data[:, :, :4].astype(np.float32)
        
        return img, action_vec, target

# --- 3. Model ---

class AnglePredictor(nn.Module):
    def __init__(self, depth_encoder_path='/data/cliang/depth_anything_vits14', action_dim=25):
        super().__init__()
        
        # 1. Depth Encoder (Frozen)
        print(f"Loading DepthAnything from {depth_encoder_path}...")
        self.depth_model = DepthAnything.from_pretrained(depth_encoder_path)
        # Freeze depth model
        for param in self.depth_model.parameters():
            param.requires_grad = False
        self.depth_model.eval()
        
        # DepthAnything ViT-S outputs features? 
        # We will use the intermediate features or the final depth map?
        # The user wants "based on my depth estimator". 
        # Let's use the depth map as a strong feature, plus maybe some intermediate features if possible.
        # For simplicity and speed, let's run the depth model to get the depth map (1 channel), 
        # then use a small CNN to process it.
        
        # 2. Action Encoder
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 3. Prediction Head
        # Input: Depth Map (1, 224, 384) -> Downsample to (C, 14, 24)
        # 224/14 = 16, 384/24 = 16. Stride 16 total.
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 112x192
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 56x96
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 28x48
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 14x24
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        
        # Fusion: Concatenate Action embedding (expanded) to Conv features
        # Conv out: (256, 14, 24)
        # Action emb: (128) -> expand to (128, 14, 24)
        # Concat: (384, 14, 24)
        
        self.head = nn.Sequential(
            nn.Conv2d(256 + 128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1) # Output: [confidence, dx, dy, angle]
        )

    def forward(self, img, action):
        # img: (B, 3, 224, 384)
        # action: (B, 25)
        
        # 1. Get Depth
        with torch.no_grad():
            depth = self.depth_model(img) # (B, H, W)
            depth = depth.unsqueeze(1) # (B, 1, H, W)
            # Normalize depth roughly to 0-1 for stability
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        # 2. Process Depth
        feat_map = self.conv_net(depth) # (B, 256, 14, 24)
        
        # 3. Process Action
        act_emb = self.action_mlp(action) # (B, 128)
        act_emb = act_emb.unsqueeze(-1).unsqueeze(-1) # (B, 128, 1, 1)
        act_emb = act_emb.expand(-1, -1, 14, 24) # (B, 128, 14, 24)
        
        # 4. Fuse
        fused = torch.cat([feat_map, act_emb], dim=1) # (B, 384, 14, 24)
        
        # 5. Predict
        out = self.head(fused) # (B, 4, 14, 24)
        
        # Split output
        # Channel 0: Logits for recoverability (Sigmoid needed later)
        # Channel 1,2,3: dx, dy, angle
        
        return out

# --- 4. Training Loop ---

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    transform = get_depth_transform()
    dataset = MisalignmentDataset(args.dataset_dir, args.labels_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Model
    model = AnglePredictor(depth_encoder_path=args.depth_ckpt).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss(reduction='none')
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_conf_loss = 0
        total_reg_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, actions, targets in pbar:
            imgs = imgs.to(device)
            actions = actions.to(device)
            targets = targets.to(device) # (B, 14, 24, 4) -> [rec, dx, dy, angle]
            
            # Forward
            preds = model(imgs, actions) # (B, 4, 14, 24)
            preds = preds.permute(0, 2, 3, 1) # (B, 14, 24, 4)
            
            pred_conf_logits = preds[..., 0]
            pred_reg = preds[..., 1:]
            
            gt_conf = targets[..., 0]
            gt_reg = targets[..., 1:]
            
            # 1. Confidence Loss (Binary Classification)
            loss_conf = bce_loss(pred_conf_logits, gt_conf)
            
            # 2. Regression Loss (Masked)
            # Only calculate regression loss where gt_conf == 1 (recoverable)
            mask = (gt_conf > 0.5).unsqueeze(-1) # (B, 14, 24, 1)
            
            loss_reg_raw = mse_loss(pred_reg, gt_reg) # (B, 14, 24, 3)
            loss_reg = (loss_reg_raw * mask).sum() / (mask.sum() + 1e-6)
            
            # Total Loss
            loss = loss_conf + args.reg_weight * loss_reg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_conf_loss += loss_conf.item()
            total_reg_loss += loss_reg.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'conf': f"{loss_conf.item():.4f}", 
                'reg': f"{loss_reg.item():.4f}"
            })
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--labels_dir", type=str, default="/data/cliang/mineworld/misalignment_dataset_labels_with_rot")
    parser.add_argument("--depth_ckpt", type=str, default="/data/cliang/depth_anything_vits14")
    parser.add_argument("--output_dir", type=str, default="checkpoints_angle")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reg_weight", type=float, default=1.0, help="Weight for regression loss")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
    
if __name__ == "__main__":
    main()
    