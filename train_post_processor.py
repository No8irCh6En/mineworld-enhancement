import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler # [新增] DDP Sampler
from torch.nn.parallel import DistributedDataParallel as DDP # [新增] DDP Wrapper
import torch.distributed as dist # [新增] Dist
import torchvision.transforms as transforms
import torchvision.models as models
import os
import argparse
from tqdm import tqdm
import sys
import numpy as np
import cv2
import math 

# --- 1. 导入必要的组件 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入 LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    # 只有主进程打印警告
    # print("Warning: 'lpips' library not found. LPIPS metric will be skipped.")

try:
    from util.dataset_reader import SequentialMultiModalDataset
    from util.attn_model import AttentionTokenPredictor
    from vae import VAE
    from util.DepthAnythingWrapper import DepthAnythingWrapper
except ImportError as e:
    print(f"Import Error: {e}. Make sure you are in the root directory.")
    sys.exit(1)

# --- 2. 定义 Refiner 网络 (轻量级 ResNet) ---
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class TextureRefiner(nn.Module):
    def __init__(self, channels=3, num_blocks=9):
        super(TextureRefiner, self).__init__()
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.body = nn.Sequential(*[ResnetBlock(64) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, channels, kernel_size=7),
            nn.Tanh() # Output [-1, 1]
        )

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        correction = self.tail(feat)
        return torch.clamp(x + correction, -1.0, 1.0)

# --- 3. 定义混合 Loss (Pixel + VGG Style) ---
class RefinementLoss(nn.Module):
    def __init__(self, device):
        super(RefinementLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:23].to(device).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def get_features(self, x):
        x = (x + 1.0) * 0.5
        x = (x - self.mean) / self.std
        return self.vgg(x)

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        feat = x.view(b, c, h*w)
        return torch.bmm(feat, feat.transpose(1,2)) / (c*h*w)

    def forward(self, pred, target):
        loss_pixel = F.l1_loss(pred, target)
        
        f_pred = self.get_features(pred)
        f_gt = self.get_features(target)
        loss_content = F.mse_loss(f_pred, f_gt)
        
        g_pred = self.gram_matrix(f_pred)
        g_gt = self.gram_matrix(f_gt)
        loss_style = F.mse_loss(g_pred, g_gt)
        
        return loss_pixel, loss_content, loss_style

# --- Helper: Metrics ---
def calculate_psnr(img1, img2):
    # img1, img2: [-1, 1]
    mse = F.mse_loss(img1, img2)
    if mse == 0: return 100.0
    return 20 * math.log10(2.0 / math.sqrt(mse.item()))

# --- 4. 训练逻辑 ---
def train(args):
    # [DDP 初始化保逻辑持不变]
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_master = (local_rank == 0)

    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save Args
        with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
            f.write(str(args))

    # [模型加载逻辑保持不变]
    # ... (VAE, Depth, AR Model loading) ...
    vae = VAE(args.vae_config, args.vae_ckpt).to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    
    depth_model = DepthAnythingWrapper(device, (384, 224))
    depth_model.eval()
    for p in depth_model.parameters(): p.requires_grad = False
    
    ar_model = AttentionTokenPredictor(input_channels=4, action_dim=11, num_tokens=8192).to(device)
    if args.ar_ckpt and os.path.exists(args.ar_ckpt):
        if is_master: print(f"Loading AR Model from {args.ar_ckpt}")
        st = torch.load(args.ar_ckpt, map_location=device)
        st = {k.replace('module.', ''): v for k, v in st.items()}
        ar_model.load_state_dict(st)
    else:
        if is_master: print("Error: AR Checkpoint not found!")
        dist.destroy_process_group()
        return
    ar_model.eval()
    for p in ar_model.parameters(): p.requires_grad = False

    # [Refiner 初始化]
    refiner = TextureRefiner().to(device)
    refiner = torch.nn.SyncBatchNorm.convert_sync_batchnorm(refiner)
    refiner = DDP(refiner, device_ids=[local_rank], output_device=local_rank)
    
    # [修改] LR 降低一点
    optimizer = optim.Adam(refiner.parameters(), lr=1e-4) # 2e-4 -> 1e-4
    criterion = RefinementLoss(device)
    
    # Dataset & DDP Sampler
    dataset = SequentialMultiModalDataset(args.image_dir, args.action_dir, args.label_dir)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    
    # --- 固定样本预处理 (Fixed Samples) ---
    fixed_vis_samples = []
    if is_master:
        print(">>> Generating 10 fixed validation samples for consistent visualization...")
        # [修复] 这里的 logic 之前有问题，导致了 tensor 形状不匹配
        # 我们手动从 dataset 取 10 个有效样本，而不是用 DataLoader
        
        count = 0
        indices = list(range(len(dataset)))
        # Simple simple shuffle to get random samples not just from start
        import random
        random.shuffle(indices)
        
        valid_indices = []
        # Find 10 valid indices (has_prev)
        for idx in indices:
            if count >= 10: break
            # dataset[idx] returns tuple, index 6 is has_prev
            # But dataset read implies reading disk, so this is slow if we iterate too many
            # Let's rely on standard loader but be careful
            pass 
        
        # Use a temp loader with batch_size=2 to be safe and accumulate
        temp_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        temp_iter = iter(temp_loader)
        
        with torch.no_grad():
            while len(fixed_vis_samples) < 10:
                try:
                    batch_data = next(temp_iter)
                except StopIteration:
                    break
                
                # Unpack and verify
                has_prev = batch_data[6]
                mask = has_prev.view(-1).bool()
                if mask.sum() == 0: continue
                
                # Apply mask immediately to EVERYTHING
                # [修复关键点] 必须对所有参与计算的 Tensor 应用同样的 mask
                input_img_t = batch_data[3][mask].to(device)   # Prev Image
                input_depth_t = batch_data[4][mask].to(device) # Prev Depth
                input_act_t = batch_data[5][mask].to(device)   # Action
                gt_img_next = batch_data[0][mask].to(device)   # GT Next
                
                # --- AR Inference ---
                d_map = depth_model(input_depth_t).unsqueeze(1)
                d_map = (d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-6)
                if d_map.shape[-1] != 384: d_map = F.interpolate(d_map, size=(224, 384), mode='bilinear')
                
                model_input = torch.cat([input_img_t, d_map], dim=1)
                pred_conf, pred_logits = ar_model(model_input, input_act_t)
                
                pred_tokens = torch.argmax(pred_logits, dim=1)
                z = F.embedding(pred_tokens, vae.model.quantize.embedding.weight).permute(0, 3, 1, 2)
                raw = vae.model.decoder(vae.model.post_quant_conv(z))
                
                if isinstance(pred_conf, tuple): pred_conf = pred_conf[0]
                conf_sig = torch.sigmoid(pred_conf)
                conf_map = F.interpolate(conf_sig, size=(224, 384), mode='bilinear', align_corners=False)
                
                # 计算 Merged
                # 这里所有tensor 维度的第一维应该都是 mask.sum() (例如 2, 3 或 4)
                # raw: [M, 3, H, W], conf_map: [M, 1, H, W], input_img_t: [M, 3, H, W]
                merged = raw * conf_map + input_img_t * (1.0 - conf_map)
                merged = torch.clamp(merged, -1.0, 1.0)
                
                # Collect
                for k in range(merged.size(0)):
                    if len(fixed_vis_samples) >= 10: break
                    fixed_vis_samples.append({
                        'prev': input_img_t[k].cpu(),
                        'merged': merged[k].cpu(),
                        'gt': gt_img_next[k].cpu()
                    })

        print(f"Collected {len(fixed_vis_samples)} fixed samples.")

    if is_master: print(f"Start Training Refiner for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        refiner.train()
        
        if is_master: pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        else: pbar = loader
            
        total_loss = 0
        total_style = 0
        total_pixel = 0
        
        for batch_idx, batch_data in enumerate(pbar):
            has_prev = batch_data[6]
            if not has_prev.all():
                valid_mask = has_prev.view(-1).bool()
                if valid_mask.sum() == 0: continue
                # [修复] 同样在训练循环中确保维度一致
                input_img_t = batch_data[3][valid_mask].to(device)
                input_depth_t = batch_data[4][valid_mask].to(device)
                input_act_t = batch_data[5][valid_mask].to(device)
                gt_img_next = batch_data[0][valid_mask].to(device)
            else:
                input_img_t = batch_data[3].to(device)
                input_depth_t = batch_data[4].to(device)
                input_act_t = batch_data[5].to(device)
                gt_img_next = batch_data[0].to(device)
            
            # --- Inference Teacher ---
            with torch.no_grad():
                d_map = depth_model(input_depth_t).unsqueeze(1)
                d_map = (d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-6)
                if d_map.shape[-1] != 384: d_map = F.interpolate(d_map, size=(224, 384), mode='bilinear')
                
                model_input = torch.cat([input_img_t, d_map], dim=1)
                # print(f"Rank {local_rank}: input shape {model_input.shape}") 调试用
                
                pred_conf, pred_logits = ar_model(model_input, input_act_t)
                z = F.embedding(torch.argmax(pred_logits, dim=1), vae.model.quantize.embedding.weight).permute(0, 3, 1, 2)
                raw = vae.model.decoder(vae.model.post_quant_conv(z))
                
                if isinstance(pred_conf, tuple): pred_conf = pred_conf[0]
                conf_sig = torch.sigmoid(pred_conf)
                conf_map = F.interpolate(conf_sig, size=(224, 384), mode='bilinear', align_corners=False)
                
                merged_input = raw * conf_map + input_img_t * (1.0 - conf_map)
                merged_input = torch.clamp(merged_input, -1.0, 1.0)
            
            # --- Train Student ---
            optimizer.zero_grad()
            refined_img = refiner(merged_input.detach()) 
            
            l_pixel, l_content, l_style = criterion(refined_img, gt_img_next)
            
            # [修改] 调整权重以提升效果
            # 降低 Pixel (10.0), 提高 Style (500.0) 和 Content (20.0)
            # 强迫模型去"画"细节，而不仅仅是把颜色对准
            loss = 10.0 * l_pixel + 20.0 * l_content + 500.0 * l_style
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pixel += l_pixel.item()
            
            if is_master:
                pbar.set_postfix({"Pix": f"{l_pixel.item():.2f}", "Sty": f"{l_style.item():.2f}"})
        
        # --- End of Epoch Logic (Master Only) ---
        if is_master:
            # 1. Save
            state_dict = refiner.module.state_dict()
            torch.save(state_dict, os.path.join(args.output_dir, f"refiner_epoch_{epoch}.pth"))
            torch.save(state_dict, os.path.join(args.output_dir, "refiner_last.pth"))
            
            # 2. Visualize Fixed Samples
            vis_dir = os.path.join(args.output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            refiner.eval()
            print("Saving visualizations...")
            with torch.no_grad():
                def tonumpy(t): return np.clip((t.permute(1,2,0).numpy() + 1)*0.5, 0, 1)
                for i, sample in enumerate(fixed_vis_samples):
                    inp = sample['merged'].unsqueeze(0).to(device)
                    # Inference
                    out = refiner.module(inp).cpu().squeeze(0) # Use .module for DDP
                    row = np.concatenate([
                        tonumpy(sample['prev']),
                        tonumpy(sample['merged']),
                        tonumpy(out),
                        tonumpy(sample['gt'])
                    ], axis=1)
                    cv2.imwrite(os.path.join(vis_dir, f"sample_{i}_epoch_{epoch}.png"), cv2.cvtColor((row*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            # 3. [新增] Run Metrics on Fixed Samples (Cheap & Fast)
            # 在每轮结束，直接用这就个固定样本算一个快速指标，看看有没有变好
            psnrs = []
            for sample in fixed_vis_samples:
                inp = sample['merged'].unsqueeze(0).to(device)
                gt = sample['gt'].unsqueeze(0).to(device)
                out = refiner.module(inp) # DDP forward
                psnr = calculate_psnr(out, gt)
                psnrs.append(psnr)
            avg_psnr = sum(psnrs)/len(psnrs)
            print(f">>> Epoch {epoch} Metrics (Fixed Set): Avg PSNR = {avg_psnr:.2f}")

        dist.barrier()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths & Checkpoints (Same as before)
    parser.add_argument("--image_dir", type=str, default="/data/cliang/mineworld/dataset/images")
    parser.add_argument("--action_dir", type=str, default="/data/cliang/mineworld/dataset/actions")
    parser.add_argument("--label_dir", type=str, default="/data/cliang/mineworld/neighbor_labels_top5")
    parser.add_argument("--output_dir", type=str, default="checkpoints_refiner")
    parser.add_argument("--vae_config", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/config.json")
    parser.add_argument("--vae_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt")
    parser.add_argument("--ar_ckpt", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4) # 会被代码覆盖成 1e-4

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        print("Please run with torchrun.")
        sys.exit(1)
        
    train(args)