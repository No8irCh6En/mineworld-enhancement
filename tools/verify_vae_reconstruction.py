import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

# 添加项目根目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.helper import instantiate_from_config

def load_model(config_path, ckpt_path=None):
    print(f"Loading config from {config_path}...")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}...")
        sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    
    model.eval()
    model.cuda()
    return model

def preprocess_image(image_path, height=224, width=384):
    if not os.path.exists(image_path):
        # 如果找不到图片，生成一张随机噪声图或者彩虹图
        print(f"Image not found at {image_path}, creating dummy image.")
        img = Image.fromarray(np.uint8(np.random.rand(height, width, 3) * 255))
    else:
        img = Image.open(image_path).convert("RGB")
    
    # 保持原始尺寸用于对比
    original_img = img.resize((width, height))
    
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor = transform(img).unsqueeze(0).cuda()
    return np.array(original_img), tensor

def decode_tokens(vae, tokens):
    """
    手动解码 Token IDs -> Image
    """
    # 获取 embedding_dim
    try:
        embedding_dim = vae.model.quantize.embedding.weight.shape[1]
    except:
        embedding_dim = 64 # 默认值，根据你的模型调整
        
    # 假设 tokens 是 [B, Seq]
    B, Seq = tokens.shape
    
    # 推断 H, W (假设下采样率是 16)
    downsample_factor = 16
    # 这里我们知道输入是 224x384 -> 14x24 = 336 tokens
    H_latent = 14
    W_latent = 24
    
    if Seq != H_latent * W_latent:
        # 尝试自动推断正方形
        side = int(np.sqrt(Seq))
        H_latent, W_latent = side, side
    
    with torch.no_grad():
        # 1. 查表获取 Quantized Vectors
        # [B, Seq] -> [B, Seq, Dim]
        z_q = vae.model.quantize.embedding(tokens) 
        
        # 2. Reshape to [B, Dim, H, W]
        z_q = z_q.view(B, H_latent, W_latent, embedding_dim).permute(0, 3, 1, 2)
        
        # 3. Decode
        quant2 = vae.model.post_quant_conv(z_q)
        dec = vae.model.decoder(quant2)
        
        # 4. Denormalize
        dec = torch.clamp(dec, -1., 1.)
        dec = (dec + 1.0) / 2.0 * 255.0
        dec = dec.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
    return dec[0]

def verify_reconstruction():
    # === 配置 ===
    CONFIG_PATH = "configs/modify.yaml"
    CKPT_PATH = None 
    
    # 找一张测试图片
    TEST_IMG_PATH = "assets/test_image.png" # 如果没有，会自动生成随机图
    # 或者尝试从数据集里找一张
    dataset_img = "/data/cliang/mineworld/dataset/images/episode_00027/image_00000.png"
    if os.path.exists(dataset_img):
        TEST_IMG_PATH = dataset_img

    OUTPUT_PATH = "analysis_results/vae_reconstruction_check.png"
    os.makedirs("analysis_results", exist_ok=True)

    # 1. 加载模型
    model = load_model(CONFIG_PATH, CKPT_PATH)
    vae = model.tokenizer

    # 2. 准备数据
    H, W = 224, 384
    original_img_np, img_tensor = preprocess_image(TEST_IMG_PATH, height=H, width=W)

    # 3. Encode (Image -> Tokens)
    print("Encoding image to tokens...")
    with torch.no_grad():
        tokens = vae.tokenize_images(img_tensor) # 返回 numpy 或 list
        tokens = torch.tensor(tokens, device=img_tensor.device).long()
        if tokens.dim() == 1: tokens = tokens.unsqueeze(0)
        if tokens.dim() == 3: tokens = tokens.view(tokens.shape[0], -1)
    
    print(f"Token shape: {tokens.shape}")
    print(f"First 10 tokens: {tokens[0, :10].cpu().numpy()}")

    # 4. Decode (Tokens -> Image)
    print("Decoding tokens back to image...")
    recon_img_np = decode_tokens(vae, tokens)

    # 5. 计算误差
    mse = np.mean((original_img_np.astype(float) - recon_img_np.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse)
    print(f"Reconstruction MSE: {mse:.2f}")
    print(f"Reconstruction PSNR: {psnr:.2f} dB (通常 > 20dB 算正常，> 30dB 算很好)")

    # 6. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(recon_img_np)
    axes[1].set_title(f"Reconstructed from Tokens\nPSNR: {psnr:.2f} dB")
    axes[1].axis('off')
    
    # 差值图 (放大 5 倍以便观察)
    diff = np.abs(original_img_np.astype(int) - recon_img_np.astype(int)).astype(np.uint8)
    axes[2].imshow(diff * 5) 
    axes[2].set_title("Difference (x5 boosted)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Saved verification result to {OUTPUT_PATH}")

if __name__ == "__main__":
    verify_reconstruction()