import os
import cv2
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def find_dataset_path(dataset_dir, sub_dir, episode_name, is_file=False, extension=""):
    variants = [episode_name]
    if not episode_name.startswith("episode_"): variants.append(f"episode_{episode_name}")
    if episode_name.startswith("episode_"): variants.append(episode_name.replace("episode_", ""))
    for v in variants:
        name = v + extension if is_file else v
        path = os.path.join(dataset_dir, sub_dir, name)
        if os.path.exists(path): return path
    return None

def load_dataset_episode(dataset_dir, episode_name):
    print(f"Attempting to load episode '{episode_name}' from {dataset_dir}...")
    img_dir = find_dataset_path(dataset_dir, "images", episode_name)
    if not img_dir:
        # Try direct path if find_dataset_path fails (e.g. for outputs_video/plain/clip_xx)
        direct_path = os.path.join(dataset_dir, episode_name)
        if os.path.exists(direct_path):
            img_dir = direct_path # Assume images are directly here or in images subdir
            if os.path.exists(os.path.join(direct_path, "images")):
                img_dir = os.path.join(direct_path, "images")
        
        if not img_dir or not os.path.exists(img_dir):
            print("Images not found.")
            return None

    fns = sorted([fn for fn in os.listdir(img_dir) if fn.endswith(('.png', '.jpg'))])
    frames = []
    for fn in fns:
        img = cv2.imread(os.path.join(img_dir, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    # Load Coords
    coords_path = find_dataset_path(dataset_dir, "token_coords", episode_name, is_file=True, extension=".npy")
    coords = None
    if coords_path:
        print(f"Found token coords at {coords_path}")
        coords = np.load(coords_path)

    # Load Inference Tokens (.npy)
    tokens = None
    
    # Extract clip ID (e.g. episode_00027 -> 00027)
    try:
        clip_id_str = episode_name.split('_')[-1] # 00027
        clip_id_int = int(clip_id_str)
        token_npy_name = f"tokens_{clip_id_int}.npy"
    except:
        token_npy_name = f"tokens_{episode_name}.npy"
        clip_id_int = 0 # Default

    # Check in dataset_dir/tokens/
    token_npy_path = os.path.join(dataset_dir, "tokens", token_npy_name)
    
    if os.path.exists(token_npy_path):
        try:
            tokens = np.load(token_npy_path)
            print(f"Found inference tokens at {token_npy_path}, shape: {tokens.shape}")
        except Exception as e:
            print(f"Error loading npy tokens: {e}")
    else:
        print(f"Inference tokens not found at {token_npy_path}, will use VAE inference.")

    # Load Actions (.jsonl)
    actions = []
    # Check in dataset_dir/clip_xx.jsonl
    action_path = os.path.join(dataset_dir, f"clip_{clip_id_int}.jsonl")
    if not os.path.exists(action_path):
        # Check in dataset_dir/actions/episode_name.jsonl (standard dataset structure)
        action_path = find_dataset_path(dataset_dir, "actions", f"clip_{clip_id_int}", is_file=True, extension=".jsonl")

    if action_path and os.path.exists(action_path):
        print(f"Found actions at {action_path}")
        try:
            with open(action_path, 'r') as f:
                for line in f:
                    if line.strip():
                        actions.append(json.loads(line))
            print(f"Loaded {len(actions)} actions.")
        except Exception as e:
            print(f"Error loading actions: {e}")
            actions = []
    else:
        print("Actions not found.")

    return {"frames": frames, "coords": coords, "tokens": tokens, "actions": actions}

def parse_action(action_dict):
    if not action_dict:
        return "No Action Data"
    
    # Camera
    cam = action_dict.get("camera", [0.0, 0.0])
    cam_str = f"Cam:[{cam[0]:.1f}, {cam[1]:.1f}]"
    
    # Movement Keys
    keys = []
    if action_dict.get("forward"): keys.append("Fwd")
    if action_dict.get("back"): keys.append("Back")
    if action_dict.get("left"): keys.append("Left")
    if action_dict.get("right"): keys.append("Right")
    if action_dict.get("jump"): keys.append("Jump")
    if action_dict.get("sneak"): keys.append("Sneak")
    if action_dict.get("sprint"): keys.append("Sprint")
    if action_dict.get("attack"): keys.append("Atk")
    if action_dict.get("use"): keys.append("Use")
    
    key_str = " ".join(keys)
    return f"{cam_str} {key_str}"

def preprocess_frame(frame_rgb, height=224, width=384):
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0).cuda()

def get_embeddings_from_ids(model, tokens):
    llama_embed = model.transformer.model.embed_tokens
    try:
        first_ln = model.transformer.model.layers[0].input_layernorm
    except AttributeError:
        first_ln = None

    with torch.no_grad():
        raw_embeddings = llama_embed(tokens)
        
        if first_ln is not None:
            embeddings = first_ln(raw_embeddings)
        else:
            embeddings = raw_embeddings

        if embeddings.dim() == 4:
            B, H, W, D = embeddings.shape
            embeddings = embeddings.view(B, H * W, D)
    return embeddings

def get_tokens_and_embeddings(model, img_tensor):
    """同时返回 Token IDs 和 经过第一层 Norm 的 Embeddings"""
    vae = model.tokenizer  
    
    with torch.no_grad():
        tokens = vae.tokenize_images(img_tensor)  
        tokens = torch.as_tensor(tokens, device=img_tensor.device)
        
        if tokens.dim() == 1: tokens = tokens.unsqueeze(0)  
        elif tokens.dim() == 3:
            B, H, W = tokens.shape
            tokens = tokens.view(B, H * W)
        elif tokens.dim() > 3:
            B = tokens.shape[0]
            tokens = tokens.view(B, -1)
        tokens = tokens.long() 

    embeddings = get_embeddings_from_ids(model, tokens)
            
    return tokens, embeddings