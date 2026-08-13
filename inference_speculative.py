import os
import cv2
import torch
import time
import numpy as np
from tqdm import tqdm
from rich import print
from PIL import Image
from pathlib import Path
from torch import autocast
from einops import rearrange
from mcdataset import MCDataset
from omegaconf import OmegaConf
from torchvision import transforms
from argparse import ArgumentParser
from util.helper import load_model
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import json

# Import new logic
from speculative_wrapper import get_inference_functions
from diagonal_decoding import speculative_img_diagd_decode_n_tokens

torch.backends.cuda.matmul.allow_tf32 = False

ACCELERATE_ALGO = [
    'speculative'
]

TARGET_SIZE=(224,384)
TOKEN_PER_IMAGE = 347 # IMAGE = PIX+ACTION
TOKEN_PER_PIX = 336
PIX_NUM = 336
FRAME_WINDOW = 4

safe_globals = {"array": np.array}

# 全局变量缓存 Draft/Action Functions
GLOBAL_DRAFT_FUNC = None
GLOBAL_ACTION_FUNC = None

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--action_model_ckpt', type=str, default="pred_model/action_predictor_latest.pth")
    parser.add_argument('--draft_model_ckpt', type=str, default="pred_model_uncertainty/best_model.pth")
    parser.add_argument('--demo_num', type=int, default=1)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--save_frames', action='store_true', help='Save individual frames and tokens')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--top_k', type=int, help='Use top-k sampling')
    group.add_argument('--top_p', type=float, help='Use top-p (nucleus) sampling')
    parser.add_argument('--val_data_num', type=int, default=500, help="number of validation data")
    
    # Speculative params
    parser.add_argument('--stagger_steps', type=int, default=5, help="Steps to wait before verification (parallel depth)")
    parser.add_argument('--num_candidates', type=int, default=5, help="Number of action candidates")
    parser.add_argument('--use_oracle', action='store_true', help='Use GT oracle draft/action instead of real models')
    
    args = parser.parse_args()
    return args

def get_latest_zero3_ckpt_dir(output_dir: str) -> str:
    norm_path = output_dir.rstrip(os.path.sep)
    if "checkpoint-" in os.path.basename(norm_path) and os.path.isdir(norm_path):
        return norm_path
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"trainer_state.json not found in {output_dir}")
    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)
    global_step = trainer_state.get("global_step", None)
    if global_step is None:
        raise ValueError("global_step not found in trainer_state.json")
    return os.path.join(output_dir, f"checkpoint-{global_step}")

def load_model_with_fallback(config, ckpt_path, gpu=True, eval_mode=True):
    if ckpt_path.endswith('.ckpt'):
        return load_model(config, ckpt_path, gpu=gpu, eval_mode=eval_mode)
    model = load_model(config, None, gpu=False, eval_mode=False)
    ckpt_dir = get_latest_zero3_ckpt_dir(ckpt_path)
    if os.path.exists(ckpt_dir):
        sub_dirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
        global_step_subdir = next((d for d in sub_dirs if d.startswith("global_step")), None)
        if global_step_subdir:
            ckpt_dir = os.path.join(ckpt_dir, global_step_subdir)
    if not os.path.exists(ckpt_dir):
         raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    parent_dir = os.path.dirname(ckpt_dir)
    tag = os.path.basename(ckpt_dir)
    print(f"[INFO] Loading ZeRO checkpoint from {parent_dir} with tag {tag}")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(parent_dir, tag)
    model.load_state_dict(state_dict, strict=True)
    if gpu: model = model.cuda()
    if eval_mode: model.eval()
    return model

def token2video(code_list, tokenizer, save_path, fps, device = 'cuda', save_frames_dir=None):
    if len(code_list) % TOKEN_PER_PIX != 0:
        print(f"code_list length {len(code_list)} is not multiple of {TOKEN_PER_PIX}")
        return
    num_images = len(code_list) // TOKEN_PER_PIX
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (384, 224))
    if save_frames_dir: os.makedirs(save_frames_dir, exist_ok=True)
    for i in range(num_images):
        code = code_list[i*TOKEN_PER_PIX:(i+1)*TOKEN_PER_PIX]
        code = torch.tensor([int(x) for x in code], dtype=torch.long).to(device)
        img = tokenizer.token2image(code) 
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(frame)
        if save_frames_dir:
            cv2.imwrite(os.path.join(save_frames_dir, f"image_{i:05d}.png"), frame)
    video.release()



def lvm_generate(args, model, output_dir, demo_video):
    global GLOBAL_DRAFT_FUNC, GLOBAL_ACTION_FUNC
    
    # Treat demo_video as base_name; construct video and action paths under standardized layout
    base_name = os.path.splitext(demo_video)[0] if demo_video.endswith('.mp4') else demo_video
    direct_video_file = os.path.join(args.data_root, base_name + ".mp4")
    direct_video_dir = os.path.join(args.data_root, base_name)
    video_dir = os.path.join(args.data_root, "video_clip", base_name)
    video_file = os.path.join(args.data_root, "video_clip", base_name + ".mp4")
    # prefer direct mp4 / frames under data_root, then video_clip layout, then fallback to legacy path
    if os.path.isdir(direct_video_dir):
        input_path = direct_video_dir
    elif os.path.exists(direct_video_file):
        input_path = direct_video_file
    elif os.path.isdir(video_dir):
        input_path = video_dir
    elif os.path.exists(video_file):
        input_path = video_file
    else:
        input_path = os.path.join(args.data_root, demo_video)
    output_mp4_path = str(output_dir / (base_name + ".mp4"))
    
    # Simple check
    if os.path.exists(output_mp4_path):
        print(f"output path {output_mp4_path} exist")
        return {}
    
    device = model.transformer.device
    
    # --- 1. Determine Input Type & Load Actions ---
    is_dir = os.path.isdir(input_path)
    
    # Preferred action location: <data_root>/action_clip/<base_name>/action.jsonl
    # If args.data_root points to a video_clip folder, use its parent for action_clip
    if os.path.basename(os.path.normpath(args.data_root)) == "video_clip":
        action_root = os.path.join(os.path.dirname(os.path.normpath(args.data_root)), "action_clip")
    else:
        action_root = os.path.join(args.data_root, "action_clip")
    input_action_path = os.path.join(action_root, base_name, "action.jsonl")
    # Fallback to <args.data_root>/<base_name>.jsonl
    if not os.path.exists(input_action_path):
        alt = os.path.join(args.data_root, base_name + ".jsonl")
        if os.path.exists(alt):
            input_action_path = alt
        else:
            # final fallback: try action_root/<base_name>.jsonl
            alt2 = os.path.join(action_root, base_name + ".jsonl")
            if os.path.exists(alt2):
                input_action_path = alt2
            else:
                print(f"[Warning] Action file {input_action_path} not found.") 
                # You might want to skip or handle gracefully
    
    action_list = []
    mcdataset = MCDataset()
    print(f"[INFO] Loading actions from {input_action_path}...")
    if os.path.exists(input_action_path):
        print(f"[INFO] Found action file: {input_action_path}")
        with open(input_action_path, 'r') as f:
            for line in f:
                line = eval(line.strip(), {"__builtins__": None}, safe_globals)
                line['camera'] = np.array(line['camera'])
                act_index = mcdataset.get_action_index_from_actiondict(line, action_vocab_offset=8192)
                action_list.append(act_index)
            
    # --- 2. Load Frames (MP4 or Image Dir) ---
    frames = []
    
    if is_dir:
        # Load Images
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        # Filter images
        image_files = [f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in valid_exts]
        
        # Sort files. 
        # Attempt numeric sort if filenames are like "0.png", "1.png" to handle "10.png" correctly
        try:
             image_files.sort(key=lambda f: int(os.path.splitext(f)[0]))
        except ValueError:
             image_files.sort() # Fallback to alphanumeric
             
        # Load demo_num frames
        for i in range(args.demo_num):
            if i >= len(image_files): break
            img_p = os.path.join(input_path, image_files[i])
            frame = cv2.imread(img_p)
            if frame is None: continue
            
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            frame = torch.from_numpy(frame)
            frames.append(frame)
            
    else:
        # Load MP4
        cap = cv2.VideoCapture(input_path)
        for frame_idx in range(args.demo_num):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            frame = torch.from_numpy(frame)
            frames.append(frame)
            
    if not frames:
        print(f"[Error] No frames loaded from {input_path}")
        return {}
        
    # Ensure all frames have TARGET_SIZE (H,W) and are uint8 numpy before stacking
    resized_frames = []
    for f in frames:
        # f may be a torch tensor (H,W,3) or numpy array; convert to numpy
        if isinstance(f, torch.Tensor):
            f_np = f.cpu().numpy()
        else:
            f_np = np.asarray(f)
        # cv2.resize expects (w,h)
        f_resized = cv2.resize(f_np, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        resized_frames.append(f_resized)

    # create tensor in (N, H, W, C) then convert to (N, C, H, W)
    frames = torch.stack([torch.from_numpy(np.clip(x,0,255).astype(np.uint8)) for x in resized_frames], dim=0)
    # move channels to dim=1, convert to float and to device
    frames = frames.permute(0, 3, 1, 2).contiguous().float().to(device) / 255.0
    # safety check: ensure channels-first
    assert frames.ndim == 4 and frames.size(1) == 3, f"frames must be (N,3,H,W), got {tuple(frames.shape)}"
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frames = normalize(frames)
    print(f"[DEBUG] frames info: shape={frames.shape}, dtype={frames.dtype}, device={frames.device}")
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        img_index = model.tokenizer.tokenize_images(frames)
        
    print(f"[DEBUG] img_index shape after tokenize_images: {img_index.shape}")
    
    # --- GT Oracle (conditional on --use_oracle flag) ---
    _oracle_draft_func = None
    _oracle_action_func = None
    if args.use_oracle:
        # Load all frames from the video for oracle testing
        is_dir = os.path.isdir(input_path)
        all_frames = []
        if is_dir:
            valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
            image_files = sorted(
                [f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in valid_exts],
                key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else 0
            )
            for f in image_files:
                frame = cv2.imread(os.path.join(input_path, f))
                if frame is not None:
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=frame)
                    all_frames.append(frame)
        else:
            cap = cv2.VideoCapture(input_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=frame)
                all_frames.append(frame)
            cap.release()
        
        # Tokenize all GT frames
        gt_frames_tokens = []  # list of [336] tensors
        if len(all_frames) > args.demo_num:
            for i in range(args.demo_num, min(len(all_frames), args.demo_num + args.frames)):
                f = all_frames[i]
                f = cv2.resize(f, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_LINEAR)
                f = torch.from_numpy(np.clip(f, 0, 255).astype(np.uint8)).permute(2, 0, 1).float().to(device) / 255.0
                f = normalize(f).unsqueeze(0)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    tokens = model.tokenizer.tokenize_images(f)  # [1, 14, 24]
                gt_frames_tokens.append(tokens.view(-1))  # [336]
            print(f"[ORACLE] Tokenized {len(gt_frames_tokens)} GT frames for oracle draft")
        
        oracle_call_count = [1]
        def _oracle_draft_func(prev_tokens, action_candidates, merge=True):
            idx = oracle_call_count[0]
            oracle_call_count[0] += 1
            K = action_candidates.size(0)
            if idx < len(gt_frames_tokens):
                gt = gt_frames_tokens[idx].to(action_candidates.device)
                return gt.unsqueeze(0).expand(K, -1)
            return torch.zeros(K, PIX_NUM, device=action_candidates.device, dtype=torch.long)
        
        def _oracle_action_func(action_history):
            hist_len = action_history.size(0) if isinstance(action_history, torch.Tensor) else len(action_history)
            idx = hist_len
            if idx < len(gt_actions):
                gt = torch.tensor(gt_actions[idx], device="cuda", dtype=torch.long)
                return gt.unsqueeze(0).expand(5, -1)
            return torch.zeros(5, 11, device="cuda", dtype=torch.long)
        
        print(f"[ORACLE] Using GT oracle draft and action functions")
    # --- End GT Oracle setup ---
    
    # Prepare Input
    img_index = rearrange(img_index, '(b t) h w -> b t (h w)', b=1)
    image_input = rearrange(img_index, 'b t c -> b (t c)') 
    
    start_act_idx = 0
    # Safety Check for action list length
    end_act_idx = min(len(action_list), args.frames+1)
    gt_actions = action_list[start_act_idx : end_act_idx]
    
    print(f"[DEBUG] start_act_idx: {start_act_idx}, end_act_idx: {end_act_idx}, action_list: {action_list}")
        
    gt_actions_tensor = torch.tensor(gt_actions, device=device, dtype=torch.long)
    
    print(f"[DEBUG] Processing {demo_video} ({'Dir' if is_dir else 'MP4'}). Input Tokens: {image_input.shape[1]}")

    start_t = time.time()
    all_generated_tokens = []
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        # Call the Speculative DiagD function
        _draft = _oracle_draft_func if args.use_oracle else GLOBAL_DRAFT_FUNC
        _action = _oracle_action_func if args.use_oracle else GLOBAL_ACTION_FUNC
        outputs = model.transformer.speculative_diag_generate_img_token(
            input_ids=image_input, max_new_tokens=TOKEN_PER_PIX * args.frames,
            action_all=gt_actions_tensor, top_k = args.top_k, top_p = args.top_p,
            draft_func=_draft, action_pred_func=_action,
        )
        
        # Output is a tensor [1, total_tokens] (all generated pixel tokens)
        if isinstance(outputs, torch.Tensor):
            all_generated_tokens = outputs.view(-1).cpu().tolist()
        elif isinstance(outputs, list):
            # Fallback: flatten list of lists
            flat = []
            for x in outputs:
                if isinstance(x, torch.Tensor):
                    flat.extend(x.view(-1).cpu().tolist())
                elif isinstance(x, list):
                    flat.extend(x)
            all_generated_tokens = flat
        else:
            all_generated_tokens = []
             
    input_len = image_input.numel()
    generated_len = len(all_generated_tokens) - input_len
    
    end_t = time.time()
    time_costed = end_t - start_t
    token_per_sec = generated_len / time_costed if time_costed > 0 else 0
    fps_gen = token_per_sec / TOKEN_PER_PIX
    
    print(f"Speculative Gen: {generated_len} tokens in {time_costed:.3f}s. {token_per_sec:.2f} tok/s ({fps_gen:.2f} fps)")

    # Extract clip_id regardless of save_frames flag
    try:
        clip_id = demo_video.split('_')[1].split('.')[0]
    except:
        clip_id = Path(demo_video).stem

    if args.save_frames:
        save_token_dir = output_dir / f"tokens"
        os.makedirs(save_token_dir, exist_ok=True)
        np.save(save_token_dir / f"tokens_{clip_id}.npy", np.array(all_generated_tokens))

    tokens_to_render = all_generated_tokens  # speculative output already excludes demo frames
    save_frames_dir = output_dir / f"images/episode_{clip_id}" if args.save_frames else None
    
    # Save video with correct extension
    out_name = demo_video + ".mp4" if not demo_video.endswith('.mp4') else demo_video
    token2video(tokens_to_render, model.tokenizer, str(output_dir / out_name), args.fps, device, save_frames_dir=save_frames_dir)

    return {"time": time_costed, "tokens": generated_len}

if __name__ == '__main__':
    args = get_args()
    config = OmegaConf.load(args.config)
    output_path = Path(args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"[INFO] Initializing Speculative Models...")
    GLOBAL_DRAFT_FUNC, GLOBAL_ACTION_FUNC = get_inference_functions(
        action_model_path=args.action_model_ckpt,
        draft_model_path=args.draft_model_ckpt
    )
    
    print(f"[INFO] Loading LVM Model...")
    model = load_model_with_fallback(config, args.model_ckpt, gpu=True, eval_mode=True)
    
    # 3. Inference Loop Enhancement
    # Build task list from <data_root>/video_clip using base_name semantics
    video_root = os.path.join(args.data_root, "video_clip")
    task_list = []
    if os.path.exists(video_root) and os.path.isdir(video_root):
        for f in sorted(os.listdir(video_root)):
            if f.startswith('.'): continue
            full = os.path.join(video_root, f)
            if os.path.isdir(full):
                task_list.append(f)  # base_name from folder
            elif os.path.isfile(full) and f.endswith('.mp4'):
                task_list.append(os.path.splitext(f)[0])  # base_name from mp4
    else:
        # fallback: scan data_root for folders/mp4s (legacy)
        all_items = sorted(os.listdir(args.data_root))
        for f in all_items:
            if f.startswith('.'): continue
            full_path = os.path.join(args.data_root, f)
            if os.path.isdir(full_path):
                if any(file.endswith(('.png', '.jpg')) for file in os.listdir(full_path)):
                    task_list.append(f)
            elif f.endswith('.mp4'):
                task_list.append(os.path.splitext(f)[0])
    task_list = sorted(task_list)
    
    count = 0
    for file in tqdm(task_list):
        try:
            lvm_generate(args, model, output_path, file)
            count += 1
            if count >= args.val_data_num: break
        except Exception as e:
            print(f"[Error] Failed on {file}: {e}")
            import traceback
            traceback.print_exc()