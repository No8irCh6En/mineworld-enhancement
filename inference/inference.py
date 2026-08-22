import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from util.helper import load_model, tensor_to_uint8
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import json
torch.backends.cuda.matmul.allow_tf32 = False

# import torch._dynamo
# torch._dynamo.config.capture_scalar_outputs = True
# # 或
# torch._dynamo.config.suppress_errors = True  # 如果 capture 不行，强制回退

ACCELERATE_ALGO = [
    'naive','image_diagd'
]

TARGET_SIZE=(224,384)
TOKEN_PER_IMAGE = 347 # IMAGE = PIX+ACTION
TOKEN_PER_PIX = 336

safe_globals = {"array": np.array}

def get_latest_zero3_ckpt_dir(output_dir: str) -> str:
    # 修改：如果传入的路径本身就是一个 checkpoint 目录（包含 checkpoint-），直接返回
    # 移除末尾可能的斜杠，以免 basename 为空
    norm_path = output_dir.rstrip(os.path.sep)
    if "checkpoint-" in os.path.basename(norm_path) and os.path.isdir(norm_path):
        return norm_path

    # 读取 trainer_state.json
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"trainer_state.json not found in {output_dir}")

    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    # 拿到 global_step，例如 2100
    global_step = trainer_state.get("global_step", None)
    if global_step is None:
        raise ValueError("global_step not found in trainer_state.json")

    # 构造 ZeRO 分片子目录路径
    return os.path.join(output_dir, f"checkpoint-{global_step}")

def load_model_with_fallback(config, ckpt_path, gpu=True, eval_mode=True):
    if ckpt_path.endswith('.ckpt'):
        # 直接加载普通 checkpoint
        return load_model(config, ckpt_path, gpu=gpu, eval_mode=eval_mode)

    # 1. 构建模型结构
    model = load_model(config, None, gpu=False, eval_mode=False)  # 注意不加载任何 state_dict


    # 3. 加载 state_dict
    # print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    # checkpoint = torch.load(ckpt_path, map_location='cpu')

    # if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    #     # case 1: 保存的是 {"state_dict": ..., "config": ...} 格式
    #     state_dict = checkpoint["state_dict"]
    #     print("[INFO] Loaded state_dict from checkpoint dict.")
    # elif isinstance(checkpoint, dict):
    #     # case 2: 保存的直接就是 state_dict 本身（无外壳）
    #     state_dict = checkpoint
    #     print("[INFO] Loaded plain state_dict.")
    # elif isinstance(checkpoint, torch.nn.Module):
    #     # case 3: 保存的是整个模型
    #     state_dict = checkpoint.state_dict()
    #     print("[WARN] Loaded full model object (not recommended).")
    # else:
    #     raise RuntimeError("Unrecognized checkpoint format.")

    ckpt_dir = get_latest_zero3_ckpt_dir(ckpt_path)  # 替换为你的输出目录路径
    
    # --- Start Fix: Handle nested global_step directory ---
    # 检查是否存在嵌套的 global_step 目录 (DeepSpeed 保存可能会多套一层)
    if os.path.exists(ckpt_dir):
        # 查找以 global_step 开头的子目录
        sub_dirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
        global_step_subdir = next((d for d in sub_dirs if d.startswith("global_step")), None)
        
        if global_step_subdir:
            # 如果找到了 global_stepXXX 文件夹，就更新路径深入进去
            print(f"[INFO] Detect nested DeepSpeed folder, descending into: {global_step_subdir}")
            ckpt_dir = os.path.join(ckpt_dir, global_step_subdir)
    # --- End Fix ---

    # --- 修改开始 ---
    # 显式分离父目录和 tag，避免 DeepSpeed 查找 'latest' 文件
    if not os.path.exists(ckpt_dir):
         raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    parent_dir = os.path.dirname(ckpt_dir)
    tag = os.path.basename(ckpt_dir)
    
    print(f"[INFO] Loading ZeRO checkpoint from {parent_dir} with tag {tag}")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(parent_dir, tag)
    # --- 修改结束 ---

    print(f"[INFO] {len(state_dict)} keys in checkpoint.")

    # 4. 加载权重到你构建的模型结构中
    model.load_state_dict(state_dict, strict=True)

    # 5. 设定为 eval + GPU
    if gpu:
        model = model.cuda()
    if eval_mode:
        model.eval()

    return model


def export_attention_weights_to_csv(inference_model, save_path, topk: int = 30, save_analysis: bool = True):
    import csv
    """
    导出 attention weights 为 CSV 文件，并可选地对每一行（每个 query）做 top-k 分析：
      - topk_indices: 每行 attention 最高的 topk 个 key 的索引
      - topk_scores: 对应的 attention 分数
      - rel_positions: topk_indices - query_index
      - topk_sum: 这 topk 个位置的 attention 之和
      - weighted_mean_rel: 用 topk_scores 做权重的平均相对位置

    Args:
        inference_model: 包含 N 层 decoder 的模型，层结构是 inference_model.model.layers
        save_path (str): CSV 保存路径，例如 './attention_weights/sample.csv'
        topk (int): 每行要取的 top-k 大小（默认为 10）
        save_analysis (bool): 是否保存 per-row 分析为额外 CSV（save_path + ".analysis.csv"）
    """
    all_layers_attn = []

    for i, layer in enumerate(inference_model.model.layers):
        attn_list = layer.self_attn.latest_attn_weights  # list of [1, Q_i, K]
        if len(attn_list) == 0:
            print(f"[Warning] Layer {i} has no attention saved.")
            continue
        attn_tensor = torch.cat(attn_list, dim=1)  # → [1, ΣQ_layer, K]
        all_layers_attn.append(attn_tensor)

    if len(all_layers_attn) == 0:
        raise RuntimeError("No attention weights found in any layer.")

    # Stack layers → [L, 1, ΣQ_layer, K]
    stacked_attn = torch.stack(all_layers_attn, dim=0)

    # Mean over layers → [1, ΣQ, K]
    attn_mean = stacked_attn.mean(dim=0)

    # Squeeze batch dim → [ΣQ, K]
    attn_result = attn_mean.squeeze(0)

    # 转换为 numpy → 保存为 CSV
    attn_np = attn_result.cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(attn_np)

    print(f"[✓] Attention weights exported to: {save_path}")
    print(f"    Shape: {attn_np.shape} (Q={attn_np.shape[0]}, K={attn_np.shape[1]})")

    # -----------------------
    # per-row top-k analysis
    # -----------------------
    if save_analysis:
        Q, K = attn_np.shape
        k = min(int(topk), K)
        # argsort 降序取前 k（注意：对大矩阵可改用 argpartition 提速）
        order = np.argsort(-attn_np, axis=1)[:, :k]  # shape [Q, k]
        # 取对应分数
        row_idx = np.arange(Q)[:, None]
        topk_scores = attn_np[row_idx, order]  # [Q, k]
        rel_pos = order - row_idx  # [Q, k], 相对位置 = key_idx - query_idx
        topk_sum = topk_scores.sum(axis=1)  # [Q]
        # 加权平均相对位置（若 topk_sum 非零）
        weighted_mean_rel = np.zeros(Q, dtype=float)
        nonzero_mask = topk_sum > 0
        weighted_mean_rel[nonzero_mask] = (rel_pos[nonzero_mask] * topk_scores[nonzero_mask]).sum(axis=1) / topk_sum[nonzero_mask]

        analysis_path = "attention_weights/analysis.csv"
        with open(analysis_path, "w", newline="") as f:
            writer = csv.writer(f)
            # header
            writer.writerow(["query_index", "topk_indices", "topk_scores", "rel_positions", "topk_sum", "weighted_mean_rel"])
            for qi in range(Q):
                idxs = ";".join(map(str, order[qi].tolist()))
                scores = ";".join([f"{s:.6f}" for s in topk_scores[qi].tolist()])
                rels = ";".join(map(str, rel_pos[qi].tolist()))
                writer.writerow([qi, idxs, scores, rels, f"{topk_sum[qi]:.6f}", f"{weighted_mean_rel[qi]:.6f}"])

        # 打印简单汇总
        print(f"[✓] Per-row top-{k} analysis saved to: {analysis_path}")
        print(f"    topk sum stats: mean={topk_sum.mean():.6f}, median={np.median(topk_sum):.6f}")
        print(f"    weighted mean relative position stats: mean={weighted_mean_rel.mean():.3f}, std={weighted_mean_rel.std():.3f}")

def token2video(code_list, tokenizer, save_path, fps, device = 'cuda', save_frames_dir=None):
    """
    change log:  we don't perform path processing inside functions to enable extensibility
    save_path: str, path to save the video, expect to endwith .mp4
    
    """
    if len(code_list) % TOKEN_PER_PIX != 0:
        print(f"code_list length {len(code_list)} is not multiple of {TOKEN_PER_PIX}")
        return
    num_images = len(code_list) // TOKEN_PER_PIX
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (384, 224))
    
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    for i in range(num_images):
        code = code_list[i*TOKEN_PER_PIX:(i+1)*TOKEN_PER_PIX]
        code = torch.tensor([int(x) for x in code], dtype=torch.long).to(device)
        img = tokenizer.token2image(code) # pixel
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(frame)
        
        if save_frames_dir:
            cv2.imwrite(os.path.join(save_frames_dir, f"image_{i:05d}.png"), frame)
            
    video.release()

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--demo_num', type=int, default=1)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--accelerate-algo', type=str, default='naive', help=f"Accelerate Algorithm Option: {ACCELERATE_ALGO}")
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--save_frames', action='store_true', help='Save individual frames and tokens')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--top_k', type=int, help='Use top-k sampling')
    group.add_argument('--top_p', type=float, help='Use top-p (nucleus) sampling')
    parser.add_argument('--val_data_num', type=int, default=500, help="number of validation data")
    args = parser.parse_args()
    return args


def lvm_generate(args, model, output_dir, demo_video):
    """
    """
    ### 1. set video input/output path
    input_mp4_path = os.path.join(args.data_root, demo_video)
    input_action_path = os.path.join(args.data_root, demo_video.replace('mp4','jsonl'))

    output_mp4_path = str(output_dir / demo_video)
    output_action_path = output_mp4_path.replace('.mp4', '.jsonl')
    # backup action 
    os.system(f"cp {input_action_path} {output_action_path}")
    # os.makedirs(os.path.dirname(f"{output_action_path}/actions"), exist_ok=True)
    # os.system(f"cp {input_action_path} {output_action_path}/actions")
    if os.path.exists(output_mp4_path):
        print(f"output path {output_mp4_path} exist")
        return {}
    
    device = model.transformer.device
    ### 2. load action into list 
    action_list = []
    mcdataset = MCDataset()
    with open(input_action_path, 'r') as f:
        for line in f:
            line = eval(line.strip(), {"__builtins__": None}, safe_globals)
            line['camera'] = np.array(line['camera'])
            act_index = mcdataset.get_action_index_from_actiondict(line, action_vocab_offset=8192)
            action_list.append(act_index)
    ### 3. load video frames 
    cap = cv2.VideoCapture(input_mp4_path)
    start_frame = 0
    end_frame = args.demo_num
    frames = []
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error in reading frame {frame_idx}")
            continue
        cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
        frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
        frame = torch.from_numpy(frame)
        frames.append(frame)
    frames = torch.stack(frames, dim=0).to(device)
    frames = frames.permute(0, 3, 1, 2)
    frames = frames.float() / 255.0
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frames = normalize(frames)
    
    print(f"[DEBUG] frames info: shape={frames.shape}, dtype={frames.dtype}, device={frames.device}")
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        img_index = model.tokenizer.tokenize_images(frames)
        
    print(f"[DEBUG] img_index shape before rearrange: {img_index.shape}")
    img_index = rearrange(img_index, '(b t) h w -> b t (h w)', b=1)
     
    all_generated_tokens = []

    action_all = action_list[end_frame: end_frame + args.frames]
    action_all = torch.tensor(action_all).unsqueeze(1).to(device)
    image_input = rearrange(img_index, 'b t c -> b (t c)')
    print(f"[DEBUG] input_ids: shape {image_input.shape}")

    start_t = time.time() 
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        if args.accelerate_algo == 'naive':
            outputs = model.transformer.naive_generate(input_ids=image_input, max_new_tokens=TOKEN_PER_PIX*args.frames, action_all=action_all, top_k=args.top_k, top_p=args.top_p)
            # export_attention_weights_to_csv(model.transformer, "./attention_weights/naive.csv")
        elif args.accelerate_algo == 'image_diagd':
            outputs = model.transformer.img_diagd_generate(input_ids=image_input, max_new_tokens=TOKEN_PER_PIX*args.frames, action_all=action_all,windowsize = args.window_size, top_k=args.top_k, top_p=args.top_p)
        else:
            raise ValueError(f"Unknown accelerate algorithm {args.accelerate_algo}")
    end_t = time.time()
    all_generated_tokens.extend(outputs.tolist()[0])
    new_length = len(all_generated_tokens)
    time_costed = end_t - start_t 
    token_per_sec = new_length / time_costed
    frame_per_sec = token_per_sec / TOKEN_PER_PIX
    print(f"{new_length} token generated; cost {time_costed:.3f} second; {token_per_sec:.3f} token/sec {frame_per_sec:.3f} fps")
    
    save_frames_dir = None
    if args.save_frames:
        # Extract clip ID from filename (e.g. episode_00027.mp4 -> 00027)
        try:
            clip_id = demo_video.split('_')[1].split('.')[0]
        except:
            clip_id = Path(demo_video).stem
            
        save_frames_dir = output_dir / f"images/episode_{clip_id}"
        os.makedirs(save_frames_dir, exist_ok=True)
        save_token_dir = output_dir / f"tokens"
        os.makedirs(save_token_dir, exist_ok=True)
        # Save tokens
        np.save(save_token_dir / f"tokens_{clip_id}.npy", np.array(all_generated_tokens))
        print(f"[INFO] Saved frames to {save_frames_dir}")
        print(f"[INFO] Saved tokens to {save_token_dir}/tokens_{clip_id}.npy")

    token2video(all_generated_tokens, model.tokenizer, str(output_dir / demo_video), args.fps, device, save_frames_dir=save_frames_dir)  
    # return for evaluation 
    return_item = {
        "time_costed": time_costed,
        "token_num": new_length,
    }
    return return_item


if __name__ == '__main__':
    args = get_args()
    config = OmegaConf.load(args.config)
    output_path = Path(args.output_dir)
    precision_scope = autocast
    os.makedirs(output_path, exist_ok=True)
    
    start_time = time.perf_counter()
    model = load_model_with_fallback(config, args.model_ckpt, gpu=True, eval_mode=True)
    print(f"[bold magenta][MINEWORLD][INFERENCE][/bold magenta] Load Model From {args.model_ckpt}")
    print(f"[DEBUG] Model loaded time: {time.perf_counter() - start_time}")
    # get accelearte algoritm
    args.accelerate_algo = args.accelerate_algo.lower()
    if args.accelerate_algo not in ACCELERATE_ALGO:
        print(f"[bold red][Warning][/bold red] {args.accelerate_algo} is not in {ACCELERATE_ALGO}, use naive")
        args.accelerate_algo = 'naive'
    num_item = 0
    for root, _, files in os.walk(args.data_root):
        files = [f for f in files  if f.endswith('.mp4')] # mp4 would not influence progress bar 
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        for file in tqdm(files):
            print(f"[DEBUG] file name: {file}")
            return_item = lvm_generate(args, model, output_path,file)
            num_item += 1
            if num_item  >= args.val_data_num:
                print(f"[bold magenta][MINEWORLD][INFERENCE][/bold magenta]  reach val data num limit {args.val_data_num}")
                break