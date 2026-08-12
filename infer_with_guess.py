import os
import cv2
import time
import torch
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

# 禁用 TF32 以保证精度
torch.backends.cuda.matmul.allow_tf32 = False

def get_latest_zero3_ckpt_dir(output_dir: str) -> str:
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

    ckpt_dir = get_latest_zero3_ckpt_dir(ckpt_path)  # 替换为你的输出目录路径
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)

    print(f"[INFO] {len(state_dict)} keys in checkpoint.")

    # 4. 加载权重到你构建的模型结构中
    model.load_state_dict(state_dict, strict=True)

    # 5. 设定为 eval + GPU
    if gpu:
        model = model.cuda()
    if eval_mode:
        model.eval()

    return model


# 常量定义
TOKEN_PER_IMAGE = 347 # IMAGE = PIX+ACTION
TOKEN_PER_PIX = 336   # 14 * 24 patches
H_PATCHES = 14
W_PATCHES = 24

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="/data/jjli/workspace/mineworld/checkpoints/300M_16f.ckpt")
    parser.add_argument("--config", type=str, default="configs/modify.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs_guess")
    parser.add_argument("--data_root", type=str, default="/data/cliang/mineworld/validation/validation")
    parser.add_argument("--demo_num", type=int, default=1, help="Start frame index")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--guess_step", type=int, default=2, help="Parallelism degree (default 2)")
    parser.add_argument('--val_data_num', type=int, default=500, help="number of validation data")
    return parser.parse_args()

def token2video(code_list, tokenizer, save_path, fps, device = 'cuda'):
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
    for i in range(num_images):
        code = code_list[i*TOKEN_PER_PIX:(i+1)*TOKEN_PER_PIX]
        code = torch.tensor([int(x) for x in code], dtype=torch.long).to(device)
        img = tokenizer.token2image(code) # pixel
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()


safe_globals = {"array": np.array}


def lvm_generate(args, model, output_dir, demo_video):
    """
    仿照 inference.py 的 lvm_generate，但使用 speculative_generate
    """
    device = model.transformer.device
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    output_mp4_path = str(output_dir / demo_video)
    
    if os.path.exists(output_mp4_path):
        print(f"{output_mp4_path} exists, skip generation.")
        return {}
    
    # 1. 准备路径
    input_mp4_path = os.path.join(args.data_root, demo_video)
    input_action_path = os.path.join(args.data_root, demo_video.replace('mp4','jsonl'))
    
    print(f"Processing {demo_video}...")
    
    # 2. 加载动作 (Action)
    action_list = []
    mcdataset = MCDataset()
    with open(input_action_path, 'r') as f:
        for line in f:
            # 安全加载 action dict
            line = eval(line.strip(), {"__builtins__": None}, safe_globals)
            line['camera'] = np.array(line['camera'])
            act_index = mcdataset.get_action_index_from_actiondict(line, action_vocab_offset=8192)
            action_list.append(act_index)

    # 3. 加载视频帧 (Video Frames)
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
    
    # 4. Tokenize 图像
    print("Tokenizing input frames...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        # 假设 model.tokenizer.tokenize_images 返回 [B, T, H, W] 或类似 indices
        img_index = model.tokenizer.tokenize_images(frames)
        img_index = rearrange(img_index, '(b t) h w -> b t (h w)', b=1)
            
    # 5. 准备输入
    # 截取需要的 action
    action_all = action_list[args.demo_num : args.demo_num + args.frames]
    action_all = torch.tensor(action_all).unsqueeze(1).to(device) # [Frames]
    
    image_input = rearrange(img_index, 'b t c -> b (t c)')
    
    # 6. 执行推测生成 (Speculative Generation)
    print(f"Starting Speculative Generation (Step={args.guess_step})...")
    start_t = time.time()
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        
        # ==============================================================================
        # 假设 lvm.py 中的 LlamaForCausalLM 类有一个名为 speculative_generate 的接口
        # ==============================================================================
        
        # 接口定义假设 (Spec Definition):
        # ------------------------------------------------------------------------------
        # def speculative_generate(
        #     self, 
        #     input_ids: torch.Tensor,      # [1, Seq_Len] 初始上下文
        #     max_new_tokens: int,          # 需要生成的总 Token 数
        #     action_all: torch.Tensor,     # [Frames] 动作序列，用于注入
        #     guess_step: int = 2,          # 并行度/步长 (K)
        #     top_k: int = None,
        #     top_p: float = None,
        #     temperature: float = 1.0
        # ) -> torch.Tensor:                # 返回生成的完整序列 [1, Seq_Len + max_new_tokens]
        # ------------------------------------------------------------------------------
        #
        # 功能对比 (vs naive_generate):
        # 1. Naive: 
        #    Loop max_new_tokens 次:
        #       Forward([Current]) -> Next Token
        #       KV Cache Update (1 token)
        #
        # 2. Speculative:
        #    Loop (max_new_tokens / K) 次:
        #       a. Draft: 从上一帧相同位置提取 K 个 token 作为 [Guess_1, ..., Guess_K]
        #       b. Verify: 构造输入 [Current, Guess_1, ..., Guess_K-1]
        #          Forward 一次 (利用 Attention Mask 并行计算 K 个位置的 Logits)
        #       c. Tree Check: 
        #          - 检查 Pos 0 输出是否匹配 Guess_1
        #          - 检查 Pos 1 输出是否匹配 Guess_2
        #          ...
        #       d. Rollback & KV Cache Management:
        #          - 如果 Guess_i 错误，丢弃后续结果
        #          - **关键**: 回滚 KV Cache，删除错误猜测产生的 Key/Value
        # ------------------------------------------------------------------------------

        output_ids = model.transformer.speculative_generate_single_pic(
            input_ids=image_input,
            max_new_tokens=TOKEN_PER_PIX * args.frames,
            action_all=action_all,
            guess_step=args.guess_step,  # 这是新增的参数，控制并行度
            top_k=args.top_k,
            top_p=args.top_p
        )
        
    ## TODO: CHECK
        
    end_t = time.time()
    print(f"Generation done in {end_t - start_t:.2f}s")
    
    # 7. 保存视频
    # 提取生成的 tokens (去掉 context)
    all_generated_tokens = []
    all_generated_tokens.extend(output_ids.tolist()[0])
    new_length = len(all_generated_tokens)
    
    time_costed = end_t - start_t 
    token_per_sec = new_length / time_costed
    frame_per_sec = token_per_sec / TOKEN_PER_PIX
    print(f"{new_length} token generated; cost {time_costed:.3f} second; {token_per_sec:.3f} token/sec {frame_per_sec:.3f} fps")
    token2video(all_generated_tokens, model.tokenizer, str(output_path / demo_video), args.fps, device)
    
    return_item = {
        "time_costed": time_costed,
        "token_num": new_length,
    }
    return return_item

def main():
    # args = get_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # print(f"Loading model from {args.model_ckpt}")
    # model = load_model(args.config, args.model_ckpt)
    # model.to(device)
    # model.eval()
    
    # # 获取文件列表 (模拟 inference.py 的逻辑)
    # files = list(Path(args.data_root).glob("*.mp4"))
    # files.sort()
    
    # if len(files) == 0:
    #     print(f"No mp4 files found in {args.data_root}")
    #     return

    # # 处理第一个视频作为演示
    # demo_video = files[0].name
    # lvm_generate(args, model, args.output_dir, demo_video)
    
    args = get_args()
    config = OmegaConf.load(args.config)
    output_path = Path(args.output_dir)
    precision_scope = autocast
    os.makedirs(output_path, exist_ok=True)
    
    start_time = time.perf_counter()
    model = load_model_with_fallback(config, args.model_ckpt, gpu=True, eval_mode=True)
    print(f"[bold magenta][MINEWORLD][INFERENCE][/bold magenta] Load Model From {args.model_ckpt}")
    print(f"[DEBUG] Model loaded time: {time.perf_counter() - start_time}")
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

if __name__ == "__main__":
    main()