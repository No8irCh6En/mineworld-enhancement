#!/usr/bin/env python3
"""生成 one-shot 帧并行解码的乱码帧，用于论文对比图。

原理：FRAME_PARALLEL=1 启用 only_previous mask，帧内 336 token 互相独立，
一次 forward 生成整帧。保存生成帧供对比。
"""
import os, sys, cv2
import numpy as np
import torch
from torchvision import transforms
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.helper import load_model
from mcdataset import MCDataset
from diagonal_decoding import sample_n_top_p

TARGET_SIZE = (224, 384)
PIX_NUM = 336
ACT_NUM = 11
safe_globals = {"array": np.array}

def main():
    os.environ["FRAME_PARALLEL"] = "1"
    device = "cuda"
    config = OmegaConf.load("configs/modify.yaml")
    model = load_model(config, "checkpoints/300M_16f.ckpt", gpu=True, eval_mode=True)
    trans = model.transformer

    data_root = "/data/cliang/mineworld/validation/small_validation"
    base_name = "clip_25"
    num_frames = 6  # 生成 6 帧即可

    mcdataset = MCDataset()
    action_list = []
    with open(os.path.join(data_root, base_name + ".jsonl"), 'r') as f:
        for line in f:
            line = eval(line.strip(), {"__builtins__": None}, safe_globals)
            line['camera'] = np.array(line['camera'])
            action_list.append(mcdataset.get_action_index_from_actiondict(line, action_vocab_offset=8192))
    gt_actions = torch.tensor(action_list[:num_frames + 1], device=device, dtype=torch.long)

    # 加载 GT 帧
    cap = cv2.VideoCapture(os.path.join(data_root, base_name + ".mp4"))
    raw = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        raw.append(f)
    cap.release()

    def tokenize(f):
        f2 = cv2.resize(f, (384, 224), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(f2.astype(np.float32)).permute(2, 0, 1).float().cuda() / 255.0
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        t = norm(t).unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            tok = model.tokenizer.tokenize_images(t)
        return tok.view(-1)

    gt_tokens = [tokenize(raw[i]) for i in range(num_frames + 1)]

    # one-shot 生成
    generated = []
    prev = gt_tokens[0]
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for t in range(num_frames):
            act = gt_actions[t].view(1, -1)
            inp = torch.cat([prev.view(1, -1), act], dim=1)
            pos = torch.arange(0, inp.shape[1], device=device)
            logits = trans(input_ids=inp, position_ids=pos)
            nf = sample_n_top_p(logits[:, -PIX_NUM:, :], temperature=1.0, top_p=0.8).view(-1)
            generated.append(nf)
            prev = nf
    torch.cuda.synchronize()

    # 解码为图像
    imgs = []
    for tok in generated:
        img = model.tokenizer.token2image_gpu(tok.view(1, 14, 24))
        # [1,3,H,W] in [-1,1] -> numpy RGB [H,W,3] in [0,255]
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)

    # 保存：GT 前 3 帧 + one-shot 前 3 帧 拼接
    out_dir = "paper/figures"
    os.makedirs(out_dir, exist_ok=True)
    # GT 帧（对应第 1~3 帧）
    gt_imgs = []
    for i in range(1, 4):
        img = model.tokenizer.token2image_gpu(gt_tokens[i].view(1, 14, 24))
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip((img * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_imgs.append(img)

    # 拼图：GT 三帧 | one-shot 三帧
    gap = np.full((224, 6, 3), 255, dtype=np.uint8)
    row_gt = gt_imgs[0]
    for im in gt_imgs[1:]:
        row_gt = np.hstack([row_gt, gap, im])
    row_os = imgs[0]
    for im in imgs[1:3]:
        row_os = np.hstack([row_os, gap, im])
    big_gap = np.full((20, row_gt.shape[1], 3), 255, dtype=np.uint8)
    full = np.vstack([row_gt, big_gap, row_os])
    cv2.imwrite(os.path.join(out_dir, "oneshot_collapse.png"), full)
    print(f'已保存 oneshot_collapse.png ({full.shape[1]}x{full.shape[0]})')

if __name__ == '__main__':
    main()
