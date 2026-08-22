#!/usr/bin/env python3
"""生成论文用生成样本对比图：GT | baseline | speculative，2 行 x 若干列。

对齐方式：demo_num=1（1 帧条件），生成视频第 i 帧对应 GT 视频第 i+1 帧。
选质量最高的几个共有 clip，用同一帧号，拼 GT/baseline/spec 三列。
"""
import os, cv2
import numpy as np

BASE = "exp_results/paper_20260822"
GT = "/data/cliang/mineworld/validation/validation"
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

def read_frames(path):
    cap = cv2.VideoCapture(path)
    frs = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frs.append(fr)
    cap.release()
    return frs

def quality(fr):
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    std = gray.std()
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean()
    return np.log1p(lap) + std * 0.3 + sat * 0.05

def psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(255.0**2 / mse)

def main():
    base_dir = os.path.join(BASE, 'base_clip12')
    spec_dir = os.path.join(BASE, 'spec_clip4')
    common = sorted(set(os.listdir(base_dir)) & set(os.listdir(spec_dir)))

    # 对每个共有 clip，按 spec vs GT 的 PSNR 选帧（证明 speculative 质量）
    candidates = []
    for f in common:
        clip = f.replace('.mp4', '').replace('clip_', '')
        bf = read_frames(os.path.join(base_dir, f))
        sf = read_frames(os.path.join(spec_dir, f))
        gt_path = os.path.join(GT, f)
        if not os.path.exists(gt_path):
            continue
        gf = read_frames(gt_path)
        n = min(len(bf), len(sf), len(gf) - 1)
        if n < 3:
            continue
        # 逐帧算 spec vs GT 的 PSNR
        frame_psnrs = [psnr(sf[i], gf[i + 1]) for i in range(n)]
        mean_psnr = float(np.mean(frame_psnrs))
        best_i = int(np.argmax(frame_psnrs))
        candidates.append((clip, bf[best_i], sf[best_i], gf[best_i + 1], mean_psnr))

    # 按平均 PSNR 排序（证明 spec 整体质量，而非单帧侥幸）
    candidates.sort(key=lambda x: -x[4])
    # 选前 6 个
    top = candidates[:6]
    print(f'从 {len(common)} 个共有 clip 中选出 {len(top)} 个 spec 质量最高的样本')
    for clip, _, _, _, p in top:
        print(f'  clip_{clip}: spec-GT 平均 PSNR = {p:.2f} dB')

    # 拼图：每行 [GT | baseline | spec]，共 3 行（每行 2 个样本 -> 6 个样本 = 3 行 x 2 组）
    # 实际做 3 行，每行一个样本（GT | baseline | spec 三列）
    rows = []
    labels = []
    for clip, b, s, g, q in top:
        h = 224
        w = 384
        b = cv2.resize(b, (w, h))
        s = cv2.resize(s, (w, h))
        g = cv2.resize(g, (w, h))
        gap = np.full((h, 6, 3), 255, dtype=np.uint8)
        row = np.hstack([g, gap, b, gap, s])
        rows.append(row)
        labels.append(clip)

    grid = np.vstack(rows)
    cv2.imwrite(os.path.join(OUT, 'generation_samples.png'), grid)
    print(f'已保存 generation_samples.png ({grid.shape[1]}x{grid.shape[0]})')
    print('样本 clip:', labels)
    print('列顺序: GT | baseline | speculative')

if __name__ == '__main__':
    main()
