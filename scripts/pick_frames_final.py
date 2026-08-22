#!/usr/bin/env python3
"""最终版：挑选高质量 Minecraft 帧并保存到 paper/figures/frames/。

指标：清晰度(Laplacian方差) + 对比度(灰度std) + 饱和度，综合评分。
排除异常帧（lap 过高疑似噪声）。baseline 和 spec 各选 Top 8。
"""
import os, cv2
import numpy as np

BASE = "exp_results/paper_20260822"
OUT = "paper/figures/frames"
os.makedirs(OUT, exist_ok=True)

def quality(fr):
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    std = gray.std()
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean()
    return lap, std, sat

def scan(dirname):
    d = os.path.join(BASE, dirname)
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.mp4'):
            continue
        cap = cv2.VideoCapture(os.path.join(d, f))
        frames = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            frames.append(fr)
        cap.release()
        if len(frames) < 3:
            continue
        best_i, best_s = 0, -1
        for i, fr in enumerate(frames):
            lap, std, sat = quality(fr)
            s = np.log1p(lap) + std * 0.3 + sat * 0.05
            if s > best_s:
                best_s, best_i = s, i
        out.append({
            'clip': f.replace('.mp4', '').replace('clip_', ''),
            'best_i': best_i,
            'best_s': best_s,
            'frame': frames[best_i],
            'lap': quality(frames[best_i])[0],
        })
    return out

def main():
    for dirname, kind in [('base_clip12', 'baseline'), ('spec_clip4', 'spec')]:
        rs = scan(dirname)
        kept = [r for r in rs if 300 <= r['lap'] <= 2500]
        kept.sort(key=lambda r: -r['best_s'])
        print(f'\n=== {kind}: Top 8 候选帧 ===')
        for rank, r in enumerate(kept[:8], 1):
            fn = f"{kind}_candidate_{rank:02d}_clip{r['clip']}_frame{r['best_i']}.png"
            cv2.imwrite(os.path.join(OUT, fn), r['frame'])
            print(f"  #{rank}: {fn} (lap={r['lap']:.0f})")

    print(f'\n候选帧已保存到 {OUT}/')

if __name__ == '__main__':
    main()
