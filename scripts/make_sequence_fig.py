#!/usr/bin/env python3
"""生成 Background 用图：MineWorld 的连续帧序列（真实 GT 帧）。

从验证集 clip 中取连续 5 帧，拼接成横向序列图，展示 Minecraft 场景。
"""
import os, cv2
import numpy as np

GT = "/data/cliang/mineworld/validation/validation"
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

def read_frames(path, max_n=20):
    cap = cv2.VideoCapture(path)
    frs = []
    while True:
        ret, fr = cap.read()
        if not ret or len(frs) >= max_n:
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

def main():
    # 从之前选出的高质量 clip 里取序列：clip_26, clip_27, clip_28, clip_53, clip_54, clip_47
    clips = ['clip_26', 'clip_27', 'clip_28', 'clip_53', 'clip_54', 'clip_47']
    # 对每个 clip 找质量最高的连续 5 帧窗口
    best_frames, best_clip, best_q = None, None, -1
    for c in clips:
        path = os.path.join(GT, c + '.mp4')
        if not os.path.exists(path):
            continue
        frs = read_frames(path)
        for start in range(0, len(frs) - 5):
            window = frs[start:start+5]
            q = sum(quality(f) for f in window)
            if q > best_q:
                best_q, best_frames, best_clip = q, window, c
    print(f'选中 clip_{best_clip}，质量分 {best_q:.1f}')

    # 拼接 5 帧横向序列
    h, w = 224, 384
    gap = np.full((h, 6, 3), 255, dtype=np.uint8)
    imgs = [cv2.resize(f, (w, h)) for f in best_frames]
    row = imgs[0]
    for im in imgs[1:]:
        row = np.hstack([row, gap, im])
    cv2.imwrite(os.path.join(OUT, 'world_model_sequence.png'), row)
    print(f'已保存 world_model_sequence.png ({row.shape[1]}x{row.shape[0]})')

if __name__ == '__main__':
    main()
