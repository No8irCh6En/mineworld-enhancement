#!/usr/bin/env python3
"""评估每个共有 clip 的 spec 生成质量（vs GT），按 PSNR 排序找出 spec 最好的 clip。"""
import os, cv2
import numpy as np

BASE = "exp_results/paper_20260822"
GT = "/data/cliang/mineworld/validation/validation"

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

    results = []
    for f in common:
        clip = f.replace('.mp4', '').replace('clip_', '')
        bf = read_frames(os.path.join(base_dir, f))
        sf = read_frames(os.path.join(spec_dir, f))
        gt_path = os.path.join(GT, f)
        if not os.path.exists(gt_path):
            continue
        gf = read_frames(gt_path)
        n = min(len(bf), len(sf), len(gf) - 1)  # 生成帧 i 对应 GT i+1
        if n < 3:
            continue
        # spec vs GT 的逐帧 PSNR，取平均
        psnrs = []
        for i in range(n):
            psnrs.append(psnr(sf[i], gf[i + 1]))
        mean_psnr = float(np.mean(psnrs))
        results.append({'clip': clip, 'mean_psnr': mean_psnr, 'n': n})

    results.sort(key=lambda x: -x['mean_psnr'])
    print(f'共 {len(results)} 个 clip，按 spec vs GT 平均 PSNR 排序（Top 20）:')
    for r in results[:20]:
        print(f"  clip_{r['clip']}: spec-GT PSNR = {r['mean_psnr']:.2f} dB ({r['n']} 帧)")

    # 保存完整排名
    import json
    with open(os.path.join(BASE, 'spec_psnr_ranking.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\n已保存排名到 {BASE}/spec_psnr_ranking.json')

if __name__ == '__main__':
    main()
