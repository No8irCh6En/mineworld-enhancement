#!/usr/bin/env python3
"""生成论文图：197-clip 速度比分布 + 20-clip PSNR 对比。"""
import os, sys, re, cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "exp_results/paper_20260822"
FIGDIR = "paper/figures"
os.makedirs(FIGDIR, exist_ok=True)

# ---------- 图1：197-clip 速度比分布 ----------
def extract_baseline(logpath):
    out = {}
    cur = None
    for line in open(logpath, encoding='utf-8', errors='ignore'):
        m = re.search(r'file name: clip_(\d+)\.mp4', line)
        if m: cur = int(m.group(1))
        m2 = re.search(r'cost [\d.]+ second; [\d.]+ token/sec ([\d.]+) fps', line)
        if m2 and cur is not None:
            out[cur] = float(m2.group(1)); cur = None
    return out

def extract_spec(logpath):
    out = {}
    cur = None
    for line in open(logpath, encoding='utf-8', errors='ignore'):
        m = re.search(r'Processing clip_(\d+) \(MP4\)', line)
        if m: cur = int(m.group(1))
        m2 = re.search(r'\(([\d.]+) fps\)', line)
        if m2 and cur is not None and 'Speculative Gen' in line:
            out[cur] = float(m2.group(1)); cur = None
    return out

b = extract_baseline(os.path.join(BASE, "base_clip4.log"))
s = extract_spec(os.path.join(BASE, "spec_clip4.log"))
common = sorted(set(b) & set(s))
# 剔除编译预热（<1.0 fps）
common = [c for c in common if b[c] >= 1.0 and s[c] >= 1.0]
ratio = np.array([s[c]/b[c] for c in common])

fig, ax = plt.subplots(figsize=(3.3, 2.2))
ax.hist(ratio, bins=30, color='#4a90d9', edgecolor='black', linewidth=0.4, alpha=0.85)
ax.axvline(1.0, color='red', linestyle='--', linewidth=1.2, label='parity')
ax.axvline(ratio.mean(), color='green', linestyle='-', linewidth=1.2, label=f'mean={ratio.mean():.3f}')
ax.set_xlabel('Speedup ratio (spec / baseline)')
ax.set_ylabel('Clips')
ax.set_title(f'{len(common)} clips')
ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "speedup_hist.pdf"), dpi=300, bbox_inches='tight')
plt.close()
print(f"[图1] speedup_hist.pdf: n={len(common)}, mean={ratio.mean():.4f}, frac>1={(ratio>1).mean():.3f}")

# ---------- 图2：20-clip PSNR 对比 ----------
def load_frames(mp4):
    cap = cv2.VideoCapture(mp4)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
    cap.release()
    return frames

def psnr(a, b):
    mse = np.mean((a-b)**2)
    return 100.0 if mse < 1e-10 else 10*np.log10(1.0/mse)

data_root = "/data/cliang/mineworld/validation/validation"
base_mp4s = set(os.listdir(os.path.join(BASE,'base_clip12')))
spec_mp4s = set(os.listdir(os.path.join(BASE,'spec_clip4')))
common_mp4 = sorted(base_mp4s & spec_mp4s)[:20]

base_psnr, spec_psnr, labels = [], [], []
for c in common_mp4:
    gp, bp, sp = os.path.join(data_root,c), os.path.join(BASE,'base_clip12',c), os.path.join(BASE,'spec_clip4',c)
    if not all(os.path.exists(p) for p in [gp,bp,sp]): continue
    gt, bf, sf = load_frames(gp), load_frames(bp), load_frames(sp)
    n = min(len(gt)-1, len(bf), len(sf))
    if n <= 0: continue
    bp_ = np.mean([psnr(bf[i], gt[i+1]) for i in range(n)])
    sp_ = np.mean([psnr(sf[i], gt[i+1]) for i in range(n)])
    base_psnr.append(bp_); spec_psnr.append(sp_); labels.append(c.replace('.mp4','').replace('clip_',''))

fig, ax = plt.subplots(figsize=(3.3, 2.2))
x = np.arange(len(labels))
w = 0.38
ax.bar(x-w/2, base_psnr, w, label='baseline', color='#2ecc71', edgecolor='black', linewidth=0.4)
ax.bar(x+w/2, spec_psnr, w, label='speculative', color='#e74c3c', edgecolor='black', linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, fontsize=5)
ax.set_ylabel('PSNR (dB)')
ax.set_xlabel('Clip')
ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "psnr_compare.pdf"), dpi=300, bbox_inches='tight')
plt.close()
print(f"[图2] psnr_compare.pdf: n={len(labels)}, base={np.mean(base_psnr):.2f}, spec={np.mean(spec_psnr):.2f}, Δ={np.mean(base_psnr)-np.mean(spec_psnr):.2f}")
