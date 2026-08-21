#!/usr/bin/env python3
"""重建论文用 FPS 统计：按 clip 名配对 baseline/spec，剔除编译预热 demo。

数据来源：
- base_clip4.log / base_clip11.log: inference.py 全验证集（自然序）
- spec_clip4.log: inference_speculative.py 全验证集（字典序）
每行 FPS 都对应一个独立 clip。首个 demo 是编译预热（0.269 fps），剔除。
"""
import os, re, glob, json
import numpy as np

BASE = "exp_results/paper_20260822"

def extract_baseline(logpath):
    """从 baseline 日志提取 (clip_name, fps) 列表。"""
    out = []
    cur_clip = None
    with open(logpath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'file name: clip_(\d+)\.mp4', line)
            if m:
                cur_clip = int(m.group(1))
            m2 = re.search(r'cost [\d.]+ second; [\d.]+ token/sec ([\d.]+) fps', line)
            if m2 and cur_clip is not None:
                out.append((cur_clip, float(m2.group(1))))
                cur_clip = None
    return out

def extract_spec(logpath):
    """从 spec 日志提取 (clip_name, fps) 列表。"""
    out = []
    cur_clip = None
    with open(logpath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'Processing clip_(\d+) \(MP4\)', line)
            if m:
                cur_clip = int(m.group(1))
            m2 = re.search(r'\(([\d.]+) fps\)', line)
            if m2 and cur_clip is not None and 'Speculative Gen' in line:
                out.append((cur_clip, float(m2.group(1))))
                cur_clip = None
    return out

def main():
    base_all = extract_baseline(os.path.join(BASE, "base_clip4.log"))
    spec_all = extract_spec(os.path.join(BASE, "spec_clip4.log"))

    # 剔除编译预热：baseline 的第一个是 0.269；spec 里 clip_10 是第一个（预热）
    # 更稳妥：剔除 < 1.0 fps 的异常值（编译预热）
    base_clean = [(c, f) for c, f in base_all if f >= 1.0]
    spec_clean = [(c, f) for c, f in spec_all if f >= 1.0]

    print(f"baseline 总样本 {len(base_all)}，剔除预热后 {len(base_clean)}")
    print(f"spec     总样本 {len(spec_all)}，剔除预热后 {len(spec_clean)}")

    # 配对：只保留两方都有的 clip
    base_map = dict(base_clean)
    spec_map = dict(spec_clean)
    common = sorted(set(base_map) & set(spec_map))
    print(f"配对 clip 数: {len(common)}")

    b = np.array([base_map[c] for c in common])
    s = np.array([spec_map[c] for c in common])
    speedup = s / b

    print(f"\n=== 配对统计（{len(common)} clips）===")
    print(f"baseline: mean={b.mean():.3f}  std={b.std():.3f}  median={np.median(b):.3f}")
    print(f"spec:     mean={s.mean():.3f}  std={s.std():.3f}  median={np.median(s):.3f}")
    print(f"speedup:  mean={speedup.mean():.4f}  median={np.median(speedup):.4f}")
    print(f"spec 超过 baseline 的 clip 占比: {(speedup > 1.0).mean()*100:.1f}%")

    result = {
        "n_clips": len(common),
        "baseline_mean": round(float(b.mean()), 3),
        "baseline_std": round(float(b.std()), 3),
        "spec_mean": round(float(s.mean()), 3),
        "spec_std": round(float(s.std()), 3),
        "speedup_mean": round(float(speedup.mean()), 4),
        "speedup_median": round(float(np.median(speedup)), 4),
        "frac_spec_faster": round(float((speedup > 1.0).mean()), 4),
    }
    with open(os.path.join(BASE, "fps_stats_paired.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\n已写入 fps_stats_paired.json")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
