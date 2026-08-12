#!/usr/bin/env python3
# tools/get_patch_depth.py
import os
import csv
import math
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_pixel_csv(path, width=640, height=360, skip_header_lines=2):
    cam_yaw = None
    cam_pitch = None
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        first = next(reader)
        if len(first) == 1 and ',' in first[0]:
            kv = dict(p.split('=') for p in first[0].split(','))
            cam_yaw = float(kv.get('yaw', 0.0))
            cam_pitch = float(kv.get('pitch', 0.0))
        else:
            for tok in first:
                if tok.startswith('yaw='):
                    cam_yaw = float(tok.replace('yaw=', ''))
                if tok.startswith('pitch='):
                    cam_pitch = float(tok.replace('pitch=', ''))
        # skip header
        next(reader)
        total = width * height
        blocks = np.empty(total, dtype=object)
        dist_cent = np.empty(total, dtype=np.int32)
        yaw_deg = np.empty(total, dtype=np.float32)
        pitch_deg = np.empty(total, dtype=np.float32)
        idx = 0
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                parts = ','.join(row).split(',')
            else:
                parts = row
            block = parts[0]
            try:
                d = int(parts[1])
            except:
                d = -1
            try:
                ry = float(parts[2])
                rp = float(parts[3])
            except:
                ry = 0.0
                rp = 0.0
            if idx < total:
                blocks[idx] = block
                dist_cent[idx] = d
                yaw_deg[idx] = ry
                pitch_deg[idx] = rp
            idx += 1
            if idx >= total:
                break
        if idx < total:
            blocks[idx:] = ''
            dist_cent[idx:] = -1
            yaw_deg[idx:] = 0.0
            pitch_deg[idx:] = 0.0
        blocks = blocks.reshape((height, width))
        dist_cent = dist_cent.reshape((height, width))
        yaw_deg = yaw_deg.reshape((height, width))
        pitch_deg = pitch_deg.reshape((height, width))
    return cam_yaw, cam_pitch, blocks, dist_cent, yaw_deg, pitch_deg

def resize_angle_field_deg(angle_deg_arr, target_size):
    rad = np.deg2rad(angle_deg_arr.astype(np.float32))
    cx = np.cos(rad)
    sx = np.sin(rad)
    target_w, target_h = target_size
    cx_r = cv2.resize(cx, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    sx_r = cv2.resize(sx, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    ang_r = np.rad2deg(np.arctan2(sx_r, cx_r))
    return ang_r

def resize_scalar_field(arr, target_size):
    target_w, target_h = target_size
    return cv2.resize(arr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def aggregate_patches(dist_m, yaw_deg, pitch_deg, valid_mask, patch_h=16, patch_w=16):
    H, W = dist_m.shape
    n_rows = H // patch_h
    n_cols = W // patch_w
    results = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * patch_h
            y1 = y0 + patch_h
            x0 = c * patch_w
            x1 = x0 + patch_w
            dist_patch = dist_m[y0:y1, x0:x1]
            yaw_patch = yaw_deg[y0:y1, x0:x1]
            pitch_patch = pitch_deg[y0:y1, x0:x1]
            mask_patch = valid_mask[y0:y1, x0:x1]
            n_total = mask_patch.size
            n_valid = int(mask_patch.sum())
            valid_ratio = n_valid / n_total if n_total > 0 else 0.0
            if n_valid > 0:
                dist_vals = dist_patch[mask_patch]
                yaw_vals = yaw_patch[mask_patch]
                pitch_vals = pitch_patch[mask_patch]
                mean_dist = float(np.mean(dist_vals))
                median_dist = float(np.median(dist_vals))
                std_dist = float(np.std(dist_vals))
                mean_yaw = float(np.around(np.mean(yaw_vals), 4))
                std_yaw = float(np.std(yaw_vals))
                mean_pitch = float(np.mean(pitch_vals))
                std_pitch = float(np.std(pitch_vals))
            else:
                mean_dist = median_dist = std_dist = float('nan')
                mean_yaw = std_yaw = mean_pitch = std_pitch = float('nan')
            results.append({
                'patch_row': r,
                'patch_col': c,
                'patch_idx': r * n_cols + c,
                'valid_ratio': valid_ratio,
                'mean_dist_m': mean_dist,
                'median_dist_m': median_dist,
                'std_dist_m': std_dist,
                'mean_yaw_deg': mean_yaw,
                'std_yaw_deg': std_yaw,
                'mean_pitch_deg': mean_pitch,
                'std_pitch_deg': std_pitch,
            })
    return results

def generate_patches(input_csv, output_dir, src_w, src_h, grid_rows, grid_cols, patch_h, patch_w):
    cam_yaw, cam_pitch, blocks, dist_cent, yaw_deg, pitch_deg = read_pixel_csv(input_csv, width=src_w, height=src_h)
    dist_m = dist_cent.astype(np.float32) / 100.0
    valid_mask = dist_cent >= 0
    target_w = grid_cols * patch_w
    target_h = grid_rows * patch_h
    dist_m_resized = resize_scalar_field(dist_m, (target_w, target_h))
    valid_mask_resized = resize_scalar_field(valid_mask.astype(np.float32), (target_w, target_h)) >= 0.5
    pitch_resized = resize_scalar_field(pitch_deg, (target_w, target_h))
    yaw_resized = resize_angle_field_deg(yaw_deg, (target_w, target_h))
    patches = aggregate_patches(dist_m_resized, yaw_resized, pitch_resized, valid_mask_resized,
                                patch_h=patch_h, patch_w=patch_w)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_csv))[0]
    out_npz = os.path.join(output_dir, f"{base}_patches.npz")
    out_csv = os.path.join(output_dir, f"{base}_patches.csv")
    np.savez_compressed(out_npz,
                        camera_yaw_deg=cam_yaw,
                        camera_pitch_deg=cam_pitch,
                        dist_m_resized=dist_m_resized,
                        yaw_deg_resized=yaw_resized,
                        pitch_deg_resized=pitch_resized,
                        valid_mask_resized=valid_mask_resized,
                        patches_meta=patches)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['patch_idx','patch_row','patch_col','valid_ratio',
                  'mean_dist_m','median_dist_m','std_dist_m',
                  'mean_yaw_deg','std_yaw_deg','mean_pitch_deg','std_pitch_deg']
        writer.writerow(['camera_yaw_deg', cam_yaw])
        writer.writerow(['camera_pitch_deg', cam_pitch])
        writer.writerow([])
        writer.writerow(header)
        for p in patches:
            writer.writerow([p[k] for k in header])
    return out_npz, out_csv

def overlay_patch_depths(image_path, patches_npz_path, out_path=None,
                         patch_h=16, patch_w=16, grid_rows=16, grid_cols=24,
                         alpha=0.6, show_text=False, text_scale=0.45, text_thickness=1,
                         colormap='viridis', label_step=0):
    """
    label_step: int, if >0 only label patches where (row % label_step ==0 and col % label_step ==0)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    data = np.load(patches_npz_path, allow_pickle=True)
    if 'dist_m_resized' in data:
        dist_res = data['dist_m_resized']
    else:
        raise ValueError("dist_m_resized not found in npz")
    patches_meta = data.get('patches_meta', None)
    resized_h, resized_w = dist_res.shape
    scale_x = W / float(resized_w)
    scale_y = H / float(resized_h)
    n_rows = grid_rows
    n_cols = grid_cols
    mean_depths = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    if patches_meta is not None:
        try:
            meta = list(patches_meta)
            for p in meta:
                idx = int(p['patch_idx'])
                r = int(p['patch_row'])
                c = int(p['patch_col'])
                mean_depths[r, c] = float(p['mean_dist_m'])
        except Exception:
            pass
    if np.isnan(mean_depths).all():
        for r in range(n_rows):
            for c in range(n_cols):
                y0 = r * patch_h
                y1 = y0 + patch_h
                x0 = c * patch_w
                x1 = x0 + patch_w
                patch = dist_res[y0:y1, x0:x1]
                valid = ~np.isnan(patch)
                if valid.any():
                    mean_depths[r, c] = float(np.mean(patch[valid]))
                else:
                    mean_depths[r, c] = np.nan
    cmap = plt.get_cmap(colormap)
    valid_mask = ~np.isnan(mean_depths)
    if valid_mask.any():
        vmin = np.nanpercentile(mean_depths, 2)
        vmax = np.nanpercentile(mean_depths, 98)
    else:
        vmin, vmax = 0.0, 1.0
    overlay = img.copy().astype(np.float32) / 255.0
    canvas = overlay.copy()
    for r in range(n_rows):
        for c in range(n_cols):
            md = mean_depths[r, c]
            x0_r = int(round(c * patch_w * scale_x))
            x1_r = int(round((c + 1) * patch_w * scale_x))
            y0_r = int(round(r * patch_h * scale_y))
            y1_r = int(round((r + 1) * patch_h * scale_y))
            x0_r = max(0, x0_r)
            y0_r = max(0, y0_r)
            x1_r = min(W, x1_r)
            y1_r = min(H, y1_r)
            if np.isnan(md):
                continue
            t = (md - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            t = np.clip(t, 0.0, 1.0)
            color = np.array(cmap(t)[:3], dtype=np.float32)
            canvas[y0_r:y1_r, x0_r:x1_r, :] = (1.0 - alpha) * canvas[y0_r:y1_r, x0_r:x1_r, :] + alpha * color
            if show_text:
                do_label = True
                if label_step and label_step > 0:
                    do_label = (r % label_step == 0) and (c % label_step == 0)
                if do_label:
                    text = f"{md:.2f}m"
                    bgr_color = tuple(int(round(c*255)) for c in color[::-1])
                    org = (x0_r + 3, y0_r + int(12 * text_scale))
                    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0,0,0), thickness=text_thickness+2, lineType=cv2.LINE_AA)
                    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, text_scale, bgr_color, thickness=text_thickness, lineType=cv2.LINE_AA)
    out_img = (canvas * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    if out_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(os.path.dirname(patches_npz_path), f"{base}_patch_depth_overlay.png")
    cv2.imwrite(out_path, out_bgr)
    return out_path

def parse_args():
    p = argparse.ArgumentParser(description="Extract per-patch depth/angle features and optionally overlay on image.")
    p.add_argument("--input_csv", type=str, help="source pixel CSV (640x360)")
    p.add_argument("--output_dir", type=str, default="analysis_results/depth_gt/")
    p.add_argument("--src_w", type=int, default=640)
    p.add_argument("--src_h", type=int, default=360)
    p.add_argument("--grid_rows", type=int, default=16)
    p.add_argument("--grid_cols", type=int, default=24)
    p.add_argument("--patch_h", type=int, default=16)
    p.add_argument("--patch_w", type=int, default=16)
    p.add_argument("--overlay-image", dest="overlay_image", type=str, default=None, help="original image (640x360) to overlay patch depths")
    p.add_argument("--overlay-out", dest="overlay_out", type=str, default=None, help="output path for overlay image")
    p.add_argument("--patches-npz", dest="patches_npz", type=str, default=None, help="use existing npz instead of generating")
    p.add_argument("--only-overlay", action="store_true", help="only run overlay using provided --patches-npz and --overlay-image")
    # new CLI flags
    p.add_argument("--show-text", action="store_true", help="show depth text labels on overlay (default: off, use heatmap)")
    p.add_argument("--alpha", type=float, default=0.6, help="overlay alpha for heatmap (0-1)")
    p.add_argument("--colormap", type=str, default="viridis", help="matplotlib colormap name")
    p.add_argument("--label-step", type=int, default=0, help="if >0, label every label-step patches (sparse labels)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # overlay-only mode
    if args.only_overlay:
        if not args.patches_npz or not args.overlay_image:
            raise SystemExit("For --only-overlay provide --patches-npz and --overlay-image")
        out_path = args.overlay_out
        res = overlay_patch_depths(args.overlay_image, args.patches_npz, out_path,
                                   patch_h=args.patch_h, patch_w=args.patch_w,
                                   grid_rows=args.grid_rows, grid_cols=args.grid_cols,
                                   alpha=args.alpha, show_text=args.show_text,
                                   text_scale=0.45, text_thickness=1,
                                   colormap=args.colormap, label_step=args.label_step)
        print("Saved overlay:", res)
        raise SystemExit(0)
    # normal mode: generate (unless patches_npz provided)
    if args.patches_npz:
        patches_npz = args.patches_npz
    else:
        if not args.input_csv:
            raise SystemExit("Provide --input_csv to generate patches (or use --patches-npz)")
        patches_npz, patches_csv = generate_patches(args.input_csv, args.output_dir,
                                                   args.src_w, args.src_h,
                                                   args.grid_rows, args.grid_cols,
                                                   args.patch_h, args.patch_w)
        print("Generated:", patches_npz, patches_csv)
    # optional overlay
    if args.overlay_image:
        out_path = args.overlay_out
        res = overlay_patch_depths(args.overlay_image, patches_npz, out_path,
                                   patch_h=args.patch_h, patch_w=args.patch_w,
                                   grid_rows=args.grid_rows, grid_cols=args.grid_cols,
                                   alpha=args.alpha, show_text=args.show_text,
                                   text_scale=0.45, text_thickness=1,
                                   colormap=args.colormap, label_step=args.label_step)
        print("Saved overlay:", res)