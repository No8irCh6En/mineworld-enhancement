import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
import ast

from token_sampler import SpatialAngleSampler  # assumes token_sampler.py on PYTHONPATH / project root

def get_episode_name_variants(episode_name):
    """
    生成可能的文件夹/文件名变体，以应对 'episode_00027' vs '00027' 的命名不一致问题。
    """
    variants = []
    # 1. 用户输入的原始名称
    variants.append(episode_name)
    
    # 2. 如果没前缀，尝试加前缀
    if not episode_name.startswith("episode_"):
        variants.append(f"episode_{episode_name}")
    
    # 3. 如果有前缀，尝试去前缀
    if episode_name.startswith("episode_"):
        variants.append(episode_name.replace("episode_", ""))
        
    # 去重并保持顺序
    seen = set()
    unique_variants = []
    for v in variants:
        if v not in seen:
            unique_variants.append(v)
            seen.add(v)
    return unique_variants

def find_dataset_path(dataset_dir, sub_dir, episode_name, is_file=False, extension=""):
    """
    在 dataset_dir/sub_dir 下查找匹配 episode_name 的路径。
    """
    variants = get_episode_name_variants(episode_name)
    for v in variants:
        name = v + extension if is_file else v
        path = os.path.join(dataset_dir, sub_dir, name)
        if os.path.exists(path):
            return path
    return None

def load_images(image_dir):
    fns = sorted([fn for fn in os.listdir(image_dir) if fn.endswith(".png") or fn.endswith(".jpg")])
    imgs = []
    for fn in fns:
        imgs.append((fn, Image.open(os.path.join(image_dir, fn)).convert("RGB")))
    return imgs

def try_load_token_coords(dataset_dir, episode_name):
    # 查找 .npy 文件
    path = find_dataset_path(dataset_dir, "token_coords", episode_name, is_file=True, extension=".npy")
    
    if path is None:
        return None, None
        
    coords = np.load(path, allow_pickle=False)
    return coords, path

def try_load_actions(dataset_dir, episode_name):
    # 查找 action 文件夹
    action_dir = find_dataset_path(dataset_dir, "actions", episode_name, is_file=False)
    
    if action_dir is None:
        return None
        
    path = os.path.join(action_dir, "action.jsonl")
    if not os.path.exists(path):
        return None

    cams = []
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = ast.literal_eval(line)
            except Exception:
                try:
                    # fallback to unsafe eval if necessary
                    d = eval(line, {"__builtins__": None}, {})
                except Exception:
                    continue
            # common keys: 'camera' may be [yaw,pitch], 'pose' or 'position' may exist
            if "camera" in d:
                cams.append(np.array(d["camera"], dtype=float))
            else:
                cams.append(None)
            if "pose" in d:
                poses.append(np.array(d["pose"], dtype=float))
            elif "position" in d:
                poses.append(np.array(d["position"], dtype=float))
            else:
                poses.append(None)
    return {"path": path, "cameras": cams, "poses": poses}

def visualize_selection(dataset_dir, episode_name, out_dir, W, H, fov_deg, lambda_dist, dist_power, spatial_extra, thumb_size=16, max_cols=16):
    # 1. 查找图片目录
    image_dir = find_dataset_path(dataset_dir, "images", episode_name, is_file=False)
    
    if image_dir is None:
        raise FileNotFoundError(f"Image dir not found for episode '{episode_name}' in {os.path.join(dataset_dir, 'images')}")
        
    print(f"[INFO] Loading images from: {image_dir}")
    imgs = load_images(image_dir)
    N = len(imgs)
    if N == 0:
        raise RuntimeError("No images found")

    # 2. 加载坐标和动作
    frames_coords, coords_path = try_load_token_coords(dataset_dir, episode_name)
    if coords_path:
        print(f"[INFO] Loaded token coords from: {coords_path}")
    
    actions_info = try_load_actions(dataset_dir, episode_name)
    if actions_info:
        print(f"[INFO] Loaded actions from: {actions_info['path']}")

    # prepare frames_tokens: we only need indices shape [N, L]
    L = W * H
    ft = np.tile(np.arange(L, dtype=np.int32)[None, :], (N, 1))  # dummy token ids

    # frames_coords: if present must have shape [N, L, 3] or [T+1, L,3]
    if frames_coords is not None:
        if frames_coords.shape[0] < N:
            print(f"[WARN] token_coords has fewer frames ({frames_coords.shape[0]}) than images ({N}). Truncating N->{frames_coords.shape[0]}")
            N = frames_coords.shape[0]
            imgs = imgs[:N]
            ft = ft[:N]
        fc = frames_coords[:N]
    else:
        fc = None

    # build frames_camera (yaw,pitch) and frames_camera_pos if possible
    frames_camera = None
    frames_camera_pos = None
    if actions_info is not None:
        cams = actions_info["cameras"]
        poses = actions_info["poses"]
        # actions lines usually T entries for transitions; attempt to derive N frames cameras by prepending first camera if missing
        if len(cams) >= N:
            frames_camera = np.array([c if c is not None else [0.0,0.0] for c in cams[:N]])
        elif len(cams) > 0:
            # pad last
            arr = [c if c is not None else [0.0,0.0] for c in cams]
            while len(arr) < N:
                arr.append(arr[-1])
            frames_camera = np.array(arr[:N])
        if any(p is not None for p in poses):
            arrp = [p if p is not None else np.array([0.0,0.0,0.0]) for p in poses]
            while len(arrp) < N:
                arrp.append(arrp[-1])
            frames_camera_pos = np.array(arrp[:N], dtype=float)

    sampler = SpatialAngleSampler(W=W, H=H, fov_deg=fov_deg, lambda_dist=lambda_dist, dist_power=dist_power)
    # request sample
    sel_tokens, sel_meta = sampler.sample_tokens(
        frames_tokens=ft,
        frames_coords=fc,
        frames_camera=frames_camera,
        frames_camera_pos=frames_camera_pos,
        target_frame_idx=0,
        budget_total=spatial_extra,
        include_target_full=False,
    )

    print("[INFO] spatial_selected_meta:", sel_meta)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # visualize per frame: crop patches and paste thumbnails to the right of the image
    for idx, (fn, pil_img_orig) in enumerate(imgs):
        pil_img = pil_img_orig.copy()
        img_w, img_h = pil_img.size  # PIL gives (W, H)
        patch_h = img_h / H
        patch_w = img_w / W

        # collect selected tokens that belong to this frame
        toks = [r for (f, r) in sel_meta if f == idx]
        # sort tokens to have deterministic layout
        toks_sorted = sorted(toks)

        # determine thumbnail grid layout
        num = len(toks_sorted)
        if num == 0:
            # still save original with markers if any
            draw = ImageDraw.Draw(pil_img)
            out_path = os.path.join(out_dir, f"{episode_name}_frame_{idx:03d}_{fn}")
            pil_img.save(out_path)
            # print(f"[INFO] saved {out_path} (no selected patches)")
            continue

        cols = min(max_cols, max(1, num))
        rows = math.ceil(num / cols)
        thumb_w = thumb_size
        thumb_h = thumb_size

        # composite canvas: original on left, thumbnails on right
        pad = 8
        canvas_w = img_w + pad + cols * thumb_w
        canvas_h = max(img_h, rows * thumb_h)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
        canvas.paste(pil_img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        # draw original markers as small red dots for visibility
        for r in toks_sorted:
            rr = r // W
            cc = r % W
            cx = int(cc * patch_w + patch_w * 0.5)
            cy = int(rr * patch_h + patch_h * 0.5)
            rsize = max(2, int(min(img_w, img_h) * 0.01))
            draw.ellipse((cx - rsize, cy - rsize, cx + rsize, cy + rsize), outline="red", width=2)

        # paste thumbnails
        for i, r in enumerate(toks_sorted):
            rr = r // W
            cc = r % W
            x0 = int(cc * patch_w)
            y0 = int(rr * patch_h)
            x1 = int((cc + 1) * patch_w)
            y1 = int((rr + 1) * patch_h)
            # clamp
            x0 = max(0, min(img_w - 1, x0))
            y0 = max(0, min(img_h - 1, y0))
            x1 = max(x0 + 1, min(img_w, x1))
            y1 = max(y0 + 1, min(img_h, y1))
            patch = pil_img.crop((x0, y0, x1, y1))
            # resize patch to thumb_size x thumb_size
            patch_thumb = patch.resize((thumb_w, thumb_h), resample=Image.BILINEAR)
            col = i % cols
            row = i // cols
            dst_x = img_w + pad + col * thumb_w
            dst_y = row * thumb_h
            canvas.paste(patch_thumb, (dst_x, dst_y))
            # draw index label under thumbnail
            ld_x = dst_x + 1
            ld_y = dst_y + 1
            draw.text((ld_x, ld_y), str(r), fill="yellow")

        out_path = os.path.join(out_dir, f"{episode_name}_frame_{idx:03d}_{fn}")
        canvas.save(out_path)
        print(f"[INFO] saved {out_path} with {num} patches")

    # also save a summary heatmap over a chosen frame (e.g., target=0) marking selected indices across frames
    target_idx = 0
    base_img = imgs[target_idx][1].copy()
    draw = ImageDraw.Draw(base_img)
    # color by source frame (colormap)
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20")
    frame_colors = {}
    for i, (f, r) in enumerate(sel_meta):
        if f not in frame_colors:
            frame_colors[f] = tuple(int(255*x) for x in cmap(f % 20)[:3])
    for (f, r) in sel_meta:
        rr = r // W
        cc = r % W
        cx = int(cc * patch_w + patch_w * 0.5)
        cy = int(rr * patch_h + patch_h * 0.5)
        col = frame_colors.get(f, (255, 0, 0))
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=col, width=2)

    summary_path = os.path.join(out_dir, f"{episode_name}_summary_frame_{target_idx:03d}.png")
    base_img.save(summary_path)
    print(f"[INFO] saved summary {summary_path}")
    print(f"[INFO] sel_tokens len={len(sel_tokens)} sel_meta len={len(sel_meta)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/data/cliang/mineworld/dataset/", help="dataset root")
    parser.add_argument("--episode", type=str, default="episode_00027", help="episode name e.g. episode_00028 or 00028")
    parser.add_argument("--out_dir", type=str, default="./vis_out", help="output dir")
    parser.add_argument("--W", type=int, default=24, help="token grid width (columns)")
    parser.add_argument("--H", type=int, default=14, help="token grid height (rows)")
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    parser.add_argument("--dist_power", type=float, default=1.0)
    parser.add_argument("--spatial_extra", type=int, default=336*2, help="number of spatial tokens to select")
    parser.add_argument("--thumb_size", type=int, default=16, help="thumbnail size in pixels")
    parser.add_argument("--max_cols", type=int, default=16, help="max columns for thumbnails")
    args = parser.parse_args()

    visualize_selection(
        dataset_dir=args.dataset_dir,
        episode_name=args.episode,
        out_dir=args.out_dir,
        W=args.W,
        H=args.H,
        fov_deg=args.fov_deg,
        lambda_dist=args.lambda_dist,
        dist_power=args.dist_power,
        spatial_extra=args.spatial_extra,
        thumb_size=args.thumb_size,
        max_cols=args.max_cols,
    )

if __name__ == "__main__":
    main()