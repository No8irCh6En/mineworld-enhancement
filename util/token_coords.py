"""
工具：基于 action.jsonl + 深度预测，生成并缓存每帧每个 image-token 的三维坐标 (token_coords)
本版本增加了基于 action（camera, forward/back/left/right/jump）估计相机位姿的类 ActionPoseEstimator，
将移动（固定步长）和跳跃高度融合到世界坐标计算中，输出 (T, N, 3) 的 world 坐标。
导出函数：
- generate_token_coords_for_episode(episode_id, images, action_jsonl_path, depth_predictor, frame_tokenizer, out_path=None, fov_deg=70, step_length=0.5, jump_height=1.0)
- load_token_coords(path)
"""
import os
import json
from pathlib import Path
from typing import List, Tuple, Callable, Optional
from PIL import Image
from tqdm import tqdm
import math
import numpy as np
import torch

def load_actions_from_jsonl(jsonl_path: str) -> List[dict]:
    actions = []
    if not jsonl_path or not os.path.exists(jsonl_path):
        return actions
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                actions.append(json.loads(line))
            except Exception:
                continue
    return actions

def estimate_intrinsics(width: int, height: int, fov_deg: float = 70.0) -> Tuple[float,float,float,float]:
    fov = math.radians(fov_deg)
    fx = 0.5 * width / math.tan(fov/2)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def load_token_coords(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    return np.load(path)

def save_token_coords(path: str, coords: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, coords)

class ActionPoseEstimator:
    """
    基于每帧 action（包含 camera [yaw,pitch] 及 movement flags forward/back/left/right/jump）
    估计每帧的相机世界位姿（position, rotation yaw/pitch）。
    简单运动模型：每帧前进/后退/横移按 step_length（m/frame）变化；当检测到 sneak/sprint 时按速度比缩放步幅。
    输出每帧 pose dict: {"position":[x,y,z], "rotation":[yaw_deg, pitch_deg, roll_deg]}
    """
    def __init__(self, fps: float = 20.0, jump_height: float = 1.0):
        self.jump_height = float(jump_height)
        # 基准速度（m/s），用于计算 sneak/sprint 的倍数关系（按 Minecraft wiki）
        self.walk_speed = 4.317
        self.sneak_speed = 1.295
        self.sprint_speed = 5.612
        self.step_length = 1.0 / fps * self.walk_speed

    def estimate_poses(self, actions: List[dict], initial_pos: Tuple[float,float,float]=(0.0,0.0,0.0)) -> List[dict]:
        poses = []
        pos = np.array(initial_pos, dtype=float)
        for a in actions:
            cam = a.get("camera", [0.0, 0.0])
            try:
                yaw_deg = float(cam[0])
                pitch_deg = float(cam[1])
            except Exception:
                yaw_deg = 0.0
                pitch_deg = 0.0

            # movement flags/values (may be 0/1 or float)
            forward = float(a.get("forward", 0))
            back = float(a.get("back", 0))
            right = float(a.get("right", 0))
            left = float(a.get("left", 0))
            jump = float(a.get("jump", 0))

            # detect sprint / sneak and compute per-frame stride
            sprint_flag = float(a.get("sprint", 0))
            sneak_flag = float(a.get("sneak", 0))
            stride = self.step_length
            # 优先 sprint，再 sneak；按速度比缩放（stride = base_stride * (speed_x / walk_speed)）
            if sprint_flag > 0.5:
                stride *= (self.sprint_speed / self.walk_speed)
            elif sneak_flag > 0.5:
                stride *= (self.sneak_speed / self.walk_speed)

            # local movement components (m/frame)
            longitudinal = (forward - back) * stride
            lateral = (right - left) * stride

            yaw_rad = math.radians(yaw_deg)
            # forward dir in world (x,y)
            fwd = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=float)
            # right dir
            rgt = np.array([math.cos(yaw_rad + math.pi/2), math.sin(yaw_rad + math.pi/2)], dtype=float)
            delta_xy = longitudinal * fwd + lateral * rgt
            pos[0:2] += delta_xy

            # z: if jump flag > 0 treat as jump height, else ground 0
            z = self.jump_height if jump > 0.5 else 0.0

            current_pos = np.array([pos[0], pos[1], z], dtype=float)
            poses.append({"position": current_pos.tolist(), "rotation": [yaw_deg, pitch_deg, 0.0]})
        return poses

def generate_token_coords_for_episode(
    episode_id: str,
    images: List[torch.Tensor],
    action_jsonl_path: str,
    depth_predictor: Callable[[torch.Tensor], torch.Tensor],
    frame_tokenizer,   # int or object with token count
    model, # for depth_estimation
    out_path: Optional[str] = None,
    fov_deg: float = 70.0,
    cache: bool = True,
    jump_height: float = 1.0,
) -> np.ndarray:
    """
    为一个 episode 生成 token_coords (T, num_tokens, 3)（世界坐标）
    说明：
      - images 必须是 List[torch.Tensor]，每项为 CxHxW 或 1xCxHxW 或 BxCxHxW（函数会视为单帧/单 batch）。
      - depth_predictor 接受 torch.Tensor（batch）并返回 torch.Tensor（B x 1 x H x W 或 B x H x W）。
      - 函数使用 depth 的实际 HxW 来计算 patch 大小，返回 coords shape (T, N, 3)。
    """
    # 若已缓存，直接加载
    if out_path:
        cache_path = str(Path(out_path).expanduser())
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass

    # 获取 actions
    actions = load_actions_from_jsonl(action_jsonl_path) if action_jsonl_path else []

    # 获取一帧尺寸与 token grid —— images 为 torch.Tensor
    first_img = images[0]
    if not isinstance(first_img, torch.Tensor):
        raise ValueError("generate_token_coords_for_episode expects images: List[torch.Tensor]")
    # first_img shape: C,H,W or 1,C,H,W or B,C,H,W
    if first_img.ndim == 4:
        _, C, H, W = first_img.shape
    elif first_img.ndim == 3:
        C, H, W = first_img.shape
    else:
        raise ValueError(f"Unsupported tensor image shape: {first_img.shape}")

    # 固定 token 网格为 14x24 (14 行, 24 列)，对应默认分辨率 224x384（patch 大小 16x16）。
    # 如果 depth 输出尺寸不是 224x384，函数会以 depth 的实际 H,W 重新计算 patch 大小，但网格保持 14x24。
    token_h, token_w = 14, 24
    fx, fy, cx, cy = estimate_intrinsics(W, H, fov_deg=fov_deg)

    # estimate poses from actions
    estimator = ActionPoseEstimator(jump_height=jump_height)
    poses = estimator.estimate_poses(actions, initial_pos=(0.0, 0.0, 0.0))
    # if fewer poses than images, repeat last pose
    if len(poses) < len(images):
        if len(poses) == 0:
            # default zero poses
            poses = [{"position":[0.0,0.0,0.0], "rotation":[0.0,0.0,0.0]} for _ in range(len(images))]
        else:
            last = poses[-1]
            poses = poses + [last] * (len(images) - len(poses))

    coords_per_frame = []
    patch_h = H / token_h
    patch_w = W / token_w

    for t_idx, img in enumerate(tqdm(images, desc=f"gen_token_coords {episode_id}", leave=False)):
        # img: torch.Tensor -> ensure batch dim
        inp = img
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)  # 1 x C x H x W
        # call depth_predictor with tensor batch
        with torch.no_grad():
            out = depth_predictor(inp, model=model)

            print(f"[DEBUG] depth_predictor output shape: {out.shape}, type: {type(out)}")

        # --- normalize depth to 2D numpy array safely ---
        if isinstance(out, torch.Tensor):
            d = out.detach().cpu().numpy()
        else:
            d = np.array(out)

        # remove batch/channel singletons robustly
        d = np.asarray(d)
        # possible shapes: (B,1,H,W), (B,H,W), (1,H,W), (H,W), (C,H,W)
        # squeeze batch/channel dims until 2D remains
        while d.ndim > 2 and (d.shape[0] == 1 or d.shape[0] in (1,3)):
            d = d.squeeze(0)
        if d.ndim != 2:
            # last resort: try taking first channel / first batch
            if d.ndim == 3:
                d = d[0]
            else:
                raise ValueError(f"Unexpected depth shape after squeeze: {d.shape}")

        depth = d.astype(np.float32)
        # --- end normalization ---

        # 若 depth 与当前 H,W 不同，使用 depth 的尺寸并重算 patch
        H_d, W_d = depth.shape
        if (W_d, H_d) != (W, H):
            W, H = W_d, H_d
            fx, fy, cx, cy = estimate_intrinsics(W, H, fov_deg=fov_deg)
            patch_h = H / token_h
            patch_w = W / token_w

        frame_coords = np.zeros((token_h * token_w, 3), dtype=np.float32)
        pose = poses[t_idx]
        pos_w = np.array(pose.get("position", [0.0,0.0,0.0]), dtype=float)
        yaw_deg, pitch_deg, roll_deg = pose.get("rotation", [0.0,0.0,0.0])

        # rotation: build R from yaw/pitch/roll (ZYX: yaw around Z)
        y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
        cy, sy = math.cos(y), math.sin(y)
        cp, sp = math.cos(p), math.sin(p)
        cr, sr = math.cos(r), math.sin(r)
        Rz = np.array([[cy, -sy, 0],[sy, cy, 0],[0,0,1]], dtype=float)
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=float)
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=float)
        R_world_cam = Rz @ Ry @ Rx  # transforms camera coords -> world coords: p_world = R * p_cam + t

        # 在索引 depth 前确保 px,py 为 int
        for r_idx in range(token_h * token_w):
            row = r_idx // token_w
            col = r_idx % token_w
            px_f = col * patch_w + patch_w * 0.5
            py_f = row * patch_h + patch_h * 0.5
            px = int(min(max(int(px_f), 0), W - 1))
            py = int(min(max(int(py_f), 0), H - 1))
            z = float(depth[py, px])
            Xc = (px - cx) / fx * z
            Yc = (py - cy) / fy * z
            Zc = z
            p_cam = np.array([Xc, Yc, Zc], dtype=float)
            p_world = R_world_cam @ p_cam + pos_w
            frame_coords[r_idx, :] = p_world
        coords_per_frame.append(frame_coords)

    coords = np.stack(coords_per_frame, axis=0)  # (T, num_tokens, 3)

    if out_path and cache:
        try:
            save_token_coords(out_path, coords)
        except Exception:
            pass
 
    return coords