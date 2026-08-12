import math
import numpy as np
from typing import Tuple, List, Optional

def token_index_to_rc(idx: int, W: int = 16, H: int = 16) -> Tuple[int, int]:
    r = idx // W
    c = idx % W
    return r, c

def rc_to_uv(r: int, c: int, W: int = 16, H: int = 16) -> Tuple[float, float]:
    u = (c + 0.5) / W
    v = (r + 0.5) / H
    return u, v

def yaw_pitch_to_dir(yaw: float, pitch: float) -> np.ndarray:
    """
    yaw: rotation around Y (left positive), pitch: rotation around X (up positive).
    Accepts yaw/pitch in radians; if values look like degrees (> 2*pi) it converts to radians.
    Returns forward unit vector in world space.
    """
    # auto-convert degrees -> radians if needed
    if abs(yaw) > 2 * math.pi or abs(pitch) > 2 * math.pi:
        yaw = math.radians(yaw)
        pitch = math.radians(pitch)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    # camera forward in camera coords = +Z.
    # apply pitch (around X) then yaw (around Y) to (0,0,1)
    x = math.sin(yaw) * math.cos(pitch)
    y = -math.sin(pitch)
    z = math.cos(yaw) * math.cos(pitch)
    v = np.array([x, y, z], dtype=np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n

class SpatialAngleSampler:
    def __init__(
        self,
        W: int = 24,
        H: int = 14,
        fov_deg: float = 90.0,
        lambda_dist: float = 1.0,
        dist_power: float = 1.0,
        dot_threshold: Optional[float] = None,
    ):
        self.W = W
        self.H = H
        self.fov = math.radians(fov_deg)
        if dot_threshold is None:
            self.dot_threshold = math.cos(self.fov / 2.0)
        else:
            self.dot_threshold = float(dot_threshold)
        self.lambda_dist = float(lambda_dist)
        # dist_power controls exponent on normalized distance: (dist / max_dist) ** dist_power
        # 0.0 -> ignore distance; 1.0 -> linear; >1 -> stronger penalty for far points
        self.dist_power = float(dist_power)

    def sample_tokens(
        self,
        frames_tokens: np.ndarray,        # [N, 336] int
        frames_coords: Optional[np.ndarray],  # [N, 336, 3] float32 or None (world coords)
        frames_camera: Optional[np.ndarray],  # [N, 2] yaw,pitch in radians or degrees, or None
        frames_camera_pos: Optional[np.ndarray] = None,  # [N,3] camera world positions (optional)
        target_frame_idx: int = 0,
        budget_total: int = 512,
        include_target_full: bool = True,
    ):
        """
        返回:
          selected_tokens: List[int]   -- token ids selected in order (target frame first if include_target_full)
          selected_meta: List[(frame_idx, row)]  -- coordinates of each selected token
        说明:
          - frames_tokens: numpy array shape [N, 336]
          - frames_coords: numpy array shape [N, 336, 3] with world/camera points; if None, fallback uses grid direction
          - frames_camera: array-like shape [N,2] with yaw,pitch (deg or rad). If None assume (0,0).
          - budget_total includes the target frame full tokens if include_target_full=True
        """
        N, L = frames_tokens.shape
        assert L == self.W * self.H, f"expected {self.W*self.H} image tokens per frame"

        # prepare camera forward vector for target frame
        if frames_camera is None:
            v_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            yaw, pitch = frames_camera[target_frame_idx]
            v_cam = yaw_pitch_to_dir(float(yaw), float(pitch))

        # include target frame all tokens first
        selected = []
        selected_meta: List[Tuple[int,int]] = []

        if include_target_full:
            for r in range(L):
                selected.append(int(frames_tokens[target_frame_idx, r]))
                selected_meta.append((target_frame_idx, r))
            remaining = max(0, budget_total - len(selected))
            if remaining == 0:
                return selected, selected_meta
        else:
            remaining = budget_total

        # compute max_dist for normalization (use camera-centric distances if possible)
        max_dist = 1.0
        if frames_coords is not None:
            try:
                if frames_camera_pos is not None:
                    # compute distances from each point to its frame camera position
                    # frames_coords: [N, L, 3], frames_camera_pos: [N,3]
                    diffs = frames_coords - frames_camera_pos[:, None, :]  # [N,L,3]
                    dists = np.linalg.norm(diffs.reshape(-1, 3), axis=1)
                else:
                    # fallback: use distance to origin (legacy behavior)
                    dists = np.linalg.norm(frames_coords.reshape(-1, 3), axis=1)
                max_dist = float(max(1e-6, np.max(dists)))
            except Exception:
                max_dist = 1.0

        # compute scores for all non-target tokens
        scores = []
        for f in range(N):
            if f == target_frame_idx:
                continue
            for r in range(L):
                tok = int(frames_tokens[f, r])
                if frames_coords is not None:
                    p = frames_coords[f, r].astype(np.float32)  # world coords
                    if frames_camera_pos is not None:
                        cam_pos = np.asarray(frames_camera_pos[f], dtype=np.float32)
                        vec = p - cam_pos
                    else:
                        # legacy: if no camera position given, assume p is camera-centric
                        vec = p
                    dist = float(np.linalg.norm(vec) + 1e-8)
                    v_token = vec / (dist + 1e-8)
                    dot = float(np.dot(v_token, v_cam))
                else:
                    # fallback: use grid center direction in camera coords: (x, y, 1)
                    rr = r // self.W
                    cc = r % self.W
                    dx = (cc + 0.5) / self.W - 0.5
                    dy = (rr + 0.5) / self.H - 0.5
                    v_token = np.array([dx, -dy, 1.0], dtype=np.float32)
                    v_token = v_token / (np.linalg.norm(v_token) + 1e-8)
                    dot = float(np.dot(v_token, v_cam))
                    dist = 1.0

                if dot < self.dot_threshold:
                    continue
                # normalized distance in [0,1], then apply exponent
                normed = (dist / (max_dist + 1e-8)) ** (self.dist_power if self.dist_power >= 0.0 else 1.0)
                score = dot / (1.0 + self.lambda_dist * normed)
                scores.append((score, f, r, tok))

        # sort by score desc and pick top remaining, avoiding duplicates
        scores.sort(key=lambda x: x[0], reverse=True)
        added = 0
        seen = set((target_frame_idx, r) for r in range(L)) if include_target_full else set()
        for sc, f, r, tok in scores:
            if added >= remaining:
                break
            if (f, r) in seen:
                continue
            selected.append(int(tok))
            selected_meta.append((f, r))
            seen.add((f, r))
            added += 1

        return selected, selected_meta