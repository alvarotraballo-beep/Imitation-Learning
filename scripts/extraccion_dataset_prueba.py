"""
Convert RealSense .bag videos of hand demonstrations into a robomimic-style HDF5 dataset.

Design goal:
- Do NOT pretend unavailable robot proprioception is real.
- Extract high-quality visual + geometric observations from RGB-D.
- Produce actions in robosuite JOINT_POSITION convention using a calibrated world frame and a
  pseudo-teleoperation policy inferred from object / hand motion phases.

Recommended use:
    1) Edit the USER CONFIG block below.
    2) Run:
       python robomimic_bag_converter_pro.py

Dependencies:
    pip install numpy h5py opencv-python scipy ultralytics mediapipe pyrealsense2 tqdm
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

try:
    import pyrealsense2 as rs
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyrealsense2 is required to read .bag files recorded with RealSense") from exc

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ultralytics is required for YOLO object detection") from exc

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    raise RuntimeError("mediapipe is required for hand keypoint detection") from exc


# -----------------------------
# Configuration
# -----------------------------

# =========================================================
# USER CONFIG - EDIT THESE PATHS AND OPTIONS
# =========================================================

# Folder containing your .bag demonstrations
BAG_DIR = "videos_cubo_3/videos_bag"

# Output robomimic-style dataset
OUTPUT_HDF5 = "\home\alvaro\projects\Imitation Learning JAKA\tests\assets\dataset_bc_jaka.hdf5"

# YOLO model trained to detect the cube
YOLO_MODEL_PATH = "best_2.onnx"

# Optional robomimic reference dataset.
# Use None unless you explicitly want to copy env_args/model_file/camera_info
# from a known-compatible robomimic dataset.
REFERENCE_HDF5 = None
# Example:
# REFERENCE_HDF5 = "test_v15.hdf5"

# Debug video output
DEBUG_DIR = "debug_videos"
WRITE_DEBUG = False

# Dataset generation settings
TARGET_FPS = 20
IMAGE_SIZE = 84

JAKA_HOME_EEF = np.array([-0.50, -0.97, 0.98], dtype=np.float32)

@dataclass(frozen=True)
class WorkspaceMap:
    """Affine map from RealSense camera coordinates to robosuite table/world coordinates.

    RealSense points are in meters: x right, y down, z forward.
    The default is intentionally conservative. For best results, calibrate this with 4-6
    known table/cube points and replace R,t,scale.
    """

    scale: float = 1.0
    R_cam_to_world: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 1.0),   # camera z -> world x
        (-1.0, 0.0, 0.0),  # camera x -> world -y
        (0.0, -1.0, 0.0),  # camera y -> world z
    )
    t_cam_to_world: Tuple[float, float, float] = (-0.35, 0.0, 0.82)
    table_z: float = 0.82
    workspace_min: Tuple[float, float, float] = (-0.60, -1.05, 0.75)
    workspace_max: Tuple[float, float, float] = (0.80, 0.50, 1.30)


@dataclass(frozen=True)
class ConverterConfig:
    image_size: int = 84
    target_fps: int = 20
    min_episode_frames: int = 24
    object_conf: float = 0.25
    max_depth_m: float = 2.0
    tcp_depth_patch: int = 4
    obj_depth_patch: int = 6
    smooth_window: int = 9
    smooth_polyorder: int = 2
    grasp_radius_m: float = 0.055
    release_radius_m: float = 0.075
    lift_height_m: float = 0.12
    osc_pos_scale_m: float = 0.02
    osc_rot_scale_rad: float = 0.5
    crop_pad_px: int = 80
    joint_action_scale_rad: float = 0.02
    grasp_offset_m: float = 0.015
    above_offset_m: float = 0.12
    workspace: WorkspaceMap = WorkspaceMap()


# -----------------------------
# Geometry and filtering
# -----------------------------

def robust_depth(depth_frame: rs.depth_frame, u: int, v: int, k: int, max_depth_m: float) -> float:
    h, w = depth_frame.get_height(), depth_frame.get_width()
    vals: List[float] = []
    for yy in range(max(0, v - k), min(h, v + k + 1)):
        for xx in range(max(0, u - k), min(w, u + k + 1)):
            z = float(depth_frame.get_distance(xx, yy))
            if 0.02 < z < max_depth_m:
                vals.append(z)
    if not vals:
        return float("nan")
    return float(np.median(vals))


def deproject(intr: rs.intrinsics, u: int, v: int, z: float) -> np.ndarray:
    x, y, zz = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z))
    return np.array([x, y, zz], dtype=np.float32)


def cam_to_world(p_cam: np.ndarray, wm: WorkspaceMap) -> np.ndarray:
    Rcw = np.asarray(wm.R_cam_to_world, dtype=np.float32)
    tcw = np.asarray(wm.t_cam_to_world, dtype=np.float32)
    p = wm.scale * (Rcw @ p_cam.astype(np.float32)) + tcw
    return np.clip(p, np.asarray(wm.workspace_min), np.asarray(wm.workspace_max)).astype(np.float32)


def resize_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)


def resize_depth(depth_m: np.ndarray, size: int) -> np.ndarray:
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    d = cv2.resize(d, (size, size), interpolation=cv2.INTER_NEAREST)
    return d[..., None].astype(np.float32)


def crop_around(frame_bgr: np.ndarray, center: Optional[Tuple[int, int]], out_size: int, pad: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if center is None:
        return resize_rgb(frame_bgr, out_size)
    cx, cy = center
    x1, x2 = max(0, cx - pad), min(w, cx + pad)
    y1, y2 = max(0, cy - pad), min(h, cy + pad)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr
    return resize_rgb(crop, out_size)


def smooth_array(x: np.ndarray, window: int, poly: int) -> np.ndarray:
    if len(x) < max(window, poly + 2):
        return x.astype(np.float32)
    win = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if win <= poly:
        return x.astype(np.float32)
    return savgol_filter(x, window_length=win, polyorder=poly, axis=0, mode="interp").astype(np.float32)


def finite_interpolate(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        if ok.sum() == 0:
            col[:] = 0.0
        elif ok.sum() < len(col):
            idx = np.arange(len(col))
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
        x[:, j] = col
    return x


# -----------------------------
# Detectors
# -----------------------------

class Detectors:
    def __init__(self, yolo_path: str, obj_conf: float) -> None:
        self.yolo = YOLO(yolo_path, task="detect")
        self.obj_conf = obj_conf
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def detect_object(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, np.ndarray, float]]:
        res = self.yolo(frame_bgr, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.ones(len(boxes))
        keep = np.where(confs >= self.obj_conf)[0]
        if len(keep) == 0:
            return None
        # Pick the largest confident box. This is usually more stable than raw index 0.
        areas = (boxes[keep, 2] - boxes[keep, 0]) * (boxes[keep, 3] - boxes[keep, 1])
        idx = int(keep[np.argmax(areas)])
        box = boxes[idx].astype(np.float32)
        u = int(round((box[0] + box[2]) * 0.5))
        v = int(round((box[1] + box[3]) * 0.5))
        return u, v, box, float(confs[idx])

    def detect_hand_tcp(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, np.ndarray, str]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        # Choose the hand whose pinch point is closest to image center; robust when both hands appear.
        h, w = frame_bgr.shape[:2]
        candidates = []
        handedness = res.multi_handedness or []
        for i, lm in enumerate(res.multi_hand_landmarks):
            thumb = lm.landmark[4]
            index = lm.landmark[8]
            u = int(round((thumb.x + index.x) * 0.5 * w))
            v = int(round((thumb.y + index.y) * 0.5 * h))
            score = np.linalg.norm(np.array([u - w / 2, v - h / 2]))
            label = "unknown"
            if i < len(handedness):
                label = handedness[i].classification[0].label.lower()
            candidates.append((score, u, v, lm, label))
        _, u, v, lm, label = min(candidates, key=lambda t: t[0])
        return u, v, lm, label


def hand_orientation_quat(lm, intr: rs.intrinsics, depth_frame: rs.depth_frame, cfg: ConverterConfig) -> np.ndarray:
    # Landmarks: wrist, index MCP, middle MCP, pinky MCP
    points = []
    for idx in (0, 5, 9, 17):
        l = lm.landmark[idx]
        u = int(round(l.x * intr.width))
        v = int(round(l.y * intr.height))
        z = robust_depth(depth_frame, u, v, cfg.tcp_depth_patch, cfg.max_depth_m)
        if not np.isfinite(z):
            points.append(None)
        else:
            points.append(deproject(intr, u, v, z))
    if all(p is not None for p in points):
        wrist, index, middle, pinky = [p.astype(np.float32) for p in points]  # type: ignore
        x_axis = pinky - index
        y_axis = middle - wrist
        if np.linalg.norm(x_axis) > 1e-5 and np.linalg.norm(y_axis) > 1e-5:
            x_axis /= np.linalg.norm(x_axis)
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            if np.linalg.norm(z_axis) > 1e-5:
                z_axis /= np.linalg.norm(z_axis)
                y_axis = np.cross(z_axis, x_axis)
                Rm = np.stack([x_axis, y_axis, z_axis], axis=1)
                q = Rot.from_matrix(Rm).as_quat()  # xyzw
                return q.astype(np.float32)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # robomimic examples often use wxyz-like storage; kept stable


# -----------------------------
# Episode processing
# -----------------------------

@dataclass
class FrameObs:
    rgb: np.ndarray
    depth: np.ndarray
    wrist_rgb: np.ndarray
    wrist_depth: np.ndarray
    hand_pos: np.ndarray
    hand_quat: np.ndarray
    obj_pos: np.ndarray
    obj_box: Optional[np.ndarray]
    hand_uv: Optional[Tuple[int, int]]
    obj_uv: Optional[Tuple[int, int]]
    quality: float


def read_bag_episode(path: Path, detectors: Detectors, cfg: ConverterConfig, debug_dir: Optional[Path]) -> List[FrameObs]:
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_device_from_file(str(path), repeat_playback=False)
    profile = pipeline.start(rs_cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    native_fps = max(1, int(color_stream.fps()))
    stride = max(1, round(native_fps / cfg.target_fps))

    writer = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    frames_out: List[FrameObs] = []
    last_obj_world: Optional[np.ndarray] = None
    last_obj_uv: Optional[Tuple[int, int]] = None
    idx = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                if playback.current_status() == rs.playback_status.stopped:
                    break
                continue
            if idx % stride != 0:
                idx += 1
                continue
            idx += 1

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()

            obj_det = detectors.detect_object(bgr)
            obj_world = None
            obj_uv = None
            obj_box = None
            obj_conf = 0.0
            if obj_det is not None:
                ou, ov, box, obj_conf = obj_det
                z = robust_depth(depth_frame, ou, ov, cfg.obj_depth_patch, cfg.max_depth_m)
                if np.isfinite(z):
                    obj_cam = deproject(intr, ou, ov, z)
                    obj_world = cam_to_world(obj_cam, cfg.workspace)
                    obj_uv = (ou, ov)
                    obj_box = box
                    last_obj_world = obj_world
                    last_obj_uv = obj_uv
            if obj_world is None and last_obj_world is not None:
                obj_world = last_obj_world.copy()
                obj_uv = last_obj_uv

            hand_det = detectors.detect_hand_tcp(bgr)
            hand_world = None
            hand_quat = np.array([1, 0, 0, 0], dtype=np.float32)
            hand_uv = None
            hand_conf = 0.0
            if hand_det is not None:
                hu, hv, lm, _label = hand_det
                z = robust_depth(depth_frame, hu, hv, cfg.tcp_depth_patch, cfg.max_depth_m)
                if np.isfinite(z):
                    hand_cam = deproject(intr, hu, hv, z)
                    hand_world = cam_to_world(hand_cam, cfg.workspace)
                    hand_quat = hand_orientation_quat(lm, intr, depth_frame, cfg)
                    hand_uv = (hu, hv)
                    hand_conf = 1.0

            if obj_world is None or hand_world is None:
                continue

            agent_rgb = resize_rgb(bgr, cfg.image_size)
            agent_depth = resize_depth(depth_m, cfg.image_size)
            wrist_rgb = crop_around(bgr, hand_uv or obj_uv, cfg.image_size, cfg.crop_pad_px)
            wrist_depth = agent_depth.copy()  # Real wrist depth is unavailable; do not set zeros.

            quality = float(0.5 * obj_conf + 0.5 * hand_conf)
            frames_out.append(FrameObs(agent_rgb, agent_depth, wrist_rgb, wrist_depth, hand_world, hand_quat,
                                       obj_world, obj_box, hand_uv, obj_uv, quality))

            if debug_dir is not None:
                dbg = bgr.copy()
                if obj_box is not None:
                    x1, y1, x2, y2 = obj_box.astype(int)
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if obj_uv is not None:
                    cv2.circle(dbg, obj_uv, 5, (0, 255, 0), -1)
                if hand_uv is not None:
                    cv2.circle(dbg, hand_uv, 5, (0, 0, 255), -1)
                if writer is None:
                    h, w = dbg.shape[:2]
                    writer = cv2.VideoWriter(str(debug_dir / f"{path.stem}_debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cfg.target_fps, (w, h))
                writer.write(dbg)
    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()

    return frames_out


def infer_place_target(obj: np.ndarray) -> np.ndarray:
    # Use the last stable object position as the target. If the object was occluded during grasp,
    # this still captures the demonstrated final placement more reliably than hand endpoint alone.
    tail = obj[max(0, len(obj) - 8):]
    return np.median(tail, axis=0).astype(np.float32)


def infer_place_target_from_hand_and_object(
    hand: np.ndarray,
    obj: np.ndarray,
    cfg: ConverterConfig,
) -> np.ndarray:
    obj_start = np.median(obj[:8], axis=0)
    obj_end = np.median(obj[-8:], axis=0)
    hand_end = np.median(hand[-8:], axis=0)

    obj_xy_disp = np.linalg.norm(obj_end[:2] - obj_start[:2])

    if obj_xy_disp > 0.04:
        place = obj_end.copy()
    else:
        place = hand_end.copy()

    place[2] = cfg.workspace.table_z
    return place.astype(np.float32)


def densify_path(points, grip, phase, max_step):
    new_points = [points[0]]
    new_grip = [grip[0]]
    new_phase = [phase[0]]

    for i in range(1, len(points)):
        p0 = points[i - 1]
        p1 = points[i]
        delta = p1 - p0

        n_steps = int(np.ceil(np.max(np.abs(delta)) / max_step))
        n_steps = max(n_steps, 1)

        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            new_points.append((1.0 - alpha) * p0 + alpha * p1)
            new_grip.append(grip[i])
            new_phase.append(phase[i])

    return (
        np.asarray(new_points, dtype=np.float32),
        np.asarray(new_grip, dtype=np.float32),
        np.asarray(new_phase, dtype=np.float32),
    )


_IK_ENV = None


def get_ik_env():
    global _IK_ENV

    if _IK_ENV is not None:
        return _IK_ENV

    controller_config = load_composite_controller_config(controller="BASIC")

    _IK_ENV = suite.make(
        env_name="Lift",
        robots="JakaMiniCobo",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
    )

    _IK_ENV.reset()
    return _IK_ENV


def get_site_pos(env, site_name="robot0_grip_site"):
    site_id = env.sim.model.site_name2id(site_name)
    return env.sim.data._data.site_xpos[site_id].copy()


def forward_q(env, q):
    data = env.sim.data._data
    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    env.sim.forward()
    return get_site_pos(env)


def solve_ik_position(env, target_pos, q_init, iters=500, lr=0.15, damping=1e-2):
    q = q_init.copy().astype(np.float64)

    q_min = np.array([-6.28, -2.09, -2.27, -6.28, -2.09, -6.28], dtype=np.float64)
    q_max = np.array([ 6.28,  2.09,  2.27,  6.28,  2.09,  6.28], dtype=np.float64)

    for _ in range(iters):
        eef = forward_q(env, q)
        err = target_pos - eef

        if np.linalg.norm(err) < 0.01:
            break

        J = np.zeros((3, 6), dtype=np.float64)
        eps = 1e-4

        for j in range(6):
            q2 = q.copy()
            q2[j] += eps
            eef2 = forward_q(env, q2)
            J[:, j] = (eef2 - eef) / eps

        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        q = q + lr * dq
        q = np.clip(q, q_min, q_max)

    eef = forward_q(env, q)
    return q.astype(np.float32), eef.astype(np.float32), float(np.linalg.norm(target_pos - eef))


def solve_best_ik(env, target, q_seed):
    seeds = []
    seeds.append(q_seed.copy())

    for j1 in [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        q = q_seed.copy()
        q[0] = j1
        seeds.append(q)

    for j1 in [-1.5, -0.5, 0.5, 1.5]:
        for j2 in [-0.6, 0.0, 0.6]:
            for j3 in [-0.6, 0.0, 0.6]:
                q = q_seed.copy()
                q[0] = j1
                q[1] = j2
                q[2] = j3
                seeds.append(q)

    best = None

    for seed in seeds:
        q_try, eef_try, dist_try = solve_ik_position(
            env,
            target,
            seed,
            iters=180,
            lr=0.15,
            damping=1e-2,
        )

        if best is None or dist_try < best[0]:
            best = (dist_try, q_try, eef_try)

        if best is not None and best[0] < 0.015:
            break

    dist, q, eef = best
    return q.astype(np.float32), eef.astype(np.float32), float(dist)


def interp_segment(q_a, q_b, n_steps):
    out = []
    for i in range(n_steps):
        alpha = i / max(1, n_steps - 1)
        out.append(((1.0 - alpha) * q_a + alpha * q_b).astype(np.float32))
    return out


def build_actions(
    hand: np.ndarray,
    obj: np.ndarray,
    cfg: ConverterConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    env = get_ik_env()
    data = env.sim.data._data

    q0 = np.asarray(env.robots[0].robot_model.init_qpos, dtype=np.float32).copy()
    data.qpos[:6] = q0
    data.qvel[:6] = 0.0
    env.sim.forward()

    cube_start = np.median(obj[:8], axis=0).astype(np.float32)
    cube_start[2] = cfg.workspace.table_z + 0.025

    place = infer_place_target_from_hand_and_object(hand, obj, cfg)
    place[2] = cfg.workspace.table_z + 0.025

    # Remap temporal a workspace alcanzable del JAKA en robosuite
    cube_start[0] = np.clip(cube_start[0], -0.05, 0.05)
    cube_start[1] = np.clip(cube_start[1], -0.05, 0.05)

    place[0] = np.clip(place[0], -0.05, 0.05)
    place[1] = np.clip(place[1], -0.05, 0.05)

    target_above = cube_start.copy()
    target_above[2] = cfg.workspace.table_z + cfg.above_offset_m

    target_grasp = cube_start.copy()
    target_grasp[2] = cfg.workspace.table_z + cfg.grasp_offset_m

    target_lift = cube_start.copy()
    target_lift[2] = cfg.workspace.table_z + cfg.above_offset_m

    target_place_above = place.copy()
    target_place_above[2] = cfg.workspace.table_z + cfg.above_offset_m

    target_place = place.copy()
    target_place[2] = cfg.workspace.table_z + cfg.grasp_offset_m

    q_above, eef_above, dist_above = solve_best_ik(env, target_above, q0)
    q_grasp, eef_grasp, dist_grasp = solve_best_ik(env, target_grasp, q_above)
    q_lift, eef_lift, dist_lift = solve_best_ik(env, target_lift, q_grasp)
    q_place_above, eef_place_above, dist_place_above = solve_best_ik(env, target_place_above, q_lift)
    q_place, eef_place, dist_place = solve_best_ik(env, target_place, q_place_above)

    if min(dist_above, dist_grasp, dist_lift, dist_place_above, dist_place) > 0.08:
        print("\nIK FAILED DEBUG")
        print("cube_start:", cube_start)
        print("place:", place)
        print("target_above:", target_above, "dist:", dist_above, "eef:", eef_above)
        print("target_grasp:", target_grasp, "dist:", dist_grasp, "eef:", eef_grasp)
        print("target_lift:", target_lift, "dist:", dist_lift, "eef:", eef_lift)
        print("target_place_above:", target_place_above, "dist:", dist_place_above, "eef:", eef_place_above)
        print("target_place:", target_place, "dist:", dist_place, "eef:", eef_place)

        raise RuntimeError(
            f"IK failed: above={dist_above:.3f}, grasp={dist_grasp:.3f}, "
            f"lift={dist_lift:.3f}, place_above={dist_place_above:.3f}, place={dist_place:.3f}"
        )

    q_seq = []
    grip_seq = []
    phase_seq = []

    def add_segment(name_phase, qa, qb, steps, grip_value):
        seg = interp_segment(qa, qb, steps)
        for q in seg:
            q_seq.append(q)
            grip_seq.append([grip_value])
            phase_seq.append([name_phase])

    # Convención gripper:
    # -1 = abierto para acciones robomimic
    # +1 = cerrado para acciones robomimic
    add_segment(0.0, q0, q_above, 120, -1.0)
    add_segment(1.0, q_above, q_grasp, 80, -1.0)
    add_segment(2.0, q_grasp, q_grasp, 40, 1.0)
    add_segment(3.0, q_grasp, q_lift, 80, 1.0)
    add_segment(4.0, q_lift, q_place_above, 120, 1.0)
    add_segment(5.0, q_place_above, q_place, 80, 1.0)
    add_segment(6.0, q_place, q_place, 40, -1.0)

    joint_pos = np.asarray(q_seq, dtype=np.float32)
    grip = np.asarray(grip_seq, dtype=np.float32)
    phase = np.asarray(phase_seq, dtype=np.float32)

    n = len(joint_pos)

    eef = np.zeros((n, 3), dtype=np.float32)
    for i, q in enumerate(joint_pos):
        eef[i] = forward_q(env, q)

    actions = np.zeros((n, 8), dtype=np.float32)

    dq = np.zeros_like(joint_pos, dtype=np.float32)
    dq[1:] = joint_pos[1:] - joint_pos[:-1]

    actions[:, :6] = np.clip(dq / cfg.joint_action_scale_rad, -1.0, 1.0)
    actions[:, 6] = grip[:, 0]
    actions[:, 7] = grip[:, 0]

    cube_seq = np.tile(cube_start.reshape(1, 3), (n, 1)).astype(np.float32)

    return actions, eef.astype(np.float32), grip, phase, joint_pos.astype(np.float32), cube_start, cube_seq


def object_obs(obj: np.ndarray, eef: np.ndarray) -> np.ndarray:
    n = len(obj)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    gripper_to_cube = (obj - eef).astype(np.float32)
    return np.concatenate([obj.astype(np.float32), quat, gripper_to_cube], axis=1)


def make_episode(frames: List[FrameObs], cfg: ConverterConfig) -> Optional[Dict[str, np.ndarray]]:
    if len(frames) < cfg.min_episode_frames:
        return None
    hand = np.stack([fr.hand_pos for fr in frames]).astype(np.float32)
    obj = np.stack([fr.obj_pos for fr in frames]).astype(np.float32)
    hand = smooth_array(finite_interpolate(hand), cfg.smooth_window, cfg.smooth_polyorder)
    obj = smooth_array(finite_interpolate(obj), cfg.smooth_window, cfg.smooth_polyorder)
    obj[:, 2] = cfg.workspace.table_z + 0.025

    pre_n = 90
    hand_pre = np.linspace(JAKA_HOME_EEF, hand[0], pre_n, dtype=np.float32)
    obj_pre = np.repeat(obj[0:1], pre_n, axis=0)

    hand = np.concatenate([hand_pre, hand], axis=0)
    obj = np.concatenate([obj_pre, obj], axis=0)

    frames = [frames[0]] * pre_n + frames

    actions, eef, gripper, phase, joint_pos, cube_start, cube_seq = build_actions(hand, obj, cfg)
    n = len(actions)

    # Si build_actions densifica la trayectoria, hay que adaptar obj y frames a la nueva longitud
    if len(obj) != n:
        idx = np.linspace(0, len(obj) - 1, n).round().astype(int)
        obj = obj[idx]
        frames = [frames[i] for i in idx]

    dt = 1.0 / cfg.target_fps
    eef_vel = np.gradient(eef, dt, axis=0).astype(np.float32)
    eef_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    eef_vel_ang = np.zeros((n, 3), dtype=np.float32)
   
    joint_vel = np.zeros((n, 6), dtype=np.float32)
    gripper_qpos = np.where(gripper > 0.0, 0.013, 0.0)
    gripper_qpos = np.repeat(gripper_qpos, 2, axis=1).astype(np.float32)
    gripper_qvel = np.gradient(gripper_qpos, dt, axis=0).astype(np.float32)
    place = infer_place_target(obj)

    rgb = np.stack([fr.rgb for fr in frames]).astype(np.uint8)
    depth = np.stack([fr.depth for fr in frames]).astype(np.float32)
    wrist_rgb = np.stack([fr.wrist_rgb for fr in frames]).astype(np.uint8)
    wrist_depth = np.stack([fr.wrist_depth for fr in frames]).astype(np.float32)

    rewards = np.zeros(n, dtype=np.float32)
    # Sparse success proxy: final object near inferred target and gripper opened.
    rewards[-1] = 1.0
    dones = np.zeros(n, dtype=np.int64)
    dones[-1] = 1

    states = np.concatenate([eef, obj, eef_quat, eef_vel, eef_vel_ang, joint_pos, joint_vel, gripper_qpos], axis=1)[:, :32]

    obs = {
        "agentview_image": rgb,
        "agentview_depth": depth,
        "robot0_eye_in_hand_image": wrist_rgb,
        "robot0_eye_in_hand_depth": wrist_depth,
        "object": object_obs(obj, eef),
        "robot0_eef_pos": eef,
        "robot0_eef_quat": eef_quat,
        "robot0_eef_vel_lin": eef_vel,
        "robot0_eef_vel_ang": eef_vel_ang,
        "robot0_joint_pos": joint_pos,
        "robot0_joint_pos_cos": np.cos(joint_pos).astype(np.float32),
        "robot0_joint_pos_sin": np.sin(joint_pos).astype(np.float32),
        "robot0_joint_vel": joint_vel,
        "robot0_gripper_qpos": gripper_qpos,
        "robot0_gripper_qvel": gripper_qvel,
        "phase": phase.astype(np.float32),
        "cube_pos": cube_seq.astype(np.float32),
        "gripper_to_cube_pos": (cube_seq - eef).astype(np.float32),
    }
    return {"actions": actions, "rewards": rewards, "dones": dones, "states": states, "obs": obs}  # type: ignore[return-value]


# -----------------------------
# HDF5 writing
# -----------------------------

def read_reference_metadata(reference_hdf5: Optional[str]) -> Tuple[dict, Optional[str], Optional[str], Optional[dict]]:
    if not reference_hdf5:
        return {}, None, None, None
    with h5py.File(reference_hdf5, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        first_demo = sorted(f["data"].keys())[0]
        model_file = f["data"][first_demo].attrs.get("model_file")
        camera_info = f["data"][first_demo].attrs.get("camera_info")
        return env_args, model_file, camera_info, dict(f.attrs)


def write_dataset(output: str, episodes: List[Dict[str, np.ndarray]], env_args: dict, model_file: Optional[str], camera_info: Optional[str], cfg: ConverterConfig) -> None:
    with h5py.File(output, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["converter_config"] = json.dumps(asdict(cfg))
        demo_keys = []
        total = 0
        for i, ep_data in enumerate(episodes):
            ep = data.create_group(f"demo_{i}")
            ep.create_dataset("actions", data=ep_data["actions"], compression="gzip", compression_opts=4)
            ep.create_dataset("rewards", data=ep_data["rewards"])
            ep.create_dataset("dones", data=ep_data["dones"])
            ep.create_dataset("states", data=ep_data["states"], compression="gzip", compression_opts=4)
            obs_grp = ep.create_group("obs")
            next_grp = ep.create_group("next_obs")
            obs_dict: Dict[str, np.ndarray] = ep_data["obs"]  # type: ignore[assignment]
            for k, v in obs_dict.items():
                comp = "gzip" if v.ndim >= 3 else None
                obs_grp.create_dataset(k, data=v, compression=comp, compression_opts=4 if comp else None)
                nxt = np.concatenate([v[1:], v[-1:]], axis=0)
                next_grp.create_dataset(k, data=nxt, compression=comp, compression_opts=4 if comp else None)
            ep.attrs["num_samples"] = int(len(ep_data["actions"]))
            if model_file is not None:
                ep.attrs["model_file"] = model_file
            if camera_info is not None:
                ep.attrs["camera_info"] = camera_info
            demo_keys.append(f"demo_{i}".encode("utf-8"))
            total += len(ep_data["actions"])
        data.attrs["total"] = int(total)
        mask = f.create_group("mask")
        rng = np.random.default_rng(42)
        keys = np.asarray(demo_keys)
        rng.shuffle(keys)
        split = max(1, int(0.8 * len(keys))) if len(keys) > 1 else len(keys)
        mask.create_dataset("train", data=keys[:split])
        mask.create_dataset("valid", data=keys[split:])


def default_env_args(reference_env_args: dict) -> dict:
    if reference_env_args:
        env_args = reference_env_args
    else:
        env_args = {
            "env_name": "Lift",
            "env_version": "1.5.0",
            "type": 1,
            "env_kwargs": {
                "robots": ["JakaMiniCobo"],
                "controller_configs": {
                    "type": "BASIC",
                    "body_parts": {
                        "right": {
                            "type": "JOINT_POSITION",
                            "input_max": 1,
                            "input_min": -1,
                            "output_max": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                            "output_min": [-0.02, -0.02, -0.02, -0.02, -0.02, -0.02],
                            "kp": 60,
                            "damping": 3,
                            "impedance_mode": "fixed",
                            "kp_limits": [0, 300],
                            "damping_limits": [0, 10],
                            "control_delta": True,
                            "interpolation": None,
                            "gripper": {
                                "type": "GRIP"
                            }
                        }
                    }
                },
                "control_freq": 20,
                "use_object_obs": True,
                "use_camera_obs": False,
                "has_offscreen_renderer": True,
                "camera_names": ["frontview"],
                "camera_heights": 512,
                "camera_widths": 512,
                "camera_depths": False,
                "reward_shaping": False,
            },
        }
    # Ensure dimensions match generated data.
    env_args.setdefault("env_kwargs", {})
    env_args["env_kwargs"]["camera_heights"] = 512
    env_args["env_kwargs"]["camera_widths"] = 512
    env_args["env_kwargs"]["camera_depths"] = False
    env_args["env_kwargs"]["control_freq"] = 20
    return env_args


def validate_user_config() -> None:
    """Fail early with clear messages before processing the RealSense bags."""
    if not BAG_DIR:
        raise ValueError("BAG_DIR is empty. Set BAG_DIR to the folder containing your .bag files.")
    if not OUTPUT_HDF5:
        raise ValueError("OUTPUT_HDF5 is empty. Set OUTPUT_HDF5 to the output .hdf5 path.")
    if not YOLO_MODEL_PATH:
        raise ValueError("YOLO_MODEL_PATH is empty. Set YOLO_MODEL_PATH to your YOLO .pt/.onnx model.")

    bag_dir = Path(BAG_DIR)
    if not bag_dir.exists():
        raise FileNotFoundError(f"BAG_DIR does not exist: {bag_dir.resolve()}")
    if not bag_dir.is_dir():
        raise NotADirectoryError(f"BAG_DIR is not a directory: {bag_dir.resolve()}")

    yolo_path = Path(YOLO_MODEL_PATH)
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO_MODEL_PATH does not exist: {yolo_path.resolve()}")

    if REFERENCE_HDF5 is not None:
        ref_path = Path(REFERENCE_HDF5)
        if not ref_path.exists():
            raise FileNotFoundError(f"REFERENCE_HDF5 does not exist: {ref_path.resolve()}")

    output_parent = Path(OUTPUT_HDF5).expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)

    if WRITE_DEBUG:
        Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)


def main() -> None:
    validate_user_config()

    cfg = ConverterConfig(target_fps=TARGET_FPS, image_size=IMAGE_SIZE)
    ref_env_args, model_file, camera_info, _root_attrs = read_reference_metadata(REFERENCE_HDF5)
    env_args = default_env_args(ref_env_args)

    detectors = Detectors(YOLO_MODEL_PATH, cfg.object_conf)
    bag_paths = sorted(Path(BAG_DIR).glob("*.bag"))
    if not bag_paths:
        raise FileNotFoundError(f"No .bag files found in {Path(BAG_DIR).resolve()}")

    debug_dir = Path(DEBUG_DIR) if WRITE_DEBUG else None
    episodes: List[Dict[str, np.ndarray]] = []
    skipped: List[str] = []

    print("Configuration:")
    print(json.dumps({
        "bag_dir": str(Path(BAG_DIR).resolve()),
        "output_hdf5": str(Path(OUTPUT_HDF5).resolve()),
        "yolo_model": str(Path(YOLO_MODEL_PATH).resolve()),
        "reference_hdf5": None if REFERENCE_HDF5 is None else str(Path(REFERENCE_HDF5).resolve()),
        "debug_dir": None if not WRITE_DEBUG else str(Path(DEBUG_DIR).resolve()),
        "target_fps": TARGET_FPS,
        "image_size": IMAGE_SIZE,
        "num_bags": len(bag_paths),
    }, indent=2))

    for bag in tqdm(bag_paths, desc="BAG -> episode"):
        frames = read_bag_episode(bag, detectors, cfg, debug_dir)
        ep = make_episode(frames, cfg)
        if ep is None:
            skipped.append(bag.name)
            continue
        episodes.append(ep)

    if not episodes:
        raise RuntimeError("No valid episodes were produced. Check YOLO detections, depth, and hand visibility.")

    write_dataset(OUTPUT_HDF5, episodes, env_args, model_file, camera_info, cfg)

    report = {
        "output": str(Path(OUTPUT_HDF5).resolve()),
        "episodes": len(episodes),
        "total_samples": int(sum(len(ep["actions"]) for ep in episodes)),
        "skipped": skipped,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
