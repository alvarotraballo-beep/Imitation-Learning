#!/usr/bin/env python3
"""Generate direct-control JAKA demos from RealSense .bag human demonstrations."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from extraccion_dataset_prueba import (
    ConverterConfig,
    Detectors,
    JAKA_HOME_EEF,
    finite_interpolate,
    infer_place_target_from_hand_and_object,
    read_bag_episode,
    smooth_array,
)
from generate_scripted_lift_dataset import collect_obs, make_env, object_obs
from jaka_mount_utils import apply_base_longitudinal_twist, reject_global_yaw
from scripted_joint_ik_lift_smoke import (
    boost_arm_actuators,
    boost_gripper_grasp,
    build_cartesian_q_path,
    cube_geom_pos,
    disable_arm_collisions,
    disable_gripper_nonpad_collisions,
    get_pad_mid_pos,
    set_ctrl,
    set_cube_appearance,
    set_cube_pos,
)


def unit_xy(vec, fallback):
    vec = np.asarray(vec, dtype=np.float64)
    if vec.shape[0] >= 3:
        vec = vec[:2]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        vec = np.asarray(fallback, dtype=np.float64)[:2]
        norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([-0.7, -0.7], dtype=np.float64) / np.sqrt(0.98)
    return vec / norm


def min_jerk(alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    return alpha * alpha * alpha * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha)


def smooth_waypoint_segments(waypoints, samples_per_segment):
    """Add eased intermediate targets so the IK path inherits smoother human-like timing."""
    if samples_per_segment <= 0 or len(waypoints) < 2:
        return waypoints
    smoothed = [waypoints[0]]
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        start_label, start_pos, _start_close, _start_steps = start
        end_label, end_pos, end_close, end_steps = end
        for i, alpha in enumerate(min_jerk(np.linspace(0.0, 1.0, samples_per_segment + 2)[1:-1]), start=1):
            label = f"{start_label}_to_{end_label}"
            pos = (1.0 - alpha) * start_pos + alpha * end_pos
            smoothed.append((label, pos.astype(np.float64), end_close, max(4, end_steps // (samples_per_segment + 1))))
        smoothed.append((end_label, end_pos, end_close, end_steps))
    return smoothed


def parse_float_list(text):
    if isinstance(text, (list, tuple)):
        return [float(v) for v in text]
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def parse_color_list(text):
    colors = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        vals = [float(v.strip()) for v in item.split(",")]
        if len(vals) == 3:
            vals.append(1.0)
        if len(vals) != 4:
            raise argparse.ArgumentTypeError("Colors must be r,g,b or r,g,b,a entries separated by semicolons")
        colors.append(np.asarray(vals, dtype=np.float64))
    if not colors:
        colors.append(np.array([0.8, 0.05, 0.05, 1.0], dtype=np.float64))
    return colors


def make_variant(args, bag_index, variant_index):
    global_index = bag_index * max(1, args.cube_variants) + variant_index
    rng = np.random.default_rng(args.seed + 1009 * bag_index + 9176 * variant_index)
    if args.cube_pos_jitter > 0.0:
        if args.variant_sampling == "halton":
            xy_offset = (2.0 * np.array([halton(global_index + 1, 2), halton(global_index + 1, 3)]) - 1.0)
            xy_offset *= args.cube_pos_jitter
        else:
            xy_offset = rng.uniform(-args.cube_pos_jitter, args.cube_pos_jitter, size=2)
    else:
        xy_offset = np.zeros(2, dtype=np.float64)
    sizes = parse_float_list(args.cube_size_values)
    colors = parse_color_list(args.cube_colors)
    size = float(sizes[(bag_index + variant_index) % len(sizes)])
    color = colors[(bag_index * max(1, args.cube_variants) + variant_index) % len(colors)]
    return {
        "variant_index": int(variant_index),
        "global_variant_index": int(global_index),
        "xy_offset": xy_offset.astype(np.float64),
        "cube_half_size": size,
        "cube_rgba": color.astype(np.float64),
    }


def halton(index, base):
    value = 0.0
    fraction = 1.0 / float(base)
    while index > 0:
        value += fraction * (index % base)
        index //= base
        fraction /= float(base)
    return value


def make_waypoints(
    cube,
    lift_height,
    lateral_grasp_z,
    style,
    approach_xy=None,
    smooth_segments=0,
    grasp_cube_z_offset=0.0,
    side_grasp_xy_offset=0.012,
):
    table_z = 0.820
    approach_xy = unit_xy(approach_xy if approach_xy is not None else [-0.7, -0.7], [-0.7, -0.7])
    tangent_xy = np.array([-approach_xy[1], approach_xy[0]], dtype=np.float64)
    if style == "scripted_grasp":
        return [
            ("above", cube + np.array([0.0, 0.0, 0.18]), False, 90),
            ("pregrasp", cube + np.array([0.0, 0.0, 0.055]), False, 70),
            ("grasp", cube + np.array([0.0, 0.0, -0.025]), False, 50),
            ("close", cube + np.array([0.0, 0.0, -0.025]), True, 80),
            ("lift", cube + np.array([0.0, 0.0, 0.26]), True, 140),
        ]
    if style == "bag_center_smooth":
        waypoints = [
            ("above", cube + np.array([0.0, 0.0, 0.18], dtype=np.float64), False, 90),
            ("soft_descent", cube + np.array([0.0, 0.0, 0.075], dtype=np.float64), False, 75),
            ("pregrasp", cube + np.array([0.0, 0.0, 0.035], dtype=np.float64), False, 60),
            ("grasp", cube + np.array([0.0, 0.0, grasp_cube_z_offset], dtype=np.float64), False, 60),
            ("close", cube + np.array([0.0, 0.0, grasp_cube_z_offset], dtype=np.float64), True, 105),
            ("lift_settle", cube + np.array([0.0, 0.0, lift_height * 0.45], dtype=np.float64), True, 90),
            ("lift", cube + np.array([0.0, 0.0, lift_height], dtype=np.float64), True, 150),
        ]
        return smooth_waypoint_segments(waypoints, smooth_segments)
    if style == "bag_side_grasp":
        side_z = max(table_z + lateral_grasp_z, cube[2] + grasp_cube_z_offset)
        side_offset = np.r_[approach_xy * side_grasp_xy_offset, 0.0]
        waypoints = [
            ("home_arc", cube + np.r_[approach_xy * 0.22 + tangent_xy * 0.035, 0.30], False, 120),
            ("shoulder_arc", cube + np.r_[approach_xy * 0.14 + tangent_xy * 0.020, 0.22], False, 90),
            ("side_pregrasp", cube + np.r_[approach_xy * 0.075, 0.115], False, 70),
            ("side_grasp", np.array([cube[0], cube[1], side_z], dtype=np.float64) + side_offset, False, 70),
            ("close", np.array([cube[0], cube[1], side_z], dtype=np.float64) + side_offset, True, 105),
            ("lift_settle", cube + np.r_[approach_xy * 0.010, lift_height * 0.45], True, 90),
            ("lift", cube + np.array([0.0, 0.0, lift_height], dtype=np.float64), True, 150),
        ]
        return smooth_waypoint_segments(waypoints, smooth_segments)
    if style == "bag_top_grasp":
        grasp_z = max(table_z + lateral_grasp_z, cube[2] + grasp_cube_z_offset)
        waypoints = [
            ("home_arc", cube + np.r_[approach_xy * 0.18 + tangent_xy * 0.030, 0.30], False, 120),
            ("over_cube", cube + np.array([0.0, 0.0, 0.22], dtype=np.float64), False, 95),
            ("top_pregrasp", cube + np.array([0.0, 0.0, 0.105], dtype=np.float64), False, 75),
            ("top_grasp", np.array([cube[0], cube[1], grasp_z], dtype=np.float64), False, 65),
            ("close", np.array([cube[0], cube[1], grasp_z], dtype=np.float64), True, 105),
            ("lift", cube + np.array([0.0, 0.0, lift_height], dtype=np.float64), True, 155),
        ]
        return smooth_waypoint_segments(waypoints, smooth_segments)
    lateral_z = max(table_z + lateral_grasp_z, cube[2] + 0.018)
    return [
        ("home_arc", cube + np.array([-0.18, -0.24, 0.28]), False, 120),
        ("above", cube + np.array([-0.06, -0.08, 0.20]), False, 100),
        ("pregrasp", cube + np.array([-0.018, -0.024, 0.070]), False, 80),
        ("grasp", np.array([cube[0], cube[1], lateral_z], dtype=np.float64), False, 60),
        ("close", np.array([cube[0], cube[1], lateral_z], dtype=np.float64), True, 90),
        ("lift", cube + np.array([0.0, 0.0, lift_height]), True, 150),
    ]


def make_episode_from_bag(bag, detectors, args, bag_index=0, variant=None):
    cfg = ConverterConfig(
        target_fps=args.target_fps,
        image_size=84,
        object_conf=args.object_conf,
        grasp_offset_m=args.lateral_grasp_z,
        above_offset_m=args.lift_height,
    )
    frames = read_bag_episode(bag, detectors, cfg, None)
    if len(frames) < args.min_frames:
        print("skip short:", bag.name, "frames:", len(frames))
        return None

    hand = np.stack([fr.hand_pos for fr in frames]).astype(np.float32)
    obj = np.stack([fr.obj_pos for fr in frames]).astype(np.float32)
    hand = smooth_array(finite_interpolate(hand), cfg.smooth_window, cfg.smooth_polyorder)
    obj = smooth_array(finite_interpolate(obj), cfg.smooth_window, cfg.smooth_polyorder)

    cube = np.median(obj[: min(8, len(obj))], axis=0).astype(np.float64)
    if args.canonical_cube_xy is not None:
        cube[0] = args.canonical_cube_xy[0]
        cube[1] = args.canonical_cube_xy[1]
    else:
        cube[0] = float(np.clip(cube[0], -args.workspace_xy, args.workspace_xy))
        cube[1] = float(np.clip(cube[1], -args.workspace_xy, args.workspace_xy))
    if variant is not None:
        cube[:2] += variant["xy_offset"]
        cube[0] = float(np.clip(cube[0], -args.workspace_xy, args.workspace_xy))
        cube[1] = float(np.clip(cube[1], -args.workspace_xy, args.workspace_xy))
    cube_half_size = float(variant["cube_half_size"]) if variant is not None else float(parse_float_list(args.cube_size_values)[0])
    cube_rgba = (
        np.asarray(variant["cube_rgba"], dtype=np.float64)
        if variant is not None
        else parse_color_list(args.cube_colors)[0]
    )
    cube[2] = args.cube_z + (cube_half_size - args.reference_cube_half_size)

    hand_start = hand[0]
    hand_end = hand[-1]
    inferred_place = infer_place_target_from_hand_and_object(hand, obj, cfg)
    hand_disp_xy = hand_end[:2] - hand_start[:2]
    approach_hint = hand_start[:2] - np.median(obj[: min(8, len(obj))], axis=0)[:2]
    if np.linalg.norm(hand_disp_xy) > np.linalg.norm(approach_hint):
        approach_hint = -hand_disp_xy
    print(
        "bag:",
        bag.name,
        "variant:",
        0 if variant is None else variant["variant_index"],
        "frames:",
        len(frames),
        "cube:",
        np.round(cube, 4),
        "half_size:",
        round(cube_half_size, 4),
        "hand_disp:",
        np.round(hand_end - hand_start, 4),
        "approach_xy:",
        np.round(unit_xy(approach_hint, [-0.7, -0.7]), 4),
        "place_hint:",
        np.round(inferred_place, 4),
    )

    env = make_env()
    env.reset()
    reject_global_yaw(args.base_yaw_deg)
    apply_base_longitudinal_twist(env, args.base_twist_deg)
    set_cube_appearance(env, half_size=cube_half_size, rgba=cube_rgba)
    set_cube_pos(env, cube)
    env.sim.data.qpos[:6] = np.asarray(args.initial_q, dtype=np.float64)
    env.sim.data.qvel[:6] = 0.0
    env.sim.forward()
    disable_arm_collisions(env)
    disable_gripper_nonpad_collisions(env)
    boost_arm_actuators(env, kp=args.boost_kp, force=args.boost_force, damping=args.boost_damping)
    boost_gripper_grasp(env, kp=args.gripper_boost_kp, force=args.gripper_boost_force, friction=args.grasp_friction)

    q_path, close_path, label_path = build_cartesian_q_path(
        env,
        make_waypoints(
            cube_geom_pos(env).copy(),
            args.lift_height,
            args.lateral_grasp_z,
            args.waypoint_style,
            approach_xy=approach_hint,
            smooth_segments=args.smooth_waypoint_segments,
            grasp_cube_z_offset=args.grasp_cube_z_offset,
            side_grasp_xy_offset=args.side_grasp_xy_offset,
        ),
        q_seed=np.asarray(args.ik_seed_q, dtype=np.float64) if args.ik_seed_q is not None else env.sim.data.qpos[:6].copy(),
        points_per_meter=args.path_points_per_meter,
        feature="pad_mid",
        wrist_q6=args.wrist_q6,
    )

    obs_list = []
    actions = []
    states = []
    rewards = []
    dones = []

    path_steps = []
    for label in label_path:
        steps = args.steps_per_path
        if label == "close":
            steps = args.close_steps
        elif label == "lift":
            steps = args.lift_steps
        path_steps.append(steps)

    total_steps = int(sum(path_steps))
    timestep = float(env.sim.model.opt.timestep)
    substeps = max(1, int(round(1.0 / (20 * timestep))))
    sample_idx = 0
    for q_target, close, _label, steps in zip(q_path, close_path, label_path, path_steps):
        for _ in range(steps):
            q_now = env.sim.data.qpos[:6].copy()
            q_cmd = q_now + np.clip(q_target - q_now, -args.max_joint_step, args.max_joint_step)
            action = np.concatenate(
                [
                    (q_cmd - q_now).astype(np.float32),
                    np.array([1.0, 1.0], dtype=np.float32) if close else np.array([-1.0, -1.0], dtype=np.float32),
                ]
            )
            progress = sample_idx / max(1, total_steps - 1)
            obs = collect_obs(env, progress=progress)
            obs["object"] = object_obs(cube_geom_pos(env).astype(np.float32), obs["robot0_eef_pos"])
            obs["bag_hand_pos"] = hand[min(len(hand) - 1, int(progress * (len(hand) - 1)))].astype(np.float32)
            obs["bag_cube_pos"] = cube.astype(np.float32)
            obs["cube_size"] = np.array([cube_half_size], dtype=np.float32)
            obs["cube_color"] = cube_rgba.astype(np.float32)
            obs["pad_mid_pos"] = get_pad_mid_pos(env).astype(np.float32)
            obs_list.append(obs)
            states.append(env.sim.get_state().flatten().astype(np.float32))
            actions.append(action)
            rewards.append(float(env._check_success()))
            dones.append(0)
            set_ctrl(env, q_cmd, close)
            for _ in range(substeps):
                env.sim.step()
            sample_idx += 1

    rewards[-1] = float(env._check_success())
    dones[-1] = 1
    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.int32),
        "obs": {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0].keys()},
        "success": bool(env._check_success()),
        "variant": variant or {},
        "bag_name": bag.name,
        "bag_index": int(bag_index),
        "cube": cube.astype(np.float64),
        "cube_half_size": float(cube_half_size),
        "cube_rgba": cube_rgba.astype(np.float64),
    }


def write_hdf5(path, episodes, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": "Lift",
                "type": 1,
                "env_version": "1.5.0",
                "env_kwargs": {
                    "robots": "JakaMiniCobo",
                    "gripper_types": "PandaGripper",
                    "controller_configs": "BASIC/JOINT_POSITION_DIRECT_CTRL_BAG_DEMOS",
                    "use_object_obs": True,
                    "control_freq": 20,
                },
            }
        )
        data.attrs["dataset_note"] = "Generated from RealSense .bag human demos via YOLO/MediaPipe + JAKA IK retargeting."
        data.attrs["script_args"] = json.dumps(vars(args))
        total = 0
        demo_keys = []
        for i, ep_data in enumerate(episodes):
            ep = data.create_group(f"demo_{i}")
            ep.create_dataset("actions", data=ep_data["actions"], compression="gzip")
            ep.create_dataset("states", data=ep_data["states"], compression="gzip")
            ep.create_dataset("rewards", data=ep_data["rewards"])
            ep.create_dataset("dones", data=ep_data["dones"])
            obs_group = ep.create_group("obs")
            next_group = ep.create_group("next_obs")
            for key, value in ep_data["obs"].items():
                obs_group.create_dataset(key, data=value, compression="gzip")
                next_group.create_dataset(key, data=np.concatenate([value[1:], value[-1:]], axis=0), compression="gzip")
            ep.attrs["num_samples"] = int(len(ep_data["actions"]))
            ep.attrs["success"] = bool(ep_data.get("success", False))
            ep.attrs["source_bag"] = ep_data.get("bag_name", "")
            ep.attrs["bag_index"] = int(ep_data.get("bag_index", -1))
            ep.attrs["cube_xyz"] = np.asarray(ep_data.get("cube", np.zeros(3)), dtype=np.float64)
            ep.attrs["cube_xy"] = np.asarray(ep_data.get("cube", np.zeros(3)), dtype=np.float64)[:2]
            ep.attrs["cube_half_size"] = float(ep_data.get("cube_half_size", np.nan))
            ep.attrs["cube_rgba"] = np.asarray(ep_data.get("cube_rgba", np.zeros(4)), dtype=np.float64)
            ep.attrs["variant"] = json.dumps(
                {
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in ep_data.get("variant", {}).items()
                }
            )
            total += int(len(ep_data["actions"]))
            demo_keys.append(f"demo_{i}".encode("utf-8"))
        data.attrs["total"] = total
        mask = f.create_group("mask")
        keys = np.asarray(demo_keys)
        split = max(1, int(0.8 * len(keys))) if len(keys) > 1 else len(keys)
        mask.create_dataset("train", data=keys[:split])
        mask.create_dataset("valid", data=keys[split:])


def parse_vec(text, n):
    values = [float(v) for v in text.split(",")]
    if len(values) != n:
        raise argparse.ArgumentTypeError(f"Expected {n} comma-separated values")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-dir", default="videos_cubo_3/videos_bag")
    parser.add_argument("--output", default="tests/assets/bag_lift_direct_ctrl.hdf5")
    parser.add_argument("--yolo-model", default="best_2.onnx")
    parser.add_argument("--max-bags", type=int, default=8)
    parser.add_argument("--cube-variants", type=int, default=1)
    parser.add_argument("--cube-pos-jitter", type=float, default=0.0)
    parser.add_argument("--variant-sampling", choices=["halton", "random"], default="halton")
    parser.add_argument("--cube-size-values", default="0.020")
    parser.add_argument("--cube-colors", default="0.80,0.05,0.05,1;0.05,0.35,0.90,1;0.05,0.65,0.25,1;0.95,0.72,0.12,1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--object-conf", type=float, default=0.25)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--initial-q", type=lambda s: parse_vec(s, 6), default=[-0.0007, 1.5797, -0.0322, 0.001, 0.0329, 0.0193])
    parser.add_argument("--ik-seed-q", type=lambda s: parse_vec(s, 6), default=None)
    parser.add_argument("--wrist-q6", type=float, default=-1.74533)
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--base-twist-deg", type=float, default=0.0)
    parser.add_argument("--cube-z", type=float, default=0.8307)
    parser.add_argument("--reference-cube-half-size", type=float, default=0.020)
    parser.add_argument("--canonical-cube-xy", type=lambda s: parse_vec(s, 2), default=None)
    parser.add_argument("--workspace-xy", type=float, default=0.045)
    parser.add_argument("--lateral-grasp-z", type=float, default=0.045)
    parser.add_argument("--lift-height", type=float, default=0.28)
    parser.add_argument("--waypoint-style", choices=["natural_arc", "scripted_grasp", "bag_center_smooth", "bag_side_grasp", "bag_top_grasp"], default="natural_arc")
    parser.add_argument("--smooth-waypoint-segments", type=int, default=0)
    parser.add_argument("--grasp-cube-z-offset", type=float, default=0.0)
    parser.add_argument("--side-grasp-xy-offset", type=float, default=0.012)
    parser.add_argument("--path-points-per-meter", type=float, default=160.0)
    parser.add_argument("--max-joint-step", type=float, default=0.10)
    parser.add_argument("--steps-per-path", type=int, default=6)
    parser.add_argument("--close-steps", type=int, default=26)
    parser.add_argument("--lift-steps", type=int, default=10)
    parser.add_argument("--boost-kp", type=float, default=1200.0)
    parser.add_argument("--boost-force", type=float, default=2500.0)
    parser.add_argument("--boost-damping", type=float, default=8.0)
    parser.add_argument("--gripper-boost-kp", type=float, default=1800.0)
    parser.add_argument("--gripper-boost-force", type=float, default=420.0)
    parser.add_argument("--grasp-friction", type=float, default=8.0)
    args = parser.parse_args()

    bag_paths = sorted(Path(args.bag_dir).glob("*.bag"))
    if args.max_bags <= 0:
        bag_paths = bag_paths[args.start_index :]
    else:
        bag_paths = bag_paths[args.start_index : args.start_index + args.max_bags]
    if not bag_paths:
        raise FileNotFoundError("No bag files selected")

    detectors = Detectors(args.yolo_model, args.object_conf)
    episodes = []
    for bag_index, bag in enumerate(bag_paths):
        for variant_index in range(max(1, args.cube_variants)):
            variant = make_variant(args, bag_index, variant_index)
            try:
                ep = make_episode_from_bag(bag, detectors, args, bag_index=bag_index, variant=variant)
            except Exception as exc:
                print("skip failed:", bag.name, "variant:", variant_index, repr(exc))
                continue
            if ep is not None:
                episodes.append(ep)
    if not episodes:
        raise RuntimeError("No bag episodes generated")
    write_hdf5(args.output, episodes, args)
    print("wrote:", args.output, "episodes:", len(episodes), "samples:", sum(len(ep["actions"]) for ep in episodes))


if __name__ == "__main__":
    main()
