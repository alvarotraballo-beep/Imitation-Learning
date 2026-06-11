#!/usr/bin/env python3
"""Generate robust BC-MLP data from the validated phase-sequence teacher.

This writes only into bc_mlp_policy_experiment/robust_v2 by default.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_phase_sequence_policy import policy_q_reference, table_contact_count  # noqa: E402
from generate_scripted_lift_dataset import collect_obs, make_env, object_obs  # noqa: E402
from jaka_mount_utils import apply_base_longitudinal_twist, reject_global_yaw, style_vertical_base_support  # noqa: E402
from scripted_joint_ik_lift_smoke import (  # noqa: E402
    boost_arm_actuators,
    boost_gripper_grasp,
    cube_geom_pos,
    disable_arm_collisions,
    disable_gripper_nonpad_collisions,
    get_pad_mid_pos,
    set_ctrl,
    set_cube_appearance,
    set_cube_pos,
)


def parse_vec(text, n):
    values = [float(v) for v in str(text).split(",")]
    if len(values) != n:
        raise argparse.ArgumentTypeError(f"Expected {n} comma-separated values")
    return np.asarray(values, dtype=np.float64)


def parse_float_list(text):
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def parse_color_list(text):
    colors = []
    for item in str(text).split(";"):
        vals = [float(v.strip()) for v in item.split(",") if v.strip()]
        if len(vals) == 3:
            vals.append(1.0)
        if len(vals) != 4:
            raise argparse.ArgumentTypeError("Colors must be r,g,b,a entries separated by semicolons")
        colors.append(np.asarray(vals, dtype=np.float64))
    return colors


def stack_obs(obs_list):
    return {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0].keys()}


def sample_scenario(args, rng, index):
    if args.sampling == "grid":
        x_values = np.linspace(args.cube_x_range[0], args.cube_x_range[1], args.grid_x)
        y_values = np.linspace(args.cube_y_range[0], args.cube_y_range[1], args.grid_y)
        xy_pairs = [(x, y) for x in x_values for y in y_values]
        x, y = xy_pairs[index % len(xy_pairs)]
    else:
        x = rng.uniform(args.cube_x_range[0], args.cube_x_range[1])
        y = rng.uniform(args.cube_y_range[0], args.cube_y_range[1])
    sizes = parse_float_list(args.cube_size_values)
    colors = parse_color_list(args.cube_colors)
    cube_half_size = float(sizes[index % len(sizes)])
    cube_rgba = colors[index % len(colors)]
    cube = np.array(
        [x, y, args.cube_z + (cube_half_size - args.reference_cube_half_size)],
        dtype=np.float64,
    )
    q_noise = rng.uniform(-args.initial_q_noise, args.initial_q_noise)
    initial_q = args.initial_q + q_noise
    return cube, cube_half_size, cube_rgba, initial_q, q_noise


def obs_with_extra(env, progress, cube_initial_pos, cube_half_size, cube_rgba):
    obs = collect_obs(env, progress=progress)
    cube_now = cube_geom_pos(env).astype(np.float32)
    obs["object"] = object_obs(cube_now, obs["robot0_eef_pos"])
    obs["cube_initial_pos"] = cube_initial_pos.astype(np.float32)
    obs["cube_size"] = np.array([cube_half_size], dtype=np.float32)
    obs["cube_color"] = cube_rgba.astype(np.float32)
    obs["pad_mid_pos"] = get_pad_mid_pos(env).astype(np.float32)
    return obs


def rollout_teacher(args, policy, episode_idx, rng):
    cube, cube_half_size, cube_rgba, initial_q, q_noise = sample_scenario(args, rng, episode_idx)
    env = make_env(has_offscreen_renderer=False)
    env.reset()
    reject_global_yaw(args.base_yaw_deg)
    apply_base_longitudinal_twist(env, args.base_twist_deg)
    if not args.hide_base_stand:
        style_vertical_base_support(env)
    set_cube_appearance(env, half_size=cube_half_size, rgba=cube_rgba)
    set_cube_pos(env, cube)
    cube_initial_pos = cube_geom_pos(env).copy()
    env.sim.data.qpos[:6] = initial_q.copy()
    env.sim.data.qvel[:6] = 0.0
    env.sim.forward()

    disable_arm_collisions(env)
    disable_gripper_nonpad_collisions(env)
    boost_arm_actuators(env, kp=args.boost_kp, force=args.boost_force, damping=args.boost_damping)
    boost_gripper_grasp(env, kp=args.gripper_boost_kp, force=args.gripper_boost_force, friction=args.grasp_friction)

    q_ref = policy_q_reference(policy, cube_initial_pos.copy(), cube_half_size)
    close_ref = policy["close_ref"].astype(bool)
    teacher_horizon = len(q_ref)
    total_steps = args.warmup_steps + teacher_horizon

    timestep = float(env.sim.model.opt.timestep)
    substeps = max(1, int(round(1.0 / (20 * timestep))))
    prev_q_cmd = env.sim.data.qpos[:6].copy()

    obs_list = []
    actions = []
    states = []
    rewards = []
    dones = []
    max_bad_table_contacts = 0
    success_step = None

    for step in range(total_steps):
        phase = 0 if step < args.warmup_steps else min(teacher_horizon - 1, step - args.warmup_steps)
        progress = phase / max(1, teacher_horizon - 1)
        q_now = env.sim.data.qpos[:6].copy()
        delta = args.kp * (q_ref[phase] - q_now)
        q_cmd = q_now + np.clip(delta, -args.max_joint_step, args.max_joint_step)
        if args.command_smoothing > 0.0:
            alpha = float(np.clip(args.command_smoothing, 0.0, 0.98))
            q_cmd = alpha * prev_q_cmd + (1.0 - alpha) * q_cmd
        prev_q_cmd = q_cmd.copy()
        close = bool(close_ref[phase])
        if progress < args.min_close_progress:
            close = False
        action = np.concatenate(
            [
                (q_cmd - q_now).astype(np.float32),
                np.array([1.0, 1.0], dtype=np.float32) if close else np.array([-1.0, -1.0], dtype=np.float32),
            ]
        )

        obs_list.append(obs_with_extra(env, progress, cube_initial_pos, cube_half_size, cube_rgba))
        states.append(env.sim.get_state().flatten().astype(np.float32))
        actions.append(action)
        rewards.append(float(env._check_success()))
        dones.append(0)

        set_ctrl(env, q_cmd, close)
        for _ in range(substeps):
            env.sim.step()

        bad_contacts, _ = table_contact_count(env)
        max_bad_table_contacts = max(max_bad_table_contacts, bad_contacts)
        if env._check_success() and success_step is None:
            success_step = step

    dones[-1] = 1
    final_cube = cube_geom_pos(env).copy()
    success = bool(env._check_success())
    if args.min_final_cube_z is not None:
        success = success and float(final_cube[2]) >= args.min_final_cube_z
    return {
        "success": success,
        "success_step": -1 if success_step is None else int(success_step),
        "final_cube": final_cube.astype(np.float32),
        "max_bad_table_contacts": int(max_bad_table_contacts),
        "actions": np.asarray(actions, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.int32),
        "obs": stack_obs(obs_list),
        "cube": cube.astype(np.float64),
        "cube_initial_pos": cube_initial_pos.astype(np.float64),
        "cube_half_size": float(cube_half_size),
        "cube_rgba": cube_rgba.astype(np.float64),
        "initial_q": initial_q.astype(np.float64),
        "initial_q_noise": q_noise.astype(np.float64),
    }


def write_dataset(path, episodes, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["dataset_note"] = (
            "Robust BC-MLP delta dataset generated from the validated bag-derived "
            "phase-sequence teacher with randomized cube positions and initial_q."
        )
        data.attrs["source_teacher"] = str(args.teacher_policy)
        data.attrs["script_args"] = json.dumps(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(args).items()},
            sort_keys=True,
        )
        total = 0
        demo_keys = []
        for i, ep_data in enumerate(episodes):
            ep = data.create_group(f"demo_{i}")
            ep.create_dataset("actions", data=ep_data["actions"], compression="gzip")
            ep.create_dataset("states", data=ep_data["states"], compression="gzip")
            ep.create_dataset("rewards", data=ep_data["rewards"])
            ep.create_dataset("dones", data=ep_data["dones"])
            obs_group = ep.create_group("obs")
            next_obs_group = ep.create_group("next_obs")
            for key, value in ep_data["obs"].items():
                obs_group.create_dataset(key, data=value, compression="gzip")
                next_value = np.concatenate([value[1:], value[-1:]], axis=0)
                next_obs_group.create_dataset(key, data=next_value, compression="gzip")
            ep.attrs["num_samples"] = int(len(ep_data["actions"]))
            ep.attrs["success"] = bool(ep_data["success"])
            ep.attrs["success_step"] = int(ep_data["success_step"])
            ep.attrs["final_cube"] = ep_data["final_cube"]
            ep.attrs["max_bad_table_contacts"] = int(ep_data["max_bad_table_contacts"])
            ep.attrs["cube_xyz"] = ep_data["cube"]
            ep.attrs["cube_initial_pos"] = ep_data["cube_initial_pos"]
            ep.attrs["cube_half_size"] = float(ep_data["cube_half_size"])
            ep.attrs["cube_rgba"] = ep_data["cube_rgba"]
            ep.attrs["initial_q"] = ep_data["initial_q"]
            ep.attrs["initial_q_noise"] = ep_data["initial_q_noise"]
            total += int(len(ep_data["actions"]))
            demo_keys.append(f"demo_{i}".encode("utf-8"))
        data.attrs["total"] = total
        mask = f.create_group("mask")
        keys = np.asarray(demo_keys)
        split = max(1, int(0.85 * len(keys))) if len(keys) > 1 else len(keys)
        mask.create_dataset("train", data=keys[:split])
        mask.create_dataset("valid", data=keys[split:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-policy", default="bc_trained_models/phase_sequence_policy_bag_basetwist30_openfix_variants_allbags_3var_cond_smooth31.npz")
    parser.add_argument("--output", default="bc_mlp_policy_experiment/robust_v2/datasets/robust_teacher_delta_96eps.hdf5")
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--max-attempts", type=int, default=180)
    parser.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--sampling", choices=["random", "grid"], default="random")
    parser.add_argument("--grid-x", type=int, default=7)
    parser.add_argument("--grid-y", type=int, default=7)
    parser.add_argument("--cube-x-range", type=lambda s: parse_vec(s, 2), default="-0.020,0.050")
    parser.add_argument("--cube-y-range", type=lambda s: parse_vec(s, 2), default="-0.050,0.030")
    parser.add_argument("--cube-z", type=float, default=0.8307)
    parser.add_argument("--cube-size-values", default="0.016,0.018,0.020,0.021,0.023")
    parser.add_argument("--reference-cube-half-size", type=float, default=0.020)
    parser.add_argument("--cube-colors", default="0.80,0.05,0.05,1;0.05,0.35,0.90,1;0.05,0.65,0.25,1;0.95,0.72,0.12,1")
    parser.add_argument("--initial-q", type=lambda s: parse_vec(s, 6), default="-0.0007,1.5797,-0.0322,0.001,0.0329,0.0193")
    parser.add_argument("--initial-q-noise", type=lambda s: parse_vec(s, 6), default="0.18,0.16,0.16,0.10,0.16,0.22")
    parser.add_argument("--warmup-steps", type=int, default=260)
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--base-twist-deg", type=float, default=30.0)
    parser.add_argument("--hide-base-stand", action="store_true")
    parser.add_argument("--kp", type=float, default=0.9)
    parser.add_argument("--max-joint-step", type=float, default=0.09)
    parser.add_argument("--command-smoothing", type=float, default=0.25)
    parser.add_argument("--min-close-progress", type=float, default=0.45)
    parser.add_argument("--min-final-cube-z", type=float, default=0.90)
    parser.add_argument("--boost-kp", type=float, default=1200.0)
    parser.add_argument("--boost-force", type=float, default=2500.0)
    parser.add_argument("--boost-damping", type=float, default=8.0)
    parser.add_argument("--gripper-boost-kp", type=float, default=1800.0)
    parser.add_argument("--gripper-boost-force", type=float, default=220.0)
    parser.add_argument("--grasp-friction", type=float, default=5.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    policy = np.load(args.teacher_policy, allow_pickle=True)
    episodes = []
    attempts = 0
    while len(episodes) < args.episodes and attempts < args.max_attempts:
        ep = rollout_teacher(args, policy, attempts, rng)
        attempts += 1
        accepted = ep["success"] or not args.require_success
        print(
            f"attempt={attempts:03d}",
            f"accepted={accepted}",
            f"success={ep['success']}",
            f"success_step={ep['success_step']}",
            f"cube={np.round(ep['cube_initial_pos'], 4)}",
            f"size={ep['cube_half_size']:.3f}",
            f"q_noise={np.round(ep['initial_q_noise'], 3)}",
            f"final_cube={np.round(ep['final_cube'], 4)}",
            f"bad_table={ep['max_bad_table_contacts']}",
        )
        if accepted:
            episodes.append(ep)
    if len(episodes) < args.episodes:
        raise RuntimeError(f"Only generated {len(episodes)} accepted episodes from {attempts} attempts")
    write_dataset(args.output, episodes, args)
    print("wrote:", args.output, "episodes:", len(episodes), "attempts:", attempts, "samples:", sum(len(ep["actions"]) for ep in episodes))


if __name__ == "__main__":
    main()
