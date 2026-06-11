#!/usr/bin/env python3
"""Roll out a trained BC-MLP checkpoint in the JAKA lift environment."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_scripted_lift_dataset import collect_obs, make_env  # noqa: E402
from jaka_mount_utils import apply_base_longitudinal_twist, reject_global_yaw, style_vertical_base_support  # noqa: E402
from scripted_joint_ik_lift_smoke import (  # noqa: E402
    VideoRecorder,
    boost_arm_actuators,
    boost_gripper_grasp,
    cube_geom_pos,
    disable_arm_collisions,
    disable_gripper_nonpad_collisions,
    set_cube_appearance,
    set_ctrl,
    set_cube_pos,
)
from train_bc_mlp import BCMLP  # noqa: E402


def parse_vec(text, n):
    values = [float(v) for v in text.split(",")]
    if len(values) != n:
        raise argparse.ArgumentTypeError(f"Expected {n} comma-separated values")
    return np.asarray(values, dtype=np.float64)


def obs_vector(env, obs_keys, progress, cube_half_size, cube_initial_pos):
    obs = collect_obs(env, progress=progress)
    obs["cube_initial_pos"] = np.asarray(cube_initial_pos, dtype=np.float32)
    obs["cube_pos"] = cube_geom_pos(env).astype(np.float32)
    obs["cube_xy"] = obs["cube_pos"][:2].astype(np.float32)
    obs["cube_size"] = np.array([cube_half_size], dtype=np.float32)
    values = []
    for key in obs_keys:
        if key not in obs:
            raise KeyError(f"Observation key cannot be built at rollout time: {key}")
        value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        values.append(value)
    return np.concatenate(values, axis=0).astype(np.float32)


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BCMLP(
        int(ckpt["obs_dim"]),
        hidden_sizes=tuple(int(v) for v in ckpt["hidden_sizes"]),
        activation=ckpt.get("activation", "relu"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tensors = {
        "obs_mean": torch.from_numpy(ckpt["obs_mean"].astype(np.float32)).to(device),
        "obs_std": torch.from_numpy(ckpt["obs_std"].astype(np.float32)).to(device),
        "q_mean": torch.from_numpy(ckpt["q_mean"].astype(np.float32)).to(device),
        "q_std": torch.from_numpy(ckpt["q_std"].astype(np.float32)).to(device),
    }
    return ckpt, model, tensors


def table_contact_count(env):
    model = env.sim.model
    count = 0
    names = []
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        name1 = model.geom_id2name(contact.geom1) or ""
        name2 = model.geom_id2name(contact.geom2) or ""
        pair = {name1, name2}
        has_table = any("table" in name for name in pair)
        allowed = any("cube" in name for name in pair)
        allowed = allowed or any("fingerpad" in name or "pad_collision" in name for name in pair)
        if has_table and not allowed:
            count += 1
            names.append((name1, name2))
    return count, names[:5]


def run_rollout(args, rollout_idx, video_path=None):
    device = torch.device(args.device)
    ckpt, model, tensors = load_policy(args.checkpoint, device)

    env = make_env(has_offscreen_renderer=video_path is not None)
    env.reset()
    reject_global_yaw(args.base_yaw_deg)
    apply_base_longitudinal_twist(env, args.base_twist_deg)
    if not args.hide_base_stand:
        style_vertical_base_support(env)

    cube_pos = args.cube.copy()
    if not args.keep_cube_z:
        cube_pos[2] += args.cube_half_size - args.reference_cube_half_size
    set_cube_appearance(env, half_size=args.cube_half_size, rgba=args.cube_rgba)
    set_cube_pos(env, cube_pos)
    cube_initial_pos = cube_geom_pos(env).copy()

    env.sim.data.qpos[:6] = args.initial_q.copy()
    env.sim.data.qvel[:6] = 0.0
    env.sim.forward()

    disable_arm_collisions(env)
    disable_gripper_nonpad_collisions(env)
    boost_arm_actuators(env, kp=args.boost_kp, force=args.boost_force, damping=args.boost_damping)
    boost_gripper_grasp(env, kp=args.gripper_boost_kp, force=args.gripper_boost_force, friction=args.grasp_friction)

    recorder = None
    if video_path:
        recorder = VideoRecorder(env, video_path, camera=args.video_camera, width=args.video_width, height=args.video_height)
        recorder.capture()

    timestep = float(env.sim.model.opt.timestep)
    substeps = max(1, int(round(1.0 / (20 * timestep))))
    prev_q_cmd = env.sim.data.qpos[:6].copy()
    success_step = None
    close_latched = False
    max_bad_table_contacts = 0
    last_bad_pairs = []

    for step in range(args.horizon):
        progress = step / max(1, args.horizon - 1)
        x_np = obs_vector(
            env,
            ckpt["obs_keys"],
            progress=progress,
            cube_half_size=args.cube_half_size,
            cube_initial_pos=cube_initial_pos,
        )[None]
        x = torch.from_numpy(x_np).to(device)
        with torch.no_grad():
            x_norm = (x - tensors["obs_mean"]) / tensors["obs_std"]
            pred_q_norm, pred_grip_logit = model(x_norm)
            q_target = (pred_q_norm * tensors["q_std"] + tensors["q_mean"]).detach().cpu().numpy()[0]
            close_prob = float(torch.sigmoid(pred_grip_logit).detach().cpu().numpy()[0])

        q_now = env.sim.data.qpos[:6].copy()
        delta = args.kp * (q_target - q_now)
        q_cmd = q_now + np.clip(delta, -args.max_joint_step, args.max_joint_step)
        if args.command_smoothing > 0.0:
            alpha = float(np.clip(args.command_smoothing, 0.0, 0.98))
            q_cmd = alpha * prev_q_cmd + (1.0 - alpha) * q_cmd
        prev_q_cmd = q_cmd.copy()

        close = close_prob >= args.gripper_threshold
        if progress < args.min_close_progress:
            close = False
        if args.hold_close_once:
            close_latched = close_latched or close
            close = close_latched

        set_ctrl(env, q_cmd, close)
        for _ in range(substeps):
            env.sim.step()
        if recorder is not None:
            recorder.capture()

        bad_contacts, bad_pairs = table_contact_count(env)
        max_bad_table_contacts = max(max_bad_table_contacts, bad_contacts)
        if bad_pairs:
            last_bad_pairs = bad_pairs

        if step % args.print_every == 0 or step == args.horizon - 1:
            print(
                f"rollout={rollout_idx} step={step:04d} progress={progress:.3f} "
                f"close_prob={close_prob:.3f} close={close} "
                f"cube={np.round(cube_geom_pos(env), 4)} "
                f"q_target={np.round(q_target, 3)} "
                f"bad_table_contacts={bad_contacts}"
            )

        if env._check_success() and success_step is None:
            success_step = step
            if args.stop_on_success:
                break

    final_cube = cube_geom_pos(env).copy()
    success = bool(env._check_success())
    if args.min_final_cube_z is not None:
        success = success and float(final_cube[2]) >= args.min_final_cube_z
    if recorder is not None:
        recorder.close()
    return {
        "success": success,
        "success_step": success_step,
        "final_cube": final_cube,
        "max_bad_table_contacts": max_bad_table_contacts,
        "last_bad_pairs": last_bad_pairs,
    }


def load_scenarios(path):
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw["scenarios"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="bc_mlp_policy_experiment/models/bc_mlp_cube_initial_progress_absq_allbags.pt")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=2694)
    parser.add_argument("--cube", type=lambda s: parse_vec(s, 3), default="0.0144,-0.0123,0.8307")
    parser.add_argument("--cube-half-size", type=float, default=0.020)
    parser.add_argument("--reference-cube-half-size", type=float, default=0.020)
    parser.add_argument("--keep-cube-z", action="store_true")
    parser.add_argument("--cube-rgba", type=lambda s: parse_vec(s, 4), default="0.80,0.05,0.05,1")
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--initial-q", type=lambda s: parse_vec(s, 6), default="-0.0007,1.5797,-0.0322,0.001,0.0329,0.0193")
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--base-twist-deg", type=float, default=30.0)
    parser.add_argument("--hide-base-stand", action="store_true")
    parser.add_argument("--kp", type=float, default=0.9)
    parser.add_argument("--max-joint-step", type=float, default=0.09)
    parser.add_argument("--command-smoothing", type=float, default=0.25)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--min-close-progress", type=float, default=0.0)
    parser.add_argument("--hold-close-once", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-final-cube-z", type=float, default=0.90)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-camera", default="frontview")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--boost-kp", type=float, default=1200.0)
    parser.add_argument("--boost-force", type=float, default=2500.0)
    parser.add_argument("--boost-damping", type=float, default=8.0)
    parser.add_argument("--gripper-boost-kp", type=float, default=1800.0)
    parser.add_argument("--gripper-boost-force", type=float, default=220.0)
    parser.add_argument("--grasp-friction", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file) if args.scenario_file else [None] * args.rollouts
    video_base = Path(args.video_path) if args.video_path else None
    results = []
    for rollout_idx, scenario in enumerate(scenarios[: args.rollouts]):
        if scenario:
            args.cube = np.asarray(scenario.get("cube", args.cube), dtype=np.float64)
            args.cube_half_size = float(scenario.get("cube_half_size", args.cube_half_size))
            args.cube_rgba = np.asarray(scenario.get("cube_rgba", args.cube_rgba), dtype=np.float64)
        rollout_video = None
        if video_base is not None:
            rollout_video = video_base if args.rollouts == 1 else video_base.with_name(f"{video_base.stem}_{rollout_idx:02d}{video_base.suffix}")
        result = run_rollout(args, rollout_idx=rollout_idx, video_path=rollout_video)
        results.append(result)
        print(
            f"rollout={rollout_idx} success={result['success']} "
            f"success_step={result['success_step']} final_cube={np.round(result['final_cube'], 4)} "
            f"max_bad_table_contacts={result['max_bad_table_contacts']}"
        )
        if result["last_bad_pairs"]:
            print("last_bad_table_pairs:", result["last_bad_pairs"])

    successes = sum(int(r["success"]) for r in results)
    print("bc_mlp successes:", successes, "/", len(results))
    if successes != len(results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
