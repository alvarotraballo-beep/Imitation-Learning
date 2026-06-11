#!/usr/bin/env python3
"""Evaluate a phase-indexed direct-control policy on the JAKA lift setup."""

import argparse
from pathlib import Path

import numpy as np

from generate_scripted_lift_dataset import collect_obs, make_env
from jaka_mount_utils import apply_base_longitudinal_twist, reject_global_yaw, style_vertical_base_support
from scripted_joint_ik_lift_smoke import (
    VideoRecorder,
    boost_arm_actuators,
    boost_gripper_grasp,
    cube_geom_pos,
    disable_arm_collisions,
    disable_gripper_nonpad_collisions,
    get_pad_mid_pos,
    get_site_pos,
    set_cube_appearance,
    set_ctrl,
    set_cube_pos,
    solve_best_ik,
)


def parse_vec(text, n):
    values = [float(v) for v in text.split(",")]
    if len(values) != n:
        raise argparse.ArgumentTypeError(f"Expected {n} comma-separated values")
    return np.asarray(values, dtype=np.float64)


def policy_q_reference(policy, cube_pos, cube_half_size):
    q_coef = policy["q_coef"] if "q_coef" in policy else np.zeros((0,), dtype=np.float32)
    if q_coef.size:
        feature = np.r_[1.0, cube_pos.astype(np.float64), float(cube_half_size)]
        return np.einsum("f,fnj->nj", feature, q_coef.astype(np.float64))
    return policy["q_ref"].astype(np.float64)


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
    env.sim.data.qpos[:6] = args.initial_q.copy()
    env.sim.data.qvel[:6] = 0.0
    env.sim.forward()

    disable_arm_collisions(env)
    disable_gripper_nonpad_collisions(env)
    boost_arm_actuators(env, kp=args.boost_kp, force=args.boost_force, damping=args.boost_damping)
    boost_gripper_grasp(env, kp=args.gripper_boost_kp, force=args.gripper_boost_force, friction=args.grasp_friction)

    policy = np.load(args.policy, allow_pickle=True)
    q_ref = policy_q_reference(policy, cube_geom_pos(env).copy(), args.cube_half_size)
    close_ref = policy["close_ref"].astype(bool)

    recorder = None
    if video_path:
        recorder = VideoRecorder(env, video_path, camera=args.video_camera, width=args.video_width, height=args.video_height)
        recorder.capture()

    timestep = float(env.sim.model.opt.timestep)
    substeps = max(1, int(round(1.0 / (20 * timestep))))
    horizon = args.horizon if args.horizon > 0 else len(q_ref)
    success_step = None
    adaptive_lift_started = False
    max_bad_table_contacts = 0
    last_bad_pairs = []
    prev_q_cmd = env.sim.data.qpos[:6].copy()

    for step in range(horizon):
        phase = min(len(q_ref) - 1, step)
        q_now = env.sim.data.qpos[:6].copy()
        delta = args.kp * (q_ref[phase] - q_now)
        q_cmd = q_now + np.clip(delta, -args.max_joint_step, args.max_joint_step)
        if args.command_smoothing > 0.0:
            alpha = float(np.clip(args.command_smoothing, 0.0, 0.98))
            q_cmd = alpha * prev_q_cmd + (1.0 - alpha) * q_cmd
        prev_q_cmd = q_cmd.copy()
        close = bool(close_ref[phase])
        set_ctrl(env, q_cmd, close)
        for _ in range(substeps):
            env.sim.step()
        if recorder is not None:
            recorder.capture()

        bad_contacts, bad_pairs = table_contact_count(env)
        max_bad_table_contacts = max(max_bad_table_contacts, bad_contacts)
        if bad_pairs:
            last_bad_pairs = bad_pairs

        if step % args.print_every == 0 or step == horizon - 1:
            obs = collect_obs(env, progress=phase / max(1, len(q_ref) - 1))
            pad_mid = get_pad_mid_pos(env)
            cube = cube_geom_pos(env)
            print(
                f"rollout={rollout_idx} step={step:04d} close={close} "
                f"eef={np.round(get_site_pos(env), 4)} pad={np.round(pad_mid, 4)} "
                f"cube={np.round(cube, 4)} bad_table_contacts={bad_contacts} "
                f"progress_obs={float(obs['progress'][0]):.3f}"
            )

        if env._check_success() and success_step is None:
            success_step = step
            if args.adaptive_lift_on_success:
                pad_now = get_pad_mid_pos(env)
                q_lift, _, dist = solve_best_ik(
                    env,
                    pad_now + np.array([0.0, 0.0, args.adaptive_lift_height], dtype=np.float64),
                    env.sim.data.qpos[:6].copy(),
                    feature="pad_mid",
                    wrist_q6=args.wrist_q6,
                )
                end = min(len(q_ref), phase + args.adaptive_lift_steps)
                if end > phase:
                    start_q = env.sim.data.qpos[:6].copy()
                    for local_idx, alpha in enumerate(np.linspace(0.0, 1.0, end - phase)):
                        q_ref[phase + local_idx] = (1.0 - alpha) * start_q + alpha * q_lift
                    q_ref[end:] = q_lift
                    close_ref[phase:] = True
                adaptive_lift_started = True
                print(
                    "adaptive_lift:",
                    "phase",
                    phase,
                    "target_dist",
                    round(float(dist), 5),
                    "target_q",
                    np.round(q_lift, 4),
                )
            if args.hold_on_success:
                q_ref[phase:] = env.sim.data.qpos[:6].copy()
                close_ref[phase:] = True
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
        "adaptive_lift_started": adaptive_lift_started,
        "max_bad_table_contacts": max_bad_table_contacts,
        "last_bad_pairs": last_bad_pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="bc_trained_models/phase_sequence_policy.npz")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--cube", type=lambda s: parse_vec(s, 3), default="0.0144,-0.0123,0.8307")
    parser.add_argument("--cube-half-size", type=float, default=0.020)
    parser.add_argument("--reference-cube-half-size", type=float, default=0.020)
    parser.add_argument("--keep-cube-z", action="store_true")
    parser.add_argument("--cube-rgba", type=lambda s: parse_vec(s, 4), default="0.80,0.05,0.05,1")
    parser.add_argument("--initial-q", type=lambda s: parse_vec(s, 6), default="-0.0007,1.5797,-0.0322,0.001,0.0329,0.0193")
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--base-twist-deg", type=float, default=0.0)
    parser.add_argument("--hide-base-stand", action="store_true")
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--max-joint-step", type=float, default=0.12)
    parser.add_argument("--command-smoothing", type=float, default=0.0)
    parser.add_argument("--hold-on-success", action="store_true")
    parser.add_argument("--adaptive-lift-on-success", action="store_true")
    parser.add_argument("--adaptive-lift-height", type=float, default=0.28)
    parser.add_argument("--adaptive-lift-steps", type=int, default=520)
    parser.add_argument("--wrist-q6", type=float, default=-1.74533)
    parser.add_argument("--min-final-cube-z", type=float, default=None)
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
    args = parser.parse_args()

    video_base = Path(args.video_path) if args.video_path else None
    results = []
    for rollout_idx in range(args.rollouts):
        rollout_video = None
        if video_base is not None:
            if args.rollouts == 1:
                rollout_video = video_base
            else:
                rollout_video = video_base.with_name(f"{video_base.stem}_{rollout_idx:02d}{video_base.suffix}")
        result = run_rollout(args, rollout_idx=rollout_idx, video_path=rollout_video)
        results.append(result)
        print(
            f"rollout={rollout_idx} success={result['success']} "
            f"success_step={result['success_step']} final_cube={np.round(result['final_cube'], 4)} "
            f"adaptive_lift_started={result['adaptive_lift_started']} "
            f"max_bad_table_contacts={result['max_bad_table_contacts']}"
        )
        if result["last_bad_pairs"]:
            print("last_bad_table_pairs:", result["last_bad_pairs"])

    successes = sum(int(r["success"]) for r in results)
    print("policy successes:", successes, "/", len(results))
    if successes != len(results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
