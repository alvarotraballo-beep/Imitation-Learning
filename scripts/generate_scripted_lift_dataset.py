#!/usr/bin/env python3
"""Generate robomimic-style low-dimensional demos from the scripted JAKA lift."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite

from scripted_joint_ik_lift_smoke import (
    boost_arm_actuators,
    boost_gripper_grasp,
    build_cartesian_q_path,
    cube_geom_pos,
    disable_arm_collisions,
    disable_gripper_nonpad_collisions,
    get_pad_mid_pos,
    get_site_pos,
    make_controller,
    set_ctrl,
    set_cube_pos,
)


def make_env(has_offscreen_renderer=False):
    return suite.make(
        env_name="Lift",
        robots="JakaMiniCobo",
        gripper_types="PandaGripper",
        controller_configs=make_controller(),
        has_renderer=False,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=600,
        ignore_done=True,
    )


def object_obs(cube, eef):
    rel = cube - eef
    return np.concatenate([cube, rel], dtype=np.float32)


def collect_obs(env, progress=0.0):
    q = env.sim.data.qpos[:6].copy().astype(np.float32)
    qvel = env.sim.data.qvel[:6].copy().astype(np.float32)
    cube = cube_geom_pos(env).astype(np.float32)
    eef = get_site_pos(env).astype(np.float32)
    grip_qpos = env.sim.data.qpos[6:8].copy().astype(np.float32)
    grip_qvel = env.sim.data.qvel[6:8].copy().astype(np.float32)
    return {
        "object": object_obs(cube, eef),
        "robot0_eef_pos": eef,
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_joint_pos": q,
        "robot0_joint_pos_cos": np.cos(q).astype(np.float32),
        "robot0_joint_pos_sin": np.sin(q).astype(np.float32),
        "robot0_joint_vel": qvel,
        "robot0_gripper_qpos": grip_qpos,
        "robot0_gripper_qvel": grip_qvel,
        "pad_mid_pos": get_pad_mid_pos(env).astype(np.float32),
        "progress": np.array([progress], dtype=np.float32),
    }


def stack_obs(obs_list):
    keys = obs_list[0].keys()
    return {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in keys}


def make_waypoints(cube, pad_grasp_z_offset):
    return [
        ("above", cube + np.array([0.0, 0.0, 0.18]), False, 90),
        ("pregrasp", cube + np.array([0.0, 0.0, 0.055]), False, 70),
        ("grasp", cube + np.array([0.0, 0.0, pad_grasp_z_offset]), False, 50),
        ("close", cube + np.array([0.0, 0.0, pad_grasp_z_offset]), True, 80),
        ("lift", cube + np.array([0.0, 0.0, 0.26]), True, 140),
    ]


def rollout_demo(args, demo_idx):
    env = make_env()
    obs = env.reset()
    del obs

    cube = np.array(args.cube, dtype=np.float64)
    if args.jitter > 0:
        rng = np.random.default_rng(args.seed + demo_idx)
        cube[:2] += rng.uniform(-args.jitter, args.jitter, size=2)
    set_cube_pos(env, cube)

    q_init = np.array(args.initial_q, dtype=np.float64)
    env.sim.data.qpos[:6] = q_init
    env.sim.data.qvel[:6] = 0.0
    env.sim.forward()

    disable_arm_collisions(env)
    disable_gripper_nonpad_collisions(env)
    boost_arm_actuators(env, kp=args.boost_kp, force=args.boost_force, damping=args.boost_damping)
    boost_gripper_grasp(env, kp=args.gripper_boost_kp, force=args.gripper_boost_force, friction=args.grasp_friction)

    cube = cube_geom_pos(env).copy()
    waypoints = make_waypoints(cube, args.pad_grasp_z_offset)
    q_path, close_path, label_path = build_cartesian_q_path(
        env,
        waypoints,
        q_seed=env.sim.data.qpos[:6].copy(),
        points_per_meter=args.path_points_per_meter,
        feature="pad_mid",
        wrist_q6=args.wrist_q6,
    )

    timestep = float(env.sim.model.opt.timestep)
    substeps = max(1, int(round(1.0 / (20 * timestep))))

    obs_list = []
    actions = []
    states = []
    rewards = []
    dones = []
    path_steps = []
    for label in label_path:
        steps = 8
        if label == "close":
            steps = 24
        elif label == "lift":
            steps = 12
        path_steps.append(steps)
    total_steps = int(sum(path_steps))
    sample_idx = 0

    for q_target, close, label, steps in zip(q_path, close_path, label_path, path_steps):
        for _ in range(steps):
            q_now = env.sim.data.qpos[:6].copy()
            q_cmd = q_now + np.clip(q_target - q_now, -args.max_joint_step, args.max_joint_step)
            grip_cmd = np.array([1.0, 1.0], dtype=np.float32) if close else np.array([-1.0, -1.0], dtype=np.float32)
            dq_cmd = np.clip(q_cmd - q_now, -args.max_joint_step, args.max_joint_step)
            action = np.concatenate([dq_cmd.astype(np.float32), grip_cmd], dtype=np.float32)

            progress = sample_idx / max(1, total_steps - 1)
            obs_list.append(collect_obs(env, progress=progress))
            states.append(env.sim.get_state().flatten().astype(np.float32))
            actions.append(action)
            set_ctrl(env, q_cmd, close)
            for _ in range(substeps):
                env.sim.step()
            rewards.append(float(env._check_success()))
            dones.append(0)
            sample_idx += 1

    dones[-1] = 1
    success = bool(env._check_success())
    return {
        "success": success,
        "actions": np.asarray(actions, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.int32),
        "obs": stack_obs(obs_list),
    }


def write_hdf5(path, episodes, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env_args = {
        "env_name": "Lift",
        "type": 1,
        "env_version": "1.5.0",
        "env_kwargs": {
            "robots": "JakaMiniCobo",
            "gripper_types": "PandaGripper",
            "controller_configs": "BASIC/JOINT_POSITION_DIRECT_CTRL_DATASET",
            "use_camera_obs": False,
            "use_object_obs": True,
            "control_freq": 20,
        },
    }
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["dataset_note"] = "Actions are direct-control delta_q commands plus gripper close/open flags."
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
            total += int(len(ep_data["actions"]))
            demo_keys.append(f"demo_{i}".encode("utf-8"))
        data.attrs["total"] = total
        data.attrs["script_args"] = json.dumps(vars(args))
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
    parser.add_argument("--output", default="tests/assets/scripted_lift_direct_ctrl.hdf5")
    parser.add_argument("--num-demos", type=int, default=4)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--cube", type=lambda s: parse_vec(s, 3), default=[0.0144, -0.0123, 0.8307])
    parser.add_argument("--initial-q", type=lambda s: parse_vec(s, 6), default=[-0.0007, 1.5797, -0.0322, 0.001, 0.0329, 0.0193])
    parser.add_argument("--wrist-q6", type=float, default=-1.74533)
    parser.add_argument("--pad-grasp-z-offset", type=float, default=-0.025)
    parser.add_argument("--path-points-per-meter", type=float, default=180.0)
    parser.add_argument("--max-joint-step", type=float, default=0.12)
    parser.add_argument("--boost-kp", type=float, default=1200.0)
    parser.add_argument("--boost-force", type=float, default=2500.0)
    parser.add_argument("--boost-damping", type=float, default=8.0)
    parser.add_argument("--gripper-boost-kp", type=float, default=1800.0)
    parser.add_argument("--gripper-boost-force", type=float, default=220.0)
    parser.add_argument("--grasp-friction", type=float, default=5.0)
    args = parser.parse_args()

    episodes = []
    attempts = 0
    while len(episodes) < args.num_demos:
        attempts += 1
        ep = rollout_demo(args, attempts)
        print(f"attempt={attempts} success={ep['success']} samples={len(ep['actions'])}")
        if ep["success"] or not args.require_success:
            episodes.append(ep)
        if attempts > args.num_demos * 5:
            raise RuntimeError("Too many failed attempts while generating successful demos")

    write_hdf5(args.output, episodes, args)
    print("wrote:", args.output, "episodes:", len(episodes), "attempts:", attempts)


if __name__ == "__main__":
    main()
