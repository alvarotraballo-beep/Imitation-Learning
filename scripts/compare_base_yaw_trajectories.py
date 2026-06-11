#!/usr/bin/env python3
"""Compare bag-retargeted JAKA trajectories across base twist candidates."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


def wrap_to_pi(values):
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def read_script_args(data_group):
    raw = data_group.attrs.get("script_args", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def safe_obs(obs_group, key):
    if key not in obs_group:
        return None
    return obs_group[key][()]


def first_close_index(actions):
    if actions.shape[1] < 8:
        return len(actions) - 1
    close = np.mean(actions[:, 6:8], axis=1) > 0.0
    if not np.any(close):
        return len(actions) - 1
    return int(np.argmax(close))


def summarize_demo(demo):
    obs = demo["obs"]
    actions = demo["actions"][()].astype(np.float64)
    q = safe_obs(obs, "robot0_joint_pos")
    if q is None:
        raise KeyError("obs/robot0_joint_pos not found")
    q = q[:, :6].astype(np.float64)
    q_wrapped = wrap_to_pi(q)
    q_unwrapped = np.unwrap(q_wrapped, axis=0)
    dq = np.diff(q_unwrapped, axis=0)
    dq_norm = np.linalg.norm(dq, axis=1) if len(dq) else np.zeros(1)
    close_idx = first_close_index(actions)

    pad = safe_obs(obs, "pad_mid_pos")
    eef = safe_obs(obs, "robot0_eef_pos")
    cube = safe_obs(obs, "cube_pos")
    if cube is None:
        cube = safe_obs(obs, "bag_cube_pos")

    wrist_cols = [3, 4, 5]
    upper_cols = [1, 2, 3]
    q_close = q_wrapped[close_idx]
    q_final = q_wrapped[-1]
    q_abs_mean = np.mean(np.abs(q_wrapped), axis=0)

    return {
        "samples": len(q),
        "success": bool(demo.attrs.get("success", False)),
        "close_idx": close_idx,
        "q_path_len": float(np.sum(dq_norm)),
        "q_step_p95": float(np.percentile(dq_norm, 95)),
        "q_step_max": float(np.max(dq_norm)),
        "mean_abs_q": q_abs_mean,
        "mean_abs_wrist": float(np.mean(np.abs(q_wrapped[:, wrist_cols]))),
        "mean_abs_upper": float(np.mean(np.abs(q_wrapped[:, upper_cols]))),
        "close_abs_wrist": float(np.mean(np.abs(q_close[wrist_cols]))),
        "close_abs_upper": float(np.mean(np.abs(q_close[upper_cols]))),
        "q_close": q_close,
        "q_final": q_final,
        "pad_min_z": float(np.min(pad[:, 2])) if pad is not None else np.nan,
        "pad_close_z": float(pad[close_idx, 2]) if pad is not None else np.nan,
        "eef_min_z": float(np.min(eef[:, 2])) if eef is not None else np.nan,
        "cube_final_z": float(cube[-1, 2]) if cube is not None else np.nan,
        "action_abs_mean": np.mean(np.abs(actions[:, :6]), axis=0),
        "action_abs_max": np.max(np.abs(actions[:, :6]), axis=0),
    }


def aggregate(items):
    numeric = [
        "samples",
        "q_path_len",
        "q_step_p95",
        "q_step_max",
        "mean_abs_wrist",
        "mean_abs_upper",
        "close_abs_wrist",
        "close_abs_upper",
        "pad_min_z",
        "pad_close_z",
        "eef_min_z",
        "cube_final_z",
    ]
    result = {
        "successes": sum(int(item["success"]) for item in items),
        "demos": len(items),
    }
    for key in numeric:
        result[key] = float(np.mean([item[key] for item in items]))
    for key in ["mean_abs_q", "q_close", "q_final", "action_abs_mean", "action_abs_max"]:
        result[key] = np.mean(np.stack([item[key] for item in items], axis=0), axis=0)
    return result


def score_summary(summary):
    table_penalty = max(0.0, 0.822 - summary["pad_close_z"]) * 25.0
    return (
        0.35 * summary["q_path_len"]
        + 1.20 * summary["close_abs_wrist"]
        + 0.80 * summary["close_abs_upper"]
        + 6.00 * summary["q_step_p95"]
        + table_penalty
    )


def print_summary(path, summary, script_args):
    twist = script_args.get("base_twist_deg", "unknown")
    style = script_args.get("waypoint_style", "unknown")
    z_offset = script_args.get("grasp_cube_z_offset", "unknown")
    score = score_summary(summary)
    print("=" * 88)
    print(f"dataset: {path}")
    print(f"base_twist_deg={twist} style={style} grasp_cube_z_offset={z_offset}")
    print(f"success demos: {summary['successes']} / {summary['demos']} samples_mean={summary['samples']:.1f}")
    print(
        "score_lower_is_better="
        f"{score:.4f} q_path_len={summary['q_path_len']:.4f} "
        f"q_step_p95={summary['q_step_p95']:.5f} q_step_max={summary['q_step_max']:.5f}"
    )
    print(
        "posture:"
        f" close_abs_upper={summary['close_abs_upper']:.4f}"
        f" close_abs_wrist={summary['close_abs_wrist']:.4f}"
        f" mean_abs_upper={summary['mean_abs_upper']:.4f}"
        f" mean_abs_wrist={summary['mean_abs_wrist']:.4f}"
    )
    print(
        "clearance:"
        f" pad_min_z={summary['pad_min_z']:.4f}"
        f" pad_close_z={summary['pad_close_z']:.4f}"
        f" eef_min_z={summary['eef_min_z']:.4f}"
        f" cube_final_z={summary['cube_final_z']:.4f}"
    )
    print("q_close:", np.array2string(summary["q_close"], precision=4, suppress_small=True))
    print("q_final :", np.array2string(summary["q_final"], precision=4, suppress_small=True))
    print("mean_abs_q:", np.array2string(summary["mean_abs_q"], precision=4, suppress_small=True))
    print("mean_abs_action:", np.array2string(summary["action_abs_mean"], precision=5, suppress_small=True))
    print("max_abs_action :", np.array2string(summary["action_abs_max"], precision=5, suppress_small=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", help="HDF5 datasets to compare")
    parser.add_argument("--successful-only", action="store_true")
    args = parser.parse_args()

    ranked = []
    for dataset in args.datasets:
        path = Path(dataset)
        with h5py.File(path, "r") as f:
            data = f["data"]
            script_args = read_script_args(data)
            demos = []
            for key in sorted(data.keys()):
                demo = data[key]
                if args.successful_only and not bool(demo.attrs.get("success", False)):
                    continue
                demos.append(summarize_demo(demo))
        if not demos:
            print("=" * 88)
            print(f"dataset: {path}")
            print("no demos selected")
            continue
        summary = aggregate(demos)
        print_summary(path, summary, script_args)
        ranked.append((score_summary(summary), path))

    if ranked:
        print("=" * 88)
        print("ranking lower is better:")
        for score, path in sorted(ranked):
            print(f"{score:.4f} {path}")


if __name__ == "__main__":
    main()
