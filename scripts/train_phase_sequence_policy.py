#!/usr/bin/env python3
"""Fit a phase-indexed direct-control policy from successful scripted demos."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_demo_reference(demo):
    obs = demo["obs"]
    progress = obs["progress"][()].reshape(-1).astype(np.float64)
    q_now = obs["robot0_joint_pos"][()].astype(np.float64)
    actions = demo["actions"][()].astype(np.float64)
    q_ref = q_now + actions[:, :6]
    gripper_close = (np.mean(actions[:, 6:8], axis=1) > 0.0).astype(np.float64)
    cube_pos = obs["object"][0, :3].astype(np.float64)
    cube_size = obs["cube_size"][0].astype(np.float64) if "cube_size" in obs else np.array([0.020], dtype=np.float64)
    return progress, q_ref, gripper_close, np.concatenate([cube_pos, cube_size], axis=0)


def resample_demo(progress, q_ref, gripper_close, grid):
    order = np.argsort(progress)
    progress = progress[order]
    q_ref = q_ref[order]
    gripper_close = gripper_close[order]

    unique_progress, unique_idx = np.unique(progress, return_index=True)
    q_ref = q_ref[unique_idx]
    gripper_close = gripper_close[unique_idx]

    q_grid = np.stack(
        [np.interp(grid, unique_progress, q_ref[:, j]) for j in range(q_ref.shape[1])],
        axis=1,
    )
    close_grid = np.interp(grid, unique_progress, gripper_close)
    return q_grid, close_grid


def smooth_q_ref(q_ref, window):
    if window <= 1:
        return q_ref
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(q_ref, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.stack(
        [np.convolve(padded[:, j], kernel, mode="valid") for j in range(q_ref.shape[1])],
        axis=1,
    )
    smoothed[0] = q_ref[0]
    smoothed[-1] = q_ref[-1]
    return smoothed


def fit_conditioned_q_ref(q_refs, features, ridge=1e-4):
    y = np.stack(q_refs, axis=0).astype(np.float64)
    x = np.concatenate([np.ones((len(features), 1), dtype=np.float64), np.stack(features, axis=0)], axis=1)
    xtx = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
    xty = np.einsum("df,dnj->fnj", x, y)
    coef = np.linalg.solve(xtx, xty.reshape(x.shape[1], -1)).reshape(x.shape[1], y.shape[1], y.shape[2])
    feature_mean = np.mean(np.stack(features, axis=0), axis=0)
    q_ref = np.einsum("f,fnj->nj", np.r_[1.0, feature_mean], coef)
    return q_ref.astype(np.float32), coef.astype(np.float32), feature_mean.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/assets/scripted_lift_delta_progress_4demo.hdf5")
    parser.add_argument("--output", default="bc_trained_models/phase_sequence_policy.npz")
    parser.add_argument("--num-phases", type=int, default=2600)
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--demo-key", default=None)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--conditioning", choices=["none", "cube_linear"], default="none")
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()

    grid = np.linspace(0.0, 1.0, args.num_phases, dtype=np.float64)
    q_refs = []
    close_refs = []
    features = []
    demo_names = []

    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        script_args = data.attrs.get("script_args", "{}")
        for demo_key in sorted(data.keys()):
            if args.demo_key is not None and demo_key != args.demo_key:
                continue
            demo = data[demo_key]
            if bool(demo.attrs.get("success", False)) is False:
                continue
            progress, q_ref, gripper_close, feature = load_demo_reference(demo)
            q_grid, close_grid = resample_demo(progress, q_ref, gripper_close, grid)
            q_refs.append(q_grid)
            close_refs.append(close_grid)
            features.append(feature)
            demo_names.append(demo_key)

    if not q_refs:
        raise RuntimeError("No successful demos found in dataset")

    q_coef = None
    feature_mean = None
    if args.conditioning == "cube_linear":
        q_ref, q_coef, feature_mean = fit_conditioned_q_ref(q_refs, features, ridge=args.ridge)
        if args.smooth_window > 1:
            q_coef = np.stack([smooth_q_ref(q_coef[i].astype(np.float64), args.smooth_window) for i in range(q_coef.shape[0])], axis=0).astype(np.float32)
            q_ref = np.einsum("f,fnj->nj", np.r_[1.0, feature_mean], q_coef).astype(np.float32)
    else:
        q_ref = np.mean(np.stack(q_refs, axis=0), axis=0).astype(np.float32)
        q_ref = smooth_q_ref(q_ref.astype(np.float64), args.smooth_window).astype(np.float32)
    close_score = np.mean(np.stack(close_refs, axis=0), axis=0)
    close_ref = (close_score >= args.close_threshold).astype(np.bool_)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        progress_grid=grid.astype(np.float32),
        q_ref=q_ref,
        close_ref=close_ref,
        close_score=close_score.astype(np.float32),
        conditioning=np.asarray(args.conditioning),
        q_coef=q_coef if q_coef is not None else np.zeros((0,), dtype=np.float32),
        feature_mean=feature_mean if feature_mean is not None else np.zeros((0,), dtype=np.float32),
        feature_names=np.asarray(["cube_x", "cube_y", "cube_z", "cube_half_size"]),
        dataset=str(args.dataset),
        demo_names=np.asarray(demo_names),
        script_args=str(script_args),
    )
    first_close = int(np.argmax(close_ref)) if np.any(close_ref) else -1
    print(
        "saved:",
        output,
        "successful_demos:",
        len(q_refs),
        "phases:",
        len(grid),
        "first_close_phase:",
        first_close,
    )


if __name__ == "__main__":
    main()
