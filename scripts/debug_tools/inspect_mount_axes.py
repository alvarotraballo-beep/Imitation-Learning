#!/usr/bin/env python3
"""Inspect local base axes against the initial extended JAKA direction."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_JAKA_RS = PROJECT_ROOT / "jaka_robosuite_integration_package" / "jaka_robosuite_integration" / "python"
if LOCAL_JAKA_RS.exists():
    sys.path.insert(0, str(LOCAL_JAKA_RS))

from generate_scripted_lift_dataset import make_env
from scripted_joint_ik_lift_smoke import get_site_pos


INITIAL_Q = np.array([-0.0007, 1.5797, -0.0322, 0.001, 0.0329, 0.0193], dtype=np.float64)


def find_body_id(env, suffix):
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i) or ""
        if name.endswith(suffix):
            return i, name
    raise RuntimeError(f"No body ending with {suffix}")


def main():
    env = make_env()
    env.reset()
    env.sim.data.qpos[:6] = INITIAL_Q
    env.sim.data.qvel[:6] = 0.0
    env.sim.forward()

    base_id, base_name = find_body_id(env, "jaka_base")
    link1_id, link1_name = find_body_id(env, "link1")
    link2_id, link2_name = find_body_id(env, "link2")
    eef = get_site_pos(env)
    base = env.sim.data.xpos[base_id].copy()
    link1 = env.sim.data.xpos[link1_id].copy()
    link2 = env.sim.data.xpos[link2_id].copy()
    arm_dir = eef - base
    arm_dir = arm_dir / np.linalg.norm(arm_dir)
    shoulder_dir = link2 - link1
    shoulder_dir = shoulder_dir / np.linalg.norm(shoulder_dir)
    rot = env.sim.data.xmat[base_id].reshape(3, 3).copy()

    print("base:", base_name, np.round(base, 5))
    print("link1:", link1_name, np.round(link1, 5))
    print("link2:", link2_name, np.round(link2, 5))
    print("eef:", np.round(eef, 5))
    print("base_to_eef_unit:", np.round(arm_dir, 5))
    print("link1_to_link2_unit:", np.round(shoulder_dir, 5))
    for axis_idx, axis_name in enumerate(["local_x", "local_y", "local_z"]):
        axis = rot[:, axis_idx]
        print(
            axis_name,
            "world=",
            np.round(axis, 5),
            "dot_base_to_eef=",
            round(float(np.dot(axis, arm_dir)), 5),
            "dot_link1_to_link2=",
            round(float(np.dot(axis, shoulder_dir)), 5),
        )


if __name__ == "__main__":
    main()
