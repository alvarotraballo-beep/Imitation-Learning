#!/usr/bin/env python3
"""Inspect runtime gripper joint, actuator, and pad geometry limits."""

import numpy as np

from generate_scripted_lift_dataset import make_env
from scripted_joint_ik_lift_smoke import get_pad_geom_names, set_ctrl


def main():
    env = make_env()
    env.reset()
    model = env.sim.model
    data = env.sim.data
    print("nu", model.nu, "nq", model.nq, "njnt", model.njnt)
    print("actuator ctrlrange")
    for i in range(model.nu):
        name = model.actuator_id2name(i)
        print(i, name, np.round(model.actuator_ctrlrange[i], 6))
    print("joints")
    for i in range(model.njnt):
        name = model.joint_id2name(i)
        adr = int(model.jnt_qposadr[i])
        print(i, name, "qposadr", adr, "range", np.round(model.jnt_range[i], 6), "qpos", round(float(data.qpos[adr]), 6))
    pads = get_pad_geom_names(env)
    for label, close in [("open_command", False), ("close_command", True)]:
        set_ctrl(env, data.qpos[:6].copy(), close)
        for _ in range(100):
            env.sim.step()
        pad_pos = [data.geom_xpos[model.geom_name2id(name)].copy() for name in pads]
        sep = pad_pos[1] - pad_pos[0]
        gq = []
        for i in range(6, min(8, model.njnt)):
            adr = int(model.jnt_qposadr[i])
            gq.append(float(data.qpos[adr]))
        print(label, "ctrl", np.round(data.ctrl[6:], 6), "gq", np.round(gq, 6), "sep", np.round(sep, 5), "width", round(float(np.linalg.norm(sep)), 5))


if __name__ == "__main__":
    main()
