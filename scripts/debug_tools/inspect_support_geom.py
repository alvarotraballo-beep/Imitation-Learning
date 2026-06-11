#!/usr/bin/env python3
"""Inspect the visual support geom in the instantiated JAKA scene."""

import numpy as np

from generate_scripted_lift_dataset import make_env


def main():
    env = make_env()
    env.reset()
    model = env.sim.model
    data = env.sim.data
    name = "robot0_vertical_torso_support"
    gid = model.geom_name2id(name)
    bid = int(model.geom_bodyid[gid])
    print("geom", name)
    print("body", bid, model.body_id2name(bid))
    print("type", int(model.geom_type[gid]))
    print("pos_local", np.round(model.geom_pos[gid], 5))
    print("size", np.round(model.geom_size[gid], 5))
    print("rgba", np.round(model.geom_rgba[gid], 5))
    print("xpos", np.round(data.geom_xpos[gid], 5))
    print("xmat", np.round(data.geom_xmat[gid].reshape(3, 3), 5))


if __name__ == "__main__":
    main()
