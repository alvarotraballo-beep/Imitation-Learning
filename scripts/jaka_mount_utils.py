"""Utilities for changing the simulated JAKA mounting orientation."""

import math

import numpy as np


def quat_z(theta):
    return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)], dtype=np.float64)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def normalize_quat(q):
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion")
    return q / norm


def quat_from_mat(mat):
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return normalize_quat(
            np.array(
                [
                    0.25 * s,
                    (mat[2, 1] - mat[1, 2]) / s,
                    (mat[0, 2] - mat[2, 0]) / s,
                    (mat[1, 0] - mat[0, 1]) / s,
                ],
                dtype=np.float64,
            )
        )
    axis = int(np.argmax(np.diag(mat)))
    if axis == 0:
        s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        quat = np.array(
            [
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif axis == 1:
        s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        quat = np.array(
            [
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        quat = np.array(
            [
                (mat[1, 0] - mat[0, 1]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
                (mat[1, 2] + mat[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return normalize_quat(quat)


def find_body_id(env, suffix):
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i) or ""
        if name.endswith(suffix):
            return i, name
    raise RuntimeError(f"No body ending with {suffix}")


def apply_base_longitudinal_twist(env, twist_deg):
    """Roll the base around its own local Z axis, which is the extended arm axis.

    In the robosuite-mounted JAKA scene, the base local Z axis points along the
    original horizontal arm direction. Post-multiplying preserves that direction
    and only changes the mounting roll, like tightening or loosening a screw.
    """
    if abs(twist_deg) < 1e-9:
        return None
    body_id, name = find_body_id(env, "jaka_base")
    base_quat = env.sim.model.body_quat[body_id].copy()
    twist_quat = quat_z(math.radians(twist_deg))
    env.sim.model.body_quat[body_id] = normalize_quat(quat_mul(base_quat, twist_quat))
    env.sim.forward()
    print("applied base longitudinal twist:", twist_deg, "deg about local_z to body:", name)
    return name


def reject_global_yaw(yaw_deg):
    if abs(yaw_deg) > 1e-9:
        raise ValueError(
            "--base-yaw-deg rotated the whole mounting in the world and is deprecated here; "
            "use --base-twist-deg for the longitudinal screw-like base rotation."
        )


def style_vertical_base_support(env, floor_z=0.0, top_z=1.08):
    """Turn the existing support placeholder into a centered vertical stand."""
    model = env.sim.model
    data = env.sim.data
    try:
        geom_id = model.geom_name2id("robot0_vertical_torso_support")
    except Exception:
        return False
    body_id = int(model.geom_bodyid[geom_id])
    body_pos = data.xpos[body_id].copy()
    body_rot = data.xmat[body_id].reshape(3, 3).copy()

    height = max(0.05, float(top_z - floor_z))
    center_world = np.array([body_pos[0], body_pos[1] + 0.06, floor_z + 0.5 * height], dtype=np.float64)
    model.geom_pos[geom_id] = body_rot.T @ (center_world - body_pos)
    model.geom_quat[geom_id] = quat_from_mat(body_rot.T)
    model.geom_size[geom_id] = np.array([0.11, 0.11, 0.5 * height], dtype=np.float64)
    model.geom_rgba[geom_id] = np.array([0.20, 0.22, 0.24, 1.0], dtype=np.float64)
    model.geom_group[geom_id] = 1
    model.geom_contype[geom_id] = 0
    model.geom_conaffinity[geom_id] = 0
    env.sim.forward()
    print("styled vertical base support:", "center", np.round(center_world, 4), "height", round(height, 4))
    return True
