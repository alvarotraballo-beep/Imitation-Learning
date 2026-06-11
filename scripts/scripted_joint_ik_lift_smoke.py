#!/usr/bin/env python3
"""Scripted lift attempt using numerical IK and direct MuJoCo joint position controls."""

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_JAKA_RS = PROJECT_ROOT / "jaka_robosuite_integration_package" / "jaka_robosuite_integration" / "python"
if LOCAL_JAKA_RS.exists():
    sys.path.insert(0, str(LOCAL_JAKA_RS))

import robosuite as suite  # noqa: E402
from robosuite.controllers import load_composite_controller_config  # noqa: E402

from robosuite.models.robots.manipulators.jaka_minicobo_robot import JakaMiniCobo  # noqa: E402,F401
from robosuite.models.grippers.pge50_26_gripper import PGE50_26  # noqa: E402,F401


class VideoRecorder:
    def __init__(self, env, path, camera="frontview", width=960, height=720, fps=20):
        self.env = env
        self.path = Path(path)
        self.camera = camera
        self.width = width
        self.height = height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=fps)
        self.frames = 0

    def capture(self):
        frame = self.env.sim.render(
            camera_name=self.camera,
            width=self.width,
            height=self.height,
            depth=False,
        )
        self.writer.append_data(np.asarray(frame)[::-1])
        self.frames += 1

    def close(self):
        self.writer.close()
        print("saved video:", self.path, "frames:", self.frames)


def make_controller():
    config = load_composite_controller_config(controller="BASIC")
    part = config.get("body_parts", {}).get("right", config)
    part.update({"type": "JOINT_POSITION", "control_delta": True})
    return config


def cube_pos_from_obs(obs):
    if "cube_pos" in obs:
        return np.asarray(obs["cube_pos"], dtype=np.float64)
    if "object-state" in obs:
        return np.asarray(obs["object-state"][:3], dtype=np.float64)
    if "object" in obs:
        return np.asarray(obs["object"][:3], dtype=np.float64)
    raise KeyError(f"Cannot infer cube position from obs keys: {sorted(obs.keys())}")


def cube_geom_pos(env):
    try:
        return env.sim.data.geom_xpos[env.sim.model.geom_name2id("cube_g0")].copy()
    except Exception:
        return None


def set_cube_pos(env, cube_pos):
    qpos = np.zeros(7, dtype=np.float64)
    qpos[:3] = cube_pos
    qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for joint_name in ("cube_joint0", "cube_main_joint0"):
        try:
            env.sim.data.set_joint_qpos(joint_name, qpos)
            env.sim.forward()
            print("set cube joint:", joint_name, "pos:", np.round(cube_pos, 4))
            return
        except Exception:
            pass
    raise RuntimeError("Could not set cube pose; expected cube_joint0 or cube_main_joint0")


def cube_geom_ids(env):
    ids = []
    for name in ("cube_g0", "cube_g0_vis"):
        try:
            ids.append(env.sim.model.geom_name2id(name))
        except Exception:
            pass
    if not ids:
        raise RuntimeError("Could not find cube geoms")
    return ids


def set_cube_appearance(env, half_size=None, rgba=None):
    model = env.sim.model
    for gid in cube_geom_ids(env):
        if half_size is not None:
            model.geom_size[gid, :3] = float(half_size)
        if rgba is not None:
            model.geom_rgba[gid, :] = np.asarray(rgba, dtype=np.float64)
    env.sim.forward()


def get_site_pos(env, site_name="robot0_grip_site"):
    site_id = env.sim.model.site_name2id(site_name)
    return env.sim.data.site_xpos[site_id].copy()


def get_pad_geom_names(env):
    names = []
    for candidate in (
        "gripper0_right_finger1_pad_collision",
        "gripper0_right_finger2_pad_collision",
        "gripper0_right_left_fingerpad_collision",
        "gripper0_right_right_fingerpad_collision",
    ):
        try:
            env.sim.model.geom_name2id(candidate)
            names.append(candidate)
        except Exception:
            pass
    if len(names) < 2:
        raise RuntimeError(f"Could not find two gripper pad geoms. Found: {names}")
    return names[:2]


def get_pad_mid_pos(env):
    pad_a, pad_b = get_pad_geom_names(env)
    model = env.sim.model
    data = env.sim.data
    a = data.geom_xpos[model.geom_name2id(pad_a)].copy()
    b = data.geom_xpos[model.geom_name2id(pad_b)].copy()
    return 0.5 * (a + b)


def forward_q(env, q, feature="eef"):
    data = env.sim.data
    old_qpos = data.qpos.copy()
    old_qvel = data.qvel.copy()
    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    env.sim.forward()
    if feature == "pad_mid":
        pos = get_pad_mid_pos(env)
    else:
        pos = get_site_pos(env)
    data.qpos[:] = old_qpos
    data.qvel[:] = old_qvel
    env.sim.forward()
    return pos


def solve_ik_position(env, target_pos, q_init, iters=260, lr=0.22, damping=1e-2, feature="eef", wrist_q6=None):
    q = q_init.copy().astype(np.float64)
    q_min = np.array([-6.28, -2.09, -2.27, -6.28, -2.09, -6.28], dtype=np.float64)
    q_max = np.array([6.28, 2.09, 2.27, 6.28, 2.09, 6.28], dtype=np.float64)
    active_joints = range(6)
    if wrist_q6 is not None:
        q[5] = wrist_q6
        active_joints = range(5)

    for _ in range(iters):
        pos = forward_q(env, q, feature=feature)
        err = target_pos - pos
        if np.linalg.norm(err) < 0.01:
            break

        jac = np.zeros((3, 6), dtype=np.float64)
        eps = 1e-4
        for j in active_joints:
            q2 = q.copy()
            q2[j] += eps
            jac[:, j] = (forward_q(env, q2, feature=feature) - pos) / eps

        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), err)
        q = np.clip(q + lr * dq, q_min, q_max)
        if wrist_q6 is not None:
            q[5] = wrist_q6

    pos = forward_q(env, q, feature=feature)
    return q.astype(np.float32), pos.astype(np.float32), float(np.linalg.norm(target_pos - pos))


def solve_best_ik(env, target, q_seed, feature="eef", wrist_q6=None):
    seeds = [q_seed.copy()]
    for j1 in np.linspace(-3.0, 3.0, 7):
        q = q_seed.copy()
        q[0] = j1
        seeds.append(q)
    for j1 in np.linspace(-2.5, 2.5, 5):
        for j2 in np.linspace(-1.6, 1.6, 5):
            for j3 in np.linspace(-1.8, 1.8, 5):
                q = q_seed.copy()
                q[:3] = [j1, j2, j3]
                seeds.append(q)

    best = None
    for seed in seeds:
        if wrist_q6 is not None:
            seed = seed.copy()
            seed[5] = wrist_q6
        q, eef, dist = solve_ik_position(env, target, seed, feature=feature, wrist_q6=wrist_q6)
        score = dist + 0.01 * np.linalg.norm(q - q_seed)
        if best is None or score < best[0]:
            best = (score, dist, q, eef)
    _, dist, q, eef = best
    return q, eef, dist


def gripper_targets(env, close):
    ctrlrange = env.sim.model.actuator_ctrlrange
    if env.sim.model.nu <= 6:
        return np.array([], dtype=np.float32)
    gripper_range = ctrlrange[6:]
    lo = gripper_range[:, 0]
    hi = gripper_range[:, 1]
    has_positive = np.any(hi > 0.0)
    has_negative = np.any(lo < 0.0)
    if has_positive and has_negative:
        # Panda-style opposed finger joints: open means each actuator moves away
        # from zero, close means both return toward zero.
        open_target = np.where(np.abs(hi) >= np.abs(lo), hi, lo)
        close_target = np.where(np.abs(hi) < np.abs(lo), hi, lo)
    else:
        # PGE-style same-sign sliders in this project use qpos=0 open.
        open_target = lo
        close_target = hi
    return (close_target if close else open_target).astype(np.float32)


def set_ctrl(env, q, close):
    env.sim.data.ctrl[:6] = q
    g = gripper_targets(env, close=close)
    if len(g):
        env.sim.data.ctrl[6 : 6 + len(g)] = g


def boost_arm_actuators(env, kp=1200.0, force=2500.0, damping=8.0):
    model = env.sim.model
    for i in range(min(6, model.nu)):
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_forcerange[i, 0] = -force
        model.actuator_forcerange[i, 1] = force
    for j in range(min(6, model.njnt)):
        model.dof_damping[j] = damping


def boost_gripper_grasp(env, kp=1800.0, force=220.0, friction=5.0):
    model = env.sim.model
    for i in range(6, model.nu):
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_forcerange[i, 0] = -force
        model.actuator_forcerange[i, 1] = force
    for name in ["cube_g0", *get_pad_geom_names(env)]:
        try:
            gid = model.geom_name2id(name)
        except Exception:
            continue
        model.geom_friction[gid, 0] = friction
        model.geom_friction[gid, 1] = max(model.geom_friction[gid, 1], 0.1)
        model.geom_friction[gid, 2] = max(model.geom_friction[gid, 2], 0.1)
    print(
        "boosted gripper grasp:",
        "actuators",
        list(range(6, model.nu)),
        "kp",
        kp,
        "force",
        force,
        "friction",
        friction,
    )


def geom_name(env, geom_id):
    model = env.sim.model
    try:
        return model.geom_id2name(int(geom_id))
    except Exception:
        pass
    try:
        return model.id2name(int(geom_id), "geom")
    except Exception:
        return str(int(geom_id))


def body_name(env, body_id):
    model = env.sim.model
    try:
        return model.body_id2name(int(body_id))
    except Exception:
        pass
    try:
        return model.id2name(int(body_id), "body")
    except Exception:
        return str(int(body_id))


def print_contacts(env, prefix="contacts"):
    data = env.sim.data
    if data.ncon == 0:
        print(f"{prefix}: none")
        return
    pairs = []
    for i in range(min(data.ncon, 12)):
        c = data.contact[i]
        pairs.append(f"{geom_name(env, c.geom1)} <-> {geom_name(env, c.geom2)}")
    print(f"{prefix}: ncon={data.ncon} " + " | ".join(pairs))


def print_grasp_geometry(env, prefix="geom"):
    model = env.sim.model
    data = env.sim.data
    names = [
        "cube_g0",
        "gripper0_right_hand_collision",
        "gripper0_right_finger1_pad_collision",
        "gripper0_right_finger2_pad_collision",
        "gripper0_right_left_fingerpad_collision",
        "gripper0_right_right_fingerpad_collision",
    ]
    entries = []
    for name in names:
        try:
            gid = model.geom_name2id(name)
            entries.append(f"{name}={np.round(data.geom_xpos[gid], 4)}")
        except Exception:
            pass
    print(f"{prefix}: " + " | ".join(entries))


def disable_arm_collisions(env):
    model = env.sim.model
    disabled = []
    for i in range(model.ngeom):
        name = geom_name(env, i) or ""
        body = body_name(env, model.geom_bodyid[i]) or ""
        is_arm_geom = name.startswith("robot0_link") or body.startswith("robot0_link") or body == "robot0_jaka_base"
        if is_arm_geom:
            lname = name.lower()
            if "finger" not in lname and "gripper" not in lname:
                model.geom_contype[i] = 0
                model.geom_conaffinity[i] = 0
                disabled.append(name)
    print("disabled arm collision geoms:", disabled)


def disable_gripper_nonpad_collisions(env):
    model = env.sim.model
    disabled = []
    for i in range(model.ngeom):
        name = geom_name(env, i) or ""
        body = body_name(env, model.geom_bodyid[i]) or ""
        lname = name.lower()
        lbody = body.lower()
        is_gripper_nonpad = (
            "gripper" in lname
            and ("hand" in lname or "hand" in lbody or "right_gripper" in lbody)
            and "finger" not in lname
            and "pad" not in lname
        )
        is_finger_nonpad = "gripper" in lname and "finger" in lname and "pad" not in lname
        if is_gripper_nonpad or is_finger_nonpad:
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0
            disabled.append(name)
    print("disabled gripper non-pad collision geoms:", disabled)


def step_to_q(env, q_target, close, steps, label, substeps, max_joint_step, contacts, debug_grasp_geometry, recorder):
    print(f"\nphase={label} close={close} q_target={np.round(q_target, 4)}")
    q_cmd = env.sim.data.qpos[:6].copy()
    for i in range(steps):
        q_now = env.sim.data.qpos[:6].copy()
        q_cmd = q_now + np.clip(q_target - q_now, -max_joint_step, max_joint_step)
        set_ctrl(env, q_cmd, close)
        for _ in range(substeps):
            env.sim.step()
        if recorder is not None:
            recorder.capture()
        if i % 20 == 0 or i == steps - 1:
            q = env.sim.data.qpos[:6].copy()
            eef = get_site_pos(env)
            grip = env.sim.data.qpos[6 : min(len(env.sim.data.qpos), 12)].copy()
            q = env.sim.data.qpos[:6].copy()
            ctrl = env.sim.data.ctrl[:6].copy()
            qfrc = env.sim.data.qfrc_actuator[:6].copy()
            print(
                f"  {i:03d} q_err={np.linalg.norm(q - q_target):.4f} "
                f"q={np.round(q, 4)} q_cmd={np.round(q_cmd, 4)} "
                f"ctrl={np.round(ctrl, 4)} qfrc={np.round(qfrc, 2)} "
                f"eef={np.round(eef, 4)} grip={np.round(grip, 5)}"
            )
            if contacts:
                print_contacts(env, prefix="       contacts")
            if debug_grasp_geometry:
                print_grasp_geometry(env, prefix="       geom")


def execute_q_path(env, q_path, close_path, label_path, substeps, max_joint_step, contacts, debug_grasp_geometry, path_print_every, recorder):
    for idx, (q_target, close, label) in enumerate(zip(q_path, close_path, label_path)):
        steps = 8
        if label == "close":
            steps = 24
        elif label == "lift":
            steps = 12
        if idx == 0:
            print(f"\npath start: {label}")
        should_print_periodic = path_print_every > 0 and idx % path_print_every == 0
        if should_print_periodic or idx == len(q_path) - 1 or label != label_path[idx - 1]:
            eef = get_site_pos(env)
            q_err = float(np.linalg.norm(env.sim.data.qpos[:6].copy() - q_target))
            try:
                pad_mid = get_pad_mid_pos(env)
                pad_text = f" pad_mid={np.round(pad_mid, 4)}"
            except Exception:
                pad_text = ""
            print(
                f"path idx={idx:03d}/{len(q_path)-1:03d} label={label} "
                f"q_err={q_err:.4f} eef={np.round(eef, 4)}{pad_text} "
                f"target_q={np.round(q_target, 4)} close={close}"
            )
        q_cmd = env.sim.data.qpos[:6].copy()
        for _ in range(steps):
            q_now = env.sim.data.qpos[:6].copy()
            q_cmd = q_now + np.clip(q_target - q_now, -max_joint_step, max_joint_step)
            set_ctrl(env, q_cmd, close)
            for _ in range(substeps):
                env.sim.step()
            if recorder is not None:
                recorder.capture()
        if contacts and (should_print_periodic or idx == len(q_path) - 1):
            print_contacts(env, prefix="       contacts")
        if debug_grasp_geometry and (should_print_periodic or idx == len(q_path) - 1):
            print_grasp_geometry(env, prefix="       geom")


def build_cartesian_q_path(env, waypoints, q_seed, points_per_meter=120, feature="eef", wrist_q6=None):
    q_path = []
    close_path = []
    label_path = []
    current_pos = get_pad_mid_pos(env) if feature == "pad_mid" else get_site_pos(env)
    current_q = q_seed.copy()
    print("\n--- Cartesian IK path ---")
    print("cartesian feature:", feature, "wrist_q6:", wrist_q6)
    for label, target, close, _steps in waypoints:
        dist = float(np.linalg.norm(target - current_pos))
        n = max(4, int(np.ceil(dist * points_per_meter)))
        for i in range(1, n + 1):
            alpha = i / n
            subtarget = (1.0 - alpha) * current_pos + alpha * target
            q, eef, err = solve_ik_position(
                env,
                subtarget,
                current_q,
                iters=80,
                lr=0.18,
                damping=1e-2,
                feature=feature,
                wrist_q6=wrist_q6,
            )
            if err > 0.04:
                q_best, eef_best, err_best = solve_best_ik(
                    env,
                    subtarget,
                    current_q,
                    feature=feature,
                    wrist_q6=wrist_q6,
                )
                if err_best < err:
                    q, eef, err = q_best, eef_best, err_best
            if err > 0.04:
                print(f"WARNING path ik err label={label} i={i}/{n} err={err:.4f} target={np.round(subtarget, 4)} eef={np.round(eef, 4)}")
            q_path.append(q)
            close_path.append(close)
            label_path.append(label)
            current_q = q
        current_pos = target.copy()
        print(f"{label:8s} n={n:03d} target={np.round(target, 4)} q={np.round(current_q, 4)}")
    return q_path, close_path, label_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Lift")
    parser.add_argument("--robot", default="JakaMiniCobo")
    parser.add_argument("--gripper", default="PandaGripper")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--manual-cube", default=None, help="Optional x,y,z cube target override")
    parser.add_argument("--initial-q", default=None, help="Optional comma-separated 6 arm qpos values after reset")
    parser.add_argument("--substeps", type=int, default=0, help="Physics steps per control command; 0 uses control_freq")
    parser.add_argument("--max-joint-step", type=float, default=0.04, help="Max radian change per commanded control step")
    parser.add_argument("--boost-actuators", action="store_true")
    parser.add_argument("--boost-kp", type=float, default=1200.0)
    parser.add_argument("--boost-force", type=float, default=2500.0)
    parser.add_argument("--boost-damping", type=float, default=8.0)
    parser.add_argument("--boost-gripper-grasp", action="store_true")
    parser.add_argument("--gripper-boost-kp", type=float, default=1800.0)
    parser.add_argument("--gripper-boost-force", type=float, default=220.0)
    parser.add_argument("--grasp-friction", type=float, default=5.0)
    parser.add_argument("--disable-arm-collisions", action="store_true")
    parser.add_argument("--disable-gripper-base-collisions", action="store_true")
    parser.add_argument("--disable-gripper-nonpad-collisions", action="store_true")
    parser.add_argument("--contacts", action="store_true")
    parser.add_argument("--debug-grasp-geometry", action="store_true")
    parser.add_argument("--cartesian-path", action="store_true")
    parser.add_argument("--path-points-per-meter", type=float, default=120.0)
    parser.add_argument("--pad-mid-ik", action="store_true", help="Target the physical midpoint between gripper pads instead of the eef site")
    parser.add_argument("--wrist-q6", type=float, default=None, help="Optional fixed joint_6 value during IK, useful to keep Panda pads horizontal")
    parser.add_argument("--pad-grasp-z-offset", type=float, default=0.0, help="Extra z offset for pad-mid grasp / close targets")
    parser.add_argument("--path-print-every", type=int, default=10, help="Print every N cartesian path points; 0 prints only transitions and final point")
    parser.add_argument("--video-path", default=None, help="Optional mp4 path to save an offscreen render")
    parser.add_argument("--video-camera", default="frontview")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    args = parser.parse_args()

    env = suite.make(
        env_name=args.env_name,
        robots=args.robot,
        gripper_types=args.gripper,
        controller_configs=make_controller(),
        has_renderer=args.render,
        has_offscreen_renderer=args.video_path is not None,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=500,
        ignore_done=True,
    )
    obs = env.reset()
    if args.manual_cube:
        set_cube_pos(env, np.array([float(v) for v in args.manual_cube.split(",")], dtype=np.float64))
        obs = env._get_observations()
    if args.initial_q:
        q_init = np.array([float(v) for v in args.initial_q.split(",")], dtype=np.float64)
        if q_init.shape != (6,):
            raise ValueError("--initial-q must contain exactly 6 comma-separated values")
        env.sim.data.qpos[:6] = q_init
        env.sim.data.qvel[:6] = 0.0
        env.sim.forward()
        print("set initial q:", np.round(q_init, 4))
    recorder = None
    if args.video_path is not None:
        recorder = VideoRecorder(
            env,
            args.video_path,
            camera=args.video_camera,
            width=args.video_width,
            height=args.video_height,
            fps=20,
        )
        recorder.capture()
    if args.disable_arm_collisions:
        disable_arm_collisions(env)
    if args.disable_gripper_base_collisions or args.disable_gripper_nonpad_collisions:
        disable_gripper_nonpad_collisions(env)
    if args.boost_actuators:
        boost_arm_actuators(env, kp=args.boost_kp, force=args.boost_force, damping=args.boost_damping)
    if args.boost_gripper_grasp:
        boost_gripper_grasp(
            env,
            kp=args.gripper_boost_kp,
            force=args.gripper_boost_force,
            friction=args.grasp_friction,
        )
    q0 = env.sim.data.qpos[:6].copy()
    cube = cube_geom_pos(env)
    if cube is None:
        cube = cube_pos_from_obs(obs)

    print("created env:", env)
    print("robot:", args.robot, "gripper:", args.gripper, "nu:", env.sim.model.nu, "qpos:", len(env.sim.data.qpos))
    print("initial q:", np.round(q0, 4))
    print("initial eef:", np.round(get_site_pos(env), 4), "cube target:", np.round(cube, 4))
    try:
        print("initial pad_mid:", np.round(get_pad_mid_pos(env), 4), "pad geoms:", get_pad_geom_names(env))
    except Exception as exc:
        print("initial pad_mid: unavailable", exc)
    print("actuator ctrlrange:", np.round(env.sim.model.actuator_ctrlrange, 5))
    print("actuator forcerange:", np.round(env.sim.model.actuator_forcerange[:8], 2))
    print("actuator gain:", np.round(env.sim.model.actuator_gainprm[:8, 0], 2))
    print("dof damping:", np.round(env.sim.model.dof_damping[:6], 2))
    timestep = float(env.sim.model.opt.timestep)
    substeps = args.substeps or max(1, int(round(1.0 / (20 * timestep))))
    print("mujoco timestep:", timestep, "substeps/control:", substeps, "max_joint_step:", args.max_joint_step)

    ik_feature = "pad_mid" if args.pad_mid_ik else "eef"
    if args.pad_mid_ik:
        waypoints = [
            ("above", cube + np.array([0.0, 0.0, 0.18]), False, 90),
            ("pregrasp", cube + np.array([0.0, 0.0, 0.055]), False, 70),
            ("grasp", cube + np.array([0.0, 0.0, args.pad_grasp_z_offset]), False, 50),
            ("close", cube + np.array([0.0, 0.0, args.pad_grasp_z_offset]), True, 80),
            ("lift", cube + np.array([0.0, 0.0, 0.26]), True, 140),
        ]
    else:
        waypoints = [
            ("above", cube + np.array([0.0, 0.0, 0.18]), False, 90),
            ("pregrasp", cube + np.array([0.0, 0.0, 0.065]), False, 70),
            ("grasp", cube + np.array([0.0, 0.0, 0.035]), False, 50),
            ("close", cube + np.array([0.0, 0.0, 0.035]), True, 60),
            ("lift", cube + np.array([0.0, 0.0, 0.28]), True, 120),
        ]

    q_seed = q0
    solved = []
    print("\n--- IK targets ---")
    print("ik feature:", ik_feature, "wrist_q6:", args.wrist_q6)
    for label, target, close, steps in waypoints:
        q, eef, dist = solve_best_ik(env, target, q_seed, feature=ik_feature, wrist_q6=args.wrist_q6)
        print(f"{label:8s} target={np.round(target, 4)} {ik_feature}={np.round(eef, 4)} dist={dist:.4f} q={np.round(q, 4)}")
        solved.append((label, q, close, steps, dist))
        q_seed = q

    if any(dist > 0.08 for _, _, _, _, dist in solved[:3]):
        print("\nWARNING: at least one approach/grasp target is not reachable within 8 cm. Check base pose or cube placement.")

    if args.cartesian_path:
        q_path, close_path, label_path = build_cartesian_q_path(
            env,
            waypoints,
            q_seed=q0,
            points_per_meter=args.path_points_per_meter,
            feature=ik_feature,
            wrist_q6=args.wrist_q6,
        )
        execute_q_path(
            env,
            q_path,
            close_path,
            label_path,
            substeps=substeps,
            max_joint_step=args.max_joint_step,
            contacts=args.contacts,
            debug_grasp_geometry=args.debug_grasp_geometry,
            path_print_every=args.path_print_every,
            recorder=recorder,
        )
    else:
        for label, q, close, steps, _ in solved:
            step_to_q(
                env,
                q,
                close,
                steps,
                label,
                substeps=substeps,
                max_joint_step=args.max_joint_step,
                contacts=args.contacts,
                debug_grasp_geometry=args.debug_grasp_geometry,
                recorder=recorder,
            )
            if args.render:
                env.render()

    try:
        success = env._check_success()
        print("\nfinal success:", success)
    except Exception:
        success = None
    final_cube = cube_geom_pos(env)
    if final_cube is None:
        final_cube = cube_pos_from_obs(env._get_observations())
    print("final eef:", np.round(get_site_pos(env), 4), "cube:", np.round(final_cube, 4))
    if recorder is not None:
        recorder.close()


if __name__ == "__main__":
    main()
