#!/usr/bin/env python3
"""Continuous demonstration runner for the final JAKA lift policy.

Default mode loads the robust BC-MLP .pt policy included in this package.
The previous phase-sequence .npz policy is still supported by passing it with
--policy.

Interactive display controls with --display:
  q / Esc : quit
  n       : skip to next cube
  r       : reset current cube
  p       : pause / resume
  h       : print controls

Browser display with --web-display:
  Open http://127.0.0.1:8765/ from Windows. This avoids cv2.imshow and is
  the recommended mode for demos on WSL / Windows PCs.
"""

import argparse
import http.server
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import imageio.v2 as imageio
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
    from torch import nn
except Exception:  # torch is only required for .pt BC-MLP policies
    torch = None
    nn = None

from eval_phase_sequence_policy import policy_q_reference  # noqa: E402
from generate_scripted_lift_dataset import collect_obs, make_env, object_obs  # noqa: E402
from jaka_mount_utils import (  # noqa: E402
    apply_base_longitudinal_twist,
    find_body_id,
    reject_global_yaw,
    style_vertical_base_support,
)
from scripted_joint_ik_lift_smoke import (  # noqa: E402
    boost_arm_actuators,
    boost_gripper_grasp,
    cube_geom_pos,
    disable_arm_collisions,
    disable_gripper_nonpad_collisions,
    get_pad_mid_pos,
    set_ctrl,
    set_cube_appearance,
    set_cube_pos,
)


DEFAULT_BC_MLP_POLICY = PACKAGE_ROOT / "models" / "bc_mlp" / "bc_mlp_abs_ref_robust_128eps.pt"
DEFAULT_PHASE_POLICY = PACKAGE_ROOT / "models" / "phase_sequence_policy_bag_basetwist30_openfix_variants_allbags_3var_cond_smooth31.npz"
DEFAULT_SCENARIO_FILE = PACKAGE_ROOT / "configs" / "bc_mlp" / "robust_eval_scenarios_8_feasible.json"

DEFAULT_COLORS = [
    ("red", np.array([0.80, 0.05, 0.05, 1.0], dtype=np.float64)),
    ("blue", np.array([0.05, 0.35, 0.90, 1.0], dtype=np.float64)),
    ("green", np.array([0.05, 0.65, 0.25, 1.0], dtype=np.float64)),
    ("yellow", np.array([0.95, 0.72, 0.12, 1.0], dtype=np.float64)),
]


WEB_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>JAKA BC-MLP continuous demo</title>
  <style>
    html, body { margin: 0; min-height: 100%; background: #111; color: #eee; font-family: Arial, sans-serif; }
    .bar { display: flex; gap: 10px; align-items: center; padding: 10px 14px; background: #202020; position: sticky; top: 0; z-index: 2; }
    button { font-size: 15px; padding: 7px 12px; border: 1px solid #666; background: #333; color: #fff; border-radius: 4px; cursor: pointer; }
    button:hover { background: #444; }
    .hint { margin-left: 8px; color: #bbb; font-size: 14px; }
    .stage { display: flex; justify-content: center; padding: 14px; }
    img { max-width: calc(100vw - 28px); max-height: calc(100vh - 76px); object-fit: contain; background: #000; }
  </style>
</head>
<body>
  <div class="bar">
    <strong>JAKA BC-MLP continuous demo</strong>
    <button onclick="cmd('pause')">Pause / Resume</button>
    <button onclick="cmd('next')">Next cube</button>
    <button onclick="cmd('reset')">Reset</button>
    <button onclick="cmd('quit')">Quit</button>
    <span class="hint">Keyboard: p pause, n next, r reset, q quit</span>
  </div>
  <div class="stage"><img src="/stream" /></div>
  <script>
    function cmd(name) { fetch('/cmd?name=' + encodeURIComponent(name)).catch(() => {}); }
    document.addEventListener('keydown', (ev) => {
      const k = ev.key.toLowerCase();
      if (k === 'p') cmd('pause');
      if (k === 'n') cmd('next');
      if (k === 'r') cmd('reset');
      if (k === 'q' || ev.key === 'Escape') cmd('quit');
    });
  </script>
</body>
</html>
"""


class WebDisplayServer:
    def __init__(self, host, port, jpeg_quality=85):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.latest_jpeg = None
        self.command = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.httpd = None
        self.thread = None

    def start(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, _fmt, *_args):
                return

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path in ("/", "/index.html"):
                    body = WEB_PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/cmd":
                    name = parse_qs(parsed.query).get("name", [""])[0]
                    if name in {"pause", "next", "reset", "quit"}:
                        outer.set_command(name)
                    body = b"ok\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/stream":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = None
                    while True:
                        with outer.condition:
                            outer.condition.wait(timeout=1.0)
                            frame = outer.latest_jpeg
                        if frame is None or frame is last:
                            continue
                        last = frame
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    return
                self.send_error(404)

        self.httpd = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        actual_host, actual_port = self.httpd.server_address
        print(f"web_display=http://127.0.0.1:{actual_port}/")
        if actual_host not in ("127.0.0.1", "localhost"):
            print(f"web_display_bound=http://{actual_host}:{actual_port}/")

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    def set_command(self, command):
        with self.lock:
            self.command = command
            self.condition.notify_all()

    def pop_command(self):
        with self.lock:
            command = self.command
            self.command = None
            return command

    def update_frame(self, rgb, bgr=None):
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("--web-display requires OpenCV for JPEG encoding") from exc
        if bgr is None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self.condition:
            self.latest_jpeg = encoded.tobytes()
            self.condition.notify_all()


@dataclass
class Scenario:
    cube: np.ndarray
    half_size: float
    rgba: np.ndarray
    color_name: str
    initial_q: np.ndarray


def parse_float_list(text):
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def parse_vec(text, expected):
    values = parse_float_list(text)
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"Expected {expected} comma-separated values")
    return np.asarray(values, dtype=np.float64)


def parse_range(text, expected=2):
    values = parse_float_list(text)
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"Expected {expected} comma-separated values")
    return values


def parse_color_palette(text):
    if not text:
        return DEFAULT_COLORS
    colors = []
    for idx, item in enumerate(text.split(";")):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, raw = item.split(":", 1)
            name = name.strip()
        else:
            name, raw = f"color_{idx}", item
        vals = parse_float_list(raw)
        if len(vals) == 3:
            vals.append(1.0)
        if len(vals) != 4:
            raise argparse.ArgumentTypeError("Colors must be name:r,g,b,a entries separated by ';'")
        colors.append((name, np.asarray(vals, dtype=np.float64)))
    return colors or DEFAULT_COLORS


def safe_name(text):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "scenario"


def make_activation(name):
    if nn is None:
        raise RuntimeError("PyTorch is required to load BC-MLP .pt policies")
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class AbsRefBCMLP(nn.Module if nn is not None else object):
    """Architecture used by bc_mlp_abs_ref_robust_128eps.pt."""

    def __init__(self, obs_dim, hidden_sizes=(512, 512, 256), activation="silu"):
        if nn is None:
            raise RuntimeError("PyTorch is required to load BC-MLP .pt policies")
        super().__init__()
        layers = []
        last_dim = int(obs_dim)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(hidden)))
            layers.append(make_activation(activation))
            last_dim = int(hidden)
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(last_dim, 6)
        self.gripper_head = nn.Linear(last_dim, 1)

    def forward(self, x):
        z = self.backbone(x)
        return self.q_head(z), self.gripper_head(z).squeeze(-1)


class PhaseSequenceRuntime:
    def __init__(self, path):
        self.path = Path(path)
        self.policy = np.load(str(self.path), allow_pickle=True)
        self.name = self.path.stem

    def begin_episode(self, env, scenario, args):
        q_ref = policy_q_reference(self.policy, cube_geom_pos(env).copy(), scenario.half_size)
        close_ref = self.policy["close_ref"].astype(bool)
        return {"q_ref": q_ref, "close_ref": close_ref}

    def default_horizon(self, args, episode_state):
        return len(episode_state["q_ref"])

    def predict(self, env, scenario, args, step, episode_state):
        phase = min(len(episode_state["q_ref"]) - 1, step)
        q_target = episode_state["q_ref"][phase]
        close = bool(episode_state["close_ref"][phase])
        return q_target, close, 1.0 if close else 0.0, float(phase) / max(1, len(episode_state["q_ref"]) - 1)


class BCMLPRuntime:
    def __init__(self, path, device):
        if torch is None:
            raise RuntimeError("PyTorch is required to load BC-MLP .pt policies")
        self.path = Path(path)
        self.device = torch.device(device)
        self.ckpt = torch.load(str(self.path), map_location=self.device, weights_only=False)
        self.model = AbsRefBCMLP(
            int(self.ckpt["obs_dim"]),
            hidden_sizes=tuple(int(v) for v in self.ckpt["hidden_sizes"]),
            activation=self.ckpt.get("activation", "silu"),
        ).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state"])
        self.model.eval()
        self.tensors = {
            "obs_mean": torch.from_numpy(self.ckpt["obs_mean"].astype(np.float32)).to(self.device),
            "obs_std": torch.from_numpy(self.ckpt["obs_std"].astype(np.float32)).to(self.device),
            "q_mean": torch.from_numpy(self.ckpt["q_mean"].astype(np.float32)).to(self.device),
            "q_std": torch.from_numpy(self.ckpt["q_std"].astype(np.float32)).to(self.device),
        }
        self.name = self.path.stem

    def begin_episode(self, env, scenario, args):
        return {
            "cube_initial_pos": cube_geom_pos(env).copy(),
            "close_latched": False,
        }

    def default_horizon(self, args, episode_state):
        return int(args.warmup_steps + args.teacher_horizon)

    def obs_vector(self, env, obs_keys, progress, cube_initial_pos, scenario):
        obs = collect_obs(env, progress=progress)
        cube_now = cube_geom_pos(env).astype(np.float32)
        obs["object"] = object_obs(cube_now, obs["robot0_eef_pos"])
        obs["cube_initial_pos"] = np.asarray(cube_initial_pos, dtype=np.float32)
        obs["cube_size"] = np.array([scenario.half_size], dtype=np.float32)
        obs["cube_color"] = np.asarray(scenario.rgba, dtype=np.float32)
        obs["pad_mid_pos"] = get_pad_mid_pos(env).astype(np.float32)
        return np.concatenate(
            [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in obs_keys],
            axis=0,
        ).astype(np.float32)

    def predict(self, env, scenario, args, step, episode_state):
        phase_step = max(0, step - args.warmup_steps)
        progress = phase_step / max(1, args.teacher_horizon - 1)
        progress = float(np.clip(progress, 0.0, 1.0))
        x_np = self.obs_vector(
            env,
            self.ckpt["obs_keys"],
            progress,
            episode_state["cube_initial_pos"],
            scenario,
        )[None]
        x = torch.from_numpy(x_np).to(self.device)
        with torch.no_grad():
            x_norm = (x - self.tensors["obs_mean"]) / self.tensors["obs_std"]
            pred_q_norm, pred_grip_logit = self.model(x_norm)
            q_target = (pred_q_norm * self.tensors["q_std"] + self.tensors["q_mean"]).detach().cpu().numpy()[0]
            close_prob = float(torch.sigmoid(pred_grip_logit).detach().cpu().numpy()[0])

        close = close_prob >= args.gripper_threshold
        if progress < args.min_close_progress:
            close = False
        if args.hold_close_once:
            episode_state["close_latched"] = bool(episode_state["close_latched"] or close)
            close = bool(episode_state["close_latched"])
        return q_target, close, close_prob, progress


def load_runtime(policy_path, device):
    path = Path(policy_path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return PhaseSequenceRuntime(path)
    if suffix == ".pt":
        return BCMLPRuntime(path, device)
    raise ValueError(f"Unsupported policy format: {path}. Expected .pt or .npz")


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


def sample_initial_q(args, rng, base_q=None):
    q = np.asarray(args.initial_q if base_q is None else base_q, dtype=np.float64).copy()
    noise = np.asarray(args.initial_q_noise, dtype=np.float64)
    if np.any(noise != 0.0):
        q += rng.uniform(-noise, noise)
    return q


def load_scenarios(path, args):
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("scenarios", [])
    scenarios = []
    for idx, item in enumerate(raw):
        cube = np.asarray(item.get("cube", args.cube), dtype=np.float64)
        half_size = float(item.get("cube_half_size", item.get("half_size", args.cube_half_size)))
        rgba = np.asarray(
            item.get("cube_rgba", item.get("rgba", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)][1])),
            dtype=np.float64,
        )
        name = str(item.get("name", item.get("color_name", f"scenario_{idx}")))
        initial_q = np.asarray(item.get("initial_q", args.initial_q), dtype=np.float64)
        scenarios.append(Scenario(cube=cube, half_size=half_size, rgba=rgba, color_name=name, initial_q=initial_q))
    return scenarios


def sample_scenario(args, rng, episode_idx, scenarios):
    if scenarios:
        base = scenarios[episode_idx % len(scenarios)]
        initial_q = base.initial_q.copy()
        if args.apply_initial_q_noise_to_scenarios:
            initial_q = sample_initial_q(args, rng, initial_q)
        return Scenario(
            cube=base.cube.copy(),
            half_size=base.half_size,
            rgba=base.rgba.copy(),
            color_name=base.color_name,
            initial_q=initial_q,
        )

    if args.cube is not None:
        cube = args.cube.copy()
    else:
        x_min, x_max = args.x_range
        y_min, y_max = args.y_range
        cube = np.array(
            [
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max),
                args.cube_z,
            ],
            dtype=np.float64,
        )

    sizes = args.size_values
    half_size = float(args.cube_half_size if args.cube_half_size is not None else sizes[episode_idx % len(sizes)])
    color_name, rgba = args.colors[episode_idx % len(args.colors)]
    return Scenario(
        cube=cube,
        half_size=half_size,
        rgba=rgba.copy(),
        color_name=color_name,
        initial_q=sample_initial_q(args, rng),
    )


def configure_scene(env, args, scenario, base_body_id, base_quat0):
    env.reset()
    env.sim.model.body_quat[base_body_id] = base_quat0.copy()
    env.sim.forward()
    reject_global_yaw(args.base_yaw_deg)
    apply_base_longitudinal_twist(env, args.base_twist_deg)
    if not args.hide_base_stand:
        style_vertical_base_support(env)

    cube = scenario.cube.copy()
    if not args.keep_cube_z:
        cube[2] += scenario.half_size - args.reference_cube_half_size
    set_cube_appearance(env, half_size=scenario.half_size, rgba=scenario.rgba)
    set_cube_pos(env, cube)

    env.sim.data.qpos[:6] = scenario.initial_q.copy()
    env.sim.data.qvel[:6] = 0.0
    env.sim.forward()

    disable_arm_collisions(env)
    disable_gripper_nonpad_collisions(env)
    boost_arm_actuators(env, kp=args.boost_kp, force=args.boost_force, damping=args.boost_damping)
    boost_gripper_grasp(env, kp=args.gripper_boost_kp, force=args.gripper_boost_force, friction=args.grasp_friction)


def render_frame(env, args, status_lines):
    frame = env.sim.render(
        camera_name=args.video_camera,
        width=args.video_width,
        height=args.video_height,
        depth=False,
    )
    frame = np.asarray(frame)[::-1].copy()
    try:
        import cv2

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        y = 24
        for line in status_lines:
            cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            y += 22
        return frame, bgr
    except Exception:
        return frame, None


def print_controls():
    print("Controls: q/Esc=quit | n=next cube | r=reset current | p=pause/resume | h=help")


def run_demo(args):
    if args.checkpoint:
        args.policy = args.checkpoint
    if args.random_scenarios:
        args.scenario_file = None

    scenarios = load_scenarios(args.scenario_file, args) if args.scenario_file else []
    rng = np.random.default_rng(args.seed)
    runtime = load_runtime(args.policy, args.device)
    web_server = None
    if args.web_display:
        web_server = WebDisplayServer(args.web_host, args.web_port, jpeg_quality=args.web_jpeg_quality)
        web_server.start()

    needs_render = args.display or args.web_display or args.record_dir is not None
    env = make_env(has_offscreen_renderer=needs_render)
    env.reset()
    base_body_id, _base_name = find_body_id(env, "jaka_base")
    base_quat0 = env.sim.model.body_quat[base_body_id].copy()

    timestep = float(env.sim.model.opt.timestep)
    substeps = max(1, int(round(1.0 / (args.control_freq * timestep))))
    episode_idx = 0
    completed = 0
    paused = False
    quit_requested = False
    reset_current = False
    print_controls()
    print(f"policy={runtime.path}")
    if scenarios:
        print(f"scenario_file={args.scenario_file} scenarios={len(scenarios)}")
    else:
        print("scenario_mode=random")

    if args.record_dir:
        Path(args.record_dir).mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except Exception:
        cv2 = None
        if args.display:
            raise RuntimeError("--display requires OpenCV with GUI support")
        if args.web_display:
            raise RuntimeError("--web-display requires OpenCV for JPEG encoding")

    while not quit_requested and (args.num_runs <= 0 or completed < args.num_runs):
        scenario = sample_scenario(args, rng, episode_idx, scenarios)
        configure_scene(env, args, scenario, base_body_id, base_quat0)
        episode_state = runtime.begin_episode(env, scenario, args)
        horizon = args.horizon if args.horizon > 0 else runtime.default_horizon(args, episode_state)
        prev_q_cmd = env.sim.data.qpos[:6].copy()
        success_step = None
        success_hold = 0
        max_bad_contacts = 0
        last_bad_pairs = []
        writer = None
        video_path = None
        if args.record_dir:
            video_path = Path(args.record_dir) / f"continuous_demo_{episode_idx:04d}_{safe_name(scenario.color_name)}.mp4"
            writer = imageio.get_writer(str(video_path), fps=args.control_freq)

        print(
            f"episode={episode_idx} cube={np.round(scenario.cube, 4)} "
            f"half_size={scenario.half_size:.4f} initial_q={np.round(scenario.initial_q, 3)} "
            f"name={scenario.color_name}"
        )

        step = 0
        while step < horizon and not quit_requested:
            loop_start = time.perf_counter()
            if paused:
                command = web_server.pop_command() if web_server is not None else None
                if command == "quit":
                    quit_requested = True
                elif command == "pause":
                    paused = False
                elif command == "next":
                    reset_current = False
                    break
                elif command == "reset":
                    reset_current = True
                    break
                if args.web_display and needs_render:
                    _rgb, _bgr = render_frame(env, args, ["PAUSED", "p resume | n next | r reset | q quit"])
                    web_server.update_frame(_rgb, _bgr)
                if args.display and cv2 is not None:
                    _rgb, bgr = render_frame(env, args, ["PAUSED", "q quit | p resume | n next | r reset"])
                    cv2.imshow(args.window_name, bgr)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), 27):
                        quit_requested = True
                    elif key == ord("p"):
                        paused = False
                    elif key == ord("n"):
                        reset_current = False
                        break
                    elif key == ord("r"):
                        reset_current = True
                        break
                else:
                    time.sleep(0.1)
                continue

            q_target, close, close_prob, progress = runtime.predict(env, scenario, args, step, episode_state)
            q_now = env.sim.data.qpos[:6].copy()
            delta = args.kp * (q_target - q_now)
            q_cmd = q_now + np.clip(delta, -args.max_joint_step, args.max_joint_step)
            if args.command_smoothing > 0.0:
                alpha = float(np.clip(args.command_smoothing, 0.0, 0.98))
                q_cmd = alpha * prev_q_cmd + (1.0 - alpha) * q_cmd
            prev_q_cmd = q_cmd.copy()
            set_ctrl(env, q_cmd, close)
            for _ in range(substeps):
                env.sim.step()

            bad_contacts, bad_pairs = table_contact_count(env)
            max_bad_contacts = max(max_bad_contacts, bad_contacts)
            if bad_pairs:
                last_bad_pairs = bad_pairs
            cube_now = cube_geom_pos(env)
            success = bool(env._check_success())
            if args.min_final_cube_z is not None:
                success = success and float(cube_now[2]) >= args.min_final_cube_z
            if success and success_step is None:
                success_step = step
                print(
                    f"episode={episode_idx} success step={step} "
                    f"final_cube={np.round(cube_now, 4)} max_bad_table_contacts={max_bad_contacts}"
                )

            if args.print_every > 0 and (step % args.print_every == 0 or step == horizon - 1):
                print(
                    f"episode={episode_idx} step={step:04d}/{horizon} progress={progress:.3f} "
                    f"close_prob={close_prob:.3f} close={close} cube={np.round(cube_now, 4)} "
                    f"bad_table_contacts={bad_contacts}"
                )

            status = [
                f"{runtime.name} | episode {episode_idx} step {step}/{horizon}",
                f"cube {scenario.cube[0]:+.3f},{scenario.cube[1]:+.3f} size {scenario.half_size:.3f} {scenario.color_name}",
                f"z {cube_now[2]:.3f} success {success_step is not None} contacts {max_bad_contacts}",
                "q quit | n next | r reset | p pause | h help",
            ]
            if needs_render:
                rgb, bgr = render_frame(env, args, status)
                if writer is not None:
                    writer.append_data(rgb)
                if args.web_display and web_server is not None:
                    web_server.update_frame(rgb, bgr)
                    command = web_server.pop_command()
                    if command == "quit":
                        quit_requested = True
                    elif command == "next":
                        break
                    elif command == "reset":
                        reset_current = True
                        break
                    elif command == "pause":
                        paused = True
                if args.display and cv2 is not None:
                    cv2.imshow(args.window_name, bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        quit_requested = True
                    elif key == ord("n"):
                        break
                    elif key == ord("r"):
                        reset_current = True
                        break
                    elif key == ord("p"):
                        paused = True
                    elif key == ord("h"):
                        print_controls()

            if success_step is not None:
                success_hold += 1
                if success_hold >= args.success_hold_steps:
                    break

            if args.real_time or args.display or args.web_display:
                elapsed = time.perf_counter() - loop_start
                sleep_s = max(0.0, (1.0 / args.control_freq) - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            step += 1

        if writer is not None:
            writer.close()
            print(f"video={video_path}")

        if reset_current:
            print(f"episode={episode_idx} manual reset")
            reset_current = False
            continue

        if success_step is None and not quit_requested:
            print(
                f"episode={episode_idx} timeout success=False "
                f"final_cube={np.round(cube_geom_pos(env), 4)} max_bad_table_contacts={max_bad_contacts}"
            )
            if last_bad_pairs:
                print("last_bad_table_pairs:", last_bad_pairs)
        completed += int(success_step is not None)
        episode_idx += 1
        if args.reset_pause > 0 and not quit_requested:
            time.sleep(args.reset_pause)

    if args.display and cv2 is not None:
        cv2.destroyAllWindows()
    if web_server is not None:
        web_server.stop()
    print(f"finished episodes_started={episode_idx} successes={completed}")


def main():
    default_policy = DEFAULT_BC_MLP_POLICY if DEFAULT_BC_MLP_POLICY.exists() else DEFAULT_PHASE_POLICY
    default_scenarios = str(DEFAULT_SCENARIO_FILE) if DEFAULT_SCENARIO_FILE.exists() else None
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Continuous JAKA lift policy demo")
    parser.add_argument("--policy", default=str(default_policy), help="Path to trained .pt BC-MLP or .npz phase policy")
    parser.add_argument("--checkpoint", default=None, help="Alias for --policy, kept for BC-MLP eval script compatibility")
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--num-runs", type=int, default=0, help="Number of successful episodes; 0 means infinite")
    parser.add_argument("--scenario-file", default=default_scenarios, help="Optional JSON list of fixed scenarios")
    parser.add_argument("--random-scenarios", action="store_true", help="Ignore --scenario-file and sample random cubes")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cube", type=lambda s: parse_vec(s, 3), default=None)
    parser.add_argument("--cube-z", type=float, default=0.8307)
    parser.add_argument("--x-range", type=lambda s: parse_range(s, 2), default=[-0.020, 0.050])
    parser.add_argument("--y-range", type=lambda s: parse_range(s, 2), default=[-0.050, 0.020])
    parser.add_argument("--cube-half-size", type=float, default=0.020, help="Fixed cube half-size when --cube is used")
    parser.add_argument("--size-values", type=lambda s: parse_float_list(s), default=[0.016, 0.018, 0.020, 0.021])
    parser.add_argument("--colors", type=parse_color_palette, default=DEFAULT_COLORS)
    parser.add_argument("--reference-cube-half-size", type=float, default=0.020)
    parser.add_argument("--keep-cube-z", action="store_true")
    parser.add_argument("--initial-q", type=lambda s: parse_vec(s, 6), default="-0.0007,1.5797,-0.0322,0.001,0.0329,0.0193")
    parser.add_argument("--initial-q-noise", type=lambda s: parse_vec(s, 6), default="0,0,0,0,0,0")
    parser.add_argument("--apply-initial-q-noise-to-scenarios", action="store_true")
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--base-twist-deg", type=float, default=30.0)
    parser.add_argument("--hide-base-stand", action="store_true")
    parser.add_argument("--kp", type=float, default=0.9)
    parser.add_argument("--max-joint-step", type=float, default=0.09)
    parser.add_argument("--command-smoothing", type=float, default=0.25)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--min-close-progress", type=float, default=0.45)
    parser.add_argument("--hold-close-once", dest="hold_close_once", action="store_true", default=True)
    parser.add_argument("--no-hold-close-once", dest="hold_close_once", action="store_false")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--teacher-horizon", type=int, default=2694)
    parser.add_argument("--warmup-steps", type=int, default=260)
    parser.add_argument("--success-hold-steps", type=int, default=70)
    parser.add_argument("--reset-pause", type=float, default=0.35)
    parser.add_argument("--control-freq", type=float, default=20.0)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--web-display", action="store_true", help="Show the rollout as an MJPEG stream in a browser")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--web-jpeg-quality", type=int, default=85)
    parser.add_argument("--window-name", default="JAKA continuous BC-MLP policy demo")
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--video-camera", default="frontview")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--min-final-cube-z", type=float, default=0.90)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--boost-kp", type=float, default=1200.0)
    parser.add_argument("--boost-force", type=float, default=2500.0)
    parser.add_argument("--boost-damping", type=float, default=8.0)
    parser.add_argument("--gripper-boost-kp", type=float, default=1800.0)
    parser.add_argument("--gripper-boost-force", type=float, default=220.0)
    parser.add_argument("--grasp-friction", type=float, default=5.0)
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
