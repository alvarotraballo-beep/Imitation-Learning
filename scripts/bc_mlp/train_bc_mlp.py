#!/usr/bin/env python3
"""Train a low-dimensional BC-MLP for the JAKA lift task.

The model predicts absolute 6-DoF joint targets plus a gripper-close logit.
It is intentionally stored in this experiment folder and does not alter the
existing phase-sequence policy or dataset.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OBS_KEYS = [
    "cube_initial_pos",
    "progress",
    "cube_size",
]


def parse_csv_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def natural_demo_key(key):
    try:
        return int(key.rsplit("_", 1)[1])
    except Exception:
        return key


def make_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class BCMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(512, 512, 256), activation="relu"):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(make_activation(activation))
            last_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(last_dim, 6)
        self.gripper_head = nn.Linear(last_dim, 1)

    def forward(self, x):
        z = self.backbone(x)
        return self.q_head(z), self.gripper_head(z).squeeze(-1)


def obs_matrix(demo, obs_keys):
    obs = demo["obs"]
    rows = []
    for key in obs_keys:
        if key == "cube_initial_pos":
            if "object" in obs:
                cube = obs["object"][()][0, :3].astype(np.float32)
            elif "bag_cube_pos" in obs:
                cube = obs["bag_cube_pos"][()][0].astype(np.float32)
            else:
                cube = np.asarray(demo.attrs.get("cube_xyz", [0.0144, -0.0123, 0.8307]), dtype=np.float32)
            value = np.repeat(cube[None, :], demo["actions"].shape[0], axis=0)
        elif key == "cube_pos":
            if "object" in obs:
                value = obs["object"][()][:, :3].astype(np.float32)
            elif "bag_cube_pos" in obs:
                value = obs["bag_cube_pos"][()].astype(np.float32)
            else:
                cube = np.asarray(demo.attrs.get("cube_xyz", [0.0144, -0.0123, 0.8307]), dtype=np.float32)
                value = np.repeat(cube[None, :], demo["actions"].shape[0], axis=0)
        elif key == "cube_xy":
            if "object" in obs:
                value = obs["object"][()][:, :2].astype(np.float32)
            else:
                cube = np.asarray(demo.attrs.get("cube_xyz", [0.0144, -0.0123, 0.8307]), dtype=np.float32)
                value = np.repeat(cube[None, :2], demo["actions"].shape[0], axis=0)
        elif key in obs:
            value = obs[key][()].astype(np.float32)
        elif key == "cube_size":
            size = float(demo.attrs.get("cube_half_size", 0.020))
            value = np.full((demo["actions"].shape[0], 1), size, dtype=np.float32)
        else:
            raise KeyError(f"Observation key missing in {demo.name}: {key}")
        if value.ndim == 1:
            value = value[:, None]
        rows.append(value)
    return np.concatenate(rows, axis=1).astype(np.float32)


def load_dataset(path, obs_keys, valid_fraction, seed, success_only=True):
    rng = random.Random(seed)
    with h5py.File(path, "r") as f:
        data = f["data"]
        demo_keys = sorted(list(data.keys()), key=natural_demo_key)
        if success_only:
            demo_keys = [key for key in demo_keys if bool(data[key].attrs.get("success", False))]
        if not demo_keys:
            raise RuntimeError("No demos available after filtering")
        shuffled = demo_keys[:]
        rng.shuffle(shuffled)
        valid_count = max(1, int(round(len(shuffled) * valid_fraction))) if len(shuffled) > 1 else 0
        valid_set = set(shuffled[:valid_count])
        split = {
            "train": [key for key in demo_keys if key not in valid_set],
            "valid": [key for key in demo_keys if key in valid_set],
        }

        arrays = {}
        for split_name, keys in split.items():
            obs_parts = []
            q_target_parts = []
            close_parts = []
            for key in keys:
                demo = data[key]
                obs_vec = obs_matrix(demo, obs_keys)
                q_now = demo["obs"]["robot0_joint_pos"][()].astype(np.float32)
                actions = demo["actions"][()].astype(np.float32)
                q_target = q_now + actions[:, :6]
                gripper_close = (np.mean(actions[:, 6:8], axis=1) > 0.0).astype(np.float32)
                obs_parts.append(obs_vec)
                q_target_parts.append(q_target)
                close_parts.append(gripper_close)
            if keys:
                arrays[split_name] = {
                    "obs": np.concatenate(obs_parts, axis=0),
                    "q_target": np.concatenate(q_target_parts, axis=0),
                    "close": np.concatenate(close_parts, axis=0),
                    "demos": keys,
                }
            else:
                arrays[split_name] = {
                    "obs": np.zeros((0, 1), dtype=np.float32),
                    "q_target": np.zeros((0, 6), dtype=np.float32),
                    "close": np.zeros((0,), dtype=np.float32),
                    "demos": keys,
                }
        attrs = dict(data.attrs)
    arrays["attrs"] = attrs
    return arrays


def standardize(values, eps=1e-6):
    mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    std = (values.std(axis=0, keepdims=True) + eps).astype(np.float32)
    return mean, std


def evaluate(model, loader, q_mean_t, q_std_t, device):
    model.eval()
    losses = []
    q_abs_errors = []
    grip_correct = []
    bce = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for xb, q_norm_b, close_b in loader:
            xb = xb.to(device)
            q_norm_b = q_norm_b.to(device)
            close_b = close_b.to(device)
            pred_q_norm, pred_grip = model(xb)
            q_loss = (pred_q_norm - q_norm_b).pow(2).mean(dim=1)
            g_loss = bce(pred_grip, close_b)
            losses.append((q_loss + 0.5 * g_loss).detach().cpu())
            pred_q = pred_q_norm * q_std_t + q_mean_t
            true_q = q_norm_b * q_std_t + q_mean_t
            q_abs_errors.append((pred_q - true_q).abs().detach().cpu())
            pred_close = (torch.sigmoid(pred_grip) >= 0.5).float()
            grip_correct.append((pred_close == close_b).float().detach().cpu())
    loss = torch.cat(losses).mean().item()
    q_mae = torch.cat(q_abs_errors).mean(dim=0).numpy()
    grip_acc = torch.cat(grip_correct).mean().item()
    return {
        "loss": float(loss),
        "q_mae_mean": float(np.mean(q_mae)),
        "q_mae_per_joint": [float(v) for v in q_mae],
        "gripper_acc": float(grip_acc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/assets/bag_lift_direct_ctrl_basetwist30_openfix_variants_allbags_3var.hdf5")
    parser.add_argument("--output", default="bc_mlp_policy_experiment/models/bc_mlp_cube_initial_progress_absq_allbags.pt")
    parser.add_argument("--obs-keys", default=",".join(DEFAULT_OBS_KEYS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-sizes", default="512,512,256")
    parser.add_argument("--activation", default="relu", choices=["relu", "gelu", "silu"])
    parser.add_argument("--valid-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gripper-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    obs_keys = parse_csv_list(args.obs_keys)
    hidden_sizes = tuple(int(v) for v in parse_csv_list(args.hidden_sizes))
    dataset = load_dataset(
        args.dataset,
        obs_keys=obs_keys,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        success_only=True,
    )

    train = dataset["train"]
    valid = dataset["valid"]
    obs_mean, obs_std = standardize(train["obs"])
    q_mean, q_std = standardize(train["q_target"])

    train_x = torch.from_numpy((train["obs"] - obs_mean) / obs_std)
    train_q = torch.from_numpy((train["q_target"] - q_mean) / q_std)
    train_c = torch.from_numpy(train["close"])
    valid_x = torch.from_numpy((valid["obs"] - obs_mean) / obs_std)
    valid_q = torch.from_numpy((valid["q_target"] - q_mean) / q_std)
    valid_c = torch.from_numpy(valid["close"])

    train_loader = DataLoader(
        TensorDataset(train_x, train_q, train_c),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        TensorDataset(valid_x, valid_q, valid_c),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = BCMLP(train_x.shape[1], hidden_sizes=hidden_sizes, activation=args.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    bce = nn.BCEWithLogitsLoss()
    q_mean_t = torch.from_numpy(q_mean).to(device)
    q_std_t = torch.from_numpy(q_std).to(device)

    best = None
    best_state = None
    history = []
    print(
        "loaded",
        "train_samples", len(train_x),
        "valid_samples", len(valid_x),
        "train_demos", len(train["demos"]),
        "valid_demos", len(valid["demos"]),
        "obs_dim", train_x.shape[1],
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for xb, q_norm_b, close_b in train_loader:
            xb = xb.to(device, non_blocking=True)
            q_norm_b = q_norm_b.to(device, non_blocking=True)
            close_b = close_b.to(device, non_blocking=True)
            pred_q_norm, pred_grip = model(xb)
            q_loss = (pred_q_norm - q_norm_b).pow(2).mean()
            gripper_loss = bce(pred_grip, close_b)
            loss = q_loss + args.gripper_loss_weight * gripper_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        metrics = evaluate(model, valid_loader, q_mean_t, q_std_t, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(batch_losses))
        metrics["lr"] = float(scheduler.get_last_lr()[0])
        history.append(metrics)
        is_best = best is None or metrics["loss"] < best["loss"]
        if is_best:
            best = metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or is_best:
            print(
                f"epoch={epoch:03d}",
                f"train_loss={metrics['train_loss']:.6f}",
                f"valid_loss={metrics['loss']:.6f}",
                f"q_mae={metrics['q_mae_mean']:.6f}",
                f"grip_acc={metrics['gripper_acc']:.4f}",
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "jaka_bc_mlp_absq_v1",
        "model_state": best_state,
        "obs_keys": obs_keys,
        "obs_dim": int(train_x.shape[1]),
        "hidden_sizes": list(hidden_sizes),
        "activation": args.activation,
        "target_mode": "absolute_joint_target_plus_gripper_logit",
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std": obs_std.astype(np.float32),
        "q_mean": q_mean.astype(np.float32),
        "q_std": q_std.astype(np.float32),
        "dataset": args.dataset,
        "train_demos": train["demos"],
        "valid_demos": valid["demos"],
        "dataset_attrs": {k: str(v) for k, v in dataset["attrs"].items()},
        "best_metrics": best,
        "history": history,
        "args": vars(args),
    }
    torch.save(checkpoint, output)
    with output.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output": str(output),
                "dataset": args.dataset,
                "obs_keys": obs_keys,
                "hidden_sizes": list(hidden_sizes),
                "activation": args.activation,
                "train_samples": int(len(train_x)),
                "valid_samples": int(len(valid_x)),
                "train_demos": train["demos"],
                "valid_demos": valid["demos"],
                "best_metrics": best,
                "target_mode": "absolute_joint_target_plus_gripper_logit",
            },
            f,
            indent=2,
        )
    print("saved:", output)
    print("best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
