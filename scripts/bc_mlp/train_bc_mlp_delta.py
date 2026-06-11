#!/usr/bin/env python3
"""Train a robust BC-MLP that predicts delta_q commands."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OBS_KEYS = [
    "robot0_joint_pos",
    "robot0_joint_vel",
    "cube_initial_pos",
    "progress",
    "cube_size",
]


def parse_csv_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def natural_demo_key(key):
    try:
        return int(key.rsplit("_", 1)[1])
    except Exception:
        return key


def make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class DeltaBCMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(512, 512, 256), activation="silu"):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(make_activation(activation))
            last_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.delta_head = nn.Linear(last_dim, 6)
        self.gripper_head = nn.Linear(last_dim, 1)

    def forward(self, x):
        z = self.backbone(x)
        return self.delta_head(z), self.gripper_head(z).squeeze(-1)


def obs_matrix(demo, obs_keys):
    obs = demo["obs"]
    rows = []
    n = demo["actions"].shape[0]
    for key in obs_keys:
        if key in obs:
            value = obs[key][()].astype(np.float32)
        elif key == "cube_initial_pos":
            cube = np.asarray(demo.attrs.get("cube_initial_pos", demo.attrs.get("cube_xyz", [0.0144, -0.0123, 0.8307])), dtype=np.float32)
            value = np.repeat(cube[None, :], n, axis=0)
        elif key == "cube_size":
            size = float(demo.attrs.get("cube_half_size", 0.020))
            value = np.full((n, 1), size, dtype=np.float32)
        else:
            raise KeyError(f"Missing obs key {key} in {demo.name}")
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
            delta_parts = []
            close_parts = []
            for key in keys:
                demo = data[key]
                actions = demo["actions"][()].astype(np.float32)
                obs_parts.append(obs_matrix(demo, obs_keys))
                delta_parts.append(actions[:, :6])
                close_parts.append((np.mean(actions[:, 6:8], axis=1) > 0.0).astype(np.float32))
            arrays[split_name] = {
                "obs": np.concatenate(obs_parts, axis=0),
                "delta": np.concatenate(delta_parts, axis=0),
                "close": np.concatenate(close_parts, axis=0),
                "demos": keys,
            }
        attrs = dict(data.attrs)
    arrays["attrs"] = attrs
    return arrays


def standardize(values, eps=1e-6):
    mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    std = (values.std(axis=0, keepdims=True) + eps).astype(np.float32)
    return mean, std


def evaluate(model, loader, delta_mean_t, delta_std_t, device, gripper_loss_weight):
    model.eval()
    losses = []
    delta_errors = []
    grip_correct = []
    bce = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for xb, delta_norm_b, close_b in loader:
            xb = xb.to(device)
            delta_norm_b = delta_norm_b.to(device)
            close_b = close_b.to(device)
            pred_delta_norm, pred_grip = model(xb)
            delta_loss = (pred_delta_norm - delta_norm_b).pow(2).mean(dim=1)
            grip_loss = bce(pred_grip, close_b)
            losses.append((delta_loss + gripper_loss_weight * grip_loss).detach().cpu())
            pred_delta = pred_delta_norm * delta_std_t + delta_mean_t
            true_delta = delta_norm_b * delta_std_t + delta_mean_t
            delta_errors.append((pred_delta - true_delta).abs().detach().cpu())
            pred_close = (torch.sigmoid(pred_grip) >= 0.5).float()
            grip_correct.append((pred_close == close_b).float().detach().cpu())
    loss = torch.cat(losses).mean().item()
    delta_mae = torch.cat(delta_errors).mean(dim=0).numpy()
    grip_acc = torch.cat(grip_correct).mean().item()
    return {
        "loss": float(loss),
        "delta_mae_mean": float(np.mean(delta_mae)),
        "delta_mae_per_joint": [float(v) for v in delta_mae],
        "gripper_acc": float(grip_acc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bc_mlp_policy_experiment/robust_v2/datasets/robust_teacher_delta_96eps.hdf5")
    parser.add_argument("--output", default="bc_mlp_policy_experiment/robust_v2/models/bc_mlp_delta_robust.pt")
    parser.add_argument("--obs-keys", default=",".join(DEFAULT_OBS_KEYS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-sizes", default="512,512,256")
    parser.add_argument("--activation", default="silu", choices=["relu", "gelu", "silu"])
    parser.add_argument("--valid-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=205)
    parser.add_argument("--gripper-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    obs_keys = parse_csv_list(args.obs_keys)
    hidden_sizes = tuple(int(v) for v in parse_csv_list(args.hidden_sizes))
    dataset = load_dataset(args.dataset, obs_keys, args.valid_fraction, args.seed, success_only=True)
    train = dataset["train"]
    valid = dataset["valid"]
    obs_mean, obs_std = standardize(train["obs"])
    delta_mean, delta_std = standardize(train["delta"])

    train_x = torch.from_numpy((train["obs"] - obs_mean) / obs_std)
    train_delta = torch.from_numpy((train["delta"] - delta_mean) / delta_std)
    train_close = torch.from_numpy(train["close"])
    valid_x = torch.from_numpy((valid["obs"] - obs_mean) / obs_std)
    valid_delta = torch.from_numpy((valid["delta"] - delta_mean) / delta_std)
    valid_close = torch.from_numpy(valid["close"])

    train_loader = DataLoader(TensorDataset(train_x, train_delta, train_close), batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(TensorDataset(valid_x, valid_delta, valid_close), batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    device = torch.device(args.device)
    model = DeltaBCMLP(train_x.shape[1], hidden_sizes=hidden_sizes, activation=args.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    bce = nn.BCEWithLogitsLoss()
    delta_mean_t = torch.from_numpy(delta_mean).to(device)
    delta_std_t = torch.from_numpy(delta_std).to(device)

    best = None
    best_state = None
    history = []
    print("loaded", "train_samples", len(train_x), "valid_samples", len(valid_x), "obs_dim", train_x.shape[1], "train_demos", len(train["demos"]), "valid_demos", len(valid["demos"]))
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for xb, delta_norm_b, close_b in train_loader:
            xb = xb.to(device, non_blocking=True)
            delta_norm_b = delta_norm_b.to(device, non_blocking=True)
            close_b = close_b.to(device, non_blocking=True)
            pred_delta_norm, pred_grip = model(xb)
            delta_loss = (pred_delta_norm - delta_norm_b).pow(2).mean()
            grip_loss = bce(pred_grip, close_b)
            loss = delta_loss + args.gripper_loss_weight * grip_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = evaluate(model, valid_loader, delta_mean_t, delta_std_t, device, args.gripper_loss_weight)
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
                f"delta_mae={metrics['delta_mae_mean']:.6f}",
                f"grip_acc={metrics['gripper_acc']:.4f}",
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "format": "jaka_bc_mlp_delta_robust_v1",
        "model_state": best_state,
        "obs_keys": obs_keys,
        "obs_dim": int(train_x.shape[1]),
        "hidden_sizes": list(hidden_sizes),
        "activation": args.activation,
        "target_mode": "delta_q_plus_gripper_logit",
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std": obs_std.astype(np.float32),
        "delta_mean": delta_mean.astype(np.float32),
        "delta_std": delta_std.astype(np.float32),
        "dataset": args.dataset,
        "train_demos": train["demos"],
        "valid_demos": valid["demos"],
        "dataset_attrs": {k: str(v) for k, v in dataset["attrs"].items()},
        "best_metrics": best,
        "history": history,
        "args": vars(args),
    }
    torch.save(ckpt, output)
    with output.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output": str(output),
                "dataset": args.dataset,
                "obs_keys": obs_keys,
                "hidden_sizes": list(hidden_sizes),
                "activation": args.activation,
                "target_mode": "delta_q_plus_gripper_logit",
                "train_samples": int(len(train_x)),
                "valid_samples": int(len(valid_x)),
                "train_demos": train["demos"],
                "valid_demos": valid["demos"],
                "best_metrics": best,
            },
            f,
            indent=2,
        )
    print("saved:", output)
    print("best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
