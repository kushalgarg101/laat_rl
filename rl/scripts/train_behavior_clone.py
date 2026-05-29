from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb

from rl.laat_game.bc_policy import BehaviorClonePolicy, masked_cross_entropy, save_bc_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/model2_bc"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="laat-card-rl")
    parser.add_argument("--wandb-run-name", default="model2-bc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        device = "cpu"
    train_behavior_clone(args, device)


def train_behavior_clone(args: argparse.Namespace, device: str) -> Path:
    rng = np.random.default_rng(args.seed)
    data = np.load(args.data)
    observations = data["observations"].astype(np.float32)
    masks = data["masks"].astype(bool)
    actions = data["actions"].astype(np.int64)

    indices = np.arange(len(actions))
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * args.val_split))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_loader = make_loader(observations, masks, actions, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(observations, masks, actions, val_idx, args.batch_size, shuffle=False)

    policy = BehaviorClonePolicy(observations.shape[1], hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    run = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_path = args.model_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(policy, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(policy, val_loader, device)
        if run is not None:
            wandb.log(
                {
                    "bc/train_loss": train_loss,
                    "bc/train_acc": train_acc,
                    "bc/val_loss": val_loss,
                    "bc/val_acc": val_acc,
                    "bc/epoch": epoch,
                }
            )
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        save_bc_policy(args.model_dir / "latest.pt", policy, {"epoch": epoch, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            save_bc_policy(best_path, policy, {"epoch": epoch, "val_acc": val_acc})

    if run is not None:
        run.finish()
    return best_path


def make_loader(observations, masks, actions, indices, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(observations[indices], dtype=torch.float32),
        torch.as_tensor(masks[indices], dtype=torch.bool),
        torch.as_tensor(actions[indices], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(policy, loader, optimizer, device: str) -> tuple[float, float]:
    policy.train()
    total_loss = 0.0
    correct = 0
    count = 0
    for obs, masks, actions in loader:
        obs = obs.to(device)
        masks = masks.to(device)
        actions = actions.to(device)
        optimizer.zero_grad()
        logits = policy(obs)
        loss = masked_cross_entropy(logits, actions, masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(actions)
        correct += int((logits.masked_fill(~masks, -1e9).argmax(dim=1) == actions).sum().item())
        count += len(actions)
    return total_loss / max(count, 1), correct / max(count, 1)


def evaluate(policy, loader, device: str) -> tuple[float, float]:
    policy.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for obs, masks, actions in loader:
            obs = obs.to(device)
            masks = masks.to(device)
            actions = actions.to(device)
            logits = policy(obs)
            loss = masked_cross_entropy(logits, actions, masks)
            total_loss += float(loss.item()) * len(actions)
            correct += int((logits.masked_fill(~masks, -1e9).argmax(dim=1) == actions).sum().item())
            count += len(actions)
    return total_loss / max(count, 1), correct / max(count, 1)


if __name__ == "__main__":
    main()
