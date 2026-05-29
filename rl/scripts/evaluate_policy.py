from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import torch

from rl.laat_game.bc_policy import load_bc_policy
from rl.laat_game.env import LaatCardEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/maskable_ppo_laat/latest.zip"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--hand-threshold", type=int, default=52)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--opponent-pool",
        choices=["baseline", "league", "random", "low_card", "high_card", "safe_suit", "pressure_suit", "checkpoint"],
        default="baseline",
    )
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        device = "cpu"
    env = LaatCardEnv(
        player_count=args.players,
        max_rounds=args.max_rounds,
        hand_threshold=args.hand_threshold,
        seed=args.seed,
        opponent_pool=args.opponent_pool,
        opponent_checkpoint=args.opponent_checkpoint,
        opponent_device=device,
    )
    is_bc_model = args.model.suffix == ".pt"
    if is_bc_model:
        bc_policy, metadata = load_bc_policy(args.model, device=device)
        print(f"Loaded behavior clone model metadata={metadata}")
    else:
        model = MaskablePPO.load(args.model, env=env, device=device)

    wins = 0
    losses = 0
    invalid = 0
    turns: list[int] = []
    final_hands: list[int] = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        last_info = info
        while not done:
            if is_bc_model:
                action = bc_policy.predict(obs, env.action_masks(), device=device)
            else:
                action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _, terminated, truncated, last_info = env.step(int(action))
            done = terminated or truncated
            invalid += int(last_info.get("invalid_action", False))

        hand_counts = last_info["hand_counts"]
        final_hands.append(hand_counts[0])
        turns.append(last_info["turn"])
        wins += int(0 in last_info["winner_ids"])
        losses += int(0 in last_info["loser_ids"])

    print(f"episodes={args.episodes}")
    print(f"win_rate={wins / args.episodes:.3f}")
    print(f"loss_rate={losses / args.episodes:.3f}")
    print(f"avg_final_hand={float(np.mean(final_hands)):.2f}")
    print(f"avg_turns={float(np.mean(turns)):.2f}")
    print(f"invalid_actions={invalid}")


if __name__ == "__main__":
    main()
