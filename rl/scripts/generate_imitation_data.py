from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rl.laat_game.env import LaatCardEnv
from rl.laat_game.teachers import MixedTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=50_000)
    parser.add_argument("--teacher-checkpoint", type=Path, action="append", default=None)
    parser.add_argument("--teacher-checkpoint-weight", type=float, action="append", default=None)
    parser.add_argument("--heuristic-teacher-weight", type=float, default=None)
    parser.add_argument("--random-teacher-weight", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/model2/imitation_50k.npz"))
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--hand-threshold", type=int, default=52)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--opponent-pool",
        choices=["baseline", "league", "random", "low_card", "high_card", "safe_suit", "pressure_suit", "checkpoint"],
        default="league",
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
    teacher_checkpoints = args.teacher_checkpoint
    if teacher_checkpoints is None:
        teacher_checkpoints = [Path("models/maskable_ppo_laat/checkpoints/laat_300000_steps.zip")]
    teacher = MixedTeacher(
        teacher_checkpoints,
        device=device,
        seed=args.seed,
        checkpoint_weight=args.teacher_checkpoint_weight,
        heuristic_weight=args.heuristic_teacher_weight,
        random_weight=args.random_teacher_weight,
    )

    observations = []
    masks = []
    actions = []
    teacher_ids = []
    episode_ids = []
    turns = []

    obs, _ = env.reset(seed=args.seed)
    episode_id = 0
    while len(actions) < args.decisions:
        mask = env.action_masks()
        action, teacher_name = teacher.choose(env)
        if not mask[action]:
            raise RuntimeError(f"Teacher selected illegal action {action}.")
        observations.append(obs.astype(np.float32))
        masks.append(mask.astype(bool))
        actions.append(action)
        teacher_ids.append(teacher.teacher_id_map[teacher_name])
        episode_ids.append(episode_id)
        turns.append(env.state.turn if env.state is not None else 0)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episode_id += 1
            obs, _ = env.reset()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=np.asarray(observations, dtype=np.float32),
        masks=np.asarray(masks, dtype=bool),
        actions=np.asarray(actions, dtype=np.int64),
        teacher_ids=np.asarray(teacher_ids, dtype=np.int64),
        teacher_names=np.asarray(teacher.teacher_names),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        turns=np.asarray(turns, dtype=np.int64),
    )
    print(f"saved={args.output}")
    print(f"decisions={len(actions)} episodes={episode_id + 1}")


if __name__ == "__main__":
    main()
