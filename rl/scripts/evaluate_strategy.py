from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import torch

from rl.laat_game.bc_policy import load_bc_policy
from rl.laat_game.cards import get_card
from rl.laat_game.engine import GameState, get_legal_moves
from rl.laat_game.env import LaatCardEnv
from rl.laat_game.teachers import choose_heuristic_action


@dataclass
class BucketMetrics:
    n: int = 0
    avg_chosen_rank: float = 0.0
    avg_legal_rank: float = 0.0
    avg_rank_percentile: float = 0.0
    high_card_rate_p75: float = 0.0
    low_card_rate_p25: float = 0.0
    avg_legal_count: float = 0.0


@dataclass
class StrategyMetrics:
    name: str
    episodes: int
    win_rate: float
    loss_rate: float
    avg_final_hand: float
    avg_turns: float
    invalid_actions: int
    composite_score: float
    laat: BucketMetrics
    lead_open: BucketMetrics
    follow_suit: BucketMetrics


class ActionPolicy(Protocol):
    def predict(self, env: LaatCardEnv, obs: np.ndarray) -> int:
        ...


class PpoActionPolicy:
    def __init__(self, path: Path, env: LaatCardEnv, device: str) -> None:
        self.model = MaskablePPO.load(path, env=env, device=device)

    def predict(self, env: LaatCardEnv, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
        return int(action)


class BcActionPolicy:
    def __init__(self, path: Path, device: str) -> None:
        self.policy, self.metadata = load_bc_policy(path, device=device)
        self.device = device

    def predict(self, env: LaatCardEnv, obs: np.ndarray) -> int:
        return self.policy.predict(obs, env.action_masks(), device=self.device)


class RandomActionPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def predict(self, env: LaatCardEnv, obs: np.ndarray) -> int:
        assert env.state is not None
        return int(self.rng.choice(get_legal_moves(env.state, 0)))


class HeuristicActionPolicy:
    def predict(self, env: LaatCardEnv, obs: np.ndarray) -> int:
        assert env.state is not None
        return int(choose_heuristic_action(env.state))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model to evaluate. Use name=path or just path. Supports .zip PPO and .pt BC checkpoints.",
    )
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--hand-threshold", type=int, default=52)
    parser.add_argument("--seed", type=int, default=20200)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--opponent-pool",
        choices=["baseline", "league", "random", "low_card", "high_card", "safe_suit", "pressure_suit", "checkpoint"],
        default="baseline",
    )
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-project", default="laat-card-rl")
    parser.add_argument("--wandb-run-name", default="strategy-eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    specs = parse_model_specs(args.model)
    if args.include_baselines:
        specs.extend([("random", Path("__random__")), ("heuristic", Path("__heuristic__"))])
    if not specs:
        raise SystemExit("Pass at least one --model or use --include-baselines.")

    results: list[StrategyMetrics] = []
    for name, path in specs:
        env = LaatCardEnv(
            player_count=args.players,
            max_rounds=args.max_rounds,
            hand_threshold=args.hand_threshold,
            seed=args.seed,
            opponent_pool=args.opponent_pool,
            opponent_checkpoint=args.opponent_checkpoint,
            opponent_device=device,
        )
        policy = build_policy(name, path, env, device, args.seed)
        metrics = evaluate_strategy(name, policy, env, args.episodes, args.seed)
        results.append(metrics)

    print_strategy_table(results)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    if args.wandb_mode != "disabled":
        log_to_wandb(args, results)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"
    return device


def parse_model_specs(raw_specs: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for raw in raw_specs:
        if "=" in raw:
            name, path = raw.split("=", 1)
            specs.append((name, Path(path)))
        else:
            path = Path(raw)
            specs.append((path.stem, path))
    return specs


def build_policy(name: str, path: Path, env: LaatCardEnv, device: str, seed: int) -> ActionPolicy:
    if name == "random" or str(path) == "__random__":
        return RandomActionPolicy(seed)
    if name == "heuristic" or str(path) == "__heuristic__":
        return HeuristicActionPolicy()
    if not path.exists():
        raise FileNotFoundError(f"Model does not exist: {path}")
    if path.suffix == ".pt":
        return BcActionPolicy(path, device)
    return PpoActionPolicy(path, env, device)


def evaluate_strategy(
    name: str,
    policy: ActionPolicy,
    env: LaatCardEnv,
    episodes: int,
    seed: int,
) -> StrategyMetrics:
    buckets: dict[str, list[tuple[float, float, float, float]]] = {
        "laat": [],
        "lead_open": [],
        "follow_suit": [],
    }
    wins = 0
    losses = 0
    invalid_actions = 0
    final_hands: list[int] = []
    turns: list[int] = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        last_info = info
        while not done:
            assert env.state is not None
            legal = get_legal_moves(env.state, 0)
            if not legal:
                break
            action = int(policy.predict(env, obs))
            record_decision(buckets, env.state, action, legal)
            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated
            invalid_actions += int(last_info.get("invalid_action", False))

        final_hands.append(int(last_info["hand_counts"][0]))
        turns.append(int(last_info["turn"]))
        wins += int(0 in last_info["winner_ids"])
        losses += int(0 in last_info["loser_ids"])

    win_rate = wins / max(episodes, 1)
    loss_rate = losses / max(episodes, 1)
    avg_final_hand = float(np.mean(final_hands)) if final_hands else 0.0
    return StrategyMetrics(
        name=name,
        episodes=episodes,
        win_rate=win_rate,
        loss_rate=loss_rate,
        avg_final_hand=avg_final_hand,
        avg_turns=float(np.mean(turns)) if turns else 0.0,
        invalid_actions=invalid_actions,
        composite_score=composite_score(win_rate, loss_rate, avg_final_hand, invalid_actions),
        laat=summarize_bucket(buckets["laat"]),
        lead_open=summarize_bucket(buckets["lead_open"]),
        follow_suit=summarize_bucket(buckets["follow_suit"]),
    )


def record_decision(
    buckets: dict[str, list[tuple[float, float, float, float]]],
    state: GameState,
    action: int,
    legal: list[int],
) -> None:
    row = (
        float(card_rank(action)),
        float(np.mean([card_rank(card) for card in legal])),
        rank_percentile(action, legal),
        float(len(legal)),
    )
    if state.lead_suit is None:
        buckets["lead_open"].append(row)
        return

    agent_has_lead_suit = any(get_card(card).suit == state.lead_suit for card in state.players[0].hand)
    if agent_has_lead_suit:
        buckets["follow_suit"].append(row)
    else:
        buckets["laat"].append(row)


def card_rank(card_id: int) -> int:
    return get_card(card_id).rank


def rank_percentile(chosen: int, legal: list[int]) -> float:
    if not legal:
        return 0.0
    ranks = sorted(card_rank(card) for card in legal)
    if len(ranks) == 1:
        return 1.0
    chosen_rank = card_rank(chosen)
    upper_tie_index = max(index for index, rank in enumerate(ranks) if rank <= chosen_rank)
    return (upper_tie_index + 1) / len(ranks)


def summarize_bucket(rows: list[tuple[float, float, float, float]]) -> BucketMetrics:
    if not rows:
        return BucketMetrics()
    values = np.asarray(rows, dtype=np.float32)
    return BucketMetrics(
        n=int(len(rows)),
        avg_chosen_rank=float(values[:, 0].mean()),
        avg_legal_rank=float(values[:, 1].mean()),
        avg_rank_percentile=float(values[:, 2].mean()),
        high_card_rate_p75=float((values[:, 2] >= 0.75).mean()),
        low_card_rate_p25=float((values[:, 2] <= 0.25).mean()),
        avg_legal_count=float(values[:, 3].mean()),
    )


def composite_score(win_rate: float, loss_rate: float, avg_final_hand: float, invalid_actions: int) -> float:
    return (100.0 * win_rate) - (80.0 * loss_rate) - (1.5 * avg_final_hand) - (10.0 * invalid_actions)


def print_strategy_table(results: list[StrategyMetrics]) -> None:
    columns = [
        "model",
        "win",
        "loss",
        "final",
        "score",
        "laat_pct",
        "laat_hi",
        "laat_low",
        "lead_pct",
        "invalid",
    ]
    print(" | ".join(columns))
    print(" | ".join(["---"] * len(columns)))
    for result in results:
        print(
            " | ".join(
                [
                    result.name,
                    f"{result.win_rate:.3f}",
                    f"{result.loss_rate:.3f}",
                    f"{result.avg_final_hand:.2f}",
                    f"{result.composite_score:.2f}",
                    f"{result.laat.avg_rank_percentile:.3f}",
                    f"{result.laat.high_card_rate_p75:.3f}",
                    f"{result.laat.low_card_rate_p25:.3f}",
                    f"{result.lead_open.avg_rank_percentile:.3f}",
                    str(result.invalid_actions),
                ]
            )
        )


def log_to_wandb(args: argparse.Namespace, results: list[StrategyMetrics]) -> None:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    for result in results:
        prefix = f"strategy/{result.name}"
        wandb.log(
            {
                f"{prefix}/win_rate": result.win_rate,
                f"{prefix}/loss_rate": result.loss_rate,
                f"{prefix}/avg_final_hand": result.avg_final_hand,
                f"{prefix}/avg_turns": result.avg_turns,
                f"{prefix}/invalid_actions": result.invalid_actions,
                f"{prefix}/composite_score": result.composite_score,
                f"{prefix}/laat_rank_percentile": result.laat.avg_rank_percentile,
                f"{prefix}/laat_high_card_rate": result.laat.high_card_rate_p75,
                f"{prefix}/laat_low_card_rate": result.laat.low_card_rate_p25,
                f"{prefix}/lead_rank_percentile": result.lead_open.avg_rank_percentile,
                f"{prefix}/follow_rank_percentile": result.follow_suit.avg_rank_percentile,
            }
        )
    run.finish()


if __name__ == "__main__":
    main()
