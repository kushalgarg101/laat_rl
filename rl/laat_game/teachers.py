from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from rl.laat_game.cards import SUITS, get_card
from rl.laat_game.engine import GameState, get_legal_moves
from rl.laat_game.env import LaatCardEnv


TEACHER_IDS = {
    "heuristic": 0,
    "checkpoint": 1,
    "random": 2,
}


class MixedTeacher:
    def __init__(
        self,
        checkpoint_path: Path | list[Path] | None,
        device: str,
        seed: int = 7,
        checkpoint_weight: float | list[float] | None = None,
        heuristic_weight: float | None = None,
        random_weight: float | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.device = device
        self.checkpoint_models: dict[str, MaskablePPO] = {}
        checkpoint_paths = normalize_checkpoint_paths(checkpoint_path)
        for index, path in enumerate(checkpoint_paths):
            if not path.exists():
                raise FileNotFoundError(f"Teacher checkpoint does not exist: {path}")
            self.checkpoint_models[f"checkpoint_{index}"] = MaskablePPO.load(path, device=device)

        checkpoint_weights = normalize_checkpoint_weights(
            checkpoint_weight,
            checkpoint_count=len(checkpoint_paths),
            heuristic_weight=heuristic_weight,
            random_weight=random_weight,
        )
        default_heuristic = 0.4 if len(checkpoint_paths) <= 1 else 0.20
        default_random = 0.2 if len(checkpoint_paths) <= 1 else 0.05
        teacher_weights: list[tuple[str, float]] = [
            ("heuristic", default_heuristic if heuristic_weight is None else heuristic_weight),
            ("random", default_random if random_weight is None else random_weight),
        ]
        for index, weight in enumerate(checkpoint_weights):
            name = f"checkpoint_{index}"
            teacher_weights.append((name, weight if name in self.checkpoint_models else 0.0))

        weights = [weight for _, weight in teacher_weights]
        if any(weight < 0 for weight in weights):
            raise ValueError("Teacher weights must be non-negative.")
        total = sum(weights)
        if total <= 0:
            teacher_weights = [("heuristic", 1.0)]
            weights = [1.0]
            total = 1.0
        self.teacher_names = [name for name, _ in teacher_weights]
        self.weights = [weight / total for weight in weights]
        self.teacher_id_map = {
            **TEACHER_IDS,
            **{f"checkpoint_{index}": 10 + index for index in range(len(checkpoint_paths))},
        }

    def choose(self, env: LaatCardEnv) -> tuple[int, str]:
        teacher = self.rng.choices(self.teacher_names, weights=self.weights, k=1)[0]
        if teacher in self.checkpoint_models:
            action, _ = self.checkpoint_models[teacher].predict(env.current_observation(), action_masks=get_action_masks(env), deterministic=True)
            return int(action), teacher
        if teacher == "random":
            legal = get_legal_moves(env.state)
            return int(self.rng.choice(legal)), teacher
        return choose_heuristic_action(env.state), "heuristic"


def normalize_checkpoint_paths(checkpoint_path: Path | list[Path] | None) -> list[Path]:
    if checkpoint_path is None:
        return []
    if isinstance(checkpoint_path, list):
        return checkpoint_path
    return [checkpoint_path]


def normalize_checkpoint_weights(
    checkpoint_weight: float | list[float] | None,
    checkpoint_count: int,
    heuristic_weight: float | None,
    random_weight: float | None,
) -> list[float]:
    if checkpoint_count <= 0:
        return []
    if isinstance(checkpoint_weight, list):
        if len(checkpoint_weight) != checkpoint_count:
            raise ValueError("checkpoint_weight list must match checkpoint count.")
        return checkpoint_weight
    if checkpoint_weight is not None:
        return [checkpoint_weight / checkpoint_count] * checkpoint_count
    if checkpoint_count == 1:
        return [0.4]

    remaining = 1.0 - (0.20 if heuristic_weight is None else heuristic_weight) - (0.05 if random_weight is None else random_weight)
    defaults = [0.45, 0.30]
    if checkpoint_count <= len(defaults):
        weights = defaults[:checkpoint_count]
    else:
        extra = checkpoint_count - len(defaults)
        weights = defaults + [max(remaining - sum(defaults), 0.0) / extra] * extra
    total = sum(weights)
    return [weight * remaining / total for weight in weights] if total > 0 else [0.0] * checkpoint_count


def choose_heuristic_action(state: GameState) -> int:
    legal = get_legal_moves(state)
    if not legal:
        raise ValueError("No legal teacher action.")
    if state.lead_suit is not None:
        return min(legal, key=lambda card: (get_card(card).rank, card))

    safe = [card for card in legal if not suit_has_known_failure(state, get_card(card).suit, state.current_player)]
    candidates = safe if safe else legal
    suit_counts = {suit: sum(1 for card in state.players[state.current_player].hand if get_card(card).suit == suit) for suit in SUITS}
    return min(candidates, key=lambda card: (-suit_counts[get_card(card).suit], get_card(card).rank, card))


def suit_has_known_failure(state: GameState, suit: str, player_id: int) -> bool:
    suit_index = SUITS.index(suit)
    return any(idx != player_id and failures[suit_index] for idx, failures in enumerate(state.suit_failures))
