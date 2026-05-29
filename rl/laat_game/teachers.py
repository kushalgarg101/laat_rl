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
        checkpoint_path: Path | None,
        device: str,
        seed: int = 7,
        checkpoint_weight: float = 0.4,
        heuristic_weight: float = 0.4,
        random_weight: float = 0.2,
    ) -> None:
        self.rng = random.Random(seed)
        self.device = device
        self.checkpoint_model = None
        if checkpoint_path is not None and checkpoint_path.exists():
            self.checkpoint_model = MaskablePPO.load(checkpoint_path, device=device)
        weights = [heuristic_weight, checkpoint_weight if self.checkpoint_model is not None else 0.0, random_weight]
        total = sum(weights)
        if total <= 0:
            weights = [1.0, 0.0, 0.0]
            total = 1.0
        self.teacher_names = ["heuristic", "checkpoint", "random"]
        self.weights = [weight / total for weight in weights]

    def choose(self, env: LaatCardEnv) -> tuple[int, str]:
        teacher = self.rng.choices(self.teacher_names, weights=self.weights, k=1)[0]
        if teacher == "checkpoint" and self.checkpoint_model is not None:
            action, _ = self.checkpoint_model.predict(env.current_observation(), action_masks=get_action_masks(env), deterministic=True)
            return int(action), teacher
        if teacher == "random":
            legal = get_legal_moves(env.state)
            return int(self.rng.choice(legal)), teacher
        return choose_heuristic_action(env.state), "heuristic"


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
