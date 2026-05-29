from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import numpy as np

from rl.laat_game.cards import SUITS, get_card
from rl.laat_game.engine import GameState, get_legal_moves
from rl.laat_game.observation import action_mask, encode_observation


class OpponentPolicy(Protocol):
    def set_seed(self, seed: int | None) -> None:
        ...

    def start_episode(self, state: GameState) -> None:
        ...

    def choose_action(self, state: GameState) -> int:
        ...


class BaseOpponentPolicy:
    def set_seed(self, seed: int | None) -> None:
        return None

    def start_episode(self, state: GameState) -> None:
        return None


class LowestSafePolicy(BaseOpponentPolicy):
    def choose_action(self, state: GameState) -> int:
        legal = require_legal_moves(state)
        if state.lead_suit is not None:
            return lowest_rank(legal)
        safe_moves = [card for card in legal if not suit_has_known_failure(state, get_card(card).suit, state.current_player)]
        return lowest_rank(safe_moves if safe_moves else legal)


class RandomLegalPolicy(BaseOpponentPolicy):
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def set_seed(self, seed: int | None) -> None:
        self.rng.seed(seed)

    def choose_action(self, state: GameState) -> int:
        return int(self.rng.choice(require_legal_moves(state)))


class LowCardPolicy(BaseOpponentPolicy):
    def choose_action(self, state: GameState) -> int:
        return lowest_rank(require_legal_moves(state))


class HighCardPolicy(BaseOpponentPolicy):
    def choose_action(self, state: GameState) -> int:
        return highest_rank(require_legal_moves(state))


class SafeSuitPolicy(BaseOpponentPolicy):
    def choose_action(self, state: GameState) -> int:
        legal = require_legal_moves(state)
        if state.lead_suit is not None:
            return lowest_rank(legal)
        safe_moves = [card for card in legal if not suit_has_known_failure(state, get_card(card).suit, state.current_player)]
        return lowest_rank(safe_moves if safe_moves else legal)


class PressureSuitPolicy(BaseOpponentPolicy):
    def choose_action(self, state: GameState) -> int:
        legal = require_legal_moves(state)
        if state.lead_suit is not None:
            return highest_rank(legal)
        return min(
            legal,
            key=lambda card: (
                -known_failure_count(state, get_card(card).suit, state.current_player),
                get_card(card).rank,
                card,
            ),
        )


class CheckpointPolicy(BaseOpponentPolicy):
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(checkpoint_path, device=device)

    def choose_action(self, state: GameState) -> int:
        legal = require_legal_moves(state)
        obs = encode_observation(state, state.current_player)
        mask = action_mask(state, state.current_player)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)
        return action if action in legal else lowest_rank(legal)


class LeagueOpponentPolicy(BaseOpponentPolicy):
    def __init__(
        self,
        rng: random.Random,
        checkpoint_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.rng = rng
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.assignments: dict[int, BaseOpponentPolicy] = {}
        self.checkpoint_policy = (
            CheckpointPolicy(checkpoint_path, device)
            if checkpoint_path is not None and checkpoint_path.exists()
            else None
        )

    def start_episode(self, state: GameState) -> None:
        self.assignments.clear()
        for player in state.players:
            if player.id == 0:
                continue
            self.assignments[player.id] = self._sample_policy()

    def set_seed(self, seed: int | None) -> None:
        self.rng.seed(seed)

    def choose_action(self, state: GameState) -> int:
        policy = self.assignments.get(state.current_player)
        if policy is None:
            policy = self._sample_policy()
            self.assignments[state.current_player] = policy
        return policy.choose_action(state)

    def _sample_policy(self) -> BaseOpponentPolicy:
        names = ["lowest_safe", "random", "low_card", "high_card", "safe_suit", "pressure_suit"]
        weights = [0.30, 0.20, 0.12, 0.10, 0.18, 0.10]
        if self.checkpoint_policy is not None:
            names.append("checkpoint")
            weights.append(0.25)
        choice = self.rng.choices(names, weights=weights, k=1)[0]
        if choice == "random":
            return RandomLegalPolicy(self.rng)
        if choice == "low_card":
            return LowCardPolicy()
        if choice == "high_card":
            return HighCardPolicy()
        if choice == "safe_suit":
            return SafeSuitPolicy()
        if choice == "pressure_suit":
            return PressureSuitPolicy()
        if choice == "checkpoint" and self.checkpoint_policy is not None:
            return self.checkpoint_policy
        return LowestSafePolicy()


def build_opponent_policy(
    opponent_pool: str = "baseline",
    seed: int | None = None,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> OpponentPolicy:
    rng = random.Random(seed)
    if opponent_pool == "league":
        return LeagueOpponentPolicy(rng, checkpoint_path=checkpoint_path, device=device)
    if opponent_pool == "random":
        return RandomLegalPolicy(rng)
    if opponent_pool == "low_card":
        return LowCardPolicy()
    if opponent_pool == "high_card":
        return HighCardPolicy()
    if opponent_pool == "safe_suit":
        return SafeSuitPolicy()
    if opponent_pool == "pressure_suit":
        return PressureSuitPolicy()
    if opponent_pool == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("--opponent-checkpoint is required for checkpoint opponents.")
        return CheckpointPolicy(checkpoint_path, device=device)
    return LowestSafePolicy()


def require_legal_moves(state: GameState) -> list[int]:
    legal = get_legal_moves(state)
    if not legal:
        raise ValueError("Opponent has no legal move.")
    return legal


def lowest_rank(cards: list[int]) -> int:
    return min(cards, key=lambda card: (get_card(card).rank, card))


def highest_rank(cards: list[int]) -> int:
    return max(cards, key=lambda card: (get_card(card).rank, card))


def suit_has_known_failure(state: GameState, suit: str, player_id: int) -> bool:
    return known_failure_count(state, suit, player_id) > 0


def known_failure_count(state: GameState, suit: str, player_id: int) -> int:
    suit_index = SUITS.index(suit)
    return sum(
        1
        for idx, failures in enumerate(state.suit_failures)
        if idx != player_id and failures[suit_index]
    )
