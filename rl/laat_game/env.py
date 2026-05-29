from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from rl.laat_game.engine import (
    GameConfig,
    GameState,
    add_event,
    apply_move,
    create_game,
    finish_by_largest_hand,
    get_legal_moves,
    run_safety_checks,
)
from rl.laat_game.observation import OBS_SIZE, action_mask, encode_observation
from rl.laat_game.opponents import OpponentPolicy, build_opponent_policy


class LaatCardEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        player_count: int = 4,
        max_rounds: int = 12,
        hand_threshold: int = 52,
        max_actions_per_game: int = 2000,
        seed: int | None = None,
        opponent_pool: str = "baseline",
        opponent_checkpoint: str | Path | None = None,
        opponent_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.config = GameConfig(
            player_count=player_count,
            max_rounds=max_rounds,
            hand_threshold=hand_threshold,
            max_actions_per_game=max_actions_per_game,
            seed=seed,
        )
        self.action_space = spaces.Discrete(52)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.state: GameState | None = None
        self.agent_player_id = 0
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)
        self.opponent_pool = opponent_pool
        self.opponent_checkpoint = Path(opponent_checkpoint) if opponent_checkpoint is not None else None
        self.opponent_device = opponent_device
        self.opponent_policy: OpponentPolicy = build_opponent_policy(
            opponent_pool=opponent_pool,
            seed=seed,
            checkpoint_path=self.opponent_checkpoint,
            device=opponent_device,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        config = GameConfig(**self.config.__dict__)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            game_seed = seed
        else:
            game_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        config.seed = game_seed
        self.opponent_policy.set_seed(game_seed)
        self.state = create_game(config)
        self.opponent_policy.start_episode(self.state)
        self._advance_bots_until_agent_turn()
        return self._observation(), self._info()

    def step(self, action: int):
        assert self.state is not None, "Call reset() before step()."
        if self.state.game_over:
            return self._observation(), 0.0, True, False, self._info()

        legal = get_legal_moves(self.state, self.agent_player_id)
        if int(action) not in legal:
            info = self._info()
            info["invalid_action"] = True
            return self._observation(), -1.0, True, False, info

        before_agent = len(self.state.players[self.agent_player_id].hand)
        before_event_count = len(self.state.events)
        apply_move(self.state, int(action))
        self._advance_bots_until_agent_turn()
        reward = self._reward(before_agent, self.state.events[before_event_count:])
        terminated = self.state.game_over
        truncated = False
        return self._observation(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        assert self.state is not None, "Call reset() before action_masks()."
        return action_mask(self.state, self.agent_player_id)

    def current_observation(self) -> np.ndarray:
        return self._observation()

    def render(self):
        assert self.state is not None, "Call reset() before render()."
        return "\n".join(
            [
                f"Round {self.state.round} Turn {self.state.turn}",
                f"Current player: {self.state.current_player}",
                f"Hands: {[len(player.hand) for player in self.state.players]}",
                f"Lead suit: {self.state.lead_suit or 'open'}",
            ]
        )

    def _observation(self) -> np.ndarray:
        assert self.state is not None
        return encode_observation(self.state, self.agent_player_id)

    def _reward(self, before_agent: int, new_events: list) -> float:
        assert self.state is not None
        after_agent = len(self.state.players[self.agent_player_id].hand)
        reward = 0.0

        hand_delta = before_agent - after_agent
        if hand_delta > 0:
            reward += 0.01 * hand_delta
        elif hand_delta < 0:
            reward += 0.02 * hand_delta

        for event in new_events:
            if event.type != "laat":
                continue
            collector_id = event.metadata.get("collector_id")
            card_count = int(event.metadata.get("card_count", 0))
            if collector_id == self.agent_player_id:
                reward -= 0.03 * card_count
            elif collector_id is not None:
                reward += 0.01 * card_count

        if self.state.game_over:
            if self.agent_player_id in self.state.winner_ids:
                reward += 5.0
            if self.agent_player_id in self.state.loser_ids:
                reward -= 5.0
        return float(reward)

    def _info(self) -> dict:
        assert self.state is not None
        safety_errors = run_safety_checks(self.state)
        return {
            "round": self.state.round,
            "turn": self.state.turn,
            "hand_counts": [len(player.hand) for player in self.state.players],
            "winner_ids": list(self.state.winner_ids),
            "loser_ids": list(self.state.loser_ids),
            "safety_errors": safety_errors,
        }

    def _advance_bots_until_agent_turn(self) -> None:
        assert self.state is not None
        guard = 0
        while (
            not self.state.game_over
            and self.state.current_player != self.agent_player_id
            and guard < self.state.config.max_actions_per_game
        ):
            apply_move(self.state, self.opponent_policy.choose_action(self.state))
            guard += 1
        if (
            not self.state.game_over
            and self.state.current_player != self.agent_player_id
            and guard >= self.state.config.max_actions_per_game
        ):
            add_event(self.state, "safety_stop", "Game stopped while advancing opponent turns.")
            finish_by_largest_hand(self.state)

    def _opponent_hand_total(self) -> int:
        assert self.state is not None
        return sum(len(player.hand) for player in self.state.players if player.id != self.agent_player_id)
