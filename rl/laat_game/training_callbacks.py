from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl.laat_game.cards import format_card
from rl.laat_game.env import LaatCardEnv


class WandbPredictionCallback(BaseCallback):
    def __init__(self, log_every: int = 250, top_k: int = 5, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_every = log_every
        self.top_k = top_k

    def _on_step(self) -> bool:
        if self.log_every <= 0 or self.num_timesteps % self.log_every != 0:
            return True

        try:
            import wandb
        except ImportError:
            return True

        env = self._base_env()
        if env is None or env.state is None:
            return True

        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos") or [{}]
        action = int(np.asarray(actions).reshape(-1)[0]) if actions is not None else -1
        reward = float(np.asarray(rewards).reshape(-1)[0]) if rewards is not None else 0.0
        legal_mask = env.action_masks()
        state = env.state
        top_predictions = self._top_predictions(env, legal_mask)
        value_estimate = self._value_estimate(env)

        log_data: dict[str, Any] = {
            "train/reward_step": reward,
            "train/latest_action_id": action,
            "train/legal_action_count": int(legal_mask.sum()),
            "value/estimate": value_estimate,
            "game/round": state.round,
            "game/turn": state.turn,
            "game/discard_count": len(state.round_discard),
            "game/agent_hand_count": len(state.players[0].hand),
            "game/opponent_hand_total": sum(len(player.hand) for player in state.players[1:]),
            "game/safety_error_count": len(infos[0].get("safety_errors", [])),
        }

        if 0 <= action < 52:
            log_data["train/latest_action_card"] = format_card(action)
        if top_predictions:
            for rank, action_id, card, probability, legal in top_predictions:
                prefix = f"predictions/top_{rank}"
                log_data[f"{prefix}_action_id"] = action_id
                log_data[f"{prefix}_card"] = card
                log_data[f"{prefix}_probability"] = probability
                log_data[f"{prefix}_legal"] = int(legal)

        try:
            wandb.log(log_data)
        except Exception as exc:
            if self.verbose:
                print(f"WandB prediction logging skipped after error: {exc}")
        return True

    def _base_env(self) -> LaatCardEnv | None:
        training_env = self.training_env
        if training_env is None:
            return None
        envs = getattr(training_env, "envs", None)
        if not envs:
            return None
        env = envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env if isinstance(env, LaatCardEnv) else None

    def _top_predictions(self, env: LaatCardEnv, legal_mask: np.ndarray) -> list[list[Any]]:
        try:
            obs = env.current_observation()
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
            distribution = self.model.policy.get_distribution(
                obs_tensor,
                action_masks=legal_mask.reshape(1, -1),
            )
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
        except Exception:
            return []

        ranked = np.argsort(probs)[::-1][: self.top_k]
        return [
            [index + 1, int(action), format_card(int(action)), float(probs[action]), bool(legal_mask[action])]
            for index, action in enumerate(ranked)
        ]

    def _value_estimate(self, env: LaatCardEnv) -> float:
        try:
            obs = env.current_observation()
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
            value = self.model.policy.predict_values(obs_tensor)
            return float(value.detach().cpu().numpy().reshape(-1)[0])
        except Exception:
            return 0.0


class StrategyEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        model_dir: Path,
        players: int,
        max_rounds: int,
        hand_threshold: int,
        opponent_pool: str,
        opponent_checkpoint: Path | None,
        opponent_device: str,
        use_wandb: bool,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.model_dir = model_dir
        self.players = players
        self.max_rounds = max_rounds
        self.hand_threshold = hand_threshold
        self.opponent_pool = opponent_pool
        self.opponent_checkpoint = opponent_checkpoint
        self.opponent_device = opponent_device
        self.use_wandb = use_wandb
        self.best_composite = float("-inf")
        self.best_win_rate = float("-inf")
        self.best_selection_key: tuple[float, float, float, int] | None = None

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        from rl.scripts.evaluate_strategy import evaluate_strategy

        env = LaatCardEnv(
            player_count=self.players,
            max_rounds=self.max_rounds,
            hand_threshold=self.hand_threshold,
            seed=self.eval_seed,
            opponent_pool=self.opponent_pool,
            opponent_checkpoint=self.opponent_checkpoint,
            opponent_device=self.opponent_device,
        )
        metrics = evaluate_strategy(
            "current",
            CurrentModelPolicy(self.model),
            env,
            self.eval_episodes,
            self.eval_seed,
        )
        self._log(metrics)
        self._save_best(metrics)
        return True

    def _log(self, metrics) -> None:
        if self.verbose:
            print(
                "eval "
                f"steps={self.num_timesteps} "
                f"win={metrics.win_rate:.3f} "
                f"loss={metrics.loss_rate:.3f} "
                f"final={metrics.avg_final_hand:.2f} "
                f"score={metrics.composite_score:.2f} "
                f"laat_hi={metrics.laat_high_card_rate:.3f}"
            )
        if not self.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            return
        try:
            wandb.log(
                {
                    "eval/win_rate": metrics.win_rate,
                    "eval/loss_rate": metrics.loss_rate,
                    "eval/avg_final_hand": metrics.avg_final_hand,
                    "eval/avg_turns": metrics.avg_turns,
                    "eval/composite_score": metrics.composite_score,
                    "eval/invalid_actions": metrics.invalid_actions,
                    "eval/laat_rank_percentile": metrics.laat.avg_rank_percentile,
                    "eval/laat_high_card_rate": metrics.laat_high_card_rate,
                    "eval/laat_low_card_rate": metrics.laat_low_card_rate,
                    "eval/lead_rank_percentile": metrics.lead_open.avg_rank_percentile,
                    "eval/follow_rank_percentile": metrics.follow_suit.avg_rank_percentile,
                    "eval/timesteps": self.num_timesteps,
                }
            )
        except Exception as exc:
            if self.verbose:
                print(f"WandB eval logging skipped after error: {exc}")

    def _save_best(self, metrics) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if metrics.composite_score > self.best_composite:
            self.best_composite = metrics.composite_score
            self.model.save(self.model_dir / "best_composite_score")
        if metrics.win_rate > self.best_win_rate:
            self.best_win_rate = metrics.win_rate
            self.model.save(self.model_dir / "best_win_rate")
        selection_key = (
            metrics.win_rate,
            -metrics.loss_rate,
            -metrics.avg_final_hand,
            -metrics.invalid_actions,
        )
        if self.best_selection_key is None or selection_key > self.best_selection_key:
            self.best_selection_key = selection_key
            self.model.save(self.model_dir / "best_final_candidate")


class CurrentModelPolicy:
    def __init__(self, model) -> None:
        self.model = model

    def predict(self, env: LaatCardEnv, obs: np.ndarray) -> int:
        from sb3_contrib.common.maskable.utils import get_action_masks

        action, _ = self.model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
        return int(action)
