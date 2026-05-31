from argparse import Namespace
from types import SimpleNamespace

from stable_baselines3.common.callbacks import CallbackList

from rl.laat_game.training_callbacks import StrategyEvalCallback
from rl.scripts.train_maskable_ppo import build_callbacks


def test_build_callbacks_includes_checkpoint_callback(tmp_path):
    args = Namespace(
        checkpoint_freq=100,
        model_dir=tmp_path,
        log_predictions_every=50,
    )
    callbacks = build_callbacks(args, use_wandb=True)

    assert isinstance(callbacks, CallbackList)
    assert (tmp_path / "checkpoints").exists()
    assert len(callbacks.callbacks) == 2


def test_build_callbacks_can_disable_all_callbacks(tmp_path):
    args = Namespace(
        checkpoint_freq=0,
        model_dir=tmp_path,
        log_predictions_every=50,
    )
    assert build_callbacks(args, use_wandb=False) is None


def test_strategy_eval_saves_best_final_candidate_by_selection_priority(tmp_path):
    saved_paths = []

    class DummyModel:
        def save(self, path):
            saved_paths.append(path.name)

    callback = StrategyEvalCallback(
        eval_freq=1,
        eval_episodes=1,
        eval_seed=1,
        model_dir=tmp_path,
        players=4,
        max_rounds=2,
        hand_threshold=52,
        opponent_pool="random",
        opponent_checkpoint=None,
        opponent_device="cpu",
        use_wandb=False,
    )
    callback.model = DummyModel()
    first = SimpleNamespace(
        composite_score=1.0,
        win_rate=0.8,
        loss_rate=0.2,
        avg_final_hand=5.0,
        invalid_actions=0,
    )
    better_tiebreak = SimpleNamespace(
        composite_score=0.5,
        win_rate=0.8,
        loss_rate=0.1,
        avg_final_hand=7.0,
        invalid_actions=0,
    )

    callback._save_best(first)
    callback._save_best(better_tiebreak)

    assert saved_paths.count("best_final_candidate") == 2
