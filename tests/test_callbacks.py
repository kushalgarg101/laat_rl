from argparse import Namespace

from stable_baselines3.common.callbacks import CallbackList

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
