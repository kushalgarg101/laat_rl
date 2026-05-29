from argparse import Namespace

from sb3_contrib import MaskablePPO

from rl.laat_game.env import LaatCardEnv
from rl.scripts.train_maskable_ppo import build_or_load_model


def test_build_or_load_model_resumes_checkpoint_with_updated_hyperparams(tmp_path):
    env = LaatCardEnv(player_count=4, max_rounds=2, seed=1)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        learning_rate=3e-4,
        gamma=0.99,
        device="cpu",
    )
    checkpoint = tmp_path / "checkpoint.zip"
    model.save(checkpoint)

    args = Namespace(
        resume=True,
        resume_path=checkpoint,
        model_dir=tmp_path,
        log_dir=tmp_path / "runs",
        learning_rate=1e-4,
        gamma=0.995,
        n_steps=64,
        batch_size=32,
        seed=7,
    )
    loaded = build_or_load_model(args, LaatCardEnv(player_count=4, max_rounds=2, seed=2), "cpu")

    assert loaded.learning_rate == 1e-4
    assert loaded.lr_schedule(1.0) == 1e-4
    assert loaded.gamma == 0.995
    assert loaded.n_steps == 64
    assert loaded.batch_size == 32
