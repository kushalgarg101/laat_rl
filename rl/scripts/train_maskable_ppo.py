from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import torch
import wandb

from rl.laat_game.env import LaatCardEnv
from rl.laat_game.bc_policy import copy_bc_to_maskable_ppo, load_bc_policy
from rl.laat_game.training_callbacks import StrategyEvalCallback, WandbPredictionCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--hand-threshold", type=int, default=52)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, default=Path("models/maskable_ppo_laat"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/maskable_ppo_laat"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--reset-num-timesteps", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--init-policy", type=Path, default=None)
    parser.add_argument("--net-hidden-size", type=int, default=256)
    parser.add_argument("--wandb-project", default="laat-card-rl")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--log-predictions-every", type=int, default=250)
    parser.add_argument(
        "--opponent-pool",
        choices=["baseline", "league", "random", "low_card", "high_card", "safe_suit", "pressure_suit", "checkpoint"],
        default="baseline",
    )
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument(
        "--eval-opponent-pool",
        choices=["baseline", "league", "random", "low_card", "high_card", "safe_suit", "pressure_suit", "checkpoint"],
        default="league",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

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
    use_wandb = args.wandb_mode != "disabled"
    run = None
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
            config={
                **vars(args),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        )

    model = build_or_load_model(args, env, device)
    if args.init_policy is not None and not args.resume:
        bc_policy, metadata = load_bc_policy(args.init_policy, device=device)
        copy_bc_to_maskable_ppo(bc_policy, model)
        print(f"Initialized PPO policy from {args.init_policy} metadata={metadata}")
    callback = build_callbacks(args, use_wandb, device)
    model.learn(
        total_timesteps=args.timesteps,
        use_masking=True,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=args.reset_num_timesteps or not args.resume,
    )
    model.save(args.model_dir / "latest")
    if run is not None:
        artifact = wandb.Artifact("maskable-ppo-latest", type="model")
        artifact.add_file(str(args.model_dir / "latest.zip"))
        run.log_artifact(artifact)

    obs, _ = env.reset(seed=args.seed + 1)
    done = False
    while not done:
        action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    if run is not None:
        run.finish()


def build_or_load_model(args: argparse.Namespace, env: LaatCardEnv, device: str) -> MaskablePPO:
    resume_path = args.resume_path or args.model_dir / "latest.zip"
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"Cannot resume because checkpoint does not exist: {resume_path}")
        print(f"Resuming training from {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=env,
            device=device,
            tensorboard_log=str(args.log_dir),
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.gamma = args.gamma
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        return model

    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=device,
        policy_kwargs={"net_arch": [args.net_hidden_size, args.net_hidden_size]},
    )


def build_callbacks(args: argparse.Namespace, use_wandb: bool, device: str = "cpu"):
    callbacks = []
    if args.checkpoint_freq > 0:
        checkpoint_dir = args.model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="laat",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
    if use_wandb:
        callbacks.append(WandbPredictionCallback(log_every=args.log_predictions_every))
    if getattr(args, "eval_every", 0) > 0:
        callbacks.append(
            StrategyEvalCallback(
                eval_freq=args.eval_every,
                eval_episodes=args.eval_episodes,
                eval_seed=args.eval_seed,
                model_dir=args.model_dir,
                players=args.players,
                max_rounds=args.max_rounds,
                hand_threshold=args.hand_threshold,
                opponent_pool=args.eval_opponent_pool,
                opponent_checkpoint=args.opponent_checkpoint,
                opponent_device=device,
                use_wandb=use_wandb,
                verbose=1,
            )
        )
    if not callbacks:
        return None
    return CallbackList(callbacks)


if __name__ == "__main__":
    main()
