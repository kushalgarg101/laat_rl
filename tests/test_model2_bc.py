from argparse import Namespace

import numpy as np
from sb3_contrib import MaskablePPO
import torch

from rl.laat_game.bc_policy import BehaviorClonePolicy, copy_bc_to_maskable_ppo, load_bc_policy, save_bc_policy
from rl.laat_game.env import LaatCardEnv, OBS_SIZE
from rl.scripts.train_behavior_clone import train_behavior_clone


def test_bc_policy_predict_respects_mask():
    policy = BehaviorClonePolicy(OBS_SIZE, hidden_size=256)
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(52, dtype=bool)
    mask[7] = True

    assert policy.predict(obs, mask) == 7


def test_bc_policy_save_and_load_roundtrip(tmp_path):
    policy = BehaviorClonePolicy(OBS_SIZE, hidden_size=256)
    path = tmp_path / "policy.pt"
    save_bc_policy(path, policy, {"source": "test"})

    loaded, metadata = load_bc_policy(path)

    assert metadata["source"] == "test"
    assert loaded.obs_size == OBS_SIZE
    assert loaded.hidden_size == 256


def test_copy_bc_weights_to_maskable_ppo_policy():
    env = LaatCardEnv(seed=4)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        device="cpu",
        policy_kwargs={"net_arch": [256, 256]},
    )
    bc_policy = BehaviorClonePolicy(OBS_SIZE, hidden_size=256)

    copy_bc_to_maskable_ppo(bc_policy, model)

    bc_layers = [layer for layer in bc_policy.net if isinstance(layer, torch.nn.Linear)]
    ppo_layers = [layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, torch.nn.Linear)]
    assert torch.allclose(ppo_layers[0].weight.cpu(), bc_layers[0].weight)
    assert torch.allclose(model.policy.action_net.bias.cpu(), bc_layers[2].bias)


def test_train_behavior_clone_one_epoch_on_tiny_dataset(tmp_path):
    obs = np.zeros((8, OBS_SIZE), dtype=np.float32)
    masks = np.zeros((8, 52), dtype=bool)
    actions = np.zeros(8, dtype=np.int64)
    for idx in range(8):
        action = idx % 4
        masks[idx, action] = True
        actions[idx] = action
    data_path = tmp_path / "tiny.npz"
    np.savez_compressed(
        data_path,
        observations=obs,
        masks=masks,
        actions=actions,
        teacher_ids=np.zeros(8, dtype=np.int64),
        episode_ids=np.zeros(8, dtype=np.int64),
        turns=np.arange(8, dtype=np.int64),
    )

    args = Namespace(
        data=data_path,
        model_dir=tmp_path / "model",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden_size=256,
        val_split=0.25,
        seed=1,
        device="cpu",
        wandb_mode="disabled",
        wandb_project="test",
        wandb_run_name="test",
    )
    best_path = train_behavior_clone(args, "cpu")

    assert best_path.exists()
    assert (tmp_path / "model" / "latest.pt").exists()


def test_train_behavior_clone_survives_wandb_init_error(tmp_path, monkeypatch):
    class BrokenWandb:
        def init(self, **kwargs):
            raise RuntimeError("wandb unavailable")

    import rl.scripts.train_behavior_clone as train_script

    monkeypatch.setattr(train_script, "wandb", BrokenWandb())
    obs = np.zeros((8, OBS_SIZE), dtype=np.float32)
    masks = np.zeros((8, 52), dtype=bool)
    actions = np.zeros(8, dtype=np.int64)
    for idx in range(8):
        masks[idx, 0] = True
    data_path = tmp_path / "tiny.npz"
    np.savez_compressed(
        data_path,
        observations=obs,
        masks=masks,
        actions=actions,
        teacher_ids=np.zeros(8, dtype=np.int64),
        episode_ids=np.zeros(8, dtype=np.int64),
        turns=np.arange(8, dtype=np.int64),
    )
    args = Namespace(
        data=data_path,
        model_dir=tmp_path / "model",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden_size=256,
        val_split=0.25,
        seed=1,
        device="cpu",
        wandb_mode="online",
        wandb_project="test",
        wandb_run_name="test",
    )

    best_path = train_behavior_clone(args, "cpu")

    assert best_path.exists()
