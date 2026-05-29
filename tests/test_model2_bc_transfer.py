import pytest
import os
import torch
import numpy as np
from pathlib import Path
from sb3_contrib import MaskablePPO

from rl.laat_game.env import LaatCardEnv
from rl.laat_game.bc_policy import BehaviorClonePolicy, copy_bc_to_maskable_ppo
import torch.nn as nn

def test_bc_to_ppo_transfer():
    # Setup
    env = LaatCardEnv(player_count=4)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_size = 256
    
    # 1. Create a dummy BC model and mutate its weights slightly so we can track them
    bc_model = BehaviorClonePolicy(obs_size=obs_size, hidden_size=hidden_size, action_size=action_size)
    for param in bc_model.parameters():
        param.data.fill_(0.5)

    # 2. Create MaskablePPO model
    ppo_model = MaskablePPO("MlpPolicy", env, n_steps=128, batch_size=32, policy_kwargs=dict(net_arch=[256, 256]))

    # Note down original PPO value head weights for checking later
    orig_value_head_weight = ppo_model.policy.value_net.weight.clone()

    # 3. Initialize PPO from BC
    copy_bc_to_maskable_ppo(bc_model, ppo_model)

    # 4. Confirm:
    # - policy hidden layers copied
    ppo_layers = [layer for layer in ppo_model.policy.mlp_extractor.policy_net if isinstance(layer, nn.Linear)]
    assert torch.allclose(ppo_layers[0].weight, torch.full_like(ppo_layers[0].weight, 0.5))
    assert torch.allclose(ppo_layers[1].weight, torch.full_like(ppo_layers[1].weight, 0.5))

    # - action head copied
    assert torch.allclose(ppo_model.policy.action_net.weight, torch.full_like(ppo_model.policy.action_net.weight, 0.5))

    # - value hidden layers copied
    value_layers = [layer for layer in ppo_model.policy.mlp_extractor.value_net if isinstance(layer, nn.Linear)]
    assert torch.allclose(value_layers[0].weight, torch.full_like(value_layers[0].weight, 0.5))
    assert torch.allclose(value_layers[1].weight, torch.full_like(value_layers[1].weight, 0.5))

    # - value head remains valid (not overwritten with 0.5)
    assert not torch.allclose(ppo_model.policy.value_net.weight, torch.full_like(ppo_model.policy.value_net.weight, 0.5))
    assert torch.allclose(ppo_model.policy.value_net.weight, orig_value_head_weight)

    # 5. Run tiny PPO fine-tune smoke test.
    ppo_model.learn(total_timesteps=128)
