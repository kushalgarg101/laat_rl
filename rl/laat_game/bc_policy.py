from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class BehaviorClonePolicy(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int = 256, action_size: int = 52) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def predict(self, obs: np.ndarray, mask: np.ndarray, device: str = "cpu") -> int:
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = self(obs_tensor).squeeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~mask_tensor, -1e9)
            return int(torch.argmax(logits).item())


def masked_cross_entropy(logits: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~masks.bool(), -1e9)
    return nn.functional.cross_entropy(masked_logits, actions.long())


def save_bc_policy(path: Path, policy: BehaviorClonePolicy, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_size": policy.obs_size,
            "hidden_size": policy.hidden_size,
            "action_size": policy.action_size,
            "metadata": metadata,
        },
        path,
    )


def load_bc_policy(path: Path, device: str = "cpu") -> tuple[BehaviorClonePolicy, dict]:
    checkpoint = torch.load(path, map_location=device)
    policy = BehaviorClonePolicy(
        obs_size=int(checkpoint["obs_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        action_size=int(checkpoint.get("action_size", 52)),
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, dict(checkpoint.get("metadata", {}))


def copy_bc_to_maskable_ppo(policy: BehaviorClonePolicy, model) -> None:
    policy_layers = [layer for layer in policy.net if isinstance(layer, nn.Linear)]
    ppo_layers = [layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, nn.Linear)]
    value_layers = [layer for layer in model.policy.mlp_extractor.value_net if isinstance(layer, nn.Linear)]
    if len(ppo_layers) < 2 or len(value_layers) < 2:
        raise ValueError("Expected MaskablePPO policy and value networks to have at least two Linear layers.")
    if len(policy_layers) != 3:
        raise ValueError("Expected behavior clone policy to have exactly three Linear layers.")

    with torch.no_grad():
        for bc_layer, ppo_layer, value_layer in zip(policy_layers[:2], ppo_layers[:2], value_layers[:2]):
            if bc_layer.weight.shape != ppo_layer.weight.shape:
                raise ValueError(f"BC layer shape {bc_layer.weight.shape} does not match PPO layer {ppo_layer.weight.shape}.")
            if bc_layer.weight.shape != value_layer.weight.shape:
                raise ValueError(f"BC layer shape {bc_layer.weight.shape} does not match Value layer {value_layer.weight.shape}.")
            ppo_layer.weight.copy_(bc_layer.weight.to(ppo_layer.weight.device))
            ppo_layer.bias.copy_(bc_layer.bias.to(ppo_layer.bias.device))
            value_layer.weight.copy_(bc_layer.weight.to(value_layer.weight.device))
            value_layer.bias.copy_(bc_layer.bias.to(value_layer.bias.device))

        action_layer = policy_layers[2]
        if action_layer.weight.shape != model.policy.action_net.weight.shape:
            raise ValueError(
                f"BC action shape {action_layer.weight.shape} does not match PPO action shape {model.policy.action_net.weight.shape}."
            )
        model.policy.action_net.weight.copy_(action_layer.weight.to(model.policy.action_net.weight.device))
        model.policy.action_net.bias.copy_(action_layer.bias.to(model.policy.action_net.bias.device))
