"""Trick-conditioned policy/value network for PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class PolicySample:
    """A sampled policy action and associated PPO bookkeeping values."""

    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


class TrickConditionedPolicy(nn.Module):
    """Policy that maps trick indices to continuous action vectors."""

    def __init__(
        self,
        *,
        num_tricks: int,
        action_dim: int = 42,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self._num_tricks = num_tricks
        self._action_dim = action_dim

        self.embedding = nn.Embedding(num_tricks, embedding_dim)
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._squash_eps = 1e-6

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _encode(self, trick_idx: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.embedding(trick_idx))

    def forward(self, trick_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action means, stds, and state value."""
        h = self._encode(trick_idx)
        means = self.actor_mean(h)
        std = torch.exp(self.log_std).expand_as(means)
        value = self.critic(h).squeeze(-1)
        return means, std, value

    def act(self, trick_idx: torch.Tensor) -> PolicySample:
        """Sample a tanh-squashed action and corrected log-prob."""
        means, std, value = self.forward(trick_idx)
        dist = Normal(means, std)
        pre_tanh_action = dist.rsample()
        action = torch.tanh(pre_tanh_action)
        log_prob = self._squashed_log_prob(
            dist=dist, pre_tanh_action=pre_tanh_action, squashed_action=action
        )
        return PolicySample(action=action, log_prob=log_prob, value=value)

    def act_deterministic(self, trick_idx: torch.Tensor) -> PolicySample:
        """Use the policy mean as the action (for eval/replay)."""
        means, _, value = self.forward(trick_idx)
        action = torch.tanh(means)
        log_prob = torch.zeros_like(value)
        return PolicySample(action=action, log_prob=log_prob, value=value)

    def evaluate_actions(
        self, trick_idx: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PPO terms for fixed actions."""
        means, std, value = self.forward(trick_idx)
        dist = Normal(means, std)
        squashed_action = torch.clamp(
            actions, -1.0 + self._squash_eps, 1.0 - self._squash_eps
        )
        pre_tanh_action = self._atanh(squashed_action)
        log_prob = self._squashed_log_prob(
            dist=dist,
            pre_tanh_action=pre_tanh_action,
            squashed_action=squashed_action,
        )
        # Approximate entropy with the base Gaussian entropy.
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    def _squashed_log_prob(
        self,
        *,
        dist: Normal,
        pre_tanh_action: torch.Tensor,
        squashed_action: torch.Tensor,
    ) -> torch.Tensor:
        """Log-prob with tanh change-of-variables correction."""
        base_log_prob = dist.log_prob(pre_tanh_action)
        correction = torch.log(1.0 - squashed_action.pow(2) + self._squash_eps)
        return (base_log_prob - correction).sum(dim=-1)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable inverse tanh for actions in (-1, 1)."""
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
