"""Simple rollout batch container for PPO updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutBatch:
    trick_idx: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.trick_idx.shape[0])

    def iter_minibatches(self, minibatch_size: int, *, generator: torch.Generator):
        indices = torch.randperm(self.size, generator=generator)
        for start in range(0, self.size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            yield (
                self.trick_idx[mb_idx],
                self.actions[mb_idx],
                self.old_log_probs[mb_idx],
                self.returns[mb_idx],
                self.advantages[mb_idx],
            )

