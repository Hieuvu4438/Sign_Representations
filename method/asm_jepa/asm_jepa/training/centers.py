"""Per-group teacher centres for centred CE (§10.1).

Raw teacher features z (before subtraction) are accumulated as confidence-weighted sums and weight totals over
all microbatches of an optimizer step (and all ranks), then each group with a positive total is updated once:
c_g <- beta * c_g + (1 - beta) * sum_g / count_g. Groups without targets keep their centre. FP32, no gradients.
"""
from __future__ import annotations

import torch
from torch import nn

from asm_jepa.training import distributed as dist_utils


class Centers(nn.Module):
    def __init__(self, n_groups: int, dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(n_groups, dim, dtype=torch.float32))
        self.register_buffer("updates", torch.zeros(n_groups, dtype=torch.long))
        self._sum = None
        self._count = None
        self.reset_accumulators()

    def reset_accumulators(self):
        self._sum = torch.zeros_like(self.center)
        self._count = torch.zeros(self.center.shape[0], dtype=torch.float64, device=self.center.device)

    @torch.no_grad()
    def accumulate(self, z: torch.Tensor, groups: torch.Tensor, weights: torch.Tensor) -> None:
        if self._sum.device != self.center.device:
            self.reset_accumulators()
        w = weights.float()
        self._sum.index_add_(0, groups, z.float() * w[:, None])
        self._count.index_add_(0, groups, w.double())

    @torch.no_grad()
    def step(self) -> None:
        s, c = dist_utils.all_reduce_sum(self._sum), dist_utils.all_reduce_sum(self._count)
        has = c > 0
        if has.any():
            mean = s[has] / c[has, None].float()
            self.center[has] = self.momentum * self.center[has] + (1.0 - self.momentum) * mean
            self.updates[has] += 1
        self.reset_accumulators()
