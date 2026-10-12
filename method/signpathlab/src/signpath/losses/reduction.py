from dataclasses import dataclass
import torch
from ..training.distributed import sum_detached, world_size


@dataclass(frozen=True)
class GroupLoss:
    loss: torch.Tensor
    sums: torch.Tensor    # globally reduced, detached; logging only
    counts: torch.Tensor

    @property
    def mean(self):
        active = self.counts > 0
        return ((self.sums / self.counts.clamp_min(1)) * active).sum() / active.sum().clamp_min(1)


def global_group_mean(errors, group_ids, eligible, num_groups):
    if errors.shape != group_ids.shape or errors.shape != eligible.shape or eligible.dtype != torch.bool:
        raise ValueError("Loss values, group IDs and eligibility must have the same shape")
    if (group_ids < 0).any() or (group_ids >= num_groups).any():
        raise ValueError("Loss group IDs outside declared range")
    values = torch.where(eligible, errors.float(), 0).flatten()
    groups = group_ids.flatten()
    local_sums = errors.new_zeros(num_groups, dtype=torch.float32).scatter_add(0, groups, values)
    local_counts = errors.new_zeros(num_groups, dtype=torch.float32).scatter_add(0, groups, eligible.flatten().float())
    counts = sum_detached(local_counts)
    sums = sum_detached(local_sums)
    active = counts > 0
    # DDP averages gradients; W recovers the true global type-balanced mean.
    loss = world_size() * ((local_sums / counts.clamp_min(1)) * active).sum() / active.sum().clamp_min(1)
    return GroupLoss(loss, sums, counts)
