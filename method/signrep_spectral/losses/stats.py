"""Loss sufficient statistics and fixed-denominator reduction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class LossStat:
    """One loss term as (differentiable numerator, detached count)."""

    name: str
    numerator: torch.Tensor
    count: int
    coefficient: float = 1.0

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return self.numerator * 0.0
        return self.numerator / self.count


def reduce_loss_stats(
    stats: Sequence[LossStat], num_configured_terms: int, distributed_context=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Mean of per-term support means with a FIXED configured denominator.

    Empty terms contribute a graph-connected zero and still count in the
    denominator, so missingness never re-weights the remaining terms.
    """
    if distributed_context is not None:
        raise NotImplementedError("DDP reduction is not implemented in the MVP")
    if len(stats) != num_configured_terms:
        raise ValueError(f"expected {num_configured_terms} terms, got {len(stats)}")
    total = None
    logs: Dict[str, float] = {}
    for st in stats:
        term = st.coefficient * st.mean()
        total = term if total is None else total + term
        logs[f"{st.name}/num"] = float(st.numerator.detach())
        logs[f"{st.name}/count"] = int(st.count)
    weight_sum = float(sum(st.coefficient for st in stats))
    return total / weight_sum, logs


def aggregate_epoch(records: List[Dict[str, float]]) -> Dict[str, float]:
    """Support-weighted epoch means from per-batch numerator/count logs."""
    nums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for rec in records:
        for key, val in rec.items():
            if key.endswith("/num"):
                base = key[:-4]
                nums[base] = nums.get(base, 0.0) + val
                counts[base] = counts.get(base, 0) + int(rec[base + "/count"])
    return {k: (nums[k] / counts[k] if counts[k] else float("nan")) for k in nums}
