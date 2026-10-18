"""Step-indexed schedules (indexed by successful optimizer steps, never microbatches)."""
from __future__ import annotations

import math


def warmup_cosine(step: int, max_steps: int, peak: float, final: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    span = max(max_steps - warmup, 1)
    prog = min(max(step - warmup, 0) / span, 1.0)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * prog))


def ema_momentum(step: int, max_steps: int, start: float, end: float, schedule: str = "cosine") -> float:
    prog = min(max(step / max(max_steps, 1), 0.0), 1.0)
    if schedule == "cosine":
        return end - (end - start) * (math.cos(math.pi * prog) + 1.0) / 2.0
    if schedule == "linear":
        return start + (end - start) * prog
    if schedule == "constant":
        return start
    raise ValueError(f"unknown EMA schedule {schedule}")


def teacher_temperature(step: int, start: float, final: float, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return final
    return start + (final - start) * step / warmup_steps
