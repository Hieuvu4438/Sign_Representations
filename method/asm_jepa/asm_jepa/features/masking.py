"""Raw target masks over valid state part-time units (§9.1).

Mixture per sample: random units (50%), contiguous per-part blocks of 2-4 patches (30%), temporal blocks of
2-3 patches across all parts (20%); blocks stop before overshooting the target count and random valid units
fill the remainder. Constraints: >= 25% of a hand's valid patches stay visible when the hand has evidence,
>= min_visible_state_tokens visible state units overall. Up to max_resample_attempts tries per ratio, then the
ratio shrinks by ratio_fallback_factor (logged). The sampler only reads validity flags, never coordinates.
"""
from __future__ import annotations

import math

import numpy as np

from asm_jepa.data.schema import N_PARTS

STRATEGIES = ("random", "part_blocks", "temporal_blocks")


def _fill_random(mask, valid, n_target, rng):
    free = np.flatnonzero(valid.ravel() & ~mask.ravel())
    need = n_target - int(mask.sum())
    if need > 0 and len(free):
        pick = rng.choice(free, size=min(need, len(free)), replace=False)
        flat = mask.ravel()
        flat[pick] = True
    return mask


def _propose(valid: np.ndarray, n_target: int, strategy: str, rng, cfg) -> np.ndarray:
    n_parts, n_p = valid.shape
    mask = np.zeros_like(valid)
    if strategy == "part_blocks":
        parts = np.flatnonzero(valid.any(1))
        for _ in range(4 * n_p * n_parts):
            if mask.sum() >= n_target or len(parts) == 0:
                break
            a = int(rng.choice(parts))
            L = int(rng.integers(cfg["block_min_patches"], cfg["block_max_patches"] + 1))
            s = int(rng.integers(0, max(n_p - L, 0) + 1))
            add = valid[a, s:s + L] & ~mask[a, s:s + L]
            if mask.sum() + add.sum() > n_target:
                break
            mask[a, s:s + L] |= add
    elif strategy == "temporal_blocks":
        for _ in range(4 * n_p):
            if mask.sum() >= n_target:
                break
            L = int(rng.integers(cfg["temporal_block_min_patches"], cfg["temporal_block_max_patches"] + 1))
            s = int(rng.integers(0, max(n_p - L, 0) + 1))
            add = valid[:, s:s + L] & ~mask[:, s:s + L]
            if mask.sum() + add.sum() > n_target:
                break
            mask[:, s:s + L] |= add
    elif strategy != "random":
        raise ValueError(strategy)
    return _fill_random(mask, valid, n_target, rng)


def _ok(mask, valid, cfg, hand_parts=(1, 2)) -> bool:
    visible = valid & ~mask
    if visible.sum() < min(cfg["min_visible_state_tokens"], int(valid.sum())):
        return False
    if valid.any() and visible.sum() < 1:
        return False
    for a in hand_parts:
        nv = int(valid[a].sum())
        if nv and visible[a].sum() < math.ceil(cfg["hand_visible_fraction_min"] * nv):
            return False
    return True


def sample_raw_mask(state_valid: np.ndarray, rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, dict]:
    """state_valid [N_PARTS, n_p] -> raw mask units [N_PARTS, n_p] bool + info."""
    assert state_valid.shape[0] == N_PARTS
    probs = np.array([cfg["mixture_random"], cfg["mixture_part_blocks"], cfg["mixture_temporal_blocks"]], float)
    probs = probs / probs.sum()
    strategy = STRATEGIES[int(rng.choice(3, p=probs))]
    n_valid = int(state_valid.sum())
    ratio = float(cfg["ratio"])
    info = {"strategy": strategy, "fallbacks": 0, "attempts": 0, "ratio_used": ratio}
    for fb in range(int(cfg["max_ratio_fallbacks"]) + 1):
        n_target = int(round(ratio * n_valid))
        if n_target == 0:
            break
        for _ in range(int(cfg["max_resample_attempts"])):
            info["attempts"] += 1
            mask = _propose(state_valid, n_target, strategy, rng, cfg)
            if _ok(mask, state_valid, cfg):
                info.update(fallbacks=fb, ratio_used=ratio, n_masked=int(mask.sum()), n_valid=n_valid)
                return mask, info
        ratio *= float(cfg["ratio_fallback_factor"])
    info.update(fallbacks=int(cfg["max_ratio_fallbacks"]) + 1, ratio_used=0.0, n_masked=0, n_valid=n_valid,
                empty=True)
    return np.zeros_like(state_valid), info
