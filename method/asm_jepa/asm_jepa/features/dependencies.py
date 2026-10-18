"""Raw dependencies and dependency closure (§9.2).

Raw units are (part, patch) pairs, unit id = part * n_patches + patch (parts in PART_ORDER). A body unit hides
the elbows only (shoulders are public normalisation anchors); hand/face units hide every landmark of the part
at the real frames of the patch. Each token lists the units whose raw values it reads:

  state (a, p)        -> unit (a, p)        (per-frame hand scale: same hand, same frame; anchors public)
  relation (r, p)     -> the two parts of r at patch p (wrists live in hand units, the nose in face units)
  path (f, c, s, v)   -> every patch touched by the interval support, for the hands of family f

context_safe = token_valid AND no raw dependency is masked; targets = valid tokens removed by mask or closure.
"""
from __future__ import annotations

import numpy as np

from asm_jepa.data.schema import N_PARTS, PATH_FAMILIES, PATH_FAMILY_PARTS, RELATION_PARTS
from asm_jepa.features.tokens import TYPE_GEOMETRY, TYPE_RELATION, TYPE_STATE, TokenLayout


def token_dependencies(layout: TokenLayout, support_patches: np.ndarray | None) -> np.ndarray:
    """-> deps [K, N_PARTS * n_patches] bool. support_patches [F,C,S,n_p] (motion only)."""
    n_p = layout.n_patches
    A = layout.arrays
    deps = np.zeros((layout.n_tokens, N_PARTS * n_p), bool)
    for k in range(layout.n_tokens):
        t, sub, p = A["type"][k], A["sub"][k], A["patch"][k]
        if t == TYPE_STATE:
            deps[k, sub * n_p + p] = True
        elif t == TYPE_RELATION:
            for part in RELATION_PARTS[sub]:
                deps[k, part * n_p + p] = True
        else:
            if support_patches is None:
                raise ValueError("motion tokens need interval supports")
            sp = support_patches[sub, A["center"][k], A["scale"][k]]
            for part in PATH_FAMILY_PARTS[PATH_FAMILIES[sub]]:
                deps[k, part * n_p:(part + 1) * n_p] = sp
    return deps


def static_dependencies(layout: TokenLayout) -> np.ndarray:
    """State + relation dependencies (do not depend on timestamps); motion rows are filled per clip."""
    n_p = layout.n_patches
    A = layout.arrays
    deps = np.zeros((layout.n_tokens, N_PARTS * n_p), bool)
    st = A["type"] == TYPE_STATE
    deps[np.flatnonzero(st), A["sub"][st] * n_p + A["patch"][st]] = True
    rel = np.flatnonzero(A["type"] == TYPE_RELATION)
    for k in rel:
        for part in RELATION_PARTS[A["sub"][k]]:
            deps[k, part * n_p + A["patch"][k]] = True
    return deps


def motion_dependencies(layout: TokenLayout, deps: np.ndarray, support_patches: np.ndarray) -> np.ndarray:
    n_p = layout.n_patches
    A = layout.arrays
    motion = np.flatnonzero(A["type"] >= TYPE_GEOMETRY)
    if len(motion) == 0:
        return deps
    sp = support_patches[A["sub"][motion], A["center"][motion], A["scale"][motion]]  # [M, n_p]
    deps[motion] = False
    for i, k in enumerate(motion):
        for part in PATH_FAMILY_PARTS[PATH_FAMILIES[A["sub"][k]]]:
            deps[k, part * n_p:(part + 1) * n_p] = sp[i]
    return deps


def closure(token_valid: np.ndarray, deps: np.ndarray, raw_mask_units: np.ndarray, enabled: bool = True,
            is_state: np.ndarray | None = None):
    """-> context_safe [K], targets [K]. With closure disabled (leakage ablation only) only directly masked
    state tokens are removed — hidden values then reach relation/motion context; never the default."""
    hit = (deps & raw_mask_units[None, :]).any(1)
    if not enabled:
        if is_state is None:
            raise ValueError("closure ablation needs is_state")
        hit &= is_state
    context = token_valid & ~hit
    targets = token_valid & hit
    return context, targets
