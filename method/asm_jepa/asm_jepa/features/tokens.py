"""Token slot layout. Deterministic order: state (part-major, then patch), relations (relation-major, then patch),
motion (family, centre, scale, view). v1 = 64 state + 48 relation = 112 slots, v2 = 112 + 72 motion = 184.

Group ids for centred CE: 0-3 state parts, 4-6 relations, 7 geometry, 8 timing (9 centres).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from asm_jepa.data.schema import N_PARTS, N_RELATIONS, PATH_FAMILIES

TYPE_STATE, TYPE_RELATION, TYPE_GEOMETRY, TYPE_TIMING = 0, 1, 2, 3
N_GROUPS = 9
GROUP_NAMES = ["state_body", "state_lh", "state_rh", "state_face", "rel_lh_rh", "rel_lh_nose", "rel_rh_nose",
               "geometry", "timing"]
FAMILY_OF_GROUP = ["state"] * 4 + ["relation"] * 3 + ["geometry", "timing"]


@dataclass
class TokenLayout:
    n_patches: int
    relations: bool
    geometry: bool
    timing: bool
    n_centers: int = 4
    n_scales: int = 3
    n_families: int = len(PATH_FAMILIES)
    arrays: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        n_p = self.n_patches
        rows = []  # (type, sub, patch, center, scale, view)
        for a in range(N_PARTS):
            for p in range(n_p):
                rows.append((TYPE_STATE, a, p, -1, -1, -1))
        if self.relations:
            for r in range(N_RELATIONS):
                for p in range(n_p):
                    rows.append((TYPE_RELATION, r, p, -1, -1, -1))
        views = [v for v, on in ((TYPE_GEOMETRY, self.geometry), (TYPE_TIMING, self.timing)) if on]
        for f in range(self.n_families):
            for c in range(self.n_centers):
                for s in range(self.n_scales):
                    for v in views:
                        rows.append((v, f, -1, c, s, v - TYPE_GEOMETRY))
        arr = np.array(rows, dtype=np.int64)
        t = arr[:, 0]
        group = np.where(t == TYPE_STATE, arr[:, 1], 0)
        group = np.where(t == TYPE_RELATION, 4 + arr[:, 1], group)
        group = np.where(t == TYPE_GEOMETRY, 7, group)
        group = np.where(t == TYPE_TIMING, 8, group)
        subgroup = np.where(t >= TYPE_GEOMETRY, arr[:, 1] * self.n_scales + arr[:, 4], 0)
        self.arrays = {"type": t, "sub": arr[:, 1], "patch": arr[:, 2], "center": arr[:, 3], "scale": arr[:, 4],
                       "view": arr[:, 5], "group": group, "subgroup": subgroup}

    @property
    def n_tokens(self) -> int:
        return len(self.arrays["type"])

    @property
    def n_state(self) -> int:
        return N_PARTS * self.n_patches

    @property
    def n_motion(self) -> int:
        return int((self.arrays["type"] >= TYPE_GEOMETRY).sum())

    def slots(self, type_id: int) -> np.ndarray:
        return np.flatnonzero(self.arrays["type"] == type_id)

    @property
    def motion_enabled(self) -> bool:
        return self.geometry or self.timing

    def active_groups(self) -> list[int]:
        return sorted(set(self.arrays["group"].tolist()))


def layout_from_config(cfg: dict) -> TokenLayout:
    d, f = cfg["data"], cfg["features"]
    if d["max_frames"] % d["patch_frames"]:
        raise ValueError("max_frames must be divisible by patch_frames")
    if list(f["path_families"]) != PATH_FAMILIES:
        raise ValueError(f"path_families must be {PATH_FAMILIES} (family order fixes token ids)")
    if f["logsignature_level"] != 2:
        raise ValueError("only level-2 log-signatures are implemented (level 3 is an extension)")
    if f["path_gap_policy"] != "strict_all_observed":
        raise ValueError("only path_gap_policy=strict_all_observed is implemented")
    return TokenLayout(n_patches=d["max_frames"] // d["patch_frames"], relations=bool(f["relations"]),
                       geometry=bool(f["geometry_paths"]), timing=bool(f["timing_paths"]),
                       n_centers=int(f["centers_per_clip"]), n_scales=len(f["interval_widths_sec"]))
