"""Public identities are separate from feature/target values."""
from dataclasses import dataclass
import numpy as np
from .geometry.supports import Primitive


@dataclass(frozen=True)
class Token:
    id: str
    stream: int
    timestamp: float
    values: np.ndarray
    valid: bool
    target_valid: bool
    raw_support: frozenset[Primitive]


@dataclass(frozen=True)
class TokenBundle:
    sample_id: str
    tokens: tuple[Token, ...]
    geometry: tuple


@dataclass(frozen=True)
class RawMask:
    cells: frozenset[Primitive]
    requested_ratio: float
    actual_source_ratio: float
    regime: str
    strategy: str = "synchronized_temporal"
    region: str | None = None

    def __post_init__(self):
        if self.regime not in {"strict_temporal", "cross_modal"} or not 0 < self.requested_ratio < 1 or not 0 <= self.actual_source_ratio <= 1:
            raise ValueError("RawMask requires a declared regime and finite mask fractions")
        if not isinstance(self.cells, frozenset) or any(not isinstance(p, Primitive) for p in self.cells):
            raise ValueError("RawMask cells require a frozen typed primitive set")


@dataclass(frozen=True)
class MaskPlan:
    visible: tuple[int, ...]
    hidden: tuple[int, ...]
    semantic_targets: tuple[int, ...]
    geometry_visible: tuple[int, ...]
    geometry_targets: tuple[int, ...]
    raw_mask: RawMask
    diagnostics: dict
