from dataclasses import dataclass
import numpy as np
from .extract import RawLandmarks


@dataclass(frozen=True)
class ExtractionContext:
    sample_id: str
    source_sha256: str
    source_frame_index: np.ndarray
    landmarks: RawLandmarks
    mirror_metadata: dict
    epsilon_px: float = 1.0
    min_body_visibility: float = .5
