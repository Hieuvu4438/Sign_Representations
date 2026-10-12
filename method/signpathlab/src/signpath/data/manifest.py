"""Versioned annotations; pretraining receives a deliberately smaller view."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path

from ..cache.hashing import is_sha256


@dataclass(frozen=True)
class PretrainRecord:
    sample_id: str
    video_path: str
    source_group_id: str
    start_s: float
    end_s: float | None
    source_sha256: str | None


@dataclass(frozen=True)
class ManifestRecord:
    schema_version: int
    sample_id: str
    dataset: str
    language: str
    split: str
    source_group_id: str
    video_path: str
    source_sha256: str | None = None
    signer_id: str | None = None
    start_s: float = 0.0
    end_s: float | None = None
    nominal_fps: float | None = None
    label_id: int | None = None
    gloss: str | None = None
    text: str | None = None
    official_split_version: str | None = None
    availability: str = "missing"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.split not in {"train", "val", "test"}:
            raise ValueError("Unsupported manifest schema or split")
        for name in ("sample_id", "dataset", "language", "source_group_id", "video_path"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Manifest requires nonempty {name}")
        if self.source_sha256 is not None and not is_sha256(self.source_sha256):
            raise ValueError("source_sha256 must be a real SHA-256 or null")
        if not math.isfinite(self.start_s) or self.start_s < 0 or (self.end_s is not None and (not math.isfinite(self.end_s) or self.end_s <= self.start_s)):
            raise ValueError("Invalid manifest time interval")
        if self.label_id is not None and (type(self.label_id) is not int or self.label_id < 0):
            raise ValueError("label_id must be a nonnegative integer or null")
        if self.availability not in {"available", "missing", "decode_failed"}:
            raise ValueError("Invalid availability state")
        if self.nominal_fps is not None and (not math.isfinite(self.nominal_fps) or self.nominal_fps <= 0):
            raise ValueError("nominal_fps must be positive or null")

    def pretrain_view(self) -> PretrainRecord:
        if self.split != "train":
            raise ValueError("Pretraining cannot read held-out split")
        return PretrainRecord(self.sample_id, self.video_path, self.source_group_id,
                              self.start_s, self.end_s, self.source_sha256)


def read_manifest(path):
    records = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if set(value) - {f.name for f in fields(ManifestRecord)}:
                    raise ValueError("Unknown manifest fields")
                records.append(ManifestRecord(**value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return records


def write_manifest(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)
