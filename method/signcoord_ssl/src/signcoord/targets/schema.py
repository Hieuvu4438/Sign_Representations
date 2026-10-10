"""Semantic identities pinned to the official COCO-WholeBody metadata."""
import json
from pathlib import Path

from signcoord.provenance import object_hash

SCHEMA_VERSION = "coco_wholebody133_to_signcoord119_v1"
RAW_NAMES = json.loads(Path(__file__).with_name("canonical_names.json").read_text())
BODY_RAW = (0, 5, 6, 7, 8, 9, 10, 11, 12)
PART_RAW = {"L": tuple(range(91, 112)), "R": tuple(range(112, 133)),
            "F": tuple(range(23, 91)), "B": BODY_RAW}
PACKED_RAW = tuple(i for indices in PART_RAW.values() for i in indices)
PART_SLICES = {}
start = 0
for part, indices in PART_RAW.items():
    PART_SLICES[part] = slice(start, start + len(indices))
    start += len(indices)
ANCHORS = tuple(PACKED_RAW.index(i) for i in (9, 10, 0))
SCHEMA_HASH = object_hash({"version": SCHEMA_VERSION, "raw_names": RAW_NAMES, "packed": PACKED_RAW})
assert start == 119 and ANCHORS == (115, 116, 110)
assert [RAW_NAMES[i] for i in (0, 9, 10)] == ["nose", "left_wrist", "right_wrist"]
assert RAW_NAMES[91] == "left_hand_root" and RAW_NAMES[112] == "right_hand_root"
