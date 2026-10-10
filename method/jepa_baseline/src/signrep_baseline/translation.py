"""Feature transport and launch descriptions for the pinned original SLT trainers."""

import csv
import json
from pathlib import Path
import numpy as np
from .data import read_manifest
from .cache import sha256


def export_how2sign(manifest, output_tsv, feature_directory):
    rows = read_manifest(manifest)
    output_tsv, feature_directory = Path(output_tsv), Path(feature_directory)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    feature_directory.mkdir(parents=True, exist_ok=True)
    exported = []
    pins = set()
    for index, row in enumerate(rows):
        if row.get("split") not in {"train", "val", "test"} or "translation" not in row:
            raise ValueError("How2Sign export requires official split and aligned translation")
        with np.load(row["features"], allow_pickle=False) as file:
            metadata = json.loads(str(file["metadata_json"]))
            values = file["features"].copy()
        if metadata.get("schema") != "signrep_features_v1" or metadata["stride"] != 2 or values.ndim != 2 or values.shape[1] != 768:
            raise ValueError("How2Sign requires unpooled stride2 T×768 SignRep features")
        pin = metadata.get("encoder_checkpoint_sha256")
        if not pin:
            raise ValueError("Feature cache is missing encoder checkpoint provenance")
        pins.add(pin)
        path = feature_directory / f"{index:08d}.npy"
        np.save(path, values.astype(np.float32))
        exported.append({"id": row["video_id"], "signs_file": str(path.resolve()), "signs_offset": 0,
                         "signs_length": len(values), "signs_type": "i3d", "translation": row["translation"]})
    if len(pins) != 1:
        raise ValueError("Feature caches use different encoder checkpoints")
    with output_tsv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(exported[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(exported)
    report = {"source_manifest_sha256": sha256(manifest), "tsv_sha256": sha256(output_tsv),
              "encoder_checkpoint_sha256": next(iter(pins)), "input_dimension": 768,
              "transport_label": "i3d denotes original Fairseq npy loader; values are SignRep",
              "model": "signrep_sign2text_transformer (--user-dir integrations/how2sign_user)",
              "samples": len(exported)}
    output_tsv.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2))
    return report
