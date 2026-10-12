"""Versioned predictions and declared detector-quality buckets, never calibrated confidence."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class PredictionRecord:
    schema_version: int
    sample_id: str
    dataset: str
    split: str
    reference_label_id: int | None
    prediction_label_id: int
    top5_label_ids: tuple[int, ...]
    source_group_id: str
    signer_id: str | None
    quality_bucket: str
    model_hash: str
    protocol_signature: str
    class_ids: tuple[int, ...]
    logits: tuple[float, ...]

    def as_dict(self):
        value = asdict(self)
        # Retain the phase3 names for consumers while publishing the proposal names.
        value.update(reference_id=self.reference_label_id, predicted_id=self.prediction_label_id)
        return value


def quality_summary(bundle):
    coordinate_fraction = float(bundle.pose_coordinate_valid.mean())
    visual_fraction = float(bundle.visual_valid.mean())
    # Fixed uncalibrated availability thresholds; do not tune on evaluation labels.
    bucket = "high" if coordinate_fraction >= .9 and visual_fraction >= .9 else "medium" if coordinate_fraction >= .5 and visual_fraction >= .5 else "low"
    return {"bucket": bucket, "coordinate_valid_fraction": coordinate_fraction, "visual_valid_fraction": visual_fraction,
            "kind": "uncalibrated_observation_availability", "thresholds": {"high": .9, "medium": .5},
            "quality_semantics": bundle.meta["quality_kind"]}


def prediction_records(corpus, logits, classes, model_hash, protocol_signature):
    scores = np.asarray(logits)
    if scores.shape != (len(corpus.available), len(classes)) or not np.isfinite(scores).all():
        raise ValueError("Predictions do not match the locked sample/class order")
    records, qualities = [], {}
    for i, name in enumerate(corpus.available):
        annotation = corpus.records[name]
        quality = quality_summary(corpus.load(name))
        qualities[name] = quality
        order = np.argsort(-scores[i], kind="stable")
        records.append(PredictionRecord(1, name, annotation.dataset, annotation.split, annotation.label_id,
            int(classes[int(order[0])]), tuple(int(classes[j]) for j in order[:min(5, len(classes))]),
            annotation.source_group_id, annotation.signer_id, quality["bucket"], model_hash, protocol_signature,
            tuple(int(c) for c in classes), tuple(float(x) for x in scores[i])).as_dict())
    return records, qualities


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)
