"""Explicit video manifests and frame-aligned paired sampling (supplement A2)."""

from collections import OrderedDict
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np
import torch
from .cache import load_standardized, clip_targets, sha256
from .schema import FidelityChoices


def youtube_id(value):
    value = str(value).strip()
    if "://" not in value:
        return value
    parsed = urlparse(value)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0]
    if parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path in {"/watch", "/watch/"}:
            ids = parse_qs(parsed.query).get("v", [])
            if ids:
                return ids[0]
        segments = parsed.path.strip("/").split("/")
        if len(segments) == 2 and segments[0] in {"embed", "shorts", "live"}:
            return segments[1]
    raise ValueError(f"Cannot identify YouTube video: {value}")


def wlasl_youtube_ids(annotation):
    rows = json.loads(Path(annotation).read_text())
    return {youtube_id(instance["url"]) for row in rows for instance in row["instances"]
            if "youtu" in instance.get("url", "").lower()}


def read_manifest(path, required_split=None):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    for row in rows:
        if not row.get("video_id") or row["video_id"] in seen:
            raise ValueError("Manifest requires unique nonempty video_id")
        seen.add(row["video_id"])
        if required_split and row.get("split") != required_split:
            raise ValueError(f"Manifest must contain only {required_split} examples")
        for key in ["video", "pose_cache", "raw_cache", "features"]:
            if key in row:
                candidate = Path(row[key])
                row[key] = str((path.parent / candidate).resolve() if not candidate.is_absolute() else candidate)
    return rows


def source_identity(row):
    """Source video identity: real YouTube ID or a dataset-scoped local ID."""
    if row.get("youtube_id"):
        return "youtube:" + youtube_id(row["youtube_id"])
    if row.get("dataset") and row.get("source_id"):
        return str(row["dataset"]) + ":" + str(row["source_id"])
    raise ValueError("Source identity requires youtube_id or dataset plus source_id")


def deduplicate_local_manifest(source, held_out_manifest, output, training_reference_manifest):
    """Audit local training membership against all official held-out sources."""
    rows = read_manifest(source, "pretrain")
    held_out = read_manifest(held_out_manifest)
    reference = {source_identity(row): row for row in read_manifest(training_reference_manifest, "train")}
    if any(row.get("source_split") != "train" for row in rows):
        raise ValueError("Local pretraining source_split must be train")
    if any(row.get("split") not in {"val", "test"} for row in held_out):
        raise ValueError("Held-out exclusion manifest must contain val/test only")
    for row in rows:
        original = reference.get(source_identity(row))
        if original is None or row.get("video") != original.get("video"):
            raise ValueError("Local pretraining membership differs from official training reference")
    blocked = {source_identity(row) for row in held_out}
    blocked_paths = {row["video"] for row in held_out if row.get("video")}
    blocked_hashes = {row["video_sha256"] for row in held_out if row.get("video_sha256")}
    if any(source_identity(row) in blocked or row.get("video") in blocked_paths or
           row.get("video_sha256") in blocked_hashes for row in rows):
        raise ValueError("Held-out source overlaps local pretraining membership")
    identities = [source_identity(row) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Local pretraining requires unique source videos")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {"audit_kind": "local_train_only", "input_sha256": sha256(source),
              "training_reference_manifest": str(Path(training_reference_manifest).resolve()),
              "training_reference_sha256": sha256(training_reference_manifest),
              "held_out_manifest": str(Path(held_out_manifest).resolve()),
              "held_out_sha256": sha256(held_out_manifest), "output_sha256": sha256(output),
              "excluded_source_ids": sorted(blocked), "excluded_ids": [],
              "removed_video_ids": [], "retained": len(rows)}
    output.with_suffix(output.suffix + ".audit.json").write_text(json.dumps(report, indent=2))
    return report


def deduplicate_manifest(source, wlasl_annotations, output):
    rows = read_manifest(source, "pretrain")
    blocked = wlasl_youtube_ids(wlasl_annotations)
    kept, removed = [], []
    for row in rows:
        if not row.get("youtube_id"):
            raise ValueError("Deduplication requires source YouTube IDs for every row")
        (removed if youtube_id(row["youtube_id"]) in blocked else kept).append(row)
    if not kept:
        raise ValueError("Deduplication removed every pretraining video")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in kept))
    report = {"input_sha256": sha256(source), "wlasl_sha256": sha256(wlasl_annotations),
              "output_sha256": sha256(output), "excluded_ids": sorted(blocked),
              "removed_video_ids": [row["video_id"] for row in removed], "retained": len(kept)}
    output.with_suffix(output.suffix + ".audit.json").write_text(json.dumps(report, indent=2))
    return report


def decode_indices(path, indices):
    """Decode sequentially so codec seeking cannot shift pose supervision by a frame."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or indices.min() < 0:
        raise ValueError("Expected nonempty nonnegative frame indices")
    wanted, decoded = set(indices.tolist()), {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {path}")
    try:
        for index in range(int(indices.max()) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ended before teacher frame {index}: {path}")
            if index in wanted:
                decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return np.stack([decoded[index] for index in indices])


class PairedVideoDataset:
    """One item -> two views; use StatefulPairSampler with num_workers=0 for replay."""
    def __init__(self, manifest, transform, choices=None, cache_capacity=2):
        self.manifest = Path(manifest)
        self.rows = read_manifest(manifest, "pretrain")
        self.manifest_sha256 = sha256(manifest)
        self.transform = transform
        self.choices = choices or FidelityChoices()
        self.cache_capacity = cache_capacity
        self._cache = OrderedDict()
        self._verified = set()
        if len(self.rows) < 2:
            raise ValueError("Style adversary requires at least two distinct source videos")
        if len({source_identity(row) for row in self.rows}) != len(self.rows):
            raise ValueError("Paired sampling requires one row per source video")
        for row in self.rows:
            for key in ["video", "pose_cache", "pose_cache_sha256", "video_sha256", "bone_statistics_sha256", "signer_bbox_xyxy"]:
                if key not in row:
                    raise ValueError(f"Pretraining manifest missing {key}")
            bbox = row["signer_bbox_xyxy"]
            if len(bbox) != 4 or not np.isfinite(bbox).all() or not 0 <= bbox[0] < bbox[2] or not 0 <= bbox[1] < bbox[3]:
                raise ValueError("Signer crop must be explicit nonnegative [x0,y0,x1,y1]")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if index not in self._verified:
            if sha256(row["video"]) != row["video_sha256"]:
                raise ValueError("Video changed since pose extraction")
            if sha256(row["pose_cache"]) != row["pose_cache_sha256"]:
                raise ValueError("Pose cache changed since corpus preparation")
            self._verified.add(index)
        if index not in self._cache:
            data, _ = load_standardized(row["pose_cache"], row["bone_statistics_sha256"], row["video_sha256"])
            self._cache[index] = data
            if len(self._cache) > self.cache_capacity:
                self._cache.popitem(last=False)
        data = self._cache[index]
        self._cache.move_to_end(index)
        valid = np.convolve(data["body_detected"].astype(int), np.ones(16, int), mode="valid") == 16 if len(data["pose_3d"]) >= 16 else np.zeros(0, bool)
        starts = np.flatnonzero(valid)
        if not len(starts):
            raise ValueError(f"No 16-frame signer-present window: {row['video_id']}")
        selected = starts[torch.randint(len(starts), (2,)).numpy()]
        all_indices = np.concatenate([np.arange(start, start + 16) for start in selected])
        frames = decode_indices(row["video"], all_indices)
        x0, y0, x1, y1 = [int(value) for value in row["signer_bbox_xyxy"]]
        if x1 > frames.shape[2] or y1 > frames.shape[1]:
            raise ValueError("Signer crop exceeds video bounds")
        frames = frames[:, y0:y1, x0:x1]
        images, targets, masks = [], [], []
        for view in range(2):
            indices = all_indices[view * 16:(view + 1) * 16]
            target, mask = clip_targets(data, indices, self.choices)
            images.append(self.transform(frames[view * 16:(view + 1) * 16]))
            targets.append(target)
            masks.append(mask)
        return {"images": torch.stack(images), "targets": {k: torch.stack([t[k] for t in targets]) for k in targets[0]},
                "masks": {k: torch.stack([m[k] for m in masks]) for k in masks[0]},
                "video_ids": [row["video_id"]] * 2, "frame_indices": torch.from_numpy(all_indices.reshape(2, 16))}


def collate_pairs(items):
    return {"images": torch.cat([item["images"] for item in items]),
            "targets": {key: torch.cat([item["targets"][key] for item in items]) for key in items[0]["targets"]},
            "masks": {key: torch.cat([item["masks"][key] for item in items]) for key in items[0]["masks"]},
            "video_ids": [key for item in items for key in item["video_ids"]],
            "frame_indices": torch.cat([item["frame_indices"] for item in items])}


class StatefulPairSampler:
    """Persistent unique-video permutations with no prefetch/replayed RNG ambiguity."""
    def __init__(self, size, pairs_per_batch, seed=0):
        if not 2 <= pairs_per_batch <= size:
            raise ValueError("Batch needs >=2 unique videos and cannot exceed dataset")
        self.size, self.pairs_per_batch = size, pairs_per_batch
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor, self.epoch = 0, 0

    def next_indices(self):
        if self.cursor + self.pairs_per_batch > self.size:
            self.order = torch.randperm(self.size, generator=self.generator)
            self.cursor, self.epoch = 0, self.epoch + 1
        result = self.order[self.cursor:self.cursor + self.pairs_per_batch].tolist()
        self.cursor += self.pairs_per_batch
        return result

    def state_dict(self):
        return {"size": self.size, "pairs_per_batch": self.pairs_per_batch,
                "generator": self.generator.get_state(), "order": self.order.clone(),
                "cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        if state["size"] != self.size or state["pairs_per_batch"] != self.pairs_per_batch:
            raise ValueError("Sampler dataset or batch changed on resume")
        self.generator.set_state(state["generator"])
        self.order = state["order"].clone()
        self.cursor, self.epoch = state["cursor"], state["epoch"]
