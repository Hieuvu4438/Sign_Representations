"""64-frame fine-tuning and train-derived class-distribution equation13."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .augmentation import VideoTransform
from .data import read_manifest, decode_indices
from .cache import sha256


def class_distribution(features, labels, classes, temperatures=None, include_self=True):
    features = torch.as_tensor(features, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if features.ndim != 2 or labels.shape != (len(features),) or not torch.isfinite(features).all() or (features.norm(dim=1) == 0).any():
        raise ValueError("Invalid training class features")
    if set(labels.tolist()) != set(range(classes)) or classes < 3:
        raise ValueError("Every class needs training samples; <.5 max probability requires >=3 classes")
    normalized = F.normalize(features, dim=1)
    means = torch.stack([normalized[labels == label].mean(0) for label in range(classes)])
    similarities = means @ means.T  # exactly mean across all sample-pair cosines
    if not include_self:
        for label in range(classes):
            count = (labels == label).sum().item()
            if count < 2:
                raise ValueError("Excluding self similarities requires >=2 samples per class")
            similarities[label, label] = (similarities[label, label] * count**2 - count) / (count * (count - 1))
    grid = torch.as_tensor(temperatures if temperatures is not None else np.arange(1, 101) / 1000,
                           dtype=features.dtype, device=features.device)
    if grid.ndim != 1 or not len(grid) or (grid < .001).any() or (grid > .1).any():
        raise ValueError("Temperature grid must lie within [.001,.1]")
    rows, selected = [], []
    for label in range(classes):
        probabilities = (similarities[label][None] / grid[:, None]).softmax(dim=-1)
        maxima = probabilities.max(-1).values
        feasible = maxima < .5
        if not feasible.any():
            raise ValueError(f"No temperature in configured paper interval meets <.5 for class {label} (U09)")
        index = torch.where(feasible, maxima, torch.full_like(maxima, -1.)).argmax()
        rows.append(probabilities[index])
        selected.append(grid[index])
    return torch.stack(rows).float(), torch.stack(selected).float(), similarities.float()


def recognition_loss(logits, labels, distribution, kappa=.2):
    labels = labels.long()
    table = torch.as_tensor(distribution, dtype=logits.dtype, device=logits.device)
    classes = logits.shape[1]
    if table.shape != (classes, classes) or not torch.isfinite(table).all() or (table < 0).any() or not torch.allclose(table.sum(1), torch.ones(classes, device=table.device), atol=1e-5):
        raise ValueError("Invalid class probability distribution")
    ce = F.cross_entropy(logits, labels, label_smoothing=.1)
    targets = table[labels]
    divergence = F.kl_div(logits.log_softmax(-1), targets, reduction="none").sum(-1)
    kl = (kappa * targets.gather(1, labels[:, None]).squeeze(1) * divergence).mean()
    return ce + kl, {"cross_entropy": ce, "class_distribution": kl}


def save_class_distribution(path, features, labels, classes, train_manifest, temperatures=None, include_self=True,
                            encoder_checkpoint_sha256=None):
    rows = read_manifest(train_manifest, "train")
    if len(rows) != len(features) or [row["label"] for row in rows] != list(labels):
        raise ValueError("Class table must use every training sample in manifest order")
    table, temps, similarities = class_distribution(features, labels, classes, temperatures, include_self)
    metadata = {"schema": "signrep_class_distribution_v1", "split": "train", "train_manifest_sha256": sha256(train_manifest),
                "aggregation": "mean of all normalized sample-pair cosines", "include_self": include_self,
                "temperature_grid": list(temperatures) if temperatures is not None else (np.arange(1, 101) / 1000).tolist()}
    if encoder_checkpoint_sha256:
        metadata["encoder_checkpoint_sha256"] = encoder_checkpoint_sha256
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, distribution=table.numpy(), temperatures=temps.numpy(), similarities=similarities.numpy(),
                        metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    return metadata


def load_class_distribution(path, train_manifest, expected_checkpoint_sha256=None):
    read_manifest(train_manifest, "train")
    with np.load(path, allow_pickle=False) as file:
        metadata = json.loads(str(file["metadata_json"]))
        if metadata.get("schema") != "signrep_class_distribution_v1" or metadata.get("split") != "train" or metadata["train_manifest_sha256"] != sha256(train_manifest):
            raise ValueError("Class table does not match training-only manifest")
        if expected_checkpoint_sha256 and metadata.get("encoder_checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError("Class table derived from a different initial encoder")
        return torch.from_numpy(file["distribution"].copy()), metadata


class RecognitionDataset:
    def __init__(self, manifest, training, short_policy="repeat_last"):
        self.rows = read_manifest(manifest, "train" if training else None)
        self.transform = VideoTransform(training)
        self.training, self.short_policy = training, short_policy
        if short_policy not in {"reject", "repeat_last"}:
            raise ValueError("Unknown recognition short-video policy")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        row = self.rows[index]
        cap = cv2.VideoCapture(row["video"])
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count <= 0 or (count < 64 and self.short_policy == "reject"):
            raise ValueError("Recognition video too short/empty")
        # Uniform temporal bins with independent random train selection; eval bin centers.
        if count >= 64:
            edges = np.linspace(0, count, 65)
            offset = torch.rand(64).numpy() if self.training else np.full(64, .5)
            indices = np.minimum((edges[:-1] + np.diff(edges) * offset).astype(int), count - 1)
        else:
            indices = np.minimum(np.arange(64), count - 1)
        frames = decode_indices(row["video"], indices)
        if "signer_bbox_xyxy" in row:
            x0, y0, x1, y1 = [int(v) for v in row["signer_bbox_xyxy"]]
            if not 0 <= x0 < x1 <= frames.shape[2] or not 0 <= y0 < y1 <= frames.shape[1]:
                raise ValueError("Invalid recognition crop")
            frames = frames[:, y0:y1, x0:x1]
        return self.transform(frames), torch.tensor(row["label"], dtype=torch.long)
