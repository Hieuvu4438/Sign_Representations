"""Frozen stride-two dictionary retrieval, equations11–12 and benchmark metrics."""

from collections import deque
from pathlib import Path
import json
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .augmentation import VideoTransform
from .cache import sha256


def pool_windows(features, activity=None, activity_mode="raw"):
    features = torch.as_tensor(features, dtype=torch.float32)
    if features.ndim != 2 or not len(features) or not torch.isfinite(features).all():
        raise ValueError("Expected nonempty finite window representations")
    if activity is None:
        pooled = features.mean(0)
    else:
        activity = torch.as_tensor(activity, device=features.device, dtype=features.dtype)
        if activity.shape != (len(features), 2) or not torch.isfinite(activity).all():
            raise ValueError("Activity must have finite LH/RH outputs per window")
        weights = activity.max(dim=1).values
        if activity_mode == "clamp_0_1":
            weights = weights.clamp(0, 1)
        elif activity_mode != "raw":
            raise ValueError("Unknown activity-weight convention")
        denominator = weights.sum()
        if not torch.isfinite(denominator) or denominator.abs() <= torch.finfo(weights.dtype).eps:
            raise ValueError("Activity-weight denominator is zero/nonfinite (U08)")
        pooled = (features * weights[:, None]).sum(0) / denominator
    if not torch.isfinite(pooled).all() or pooled.norm() <= torch.finfo(pooled.dtype).eps:
        raise ValueError("Cannot normalize zero/nonfinite pooled features")
    return F.normalize(pooled, dim=0)


@torch.inference_mode()
def extract_windows(model, video, stride=2, batch_size=2, bbox=None, short_policy="reject"):
    if model.frames != 16 or stride < 1 or batch_size < 1 or short_policy not in {"reject", "repeat_last"}:
        raise ValueError("Extraction requires 16-frame encoder and explicit valid window policy")
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    transform = VideoTransform(False)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode {video}")
    buffer, pending, pending_starts = deque(maxlen=16), [], []
    outputs, activities, starts = [], [], []

    def flush():
        if not pending:
            return
        images = torch.stack(pending).to(device)
        keep = torch.ones(len(images), math.prod(model.backbone.mask_spatial_shape), device=device, dtype=torch.bool)
        result = model.encode(images, mask=keep, mask_ratio=0)
        # Cache transport remains float32 even when caller uses BF16 inference.
        outputs.append(result["features"].float().cpu())
        activities.append(model.active_head(result["features"]).float().cpu())
        starts.extend(pending_starts)
        pending.clear()
        pending_starts.clear()

    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if bbox is not None:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                if not 0 <= x0 < x1 <= frame.shape[1] or not 0 <= y0 < y1 <= frame.shape[0]:
                    raise ValueError("Invalid inference signer crop")
                frame = frame[y0:y1, x0:x1]
            buffer.append(frame)
            frames += 1
            if frames >= 16 and (frames - 16) % stride == 0:
                pending.append(transform(np.stack(buffer)))
                pending_starts.append(frames - 16)
                if len(pending) == batch_size:
                    flush()
        if frames < 16:
            if not frames or short_policy == "reject":
                raise ValueError("Video shorter than 16 frames; set explicit short policy (U06)")
            clip = list(buffer)
            clip.extend([clip[-1]] * (16 - len(clip)))
            pending.append(transform(np.stack(clip)))
            pending_starts.append(0)
        flush()
    finally:
        capture.release()
        model.train(training)
    return {"features": torch.cat(outputs).numpy(), "activity": torch.cat(activities).numpy(),
            "window_starts": np.array(starts), "metadata_json": np.array(json.dumps({
                "schema": "signrep_features_v1", "video_sha256": sha256(video), "decoded_frames": frames,
                "stride": stride, "window_frames": 16, "short_policy": short_policy, "bbox": bbox}))}


def feature_record(row, weighted=True, activity_mode="raw"):
    with np.load(row["features"], allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("schema") != "signrep_features_v1" or metadata["stride"] != 2:
            raise ValueError("Benchmark feature cache requires SignRep stride2 windows")
        return pool_windows(data["features"], data["activity"] if weighted else None, activity_mode)


def ranking_metrics(class_ranking, labels):
    """ASL-Citizen eval_metrics: DCG=1/log2(rank+1), MRR=1/rank (one-based)."""
    ranking, labels = torch.as_tensor(class_ranking), torch.as_tensor(labels)
    if ranking.ndim != 2 or labels.shape != (len(ranking),):
        raise ValueError("Invalid ranking/labels")
    matches = ranking == labels[:, None]
    if not (matches.sum(1) == 1).all():
        raise ValueError("Each query label must occur exactly once in class ranking")
    rank = matches.long().argmax(1) + 1
    scores = {"per_instance_top1": (rank <= 1).float().mean().item() * 100,
              "per_instance_top5": (rank <= 5).float().mean().item() * 100,
              "MRR": rank.float().reciprocal().mean().item(),
              "DCG": (rank.float() + 1).log2().reciprocal().mean().item()}
    for k in [1, 5]:
        scores[f"per_class_top{k}"] = torch.stack([
            (rank[labels == label] <= k).float().mean() for label in labels.unique()
        ]).mean().item() * 100
    return scores


def dictionary_retrieval(gallery_features, gallery_labels, query_features, query_labels,
                         gallery_ids, query_ids, split="test"):
    if split not in {"val", "test"} or set(gallery_ids) & set(query_ids):
        raise ValueError("Gallery/query must be disjoint training versus val/test videos")
    gallery, query = torch.as_tensor(gallery_features).float(), torch.as_tensor(query_features).float()
    labels, truth = torch.as_tensor(gallery_labels), torch.as_tensor(query_labels)
    if gallery.ndim != 2 or query.ndim != 2 or gallery.shape[1] != query.shape[1] or not len(query):
        raise ValueError("Invalid gallery/query shapes")
    if len(gallery) != len(labels) or len(query) != len(truth) or len(gallery_ids) != len(gallery) or len(query_ids) != len(query):
        raise ValueError("Feature/label/ID counts differ")
    if not torch.isfinite(gallery).all() or not torch.isfinite(query).all() or (gallery.norm(dim=1) == 0).any() or (query.norm(dim=1) == 0).any():
        raise ValueError("Invalid dictionary features")
    similarities = F.normalize(query, dim=1) @ F.normalize(gallery, dim=1).T
    video_order = similarities.argsort(dim=1, descending=True, stable=True)
    classes = labels.unique(sorted=True)
    if not set(truth.tolist()) <= set(classes.tolist()):
        raise ValueError("Query class missing from training dictionary")
    # Rank each class by its best individual training video; no class prototype.
    scores, winning_video_indices = [], []
    for label in classes:
        positions = (labels == label).nonzero().flatten()
        maximum = similarities[:, positions].max(1)
        scores.append(maximum.values)
        winning_video_indices.append(positions[maximum.indices])
    class_scores = torch.stack(scores, 1)
    # Preserve original gallery order for tied best-video scores at every rank.
    tie_order = torch.stack(winning_video_indices, 1).argsort(dim=1, stable=True)
    score_order = class_scores.gather(1, tie_order).argsort(dim=1, descending=True, stable=True)
    class_order = tie_order.gather(1, score_order)
    class_ranking = classes[class_order]
    metrics = ranking_metrics(class_ranking, truth)
    video_top5 = (labels[video_order[:, :5]] == truth[:, None]).any(1)
    metrics.update(task="dictionary_retrieval", split=split, queries=len(query), gallery_videos=len(gallery),
                   topk_protocol="distinct classes ranked by best individual dictionary video",
                   per_instance_top5_video_hits=video_top5.float().mean().item() * 100)
    # Exact equation12/nearest-video assignment, including ties resolved by gallery order.
    nearest_labels = labels[video_order[:, 0]]
    metrics["per_instance_top1"] = (nearest_labels == truth).float().mean().item() * 100
    metrics["per_class_top1"] = torch.stack([(nearest_labels[truth == c] == c).float().mean() for c in truth.unique()]).mean().item() * 100
    return metrics, {"nearest_video_indices": video_order[:, 0], "class_ranking": class_ranking,
                     "class_scores": class_scores}
