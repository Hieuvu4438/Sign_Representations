"""Deterministic geometric tracker. Missing detections retain calendar time."""
import numpy as np
from .schema import PART_INDICES


def bbox_and_quality(xy, scores, threshold=.3):
    ids = PART_INDICES['body']
    v = (scores[ids] >= threshold) & np.isfinite(xy[ids]).all(-1)
    if v.sum() < 5:
        return None
    points = xy[ids][v]
    lo, hi = points.min(0), points.max(0)
    if np.any(hi-lo < 1e-5):
        return None
    return np.r_[lo, hi], float(scores[ids][v].mean())


def iou(a, b):
    overlap = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    area_a, area_b = (a[2:]-a[:2]).prod(), (b[2:]-b[:2]).prod()
    return float(overlap/max(area_a+area_b-overlap, 1e-8))


class PersonTracker:
    def __init__(self, roi=None, min_score=.25, max_center_distance=1.0, max_missing_frames=25):
        self.roi = np.asarray(roi, dtype=np.float32) if roi is not None else None
        self.previous_box = None
        self.previous_center = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_seen = None
        self.frame = -1
        self.min_score, self.max_center_distance = min_score, max_center_distance
        self.max_missing_frames = max_missing_frames
        self.report = {'frames': 0, 'multiple_person_frames': 0, 'missing_frames': 0,
                       'ambiguous_frames': 0, 'track_switches': 0}

    def step(self, keypoints, scores):
        self.frame += 1
        self.report['frames'] += 1
        if keypoints.shape != (len(keypoints), 133, 2) or scores.shape != (len(keypoints), 133):
            raise ValueError('Detector must return [P,133,2] xy and [P,133] confidence')
        if len(keypoints) > 1:
            self.report['multiple_person_frames'] += 1
        candidates = [(i, bbox_and_quality(x, q)) for i, (x, q) in enumerate(zip(keypoints, scores))]
        candidates = [(i, info) for i, info in candidates if info is not None]
        ranked = []
        if self.previous_box is None:
            for i, (box, quality) in candidates:
                if self.roi is not None:
                    score = iou(box, self.roi)*quality
                    if score <= 0:
                        continue
                else:
                    score = float((box[2:]-box[:2]).prod())*quality
                ranked.append((score, i, box))
        elif self.frame-self.last_seen <= self.max_missing_frames:
            gap = self.frame-self.last_seen
            displacement = self.velocity*gap
            predicted = self.previous_box + np.r_[displacement, displacement]
            size = max(float(np.linalg.norm(predicted[2:]-predicted[:2])), 1e-5)
            center = (predicted[:2]+predicted[2:])/2
            for i, (box, quality) in candidates:
                distance = float(np.linalg.norm((box[:2]+box[2:])/2-center))/size
                if distance > self.max_center_distance:
                    continue
                score = .6*iou(predicted, box)+.3*max(0., 1-distance)+.1*quality
                if score >= self.min_score:
                    ranked.append((score, i, box))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            self.report['missing_frames'] += 1
            return None, False
        if len(ranked) > 1 and ranked[0][0]-ranked[1][0] < .03:
            self.report['ambiguous_frames'] += 1
        _, index, box = ranked[0]
        center = (box[:2]+box[2:])/2
        continuous = self.last_seen == self.frame-1
        if self.previous_center is not None:
            self.velocity = (center-self.previous_center)/(self.frame-self.last_seen)
        self.previous_box, self.previous_center, self.last_seen = box, center, self.frame
        return index, continuous
