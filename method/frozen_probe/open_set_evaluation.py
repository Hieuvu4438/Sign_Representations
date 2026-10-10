"""Explicitly adapted inclusive-tie FAR/CCR counts and grouped trapezoid curve."""
from math import fsum, isnan
from numbers import Integral, Real

import numpy as np


def _ids(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not array.size:
        raise ValueError(name + ' must be nonempty one-dimensional IDs')
    items = array.tolist()
    if any(not isinstance(item, str) or not item for item in values) or len(set(items)) != len(items):
        raise ValueError(name + ' must contain distinct nonempty strings')
    return items


def _classes(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not array.size:
        raise ValueError(name + ' must be nonempty one-dimensional class IDs')
    # Inspect original values too: mixed bool/int must not silently coerce bool.
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) or item < 0 or item > np.iinfo(np.int64).max for item in values):
        raise ValueError(name + ' requires nonnegative integer class IDs')
    return np.array(values, dtype=np.int64, copy=True)


class JointOpenSet:
    """Validate complete ordered predictions before reporting joint rates."""

    def __init__(self, sample_ids, labels, predicted_classes, scores, gallery_classes, expected_ids):
        self.sample_ids = _ids(sample_ids, 'sample_ids')
        if self.sample_ids != _ids(expected_ids, 'expected_ids'):
            raise ValueError('Prediction IDs differ from expected ordered cohort')
        self.labels = _classes(labels, 'labels')
        self.predictions = _classes(predicted_classes, 'predicted_classes')
        self.gallery = _classes(gallery_classes, 'gallery_classes')
        if len(np.unique(self.gallery)) != len(self.gallery):
            raise ValueError('Duplicate gallery class ID')
        array = np.asarray(scores)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
            raise ValueError('Scores must be one-dimensional real numbers')
        self.scores = np.array(array, dtype=np.float64, copy=True)
        if not np.isfinite(self.scores).all():
            raise ValueError('Nonfinite acceptance score')
        if any(len(values) != len(self.sample_ids) for values in (self.labels, self.predictions, self.scores)):
            raise ValueError('Incomplete or mismatched prediction vectors')
        if not np.isin(self.predictions, self.gallery).all():
            raise ValueError('Predicted class outside gallery')
        self.known = np.isin(self.labels, self.gallery)
        self.n_known = int(self.known.sum())
        self.n_unknown = len(self.known) - self.n_known
        if not self.n_known or not self.n_unknown:
            raise ValueError('Joint evaluator requires both known and unknown queries')
        self.correct = self.known & (self.predictions == self.labels)

    def at_threshold(self, threshold):
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real) or isnan(float(threshold)):
            raise ValueError('Threshold must be real scalar, not Boolean/NaN')
        threshold = float(threshold)
        accepted = self.scores >= threshold
        ka = int((accepted & self.known).sum())
        kca = int((accepted & self.correct).sum())
        ua = int((accepted & ~self.known).sum())
        return dict(threshold=threshold if np.isfinite(threshold) else '+inf' if threshold > 0 else '-inf',
                    n_known=self.n_known, n_unknown=self.n_unknown,
                    known_accept_count=ka, known_correct_accept_count=kca, unknown_accept_count=ua,
                    FAR=ua / self.n_unknown, CCR=kca / self.n_known,
                    known_acceptance=ka / self.n_known,
                    closed_known_accuracy=int(self.correct.sum()) / self.n_known)

    def curve(self):
        points = [self.at_threshold(float('inf'))]
        points.extend(self.at_threshold(float(score)) for score in np.unique(self.scores)[::-1])
        area = fsum((right['FAR'] - left['FAR']) * (left['CCR'] + right['CCR']) / 2
                    for left, right in zip(points, points[1:]))
        return dict(points=points, grouped_trapezoid_AUC=area,
                    variant='adapted inclusive >= threshold, grouped ties, closed-accuracy endpoint')
