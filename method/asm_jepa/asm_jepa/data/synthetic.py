"""In-memory synthetic 133-point pose clips for smoke tests (never reported as ASL-Citizen results).

Each clip: plausible upper-body layout in pixels, smooth wrist trajectories with templated fingers, a face
template, variable length / fps, hand drop-outs, invalid frames, low-score joints and occasional unusable clips
(no shoulders). Deterministic per (seed, index).
"""
from __future__ import annotations

import numpy as np

from asm_jepa.data.pose_store import PoseClip

_HAND_TEMPLATE = None
_FACE_TEMPLATE = None


def _templates():
    global _HAND_TEMPLATE, _FACE_TEMPLATE
    if _HAND_TEMPLATE is None:
        r = np.random.default_rng(1234)
        hand = [np.zeros(2)]
        for finger in range(5):
            ang = np.deg2rad(-60 + 30 * finger)
            for k in range(1, 5):
                hand.append(np.array([np.sin(ang), -np.cos(ang)]) * (0.35 * k + 0.4))
        _HAND_TEMPLATE = np.stack(hand)  # [21,2] in units of the wrist->MCP distance
        th = np.linspace(0, 2 * np.pi, 68, endpoint=False)
        _FACE_TEMPLATE = np.stack([0.5 * np.sin(th), 0.6 * np.cos(th)], -1) + r.normal(0, 0.03, (68, 2))
        _FACE_TEMPLATE[30] = [0.0, 0.0]  # nose tip at the face centre
    return _HAND_TEMPLATE, _FACE_TEMPLATE


def synthetic_clip(index: int, seed: int = 0, unusable_every: int = 37) -> PoseClip:
    rng = np.random.default_rng([seed, index, 99])
    hand_t, face_t = _templates()
    T = int(rng.integers(12, 160))
    fps = float(rng.choice([25.0, 29.97, 30.0]))
    ts = np.arange(T) / fps + np.cumsum(rng.uniform(0, 0.002, T))  # mildly variable frame times
    W, H = 640, 480
    sb = rng.uniform(90, 150)  # shoulder distance in px
    root = np.array([W / 2, H * 0.62]) + rng.normal(0, 10, 2)
    xy = np.zeros((T, 133, 2))
    t = np.linspace(0, 1, T)
    root_t = root + np.stack([3 * np.sin(2 * np.pi * t), 2 * np.cos(2 * np.pi * t)], -1)
    xy[:, 5] = root_t + [sb / 2, 0]   # person-left shoulder appears on image right
    xy[:, 6] = root_t - [sb / 2, 0]
    xy[:, 7] = xy[:, 5] + [0.25 * sb, 0.8 * sb]
    xy[:, 8] = xy[:, 6] + [-0.25 * sb, 0.8 * sb]
    phase = rng.uniform(0, 2 * np.pi, 4)
    freq = rng.uniform(0.5, 2.5, 4)
    for hand, base, side in ((0, 91, 1.0), (1, 112, -1.0)):
        center = root_t + [side * 0.4 * sb, -0.3 * sb]
        wrist = center + sb * 0.35 * np.stack([np.sin(2 * np.pi * freq[2 * hand] * t + phase[2 * hand]),
                                               np.cos(2 * np.pi * freq[2 * hand + 1] * t + phase[2 * hand + 1])], -1)
        ang = 0.6 * np.sin(2 * np.pi * t * freq[hand] + phase[hand])
        rot = np.stack([np.stack([np.cos(ang), -np.sin(ang)], -1), np.stack([np.sin(ang), np.cos(ang)], -1)], -2)
        pts = np.einsum("tij,kj->tki", rot, hand_t) * (0.18 * sb)
        xy[:, base:base + 21] = wrist[:, None] + pts
        xy[:, 9 + hand] = wrist
    face_c = root_t + [0, -0.9 * sb]
    xy[:, 23:91] = face_c[:, None] + face_t[None] * 0.6 * sb
    xy[:, 0:5] = face_c[:, None] + rng.normal(0, 3, (5, 2))
    xy[:, 11:23] = root_t[:, None] + [0, 2 * sb]
    xy += rng.normal(0, 0.5, xy.shape)
    score = rng.uniform(2.0, 9.0, (T, 133)).astype(np.float32)
    observed = np.ones((T, 133), bool)
    observed[:, 11:23] = False
    # hand drop-outs
    for base in (91, 112):
        if rng.random() < 0.6:
            s = int(rng.integers(0, T))
            e = min(T, s + int(rng.integers(1, max(2, T // 3))))
            observed[s:e, base:base + 21] = False
            score[s:e, base:base + 21] = rng.uniform(0.0, 0.1)
    frame_valid = np.ones(T, bool)
    if T > 10 and rng.random() < 0.3:
        bad = rng.choice(T, size=max(1, T // 20), replace=False)
        frame_valid[bad] = False
        observed[bad] = False
    if unusable_every and index % unusable_every == unusable_every - 1:
        observed[:, 5:7] = False
    xy = np.where(observed[..., None], xy, np.nan).astype(np.float32)
    return PoseClip(xy=xy, score=score, observed=observed, timestamps_sec=ts.astype(np.float64),
                    frame_valid=frame_valid, image_wh=np.array([W, H], np.int32),
                    source_frame_index=np.arange(T, dtype=np.int32),
                    meta={"synthetic": True, "seed": seed, "index": index, "timestamp_mode": "synthetic"})


class SyntheticSource:
    def __init__(self, n: int, seed: int = 0):
        self.n, self.seed = n, seed
        self.rows = [{"sample_id": f"synthetic_{i:06d}", "class_id": i % 10, "signer_id": f"S{i % 4}"} for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return synthetic_clip(i, self.seed)
