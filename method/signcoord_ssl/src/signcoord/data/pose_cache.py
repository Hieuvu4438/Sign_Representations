import json
from pathlib import Path

import numpy as np

from signcoord.targets.schema import SCHEMA_HASH


def read_pose_cache(path, expected_media_hash=None):
    with np.load(path, allow_pickle=False) as cache:
        required = ("xy_px", "confidence", "observed", "status", "pts_s", "metadata")
        if any(k not in cache for k in required):
            raise ValueError("incomplete pose cache contract")
        metadata = json.loads(str(cache["metadata"]))
        arrays = {k: cache[k].copy() for k in required if k != "metadata"}
        if 'track_ids' in cache:arrays['track_ids']=cache['track_ids'].copy()
    xy, conf, observed, pts = (arrays[k] for k in ("xy_px", "confidence", "observed", "pts_s"))
    if metadata.get("schema_hash") != SCHEMA_HASH or metadata.get("schema_version") != "pose_cache_v1":
        raise ValueError("pose cache semantic schema mismatch")
    if expected_media_hash is not None and metadata.get("media_sha256") != expected_media_hash:
        raise ValueError("pose belongs to different media content")
    if xy.shape != (len(pts), 133, 2) or conf.shape != (len(pts), 133) or observed.shape != conf.shape:
        raise ValueError("pose cache shapes mismatch")
    if observed.dtype != bool or arrays["status"].shape != conf.shape:
        raise ValueError("pose cache observed/status contract mismatch")
    if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("pose confidence must be a documented score in [0,1]")
    if not np.isfinite(pts).all() or not np.all(np.diff(pts) > 0):
        raise ValueError("pose PTS must increase")
    if metadata.get('tracking_identity_verified'):
        ids=arrays.get('track_ids')
        if ids is None or ids.shape!=(len(pts),) or not np.issubdtype(ids.dtype,np.integer):
            raise ValueError('verified pose tracking requires source-frame integer track IDs')
    return arrays, metadata
