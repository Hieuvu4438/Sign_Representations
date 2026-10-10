from __future__ import annotations

import torch

from .schema import PACKED_RAW, PART_SLICES


def normalized_geometry(xy_px, confidence, observed, frame_valid, image_size,
                        min_shoulder_frames=4, min_scale_fraction=0.02, min_conf=0.30):
    """Raw133 transformed pixels -> packed119 physical shoulder-width units."""
    if xy_px.ndim != 4 or xy_px.shape[-2:] != (133, 2):
        raise ValueError("xy_px must be [B,T,133,2]")
    if confidence.shape != xy_px.shape[:-1] or observed.shape != confidence.shape:
        raise ValueError("pose confidence/observed shape mismatch")
    h, w = image_size
    finite = torch.isfinite(xy_px).all(-1) & torch.isfinite(confidence)
    inside = (xy_px[..., 0] >= 0) & (xy_px[..., 0] < w) & (xy_px[..., 1] >= 0) & (xy_px[..., 1] < h)
    basic = observed.bool() & finite & inside & frame_valid[..., None]
    conf = torch.nan_to_num(confidence).clamp(0, 1)
    xy = torch.where(basic[..., None], xy_px, 0)
    origin = xy.new_zeros(xy.shape[0], 2)
    scale = xy.new_ones(xy.shape[0])
    norm_valid = torch.zeros(xy.shape[0], dtype=torch.bool, device=xy.device)
    shoulder_valid = basic[..., 5] & basic[..., 6] & (conf[..., 5] >= min_conf) & (conf[..., 6] >= min_conf)
    for b in range(len(xy)):
        use = shoulder_valid[b]
        if use.sum() < min_shoulder_frames:
            continue
        centers = (xy[b, use, 5] + xy[b, use, 6]) / 2
        widths = (xy[b, use, 5] - xy[b, use, 6]).norm(dim=-1)
        # torch.quantile gives the midpoint for an even sample count.
        candidate = torch.quantile(widths, 0.5)
        if candidate > min_scale_fraction * min(h, w):
            origin[b] = torch.quantile(centers, 0.5, dim=0)
            scale[b] = candidate
            norm_valid[b] = True
    valid = basic[..., PACKED_RAW] & norm_valid[:, None, None]
    glob = (xy[..., PACKED_RAW, :] - origin[:, None, None, :]) / scale[:, None, None, None]
    glob = torch.where(valid[..., None], glob, 0)
    rel = conf[..., PACKED_RAW] * valid
    return {"global": glob, "joint_valid": valid, "reliability": rel,
            "origin_px": origin, "scale_px": scale, "normalization_valid": norm_valid}


def local_hands(global_coords, reliability):
    coords, rel = {}, {}
    for part in ("L", "R"):
        s = PART_SLICES[part]
        x, r = global_coords[..., s, :], reliability[..., s]
        coords[part] = x[..., 1:, :] - x[..., :1, :]
        rel[part] = torch.minimum(r[..., 1:], r[..., :1])
    return coords, rel
