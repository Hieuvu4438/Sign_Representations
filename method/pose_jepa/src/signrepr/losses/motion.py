import torch


def motion_targets(batch, stream, reference_fps=25.0, max_gap_sec=0.2):
    g, q = stream["global_xy"], stream["confidence"]
    # Global trajectories do not depend on an observed hand/face local root.
    v = stream.get("global_joint_valid", stream["joint_valid"])
    target = torch.zeros_like(g)
    valid = torch.zeros_like(v)
    weights = torch.zeros_like(q)
    dt = batch["timestamps_sec"][:, 1:] - batch["timestamps_sec"][:, :-1]
    pair = (batch["real"][:, 1:] & batch["real"][:, :-1] & batch["track_continuity"]
            & batch["timing_known"][:, None] & torch.isfinite(dt) & (dt > 0) & (dt <= max_gap_sec))
    valid[:, :-1] = v[:, :-1] & v[:, 1:] & pair[..., None]
    delta = (g[:, 1:] - g[:, :-1]) * ((1/reference_fps)/torch.where(pair, dt, 1)).to(g.dtype)[..., None, None]
    target[:, :-1] = torch.where(valid[:, :-1, :, None], delta, 0)
    weights[:, :-1] = q[:, :-1]*q[:, 1:]
    return target, valid, weights
