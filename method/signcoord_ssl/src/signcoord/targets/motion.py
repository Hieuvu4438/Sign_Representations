import torch


def interval_velocity(coords, times, reliability):
    if coords.ndim != 4 or times.shape != coords.shape[:2] or reliability.shape != coords.shape[:-1]:
        raise ValueError("velocity requires coords[B,T,P,C], times[B,T], reliability[B,T,P]")
    dt = times[:, 1:] - times[:, :-1]
    if not torch.isfinite(times).all() or not torch.all(dt > 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    vel = (coords[:, 1:] - coords[:, :-1]) / dt.to(coords.dtype)[..., None, None]
    rel = torch.minimum(reliability[:, 1:], reliability[:, :-1])
    return vel, rel
