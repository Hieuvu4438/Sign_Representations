"""A5 target-gated normalized log auto-power; no cross-spectrum or phase."""
import torch

from .spectral import window_fft, window_starts


def auto_power(signal, reliability, window, hop, min_conf=.30, eps=1e-6):
    r = reliability.unfold(1, window, hop)  # B,M,P,W
    weights = r.mean(-1) * (r >= min_conf).all(-1)
    u = window_fft(signal, window, hop)
    power = (u.abs().square().sum(3) * weights[..., None]).sum(1)
    power = power / weights.sum(1).clamp_min(eps)[..., None]
    count = (weights > 0).sum(1)
    mean_rel = weights.sum(1) / count.clamp_min(1)
    return power[..., 1:window//2], count, mean_rel, weights


def fit_magnitude_calibration(signals, reliability, window=16, hop=8,
                              min_samples=8, percentile=10, floor=1e-6, min_conf=.30):
    with torch.no_grad():
        power, count, _, _ = auto_power(signals, reliability, window, hop, min_conf)
        shape = power.shape[1:]
        tau, mean, scale = (torch.empty(shape, device=power.device, dtype=power.dtype) for _ in range(3))
        counts = torch.empty(shape, device=power.device, dtype=torch.long)
        for anchor in range(3):
            for k in range(shape[1]):
                values = power[:, anchor, k]
                values = values[(count[:, anchor] >= 2) & torch.isfinite(values) & (values > 0)]
                counts[anchor, k] = len(values)
                if len(values) < min_samples:
                    raise ValueError(f"magnitude calibration failure: anchor={anchor} bin={k+1}: {len(values)} < {min_samples}")
                logs = values.clamp_min(floor).log()
                tau[anchor, k] = torch.quantile(values, percentile/100).clamp_min(floor)
                mean[anchor, k] = logs.mean()
                scale[anchor, k] = logs.std(unbiased=False).clamp_min(.05)
        return {"floor": floor, "tau_auto": tau.tolist(), "log_mean": mean.tolist(),
                "log_scale": scale.tolist(), "counts": counts.tolist(),
                "min_confidence": min_conf, "window": window, "hop": hop,
                "gate": "target_reliability_and_auto_energy_only"}


def magnitude_loss_terms(pred_signal, target_signal, target_rel, calibration, window=16, hop=8):
    if pred_signal.shape != target_signal.shape or target_signal.shape[2:] != (3,2):
        raise ValueError("magnitude expects matching anchor velocities[B,N,3,2]")
    if len(window_starts(pred_signal.shape[1], window, hop)) < 2:
        raise ValueError("magnitude requires two planned windows")
    if (calibration["window"], calibration["hop"]) != (window, hop):
        raise ValueError("magnitude calibration window/hop mismatch")
    with torch.autocast(device_type=pred_signal.device.type, enabled=False):
        dtype = torch.float64 if pred_signal.dtype == torch.float64 else torch.float32
        pred, target = pred_signal.to(dtype), target_signal.detach().to(dtype)
        rel = target_rel.detach().to(dtype)
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("magnitude signals must be finite")
        tau, mean, scale = (pred.new_tensor(calibration[k]) for k in ("tau_auto","log_mean","log_scale"))
        floor = calibration["floor"]
        if floor <= 0 or not torch.isfinite(mean).all() or not ((tau > 0) & torch.isfinite(tau) & (scale > 0) & torch.isfinite(scale)).all():
            raise ValueError("invalid magnitude calibration")
        with torch.no_grad():
            power_t, count, mean_rel, weights = auto_power(target, rel, window, hop, calibration["min_confidence"])
            target_log = (power_t.clamp_min(floor).log()-mean) / scale
            gate = (count >= 2)[...,None] * mean_rel[...,None] * (power_t / tau).clamp(0,1)
        u = window_fft(pred, window, hop)
        power_s = (u.abs().square().sum(3) * weights[...,None]).sum(1)
        power_s = (power_s / weights.sum(1).clamp_min(1e-6)[...,None])[...,1:window//2]
        student_log = (power_s.clamp_min(floor).log()-mean) / scale
        error = (student_log-target_log).square()
        return (gate*error).sum(), gate.sum(), {"gate":gate, "valid_window_count":count,
                "auto_energy":power_t, "error":error.detach()}
