"""Temporal spectra on physical anchor velocities; never confidence FFT."""
from __future__ import annotations

import torch

PAIRS = ((0, 1), (0, 2), (1, 2))


def window_starts(length, window=16, hop=8):
    if window < 4 or window % 2 or hop <= 0:
        raise ValueError("MVP requires even window >=4 and positive hop")
    return tuple(range(0, length - window + 1, hop))


def window_fft(signal, window=16, hop=8):
    if signal.ndim != 4 or signal.shape[1] < window:
        raise ValueError("insufficient observed signal length")
    window_starts(signal.shape[1], window, hop)
    x = signal.unfold(1, window, hop)  # B,M,P,C,W
    x = x - x.mean(dim=-1, keepdim=True)
    hann = torch.hann_window(window, periodic=True, device=x.device, dtype=x.dtype)
    return torch.fft.rfft(x * hann, dim=-1, norm="ortho")


def pair_window_weights(reliability, window=16, hop=8, min_conf=0.30, basic_valid=None):
    r = reliability.unfold(1, window, hop)
    weights = []
    for p, q in PAIRS:
        common = torch.minimum(r[:, :, p], r[:, :, q])
        if basic_valid is None:
            eligible = (common >= min_conf).all(-1)
            weights.append(common.mean(-1) * eligible)
        else:
            v = basic_valid.unfold(1, window, hop)
            eligible = (v[:, :, p] & v[:, :, q]).all(-1)
            weights.append(eligible.to(reliability.dtype))
    return torch.stack(weights, -1)


def cross_spectra(u, weights, eps=1e-6):
    cross, power_p, power_q = [], [], []
    for qi, (p, q) in enumerate(PAIRS):
        a = weights[:, :, qi]
        den = a.sum(1).clamp_min(eps)[:, None]
        cross.append(((u[:, :, p] * u[:, :, q].conj()).sum(2) * a[..., None]).sum(1) / den)
        power_p.append((u[:, :, p].abs().square().sum(2) * a[..., None]).sum(1) / den)
        power_q.append((u[:, :, q].abs().square().sum(2) * a[..., None]).sum(1) / den)
    return torch.stack(cross, 1), torch.stack(power_p, 1), torch.stack(power_q, 1)


def phase_loss_terms(pred_signal, target_signal, target_rel, tau_p, tau_q, tau_cross,
                     window=16, hop=8, min_conf=0.30, eps=1e-6, basic_valid=None):
    """A7 quality-gated or A8 basic-valid-only phase numerator/denominator.

    The caller zeroes reliability/basic validity for spectral-ineligible timing.
    Target-derived gates cannot change when the student changes its prediction.
    """
    if pred_signal.shape != target_signal.shape or target_signal.shape[2:] != (3, 2):
        raise ValueError("phase expects matching anchor signals[B,N,3,2]")
    if len(window_starts(pred_signal.shape[1], window, hop)) < 2:
        raise ValueError("need at least two planned windows")
    for tau in (tau_p, tau_q, tau_cross):
        if torch.as_tensor(tau).requires_grad or not torch.all(torch.isfinite(torch.as_tensor(tau)) & (torch.as_tensor(tau) > 0)):
            raise ValueError("calibration thresholds must be frozen, finite and positive")
    with torch.autocast(device_type=pred_signal.device.type, enabled=False):
        dtype = torch.float64 if pred_signal.dtype == torch.float64 else torch.float32
        pred, target, rel = pred_signal.to(dtype), target_signal.detach().to(dtype), target_rel.detach().to(dtype)
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("phase input must be finite; missing values require validity masks")
        with torch.no_grad():
            weights = pair_window_weights(rel, window, hop, min_conf, basic_valid)
            s_t, p_t, q_t = cross_spectra(window_fft(target, window, hop), weights, eps)
            keep = slice(1, window // 2)
            s_t, p_t, q_t = s_t[..., keep], p_t[..., keep], q_t[..., keep]
            unit_t = s_t / s_t.abs().clamp_min(eps)
            count = (weights > 0).sum(1)
            eligible = (count >= 2)[..., None] & (s_t.abs() > eps)
            if basic_valid is None:
                mean_rel = weights.sum(1) / count.clamp_min(1)
                gate = eligible * mean_rel[..., None]
                gate = gate * (p_t / tau_p).clamp(0, 1) * (q_t / tau_q).clamp(0, 1)
                gate = gate * (s_t.abs() / tau_cross).clamp(0, 1)
            else:
                gate = eligible.to(dtype)
        s_s, _, _ = cross_spectra(window_fft(pred, window, hop), weights, eps)
        s_s = s_s[..., keep]
        unit_s = s_s / s_s.abs().clamp_min(eps)
        error = (unit_s.real - unit_t.real).square() + (unit_s.imag - unit_t.imag).square()
        return (gate * error).sum(), gate.sum(), {"gate": gate, "q_target": unit_t,
                "valid_window_count": count, "cross_energy": s_t.abs(), "error": error.detach()}


def fit_energy_thresholds(signals, reliability, window=16, hop=8, min_samples=8,
                          percentile=10, floor=1e-6, min_conf=.30):
    """Fit positive finite train spectra. Split provenance is enforced by caller."""
    if signals.shape[0] != reliability.shape[0]:
        raise ValueError("signal/reliability batch mismatch")
    with torch.no_grad():
        weights = pair_window_weights(reliability, window, hop, min_conf)
        s, p, q = cross_spectra(window_fft(signals, window, hop), weights)
        eligible = (weights > 0).sum(1) >= 2
        out = {}
        for name, spectrum in (("tau_p", p), ("tau_q", q), ("tau_cross", s.abs())):
            spectrum = spectrum[..., 1:window//2]
            thresholds = torch.empty(spectrum.shape[1:], dtype=spectrum.dtype, device=spectrum.device)
            counts = torch.empty_like(thresholds, dtype=torch.long)
            for pair in range(3):
                for k in range(window//2-1):
                    values = spectrum[:, pair, k]
                    values = values[eligible[:, pair] & torch.isfinite(values) & (values > 0)]
                    counts[pair, k] = len(values)
                    if len(values) < min_samples:
                        raise ValueError(f"calibration failure: {name} pair={pair} bin={k+1}: {len(values)} < {min_samples}")
                    thresholds[pair, k] = torch.quantile(values, percentile/100).clamp_min(floor)
            out[name] = thresholds
            out[name + "_counts"] = counts
        return out


def rfft_energy(x):
    u = torch.fft.rfft(x, dim=-1, norm="ortho")
    weights = torch.full((u.shape[-1],), 2., dtype=x.dtype, device=x.device)
    weights[0] = 1
    if x.shape[-1] % 2 == 0:
        weights[-1] = 1
    return (u.abs().square() * weights).sum(-1)
