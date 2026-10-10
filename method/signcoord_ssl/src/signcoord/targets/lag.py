"""Matched noncircular correlation control using the same anchors/windows."""
import torch

from .spectral import PAIRS, pair_window_weights, window_starts


def window_correlations(signal, window=16, hop=8, lag_min=-4, lag_max=4, eps=1e-6):
    if not -window < lag_min <= lag_max < window:
        raise ValueError("lag must leave observed overlap")
    x = signal.unfold(1, window, hop)
    x = x - x.mean(-1, keepdim=True)
    values, energies = [], []
    for p, q in PAIRS:
        pair, energy = [], []
        for lag in range(lag_min, lag_max+1):
            if lag > 0:
                a, b = x[:, :, p, :, :-lag], x[:, :, q, :, lag:]
            elif lag < 0:
                a, b = x[:, :, p, :, -lag:], x[:, :, q, :, :lag]
            else:
                a, b = x[:, :, p], x[:, :, q]
            aa, bb = a.square().sum((-1, -2)), b.square().sum((-1, -2))
            # clamp before sqrt to keep zero/hold gradients finite.
            pair.append((a*b).sum((-1, -2)) / (aa*bb).clamp_min(eps**2).sqrt().add(eps))
            energy.append(torch.stack((aa, bb), -1))
        values.append(torch.stack(pair, -1))
        energies.append(torch.stack(energy, -2))
    return torch.stack(values, 2), torch.stack(energies, 2)


def lag_loss_terms(pred_signal, target_signal, reliability, window=16, hop=8,
                   lag_min=-4, lag_max=4, energy_floor=1e-6, min_conf=.30):
    if len(window_starts(pred_signal.shape[1], window, hop)) < 2:
        raise ValueError("need at least two planned lag windows")
    with torch.autocast(device_type=pred_signal.device.type, enabled=False):
        dtype = torch.float64 if pred_signal.dtype == torch.float64 else torch.float32
        with torch.no_grad():
            weights = pair_window_weights(reliability.detach().to(dtype), window, hop, min_conf)
            target, energy = window_correlations(target_signal.detach().to(dtype), window, hop, lag_min, lag_max)
            count = (weights > 0).sum(1)
            eligible = (count >= 2)[..., None] & ((energy[..., 0] * weights[..., None]).sum(1) > energy_floor)
            eligible &= ((energy[..., 1] * weights[..., None]).sum(1) > energy_floor)
            den = weights.sum(1).clamp_min(energy_floor)[..., None]
            r_target = (target*weights[..., None]).sum(1)/den
            gate = eligible * (weights.sum(1)/count.clamp_min(1))[..., None]
        predicted, _ = window_correlations(pred_signal.to(dtype), window, hop, lag_min, lag_max)
        r_pred = (predicted*weights[..., None]).sum(1)/den
        error = (r_pred-r_target).square()
        return (error*gate).sum(), gate.sum(), {"gate": gate, "r_target": r_target, "valid_window_count": count}
