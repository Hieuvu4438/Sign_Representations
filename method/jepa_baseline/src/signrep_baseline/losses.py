"""Main equations 2–10 and supplement Table9; engineering reductions explicit."""

import torch
from torch.nn import functional as F
from .schema import FidelityChoices, LOSS_WEIGHTS, TARGET_SCALES


def reconstruction_loss(predictions, targets, masks, choices=None):
    choices = choices or FidelityChoices()
    if set(targets) != set(LOSS_WEIGHTS) or set(masks) != set(targets):
        raise ValueError("Expected all nine priors plus activity, each with an explicit mask")
    terms = {}
    for key, weight in LOSS_WEIGHTS.items():
        predicted, target, valid = predictions[key], targets[key], masks[key]
        if predicted.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
            raise ValueError(f"Prediction/target/mask shape mismatch: {key}")
        if not torch.isfinite(predicted).all() or not torch.isfinite(target[valid]).all():
            raise ValueError(f"Nonfinite {key} predictions or observed supervision")
        target = torch.where(valid, target, torch.zeros_like(target)) * TARGET_SCALES.get(key, 1.0)
        loss = F.smooth_l1_loss(predicted.float(), target.float(), beta=choices.smooth_l1_beta,
                                reduction="none")
        terms[key] = weight * loss.masked_select(valid).sum() / valid.sum().clamp_min(1)
    return sum(terms.values()), terms


def feature_regularization(features):
    if features.ndim != 2 or features.shape[0] < 2 or not torch.isfinite(features).all():
        raise ValueError("Regularization requires at least two finite representations")
    features = features.float()
    variance = (1 - features.std(dim=0, correction=1)).clamp_min(0).sum()
    centered = features - features.mean(0)
    covariance = centered.T @ centered / len(features)
    off_diagonal = ~torch.eye(features.shape[1], dtype=torch.bool, device=features.device)
    cov_loss = covariance[off_diagonal].square().sum()
    return variance, cov_loss


def adversarial_loss(matched, unmatched, expected_matched, expected_unmatched):
    """Equations 7–9, averaged across anchors; EMA constants never get gradients."""
    if matched.shape != unmatched.shape:
        raise ValueError("Matched/unmatched probability shapes differ")
    if not torch.isfinite(matched).all() or not torch.isfinite(unmatched).all():
        raise ValueError("Nonfinite discriminator probabilities")
    if (matched.detach() < 0).any() or (matched.detach() > 1).any() or (unmatched.detach() < 0).any() or (unmatched.detach() > 1).any():
        raise ValueError("Adversarial equations use probabilities, not logits")
    em = torch.as_tensor(expected_matched, device=matched.device, dtype=matched.dtype).detach()
    eu = torch.as_tensor(expected_unmatched, device=matched.device, dtype=matched.dtype).detach()
    per_anchor = (matched - eu).clamp_min(0).square() + (em - unmatched).clamp_min(0).square()
    return per_anchor.mean() * (em > eu).to(per_anchor.dtype)


def pretraining_loss(predictions, targets, masks, features, matched, unmatched, ema, choices=None):
    recon, components = reconstruction_loss(predictions, targets, masks, choices)
    variance, covariance = feature_regularization(features)
    adv = adversarial_loss(matched, unmatched, ema.matched, ema.unmatched)
    return recon + variance + .004 * covariance + 2 * adv, {
        **{f"recon/{k}": v for k, v in components.items()}, "reconstruction": recon,
        "variance": variance, "covariance": covariance, "adversarial": adv}
