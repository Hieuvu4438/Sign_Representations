"""Synthetic cache-localization perturbations; no semantic training inputs."""
import torch


def jitter_boxes(boxes, confidence, sigma, seed=42):
    """Translate each slot by sigma * its size, consistently across a clip.

    The same seeded Gaussian direction is reused at all noise levels. Confidence
    is retained for positive clipped boxes; invalid boxes remain invalid.
    """
    if boxes.ndim != 3 or boxes.shape[-2:] != (4,4) or confidence.shape != boxes.shape[:-1]:
        raise ValueError("Expected [T,4,4] boxes and [T,4] confidence")
    if sigma < 0:
        raise ValueError("Noise sigma must be nonnegative")
    noise = torch.randn((1,4,2),generator=torch.Generator().manual_seed(seed)).to(boxes.device)*sigma
    size = (boxes[...,2:]-boxes[...,:2]).clamp_min(0)
    delta = noise*size
    shifted = (boxes+torch.cat((delta,delta),-1)).clamp(0,1)
    positive = (shifted[...,2:] > shifted[...,:2]).all(-1)
    return shifted,confidence*positive


def box_iou(a,b):
    intersection = (torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp_min(0).prod(-1)
    area_a = (a[...,2:]-a[...,:2]).clamp_min(0).prod(-1)
    area_b = (b[...,2:]-b[...,:2]).clamp_min(0).prod(-1)
    return intersection/(area_a+area_b-intersection).clamp_min(1e-12)
