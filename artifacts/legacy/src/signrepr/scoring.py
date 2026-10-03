"""Teacher-forced token scoring with explicit padding and length accounting."""
import torch


def token_nll(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError('Logit/label shape mismatch')
    mask = labels != -100
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError('Candidate has no scored tokens')
    losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels,
                                                ignore_index=-100, reduction='none')
    summed = (losses * mask).sum(dim=1)
    if not torch.isfinite(summed).all():
        raise ValueError('Candidate NLL is not finite')
    return {'sum_nll': summed, 'token_count': counts, 'mean_nll': summed / counts}
