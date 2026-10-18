import numpy as np


def recognition_metrics(logits, labels, valid=None):
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or not len(labels) or len(logits) != len(labels):
        raise ValueError('Invalid prediction/label shape')
    valid = np.isfinite(logits).all(-1) if valid is None else np.asarray(valid, bool) & np.isfinite(logits).all(-1)
    # Unknown train-vocabulary classes stay in the evaluation failure denominator.
    order = np.argsort(-np.nan_to_num(logits, nan=-np.inf), axis=1, kind='stable')
    top1 = valid & (order[:, 0] == labels)
    top5 = valid & (order[:, :min(5, logits.shape[1])] == labels[:, None]).any(-1)
    classes = np.unique(labels)
    return {'Top1_micro': float(top1.mean()), 'Top5_micro': float(top5.mean()),
            'Top1_macro': float(np.mean([top1[labels == g].mean() for g in classes])),
            'Top5_macro': float(np.mean([top5[labels == g].mean() for g in classes])),
            'num_samples': len(labels), 'valid_samples': int(valid.sum()), 'unit': 'fraction'}
