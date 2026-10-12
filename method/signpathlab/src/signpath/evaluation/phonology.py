"""Annotation-driven phonological probes; English glosses are never converted into targets."""
import numpy as np
import torch
from ..cache.hashing import is_sha256
from ..models.heads import LinearHead


def macro_f1(reference, prediction, classes):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape or not len(reference):
        raise ValueError("F1 inputs must be aligned and nonempty")
    values = []
    for label in classes:
        if not (reference == label).any():
            continue
        tp = np.sum((reference == label) & (prediction == label))
        fp = np.sum((reference != label) & (prediction == label))
        fn = np.sum((reference == label) & (prediction != label))
        values.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
    return float(np.mean(values)) if values else None


def fit_phonology_probe(train_embeddings, train_labels, val_embeddings, val_labels, *,
                        annotation_provenance=None, seed=17, epochs=50, learning_rate=.001):
    if annotation_provenance is None:
        return {"status": "not_run", "reason": "No verified phonological annotation source supplied"}
    if not is_sha256(annotation_provenance.get("source_sha256")):
        raise ValueError("Phonological probes require actual annotation provenance")
    x, vx = np.asarray(train_embeddings, dtype=np.float32), np.asarray(val_embeddings, dtype=np.float32)
    labels, vl = np.asarray(train_labels), np.asarray(val_labels)
    if x.ndim != 2 or vx.ndim != 2 or x.shape[1] != vx.shape[1] or labels.shape != (len(x),) or vl.shape != (len(vx),):
        raise ValueError("Phonological embeddings/annotation arrays are misaligned")
    if not np.isfinite(x).all() or not np.isfinite(vx).all() or not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(vl.dtype, np.integer):
        raise ValueError("Phonological probes need finite embeddings and integer labels (-100 unknown)")
    known, vknown = labels != -100, vl != -100
    if not known.any() or not vknown.any():
        return {"status": "not_run", "reason": "No known annotations in train or validation", "train_known": int(known.sum()), "val_known": int(vknown.sum())}
    classes = tuple(sorted(set(labels[known].tolist())))
    if not set(vl[vknown].tolist()) <= set(classes):
        raise ValueError("Validation phonological taxonomy contains labels absent from training")
    torch.manual_seed(seed)
    head = LinearHead(x.shape[1], len(classes))
    target_map = {label: i for i, label in enumerate(classes)}
    target = torch.tensor([target_map[int(label)] for label in labels[known]])
    features = torch.from_numpy(x[known])
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(features), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite phonological probe objective")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = np.asarray(classes)[head(torch.from_numpy(vx[vknown])).argmax(-1).numpy()]
    return {"status": "completed", "accuracy": float((prediction == vl[vknown]).mean()),
        "macro_f1": macro_f1(vl[vknown], prediction, classes), "class_ids": list(classes), "train_known": int(known.sum()),
        "val_known": int(vknown.sum()), "unknown_policy": "-100 ignored", "annotation_provenance": annotation_provenance,
        "seed": seed, "epochs": epochs, "learning_rate": learning_rate}
