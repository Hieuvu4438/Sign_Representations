import numpy as np


def classification_metrics(logits, labels, class_ids=None):
    logits, labels = np.asarray(logits), np.asarray(labels)
    if logits.ndim != 2 or labels.shape != (len(logits),) or not len(labels) or not np.isfinite(logits).all():
        raise ValueError("Classification metrics need finite logits[N,C] and nonempty labels[N]")
    classes = np.arange(logits.shape[1]) if class_ids is None else np.asarray(class_ids)
    if len(classes) != logits.shape[1] or len(set(classes.tolist())) != len(classes) or not set(labels.tolist()) <= set(classes.tolist()):
        raise ValueError("Labels or logits mismatch the fixed class taxonomy")
    order = np.argsort(-logits, axis=1, kind="stable")
    predictions = classes[order[:, 0]]
    supported = [c for c in classes if (labels == c).any()]
    per_class = {str(c): float((predictions[labels == c] == c).mean()) for c in supported}
    return {"per_instance_top1": float((predictions == labels).mean()),
            "per_class_top1": float(np.mean(list(per_class.values()))),
            "top5": float((classes[order[:, :min(5, len(classes))]] == labels[:, None]).any(axis=1).mean()),
            "class_counts": {str(c): int((labels == c).sum()) for c in classes},
            "absent_classes": [int(c) for c in classes if c not in supported], "per_class_accuracy": per_class}


def edit_counts(reference, hypothesis):
    """Levenshtein alignment with deterministic ties; returns S/D/I and reference length."""
    n, m = len(reference), len(hypothesis)
    table = [[None] * (m + 1) for _ in range(n + 1)]
    table[0][0] = (0, 0, 0, 0)
    for i in range(1, n + 1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                cost, s, d, ins = table[i - 1][j - 1]
                candidates = [(cost + 1, s + 1, d, ins)]
                cost, s, d, ins = table[i - 1][j]
                candidates.append((cost + 1, s, d + 1, ins))
                cost, s, d, ins = table[i][j - 1]
                candidates.append((cost + 1, s, d, ins + 1))
                table[i][j] = min(candidates, key=lambda value: value[0])
    _, substitutions, deletions, insertions = table[n][m]
    return {"substitutions": substitutions, "deletions": deletions, "insertions": insertions, "reference_tokens": n}


def corpus_wer(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError("WER corpus lengths differ")
    counts = {key: 0 for key in ("substitutions", "deletions", "insertions", "reference_tokens")}
    for ref, hyp in zip(references, hypotheses):
        for key, value in edit_counts(ref, hyp).items():
            counts[key] += value
    counts["wer"] = ((counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts["reference_tokens"]
                     if counts["reference_tokens"] else None)
    return counts
