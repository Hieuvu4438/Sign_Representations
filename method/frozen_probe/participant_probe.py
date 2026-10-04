"""H005 diagnostic: cross-gloss participant centroids, not a retrieval method."""
import hashlib
import numpy as np


def gloss_folds(glosses, seed):
    if not glosses or len(set(glosses)) != len(glosses) or len(glosses) % 2:
        raise ValueError('Need a nonempty even unique gloss vocabulary')
    ordered = sorted(glosses, key=lambda g: (hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest(), g))
    return ordered[:len(ordered)//2], ordered[len(ordered)//2:]


def score_participants(truth, predictions):
    if not truth or len(truth) != len(predictions):
        raise ValueError('Incomplete predictions')
    groups = sorted(set(truth))
    per_group = {}
    for group in groups:
        positions = [i for i, p in enumerate(truth) if p == group]
        correct = sum(predictions[i] == group for i in positions)
        per_group[group] = dict(n=len(positions), correct=correct, recall=correct/len(positions))
    return dict(micro=sum(t == p for t, p in zip(truth, predictions))/len(truth),
                balanced=float(np.mean([r['recall'] for r in per_group.values()])),
                by_participant=per_group)


def centroid_predictions(features, labels, fit_rows, eval_rows):
    x = np.asarray(features, dtype=np.float64)
    if (x.ndim != 2 or len(x) != len(labels) or not np.isfinite(x).all()
            or not fit_rows or not eval_rows or set(fit_rows) & set(eval_rows)
            or len(set(fit_rows + eval_rows)) != len(fit_rows + eval_rows)
            or any(i < 0 or i >= len(x) for i in fit_rows + eval_rows)):
        raise ValueError('Invalid features or row partition')
    people = sorted(set(labels[i] for i in fit_rows))
    centroids, candidates, degenerate = [], [], []
    for person in people:
        mean = x[[i for i in fit_rows if labels[i] == person]].mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm <= 1e-12:
            degenerate.append(person)
            continue
        centroids.append(mean/norm)
        candidates.append(person)
    if not candidates:
        raise ValueError('No nondegenerate participant prototype')
    scores = x[eval_rows] @ np.stack(centroids).T
    # Candidates sorted lexicographically; argmax resolves exact ties to first.
    predictions = [candidates[i] for i in np.argmax(scores, axis=1)]
    truth = [labels[i] for i in eval_rows]
    counts = {p: sum(labels[i] == p for i in fit_rows) for p in people}
    majority = min(people, key=lambda p: (-counts[p], p))
    return dict(predictions=predictions, scores=score_participants(truth, predictions),
                majority=score_participants(truth, [majority]*len(eval_rows)),
                fit_counts=counts, candidate_participants=candidates,
                degenerate_participants=degenerate,
                fit_coverage=sum(p in people for p in truth)/len(truth),
                candidate_coverage=sum(p in candidates for p in truth)/len(truth))


def permute_within_gloss(labels, glosses, rng):
    if len(labels) != len(glosses):
        raise ValueError('Label/gloss coverage differs')
    output = list(labels)
    for gloss in sorted(set(glosses)):
        positions = [i for i, g in enumerate(glosses) if g == gloss]
        shuffled = rng.permutation([labels[i] for i in positions]).tolist()
        for i, value in zip(positions, shuffled):
            output[i] = value
    return output
