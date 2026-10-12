import numpy as np


def paired_classification_bootstrap(first, second, labels, groups, *, samples=1000, seed=17, resampling_unit="source_group_id"):
    """Paired PC/PI differences, resampling whole source/signer groups together."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    labels, groups = np.asarray(labels), np.asarray(groups)
    if first.ndim != 1 or any(x.shape != first.shape for x in (second, labels, groups)) or not len(first) or samples < 1:
        raise ValueError("Paired classification requires aligned nonempty outcomes, labels and groups")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Paired outcomes must be finite")
    def metrics(indices):
        difference = second[indices] - first[indices]
        taxonomy = labels[indices]
        return [difference.mean(), np.mean([difference[taxonomy == label].mean() for label in np.unique(taxonomy)])]
    point = metrics(np.arange(len(first)))
    units = sorted(set(groups.tolist()))
    report = {"per_instance_difference": float(point[0]), "per_class_difference": float(point[1]),
        "group_count": len(units), "sample_count": len(first), "resampling_unit": resampling_unit, "samples": samples, "seed": seed,
        "scope": "sampling uncertainty conditional on fitted paired models and evaluation pool; separate from seed dispersion"}
    if len(units) < 2:
        return {**report, "status": "insufficient_independent_groups", "per_instance_difference_ci95": None, "per_class_difference_ci95": None}
    members = {g: np.flatnonzero(groups == g) for g in units}
    rng = np.random.default_rng(seed)
    draws = np.asarray([metrics(np.concatenate([members[g] for g in rng.choice(units, len(units), replace=True)])) for _ in range(samples)])
    intervals = np.quantile(draws, [.025, .975], axis=0)
    return {**report, "status": "measured", "per_instance_difference_ci95": intervals[:, 0].tolist(),
        "per_class_difference_ci95": intervals[:, 1].tolist(), "absent_class_policy": "mean over sampled supported classes in fixed taxonomy"}


def paired_source_bootstrap(first, second, sources, *, samples=1000, seed=17):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or len(first) != len(sources) or not len(first) or samples < 1:
        raise ValueError("Bootstrap needs aligned paired outcomes and source IDs")
    groups = sorted(set(sources))
    by_source = {source: np.flatnonzero(np.asarray(sources) == source) for source in groups}
    rng, differences = np.random.default_rng(seed), []
    for _ in range(samples):
        indices = np.concatenate([by_source[source] for source in rng.choice(groups, len(groups), replace=True)])
        differences.append(float((second[indices] - first[indices]).mean()))
    low, high = np.quantile(differences, [.025, .975])
    return {"mean_difference": float((second - first).mean()), "ci95": [float(low), float(high)],
            "resampling_unit": "source_group", "samples": samples, "seed": seed}


def classification_bootstrap(correct, labels, sources, *, samples=1000, seed=17, resampling_unit="source_group_id"):
    correct, labels, sources = np.asarray(correct, dtype=float), np.asarray(labels), np.asarray(sources)
    if correct.ndim != 1 or labels.shape != correct.shape or sources.shape != correct.shape or not len(correct) or samples < 1:
        raise ValueError("Bootstrap classification inputs must be aligned and nonempty")
    groups = sorted(set(sources.tolist()))
    if len(groups) < 2:
        return {"status": "insufficient_independent_sources", "group_count": len(groups), "resampling_unit": resampling_unit, "ci95": None}
    members = {g: np.flatnonzero(sources == g) for g in groups}
    rng, instance, macro = np.random.default_rng(seed), [], []
    for _ in range(samples):
        idx = np.concatenate([members[g] for g in rng.choice(groups, len(groups), replace=True)])
        values, classes = correct[idx], labels[idx]
        instance.append(float(values.mean()))
        macro.append(float(np.mean([values[classes == label].mean() for label in np.unique(classes)])))
    return {"status": "measured", "resampling_unit": resampling_unit, "group_count": len(groups),
        "samples": samples, "seed": seed, "per_instance_top1_ci95": np.quantile(instance, [.025, .975]).tolist(),
        "per_class_top1_ci95": np.quantile(macro, [.025, .975]).tolist(),
        "absent_class_policy": "mean over sampled supported classes in fixed taxonomy",
        "scope": "percentile sampling intervals conditional on fitted model/pool; separate from seed dispersion"}
