"""Descriptive complete query/gallery factorial; original primary remains unchanged."""
from collections import Counter

import numpy as np

from metrics import evaluate


def pair_key(query, gallery, seed):
    return f'q_{query}__g_{gallery}__s{seed}'


def summarize_pairs(schema, rankings, methods, seeds):
    ids, labels, people, classes = (schema[k] for k in ('query_ids', 'labels', 'participant_ids', 'class_ids'))
    if (schema['split'] != 'test' or not ids or len(set(ids)) != len(ids)
            or len(labels) != len(ids) or len(people) != len(ids)
            or classes != list(range(len(classes))) or not classes
            or not methods or len(set(methods)) != len(methods) or 'full_activity' not in methods
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError('Invalid complete factorial schema')
    expected = {pair_key(q, g, s) for q in methods for g in methods for s in seeds}
    if set(rankings) != expected:
        raise ValueError('Incomplete/unknown query-gallery factorial cells')
    reference = [dict(sample_id=k, label=label) for k, label in zip(ids, labels)]
    measured, dcg, per_pair = {}, {}, []
    for q in methods:
        for g in methods:
            per_seed = {}
            for seed in seeds:
                key = pair_key(q, g, seed)
                matrix = np.asarray(rankings[key])
                if matrix.shape != (len(ids), len(classes)) or not np.issubdtype(matrix.dtype, np.integer):
                    raise ValueError('Invalid factorial full-ranking shape/dtype')
                predictions = [dict(sample_id=k, ranking=row.tolist()) for k, row in zip(ids, matrix)]
                metrics = evaluate(predictions, reference, classes)
                measured[key] = metrics
                rank = np.argmax(matrix == np.asarray(labels)[:, None], axis=1) + 1
                dcg[key] = 1 / np.log2(rank + 1)
                per_seed[str(seed)] = metrics
            mean = {metric: float(np.mean([values[metric] for values in per_seed.values()]))
                    for metric in next(iter(per_seed.values()))}
            per_pair.append(dict(query_method=q, gallery_method=g, per_seed=per_seed, mean_metrics=mean))
    contrasts = {}
    dense = 'full_activity'
    for sparse in methods:
        if sparse == dense:
            continue
        by_seed = {}
        vectors = {}
        for seed in seeds:
            keys = [pair_key(q, g, seed) for q, g in
                    ((dense, dense), (sparse, dense), (dense, sparse), (sparse, sparse))]
            values = [dcg[key] for key in keys]
            vectors[seed] = dict(query_only=values[1]-values[0], gallery_only=values[2]-values[0],
                joint=values[3]-values[0], interaction=values[3]-values[1]-values[2]+values[0])
            metric_contrasts = {}
            for metric in ('dcg', 'top1'):
                dd, sd, ds, ss = [measured[key][metric] for key in keys]
                metric_contrasts[metric] = dict(query_only=sd-dd, gallery_only=ds-dd,
                                               joint=ss-dd, interaction=ss-sd-ds+dd)
            by_seed[str(seed)] = metric_contrasts
        mean = {metric: {name: float(np.mean([row[metric][name] for row in by_seed.values()]))
                         for name in ('query_only', 'gallery_only', 'joint', 'interaction')}
                for metric in ('dcg', 'top1')}
        averaged = {name: np.mean([vectors[seed][name] for seed in seeds], axis=0)
                    for name in ('query_only', 'gallery_only', 'joint', 'interaction')}
        participants = {}
        for person in sorted(set(people)):
            mask = np.array(people) == person
            participants[person] = dict(N=int(mask.sum()), dcg={name: float(value[mask].mean())
                                                              for name, value in averaged.items()})
        contrasts[sparse] = dict(per_seed=by_seed, mean=mean, by_participant=participants)
    return dict(N_expected=len(ids), N_scored=len(ids), classes=len(classes),
        participant_counts=dict(Counter(people)), methods=list(methods), seeds=list(seeds),
        cells=len(per_pair), ranking_matrices=len(rankings), per_pair=per_pair, contrasts=contrasts,
        scope='post-outcome exploratory descriptive fixed query/gallery factorial; no new primary selection, inferential tests or causal/linguistic mechanism proof')
