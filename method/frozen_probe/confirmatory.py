"""P003 held-out contracts: train-only galleries, fixed inference and analysis."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import participant_bootstrap
from baseline import support_ids
from cache import file_sha, sample_map
from metrics import evaluate, ranked_classes
from probes import normalize_rows
from window_budget import pooled_vector, selected_indices

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
METHODS = ('full_mean', 'full_activity', 'uniform8', 'random8_s2718', 'random8_s31415')
RECIPES = METHODS[1:]
SEEDS = (1701, 1702, 1703)


def check_freeze(gate_path, config_path):
    gate = json.loads(Path(gate_path).read_text())
    config_path = Path(config_path).resolve()
    if gate['verdict'] in ('VALIDATED_TECHNICAL_ERRATUM_BEFORE_SCORING',
                           'VALIDATED_SERIALIZATION_ERRATUM_AFTER_EXECUTION'):
        original_path = REPO / gate['original_freeze']
        if file_sha(original_path) != gate['original_freeze_sha256']:
            raise ValueError('Original pre-test registration changed')
        original = json.loads(original_path.read_text())
        if (original['verdict'] != 'FROZEN_BEFORE_SELECTED_TEST_ACCESS'
                or gate['scientific_config_changes'] or gate['config_hashes'] != original['config_hashes']):
            raise ValueError('Technical erratum cannot change scientific configuration')
    elif gate['verdict'] != 'FROZEN_BEFORE_SELECTED_TEST_ACCESS':
        raise ValueError('Unissued confirmatory freeze')
    if gate['config_hashes'].get(str(config_path.relative_to(REPO))) != file_sha(config_path):
        raise ValueError('Confirmatory config not bound to the pre-test freeze')
    for name, digest in gate['file_hashes'].items():
        if file_sha(REPO / name) != digest:
            raise ValueError(f'Frozen scientific source/config changed: {name}')
    return gate


def query_map(cohort, test):
    """Never alias held-out test to val, and never allow test in a gallery."""
    known = sample_map(cohort)
    if test['classes'] != cohort['classes'] or not test['samples']:
        raise ValueError('Held-out vocabulary/coverage mismatch')
    ids, hashes = set(known), {s['source_sha256'] for s in known.values()}
    people = {s['participant_id'] for s in known.values()}
    indexed = {}
    for sample in test['samples']:
        key = sample['sample_id']
        if (sample['split'] != 'test' or key in ids or key in indexed
                or sample['source_sha256'] in hashes or sample['participant_id'] in people
                or type(sample['class_id']) is not int or not 0 <= sample['class_id'] < len(cohort['classes'])
                or cohort['classes'][sample['class_id']] != sample['gloss']):
            raise ValueError('Test identity/split/participant/label leakage')
        indexed[key] = sample
        hashes.add(sample['source_sha256'])
    if set(s['class_id'] for s in indexed.values()) != set(range(len(cohort['classes']))):
        raise ValueError('Selected test must cover every fixed train class')
    return indexed


def activity_proxy(activity):
    """Effective activity mass fraction; diagnostic proxy, no linguistic labels."""
    value = np.asarray(activity, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Invalid native activity array')
    weights = np.clip(value, 0, 1).max(axis=1)
    total = float(weights.sum())
    return dict(activity_mass_mean=float(weights.mean()), zero_activity=total <= 1e-12,
                effective_activity_fraction=(float(total * total / (len(weights) * (weights @ weights)))
                                             if total > 1e-12 else 0.0))


def train_thresholds(reader, cohort, audit):
    indexed = sample_map(cohort)
    native = {s['sample_id']: s for s in audit['samples']}
    ids = sorted(k for k, s in indexed.items() if s['split'] == 'train')
    lengths, activity = [], []
    for key in ids:
        _, raw, _ = reader.load(key)
        lengths.append(native[key]['windows'])
        activity.append(activity_proxy(raw)['effective_activity_fraction'])
    return dict(train_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(), count=len(ids),
                native_window_count_quartiles=np.quantile(lengths, [.25, .5, .75]).tolist(),
                effective_activity_fraction_quartiles=np.quantile(activity, [.25, .5, .75]).tolist(),
                slice_rule='Q1 <=q25; Q2 (q25,q50]; Q3 (q50,q75]; Q4 >q75',
                fit_split='train only; all fixed support-union sources; no test calibration')


def build_galleries(reader, cohort, budget=8, seeds=SEEDS):
    indexed = sample_map(cohort)
    supports = {str(seed): support_ids(cohort, seed, budget) for seed in seeds}
    ids = sorted(set(k for group in supports.values() for k in group))
    pooled = {method: [] for method in METHODS}
    counts = {method: 0 for method in METHODS}
    for key in ids:
        x, a, centers = reader.load(key)
        for method in METHODS:
            selection = selected_indices(method, len(centers), indexed[key]['source_sha256'])
            pooled[method].append(pooled_vector(x, a, method, selection))
            counts[method] += len(selection)
    positions = {key: i for i, key in enumerate(ids)}
    galleries = {}
    for method in METHODS:
        matrix = normalize_rows(np.stack(pooled[method]))
        galleries[method] = {}
        for seed in seeds:
            selected = supports[str(seed)]
            labels = np.array([indexed[k]['class_id'] for k in selected])
            galleries[method][str(seed)] = (matrix[[positions[k] for k in selected]],
                [np.flatnonzero(labels == c) for c in range(len(cohort['classes']))])
    return galleries, dict(support_ids=supports, unique_source_count=len(ids), window_counts=counts,
                           origin='validated CACHE005 native outputs; per-source fixed pooling only; no test fit',
                           setup_cost='CPU pooling/search-index setup only; gallery extraction cost excluded from query latency')


def search(vector, galleries, classes):
    vector = normalize_rows(np.asarray(vector)[None])[0]
    result = {}
    for seed, (matrix, rows) in galleries.items():
        similarities = matrix @ vector
        scores = [float(similarities[selected].max()) for selected in rows]
        result[seed] = ranked_classes(scores, list(range(classes)))
    return result


def verify_rankings(schema, rankings):
    ids, labels, people, classes = (schema[k] for k in ('query_ids', 'labels', 'participant_ids', 'class_ids'))
    if (not ids or len(set(ids)) != len(ids) or len(labels) != len(ids) or len(people) != len(ids)
            or classes != list(range(len(classes))) or not classes or schema['split'] != 'test'):
        raise ValueError('Invalid held-out ranking schema')
    expected = {f'{method}_s{seed}' for method in METHODS for seed in SEEDS}
    if set(rankings) != expected:
        raise ValueError('Incomplete held-out method/gallery coverage')
    refs = [dict(sample_id=k, label=label) for k, label in zip(ids, labels)]
    results, dcg, correct = [], {}, {}
    for name in sorted(expected):
        rank = np.asarray(rankings[name])
        if rank.shape != (len(ids), len(classes)) or not np.issubdtype(rank.dtype, np.integer):
            raise ValueError('Invalid complete ranking shape/type')
        predictions = [dict(sample_id=k, ranking=row.tolist()) for k, row in zip(ids, rank)]
        measured = evaluate(predictions, refs, classes)  # validates every full permutation
        position = np.argmax(rank == np.asarray(labels)[:, None], axis=1) + 1
        dcg[name] = 1 / np.log2(position + 1)
        correct[name] = (position == 1)
        results.append(dict(name=name, metrics=measured))
    return results, dcg, correct


def cost_summary(schema, observations):
    ids = schema['query_ids']
    indexed = {}
    for row in observations:
        key = (row['sample_id'], row['method'])
        if key in indexed or row['method'] not in RECIPES or row['sample_id'] not in ids:
            raise ValueError('Duplicate/unknown actual timing observation')
        components = [row[k] for k in ('decode_seconds', 'preprocess_seconds', 'encoder_wrapper_seconds',
                                       'pool_seconds', 'three_gallery_search_seconds')]
        if (any(not np.isfinite(v) or v < 0 for v in components) or row['end_to_end_seconds'] <= 0
                or type(row['encoded_windows']) is not int or type(row['source_windows']) is not int or row['encoded_windows'] < 1
                or row['source_windows'] < row['encoded_windows']
                or not np.isclose(sum(components), row['end_to_end_seconds'], atol=1e-9, rtol=0)):
            raise ValueError('Invalid actual timing component/count ledger')
        indexed[key] = row
    if set(indexed) != {(key, method) for key in ids for method in RECIPES}:
        raise ValueError('Incomplete held-out actual inference/timing coverage')
    summaries = {}
    dense = sum(indexed[(key, 'full_activity')]['encoded_windows'] for key in ids)
    for method in RECIPES:
        rows = [indexed[(key, method)] for key in ids]
        if any(r['source_windows'] != indexed[(r['sample_id'], 'full_activity')]['source_windows'] for r in rows):
            raise ValueError('Methods saw different native source windows')
        times = np.array([r['end_to_end_seconds'] for r in rows])
        ratios = [r['end_to_end_seconds'] / indexed[(r['sample_id'], 'full_activity')]['end_to_end_seconds'] for r in rows]
        counts = sum(r['encoded_windows'] for r in rows)
        summaries[method] = dict(N=len(rows), encoded_windows=counts, dense_windows=dense,
            window_reduction_factor=dense / counts, median_paired_query_ratio=float(np.median(ratios)),
            p50_seconds=float(np.quantile(times, .5)), p95_seconds=float(np.quantile(times, .95)),
            sum_pipeline_seconds=float(times.sum()), actual_cost_screen=bool(counts * 2 <= dense and np.median(ratios) <= .8))
    return summaries


def analyze(schema, rankings, observations, diagnostics, thresholds):
    results, dcg, correct = verify_rankings(schema, rankings)
    costs = cost_summary(schema, observations)
    indexed = {r['name']: r['metrics'] for r in results}
    mean_metrics = {method: {metric: float(np.mean([indexed[f'{method}_s{s}'][metric] for s in SEEDS]))
                            for metric in results[0]['metrics']} for method in METHODS}
    paired = np.mean([dcg[f'uniform8_s{s}'] - dcg[f'full_activity_s{s}'] for s in SEEDS], axis=0)
    boot = participant_bootstrap(paired, schema['participant_ids'], 2000, 1701)
    interval = [v / 100 for v in boot['interval95_pp']]
    boot.update(estimate_dcg=float(paired.mean()), interval95_dcg=interval,
                limitation='conditional on fixed train-selected classes/checkpoint/galleries; few participants; no population equivalence')
    losses = {str(s): float(indexed[f'full_activity_s{s}']['dcg'] - indexed[f'uniform8_s{s}']['dcg']) for s in SEEDS}
    quality = paired.mean() >= -.02 and interval[0] >= -.02 and all(v <= .02 for v in losses.values())
    if set(diagnostics) != set(schema['query_ids']):
        raise ValueError('Incomplete diagnostic source coverage')
    slices = {}
    for kind, field, cuts in (
        ('length', 'source_windows', thresholds['native_window_count_quartiles']),
        ('activity_proxy', 'effective_activity_fraction', thresholds['effective_activity_fraction_quartiles'])):
        values = [diagnostics[k][field] for k in schema['query_ids']]
        assignment = np.searchsorted(cuts, values, side='left')
        slices[kind] = {}
        for group in range(4):
            mask = assignment == group
            slices[kind][f'Q{group+1}'] = dict(N=int(mask.sum()), participant_count=len(set(
                np.array(schema['participant_ids'])[mask].tolist())),
                uniform_minus_dense_dcg=float(paired[mask].mean()) if mask.any() else None,
                mean_dcg={m: float(np.mean([dcg[f'{m}_s{s}'][mask].mean() for s in SEEDS])) if mask.any() else None for m in METHODS})
    by_participant = {}
    for person in sorted(set(schema['participant_ids'])):
        mask = np.array(schema['participant_ids']) == person
        by_participant[person] = dict(N=int(mask.sum()), uniform_minus_dense_dcg=float(paired[mask].mean()))
    # Qualitative sampling categories fixed before test; primary gallery1701 only.
    ids = schema['query_ids']
    dense, sparse = correct['full_activity_s1701'], correct['uniform8_s1701']
    categories = dict(both_success=dense & sparse, both_failure=~dense & ~sparse,
                      dense_only=dense & ~sparse, sparse_only=~dense & sparse)
    cases = {}
    for category, mask in categories.items():
        eligible = [k for k, flag in zip(ids, mask) if flag]
        eligible.sort(key=lambda k: (hashlib.sha256(f'1701:{category}:{k}'.encode()).hexdigest(), k))
        chosen = eligible[:5]
        cases[category] = dict(eligible_N=len(eligible), samples=[dict(sample_id=k,
            label=schema['labels'][ids.index(k)], rankings={m: rankings[f'{m}_s1701'][ids.index(k), :10].tolist() for m in METHODS},
            diagnostics=diagnostics[k]) for k in chosen])
    return dict(primary_comparison='uniform8 minus full_activity; mean DCG over three fixed train galleries',
        metrics_by_gallery=results, mean_metrics=mean_metrics, per_gallery_dcg_loss=losses,
        conditional_participant_bootstrap=boot, quality_noninferiority_screen=bool(quality), costs=costs,
        joint_confirmatory_screen=bool(quality and costs['uniform8']['actual_cost_screen']),
        participant_counts=dict(Counter(schema['participant_ids'])), participant_differences=by_participant,
        descriptive_slices=slices, qualitative_cases=cases,
        secondary_scope='controls/slices/cases descriptive; no multiplicity-adjusted inferential claims',
        scope='100 fixed train-selected ASL classes, existing checkpoint, eight labeled train examples/class; adapted held-out lookup')
