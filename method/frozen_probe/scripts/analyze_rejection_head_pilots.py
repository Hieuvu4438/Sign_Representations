"""Separate saved-weight replay, exact semantic recount and locked head screen.

No head imports, fitting, target selection or new encoder calls. NumPy affine
layers replay the explicit saved parameter schema; all decisions use saved GPU
logits, not approximately reproduced CPU logits.
"""
import argparse
import csv
import io
import json
from math import fsum
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store
from msasl_saved_features import load_complete_features
from rejection_analysis import cluster_mean_rate, conditional_interval
from resources import scratch_preflight
from cache import file_sha


def read(path):
    return json.loads(Path(path).read_text())


def threshold(scores, known):
    values = sorted(float(x) for x in scores[known])
    needed = (19 * len(values) + 19) // 20
    tau = values[len(values) - needed]
    return tau, needed, sum(x >= tau for x in values)


def point(scores, tau, known, correct):
    accepted = scores >= tau
    nk, nu = int(known.sum()), int((~known).sum())
    ka = int(np.count_nonzero(accepted & known))
    ca = int(np.count_nonzero(accepted & correct))
    ua = int(np.count_nonzero(accepted & ~known))
    return dict(threshold=float(tau), n_known=nk, n_unknown=nu,
                known_accept_count=ka, known_correct_accept_count=ca,
                unknown_accept_count=ua, FAR=ua / nu, CCR=ca / nk,
                known_acceptance=ka / nk,
                closed_known_accuracy=int(correct.sum()) / nk)


def count_curve(scores, known, correct):
    # Independent ordered cumulative integer counts, including whole tie groups.
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(order) - 1]
    counts = np.column_stack((np.cumsum(known[order]), np.cumsum(correct[order]),
                              np.cumsum((~known)[order])))[ends]
    rows = np.vstack((np.zeros(4), np.column_stack((ordered[ends], counts))))
    x, y = rows[:, 3] / int((~known).sum()), rows[:, 2] / int(known.sum())
    auc = fsum(float((x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2)
               for i in range(len(x) - 1))
    return rows, auc


def table_bytes(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def manual_oracles():
    known = np.array([True, True, False, False])
    correct = np.array([True, False, False, False])
    scores = np.array([.9, .8, .8, .1])
    p = point(scores, .8, known, correct)
    assert (p['known_accept_count'], p['known_correct_accept_count'], p['unknown_accept_count']) == (2, 1, 1)
    assert p['CCR'] == .5 and p['FAR'] == .5
    rows, auc = count_curve(scores, known, correct)
    assert np.array_equal(rows[:, 1:], [[0, 0, 0], [1, 1, 0], [2, 1, 1], [2, 1, 2]]) and auc == .5
    assert threshold(np.arange(20), np.ones(20, dtype=bool)) == (1., 19, 19)
    assert threshold(np.ones(20), np.ones(20, dtype=bool)) == (1., 19, 20)
    schema = [dict(head='residual_linear', field='fc.weight', start=0, stop=2, shape=[1, 2]),
              dict(head='residual_linear', field='fc.bias', start=2, stop=3, shape=[1])]
    assert replay(np.array([2, -3, 5]), schema, 'residual_linear', np.array([[7, 11], [-2, 4]])).tolist() == [-14., -11.]
    return dict(passed=True, groups=5, scope='Wrong-known/inclusive tie counts, grouped AUC, known95 threshold, tie overshoot, affine replay hand calculations.')


def manual_source_counts(pool):
    roster = read(pool / 'source_roster.json')
    plan = np.load(pool / 'training_plan.npy', allow_pickle=False)
    vectors = np.load(pool / 'fit_vectors.npy', allow_pickle=False)
    assert len(roster) == 1027 and plan.shape == (1000, 16, 52)
    assert vectors.shape == (1027, 768) and np.isfinite(vectors).all()
    labels = np.array([row['class_id'] for row in roster])
    repool_error = 0.; windows = 0
    for i, row in enumerate(roster):
        assert file_sha(row['feature']['path']) == row['feature']['sha256']
        with np.load(row['feature']['path'], allow_pickle=False) as cache:
            # Native cache schema is checked below before any averaging.
            x = cache['features'].astype(np.float64)
            activity = cache['raw_activity'].astype(np.float64)
        assert len(x) == row['windows'] and activity.shape == (len(x), 2)
        w = np.maximum(np.clip(activity[:, 0], 0, 1), np.clip(activity[:, 1], 0, 1))
        vector = x.mean(axis=0) if w.sum() <= 1e-12 else np.sum(x * w[:, None], axis=0) / w.sum()
        vector /= np.sqrt(np.dot(vector, vector))
        repool_error = max(repool_error, float(np.max(np.abs(vector - vectors[i]))))
        windows += len(x)
    assert repool_error <= 1e-12
    counts = []
    for episode in plan:
        correct = 0
        for pair in episode:
            q = roster[int(pair[0])]
            supports = [roster[int(j)] for j in pair[1:]]
            assert supports[0]['class_id'] == q['class_id']
            assert len({r['class_id'] for r in supports}) == 51
            assert all(r['signer'] != q['signer'] and r['source_group'] != q['source_group'] for r in supports)
            positive = np.r_[pair[1:2], pair[3:]]
            positive = positive[np.argsort(labels[positive], kind='stable')]
            # Match the frozen FP64 accumulation, independent membership logic.
            similarities = np.clip(np.einsum('d,jd->j', vectors[pair[0]], vectors[positive]), -1, 1)
            correct += int(labels[positive[int(similarities.argmax())]] == q['class_id'])
        counts.append([correct, correct, int(correct > 0)])
    return np.asarray(counts, dtype=np.int16), dict(clips=len(roster), windows=windows,
        pairs=int(plan.shape[0] * plan.shape[1]), max_repool_error=repool_error,
        official_signers=len({r['signer'] for r in roster}),
        scope='Saved native-window activity pooling and all paired membership/count checks; no full raw-video decode or refitting.')


def replay(flat, schema, method, residual):
    fields = {r['field']: flat[r['start']:r['stop']].reshape(r['shape']).astype(np.float64)
              for r in schema if r['head'] == method}
    layers = ['fc'] if method == 'residual_linear' else ['fc1', 'fc2', 'fc3', 'fc4']
    x = residual.astype(np.float64)
    for i, layer in enumerate(layers):
        x = x @ fields[layer + '.weight'].T + fields[layer + '.bias']
        if i + 1 < len(layers): x = np.maximum(x, 0)
    return x[:, 0]


def independent_bootstrap(success, masks, groups, weights):
    names = sorted(set(groups))
    numerator = np.zeros((len(names), success.shape[0]), dtype=np.int64)
    denominator = np.zeros_like(numerator)
    for j, name in enumerate(names):
        cols = np.asarray(groups) == name
        for i in range(success.shape[0]):
            numerator[j, i] = np.count_nonzero(success[i, cols] & masks[i, cols])
            denominator[j, i] = np.count_nonzero(masks[i, cols])
    values = []; validity = []
    for w in weights:
        n, d = np.sum(numerator * w[:, None], axis=0), np.sum(denominator * w[:, None], axis=0)
        validity.append(bool(np.all(d > 0)))
        values.append(float(np.mean(np.divide(n, d, out=np.zeros(len(n)), where=d > 0))))
    return np.asarray(values), np.asarray(validity)


def main():
    p = argparse.ArgumentParser()
    for key in ('config', 'run-id', 'output-dir'): p.add_argument('--' + key, required=True)
    p.add_argument('--seed', type=int, required=True); a = p.parse_args()
    with BoundedCPUContext(a.output_dir, a.run_id, a.config, a.seed) as run, threadpool_limits(limits=1):
        c = run.config; resource = scratch_preflight('/dev/shm', 512 << 20); oracles = manual_oracles()
        measured = Path(c['measured_runtime']); pool = Path(c['pool_runtime'])
        verify_store(measured); verify_store(pool)
        meta = read(measured / 'rejection_metadata.json'); old = read(measured / 'rejection_eval.json')
        assert meta['gallery_sizes'] == [10, 25, 50] and meta['support_seeds'] == [1701, 1702, 1703]
        assert meta['score_names'] == ['maxcosine', 'margin', 'maxsoftmax', 'NAC16']
        pred = np.load(measured / 'predictions.npy', allow_pickle=False)
        ranks = np.load(measured / 'known_rank.npy', allow_pickle=False)
        frozen_scores = np.load(measured / 'scores.npy', allow_pickle=False)
        y = np.asarray(meta['labels']); methods = meta['score_names'] + ['residual_linear', 'FRDisc_frozen_MLP']
        expected_counts, source_check = manual_source_counts(pool)
        mc = read(measured / 'config.json'); base = np.load(measured / 'base_vectors.npy', allow_pickle=False)
        fc = read(Path(mc['feature_runtime']) / 'config.json'); items = read(fc['extraction_manifest'])
        ms, _, feature_check = load_complete_features(mc['feature_runtime'], items)
        sources = {'ASLCitizen_base': (base, {key: i for i, key in enumerate(meta['base_ids'])}),
                   'MSASL100': (ms, {row['sample_id']: i for i, row in enumerate(items)})}
        scores_by_train = {}; checks = []; all_rows = []; curves = []; segments = []
        for directory in map(Path, c['head_runtimes']):
            verify_store(directory)
            report = read(directory / 'head_evaluation.json'); seed = report['training_seed']
            logits = np.load(directory / 'head_scores.npy', allow_pickle=False)
            flat = np.load(directory / 'head_weights.npy', allow_pickle=False)
            counts = np.load(directory / 'training_counts.npy', allow_pickle=False)
            losses = np.load(directory / 'train_losses.npy', allow_pickle=False)
            assert np.array_equal(counts, expected_counts)
            assert report['actual_updates'] == int(expected_counts[:, 2].sum())
            assert np.isfinite(flat).all() and np.isfinite(logits).all() and np.isfinite(losses).all()
            assert np.all(losses[counts[:, 2] == 0] == 0) and np.all(losses[counts[:, 2] == 1] >= 0)
            assert flat.shape == (2411650,) and logits.shape == (3, 3, 2, 1412)
            schema = report['parameter_schema']; stop = 0
            assert [r['shape'] for r in schema] == [[1, 768], [1], [1536, 768], [1536], [768, 1536], [768], [64, 768], [64], [1, 64], [1]]
            for row in schema:
                assert row['start'] == stop and row['stop'] - stop == int(np.prod(row['shape']))
                stop = row['stop']
            assert stop == len(flat)
            max_error = 0.; calibration_checked = 0; rows_checked = 0
            for si, support_seed in enumerate(meta['support_seeds']):
                cal = meta['parts'][0]; cal_sl = slice(cal['start'], cal['stop'])
                for ni, size in enumerate(meta['gallery_sizes']):
                    ce = next(g for g in meta['gallery_metadata'] if g['part'] == cal['name'] and g['seed'] == support_seed)
                    known_cal = np.isin(y[cal_sl], ce['galleries'][str(size)]['class_ids'])
                    for hi, method in enumerate(report['score_names']):
                        tau, needed, accepted = threshold(logits[si, ni, hi, cal_sl], known_cal)
                        registered = report['calibration'][f'{support_seed}:{size}:{method}']
                        assert tau == registered['threshold'] and needed == registered['minimum_accept_count']
                        assert accepted == registered['accepted_known_count']
                        assert accepted - needed == registered['tie_overshoot']
                        calibration_checked += 1
                    for part in meta['parts']:
                        sl = slice(part['start'], part['stop'])
                        entry = next(g for g in meta['gallery_metadata'] if g['part'] == part['name'] and g['seed'] == support_seed)
                        gallery = entry['galleries'][str(size)]; classes = gallery['class_ids']
                        values, positions = sources[part['corpus']]
                        query = values[[positions[key] for key in meta['query_ids'][sl]]].astype(np.float32)
                        support = values[[positions[key] for key in gallery['support_ids']]].astype(np.float32)
                        columns = [classes.index(int(cls)) for cls in pred[si, ni, sl]]
                        residual = query - support[columns]
                        known = np.isin(y[sl], classes); correct = known & (pred[si, ni, sl] == y[sl])
                        kr = ranks[si, ni, sl][known].astype(np.float64)
                        retrieval = dict(Top1=float(np.mean(kr == 1)), Top5=float(np.mean(kr <= 5)),
                                         DCG_FP64=float(np.mean(1 / np.log2(kr + 1))), MRR=float(np.mean(1 / kr)))
                        for hi, method in enumerate(report['score_names']):
                            cpu = replay(flat, schema, method, residual)
                            gpu = logits[si, ni, hi, sl].astype(np.float64)
                            error = np.abs(cpu - gpu); max_error = max(max_error, float(error.max()))
                            assert np.all(error <= 1e-4 + 1e-4 * np.abs(gpu)), (seed, method, float(error.max()))
                            curve, auc = count_curve(gpu, known, correct)
                            start = sum(len(x) for x in curves); curves.append(curve)
                            segments.append(dict(training_seed=seed, support_seed=support_seed, size=size,
                                part=part['name'], method=method, start=start, stop=start + len(curve),
                                first_threshold='+inf', grouped_trapezoid_AUC=auc))
                            for policy in ('FROZEN50', 'MATCHED_SOURCE_SIZE'):
                                tau = report['calibration'][f'{support_seed}:{50 if policy == "FROZEN50" else size}:{method}']['threshold']
                                actual = point(gpu, tau, known, correct)
                                row = next(r for r in report['metric_rows'] if r['support_seed'] == support_seed
                                    and r['size'] == size and r['part'] == part['name'] and r['method'] == method and r['policy'] == policy)
                                assert actual == row['point'] and retrieval == row['known_retrieval']
                                assert abs(auc - row['adapted_grouped_trapezoid_AUC']) <= 1e-12
                                all_rows.append(dict(training_seed=seed, support_seed=support_seed, size=size,
                                    part=part['name'], method=method, policy=policy, **actual, **retrieval, adapted_AUC=auc))
                                rows_checked += 1
            assert rows_checked == 108 and calibration_checked == 18
            scores_by_train[seed] = (logits, report['calibration'])
            checks.append(dict(training_seed=seed, metric_rows_checked=rows_checked,
                calibration_cells_checked=calibration_checked, replay_logits=int(logits.size),
                max_CPU_FP64_GPU_FP32_error=max_error, actual_updates=report['actual_updates'],
                first100_mean_losses=losses[:100].mean(axis=0).tolist(),
                last100_mean_losses=losses[-100:].mean(axis=0).tolist()))
        assert sorted(scores_by_train) == [1701, 1702, 1703]
        # Existing score controls keep their original FP32 DCG fields untouched.
        for row in old['metric_rows']:
            si = meta['support_seeds'].index(row['seed']); ni = meta['gallery_sizes'].index(row['size'])
            hi = meta['score_names'].index(row['method']); part = next(p for p in meta['parts'] if p['name'] == row['part'])
            sl = slice(part['start'], part['stop'])
            entry = next(g for g in meta['gallery_metadata'] if g['part'] == part['name'] and g['seed'] == row['seed'])
            known = np.isin(y[sl], entry['galleries'][str(row['size'])]['class_ids'])
            correct = known & (pred[si, ni, sl] == y[sl])
            scores = frozen_scores[si, ni, hi, sl]
            assert point(scores, row['point']['threshold'], known, correct) == row['point']
            _, auc = count_curve(scores, known, correct)
            assert abs(auc - row['adapted_grouped_trapezoid_AUC']) <= 1e-12
            kr = ranks[si, ni, sl][known].astype(np.float64)
            retrieval = dict(row['known_retrieval']); retrieval['DCG_FP64'] = float(np.mean(1 / np.log2(kr + 1))); retrieval.pop('DCG')
            all_rows.append(dict(training_seed=0, support_seed=row['seed'], size=row['size'], part=row['part'],
                method=row['method'], policy=row['policy'], **row['point'], **retrieval, adapted_AUC=auc))
        assert len(all_rows) == 540
        means = []
        rate_keys = ['FAR', 'CCR', 'known_acceptance', 'Top1', 'Top5', 'DCG_FP64', 'MRR', 'adapted_AUC']
        for part in meta['parts']:
            for size in meta['gallery_sizes']:
                for method in methods:
                    for policy in ('FROZEN50', 'MATCHED_SOURCE_SIZE'):
                        rows = [r for r in all_rows if (r['part'], r['size'], r['method'], r['policy']) == (part['name'], size, method, policy)]
                        means.append(dict(part=part['name'], size=size, method=method, policy=policy,
                            cells=len(rows), training_seeds=3 if method not in meta['score_names'] else 0,
                            support_seeds=3, **{k: float(np.mean([r[k] for r in rows])) for k in rate_keys}))
        primary = {}; deltas = []; valid_arrays = []
        for name, bootstrap_name, metrics in [('MS_dev', 'MS', ('CCR', 'FAR')), ('base_dev', 'base', ('CCR',))]:
            part = next(p for p in meta['parts'] if p['name'] == name); sl = slice(part['start'], part['stop'])
            groups = meta['signers'][sl]; weights = np.load(measured / f'bootstrap_{bootstrap_name}.npy', allow_pickle=False)
            assert sorted(set(groups)) == meta['group_order'][name] and len(weights) == 2000
            masks = []; corrects = []; raw_accept = []; mlp_accept = []
            for train_seed in sorted(scores_by_train):
                logits, calibration = scores_by_train[train_seed]
                for si, support_seed in enumerate(meta['support_seeds']):
                    entry = next(g for g in meta['gallery_metadata'] if g['part'] == name and g['seed'] == support_seed)
                    known = np.isin(y[sl], entry['galleries']['50']['class_ids'])
                    raw_row = next(r for r in old['metric_rows'] if (r['part'], r['seed'], r['size'], r['method'], r['policy']) == (name, support_seed, 50, 'maxcosine', 'FROZEN50'))
                    raw_accept.append(frozen_scores[si, 2, 0, sl] >= raw_row['point']['threshold'])
                    mlp_accept.append(logits[si, 2, 1, sl] >= calibration[f'{support_seed}:50:FRDisc_frozen_MLP']['threshold'])
                    masks.append(known); corrects.append(known & (pred[si, 2, sl] == y[sl]))
            masks, corrects, raw_accept, mlp_accept = map(np.asarray, (masks, corrects, raw_accept, mlp_accept))
            for metric in metrics:
                denominator = masks if metric == 'CCR' else ~masks
                rs = raw_accept & corrects if metric == 'CCR' else raw_accept
                hs = mlp_accept & corrects if metric == 'CCR' else mlp_accept
                rb, rv = cluster_mean_rate(rs, denominator, groups, weights)
                hb, hv = cluster_mean_rate(hs, denominator, groups, weights)
                for success, expected, valid in [(rs, rb, rv), (hs, hb, hv)]:
                    manual, mv = independent_bootstrap(success, denominator, groups, weights)
                    assert np.array_equal(mv, valid) and np.allclose(manual, expected, atol=1e-12, rtol=0)
                delta, valid = hb - rb, hv & rv
                seed_values = ((hs & denominator).sum(axis=1) / denominator.sum(axis=1)
                               - (rs & denominator).sum(axis=1) / denominator.sum(axis=1)).reshape(3, 3).mean(axis=1)
                primary[f'{name}_{metric}'] = dict(mean_delta=float(seed_values.mean()),
                    training_seed_support_mean_deltas=seed_values.tolist(), **conditional_interval(delta, valid))
                deltas.append(delta); valid_arrays.append(valid)
        ccr = primary['MS_dev_CCR']; far = primary['MS_dev_FAR']; base_ccr = primary['base_dev_CCR']
        gates = dict(MS_CCR_gain_at_least_2pp=ccr['mean_delta'] >= .02,
            every_initialization_CCR_gain_positive=all(x > 0 for x in ccr['training_seed_support_mean_deltas']),
            MS_CCR_conditional_lower_positive=ccr['interval95'] is not None and ccr['interval95'][0] > 0,
            MS_FAR_mean_increase_at_most_2pp=far['mean_delta'] <= .02,
            MS_FAR_conditional_upper_at_most_2pp=far['interval95'] is not None and far['interval95'][1] <= .02,
            base_CCR_loss_at_most_2pp=base_ccr['mean_delta'] >= -.02)
        used = sum(len(x) for x in curves); assert used <= 77000 and len(segments) == 162
        packed = np.zeros((77000, 4), dtype=np.float64); packed[:used] = np.vstack(curves)
        run.save_array('head_count_curves.npy', packed, c['array_specs']['head_count_curves.npy'])
        run.save_array('bootstrap_head_deltas.npy', np.column_stack(deltas), c['array_specs']['bootstrap_head_deltas.npy'])
        run.save_array('bootstrap_head_valid.npy', np.column_stack(valid_arrays), c['array_specs']['bootstrap_head_valid.npy'])
        run.store.write_bytes('per_seed.csv', table_bytes(all_rows))
        run.store.write_bytes('mean.csv', table_bytes(means))
        run.save('head_analysis.json', dict(passed=True, verification_passed=True,
            scientific_screen_passed=all(gates.values()), primary=primary, gates=gates, manual_oracles=oracles,
            primary_order=list(primary), source_check=source_check, head_checks=checks,
            existing_metric_rows_checked=216, head_metric_rows_checked=324,
            curve_segments=segments, curve_points_used=used, curve_unused_tail=77000 - used,
            cells_per_seed=len(all_rows), mean_cells=len(means), resource=resource, feature_check=feature_check,
            replay_tolerance='abs(CPU_FP64 - saved_GPU_FP32) <= 1e-4 + 1e-4*abs(saved_GPU_FP32)',
            bootstrap_scope='2000 fixed original signer multiplicities; equal mean of nine per-cell ratios; supports, source calibration, parameters fixed; three initialization outcomes descriptive.',
            scientific_scope='Exploratory existing frozen-feature controls with extra source labels, unchanged cosine ranking. A utility screen is not novelty or full FR-Disc reproduction. Fail prohibits same-MS rescue tuning of this fixed recipe.',
            GPU_seconds=0, new_encoder_calls=0, fit_calls=0))


if __name__ == '__main__': main()
