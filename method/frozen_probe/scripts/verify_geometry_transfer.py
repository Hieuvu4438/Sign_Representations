"""Rebuild H006 rankings and covariance equations without measurement helpers."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha
from run_context import RunContext


def main():
    p = argparse.ArgumentParser()
    for key in ('config', 'output-dir'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--run-id', required=True)
    p.add_argument('--seed', required=True, type=int)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    with RunContext(a.output_dir, a.run_id, a.config, a.seed) as run:
        import numpy as np
        for name, digest in c['input_hashes'].items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Changed verification input ' + name)
        parent = REPO / c['parent_runtime']
        status = json.loads((parent / 'status.json').read_text())
        if status['status'] != 'completed' or status['exit_code'] != 0:
            raise ValueError('Parent did not complete')
        original = json.loads((parent / 'config.json').read_text())
        for name, digest in original['input_hashes'].items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Changed measured input ' + name)
        for name, digest in status['output_hashes'].items():
            if file_sha(parent / name) != digest:
                raise ValueError('Changed measured output ' + name)
        metadata = json.loads((parent / 'fit_support_metadata.json').read_text())
        report = json.loads((parent / 'geometry_transfer.json').read_text())
        rows = json.loads((parent / 'predictions.json').read_text())
        fitted = np.load(parent / 'fitted_geometry.npz', allow_pickle=False)
        cohorts, samples, readers = {}, {}, {}
        for role in ('base', 'target'):
            identity = json.loads((REPO / original[role + '_identity_config']).read_text())
            cache = json.loads((REPO / original[role + '_cache_report']).read_text())
            cohorts[role] = json.loads((REPO / identity['cohort_manifest']).read_text())
            samples[role] = {s['sample_id']: s for s in cohorts[role]['samples']}
            audit = json.loads((REPO / identity['video_audit']).read_text())
            if file_sha(cache['cache_manifest']) != metadata[role + '_manifest_sha256']:
                raise ValueError('Changed native manifest')
            readers[role] = CacheReader(cache['cache_manifest'], identity['cache_identity'], cohorts[role], audit)
        for field in ('sample_id', 'source_sha256'):
            if {s[field] for s in samples['base'].values()} & {s[field] for s in samples['target'].values()}:
                raise ValueError('Cohorts overlap')
        if set(cohorts['base']['classes']) & set(cohorts['target']['classes']):
            raise ValueError('Class overlap')
        fit_ids = sorted(k for k, s in samples['base'].items() if s['split'] == 'train')
        query_ids = sorted(k for k, s in samples['target'].items() if s['split'] == 'val')
        assert len(fit_ids) == 1313 and len(query_ids) == 383
        assert fit_ids == metadata['fit_ids'] and query_ids == metadata['query_ids']
        labels = [samples['base'][k]['class_id'] for k in fit_ids]
        assert labels == metadata['fit_labels']
        assert Counter(labels) == Counter(metadata['shuffled_labels'])
        assert np.random.default_rng(1701).permutation(labels).tolist() == metadata['shuffled_labels']
        supports = metadata['supports']
        assert set(supports) == {'1701', '1702', '1703'}
        for seed, ids in supports.items():
            expected = [cohorts['target']['supports'][seed]['1'][g][0] for g in cohorts['target']['classes']]
            assert ids == expected and len(set(ids)) == 100
            assert [samples['target'][k]['class_id'] for k in ids] == list(range(100))
            assert all(samples['target'][k]['split'] == 'train' for k in ids)
            assert not ({samples['target'][k]['participant_id'] for k in ids} &
                        {samples['target'][k]['participant_id'] for k in query_ids})
        assert not ({samples['base'][k]['participant_id'] for k in fit_ids} &
                    {samples['target'][k]['participant_id'] for k in query_ids})
        target_ids = sorted(set(query_ids) | {k for ids in supports.values() for k in ids})
        assert metadata['target_loaded_ids'] == target_ids

        def pool(role, ids):
            values = []
            windows = 0
            for key in ids:
                x, activity, _ = readers[role].load(key)
                x = np.asarray(x, dtype=np.float64)
                weights = np.maximum(np.clip(activity[:, 0].astype(np.float64), 0, 1),
                                     np.clip(activity[:, 1].astype(np.float64), 0, 1))
                v = x.mean(axis=0) if weights.sum() <= 1e-12 else (x * weights[:, None]).sum(axis=0) / weights.sum()
                assert np.linalg.norm(v) > 1e-12
                values.append(v / np.linalg.norm(v))
                windows += len(x)
            return np.stack(values), windows

        x, base_windows = pool('base', fit_ids)
        values, target_windows = pool('target', target_ids)
        assert base_windows == metadata['base_windows'] and target_windows == metadata['target_loaded_windows']
        assert np.max(np.abs(x.mean(axis=0) - fitted['mean'])) <= c['numeric_atol']
        covariance_errors = {}
        for mode in ('random', 'total_pca', 'shuffled_pca', 'residual_pca'):
            u = fitted[mode]
            assert u.shape == (768, 8)
            assert np.max(np.abs(u.T @ u - np.eye(8))) <= c['orthogonality_atol']
            if mode == 'random':
                expected, _ = np.linalg.qr(np.random.default_rng(1701).normal(size=(768, 8)), mode='reduced')
                assert np.max(np.abs(expected - u)) <= c['numeric_atol']
                continue
            cov = x.T @ x
            if mode == 'total_pca':
                summed = x.sum(axis=0)
                cov -= np.outer(summed, summed) / len(x)
                singular = fitted['total_singular']
            else:
                used_labels = labels if mode == 'residual_pca' else metadata['shuffled_labels']
                for label in sorted(set(used_labels)):
                    local = x[np.asarray(used_labels) == label]
                    summed = local.sum(axis=0)
                    cov -= np.outer(summed, summed) / len(local)
                singular = fitted['residual_singular' if mode == 'residual_pca' else 'shuffled_singular']
            error = float(np.linalg.norm(cov @ u - u * singular[:8] ** 2) / max(1, np.linalg.norm(cov)))
            assert error <= c['covariance_relative_atol']
            covariance_errors[mode] = error
        modes = ('raw', 'center', 'random', 'total_pca', 'shuffled_pca', 'residual_pca')
        indexed = {(r['mode'], str(r['support_seed']), r['sample_id']): r for r in rows}
        assert len(indexed) == len(rows) == 6894
        assert set(indexed) == {(m, s, k) for m in modes for s in supports for k in query_ids}
        positions = {k: i for i, k in enumerate(target_ids)}
        old_dir = REPO / c['earlier_raw_runtime']
        old_schema = json.loads((old_dir / 'schema.json').read_text())
        old_ranks = np.load(old_dir / 'rankings.npz', allow_pickle=False)
        assert old_schema['query_ids'] == query_ids
        means = {}
        verified = 0
        for mode in modes:
            z = values.copy()
            if mode != 'raw':
                z -= fitted['mean']
                if mode != 'center':
                    u = fitted[mode]
                    z -= (z @ u) @ u.T
            norms = np.sqrt(np.sum(z * z, axis=1))
            assert np.min(norms) > 1e-12
            z /= norms[:, None]
            all_dcg = []
            for seed, support in supports.items():
                gallery = z[[positions[k] for k in support]]
                measured = []
                for q, key in enumerate(query_ids):
                    # Direct squared Euclidean distances, independent of cosine sorting.
                    distances = np.sum((gallery - z[positions[key]]) ** 2, axis=1)
                    ranking = np.lexsort((np.arange(100), distances)).tolist()
                    row = indexed[mode, seed, key]
                    assert row['ranking'] == ranking and sorted(ranking) == list(range(100))
                    truth = samples['target'][key]
                    assert row['label'] == truth['class_id'] and row['participant_id'] == truth['participant_id']
                    rank = ranking.index(truth['class_id']) + 1
                    metrics = {'dcg': 1 / math.log2(rank + 1), 'mrr': 1 / rank,
                               **{'top' + str(k): float(rank <= k) for k in (1, 5, 10, 20)}}
                    assert all(abs(row[k] - v) <= c['numeric_atol'] for k, v in metrics.items())
                    if mode == 'raw':
                        assert ranking == old_ranks['full_activity_s' + seed][q].tolist()
                    measured.append(metrics)
                    verified += 1
                aggregate = {k: math.fsum(r[k] for r in measured) / 383 for k in measured[0]}
                assert all(abs(report['metrics'][mode][seed][k] - v) <= c['numeric_atol'] for k, v in aggregate.items())
                all_dcg.extend(r['dcg'] for r in measured)
            means[mode] = math.fsum(all_dcg) / len(all_dcg)
            assert abs(means[mode] - report['mean_DCG'][mode]) <= c['numeric_atol']
        deltas = {m: means['residual_pca'] - means[m] for m in modes if m != 'residual_pca'}
        assert all(abs(v - report['residual_minus_controls_DCG'][m]) <= c['numeric_atol'] for m, v in deltas.items())
        nonnegative = all(report['metrics']['residual_pca'][s]['dcg'] >= report['metrics']['raw'][s]['dcg'] for s in supports)
        passed = deltas['raw'] >= original['raw_gain_floor'] and min(deltas[m] for m in ('center', 'random', 'total_pca', 'shuffled_pca')) >= original['control_gain_floor'] and nonnegative
        assert passed == report['passed_diagnostic'] and nonnegative == report['all_support_seeds_nonnegative']
        run.save('verification.json', dict(passed=True, rebuilt_rankings=verified, query_count=383,
                 covariance_relative_errors=covariance_errors, earlier_raw_rankings_equal=1149,
                 measured_outputs_sha256=status['output_hashes'], original_inputs_verified=True,
                 mean_DCG=means, supervision_and_access_verified=True, GPU_hours=0,
                 scope='same-agent independent arithmetic implementation; no new scientific trial or independent review'))
        run.status['output_hashes'] = {'verification.json': file_sha(run.output / 'verification.json')}


if __name__ == '__main__':
    main()
