"""Prediction/clock audit and explicitly post-hoc duration-prior sensitivity."""
import csv
import hashlib
import json

import numpy as np

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.interval_readouts import interval_iou, macro_values, cluster_interval_bootstrap


def main():
    directory = ROOT / 'runs/ncslgr_intervals_v1'
    protocol_path = ROOT / 'configs/protocol_ncslgr_intervals_v1.json'
    p = json.loads(protocol_path.read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['protocol_sha256'] != sha256(protocol_path) or sha256(ROOT / p['targets']) != p['targets_sha256']:
        raise ValueError('Interval protocol or targets changed')
    data = {r['sample_id']: r for _, r in read_jsonl(ROOT / p['targets'])}
    ids = sorted(sid for sid, r in data.items() if r['split'] == 'test')
    retained = np.asarray([data[s]['scope_supervised'] for s in ids])
    gold = np.asarray([p['classes'].index(data[s]['label']) for s in ids])
    targets = np.asarray([data[s]['recorded_functional_interval_sec'] or [0., 0.] for s in ids])
    supports = np.asarray([data[s]['utterance_support_sec'] for s in ids])
    # Chosen after primary results: sensitivity only, never retrain/select heads.
    means = []
    for label in p['classes']:
        train = [r for r in data.values() if r['split'] == 'train' and r['label'] == label and r['scope_supervised']]
        means.append(np.mean([(np.asarray(r['recorded_functional_interval_sec']) - r['utterance_support_sec'][0]) /
                               np.diff(r['utterance_support_sec'])[0] for r in train], axis=0))
    means = np.asarray(means)
    hashes, scores, prior_scores, error_ids = {}, {}, {}, set()
    for kind in p['readouts']:
        values, priors = [], []
        for seed in p['seeds']:
            run = directory / f'{kind}_seed{seed}'
            path = run / 'predictions.csv'
            with path.open() as f:
                rows = list(csv.DictReader(f))
            table = {r['sample_id']: r for r in rows}
            if len(rows) != len(ids) or set(table) != set(ids):
                raise ValueError('Prediction IDs or exact test coverage differ')
            pred = np.asarray([int(table[s]['prediction']) for s in ids])
            intervals = np.asarray([[float(table[s]['predicted_start_sec']), float(table[s]['predicted_end_sec'])] for s in ids])
            if (intervals[:, 0] < supports[:, 0] - 1e-6).any() or (intervals[:, 1] > supports[:, 1] + 1e-6).any():
                raise ValueError('Predicted intervals outside source support')
            for i, sid in enumerate(ids):
                row, reference = table[sid], data[sid]
                if (row['split'] != 'test' or int(row['gold']) != gold[i]
                        or int(row['scope_supervised']) != int(retained[i])
                        or any(row[k] != reference[k] for k in ['signer', 'recording_group', 'conservative_xml_group'])):
                    raise ValueError('Prediction/source metadata mismatch')
                if retained[i] and not np.array_equal([float(row['gold_start_sec']), float(row['gold_end_sec'])], targets[i]):
                    raise ValueError('Prediction gold boundary differs from source')
                if not retained[i] and (row['gold_start_sec'] or row['gold_end_sec']):
                    raise ValueError('Excluded multiple intervals wrongly treated as single/absent')
            iou = interval_iou(intervals[retained], targets[retained])
            score = iou * (pred[retained] == gold[retained])
            m = json.loads((run / 'metrics.json').read_text())
            if not np.isclose(m['macro_class_aware_interval_IoU'], macro_values(gold[retained], score)[0], atol=1e-10):
                raise ValueError('Saved metric does not reproduce from predictions')
            if kind.endswith('_temporal'):
                boundary_files = list(run.glob('*_boundary.npz'))
                if len(boundary_files) != len(ids):
                    raise ValueError('Boundary probability coverage differs')
                seen = set()
                for file in boundary_files:
                    with np.load(file, allow_pickle=False) as d:
                        sid = str(d['sample_id']); prob, nodes = d['boundary_probabilities'], d['source_node_sec']
                        if sid not in table or sid in seen or prob.shape != (len(nodes), 3, 2):
                            raise ValueError('Boundary probability identity or shape differs')
                        seen.add(sid)
                        if not np.isfinite(prob).all() or (prob < 0).any() or not np.allclose(prob.sum(0), 1, atol=2e-6):
                            raise ValueError('Invalid boundary position distribution')
                        r = table[sid]
                        expected = (prob[:, int(r['prediction'])] * nodes[:, None]).sum(0)
                        if not np.allclose(expected, [float(r['predicted_start_sec']), float(r['predicted_end_sec'])], atol=2e-6, rtol=1e-6):
                            raise ValueError('Probability-weighted original clock does not reproduce interval')
                    hashes[str(file.relative_to(ROOT))] = sha256(file)
            fixed_prior = supports[:, :1] + means[pred] * np.diff(supports, axis=1)
            priors.append(interval_iou(fixed_prior[retained], targets[retained]) * (pred[retained] == gold[retained]))
            values.append(score)
            error_ids.update(sid for sid, keep, value in zip(np.asarray(ids)[retained], np.ones(retained.sum(), bool), score) if value < .5)
            hashes[str(path.relative_to(ROOT))] = sha256(path)
        scores[kind], prior_scores[kind] = np.stack(values), np.stack(priors)
    groups = [data[s]['recording_group'] for s, keep in zip(ids, retained) if keep]
    xml = [data[s]['conservative_xml_group'] for s, keep in zip(ids, retained) if keep]
    report = {'status': 'PASS_INTERVAL_PREDICTION_AND_BOUNDARY_CLOCK_INTEGRITY_WITH_LIMITS',
              'protocol_sha256': sha256(protocol_path), 'summary_sha256': sha256(directory / 'summary.json'),
              'prediction_and_distribution_sha256': hashes, 'script_sha256': sha256(ROOT / 'scripts/audit_ncslgr_intervals.py'),
              'classification_samples': len(ids), 'interval_samples': int(retained.sum()),
              'boundary_distribution_files_verified': sum(k.endswith('.npz') for k in hashes),
              'posthoc_same_classifier_duration_prior': {
                  'status': 'POST_HOC_SENSITIVITY_NO_REFIT_OR_SELECTION',
                  'train_only_class_mean_normalized_endpoints': means.tolist(),
                  'readouts': {k: cluster_interval_bootstrap(gold[retained], groups, v, reference=prior_scores[k]) for k, v in scores.items()},
                  'semantics': 'Use fixed already-selected class predictions, replace boundary prediction with train-only positive single-event class mean fraction. Gold test class is never supplied to predictor.'},
              'posthoc_xml_cluster_primary': cluster_interval_bootstrap(gold[retained], xml, scores['body_temporal'], reference=scores['body_global']),
              'fixed_error_ids': sorted(error_ids, key=lambda sid: hashlib.sha256(('42:' + sid).encode()).hexdigest())[:30],
              'expert_linguistic_reviews_completed': 0,
              'limits': p['limits'] + ['Post-hoc sensitivity cannot replace the locked primary estimator or establish an independently replicated gain.']}
    output = ROOT / 'reports/ncslgr_interval_audit.json'
    if output.exists():
        raise ValueError('Preserve completed audit')
    write_json(output, report)
    print(json.dumps({'status': report['status'], 'interval_samples': report['interval_samples'],
                      'duration_prior_sensitivity': {k: {'learned': v['point'], 'prior': v['reference_point'], 'delta': v['delta'], 'delta_ci95': v['delta_ci95']}
                        for k, v in report['posthoc_same_classifier_duration_prior']['readouts'].items()}}))


if __name__ == '__main__':
    main()
