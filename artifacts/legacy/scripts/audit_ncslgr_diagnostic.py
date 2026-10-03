"""Post-hoc grouping sensitivity of fixed predictions; never fit or select a head."""
import csv
import hashlib
import json

import numpy as np

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.statistics import grouped_bootstrap


def main():
    directory = ROOT / 'runs/ncslgr_diagnostic_v1'
    summary = json.loads((directory / 'summary.json').read_text())
    manifest = ROOT / 'data/external/ncslgr/diagnostic_v1/manifest.jsonl'
    protocol_path = manifest.parent / 'protocol.lock.json'
    protocol = json.loads(protocol_path.read_text())
    if sha256(protocol_path) != summary['protocol_sha256'] or sha256(manifest) != protocol['manifest_sha256']:
        raise ValueError('Locked diagnostic artifacts changed')
    metadata = {row['utterance_sample_id']: row for _, row in read_jsonl(manifest)}
    test_ids = sorted(sid for sid, row in metadata.items() if row['split'] == 'test')
    other_groups = {row['conservative_xml_group'] for row in metadata.values() if row['split'] != 'test'}
    gold = np.asarray([protocol['classes'].index(metadata[sid]['label']) for sid in test_ids])
    xml_groups = [metadata[sid]['conservative_xml_group'] for sid in test_ids]
    strict = np.asarray([group not in other_groups for group in xml_groups])
    source_hashes, correct = {}, {}
    for kind in protocol['readouts'][1:]:
        tables = []
        for seed in protocol['training_seeds']:
            path = directory / f'{kind}_seed{seed}/predictions.csv'
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            by_id = {row['sample_id']: row for row in rows}
            if len(rows) != len(test_ids) or set(by_id) != set(test_ids):
                raise ValueError('Prediction coverage or uniqueness mismatch')
            for i, sid in enumerate(test_ids):
                row, meta = by_id[sid], metadata[sid]
                if (int(row['gold']) != gold[i] or row['split'] != 'test'
                    or row['recording_group'] != meta['recording_group']
                    or row['conservative_xml_group'] != meta['conservative_xml_group']
                    or row['signer_group'] != meta['signer']
                    or int(row['correct']) != int(row['prediction'] == row['gold'])):
                    raise ValueError('Prediction-to-source metadata mismatch')
            tables.append([int(by_id[sid]['correct']) for sid in test_ids])
            source_hashes[str(path.relative_to(ROOT))] = sha256(path)
        correct[kind] = np.asarray(tables, dtype=float)
    analyses = {}
    for name, keep in [('all_test_clustered_by_xml', np.ones(len(test_ids), dtype=bool)),
                       ('exclude_shared_xml_test_subset', strict)]:
        selected_groups = [group for group, retained in zip(xml_groups, keep) if retained]
        analyses[name] = {
            'status': 'POST_HOC_SENSITIVITY_NO_REFIT', 'samples': int(keep.sum()),
            'xml_groups': sorted(set(selected_groups)),
            'class_counts': {label: int((gold[keep] == c).sum()) for c, label in enumerate(protocol['classes'])},
            'readouts': {kind: grouped_bootstrap(gold[keep], selected_groups, table[:, keep],
                                                 resamples=2000, seed=42)
                         for kind, table in correct.items()},
            'paired_vs_body': grouped_bootstrap(gold[keep], selected_groups,
                correct['paired_normalized_mean_linear'][:, keep],
                reference=correct['body_mean_linear'][:, keep], resamples=2000, seed=42)}
    error_ids = {sid for i, sid in enumerate(test_ids)
                 if any((table[:, i] == 0).any() for table in correct.values())}
    selected_errors = sorted(error_ids, key=lambda sid: hashlib.sha256(('42:' + sid).encode()).hexdigest())[:30]
    report = {
        'status': 'PASS_PREDICTION_INTEGRITY_WITH_LIMITS', 'protocol_sha256': sha256(protocol_path),
        'summary_sha256': sha256(directory / 'summary.json'), 'manifest_sha256': sha256(manifest),
        'script_sha256': sha256(ROOT / 'scripts/audit_ncslgr_diagnostic.py'),
        'prediction_sha256': source_hashes, 'primary_test_samples': len(test_ids),
        'shared_xml_groups': sorted(set(xml_groups) & other_groups), 'analyses': analyses,
        'fixed_error_ids': selected_errors, 'expert_linguistic_reviews_completed': 0,
        'limits': ['Sensitivity specified after seeing primary results; no new primary claim, fitting or selection.',
                   'Excluding shared test XML files cannot remove a possible shared acquisition process elsewhere.',
                   'Few XML groups and one test signer limit uncertainty estimates; no signer-population inference.',
                   'Prediction integrity checks are machine checks, not review of ASL semantics or video identity.']}
    output = ROOT / 'reports/ncslgr_diagnostic_audit.json'
    if output.exists():
        raise ValueError('Preserve completed diagnostic audit')
    write_json(output, report)
    print(json.dumps({'status': report['status'], 'analyses': {
        name: {'samples': entry['samples'], 'class_counts': entry['class_counts'],
               'macro_recall': {kind: stats['metrics']['macro_recall']['point']
                                for kind, stats in entry['readouts'].items()}}
        for name, entry in analyses.items()}}))


if __name__ == '__main__':
    main()
