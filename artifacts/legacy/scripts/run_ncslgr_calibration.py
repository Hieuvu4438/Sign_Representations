"""Locked validation-temperature sensitivity for existing categorical heads."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.ncslgr import inclusive_frame_bounds
from signrepr.calibration import categorical_scores, select_temperature, paired_mean_bootstrap

CONFIG = ROOT / 'configs/protocol_ncslgr_calibration_v1.json'
OUTPUT = ROOT / 'runs/ncslgr_calibration_v1'
SOURCE = ROOT / 'runs/ncslgr_diagnostic_v1'
MANIFEST = ROOT / 'data/external/ncslgr/diagnostic_v1/manifest.jsonl'
FEATURES = ROOT / 'features/signrep_ncslgr_diagnostic_v1'
KINDS = ['body_mean_linear', 'face_mean_linear', 'paired_normalized_mean_linear']


def identities():
    files = [Path(__file__), ROOT / 'src/signrepr/calibration.py', ROOT / 'src/signrepr/ncslgr.py',
             ROOT / 'src/signrepr/io.py', MANIFEST, MANIFEST.parent / 'protocol.lock.json',
             FEATURES / 'extraction_report.json', FEATURES / 'index.jsonl', SOURCE / 'summary.json']
    files += [SOURCE / f'{kind}_seed{seed}' / name for kind in KINDS for seed in [0, 1, 2]
              for name in ['head.pt', 'selection.json', 'predictions.csv']]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def lock():
    if CONFIG.exists() or OUTPUT.exists():
        raise ValueError('Preserve existing calibration protocol/run')
    write_json(CONFIG, {'status': 'LOCKED_EXPLORATORY_VALIDATION_TEMPERATURE_SENSITIVITY',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'identity_sha256': identities(),
        'kinds': KINDS, 'seeds': [0, 1, 2], 'temperature_grid': [1., .25, .5, 2., 4., 8.], 'bins': 5,
        'selection': 'Minimum mean categorical validation NLL; deterministic first temperature wins ties. Save selection before test.',
        'primary': 'Calibrated-minus-uncalibrated mean NLL for body head, arithmetic mean over three trained seeds.',
        'secondary': 'Face/paired NLL and multiclass Brier differences; descriptive ECE/reliability bins. No best view selection.',
        'normalization': 'Reuse immutable original train-only mean/SD and head; no head retraining or test-population normalization.',
        'probability_definition': 'Three mutually exclusive positively recorded functional types NEG/WH/YN, conditional on selected cohort. Not absence or interval position/presence probabilities.',
        'statistics': {'resamples': 2000, 'seed': 42, 'group': 'recording_group'}, 'device': 'cpu', 'torch_threads': 1,
        'limits': ['Same-cohort test classification already observed: exploratory sensitivity, not independent confirmatory calibration.',
                   'One validation signer, only27validation utterances, sparse NEG/YN. Validation reused for original LR selection; calibration selection is not independent of head selection.',
                   'One test signer and four shared XML archive groups; conditional source-group CIs are not signer-population inference.',
                   'Finite-bin ECE on88test utterances is noisy; improved NLL alone does not establish calibrated uncertainty.',
                   'No representation adaptation, semantic minimal-pair accuracy or method novelty.']})
    print(json.dumps({'status': 'LOCKED', 'config_sha256': sha256(CONFIG)}))


def load_features():
    index = {r['sample_id']: r for _, r in read_jsonl(FEATURES / 'index.jsonl')}
    values, metadata = {}, {}
    for _, row in read_jsonl(MANIFEST):
        entry = index[row['sample_id']]
        path = Path(entry['shard'])
        meta = json.loads(path.with_suffix('.json').read_text())
        if entry['status'] != 'SUCCESS' or sha256(path) != meta['shard_sha256']:
            raise ValueError('Immutable feature identity differs')
        a, b = inclusive_frame_bounds(row['utterance_start_ms'], row['utterance_end_ms'], 30)
        with np.load(path, allow_pickle=False) as shard:
            keep = shard['valid_mask'] & (shard['center_sec'] >= a/30) & (shard['center_sec'] < b/30)
            if not keep.any():
                raise ValueError('No utterance-supported frozen features')
            raw = shard['embeddings'][keep].mean(0)
        if not np.isfinite(raw).all() or np.linalg.norm(raw) < 1e-10:
            raise ValueError('Invalid pooled feature')
        sid = row['utterance_sample_id']
        values.setdefault(sid, {})[row['view_role']] = (raw/np.linalg.norm(raw)).astype(np.float32)
        metadata[sid] = row
    ids = sorted(values)
    data = {'body_mean_linear': np.stack([values[i]['body'] for i in ids]),
            'face_mean_linear': np.stack([values[i]['face'] for i in ids])}
    paired = np.stack([values[i]['body'] + values[i]['face'] for i in ids])
    data['paired_normalized_mean_linear'] = paired / np.linalg.norm(paired, axis=1, keepdims=True)
    return ids, metadata, data


def run():
    if OUTPUT.exists():
        raise ValueError('Preserve previous attempted calibration suite')
    p = json.loads(CONFIG.read_text())
    if identities() != p['identity_sha256']:
        raise ValueError('Code/source/heads changed after lock')
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'config_sha256': sha256(CONFIG), 'device': 'cpu'}
    write_json(OUTPUT / 'status.json', state)
    try:
        torch.set_num_threads(p['torch_threads'])
        ids, rows, features = load_features()
        labels = {name: i for i, name in enumerate(json.loads((MANIFEST.parent / 'protocol.lock.json').read_text())['classes'])}
        gold = np.array([labels[rows[i]['label']] for i in ids])
        val, test = [np.array([i for i, sid in enumerate(ids) if rows[sid]['split'] == s]) for s in ['val', 'test']]
        groups = [rows[ids[i]]['recording_group'] for i in test]
        results = {}
        for kind in p['kinds']:
            records, arrays = [], {name: [] for name in ['nll', 'brier', 'raw_nll', 'raw_brier']}
            for seed in p['seeds']:
                original = SOURCE / f'{kind}_seed{seed}'
                # Own locally produced, hash-verified checkpoint includes numpy train statistics.
                saved = torch.load(original / 'head.pt', map_location='cpu', weights_only=False)
                head = torch.nn.Linear(768, 3); head.load_state_dict(saved['state_dict'], strict=True); head.eval()
                x = torch.from_numpy(((features[kind]-saved['train_mean'])/np.maximum(saved['train_sd'], 1e-5)).astype(np.float32))
                with torch.inference_mode():
                    val_logits = head(x[val]).numpy()
                temperature, candidates = select_temperature(val_logits, gold[val], p['temperature_grid'])
                directory = OUTPUT / f'{kind}_seed{seed}'; directory.mkdir()
                selection = {'temperature': temperature, 'validation_candidates': candidates, 'test_used_for_selection': False,
                             'validation_samples': len(val), 'original_head_sha256': sha256(original / 'head.pt')}
                write_json(directory / 'selection.json', selection)
                with torch.inference_mode():
                    logits = head(x[test]).numpy()
                raw = categorical_scores(logits, gold[test], 1., p['bins'])
                calibrated = categorical_scores(logits, gold[test], temperature, p['bins'])
                if not np.array_equal(raw['prediction'], calibrated['prediction']):
                    raise ValueError('Positive scalar temperature changed class predictions')
                with (original / 'predictions.csv').open(newline='') as f:
                    previous = {r['sample_id']: r for r in csv.DictReader(f)}
                if set(previous) != {ids[i] for i in test}:
                    raise ValueError('Original prediction population differs')
                prediction_rows = []
                for j, i in enumerate(test):
                    sid = ids[i]
                    if int(previous[sid]['prediction']) != int(raw['prediction'][j]) or int(previous[sid]['gold']) != gold[i]:
                        raise ValueError('Original immutable head/feature predictions differ')
                    if not np.allclose(raw['probability'][j], [float(previous[sid]['probability_'+name]) for name in labels], atol=1e-6):
                        raise ValueError('Original immutable head probabilities differ')
                    prediction_rows.append({'sample_id': sid, 'split': 'test', 'gold': int(gold[i]), 'prediction': int(raw['prediction'][j]),
                        'recording_group': groups[j], 'temperature': temperature, 'raw_nll': float(raw['nll'][j]),
                        'calibrated_nll': float(calibrated['nll'][j]), 'raw_brier': float(raw['brier'][j]), 'calibrated_brier': float(calibrated['brier'][j]),
                        **{f'probability_{name}': float(calibrated['probability'][j, c]) for name, c in labels.items()}})
                with (directory / 'predictions.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
                metrics = {'raw': raw['metrics'], 'calibrated': calibrated['metrics']}
                write_json(directory / 'metrics.json', metrics)
                for name in ['nll', 'brier']:
                    arrays[name].append(calibrated[name]); arrays['raw_'+name].append(raw[name])
                records.append({'seed': seed, 'selection': selection, 'metrics': metrics})
                print(json.dumps({'kind': kind, 'seed': seed, 'temperature': temperature, 'raw_nll': raw['metrics']['mean_nll'],
                                  'calibrated_nll': calibrated['metrics']['mean_nll']}), flush=True)
            results[kind] = {'seeds': records, 'comparisons': {name: paired_mean_bootstrap(groups, np.stack(arrays[name]), np.stack(arrays['raw_'+name])) for name in ['nll', 'brier']}}
        summary = {'status': 'PASS_EXPLORATORY_VALIDATION_TEMPERATURE_SENSITIVITY', 'config_sha256': sha256(CONFIG),
                   'elapsed_seconds': time.perf_counter()-started, 'identity_sha256': identities(), 'results': results,
                   'primary': results['body_mean_linear']['comparisons']['nll'], 'limits': p['limits'],
                   'contribution_status': 'Categorical calibration sensitivity only; no established calibrated uncertainty, adaptation or novelty.'}
        write_json(OUTPUT / 'summary.json', summary); state.update(status=summary['status'], elapsed_seconds=summary['elapsed_seconds'])
        write_json(OUTPUT / 'status.json', state)
    except Exception as error:
        state.update(status='FAIL', failure_reason=type(error).__name__+': '+str(error), elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT / 'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-only', action='store_true')
    args = parser.parse_args()
    lock() if args.lock_only else run()
