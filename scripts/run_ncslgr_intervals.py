"""Prospectively locked, exploratory recorded-functional-interval diagnostic.

No absent-target detector or grammatical-scope gold is fabricated. All 222
utterances train classification; endpoint loss uses only 179 single-event
utterances. Classes and intervals are supervision, never forward inputs.
"""
import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json, write_jsonl
from signrepr.ncslgr import inclusive_frame_bounds
from signrepr.interval_readouts import IntervalReadout, interval_iou, macro_values, cluster_interval_bootstrap

PROTOCOL = ROOT / 'configs/protocol_ncslgr_intervals_v1.json'
OUTPUT = ROOT / 'runs/ncslgr_intervals_v1'
MANIFEST = ROOT / 'data/external/ncslgr/diagnostic_v1/manifest.jsonl'
FEATURES = ROOT / 'features/signrep_ncslgr_diagnostic_v1'


def identities():
    files = [Path(__file__), ROOT / 'src/signrepr/interval_readouts.py', ROOT / 'src/signrepr/ncslgr.py',
             ROOT / 'src/signrepr/io.py', MANIFEST, MANIFEST.parent / 'protocol.lock.json',
             FEATURES / 'extraction_report.json', FEATURES / 'index.jsonl']
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def locked_rows():
    rows = {r['utterance_sample_id']: r for _, r in read_jsonl(MANIFEST) if r['view_role'] == 'body'}
    result = []
    for sid in sorted(rows):
        r = rows[sid]
        start, stop = inclusive_frame_bounds(r['utterance_start_ms'], r['utterance_end_ms'], 30)
        cores = r['functional_core_intervals']
        if not cores or any(not core['is_functional'] for core in cores):
            raise ValueError('Only positively recorded functional tiers are eligible')
        target = None
        if len(cores) == 1:
            a, b = inclusive_frame_bounds(cores[0]['absolute_start_ms'], cores[0]['absolute_end_ms'], 30)
            if not start <= a < b <= stop:
                raise ValueError('Functional interval outside source utterance support')
            target = [a/30, b/30]
        result.append({'sample_id': sid, 'label': r['label'], 'split': r['split'],
                       'signer': r['signer'], 'recording_group': r['recording_group'],
                       'conservative_xml_group': r['conservative_xml_group'],
                       'utterance_support_sec': [start/30, stop/30],
                       'scope_supervised': target is not None,
                       'recorded_functional_interval_sec': target,
                       'source_interval_count': len(cores),
                       'scope_exclusion': None if target else 'MULTIPLE_RECORDED_FUNCTIONAL_INTERVALS',
                       'event_id': cores[0]['event_id'] if target else None})
    return result


def lock():
    if PROTOCOL.exists() or OUTPUT.exists():
        raise ValueError('Do not replace a locked or attempted diagnostic')
    rows = locked_rows()
    counts = {split: dict(Counter(r['label'] for r in rows if r['split'] == split and r['scope_supervised']))
              for split in ['train', 'val', 'test']}
    metadata = ROOT / 'data/external/ncslgr/intervals_v1/targets.jsonl'
    write_jsonl(metadata, rows)
    p = {'status': 'LOCKED_EXPLORATORY_RECORDED_INTERVAL_DIAGNOSTIC', 'protocol_version': 'ncslgr_intervals_v1',
         'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'identity_sha256': identities(), 'targets': str(metadata.relative_to(ROOT)), 'targets_sha256': sha256(metadata),
         'classes': ['NEG', 'WH', 'YN'], 'classification_utterances': len(rows),
         'interval_utterances': sum(r['scope_supervised'] for r in rows), 'interval_counts': counts,
         'readouts': ['body_global', 'body_temporal', 'face_global', 'face_temporal'],
         'primary_comparison': 'body_temporal_minus_body_global',
         'primary_metric': 'macro_class_aware_interval_IoU',
         'validation_selection': 'final-epoch macro class-aware interval IoU on single-interval validation utterances; first LR wins ties',
         'secondary_metrics': ['classification_macro_recall_all_utterances', 'class_agnostic_mean_IoU',
                               'class_aware_interval_success_at_IoU_0.3_0.5_0.7', 'mean_onset_offset_error_ms'],
         'seeds': [0, 1, 2], 'learning_rates': [0.01, 0.03, 0.1], 'epochs': 100,
         'optimizer': 'AdamW', 'weight_decay': 0.0001, 'endpoint_loss_weight': 1.0,
         'endpoint_loss': 'SmoothL1 beta0.1, normalized to utterance support, only source-positive single interval; class index gathers supervision after forward',
         'head_parameters': 6915, 'capacity': 'Same Linear768->3 with bias plus Linear768->6 without bias; every counted parameter affects outputs. No encoder adaptation.',
         'input_policy': 'Frozen SignRep embeddings; valid window centers in known utterance support; per-token L2 then train-only utterance-weighted mean/SD. Add source utterance endpoint nodes with nearest supported embedding, identically for both controls. No gold core, phenotype or text enters model inputs.',
         'temporal_policy': 'Normalized start/end boundary distributions over candidate nodes; expected positions, sorted and probability channels reordered with endpoints. Distributions express boundary position, not phenotype presence/absence.',
         'global_policy': 'Uniform mean of identical candidate tokens, six sigmoid endpoint regressions sorted per predicted class. Same supplied duration, parameters, seeds, LR grid and endpoint labels as temporal control.',
         'timestamp_policy': 'Verified source30FPS; official round(start_ms*FPS/1000), inclusive end round(end_ms*FPS/1000)+1. Intervals half-open, node clocks original source timestamps.',
         'annotation_resolution_ms': 1000/30,
         'negative_policy': 'No absence classes or background BCE. Multiple-core utterances retain class CE but no endpoint loss/IoU. Functional-core annotation is not independently adjudicated complete grammatical scope.',
         'statistics': {'resamples': 2000, 'seed': 42, 'group': 'recording_group', 'seed_aggregation': 'all3arithmeticmean'},
         'prospective_scope': 'Locked before new interval model metrics; earlier same-cohort classification test results already seen. Exploratory development diagnostic, not independent confirmatory evaluation.',
         'limits': ['One test signer; very small NEG/YN validation supports.', 'Four XML archive groups cross splits; source-sequence disjoint does not establish independent acquisition or pretraining exclusion.',
                    'Source functional-core interval is distinct from anatomical phases and fully adjudicated grammatical scope.',
                    'Camera comparisons do not isolate facial information causally.', 'Single-interval subset excludes43utterances; no general multiple-event or absent-phenomenon detection claim.',
                    'Boundary uncertainty distributions are uncalibrated. Frozen readout diagnostic is not a new representation method.']}
    write_json(PROTOCOL, p)
    print(json.dumps({'status': p['status'], 'protocol_sha256': sha256(PROTOCOL), 'counts': counts}))


def load_data(protocol):
    if identities() != protocol['identity_sha256']:
        raise ValueError('Code, extraction, input or source protocol identity changed')
    if sha256(ROOT / protocol['targets']) != protocol['targets_sha256']:
        raise ValueError('Locked interval targets changed')
    rows = [r for _, r in read_jsonl(ROOT / protocol['targets'])]
    source = {r['sample_id']: r for _, r in read_jsonl(MANIFEST)}
    index = {r['sample_id']: r for _, r in read_jsonl(FEATURES / 'index.jsonl')}
    sequences = {'body': [], 'face': []}
    clocks = []
    for row in rows:
        a, b = row['utterance_support_sec']
        view_clocks = []
        for view in sequences:
            sid = row['sample_id'] + ':' + view
            if index[sid]['status'] != 'SUCCESS':
                raise ValueError('Failed features must not silently disappear')
            path = Path(index[sid]['shard'])
            meta = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != meta['shard_sha256']:
                raise ValueError('Feature shard checksum differs')
            with np.load(path, allow_pickle=False) as shard:
                keep = shard['valid_mask'] & (shard['center_sec'] >= a) & (shard['center_sec'] < b)
                tokens, clock = shard['embeddings'][keep], shard['center_sec'][keep]
                if not len(tokens) or not np.isfinite(tokens).all() or not (np.diff(clock) > 0).all():
                    raise ValueError('Invalid feature or source clock support')
                tokens = tokens / np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-10)
                tokens = np.vstack([tokens[0], tokens, tokens[-1]]).astype(np.float32)
                clock = np.concatenate([[a], clock, [b]])
            if source[sid]['split'] != row['split'] or source[sid]['label'] != row['label']:
                raise ValueError('View/target identity differs')
            sequences[view].append(tokens)
            view_clocks.append(clock)
        if not np.array_equal(view_clocks[0], view_clocks[1]):
            raise ValueError('Body/face candidate clocks differ')
        clocks.append(view_clocks[0])
    return rows, sequences, clocks


def evaluate(model, x, mask, positions, indices, y, supports, targets, supervised):
    model.eval()
    with torch.inference_mode():
        logits, bounds, distributions = model(x[indices], mask[indices], positions[indices])
        pred = logits.argmax(1).numpy()
        p = bounds[torch.arange(len(indices)), torch.from_numpy(pred)].numpy()
        intervals = supports[indices, :1] + p * np.diff(supports[indices], axis=1)
        retained = supervised[indices]
        iou = interval_iou(intervals[retained], targets[indices][retained])
        correct = pred == y[indices].numpy()
        class_aware = iou * correct[retained]
        g = y[indices].numpy()
        metrics = {'classification_macro_recall': float(macro_values(g, correct)[0]),
                   'classification_top1': float(correct.mean()), 'classification_samples': len(indices),
                   'macro_class_aware_interval_IoU': float(macro_values(g[retained], class_aware)[0]),
                   'class_agnostic_mean_IoU': float(iou.mean()), 'interval_samples': int(retained.sum()),
                   'mean_onset_offset_error_ms': (np.abs(intervals[retained] - targets[indices][retained]).mean(0) * 1000).tolist(),
                   'class_aware_success': {str(t): float((correct[retained] & (iou >= t)).mean()) for t in [.3, .5, .7]}}
    return metrics, pred, intervals, class_aware, distributions, logits.softmax(1).numpy()


def run():
    if OUTPUT.exists():
        raise ValueError('Preserve attempted or completed suite; reviewed recovery needs a new run ID')
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'LOCKED_EXPLORATORY_RECORDED_INTERVAL_DIAGNOSTIC':
        raise ValueError('Prospective interval protocol missing')
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows, sequences, clocks = load_data(protocol)
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(), 'protocol_sha256': sha256(PROTOCOL)})
    try:
        labels = {name: i for i, name in enumerate(protocol['classes'])}
        y = torch.tensor([labels[r['label']] for r in rows])
        splits = {s: np.asarray([i for i, r in enumerate(rows) if r['split'] == s]) for s in ['train', 'val', 'test']}
        supervised = np.asarray([r['scope_supervised'] for r in rows])
        supports = np.asarray([r['utterance_support_sec'] for r in rows])
        targets = np.asarray([r['recorded_functional_interval_sec'] or [0., 0.] for r in rows])
        normalized_targets = torch.from_numpy(((targets - supports[:, :1]) / np.diff(supports, axis=1)).astype(np.float32))
        train, val, test = [splits[s] for s in ['train', 'val', 'test']]
        scope_train = np.flatnonzero(supervised[train])
        width = max(map(len, clocks))
        mask = torch.zeros(len(rows), width, dtype=torch.bool)
        positions = torch.zeros(len(rows), width)
        for i, clock in enumerate(clocks):
            mask[i, :len(clock)] = True
            positions[i, :len(clock)] = torch.from_numpy(((clock - supports[i, 0]) / np.diff(supports[i])[0]).astype(np.float32))
        summaries, tables = {}, {}
        for kind in protocol['readouts']:
            view, mode = kind.split('_')
            raw = sequences[view]
            # Give each train utterance equal weight despite variable token counts.
            mean = np.stack([raw[i].mean(0) for i in train]).mean(0)
            second = np.stack([(raw[i] ** 2).mean(0) for i in train]).mean(0)
            sd = np.sqrt(np.maximum(second - mean**2, 1e-10))
            x = torch.zeros(len(rows), width, 768)
            for i, seq in enumerate(raw):
                x[i, :len(seq)] = torch.from_numpy((seq - mean) / sd)
            models, scores = [], []
            for seed in protocol['seeds']:
                selected, candidates = None, []
                for lr in protocol['learning_rates']:
                    torch.manual_seed(seed)
                    model = IntervalReadout(768, 3, temporal=mode == 'temporal')
                    if sum(p.numel() for p in model.parameters()) != protocol['head_parameters']:
                        raise ValueError('Capacity matching changed')
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=protocol['weight_decay'])
                    for epoch in range(protocol['epochs']):
                        model.train(); optimizer.zero_grad()
                        logits, bounds, _ = model(x[train], mask[train], positions[train])
                        ce = torch.nn.functional.cross_entropy(logits, y[train])
                        p = bounds[scope_train, y[train][scope_train]]
                        loss_bounds = torch.nn.functional.smooth_l1_loss(p, normalized_targets[train][scope_train], beta=.1)
                        loss = ce + protocol['endpoint_loss_weight'] * loss_bounds
                        if not torch.isfinite(loss):
                            raise ValueError('Non-finite loss')
                        loss.backward(); optimizer.step()
                    measured = evaluate(model, x, mask, positions, val, y, supports, targets, supervised)[0]
                    candidates.append({'learning_rate': lr, 'validation': measured, 'final_train_loss': float(loss.detach())})
                    if selected is None or measured[protocol['primary_metric']] > selected['validation'][protocol['primary_metric']]:
                        selected = {'model': model, 'learning_rate': lr, 'validation': measured}
                directory = OUTPUT / f'{kind}_seed{seed}'; directory.mkdir()
                selection = {'seed': seed, 'learning_rate': selected['learning_rate'], 'validation': selected['validation'],
                             'candidates': candidates, 'head_parameters': protocol['head_parameters'],
                             'protocol_sha256': sha256(PROTOCOL), 'test_used_for_selection': False}
                write_json(directory / 'selection.json', selection)
                torch.save({'state_dict': selected['model'].state_dict(), 'train_mean': mean, 'train_sd': sd,
                            'selection': selection}, directory / 'head.pt')
                m, predicted, intervals, values, distributions, probabilities = evaluate(
                    selected['model'], x, mask, positions, test, y, supports, targets, supervised)
                write_json(directory / 'metrics.json', m)
                output_rows = []
                for j, i in enumerate(test):
                    r = rows[i]
                    output_rows.append({'sample_id': r['sample_id'], 'split': 'test', 'gold': int(y[i]),
                        'prediction': int(predicted[j]), 'scope_supervised': int(supervised[i]),
                        'predicted_start_sec': float(intervals[j, 0]), 'predicted_end_sec': float(intervals[j, 1]),
                        'gold_start_sec': float(targets[i, 0]) if supervised[i] else None,
                        'gold_end_sec': float(targets[i, 1]) if supervised[i] else None,
                        'recording_group': r['recording_group'], 'conservative_xml_group': r['conservative_xml_group'],
                        'signer': r['signer'], **{f'probability_{name}': float(probabilities[j, k]) for name, k in labels.items()}})
                    if distributions is not None:
                        path = directory / (str(i) + '_boundary.npz')
                        np.savez_compressed(path, source_node_sec=clocks[i],
                                            boundary_probabilities=distributions[j, :len(clocks[i])].numpy().reshape(-1, 3, 2),
                                            sample_id=r['sample_id'], semantics='Uncalibrated position distributions, not phenomenon presence/absence')
                with (directory / 'predictions.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(output_rows[0])); w.writeheader(); w.writerows(output_rows)
                models.append({'seed': seed, 'selection': selection, 'metrics': m})
                scores.append(values)
                print(json.dumps({'readout': kind, 'seed': seed, 'metrics': m}), flush=True)
            tables[kind] = np.stack(scores)
            groups = [rows[i]['recording_group'] for i in test if supervised[i]]
            gold = y[test][supervised[test]].numpy()
            summaries[kind] = {'seeds': models, 'primary_bootstrap': cluster_interval_bootstrap(gold, groups, tables[kind])}
        primary = cluster_interval_bootstrap(gold, groups, tables['body_temporal'], reference=tables['body_global'])
        summary = {'status': 'PASS_EXPLORATORY_RECORDED_INTERVAL_DIAGNOSTIC',
                   'protocol_sha256': sha256(PROTOCOL), 'script_sha256': sha256(Path(__file__)),
                   'elapsed_seconds': time.perf_counter() - started, 'head_parameters': protocol['head_parameters'],
                   'classification_counts': {s: len(i) for s, i in splits.items()},
                   'interval_counts': {s: int(supervised[i].sum()) for s, i in splits.items()},
                   'results': summaries, 'body_temporal_vs_global': primary, 'limits': protocol['limits'],
                   'contribution_status': 'Frozen boundary-readout diagnostic only; no representation adaptation or established method novelty.'}
        write_json(OUTPUT / 'summary.json', summary)
        write_json(OUTPUT / 'status.json', {'status': summary['status'], 'elapsed_seconds': summary['elapsed_seconds']})
        print(json.dumps({'status': summary['status'], 'primary': primary, 'elapsed_seconds': summary['elapsed_seconds']}))
    except Exception as error:
        write_json(OUTPUT / 'status.json', {'status': 'FAIL', 'failure_reason': type(error).__name__ + ': ' + str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-only', action='store_true')
    args = parser.parse_args()
    lock() if args.lock_only else run()
