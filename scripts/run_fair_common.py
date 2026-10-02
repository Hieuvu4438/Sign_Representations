"""Locked equal-grid lexical controls on complete common-backbone coverage.

The optional waiter observes artifacts only, never starts/retries extraction.
All fitting runs on CPU, leaving the one allocated GPU to native extraction.
"""
import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.probes import LinearProbe, TemporalProbe, classification_metrics, normalized_mean
from signrepr.statistics import grouped_bootstrap

CONFIG = ROOT / 'configs/protocol_fair_common_v1.json'
OUTPUT = ROOT / 'runs/fair_common_v1'
MANIFEST = ROOT / 'data/manifests/common_v1.jsonl'
NATIVE = ROOT / 'features/shubert_native_common_v1'
SIGNREP = ROOT / 'features/signrep_common_v1'


def code_identity():
    return {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / 'src/signrepr/probes.py',
             ROOT / 'src/signrepr/statistics.py', ROOT / 'src/signrepr/io.py']}


def lock():
    if CONFIG.exists() or OUTPUT.exists():
        raise ValueError('Preserve locked protocol and attempted suite')
    original = yaml.safe_load((ROOT / 'configs/protocol_common_v1.yaml').read_text())
    p = {'status': 'LOCKED_EXPLORATORY_EQUAL_GRID_CONTROLS', 'protocol_version': 'fair_common_v1',
         'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'code_sha256': code_identity(),
         'manifest_sha256': sha256(MANIFEST), 'label_mapping': original['label_mapping'],
         'label_mapping_sha256': original['label_mapping_sha256'],
         'source_protocol_sha256': sha256(ROOT / 'configs/protocol_common_v1.yaml'),
         'native_feature_protocol_sha256': sha256(ROOT / 'configs/protocol_native_common_v1.json'),
         'native_cache_identity': json.loads((NATIVE / 'cohort.lock.json').read_text()),
         'signrep_report_sha256': sha256(SIGNREP / 'extraction_report.json'),
         'backbones': ['signrep', 'shubert_last', 'shubert_average'],
         'controls': ['B00', 'B01', 'B02', 'B03'], 'seeds': [0, 1, 2],
         'learning_rates': [0.0001, 0.001, 0.01], 'epochs': 100, 'weight_decay': 0.01,
         'temporal_width': 64, 'capacity_match_relative_tolerance': 0.001,
         'batch_size': 64, 'device': 'cpu', 'torch_threads': 1,
         'selection': 'Validation macro recall each epoch, identical three LR choices and100epochs for B02/B03; deterministic first tie. Save selection before test.',
         'population': 'Original train-selected200classes; common subset where SignRep has nonpadded valid windows AND SHuBERT has at least one all-four-stream observed frame. Per-ID exclusions, split/class coverage mandatory; stop if any class loses all train/validation/test support.',
         'feature_policy': 'Keep original temporal positions and false masks for missing/padded tokens; never compress gaps into adjacent observations. Last FFN and fixed average of all12 both reported without best-test-layer selection.',
         'primary_comparison': 'B03_signrep_minus_B02_signrep',
         'secondary_comparisons': ['B02_shubert_last_minus_B02_signrep', 'B02_shubert_average_minus_B02_signrep'],
         'primary_metric': 'macro_recall', 'statistics': {'resamples': 2000, 'seed': 42, 'group': 'recording_group'},
         'limits': ['Earlier SignRep test metrics already observed: this is exploratory equal-grid sensitivity, not independent confirmatory evidence.',
                    'Different native token rates, receptive fields and cue preprocessing remain; same source clips and heads do not establish matched-context causal backbone comparison.',
                    'Mask excludes missing frames from readout but native context can include author carry-forward crops.',
                    'No encoder adaptation or full paper benchmark reproduction; pretraining overlap remains unknown.',
                    'Conditional cluster intervals do not include training randomness or signer-population uncertainty.']}
    write_json(CONFIG, p)
    print(json.dumps({'status': p['status'], 'config_sha256': sha256(CONFIG)}))


def native_ready():
    s = json.loads((NATIVE / 'status.json').read_text())
    if s['status'] == 'RUNNING':
        return False, s
    if s['status'] != 'PASS':
        raise ValueError('Native extraction is terminal without full PASS: ' + str(s['status']))
    report = NATIVE / 'extraction_report.json'
    return report.exists(), s


def prepare(protocol):
    if code_identity() != protocol['code_sha256'] or sha256(MANIFEST) != protocol['manifest_sha256']:
        raise ValueError('Frozen code or manifest changed')
    if sha256(ROOT / 'configs/protocol_native_common_v1.json') != protocol['native_feature_protocol_sha256']:
        raise ValueError('Native protocol changed')
    if sha256(ROOT / 'configs/protocol_common_v1.yaml') != protocol['source_protocol_sha256']:
        raise ValueError('Original lexical protocol changed')
    if sha256(SIGNREP / 'extraction_report.json') != protocol['signrep_report_sha256']:
        raise ValueError('SignRep extraction report changed')
    if sha256(ROOT / protocol['label_mapping']) != protocol['label_mapping_sha256']:
        raise ValueError('Train-only label vocabulary changed')
    rows = [r for _, r in read_jsonl(MANIFEST)]
    mapping = json.loads((ROOT / protocol['label_mapping']).read_text())['mapping']
    ix = {}
    for b, folder in [('signrep', SIGNREP), ('native', NATIVE)]:
        entries = [r for _, r in read_jsonl(folder / 'index.jsonl')]
        ix[b] = {r['sample_id']: r for r in entries}
        if len(entries) != len(ix[b]):
            raise ValueError('Duplicate feature IDs must not disappear in a dictionary')
    if any(set(i) != {r['sample_id'] for r in rows} for i in ix.values()):
        raise ValueError('Complete extraction coverage or unique ID sets disagree')
    extraction = json.loads((NATIVE / 'extraction_report.json').read_text())
    if extraction['status'] != 'PASS' or extraction['manifest_sha256'] != protocol['manifest_sha256']:
        raise ValueError('Terminal native extraction does not match population')
    samples = {b: {s: [] for s in ['train', 'val', 'test']} for b in protocol['backbones']}
    excluded = []
    for row in rows:
        sid, loaded = row['sample_id'], {}
        reasons = []
        for b in ix:
            entry = ix[b][sid]
            if entry['status'] != 'SUCCESS':
                raise ValueError('Extraction failure requires reviewed coverage, not silent dropping')
            path = Path(entry['shard'])
            meta = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != meta['shard_sha256']:
                raise ValueError('Cached feature checksum mismatch')
            if b == 'native':
                if any(meta['cache_material'].get(k) != v for k, v in protocol['native_cache_identity'].items()):
                    raise ValueError('Native feature cache identity differs')
            with np.load(path, allow_pickle=False) as data:
                valid = data['valid_mask'] & data['valid_frame_mask'].all(axis=1)
                if b == 'native' and not np.array_equal(valid, data['stream_observed_mask'].all(axis=1)):
                    raise ValueError('Native stream validity differs')
                if not valid.any():
                    reasons.append(b + '_ZERO_VALID_NONPADDED_TOKENS')
                keys = {'signrep': 'embeddings'} if b == 'signrep' else {
                    'shubert_last': 'embeddings', 'shubert_average': 'embeddings_layer_average'}
                for name, key in keys.items():
                    tokens = data[key].astype(np.float32)
                    clock = data['center_sec']
                    if tokens.shape != (len(valid), 768) or not np.isfinite(tokens).all() or not (np.diff(clock) > 0).all():
                        raise ValueError('Invalid tokens or source chronology')
                    loaded[name] = (tokens, valid)
        if reasons:
            excluded.append({'sample_id': sid, 'split': row['official_split'], 'lexical_id': row['lexical_id'], 'reasons': reasons})
            continue
        for name, (tokens, valid) in loaded.items():
            samples[name][row['official_split']].append({'row': row, 'tokens': tokens, 'valid': valid,
                'label': mapping[row['lexical_id']], 'pooled': normalized_mean(tokens, valid)})
    counts = {s: len(v) for s, v in samples['signrep'].items()}
    supports = {s: np.bincount([v['label'] for v in values], minlength=len(mapping)).tolist()
                for s, values in samples['signrep'].items()}
    write_json(OUTPUT / 'coverage.json', {'counts': counts, 'class_support': supports, 'exclusions': excluded,
               'native_terminal_report_sha256': sha256(NATIVE / 'extraction_report.json'),
               'native_terminal_index_sha256': sha256(NATIVE / 'index.jsonl'),
               'signrep_index_sha256': sha256(SIGNREP / 'index.jsonl')})
    if any(min(support) == 0 for support in supports.values()):
        raise ValueError('At least one locked class has no surviving train/val/test support; no automatic vocabulary shrink')
    return samples, len(mapping)


def collate(samples, indices):
    width = max(len(samples[i]['tokens']) for i in indices)
    x = np.zeros((len(indices), width, 768), np.float32)
    mask = np.zeros((len(indices), width), np.float32)
    for j, i in enumerate(indices):
        seq = samples[i]
        x[j, :len(seq['tokens'])] = seq['tokens'] * seq['valid'][:, None]
        mask[j, :len(seq['tokens'])] = seq['valid']
    return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor([samples[i]['label'] for i in indices])


def evaluate(model, samples, batch):
    model.eval(); scores = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch):
            x, mask, _ = collate(samples, list(range(start, min(start + batch, len(samples)))))
            scores.append(model(x, mask).softmax(-1).numpy())
    return np.concatenate(scores)


def run(wait):
    if OUTPUT.exists():
        raise ValueError('Attempt already exists; preserve outcomes and inspect waiter')
    p = json.loads(CONFIG.read_text())
    if code_identity() != p['code_sha256']:
        raise ValueError('Code changed after protocol lock')
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    state = {'status': 'WAITING_FOR_NATIVE', 'pid': os.getpid(), 'config_sha256': sha256(CONFIG), 'device': 'cpu'}
    try:
        while True:
            ready, native = native_ready()
            if ready:
                break
            if not wait:
                raise ValueError('Full native feature cache pending; no partial test inference permitted')
            state.update(native_completed=native.get('success', 0), native_samples=native['samples'],
                         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         elapsed_seconds=time.perf_counter() - started)
            write_json(OUTPUT / 'status.json', state)
            time.sleep(30)
        state['status'] = 'RUNNING'; write_json(OUTPUT / 'status.json', state)
        torch.set_num_threads(p['torch_threads'])
        data, classes = prepare(p)
        fit_started = time.perf_counter()
        results, correct = {}, {}
        for backbone, splits in data.items():
            train, val, test = [splits[s] for s in ['train', 'val', 'test']]
            gold = np.asarray([s['label'] for s in test]); train_gold = np.asarray([s['label'] for s in train])
            val_gold = np.asarray([s['label'] for s in val])
            counts = {s: len(v) for s, v in splits.items()}
            for baseline in p['controls']:
                tables, records = [], []
                seeds = [0] if baseline in ['B00', 'B01'] else p['seeds']
                for seed in seeds:
                    runid = f'{backbone}_{baseline}_seed{seed}'; directory = OUTPUT / runid; directory.mkdir()
                    selection, parameters = None, 0
                    if baseline == 'B00':
                        prior = np.bincount(train_gold, minlength=classes).astype(float)
                        scores = np.repeat((prior / prior.sum())[None], len(test), 0)
                    elif baseline == 'B01':
                        centroids = np.stack([np.stack([s['pooled'] for s in train if s['label'] == c]).mean(0) for c in range(classes)])
                        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
                        scores = np.stack([s['pooled'] for s in test]) @ centroids.T
                    else:
                        factory = lambda: (LinearProbe(768, classes) if baseline == 'B02' else TemporalProbe(768, classes, p['temporal_width']))
                        reference = sum(v.numel() for v in LinearProbe(768, classes).parameters())
                        parameters = sum(v.numel() for v in factory().parameters())
                        if abs(parameters - reference) / reference > p['capacity_match_relative_tolerance']:
                            raise ValueError('Parameter budget mismatch')
                        best, best_state, trials = -1., None, []
                        for lr in p['learning_rates']:
                            torch.manual_seed(seed); rng = np.random.default_rng(seed)
                            model = factory(); optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
                            trial_best = -1.
                            for epoch in range(p['epochs']):
                                model.train(); order = rng.permutation(len(train)).tolist()
                                for start in range(0, len(order), p['batch_size']):
                                    x, mask, y = collate(train, order[start:start + p['batch_size']])
                                    optimizer.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x, mask), y)
                                    if not torch.isfinite(loss):
                                        raise ValueError('Non-finite loss')
                                    loss.backward(); optimizer.step()
                                score = classification_metrics(val_gold, evaluate(model, val, p['batch_size']).argmax(1), classes)['macro_recall']
                                trial_best = max(trial_best, score)
                                if score > best:
                                    best, best_state = score, copy.deepcopy(model.state_dict())
                                    selection = {'learning_rate': lr, 'epoch': epoch + 1, 'validation_macro_recall': score}
                            trials.append({'learning_rate': lr, 'best_validation_macro_recall': trial_best})
                            print(json.dumps({'run': runid, 'trial': trials[-1]}), flush=True)
                        selection.update(trials=trials, config_sha256=sha256(CONFIG), test_used_for_selection=False)
                        write_json(directory / 'selection.json', selection)
                        torch.save({'state_dict': best_state, 'selection': selection}, directory / 'head.pt')
                        model = factory(); model.load_state_dict(best_state, strict=True)
                        scores = evaluate(model, test, p['batch_size'])
                    pred = scores.argmax(1)
                    metric = classification_metrics(gold, pred, classes)
                    metric.update(trainable_parameters=parameters, split_counts=counts,
                                  score_semantics='cosine_similarity' if baseline == 'B01' else 'class_probability')
                    write_json(directory / 'metrics.json', metric)
                    prediction_rows = [{'sample_id': s['row']['sample_id'], 'split': 'test', 'gold': int(g),
                        'prediction': int(v), 'correct': int(g == v), 'recording_group': s['row']['recording_group'],
                        'signer_id': s['row']['signer_id']} for s, g, v in zip(test, gold, pred)]
                    with (directory / 'predictions.csv').open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
                    tables.append((pred == gold).astype(float)); records.append({'seed': seed, 'metrics': metric, 'selection': selection})
                    print(json.dumps({'run': runid, 'metrics': metric}), flush=True)
                key = backbone + '_' + baseline; correct[key] = np.stack(tables)
                groups = [s['row']['recording_group'] for s in test]
                results[key] = {'runs': records, 'bootstrap': grouped_bootstrap(gold, groups, correct[key])}
        primary = grouped_bootstrap(gold, groups, correct['signrep_B03'], reference=correct['signrep_B02'])
        secondary = {name: grouped_bootstrap(gold, groups, correct[name + '_B02'], reference=correct['signrep_B02'])
                     for name in ['shubert_last', 'shubert_average']}
        summary = {'status': 'PASS_EXPLORATORY_EQUAL_GRID_CONTROLS', 'config_sha256': sha256(CONFIG),
                   'code_sha256': code_identity(), 'coverage_sha256': sha256(OUTPUT / 'coverage.json'),
                   'elapsed_seconds': time.perf_counter() - started, 'fitting_seconds': time.perf_counter() - fit_started,
                   'results': results, 'primary': primary, 'secondary': secondary, 'limits': p['limits']}
        write_json(OUTPUT / 'summary.json', summary); state.update(status=summary['status'], elapsed_seconds=summary['elapsed_seconds'])
        write_json(OUTPUT / 'status.json', state)
    except Exception as error:
        state.update(status='FAIL', failure_reason=type(error).__name__ + ': ' + str(error), elapsed_seconds=time.perf_counter() - started)
        write_json(OUTPUT / 'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--wait-for-native', action='store_true')
    args = parser.parse_args()
    lock() if args.lock_only else run(args.wait_for_native)
