"""CPU-only equal-budget native/SignRep recorded-function controls.

Wait for complete native extraction. No extraction retry, encoder adaptation,
partial test evaluation, absence label or independently verified scope claim.
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
from run_ncslgr_intervals import evaluate
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.interval_readouts import IntervalReadout, interval_iou, cluster_interval_bootstrap
from signrepr.statistics import grouped_bootstrap

CONFIG = ROOT / 'configs/protocol_fair_ncslgr_v1.json'
OUTPUT = ROOT / 'runs/fair_ncslgr_v1'
NATIVE = ROOT / 'features/shubert_native_ncslgr_v1'
SIGNREP = ROOT / 'features/signrep_ncslgr_diagnostic_v1'
TARGETS = ROOT / 'data/external/ncslgr/intervals_v1/targets.jsonl'
MANIFEST = ROOT / 'data/manifests/ncslgr_native_body_v1.jsonl'


def identities():
    paths = [Path(__file__), ROOT / 'scripts/run_ncslgr_intervals.py',
             ROOT / 'src/signrepr/interval_readouts.py', ROOT / 'src/signrepr/statistics.py',
             ROOT / 'src/signrepr/io.py', TARGETS, MANIFEST,
             ROOT / 'configs/protocol_native_ncslgr_v1.json',
             ROOT / 'configs/protocol_ncslgr_intervals_v1.json',
             SIGNREP / 'extraction_report.json', SIGNREP / 'index.jsonl']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def lock():
    if CONFIG.exists() or OUTPUT.exists():
        raise ValueError('Preserve locked or attempted suite')
    previous = json.loads((ROOT / 'configs/protocol_ncslgr_intervals_v1.json').read_text())
    p = {k: previous[k] for k in ['classes', 'seeds', 'learning_rates', 'epochs',
         'weight_decay', 'endpoint_loss_weight', 'head_parameters', 'primary_metric', 'limits']}
    p.update(status='LOCKED_EXPLORATORY_EQUAL_BUDGET_NATIVE_FUNCTION_CONTROLS',
             created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             identity_sha256=identities(), native_cache_identity=json.loads((NATIVE / 'cohort.lock.json').read_text()),
             backbones=['signrep', 'shubert_last', 'shubert_average'], modes=['global', 'temporal'],
             primary_comparison='shubert_last_temporal_minus_shubert_last_global_interval_IoU',
             secondary_comparisons=['All six interval/readout results and paired native-minus-SignRep differences',
                                    'All six learned intervals minus train-only class-duration prior',
                                    'Classification macro recall under interval-selected heads'],
             selection=previous['validation_selection'], device='cpu', torch_threads=1,
             population='Same eligible IDs for all backbones; retain all positive classes. Exclude only zero valid utterance-supported tokens, with explicit per-ID reasons. STOP if any class has zero classification or single-interval support in any split.',
             missingness='Native frames require all four streams observed; SignRep requires nonpadded windows. Preserve original source node clocks; no compressed-time gaps. Add known utterance edges with nearest supported token for both modes.',
             normalization='Per-token L2 then train-only utterance-weighted mean/SD, separately per backbone. No gold core, phenotype or caption enters forward.',
             duration_prior='Train-only mean normalized single-positive interval per class, using selected head predicted class. Predeclared comparator; no test refit or interval labels in input.',
             prospective_scope='Locked before native grammar metrics; earlier SignRep classification, interval metrics and duration-prior sensitivity known. Exploratory, no independent confirmatory claim.',
             multiplicity='One primary native temporal/global interval comparison; remaining contrasts descriptive, no selecting best backbone/metric/seed.',
             limits=previous['limits'] + ['Native and SignRep differ in cue preprocessing, token rates and receptive fields; shared clips and head budget do not establish causal backbone comparison.',
                                        'Eligibility conditioning can change population; no claim on excluded clips.',
                                        'Frozen readouts are controls, not encoder adaptation or method novelty.'])
    write_json(CONFIG, p)
    print(json.dumps({'status': p['status'], 'config_sha256': sha256(CONFIG)}))


def native_ready():
    state = json.loads((NATIVE / 'status.json').read_text())
    if state['status'] == 'RUNNING':
        return False, state
    if state['status'] != 'PASS':
        raise ValueError('Native extraction ended without full PASS: ' + state['status'])
    return (NATIVE / 'extraction_report.json').exists(), state


def load_data(p):
    if identities() != p['identity_sha256']:
        raise ValueError('Locked source, code or feature identity changed')
    report = json.loads((NATIVE / 'extraction_report.json').read_text())
    if report['status'] != 'PASS' or report['manifest_sha256'] != sha256(MANIFEST):
        raise ValueError('Full terminal native coverage is required')
    targets = [r for _, r in read_jsonl(TARGETS)]
    manifest = [r for _, r in read_jsonl(MANIFEST)]
    expected = {r['sample_id'] for r in manifest}
    indexes = {}
    for name, folder in [('native', NATIVE), ('signrep', SIGNREP)]:
        rows = [r for _, r in read_jsonl(folder / 'index.jsonl')]
        indexes[name] = {r['sample_id']: r for r in rows}
        if len(rows) != len(indexes[name]):
            raise ValueError('Duplicate feature IDs')
    if len(manifest) != len(expected) or set(indexes['native']) != expected:
        raise ValueError('Native manifest and full feature IDs disagree')
    if expected != {r['sample_id'] + ':body' for r in targets}:
        raise ValueError('Target and source video identities disagree')
    result, sequences, clocks, excluded = [], {b: [] for b in p['backbones']}, {b: [] for b in p['backbones']}, []
    for row in targets:
        sid = row['sample_id'] + ':body'
        a, b = row['utterance_support_sec']
        loaded, reasons = {}, []
        for name, index in indexes.items():
            entry = index[sid]
            if entry['status'] != 'SUCCESS':
                raise ValueError('Failed extraction must not silently disappear')
            path = Path(entry['shard'])
            meta = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != meta['shard_sha256']:
                raise ValueError('Feature checksum mismatch')
            if name == 'native' and any(meta['cache_material'].get(k) != v for k, v in p['native_cache_identity'].items()):
                raise ValueError('Native cache identity differs')
            with np.load(path, allow_pickle=False) as data:
                valid = data['valid_mask'] & data['valid_frame_mask'].all(axis=1)
                if name == 'native' and not np.array_equal(valid, data['stream_observed_mask'].all(axis=1)):
                    raise ValueError('Native stream validity differs')
                clock = data['center_sec']
                if not np.isfinite(clock).all() or not (np.diff(clock) > 0).all():
                    raise ValueError('Invalid source chronology')
                keep = valid & (clock >= a) & (clock < b)
                if not keep.any():
                    reasons.append(name + '_ZERO_VALID_UTTERANCE_TOKENS')
                    continue
                keys = {'signrep': 'embeddings'} if name == 'signrep' else {
                    'shubert_last': 'embeddings', 'shubert_average': 'embeddings_layer_average'}
                for backbone, key in keys.items():
                    raw = data[key]
                    if raw.shape != (len(clock), 768) or not np.isfinite(raw).all():
                        raise ValueError('Invalid feature dimensions or values')
                    tokens = raw[keep].astype(np.float32)
                    tokens /= np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-10)
                    loaded[backbone] = (np.vstack([tokens[0], tokens, tokens[-1]]),
                                        np.concatenate([[a], clock[keep], [b]]))
        if reasons:
            excluded.append({'sample_id': row['sample_id'], 'split': row['split'], 'label': row['label'], 'reasons': reasons})
            continue
        result.append(row)
        for backbone, (tokens, nodes) in loaded.items():
            sequences[backbone].append(tokens)
            clocks[backbone].append(nodes)
    counts = {s: dict(Counter(r['label'] for r in result if r['split'] == s)) for s in ['train', 'val', 'test']}
    interval_counts = {s: dict(Counter(r['label'] for r in result if r['split'] == s and r['scope_supervised'])) for s in counts}
    write_json(OUTPUT / 'coverage.json', {'classification_class_counts': counts, 'interval_class_counts': interval_counts,
               'excluded': excluded, 'eligible_ids': [r['sample_id'] for r in result],
               'native_report_sha256': sha256(NATIVE / 'extraction_report.json'), 'native_index_sha256': sha256(NATIVE / 'index.jsonl')})
    if any(c.get(label, 0) == 0 for table in [counts, interval_counts] for c in table.values() for label in p['classes']):
        raise ValueError('Locked class lacks train/val/test support; no automatic class or population redesign')
    return result, sequences, clocks


def run(wait):
    if OUTPUT.exists():
        raise ValueError('Preserve previous attempted suite')
    p = json.loads(CONFIG.read_text())
    if identities() != p['identity_sha256'] or p['status'] != 'LOCKED_EXPLORATORY_EQUAL_BUDGET_NATIVE_FUNCTION_CONTROLS':
        raise ValueError('Protocol identity or status differs')
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    state = {'status': 'WAITING_FOR_NATIVE', 'pid': os.getpid(), 'device': 'cpu', 'config_sha256': sha256(CONFIG)}
    try:
        while True:
            ready, native = native_ready()
            if ready:
                break
            if not wait:
                raise ValueError('No partial native test inference')
            state.update(native_completed=native['success'], native_samples=native['samples'], elapsed_seconds=time.perf_counter()-started,
                         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            write_json(OUTPUT / 'status.json', state)
            time.sleep(30)
        state['status'] = 'RUNNING'; write_json(OUTPUT / 'status.json', state)
        torch.set_num_threads(p['torch_threads'])
        rows, sequences, clocks = load_data(p)
        fit_started = time.perf_counter()
        labels = {name: i for i, name in enumerate(p['classes'])}
        y = torch.tensor([labels[r['label']] for r in rows])
        split = {s: np.array([i for i, r in enumerate(rows) if r['split'] == s]) for s in ['train', 'val', 'test']}
        train, val, test = [split[s] for s in ['train', 'val', 'test']]
        supervised = np.array([r['scope_supervised'] for r in rows])
        supports = np.array([r['utterance_support_sec'] for r in rows])
        targets = np.array([r['recorded_functional_interval_sec'] or [0., 0.] for r in rows])
        fraction = torch.from_numpy(((targets-supports[:, :1])/np.diff(supports, axis=1)).astype(np.float32))
        scope_train = np.flatnonzero(supervised[train])
        prior = np.stack([fraction[train][supervised[train] & (y[train].numpy() == c)].mean(0).numpy() for c in range(3)])
        write_json(OUTPUT / 'train_duration_prior.json', {'normalized_intervals': prior.tolist(), 'source_split': 'train', 'test_used': False})
        results, interval_tables, prior_tables, class_tables = {}, {}, {}, {}
        for backbone in p['backbones']:
            raw, nodes = sequences[backbone], clocks[backbone]
            width = max(map(len, nodes))
            mean = np.stack([raw[i].mean(0) for i in train]).mean(0)
            sd = np.sqrt(np.maximum(np.stack([(raw[i]**2).mean(0) for i in train]).mean(0)-mean**2, 1e-10))
            x = torch.zeros(len(rows), width, 768)
            mask = torch.zeros(len(rows), width, dtype=torch.bool)
            positions = torch.zeros(len(rows), width)
            for i, tokens in enumerate(raw):
                x[i, :len(tokens)] = torch.from_numpy((tokens-mean)/sd)
                mask[i, :len(tokens)] = True
                positions[i, :len(tokens)] = torch.from_numpy(((nodes[i]-supports[i, 0])/np.diff(supports[i])[0]).astype(np.float32))
            for mode in p['modes']:
                kind = backbone + '_' + mode
                records, scores, prior_scores, correct = [], [], [], []
                for seed in p['seeds']:
                    selected, candidates = None, []
                    for lr in p['learning_rates']:
                        torch.manual_seed(seed)
                        model = IntervalReadout(768, 3, temporal=mode == 'temporal')
                        if sum(v.numel() for v in model.parameters()) != p['head_parameters']:
                            raise ValueError('Head capacity differs')
                        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
                        for _ in range(p['epochs']):
                            model.train(); optimizer.zero_grad()
                            logits, bounds, _ = model(x[train], mask[train], positions[train])
                            loss = torch.nn.functional.cross_entropy(logits, y[train]) + p['endpoint_loss_weight'] * torch.nn.functional.smooth_l1_loss(
                                bounds[scope_train, y[train][scope_train]], fraction[train][scope_train], beta=.1)
                            if not torch.isfinite(loss):
                                raise ValueError('Non-finite loss')
                            loss.backward(); optimizer.step()
                        measured = evaluate(model, x, mask, positions, val, y, supports, targets, supervised)[0]
                        candidates.append({'lr': lr, 'validation': measured, 'final_train_loss': float(loss.detach())})
                        if selected is None or measured[p['primary_metric']] > selected['validation'][p['primary_metric']]:
                            selected = {'model': model, 'lr': lr, 'validation': measured}
                    directory = OUTPUT / f'{kind}_seed{seed}'; directory.mkdir()
                    selection = {'seed': seed, 'selected_lr': selected['lr'], 'validation': selected['validation'],
                                 'candidates': candidates, 'test_used_for_selection': False, 'config_sha256': sha256(CONFIG)}
                    write_json(directory / 'selection.json', selection)
                    torch.save({'state_dict': selected['model'].state_dict(), 'train_mean': mean, 'train_sd': sd, 'selection': selection}, directory / 'head.pt')
                    m, prediction, intervals, values, distributions, probabilities = evaluate(
                        selected['model'], x, mask, positions, test, y, supports, targets, supervised)
                    prior_intervals = supports[test, :1] + prior[prediction]*np.diff(supports[test], axis=1)
                    class_correct = prediction == y[test].numpy()
                    prior_value = interval_iou(prior_intervals[supervised[test]], targets[test][supervised[test]])*class_correct[supervised[test]]
                    prediction_rows = []
                    for j, i in enumerate(test):
                        r = rows[i]
                        prediction_rows.append({'sample_id': r['sample_id'], 'split': 'test', 'gold': int(y[i]), 'prediction': int(prediction[j]),
                            'correct': int(class_correct[j]), 'scope_supervised': int(supervised[i]), 'recording_group': r['recording_group'],
                            'signer': r['signer'], 'conservative_xml_group': r['conservative_xml_group'],
                            'predicted_start_sec': float(intervals[j, 0]), 'predicted_end_sec': float(intervals[j, 1]),
                            'prior_start_sec': float(prior_intervals[j, 0]), 'prior_end_sec': float(prior_intervals[j, 1]),
                            'gold_start_sec': float(targets[i, 0]) if supervised[i] else None, 'gold_end_sec': float(targets[i, 1]) if supervised[i] else None,
                            **{f'probability_{name}': float(probabilities[j, c]) for name, c in labels.items()}})
                        if distributions is not None:
                            np.savez_compressed(directory / f'{i}_boundary.npz', source_node_sec=nodes[i], sample_id=r['sample_id'],
                                boundary_probabilities=distributions[j, :len(nodes[i])].numpy().reshape(-1, 3, 2))
                    with (directory / 'predictions.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
                    write_json(directory / 'metrics.json', m)
                    records.append({'seed': seed, 'selection': selection, 'metrics': m})
                    scores.append(values); prior_scores.append(prior_value); correct.append(class_correct.astype(float))
                    print(json.dumps({'kind': kind, 'seed': seed, 'metrics': m}), flush=True)
                interval_tables[kind], prior_tables[kind], class_tables[kind] = map(np.stack, [scores, prior_scores, correct])
                igold = y[test][supervised[test]].numpy()
                igroups = [rows[i]['recording_group'] for i in test if supervised[i]]
                groups = [rows[i]['recording_group'] for i in test]
                results[kind] = {'seeds': records, 'interval_bootstrap': cluster_interval_bootstrap(igold, igroups, interval_tables[kind]),
                    'classification_bootstrap': grouped_bootstrap(y[test].numpy(), groups, class_tables[kind]),
                    'train_duration_prior_bootstrap': cluster_interval_bootstrap(igold, igroups, prior_tables[kind]),
                    'learned_minus_prior': cluster_interval_bootstrap(igold, igroups, interval_tables[kind], reference=prior_tables[kind])}
        primary = cluster_interval_bootstrap(igold, igroups, interval_tables['shubert_last_temporal'], reference=interval_tables['shubert_last_global'])
        secondary = {b+'_'+m+'_minus_signrep_'+m: cluster_interval_bootstrap(igold, igroups, interval_tables[b+'_'+m], reference=interval_tables['signrep_'+m])
                     for b in ['shubert_last', 'shubert_average'] for m in p['modes']}
        summary = {'status': 'PASS_EXPLORATORY_EQUAL_BUDGET_NATIVE_FUNCTION_CONTROLS', 'config_sha256': sha256(CONFIG),
                   'identity_sha256': identities(), 'coverage_sha256': sha256(OUTPUT / 'coverage.json'),
                   'elapsed_seconds': time.perf_counter()-started, 'fitting_seconds': time.perf_counter()-fit_started,
                   'head_parameters': p['head_parameters'], 'results': results, 'primary': primary, 'secondary': secondary,
                   'limits': p['limits'], 'contribution_status': 'Frozen controls, no encoder adaptation or established method novelty.'}
        write_json(OUTPUT / 'summary.json', summary)
        state.update(status=summary['status'], elapsed_seconds=summary['elapsed_seconds']); write_json(OUTPUT / 'status.json', state)
    except Exception as error:
        state.update(status='FAIL', failure_reason=type(error).__name__+': '+str(error), elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT / 'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--wait-for-native', action='store_true')
    args = parser.parse_args()
    lock() if args.lock_only else run(args.wait_for_native)
