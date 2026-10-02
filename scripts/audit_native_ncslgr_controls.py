"""Audit full native source clocks/cache and both completed frozen control suites.

Reproduce predictions from saved heads and immutable inputs; do not fit, tune,
change the eligible population, or add a post-hoc primary comparison.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.interval_readouts import IntervalReadout, interval_iou, macro_values
from signrepr.native_cache import verified_preprocessing
from signrepr.timestamps import source_frame_intervals


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b, message, atol=1e-6):
    require(np.allclose(a, b, atol=atol, rtol=1e-6), message)


def read(path):
    return json.loads(path.read_text())


def main():
    output = ROOT / 'reports/native_ncslgr_controls_audit.json'
    require(not output.exists(), 'Preserve previous audit')
    torch.set_num_threads(1)
    native = ROOT / 'features/shubert_native_ncslgr_v1'
    prep = ROOT / 'features/shubert_native_ncslgr_v1_preprocessing'
    protocol_path = ROOT / 'configs/protocol_native_ncslgr_v1.json'
    p = read(protocol_path)
    report = read(native / 'extraction_report.json')
    require(report['status'] == 'PASS' and read(native / 'status.json')['status'] == 'PASS', 'Native full terminal PASS required')
    require(report['protocol_sha256'] == sha256(protocol_path), 'Native protocol hash differs')
    source_rows = [r for _, r in read_jsonl(ROOT / p['manifest'])]
    require(sha256(ROOT / p['manifest']) == p['manifest_sha256'], 'Native manifest differs')
    index = {r['sample_id']: r for _, r in read_jsonl(native / 'index.jsonl')}
    require(len(source_rows) == len(index) == report['samples'] == report['success'] == 222, 'Full native coverage differs')
    require(set(index) == {r['sample_id'] for r in source_rows}, 'Native source IDs differ')
    hashes, cache, missing = {}, {}, []
    identity = read(native / 'cohort.lock.json')
    prep_identity = read(prep / 'cohort.lock.json')
    for row in source_rows:
        entry = index[row['sample_id']]
        folder = prep / hashlib.sha256(row['sample_id'].encode()).hexdigest()[:20]
        audit_path = folder / 'audit.json'
        prepared = verified_preprocessing(audit_path, row, prep_identity)
        material = entry['cache_material']
        require(all(material.get(k) == v for k, v in identity.items()), 'Native feature identity differs')
        require(material['source_sha256'] == prepared['source_sha256'] and material['preprocessing_audit_sha256'] == sha256(audit_path), 'Native source/preparation linkage differs')
        path = Path(entry['shard'])
        require(entry['status'] == 'SUCCESS' and sha256(path) == entry['shard_sha256'], 'Native shard checksum differs')
        require(read(path.with_suffix('.json')) == entry, 'Native index and metadata differ')
        with np.load(path, allow_pickle=False) as d:
            layers, observed = d['layer_embeddings'], d['stream_observed_mask']
            require(layers.shape == (12, prepared['frames'], 768) and np.isfinite(layers).all(), 'Native layer shape/values differ')
            require(observed.dtype == bool and observed.shape == (prepared['frames'], 4), 'Stream observation shape differs')
            require(np.array_equal(d['valid_mask'], observed.all(1)) and np.array_equal(d['valid_frame_mask'][:, 0], observed.all(1)), 'Native validity differs')
            require(int(observed.all(1).sum()) == entry['fully_observed_frames'], 'Native indexed observed count differs')
            near(d['embeddings'], layers[-1], 'Last FFN differs', atol=0)
            near(d['embeddings_layer_average'], layers.mean(0), 'Fixed layer mean differs', atol=0)
            clock, _ = source_frame_intervals(prepared['source_path'], prepared['frames'], prepared['fps'])
            near(d['window_start_sec'], clock[:, 0], 'Source frame start differs', atol=1e-10)
            near(d['window_end_sec'], clock[:, 1], 'Source frame end differs', atol=1e-10)
            near(d['center_sec'], clock.mean(1), 'Source frame center differs', atol=1e-10)
            cache[row['sample_id']] = {k: d[k].copy() for k in ['embeddings', 'embeddings_layer_average', 'center_sec', 'valid_mask']}
            missing.append({'sample_id': row['sample_id'], 'frames': len(observed), 'all_four_observed': int(observed.all(1).sum()), 'per_stream_observed': observed.sum(0).tolist()})
        hashes[str(path.relative_to(ROOT))] = entry['shard_sha256']
        hashes[str(audit_path.relative_to(ROOT))] = sha256(audit_path)

    signrep_index = {r['sample_id']: r for _, r in read_jsonl(ROOT / 'features/signrep_ncslgr_diagnostic_v1/index.jsonl')}
    signrep = {}
    for row in source_rows:
        sid = row['sample_id']; path = Path(signrep_index[sid]['shard'])
        require(sha256(path) == read(path.with_suffix('.json'))['shard_sha256'], 'SignRep shard checksum differs')
        with np.load(path, allow_pickle=False) as d:
            signrep[sid] = {k: d[k].copy() for k in ['embeddings', 'center_sec', 'valid_mask', 'valid_frame_mask']}
    targets = {r['sample_id']: r for _, r in read_jsonl(ROOT / 'data/external/ncslgr/intervals_v1/targets.jsonl')}
    suites = {}
    for run_id, interval in [('fair_ncslgr_v1', True), ('ncslgr_author_pool_v1', False)]:
        directory = ROOT / 'runs' / run_id
        config = ROOT / 'configs' / ('protocol_' + run_id + '.json')
        cp, summary, coverage = read(config), read(directory / 'summary.json'), read(directory / 'coverage.json')
        require(summary['status'].startswith('PASS') and summary['config_sha256'] == sha256(config), 'Control summary protocol differs')
        require(summary['coverage_sha256'] == sha256(directory / 'coverage.json'), 'Control coverage hash differs')
        for name, digest in summary['identity_sha256'].items():
            require(sha256(ROOT / name) == digest, 'Pinned control input/code differs: ' + name)
        ids = sorted(coverage['eligible_ids']); require(len(ids) == len(set(ids)), 'Duplicate eligible IDs')
        expected = set(targets) if not interval else set(targets) - {r['sample_id'] for r in coverage['excluded']}
        require(set(ids) == expected, 'Locked population coverage differs')
        rows = [targets[s] for s in ids]
        train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
        val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
        test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
        class_ids = np.array([cp['classes'].index(r['label']) for r in rows])
        supported = np.array([r['scope_supervised'] for r in rows])[test]
        node_map, pooled = {}, {}
        for backbone in cp['backbones']:
            raw, nodes = [], []
            for r in rows:
                sid = r['sample_id'] + ':body'; start, stop = r['utterance_support_sec']
                data = signrep[sid] if backbone == 'signrep' else cache[sid]
                keep = (data['center_sec'] >= start) & (data['center_sec'] < stop)
                if interval or backbone == 'signrep':
                    keep &= data['valid_mask']
                if backbone == 'signrep':
                    keep &= data['valid_frame_mask'].all(1)
                require(keep.any(), 'Saved eligible ID has no supported tokens')
                key = 'embeddings_layer_average' if backbone == 'shubert_average' else 'embeddings'
                tokens = data[key][keep].astype(np.float32)
                if interval:
                    tokens /= np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-10)
                    raw.append(np.vstack([tokens[0], tokens, tokens[-1]]))
                    nodes.append(np.concatenate([[start], data['center_sec'][keep], [stop]]))
                else:
                    mean = tokens.mean(0); raw.append(mean / np.linalg.norm(mean))
            if interval:
                mu = np.stack([raw[i].mean(0) for i in train]).mean(0)
                sd = np.sqrt(np.maximum(np.stack([(raw[i] ** 2).mean(0) for i in train]).mean(0) - mu ** 2, 1e-10))
                width = max(map(len, raw)); x = torch.zeros(len(rows), width, 768)
                mask = torch.zeros(len(rows), width, dtype=torch.bool); positions = torch.zeros(len(rows), width)
                for i, tokens in enumerate(raw):
                    x[i, :len(tokens)] = torch.from_numpy((tokens - mu) / sd); mask[i, :len(tokens)] = True
                    start, stop = rows[i]['utterance_support_sec']; positions[i, :len(tokens)] = torch.from_numpy(((nodes[i] - start) / (stop - start)).astype(np.float32))
                node_map[backbone] = nodes
                pooled[backbone] = (mu, sd, x, mask, positions)
            else:
                raw = np.stack(raw); mu, sd = raw[train].mean(0), raw[train].std(0)
                pooled[backbone] = (mu, sd, torch.from_numpy(((raw - mu) / np.maximum(sd, 1e-5)).astype(np.float32)))
        head_count, boundary_count = 0, 0
        for kind, results in summary['results'].items():
            backbone = kind.rsplit('_', 1)[0] if interval else kind
            per_seed, per_seed_prior, class_scores = [], [], []
            for record in results['seeds']:
                head_count += 1; run = directory / f"{kind}_seed{record['seed']}"
                selected = read(run / 'selection.json')
                require(not selected['test_used_for_selection'] and selected == record['selection'], 'Saved validation selection differs')
                metric = cp['primary_metric'] if interval else 'macro_recall'
                best = max(selected['candidates'], key=lambda c: c['validation'][metric])
                require(best['lr'] == selected['selected_lr'] and best['validation'] == selected['validation'], 'Final validation winner differs')
                head = torch.load(run / 'head.pt', map_location='cpu', weights_only=False)
                require(head['selection'] == selected, 'Head selection provenance differs')
                mu, sd, x, *extra = pooled[backbone]
                near(head['train_mean'], mu, 'Train mean differs'); near(head['train_sd'], sd, 'Train SD differs')
                model = IntervalReadout(temporal=kind.endswith('_temporal')) if interval else torch.nn.Linear(768, 3)
                model.load_state_dict(head['state_dict'], strict=True); model.eval()
                require(sum(v.numel() for v in model.parameters()) == cp['head_parameters'], 'Head parameters differ')
                with torch.inference_mode():
                    if interval:
                        logits, fractions, _ = model(x[test], extra[0][test], extra[1][test])
                    else:
                        logits = model(x[test])
                    probabilities = logits.softmax(1).numpy(); predicted = probabilities.argmax(1)
                with (run / 'predictions.csv').open() as f:
                    predictions = list(csv.DictReader(f))
                require(len(predictions) == len(test) and [r['sample_id'] for r in predictions] == [ids[i] for i in test], 'Exact ordered prediction IDs differ')
                for j, i in enumerate(test):
                    r, reference = predictions[j], rows[i]
                    require(r['split'] == 'test' and int(r['gold']) == class_ids[i] and int(r['prediction']) == predicted[j], 'Reconstructed prediction differs')
                    require(int(r['correct']) == int(predicted[j] == class_ids[i]), 'Saved correctness differs')
                    require(all(r[k] == reference[k] for k in ['signer', 'recording_group', 'conservative_xml_group']), 'Source grouping differs')
                    near([float(r['probability_' + c]) for c in cp['classes']], probabilities[j], 'Reconstructed probabilities differ')
                correct = predicted == class_ids[test]; class_scores.append(macro_values(class_ids[test], correct)[0])
                if interval:
                    supports = np.array([rows[i]['utterance_support_sec'] for i in test])
                    predicted_intervals = supports[:, :1] + fractions.numpy()[np.arange(len(test)), predicted] * np.diff(supports, axis=1)
                    saved = np.array([[float(r['predicted_start_sec']), float(r['predicted_end_sec'])] for r in predictions])
                    near(saved, predicted_intervals, 'Reconstructed source intervals differ', atol=2e-6)
                    gold_intervals = np.array([rows[i]['recorded_functional_interval_sec'] or [0, 0] for i in test])
                    for j, i in enumerate(test):
                        r = predictions[j]
                        require(int(r['scope_supervised']) == int(supported[j]), 'Recorded single-interval supervision differs')
                        if supported[j]:
                            near([float(r['gold_start_sec']), float(r['gold_end_sec'])], gold_intervals[j], 'Recorded gold bounds differ')
                        else:
                            require(not r['gold_start_sec'] and not r['gold_end_sec'], 'Multiple events falsely collapsed')
                    score = interval_iou(saved[supported], gold_intervals[supported]) * correct[supported]
                    per_seed.append(macro_values(class_ids[test][supported], score)[0])
                    prior = read(directory / 'train_duration_prior.json')
                    require(prior['source_split'] == 'train' and not prior['test_used'], 'Duration prior provenance differs')
                    means = np.stack([np.mean([(np.array(r['recorded_functional_interval_sec']) - r['utterance_support_sec'][0]) / np.diff(r['utterance_support_sec'])[0]
                                     for r in rows if r['split'] == 'train' and r['scope_supervised'] and r['label'] == label], axis=0) for label in cp['classes']])
                    near(prior['normalized_intervals'], means, 'Train-only duration prior differs')
                    fixed_prior = supports[:, :1] + means[predicted] * np.diff(supports, axis=1)
                    saved_prior = np.array([[float(r['prior_start_sec']), float(r['prior_end_sec'])] for r in predictions])
                    near(saved_prior, fixed_prior, 'Saved duration prior intervals differ')
                    per_seed_prior.append(macro_values(class_ids[test][supported], interval_iou(saved_prior[supported], gold_intervals[supported]) * correct[supported])[0])
                    if kind.endswith('_temporal'):
                        files = list(run.glob('*_boundary.npz')); require(len(files) == len(test), 'Boundary file coverage differs')
                        seen = set(); table = {r['sample_id']: r for r in predictions}
                        for file in files:
                            with np.load(file, allow_pickle=False) as d:
                                sid = str(d['sample_id']); require(sid in table and sid not in seen, 'Boundary identity differs'); seen.add(sid)
                                nodes = d['source_node_sec']; probability = d['boundary_probabilities']; i = ids.index(sid)
                                near(nodes, node_map[backbone][i], 'Sparse source clock was compressed', atol=1e-10)
                                require(probability.shape == (len(nodes), 3, 2) and np.isfinite(probability).all() and (probability >= 0).all(), 'Boundary probabilities invalid')
                                near(probability.sum(0), np.ones((3, 2)), 'Boundary normalization differs', atol=2e-6)
                                r = table[sid]; bound = (probability[:, int(r['prediction'])] * nodes[:, None]).sum(0)
                                near(bound, [float(r['predicted_start_sec']), float(r['predicted_end_sec'])], 'Boundary expected source clock differs', atol=2e-6)
                            hashes[str(file.relative_to(ROOT))] = sha256(file); boundary_count += 1
                else:
                    per_seed.append(class_scores[-1])
                hashes[str((run / 'head.pt').relative_to(ROOT))] = sha256(run / 'head.pt')
                hashes[str((run / 'predictions.csv').relative_to(ROOT))] = sha256(run / 'predictions.csv')
            stats = results['interval_bootstrap'] if interval else results['bootstrap']['metrics']['macro_recall']
            near(stats['per_seed'], per_seed, 'Bootstrap point inputs differ', atol=1e-10)
            near(stats['point'], np.mean(per_seed), 'Bootstrap point differs', atol=1e-10)
            if interval:
                near(results['train_duration_prior_bootstrap']['per_seed'], per_seed_prior, 'Prior bootstrap inputs differ', atol=1e-6)
                near(results['classification_bootstrap']['metrics']['macro_recall']['per_seed'], class_scores, 'Class bootstrap inputs differ', atol=1e-10)
        suites[run_id] = {'heads_reconstructed': head_count, 'boundary_files_verified': boundary_count,
                         'eligible_split_counts': {s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test']},
                         'summary_sha256': sha256(directory / 'summary.json'), 'coverage_sha256': sha256(directory / 'coverage.json')}
    result = {'status': 'PASS_NATIVE_CACHE_CLOCK_AND_SAVED_HEAD_INTEGRITY_WITH_LIMITS', 'native_clips': len(index),
              'native_source_frames': sum(r['frames'] for r in missing), 'zero_all_four_observed_clips': sum(r['all_four_observed'] == 0 for r in missing),
              'observation_counts': missing, 'suites': suites, 'artifact_sha256': hashes, 'script_sha256': sha256(Path(__file__)),
              'limits': ['Machine integrity audit, zero expert linguistic reviews.', 'Reconstructs outputs and bootstrap inputs/points; does not certify CI population coverage.',
                         'Independent source clock/checksum checks do not validate pose/hand detector correctness or full grammatical scope.',
                         'One test signer, known exploratory test and source-context differences; no method novelty.']}
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['artifact_sha256', 'observation_counts']}))


if __name__ == '__main__':
    main()
