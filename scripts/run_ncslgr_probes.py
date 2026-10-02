"""Equal-budget frozen SignRep readouts for the locked internal diagnostic."""
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
from signrepr.statistics import grouped_bootstrap


def metrics(gold, prediction):
    return {'top1': float(np.mean(gold == prediction)),
            'macro_recall': float(np.mean([np.mean(prediction[gold == c] == c)
                                          for c in np.unique(gold)])),
            'per_class_recall': {str(c): float(np.mean(prediction[gold == c] == c))
                                 for c in np.unique(gold)}}


def normalized(x):
    norm = np.linalg.norm(x)
    if norm < 1e-10 or not np.isfinite(x).all():
        raise ValueError('Invalid pooled embedding')
    return x / norm


def main():
    folder = ROOT / 'data/external/ncslgr/diagnostic_v1'
    protocol_path = folder / 'protocol.lock.json'
    protocol = json.loads(protocol_path.read_text())
    manifest = folder / 'manifest.jsonl'
    if protocol['status'] != 'LOCKED_FOR_INTERNAL_DIAGNOSTIC_WITH_LIMITS' or sha256(manifest) != protocol['manifest_sha256']:
        raise ValueError('Diagnostic protocol is not locked or data hash differs')
    feature_root = ROOT / 'features/signrep_ncslgr_diagnostic_v1'
    extraction = json.loads((feature_root / 'extraction_report.json').read_text())
    if extraction['status'] != 'PASS' or extraction['manifest_sha256'] != sha256(manifest):
        raise ValueError('Full paired feature coverage not established')
    index = {r['sample_id']: r for _, r in read_jsonl(feature_root / 'index.jsonl')}
    paired, data = {}, {}
    for _, row in read_jsonl(manifest):
        entry = index[row['sample_id']]
        if entry['status'] != 'SUCCESS':
            raise ValueError('Failed feature row cannot silently disappear')
        shard_path = Path(entry['shard'])
        meta = json.loads(shard_path.with_suffix('.json').read_text())
        if sha256(shard_path) != meta['shard_sha256']:
            raise ValueError('Feature shard checksum mismatch')
        with np.load(shard_path, allow_pickle=False) as shard:
            start, stop = inclusive_frame_bounds(row['utterance_start_ms'], row['utterance_end_ms'], 30)
            keep = shard['valid_mask'] & (shard['center_sec'] >= start/30) & (shard['center_sec'] < stop/30)
            if not keep.any():
                raise ValueError('No valid utterance-supported feature windows')
            x = normalized(shard['embeddings'][keep].mean(axis=0)).astype(np.float32)
        sid = row['utterance_sample_id']
        paired.setdefault(sid, {})[row['view_role']] = x
        data[sid] = row
    ids = sorted(paired)
    if any(set(paired[sid]) != {'body', 'face'} for sid in ids):
        raise ValueError('Unpaired camera coverage')
    labels = {name: n for n, name in enumerate(protocol['classes'])}
    gold = np.asarray([labels[data[sid]['label']] for sid in ids])
    splits = np.asarray([data[sid]['split'] for sid in ids])
    train, val, test = [np.where(splits == s)[0] for s in ['train', 'val', 'test']]
    if any(len(np.unique(gold[i])) != len(labels) for i in [train, val, test]):
        raise ValueError('Every locked split must support each target class')
    features = {'body_mean_linear': np.stack([paired[sid]['body'] for sid in ids]),
                'face_mean_linear': np.stack([paired[sid]['face'] for sid in ids]),
                'paired_normalized_mean_linear': np.stack([
                    normalized(paired[sid]['body'] + paired[sid]['face']) for sid in ids])}
    output = ROOT / 'runs/ncslgr_diagnostic_v1'
    if (output / 'summary.json').exists():
        raise ValueError('Completed probe suite exists; refusing overwriting test results')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    start_time = time.perf_counter()
    write_json(output / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(),
                                      'protocol_sha256': sha256(protocol_path),
                                      'script_sha256': sha256(Path(__file__))})
    groups = [data[ids[i]]['recording_group'] for i in test]
    results, correct_tables = {}, {}
    prior = np.bincount(gold[train], minlength=len(labels)).argmax()
    prior_pred = np.full(len(test), prior)
    results['prior'] = {'metrics': metrics(gold[test], prior_pred),
                        'train_majority_label': protocol['classes'][prior]}
    correct_tables['prior'] = (prior_pred == gold[test])[None, :]
    for kind, all_x in features.items():
        # All distributional normalization is fitted on train only.
        mean, sd = all_x[train].mean(axis=0), all_x[train].std(axis=0)
        x = torch.from_numpy(((all_x - mean) / np.maximum(sd, 1e-5)).astype(np.float32))
        y = torch.from_numpy(gold).long()
        seed_results, seed_correct = [], []
        for seed in protocol['training_seeds']:
            candidates, best = [], None
            for lr in protocol['lr_grid']:
                torch.manual_seed(seed)
                model = torch.nn.Linear(x.shape[1], len(labels))
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=protocol['weight_decay'])
                for _ in range(protocol['epochs']):
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(x[train]), y[train])
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    val_pred = model(x[val]).argmax(1).numpy()
                val_metrics = metrics(gold[val], val_pred)
                candidates.append({'lr': lr, 'validation': val_metrics,
                                   'final_train_loss': float(loss.detach())})
                if best is None or val_metrics['macro_recall'] > best['validation']['macro_recall']:
                    best = {'model': model, 'lr': lr, 'validation': val_metrics}
            run = output / f'{kind}_seed{seed}'
            run.mkdir(exist_ok=True)
            # Persist selection before evaluating the held-out test.
            selection = {'seed': seed, 'selected_lr': best['lr'], 'validation': best['validation'],
                         'candidates': candidates, 'head_parameters': sum(p.numel() for p in best['model'].parameters()),
                         'protocol_sha256': sha256(protocol_path), 'test_used_for_selection': False}
            write_json(run / 'selection.json', selection)
            torch.save({'state_dict': best['model'].state_dict(), 'train_mean': mean,
                        'train_sd': sd, 'selection': selection}, run / 'head.pt')
            with torch.inference_mode():
                logits = best['model'](x[test])
                prediction = logits.argmax(1).numpy()
                probabilities = logits.softmax(1).numpy()
            measured = metrics(gold[test], prediction)
            measured.update(test_samples=len(test), coverage=1.0)
            write_json(run / 'metrics.json', measured)
            prediction_rows = []
            for j, idx in enumerate(test):
                sid = ids[idx]
                prediction_rows.append({'sample_id': sid, 'split': 'test', 'gold': int(gold[idx]),
                                        'prediction': int(prediction[j]), 'correct': int(prediction[j] == gold[idx]),
                                        'recording_group': data[sid]['recording_group'],
                                        'signer_group': data[sid]['signer'],
                                        'conservative_xml_group': data[sid]['conservative_xml_group'],
                                        **{f'probability_{name}': float(probabilities[j, k]) for name,k in labels.items()}})
            with (run / 'predictions.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
                writer.writeheader(); writer.writerows(prediction_rows)
            seed_results.append({'seed': seed, 'metrics': measured, 'selection': selection})
            seed_correct.append((prediction == gold[test]).astype(float))
            print(json.dumps({'kind':kind,'seed':seed,'metrics':measured}),flush=True)
        correct_tables[kind] = np.stack(seed_correct)
        stats = grouped_bootstrap(gold[test], groups, correct_tables[kind], resamples=2000, seed=42)
        stats['ordered_seeds'] = protocol['training_seeds']
        write_json(output / f'{kind}_bootstrap.json', stats)
        results[kind] = {'seeds': seed_results, 'bootstrap': stats}
    # Predefined paired comparison; no choosing a best view based on test.
    comparison = grouped_bootstrap(gold[test], groups,
        correct_tables['paired_normalized_mean_linear'],
        reference=correct_tables['body_mean_linear'], resamples=2000, seed=42)
    write_json(output / 'paired_vs_body_bootstrap.json', comparison)
    summary = {'status':'PASS_INTERNAL_DIAGNOSTIC','protocol_sha256':sha256(protocol_path),
               'script_sha256':sha256(Path(__file__)),'extraction_report_sha256':sha256(feature_root/'extraction_report.json'),
               'elapsed_seconds':time.perf_counter()-start_time,'train':len(train),'val':len(val),'test':len(test),
               'results':results,'paired_vs_body':comparison,'limitations':protocol['limits'],
               'contribution_status':'Frozen recorded-function diagnostic only; no method novelty or external semantic verification.',
               'pooling_policy':'Official inclusive-frame conversion; keep valid16-frame window centers in utterance support. No functional gold interval used as model input.',
               'training_loss':'Unweighted categorical cross entropy; train-only standardization; fixed100epochs,3LRchoices per readout/seed.'}
    write_json(output/'summary.json',summary)
    write_json(output/'status.json',{'status':'PASS_INTERNAL_DIAGNOSTIC','pid':os.getpid(),
                                    'elapsed_seconds':summary['elapsed_seconds']})


if __name__=='__main__':main()
