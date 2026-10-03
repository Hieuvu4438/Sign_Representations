"""Run locked B00-B03 lexical controls; tune only on validation."""
import argparse
import copy
import csv
import hashlib
import io
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from _common import ROOT
from signrepr.io import atomic_text, read_jsonl, sha256, write_json
from signrepr.probes import LinearProbe, TemporalProbe, classification_metrics, collate, evaluate, normalized_mean


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--baseline', choices=['B00', 'B01', 'B02', 'B03'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.config.read_text())
    lock = json.loads((ROOT / 'state/protocol_common_v1.lock.json').read_text())
    if sha256(args.config) != lock['sha256'] or not protocol['locked']:
        raise ValueError('Protocol lock mismatch')
    if args.seed not in protocol['seeds']:
        raise ValueError('Seed outside registered protocol')
    manifest = ROOT / protocol['manifest']
    if sha256(manifest) != protocol['manifest_sha256']:
        raise ValueError('Manifest changed after lock')
    mapping_path = ROOT / protocol['label_mapping']
    if sha256(mapping_path) != protocol['label_mapping_sha256']:
        raise ValueError('Train label mapping changed after lock')
    run_id = protocol['protocol_version'] + '_' + args.baseline + '_signrep_seed' + str(args.seed)
    output = ROOT / 'runs' / run_id
    if args.dry_run:
        print(json.dumps({'run_id': run_id, 'baseline': args.baseline, 'manifest_sha256': protocol['manifest_sha256'], 'test_tuning_allowed': False}))
        return
    features = ROOT / protocol['feature_directory']
    extraction_report = json.loads((features / 'extraction_report.json').read_text())
    if extraction_report['status'] != 'PASS' or extraction_report['manifest_sha256'] != protocol['manifest_sha256']:
        raise ValueError('Feature extraction incomplete or failed; cannot evaluate')
    if extraction_report['config_sha256'] != protocol['model_config_sha256']:
        raise ValueError('Feature configuration differs from locked protocol')
    if output.exists():
        raise ValueError('Run directory exists; inspect its status instead of overwriting results')
    output.mkdir(parents=True)
    begun = time.perf_counter()
    write_json(output / 'status.json', {'run_id': run_id, 'status': 'RUNNING', 'pid': os.getpid(), 'command': os.sys.argv})
    try:
        mapping = json.loads(mapping_path.read_text())['mapping']
        index = {row['sample_id']: row for _, row in read_jsonl(features / 'index.jsonl')}
        data = {split: [] for split in ['train', 'val', 'test']}
        exclusions, planned_counts = [], {split: 0 for split in data}
        for _, row in read_jsonl(manifest):
            split = row['official_split']
            planned_counts[split] += 1
            entry = index[row['sample_id']]
            if entry['status'] != 'SUCCESS':
                exclusions.append({'sample_id': row['sample_id'], 'reason': entry.get('error', 'EXTRACTION_FAILED')})
                continue
            shard_path = Path(entry['shard'])
            metadata = json.loads(shard_path.with_suffix('.json').read_text())
            if metadata['shard_sha256'] != sha256(shard_path):
                raise ValueError('Feature shard checksum mismatch')
            with np.load(shard_path, allow_pickle=False) as shard:
                valid = shard['valid_mask'] & shard['valid_frame_mask'].all(axis=1)
                if not valid.any():
                    exclusions.append({'sample_id': row['sample_id'], 'reason': 'SHORT_CLIP_REQUIRING_PADDING'})
                    continue
                tokens = shard['embeddings'][valid].astype(np.float32)
            data[split].append({'row': row, 'tokens': tokens, 'label': mapping[row['lexical_id']],
                                'pooled': normalized_mean(tokens, np.ones(len(tokens), dtype=bool))})
        if not all(data.values()):
            raise ValueError('Empty train/validation/test after exclusions')
        classes = len(mapping)
        gold = np.asarray([sample['label'] for sample in data['test']])
        train_gold = np.asarray([sample['label'] for sample in data['train']])
        train_pooled = np.stack([sample['pooled'] for sample in data['train']])
        test_pooled = np.stack([sample['pooled'] for sample in data['test']])
        trials, trainable, selected = [], 0, None
        if args.baseline == 'B00':
            prior = np.bincount(train_gold, minlength=classes).astype(float)
            scores = np.repeat((prior / prior.sum())[None], len(gold), axis=0)
        elif args.baseline == 'B01':
            if any(not (train_gold == label).any() for label in range(classes)):
                raise ValueError('Class with no surviving train samples')
            centroids = np.stack([train_pooled[train_gold == label].mean(axis=0) for label in range(classes)])
            centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
            scores = test_pooled @ centroids.T
        else:
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device(yaml.safe_load((ROOT / protocol['model_config']).read_text())['device'])
            best_score, model_state = -1., None
            hidden_grid = [None] if args.baseline == 'B02' else protocol['tuning']['capacities']
            for lr, hidden in itertools.product(protocol['tuning']['learning_rates'], hidden_grid):
                torch.manual_seed(args.seed)
                generator = np.random.default_rng(args.seed)
                model = (LinearProbe(768, classes) if hidden is None else TemporalProbe(768, classes, hidden)).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=protocol['tuning']['weight_decay'])
                trial_best, trial_epoch = -1., None
                for epoch in range(protocol['tuning']['epochs']):
                    model.train()
                    indices = generator.permutation(len(data['train'])).tolist()
                    for start in range(0, len(indices), 64):
                        tokens, mask, labels = collate(data['train'], indices[start:start + 64], device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(model(tokens, mask), labels)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite training loss')
                        loss.backward()
                        optimizer.step()
                    val_scores = evaluate(model, data['val'], device)
                    val_gold = [sample['label'] for sample in data['val']]
                    metric = classification_metrics(val_gold, val_scores.argmax(axis=1), classes)['macro_recall']
                    if metric > trial_best:
                        trial_best, trial_epoch = metric, epoch + 1
                    if metric > best_score:
                        best_score, model_state = metric, copy.deepcopy(model.state_dict())
                        selected = {'learning_rate': lr, 'hidden': hidden, 'epoch': epoch + 1, 'validation_macro_recall': metric}
                trials.append({'learning_rate': lr, 'hidden': hidden, 'best_epoch': trial_epoch, 'validation_macro_recall': trial_best})
                print(json.dumps({'run_id': run_id, 'trial': trials[-1]}), flush=True)
            model = (LinearProbe(768, classes) if selected['hidden'] is None else TemporalProbe(768, classes, selected['hidden'])).to(device)
            model.load_state_dict(model_state, strict=True)
            trainable = sum(parameter.numel() for parameter in model.parameters())
            temporary = output / 'head.pt.partial'
            torch.save({'state_dict': model_state, 'selection': selected}, temporary)
            os.replace(temporary, output / 'head.pt')
            write_json(output / 'validation_selection.json', {'selected': selected, 'trials': trials, 'protocol_sha256': sha256(args.config), 'test_scores_available_at_selection': False})
            scores = evaluate(model, data['test'], device)
        predictions = scores.argmax(axis=1)
        metrics = classification_metrics(gold, predictions, classes)
        top5 = np.argsort(-scores, axis=1, kind='stable')[:, :min(5, classes)]
        metrics.update(top5=float(np.mean([label in choices for label, choices in zip(gold, top5)])),
                       coverage=len(gold) / planned_counts['test'], trainable_parameters=trainable,
                       split_counts={split: len(samples) for split, samples in data.items()},
                       selected=selected, seed=args.seed, baseline=args.baseline, protocol_sha256=sha256(args.config),
                       feature_implementation_sha256=extraction_report['implementation_sha256'],
                       metric_implementation_sha256=sha256(ROOT / 'src/signrepr/probes.py'),
                       elapsed_seconds=time.perf_counter() - begun,
                       score_semantics='cosine_similarity_not_probability' if args.baseline == 'B01' else 'class_probability',
                       scope=protocol['scientific_claim_role'])
        prediction_rows = []
        for item, label, predicted, score in zip(data['test'], gold, predictions, scores):
            row = item['row']
            prediction_rows.append({'sample_id': row['sample_id'], 'recording_group': row['recording_group'],
                                    'signer_id': row['signer_id'], 'split': 'test', 'gold_status': row['annotation_quality'],
                                    'gold': int(label), 'prediction': int(predicted), 'correct': int(label == predicted),
                                    'score': float(score[predicted]), 'baseline': args.baseline, 'seed': args.seed})
        atomic_text(output / 'predictions.csv', csv_text(prediction_rows))
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'exclusions.json', exclusions)
        write_json(output / 'config.json', {'protocol': protocol, 'protocol_sha256': sha256(args.config), 'seed': args.seed, 'baseline': args.baseline,
                                          'capacity_note': 'B02 standard linear; B03 pointwise/depthwise temporal head plus channel residual whose intermediate width matches B02 parameter budget within 0.1% at the registered 200 classes. Widths 64/128 remain the registered capacities. Actual parameter counts reported; no encoder-method novelty claim.'})
        write_json(output / 'status.json', {'run_id': run_id, 'status': 'PASS', 'metrics_path': str(output / 'metrics.json')})
        print(json.dumps({'run_id': run_id, **{key: metrics[key] for key in ['top1', 'macro_recall', 'top5', 'coverage', 'trainable_parameters']}}), flush=True)
    except Exception as error:
        write_json(output / 'status.json', {'run_id': run_id, 'status': 'FAIL', 'failure_reason': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
