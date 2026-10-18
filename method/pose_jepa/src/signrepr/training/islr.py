"""Frozen linear probe and supervised fine-tune with train-only vocabularies."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from .pretrain import seed_everything
from .checkpoints import atomic_save, rng_state
from .provenance import provenance
from ..models.pose_encoder import PoseEncoder, pool_clip
from ..data.samplers import gloss_key
from ..data.pose_dataset import PoseDataset
from ..data.collate import collate_pose, to_device
from ..data.schema import sha256_file
from ..evaluation.extract import extract_full_sequence
from ..evaluation.recognition import recognition_metrics


def load_selected_encoder(checkpoint, device='cpu', selection=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    if state.get('task') == 'islr' or (state.get('task') == 'cslr' and state.get('encoder') is not None):
        selected = state['encoder']
    elif state.get('task') in ('cslr', 'slt'):
        source = state['encoder_checkpoint']
        if sha256_file(source) != state['cache_provenance']['encoder_checkpoint_sha256']:
            raise ValueError('Frozen sequence backbone checkpoint checksum mismatch')
        encoder, _ = load_selected_encoder(source, device, selection)
        return encoder, state
    else:
        stage = state.get('metadata', {}).get('stage', 'ssl')
        selection = selection or ('online' if stage == 'metric' else cfg['ema']['export_encoder'])
        selected = state['teacher'] if selection == 'ema' else {k.removeprefix('encoder.'): v for k, v in state['system'].items() if k.startswith('encoder.')}
    encoder = PoseEncoder(cfg['model']).to(device)
    encoder.load_state_dict(selected, strict=True)
    return encoder, state


def check_splits(train, validation):
    if not train.rows or not validation.rows:
        raise ValueError('Training and validation splits must be nonempty')
    if any(r['split'] not in ('train', 'pretrain') for r in train.rows) or any(r['split'] not in ('val', 'validation') for r in validation.rows):
        raise ValueError('Training selection requires train and held-out validation manifests')
    for key in ('sample_id', 'source_video_id', 'pose_sha256'):
        if {r[key] for r in train.rows} & {r[key] for r in validation.rows}:
            raise ValueError(f'Train/validation contamination: {key}')


@torch.inference_mode()
def frozen_features(encoder, dataset, config):
    encoder.eval()
    result = []
    ev = config['evaluation']
    for i in range(len(dataset)):
        sequence = extract_full_sequence(encoder, dataset[i], ev['window_frames'], ev['window_stride_frames'])
        result.append(sequence)
    return result


def set_epoch_lr(optimizer, epoch, epochs, warmup_epochs, base_rates):
    if epoch < warmup_epochs:
        factor = (epoch+1)/max(1, warmup_epochs)
    else:
        factor = (1+math.cos(math.pi*(epoch-warmup_epochs)/max(1, epochs-warmup_epochs)))/2
    for group, rate in zip(optimizer.param_groups, base_rates):
        group['lr'] = rate*factor


class ISLRModel(nn.Module):
    def __init__(self, encoder, classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config['sequence_dim'], classes)

    def forward(self, batch):
        raw, valid_parts = pool_clip(self.encoder.encode_full(batch))
        return self.classifier(raw), valid_parts.any(-1)


@torch.inference_mode()
def evaluate_islr_model(model, dataset, vocabulary, config):
    model.eval()
    sequences = frozen_features(model.encoder, dataset, config)
    device = next(model.parameters()).device
    vectors = torch.stack([s['raw_embedding'] for s in sequences]).to(device)
    logits = model.classifier(vectors).cpu().numpy()
    mapping = {tuple(key): i for i, key in enumerate(vocabulary)}
    unknown = sorted({gloss_key(row) for row in dataset.rows}-set(mapping))
    mapping.update({key: len(vocabulary)+i for i, key in enumerate(unknown)})
    labels = [mapping[gloss_key(row)] for row in dataset.rows]
    valid = [s['has_observation'] for s in sequences]
    return recognition_metrics(logits, labels, valid), [{'sample_id': row['sample_id'], 'valid': ok,
                                                       'label_index': label, 'prediction_index': int(score.argmax()) if ok else None}
                                                      for row, ok, label, score in zip(dataset.rows, valid, labels, logits)]


def train_islr(config, initial_checkpoint=None, device='cpu'):
    seed_everything(config['run']['seed'])
    cfg = config['downstream']
    if initial_checkpoint:
        encoder, initial = load_selected_encoder(initial_checkpoint, device)
        if encoder.config != config['model']:
            raise ValueError('ISLR architecture does not match source checkpoint')
    else:
        if cfg['frozen']:
            raise ValueError('Frozen probe requires a trained initialization checkpoint')
        encoder = PoseEncoder(config['model']).to(device)
    frozen = cfg['frozen']
    train = PoseDataset(config['data']['train_manifest'], config['data'], stage='islr', sampling='full' if frozen else 'isolated')
    validation = PoseDataset(config['data']['validation_manifest'], config['data'], stage='islr')
    check_splits(train, validation)
    metadata = provenance(config, config['data']['train_manifest'])
    metadata['stage'] = 'islr'
    vocabulary = sorted({gloss_key(row) for row in train.rows})
    mapping = {label: i for i, label in enumerate(vocabulary)}
    model = ISLRModel(encoder, len(vocabulary)).to(device)
    encoder.requires_grad_(not frozen)
    rates = [cfg['head_lr']] if frozen else [cfg['encoder_lr'], cfg['head_lr']]
    groups = [{'params': model.classifier.parameters()}] if frozen else [{'params': encoder.parameters()}, {'params': model.classifier.parameters()}]
    optimizer = torch.optim.AdamW(groups, lr=cfg['head_lr'], weight_decay=cfg['weight_decay'])
    cache = frozen_features(encoder, train, config) if frozen else None
    root = Path(config['run']['output_dir']); root.mkdir(parents=True, exist_ok=True)
    best, stale, logs = -float('inf'), 0, []
    for epoch in range(cfg['epochs']):
        model.train()
        if frozen: encoder.eval()
        set_epoch_lr(optimizer, epoch, cfg['epochs'], cfg['warmup_epochs'], rates)
        order = torch.randperm(len(train)).tolist()
        total, support, skipped = 0., 0, 0
        for start in range(0, len(order), cfg['batch_size']):
            ids = order[start:start+cfg['batch_size']]
            labels = torch.tensor([mapping[gloss_key(train.rows[i])] for i in ids], device=device)
            if frozen:
                raw = torch.stack([cache[i]['raw_embedding'] for i in ids]).to(device)
                valid = torch.tensor([cache[i]['has_observation'] for i in ids], device=device)
                logits = model.classifier(raw)
            else:
                logits, valid = model(to_device(collate_pose([train[i] for i in ids]), device))
            skipped += int((~valid).sum())
            if not valid.any(): continue
            loss = nn.functional.cross_entropy(logits[valid], labels[valid], label_smoothing=0. if frozen else cfg['label_smoothing'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite supervised gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*int(valid.sum()); support += int(valid.sum())
        if not support:
            raise ValueError('ISLR train split has no valid representations')
        metrics, _ = evaluate_islr_model(model, validation, vocabulary, config)
        score = metrics[cfg['selection_metric']]
        logs.append({'epoch': epoch+1, 'train_loss': total/support, 'skipped_train_samples': skipped, 'validation': metrics})
        state = {'task': 'islr', 'config': config, 'metadata': metadata, 'encoder': encoder.state_dict(), 'classifier': model.classifier.state_dict(),
                 'vocabulary': vocabulary, 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'rng': rng_state(),
                 'initial_checkpoint_sha256': sha256_file(initial_checkpoint) if initial_checkpoint else None,
                 'train_manifest_sha256': sha256_file(config['data']['train_manifest']),
                 'validation_manifest_sha256': sha256_file(config['data']['validation_manifest']), 'validation': metrics,
                 'supervision': 'gloss_supervised_frozen_linear_probe' if frozen else 'gloss_supervised_finetuning'}
        atomic_save(state, root/'last.pt')
        if score > best:
            best, stale = score, 0; atomic_save(state, root/'best_validation.pt')
        else: stale += 1
        (root/'train.json').write_text(json.dumps(logs, indent=2))
        if stale >= cfg['patience']: break
    return {'checkpoint': str(root/'best_validation.pt'), 'best_validation': best, 'epochs': len(logs), 'classes': len(vocabulary)}


def eval_islr(checkpoint, manifest, device='cpu'):
    encoder, state = load_selected_encoder(checkpoint, device)
    if state.get('task') != 'islr':
        raise ValueError('eval_islr requires a trained ISLR classifier checkpoint')
    model = ISLRModel(encoder, len(state['vocabulary'])).to(device)
    model.classifier.load_state_dict(state['classifier'], strict=True)
    dataset = PoseDataset(manifest, state['config']['data'], stage='islr')
    metrics, predictions = evaluate_islr_model(model, dataset, state['vocabulary'], state['config'])
    return {'metrics': metrics, 'predictions': predictions, 'checkpoint_sha256': sha256_file(checkpoint), 'manifest_sha256': sha256_file(manifest)}
