"""End-to-end frozen sequence backbone workflows for CSLR and mT5 SLT."""
from pathlib import Path
import json
import math
import torch
from torch import nn
from .islr import load_selected_encoder, check_splits, frozen_features, set_epoch_lr
from .pretrain import seed_everything
from .checkpoints import atomic_save, rng_state
from .provenance import provenance
from .cslr import CTCHead, ctc_loss
from .slt import SLTBridge, translation_labels, load_mt5
from ..data.pose_dataset import PoseDataset
from ..data.schema import sha256_file
from ..evaluation.temporal import ctc_decode, ctc_minimum_length, word_error_rate
from ..evaluation.cache import save_sequence_features
from ..evaluation.extract import encode_full_sequence_train
from ..models.part_tokenizer import token_metadata
from ..data.collate import collate_pose


def sequence_batch(sequences, indices, device):
    lengths = torch.tensor([sequences[i]['length'] for i in indices], device=device)
    cap = int(lengths.max())
    dim = sequences[indices[0]]['features'].shape[-1]
    x = torch.zeros(len(indices), cap, dim, device=device)
    for row, index in enumerate(indices):
        x[row, :lengths[row]] = sequences[index]['features'].to(device)
    return x, lengths


def sequence_glosses(row):
    if row.get('gloss_sequence') is None or not row.get('gloss_namespace'):
        raise ValueError('CSLR requires verified gloss sequence and namespace')
    return [(row['language'], row['gloss_namespace'], str(token)) for token in row['gloss_sequence']]


def slt_scores(references, hypotheses, tokenize='13a'):
    import sacrebleu
    from rouge_score import rouge_scorer
    bleu = sacrebleu.metrics.BLEU(tokenize=tokenize, effective_order=False)
    score = bleu.corpus_score(hypotheses, [references])
    class CharacterTokenizer:
        def tokenize(self, text):
            return [char for char in text if not char.isspace()]
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=CharacterTokenizer() if tokenize == 'zh' else None)
    # Explicit segmentation policy shared between references and hypotheses.
    if tokenize == 'zh':
        ref_rouge = [' '.join(x) for x in references]
        hyp_rouge = [' '.join(x) for x in hypotheses]
    else:
        ref_rouge, hyp_rouge = references, hypotheses
    r = sum(rouge.score(a, b)['rougeL'].fmeasure for a, b in zip(ref_rouge, hyp_rouge))/len(references)
    return {'BLEU4': score.score, 'BLEU_signature': str(bleu.get_signature()), 'ROUGE_L': r,
            'ROUGE_policy': 'character_spaced' if tokenize == 'zh' else 'rouge_score_default_words', 'num_utterances': len(references)}


@torch.inference_mode()
def evaluate_sequence(head, sequences, rows, task, config, vocabulary=None, tokenizer=None, device='cpu'):
    head.eval()
    references, hypotheses = [], []
    for i, (row, seq) in enumerate(zip(rows, sequences)):
        if task == 'cslr':
            reference = sequence_glosses(row)
            if seq['has_observation']:
                x, lengths = sequence_batch(sequences, [i], device)
                pred = head(x, lengths)[0, :lengths[0]].argmax(-1).tolist()
                hypothesis = [tuple(vocabulary[token-1]) for token in ctc_decode(pred)]
            else: hypothesis = []
        else:
            reference = row.get('translation')
            if not isinstance(reference, str):
                raise ValueError('SLT requires actual translation text')
            if seq['has_observation']:
                x, lengths = sequence_batch(sequences, [i], device)
                ids = head.generate(x, lengths, num_beams=config['downstream']['beam'], max_new_tokens=config['downstream']['max_new_tokens'])
                hypothesis = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            else: hypothesis = ''
        references.append(reference); hypotheses.append(hypothesis)
    metrics = word_error_rate(references, hypotheses) if task == 'cslr' else slt_scores(references, hypotheses, config['downstream']['tokenize'])
    metrics['valid_utterances'] = sum(seq['has_observation'] for seq in sequences)
    return metrics, hypotheses


def train_sequence(config, encoder_checkpoint, task, device='cpu', decoder=None, tokenizer=None):
    if task not in ('cslr', 'slt'):
        raise ValueError('Unknown sequence task')
    cfg = config['downstream']
    frozen = cfg['frozen']
    if not frozen and task == 'slt':
        raise ValueError('Main SLT workflow freezes pose; use the CSLR fine-tune branch for encoder-gradient experiments')
    seed_everything(config['run']['seed'])
    encoder, initial = load_selected_encoder(encoder_checkpoint, device)
    if encoder.config != config['model']:
        raise ValueError('Sequence architecture does not match source checkpoint')
    encoder.requires_grad_(not frozen).eval()
    train = PoseDataset(config['data']['train_manifest'], config['data'], stage=task)
    validation = PoseDataset(config['data']['validation_manifest'], config['data'], stage=task)
    check_splits(train, validation)
    metadata = provenance(config, config['data']['train_manifest'])
    metadata['stage'] = task
    if frozen:
        train_seq, val_seq = frozen_features(encoder, train, config), frozen_features(encoder, validation, config)
    else:
        train_seq = [{'length': (sample['length']+1)//2,
                      'has_observation': bool(token_metadata(collate_pose([sample]))['valid'].any())} for sample in train]
        val_seq = None
    eligible = [i for i, seq in enumerate(train_seq) if seq['has_observation']]
    if not eligible:
        raise ValueError('No observed training utterances')
    vocabulary = None
    if task == 'cslr':
        vocabulary = sorted({g for row in train.rows for g in sequence_glosses(row)})
        mapping = {g: i+1 for i, g in enumerate(vocabulary)}
        targets = [[mapping[g] for g in sequence_glosses(row)] for row in train.rows]
        infeasible = [train.rows[i]['sample_id'] for i in eligible if train_seq[i]['length'] < ctc_minimum_length(targets[i])]
        if infeasible:
            raise ValueError(f'Infeasible training CTC alignments: {infeasible[:10]} ({len(infeasible)} total)')
        head = CTCHead(len(vocabulary), config['model']['sequence_dim'], cfg.get('hidden_dim', 256)).to(device)
        groups = [{'params': head.parameters()}] if frozen else [{'params': encoder.parameters()}, {'params': head.parameters()}]
        optimizer = torch.optim.AdamW(groups, lr=cfg['head_lr'], weight_decay=cfg['weight_decay'])
        rates = [cfg['head_lr']] if frozen else [cfg.get('encoder_lr', 1e-5), cfg['head_lr']]
    else:
        if decoder is None:
            decoder, tokenizer = load_mt5(cfg['decoder'], cfg.get('decoder_revision'), cfg.get('local_files_only', False))
        head = SLTBridge(decoder, config['model']['sequence_dim']).to(device)
        optimizer = torch.optim.AdamW([{'params': head.bridge.parameters()}, {'params': head.decoder.parameters()}], lr=cfg['head_lr'], weight_decay=cfg['weight_decay'])
        rates = [cfg['head_lr'], cfg['decoder_lr']]
        # Audit every target before fitting, with truncation disabled.
        for row in train.rows+validation.rows:
            if not isinstance(row.get('translation'), str):
                raise ValueError('SLT training/validation requires real translation text')
            translation_labels(tokenizer, [row['translation']], cfg.get('target_max_tokens'))
    root = Path(config['run']['output_dir']); root.mkdir(parents=True, exist_ok=True)
    cache_metadata = {'encoder_checkpoint_sha256': sha256_file(encoder_checkpoint), 'input_profile': config['data']['coordinate_profile'],
                      'train_manifest_sha256': sha256_file(config['data']['train_manifest']), 'validation_manifest_sha256': sha256_file(config['data']['validation_manifest']),
                      'window_frames': config['evaluation']['window_frames'], 'stride_frames': config['evaluation']['window_stride_frames'],
                      'stitch_version': 'aligned_unique_segment_mean_v1'}
    cache_metadata['cache_enabled'] = frozen
    (root/'feature_cache_provenance.json').write_text(json.dumps(cache_metadata, indent=2))
    for split, sequences, rows in ([('train', train_seq, train.rows), ('validation', val_seq, validation.rows)] if frozen else []):
        index = []
        for i, (sequence, row) in enumerate(zip(sequences, rows)):
            path = root/'feature_cache'/split/f'{i:08d}.npz'
            save_sequence_features(path, sequence)
            index.append({'sample_id': row['sample_id'], 'pose_sha256': row['pose_sha256'], 'path': path.name,
                          'status': sequence['status'], 'length': sequence['length']})
        (root/'feature_cache'/split/'index.json').write_text(json.dumps(index, indent=2))
    best, stale, logs = float('inf') if task == 'cslr' else -float('inf'), 0, []
    optimizer_steps = 0
    planned_steps = cfg['epochs']*((len(eligible)+cfg['batch_size']-1)//cfg['batch_size'])
    warmup_steps = max(1, round(planned_steps*cfg.get('warmup_fraction', .05))) if task == 'slt' else 0
    for epoch in range(cfg['epochs']):
        head.train()
        if not frozen: encoder.train()
        set_epoch_lr(optimizer, epoch, cfg['epochs'], cfg['warmup_epochs'], rates)
        order = [eligible[i] for i in torch.randperm(len(eligible)).tolist()]
        total, updates = 0., 0
        for start in range(0, len(order), cfg['batch_size']):
            ids = order[start:start+cfg['batch_size']]
            if task == 'slt':
                fraction = (optimizer_steps+1)/warmup_steps if optimizer_steps < warmup_steps else (1+math.cos(math.pi*(optimizer_steps-warmup_steps)/max(1, planned_steps-warmup_steps)))/2
                for group, rate in zip(optimizer.param_groups, rates): group['lr'] = rate*fraction
            if frozen:
                x, lengths = sequence_batch(train_seq, ids, device)
            else:
                features = [encode_full_sequence_train(encoder, train[i], config['evaluation']['window_frames'], config['evaluation']['window_stride_frames'])[0] for i in ids]
                lengths = torch.tensor([len(f) for f in features], device=device)
                x = nn.utils.rnn.pad_sequence(features, batch_first=True)
            if task == 'cslr':
                loss = ctc_loss(head(x, lengths), [targets[i] for i in ids], lengths)
            else:
                labels = translation_labels(tokenizer, [train.rows[i]['translation'] for i in ids], cfg.get('target_max_tokens'), device)
                loss = head(x, lengths, labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite sequence loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            parameters = list(head.parameters()) + ([] if frozen else list(encoder.parameters()))
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError('Nonfinite sequence gradients')
            torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
            optimizer_steps += 1
            total += float(loss.detach()); updates += 1
        if not frozen: val_seq = frozen_features(encoder, validation, config)
        metrics, _ = evaluate_sequence(head, val_seq, validation.rows, task, config, vocabulary, tokenizer, device)
        score = metrics['WER'] if task == 'cslr' else metrics['BLEU4']
        if score is None:
            raise ValueError('Validation has no reference gloss tokens')
        state = {'task': task, 'config': config, 'metadata': metadata, 'head': head.state_dict(), 'vocabulary': vocabulary, 'epoch': epoch+1,
                 'optimizer_step': optimizer_steps, 'warmup_steps': warmup_steps,
                 'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'validation': metrics, 'cache_provenance': cache_metadata,
                 'encoder_checkpoint': str(Path(encoder_checkpoint).resolve()),
                 'encoder': None if frozen else encoder.state_dict(),
                 'supervision': f'{task}_supervised_' + ('frozen_backbone' if frozen else 'encoder_finetuning')}
        atomic_save(state, root/'last.pt')
        better = score < best if task == 'cslr' else score > best
        if better:
            best, stale = score, 0; atomic_save(state, root/'best_validation.pt')
        else: stale += 1
        logs.append({'epoch': epoch+1, 'train_loss': total/updates, 'skipped_train_samples': len(train)-len(eligible), 'validation': metrics})
        (root/'train.json').write_text(json.dumps(logs, indent=2))
        if stale >= cfg['patience']: break
    return {'checkpoint': str(root/'best_validation.pt'), 'task': task, 'best_validation': best, 'epochs': len(logs)}
