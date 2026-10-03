"""Offline CPU author-demo semantic ranking, gated by reviewed external gold.

Requires frozen, checksummed C04 cue caches and an independently reviewed pair
manifest. Dry-run validates readiness without model loading or network access.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import yaml

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.semantic_pairs import candidate_order, matched_credit, summarize, validate_pairs, wrong_video_mapping

C04_REVISION = '69d3d77aa4a4fec89048f5417d44fa717e36f6f8'
STREAM_SHAPES = {'face_features': 384, 'left_hand_features': 384, 'right_hand_features': 384, 'pose_features': 14}


def path_in(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest cache/media path escapes declared root')
    return path


def prepare(config):
    if config.get('status') != 'LOCKED_VERIFIED_EXTERNAL_SEMANTIC_PAIRS':
        raise ValueError('External semantic protocol remains blocked or unlocked: '+config.get('status', 'UNKNOWN'))
    gate_path = path_in(ROOT, config['gate_path'])
    if sha256(gate_path) != config['gate_sha256'] or json.loads(gate_path.read_text())['A-G0']['status'] != 'PASS_VERIFIED_EXTERNAL_SEMANTIC_PAIRS':
        raise ValueError('Verified external A-G0 gate required')
    manifest = path_in(ROOT, config['manifest'])
    if sha256(manifest) != config['manifest_sha256']:
        raise ValueError('Frozen external gold manifest changed')
    rows = validate_pairs([r for _, r in read_jsonl(manifest)])
    if config['device'] != 'cpu' or config['selection'] != 'NONE_FROZEN_EXTERNAL_EVALUATION' or config['source_revision'] != C04_REVISION:
        raise ValueError('Only registered CPU frozen author-demo evaluation is supported')
    if config['primary_metric'] != 'phenomenon_macro_utterance_pair_accuracy' or config['tie_credit'] != .5 or config['tie_tolerance'] < 0:
        raise ValueError('Locked semantic estimator/tie policy differs')
    media_root = Path(config['media_root']).resolve()
    cache_root = path_in(ROOT, config['cache_root'])
    cache, video_identity = {}, {}
    for row in rows:
        if row['split'] != 'external_test':
            raise ValueError('ASL-MTP gold is external evaluation only')
        video = path_in(media_root, row['video_path'])
        streams = path_in(cache_root, row['streams_path'])
        provenance = path_in(cache_root, row['streams_provenance_path'])
        if any(sha256(p) != row[k] for p, k in [(video, 'video_sha256'), (streams, 'streams_sha256'), (provenance, 'streams_provenance_sha256')]):
            raise ValueError('Reviewed source/cue checksum differs')
        identity = (str(video), row['video_sha256'], row['streams_sha256'])
        if video_identity.setdefault(row['utterance_id'], identity) != identity:
            raise ValueError('Canonical utterance has ambiguous video/cue identity')
        meta = json.loads(provenance.read_text())
        if meta['source_revision'] != C04_REVISION or meta['video_sha256'] != row['video_sha256'] or meta['streams_sha256'] != row['streams_sha256']:
            raise ValueError('Cue cache does not use registered C04 source/video')
        if not meta['all_source_frames_verified'] or meta['gold_intervals_used_as_input']:
            raise ValueError('Full source clock needed; gold intervals cannot crop model input')
        for name, digest in meta['source_file_sha256'].items():
            if sha256(path_in(ROOT, name)) != digest:
                raise ValueError('Cue cache source-code identity changed')
        with np.load(streams, allow_pickle=False) as data:
            frames = len(data['sampled_indices']); clock = data['source_frame_intervals_sec']
            if not 1 <= frames <= config['max_frames'] or not np.array_equal(data['sampled_indices'], np.arange(frames)):
                raise ValueError('C04 whole-source stride1 frame policy differs or exceeds cap')
            if clock.shape != (frames, 2) or not np.isfinite(clock).all() or (clock[:, 1] <= clock[:, 0]).any() or not (np.diff(clock[:, 0]) > 0).all():
                raise ValueError('Cue source clock is invalid')
            for key, width in STREAM_SHAPES.items():
                if data[key].shape != (frames, width) or not np.isfinite(data[key]).all():
                    raise ValueError('Cue stream shape/values differ')
            cache[row['utterance_id']] = {key: data[key].astype(np.float32) for key in STREAM_SHAPES}
    mapping = wrong_video_mapping(rows, config['seed'])
    return rows, cache, mapping


def load_model(config):
    import torch
    registry = json.loads((ROOT / 'provenance/assets.json').read_text())['assets']
    dependencies = ['W08'] + [k for k in registry if k.startswith('C04_') or k.startswith('W08_CONFIG_')]
    verified = {}
    if config['checkpoint_sha256'] != registry['W08']['expected_sha256']:
        raise ValueError('Requested checkpoint differs from registered author-demoW08')
    for key in dependencies:
        path = ROOT / registry[key]['local_path']
        if sha256(path) != registry[key]['expected_sha256']:
            raise ValueError('Offline author-demo dependency changed: '+key)
        verified[key] = path
    sys.path[:0] = [str(ROOT / 'third_party/shubert-author-demo'), str(ROOT / 'third_party/SHuBERT/fairseq')]
    import inference
    from transformers import ByT5Tokenizer
    weights = torch.load(verified['W08'], map_location='cpu', weights_only=True)
    model_config = inference.SignLanguageByT5Config.from_pretrained(str(verified['W08'].parent), local_files_only=True)
    model = inference.SignLanguageByT5ForConditionalGeneration(model_config)
    loaded = model.load_state_dict(weights, strict=True)
    del weights
    model.eval().requires_grad_(False)
    tokenizer = ByT5Tokenizer.from_pretrained(str(ROOT / 'checkpoints/shubert_author_hf/models/byt5_base'), local_files_only=True)
    return model, tokenizer, {'strict': True, 'missing_keys': list(loaded.missing_keys), 'unexpected_keys': list(loaded.unexpected_keys),
                              'asset_sha256': {k: sha256(p) for k, p in verified.items()}, 'torch': torch.__version__,
                              'transformers': __import__('transformers').__version__}


def score(model, tokenizer, streams, texts, max_tokens):
    import torch
    from signrepr.scoring import token_nll
    scores = []
    frames = len(streams['pose_features'])
    inputs = {key: torch.from_numpy(value).unsqueeze(0) for key, value in streams.items()}
    inputs['attention_mask'] = torch.ones((1, frames), dtype=torch.long)
    for text in texts:
        labels = tokenizer(text, return_tensors='pt')['input_ids']
        if labels.shape[1] > max_tokens or labels[0, -1].item() != tokenizer.eos_token_id:
            raise ValueError('Candidate token budget/EOS policy differs; truncation forbidden')
        with torch.inference_mode():
            output = model(**inputs, labels=labels, return_dict=True)
            values = token_nll(output.logits, labels)
        scores.append({'sum_nll': float(values['sum_nll'][0]), 'token_count': int(values['token_count'][0]), 'mean_nll': float(values['mean_nll'][0])})
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT) or args.output.exists():
        raise ValueError('Use a new project-local output; preserve existing results')
    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True)
    state = {'status': 'VALIDATING_GOLD_AND_CUE_READINESS', 'config_sha256': sha256(args.config),
             'script_sha256': sha256(Path(__file__)), 'device': 'cpu', 'pid': os.getpid()}
    started = time.perf_counter()
    try:
        rows, cache, mapping = prepare(config)
        if args.dry_run:
            state.update(status='PASS_READINESS_NO_MODEL_EVALUATION', pairs=len(rows), utterances=len(cache), cached_stream_bytes=sum(v.nbytes for s in cache.values() for v in s.values()))
            write_json(args.output / 'status.json', state); print(json.dumps(state)); return
        import torch
        torch.set_num_threads(config['torch_threads'])
        model, tokenizer, audit = load_model(config)
        state.update(status='RUNNING_FROZEN_EXTERNAL_EVALUATION'); write_json(args.output / 'status.json', state)
        conditions = ['video', 'wrong_video', 'blank_visual', 'shorter_text_prior']
        credits, predictions = {k: [] for k in conditions}, []
        for row in rows:
            order = candidate_order(row['pair_id'], config['seed'])
            source_texts = [row['matched_text'], row['mismatched_text']]
            texts = [source_texts[i] for i in order]
            matched_position = order.index(0)
            streams = cache[row['utterance_id']]
            variants = {'video': streams, 'wrong_video': cache[mapping[row['utterance_id']]],
                        'blank_visual': {k: np.zeros_like(v) for k, v in streams.items()}}
            for condition, visual in variants.items():
                scores = score(model, tokenizer, visual, texts, config['max_candidate_tokens'])
                credit = matched_credit([v['mean_nll'] for v in scores], matched_position, config['tie_tolerance'])
                credits[condition].append(credit)
                predictions.append({'pair_id': row['pair_id'], 'utterance_id': row['utterance_id'], 'recording_group': row['recording_group'],
                                    'phenomenon': row['phenomenon'], 'condition': condition, 'candidate_order': order,
                                    'matched_position': matched_position, 'matched_credit': credit, 'scores': scores,
                                    'visual_source_utterance': mapping[row['utterance_id']] if condition == 'wrong_video' else row['utterance_id']})
            lengths = [len(tokenizer(t)['input_ids']) for t in texts]
            credit = matched_credit(lengths, matched_position)
            credits['shorter_text_prior'].append(credit)
            predictions.append({'pair_id': row['pair_id'], 'utterance_id': row['utterance_id'], 'recording_group': row['recording_group'],
                                'phenomenon': row['phenomenon'], 'condition': 'shorter_text_prior', 'candidate_order': order,
                                'matched_position': matched_position, 'matched_credit': credit, 'token_counts': lengths})
        # Each JSONL record preserves exact candidate permutation, EOS counts,
        # source control identity, sums and means. No model/head selection occurs.
        from signrepr.io import atomic_text
        atomic_text(args.output / 'predictions.jsonl', ''.join(json.dumps(r)+'\n' for r in predictions))
        result = summarize(rows, credits, resamples=config['resamples'], seed=config['seed'])
        summary = {**state, 'status': 'PASS_EXTERNAL_AUTHOR_DEMO_SEMANTIC_DIAGNOSTIC_WITH_LIMITS',
                   'model': 'SHuBERT-author-demo-11625', 'source_revision': C04_REVISION, 'load_audit': audit,
                   'elapsed_seconds': time.perf_counter()-started, 'metrics': result,
                   'predictions_sha256': sha256(args.output / 'predictions.jsonl'), 'trainable_parameters': 0,
                   'wrong_video_donor_mapping': mapping,
                   'total_parameters': sum(v.numel() for v in model.parameters()),
                   'limits': ['W08demo encoder differs from frozenW02; no causal W02information-loss inference.',
                              'Cue crops can be carry-forward/black. Blank inputs preserve source length/mask and are distribution-shift controls.',
                              'Shorter-text prior is a locked length heuristic, not a language-model prior.',
                              'Wrong-video donors can be reused; recipient recording bootstrap does not include donor dependence.',
                              'No training/tuning, no complete grammatical-scope or absent-phenomenon claim.']}
        write_json(args.output / 'summary.json', summary); state.update(status=summary['status'])
        write_json(args.output / 'status.json', state); print(json.dumps({'status': state['status'], 'metrics': result}))
    except Exception as error:
        state.update(status='BLOCKED_GOLD_OR_ACCESS' if config.get('status', '').startswith('BLOCKED') else 'FAIL',
                     failure_reason=type(error).__name__+': '+str(error), elapsed_seconds=time.perf_counter()-started)
        write_json(args.output / 'status.json', state); print(json.dumps(state)); raise


if __name__ == '__main__':
    main()
