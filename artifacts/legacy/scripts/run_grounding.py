"""CPU subsequence-DTW baseline on verified ASL query/occurrence/absence gold."""
import argparse
import csv
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from _common import ROOT
from signrepr.grounding import evaluate, recording_bootstrap, subsequence_dtw
from signrepr.io import atomic_text, read_jsonl, sha256, write_json, write_jsonl


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Config/cache path escapes project')
    return path


def locked_rows(config, name):
    path = project_path(config[name])
    if sha256(path) != config[name+'_sha256']:
        raise ValueError('Locked '+name+' changed')
    return [r for _, r in read_jsonl(path)]


def prepare(config):
    if config.get('status') != 'LOCKED_VERIFIED_ASL_GROUNDING':
        raise ValueError('Grounding protocol blocked or unlocked: '+config.get('status', 'UNKNOWN'))
    gate = project_path(config['gate_path'])
    if sha256(gate) != config['gate_sha256']:
        raise ValueError('Grounding gate snapshot changed')
    gates = json.loads(gate.read_text())
    if any(not gates[k]['status'].startswith('PASS') for k in ['B-G0', 'B-G1', 'B-G2']):
        raise ValueError('Verified mapping, exhaustive occurrence/absence and grouping gates required')
    selected = project_path(config['validation_selection'])
    evidence = json.loads(selected.read_text())
    policy_keys = ['maximum_cost', 'min_duration', 'max_duration', 'nms_iou']
    if sha256(selected) != config['validation_selection_sha256'] or evidence['source_split'] != 'val' or evidence['test_used']:
        raise ValueError('Search/threshold parameters require validation-only selection')
    if evidence['parameters'] != {k: config[k] for k in policy_keys}:
        raise ValueError('Locked search/threshold parameters differ from selection evidence')
    if config['algorithm'] != 'subsequence_dtw_sum_cosine_then_path_length_normalization' or config['device'] != 'cpu':
        raise ValueError('Only declared CPU Q02 baseline supported')
    identity = config.get('common_feature_identity', {})
    if any(not isinstance(identity.get(k), str) or len(identity[k]) != 64 for k in ['config_sha256', 'implementation_sha256']):
        raise ValueError('Pinned common feature config and implementation required')
    thresholds = config['secondary_iou_thresholds']
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not 0 < x <= 1 for x in thresholds) or not 0 < config['primary_iou'] <= 1:
        raise ValueError('Register primary and secondary IoU thresholds before evaluation')
    queries, targets, gold = [locked_rows(config, k) for k in ['queries', 'targets', 'occurrences']]
    if not queries or not targets:
        raise ValueError('Empty locked query/target population')
    query_ids = {r['query_id'] for r in queries}; target_ids = {r['target_id'] for r in targets}
    if len(query_ids) != len(queries) or len(target_ids) != len(targets):
        raise ValueError('Duplicate stable query/target IDs')
    for row in queries+targets:
        if row['language'] != 'ASL' or row['split'] != 'test' or any(not row.get(k) or str(row[k]).lower() in ['unknown', 'none', 'null'] for k in ['recording_group', 'canonical_signer_id']):
            raise ValueError('Verified ASL test identity/grouping required')
    if {r['recording_group'] for r in queries} & {r['recording_group'] for r in targets} or {r['canonical_signer_id'] for r in queries} & {r['canonical_signer_id'] for r in targets}:
        raise ValueError('Query/target signer or source-recording overlap')
    for q in queries:
        if q['mapping_status'] != 'EXPERT_VERIFIED_SAME_ASL_LEXEME':
            raise ValueError('English gloss string does not establish same-sign query mapping')
    for target in targets:
        if target['annotation_status'] != 'EXPERT_VERIFIED_OCCURRENCES_AND_ABSENCE' or set(target['exhaustively_reviewed_query_ids']) != query_ids:
            raise ValueError('Unannotated query/target pair cannot be a negative')
        if not np.isfinite(target['duration_sec']) or target['duration_sec'] <= 0:
            raise ValueError('Invalid target source duration')
    raw, sources = {}, {'query': set(), 'target': set()}
    cache_root = project_path(config['cache_root'])
    media_root = Path(config['media_root']).resolve()
    for role, rows in [('query', queries), ('target', targets)]:
        for row in rows:
            video = (media_root / row['video_path']).resolve()
            shard = (cache_root / row['feature_path']).resolve()
            if not video.is_relative_to(media_root) or not shard.is_relative_to(cache_root):
                raise ValueError('Source/cache manifest path escapes declared root')
            if sha256(video) != row['video_sha256'] or sha256(shard) != row['feature_sha256']:
                raise ValueError('Immutable source/feature checksum differs')
            metadata_path = shard.with_suffix('.json')
            if sha256(metadata_path) != row['metadata_sha256']:
                raise ValueError('Locked feature metadata changed')
            metadata = json.loads(metadata_path.read_text())
            if metadata['source_sha256'] != row['video_sha256'] or metadata['shard_sha256'] != row['feature_sha256']:
                raise ValueError('Source-to-feature provenance differs')
            if any(metadata[k] != v for k, v in config['common_feature_identity'].items()):
                raise ValueError('Query/target frozen feature identity incompatible')
            sources[role].add(row['video_sha256'])
            with np.load(shard, allow_pickle=False) as data:
                features, mask = data['embeddings'], data['valid_mask']
                start, end = data['window_start_sec'], data['window_end_sec']
                if features.ndim != 2 or features.shape[1] != config['feature_dimension'] or not np.isfinite(features).all() or mask.dtype != bool or mask.shape != (len(features),):
                    raise ValueError('Invalid frozen feature dimensions/masks')
                frame_mask = data['valid_frame_mask']
                if frame_mask.dtype != bool or frame_mask.ndim != 2 or frame_mask.shape[0] != len(features):
                    raise ValueError('Invalid frame padding masks')
                if start.shape != mask.shape or end.shape != mask.shape or not np.isfinite(start).all() or not np.isfinite(end).all() or (end <= start).any() or not (np.diff(start) > 0).all() or not (np.diff(end) > 0).all():
                    raise ValueError('Invalid original feature interval clock')
                keep = mask & frame_mask.all(1)
                identifier = row[role+'_id']; raw[(role, identifier)] = (features[keep], start[keep], end[keep])
                if role == 'query' and not keep.any():
                    raise ValueError('Query has no valid tokens')
                if role == 'target' and len(end) and (start.min() < 0 or end.max() > row['duration_sec']+1e-6):
                    raise ValueError('Feature window outside target video duration')
    if sources['query'] & sources['target']:
        raise ValueError('Exact query/target source duplication')
    # Validate all gold support/duplicate occurrences before prediction scoring.
    evaluate([], gold, {r['target_id']: r['duration_sec'] for r in targets}, sorted(query_ids), config['primary_iou'])
    return queries, targets, gold, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Preserve attempts; use new project-local output')
    config = yaml.safe_load(args.config.read_text()); args.output.mkdir(parents=True)
    state = {'status': 'VALIDATING_GROUNDING_READINESS', 'pid': os.getpid(), 'device': 'cpu',
             'config_sha256': sha256(args.config), 'script_sha256': sha256(Path(__file__)),
             'metric_and_search_sha256': sha256(ROOT / 'src/signrepr/grounding.py')}
    started = time.perf_counter()
    try:
        queries, targets, gold, raw = prepare(config)
        coverage = {'scope': 'Every locked query evaluated on every exhaustively reviewed target.',
                    'queries': [{'query_id': r['query_id'], 'valid_nonpadded_tokens': len(raw[('query', r['query_id'])][0])} for r in queries],
                    'targets': [{'target_id': r['target_id'], 'recording_group': r['recording_group'], 'canonical_signer_id': r['canonical_signer_id'],
                                 'duration_sec': r['duration_sec'], 'valid_nonpadded_tokens': len(raw[('target', r['target_id'])][0]),
                                 'exhaustively_reviewed_query_ids': r['exhaustively_reviewed_query_ids']} for r in targets],
                    'query_target_pairs': len(queries)*len(targets), 'gold_occurrences': len(gold), 'failed_samples': []}
        write_json(args.output / 'coverage.json', coverage)
        work = sum(len(raw[('query', q['query_id'])][0])*len(raw[('target', t['target_id'])][0]) for q in queries for t in targets)
        if work > config['maximum_total_dtw_cells']:
            raise ValueError('Locked DTW work budget exceeded; no silent corpus reduction')
        if args.dry_run:
            state.update(status='PASS_GROUNDING_READINESS_NO_METRICS', estimated_dtw_cells=work, query_target_pairs=len(queries)*len(targets))
            write_json(args.output / 'status.json', state); print(json.dumps(state)); return
        state['status'] = 'RUNNING_Q02_FROZEN_BASELINE'; write_json(args.output / 'status.json', state)
        predictions = []
        for query in queries:
            qid = query['query_id']; features = raw[('query', qid)][0]
            for target in targets:
                tid = target['target_id']; tokens, starts, ends = raw[('target', tid)]
                candidates = subsequence_dtw(features, tokens, starts, ends,
                                             **{k: config[k] for k in ['maximum_cost', 'min_duration', 'max_duration', 'nms_iou']})
                predictions.extend({**r, 'query_id': qid, 'target_id': tid, 'recording_group': target['recording_group'],
                                    'split': 'test', 'gold_status': target['annotation_status'],
                                    'maximum_cost': config['maximum_cost'], 'nms_iou': config['nms_iou']} for r in candidates)
        write_jsonl(args.output / 'predictions.jsonl', predictions)
        csv_stream = io.StringIO()
        columns = ['query_id', 'target_id', 'recording_group', 'split', 'gold_status', 'start_sec', 'end_sec',
                   'score', 'mean_path_cost', 'path_length', 'query_tokens_covered', 'maximum_cost', 'nms_iou']
        writer = csv.DictWriter(csv_stream, fieldnames=columns); writer.writeheader(); writer.writerows(predictions)
        atomic_text(args.output / 'predictions.csv', csv_stream.getvalue())
        ids = sorted(r['query_id'] for r in queries)
        metrics = evaluate(predictions, gold, {r['target_id']: r['duration_sec'] for r in targets}, ids, config['primary_iou'])
        secondary = {str(threshold): evaluate(predictions, gold, {r['target_id']: r['duration_sec'] for r in targets}, ids, threshold)
                     for threshold in config['secondary_iou_thresholds']}
        defined = [m['macro_AP'] for m in secondary.values() if m['macro_AP'] is not None]
        statistics = recording_bootstrap(predictions, gold, targets, ids, config['primary_iou'], config['resamples'], config['seed'])
        write_json(args.output / 'metrics.json', {'primary': metrics, 'secondary_by_iou': secondary, 'recording_bootstrap': statistics,
                   'implementation_sha256': state['metric_and_search_sha256'], 'coverage_sha256': sha256(args.output / 'coverage.json'),
                   'primary_iou': config['primary_iou'], 'secondary_iou_thresholds': config['secondary_iou_thresholds']})
        summary = {**state, 'status': 'PASS_Q02_VERIFIED_GROUNDING_BASELINE_WITH_LIMITS', 'metrics': metrics, 'recording_bootstrap': statistics,
                   'secondary_metrics_by_iou': secondary, 'secondary_mean_macro_AP_across_registered_iou': float(np.mean(defined)) if defined else None,
                   'elapsed_seconds': time.perf_counter()-started, 'dtw_cells': work, 'trainable_parameters': 0,
                   'evaluated_query_target_pairs': len(queries)*len(targets), 'source_manifest_sha256': {k: config[k+'_sha256'] for k in ['queries', 'targets', 'occurrences']},
                   'predictions_sha256': sha256(args.output / 'predictions.jsonl'),
                   'limits': ['Standard subsequence-DTW baseline, no method novelty or representation adaptation.',
                              'Summed cost is minimized, then normalized by path length; minimum average cost is not optimized.',
                              'Duration constraints filter best paths after alignment; alternative constrained paths are not searched.',
                              'Masks omit invalid tokens but original source interval coordinates preserve gaps.',
                              'Threshold/NMS/duration chosen on validation only. No test selection or inferred negative labels.']}
        write_json(args.output / 'summary.json', summary); state.update(status=summary['status'])
        write_json(args.output / 'status.json', state); print(json.dumps({'status': state['status'], 'metrics': metrics}))
    except Exception as error:
        state.update(status='BLOCKED_GOLD_OR_MAPPING' if config.get('status', '').startswith('BLOCKED') else 'FAIL',
                     failure_reason=type(error).__name__+': '+str(error), elapsed_seconds=time.perf_counter()-started)
        write_json(args.output / 'status.json', state); print(json.dumps(state)); raise


if __name__ == '__main__':
    main()
