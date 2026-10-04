"""Bounded paired end-to-end timing with actual pre-encoder window selection."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from budget import inspect_budget
from cache import CacheReader, file_sha, sample_map
from resources import gpu_preflight, configure_gpu, scratch_preflight
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir', 'cache-manifest', 'quality-run', 'launch-gate'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise ValueError('Register a new timing run; no partial resume')
    config = json.loads(args.config.read_text())
    gate = json.loads(args.launch_gate.read_text())
    quality_status = json.loads((args.quality_run / 'status.json').read_text())
    if (gate['verdict'] != 'VALIDATED' or gate['protocol'] != config['protocol']
            or gate['timing_config_sha256'] != file_sha(args.config)
            or gate['cache_manifest_sha256'] != file_sha(args.cache_manifest)
            or gate['quality_status_sha256'] != file_sha(args.quality_run / 'status.json')
            or gate['policy_code_sha256'] != file_sha(ROOT / 'window_budget.py')
            or quality_status['status'] != 'completed' or quality_status['protocol'] != config['protocol']):
        raise ValueError('Timing protocol/quality/cache gate mismatch')
    for name, digest in quality_status['output_hashes'].items():
        if file_sha(args.quality_run / name) != digest:
            raise ValueError('Quality outputs changed')
    quality = json.loads((args.quality_run / 'quality.json').read_text())
    if not any(p['quality_count_screen'] for p in quality['policies'].values()):
        raise ValueError('No policy meets the registered quality/count floor')
    identity = config['cache_identity']
    for name, digest in (('signrep_adapter.py', identity['preprocessing_code_sha256']),
                         ('temporal.py', identity['temporal_code_sha256'])):
        if file_sha(ROOT / name) != digest:
            raise ValueError('Encoder/temporal recipe changed')
    checkpoint = REPO / config['checkpoint']
    if file_sha(checkpoint) != identity['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    budget = inspect_budget([REPO / p for p in config['budget_runtime_roots']], 'H002', 14400, 3600)
    scratch = scratch_preflight('/dev/shm', 1, 1 << 30)
    observation = gpu_preflight(8192, 100)
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        from baseline import support_ids
        from probes import normalize_rows
        from metrics import ranked_classes
        from window_budget import POLICIES, pool_policies, pooled_vector, selected_indices
        if config['policies'] != list(POLICIES) or config['repeats'] != 3 or len(config['cost_samples']) != 5:
            raise ValueError('Registered timing sample/policy/repeat changed')
        setup_start = time.perf_counter()
        cohort_path, audit_path = (REPO / config[k] for k in ('cohort_manifest', 'video_audit'))
        if file_sha(cohort_path) != identity['cohort_sha256'] or file_sha(audit_path) != identity['audit_sha256']:
            raise ValueError('Locked manifests changed')
        cohort, audit = (json.loads(p.read_text()) for p in (cohort_path, audit_path))
        indexed = sample_map(cohort)
        videos = {s['sample_id']: s for s in audit['samples']}
        reader = CacheReader(args.cache_manifest, identity, cohort, audit)
        ids, matrices, _, _ = pool_policies(reader, cohort)
        position = {k: i for i, k in enumerate(ids)}
        galleries = {}
        methods = ['full_activity'] + list(POLICIES)
        for method in methods:
            galleries[method] = []
            for seed in (1701, 1702, 1703):
                supports = support_ids(cohort, seed, 8)
                x = normalize_rows(matrices[method][[position[k] for k in supports]])
                labels = np.array([indexed[k]['class_id'] for k in supports])
                galleries[method].append((x, [np.flatnonzero(labels == c) for c in range(100)]))
        del matrices
        expected = {key: reader.load(key) for key in config['cost_samples']}
        for key in config['cost_samples']:
            sample = indexed[key]
            path = Path(sample['input_path'])
            if path.stat().st_size != sample['source_bytes'] or file_sha(path) != sample['source_sha256']:
                raise ValueError('Timing input source changed')
        gallery_setup_seconds = time.perf_counter() - setup_start
        run.save('preflight.json', dict(GPU=observation, scratch=scratch, budget=budget,
                 gallery_setup_seconds=gallery_setup_seconds, cache_manifest_sha256=file_sha(args.cache_manifest)))
        run.start_gpu(budget)
        import cv2
        import torch
        from signrep_adapter import load_model, decode_video, preprocess_rgb, infer_windows
        if {p: importlib.metadata.version(p) for p in identity['packages']} != identity['packages']:
            raise ValueError('Encoder environment changed')
        upstream = REPO / 'third_party/SignRep'
        producer = json.loads((REPO / config['extraction_report']).read_text())
        hashes = {str(p.relative_to(upstream)): file_sha(p) for p in sorted(upstream.rglob('*.py'))}
        if hashes != producer['upstream_python_sha256']:
            raise ValueError('Upstream model changed')
        device = configure_gpu(observation)
        cv2.setNumThreads(1)
        torch.manual_seed(args.seed)
        model_start = time.perf_counter()
        model, params = load_model(checkpoint, device)
        torch.cuda.synchronize()
        model_setup_seconds = time.perf_counter() - model_start
        first = config['cost_samples'][0]
        rgb, pts, _ = decode_video(indexed[first]['input_path'], 2000)
        frames = preprocess_rgb(rgb)
        warmup_start = time.perf_counter()
        infer_windows(model, frames, pts, list(range(min(4, videos[first]['windows']))), 16, 2, 2)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_start
        del frames, rgb
        order = np.random.default_rng(1701).permutation(methods).tolist()
        rows = []
        for repeat in range(3):
            for sample_number, key in enumerate(config['cost_samples']):
                shift = (repeat * 5 + sample_number) % len(order)
                sequence = order[shift:] + order[:shift]
                source = indexed[key]
                cached_features, cached_activity, cached_centers = expected[key]
                for ordinal, method in enumerate(sequence):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    rgb, pts, _ = decode_video(source['input_path'], 2000)
                    if pts != videos[key]['pts_s']:
                        raise ValueError('Timing native PTS changed')
                    t1 = time.perf_counter()
                    frames = preprocess_rgb(rgb)
                    del rgb
                    t2 = time.perf_counter()
                    indices = selected_indices(method, (len(pts) - 16) // 2 + 1, source['source_sha256'])
                    outputs, plan = infer_windows(model, frames, pts, indices, 16, 2, 2)
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    features, activity = outputs['features'].numpy(), outputs['cls_active'].numpy()
                    vector = pooled_vector(features, activity, 'full_activity', list(range(len(indices))))
                    t4 = time.perf_counter()
                    vector = normalize_rows(vector[None])[0]
                    for gallery, class_rows in galleries[method]:
                        similarities = gallery @ vector
                        scores = np.array([similarities[positions].max() for positions in class_rows])
                        if not np.isfinite(scores).all():
                            raise ValueError('Nonfinite exact class-search score')
                        ranked_classes(scores.tolist(), list(range(100)))
                    t5 = time.perf_counter()
                    if (len(plan) != len(indices)
                            or not np.array_equal([p['center_s'] for p in plan], cached_centers[indices])
                            or not np.allclose(features, cached_features[indices], atol=1e-5, rtol=1e-4)
                            or not np.allclose(activity, cached_activity[indices], atol=1e-5, rtol=1e-4)):
                        raise ValueError('Actual selected inference/cache parity failed')
                    rows.append(dict(sample_id=key, split=source['split'], repeat=repeat, ordinal=ordinal,
                        method=method, encoded_windows=len(indices), source_windows=len(cached_centers),
                        decode_seconds=t1-t0, preprocess_seconds=t2-t1, encoder_wrapper_seconds=t3-t2,
                        pool_seconds=t4-t3, three_gallery_search_seconds=t5-t4, end_to_end_seconds=t5-t0,
                        features_max_abs_error=float(np.abs(features-cached_features[indices]).max()),
                        activity_max_abs_error=float(np.abs(activity-cached_activity[indices]).max())))
                    del frames, outputs
                print(json.dumps(dict(completed_blocks=len(rows)//7, expected_blocks=15)), flush=True)
        summary = {}
        for method in methods:
            selected = [r for r in rows if r['method'] == method]
            values = np.array([r['end_to_end_seconds'] for r in selected])
            ratios = {key: float(np.median([r['end_to_end_seconds'] for r in selected if r['sample_id'] == key]) /
                       np.median([r['end_to_end_seconds'] for r in rows if r['method'] == 'full_activity' and r['sample_id'] == key]))
                      for key in config['cost_samples']}
            windows = sum(r['encoded_windows'] for r in selected)
            full = sum(r['encoded_windows'] for r in rows if r['method'] == 'full_activity')
            ratio = float(np.median(list(ratios.values())))
            summary[method] = dict(p50_seconds=float(np.quantile(values,.5)), p95_seconds=float(np.quantile(values,.95)),
                source_paired_median_ratios=ratios, median_paired_ratio=ratio, encoded_windows_all_repeats=windows,
                sample_window_reduction_factor=full/windows, actual_cost_screen=ratio<=.8 and windows*2<=full)
        run.save('timing.json', dict(run_id=args.run_id, summaries=summary, observations=rows,
            observation_count=len(rows), source_count=5, repeats=3, quality_run=quality_status['run_id'],
            quality_status_sha256=file_sha(args.quality_run/'status.json'), model_setup_seconds=model_setup_seconds,
            gallery_setup_seconds=gallery_setup_seconds, warmup_seconds=warmup_seconds, initial_policy_order=order,
            GPU=observation, model_params=params, upstream_python_sha256=hashes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            scope='five fixed mixed train/dev sources; shared GPU paired timing; no quality score on cost examples',
            cost_exclusions='model/gallery setup, SHA identity checks and parity checks separately recorded; OS cache uncontrolled',
            test_access='none', actual_pre_encoder_selection=True, all_selected_output_cache_parity=True))
        run.status['output_sha256'] = file_sha(run.output / 'timing.json')


if __name__ == '__main__':
    main()
