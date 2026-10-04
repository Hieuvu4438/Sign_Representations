"""P003 actual pre-encoder selection for every locked test query; no scoring."""
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
from cache import CacheReader, file_sha
from confirmatory import METHODS, RECIPES, SEEDS, activity_proxy, build_galleries, check_freeze, query_map, search
from resources import configure_gpu, gpu_preflight, scratch_preflight
from run_context import RunContext
from window_budget import pooled_vector, selected_indices


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'freeze', 'prepared-run', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise ValueError('No partial scientific resume; register a diagnosed new attempt')
    check_freeze(args.freeze, args.config)
    config = json.loads(args.config.read_text())
    prepared = json.loads((args.prepared_run / 'status.json').read_text())
    if prepared['status'] != 'completed' or prepared['protocol'] != config['protocol']:
        raise ValueError('Held-out preparation incomplete')
    for name, digest in prepared['output_hashes'].items():
        if file_sha(args.prepared_run / name) != digest:
            raise ValueError('Prepared held-out artifacts changed')
    estimate = json.loads((args.prepared_run / 'preflight_estimate.json').read_text())
    if not estimate['launch_eligible']:
        raise ValueError('Registered cost/duplicate audit gate failed')
    budget = inspect_budget([REPO / p for p in config['budget_runtime_roots']], 'H002',
                            config['project_gpu_ceiling_seconds'], config['hypothesis_gpu_ceiling_seconds'])
    if estimate['estimated_gpu_allocation_seconds'] > min(budget['remaining_seconds'], config['timeout_seconds']):
        raise ValueError('Representative cost estimate exceeds actual remaining allowance')
    scratch = scratch_preflight('/dev/shm', estimate['planned_pooled_payload_bytes'] + (16 << 20), 1 << 30)
    observation = gpu_preflight(8192, 100)
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        cohort = json.loads((REPO / config['cohort_manifest']).read_text())
        old_audit = json.loads((REPO / config['video_audit']).read_text())
        test = json.loads((args.prepared_run / 'test_manifest.json').read_text())
        audit = json.loads((args.prepared_run / 'video_audit.json').read_text())
        indexed = query_map(cohort, test)
        videos = {s['sample_id']: s for s in audit['samples']}
        ids = sorted(indexed)
        if set(videos) != set(ids) or not audit['passed'] or audit['candidate_count']:
            raise ValueError('Incomplete held-out timeline/duplicate audit')
        identity = config['cache_identity']
        reader = CacheReader(config['cache_manifest'], identity, cohort, old_audit)
        gallery_started = time.perf_counter()
        galleries, gallery_meta = build_galleries(reader, cohort)
        gallery_setup_seconds = time.perf_counter() - gallery_started
        if len(cohort['classes']) != 100 or tuple(config['methods']) != METHODS or tuple(config['support_seeds']) != SEEDS:
            raise ValueError('Locked cohort/method/gallery specification changed')
        checkpoint = REPO / config['checkpoint']
        if file_sha(checkpoint) != identity['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        producer = json.loads((REPO / config['extraction_report']).read_text())
        upstream = REPO / 'third_party/SignRep'
        upstream_hashes = {str(p.relative_to(upstream)): file_sha(p) for p in sorted(upstream.rglob('*.py'))}
        if upstream_hashes != producer['upstream_python_sha256']:
            raise ValueError('Upstream encoder sources changed')
        if {p: importlib.metadata.version(p) for p in identity['packages']} != identity['packages']:
            raise ValueError('Encoder environment changed')
        run.save('preflight.json', dict(GPU=observation, scratch=scratch, budget=budget, estimate=estimate,
            prepared_status_sha256=file_sha(args.prepared_run / 'status.json'), freeze_sha256=file_sha(args.freeze),
            cache_manifest_sha256=file_sha(config['cache_manifest']), gallery_setup_seconds=gallery_setup_seconds,
            gallery_metadata=gallery_meta))
        pooled_dir = Path('/dev/shm') / f'signrep_{args.run_id}'
        pooled_dir.mkdir(exist_ok=False)
        pooled = {m: np.empty((len(ids), 768), dtype=np.float64) for m in METHODS}
        rankings = {f'{m}_s{s}': np.empty((len(ids), 100), dtype=np.int16) for m in METHODS for s in SEEDS}
        run.start_gpu(budget)
        import cv2
        import torch
        from signrep_adapter import decode_video, infer_windows, load_model, preprocess_rgb
        device = configure_gpu(observation)
        cv2.setNumThreads(1)
        torch.manual_seed(args.seed)
        setup_started = time.perf_counter()
        model, params = load_model(checkpoint, device)
        torch.cuda.synchronize()
        model_setup_seconds = time.perf_counter() - setup_started
        # Same fixed non-test source as pilot. Erratum: HATCH2 is dev, not train.
        warmup_source = next(s for s in cohort['samples'] if s['sample_id'] == config['warmup_train_source'])
        if warmup_source['split'] not in ('train', 'val') or file_sha(warmup_source['input_path']) != warmup_source['source_sha256']:
            raise ValueError('Warmup non-test source identity mismatch')
        rgb, pts, _ = decode_video(warmup_source['input_path'], 2000)
        frames = preprocess_rgb(rgb)
        warmup_start = time.perf_counter()
        infer_windows(model, frames, pts, list(range(min(4, (len(pts)-16)//2+1))), 16, 2, 2)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_start
        del rgb, frames
        order = np.random.default_rng(1701).permutation(list(RECIPES)).tolist()
        observations, diagnostics = [], {}
        max_feature_error = max_activity_error = 0.0
        for number, key in enumerate(ids):
            source = indexed[key]
            if Path(source['input_path']).stat().st_size != source['source_bytes'] or file_sha(source['input_path']) != source['source_sha256']:
                raise ValueError('Held-out source identity changed after audit')
            shift = number % len(order)
            sequence = order[shift:] + order[:shift]
            native_outputs = {}
            for ordinal, method in enumerate(sequence):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rgb, pts, _ = decode_video(source['input_path'], 2000)
                if pts != videos[key]['pts_s']:
                    raise ValueError('Held-out native timeline changed')
                t1 = time.perf_counter()
                frames = preprocess_rgb(rgb)
                del rgb
                t2 = time.perf_counter()
                selection = selected_indices(method, videos[key]['windows'], source['source_sha256'])
                outputs, plan = infer_windows(model, frames, pts, selection, 16, 2, 2)
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                features, activity = outputs['features'].numpy(), outputs['cls_active'].numpy()
                vector = pooled_vector(features, activity, 'full_activity', list(range(len(selection))))
                t4 = time.perf_counter()
                ranked = search(vector, galleries[method], 100)
                t5 = time.perf_counter()
                pooled[method][number] = vector
                for seed, rows in ranked.items():
                    rankings[f'{method}_s{seed}'][number] = rows
                observations.append(dict(sample_id=key, method=method, ordinal=ordinal, source_windows=videos[key]['windows'],
                    encoded_windows=len(selection), native_window_indices=selection,
                    decode_seconds=t1-t0, preprocess_seconds=t2-t1, encoder_wrapper_seconds=t3-t2,
                    pool_seconds=t4-t3, three_gallery_search_seconds=t5-t4, end_to_end_seconds=t5-t0))
                native_outputs[method] = (features, activity, np.array([p['center_s'] for p in plan]))
                if method == 'full_activity':
                    mean = pooled_vector(features, activity, 'full_mean', list(range(len(selection))))
                    pooled['full_mean'][number] = mean
                    for seed, rows in search(mean, galleries['full_mean'], 100).items():
                        rankings[f'full_mean_s{seed}'][number] = rows
                    diagnostics[key] = dict(source_windows=len(selection), frames=len(pts), **activity_proxy(activity))
                del frames, outputs
            # No per-query score/label used for policy decisions or progress.
            dense_x, dense_a, dense_t = native_outputs['full_activity']
            for method in RECIPES[1:]:
                selection = selected_indices(method, len(dense_t), source['source_sha256'])
                x, a, centers = native_outputs[method]
                if (not np.array_equal(centers, dense_t[selection])
                        or not np.allclose(x, dense_x[selection], atol=1e-5, rtol=1e-4)
                        or not np.allclose(a, dense_a[selection], atol=1e-5, rtol=1e-4)):
                    raise ValueError('Actual sparse/dense window output parity failed')
                max_feature_error = max(max_feature_error, float(np.abs(x-dense_x[selection]).max()))
                max_activity_error = max(max_activity_error, float(np.abs(a-dense_a[selection]).max()))
            del native_outputs
            if (number + 1) % 25 == 0 or number + 1 == len(ids):
                run.save('progress.json', dict(completed_queries=number+1, expected_queries=len(ids),
                    measured_pipeline_seconds=sum(r['end_to_end_seconds'] for r in observations), scored=False))
                print(json.dumps(dict(completed_queries=number+1, expected_queries=len(ids))), flush=True)
        np.savez(run.output / 'rankings.npz', **rankings)
        np.savez(pooled_dir / 'pooled.npz', **pooled)
        run.save('pooled_manifest.json', dict(path=str(pooled_dir / 'pooled.npz'), sha256=file_sha(pooled_dir / 'pooled.npz'),
            bytes=(pooled_dir / 'pooled.npz').stat().st_size, query_ids=ids, methods=list(METHODS), dimension=768))
        run.save('schema.json', dict(query_ids=ids, labels=[indexed[k]['class_id'] for k in ids],
            participant_ids=[indexed[k]['participant_id'] for k in ids], class_ids=list(range(100)), split='test',
            class_glosses=cohort['classes'], test_manifest_sha256=file_sha(args.prepared_run / 'test_manifest.json'),
            prepared_run=args.prepared_run.name, freeze_sha256=file_sha(args.freeze)))
        run.save('timing.json', dict(observations=observations, initial_policy_order=order,
            repeats_per_query=1, counterbalance='seed1701 permutation then cyclic rotation by sorted query ordinal',
            gallery_setup_seconds=gallery_setup_seconds, model_setup_seconds=model_setup_seconds, warmup_seconds=warmup_seconds,
            excluded='model/gallery setup, source SHA checks, parity, full_mean secondary pooling/search and persistence; OS cache uncontrolled',
            boundary='independent full decode/preprocess, actual selected encoder, activity pooling/L2 and exact three-gallery search per query/recipe',
            actual_pre_encoder_selection=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), GPU=observation))
        run.save('diagnostics.json', diagnostics)
        run.save('parity.json', dict(passed=True, N=len(ids), feature_atol=1e-5, rtol=1e-4,
            max_feature_abs_error=max_feature_error, max_activity_abs_error=max_activity_error,
            upstream_python_sha256=upstream_hashes, model_params=params, all_queries_checked=True))
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in
            ('rankings.npz', 'schema.json', 'timing.json', 'diagnostics.json', 'parity.json', 'pooled_manifest.json', 'preflight.json')}


if __name__ == '__main__':
    main()
