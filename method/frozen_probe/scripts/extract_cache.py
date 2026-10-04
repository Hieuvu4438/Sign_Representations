"""Full locked-cohort extraction into bounded RAM after GPU parity/cost gate."""
import argparse
import json
import importlib.metadata
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheWriter, atomic_json, expected_centers, file_sha, sample_map
from budget import inspect_budget
from resources import configure_gpu, gpu_preflight, scratch_preflight
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--smoke-run', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise ValueError('Partial extraction resume is unsupported; audit partial artifacts before registering a retry')
    config = json.loads(args.config.read_text())
    smoke = json.loads((args.smoke_run / 'report.json').read_text())
    smoke_status = json.loads((args.smoke_run / 'status.json').read_text())
    identity = config['cache_identity']
    if (smoke_status['status'] != 'completed' or not smoke['passed']
            or smoke_status['protocol'] != config.get('reference_smoke_protocol', config['protocol'])
            or smoke_status['output_sha256'] != file_sha(args.smoke_run / 'report.json')
            or smoke_status['config_sha256'] != config['smoke_config_sha256']
            or smoke['run_id'] != smoke_status['run_id']
            or smoke['input_hashes']['checkpoint'] != identity['checkpoint_sha256']
            or smoke['input_hashes']['video_audit'] != identity['audit_sha256']
            or smoke['packages'] != identity['packages']
            or smoke['code_hashes']['signrep_adapter.py'] != identity['preprocessing_code_sha256']
            or smoke['code_hashes']['temporal.py'] != identity['temporal_code_sha256']):
        raise ValueError('GPU parity/cost gate is invalid or feature recipe changed')
    budget = inspect_budget([REPO / p for p in config['budget_runtime_roots']], config['H_id'],
                            config['project_gpu_ceiling_seconds'], config['H001_gpu_ceiling_seconds'])
    smoke_path = str((args.smoke_run / 'status.json').resolve())
    if budget['status_sha256'].get(smoke_path) != file_sha(args.smoke_run / 'status.json'):
        raise ValueError('Smoke status is not in the owned cumulative budget ledger')
    remaining = budget['remaining_seconds']
    if smoke['estimate_full_cache_wall_seconds'] > remaining:
        raise RuntimeError('Full cohort exceeds remaining registered H001 GPU budget')
    paths = {k: REPO / config[k] for k in ('cohort_manifest', 'video_audit', 'checkpoint')}
    for key, identity_key in (('cohort_manifest', 'cohort_sha256'), ('video_audit', 'audit_sha256'),
                              ('checkpoint', 'checkpoint_sha256')):
        if file_sha(paths[key]) != identity[identity_key]: raise ValueError(f'{key} identity changed')
    if (file_sha(ROOT / 'signrep_adapter.py') != identity['preprocessing_code_sha256']
            or file_sha(ROOT / 'temporal.py') != identity['temporal_code_sha256']):
        raise ValueError('Feature code changed')
    cohort = json.loads(paths['cohort_manifest'].read_text())
    audit = json.loads(paths['video_audit'].read_text())
    indexed = sample_map(cohort)
    if len(indexed) != 1691 or len(cohort['classes']) != 100 or not audit['passed']:
        raise ValueError('Full candidate cohort is required')
    videos = {s['sample_id']: s for s in audit['samples']}
    total_windows = sum(len(expected_centers(v, identity['contract'])) for v in videos.values())
    if total_windows != 61339: raise ValueError('Window count changed')
    planned_bytes = total_windows * (768 * 4 + 2 * 4 + 8) + len(indexed) * 4096 + (4 << 20)
    scratch = scratch_preflight('/dev/shm', planned_bytes, config['cache_limit_bytes'])
    observation = gpu_preflight(config['min_free_mib'], config['max_gpu_utilization'])
    # Scratch name derives from the run ID; no source/gloss names in paths.
    name = __import__('hashlib').sha256(args.run_id.encode()).hexdigest()[:20]
    cache_dir = Path('/dev/shm') / ('signrep_frozen_probe_' + name)
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import cv2
        import numpy as np
        import torch
        from signrep_adapter import load_model, decode_video, preprocess_rgb, infer_windows
        if {p: importlib.metadata.version(p) for p in identity['packages']} != identity['packages']:
            raise ValueError('Inference environment changed since GPU smoke')
        run.save('resource_preflight.json', dict(GPU=observation, scratch=scratch, cache_dir=str(cache_dir),
                 remaining_H001_gpu_seconds=remaining, smoke_run=str(args.smoke_run),
                 smoke_report_sha256=file_sha(args.smoke_run / 'report.json')))
        run.start_gpu(budget)
        device = configure_gpu(observation)
        cv2.setNumThreads(1)
        torch.manual_seed(args.seed)
        upstream = REPO / 'third_party/SignRep'
        hashes = {str(p.relative_to(upstream)): file_sha(p) for p in sorted(upstream.rglob('*.py'))}
        if hashes != smoke['upstream_python_sha256']: raise ValueError('Upstream changed since GPU smoke')
        model, params = load_model(paths['checkpoint'], device)
        writer = CacheWriter(cache_dir, identity, cohort, audit, config['cache_limit_bytes'])
        # Lightweight durable progress copy makes RAM cache provenance reviewable.
        run.save('cache_manifest.json', writer.manifest | {'storage_path': str(cache_dir)})
        for count, key in enumerate(sorted(indexed), 1):
            sample = indexed[key]
            source = Path(sample['input_path'])
            if source.stat().st_size != sample['source_bytes'] or file_sha(source) != sample['source_sha256']:
                raise ValueError('Selected source changed')
            rgb, pts, _ = decode_video(source, config['max_frames_per_video'])
            if pts != videos[key]['pts_s']: raise ValueError('Native PTS changed')
            transformed = preprocess_rgb(rgb)
            del rgb
            outputs, plan = infer_windows(model, transformed, pts, None, 16, 2, config['batch_size'])
            centers = np.array([p['center_s'] for p in plan], dtype=np.float64)
            writer.write(key, outputs['features'].numpy(), outputs['cls_active'].numpy(), centers)
            del outputs, transformed
            run.save('cache_manifest.json', writer.manifest | {'storage_path': str(cache_dir)})
            if count % 50 == 0:
                # Dynamic reserve check includes only remaining payload, never deletes other scratch.
                scratch_preflight('/dev/shm', max(1, planned_bytes - writer.bytes), config['cache_limit_bytes'])
                print(json.dumps({'completed_sources': count, 'expected': len(indexed), 'payload_bytes': writer.bytes}), flush=True)
        complete = writer.complete()
        run.save('cache_manifest.json', complete | {'storage_path': str(cache_dir)})
        run.save('report.json', dict(run_id=args.run_id, passed=True, source_count=len(indexed), windows=total_windows,
                 test_access='none', cache_manifest=str(cache_dir / 'manifest.json'),
                 cache_manifest_sha256=file_sha(cache_dir / 'manifest.json'), payload_bytes=writer.bytes,
                 checkpoint_sha256=identity['checkpoint_sha256'], upstream_python_sha256=hashes,
                 code_hashes=run.code_hashes, model_params=params,
                 packages={p: __import__('importlib.metadata', fromlist=['version']).version(p)
                           for p in ('torch', 'timm', 'albumentations', 'numpy')},
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        run.status['output_sha256'] = file_sha(run.output / 'report.json')


if __name__ == '__main__': main()
