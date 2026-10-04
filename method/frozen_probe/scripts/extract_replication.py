"""Hash-bound P004 train/dev extraction with the unchanged native encoder."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from budget import inspect_budget
from cache import CacheWriter, expected_centers, file_sha
from replication import check_gate, validate_cohort
from resources import configure_gpu, gpu_preflight, scratch_preflight
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir', 'launch-gate'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise ValueError('No partial extraction resume; retain failure and register retry')
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        gate = check_gate(args.launch_gate, args.config, REPO)
        identity = config['cache_identity']
        paths = {k: REPO / config[k] for k in ('cohort_manifest', 'video_audit', 'checkpoint')}
        for key, identity_key in (('cohort_manifest', 'cohort_sha256'), ('video_audit', 'audit_sha256'),
                                  ('checkpoint', 'checkpoint_sha256')):
            if file_sha(paths[key]) != identity[identity_key]:
                raise ValueError('Registered native cache identity changed')
        cohort, audit = (json.loads(paths[k].read_text()) for k in ('cohort_manifest', 'video_audit'))
        indexed, videos = validate_cohort(cohort, audit, config['expected'], identity['contract'])
        smoke_dir = REPO / config['reference_gpu_smoke']
        smoke = json.loads((smoke_dir / 'report.json').read_text())
        status = json.loads((smoke_dir / 'status.json').read_text())
        if (status['status'] != 'completed' or not smoke['passed']
                or status['output_sha256'] != file_sha(smoke_dir / 'report.json')
                or smoke['input_hashes']['checkpoint'] != identity['checkpoint_sha256']
                or smoke['packages'] != identity['packages']
                or smoke['code_hashes']['signrep_adapter.py'] != identity['preprocessing_code_sha256']
                or smoke['code_hashes']['temporal.py'] != identity['temporal_code_sha256']):
            raise ValueError('Unchanged native recipe requires a valid original parity gate')
        packages = {p: importlib.metadata.version(p) for p in identity['packages']}
        if packages != identity['packages']:
            raise ValueError('Inference environment changed')
        budget = inspect_budget([REPO / p for p in config['budget_runtime_roots']], 'H002', 14400, 7200)
        if config['estimate_cache_seconds'] > budget['remaining_seconds']:
            raise RuntimeError('Registered cache estimate exceeds remaining GPU allowance')
        if budget['status_sha256'].get(str((smoke_dir / 'status.json').resolve())) != file_sha(smoke_dir / 'status.json'):
            raise ValueError('Reference GPU smoke missing from cumulative ledger')
        planned = config['expected']['windows'] * (768 * 4 + 2 * 4 + 8) + len(indexed) * 4096 + (4 << 20)
        scratch = scratch_preflight('/dev/shm', planned, config['cache_limit_bytes'])
        observation = gpu_preflight(config['min_free_mib'], config['max_gpu_utilization'])
        if observation['uuid'] != config['granted_gpu_uuid']:
            raise ValueError('Granted GPU identity mismatch')
        upstream = REPO / 'third_party/SignRep'
        hashes = {str(p.relative_to(upstream)): file_sha(p) for p in sorted(upstream.rglob('*.py'))}
        if hashes != smoke['upstream_python_sha256']:
            raise ValueError('Reference upstream inference source changed')
        cache_dir = Path('/dev/shm') / ('signrep_p004_' + hashlib.sha256(args.run_id.encode()).hexdigest()[:20])
        run.save('resource_preflight.json', dict(GPU=observation, scratch=scratch, cache_dir=str(cache_dir),
                 budget=budget, registration_sha256=file_sha(args.launch_gate),
                 reference_parity_scope='unchanged adapter/checkpoint/packages/upstream; new source audit separately bound'))
        import cv2
        import numpy as np
        import torch
        from signrep_adapter import load_model, decode_video, preprocess_rgb, infer_windows
        run.start_gpu(budget)
        device = configure_gpu(observation)
        cv2.setNumThreads(1)
        torch.manual_seed(args.seed)
        model, params = load_model(paths['checkpoint'], device)
        writer = CacheWriter(cache_dir, identity, cohort, audit, config['cache_limit_bytes'])
        for count, key in enumerate(sorted(indexed), 1):
            sample = indexed[key]
            source = Path(sample['input_path'])
            if source.stat().st_size != sample['source_bytes'] or file_sha(source) != sample['source_sha256']:
                raise ValueError('Selected source changed')
            rgb, pts, _ = decode_video(source, config['max_frames_per_video'])
            if pts != videos[key]['pts_s']:
                raise ValueError('Native PTS changed')
            transformed = preprocess_rgb(rgb)
            del rgb
            outputs, plan = infer_windows(model, transformed, pts, None, 16, 2, config['batch_size'])
            centers = np.array([p['center_s'] for p in plan], dtype=np.float64)
            if not np.array_equal(centers, expected_centers(videos[key], identity['contract'])):
                raise ValueError('Native complete-window centers changed')
            writer.write(key, outputs['features'].numpy(), outputs['cls_active'].numpy(), centers)
            del outputs, transformed
            run.save('cache_manifest.json', writer.manifest | {'storage_path': str(cache_dir)})
            if count % 25 == 0:
                scratch_preflight('/dev/shm', max(1, planned - writer.bytes), config['cache_limit_bytes'])
                print(json.dumps(dict(completed_sources=count, expected=len(indexed), payload_bytes=writer.bytes)), flush=True)
        complete = writer.complete()
        run.save('cache_manifest.json', complete | {'storage_path': str(cache_dir)})
        run.save('report.json', dict(run_id=args.run_id, passed=True, source_count=len(indexed),
                 windows=config['expected']['windows'], test_access='none',
                 cache_manifest=str(cache_dir / 'manifest.json'), cache_manifest_sha256=file_sha(cache_dir / 'manifest.json'),
                 payload_bytes=writer.bytes, code_hashes=run.code_hashes, packages=packages,
                 upstream_python_sha256=hashes, model_params=params,
                 registration_sha256=file_sha(args.launch_gate),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        run.status['output_sha256'] = file_sha(run.output / 'report.json')


if __name__ == '__main__':
    main()
