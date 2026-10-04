"""Five-sample FP32 parity and full-window cost estimate; no recognition scoring."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from budget import inspect_budget
from resources import configure_gpu, gpu_preflight
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    budget = inspect_budget([REPO / p for p in config['budget_runtime_roots']], config['H_id'],
                            config['project_gpu_ceiling_seconds'], config['H001_gpu_ceiling_seconds'])
    observation = gpu_preflight(config['min_free_mib'], config['max_gpu_utilization'])
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import cv2
        import numpy as np
        import torch
        from signrep_adapter import load_model, decode_video, preprocess_rgb, infer_windows
        from temporal import window_plan
        run.start_gpu(budget)
        device = configure_gpu(observation)
        cv2.setNumThreads(1)
        torch.manual_seed(args.seed)
        paths = {name: REPO / config[name] for name in ('checkpoint', 'cpu_report', 'cpu_features', 'video_audit')}
        for name, path in paths.items():
            if file_sha(path) != config[name + '_sha256']: raise ValueError(f'{name} hash changed')
        upstream = REPO / 'third_party/SignRep'
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip() != config['upstream_commit']:
            raise ValueError('Pinned upstream commit changed')
        packages = {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'timm', 'albumentations')}
        if packages != config['packages']: raise ValueError('Pinned inference environment changed')
        upstream_hashes = {str(p.relative_to(upstream)): file_sha(p) for p in sorted(upstream.rglob('*.py'))}
        reference = json.loads(paths['cpu_report'].read_text())
        if not reference['passed'] or upstream_hashes != reference['upstream_python_sha256']:
            raise ValueError('CPU reference/upstream changed')
        model, params = load_model(paths['checkpoint'], device)
        report = dict(run_id=args.run_id, scope='GPU numerical parity and representative cost; no task score',
                      GPU=observation, device=device, dtype='float32', TF32=False, batch_size=config['batch_size'],
                      deterministic_algorithms=False, determinism_limit='CUDA kernels not certified bitwise deterministic',
                      packages=packages,
                      code_hashes=run.code_hashes, upstream_python_sha256=upstream_hashes, model_params=params,
                      input_hashes={name: file_sha(path) for name, path in paths.items()}, samples=[])
        with np.load(paths['cpu_features'], allow_pickle=False) as arrays:
            for number, sample in enumerate(reference['samples']):
                if sample['split'] not in ('train', 'val'): raise ValueError('Test sample forbidden')
                source = Path(config['video_root']) / sample['sample_id']
                if file_sha(source) != sample['source_sha256']: raise ValueError('Smoke video changed')
                start = time.monotonic()
                rgb, pts, _ = decode_video(source, config['max_frames_per_video'])
                if pts != sample['native_pts_s']: raise ValueError('Smoke timeline changed')
                frames = preprocess_rgb(rgb)
                preprocessing_seconds = time.monotonic() - start
                del rgb
                selected, _ = infer_windows(model, frames, pts, sample['selected_window_indices'],
                                             16, 2, config['batch_size'])
                parity = {}
                for key, actual in selected.items():
                    expected = arrays[f'sample_{number}_{key}']
                    value = actual.numpy()
                    passed = value.shape == expected.shape and np.allclose(value, expected,
                        atol=config['feature_atol'], rtol=config['feature_rtol'])
                    parity[key] = dict(max_abs_difference=float(np.abs(value - expected).max()), passed=bool(passed))
                    if not passed: raise ValueError(f'GPU parity failed: {key} {parity[key]}')
                torch.cuda.synchronize()
                start = time.monotonic()
                complete, plan = infer_windows(model, frames, pts, None, 16, 2, config['batch_size'])
                torch.cuda.synchronize()
                encoder_seconds = time.monotonic() - start
                report['samples'].append(dict(sample_id=sample['sample_id'], source_sha256=sample['source_sha256'],
                    split=sample['split'], frames=len(pts), windows=len(plan), parity=parity,
                    preprocessing_seconds=preprocessing_seconds, encoder_seconds=encoder_seconds))
                del complete, selected, frames
                run.save('report.json', report)
                print(json.dumps({'sample': sample['sample_id'], 'parity': 'passed', 'full_windows': len(plan)}), flush=True)
        audit = json.loads(paths['video_audit'].read_text())
        total_frames = sum(s['frames'] for s in audit['samples'])
        total_windows = sum(s['windows'] for s in audit['samples'])
        worst_encoder = max(s['encoder_seconds'] / s['windows'] for s in report['samples'])
        worst_preproc = max(s['preprocessing_seconds'] / s['frames'] for s in report['samples'])
        estimate = config['estimate_safety_multiplier'] * (worst_encoder * total_windows +
                    worst_preproc * total_frames + config['hash_and_write_reserve_seconds'])
        report.update(passed=True, estimate_full_cache_wall_seconds=estimate,
                      estimate_is_provisional=True, estimate_rule='max per-window/per-frame rates across five fixed clips',
                      total_windows=total_windows, total_frames=total_frames,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      budget_before_launch=budget,
                      extraction_within_H001_budget=estimate + (time.monotonic() - run.gpu_started) <= budget['remaining_seconds'])
        run.save('report.json', report)
        run.status['output_sha256'] = file_sha(run.output / 'report.json')


if __name__ == '__main__': main()
