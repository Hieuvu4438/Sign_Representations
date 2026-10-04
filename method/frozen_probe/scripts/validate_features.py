"""Bounded CPU parity against SignRep's upstream preprocessing/forward recipe."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_signrep import sha

REPO = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config['device'], config['dtype'], config['window_length'], config['stride'], config['selection']) != (
            'cpu', 'float32', 16, 2, 'first_and_last_valid_window_per_video'):
        raise ValueError('This registered parity run supports only its fixed CPU contract')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'run_id': args.run_id, 'protocol': config['protocol'], 'seed': args.seed,
              'status': 'running', 'pid': os.getpid(), 'host': platform.node(),
              'start_utc': datetime.now(timezone.utc).isoformat(), 'GPU_hours': 0,
              'command': [sys.executable]+sys.argv, 'config_sha256': sha(args.config)}

    def save(name, value):
        temporary = args.output_dir/(name+'.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(args.output_dir/name)

    def timeout_handler(*_):
        raise TimeoutError('Registered CPU smoke time budget exhausted')

    save('status.json', status)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config['timeout_seconds'])
    try:
        import cv2
        import numpy as np
        import torch
        from signrep_adapter import load_model, decode_video, preprocess_rgb, infer_windows, MEAN, STD
        from temporal import window_plan
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        package_versions = {p: importlib.metadata.version(p) for p in (
            'torch', 'timm', 'albumentations', 'numpy')}
        package_versions['cv2'] = cv2.__version__
        manifest_path = REPO/config['sample_manifest']
        manifest = json.loads(manifest_path.read_text())
        samples = manifest['samples']
        if len(samples) != 5 or any(s['split'] not in ('train', 'val') for s in samples):
            raise ValueError('Only the registered five train/dev samples are permitted')
        upstream = REPO/'third_party/SignRep'
        source_hashes = {str(p.relative_to(upstream)): sha(p) for p in sorted(upstream.rglob('*.py'))}
        code_files = [Path(__file__), REPO/'method/frozen_probe/signrep_adapter.py',
                      REPO/'method/frozen_probe/temporal.py', REPO/'method/frozen_probe/scripts/audit_signrep.py']
        code_hashes = {str(p.relative_to(REPO)): sha(p) for p in code_files}
        snapshots = args.output_dir/'source_snapshot'
        snapshots.mkdir()
        for path in code_files:
            shutil.copyfile(path, snapshots/path.name)
        shutil.copyfile(args.config, args.output_dir/'config.json')
        save('environment.json', {'executable': sys.executable, 'python': sys.version,
                                 'packages': package_versions, 'device': 'cpu', 'dtype': 'float32',
                                 'threads': 1, 'deterministic_algorithms': True,
                                 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()})
        checkpoint = REPO/config['checkpoint']
        load_start = time.monotonic()
        model, params = load_model(checkpoint)
        load_seconds = time.monotonic()-load_start
        from augmentation.video.base_video_aug import Transformation

        def compare(actual, reference, atol, rtol):
            if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
                raise ValueError('Nonfinite or mismatched comparison tensors')
            result = {'shape': list(actual.shape), 'max_abs_difference': (actual-reference).abs().max().item(),
                      'atol': atol, 'rtol': rtol,
                      'passed': bool(torch.allclose(actual, reference, atol=atol, rtol=rtol))}
            if not result['passed']:
                raise ValueError(f'Parity failed: {result}')
            return result

        colors = np.array([0, 127, 255], dtype=np.uint8)
        constant = np.broadcast_to(colors, (19, 23, 3)).copy()
        synthetic = [constant, np.arange(19*23*3, dtype=np.uint16).reshape(19,23,3).astype(np.uint8)]
        synthetic_actual = preprocess_rgb(synthetic)
        analytic = torch.tensor((colors.astype(np.float64)/255-np.array(MEAN))/np.array(STD), dtype=torch.float32)
        synthetic_oracle = compare(synthetic_actual[0], analytic[:, None, None].expand(3,224,224),
                                   config['preprocessing_atol'], config['preprocessing_rtol'])
        synthetic_reference = compare(synthetic_actual, Transformation().aug_video(synthetic),
                                      config['preprocessing_atol'], config['preprocessing_rtol'])
        report = {'run_id': args.run_id, 'scope': 'CPU preprocessing and ten selected-window batch/serial parity; no benchmark',
                  'config_sha256': sha(args.config), 'manifest_sha256': sha(manifest_path),
                  'checkpoint_sha256': sha(checkpoint), 'code_hashes': code_hashes,
                  'upstream_python_sha256': source_hashes, 'model_params': params,
                  'environment': package_versions, 'strict_load_passed': True,
                  'load_seconds': load_seconds, 'synthetic_color_oracle': synthetic_oracle,
                  'synthetic_upstream_parity': synthetic_reference, 'samples': [],
                  'activity_contract': 'two raw outputs retained; no probability or pooling interpretation',
                  'end_boundary_contract': 'next native PTS; last-frame end estimated with median PTS delta'}
        arrays = {}
        for sample_number, sample in enumerate(samples):
            sample_start = time.monotonic()
            path = Path(config['video_root'])/sample['sample_id']
            if sha(path) != sample['source_sha256']:
                raise ValueError('Video source hash changed')
            rgb, pts, command = decode_video(path, config['max_frames_per_video'])
            if pts != sample['pts_seconds']:
                raise ValueError('Native PTS changed since DATA001_r1')
            transformed = preprocess_rgb(rgb)
            reference_frames = Transformation().aug_video(rgb)
            preproc = compare(transformed, reference_frames,
                              config['preprocessing_atol'], config['preprocessing_rtol'])
            plan = window_plan(pts, config['window_length'], config['stride'])
            if len(plan) != sample['reference_window16_stride2_count']:
                raise ValueError('Window coverage differs from audited inference recipe')
            indices = [0, len(plan)-1] if len(plan) > 1 else [0]
            forward_start = time.monotonic()
            actual, selected_plan = infer_windows(model, transformed, pts, indices,
                                                 config['window_length'], config['stride'], config['batch_size'])
            batch_seconds = time.monotonic()-forward_start
            # Separate literal single-window reference loop, including its start enumeration.
            reference_starts = [i for i in range(0, len(reference_frames), 2) if len(reference_frames[i:i+16]) == 16]
            if reference_starts != [w['start_frame'] for w in plan]:
                raise ValueError('Window plan differs from literal upstream loop')
            reference = {key: [] for key in actual}
            forward_start = time.monotonic()
            with torch.no_grad():
                for index in indices:
                    start = reference_starts[index]
                    result = model(reference_frames[start:start+16].unsqueeze(0).permute(0,2,1,3,4))
                    for key in reference:
                        reference[key].append(result[key].cpu())
            serial_seconds = time.monotonic()-forward_start
            parity = {key: compare(value, torch.cat(reference[key]), config['feature_atol'], config['feature_rtol'])
                      for key, value in actual.items()}
            for key, value in actual.items():
                arrays[f'sample_{sample_number}_{key}'] = value.numpy()
            report['samples'].append({'sample_id': sample['sample_id'], 'split': sample['split'],
                                      'source_sha256': sample['source_sha256'], 'native_pts_s': pts,
                                      'decoded_frames': len(rgb), 'total_valid_windows': len(plan),
                                      'selected_window_indices': indices, 'selected_windows': selected_plan,
                                      'ffprobe_command': command, 'preprocessing': preproc,
                                      'output_parity': parity, 'batch_forward_seconds': batch_seconds,
                                      'serial_forward_seconds': serial_seconds,
                                      'sample_wall_seconds': time.monotonic()-sample_start})
            save('parity.json', report)
            print(json.dumps({'sample': sample['sample_id'], 'parity': 'passed',
                              'batch_forward_seconds': batch_seconds}), flush=True)
        np.savez_compressed(args.output_dir/'selected_features.npz', **arrays)
        report['passed'] = True
        report['feature_artifact_sha256'] = sha(args.output_dir/'selected_features.npz')
        save('parity.json', report)
        status.update(status='completed', exit_code=0, code_hashes=code_hashes,
                      output_sha256=sha(args.output_dir/'parity.json'))
    except Exception as error:
        status.update(status='failed', exit_code=1, error=repr(error))
        (args.output_dir/'error.txt').write_text(traceback.format_exc())
    finally:
        signal.alarm(0)
        status.update(end_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started)
        save('status.json', status)
    print(json.dumps(status), flush=True)
    if status['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
