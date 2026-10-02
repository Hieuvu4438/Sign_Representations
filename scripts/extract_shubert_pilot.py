"""Complete a bounded native raw-video smoke using CPU preprocessing and encoders."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Hide GPUs for the default CPU smoke before importing any framework.
early = argparse.ArgumentParser(add_help=False)
early.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0'])
early_args, _ = early.parse_known_args()
if early_args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            command = (folder / 'cmdline').read_bytes()
        except OSError:
            continue
        if b'python' in command and (b'scripts/run_probe.py' in command or b'scripts/extract_features.py' in command) and b'/bin/bash' not in command:
            raise ValueError('A feature/probe worker is live; wait before initializing CUDA')

import torch
if early_args.device == 'cuda:0':
    # In this borrowed PyTorch 2.1.1/decord 0.6 runtime, importing decord before
    # first CUDA initialization reproducibly segfaults. CUDA-first initialization
    # passed isolated allocation/matmul checks; no upstream model is modified.
    torch.cuda.init()

import decord
import numpy as np
import yaml
from torchvision import transforms

from _common import ROOT
from signrepr.io import sha256, write_json
from signrepr.shubert import NativeSHuBERT, load_native_dino


def dino_features(model, path, batch_size=16):
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), width=224, height=224)
    transform = transforms.Compose([transforms.ToTensor(),
                 transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    features = []
    with torch.inference_mode():
        for start in range(0, len(reader), batch_size):
            frames = reader.get_batch(range(start, min(start + batch_size, len(reader)))).asnumpy()
            values = torch.stack([transform(frame)[:3] for frame in frames]).to(next(model.parameters()).device)
            features.append(model(values).cpu().numpy())
    result = np.concatenate(features)
    if result.shape != (len(reader), 384) or not np.isfinite(result).all():
        raise ValueError('Unexpected native DINO feature shape/values')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preprocessing-report', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/shubert.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0'])
    args = parser.parse_args()
    if args.device == 'cuda:0':
        # The common suite does not use this adapter's lock, so verify its actual
        # process commands rather than assuming a completed state means idle.
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit() or int(folder.name) == os.getpid():
                continue
            try:
                command = (folder / 'cmdline').read_bytes()
            except (OSError, PermissionError):
                continue
            if (b'python' in command and (b'scripts/run_probe.py' in command or b'scripts/extract_features.py' in command)
                    and b'/bin/bash' not in command):
                raise ValueError('Another feature/probe worker is live; native GPU smoke must wait')
    report = json.loads(args.preprocessing_report.read_text())
    if report['status'] != 'PASS_NATIVE_PREPROCESSING' or not 1 <= report['samples'] <= 20:
        raise ValueError('Native preprocessing must pass the bounded train pilot')
    if args.output.exists():
        raise ValueError('Preserve previous native feature smoke attempts')
    args.output.mkdir(parents=True)
    begun = time.perf_counter()
    implementation_hash = sha256(Path(__file__))
    adapter_hash = sha256(ROOT / 'src/signrepr/shubert.py')
    write_json(args.output / 'status.json', {'status': 'RUNNING', 'device': args.device, 'pid': os.getpid()})
    try:
        config = yaml.safe_load(args.config.read_text())
        if sha256(args.config) != report['config_sha256']:
            raise ValueError('Native preprocessing config changed')
        encoder = NativeSHuBERT(ROOT, config, device=args.device)
        print(json.dumps({'stage': 'ENCODER_STRICT_LOAD_PASS', 'device': args.device}), flush=True)
        face, face_audit = load_native_dino(ROOT, config, 'W03', device=args.device)
        print(json.dumps({'stage': 'FACE_DINO_STRICT_LOAD_PASS', 'device': args.device}), flush=True)
        hand, hand_audit = load_native_dino(ROOT, config, 'W04', device=args.device)
        print(json.dumps({'stage': 'HAND_DINO_STRICT_LOAD_PASS', 'device': args.device}), flush=True)
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        inference_begun = time.perf_counter()
        hand_file = ROOT / config['upstream_path'] / 'dataset/crop_hands.py'
        spec = importlib.util.spec_from_file_location('native_hand_selection', hand_file)
        selection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selection)
        entries = []
        for item in report['results']:
            streams = {}
            for name, path_text in item['streams'].items():
                path = Path(path_text)
                if sha256(path) != item['stream_sha256'][name]:
                    raise ValueError('Native intermediate checksum mismatch')
                streams[name] = (np.load(path, allow_pickle=False).astype(np.float32) if name == 'body_posture'
                                 else dino_features(face if name == 'face' else hand, path))
                print(json.dumps({'stage': 'STREAM_FEATURES_PASS', 'sample_id': item['sample_id'], 'stream': name}), flush=True)
            all_layers, repeated = encoder.forward_all_layers(streams), encoder.forward_all_layers(streams)
            a = all_layers[-1]
            difference = float(np.abs(all_layers - repeated).max())
            if difference > config['deterministic_tolerance']:
                raise ValueError('Native raw-video eval is nondeterministic')
            # Model input retains the author carry-forward/black-frame policy.
            # The observation mask separately records which streams were found.
            body_path = Path(item['streams']['body_posture'])
            landmarks = json.loads(body_path.with_suffix('.json').read_text())
            observed = np.zeros((len(a), 4), dtype=bool)
            for index in range(len(a)):
                frame = landmarks[str(index)]
                if frame is None or frame.get('pose_landmarks') is None:
                    continue
                observed[index, 3] = True
                observed[index, 0] = bool(frame.get('face_landmarks'))
                left, right = selection.select_hands(frame['pose_landmarks'][0], frame.get('hand_landmarks'), (1, 1, 3))
                observed[index, 1:3] = [left is not None, right is not None]
            # Imputed frames remain in native contextual inference, but are
            # excluded from downstream pooling/loss. No fabricated observations.
            valid = observed.all(axis=1)
            timestamps = np.stack([np.arange(len(a)) / item['fps'], (np.arange(len(a)) + 1) / item['fps']], axis=1)
            cache_material = {'sample_id': item['sample_id'], 'source_sha256': item['source_sha256'],
                              'config_sha256': sha256(args.config), 'preprocessing_report_sha256': sha256(args.preprocessing_report),
                              'script_sha256': implementation_hash, 'adapter_sha256': adapter_hash,
                              'checkpoint_sha256': config['checkpoint_sha256'], 'upstream_commit': config['upstream_commit'],
                              'dino_commit': config['dino_commit'], 'dino_sha256': [face_audit['checkpoint_sha256'], hand_audit['checkpoint_sha256']],
                              'pool_validity_policy': 'all_four_streams_observed', 'device': args.device}
            cache_key = hashlib.sha256(json.dumps(cache_material, sort_keys=True).encode()).hexdigest()
            output = args.output / (cache_key + '.npz')
            temporary = output.with_suffix('.npz.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, embeddings=a, window_start_sec=timestamps[:, 0],
                                    window_end_sec=timestamps[:, 1], center_sec=timestamps.mean(axis=1), valid_mask=valid,
                                    valid_frame_mask=valid[:, None], stream_observed_mask=observed,
                                    layer_embeddings=all_layers, embeddings_layer_average=all_layers.mean(axis=0))
                stream.flush()
                os.fsync(stream.fileno())
            with np.load(temporary, allow_pickle=False) as check:
                if not np.isfinite(check['layer_embeddings']).all() or not np.array_equal(check['valid_mask'], check['stream_observed_mask'].all(axis=1)):
                    raise ValueError('Native shard layer/mask integrity failure')
            os.replace(temporary, output)
            entry = {'sample_id': item['sample_id'], 'status': 'PASS', 'shard': str(output),
                     'shard_sha256': sha256(output), 'shape': list(a.shape),
                     'layer_shape': list(all_layers.shape), 'cache_key': cache_key, 'cache_material': cache_material,
                     'determinism_max_abs_difference': difference, 'fully_observed_frames': int(observed.all(axis=1).sum()),
                     'total_frames': len(a), 'source_sha256': item['source_sha256']}
            write_json(output.with_suffix('.json'), entry)
            entries.append(entry)
            print(json.dumps(entry), flush=True)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_begun
        summary = {'status': 'PASS_NATIVE_RAW_VIDEO_SMOKE', 'samples': len(entries), 'results': entries,
                   'device': args.device, 'elapsed_seconds': time.perf_counter() - begun,
                   'feature_smoke_seconds_including_repeat_check': inference_seconds,
                   'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                   'encoder_load_audit': encoder.audit, 'dino_load_audits': [face_audit, hand_audit],
                   'preprocessing_report_sha256': sha256(args.preprocessing_report), 'config_sha256': sha256(args.config),
                   'implementation_sha256': implementation_hash,
                   'adapter_implementation_sha256': adapter_hash,
                   'pool_validity_policy': 'Only frames with all four streams observed enter pooling/loss; imputed frames remain in native context, with observation masks preserved.',
                   'feature_unit': 'Native final FFN branch per original video frame, whole-clip context',
                   'limitations': ['Train pilot only; no native lexical/grammar accuracy measured.',
                                   'Pinned author preprocessing retains input FPS; downstream paper recipe unavailable.',
                                   'Missing landmarks use native carry-forward or black crops; downstream pooling/loss excludes frames lacking any stream.',
                                   'Feature time includes repeated encoder determinism checks, intermediate hashing, decoding and writing; preprocessing cost is reported separately.']}
        write_json(args.output / 'report.json', summary)
        write_json(args.output / 'status.json', {'status': summary['status']})
    except Exception as error:
        write_json(args.output / 'status.json', {'status': 'FAIL', 'failure_reason': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
