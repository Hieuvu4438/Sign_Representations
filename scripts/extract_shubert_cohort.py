"""Single GPU native feature worker, per-clip verified cache, optionally waiting on CPU preparation."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
# Reuse the verified CUDA-first import workaround and native DINO transform.
# Its early parser recognizes --device and does not run the pilot main routine.
from extract_shubert_pilot import torch, np, dino_features, NativeSHuBERT, load_native_dino
from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json, write_jsonl
from signrepr.native_cache import verified_preprocessing
from signrepr.timestamps import source_frame_intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/protocol_native_common_v1.json')
    parser.add_argument('--device', default='cuda:0', choices=['cpu', 'cuda:0'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wait-for-preprocessing', action='store_true')
    parser.add_argument('--limit-new-clips', type=int)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol['status'] != 'LOCKED_FOR_NATIVE_PUBLIC_PIPELINE_DIAGNOSTIC':
        raise ValueError('Native feature protocol is not locked')
    manifest = ROOT / protocol['manifest']
    if sha256(manifest) != protocol['manifest_sha256']:
        raise ValueError('Native cohort manifest changed')
    rows = [row for _, row in read_jsonl(manifest)]
    config_path = ROOT / protocol['encoder_config']
    if sha256(config_path) != protocol['encoder_config_sha256']:
        raise ValueError('Native encoder config changed')
    import yaml
    config = yaml.safe_load(config_path.read_text())
    preparation = ROOT / protocol['preprocessing_directory']
    identity = protocol['preprocessing_identity']
    if json.loads((preparation / 'cohort.lock.json').read_text()) != identity:
        raise ValueError('Native preprocessing identity mismatch')
    output = ROOT / protocol['feature_directory']
    if not output.resolve().is_relative_to(ROOT / 'features'):
        raise ValueError('Generated features must stay inside project features/')
    if output.exists() and not args.resume:
        raise ValueError('Existing native features need explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    cache_identity = {'protocol_sha256': sha256(args.protocol), 'manifest_sha256': sha256(manifest),
                      'config_sha256': sha256(config_path), 'implementation_sha256': sha256(Path(__file__)),
                      'pilot_helpers_sha256': sha256(ROOT / 'scripts/extract_shubert_pilot.py'),
                      'adapter_sha256': sha256(ROOT / 'src/signrepr/shubert.py'),
                      'resume_validator_sha256': sha256(ROOT / 'src/signrepr/native_cache.py'),
                      'timestamp_adapter_sha256': sha256(ROOT / 'src/signrepr/timestamps.py'), 'device': args.device}
    with (ROOT / 'features/.native_feature_worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit() or int(folder.name) == os.getpid():
                continue
            try:
                command = (folder / 'cmdline').read_bytes()
            except OSError:
                continue
            if b'python' in command and b'/bin/bash' not in command and any(
                name in command for name in [b'scripts/run_probe.py', b'scripts/extract_features.py',
                                             b'scripts/extract_shubert_pilot.py', b'scripts/extract_shubert_cohort.py']):
                raise ValueError('Another project GPU worker is live')
        cache_lock = output / 'cohort.lock.json'
        if cache_lock.exists():
            if json.loads(cache_lock.read_text()) != cache_identity:
                raise ValueError('Native feature cache implementation/input identity changed')
        else:
            write_json(cache_lock, cache_identity)
        attempts = list(output.glob('attempt_*.json'))
        attempt = output / f'attempt_{len(attempts)+1:04d}.json'
        state = {'status': 'RUNNING', 'pid': os.getpid(), 'samples': len(rows), 'success': 0,
                 'new_clips': 0, 'elapsed_seconds': 0, **cache_identity}
        write_json(output / 'status.json', state)
        entries = []
        try:
            encoder = NativeSHuBERT(ROOT, config, device=args.device)
            face, face_audit = load_native_dino(ROOT, config, 'W03', device=args.device)
            hand, hand_audit = load_native_dino(ROOT, config, 'W04', device=args.device)
            selection_path = ROOT / config['upstream_path'] / 'dataset/crop_hands.py'
            spec = importlib.util.spec_from_file_location('native_cohort_hand_selection', selection_path)
            selection = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(selection)
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            compute_wall_seconds = 0
            for row in rows:
                prep_audit = preparation / hashlib.sha256(row['sample_id'].encode()).hexdigest()[:20] / 'audit.json'
                while not prep_audit.exists():
                    prep_state = json.loads((preparation / 'status.json').read_text())
                    if not args.wait_for_preprocessing or prep_state['status'] != 'RUNNING':
                        raise ValueError('Native input not ready and preprocessing is not running')
                    pid = prep_state['pid']
                    try:
                        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
                    except OSError as error:
                        raise ValueError('Native preprocessing process vanished without terminal state') from error
                    if b'scripts/prepare_shubert_cohort.py' not in command:
                        raise ValueError('Preprocessing PID identity no longer matches')
                    state.update(waiting_for_sample_id=row['sample_id'], elapsed_seconds=time.perf_counter()-begun)
                    write_json(output / 'status.json', state)
                    time.sleep(10)
                item = verified_preprocessing(prep_audit, row, identity)
                material = {**cache_identity, 'sample_id': row['sample_id'],
                            'source_sha256': item['source_sha256'], 'preprocessing_audit_sha256': sha256(prep_audit),
                            'pool_validity_policy': protocol['pool_validity_policy']}
                key = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
                shard_path = output / (key + '.npz')
                metadata_path = shard_path.with_suffix('.json')
                if metadata_path.exists():
                    entry = json.loads(metadata_path.read_text())
                    if entry['cache_material'] != material or sha256(shard_path) != entry['shard_sha256']:
                        raise ValueError('Native shard cache changed or corrupted')
                else:
                    if args.limit_new_clips is not None and state['new_clips'] >= args.limit_new_clips:
                        break
                    import shutil
                    if shutil.disk_usage(output).free < (protocol['resource_policy']['minimum_free_disk_gib'] + 1)*1024**3:
                        raise ValueError('Native feature worker reached reserved disk limit')
                    clip_begun = time.perf_counter()
                    streams = {name: (np.load(path, allow_pickle=False).astype(np.float32) if name == 'body_posture'
                        else dino_features(face if name == 'face' else hand, Path(path)))
                        for name, path in item['streams'].items()}
                    layers = encoder.forward_all_layers(streams)
                    if state['new_clips'] == 0:
                        difference = float(np.abs(layers-encoder.forward_all_layers(streams)).max())
                        if difference > config['deterministic_tolerance']:
                            raise ValueError('Native first-clip repeat differs')
                    else:
                        difference = None
                    observed = np.zeros((layers.shape[1], 4), dtype=bool)
                    landmarks = json.loads(Path(item['landmarks_path']).read_text())
                    for i in range(len(observed)):
                        frame = landmarks[str(i)]
                        if frame is None or not frame.get('pose_landmarks'):
                            continue
                        observed[i, 3] = True
                        observed[i, 0] = bool(frame.get('face_landmarks'))
                        left, right = selection.select_hands(frame['pose_landmarks'][0], frame.get('hand_landmarks'), (1, 1, 3))
                        observed[i, 1:3] = [left is not None, right is not None]
                    valid = observed.all(axis=1)
                    clock, tail_policy = source_frame_intervals(item['source_path'], item['frames'], item['fps'])
                    temporary = shard_path.with_suffix('.npz.partial')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, embeddings=layers[-1], layer_embeddings=layers,
                            embeddings_layer_average=layers.mean(axis=0), window_start_sec=clock[:,0],
                            window_end_sec=clock[:,1], center_sec=clock.mean(axis=1), valid_mask=valid,
                            valid_frame_mask=valid[:,None], stream_observed_mask=observed)
                        stream.flush()
                        os.fsync(stream.fileno())
                    with np.load(temporary, allow_pickle=False) as check:
                        if not np.isfinite(check['layer_embeddings']).all() or not np.array_equal(
                            check['valid_mask'], check['stream_observed_mask'].all(axis=1)):
                            raise ValueError('Native shard mask/layer validation failed')
                    os.replace(temporary, shard_path)
                    elapsed = time.perf_counter() - clip_begun
                    compute_wall_seconds += elapsed
                    entry = {'sample_id': row['sample_id'], 'status': 'SUCCESS', 'shard': str(shard_path),
                             'shard_sha256': sha256(shard_path), 'cache_material': material, 'cache_key': key,
                             'shape': list(layers[-1].shape), 'layer_shape': list(layers.shape),
                             'fully_observed_frames': int(valid.sum()), 'total_frames': len(valid),
                             'determinism_max_abs_difference': difference,
                             'source_clock_tail_policy': tail_policy, 'feature_wall_seconds': elapsed}
                    write_json(metadata_path, entry)
                    state['new_clips'] += 1
                entries.append(entry)
                state.update(success=len(entries), elapsed_seconds=time.perf_counter()-begun,
                             active_feature_wall_seconds=compute_wall_seconds, last_sample_id=row['sample_id'])
                state.pop('waiting_for_sample_id', None)
                write_json(output / 'status.json', state)
                if len(entries) % 20 == 0:
                    write_jsonl(output / 'index.jsonl', entries)
                print(json.dumps({'sample_id': row['sample_id'], 'success': len(entries),
                                  'fully_observed_frames': entry['fully_observed_frames']}), flush=True)
            complete = len(entries) == len(rows)
            state.update(status='PASS' if complete else 'PARTIAL_BOUNDED_VALIDATION',
                         elapsed_seconds=time.perf_counter()-begun)
            write_jsonl(output / 'index.jsonl', entries)
            report = {**state, 'encoder_load_audit': encoder.audit, 'load_audit': encoder.audit,
                      'dino_load_audits': [face_audit, hand_audit],
                      'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                      'zero_valid_clips': sum(entry['fully_observed_frames'] == 0 for entry in entries),
                      'pool_validity_policy': protocol['pool_validity_policy'],
                      'limits': ['Public native raw-video pipeline, not exact downstream paper recipe.',
                                 'One GPU allocation wall time includes waiting for CPU; active_feature_wall_seconds includes decode/I/O, not CUDA compute time.',
                                 'Complete extraction does not imply complete usable readout coverage.']}
            write_json(output / 'extraction_report.json', report)
            write_json(attempt, state)
            write_json(output / 'status.json', state)
        except Exception as error:
            state.update(status='FAIL', elapsed_seconds=time.perf_counter()-begun,
                         failure_reason=f'{type(error).__name__}: {error}')
            write_jsonl(output / 'index.jsonl', entries)
            write_json(attempt, state)
            write_json(output / 'status.json', state)
            raise


if __name__ == '__main__':
    main()
