"""One CPU worker, pinned native preprocessing, per-clip integrity and resumable cohort."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.dont_write_bytecode = True

# The verified pilot is reused without changing its implementation/hash.
from prepare_shubert_native import module, video_info, cv2, mp, np, yaml, python, vision
from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.shubert import verify_source
from signrepr.native_cache import verified_preprocessing


def prepare(row, sample, modules, assets, config, identity):
    source_path = (Path(row['source_root']) / row['relative_path']).resolve()
    original, _ = video_info(source_path)
    if original['frames'] > config['max_frames']:
        raise ValueError('Input exceeds registered whole-context frame cap')
    sample.mkdir()
    problem, clip = sample / 'problems.txt', sample / 'signer.mp4'
    problem.touch()
    modules['crop'].crop_clip(str(source_path), str(problem), str(clip), str(assets['W07']))
    cropped, _ = video_info(clip)
    if original['frames'] != cropped['frames'] or abs(original['fps'] - cropped['fps']) > .05:
        raise ValueError('Signer crop changed frame count or FPS')
    modules['pose'].video_holistic(str(clip), str(problem), sample, sample)
    landmarks = sample / 'signer_pose.json'
    modules['face'].video_holistic(str(clip), str(sample), str(problem), str(sample))
    modules['hands'].video_holistic(str(clip), str(sample), str(problem), str(sample))
    modules['body'].keypoints_to_numpy(str(landmarks), str(sample))
    paths = {'face': sample / 'signer_face.mp4', 'left_hand': sample / 'signer_hand1.mp4',
             'right_hand': sample / 'signer_hand2.mp4', 'body_posture': sample / 'signer_pose.npy'}
    body = np.load(paths['body_posture'], allow_pickle=False)
    if body.shape != (original['frames'], 14) or not np.isfinite(body).all():
        raise ValueError('Native body feature shape/finite check failed')
    for name, path in paths.items():
        if name == 'body_posture':
            continue
        info, _ = video_info(path)
        if info['frames'] != original['frames'] or abs(info['fps'] - original['fps']) > .05:
            raise ValueError('Native cue crop frame alignment mismatch')
    import decord
    reader = decord.VideoReader(str(paths['face']), ctx=decord.cpu(0))
    _, rgb = video_info(paths['face'])
    difference = float(np.abs(reader[0].asnumpy().astype(float) - rgb.astype(float)).max())
    if difference > 3:
        raise ValueError('OpenCV/author-decord RGB disagreement')
    observed = json.loads(landmarks.read_text())
    if set(observed) != {str(i) for i in range(original['frames'])}:
        raise ValueError('Landmark observation indices do not cover every frame')
    return {'sample_id': row['sample_id'], 'status': 'PASS_NATIVE_PREPROCESSING',
            'source_path': str(source_path), 'source_sha256': sha256(source_path),
            'streams': {k: str(v) for k, v in paths.items()}, 'landmarks_path': str(landmarks),
            'stream_sha256': {k: sha256(v) for k, v in {**paths, 'landmarks': landmarks}.items()},
            'frames': original['frames'], 'fps': original['fps'],
            'missing_pose_frames': sum(v is None or not v.get('pose_landmarks') for v in observed.values()),
            'missing_face_frames': sum(v is None or not v.get('face_landmarks') for v in observed.values()),
            'rgb_decoder_max_abs_difference': difference, 'cache_identity': identity,
            'rate_note': 'Pinned author raw-video pipeline retains input FPS; not exact downstream paper reproduction.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/shubert.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit-new-clips', type=int, help='Bound validation; does not change the locked manifest')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / 'features'):
        raise ValueError('Native generated artifacts must stay inside project features/')
    rows = [row for _, row in read_jsonl(args.manifest)]
    if not 1 <= len(rows) <= 5000 or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Cohort size/duplicate ID guard failed')
    if args.limit_new_clips is not None and args.limit_new_clips < 1:
        raise ValueError('Bounded validation must process at least one new clip')
    for row in rows:
        root = Path(row['source_root']).resolve()
        path = (root / row['relative_path']).resolve()
        if not path.is_relative_to(root) or row.get('input_kind') != 'video' or not path.is_file():
            raise ValueError('Missing/escaping/unsupported native video input')
    config = yaml.safe_load(args.config.read_text())
    source = verify_source(ROOT, config['upstream_path'], config['upstream_commit'])
    registry = json.loads((ROOT / 'provenance/assets.json').read_text())['assets']
    assets = {k: ROOT / registry[k]['local_path'] for k in ['W05', 'W06', 'W07']}
    for name, path in assets.items():
        if sha256(path) != registry[name]['expected_sha256']:
            raise ValueError('Unverified native preprocessing weight: ' + name)
    relative = {'crop': 'dataset/clips_bbox.py', 'pose': 'dataset/kpe_mediapipe.py',
                'face': 'dataset/crop_face.py', 'hands': 'dataset/crop_hands.py', 'body': 'features/body_features.py'}
    identity = {'manifest_sha256': sha256(args.manifest), 'config_sha256': sha256(args.config),
                'implementation_sha256': sha256(Path(__file__)),
                'pilot_helpers_sha256': sha256(ROOT / 'scripts/prepare_shubert_native.py'),
                'resume_validator_sha256': sha256(ROOT / 'src/signrepr/native_cache.py'),
                'upstream_commit': config['upstream_commit'],
                'native_source_hashes': {k: sha256(source / v) for k, v in relative.items()},
                'preprocessing_weight_sha256': {k: sha256(v) for k, v in assets.items()},
                'device': 'cpu', 'workers': 1}
    if args.dry_run:
        print(json.dumps({'samples': len(rows), 'output': str(args.output), 'cache_identity': identity}))
        return
    if args.output.exists() and not args.resume:
        raise ValueError('Existing native cohort requires explicit --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_lock = args.output / 'cohort.lock.json'
        if run_lock.exists():
            if json.loads(run_lock.read_text()) != identity:
                raise ValueError('Native cohort implementation/input identity changed')
        else:
            write_json(run_lock, identity)
        previous = [json.loads(p.read_text()) for p in args.output.glob('attempt_*.json')]
        attempt = args.output / f'attempt_{len(previous)+1:04d}.json'
        begun = time.perf_counter()
        state = {'status': 'RUNNING', 'pid': os.getpid(), 'device': 'cpu', 'samples': len(rows),
                 'passed': 0, 'new_clips': 0, 'resume': args.resume, **identity}
        write_json(attempt, state)
        write_json(args.output / 'status.json', state)
        try:
            import torch
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            modules = {k: module(source / v, 'cohort_native_' + k) for k, v in relative.items()}
            face_options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(
                model_asset_path=str(assets['W05']), delegate=python.BaseOptions.Delegate.CPU),
                output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6)
            hand_options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(
                model_asset_path=str(assets['W06']), delegate=python.BaseOptions.Delegate.CPU),
                num_hands=6, min_hand_detection_confidence=.05)
            results = []
            with vision.FaceLandmarker.create_from_options(face_options) as face_detector, \
                 vision.HandLandmarker.create_from_options(hand_options) as hand_detector, \
                 mp.solutions.holistic.Holistic(min_detection_confidence=.1) as holistic:
                pose = modules['pose']
                pose.face_detector, pose.hand_detector, pose.mp_holistic = face_detector, hand_detector, holistic
                for row in rows:
                    sample = args.output / hashlib.sha256(row['sample_id'].encode()).hexdigest()[:20]
                    audit = sample / 'audit.json'
                    if audit.exists():
                        item = verified_preprocessing(audit, row, identity)
                    else:
                        if args.limit_new_clips is not None and state['new_clips'] >= args.limit_new_clips:
                            break
                        free = shutil.disk_usage(args.output).free
                        if free < (config['disk_free_min_gib'] + 1) * 1024**3:
                            raise ValueError('Native preprocessing reached reserved disk limit')
                        if sample.exists():
                            # Incomplete artifacts survive an interrupted process for inspection.
                            sample.rename(sample.with_name(sample.name + '.interrupted.' + str(time.time_ns())))
                        clip_begun = time.perf_counter()
                        try:
                            item = prepare(row, sample, modules, assets, config, identity)
                        except Exception as error:
                            item = {'sample_id': row['sample_id'], 'status': 'FAIL', 'cache_identity': identity,
                                    'failure_reason': f'{type(error).__name__}: {error}'}
                            sample.mkdir(exist_ok=True)
                            write_json(audit, item)
                            raise
                        item['elapsed_seconds'] = time.perf_counter() - clip_begun
                        write_json(audit, item)
                        state['new_clips'] += 1
                    results.append(item)
                    state.update(passed=len(results), elapsed_seconds=time.perf_counter()-begun,
                                 last_sample_id=row['sample_id'])
                    write_json(args.output / 'status.json', state)
                    print(json.dumps({'sample_id': row['sample_id'], 'status': item['status'],
                                      'passed': len(results), 'frames': item['frames'],
                                      'missing_face_frames': item['missing_face_frames']}), flush=True)
            state['status'] = 'PASS_NATIVE_PREPROCESSING' if len(results) == len(rows) else 'PARTIAL_BOUNDED_VALIDATION'
            state['elapsed_seconds'] = time.perf_counter() - begun
            state['cumulative_attempt_wall_seconds'] = state['elapsed_seconds'] + sum(
                p.get('elapsed_seconds', 0) for p in previous)
            write_json(attempt, state)
            write_json(args.output / 'report.json', {**state, 'results': results,
                'scope': 'Native preprocessing only. No benchmark metric; retain missing-stream masks in subsequent features.'})
            write_json(args.output / 'status.json', state)
        except Exception as error:
            state.update(status='FAIL', elapsed_seconds=time.perf_counter()-begun,
                         failure_reason=f'{type(error).__name__}: {error}')
            write_json(attempt, state)
            write_json(args.output / 'status.json', state)
            raise


if __name__ == '__main__':
    main()
