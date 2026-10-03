"""Run pinned author preprocessing on bounded train clips with CPU only."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Must precede torch/Ultralytics imports. This job cannot use the allocated GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.shubert import verify_source


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, first_rgb = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first_rgb is None:
            first_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames += 1
    cap.release()
    if frames < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Cannot decode native intermediate: ' + str(path))
    return {'frames': frames, 'fps': fps}, first_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/shubert.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-clips', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    rows = [row for _, row in read_jsonl(args.manifest)]
    if not 1 <= len(rows) <= min(args.max_clips, 20) or any(row['official_split'] != 'train' for row in rows):
        raise ValueError('Native preprocessing pilot is bounded to at most 20 train clips')
    config = yaml.safe_load(args.config.read_text())
    source = verify_source(ROOT, config['upstream_path'], config['upstream_commit'])
    registry = json.loads((ROOT / 'provenance/assets.json').read_text())['assets']
    assets = {}
    for asset_id in ['W05', 'W06', 'W07']:
        registered = registry[asset_id]
        path = ROOT / registered['local_path']
        if sha256(path) != registered['expected_sha256']:
            raise ValueError('Unverified preprocessing checkpoint: ' + asset_id)
        assets[asset_id] = path
    if args.dry_run:
        print(json.dumps({'samples': len(rows), 'device': 'cpu', 'manifest_sha256': sha256(args.manifest)}))
        return
    if args.output.exists():
        raise ValueError('Preserve native preprocessing attempts; choose a fresh output')
    args.output.mkdir(parents=True)
    begun = time.perf_counter()
    write_json(args.output / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(), 'device': 'cpu'})
    modules = {name: module(source / path, 'native_' + name) for name, path in {
        'crop': 'dataset/clips_bbox.py', 'pose': 'dataset/kpe_mediapipe.py',
        'face': 'dataset/crop_face.py', 'hands': 'dataset/crop_hands.py', 'body': 'features/body_features.py'}.items()}
    face_options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(assets['W05']), delegate=python.BaseOptions.Delegate.CPU),
                 output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6)
    hand_options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(assets['W06']), delegate=python.BaseOptions.Delegate.CPU),
                 num_hands=6, min_hand_detection_confidence=.05)
    results = []
    try:
        with vision.FaceLandmarker.create_from_options(face_options) as face_detector, \
             vision.HandLandmarker.create_from_options(hand_options) as hand_detector, \
             mp.solutions.holistic.Holistic(min_detection_confidence=.1) as holistic:
            pose = modules['pose']
            pose.face_detector, pose.hand_detector, pose.mp_holistic = face_detector, hand_detector, holistic
            for row in rows:
                source_path = Path(row['source_root']) / row['relative_path']
                sample = args.output / sha256_string(row['sample_id'])[:20]
                sample.mkdir()
                problem = sample / 'problems.txt'
                problem.touch()
                clip = sample / 'signer.mp4'
                try:
                    modules['crop'].crop_clip(str(source_path), str(problem), str(clip), str(assets['W07']))
                    original, _ = video_info(source_path)
                    cropped, _ = video_info(clip)
                    if original['frames'] != cropped['frames'] or abs(original['fps'] - cropped['fps']) > .05:
                        raise ValueError('Signer crop changed frame count or FPS')
                    pose.video_holistic(str(clip), str(problem), sample, sample)
                    landmarks = sample / 'signer_pose.json'
                    modules['face'].video_holistic(str(clip), str(sample), str(problem), str(sample))
                    modules['hands'].video_holistic(str(clip), str(sample), str(problem), str(sample))
                    modules['body'].keypoints_to_numpy(str(landmarks), str(sample))
                    body_path = sample / 'signer_pose.npy'
                    body = np.load(body_path, allow_pickle=False)
                    paths = {'face': sample / 'signer_face.mp4', 'left_hand': sample / 'signer_hand1.mp4',
                             'right_hand': sample / 'signer_hand2.mp4', 'body_posture': body_path}
                    info = {name: video_info(path)[0] for name, path in paths.items() if name != 'body_posture'}
                    if body.shape != (original['frames'], 14) or not np.isfinite(body).all():
                        raise ValueError('Native body features shape/finite check failed')
                    if any(values['frames'] != original['frames'] or abs(values['fps'] - original['fps']) > .05 for values in info.values()):
                        raise ValueError('Native cue crops lost frame alignment; cannot silently truncate')
                    # Actual decoder RGB agreement on a generated cue file.
                    import decord
                    reader = decord.VideoReader(str(paths['face']), ctx=decord.cpu(0))
                    decoded = reader[0].asnumpy()
                    _, rgb = video_info(paths['face'])
                    difference = float(np.abs(decoded.astype(float) - rgb.astype(float)).max())
                    if difference > 3:
                        raise ValueError('OpenCV RGB and author decord RGB disagree')
                    observed = json.loads(landmarks.read_text())
                    missing_pose = sum(value is None or value.get('pose_landmarks') is None for value in observed.values())
                    missing_face = sum(value is None or value.get('face_landmarks') is None for value in observed.values())
                    item = {'sample_id': row['sample_id'], 'status': 'PASS_NATIVE_PREPROCESSING', 'source_path': str(source_path),
                            'source_sha256': sha256(source_path), 'streams': {k: str(v) for k, v in paths.items()},
                            'stream_sha256': {k: sha256(v) for k, v in paths.items()}, 'frames': original['frames'],
                            'fps': original['fps'], 'missing_pose_frames': missing_pose, 'missing_face_frames': missing_face,
                            'rgb_decoder_max_abs_difference': difference,
                            'rate_note': 'Pinned raw-video scripts retain input FPS. Paper pretraining removes every other frame; downstream preprocessing is TODO upstream, so this is not an exact paper recipe reproduction.'}
                except Exception as error:
                    item = {'sample_id': row['sample_id'], 'status': 'FAIL', 'failure_reason': f'{type(error).__name__}: {error}'}
                results.append(item)
                write_json(sample / 'audit.json', item)
                print(json.dumps(item), flush=True)
        report = {'status': 'PASS_NATIVE_PREPROCESSING' if all(row['status'].startswith('PASS') for row in results) else 'FAIL',
                  'samples': len(rows), 'passed': sum(row['status'].startswith('PASS') for row in results),
                  'results': results, 'device': 'cpu', 'elapsed_seconds': time.perf_counter() - begun,
                  'manifest_sha256': sha256(args.manifest), 'config_sha256': sha256(args.config),
                  'upstream_commit': config['upstream_commit'], 'implementation_sha256': sha256(Path(__file__)),
                  'native_source_hashes': {name: sha256(Path(loaded.__file__)) for name, loaded in modules.items()},
                  'scope': 'Author script preprocessing smoke; no model accuracy or native paper performance claim.'}
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'status.json', {'status': report['status']})
        if report['status'] == 'FAIL':
            raise SystemExit(1)
    except Exception as error:
        write_json(args.output / 'status.json', {'status': 'FAIL', 'failure_reason': f'{type(error).__name__}: {error}'})
        raise


def sha256_string(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == '__main__':
    main()
