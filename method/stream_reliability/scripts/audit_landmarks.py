"""Bounded train/dev author detector smoke; raw observability before filling."""
import argparse
from contextlib import redirect_stdout
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'method/frozen_probe'))
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


class DetectorRecorder:
    def __init__(self, detector):
        self.detector = detector
        self.latest = None

    def detect(self, image):
        self.latest = self.detector.detect(image)
        return self.latest


def main():
    parser = argparse.ArgumentParser()
    for field in ('config', 'output-dir'):
        parser.add_argument('--' + field, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        for name, digest in cfg['input_hashes'].items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Pinned data/code metadata changed')
        import numpy as np
        import torch
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import decord
        torch.set_num_threads(1)
        package = Path(mp.__file__).parent
        installed_models = {}
        for name in cfg['required_installed_models']:
            path = package / name
            if not path.is_file():
                raise ValueError('Missing installed detector asset; no implicit download allowed')
            installed_models[name] = file_sha(path)
        remote = json.loads((REPO / cfg['remote_tree']).read_text())
        tasks = {}
        for filename in cfg['task_filenames']:
            matches = [row for row in remote if row['path'] == 'models/' + filename]
            path = REPO / 'checkpoints/shubert_author_hf/models' / filename
            if len(matches) != 1 or path.stat().st_size != matches[0]['lfs']['size']:
                raise ValueError('Published landmarker identity missing')
            digest = file_sha(path)
            if digest != matches[0]['lfs']['oid']:
                raise ValueError('Local landmarker differs from published author bytes')
            tasks[filename] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
        cohort = json.loads((REPO / cfg['cohort_manifest']).read_text())
        sources = {row['sample_id']: row for row in cohort['samples']}
        audit = json.loads((REPO / cfg['video_audit']).read_text())
        videos = {row['sample_id']: row for row in audit['samples']}
        artifact_root = Path('/dev/shm') / ('shubert_h003_landmarks_' + run.status['config_sha256'][:20])
        free = shutil.disk_usage('/dev/shm')
        if free.free - cfg['max_output_bytes'] < max(free.total * .2, 2 * 1024**3):
            raise ValueError('RAM reserve not satisfied')
        artifact_root.mkdir(exist_ok=False)
        run.status['artifact_root'] = str(artifact_root)
        run.save('status.json', run.status)
        spec = importlib.util.spec_from_file_location('author_landmark_reference', REPO / cfg['author_source'])
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        report = dict(run_id=args.run_id, GPU_hours=0, artifact_root=str(artifact_root),
                      task_assets=tasks, installed_model_hashes=installed_models, samples=[],
                      detector_options=cfg['detector_options'],
                      semantics='Raw API counts/handedness categories; no ground-truth tracking quality or probability calibration',
                      deviations=['Holistic reset at every clip boundary',
                                  'Sequential native frames without upstream seek(0)',
                                  'Retain raw detections before crop/fill and preserve decoded timestamps'],
                      scope='Train/dev input smoke only; no crops, DINO features, task scores or test access')
        run.save('landmark_audit.json', report)
        output_bytes = 0
        total_frames = 0
        for sample_id in cfg['sample_ids']:
            source = sources[sample_id]
            native = videos[sample_id]
            if source['split'] not in ('train', 'val') or file_sha(source['input_path']) != native['source_sha256']:
                raise ValueError('Fixed train/dev source identity changed')
            reader = decord.VideoReader(source['input_path'], ctx=decord.cpu(0), num_threads=1)
            if len(reader) != native['frames']:
                raise ValueError('Native decoded frame count differs')
            total_frames += len(reader)
            if total_frames > cfg['max_frames']:
                raise ValueError('Registered frame bound exceeded')
            timestamps = reader.get_frame_timestamp(range(len(reader)))
            if timestamps.shape != (len(reader), 2) or not np.isfinite(timestamps).all():
                raise ValueError('Invalid decoded timestamps')
            pts_delta = float(np.max(np.abs(timestamps[:, 0] - np.asarray(native['pts_s']))))
            if pts_delta > cfg['pts_tolerance_seconds'] or not np.all(np.diff(timestamps[:, 0]) > 0):
                raise ValueError('Decoder/registered PTS alignment failed')
            face_opts = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=tasks[cfg['task_filenames'][0]]['path'],
                                                delegate=python.BaseOptions.Delegate.CPU),
                output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6)
            hand_opts = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=tasks[cfg['task_filenames'][1]]['path'],
                                                delegate=python.BaseOptions.Delegate.CPU),
                num_hands=6, min_hand_detection_confidence=.05)
            counts = {'face_absent': 0, 'hand_absent': 0, 'pose_absent': 0,
                      'multi_face': 0, 'multi_hand_over_two': 0}
            rows = []
            with vision.FaceLandmarker.create_from_options(face_opts) as face, \
                    vision.HandLandmarker.create_from_options(hand_opts) as hand, \
                    mp.solutions.holistic.Holistic(min_detection_confidence=.1) as holistic:
                author.face_detector = DetectorRecorder(face)
                author.hand_detector = DetectorRecorder(hand)
                author.mp_holistic = holistic
                with (run.output / 'detector_stdout.txt').open('a') as log, redirect_stdout(log):
                    for frame_index in range(len(reader)):
                        image = reader[frame_index].asnumpy()
                        if image.dtype != np.uint8 or image.shape != (native['stream']['height'], native['stream']['width'], 3):
                            raise ValueError('Decoded RGB image shape differs')
                        observation, landmarks = author.detect_holistic(image)
                        for values in landmarks.values():
                            if values is not None and not np.isfinite(np.asarray(values)).all():
                                raise ValueError('Nonfinite observed landmark coordinates')
                        raw_hand = author.hand_detector.latest
                        handedness = [[dict(label=c.category_name, score=float(c.score), index=int(c.index))
                                       for c in categories] for categories in raw_hand.handedness]
                        counts['face_absent'] += int(observation['#face'] == 0)
                        counts['hand_absent'] += int(observation['#hands'] == 0)
                        counts['pose_absent'] += int(observation['#pose'] == 0)
                        counts['multi_face'] += int(observation['#face'] > 1)
                        counts['multi_hand_over_two'] += int(observation['#hands'] > 2)
                        rows.append(dict(frame_index=frame_index, pts_start_s=float(timestamps[frame_index, 0]),
                                         pts_end_s=float(timestamps[frame_index, 1]), counts=observation,
                                         landmarks=landmarks, handedness=handedness))
            target = artifact_root / (str(len(report['samples'])) + '.json.gz')
            with target.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
                compressed.write(json.dumps(dict(sample_id=sample_id, frames=rows), allow_nan=False).encode())
            output_bytes += target.stat().st_size
            if output_bytes > cfg['max_output_bytes']:
                raise ValueError('Raw artifact output cap exceeded')
            report['samples'].append(dict(sample_id=sample_id, split=source['split'], frames=len(rows),
                                          source_sha256=native['source_sha256'], pts_max_delta_s=pts_delta,
                                          counts=counts, raw_path=str(target), raw_bytes=target.stat().st_size,
                                          raw_sha256=file_sha(target)))
            run.save('landmark_audit.json', report)
        report.update(total_frames=total_frames, raw_bytes=output_bytes, CUDA_initialized=torch.cuda.is_initialized(),
                      mediapipe_version=mp.__version__, decord_version=decord.__version__, completed=True)
        if report['CUDA_initialized']:
            raise ValueError('CPU-only landmark audit initialized CUDA')
        run.save('landmark_audit.json', report)
        run.status['output_hashes'] = {'landmark_audit.json': file_sha(run.output / 'landmark_audit.json')}


if __name__ == '__main__':
    main()
