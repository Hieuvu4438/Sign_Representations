"""Optional streaming RTMW extraction, isolated from the torch training core."""
from pathlib import Path
import json
import numpy as np
from .person_tracking import PersonTracker
from .schema import save_pose, sha256_file


def extract_video(video_path, output_path, metadata, detector=None, mode='performance', backend='onnxruntime', device='cpu'):
    import av
    if not metadata.get('extractor_weights_sha256'):
        raise ValueError('Record actual detector/pose weight hashes before extraction')
    if detector is None:
        from rtmlib import Wholebody
        detector = Wholebody(mode=mode, backend=backend, device=device, to_openpose=False)
    profile = metadata.get('score_profile', 'raw_probability')
    if profile not in ('raw_probability', 'simcc_positive_rational_v1'):
        raise ValueError('Unknown detector score profile')
    tracker = PersonTracker(metadata.get('signer_roi'))
    positions, confidences, observations, continuous, times, raw_scores = [], [], [], [], [], []
    width = height = None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        for decoded in container.decode(stream):
            frame = decoded.to_ndarray(format='bgr24')
            h, w = frame.shape[:2]
            if width is None: width, height = w, h
            if (w, h) != (width, height):
                raise ValueError('Changing frame dimensions require a separate extraction profile')
            xy, score = detector(frame)
            xy, score = np.asarray(xy, np.float32), np.asarray(score, np.float32)
            if xy.size == score.size == 0:
                xy, score = np.empty((0, 133, 2), np.float32), np.empty((0, 133), np.float32)
            if not np.isfinite(score).all() or np.any(score < 0):
                raise ValueError('Invalid detector confidence')
            if profile == 'simcc_positive_rational_v1':
                raw_scores.append(score.copy())
                score = score/(1+score)
            elif np.any(score > 1):
                raise ValueError('Unbounded scores require an explicit SimCC conversion profile')
            xy = xy/np.array([width, height], np.float32)
            index, continuity = tracker.step(xy, score)
            observed = np.zeros(133, bool) if index is None else np.isfinite(xy[index]).all(-1) & (score[index] > 0)
            positions.append(np.zeros((133, 2), np.float32) if index is None else np.where(observed[:, None], xy[index], 0))
            confidences.append(np.zeros(133, np.float32) if index is None else score[index])
            observations.append(observed)
            continuous.append(continuity)
            times.append(float(decoded.pts*decoded.time_base) if decoded.pts is not None else np.nan)
            # Only the current decoded RGB frame exists; keep small pose arrays.
    n = len(positions)
    if not n: raise ValueError('Video decoded no frames')
    timestamps = np.asarray(times, np.float64)
    known = bool(np.isfinite(timestamps).all() and np.all(np.diff(timestamps) > 0))
    if not known: timestamps = np.zeros(n, np.float64)
    pose = {'xy': np.stack(positions).astype(np.float32), 'confidence': np.stack(confidences).astype(np.float32),
            'observed': np.stack(observations), 'timestamps_sec': timestamps, 'timing_known': np.array(known),
            'frame_indices': np.arange(n, dtype=np.int64), 'track_continuity': np.array(continuous[1:], bool),
            'person_track_id': np.array('geometric_track_0')}
    save_pose(output_path, pose)
    score_metadata = {}
    if profile != 'raw_probability':
        raw_path = Path(output_path).with_suffix('.detector_scores.npz')
        sizes = np.array([len(q) for q in raw_scores], np.int64)
        np.savez_compressed(raw_path, detector_scores=np.concatenate(raw_scores, axis=0), frame_person_offsets=np.r_[0, np.cumsum(sizes)])
        score_metadata = {'raw_detector_scores_path': str(raw_path.resolve()), 'raw_detector_scores_sha256': sha256_file(raw_path),
                          'score_conversion': 'q=s/(1+s); uncalibrated reliability, raw scores preserved'}
    report = {**metadata, **score_metadata, 'pose_path': str(Path(output_path).resolve()), 'video_path': str(Path(video_path).resolve()),
              'video_sha256': sha256_file(video_path), 'pose_sha256': sha256_file(output_path), 'num_frames': n,
              'fps': fps, 'timing_source': 'decoded_source_frame_pts' if known else 'unknown',
              'image_width': width, 'image_height': height, 'joint_layout': 'coco_wholebody_133', 'score_profile': profile,
              'coordinate_system': 'viewport_normalized',
              'extractor_profile': metadata.get('extractor_profile', f'rtmlib_Wholebody_{mode}_{backend}'),
              'tracking_report': tracker.report}
    Path(output_path).with_suffix('.metadata.json').write_text(json.dumps(report, indent=2))
    return report
