"""Import trusted Uni-Sign pickle caches without silently selecting person zero."""
from pathlib import Path
import pickle
import numpy as np
from .person_tracking import PersonTracker
from .schema import save_pose, sha256_file, write_manifest


def track_frames(keypoints, scores, tracker=None):
    if len(keypoints) != len(scores) or len(keypoints) == 0:
        raise ValueError('Mismatched or empty frame lists')
    tracker = tracker or PersonTracker()
    t = len(keypoints)
    xy, q = np.zeros((t, 133, 2), np.float32), np.zeros((t, 133), np.float32)
    obs, continuity = np.zeros((t, 133), bool), np.zeros(t-1, bool)
    for i, (points, score) in enumerate(zip(keypoints, scores)):
        points, score = np.asarray(points, np.float32), np.asarray(score, np.float32)
        if points.size == score.size == 0:
            points, score = np.empty((0, 133, 2), np.float32), np.empty((0, 133), np.float32)
        if points.shape == (133, 2) and score.shape == (133,):
            points, score = points[None], score[None]
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError('Detector confidence must be finite in [0,1]')
        index, continuous = tracker.step(points, score)
        if index is not None:
            obs[i] = np.isfinite(points[index]).all(-1) & (score[index] > 0)
            xy[i] = np.where(obs[i, :, None], points[index], 0)
            q[i] = score[index]
        if i:
            continuity[i-1] = continuous
    return xy, q, obs, continuity, tracker.report


def import_unisign_pose(input_path, metadata, output_path):
    # Pickle is a legacy input format. Load only caches from a trusted source.
    with open(input_path, 'rb') as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict) or not {'keypoints', 'scores'} <= cache.keys():
        raise ValueError('Expected Uni-Sign keypoints/scores dictionary')
    if metadata.get('joint_layout') != 'coco_wholebody_133':
        raise ValueError('Explicit COCO-WholeBody convention required')
    profile = metadata.get('score_profile', 'raw_probability')
    raw_scores = cache['scores']
    if profile == 'simcc_positive_rational_v1':
        converted_scores = []
        for scores in raw_scores:
            values = np.asarray(scores, np.float32)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('SimCC profile requires finite nonnegative detector scores')
            # Explicit uncalibrated reliability mapping, not a probability claim.
            converted_scores.append(values/(1+values))
        cache = {**cache, 'scores': converted_scores}
    elif profile != 'raw_probability':
        raise ValueError('Unknown score conversion profile')
    xy, q, obs, continuity, report = track_frames(cache['keypoints'], cache['scores'], PersonTracker(metadata.get('signer_roi')))
    if metadata.get('coordinate_system') == 'pixels':
        w, h = metadata.get('image_width'), metadata.get('image_height')
        if not w or not h or min(w, h) <= 0:
            raise ValueError('Pixel input requires image dimensions')
        xy /= np.array([w, h], np.float32)
    elif metadata.get('coordinate_system') != 'viewport_normalized':
        raise ValueError('Explicit coordinate system required')
    t = len(xy)
    fps = metadata.get('fps')
    timestamps = metadata.get('timestamps_sec')
    known = timestamps is not None or fps is not None
    if timestamps is not None:
        times = np.asarray(timestamps, np.float64)
    elif fps is not None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('FPS must be positive and verified')
        times = np.arange(t, dtype=np.float64)/fps
    else:
        times = np.zeros(t, np.float64)
    pose = {'xy': xy, 'confidence': q, 'observed': obs, 'timestamps_sec': times,
            'timing_known': np.array(known), 'frame_indices': np.asarray(metadata.get('frame_indices', np.arange(t)), np.int64),
            'track_continuity': continuity, 'person_track_id': np.array('geometric_track_0')}
    if profile != 'raw_probability':
        # Preserve all-person raw output separately; the original source pickle
        # hash plus this sidecar provide audit without unsafe pickle in training.
        raw_path = Path(output_path).with_suffix('.detector_scores.npz')
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        sizes = np.array([np.asarray(s).size//133 for s in raw_scores], np.int64)
        raw = np.concatenate([np.asarray(s, np.float32).reshape(-1, 133) for s in raw_scores], axis=0)
        np.savez_compressed(raw_path, detector_scores=raw, frame_person_offsets=np.r_[0, np.cumsum(sizes)])
        metadata = {**metadata, 'raw_detector_scores_path': str(raw_path.resolve()), 'raw_detector_scores_sha256': sha256_file(raw_path),
                    'score_conversion': 'q=s/(1+s), finite nonnegative SimCC score; uncalibrated reliability',
                    'raw_score_range': [float(raw.min()) if raw.size else None, float(raw.max()) if raw.size else None]}
    save_pose(output_path, pose)
    row = {**metadata, 'score_profile': profile, 'pose_path': str(Path(output_path).resolve()), 'num_frames': t,
           'coordinate_system': 'viewport_normalized', 'timing_source': metadata.get('timing_source', 'metadata' if known else 'unknown'),
           'pose_sha256': sha256_file(output_path), 'source_cache_sha256': sha256_file(input_path)}
    row.pop('timestamps_sec', None)
    row.pop('frame_indices', None)
    return row, report


def import_directory(input_dir, metadata_path, output_dir, manifest_path):
    import json
    metadata_path = Path(metadata_path).resolve()
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    rows, reports = [], []
    for i, row in enumerate(metadata):
        source = Path(input_dir)/row['source_pose_path']
        converted, report = import_unisign_pose(source, row, Path(output_dir)/f'{i:08d}.npz')
        rows.append(converted)
        reports.append({'sample_id': row['sample_id'], **report})
    write_manifest(manifest_path, rows)
    report_path = Path(manifest_path).with_suffix('.conversion_report.json')
    report_path.write_text(json.dumps(reports, indent=2))
    return reports
