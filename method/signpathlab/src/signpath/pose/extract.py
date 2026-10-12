"""Independent IMAGE-mode inference with same-frame wrist matching and raw retention."""
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import importlib.metadata
import numpy as np

from ..cache.hashing import file_hash
from .topology import UPPER7_MEDIAPIPE


@dataclass(frozen=True)
class RawLandmarks:
    xy: np.ndarray
    observed: np.ndarray
    quality: np.ndarray
    body_raw: np.ndarray       # [T,33,4] x,y,visibility,presence
    hand_candidates: np.ndarray  # [T,2,21,2], before anatomical matching
    face_raw: np.ndarray       # [T,478,2]
    handedness_scores: np.ndarray  # category score; NOT localization quality
    assignment: np.ndarray    # [T,2] left/right -> candidate index, -1 missing
    provenance: dict


def match_hands(body_xy, body_quality, candidates, sizes, *, min_quality=.5, max_distance=.75, min_margin=.15):
    """Assignment by same-frame anatomical wrists, independent of image x ordering."""
    candidates = np.asarray(candidates)
    output = np.full(2, -1, dtype=np.int64)
    if not len(candidates):
        return output
    if not (np.isfinite(body_xy[[1, 2]]).all() and (body_quality[[1, 2]] >= min_quality).all()):
        return output
    scale = np.linalg.norm((body_xy[1] - body_xy[2]) * sizes)
    if scale <= 1:
        return output
    distances = np.linalg.norm((candidates[:, None, 0, :] - body_xy[None, [5, 6], :]) * sizes, axis=-1) / scale
    distances = np.where(np.isfinite(distances), distances, np.inf)
    reliable = body_quality[[5, 6]] >= min_quality
    if len(candidates) == 1:
        side = int(np.argmin(distances[0]))
        if reliable[side] and distances[0, side] <= max_distance and abs(distances[0, 0] - distances[0, 1]) >= min_margin:
            output[side] = 0
        return output
    assignments = sorted((float(distances[left, 0] + distances[right, 1]), left, right)
                         for left, right in permutations(range(len(candidates)), 2))
    best = assignments[0]
    if not np.isfinite(best[0]) or assignments[1][0] - best[0] < min_margin:
        return output
    for side, candidate in enumerate(best[1:]):
        if reliable[side] and distances[candidate, side] <= max_distance:
            output[side] = candidate
    return output


class LandmarkExtractor:
    def __init__(self, pose_model, hand_model, face_model, *, min_body_visibility=.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        paths = {"pose": Path(pose_model), "hand": Path(hand_model), "face": Path(face_model)}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing MediaPipe IMAGE-mode model: {path}")
        self.mp, self.min_quality = mp, min_body_visibility
        self.provenance = {"mediapipe_version": importlib.metadata.version("mediapipe"), "detector_mode": "IMAGE",
                           "temporal_dependency_policy": "same_frame", "weights_sha256": {k: file_hash(p) for k, p in paths.items()},
                           "hand_assignment": "same_frame_body_wrist_distance_v1", "min_body_visibility": min_body_visibility,
                           "hand_quality_kind": "observed_binary", "body_quality_kind": "visibility",
                           "handedness_score_kind": "category_score_not_localization", "max_hand_distance": .75, "hand_assignment_margin": .15}
        options = lambda path: BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.CPU)
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=options(paths["pose"]), running_mode=vision.RunningMode.IMAGE, num_poses=1))
        try:
            self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=options(paths["hand"]), running_mode=vision.RunningMode.IMAGE, num_hands=2))
            self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=options(paths["face"]), running_mode=vision.RunningMode.IMAGE, num_faces=1))
        except Exception:
            self.close()
            raise

    def close(self):
        for name in ("pose", "hand", "face"):
            if hasattr(self, name):
                getattr(self, name).close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def extract(self, frames):
        n = len(frames)
        xy = np.full((n, 49, 2), np.nan, dtype=np.float32)
        observed, quality = np.zeros((n, 49), dtype=bool), np.zeros((n, 49), dtype=np.float32)
        body_raw = np.full((n, 33, 4), np.nan, dtype=np.float32)
        hand_raw = np.full((n, 2, 21, 2), np.nan, dtype=np.float32)
        face_raw = np.full((n, 478, 2), np.nan, dtype=np.float32)
        scores = np.full((n, 2), np.nan, dtype=np.float32)
        assignment = np.full((n, 2), -1, dtype=np.int64)
        for t, frame in enumerate(frames):
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            body, hands, face = self.pose.detect(image), self.hand.detect(image), self.face.detect(image)
            if body.pose_landmarks:
                for j, landmark in enumerate(body.pose_landmarks[0]):
                    body_raw[t, j] = [landmark.x, landmark.y, landmark.visibility or 0, landmark.presence or 0]
                xy[t, :7] = body_raw[t, UPPER7_MEDIAPIPE, :2]
                quality[t, :7] = body_raw[t, UPPER7_MEDIAPIPE, 2]
                observed[t, :7] = np.isfinite(xy[t, :7]).all(axis=-1)
            for candidate, hand in enumerate(hands.hand_landmarks):
                hand_raw[t, candidate] = [[p.x, p.y] for p in hand]
                scores[t, candidate] = hands.handedness[candidate][0].score
            assignment[t] = match_hands(xy[t], quality[t], hand_raw[t, :len(hands.hand_landmarks)],
                                         np.array([frame.shape[1], frame.shape[0]]), min_quality=self.min_quality)
            for side, candidate in enumerate(assignment[t]):
                if candidate >= 0:
                    start = 7 + side * 21
                    xy[t, start:start + 21] = hand_raw[t, candidate]
                    observed[t, start:start + 21] = np.isfinite(hand_raw[t, candidate]).all(axis=-1)
                    quality[t, start:start + 21] = observed[t, start:start + 21]
            if face.face_landmarks:
                points = np.array([[p.x, p.y] for p in face.face_landmarks[0]], dtype=np.float32)
                face_raw[t, :len(points)] = points
        return RawLandmarks(xy, observed, quality, body_raw, hand_raw, face_raw, scores, assignment, self.provenance)
