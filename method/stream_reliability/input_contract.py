"""Native crop geometry with explicit frame retention and observation provenance.

Adaptations: initial whole-row None retains a frame; body sentinel always14D;
degenerate body/nonempty crop failures raise before committing state. Preserve
native independent hand assignment, including duplicate detection, and flag it.
Pixel color is RGB; an encoder wrapper must reverse to BGR for OpenCV writing.
"""
import numpy as np


class AlignedNativeCrops:
    names = ('face', 'left_hand', 'right_hand', 'body_posture')

    def __init__(self, face_reference, hands_reference, body_reference):
        self.face_reference = face_reference
        self.hands_reference = hands_reference
        self.body_reference = body_reference
        self.previous = {}
        self.next_frame = 0

    def step(self, rgb, raw, frame_index):
        if frame_index != self.next_frame:
            raise ValueError('Native frame order must be complete and contiguous')
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Native RGB uint8 frame required')
        if raw is not None and set(raw) != {'face_landmarks', 'hand_landmarks', 'pose_landmarks'}:
            raise ValueError('Exact author detector schema required')
        raw = raw or dict.fromkeys(('face_landmarks', 'hand_landmarks', 'pose_landmarks'))
        pose = raw['pose_landmarks'][0] if raw['pose_landmarks'] else None
        current = {}
        left_index = right_index = None
        boxes = {}
        if pose is not None:
            values = np.asarray(pose)
            if values.shape != (33, 3) or not np.isfinite(values).all():
                raise ValueError('Finite33×3 author pose required')
            if not np.linalg.norm(values[12, :2] - values[11, :2]) > 0:
                raise ValueError('Collapsed shoulders; reject rather than emit nonfinite body')
            normalized = np.asarray(self.body_reference.normalize_pose_keypoints(pose[:25]))
            current['body_posture'] = normalized[[0, 11, 12, 13, 14, 15, 16]].reshape(14)
            if not np.isfinite(current['body_posture']).all():
                raise ValueError('Body normalization nonfinite')
            if raw['face_landmarks'] is not None:
                face = self.face_reference.select_face(pose, raw['face_landmarks'])
                pixels = self.face_reference.cues_on_grey_background(rgb, face)
                current['face'] = self.face_reference.resize_frame(pixels, (224, 224))
            left, right = self.hands_reference.select_hands(pose, raw['hand_landmarks'], rgb.shape)
            for name, selected in (('left_hand', left), ('right_hand', right)):
                if selected is None:
                    continue
                index = next(i for i, value in enumerate(raw['hand_landmarks']) if value is selected)
                if name == 'left_hand': left_index = index
                else: right_index = index
                box = self.hands_reference.adjust_bounding_box(
                    self.hands_reference.get_bounding_box(selected, rgb.shape, scale_factor=1.5), rgb.shape)
                pixels = self.hands_reference.crop_frame(rgb, box)
                current[name] = self.hands_reference.resize_frame(pixels, (224, 224))
                boxes[name] = dict(native_box=[int(v) for v in box], actual_extent=list(pixels.shape[:2]),
                                  declared_extent_matches=tuple(pixels.shape[:2]) == (box[3], box[2]))
        for name, value in current.items():
            shape = (14,) if name == 'body_posture' else (224, 224, 3)
            if value is None or value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Observed native stream output invalid; no exception-to-fill substitution')
        output = {}
        provenance = {}
        new_previous = dict(self.previous)
        for name in self.names:
            if name in current:
                value = current[name]
                origin = frame_index
                kind = 'observed'
                new_previous[name] = (value.copy(), origin)
            elif name in self.previous:
                value, origin = self.previous[name]
                kind = 'repeat_previous'
            else:
                value = np.full(14, -9999., dtype=np.float64) if name == 'body_posture' else np.zeros((224, 224, 3), dtype=np.uint8)
                origin = None
                kind = 'initial_body_sentinel' if name == 'body_posture' else 'initial_blank'
            output[name] = value.copy()
            provenance[name] = dict(observed=name in current, fill_kind=kind, source_frame_index=origin)
        metadata = dict(frame_index=frame_index, streams=provenance,
                        detector_counts={name: len(raw[name]) if raw[name] else 0 for name in raw},
                        left_detection_index=left_index, right_detection_index=right_index,
                        same_detection=left_index is not None and left_index == right_index,
                        hand_boxes=boxes,
                        semantics='API observations/geometric assignment only; not correct anatomy or natural failure labels')
        self.previous = new_previous
        self.next_frame += 1
        return output, metadata
