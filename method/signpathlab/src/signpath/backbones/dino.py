"""Frozen official DINOv2; explicit pinned code and strict weight loading."""
from pathlib import Path
import subprocess
import numpy as np
import torch

from .base import BackboneAdapter
from ..cache.hashing import canonical_hash, file_hash, is_sha256
from ..cache.schema import FeatureBundle
from ..geometry.supports import Primitive, normalized_joint_support
from ..pose.coordinates import body_coordinates
from ..pose.roi import landmark_box, letterbox
from ..pose.topology import POSE_GROUPS, TOPOLOGY_HASH


class DINOAdapter(BackboneAdapter):
    def load(self, registry_entry, device):
        entry = dict(registry_entry)
        repo, weights = Path(entry["repo_dir"]).resolve(), Path(entry["local_path"]).resolve()
        if not repo.is_dir() or not weights.is_file():
            raise FileNotFoundError(f"DINO requires pinned repository {repo} and weights {weights}")
        if not is_sha256(entry.get("sha256")) or file_hash(weights) != entry["sha256"]:
            raise ValueError("DINO weights need a verified SHA-256, with no random fallback")
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if actual != entry["code_commit"] or dirty:
            raise ValueError("DINO source commit mismatch or unrecorded source patch")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable")
        self.model = torch.hub.load(str(repo), entry["hub_entry"], source="local", pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        report = self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.output_dim = int(entry["output_dim"])
        self.extractor_id = entry["hub_entry"] + "_generic"
        if self.model.embed_dim != self.output_dim:
            raise ValueError("DINO architecture/output dimension mismatch")
        self._provenance = {"registry_key": entry.get("registry_key", entry["hub_entry"]), "code_commit": actual,
            "code_repository": entry["code_repository"], "weights_sha256": entry["sha256"],
            "initialization": "official_pretrained", "missing_keys": list(report.missing_keys),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "unexpected_keys": list(report.unexpected_keys), "source_files_sha256": {name: file_hash(repo / name) for name in
                ("dinov2/hub/backbones.py", "dinov2/models/vision_transformer.py", "hubconf.py")}}
        return self

    def provenance(self):
        return dict(self._provenance)

    @torch.inference_mode()
    def encode_crops(self, images, *, batch_size=16):
        if batch_size <= 0:
            raise ValueError("DINO extraction batch size must be positive")
        images = np.asarray(images)
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (224, 224, 3):
            raise ValueError("DINO expects uint8 RGB crops [B,224,224,3], normalized exactly once")
        if len(images) == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)
        mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        outputs = []
        for start in range(0, len(images), batch_size):
            rgb = torch.from_numpy(images[start:start + batch_size].copy()).to(self.device).permute(0, 3, 1, 2).float() / 255
            features = self.model.forward_features((rgb - mean) / std)["x_norm_clstoken"]
            if features.shape != (len(rgb), self.output_dim) or not torch.isfinite(features).all():
                raise ValueError("Invalid DINO CLS output")
            outputs.append(features.cpu().numpy())
        return np.concatenate(outputs)

    def extract(self, frames, timestamps, rois):
        """rois is an ExtractionContext with raw landmarks and original frame IDs."""
        from ..pose.extract import RawLandmarks
        raw = rois.landmarks
        if not isinstance(raw, RawLandmarks):
            raise TypeError("DINO extraction requires RawLandmarks with provenance")
        n = len(frames)
        sizes = np.array([[frame.shape[1], frame.shape[0]] for frame in frames])
        coordinates = body_coordinates(raw.xy, raw.observed, raw.quality, sizes,
            epsilon_px=rois.epsilon_px, min_shoulder_quality=rois.min_body_visibility)
        crops, locations, transforms, padding, boxes, fallback = [], [], np.zeros((n, 4, 2, 3)), np.ones((n, 4)), np.zeros((n, 4, 4)), []
        supports = []
        valid = np.zeros((n, 4), dtype=bool)
        for t, (frame, frame_id) in enumerate(zip(frames, rois.source_frame_index)):
            dims = sizes[t]
            points = raw.xy[t] * dims
            # Conservative ROI supports include all landmark candidates and body wrist matching.
            full_joints = frozenset(Primitive("joint", int(frame_id), j) for j in range(49))
            face_joints = frozenset(Primitive("face", int(frame_id), j) for j in range(478))
            stream_supports = [frozenset({Primitive("rgb", int(frame_id))}) | full_joints | face_joints for _ in range(4)]
            usable = raw.observed[t] & (raw.quality[t] >= rois.min_body_visibility)
            global_box = landmark_box(points[usable], margin=1.25, square=False)
            global_fallback = global_box is None
            fallback.append(global_fallback)
            hand_boxes = [landmark_box(points[list(joints)], margin=1.5) if raw.observed[t, joints].all() else None for joints in POSE_GROUPS[1:]]
            face_points = raw.face_raw[t] * dims
            face_box = landmark_box(face_points[np.isfinite(face_points).all(axis=-1)], margin=1.25)
            for stream, box in enumerate([global_box, *hand_boxes, face_box]):
                if stream > 0 and box is None:
                    continue
                try:
                    crop = letterbox(frame, box)
                except ValueError:
                    if stream > 0:
                        continue
                    crop = letterbox(frame)
                    fallback[-1] = True
                crops.append(crop.image)
                locations.append((t, stream))
                transforms[t, stream], padding[t, stream], boxes[t, stream] = crop.affine, crop.padding_fraction, crop.box
                valid[t, stream] = True
            supports.extend(stream_supports)
            # Matching/ROI use same-frame inputs only; include their declared dependencies.
            supports.extend(normalized_joint_support(frame_id, group) | full_joints for group in POSE_GROUPS)
        encoded = self.encode_crops(np.stack(crops))
        visual = np.zeros((n, 4, self.output_dim), dtype=np.float16)
        for value, (t, stream) in zip(encoded, locations):
            visual[t, stream] = value
        preprocessing = {"detector": raw.provenance, "topology_hash": TOPOLOGY_HASH, "roi_policy": "per_frame_letterbox",
            "hand_margin": 1.5, "global_margin": 1.25, "face_margin": 1.25, "resize_interpolation": "opencv_inter_linear",
            "fill": 0, "output_size": 224, "epsilon_px": rois.epsilon_px, "min_body_visibility": rois.min_body_visibility,
            "normalization": "per_frame_shoulder_center_isotropic_scale", "implementation_version": 1}
        source_root = Path(__file__).resolve().parents[1]
        preprocessing["implementation_sha256"] = {name: file_hash(source_root / name) for name in
            ("backbones/dino.py", "pose/extract.py", "pose/coordinates.py", "pose/roi.py", "pose/topology.py")}
        identity = {"backbone": self.provenance(), "preprocessing": preprocessing}
        meta = {"schema_version": 1, "sample_id": rois.sample_id, "extractor_id": self.extractor_id,
            "extractor_hash": canonical_hash(identity), "weights_sha256": self._provenance["weights_sha256"], "feature_dim": self.output_dim,
            "visual_stream_order": ["global", "left_hand", "right_hand", "face"], "topology_hash": TOPOLOGY_HASH,
            "detector_mode": "IMAGE", "temporal_dependency_policy": "same_frame", "normalization": preprocessing["normalization"],
            "timestamp_unit": "seconds", "source_sha256": rois.source_sha256,
            "quality_kind": "observed_binary_hand_visibility_body", "preprocessing_hash": canonical_hash(preprocessing),
            "backbone": self.provenance(), "preprocessing": preprocessing,
            "roi_affines": transforms.tolist(), "roi_boxes": boxes.tolist(), "roi_padding_fractions": padding.tolist(),
            "global_full_frame_fallback": fallback, "mirror_metadata": rois.mirror_metadata}
        return FeatureBundle(meta, np.asarray(timestamps, dtype=np.float64), np.asarray(rois.source_frame_index, dtype=np.int64),
            visual, valid, coordinates.xy, raw.observed, coordinates.quality, coordinates.coordinate_valid, tuple(supports)).validate()
