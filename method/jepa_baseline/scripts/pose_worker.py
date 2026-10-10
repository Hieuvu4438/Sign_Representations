"""Isolated original-teacher process (Python3.10/MediaPipe0.10.14).

Keep all decoded frames and source confidence. Numeric NPZ transport prevents
teacher's generic top-level module names leaking into the SignRep process.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/skeleton_3d_uplift"))
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from forward_kinematics import ForwardK
from mlp_config import getModelFromCKPT
from skeleton import HumanSkeleton
from utils import matrix_to_euler_angles


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Teacher:
    def __init__(self, weights, body_root_mode="canonical"):
        torch.set_num_threads(2)
        if body_root_mode not in {"canonical", "camera"}:
            raise ValueError("Unknown body root convention")
        self.body_root_mode = body_root_mode
        self.weights = Path(weights)
        pins = json.loads((ROOT / "evidence/teacher_weights.json").read_text())
        for name in ["mlp_body.ckpt", "mlp_hand.ckpt"]:
            # Check content before executing the original checkpoint deserializer.
            expected = pins["weights"][f"models/{name}"]["sha256"]
            if sha(self.weights / name) != expected:
                raise ValueError(f"Teacher checkpoint hash mismatch: {name}")
        self.body = getModelFromCKPT(str(self.weights / "mlp_body.ckpt"), eval=True)
        self.hand = getModelFromCKPT(str(self.weights / "mlp_hand.ckpt"), eval=True)
        self.full = HumanSkeleton("cpu", "mediapipe_spine")
        if self.body.skeleton.getNumberOfAngles() != 22 or self.hand.skeleton.getNumberOfAngles() != 41:
            raise ValueError("Public teacher does not match paper's 22/41 Euler parameters")
        if self.body.skeleton.num_non_virtual_nodes != 19 or self.hand.skeleton.num_nodes != 21:
            raise ValueError("Public teacher does not match paper's 61-point topology")
        self.body_skeleton, self.hand_skeleton = self.body.skeleton, self.hand.skeleton

    @torch.inference_mode()
    def lift(self, body2d, right2d, left2d, detected):
        frames = len(body2d)
        result = {"body_angles": torch.zeros(frames, 22), "right_angles": torch.zeros(frames, 41),
                  "left_angles": torch.zeros(frames, 41),
                  "body_offsets_full": torch.zeros(frames, 22, 3, 1),
                  "right_offsets_full": torch.zeros(frames, 21, 3, 1),
                  "left_offsets_full": torch.zeros(frames, 21, 3, 1)}
        for part, points, model in [("body", body2d, self.body), ("right", right2d, self.hand),
                                    ("left", left2d, self.hand)]:
            valid = detected[part]
            if not valid.any():
                continue
            selected = points[valid].clone()
            if part == "body":
                selected = model.skeleton.centralizePose(selected)
            else:
                selected -= selected[:, :1].clone()
                if part == "left":
                    selected[..., 0] *= -1
            out = model(selected)
            angles = out["angles"].clone()
            if "cam_rot" in out and (part != "body" or self.body_root_mode == "camera"):
                # Original demo retains hand cam_rot, but exports native body angles.
                angles[:, model.skeleton.root_angles_idxs] = matrix_to_euler_angles(
                    out["cam_rot"], model.skeleton.convention
                )
            result[f"{part}_angles"][valid] = angles
            result[f"{part}_offsets_full"][valid] = model.skeleton.getOffsetsFull(out["offsets"])
        return result

    def forward_offsets(self, skeleton, angles, full_offsets):
        pose = torch.zeros(len(angles), skeleton.num_nodes, 3)
        ForwardK.forward(skeleton.getAnglesTensor(angles), pose, full_offsets, skeleton.roots)
        return skeleton.subSamplePose(pose)

    def reconstruct(self, data, standard=None):
        body = self.forward_offsets(self.body_skeleton, torch.as_tensor(data["body_angles"]),
                                    torch.as_tensor(data["body_offsets_full"] if standard is None
                                                    else standard["body_offsets_full"]))
        hands = []
        for part in ["right", "left"]:
            pose = self.forward_offsets(self.hand_skeleton, torch.as_tensor(data[f"{part}_angles"]),
                                        torch.as_tensor(data[f"{part}_offsets_full"] if standard is None
                                                        else standard["hand_offsets_full"]))
            if part == "left":
                pose[..., 0] *= -1
            hands.append(pose)
        return self.full.concatePoses(torch.cat([body, *hands], 1))

    def metadata(self):
        return {"teacher_commit": "6885e4865bc8993ae41a4d88287a5b03281d7d91",
                "body_angles": 22, "hand_angles": 41, "pose_points": 61,
                "weights": {p.name: sha(p) for p in self.weights.glob("*.ckpt")},
                "body_root_mode": self.body_root_mode,
                "body_root_camera_rotation": "native/canonical matches public demo; camera mode explicit U01 alternative",
                "confidence": "body MediaPipe visibility; hand source detector observation indicator",
                "hand_localization_confidence": "not supplied by MediaPipe; indicator is not a calibrated score",
                "missing_frames": "retained, zero confidence, no interpolation",
                "missing_hand_image_fallback": "not used for observed targets; masked missing hands"}


def extract(teacher, video_path, output):
    import mediapipe as mp
    from media_pipe.skeleton_mediapipe import BodySkeletonMediapipe
    from process_mediapipe import extractHolisticPoseMP
    from uplift_release import processMediapipeFrameOut
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video has no valid fps")
    records, timestamps, frame_indices = [], [], []
    with mp.solutions.holistic.Holistic(model_complexity=2, refine_face_landmarks=True,
                                      min_detection_confidence=.5, min_tracking_confidence=.5) as detector:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            observation, _, success = extractHolisticPoseMP(detector, frame)
            confidence = torch.zeros(61)
            points = torch.zeros(61, 2)
            if success[0]:
                order = teacher.body_skeleton.getBodyOrder()
                confidence = torch.cat((observation[:33, 2][order], observation[33:, 2]))
                diagonal = (frame.shape[0] ** 2 + frame.shape[1] ** 2) ** .5
                points, success = processMediapipeFrameOut(observation, success, teacher.full,
                                                          BodySkeletonMediapipe(), 1.5 * diagonal / 1414)
                if not success[1]:
                    confidence[19:40] = 0
                if not success[2]:
                    confidence[40:] = 0
            records.append((points, confidence, success))
            # Frame index, fps and decoder timestamp all retained. No reindexing after failures.
            timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            frame_indices.append(index)
            index += 1
    capture.release()
    if not records:
        raise ValueError("Empty video")
    points = torch.stack([r[0] for r in records])
    confidence = torch.stack([r[1] for r in records])
    flags = torch.tensor([r[2] for r in records], dtype=torch.bool)
    detected = {part: flags[:, i] for i, part in enumerate(["body", "right", "left"])}
    result = teacher.lift(points[:, :19], points[:, 19:40], points[:, 40:], detected)
    result["pose_3d"] = teacher.reconstruct(result)
    result["confidence"] = confidence
    for part, flag in detected.items():
        result[f"{part}_detected"] = flag
    result["body2d"], result["right2d"], result["left2d"] = points[:, :19], points[:, 19:40], points[:, 40:]
    metadata = teacher.metadata()
    metadata.update(video_sha256=sha(video_path), video_path=str(Path(video_path).resolve()),
                    fps=fps, decoded_frames=len(records), schema="signrep_teacher_v1")
    arrays = {key: value.numpy() for key, value in result.items()}
    arrays.update(frame_indices=np.array(frame_indices), decoder_timestamps_s=np.array(timestamps),
                  metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return metadata


def normalize(teacher, raw, stats, output):
    from signrep_baseline.bones import load_bone_statistics
    standard, statistics_metadata = load_bone_statistics(stats)
    with np.load(raw, allow_pickle=False) as source:
        result = {key: source[key].copy() for key in source.files}
    metadata = json.loads(str(result["metadata_json"]))
    if metadata.get("schema") != "signrep_teacher_v1" or metadata["weights"] != teacher.metadata()["weights"] or metadata.get("body_root_mode") != teacher.body_root_mode:
        raise ValueError("Normalization requires raw cache from the pinned teacher")
    result["pose_3d"] = teacher.reconstruct(result, standard).numpy()
    metadata.update(schema="signrep_standardized_v1", bone_statistics_sha256=sha(stats),
                    bone_statistics=statistics_metadata)
    result["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract", "normalize", "normalize-batch", "inspect"])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--body-root-mode", choices=["canonical", "camera"], default="canonical")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    teacher = Teacher(args.weights, args.body_root_mode)
    if args.command == "inspect":
        result = teacher.metadata()
    elif args.command == "extract":
        if not args.video or not args.output:
            parser.error("extract requires --video and --output")
        result = extract(teacher, args.video, args.output)
    elif args.command == "normalize-batch":
        if not args.manifest:
            parser.error("normalize-batch requires --manifest")
        tasks = json.loads(args.manifest.read_text())
        for task in tasks:
            normalize(teacher, Path(task["raw"]), Path(task["stats"]), Path(task["output"]))
        result = {"normalized_caches": len(tasks)}
    else:
        if not args.raw or not args.stats or not args.output:
            parser.error("normalize requires --raw, --stats and --output")
        result = normalize(teacher, args.raw, args.stats, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
