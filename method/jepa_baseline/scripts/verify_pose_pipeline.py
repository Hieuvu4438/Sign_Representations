"""Run in the isolated pose environment after extracting runtime/smoke/raw.npz."""
import json
import numpy as np
import torch
from pose_worker import Teacher, normalize, ROOT, sha, ForwardK
from signrep_baseline.bones import load_bone_statistics
from signrep_baseline.cache import load_standardized, clip_targets


def main():
    smoke = ROOT / "runtime/smoke"
    teacher = Teacher(ROOT / "runtime/teacher/models")
    normalize(teacher, smoke / "raw.npz", smoke / "bones.npz", smoke / "standardized.npz")
    data, metadata = load_standardized(smoke / "standardized.npz", sha(smoke / "bones.npz"),
                                       sha(smoke / "real_pose_clip.mp4"))
    assert len(data["frame_indices"]) == 36
    assert (data["confidence"][-4:] == 0).all()
    standard, _ = load_bone_statistics(smoke / "bones.npz")
    errors = {}
    for part, model in [("body", teacher.body), ("right", teacher.hand)]:
        skeleton = model.skeleton
        valid = data[f"{part}_detected"]
        points = torch.from_numpy(data[f"{part}2d"][valid])
        points = skeleton.centralizePose(points) if part == "body" else points - points[:, :1]
        with torch.inference_mode():
            out = model(points)
            reconstructed = teacher.forward_offsets(skeleton, torch.from_numpy(data[f"{part}_angles"][valid]),
                                                      torch.from_numpy(data[f"{part}_offsets_full"][valid]))
        # Original demo's native body angles reconstruct canonical output;
        # hand root cam_rot reconstructs camera-aligned output.
        reference = out["pose_3d_can"] if part == "body" else out["pose_3d"]
        torch.testing.assert_close(reconstructed, reference, rtol=2e-4, atol=2e-5)
        errors[part] = float((reconstructed - reference).abs().max())
        offset_key = "body" if part == "body" else "hand"
        angles = torch.from_numpy(data[f"{part}_angles"]).clone()
        angles += torch.randn_like(angles) * .2
        full_pose = torch.zeros(len(angles), skeleton.num_nodes, 3)
        ForwardK.forward(skeleton.getAnglesTensor(angles), full_pose,
                         torch.from_numpy(standard[f"{offset_key}_offsets_full"]), skeleton.roots)
        queue = list(skeleton.roots)
        while queue:
            node = queue.pop()
            for child in node.children:
                lengths = (full_pose[:, child.idx] - full_pose[:, node.idx]).norm(dim=-1)
                expected = float(standard[f"{offset_key}_median_lengths"][child.idx])
                torch.testing.assert_close(lengths, torch.full_like(lengths, expected), rtol=2e-4, atol=2e-6)
                queue.append(child)
    targets, masks = clip_targets(data, np.arange(8, 24))
    assert all(torch.isfinite(value).all() for value in targets.values())
    blank_targets, blank_masks = clip_targets(data, np.arange(32, 36))
    assert all(not value.any() for value in blank_masks.values())
    report = {"result": "PASS", "decoded_frames": 36, "appended_blank_frames_retained": 4,
              "detected_frames": {p: int(data[f"{p}_detected"].sum()) for p in ["body", "right", "left"]},
              "teacher_FK_max_abs_error": errors, "bone_lengths_invariant_under_changed_angles": True,
              "target_valid_elements": {k: int(v.sum()) for k, v in masks.items()},
              "all_missing_clip_has_no_supervision": True, "statistics_scope": "technical smoke only, not YT-SL corpus",
              "video_sha256": metadata["video_sha256"], "bones_sha256": sha(smoke / "bones.npz")}
    report["body_root_mode"] = teacher.body_root_mode
    (ROOT / "evidence/S2_pose_pipeline.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
