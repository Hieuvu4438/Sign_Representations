"""Synthetic pipeline fixtures, never evidence of language representation quality."""
from pathlib import Path
import numpy as np
from .schema import save_pose, sha256_file, write_manifest


def fixture_pose(frames=16, seed=42, variant="clean"):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.25, 0.75, (frames, 133, 2)).astype(np.float32)
    xy[:, 5], xy[:, 6] = [0.4, 0.5], [0.6, 0.5]
    xy[:, 91:112] = xy[:, 91:92] + rng.normal(0, 0.025, (frames, 21, 2))
    xy[:, 112:133] = xy[:, 112:113] + rng.normal(0, 0.025, (frames, 21, 2))
    observed = np.ones((frames, 133), bool)
    if variant == "missing":
        observed[frames//2] = False
    elif variant == "all_missing":
        observed[:] = False
    elif variant == "one_part_missing":
        observed[:, 91:112] = False
    return {"xy": xy, "confidence": np.full((frames, 133), 0.9, np.float32), "observed": observed,
            "timestamps_sec": np.arange(frames, dtype=np.float64)/25,
            "timing_known": np.array(variant != "unknown_timing"),
            "frame_indices": np.arange(frames, dtype=np.int64), "track_continuity": np.ones(frames-1, bool)}


def make_fixture(output, samples=16, seed=42):
    if samples < 8:
        raise ValueError("Fixture requires at least two distinct clips per four arbitrary classes")
    root = Path(output).resolve()
    rows = []
    for i in range(samples):
        n = (1, 2, 15, 16, 23)[i % 5]
        variant = ("clean", "clean", "missing", "one_part_missing", "unknown_timing")[i % 5]
        pose = fixture_pose(n, seed+i, variant)
        path = root / "poses" / f"sample_{i:04d}.npz"
        save_pose(path, pose)
        rows.append({"sample_id": f"synthetic/train/{i:04d}", "corpus": "synthetic", "language": "synthetic",
                     "split": "train", "source_video_id": f"source_{i:04d}", "signer_id": f"synthetic_signer_{(i//4)%4}",
                     "pose_path": str(path), "video_path": None, "num_frames": n, "fps": 25 if bool(pose["timing_known"]) else None,
                     "timing_source": "synthetic_clock" if bool(pose["timing_known"]) else "unknown",
                     "coordinate_system": "viewport_normalized", "joint_layout": "coco_wholebody_133",
                     "extractor_profile": "synthetic_fixture", "extractor_weights_sha256": None,
                     "gloss_id": str(i%4), "gloss_namespace": "synthetic:synthetic",
                     "gloss_sequence": [str(i%4)], "translation": f"arbitrary fixture {i%4}",
                     "pose_sha256": sha256_file(path), "synthetic": True})
    write_manifest(root/"manifest.jsonl", rows)
    # Independent synthetic clips for downstream smoke selection/evaluation.
    for split, offset in [('val', 10000), ('test', 20000)]:
        heldout = []
        for i in range(8):
            n = (5, 7, 16, 23)[i%4]
            path = root/'poses'/split/f'sample_{i:04d}.npz'
            save_pose(path, fixture_pose(n, seed+offset+i))
            template = rows[i%len(rows)]
            heldout.append({**template, 'sample_id': f'synthetic/{split}/{i:04d}', 'split': split,
                            'source_video_id': f'{split}_source_{i:04d}', 'pose_path': str(path), 'num_frames': n,
                            'fps': 25., 'timing_source': 'synthetic_clock', 'pose_sha256': sha256_file(path),
                            'signer_id': f'synthetic_{split}_signer_{i//4}', 'gloss_id': str(i%4),
                            'gloss_sequence': [str(i%4)], 'translation': f'arbitrary fixture {i%4}'})
        write_manifest(root/f'{split}.jsonl', heldout)
    import json
    (root/'validation_protocol.json').write_text(json.dumps({'gallery_manifest': 'manifest.jsonl', 'query_manifest': 'val.jsonl',
                                                            'protocol': 'in_domain_ssl', 'model_selection_scope': 'target_validation',
                                                            'synthetic': True}, indent=2))
    return root/"manifest.jsonl"
