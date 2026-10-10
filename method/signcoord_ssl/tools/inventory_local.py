"""Read-only inventory, bounded train pose/video inspections, exact asset hashes."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import pickle
import subprocess
import sys
from pathlib import Path

import av
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))
from signcoord.provenance import atomic_json, execution_stamp, sha256_file


def inspect_cache(path, video):
    with path.open("rb") as f:
        data = pickle.load(f)
    xy_frames = [np.asarray(v) for v in data["keypoints"]]
    score_frames = [np.asarray(v) for v in data["scores"]]
    xy = np.concatenate([v.reshape(-1,2) for v in xy_frames],axis=0)
    scores = np.concatenate([v.reshape(-1) for v in score_frames],axis=0)
    people = [v.shape[0] for v in xy_frames]
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        pts = [float(frame.pts*frame.time_base) for frame in container.decode(stream)]
        w, h = stream.width, stream.height
    return {"pose_path": str(path), "video_path": str(video), "pose_sha256": sha256_file(path),
       "video_sha256": sha256_file(video), "xy_frame_shapes": sorted({str(v.shape) for v in xy_frames}),
       "score_frame_shapes": sorted({str(v.shape) for v in score_frames}),
       "person_count_histogram": {str(n):people.count(n) for n in sorted(set(people))},
       "xy_quantiles": np.nanquantile(xy, [0, .01, .5, .99, 1]).tolist(),
       "score_quantiles": np.nanquantile(scores, [0, .01, .5, .99, 1]).tolist(),
       "score_fraction_above_one": float((scores > 1).mean()),
       "nonfinite_xy": int((~np.isfinite(xy)).sum()), "decoded_frames": len(pts),
       "frame_count_matches": len(pts) == len(xy_frames), "video_size": [w, h],
       "pts_strictly_increasing": bool(np.all(np.diff(pts) > 0)),
       "median_pts_interval_s": float(np.median(np.diff(pts))) if len(pts)>1 else None,
       "reuse_status": "unverified_coordinate_units_score_semantics_schema_and_extractor"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(PROJECT/"docs/evidence/local_inventory.json"))
    p.add_argument("--inspect-per-source", type=int, default=6)
    args = p.parse_args()
    repo = PROJECT.parents[1]
    sources = {"how2sign_train": (Path("/home/shared_data/sign_language/How2Sign/train/raw_videos"),
                  Path("/home/shared_data/sign_language/How2Sign/train/train_pose")),
               "asl_citizen": (Path("/home/dongvk/datasets/ASL_Citizen/videos"), Path("/home/dongvk/datasets/ASL_Citizen/keypoints")),
               "wlasl2000": (Path("/home/dongvk/datasets/WLASL2000/WLASL2000"), Path("/home/dongvk/datasets/WLASL2000/keypoint"))}
    out = {**execution_stamp(), "datasets": {}, "third_party": {}, "checkpoints": [],
           "environment": {"python": sys.version, "executable": sys.executable}, "inspections": []}
    for package in ("torch", "numpy", "scipy", "av", "PyYAML", "pytest", "timm", "huggingface-hub"):
        try:
            out["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out["environment"][package] = None
    out["gpu_snapshot"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    for name, (video_root, pose_root) in sources.items():
        videos = {p.stem: p for p in video_root.glob("*.mp4")}
        poses = {p.stem: p for p in pose_root.glob("*.pkl")}
        out["datasets"][name] = {"video_root": str(video_root), "pose_root": str(pose_root),
             "available_videos": len(videos), "available_pose_files": len(poses),
             "paired_by_filename": len(videos.keys() & poses.keys()), "counts_are_not_decode_verification": True}
        if name == "asl_citizen":
            split_ids = {}
            for split in ("train", "val", "test"):
                path = video_root.parent/"splits"/(split+".csv")
                rows = list(csv.DictReader(path.open()))
                split_ids[split] = {r["Participant ID"] for r in rows}
                out["datasets"][name][split] = {"declared": len(rows), "available": sum(Path(r["Video file"]).stem in videos for r in rows),
                         "signer_count": len(split_ids[split]), "csv_sha256": sha256_file(path)}
            out["datasets"][name]["split_signer_intersections"] = {
                 a+"_"+b: sorted(split_ids[a] & split_ids[b]) for a,b in (("train", "val"),("train", "test"),("val", "test"))}
            train_names = sorted(Path(r["Video file"]).stem for r in csv.DictReader((video_root.parent/"splits/train.csv").open()))
        else:
            train_names = sorted(videos.keys() & poses.keys()) if name == "how2sign_train" else []
        for stem in [s for s in train_names if s in poses and s in videos][:args.inspect_per_source]:
            try:
                out["inspections"].append({"dataset": name, **inspect_cache(poses[stem], videos[stem])})
            except Exception as exc:
                out["inspections"].append({"dataset": name, "sample": stem, "error": str(exc)})
    # Remaining paths explicitly listed in docs/datasets; inventory media and
    # metadata availability without decoding/reading held-out pose contents.
    extra_roots={
      "csldaily_shared":"/home/shared_data/sign_language/CSLDaily/videos/videos",
      "csldaily_crop":"/home/dongvk/datasets/CSL_Daily_Sentence_Crop/videos",
      "how2sign_eval":"/home/shared_data/sign_language/How2Sign/eval/raw_videos",
      "how2sign_test":"/home/shared_data/sign_language/How2Sign/test/raw_videos",
      "msasl_downloaded":"/home/dongvk/datasets/MS-ASL/video",
      "msasl_processed":"/home/dongvk/datasets/MS-ASL/cut_processed_video",
      "phoenix14t_train":"/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/train",
      "phoenix14t_dev":"/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/dev",
      "phoenix14t_test":"/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/test",
      "youtube_asl":"/home/dongvk/datasets/youtubeASL"}
    for name,root in extra_roots.items():
        path=Path(root)
        out["datasets"][name]={"video_root":root,"exists":path.is_dir(),
           "available_mp4":sum(1 for _ in path.rglob('*.mp4')) if path.is_dir() else 0,
           "counts_are_not_decode_verification":True}
    out["method_inventory"]={path.name:{"path":str(path),"python_file_count":sum(1 for _ in path.rglob('*.py')),
                   "agent_instructions":str(path/'AGENTS.md') if (path/'AGENTS.md').exists() else None}
                   for path in (repo/'method').iterdir() if path.is_dir()}
    for path in sorted((repo/"third_party").iterdir()):
        if path.is_dir():
            rev = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True)
            out["third_party"][path.name] = {"revision": rev.stdout.strip(), "path": str(path)}
    for path in sorted((repo/"checkpoints").rglob("*")):
        if path.is_file() and path.suffix in (".pt", ".pth", ".bin", ".safetensors", ".task"):
            # Full hashing of reused SignRep, bounded metadata inventory for unrelated weights.
            out["checkpoints"].append({"path": str(path), "bytes": path.stat().st_size,
                  "sha256": sha256_file(path) if "signrep" in str(path).lower() else None})
    atomic_json(args.output, out)
    print(json.dumps({"output": args.output, "datasets": out["datasets"], "inspected": len(out["inspections"])}))


if __name__ == "__main__":
    main()
