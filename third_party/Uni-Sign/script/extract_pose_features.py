#!/usr/bin/env python3
"""Extract whole-body poses and pretrained Uni-Sign embeddings from one clip."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from pose_preprocessing import preprocess_pose_sequence, uniform_frame_indices
from serving.pose_runner import PoseRunner


def video_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--pose", type=Path, help="Trusted Uni-Sign pose pickle")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--mode", choices=("performance", "balanced", "lightweight"), default="performance")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pose-only", action="store_true", help="Skip the Uni-Sign encoder")
    parser.add_argument("--predict", action="store_true", help="Also generate a sign label using the checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained_weight/unisign/wlasl_pose_only_islr.pth")
    parser.add_argument("--mt5-path", type=Path, default=ROOT / "pretrained_weight/mt5-base")
    parser.add_argument("--dataset", default="WLASL", choices=("WLASL", "CoSign", "CSL_Daily", "CSL_News", "How2Sign", "OpenASL"))
    parser.add_argument("--language", default="", help="Defaults to the language of --dataset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.predict and args.pose_only:
        parser.error("--predict requires the Uni-Sign encoder; remove --pose-only")
    source_path = args.video or args.pose
    if not source_path.is_file():
        parser.error(f"Input does not exist: {source_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")
    if not args.pose_only:
        for path in (args.checkpoint, args.mt5_path):
            if not path.exists():
                parser.error(f"Required model asset does not exist: {path}")
    if not args.overwrite and any((args.output_dir / name).exists() for name in ("pose.pkl", "features.npz", "result.json")):
        parser.error("Output already exists; choose a new directory or pass --overwrite")
    if args.overwrite and args.pose_only:
        # A pose-only rerun must not leave features from a previous clip.
        (args.output_dir / "features.npz").unlink(missing_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = {"input": str(source_path.resolve()), "device": args.device, "pose_mode": args.mode}

    if args.video:
        from rtmlib import Wholebody

        estimator = Wholebody(mode=args.mode, backend="onnxruntime", device=args.device, to_openpose=False)
        providers = {name: getattr(estimator, name).session.get_providers() for name in ("det_model", "pose_model")}
        if args.device == "cuda" and any(p[0] != "CUDAExecutionProvider" for p in providers.values()):
            raise RuntimeError(f"ONNX Runtime fell back from CUDA: {providers}")
        sequence = PoseRunner(estimator).extract(video_frames(args.video))
        pose = sequence.pose
        result["onnx_providers"] = providers
        result["quality"] = asdict(sequence.quality)
        result["quality"]["primary_person_coverage"] = sequence.quality.primary_person_coverage
        result["pose_models"] = {name: getattr(estimator, name).onnx_model for name in providers}
    else:
        with args.pose.open("rb") as stream:
            pose = pickle.load(stream)
        result["pose_mode"] = None  # A pickle does not identify its extractor.

    model_input = preprocess_pose_sequence(pose, max_length=args.max_length)
    indices = uniform_frame_indices(len(pose["scores"]), args.max_length)
    result["pose_frames"] = len(pose["scores"])
    result["sampled_frames"] = len(indices)
    result["pose_seconds"] = round(time.perf_counter() - started, 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pose.pkl").open("wb") as stream:
        pickle.dump(pose, stream)

    if not args.pose_only:
        from models import Uni_Sign

        model_args = SimpleNamespace(hidden_dim=256, dataset=args.dataset, task="ISLR", rgb_support=False,
                                     label_smoothing=0.0, mt5_path=str(args.mt5_path.resolve()))
        if args.language:
            model_args.language = args.language
        model = Uni_Sign(model_args)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        del checkpoint
        model.eval().to(args.device)
        model_input = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in model_input.items()}
        with torch.inference_mode():
            encoded = model.encode(model_input)
            # encode() prepends text prompt tokens. Save only the pose frames.
            embeddings = encoded["inputs_embeds"][0, -len(indices):].float().cpu().numpy()
            if not np.isfinite(embeddings).all():
                raise RuntimeError("The pose encoder produced non-finite features")
            np.savez_compressed(args.output_dir / "features.npz", pose_embeddings=embeddings,
                                pooled_embedding=embeddings.mean(axis=0), sampled_pose_indices=indices)
            if args.predict:
                generated = model.generate(encoded, max_new_tokens=20, num_beams=4)
                result["prediction"] = model.mt5_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                result["prediction_method"] = "open_vocabulary_generation"
        result["checkpoint"] = str(args.checkpoint.resolve())
        result["dataset"] = args.dataset
        result["language"] = model.lang
        result["feature_shape"] = list(embeddings.shape)
        if args.device == "cuda":
            torch.cuda.synchronize()
            result["torch_peak_memory_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    result["total_seconds"] = round(time.perf_counter() - started, 3)
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
