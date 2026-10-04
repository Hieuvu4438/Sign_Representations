"""Frozen, read-only SignRep wrapper with explicit temporal provenance."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch

from scripts.audit_signrep import IntegerProducts, sha
from temporal import window_plan

REPO = Path(__file__).resolve().parents[2]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_model(checkpoint, device='cpu'):
    upstream = REPO/'third_party/SignRep'
    example = upstream/'example_usage.py'
    assignments = [n for n in ast.walk(ast.parse(example.read_text()))
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model_params' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Ambiguous upstream model parameters')
    params = ast.literal_eval(IntegerProducts().visit(assignments[0].value))
    sys.path.insert(0, str(upstream))
    cls = importlib.import_module('models.final_models.FINAL_hiera_latent_model_head_v25_active').Model
    model = cls(**params)
    if not all(hasattr(model, key) for key in ('cls_head', 'fc', 'active_head')):
        raise RuntimeError('Upstream silently skipped an expected head')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
    return model.to(device).eval(), params


def decode_video(path, max_frames=500):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)]
    metadata = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout)
    pts = [float(f['best_effort_timestamp_time']) for f in metadata['frames']]
    if not 1 <= len(pts) <= max_frames:
        raise ValueError('Clip exceeds bounded decode contract or is empty')
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(frames) > max_frames:
                raise ValueError('Decoded frame limit exceeded')
    finally:
        cap.release()
    if len(frames) != len(pts):
        raise ValueError('OpenCV/ffprobe frame count mismatch')
    return frames, pts, command


def preprocess_rgb(frames):
    transform = A.Compose([A.Resize(224, 224, p=1),
                           A.Normalize(mean=MEAN, std=STD, max_pixel_value=255, p=1),
                           ToTensorV2(p=1)])
    if not frames or any(f.dtype != np.uint8 or f.ndim != 3 or f.shape[-1] != 3 for f in frames):
        raise ValueError('Expected nonempty uint8 RGB frames')
    return torch.stack([transform(image=f)['image'] for f in frames])


def infer_windows(model, frames, pts, selected_indices=None, length=16, stride=2, batch_size=2):
    if len(frames) != len(pts) or tuple(frames.shape[1:]) != (3, 224, 224):
        raise ValueError('Input tensor/PTS shape mismatch')
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    plan = window_plan(pts, length, stride)
    indices = list(range(len(plan))) if selected_indices is None else list(selected_indices)
    if not indices or len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 or i >= len(plan) for i in indices):
        raise ValueError('Invalid selected windows')
    if model.training:
        raise ValueError('Frozen inference requires eval mode')
    device = next(model.parameters()).device
    outputs = {key: [] for key in ('features', 'latent', 'cls_active')}
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset:offset+batch_size]
            inputs = torch.stack([frames[plan[i]['start_frame']:plan[i]['stop_frame_exclusive']]
                                  for i in batch_indices]).permute(0, 2, 1, 3, 4).contiguous().to(device)
            result = model(inputs)
            for key, dim in (('features', 768), ('latent', 768), ('cls_active', 2)):
                value = result[key]
                if tuple(value.shape) != (len(batch_indices), dim) or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid encoder output: {key}')
                outputs[key].append(value.detach().cpu())
    return {key: torch.cat(values) for key, values in outputs.items()}, [plan[i] for i in indices]
