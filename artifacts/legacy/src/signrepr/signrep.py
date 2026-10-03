import ast
import importlib
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from signrepr.io import sha256


def constructor_literal(node):
    """Read literals and integer products in upstream config without executing it."""
    if isinstance(node, ast.Dict):
        return {constructor_literal(key): constructor_literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = constructor_literal(node.left), constructor_literal(node.right)
        if not isinstance(left, int) or not isinstance(right, int):
            raise ValueError('Constructor multiplication must use integer constants')
        return left * right
    return ast.literal_eval(node)


def windows(frame_count, segment_length, stride):
    if frame_count <= 0:
        raise ValueError('No decoded frames')
    if frame_count < segment_length:
        return [(0, frame_count, segment_length - frame_count)]
    return [(start, start + segment_length, 0) for start in range(0, frame_count - segment_length + 1, stride)]


class SignRepAdapter:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        upstream = self.root / config['upstream_path']
        commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != config['upstream_commit']:
            raise ValueError('Upstream commit differs from registered revision')
        dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        if dirty:
            raise ValueError('Upstream has changes; register patch before loading')
        checkpoint = self.root / config['checkpoint_path']
        if sha256(checkpoint) != config['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        tree = ast.parse((upstream / 'example_usage.py').read_text())
        params = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'model_params' for target in node.targets):
                params = constructor_literal(node.value)
        if params is None:
            raise ValueError('Cannot resolve official constructor parameters')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(upstream))
        model_cls = importlib.import_module('models.final_models.FINAL_hiera_latent_model_head_v25_active').Model
        transform_cls = importlib.import_module('augmentation.video.base_video_aug').Transformation
        self.model = model_cls(**params)
        if isinstance(self.model.cls_head, torch.nn.Identity):
            raise ValueError('Upstream silently skipped pretrained projection head')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        loaded = self.model.load_state_dict(state['model'], strict=True)
        self.device = torch.device(config['device'])
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.to(self.device).eval()
        self.transform = transform_cls()
        self.load_audit = {'strict': True, 'missing_keys': list(loaded.missing_keys),
                           'unexpected_keys': list(loaded.unexpected_keys), 'head_class': type(self.model.cls_head).__name__,
                           'total_parameters': sum(p.numel() for p in self.model.parameters()),
                           'trainable_parameters_for_extraction': 0, 'torch': torch.__version__,
                           'cuda_runtime': torch.version.cuda, 'upstream_commit': commit,
                           'checkpoint_sha256': config['checkpoint_sha256']}

    def extract(self, video):
        started = time.perf_counter()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError('DECODE_OPEN_FAILED')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        transformed = []
        first_rgb = None
        try:
            while True:
                valid, bgr = capture.read()
                if not valid:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if first_rgb is None:
                    first_rgb = rgb
                transformed.append(self.transform.transform_valid(image=rgb)['image'])
        finally:
            capture.release()
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('INVALID_OBSERVED_FPS')
        if not transformed:
            raise ValueError('DECODE_EMPTY')
        if declared_count > 0 and len(transformed) != declared_count:
            raise ValueError(f'DECODE_TRUNCATED:{len(transformed)}/{declared_count}')
        frames = torch.stack(transformed)
        window_list = windows(len(frames), self.config['segment_length'], self.config['stride_frames'])
        starts, ends, padding = zip(*window_list)
        embeddings, frame_masks = [], []
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        batch_size = self.config['batch_size']
        for offset in range(0, len(window_list), batch_size):
            tensors = []
            for start, end, pad in window_list[offset:offset + batch_size]:
                window = frames[start:end]
                if pad:
                    window = torch.cat([window, window[-1:].repeat(pad, 1, 1, 1)])
                tensors.append(window)
                frame_masks.append([True] * (end - start) + [False] * pad)
            batch = torch.stack(tensors).permute(0, 2, 1, 3, 4).to(self.device)
            with torch.inference_mode():
                output = self.model(batch)[self.config['output_layer']]
            embeddings.append(output.detach().cpu().numpy())
        features = np.concatenate(embeddings, axis=0)
        start_sec, end_sec = np.asarray(starts) / fps, np.asarray(ends) / fps
        duration = len(frames) / fps
        if features.shape != (len(window_list), self.config['expected_dimension']) or not np.isfinite(features).all():
            raise ValueError('INVALID_FEATURE_SHAPE_OR_VALUES')
        if np.any(end_sec > duration + 1e-6) or np.any(np.diff(start_sec) < 0):
            raise ValueError('INVALID_FEATURE_TIMESTAMPS')
        check = cv2.VideoCapture(str(video))
        ok, first_bgr = check.read()
        check.release()
        rgb_check = ok and np.array_equal(first_rgb, first_bgr[:, :, ::-1])
        direct = self.transform.aug_video([first_rgb])[0]
        transform_check = bool(torch.equal(frames[0], direct))
        mask = np.asarray(frame_masks, dtype=bool)
        return {'embeddings': features, 'window_start_sec': start_sec, 'window_end_sec': end_sec,
                'center_sec': (start_sec + end_sec) / 2, 'valid_mask': mask.any(axis=1),
                'valid_frame_mask': mask, 'valid_fraction': mask.mean(axis=1).astype(np.float32)}, {
                    'fps_observed': fps, 'num_frames_observed': len(frames), 'duration_sec': duration,
                    'width': width, 'height': height, 'windows': len(window_list),
                    'padded_frames': int(sum(padding)), 'elapsed_seconds': time.perf_counter() - started,
                    'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                    'rgb_check': bool(rgb_check), 'upstream_transform_equivalence': transform_check,
                    'representation_unit': '16-frame window', 'short_clip_caveat': 'Padded windows differ from native upstream, which drops short clips.' if sum(padding) else None}
