"""Measure actual inference parity, dimensions and cached-pose latency."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from signrepr.export.api import SignRepresentationModel
from signrepr.data.schema import load_pose, sha256_file
from signrepr.data.normalization import canonicalize_sample
from signrepr.evaluation.extract import extract_full_sequence
from signrepr.training.islr import load_selected_encoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--pose', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args(); torch.set_num_threads(2)
    model = SignRepresentationModel(args.bundle, args.device)
    encoder, _ = load_selected_encoder(args.checkpoint, args.device, model.metadata['selected_encoder'])
    pose = load_pose(args.pose); cfg = model.config['data']; ev = model.config['evaluation']
    sample = canonicalize_sample(pose, confidence_threshold=cfg['confidence_threshold'], min_part_valid_fraction=cfg['min_part_valid_fraction'], coordinate_profile=cfg['coordinate_profile'])
    ref = extract_full_sequence(encoder, sample, ev['window_frames'], ev['window_stride_frames'])
    out = model.encode_sequence(pose)
    torch.testing.assert_close(ref['features'], out['features'], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ref['raw_embedding'], model.encode_clip(pose)['embedding'], atol=1e-6, rtol=1e-5)
    for _ in range(2): model.encode_sequence(pose)
    timings = []
    for _ in range(args.repeats):
        if args.device.startswith('cuda'): torch.cuda.synchronize()
        started = time.perf_counter(); model.encode_sequence(pose)
        if args.device.startswith('cuda'): torch.cuda.synchronize()
        timings.append(time.perf_counter()-started)
    report = {'parity_verified': True, 'atol': 1e-6, 'rtol': 1e-5, 'device': args.device,
              'sequence_shape': list(out['features'].shape), 'raw_clip_dim': len(out['raw_embedding']),
              'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
              'cached_pose_sequence_latency_mean_sec': sum(timings)/len(timings), 'repeats': len(timings),
              'latency_scope': 'canonicalization, pose encoder, aligned window stitching; detector not executed',
              'detector_latency_sec': None, 'checkpoint_sha256': sha256_file(args.checkpoint),
              'bundle_sha256': model.metadata['bundle_sha256'], 'selected_encoder': model.metadata['selected_encoder'],
              'training_objective': model.metadata['method']}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__': main()
