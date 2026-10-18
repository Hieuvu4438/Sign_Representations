"""Full-sample aligned window stitching before unique-segment clip pooling."""
import torch
from ..data.normalization import select_frames
from ..data.collate import collate_pose, to_device


def encode_full_sequence_train(encoder, sample, window_frames=128, stride_frames=64):
    """Differentiable full-utterance stitching for supervised encoder fine-tuning."""
    if window_frames % 2 or stride_frames % 2 or not 0 < stride_frames <= window_frames:
        raise ValueError('Window and stride must be positive, segment-aligned, without gaps')
    device = next(encoder.parameters()).device
    t, d = sample['length'], encoder.config['encoder_dim']
    k = (t+1)//2
    sums = torch.zeros((k, 4, d), device=device)
    counts = torch.zeros((k, 4), device=device)
    start = 0
    while True:
        end = min(start+window_frames, t)
        window = select_frames(sample, torch.arange(start, end))
        encoded = encoder.encode_full(to_device(collate_pose([window]), device))
        features = encoded['features'][0].reshape(-1, 4, d)
        valid = encoded['valid'][0].reshape(-1, 4)
        indices = torch.arange(start//2, start//2+len(features), device=device)
        sums = sums.index_add(0, indices, features*valid[..., None])
        counts = counts.index_add(0, indices, valid.to(counts.dtype))
        if end >= t: break
        start += stride_frames
    return (sums/counts.clamp_min(1)[..., None]).flatten(1), counts > 0


@torch.inference_mode()
def extract_full_sequence(encoder, sample, window_frames=128, stride_frames=64):
    if window_frames % 2 or stride_frames % 2 or not 0 < stride_frames <= window_frames:
        raise ValueError('Window and stride must be positive, segment-aligned, without gaps')
    encoder.eval()
    device = next(encoder.parameters()).device
    t, d = sample['length'], encoder.config['encoder_dim']
    k = (t+1)//2
    sums = torch.zeros((k, 4, d), device=device)
    counts = torch.zeros((k, 4), device=device)
    start = 0
    while True:
        end = min(start+window_frames, t)
        window = select_frames(sample, torch.arange(start, end))
        encoded = encoder.encode_full(to_device(collate_pose([window]), device))
        feature = encoded['features'][0].reshape(-1, 4, d)
        valid = encoded['valid'][0].reshape(-1, 4)
        offset = start//2
        sums[offset:offset+len(feature)] += feature*valid[..., None]
        counts[offset:offset+len(feature)] += valid
        if end >= t:
            break
        start += stride_frames
    features = sums/counts.clamp_min(1)[..., None]
    valid = counts > 0
    real = torch.arange(t)
    times = torch.zeros(k, dtype=torch.float64)
    source_indices = torch.full((k, 2), -1, dtype=torch.long)
    for i in range(k):
        ids = real[2*i:min(2*i+2, t)]
        times[i] = sample['timestamps_sec'][ids].mean()
        source_indices[i, :len(ids)] = sample['frame_indices'][ids]
    raw = ((features*valid[..., None]).sum(0)/valid.sum(0).clamp_min(1)[..., None]).flatten()
    return {'features': features.flatten(1).cpu(), 'length': k, 'lengths': torch.tensor([k]), 'timestamps_sec': times,
            'timing_known': sample['timing_known'], 'source_frame_indices': source_indices,
            'segment_indices': torch.arange(k), 'part_valid': valid.cpu(),
            'observed_time_mask': valid.any(-1).cpu(), 'has_observation': bool(valid.any()), 'raw_embedding': raw.cpu(),
            'status': 'ok' if valid.any() else sample['status'] if sample['status'] != 'ok' else 'no_observation',
            'stitch_version': 'aligned_unique_segment_mean_v1'}
