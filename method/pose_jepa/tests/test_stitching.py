import torch
from torch import nn
from signrepr.data.fixture import fixture_pose
from signrepr.data.normalization import canonicalize_sample
from signrepr.models.part_tokenizer import token_metadata
from signrepr.evaluation.extract import extract_full_sequence
from signrepr.evaluation.extract import encode_full_sequence_train
from signrepr.models.pose_encoder import PoseEncoder


class PositionOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('dummy', nn.Parameter(torch.zeros(())))
        self.config = {'encoder_dim': 1}

    def encode_full(self, batch):
        meta = token_metadata(batch)
        features = meta['position']+meta['indices']%4
        features = torch.where(meta['valid'], features, 0)[..., None]
        return {'features': features, **meta}


def test_stitch_unique_segments_odd_tail_internal_missing_source_offsets():
    pose = fixture_pose(139)
    pose['frame_indices'] = pose['frame_indices']*3+101
    pose['observed'][62:64] = False
    sample = canonicalize_sample(pose)
    result = extract_full_sequence(PositionOracle(), sample, 128, 64)
    assert result['length'] == 70
    expected = torch.tensor([i+.5 for i in range(0, 138, 2)]+[138.])[:, None]+torch.arange(4)[None]
    expected[31] = 0
    torch.testing.assert_close(result['features'], expected, atol=1e-5, rtol=1e-5)
    valid = torch.ones(70, dtype=torch.bool); valid[31] = False
    torch.testing.assert_close(result['raw_embedding'], expected[valid].mean(0))
    assert result['source_frame_indices'][32].tolist() == [293, 296]
    assert result['source_frame_indices'][-1].tolist() == [515, -1]
    assert not result['observed_time_mask'][31]


def test_long_sequence_stitch_is_differentiable_and_matches_inference(config):
    encoder = PoseEncoder(config['model']).eval()
    sample = canonicalize_sample(fixture_pose(139))
    features, valid = encode_full_sequence_train(encoder, sample)
    reference = extract_full_sequence(encoder, sample)
    torch.testing.assert_close(features.detach(), reference['features'], atol=1e-6, rtol=1e-5)
    assert features.shape[0] == 70
    features[-1, 0].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
