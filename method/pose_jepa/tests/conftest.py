from pathlib import Path
import torch
import pytest
import yaml
from signrepr.data.fixture import fixture_pose
from signrepr.data.normalization import canonicalize_sample
from signrepr.data.collate import collate_pose


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(7)


@pytest.fixture
def config():
    cfg = yaml.safe_load((Path(__file__).parents[1]/"configs/ssl_pose_jepa.yaml").read_text())
    cfg['model'].update(encoder_dim=16, encoder_layers=2, encoder_heads=4, encoder_ffn_dim=32,
                        predictor_dim=8, predictor_layers=1, predictor_heads=2, predictor_ffn_dim=16,
                        latent_dim=16, sequence_dim=64, spatial_channels=[8, 8, 16, 16], anchor_dim=8)
    cfg['optim'].update(max_steps=20, warmup_steps=2, gradient_accumulation_steps=1, per_rank_batch=2, precision='fp32')
    return cfg


@pytest.fixture
def batch():
    return collate_pose([canonicalize_sample(fixture_pose(16, seed=s)) for s in (1, 2)])
