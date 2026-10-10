import json
import numpy as np
import pytest
import torch
from signrep_baseline.augmentation import VideoTransform, AugmentationConfig
from signrep_baseline.data import youtube_id, deduplicate_manifest, StatefulPairSampler
from signrep_baseline.optim import WarmupCosine, layerwise_parameter_groups
from signrep_baseline.upstream_signrep.augmentation.video.base_video_aug import Transformation


def test_youtube_dedup_covers_all_splits_and_url_variants(tmp_path):
    source, annotations, output = [tmp_path / name for name in ["input.jsonl", "wlasl.json", "dedup.jsonl"]]
    source.write_text(''.join(json.dumps({"video_id": video,"youtube_id": video,"split": "pretrain"})+'\n' for video in ['train','val','test','kept']))
    annotations.write_text(json.dumps([{"instances": [{"url": "https://youtu.be/train?t=4", "split": "train"},
        {"url": "https://www.youtube.com/watch?v=val", "split": "val"}, {"url": "https://youtube.com/embed/test", "split": "test"}]}]))
    report = deduplicate_manifest(source, annotations, output)
    assert report['retained'] == 1 and report['excluded_ids'] == ['test', 'train', 'val']
    assert json.loads(output.read_text())['video_id'] == 'kept'
    assert youtube_id('https://youtube.com/shorts/abc') == 'abc'


def test_temporally_coherent_augmentation_and_exact_author_inference():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)
    frames = np.repeat(frame[None], 16, axis=0)
    config = AugmentationConfig(planckian_probability=1., blur_probability=1., grayscale_probability=1.)
    augmented = VideoTransform(True, config)(frames)
    assert augmented.shape == (3, 16, 224, 224)
    assert torch.equal(augmented[:, 0], augmented[:, -1])
    actual = VideoTransform(False)(frames)
    expected = Transformation().aug_video(frames).permute(1, 0, 2, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sampler_resume_unique_batches_and_warmup_cosine_endpoints():
    sampler = StatefulPairSampler(7, 3, seed=3)
    assert len(set(sampler.next_indices())) == 3
    state = sampler.state_dict()
    expected = [sampler.next_indices() for _ in range(4)]
    restored = StatefulPairSampler(7, 3, seed=40)
    restored.load_state_dict(state)
    assert [restored.next_indices() for _ in range(4)] == expected
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    schedule = WarmupCosine(optimizer, 1e-4, 50000, 500000)
    assert schedule.rate(1) == 1e-4/50000
    assert schedule.rate(50000) == 1e-4
    assert schedule.rate(500000) == 0
    assert 0 < schedule.rate(275000) < schedule.rate(50001) < 1e-4
    with pytest.raises(ValueError):
        schedule.before_update(2)


def test_layerwise_decay_covers_every_parameter_once():
    # A small independent architecture has the same naming contract, without Hiera cost.
    class Mini(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.patch_embed = torch.nn.Linear(2, 2)
            self.backbone.pos_embed = torch.nn.Parameter(torch.ones(1, 2))
            self.backbone.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])
            self.fc = torch.nn.LayerNorm(2)
    model = Mini()
    groups = layerwise_parameter_groups(model)
    ids = [id(p) for group in groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    by_name = {name: group for group in groups for name in group['parameter_names']}
    assert by_name['backbone.patch_embed.weight']['lr_scale'] == .85**3
    assert by_name['backbone.blocks.1.weight']['lr_scale'] == .85
    assert by_name['fc.weight']['weight_decay'] == 0
    assert by_name['backbone.pos_embed']['weight_decay'] == 0
