import copy
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from signrep_baseline.data import StatefulPairSampler
from signrep_baseline.job_support import (OrderedPairLoader, chunked_retrieval,
                                          trim_metrics, rotate_previous)
from signrep_baseline.model import SignRep
from signrep_baseline.retrieval import dictionary_retrieval, extract_windows
from signrep_baseline.schema import PRIOR_SHAPES
from signrep_baseline.training import Pretrainer, seed_everything


class RandomPairs:
    def __len__(self):
        return 19

    def __getitem__(self, index):
        values = torch.rand(2, 4) + np.random.rand() + random.random() + index
        return {"images": values, "targets": {"x": values.clone()},
                "masks": {"x": torch.ones_like(values, dtype=torch.bool)},
                "video_ids": [str(index)] * 2,
                "frame_indices": torch.full((2, 16), index, dtype=torch.long)}


def equal_batch(left, right):
    assert left["video_ids"] == right["video_ids"]
    for key in ("images", "frame_indices"):
        assert torch.equal(left[key], right[key])
    for key in ("targets", "masks"):
        for name in left[key]:
            assert torch.equal(left[key][name], right[key][name])


def test_prefetched_workers_resume_committed_sampler_and_identical_batches():
    torch.set_num_threads(2)
    dataset = RandomPairs()
    serial_sampler = StatefulPairSampler(19, 3, 101)
    parallel_sampler = StatefulPairSampler(19, 3, 101)
    serial = OrderedPairLoader(dataset, serial_sampler, 101, workers=0)
    parallel = OrderedPairLoader(dataset, parallel_sampler, 101, workers=2, prefetch=2)
    try:
        before = torch.random.get_rng_state().clone()
        for step in (1, 2):
            equal_batch(serial.next_batch(), parallel.next_batch())
            serial.commit(step)
            parallel.commit(step)
        assert torch.equal(before, torch.random.get_rng_state())
        saved = copy.deepcopy(parallel_sampler.state_dict())
        expected = parallel.next_batch()
        # Simulate forced interruption: the queue already planned later batches,
        # but the checkpoint contains only the two completed optimizer updates.
        parallel.close()
        restored_sampler = StatefulPairSampler(19, 3, 999)
        restored_sampler.load_state_dict(saved)
        resumed = OrderedPairLoader(dataset, restored_sampler, 101, completed_steps=2,
                                    workers=2, prefetch=2)
        try:
            equal_batch(expected, resumed.next_batch())
            resumed.commit(3)
            assert torch.equal(restored_sampler.order, parallel_sampler.order)
            assert restored_sampler.cursor == 9
        finally:
            resumed.close()
    finally:
        serial.close()
        parallel.close()


class TinySignRep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_ratio = .8
        self.backbone = torch.nn.Module()
        self.backbone.patch_embed = torch.nn.Linear(4, 768)
        self.backbone.blocks = torch.nn.ModuleList([])
        self.fc = torch.nn.Linear(768, 768)
        self.heads = torch.nn.ModuleDict({k: torch.nn.Linear(768, int(np.prod(v)))
                                         for k, v in PRIOR_SHAPES.items()})
        self.active_head = torch.nn.Linear(768, 2)

    def forward(self, images):
        features = self.fc(self.backbone.patch_embed(images))
        features = torch.nn.functional.dropout(features, .1, self.training)
        return {"features": features, "tokens": features[:, None, :],
                "cls_head": {k: h(features)[:, None, :] for k, h in self.heads.items()},
                "cls_active": self.active_head(features)}


def model_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_complete_trainer_resume_restores_optimizers_rng_and_update(tmp_path):
    torch.set_num_threads(2)
    seed_everything(23)
    trainer = Pretrainer(TinySignRep(), total_steps=4, warmup_steps=1)
    sampler = StatefulPairSampler(19, 2, 23)
    targets = {k: torch.zeros(4, 1, int(np.prod(v))) for k, v in PRIOR_SHAPES.items()}
    targets["activity"] = torch.zeros(4, 2)
    batch = {"images": torch.randn(4, 4), "targets": targets,
             "masks": {k: torch.ones_like(v, dtype=torch.bool) for k, v in targets.items()},
             "video_ids": ["a", "a", "b", "b"]}
    trainer.update(batch)
    sampler.next_indices()
    contract = {"scope": "technical_resume_test"}
    path = tmp_path / "last.pt"
    (tmp_path / "metrics.jsonl").write_text('{"step":1}\n')
    trainer.save(path, sampler, contract)
    expected_metrics = trainer.update(batch)
    expected = {k: model_state(getattr(trainer, k)) for k in ("model", "discriminator", "ema")}
    restored = Pretrainer(TinySignRep(), total_steps=4, warmup_steps=1)
    restored_sampler = StatefulPairSampler(19, 2, 999)
    restored.resume(path, restored_sampler, contract)
    actual_metrics = restored.update(batch)
    assert actual_metrics == expected_metrics
    for key, state in expected.items():
        assert all(torch.equal(v, getattr(restored, key).state_dict()[k]) for k, v in state.items())
    assert restored.scheduler.completed_steps == restored.d_scheduler.completed_steps == 2
    assert restored_sampler.cursor == sampler.cursor
    assert restored.optimizer.state_dict()["state"]
    assert restored.d_optimizer.state_dict()["state"]


class PixelEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frames = 16
        self.backbone = torch.nn.Module()
        self.backbone.mask_spatial_shape = (1, 1, 1)
        self.active_head = torch.nn.Linear(768, 2)

    def encode(self, images, **kwargs):
        values = images.mean((2, 3, 4))
        return {"features": values.repeat(1, 256)}


def test_unique_frame_preprocessing_preserves_every_window(tmp_path):
    torch.set_num_threads(2)
    path = str(tmp_path / "video.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    rng = np.random.default_rng(10)
    for _ in range(35):
        writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    writer.release()
    model = PixelEncoder().eval()
    old = extract_windows(model, path, batch_size=4, preprocess_once=False)
    new = extract_windows(model, path, batch_size=4, preprocess_once=True)
    for key in ("features", "activity", "window_starts"):
        np.testing.assert_array_equal(old[key], new[key])


def test_chunked_full_dictionary_matches_original_ties_and_metrics():
    gallery = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [-1., 0.], [0., -1.]])
    labels = [9, 0, 9, 5, 5]
    query = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    truth = [0, 9, 5]
    expected, detail = dictionary_retrieval(gallery, labels, query, truth,
                                          list("abcde"), list("xyz"), split="val")
    actual, records = chunked_retrieval(gallery, labels, query, truth, chunk=2)
    for key in ("class_ranking", "nearest_video_indices"):
        assert torch.equal(records[key], detail[key])
    for key in ("per_instance_top1", "per_instance_top5", "per_class_top1", "per_class_top5", "MRR", "DCG"):
        assert actual[key] == expected[key]


def test_history_recovers_partial_write_and_discards_uncheckpointed_updates(tmp_path):
    history = tmp_path / "metrics.jsonl"
    history.write_text('{"step":1}\n{"step":2}\n{"step":3}\n{"ste')
    trim_metrics(history, 2)
    assert history.read_text() == '{"step": 1}\n{"step": 2}\n'
    path = tmp_path / "last.pt"
    path.write_bytes(b"old-valid")
    rotate_previous(path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"new-valid")
    replacement.replace(path)
    assert (tmp_path / "last.previous.pt").read_bytes() == b"old-valid"
