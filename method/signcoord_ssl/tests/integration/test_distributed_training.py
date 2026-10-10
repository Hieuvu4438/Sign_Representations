import copy
from datetime import timedelta
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from signcoord.config import load_config
from signcoord.engine.checkpoint import load_checkpoint
from signcoord.engine.train import make_dataset
from signcoord.targets.statistics import fit_target_statistics

PROJECT = Path(__file__).resolve().parents[2]


def capturing_optimizer(base_class, path):
    class CaptureAdamW(base_class):
        def step(self, *args, **kwargs):
            parameters = [p for group in self.param_groups for p in group['params']]
            torch.save({i: None if p.grad is None else p.grad.detach().cpu().clone()
                        for i, p in enumerate(parameters)}, path)
            return super().step(*args, **kwargs)
    return CaptureAdamW


@pytest.fixture(scope="module")
def distributed_config(tmp_path_factory):
    root = tmp_path_factory.mktemp("ddp_source") / "fixture"
    subprocess.run([sys.executable, str(PROJECT / "tools/generate_synthetic_fixture.py"),
                    "--output", str(root)], check=True, capture_output=True)
    config = load_config(root / "fixture_smoke.yaml")
    fit_target_statistics(make_dataset(config), config["data"]["ssl_train_manifest"],
                          config["data"]["target_stats_path"], config["data"]["phase_calibration_path"])
    config["train"].update(microbatch_per_gpu=2, accumulation_steps=2, max_updates=4, save_every_updates=2)
    config["model"]["decoder"]["dropout"] = .1
    return config


def trainer_worker(rank, rendezvous, config, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    module = importlib.import_module("signcoord.engine.train")
    original_dataset = module.make_dataset
    original_model = module.make_model
    if mode == 'gradient':
        module.torch.optim.AdamW = capturing_optimizer(torch.optim.AdamW,
                                      Path(config['run_dir']) / f'gradient_rank{rank}.pt')
    if mode in ("empty_rank", "load_failure"):
        class DatasetWrapper:
            def __init__(self, dataset):
                self.dataset = dataset
                self.records = dataset.records

            def __getattr__(self, key):
                return getattr(self.dataset, key)

            @property
            def sample_epoch(self):
                return self.dataset.sample_epoch

            @sample_epoch.setter
            def sample_epoch(self, value):
                self.dataset.sample_epoch = value

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, index):
                if mode == "load_failure" and rank == 1:
                    raise OSError("injected single-rank video decode failure")
                sample = self.dataset[index]
                if mode == "empty_rank" and rank == 0:
                    sample["targets"]["reliability"].zero_()
                    sample["targets"]["joint_valid"].zero_()
                    sample["targets"]["global"].fill_(float("nan"))
                return sample
        module.make_dataset = lambda c: DatasetWrapper(original_dataset(c))
    if mode == "nonfinite_prediction":
        def make_bad_model(c):
            model = original_model(c)
            if rank == 1:
                def invalidate(_model, _inputs, outputs):
                    outputs["pred_standardized"] = outputs["pred_standardized"] * float("nan")
                    return outputs
                model.register_forward_hook(invalidate)
            return model
        module.make_model = make_bad_model
    try:
        if mode == "resume":
            module.train(config, stop_after=2)
            module.train(config, resume=Path(config["run_dir"]) / "checkpoints/last.pt")
        elif mode in ("load_failure", "nonfinite_prediction"):
            try:
                module.train(config)
            except RuntimeError as exc:
                expected = "load_microbatch" if mode == "load_failure" else "model_prediction"
                assert expected in str(exc)
                Path(config["run_dir"], f"caught_rank{rank}.txt").write_text(str(exc))
            else:
                raise AssertionError("all ranks must terminate together on the injected failure")
        else:
            module.train(config)
    finally:
        dist.destroy_process_group()


def launch(root, config, mode):
    current = copy.deepcopy(config)
    current["run_dir"] = str(root / mode)
    mp.spawn(trainer_worker, args=(str(root / (mode + "_rendezvous")), current, mode), nprocs=2, join=True)
    return Path(current["run_dir"])


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_real_trainer_two_ranks_accumulation_and_rank_rng_exact_resume(distributed_config, tmp_path):
    full = launch(tmp_path, distributed_config, "full")
    resumed = launch(tmp_path, distributed_config, "resume")
    a, b = (load_checkpoint(path / "checkpoints/final.pt") for path in (full, resumed))
    for key in ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "sampler_state"):
        assert_nested_equal(a[key], b[key])
    assert a["world_size"] == 2
    assert a["compute_ledger"]["stats_batch"] == 4
    assert a["compute_ledger"]["optimizer_batch"] == 8
    assert len(a["rng_states_by_rank"]) == 2
    for rank in range(2):
        assert torch.equal(a["rng_states_by_rank"][rank]["torch"], b["rng_states_by_rank"][rank]["torch"])
    assert not torch.equal(a["rng_states_by_rank"][0]["torch"], a["rng_states_by_rank"][1]["torch"])
    logs = [json.loads(line) for line in (full / "metrics.jsonl").read_text().splitlines()]
    assert logs == [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    for record in logs:
        for micro in zip(*record["rank_microsteps"]):
            assert not set(micro[0]["sample_ids"]) & set(micro[1]["sample_ids"])
    assert all(torch.isfinite(v).all() for v in a["model_state"].values())


def test_rank_with_no_geometry_uses_global_valid_denominator(distributed_config, tmp_path):
    output = launch(tmp_path, distributed_config, "empty_rank")
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    for record in records:
        assert torch.isfinite(torch.tensor(record["global_mean_loss"]))
        assert all(m["geometry"] == 0 for m in record["rank_microsteps"][0])
        assert all(m["geometry_groups"]["global_L"]["denominator"] > 0 for m in record["rank_microsteps"][0])


@pytest.mark.parametrize("mode", ["load_failure", "nonfinite_prediction"])
def test_single_rank_failure_terminates_collectively(distributed_config, tmp_path, mode):
    output = launch(tmp_path, distributed_config, mode)
    for rank in range(2):
        assert (output / f"caught_rank{rank}.txt").exists()
    assert not (output / "checkpoints/last.pt").exists()


def test_a9_deranges_across_ranks_with_one_sample_each(distributed_config, tmp_path):
    config = copy.deepcopy(distributed_config)
    config["arm"] = "A9"
    config["loss"]["relation"]["kind"] = "phase_shuffle"
    config["train"]["microbatch_per_gpu"] = 1
    output = launch(tmp_path, config, "shuffle")
    changed = 0
    for line in (output / "metrics.jsonl").read_text().splitlines():
        for micro in zip(*json.loads(line)["rank_microsteps"]):
            for rank in range(2):
                shuffle = micro[rank]["relation_shuffle"]
                assert shuffle["scope"] == "global_ddp_microbatch_detached_targets"
                target = shuffle["target_sample_ids"][0]
                if target is not None:
                    assert target == micro[1-rank]["sample_ids"][0]
                    changed += 1
    assert changed > 0


def test_trainer_gradient_matches_centralized_same_stats_batch(distributed_config, tmp_path, monkeypatch):
    config = copy.deepcopy(distributed_config)
    config['model']['decoder']['dropout'] = 0.
    config['train'].update(max_updates=1, save_every_updates=1)
    config['loss']['relation'].update(warmup_zero_updates=-1, ramp_end_update=0)
    distributed = launch(tmp_path, config, 'gradient')
    central = copy.deepcopy(config)
    central['train']['microbatch_per_gpu'] *= 2
    central['run_dir'] = str(tmp_path / 'central')
    monkeypatch.setattr(torch.optim, 'AdamW', capturing_optimizer(torch.optim.AdamW, tmp_path / 'central_gradients.pt'))
    importlib.import_module('signcoord.engine.train').train(central)
    gradients = torch.load(tmp_path / 'central_gradients.pt', weights_only=True)
    rank_gradients = torch.load(distributed / 'gradient_rank0.pt', weights_only=True)
    for key, gradient in gradients.items():
        if gradient is None:
            assert rank_gradients[key] is None
        else:
            torch.testing.assert_close(gradient, rank_gradients[key], atol=5e-6, rtol=2e-4)
    # Batch-shaped float32 kernels can accumulate differently. Double-precision
    # analytic reduction/gather checks are separate; bound the full engine drift.
    central_flat=torch.cat([value.flatten() for value in gradients.values() if value is not None])
    rank_flat=torch.cat([value.flatten() for value in rank_gradients.values() if value is not None])
    assert (central_flat-rank_flat).norm()/central_flat.norm() < 3e-5
    central_record = json.loads((tmp_path / 'central/metrics.jsonl').read_text())
    rank_record = json.loads((distributed / 'metrics.jsonl').read_text())
    central_loss = sum(m['total'] for m in central_record['microsteps']) / 2
    assert rank_record['global_mean_loss'] == pytest.approx(central_loss, rel=2e-5, abs=2e-6)
