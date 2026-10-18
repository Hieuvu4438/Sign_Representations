"""Resume continues the same batches/schedules; sampler determinism; frozen inputs unchanged by probing."""
import json

import numpy as np
import torch

from asm_jepa.config import load_config
from asm_jepa.evaluation.embeddings import save_embeddings
from asm_jepa.training.engine import PretrainEngine
from asm_jepa.training.sampler import StepBatchSampler
from asm_jepa.utils import read_jsonl

from conftest import cfg_for

SETS = ["train.warmup_steps=1", "train.batch_size_per_gpu=4", "train.accum_steps=2", "data.num_workers=0",
        "eval.frozen_every_steps=0", "train.checkpoint_every_steps=2", "train.log_every_steps=1",
        "eval.diagnostic_batch_size=8", "train.precision=fp32"]


def test_sampler_is_deterministic_and_resumable():
    a = StepBatchSampler(50, 4, base_seed=3)
    full = [next(x for x in [a.batch(m)]) for m in range(30)]
    b = StepBatchSampler(50, 4, base_seed=3, start_micro=17)
    it = iter(b)
    assert [next(it) for _ in range(13)] == full[17:]
    epoch0 = [i for m in range(12) for i, _ in full[m]]
    assert len(set(epoch0)) == 48  # drop_last epoch covers distinct items


def test_resume_matches_uninterrupted_run(tmp_path):
    cfg_a = cfg_for("state", "train.max_steps=4", *SETS)
    eng = PretrainEngine(cfg_a, tmp_path / "a", synthetic=True, device="cpu")
    eng.train()
    teacher_a = {k: v.clone() for k, v in eng.teacher.state_dict().items()}
    losses_a = [r["loss/total"] for r in read_jsonl(tmp_path / "a" / "metrics.jsonl") if "loss/total" in r]

    cfg_b = cfg_for("state", "train.max_steps=4", *SETS)
    eng_b = PretrainEngine(cfg_b, tmp_path / "b", synthetic=True, device="cpu")
    eng_b.train()  # identical first run, then resume a copy from its step-2 checkpoint
    cfg_c = cfg_for("state", "train.max_steps=4", *SETS)
    eng_c = PretrainEngine(cfg_c, tmp_path / "c", synthetic=True, device="cpu",
                           resume=str(tmp_path / "b" / "checkpoints" / "step_0000002.pt"))
    assert eng_c.step == 2 and eng_c.micro_consumed == 4
    eng_c.train()
    losses_c = [r["loss/total"] for r in read_jsonl(tmp_path / "c" / "metrics.jsonl") if "loss/total" in r]
    assert np.allclose(losses_a[2:], losses_c, rtol=1e-4, atol=1e-5)
    for k, v in eng_c.teacher.state_dict().items():
        assert torch.allclose(v, teacher_a[k], atol=1e-5), k
    assert eng_c.optimizer.param_groups[0]["lr"] == eng.optimizer.param_groups[0]["lr"]


def test_resume_refuses_changed_architecture(tmp_path):
    cfg = cfg_for("state", "train.max_steps=2", *SETS)
    PretrainEngine(cfg, tmp_path / "a", synthetic=True, device="cpu").train()
    cfg2 = cfg_for("full", "train.max_steps=2", *SETS)
    try:
        PretrainEngine(cfg2, tmp_path / "b", synthetic=True, device="cpu",
                       resume=str(tmp_path / "a" / "checkpoints" / "last.pt"))
    except SystemExit as e:
        assert "refusing to resume" in str(e)
    else:
        raise AssertionError("resume with a different architecture must fail")


def test_linear_probe_leaves_frozen_inputs_unchanged(tmp_path):
    from asm_jepa.training.classifier import train_linear_probe

    rng = np.random.default_rng(0)
    meta = {"checkpoint_sha256": "x", "stats_sha256": "s", "feature_key": "state_part_concat1024_l2",
            "schema": "v1", "arch_hash": "a", "encoder": "teacher", "sampler": "t"}
    paths = {}
    for split, n in (("train", 200), ("val", 50)):
        y = rng.integers(0, 5, n)
        x = rng.normal(size=(n, 1024)).astype(np.float32) * 0.1
        x[np.arange(n), y] += 1.0
        arrays = {"state_part_concat1024_l2": x, "state_balanced_mean256_l2": x[:, :256], "usable": np.ones(n, bool),
                  "class_ids": y}
        paths[split] = tmp_path / f"{split}.npz"
        save_embeddings(paths[split], arrays, [f"{split}{i}" for i in range(n)], ["S"] * n, {**meta, "split": split})
    cfg = load_config(None, "linear", ["train.epochs=30", "train.warmup_epochs=1", "train.batch_size=16", "num_classes_from_vocab=false"])
    rep = train_linear_probe(cfg, paths["train"], paths["val"], tmp_path / "lin", "cpu")
    assert rep["frozen_inputs_unchanged"] and rep["best_val"]["top1_instance"] > 0.5
    assert json.loads((tmp_path / "lin" / "reports" / "val.json").read_text())["test_exposure"] is False


def test_few_shot_subset_counts():
    from asm_jepa.training.classifier import few_shot_subset

    y = np.array([0] * 5 + [1] * 1 + [2] * 3)
    usable = np.ones(len(y), bool)
    usable[0] = False
    sel = few_shot_subset(y, usable, 2, seed=0)
    counts = np.bincount(y[sel], minlength=3)
    assert counts.tolist() == [2, 1, 2] and usable[sel].all()
    assert np.array_equal(sel, few_shot_subset(y, usable, 2, seed=0))
