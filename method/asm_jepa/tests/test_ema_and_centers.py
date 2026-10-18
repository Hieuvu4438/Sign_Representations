"""Teacher gradients, exact EMA, EMA/centre updates once per optimizer step, count-weighted centres."""
import torch

from asm_jepa.data.dataset import SampleBuilder, SSLDataset
from asm_jepa.data.synthetic import SyntheticSource
from asm_jepa.models.asm_jepa import ASMJEPA, ema_pairs, make_teacher, teacher_forward
from asm_jepa.training import engine as E
from asm_jepa.training.centers import Centers
from asm_jepa.training.ema import ema_update

from conftest import cfg_for


def _batch(cfg, stats, n=16):
    ds = SSLDataset(SyntheticSource(n, 0), SampleBuilder(cfg, stats))
    return torch.utils.data.default_collate([ds[(i, i)] for i in range(n)])


def test_teacher_has_no_grads_and_online_heads_do(stats):
    cfg = cfg_for("full")
    torch.manual_seed(0)
    m = ASMJEPA(cfg)
    t = make_teacher(m)
    b = _batch(cfg, stats)
    out = teacher_forward(t, b, True)
    loss, logs, _ = m(b, out, torch.zeros(9, 256), 0.1, 0.06, True)
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in t.parameters())
    for mod in (m.online, m.predictor, m.anchor, m.global_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
    m.train()
    t.train()
    assert not t.training and not any(x.training for x in t.modules())


def test_ema_exact_update():
    a, b = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    a0 = {k: v.clone() for k, v in a.state_dict().items()}
    ema_update([(a, b)], 0.9)
    for k, v in a.state_dict().items():
        assert torch.allclose(v, 0.9 * a0[k] + 0.1 * b.state_dict()[k])


def test_centers_count_weighted_and_untouched_groups():
    c = Centers(9, 4, momentum=0.5)
    z = torch.tensor([[2.0] * 4, [4.0] * 4, [10.0] * 4])
    c.accumulate(z[:2], torch.tensor([0, 0]), torch.tensor([1.0, 3.0]))  # microbatch 1
    c.accumulate(z[2:], torch.tensor([1]), torch.tensor([2.0]))  # microbatch 2
    c.step()
    assert torch.allclose(c.center[0], torch.full((4,), 0.5 * (2 * 1 + 4 * 3) / 4))  # weighted, not mean of means
    assert torch.allclose(c.center[1], torch.full((4,), 5.0))
    assert (c.center[2:] == 0).all() and c.center.dtype == torch.float32
    assert c.updates.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    sd = c.state_dict()
    c2 = Centers(9, 4, 0.5)
    c2.load_state_dict(sd)
    assert torch.equal(c2.center, c.center)


def test_engine_updates_ema_and_centres_once_per_optimizer_step(tmp_path, monkeypatch):
    cfg = cfg_for("state", "train.max_steps=3", "train.warmup_steps=1", "train.batch_size_per_gpu=4",
                  "train.accum_steps=3", "data.num_workers=0", "eval.frozen_every_steps=0",
                  "train.checkpoint_every_steps=0", "train.log_every_steps=1", "eval.diagnostic_batch_size=8")
    calls = {"ema": 0, "center": 0}
    real_ema, real_step = E.ema_update, Centers.step

    def count_ema(*a, **k):
        calls["ema"] += 1
        return real_ema(*a, **k)

    def count_center(self):
        calls["center"] += 1
        return real_step(self)

    monkeypatch.setattr(E, "ema_update", count_ema)
    monkeypatch.setattr(Centers, "step", count_center)
    eng = E.PretrainEngine(cfg, tmp_path / "run", synthetic=True, device="cpu")
    eng.train()
    assert calls == {"ema": 3, "center": 3}
    assert eng.micro_consumed == 9 and eng.step == 3
    assert not eng.teacher.training


def test_teacher_changes_only_through_ema(stats):
    cfg = cfg_for("state")
    m = ASMJEPA(cfg)
    t = make_teacher(m)
    before = {k: v.clone() for k, v in t.state_dict().items()}
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    b = _batch(cfg, stats, 8)
    loss, _, _ = m(b, teacher_forward(t, b, False), torch.zeros(9, 256), 0.1, 0.06, False)
    loss.backward()
    opt.step()
    assert all(torch.equal(before[k], v) for k, v in t.state_dict().items())
    ema_update(ema_pairs(t, m), 0.5)
    assert any(not torch.equal(before[k], v) for k, v in t.state_dict().items())


def test_four_interval_widths_keep_geometry_and_timing_buckets_apart():
    from asm_jepa.features.stats import identity_stats

    cfg = cfg_for("full", "features.interval_widths_sec=[0.32,0.64,1.28,2.0]")
    m = ASMJEPA(cfg)
    t = make_teacher(m)
    b = _batch(cfg, identity_stats(4), 16)
    loss, logs, _ = m(b, teacher_forward(t, b, True), torch.zeros(9, 256), 0.1, 0.06, True)
    assert torch.isfinite(loss) and torch.isfinite(logs["loss/geometry"]) and torch.isfinite(logs["loss/timing"])
