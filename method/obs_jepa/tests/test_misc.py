"""Remaining §11.6 checks: finite losses with no valid targets, EMA initialisation, letterbox registration,
graph-encoder isolation of hidden/missing cells."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.data.clips import make_letterbox  # noqa: E402
from oj.data.landmarks import J  # noqa: E402
from oj.losses import CenteredCE, masked_smooth_l1, region_masked_loss  # noqa: E402
from oj.models.graph import GraphJepa  # noqa: E402


def test_losses_all_invalid():
    err = torch.rand(2, 5, 4, requires_grad=True)
    l, _ = region_masked_loss(err, torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5))
    assert torch.isfinite(l) and l.item() == 0.0; l.backward()
    assert torch.isfinite(masked_smooth_l1(torch.rand(3, 4), torch.rand(3, 4), torch.zeros(3, 4)))
    c = CenteredCE(8, device="cpu"); assert torch.isfinite(c(torch.rand(2, 3, 8), torch.rand(2, 3, 8), torch.zeros(2, 3)))


def test_letterbox_roundtrip():
    for w, h, s in ((640, 480, 224), (960, 540, 384), (480, 640, 224)):
        lb = make_letterbox(w, h, s); p = np.random.rand(50, 2).astype(np.float32) * [w, h]
        assert np.allclose(lb.invert(lb.apply(p)), p, atol=1e-3)
        tv = lb.token_valid(); assert tv.any() and (tv.sum() <= (s // 16) ** 2)


def test_graph_ema_init_and_isolation():
    torch.manual_seed(0)
    m = GraphJepa("anchors").eval()
    for a, b in zip(m.teacher.parameters(), m.student.parameters()):
        assert torch.equal(a, b)
    B, Tb = 2, 32
    node = torch.randn(B, Tb, J, 38); loc = {"lh": torch.randn(B, Tb, 21, 16), "rh": torch.randn(B, Tb, 21, 16), "face": torch.randn(B, Tb, 18, 16)}
    vis = torch.rand(B, Tb, J) > 0.5; t = torch.arange(Tb).float()[None].repeat(B, 1) * 0.08
    with torch.no_grad():
        h1, _, _, _ = m.student(node, loc, vis, t)
        node2 = node.clone(); node2[~vis] = torch.randn(int((~vis).sum()), 38) * 10
        loc2 = {k: v.clone() for k, v in loc.items()}
        from oj.data.landmarks import FACE, LH, RH
        for k, nodes in (("lh", LH), ("rh", RH), ("face", FACE)):
            inv = ~vis[:, :, nodes]; loc2[k][inv] = torch.randn(int(inv.sum()), 16) * 10
        h2, _, _, _ = m.student(node2, loc2, vis, t)
    d = (h1 - h2)[vis].abs().max().item()
    assert d < 1e-5, f"visible states depend on hidden cells: {d}"
