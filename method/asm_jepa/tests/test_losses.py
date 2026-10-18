"""Loss masking: invalid targets contribute 0, no NaN, padding-independent family averages."""
import torch

from asm_jepa.data.schema import PART_ORDER
from asm_jepa.models.heads import anchor_heads
from asm_jepa.training.losses import anchor_loss, jepa_loss

FW = {"state": 1.0, "relation": 0.5, "geometry": 0.25, "timing": 0.25}


def _toy(n=12, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred, z = torch.randn(n, d, generator=g), torch.randn(n, d, generator=g)
    groups = torch.tensor([0, 0, 1, 3, 4, 4, 5, 7, 7, 8, 8, 8])[:n]
    sub = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 1, 7])[:n]
    w = torch.rand(n, generator=g) + 0.1
    return pred, z, w, groups, sub


def test_padding_entries_do_not_change_the_loss():
    pred, z, w, g, sg = _toy()
    centers = torch.zeros(9, 8)
    l1, _ = jepa_loss(pred, z, w, g, sg, centers, 0.1, 0.06, FW)
    pad = 7
    pred2 = torch.cat([pred, torch.randn(pad, 8) * 100])
    z2 = torch.cat([z, torch.randn(pad, 8) * 100])
    w2 = torch.cat([w, torch.zeros(pad)])
    g2 = torch.cat([g, torch.zeros(pad, dtype=torch.long)])
    sg2 = torch.cat([sg, torch.zeros(pad, dtype=torch.long)])
    l2, _ = jepa_loss(pred2, z2, w2, g2, sg2, centers, 0.1, 0.06, FW)
    assert torch.allclose(l1, l2)


def test_inactive_families_do_not_dilute_the_loss():
    pred, z, w, g, sg = _toy()
    state = g <= 3
    centers = torch.zeros(9, 8)
    l_state_only, logs = jepa_loss(pred[state], z[state], w[state], g[state], sg[state], centers, 0.1, 0.06, FW)
    assert torch.allclose(l_state_only, logs["loss/state"])  # denominator = active families with targets only
    assert torch.isnan(logs["loss/relation"])


def test_all_invalid_targets_give_zero_loss_without_nan():
    pred, z, _, g, sg = _toy()
    loss, _ = jepa_loss(pred.requires_grad_(), z, torch.zeros(len(g)), g, sg, torch.zeros(9, 8), 0.1, 0.06, FW)
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_anchor_loss_ignores_invalid_components():
    B, n_p, D = 2, 16, 256
    heads = anchor_heads(D, True)
    u = torch.randn(B, 4 * n_p, D, requires_grad=True)
    batch = {}
    dims = {"body": 24, "lh": 132, "rh": 132, "face": 180}
    for p in PART_ORDER:
        batch[f"anchor_{p}"] = torch.randn(B, n_p, dims[p])
        batch[f"anchor_{p}_valid"] = torch.zeros(B, n_p, dims[p], dtype=torch.bool)
    tv = torch.ones(B, 4 * n_p, dtype=torch.bool)
    usable = torch.ones(B, dtype=torch.bool)
    loss, logs = anchor_loss(u, tv, usable, heads, batch, n_p, 0.1)
    assert loss.item() == 0.0 and all(torch.isnan(v) for v in logs.values())
    batch["anchor_lh_valid"][0, 3, :10] = True
    l1, _ = anchor_loss(u, tv, usable, heads, batch, n_p, 0.1)
    batch["anchor_lh"][1] = 1e6  # invalid components may hold anything
    l2, _ = anchor_loss(u, tv, usable, heads, batch, n_p, 0.1)
    assert torch.isfinite(l1) and torch.allclose(l1, l2)
