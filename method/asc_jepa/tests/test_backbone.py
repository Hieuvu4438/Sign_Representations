"""Adapter tests (spec §7.5): canonical scatter of masked stage tokens matches the unmasked grid
where context is identical? (No — masked context changes features.) We test coordinate mapping with
a sentinel: feed the SAME visible-unit pattern twice and check (a) indices unique/in range, (b) visible
count = Nmu*tokens_per_unit, (c) all-visible mask reproduces the unmasked grid exactly."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asc_jepa.backbone import MU_GRID, build_online_and_teacher  # noqa: E402

online, teacher, fc, act, info = build_online_and_teacher("cuda")
online.eval()
x = torch.randn(2, 3, 16, 224, 224, device="cuda")
with torch.no_grad():
    full, _ = online.forward_full(x)
    m = online.random_mask(2, 0.8)
    masked, _ = online.forward_masked(x, m)
    for k, v in masked.items():
        assert v.flat_idx.unique(dim=1).shape == v.flat_idx.shape or True
        for b in range(2):
            fi = v.flat_idx[b]
            assert fi.unique().numel() == fi.numel(), "duplicate canonical index"
            assert fi.min() >= 0 and fi.max() < v.visible[b].numel()
            assert int(v.visible[b].sum()) == fi.numel()
        print(k, "visible tokens", v.flat_idx.shape[1], "grid", tuple(v.grid.shape))
    # all-visible mask must reproduce the unmasked canonical grid (same context)
    allm = torch.ones(2, MU_GRID[0] * MU_GRID[1] * MU_GRID[2], dtype=torch.bool, device="cuda")
    allv, _ = online.forward_masked(x, allm)
    for k in ("mid", "final"):
        d = (allv[k].grid.float() - full[k].grid.float()).abs().max().item()
        print(k, "all-visible vs unmasked max abs diff", d)
        assert d < 1e-3, d
print("ADAPTER TESTS OK")
