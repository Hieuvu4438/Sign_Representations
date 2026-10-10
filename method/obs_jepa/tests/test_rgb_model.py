"""RGB-JEPA wiring: shapes, teacher gradient isolation, hidden-value perturbation (proposal11 §11.6)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.models.rgb import RGBJepa, normalize_video  # noqa: E402


def check(recipe, size=224, B=2):
    torch.manual_seed(0)
    m = RGBJepa(size=size, recipe=recipe).cuda()
    g = size // 16; N = 8 * g * g
    x = torch.randint(0, 255, (B, 16, size, size, 3), dtype=torch.uint8).cuda()
    perm = torch.stack([torch.randperm(N) for _ in range(B)]).cuda()
    ctx, tgt = perm[:, : N // 5].sort(1).values, perm[:, N // 5:].sort(1).values
    v = normalize_video(x)
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pt, pc, z = m(v, ctx, tgt)
        h = m.targets(v)
    assert pt.shape == (B, tgt.shape[1], m.n_out * 768) and pc.shape == (B, ctx.shape[1], m.n_out * 768), (pt.shape, pc.shape)
    assert h.shape == (B, N, m.n_out * 768)
    # hidden-value perturbation: overwrite pixels of every hidden tubelet; context features must not change
    x2 = x.clone()
    hid = torch.zeros(B, N, dtype=torch.bool, device="cuda"); hid.scatter_(1, tgt, True)
    hid = hid.view(B, 8, g, g).repeat_interleave(2, 1).repeat_interleave(16, 2).repeat_interleave(16, 3)
    x2[hid] = torch.randint(0, 255, (int(hid.sum()), 3), dtype=torch.uint8, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, z2 = m(normalize_video(x2), ctx, tgt)
    diff = (z.float() - z2.float()).abs().max().item()
    assert diff == 0.0, f"context features depend on hidden pixels: {diff}"
    # teacher receives no gradient
    m.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pt, pc, _ = m(v, ctx, tgt)
        loss = (pt.float() - torch.cat([m.targets(v)], 0).gather(1, tgt[..., None].expand(-1, -1, pt.shape[-1]))).abs().mean()
    loss.backward()
    assert all(p.grad is None for p in m.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.student.parameters())
    return diff


def test_ar():
    check("AR")


def test_ad():
    check("AD")


if __name__ == "__main__":
    for r in ("AR", "AD"):
        print(r, "max |dz| under hidden perturbation:", check(r))
    print(check("AD", size=384, B=1))
    print("OK")
