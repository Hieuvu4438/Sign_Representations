"""Continue SignRep with cross-window latent prediction (H002) or its matched controls.

Arms (same data, init, steps, batch, masking, predictor size):
  --deltas -8,8 --lam 1.0   M0  cross-window prediction (past + future, half-overlap)
  --deltas 0    --lam 1.0   C0  same-window prediction through the predictor (matched compute)
  --lam 0.0                 A0  anchor only
Unlabelled ASL Citizen **train** videos only (labels never read). Fixed step budget,
last checkpoint is the one evaluated (no selection).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4 import xwp  # noqa: E402
from p4.data import aslc_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deltas", default="-8,8")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-pred", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--mask", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-steps-run", type=int, default=0, help="stop early (smoke tests only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    deltas = [int(x) for x in a.deltas.split(",")]
    teacher, student, info = xwp.build_models("cuda", a.mask)
    predictor = xwp.DeltaPredictor(deltas).cuda()
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_back = sum(p.numel() for n, p in student.named_parameters() if p.requires_grad and n.startswith("backbone."))
    assert n_back > 50e6, f"backbone not trainable ({n_back})"
    assert all(not p.requires_grad for p in teacher.parameters())
    print(f"trainable student params {n_train/1e6:.2f}M (backbone {n_back/1e6:.2f}M)", flush=True)
    ref = {k: v.detach().clone() for k, v in student.state_dict().items() if k.startswith(("backbone.blocks.0.", "fc."))}
    opt = torch.optim.AdamW(xwp.param_groups(student, predictor, a.lr, a.lr_pred), betas=(0.9, 0.999))
    rows = [{k: r[k] for k in ("sample_id", "path")} for r in aslc_rows("train")]  # no labels passed on
    ds = xwp.PairDataset(rows, deltas, seed=a.seed)
    g = torch.Generator().manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                                     generator=g, persistent_workers=True, prefetch_factor=4)
    json.dump(dict(vars(a), init=info), open(out / "config.json", "w"), indent=1)
    log = open(out / "train_log.jsonl", "w")
    student.train()
    step, t0 = 0, time.time()
    while step < a.steps:
        for ua, ub, di in dl:
            if step >= a.steps or (a.max_steps_run and step >= a.max_steps_run):
                break
            lr_scale = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for gr in opt.param_groups:
                gr["lr"] = gr["base_lr"] * lr_scale
            x_a, x_b = xwp.to_input(ua.cuda(non_blocking=True), "cuda"), xwp.to_input(ub.cuda(non_blocking=True), "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, st = xwp.xwp_loss(student, teacher, predictor, x_a, x_b, di.cuda(), a.lam)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for gr in opt.param_groups for p in gr["params"]], 1.0)
            opt.step()
            step += 1
            if step % 25 == 0 or step == 1:
                st.update(step=step, loss=loss.item(), grad_norm=gn.item(), lr_scale=lr_scale,
                          sec=time.time() - t0, samples_per_s=step * a.batch / (time.time() - t0))
                log.write(json.dumps(st) + "\n")
                log.flush()
            if step % 500 == 0:
                print(json.dumps(st), flush=True)
        if a.max_steps_run and step >= a.max_steps_run:
            break
    sd = {k: v for k, v in student.state_dict().items() if k.startswith(("backbone.", "fc."))}
    drift = max((sd[k].float() - ref[k].float()).abs().max().item() for k in ref)
    print("max weight drift (block0/fc)", drift, flush=True)
    assert drift > 0, "student weights did not change"
    torch.save({"student": sd, "predictor": predictor.state_dict(), "step": step}, out / "last.pt")
    print("DONE", step, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
