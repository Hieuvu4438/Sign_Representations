"""EX11 / H007: train the 32-frame SignRep student (arm H7 ordered or C7 order-free), then
extract val features (32-frame windows, stride 4) with the trained student."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4 import longwin as LW  # noqa: E402
from p4 import signrep as S  # noqa: E402
from p4.data import VideoDataset, aslc_rows, collate_single  # noqa: E402
from p4.xwp import to_input  # noqa: E402


def groups(student, heads, lr, lr_h, decay=0.85, wd=0.05):
    nb = len(student.backbone.blocks)
    gs = []
    for n, p in student.named_parameters():
        lid = int(n.split(".")[2]) + 1 if n.startswith("backbone.blocks.") else (0 if n.startswith("backbone.") else nb + 1)
        sc = decay ** (nb + 1 - lid)
        gs.append({"params": [p], "lr": lr * sc, "base_lr": lr * sc, "weight_decay": 0.0 if p.ndim == 1 else wd})
    gs.append({"params": list(heads.parameters()), "lr": lr_h, "base_lr": lr_h, "weight_decay": wd})
    return gs


@torch.no_grad()
def extract_val(student, out, stride=4, batch=16):
    rows = aslc_rows("val")
    dl = torch.utils.data.DataLoader(VideoDataset(rows), batch_size=1, num_workers=1, collate_fn=collate_single, prefetch_factor=4)
    feats, starts, nwin = [], [], []
    idx = torch.arange(32, device="cuda")
    for i, arr, fps in dl:
        if len(arr) < 32:
            arr = torch.cat([arr, arr[-1:].expand(32 - len(arr), -1, -1, -1)])
        st = list(range(0, len(arr) - 32 + 1, stride))
        fr = S.normalize_uint8(arr, "cuda").half()
        zs = []
        for j in range(0, len(st), batch):
            stt = torch.tensor(st[j:j + batch], device="cuda")
            x = fr[stt[:, None] + idx[None]].permute(0, 2, 1, 3, 4).contiguous()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zs.append(LW.student_features(student, x, train=False).float().cpu().numpy())
        feats.append(np.concatenate(zs).astype(np.float16)); starts.append(np.asarray(st, np.int32)); nwin.append(len(st))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "windows_features.npy", np.concatenate(feats)); np.save(out / "window_start.npy", np.concatenate(starts))
    np.save(out / "offsets.npy", np.concatenate([[0], np.cumsum(nwin)]))
    with open(out / "clips.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["H7", "C7"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    teacher, student, info = LW.build_student32("cuda")
    heads = LW.Heads().cuda()
    opt = torch.optim.AdamW(groups(student, heads, 2e-5, 5e-4), betas=(0.9, 0.999))
    rows = [{k: r[k] for k in ("sample_id", "path")} for r in aslc_rows("train")]  # labels never passed on
    dl = torch.utils.data.DataLoader(LW.Win32Dataset(rows, a.seed), batch_size=a.batch, shuffle=True, num_workers=1,
                                     drop_last=True, generator=torch.Generator().manual_seed(a.seed), persistent_workers=True, prefetch_factor=6)
    json.dump(dict(vars(a), init=info), open(out / "config.json", "w"), indent=1, default=str)
    log = open(out / "train_log.jsonl", "w")
    student.train(); step, t0 = 0, time.time()
    ref = {k: v.detach().clone() for k, v in student.state_dict().items() if k.startswith("backbone.blocks.0.")}
    while step < a.steps:
        for u in dl:
            if step >= a.steps:
                break
            sc = min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * sc
            x = to_input(u.cuda(non_blocking=True), "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, st = LW.loss_fn(student, teacher, heads, x, ordered=(a.arm == "H7"))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at {step}")
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step(); step += 1
            if step % 25 == 0 or step == 1:
                st.update(step=step, loss=loss.item(), grad_norm=gn.item(), sec=time.time() - t0)
                log.write(json.dumps(st) + "\n"); log.flush()
                if step % 500 == 0:
                    print(json.dumps(st), flush=True)
    drift = max((student.state_dict()[k].float() - v.float()).abs().max().item() for k, v in ref.items())
    assert drift > 0, "student did not change"
    print("drift", drift, "train sec", time.time() - t0, flush=True)
    student.eval()
    if a.no_extract:
        return
    torch.save({k: v for k, v in student.state_dict().items()}, out / "last.pt")
    extract_val(student, out / "val_feats")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
