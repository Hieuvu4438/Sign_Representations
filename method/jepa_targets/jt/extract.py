"""Frozen window features on ASL Citizen VAL for EX15 evaluation.
Windows: 16 consecutive frames, start every 8 frames (clips < 16 frames are padded by repeating the last
frame). Feature per window = mean of the encoder's final (normed) tokens. Encoder: the EMA encoder of a
run (`--ckpt runtime/<arm>/last.pt`), the released V-JEPA 2.1 (`--ckpt vjepa`) or random init (`--ckpt random`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "p4_signssl"))
from jt.vjepa import build_encoder, normalize_video  # noqa: E402
from p4.data import VideoDataset, aslc_rows  # noqa: E402

RT = ROOT / "runtime"
STRIDE, LEN = 8, 16


def window_starts(n):
    return list(range(0, max(1, n - LEN + 1), STRIDE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--subset", default="", help="optional file of sample_ids")
    ap.add_argument("--kind", default="vjepa", choices=["vjepa", "hiera", "vjepa_fc"], help="EX16: hiera = SignRep-layout model (features = fc(mean tokens)); vjepa_fc = jv3d encoder + fc")
    ap.add_argument("--weights", default="model", help="EX16 checkpoint key: model / ema_model (hiera), encoder / ema_encoder (vjepa_fc)")
    a = ap.parse_args()
    assert a.split != "test"
    out = RT / "feats" / a.name; out.mkdir(parents=True, exist_ok=False)
    dev = "cuda"
    if a.kind == "hiera":
        sys.path.insert(0, str(ROOT.parents[0] / "jepa_baseline" / "src"))
        from signrep_baseline.model import SignRep
        m = SignRep(); m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)[a.weights], strict=True)
        m = m.to(dev).eval()
        enc = lambda x: m.encode(x, mask_ratio=0.0)["features"][:, None]
    elif a.kind == "vjepa_fc":
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        e = build_encoder(pretrained=False, checkpointing=False); e.load_state_dict(ck[a.weights]); e = e.to(dev).eval()
        fc = torch.nn.Sequential(torch.nn.LayerNorm(768), torch.nn.Linear(768, 768, bias=False)); fc.load_state_dict(ck["fc"]); fc = fc.to(dev).eval()
        enc = lambda x: fc(e(x).float().mean(1))[:, None]
    elif a.ckpt == "vjepa":
        enc = build_encoder(checkpointing=False)
    elif a.ckpt == "random":
        torch.manual_seed(0); enc = build_encoder(pretrained=False, checkpointing=False)
    else:
        enc = build_encoder(pretrained=False, checkpointing=False)
        enc.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["ema_encoder"])
    if a.kind == "vjepa":
        enc = enc.to(dev).eval()
    rows = aslc_rows(a.split)
    if a.subset:
        keep = {l.strip() for l in open(a.subset) if l.strip()}
        rows = [r for r in rows if r["sample_id"] in keep]
    dl = torch.utils.data.DataLoader(VideoDataset(rows), batch_size=1, num_workers=2, collate_fn=lambda b: b[0])
    feats, offs, starts, nfr = [], [0], [], []
    t0 = time.time()
    buf, bidx = [], []

    def flush():
        if not buf:
            return
        x = normalize_video(torch.stack(buf).to(dev))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = enc(x).float().mean(1)
        for k, j in enumerate(bidx):
            feats[j] = z[k].cpu().numpy().astype(np.float16)
        buf.clear(); bidx.clear()

    for i, arr, fps in dl:
        n = len(arr)
        if n < LEN:
            arr = torch.cat([arr, arr[-1:].repeat(LEN - n, 1, 1, 1)]) if n else torch.zeros(LEN, 224, 224, 3, dtype=torch.uint8)
        st = window_starts(max(n, LEN))
        for s in st:
            feats.append(None); buf.append(arr[s:s + LEN]); bidx.append(len(feats) - 1)
            if len(buf) == 32:
                flush()
        starts += st; offs.append(len(starts)); nfr.append(n)
        if i % 1000 == 0:
            print(i, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    flush()
    np.save(out / "windows.npy", np.stack(feats))
    np.save(out / "offsets.npy", np.array(offs)); np.save(out / "starts.npy", np.array(starts)); np.save(out / "nframes.npy", np.array(nfr))
    with open(out / "clips.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"ckpt": a.ckpt, "kind": a.kind, "weights": a.weights, "split": a.split, "stride": STRIDE, "len": LEN, "seconds": time.time() - t0}, open(out / "meta.json", "w"))
    print("DONE", len(starts), "windows", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
