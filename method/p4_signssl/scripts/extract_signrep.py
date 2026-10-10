"""Extract per-window SignRep outputs (features z^avg, latent, activity logits) for an ASL
Citizen split. Windows: 16 consecutive native frames, stride 2 (paper Sec 7.2 / upstream
example). Videos shorter than 16 frames are padded by repeating the last frame (flag
`padded`), so every clip gets >= 1 window.

Output dir contains: windows_features.npy [Nw,768] fp16, windows_latent.npy [Nw,768] fp16,
windows_active.npy [Nw,2] fp32, window_start.npy [Nw] int32, offsets.npy [Nclip+1] int64,
clips.jsonl (manifest rows + n_frames, fps, padded), meta.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4 import signrep as S  # noqa: E402
from p4.data import VideoDataset, aslc_rows, collate_single, manifest_hash, msasl_rows, wlasl_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--dataset", default="aslc", choices=["aslc", "wlasl", "msasl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32, help="windows per forward chunk (GPU memory)")
    ap.add_argument("--shard", type=int, default=2000, help="clips per resumable shard")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--perturb", default="none", choices=["none", "reverse", "static_mid", "shuffle"])
    ap.add_argument("--ckpt", default="", help="student checkpoint (backbone+fc) from train_xwp.py")
    ap.add_argument("--subset", default="", help="text file of sample_ids to keep (order of split file kept)")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        sys.exit("test split is quarantined; pass --allow-test only under a LOCKED protocol")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        sys.exit(f"{out} already complete")
    rows = {"aslc": aslc_rows, "wlasl": wlasl_rows, "msasl": msasl_rows}[a.dataset](a.split)
    if a.subset:
        keep = set(l.strip() for l in open(a.subset) if l.strip())
        rows = [r for r in rows if r["sample_id"] in keep]
    if a.limit:
        rows = rows[: a.limit]
    gen = torch.Generator().manual_seed(0)
    perm16 = torch.randperm(16, generator=gen).cuda()
    model, info = S.load_signrep("cuda")
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)["student"]
        res = model.load_state_dict(sd, strict=False)
        bad = [k for k in res.missing_keys if k.startswith(("backbone.", "fc."))]
        if bad or res.unexpected_keys:
            raise RuntimeError(f"student load mismatch {bad} {res.unexpected_keys}")
        info["student_ckpt"] = a.ckpt
        info["student_ckpt_sha256"] = S.sha256_file(a.ckpt)
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    idx16 = torch.arange(16, device="cuda")
    if a.perturb == "static_mid":
        idx16 = torch.full((16,), 8, device="cuda")
    elif a.perturb == "shuffle":
        idx16 = perm16
    t0 = time.time()
    total_w = 0
    for s0 in range(0, len(rows), a.shard):
        sp = shard_dir / f"{s0:07d}.npz"
        if sp.exists():
            continue
        sub = rows[s0 : s0 + a.shard]
        dl = torch.utils.data.DataLoader(VideoDataset(sub), batch_size=1, num_workers=a.workers,
                                         collate_fn=collate_single, prefetch_factor=4)
        F_, L_, A_, W_, N_, M_ = [], [], [], [], [], []
        for i, arr, fps in dl:
            n = arr.shape[0]
            padded = False
            if n == 0:
                raise RuntimeError(f"decode failed: {sub[i]['path']}")
            if n < 16:
                arr = torch.cat([arr, arr[-1:].expand(16 - n, -1, -1, -1)])
                padded = True
            starts = S.window_starts(arr.shape[0], 16, a.stride)
            if a.perturb == "reverse":  # whole clip reversed = every window reversed in time
                arr = arr.flip(0)
            fr = S.normalize_uint8(arr, "cuda").half()
            stt = torch.tensor(starts, device="cuda")
            for j in range(0, len(starts), a.batch):
                x = fr[stt[j : j + a.batch, None] + idx16[None]].permute(0, 2, 1, 3, 4).contiguous()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    y = model(x)
                F_.append(y["features"].float().cpu().numpy().astype(np.float16))
                L_.append(y["latent"].float().cpu().numpy().astype(np.float16))
                A_.append(y["cls_active"].float().cpu().numpy())
            W_.append(np.asarray(starts, np.int32))
            N_.append(len(starts))
            M_.append(json.dumps(dict(sub[i], n_frames=int(n), fps=fps, padded=padded)))
            total_w += len(starts)
        tmp = shard_dir / f"{s0:07d}.tmp.npz"
        np.savez(tmp, features=np.concatenate(F_), latent=np.concatenate(L_), active=np.concatenate(A_),
                 start=np.concatenate(W_), nwin=np.asarray(N_, np.int64), meta=np.asarray(M_))
        tmp.rename(sp)
        el = time.time() - t0
        print(f"shard {s0}+{len(sub)}/{len(rows)} done; {total_w} new windows {total_w/max(el,1e-6):.1f} win/s", flush=True)
    shards = [np.load(shard_dir / f"{s0:07d}.npz") for s0 in range(0, len(rows), a.shard)]
    feats = [z["features"] for z in shards]
    lats = [z["latent"] for z in shards]
    acts = [z["active"] for z in shards]
    wstart = [z["start"] for z in shards]
    n_win = np.concatenate([z["nwin"] for z in shards])
    meta_rows = [json.loads(m) for z in shards for m in z["meta"]]
    assert [m["sample_id"] for m in meta_rows] == [r["sample_id"] for r in rows]
    total_w = int(n_win.sum())
    # shards are in row order (asserted above)
    np.save(out / "windows_features.npy", np.concatenate(feats))
    np.save(out / "windows_latent.npy", np.concatenate(lats))
    np.save(out / "windows_active.npy", np.concatenate(acts))
    np.save(out / "window_start.npy", np.concatenate(wstart))
    np.save(out / "offsets.npy", np.concatenate([[0], np.cumsum(n_win)]))
    with open(out / "clips.jsonl", "w") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(info, split=a.split, stride=a.stride, n_clips=len(rows), n_windows=int(total_w),
                manifest_sha256=manifest_hash(rows), seconds=time.time() - t0,
                precision="bf16 autocast, fp16 storage", short_clip_policy="pad-last-frame",
                perturb=a.perturb, subset=a.subset, dataset=a.dataset)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    import shutil
    shutil.rmtree(shard_dir)
    print("DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
