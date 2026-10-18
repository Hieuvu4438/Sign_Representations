"""R1-F3/F10 fix: does the CLS signer code exist EARLY in training, is it stable across window offsets, and does the proposed gloss monitor work?
(proposal17 diagnostic G19b)

For three checkpoints of the same architecture - the 4k-step PLctx pilot (PJ_PLctx, 3,000 training clips), the 16k-step pilot (PJ_PLctx_16k) and the
final 160-epoch model - we extract teacher CLS and group embeddings (stride 16) on train every-4th (10,0xx clips) and val every-4th and measure:
  AMI / purity of k-means(K=35 train, 6 val) CLS clusters against true signers and against glosses (permutation baseline),
  eta^2_signer, same-signer NN, linear signer decode of the CLS (standardised, 70/30 clip split),
  ARI between the cluster assignments obtained from the EVEN windows and the ODD windows of the same clip (centroids fit on all-window CLS),
  raw R@1 of the group embedding (context).
It also replaces the v1.0 NMI(cluster, gloss) <= 0.2 rule: NMI vs AMI for CLS clusters vs random clusters on a 16,384-clip bank.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

import emb_variants as E
from analyze_emb import eta2, meta, nn_stats
from common import *  # noqa
from oj.data.clips import video_meta  # noqa: E402

torch.set_num_threads(6)
RUNS = "/home/haipd/Sign_Representations/method/phonojepa/runtime/runs"
CKPTS = {"pilot_4k": f"{RUNS}/PJ_PLctx/final.pt", "pilot_16k": f"{RUNS}/PJ_PLctx_16k/final.pt", "final_160ep": E.CKPT}


def load_enc(path, weights="teacher"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    enc = E.Encoder(c["d"], c["depth"], c["heads"]).eval()
    enc.load_state_dict(ck[weights])
    return enc


def extract(enc, rows, metas, workers=3):
    dl = torch.utils.data.DataLoader(E.DS(rows, metas), batch_size=None, num_workers=workers, prefetch_factor=4,
                                     worker_init_fn=lambda _: (torch.set_num_threads(1), __import__("cv2").setNumThreads(1)))
    cls_all, cls_even, cls_odd, grp, nwin = [], [], [], [], []
    for p in dl:
        p = {k: v.numpy() if torch.is_tensor(v) else v for k, v in p.items()}
        r = E.encode_windows({"m": enc}, p)["m"]
        c = r["cls_fin"]
        nwin.append(len(c))
        cls_all.append(c.mean(0))
        cls_even.append(c[0::2].mean(0))
        cls_odd.append(c[1::2].mean(0) if len(c) > 1 else c.mean(0))
        grp.append(r["grp_fin"].mean(0))
    return [np.stack(a) for a in (cls_all, cls_even, cls_odd, grp)] + [np.array(nwin)]


def main():
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    rows_t = man["train"][::4]
    rows_v = man["val"][::4]
    metas_t = [video_meta(r["video"])[:4] for r in rows_t]
    metas_v = [video_meta(r["video"])[:4] for r in rows_v]
    yt = np.array([gi[r["gloss"]] for r in rows_t])
    st = np.array([r["signer"] for r in rows_t])
    yv = np.array([gi[r["gloss"]] for r in rows_v])
    sv = np.array([r["signer"] for r in rows_v])
    out = {}
    for name, path in CKPTS.items():
        t0 = time.time()
        enc = load_enc(path)
        Ct, Ce_t, Co_t, Gt, nw_t = extract(enc, rows_t, metas_t)
        Cv, Ce_v, Co_v, Gv, nw_v = extract(enc, rows_v, metas_v)
        nz = np.linalg.norm(Ct, axis=1) > 0
        Ctn, Cvn = l2(Ct), l2(Cv)
        km = KMeans(35, n_init=3, random_state=0).fit(Ctn[nz])
        lab_t = km.labels_
        lab_v = KMeans(6, n_init=3, random_state=0).fit(Cvn).labels_
        # even/odd stability: assign each half-window mean to the train centroids
        ae = km.predict(l2(Ce_v)) if False else None
        cen = km.cluster_centers_
        asg = lambda A: np.argmax(l2(A) @ l2(cen).T, 1)
        mt4, mv4 = nw_t >= 4, nw_v >= 4  # R2-N2: clips with a single window make even == odd (trivial); use clips with >= 4 windows only
        ari_t = adjusted_rand_score(asg(Ce_t[mt4]), asg(Co_t[mt4]))
        ari_v = adjusted_rand_score(asg(Ce_v[mv4]), asg(Co_v[mv4]))
        Zs = StandardScaler().fit_transform(Ct[nz])
        s_nz = st[nz]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(Zs))
        cut = int(0.7 * len(Zs))
        clf = LogisticRegression(max_iter=300).fit(Zs[perm[:cut]], s_nz[perm[:cut]])
        dec = float((clf.predict(Zs[perm[cut:]]) == s_nz[perm[cut:]]).mean() * 100)
        r = {"clips_train": int(len(rows_t)), "clips_val": int(len(rows_v)),
             "AMI_signer_train_K35": float(adjusted_mutual_info_score(st[nz], lab_t)),
             "AMI_signer_val_K6": float(adjusted_mutual_info_score(sv, lab_v)),
             "AMI_gloss_train_K35": float(adjusted_mutual_info_score(yt[nz], lab_t)),
             "ARI_even_vs_odd_windows_train_ge4win": float(ari_t), "n_clips_ge4win_train": int(mt4.sum()), "n_clips_ge4win_val": int(mv4.sum()), "ARI_even_vs_odd_windows_val_ge4win": float(ari_v),
             "CLS_eta2_signer": eta2(Ct[nz], st[nz]), "CLS_signer_decode_linear_standardised": dec,
             "CLS_same_signer_NN": nn_stats(Ct[nz], yt[nz], st[nz], n=4000)[0],
             "grp_raw_R1_val_vs_train_subset": float(metrics(class_max_ranks(Gv, yv, Gt, yt))["R1"]), "seconds": time.time() - t0}
        out[name] = r
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        json.dump(out, open("results/cls_early.json", "w"), indent=1)
    # NMI vs AMI monitor on a 16,384-clip bank (final checkpoint CLS from the stored half-subset embeddings)
    z = np.load(f"{SCRATCH}/emb/train_teacher.npz")
    mt = meta(z["ids"], man["train"])
    yb = np.array([gi[m["gloss"]] for m in mt])
    sb = np.array([m["signer"] for m in mt])
    C = l2(z["cls_fin"].astype(np.float32))
    ok = np.where(np.linalg.norm(z["cls_fin"], axis=1) > 0)[0]
    bank = np.random.default_rng(1).choice(ok, 16384, replace=False)
    lab = KMeans(48, n_init=3, random_state=0).fit(C[bank]).labels_
    rl = np.random.default_rng(2).integers(0, 48, len(bank))
    out["monitor_bank16384_K48"] = {
        "CLS_clusters": {"NMI_gloss": float(normalized_mutual_info_score(yb[bank], lab)), "AMI_gloss": float(adjusted_mutual_info_score(yb[bank], lab)),
                         "NMI_signer": float(normalized_mutual_info_score(sb[bank], lab)), "AMI_signer": float(adjusted_mutual_info_score(sb[bank], lab))},
        "random_clusters": {"NMI_gloss": float(normalized_mutual_info_score(yb[bank], rl)), "AMI_gloss": float(adjusted_mutual_info_score(yb[bank], rl))}}
    print(out["monitor_bank16384_K48"])
    json.dump(out, open("results/cls_early.json", "w"), indent=1)


if __name__ == "__main__":
    sys.path.insert(0, "/home/haipd/Sign_Representations/method/obs_jepa")
    main()
