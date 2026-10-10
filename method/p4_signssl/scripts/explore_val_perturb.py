"""EXPLORATORY (val LOSO): perturbed 2000-query subset vs unperturbed gallery of other val
signers; retrieval + ASL-LEX minimal-pair accuracy + error attribute sharing."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, pool_clips, ranks_from_scores
from p4.phonology import load_asllex, gloss_codes, minimal_pairs, minimal_pair_accuracy, error_attribute_sharing
R = Path("method/p4_signssl/runtime/feats")
def load(d, pool):
    clips = [json.loads(l) for l in open(d / "clips.jsonl")]
    off = np.load(d / "offsets.npy"); act = np.load(d / "windows_active.npy"); win = np.load(d / "windows_features.npy")
    p, _ = pool_clips(win, off, activity_weights(act) if pool == "activity" else None)
    return clips, l2n(p)
lex = load_asllex()
out = {}
for pool in ("activity", "mean"):
    gc, G = load(R / "aslc_val_s2", pool)
    gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
    y = np.array([lab[c["gloss"]] for c in gc]); sg = np.array([c["signer"] for c in gc])
    pairs, sigs = minimal_pairs(gl_, gloss_codes(gc), lex)
    idx = {c["sample_id"]: i for i, c in enumerate(gc)}
    for name in ("none", "reverse", "static_mid", "shuffle"):
        d = R / ("aslc_val_s2" if name == "none" else f"aslc_val2000_{name}")
        if not (d / "meta.json").exists(): continue
        if name == "none":
            sub = [l.strip() for l in open("method/p4_signssl/configs/val_subset2000.txt") if l.strip()]
            qi = np.array(sorted(idx[s] for s in sub)); Q = G[qi]; qc = [gc[i] for i in qi]
        else:
            qc, Q = load(d, pool); qi = np.array([idx[c["sample_id"]] for c in qc])
        S = np.zeros((len(qi), len(gl_)), np.float32)
        for s in np.unique(sg[qi]):
            m = sg[qi] == s; g = sg != s
            S[m] = class_max_scores(Q[m], G[g], y[g], len(gl_))
        r = ranks_from_scores(S, y[qi])
        out[f"{name}|{pool}"] = {"retrieval": metrics_from_ranks(r), "mp": minimal_pair_accuracy(S, y[qi], pairs),
                                 "err": error_attribute_sharing(S, y[qi], sigs)}
        o = out[f"{name}|{pool}"]
        print(f"{name:10s} {pool:8s} n={len(qi)} R@1 {o['retrieval']['R@1']:.4f} MRR {o['retrieval']['MRR']:.4f} | MP acc " +
              " ".join(f"{k}:{v['query_acc']:.3f}({v['n_trials']})" for k, v in o['mp'].items()) +
              " | err-share " + " ".join(f"{k}:{v['error_share']:.2f}/{v['chance_share']:.2f}" for k, v in o['err'].items() if k != 'n_errors'), flush=True)
Path("method/p4_signssl/runtime/explore").mkdir(exist_ok=True, parents=True)
json.dump(out, open("method/p4_signssl/runtime/explore/val_perturb_loso.json", "w"), indent=1)
