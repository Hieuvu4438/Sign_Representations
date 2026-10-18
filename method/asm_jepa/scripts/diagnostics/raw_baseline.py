"""D3: retrieval from hand-crafted pooled pose features (no learning), same gallery/query protocol (train -> val)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from asm_jepa.config import load_config
from asm_jepa.data.dataset import LabeledDataset, PoseStoreSource, SampleBuilder
from asm_jepa.data.manifest import load_split
from asm_jepa.features.stats import load_stats
from asm_jepa.utils import resolve
from asm_jepa.evaluation.retrieval import evaluate_retrieval

cfg = load_config("configs/pretrain/final.yaml", "pretrain")
root = resolve(cfg["data"]["processed_root"])
st = load_stats(root / cfg["data"]["stats_path"], 3)
builder = SampleBuilder(cfg, st)
SEG = 4

class Feat(torch.utils.data.Dataset):
    def __init__(self, split):
        self.rows = load_split(root, split)
        self.ds = LabeledDataset(PoseStoreSource(self.rows, root), builder, [r["class_id"] for r in self.rows], "eval")
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        b = self.ds[i]
        j = np.asarray(b["joints"])  # [T,114,7]
        T = j.shape[0]; out = {k: [] for k in ("shape", "pos", "face")}
        for s in range(SEG):
            sl = slice(s * T // SEG, (s + 1) * T // SEG)
            for name, js in (("lh", slice(4, 25)), ("rh", slice(25, 46))):
                v = j[sl, js, 6:7]; w = v.sum(0).clip(1e-6)
                out["shape"].append(((j[sl, js, 2:4] * v).sum(0) / w).ravel())
                vb = j[sl, js.start:js.start + 1, 5:6]; wb = vb.sum(0).clip(1e-6)
                out["pos"].append(((j[sl, js.start:js.start + 1, 0:2] * vb).sum(0) / wb).ravel())
        return {"shape": np.concatenate(out["shape"]).astype(np.float32), "pos": np.concatenate(out["pos"]).astype(np.float32),
                "usable": bool(b["usable"]), "class_id": int(self.rows[i]["class_id"]), "signer": self.rows[i]["signer_id"]}

def collect(split):
    dl = torch.utils.data.DataLoader(Feat(split), batch_size=256, num_workers=10)
    S, P, U, Y, G = [], [], [], [], []
    for b in dl:
        S.append(b["shape"].numpy()); P.append(b["pos"].numpy()); U.append(b["usable"].numpy()); Y.append(b["class_id"].numpy()); G += list(b["signer"])
    return np.concatenate(S), np.concatenate(P), np.concatenate(U), np.concatenate(Y), np.array(G)

gS, gP, gU, gY, gG = collect("train"); qS, qP, qU, qY, qG = collect("val")
n = int(max(gY.max(), qY.max()) + 1)
l2 = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
def run(name, gx, qx, whiten):
    if whiten:
        mu = gx[gU].mean(0); w, V = np.linalg.eigh(np.cov((gx[gU] - mu).T)); W = V / np.sqrt(w + 1e-3 * w.max())
        gx, qx = (gx - mu) @ W, (qx - mu) @ W
    res, _, _ = evaluate_retrieval(l2(qx), qY, l2(gx), gY, n, query_usable=qU, gallery_usable=gU, device="cpu")
    m = res["full"]; print(f"{name:40s} R@1 {100*m['r1']:6.2f} R@5 {100*m['r5']:6.2f} R@10 {100*m['r10']:6.2f} DCG {100*m['dcg']:6.2f}", flush=True)
    return {k: m[k] for k in ("r1", "r5", "r10", "dcg")}
out = {}
for whiten in (False, True):
    t = "whitened" if whiten else "raw"
    out[f"hand_shape {t}"] = run(f"hand shape (local xy, 4 segments) {t}", gS, qS, whiten)
    out[f"hand_shape+pos {t}"] = run(f"hand shape + wrist position {t}", np.concatenate([gS, gP], 1), np.concatenate([qS, qP], 1), whiten)
Path("runs/diagnostics").mkdir(parents=True, exist_ok=True)
Path("runs/diagnostics/raw_baseline.json").write_text(json.dumps(out, indent=2))
