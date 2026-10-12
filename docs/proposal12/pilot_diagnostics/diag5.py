"""D5: official-protocol ASL Citizen dictionary retrieval (gallery = full train split, query = val, class-max cosine)
for training-free descriptors (raw / factorised / factorised + bone-normalised)."""
import sys, json, pickle
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1]]
import diag
D = sys.argv[1]
Ftr = diag.build(f'{D}/train_full.pkl'); Fva = diag.build(f'{D}/val.pkl')
res = {'n_train': len(Ftr), 'n_val': len(Fva)}
ga = np.array([f['gloss'] for f in Ftr]); gb = np.array([f['gloss'] for f in Fva])
classes = np.unique(ga); cidx = {c: i for i, c in enumerate(classes)}
ya = np.array([cidx[g] for g in ga]); keep = np.isin(gb, classes); yb = np.array([cidx.get(g, -1) for g in gb])
for k in ['raw', 'fact', 'fact_bn']:
    A = np.nan_to_num(diag.stack(Ftr, k)); B = np.nan_to_num(diag.stack(Fva, k))
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A = (A - mu) / sd; B = (B - mu) / sd
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-6; B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-6
    r1 = r5 = 0; n = 0
    for s in range(0, len(B), 1024):
        S = B[s:s + 1024] @ A.T                                   # q x gallery
        C = np.full((len(S), len(classes)), -9.0, np.float32)
        np.maximum.at(C.T, ya, S.T) if False else None
        for c0 in range(0, len(classes), 512):                  # class-max via sorting trick
            pass
        order = np.argsort(ya, kind='stable'); ys = ya[order]; bounds = np.r_[0, np.flatnonzero(np.diff(ys)) + 1, len(ys)]
        Ss = S[:, order]
        C = np.maximum.reduceat(Ss, bounds[:-1], axis=1)            # q x classes (sorted unique ys == range)
        rank = (C > C[np.arange(len(C)), np.clip(yb[s:s + 1024], 0, None)][:, None]).sum(1)
        kk = keep[s:s + 1024]
        r1 += ((rank == 0) & kk).sum(); r5 += ((rank < 5) & kk).sum(); n += kk.sum()
    res[k] = dict(R1=round(r1 / n, 4), R5=round(r5 / n, 4), n=int(n))
    print(k, res[k], flush=True)
json.dump(res, open(f'{D}/diag5_results.json', 'w'), indent=1)
