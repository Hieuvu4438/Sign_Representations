"""D4: per-channel gloss information (cross-signer 1-NN) vs signer information (signer probe), training-free descriptors."""
import sys, json
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1]]
import diag
D = sys.argv[1]
Ftr = diag.build(f'{D}/train14k.pkl'); Fva = diag.build(f'{D}/val.pkl')
NT = diag.NT
def add(F):
    for f in F:
        fa = f['fact'].reshape(NT, -1)      # [dloc2, dori4, dshape40, nloc2, nori4, nshape40]
        f['c_dloc'] = fa[:, 0:2].reshape(-1); f['c_dori'] = fa[:, 2:6].reshape(-1); f['c_dshape'] = fa[:, 6:46].reshape(-1)
        f['c_nloc'] = fa[:, 46:48].reshape(-1); f['c_nori'] = fa[:, 48:52].reshape(-1); f['c_nshape'] = fa[:, 52:92].reshape(-1)
        f['c_mouth'] = f['mouth']
        f['c_dloc_vel'] = np.diff(fa[:, 0:2], axis=0).reshape(-1)
add(Ftr); add(Fva)
res = {}
for k in ['c_dloc', 'c_dloc_vel', 'c_dori', 'c_dshape', 'c_nloc', 'c_nori', 'c_nshape', 'c_mouth']:
    res[k] = dict(signer_bal_acc=round(diag.signer_probe(Fva, k), 4), gloss_1nn=round(diag.gloss_1nn(Ftr, Fva, k)[0], 4))
    print(k, res[k], flush=True)
json.dump(res, open(f'{D}/diag4_results.json', 'w'), indent=1)
