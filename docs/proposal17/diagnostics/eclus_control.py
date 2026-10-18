"""Controls for emb_signer_cluster.py: random query groups (should give ~whiten-only) and group-size sweep for random groups."""
import json, numpy as np
from common import *
import geo_headroom as G, signer_norm as SN
d=load_baseline(); tr,va=d['train'],d['val']
Xg,yg,sg=tr['X'],tr['y'],tr['signer']; Xq,yq,sq=va['X'],va['y'],va['signer']
r1=lambda Q,Gl: float(metrics(class_max_ranks(Q,yq,Gl,yg))["R1"])
rng=np.random.default_rng(0); out={}
Gn=SN.coral(Xg,sg); m,W=G.fit_pca_whiten(Gn,512,1.0)
for K in (6,12):
    lq=rng.integers(0,K,len(Xq))
    Qn=SN.coral(Xq,lq)
    out[f'random_query_groups_K{K}_gallery-true']=r1(G.apply_lin(Qn,m,W),G.apply_lin(Gn,m,W))
# random gallery groups (70) with random query groups (6): fully random control
lg=rng.integers(0,70,len(Xg)); Gr=SN.coral(Xg,lg); m2,W2=G.fit_pca_whiten(Gr,512,1.0)
lq=rng.integers(0,6,len(Xq)); Qr=SN.coral(Xq,lq)
out['random_both']=r1(G.apply_lin(Qr,m2,W2),G.apply_lin(Gr,m2,W2))
# signer-balanced check: true val signers but gallery random groups
Qt=SN.coral(Xq,sq)
out['true_query_signers_gallery_random70']=r1(G.apply_lin(Qt,m2,W2),G.apply_lin(Gr,m2,W2))
print(out); json.dump(out,open('results/eclus_control.json','w'),indent=1)
