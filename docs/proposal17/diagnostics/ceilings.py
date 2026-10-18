"""Compact, comparable ceilings (proposal17 diagnostic G16): on the SAME half-gallery / half-query subsets (20,077 / 5,152 clips),
for a set of readouts compute  R@1 raw | label-free whitened (PCA-512) | LDA (PCA-512 -> LDA-96, shrink 0.05, gloss-supervised linear ceiling)
and the share of signer / gloss variance.  Rows: PLctx teacher readouts (fin, L1, L2, hands-only, CONF-only), RANDOM-INIT encoder,
and the INPUT-token mean (no encoder).  The added value of the trained encoder over its own input features is the difference."""
import json, numpy as np
from common import *
import geo_headroom as G
from analyze_emb import meta, eta2, nn_stats
EMB=f"{SCRATCH}/emb"; man=manifests(); gl=sorted({m["gloss"] for sp in man.values() for m in sp}); gi={g:i for i,g in enumerate(gl)}
def load(v):
    zt,zv=np.load(f"{EMB}/train_{v}.npz"),np.load(f"{EMB}/val_{v}.npz"); mt,mv=meta(zt["ids"],man["train"]),meta(zv["ids"],man["val"])
    return zt,zv,np.array([gi[m["gloss"]] for m in mt]),np.array([m["signer"] for m in mt]),np.array([gi[m["gloss"]] for m in mv])
rows={}
zt,zv,yg,sg,yq=load("teacher")
ch_g=zt["ch_fin"].astype(np.float32).reshape(len(yg),7,256); ch_q=zv["ch_fin"].astype(np.float32).reshape(len(yq),7,256)
rows["teacher grp_fin (protocol readout)"]=(zt["grp_fin"].astype(np.float32),zv["grp_fin"].astype(np.float32))
rows["teacher grp_L1 (after block 1)"]=(zt["grp_L1"].astype(np.float32),zv["grp_L1"].astype(np.float32))
rows["teacher grp_L2"]=(zt["grp_L2"].astype(np.float32),zv["grp_L2"].astype(np.float32))
rows["teacher hands-only (ch 0-3)"]=(ch_g[:,:4].reshape(len(yg),-1),ch_q[:,:4].reshape(len(yq),-1))
rows["teacher CONF-only (ch 0,2)"]=(ch_g[:,[0,2]].reshape(len(yg),-1),ch_q[:,[0,2]].reshape(len(yq),-1))
rows["teacher face-only (ch 4-6)"]=(ch_g[:,4:].reshape(len(yg),-1),ch_q[:,4:].reshape(len(yq),-1))
ri=load("random"); rows["RANDOM-INIT grp_fin"]=(ri[0]["grp_fin"].astype(np.float32),ri[1]["grp_fin"].astype(np.float32))
ci_g=ri[0]["ch_fin"].astype(np.float32).reshape(len(yg),7,256); ci_q=ri[1]["ch_fin"].astype(np.float32).reshape(len(yq),7,256)
rows["RANDOM-INIT hands-only"]=(ci_g[:,:4].reshape(len(yg),-1),ci_q[:,:4].reshape(len(yq),-1))
try:
    ii=np.load(f"{EMB}/train_inputs.npz"),np.load(f"{EMB}/val_inputs.npz"); assert list(ii[0]["ids"])==list(zt["ids"]) and list(ii[1]["ids"])==list(zv["ids"])
    rows["INPUT tokens mean (no encoder), all channels"]=(ii[0]["inp"].astype(np.float32),ii[1]["inp"].astype(np.float32))
    # hands only inputs: dims CONF 170, LOC 16, CONF 170, LOC 16, UP 108, MO 126, HD 20 -> first 372
    rows["INPUT tokens mean, hands only (ch 0-3)"]=(ii[0]["inp"].astype(np.float32)[:,:372],ii[1]["inp"].astype(np.float32)[:,:372])
except Exception as e: print("no inputs yet",e)
res={}
for name,(Xg,Xq) in rows.items():
    # standardise input-like features so PCA/cosine are not dominated by large-scale dims (the encoder outputs are LayerNormed already)
    if name.startswith("INPUT"):
        m_,s_=Xg.mean(0),Xg.std(0)+1e-6; Xg,Xq=(Xg-m_)/s_,(Xq-m_)/s_
    k=min(512,Xg.shape[1]-1); mu,W=G.fit_pca_whiten(Xg,k,1.0); Gw,Qw=G.apply_lin(Xg,mu,W),G.apply_lin(Xq,mu,W)
    r=dict(dim=int(Xg.shape[1]),R1_raw=metrics(class_max_ranks(Xq,yq,Xg,yg))["R1"],R1_whiten=metrics(class_max_ranks(Qw,yq,Gw,yg))["R1"])
    m2,Wl=G.fit_lda(Gw,yg,min(96,k-1),0.05); r["R1_lda96"]=metrics(class_max_ranks(G.apply_lin(Qw,m2,Wl),yq,G.apply_lin(Gw,m2,Wl),yg))["R1"]
    r["eta2_gloss"]=eta2(Xg,yg); r["eta2_signer"]=eta2(Xg,sg); r["nn_same_signer"],r["nn_same_gloss"]=nn_stats(Xg,yg,sg)
    res[name]=r; print(name,{k_:(round(v,2) if isinstance(v,float) else v) for k_,v in r.items()},flush=True)
json.dump(res,open("results/ceilings.json","w"),indent=1)
