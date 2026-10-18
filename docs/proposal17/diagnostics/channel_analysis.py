"""D-CHAN: which PLctx channels carry gloss information and which carry signer identity? (proposal17 diagnostic G14)
Uses the per-channel mean of the final-layer tokens (ch_fin: 7 x 256) of the teacher, train-half gallery, val-half queries."""
import json, numpy as np
from common import *
import geo_headroom as G
from analyze_emb import meta, eta2, nn_stats
EMB=f"{SCRATCH}/emb"
man=manifests(); gl=sorted({m["gloss"] for sp in man.values() for m in sp}); gi={g:i for i,g in enumerate(gl)}
zt=np.load(f"{EMB}/train_teacher.npz"); zv=np.load(f"{EMB}/val_teacher.npz")
mt,mv=meta(zt["ids"],man["train"]),meta(zv["ids"],man["val"])
yg=np.array([gi[m["gloss"]] for m in mt]); sg=np.array([m["signer"] for m in mt]); yq=np.array([gi[m["gloss"]] for m in mv])
CH=["D_CONF","D_LOC","N_CONF","N_LOC","F_UP","F_MO","F_HD"]
Xg_all,Xq_all=zt["ch_fin"].astype(np.float32).reshape(len(yg),7,256),zv["ch_fin"].astype(np.float32).reshape(len(yq),7,256)
combos={**{c:[i] for i,c in enumerate(CH)},"hands(0-3)":[0,1,2,3],"D hand(0,1)":[0,1],"N hand(2,3)":[2,3],"CONF(0,2)":[0,2],"LOC(1,3)":[1,3],"face(4-6)":[4,5,6],"hands+HD":[0,1,2,3,6],"all7":list(range(7))}
res={}
for name,ch in combos.items():
    Xg=Xg_all[:,ch].reshape(len(yg),-1); Xq=Xq_all[:,ch].reshape(len(yq),-1)
    row={"dim":int(Xg.shape[1]),"R1_raw":metrics(class_max_ranks(Xq,yq,Xg,yg))["R1"]}
    k=min(256,Xg.shape[1]-1); mu,W=G.fit_pca_whiten(Xg,k,1.0); Gw,Qw=G.apply_lin(Xg,mu,W),G.apply_lin(Xq,mu,W)
    row["R1_whiten"]=metrics(class_max_ranks(Qw,yq,Gw,yg))["R1"]
    try:
        m_,Wl=G.fit_lda(Gw,yg,min(96,k-1),0.05); row["R1_lda"]=metrics(class_max_ranks(G.apply_lin(Qw,m_,Wl),yq,G.apply_lin(Gw,m_,Wl),yg))["R1"]
    except Exception: row["R1_lda"]=None
    row["eta2_gloss"]=eta2(Xg,yg); row["eta2_signer"]=eta2(Xg,sg); row["nn_same_signer"],row["nn_same_gloss"]=nn_stats(Xg,yg,sg)
    res[name]=row; print(name,{k_:(round(v,2) if isinstance(v,float) else v) for k_,v in row.items()},flush=True)
json.dump(res,open("results/channel_analysis.json","w"),indent=1)
