"""D-CLSSIGN: the CLS token is a near-perfect signer fingerprint (eta^2_signer 0.44, same-signer NN 96.6 %).  Use it as a LABEL-FREE
pseudo-signer code (proposal17 diagnostic G19).
(a) how pure are k-means clusters of the CLS embedding w.r.t. true signers (AMI, purity), train K in {35,70,140}, val K in {6,12}?
(b) per-cluster CORAL (+ label-free whiten) on the group embedding `grp_fin`, clusters from CLS, vs clusters from grp_fin, vs true signers.
Gallery = train-half (20,077), queries = val-half (5,152), teacher, stride 16 (same subsets as emb_variants.py)."""
import json, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from common import *
import geo_headroom as G, signer_norm as SN
from analyze_emb import meta
EMB=f"{SCRATCH}/emb"; man=manifests(); gl=sorted({m["gloss"] for sp in man.values() for m in sp}); gi={g:i for i,g in enumerate(gl)}
zt,zv=np.load(f"{EMB}/train_teacher.npz"),np.load(f"{EMB}/val_teacher.npz"); mt,mv=meta(zt["ids"],man["train"]),meta(zv["ids"],man["val"])
yg=np.array([gi[m["gloss"]] for m in mt]); sg=np.array([m["signer"] for m in mt]); yq=np.array([gi[m["gloss"]] for m in mv]); sq=np.array([m["signer"] for m in mv])
Xg,Xq=zt["grp_fin"].astype(np.float32),zv["grp_fin"].astype(np.float32); Cg,Cq=zt["cls_fin"].astype(np.float32),zv["cls_fin"].astype(np.float32)
r1=lambda Q,Gl: float(metrics(class_max_ranks(Q,yq,Gl,yg))["R1"])
out={"raw":r1(Xq,Xg)}
mu,W=G.fit_pca_whiten(Xg,512,1.0); out["whiten"]=r1(G.apply_lin(Xq,mu,W),G.apply_lin(Xg,mu,W))
def purity(lab,true):
    return float(sum(np.bincount(np.unique(true[lab==c],return_inverse=True)[1]).max() for c in np.unique(lab))/len(lab))
Gt,Qt=SN.coral(Xg,sg),SN.coral(Xq,sq); m2,W2=G.fit_pca_whiten(Gt,512,1.0); out["true_signers_CORAL+whiten"]=r1(G.apply_lin(Qt,m2,W2),G.apply_lin(Gt,m2,W2))
print(out,flush=True)
Cgn,Cqn=l2(Cg),l2(Cq)
for Kg in (35,70,140):
    kg=KMeans(Kg,n_init=3,random_state=0).fit(Cgn); lg=kg.labels_
    out[f"CLS_K{Kg}_train_AMI"]=float(adjusted_mutual_info_score(sg,lg)); out[f"CLS_K{Kg}_train_purity"]=purity(lg,sg)
    for Kq in (6,12):
        lq=KMeans(Kq,n_init=3,random_state=0).fit(Cqn).labels_
        out[f"CLS_Kq{Kq}_val_AMI"]=float(adjusted_mutual_info_score(sq,lq)); out[f"CLS_Kq{Kq}_val_purity"]=purity(lq,sq)
        Gn,Qn=SN.coral(Xg,lg),SN.coral(Xq,lq); m3,W3=G.fit_pca_whiten(Gn,512,1.0)
        out[f"CLS_clusters_g{Kg}_q{Kq}_CORAL+whiten"]=r1(G.apply_lin(Qn,m3,W3),G.apply_lin(Gn,m3,W3))
        print(Kg,Kq,{k:round(v,3) for k,v in out.items() if k.startswith(f"CLS_clusters_g{Kg}_q{Kq}") or k.startswith(f"CLS_K{Kg}") or k.startswith(f"CLS_Kq{Kq}")},flush=True)
# gallery clusters from CLS but queries with TRUE signers / vice versa, for attribution
lg=KMeans(70,n_init=3,random_state=0).fit(Cgn).labels_; Gn=SN.coral(Xg,lg); m3,W3=G.fit_pca_whiten(Gn,512,1.0)
out["gallery CLS70 + queries true signers"]=r1(G.apply_lin(SN.coral(Xq,sq),m3,W3),G.apply_lin(Gn,m3,W3))
lq=KMeans(6,n_init=3,random_state=0).fit(Cqn).labels_; out["gallery true signers + queries CLS6"]=r1(G.apply_lin(SN.coral(Xq,lq),m2,W2),G.apply_lin(Gt,m2,W2))
print({k:v for k,v in out.items() if "gallery" in k}); json.dump(out,open("results/cls_cluster.json","w"),indent=1)
