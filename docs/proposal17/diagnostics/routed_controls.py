"""R2-N4: controls for the inductive CLS-routed normalisation N (half subset; whiten 24.94; routed K=35 ~28.3).
(a) K=1 (global; must equal whiten);  (b) routing by `grp` clusters instead of CLS (is CLS special?);  (c) 3 K-means seeds, mean/SD for K in {16,35,48};
(d) held-out-signer CV INSIDE the half-subset: 5 splits, 6 train signers as queries, clusters/statistics fit on the other signers only (queries
    are signers absent from the gallery AND from all fits; the encoder still saw them in SSL).  Reports whiten vs N and the paired difference."""
import json, numpy as np
from sklearn.cluster import KMeans
from analyze_emb import meta
from common import *
import geo_headroom as G
from routed_coral import fit_cluster_transforms, apply
EMB=f"{SCRATCH}/emb"; man=manifests(); gl=sorted({m["gloss"] for sp in man.values() for m in sp}); gi={g:i for i,g in enumerate(gl)}
zt,zv=np.load(f"{EMB}/train_teacher.npz"),np.load(f"{EMB}/val_teacher.npz"); mt,mv=meta(zt["ids"],man["train"]),meta(zv["ids"],man["val"])
yg=np.array([gi[m["gloss"]] for m in mt]); sg=np.array([m["signer"] for m in mt]); yq=np.array([gi[m["gloss"]] for m in mv]); sq=np.array([m["signer"] for m in mv])
Xg,Xq=zt["grp_fin"].astype(np.float32),zv["grp_fin"].astype(np.float32); Cg,Cq=l2(zt["cls_fin"].astype(np.float32)),l2(zv["cls_fin"].astype(np.float32))
def routed(Xg,Cg,Xq,Cq,yq_,yg_,K,seed=0,feat="cls",Fg=None,Fq=None):
    A,B=(Cg,Cq) if feat=="cls" else (Fg,Fq)
    km=KMeans(K,n_init=3,random_state=seed).fit(A); lg=km.labels_
    tr=fit_cluster_transforms(Xg,lg); Gn=apply(Xg,lg,tr); m3,W3=G.fit_pca_whiten(Gn,512,1.0)
    cen=km.cluster_centers_; route=np.argmax(l2(B)@l2(cen).T,1); Qn=apply(Xq,route,tr)
    return float(metrics(class_max_ranks(G.apply_lin(Qn,m3,W3),yq_,G.apply_lin(Gn,m3,W3),yg_))["R1"])
out={}
mu,W=G.fit_pca_whiten(Xg,512,1.0); out["whiten"]=float(metrics(class_max_ranks(G.apply_lin(Xq,mu,W),yq,G.apply_lin(Xg,mu,W),yg))["R1"])
out["K1_global"]=routed(Xg,Cg,Xq,Cq,yq,yg,1)
Zg,Zq=G.apply_lin(Xg,mu,W),G.apply_lin(Xq,mu,W)
for K in (16,35,48):
    out[f"K{K}_grp_clusters_routing"]=routed(Xg,Cg,Xq,Cq,yq,yg,K,feat="grp",Fg=Zg[:,:64],Fq=Zq[:,:64])
    v=[routed(Xg,Cg,Xq,Cq,yq,yg,K,seed=s) for s in (0,1,2)]
    out[f"K{K}_CLS_seeds"]={"values":v,"mean":float(np.mean(v)),"sd":float(np.std(v))}
    print(K,out[f"K{K}_grp_clusters_routing"],out[f"K{K}_CLS_seeds"],flush=True)
print(out,flush=True)
# (d) CV inside the half subset
rng=np.random.default_rng(0); sgu=np.unique(sg); cv=[]
for split in range(5):
    hold=rng.choice(sgu,size=6,replace=False); mq=np.isin(sg,hold)
    Xg_,yg_,Cg_=Xg[~mq],yg[~mq],Cg[~mq]; Xq_,yq_,Cq_=Xg[mq],yg[mq],Cg[mq]
    ok=np.isin(yq_,np.unique(yg_)); Xq_,yq_,Cq_=Xq_[ok],yq_[ok],Cq_[ok]
    mu_,W_=G.fit_pca_whiten(Xg_,512,1.0)
    wh=float(metrics(class_max_ranks(G.apply_lin(Xq_,mu_,W_),yq_,G.apply_lin(Xg_,mu_,W_),yg_))["R1"])
    rt=routed(Xg_,Cg_,Xq_,Cq_,yq_,yg_,35)
    cv.append({"whiten":wh,"routed_K35":rt,"delta":rt-wh,"n_queries":int(len(Xq_))}); print("cv",split,cv[-1],flush=True)
out["cv_heldout_train_signers"]={"splits":cv,"mean_delta":float(np.mean([c["delta"] for c in cv])),"min_delta":float(np.min([c["delta"] for c in cv]))}
json.dump(out,open("results/routed_controls.json","w"),indent=1)
