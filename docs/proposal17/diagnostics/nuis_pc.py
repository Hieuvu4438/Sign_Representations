"""D-NUISPC: which interpretable variables explain the WITHIN-GLOSS variance of the embedding on HELD-OUT SIGNERS? (proposal17 G18)
R = embedding - gloss mean (gloss means from all train clips).  Features: 30 clip-level style variables (style_vars.py).  Ridge regression with
signer-grouped 5-fold CV (held-out signers).  Reports R^2 for the top PCs of R, the variance-weighted total, and the incremental/ablation share
per variable group (MORPH, POSTURE, BEHAV, CAM); also for the signer-centred residual (within-signer, within-gloss variation).
"""
import json, numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from common import *
import style_vars as SV
d=load_baseline(); tr=d["train"]
names=[n for g in SV.GROUPS.values() for n in g]
ids,rows=SV.build("train",1); F=np.array([[r.get(n,np.nan) for n in names] for r in rows],dtype=np.float64)
pos={i:k for k,i in enumerate(tr["ids"])}; sel=np.array([k for k,i in enumerate(ids) if i in pos]); ti=[pos[ids[k]] for k in sel]
F=F[sel]; ok=np.isfinite(F).all(1); F=F[ok]; ti=np.array(ti)[ok]
X=l2(tr["X"]); y=tr["y"]; s=tr["signer"]
gm=np.zeros((y.max()+1,X.shape[1])); cnt=np.bincount(y,minlength=y.max()+1); np.add.at(gm,y,X); gm=gm/np.maximum(cnt,1)[:,None]
R=(X-gm[y])[ti]; sg=s[ti]
Z=StandardScaler().fit_transform(F)
U,sv,Vt=np.linalg.svd(R-R.mean(0),full_matrices=False)
Pc=(R-R.mean(0))@Vt[:20].T
def cv_r2(Zf,T,groups):
    num=0;den=0;per=np.zeros(T.shape[1]); denp=np.zeros(T.shape[1])
    for a,b in GroupKFold(5).split(Zf,groups=groups):
        m=Ridge(alpha=30.0).fit(Zf[a],T[a]); p=m.predict(Zf[b])
        num+=((T[b]-p)**2).sum(); den+=((T[b]-T[a].mean(0))**2).sum()
        per+=((T[b]-p)**2).sum(0); denp+=((T[b]-T[a].mean(0))**2).sum(0)
    return 1-num/den, 1-per/denp
out={"n":int(len(R)),"within_gloss_total_var":float(((R-R.mean(0))**2).sum())}
out["R2_all_dims_heldout_signers"]={}
for gname,cols in list(SV.GROUPS.items())+[("ALL",names)]:
    idx=[names.index(c) for c in cols]
    r_all,_=cv_r2(Z[:,idx],R-R.mean(0),sg); r_pc,per=cv_r2(Z[:,idx],Pc,sg)
    out["R2_all_dims_heldout_signers"][gname]={"R2_full768":float(r_all*100),"R2_top20PC":float(r_pc*100),"R2_per_PC1-8":[round(float(x)*100,1) for x in per[:8]]}
    print(gname,out["R2_all_dims_heldout_signers"][gname],flush=True)
# within-signer residual
Rs=R.copy()
for u in np.unique(sg): Rs[sg==u]-=R[sg==u].mean(0)
out["share_within_gloss_var_that_is_signer_mean"]=float(1-((Rs-Rs.mean(0))**2).sum()/((R-R.mean(0))**2).sum())
r_all,_=cv_r2(Z,Rs-Rs.mean(0),sg); out["R2_within_signer_residual_ALL"]=float(r_all*100)
for gname,cols in SV.GROUPS.items():
    idx=[names.index(c) for c in cols]; r_,_=cv_r2(Z[:,idx],Rs-Rs.mean(0),sg); out[f"R2_within_signer_residual_{gname}"]=float(r_*100)
# per-variable standardised importance for PC1-3 (full-data ridge)
imp={}
m=Ridge(alpha=30.0).fit(Z,Pc[:,:3])
for k in range(3): imp[f"PC{k+1}"]=sorted(zip(names,[round(float(c),3) for c in m.coef_[k]]),key=lambda t:-abs(t[1]))[:6]
out["top_vars_per_PC"]=imp
json.dump(out,open("results/nuis_pc.json","w"),indent=1); print(json.dumps({k:v for k,v in out.items() if k!="R2_all_dims_heldout_signers"},indent=1))
