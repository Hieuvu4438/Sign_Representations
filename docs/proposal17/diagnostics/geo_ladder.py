"""Unified 'retrieval ladder' for the PLctx full160 embedding with per-signer breakdown and signer x gloss bootstrap CIs
(proposal17 diagnostic G12).  Conditions (all fit on TRAIN only; val queries vs train gallery; class-max cosine):
  raw (protocol) | whiten (label-free) | local-mean k200 (inductive, label-free) | CMN signer oracle + whiten |
  CORAL signer oracle + whiten | label-free transductive cluster-CORAL (Kq=6 val clusters, 70 gallery clusters) + whiten |
  LEACE signer + whiten | LDA (gloss-supervised linear ceiling, extended grid chosen on a signer-disjoint hold-out of train)
Paired bootstrap (1000 resamples of signers then glosses) gives 95% CI for R@1 and for the difference to raw / to whiten.
"""
import json, numpy as np, torch
from sklearn.cluster import KMeans
from common import *
import geo_headroom as G, signer_norm as SN
torch.set_num_threads(4)
d=load_baseline(); tr,va=d['train'],d['val']
Xg,yg,sg=tr['X'],tr['y'],tr['signer']; Xq,yq,sq,gq=va['X'],va['y'],va['signer'],va['gloss']
ranks={}
def add(name,Q,Gl): ranks[name]=class_max_ranks(Q,yq,Gl,yg); print(name,metrics(ranks[name])['R1'],flush=True)
add('raw',Xq,Xg)
mu,W=G.fit_pca_whiten(Xg,512,1.0); add('whiten',G.apply_lin(Xq,mu,W),G.apply_lin(Xg,mu,W))
# local mean k200 (inductive)
def knn_mean(A,Gall,k,excl,bs=2048):
    At=torch.as_tensor(l2(A)); Gt=torch.as_tensor(l2(Gall)); GA=torch.as_tensor(Gall); out=np.zeros((len(A),Gall.shape[1]),np.float32)
    for a in range(0,len(A),bs):
        s=At[a:a+bs]@Gt.T
        if excl: s[torch.arange(s.shape[0]),torch.arange(a,a+s.shape[0])]=-2
        out[a:a+bs]=GA[s.topk(k,dim=1)[1]].mean(1).numpy()
    return out
mg=knn_mean(Xg,Xg,200,True); mq=knn_mean(Xq,Xg,200,False); add('local-mean k200 (inductive)',Xq-mq,Xg-mg)
Gc,Qc=SN.coral(Xg,sg),SN.coral(Xq,sq); m2,W2=G.fit_pca_whiten(Gc,512,1.0); add('CORAL signer oracle + whiten',G.apply_lin(Qc,m2,W2),G.apply_lin(Gc,m2,W2))
Gm,Qm=SN.cmn(Xg,sg),SN.cmn(Xq,sq); m1,W1=G.fit_pca_whiten(Gm,512,1.0); add('CMN signer oracle + whiten',G.apply_lin(Qm,m1,W1),G.apply_lin(Gm,m1,W1))
Zg,Zq=G.apply_lin(Xg,mu,W),G.apply_lin(Xq,mu,W)
lq=KMeans(6,n_init=4,random_state=0).fit(Zq[:,:64]).labels_; lg=KMeans(70,n_init=2,random_state=0).fit(Zg[:,:64]).labels_
Gu,Qu=SN.coral(Xg,lg),SN.coral(Xq,lq); m3,W3=G.fit_pca_whiten(Gu,512,1.0); add('cluster-CORAL label-free transductive + whiten',G.apply_lin(Qu,m3,W3),G.apply_lin(Gu,m3,W3))
sgu=np.unique(sg); z=np.array([{u:i for i,u in enumerate(sgu)}[u] for u in sg]); ml,M=G.fit_leace(Xg,z); Gl=ml+(Xg-ml)@M; Ql=ml+(Xq-ml)@M
m4,W4=G.fit_pca_whiten(Gl,512,1.0); add('LEACE signer + whiten',G.apply_lin(Ql,m4,W4),G.apply_lin(Gl,m4,W4))
# LDA extended grid on signer-disjoint inner hold-out
rng=np.random.default_rng(0); hold=set(rng.choice(sgu,size=6,replace=False).tolist()); mh=np.array([s in hold for s in sg])
Xa,ya=Xg[~mh],yg[~mh]; Xh,yh=Xg[mh],yg[mh]; keep=np.isin(yh,np.unique(ya)); Xh,yh=Xh[keep],yh[keep]; sub=rng.choice(len(Xh),size=min(3000,len(Xh)),replace=False); Xh,yh=Xh[sub],yh[sub]
best=None; grid=[]
for k in (96,128,192,256,384):
    for sh in (0.005,0.01,0.02,0.05,0.1):
        m_,Wl=G.fit_lda(Xa,ya,k,sh); sc=float(metrics(class_max_ranks(G.apply_lin(Xh,m_,Wl),yh,G.apply_lin(Xa,m_,Wl),ya))['R1']); grid.append((k,sh,sc))
        if best is None or sc>best[0]: best=(sc,k,sh)
print('LDA grid best',best,flush=True)
m_,Wl=G.fit_lda(Xg,yg,best[1],best[2]); add('LDA gloss-supervised linear ceiling',G.apply_lin(Xq,m_,Wl),G.apply_lin(Xg,m_,Wl))
# per-signer and bootstrap
names=list(ranks); res={'lda_grid':grid,'lda_best':best,'conditions':{}}
hit={n:(ranks[n]<=1).astype(float) for n in names}
sgs=np.unique(sq); by={s:np.where(sq==s)[0] for s in sgs}
res['per_signer_R1']={n:{s:float(hit[n][by[s]].mean()*100) for s in sgs} for n in names}; res['per_signer_n']={s:int(len(by[s])) for s in sgs}
B=1000; bs={n:[] for n in names}
for _ in range(B):
    ss=rng.choice(sgs,size=len(sgs),replace=True); idx=np.concatenate([by[s] for s in ss]); gl=np.unique(gq[idx]); gs=set(rng.choice(gl,size=len(gl),replace=True).tolist())
    keepi=idx[np.array([g in gs for g in gq[idx]])]
    for n in names: bs[n].append(hit[n][keepi].mean()*100)
for n in names:
    a=np.array(bs[n]); res['conditions'][n]={'R1':float(hit[n].mean()*100),'ci':[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))],
        'delta_vs_raw_ci':[float(np.percentile(a-np.array(bs['raw']),2.5)),float(np.percentile(a-np.array(bs['raw']),97.5))],
        'delta_vs_whiten_ci':[float(np.percentile(a-np.array(bs['whiten']),2.5)),float(np.percentile(a-np.array(bs['whiten']),97.5))]}
json.dump(res,open('results/geo_ladder.json','w'),indent=1)
for n in names: print(n,res['conditions'][n])
