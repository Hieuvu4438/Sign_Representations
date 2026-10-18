"""D-LOCAL: inductive, label-free, per-clip neighbourhood corrections of the frozen PLctx embedding (proposal17 diagnostic G11).
Every correction is computed from the clip itself and the gallery only (no query batch statistics, no signer id, no gloss label):
  csls        : cosine - r_gallery(g)/2 - r_query(q)/2, r = mean cosine to the k nearest gallery clips (hubness correction)
  local-mean  : x' = x - beta * mean(kNN gallery neighbours of x)  (neighbourhood centring ~ pseudo-signer centring), for gallery
                clips (self excluded) and queries alike, optionally followed by the label-free PCA whitening.
"""
import json, numpy as np, torch
from common import *
import geo_headroom as G
torch.set_num_threads(4)
d=load_baseline(); tr,va=d['train'],d['val']
Xg,yg=tr['X'],tr['y']; Xq,yq=va['X'],va['y']
r1=lambda Q,Gl: float(metrics(class_max_ranks(Q,yq,Gl,yg))["R1"])
def knn_mean(A, Gall, k, exclude_self, bs=2048):
    A=torch.as_tensor(l2(A)); Gt=torch.as_tensor(l2(Gall)); GA=torch.as_tensor(Gall)
    out=np.zeros((len(A),Gall.shape[1]),np.float32); rk=np.zeros(len(A),np.float32)
    for a in range(0,len(A),bs):
        s=A[a:a+bs]@Gt.T
        if exclude_self: s[torch.arange(s.shape[0]),torch.arange(a,a+s.shape[0])]=-2
        v,i=s.topk(k,dim=1)
        out[a:a+bs]=GA[i].mean(1).numpy(); rk[a:a+bs]=v.mean(1).numpy()
    return out, rk
out={'raw':r1(Xq,Xg)}
mu,W=G.fit_pca_whiten(Xg,512,1.0); out['whiten']=r1(G.apply_lin(Xq,mu,W),G.apply_lin(Xg,mu,W))
print(out,flush=True)
for k in (10,50,200):
    mg,rg=knn_mean(Xg,Xg,k,True); mq,rq=knn_mean(Xq,Xg,k,False)
    for beta in (0.5,1.0):
        Gl=Xg-beta*mg; Ql=Xq-beta*mq
        out[f'local-mean k{k} b{beta}']=r1(Ql,Gl)
        m2,W2=G.fit_pca_whiten(Gl,512,1.0)
        out[f'local-mean k{k} b{beta} +whiten']=r1(G.apply_lin(Ql,m2,W2),G.apply_lin(Gl,m2,W2))
        print(k,beta,out[f'local-mean k{k} b{beta}'],out[f'local-mean k{k} b{beta} +whiten'],flush=True)
    # csls on top of raw and whitened
    for tag,(A,B) in {'raw':(Xq,Xg),'whiten':(G.apply_lin(Xq,mu,W),G.apply_lin(Xg,mu,W))}.items():
        _,rg2=knn_mean(B,B,k,True); _,rq2=knn_mean(A,B,k,False)
        An,Bn=torch.as_tensor(l2(A)),torch.as_tensor(l2(B))
        order=np.argsort(yg,kind='stable'); ygs=yg[order]; classes,starts=np.unique(ygs,return_index=True); pos={c:i for i,c in enumerate(classes)}
        Bs=Bn[order]; rgs=torch.as_tensor(rg2[order])
        ranks=np.empty(len(A),np.int64)
        for a in range(0,len(A),512):
            s=An[a:a+512]@Bs.T
            s=2*s-rgs[None,:]-torch.as_tensor(rq2[a:a+512])[:,None]
            sc=np.maximum.reduceat(s.numpy(),starts,axis=1)
            tgt=np.array([sc[i,pos[y]] for i,y in enumerate(yq[a:a+512])])
            ranks[a:a+512]=(sc>=tgt[:,None]).sum(1)
        out[f'csls k{k} on {tag}']=float(metrics(ranks)['R1']); print('csls',k,tag,out[f'csls k{k} on {tag}'],flush=True)
json.dump(out,open('results/local_norm.json','w'),indent=1)
