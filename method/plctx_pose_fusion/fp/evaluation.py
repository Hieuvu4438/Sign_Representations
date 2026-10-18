"""Dictionary rank: cosine -> max gallery per gloss -> pessimistic class ties."""
from __future__ import annotations
import numpy as np
import torch


def dictionary(gallery,queries,gallery_rows,query_rows,device='cpu',stop=lambda:False):
    if len(gallery)!=len(gallery_rows) or len(queries)!=len(query_rows):raise ValueError('Metadata length mismatch')
    if {r['sample_id'] for r in gallery_rows}&{r['sample_id'] for r in query_rows}:raise ValueError('Gallery/query overlap')
    classes=sorted({r['gloss'] for r in gallery_rows}|{r['gloss'] for r in query_rows})
    lab={c:i for i,c in enumerate(classes)}
    gy=np.array([lab[r['gloss']] for r in gallery_rows]);qy=np.array([lab[r['gloss']] for r in query_rows])
    gv=np.isfinite(gallery).all(-1)&(np.linalg.norm(gallery,axis=-1)>1e-12)
    qv=np.isfinite(queries).all(-1)&(np.linalg.norm(queries,axis=-1)>1e-12)
    g=torch.as_tensor(gallery[gv],dtype=torch.float32,device=device)
    g=torch.nn.functional.normalize(g,dim=-1);labels=torch.as_tensor(gy[gv],device=device)
    present=np.zeros(len(classes),bool);present[gy[gv]]=True
    ranks=np.full(len(queries),np.inf)
    for begin in range(0,len(queries),32):
        if stop():raise InterruptedError('Ranking interrupted')
        end=min(begin+32,len(queries));q=torch.as_tensor(queries[begin:end],dtype=torch.float32,device=device)
        q=torch.nn.functional.normalize(q,dim=-1)
        scores=torch.full((end-begin,len(classes)),-torch.inf,device=device)
        for offset in range(0,len(g),8192):
            sim=q@g[offset:offset+8192].T
            scores.scatter_reduce_(1,labels[offset:offset+8192][None].expand(len(q),-1),sim,reduce='amax',include_self=True)
        true=torch.as_tensor(qy[begin:end],device=device)
        truth=scores.gather(1,true[:,None])
        rank=(scores>=truth).sum(-1).cpu().numpy()
        good=qv[begin:end]&present[qy[begin:end]]
        ranks[begin:end]=np.where(good,rank,np.inf)
    metrics={'DCG':float((1/np.log2(1+ranks)).mean()),'MRR':float((1/ranks).mean()),
             **{f'Rec@{k}':float((ranks<=k).mean()) for k in (1,5,10)},
             'n_queries':len(queries),'gallery_valid':int(gv.sum()),'query_valid':int(qv.sum()),
             'macro_Rec@1':float(np.mean([(ranks[qy==i]<=1).mean() for i in np.unique(qy)]))}
    return metrics,ranks
