"""SignRep dictionary protocol: cosine, class-max, pessimistic ties; train-only PCA."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pp.train import atomic_json
from pp.fit_stats import fingerprint


def normalize(x):
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite embeddings')
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def class_max_ranks(gallery,query,glabels,qlabels,device='cpu',query_batch=128,gallery_batch=2048):
    glabels=np.asarray(glabels);qlabels=np.asarray(qlabels)
    classes=np.unique(glabels)
    if not set(qlabels)<=set(classes):raise ValueError('Query class absent from gallery')
    gi=np.searchsorted(classes,glabels);qi=np.searchsorted(classes,qlabels)
    g=torch.tensor(normalize(gallery),device=device);q=torch.tensor(normalize(query),device=device)
    ranks=[]
    for start in range(0,len(q),query_batch):
        qb=q[start:start+query_batch];scores=torch.full((len(qb),len(classes)),-float('inf'),device=device)
        for s in range(0,len(g),gallery_batch):
            sim=qb@g[s:s+gallery_batch].T
            idx=torch.tensor(gi[s:s+gallery_batch],device=device)[None].expand(len(qb),-1)
            scores.scatter_reduce_(1,idx,sim,reduce='amax',include_self=True)
        target=torch.tensor(qi[start:start+len(qb)],device=device)
        true=scores.gather(1,target[:,None]);rank=(scores>=true).sum(1) # includes correct class -> pessimistic ties
        ranks.append(rank.cpu().numpy())
    return np.concatenate(ranks)


def metrics(ranks,labels):
    r=np.asarray(ranks);labels=np.asarray(labels)
    return {**{f'R@{k}':float(100*np.mean(r<=k)) for k in (1,5,10)},'MRR':float(np.mean(1/r)),
            'DCG':float(np.mean(1/np.log2(1+r))),'macro_R@1':float(100*np.mean([np.mean(r[labels==c]==1) for c in np.unique(labels)])),'n':len(r)}


def pca_pair(g,q,dim=768):
    from sklearn.decomposition import PCA
    if min(g.shape)<dim:raise ValueError(f'Gallery too small for preregistered PCA {dim}: {g.shape}')
    fit=PCA(dim,svd_solver='full').fit(g)
    return fit.transform(g),fit.transform(q),dict(mean=fit.mean_,components=fit.components_)


def bootstrap_delta(a,b,signer,gloss,repetitions=2000,seed=0):
    """Paired crossed signer × gloss (pigeonhole) bootstrap, percentage points."""
    a=np.asarray(a);b=np.asarray(b);signer=np.asarray(signer);gloss=np.asarray(gloss)
    delta=(a==1).astype(float)-(b==1).astype(float)
    us,si=np.unique(signer,return_inverse=True);ug,gi=np.unique(gloss,return_inverse=True)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repetitions):
        sw=np.bincount(rng.integers(len(us),size=len(us)),minlength=len(us))
        gw=np.bincount(rng.integers(len(ug),size=len(ug)),minlength=len(ug))
        w=sw[si]*gw[gi]
        if w.sum():values.append(100*np.dot(delta,w)/w.sum())
    if not values:raise ValueError('No valid bootstrap samples')
    lo,hi=np.percentile(values,[2.5,97.5])
    observed=float(100*delta.mean());centered=np.asarray(values)-observed
    lower_p=(np.sum(centered>=observed+1.)+1)/(len(values)+1)
    upper_p=(np.sum(centered<=observed-1.)+1)/(len(values)+1)
    tost_p=float(max(lower_p,upper_p))
    p=min(1.,2*min((np.sum(np.array(values)<=0)+1)/(len(values)+1),(np.sum(np.array(values)>=0)+1)/(len(values)+1)))
    per_signer={s:float(100*delta[signer==s].mean()) for s in us}
    return dict(delta=observed,ci95=[float(lo),float(hi)],bootstrap_sign_p=float(p),tost_margin=1.,tost_p=tost_p,
                signer_delta=per_signer,large_signers_improved=int(sum(per_signer[s]>0 for s in us if (signer==s).sum()>=600)))


def holm(pvalues):
    order=np.argsort(pvalues);adjusted=np.zeros(len(order));prev=0.
    for rank,i in enumerate(order):prev=max(prev,min(1.,(len(order)-rank)*pvalues[i]));adjusted[i]=prev
    return adjusted.tolist()


def decision(comparison,seed_sd,adjusted_p,mechanism=False):
    delta=comparison['delta'];lo,hi=comparison['ci95'];tau=max(1.5,2.5*seed_sd)
    if delta>=tau and lo>0 and comparison['large_signers_improved']>=4 and adjusted_p<.05:return 'KEEP (candidate; confirm 3 seeds)'
    if abs(delta)<=1.5 and mechanism:return 'KEEP-MECH (exploratory)'
    if delta<=-1.5:return 'DROP'
    if comparison.get('tost_p',1.)<.05:return 'DROP (equivalent within ±1 point)'
    return 'INCONCLUSIVE'


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--gallery',required=True);ap.add_argument('--query',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--readout',default='grp');ap.add_argument('--pca',type=int,default=0)
    ap.add_argument('--phase',type=int,default=2);ap.add_argument('--ledger',default='runtime/test_ledger.json')
    ap.add_argument('--dataset',choices=['aslc','wlasl','custom'],default='aslc')
    a=ap.parse_args(argv);g=np.load(a.gallery);q=np.load(a.query)
    gm=json.loads(str(g['metadata']));qm=json.loads(str(q['metadata']))
    if gm['split']!='train' or qm['split'] not in ('val','test'):raise ValueError('Expected train gallery and val/test query')
    if gm['checkpoint_sha256']!=qm['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    if set(g['ids'])&set(q['ids']):raise ValueError('Gallery/query overlap')
    if qm['limited'] or gm['limited']:raise ValueError('Partial extraction cannot claim full protocol metrics')
    expected={'aslc':{'val':(40154,10304),'test':(40154,32941)},'wlasl':{'test':(14296,2879)}}
    counts=expected.get(a.dataset,{}).get(qm['split'])
    if counts and (len(g['ids']),len(q['ids']))!=counts:
        raise ValueError(f'Incomplete {a.dataset} official protocol: expected {counts}; custom subsets require --dataset custom')
    if a.readout in ('grp+P','mw','ord2') and a.pca!=768:raise ValueError('Preregistered P readouts require train-fitted PCA768')
    ledger=Path(a.ledger);entry=None
    if qm['split']=='test':
        if a.phase!=4:raise ValueError('Official test is reserved for Phase 4')
        entries=json.loads(ledger.read_text()) if ledger.exists() else []
        key=(qm['checkpoint_sha256'],qm['source_sha256'],a.readout)
        if any(tuple(e['key'])==key for e in entries):raise ValueError('This checkpoint/dataset/readout has already used test')
        entry=dict(key=key,gallery=fingerprint(a.gallery),query=fingerprint(a.query),status='started')
        entries.append(entry);ledger.parent.mkdir(parents=True,exist_ok=True);atomic_json(ledger,entries)
    G,Q=g[a.readout],q[a.readout]
    if a.pca:G,Q,transform=pca_pair(G,Q,a.pca)
    ranks=class_max_ranks(G,Q,g['gloss'],q['gloss'],a.device)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.save(out.with_suffix('.ranks.npy'),ranks)
    result=dict(metrics=metrics(ranks,q['gloss']),readout=a.readout,pca=a.pca,split=qm['split'],
                dataset=a.dataset,gallery_n=len(G),query_n=len(Q),ties='pessimistic class-max',checkpoint=qm['checkpoint_sha256'])
    atomic_json(out,result)
    if a.pca:np.savez(out.with_suffix('.pca.npz'),**transform)
    if entry is not None:entry['status']='complete';atomic_json(ledger,entries)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
