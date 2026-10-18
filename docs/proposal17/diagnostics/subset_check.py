"""Check how the ladder numbers depend on the gallery size and on window stride: recompute raw / whiten / LDA on the stride-2 protocol
embeddings restricted to the same every-2nd train/val subsets used by emb_variants.py (stride 16)."""
import json, numpy as np
from common import *
import geo_headroom as G
d=load_baseline(); man=manifests()
for sp in ('train','val'):
    ids=set(m['sample_id'] for m in man[sp][::2]); d[sp]['keep']=np.array([i in ids for i in d[sp]['ids']])
tr,va=d['train'],d['val']; kt,kv=tr['keep'],va['keep']
Xg,yg,Xq,yq=tr['X'][kt],tr['y'][kt],va['X'][kv],va['y'][kv]
out={'n_gallery':int(kt.sum()),'n_query':int(kv.sum())}
out['raw']=metrics(class_max_ranks(Xq,yq,Xg,yg))['R1']
mu,W=G.fit_pca_whiten(Xg,512,1.0); Gw,Qw=G.apply_lin(Xg,mu,W),G.apply_lin(Xq,mu,W); out['whiten']=metrics(class_max_ranks(Qw,yq,Gw,yg))['R1']
for k in (96,256):
    m_,Wl=G.fit_lda(Gw,yg,k,0.05); out[f'lda_pca512_k{k}']=metrics(class_max_ranks(G.apply_lin(Qw,m_,Wl),yq,G.apply_lin(Gw,m_,Wl),yg))['R1']
m_,Wl=G.fit_lda(Xg,yg,96,0.05); out['lda_raw768_k96']=metrics(class_max_ranks(G.apply_lin(Xq,m_,Wl),yq,G.apply_lin(Xg,m_,Wl),yg))['R1']
print(out); json.dump(out,open('results/subset_check.json','w'),indent=1)
