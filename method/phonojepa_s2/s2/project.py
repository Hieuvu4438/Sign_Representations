"""Fit group-readout PCA on ASLC300 gallery windows ONLY, then transform both sets.

No whitening here. The unchanged evaluator applies its historical whitening.
Output dimensions default to 768, matching baseline D/N/face concatenation.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from s2.paths import RUNTIME


def gallery_fit(windows,offsets,clips,dim=768):
    indices=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i,c in enumerate(clips) if c['role']=='gallery'])
    x=windows[indices].astype(np.float32)
    if dim>min(x.shape):raise ValueError('Insufficient gallery rows/features for requested projection dimension')
    with threadpool_limits(limits=4):
        pca=PCA(n_components=dim,svd_solver='full').fit(x)
    return pca.mean_,pca.components_,indices


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--root',type=Path,default=RUNTIME/'feats');p.add_argument('--dim',type=int,default=768)
    a=p.parse_args(argv)
    source=a.root/(a.name+'_grpraw');destination=a.root/(a.name+'_grp')
    if destination.exists():raise FileExistsError(destination)
    def load(setname):
        d=source/setname
        return d,np.load(d/'windows.npy'),np.load(d/'offsets.npy'),[json.loads(l) for l in (d/'clips.jsonl').read_text().splitlines()]
    # Validate both inputs before creating output, including checkpoint/view identity.
    for setname in ['aslc300','val']:
        for filename in ['windows.npy','offsets.npy','src_span.npy','clips.jsonl','meta.json']:
            if not (source/setname/filename).exists():raise FileNotFoundError(source/setname/filename)
    metas=[json.loads((source/s/'meta.json').read_text()) for s in ['aslc300','val']]
    for key in ['checkpoint_sha256','extraction_source_sha256','training_config','weights']:
        if metas[0].get(key)!=metas[1].get(key):raise ValueError(f'Mixed extraction provenance: {key}')
    d,win,off,clips=load('aslc300')
    mu,components,idx=gallery_fit(win,off,clips,a.dim)
    gallery_ids=[c['sample_id'] for c in clips if c['role']=='gallery']
    destination.mkdir(parents=True)
    np.savez(destination/'projection.npz',mean=mu,components=components)
    projection_hash=hashlib.sha256((destination/'projection.npz').read_bytes()).hexdigest()
    for setname in ['aslc300','val']:
        d,win,off,clips=load(setname);out=destination/setname;out.mkdir()
        with threadpool_limits(limits=4):z=((win.astype(np.float32)-mu)@components.T).astype(np.float16)
        np.save(out/'windows.npy',z)
        for name in ['offsets.npy','src_span.npy','clips.jsonl']:(out/name).write_bytes((d/name).read_bytes())
        meta=json.loads((d/'meta.json').read_text())
        meta.update(readout='_grp',group_projection_required=False,projection={'fit':'aslc300 gallery windows only','dim':a.dim,
            'gallery_ids_sha256':hashlib.sha256(json.dumps(gallery_ids).encode()).hexdigest(),'n_fit_windows':len(idx),
            'sha256':projection_hash,'whiten':False})
        (out/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Projected',a.name,'to',a.dim,'dimensions; gallery-only fit')

if __name__=='__main__':main()
