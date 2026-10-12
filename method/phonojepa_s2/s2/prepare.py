"""Read-only dataset/protocol preflight; writes an evidence manifest under S2 runtime."""
from pathlib import Path
import argparse,json,importlib.metadata,sys
import numpy as np
from s2.paths import ROOT,REPO,RUNTIME,ASLC_KP,OBS_JEPA
from s2.train import DEFAULT_CACHE,provenance
from s2.data.cache import ClipCache


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--out',type=Path,default=RUNTIME/'setup_audit.json');a=p.parse_args(argv)
    cache=ClipCache(a.cache);proof=provenance(cache,a.cache)
    assert cache.xy.shape[1:]==(103,2) and cache.conf.shape==cache.xy.shape[:2]
    assert cache.offs[0]==0 and cache.offs[-1]==len(cache.xy) and np.all(np.diff(cache.offs)>0)
    assert np.all(cache.fps>0)
    assert np.isfinite(cache.conf).all() and np.all((cache.conf>=0)&(cache.conf<=1))
    assert np.array_equal(cache.conf==0,~np.isfinite(cache.xy).all(-1))
    assert 'gloss' not in np.load(a.cache).files and 'signer' not in np.load(a.cache).files
    sys.path.insert(0,str(OBS_JEPA))
    from oj.extract import rows_for
    val=rows_for('val');aslc=rows_for('aslc300')
    gallery=[r for r in aslc if r['role']=='gallery'];query=[r for r in aslc if r['role']=='query']
    assert len(gallery)==3000 and len(query)==1500 and len({r['label'] for r in gallery})==300
    assert set(cache.ids)=={r['sample_id'] for r in gallery}
    assert not set(cache.ids)&{r['sample_id'] for r in query+val}
    assert all((ASLC_KP/(r['sample_id']+'.pkl')).exists() for r in val+aslc)
    prior=REPO/'method/jepa_targets/runtime/prior_stats_aslc_train.npz'
    assert prior.exists()
    proof.update(evaluation={'val_clips':len(val),'val_signers':len({r['signer'] for r in val}),
        'gallery_clips':len(gallery),'query_clips':len(query),'classes':300,'evaluator_path':str(OBS_JEPA/'oj/evaluate.py'),
        'evaluator_sha256':__import__('s2.train',fromlist=['sha256']).sha256(OBS_JEPA/'oj/evaluate.py'),
        'prior_stats_sha256':__import__('s2.train',fromlist=['sha256']).sha256(prior)},
        environment={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','scikit-learn','opencv-python','pytest','threadpoolctl']},
        python=sys.version,status='PASS')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ['n_clips','evaluation','status']},indent=2))

if __name__=='__main__':main()
