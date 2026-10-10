import numpy as np


def paired_gloss_bootstrap(first,second,replicates=1000,seed=2026,metric='r1'):
    a,b = ({r['video_id']:r for r in rows} for rows in (first,second))
    if set(a)!=set(b): raise ValueError('Paired bootstrap requires identical query coverage')
    if any(a[k]['gloss_id']!=b[k]['gloss_id'] for k in a): raise ValueError('Paired labels differ')
    glosses = sorted(set(r['gloss_id'] for r in a.values()))
    deltas = np.array([np.mean([b[k][metric]-a[k][metric] for k in a if a[k]['gloss_id']==g]) for g in glosses])
    rng = np.random.default_rng(seed)
    draws = deltas[rng.integers(len(glosses),size=(replicates,len(glosses)))].mean(1)
    return {'mean_delta':float(deltas.mean()),'ci95':[float(v) for v in np.quantile(draws,[.025,.975])],
            'replicates':replicates,'clusters':len(glosses),'cluster':'gloss','metric':metric,
            'scope':'Evaluation-gloss sampling uncertainty at these fixed checkpoints; not training-seed uncertainty'}
