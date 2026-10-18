#!/usr/bin/env python3
"""G0/G1 decisions, without initializing any SSL model from supervised weights."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp.train import atomic_json
from pp.evaluate import bootstrap_delta


def g0(runs):
    """runs list: mode alpha/beta, arm A0/A1/A2, seed, val_top1.

    Uses paired seed differences and pooled seed SD. A veto requires BOTH
    signature and moment streams to harm BOTH low-label regimes by >2 SD.
    Bootstrap evidence can be supplied in each run card independently.
    """
    result={};all_harm=True;moment_better=True
    for mode in ('alpha','beta'):
        data={arm:{int(r['seed']):float(r['val_top1']) for r in runs if r['mode']==mode and r['arm']==arm} for arm in ('A0','A1','A2')}
        seeds=set(data['A0'])&set(data['A1'])&set(data['A2'])
        if len(seeds)<2:raise ValueError(f'G0 {mode} requires >=2 shared seeds per arm')
        values={arm:np.array([data[arm][s] for s in sorted(seeds)]) for arm in data}
        sd=max(float(v.std(ddof=1)) for v in values.values());ref=values['A0'].mean()
        deltas={arm:float(v.mean()-ref) for arm,v in values.items()}
        bootstrap={}
        for arm in ('A1','A2'):
            ar=[];br=[];ss=[];gg=[]
            for seed in sorted(seeds):
                candidate=next(r for r in runs if r['mode']==mode and r['arm']==arm and int(r['seed'])==seed)
                reference=next(r for r in runs if r['mode']==mode and r['arm']=='A0' and int(r['seed'])==seed)
                if 'predictions' not in candidate or 'predictions' not in reference:
                    raise ValueError('G0 decisions need per-query predictions for signer × gloss bootstrap')
                a=np.load(candidate['predictions']);b=np.load(reference['predictions'])
                if not np.array_equal(a['ids'],b['ids']) or not np.array_equal(a['truth'],b['truth']):
                    raise ValueError('G0 query ordering or label mapping differs')
                ar.extend(np.where(a['pred']==a['truth'],1,2));br.extend(np.where(b['pred']==b['truth'],1,2))
                ss.extend(a['signer']);gg.extend(a['gloss'])
            bootstrap[arm]=bootstrap_delta(ar,br,ss,gg)
        harm=all(deltas[arm]<-2*sd and bootstrap[arm]['ci95'][1]<0 for arm in ('A1','A2'))
        all_harm &= harm
        moment_better &= values['A2'].mean()-values['A1'].mean()>2*sd
        result[mode]=dict(means={a:float(v.mean()) for a,v in values.items()},sd=sd,deltas=deltas,harm=harm,bootstrap=bootstrap)
    recommendation='Plan B (veto P in SSL)' if all_harm else 'A2 primary; A1 secondary' if moment_better else 'A1 primary; open Phase 2'
    return dict(regimes=result,recommendation=recommendation,heuristic=True,supervised_weights_must_not_initialize_ssl=True)


def g1(a0,a1,a2):
    if a1<a0-1 and a2<a0-1:return 'Run A8a/A8b and lambda_P=0.5 before concluding P harms SSL'
    if abs(a1-a2)<.5 and min(a1,a2)>a0:return 'Movement-token benefit; no evidence for H3. Prefer cheaper A2'
    return 'Continue registered screening; confirm promising arms with three seeds'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--g0');ap.add_argument('--g1',nargs=3,type=float,metavar=('A0','A1','A2'))
    ap.add_argument('--out',required=True);a=ap.parse_args()
    result=g0(json.loads(Path(a.g0).read_text())) if a.g0 else dict(recommendation=g1(*a.g1)) if a.g1 else None
    if result is None:raise ValueError('Supply --g0 or --g1')
    atomic_json(a.out,result)
if __name__=='__main__':main()
