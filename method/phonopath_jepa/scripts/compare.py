#!/usr/bin/env python3
"""Preregistered retrieval comparisons and Holm correction for a family of arms."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp.evaluate import bootstrap_delta,holm,decision
from pp.train import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,help='JSON list: name,candidate_ranks,reference_ranks,query')
    ap.add_argument('--a0-seed-r1',required=True,help='Comma-separated THREE A0 seed R@1 percentages')
    ap.add_argument('--out',required=True);a=ap.parse_args()
    seeds=np.array([float(x) for x in a.a0_seed_r1.split(',')])
    if len(seeds)!=3:raise ValueError('A0 needs three seeds before arm comparisons')
    comparisons=[]
    for p in json.loads(Path(a.plan).read_text()):
        q=np.load(p['query']);meta=json.loads(str(q['metadata']))
        if meta['split']!='val' or meta['limited']:raise ValueError('Comparison requires full VAL-DICT')
        c=bootstrap_delta(np.load(p['candidate_ranks']),np.load(p['reference_ranks']),q['signer'],q['gloss'])
        comparisons.append(dict(name=p['name'],**c))
    adjusted=holm([c['bootstrap_sign_p'] for c in comparisons]);sd=float(seeds.std(ddof=1))
    for c,p in zip(comparisons,adjusted):c.update(holm_p=p,decision=decision(c,sd,p))
    atomic_json(a.out,dict(a0_seed_sd=sd,tau=max(1.5,2.5*sd),comparisons=comparisons,
                          caveat='Bootstrap inference is weak with only five sizable signer clusters; Phase 2 is screening.'))
if __name__=='__main__':main()
