#!/usr/bin/env python3
"""Mechanism diagnostics from frozen embeddings: signer/gloss NN, reversal, minimal pairs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp.evaluate import normalize
from pp.probe import minimal_pairs
from pp.train import atomic_json


def nn_accuracy(embedding,labels,batch=256):
    z=normalize(embedding);correct=0
    for start in range(0,len(z),batch):
        sim=z[start:start+batch]@z.T
        sim[np.arange(len(sim)),np.arange(start,start+len(sim))]=-np.inf
        correct+=int((labels[sim.argmax(-1)]==labels[start:start+len(sim)]).sum())
    return 100*correct/len(z)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--reverse')
    ap.add_argument('--annotations',help='JSON gloss -> independently annotated phonological parameter dictionary')
    ap.add_argument('--parameter',default='Movement');ap.add_argument('--readout',default='grp');ap.add_argument('--out',required=True)
    a=ap.parse_args();z=np.load(a.features);meta=json.loads(str(z['metadata']))
    if meta['split']!='val':raise ValueError('Mechanism selection uses validation only')
    if len(z['ids'])<2:raise ValueError('Need at least two clips for NN')
    feature=z[a.readout];signer=nn_accuracy(feature,z['signer']);gloss=nn_accuracy(feature,z['gloss'])
    result=dict(signer_NN=signer,gloss_NN=gloss,signer_to_gloss_NN_ratio=signer/max(gloss,1e-9),readout=a.readout,
                nn_is_not_a_linear_probe=True)
    if a.reverse:
        r=np.load(a.reverse)
        if not np.array_equal(z['ids'],r['ids']):raise ValueError('Reverse feature order mismatch')
        cos=(normalize(feature)*normalize(r[a.readout])).sum(-1)
        result['time_reverse_cosine']=dict(mean=float(cos.mean()),median=float(np.median(cos)))
    if a.annotations:
        labels=json.loads(Path(a.annotations).read_text())
        result['minimal_pairs']=minimal_pairs(feature,z['gloss'],labels,a.parameter)
    atomic_json(a.out,result)
if __name__=='__main__':main()
