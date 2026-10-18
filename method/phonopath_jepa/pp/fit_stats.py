"""Fit bounded, train-only robust tokenizer statistics (never val/test)."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from pp.config import Config
from pp.data import PoseSource,Windows,collate
from pp.ptokens import build
from pp.signature import RobustStats


def fingerprint(path):
    p=Path(path); h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def fit(source,cfg,n=640,batch=16):
    if source.split!='train': raise ValueError('Fit is restricted to train')
    ds=Windows(source,cfg,seed=123); records={}
    idx=np.random.default_rng(123).choice(len(ds),min(n,len(ds)),replace=False)
    for start in range(0,len(idx),batch):
        v=collate([ds[int(i)] for i in idx[start:start+batch]])['near']
        v={k:x.to(cfg.device) for k,x in v.items()}
        p=build(v['t_xy'],v['t_obs'],v['t_conf'],v['valid_f'],torch.zeros_like(v['output_hidden']),cfg,raw=True)
        for k,vs in p['records'].items():
            records.setdefault(k,[]).extend([tuple(x.cpu() for x in value) for value in vs])
    return RobustStats.fit(records,dict(split='train',source=str(source.path.resolve()),source_sha256=fingerprint(source.path),
                                        clips=int(len(idx)),seed=123,config=cfg.to_dict()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--clips',type=int,default=640); a=ap.parse_args()
    cfg=Config.load(a.config).validate();torch.set_num_threads(4)
    if not cfg.paths: raise ValueError('A0 does not require P statistics')
    stats=fit(PoseSource(cfg.cache),cfg,a.clips)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);stats.save(a.out)
    print(a.out)
if __name__=='__main__':main()
