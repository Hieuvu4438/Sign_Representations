#!/usr/bin/env python3
"""Small real-cache CPU smoke with train/save/resume/extract; no official test use."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp.config import Config
from pp.data import PoseSource,Windows,collate
from pp.fit_stats import fit,fingerprint
from pp.train import Trainer,to_device,atomic_json
from pp.extract import Extractor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_train.jsonl')
    ap.add_argument('--out',default='evidence/real_cpu_smoke');ap.add_argument('--full-model',action='store_true');a=ap.parse_args()
    # Repeated-index bias gradients differ by ~1e-10 with multithread CPU scatter.
    # Single-thread replay makes checkpoint equivalence an exact assertion.
    torch.set_num_threads(1);torch.manual_seed(0)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=Config.for_arm('A1',batch=2,device='cpu',workers=0,cache=a.source,
                       **({} if a.full_model else dict(d=32,dp=16,depth=2,pdepth=1)))
    src=PoseSource(a.source);stats=fit(src,cfg,n=4,batch=2);stats.save(out/'stats.pt')
    ds=Windows(src,cfg);b=collate([ds[0],ds[1]]);tr=Trainer(cfg,stats)
    results=[]
    for i in range(2):
        loss,gn,terms=tr.step(b,i,4);results.append(dict(step=i+1,loss=loss,gn=gn))
    sha=fingerprint(src.path);tr.save(out/'checkpoint.pt',2,0,1,sha,4)
    reloaded=Trainer(cfg,stats);assert reloaded.load(out/'checkpoint.pt',sha,4)==(2,0,1)
    torch_state=torch.get_rng_state();loss1,gn1,_=tr.step(b,2,4)
    torch.set_rng_state(torch_state);loss2,gn2,_=reloaded.step(b,2,4)
    assert loss1==loss2
    for k,v in tr.model.state_dict().items():assert torch.equal(v,reloaded.model.state_dict()[k]),k
    ext=Extractor(out/'checkpoint.pt','cpu');x,c,fps=src.clip(0);features=ext.clip(x,c,fps)
    assert all(np.isfinite(v).all() for v in features.values())
    report=dict(source=a.source,source_sha256=sha,ids=src.ids[:2].tolist(),steps=results,resume_bitwise=True,config=cfg.to_dict(),
                readout_shapes={k:list(v.shape) for k,v in features.items()},device='cpu',official_test_used=False)
    atomic_json(out/'results.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
