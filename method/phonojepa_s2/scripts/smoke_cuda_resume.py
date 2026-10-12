"""Real-data bf16 CUDA checkpoint continuation check, not research training."""
from pathlib import Path
import json,random,time
import numpy as np
import torch
from s2.train import args,Trainer,Windows,Batches,DEFAULT_CACHE,provenance
from s2.paths import RUNTIME
from s2.data.cache import ClipCache
from s2.data.dataset import Cfg,collate


def main():
    torch.set_num_threads(4);torch.manual_seed(71);np.random.seed(71);random.seed(71)
    a=args(['--out',str(RUNTIME/'review'),'--smoke','--steps','4','--stats-clips','4',
            '--workers','0','--descriptor-input','both','--signature','leadlag','--lambda-signature','.1'])
    cache=ClipCache(DEFAULT_CACHE);ds=Windows(cache,Cfg(),a.seed);proof=provenance(cache,DEFAULT_CACHE)
    tr=Trainer(a);tr.fit_stats(ds)
    batches=list(Batches(3000,32,a.seed,0,3))
    t0=time.monotonic()
    for step in range(2):tr.step(collate([ds[i] for i in batches[step]]),step)
    path=RUNTIME/'review/cuda_resume.pt';tr.save(path,2,proof)
    b=collate([ds[i] for i in batches[2]])
    original=tr.step(b,2)
    other=Trainer(a);assert other.load(path,proof)==2
    resumed=other.step(b,2)
    assert original==resumed,[(k,original[k],resumed[k]) for k in original if original[k]!=resumed[k]]
    for key in ('model','aux','intervals'):
        for name,x in getattr(tr,key).state_dict().items():
            torch.testing.assert_close(x,getattr(other,key).state_dict()[name],rtol=0,atol=0)
    torch.testing.assert_close(tr.cce.c,other.cce.c,rtol=0,atol=0)
    report={'status':'PASS','kind':'smoke, real training clips, exact bf16 CUDA continuation',
            'batch':32,'descriptor_input':'both','signature':'leadlag','step_resumed':2,
            'comparison':'all logged metrics and model/aux/interval/center state exactly equal',
            'source_sha256':proof['source_sha256'],'seconds':time.monotonic()-t0}
    (RUNTIME/'review/cuda_resume.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
