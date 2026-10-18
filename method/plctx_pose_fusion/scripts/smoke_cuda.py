"""Production-sized real-data CUDA correctness/continuation and memory audit."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fp.utils import load_config,atomic_json,atomic_torch,sha256
from fp.trainer import Trainer
from fp.data.stream import Windows,Batches
from fp.data.dataset import Cfg,collate
from fp.preflight import verify


def main():
    out=ROOT/'runtime/review';out.mkdir(parents=True,exist_ok=True)
    c=load_config(ROOT/'configs/smoke.json');c['warmup_steps']=1;c['gate_ramp_steps']=1;c['steps']=8
    torch.set_num_threads(4);torch.manual_seed(91)
    cache,proof=verify(c);ds=Windows(cache,Cfg(),c['seed']);tr=Trainer(c);tr.fit_stats(ds)
    plan=list(Batches(len(ds),32,c['seed'],0,4));b=collate([ds[i] for i in plan[2]])
    old={n:x.detach().cpu().clone() for n,x in tr.model.student.pl.state_dict().items()}
    teacher={n:x.detach().cpu().clone() for n,x in tr.model.teacher.graph.state_dict().items()}
    t0=time.monotonic();first=tr.update(b,2)
    assert all(np.isfinite(x) for x in first.values())
    assert first['near_graph_visible']>0 and first['near_gate_max_actual']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in tr.model.student.graph.parameters())
    assert tr.model.student.gate.grad.abs().sum()>0
    assert all(p.grad is None for p in tr.model.student.pl.parameters())
    assert all(p.grad is None for p in tr.model.teacher.parameters())
    for n,x in old.items():torch.testing.assert_close(x,tr.model.student.pl.state_dict()[n].cpu(),rtol=0,atol=0)
    assert any(not torch.equal(x,tr.model.teacher.graph.state_dict()[n].cpu()) for n,x in teacher.items())
    path=out/'cuda_resume.pt';atomic_torch(path,tr.state(3,proof,{'validated':[0]}))
    original=tr.update(b,3)
    other=Trainer(c);ck=torch.load(path,map_location='cpu',weights_only=False)
    assert other.restore(ck,proof)[0]==3
    restored=other.update(b,3)
    assert original==restored,[(k,original[k],restored[k]) for k in original if original[k]!=restored[k]]
    for name,x in tr.model.state_dict().items():torch.testing.assert_close(x,other.model.state_dict()[name],rtol=0,atol=0)
    torch.testing.assert_close(tr.center.c,other.center.c,rtol=0,atol=0)
    record={'status':'PASS','scope':'real full-corpus windows, production architecture, batch32/BF16; execution only',
        'proof':proof,'step':3,'first':first,'exact_cuda_resume':True,'frozen_PL_unchanged':True,
        'gate_and_graph_gradients':True,'teacher_no_grad':True,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'seconds':time.monotonic()-t0}
    atomic_json(out/'cuda_smoke.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='proof'},indent=2))


if __name__=='__main__':main()
