#!/usr/bin/env python3
"""Build reviewable sequential plans for G0 / screening / full160 (no test evaluation)."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pp.train import atomic_json
from pp.config import Config


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['g0','screen','full160'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda')
    ap.add_argument('--train',default='/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_train.jsonl')
    ap.add_argument('--val',default='/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_val.jsonl')
    ap.add_argument('--asllex');ap.add_argument('--splits',default='/home/dongvk/datasets/ASL_Citizen/splits');ap.add_argument('--g0-steps',type=int,default=2000)
    a=ap.parse_args();jobs=[];py=sys.executable
    def add(name,module,args):jobs.append(dict(name=name,argv=[py,'-m',module,*args]))
    def configured(original,name):
        cfg=Config.load(ROOT/original);cfg.device=a.device;cfg.cache=a.train
        target=ROOT/'runtime/plan_configs'/f'{name}.json';target.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(target,cfg.to_dict());return str(target)
    if a.stage=='g0':
        for arm in ('A0','A1','A2'):
            config=configured(f'configs/G0_{arm}.json',f'G0_{arm}')
            if arm!='A0':add(f'stats_{arm}','pp.fit_stats',['--config',config,'--out',f'runtime/stats/{arm}.pt'])
            for mode,args in [('alpha',['--fraction','.4']),('beta',['--k','5','--glosses','300']),('full',[])]:
                for seed in (0,1):
                    name=f'G0_{arm}_{mode}_s{seed}'
                    add(name,'pp.downstream',['--config',config,'--train',a.train,'--val',a.val,'--out',f'runtime/g0/{name}',
                        '--steps',str(a.g0_steps),'--seed',str(seed),'--device',a.device,*args])
    else:
        runs=[('A0',s) for s in range(3)]+[(arm,0) for arm in ('A1','A2','A5','A6','A4','A9','A3','A7','A8a','A8b','A10')] if a.stage=='screen' else [('A1',0)]
        for arm,seed in runs:
            name=f'{arm}_s{seed}' if a.stage=='screen' else 'A1_full160'
            config=f'configs/{arm}_s{seed}.json' if a.stage=='screen' else 'configs/full160.json'
            config=configured(config,name)
            if arm!='A0':add(f'{name}_stats','pp.fit_stats',['--config',config,'--out',f'runtime/stats/{arm}.pt'])
            add(name,'pp.train',['--config',config,'--out',f'runtime/runs/{name}'])
            for epoch in ((10,20,40) if a.stage=='screen' else (40,80,120,160)):
                ck=f'runtime/runs/{name}/epoch_{epoch:03d}.pt';prefix=f'runtime/eval/{name}/e{epoch}'
                for split,source in (('train',a.train),('val',a.val)):
                    add(f'{name}_{epoch}_{split}','pp.extract',['--ckpt',ck,'--source',source,'--split',split,'--out',f'{prefix}_{split}.npz','--device',a.device])
                add(f'{name}_{epoch}_valdict','pp.evaluate',['--gallery',f'{prefix}_train.npz','--query',f'{prefix}_val.npz','--out',f'{prefix}_grp.json','--readout','grp','--device',a.device])
                if a.asllex:add(f'{name}_{epoch}_probe','pp.probe',['--train',f'{prefix}_train.npz','--val',f'{prefix}_val.npz','--asllex',a.asllex,'--splits',a.splits,'--out',f'{prefix}_probe.json'])
    atomic_json(a.out,jobs);print(f'{len(jobs)} sequential commands -> {a.out}')
if __name__=='__main__':main()
