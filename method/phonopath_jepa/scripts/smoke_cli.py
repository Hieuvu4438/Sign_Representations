#!/usr/bin/env python3
"""Exercise public CLI commands on a tiny subset of REAL train/val pose files."""
from pathlib import Path
import json
import sys
import subprocess
import argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pp.train import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='runtime/cli_smoke');a=ap.parse_args()
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    source=ROOT.parent/'asm_jepa/data/processed/aslc'
    train=[json.loads(l) for l in (source/'metadata_train.jsonl').open()][:8]
    gloss={r['gloss'] for r in train};val=[]
    for line in (source/'metadata_val.jsonl').open():
        r=json.loads(line)
        if r['gloss'] in gloss:val.append(r)
        if len(val)>=8:break
    for split,rows in (('train',train),('val',val)):
        for r in rows:r['pose_relpath']=str((source/r['pose_relpath']).resolve())
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cfg=dict(arm='A1',d=32,dp=16,depth=2,pdepth=1,batch=2,device='cpu',workers=0,epochs=2,
             cache=str(out/'train.jsonl'),stats=str(out/'stats.pt'))
    atomic_json(out/'config.json',cfg)
    commands=[
      ['-m','pp.fit_stats','--config',str(out/'config.json'),'--out',str(out/'stats.pt'),'--clips','4'],
      ['-m','pp.train','--config',str(out/'config.json'),'--out',str(out/'run'),'--max-steps','3'],
      ['-m','pp.train','--config',str(out/'config.json'),'--out',str(out/'run'),'--max-steps','3','--resume'],
    ]
    for split in ('train','val'):
        commands.append(['-m','pp.extract','--ckpt',str(out/'run/final.pt'),'--source',str(out/f'{split}.jsonl'),
                         '--split',split,'--out',str(out/f'{split}.npz'),'--device','cpu'])
    commands.extend([
      ['-m','pp.evaluate','--gallery',str(out/'train.npz'),'--query',str(out/'val.npz'),'--out',str(out/'retrieval.json'),'--dataset','custom'],
      ['-m','pp.downstream','--config',str(out/'config.json'),'--train',str(out/'train.jsonl'),'--val',str(out/'val.jsonl'),
       '--out',str(out/'g0'),'--steps','2','--device','cpu'],
      ['-m','pp.downstream','--ckpt',str(out/'run/final.pt'),'--train',str(out/'train.jsonl'),'--val',str(out/'val.jsonl'),
       '--out',str(out/'linear'),'--linear','--steps','2','--device','cpu'],
      ['-m','pp.downstream','--ckpt',str(out/'run/final.pt'),'--train',str(out/'train.jsonl'),'--val',str(out/'val.jsonl'),
       '--out',str(out/'finetune'),'--steps','2','--device','cpu']
    ])
    for command in commands:
        print('RUN',sys.executable,*command,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    report=dict(real_train_clips=len(train),real_val_clips=len(val),commands=commands,
                retrieval=json.loads((out/'retrieval.json').read_text()),official_test_used=False,
                scientific_results=False,passed=True)
    target=ROOT/'evidence/cli_smoke.json';atomic_json(target,report)
if __name__=='__main__':main()
