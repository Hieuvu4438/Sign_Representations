"""Frozen inference of a G0/linear/fine-tuned classifier on val or official test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pp.config import Config
from pp.data import PoseSource
from pp.downstream import scores
from pp.extract import Extractor
from pp.model import Encoder
from pp.signature import RobustStats
from pp.fit_stats import fingerprint
from pp.train import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ckpt',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--split',choices=['val','test'],required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--ledger',default='runtime/classification_test_ledger.json')
    a=ap.parse_args();torch.set_num_threads(4)
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    if not ck.get('supervised'):raise ValueError('Expected downstream best.pt, not an SSL checkpoint')
    cfg=Config(**ck['config']);ext=Extractor.__new__(Extractor);ext.cfg=cfg;ext.device=a.device
    ext.encoder=Encoder(cfg).to(a.device).eval().requires_grad_(False);ext.encoder.load_state_dict(ck['encoder'])
    ext.stats=RobustStats(ck['stats'],cfg.asinh) if cfg.paths else None
    dim=next(v for k,v in ck['head'].items() if k=='weight').shape[1]
    head=torch.nn.Linear(dim,len(ck['classes'])).to(a.device).eval();head.load_state_dict(ck['head'])
    labels={str(c):i for i,c in enumerate(ck['classes'])};source=PoseSource(a.source,a.split)
    if source.rows is None:raise ValueError('Classifier evaluation needs labeled metadata')
    key=[fingerprint(a.ckpt),fingerprint(a.source)];ledger=Path(a.ledger);entries=[]
    if a.split=='test':
        entries=json.loads(ledger.read_text()) if ledger.exists() else []
        if any(e['key']==key for e in entries):raise ValueError('Classifier checkpoint already evaluated on this official test')
        entries.append(dict(key=key,status='started'));ledger.parent.mkdir(parents=True,exist_ok=True);atomic_json(ledger,entries)
    logits=[];truth=[];ids=[]
    with torch.inference_mode():
        for i,r in enumerate(source.rows):
            if r['gloss'] not in labels:continue
            x,c,fps=source.clip(i);windows=ext.windows(x,c,fps)['grp+P']
            z=torch.tensor(windows,device=a.device);logits.append(head(z).mean(0).cpu().numpy())
            truth.append(labels[r['gloss']]);ids.append(r['sample_id'])
    if not logits:raise ValueError('No supported evaluation glosses')
    result=dict(metrics=scores(np.stack(logits),np.array(truth)),split=a.split,checkpoint_sha256=key[0],source_sha256=key[1],
                selected_glosses=ck['classes'],n_source=len(source),n_scored=len(ids),readout='grp+P',stride=2)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,result)
    np.savez(path.with_suffix('.predictions.npz'),logits=np.stack(logits),truth=np.array(truth),ids=np.array(ids))
    if a.split=='test':entries[-1]['status']='complete';atomic_json(ledger,entries)
    print(json.dumps(result))
if __name__=='__main__':main()
