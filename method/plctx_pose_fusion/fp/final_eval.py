"""Selected-checkpoint full official test; resumable extraction and frozen LP."""
from __future__ import annotations
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from fp.utils import load_config,sha256,atomic_json,atomic_torch,rng_state,restore_rng
from fp.preflight import verify
from fp.trainer import Trainer
from fp.inference import split_rows,extract,l2
from fp.evaluation import dictionary


def accuracy(logits,y,valid):
    top=np.argsort(-logits,axis=-1,kind='stable')[:,:5]
    one=(top[:,0]==y)&valid;five=(top==y[:,None]).any(-1)&valid
    return {'instance_top1':float(one.mean()),'instance_top5':float(five.mean()),
            'macro_top1':float(np.mean([one[y==c].mean() for c in np.unique(y)])),
            'n_queries':len(y),'valid_queries':int(valid.sum())}


def linear_probe(features,rows,out,identity,device,stop=lambda:False,max_epochs=30):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        j=json.loads((out/'complete.json').read_text())
        if j['identity']!=identity:raise ValueError('Probe checkpoint identity mismatch')
        return j
    labels=sorted({r['gloss'] for r in rows['train']});lookup={s:i for i,s in enumerate(labels)}
    y={s:np.array([lookup[r['gloss']] for r in rs]) for s,rs in rows.items()}
    valid={s:np.isfinite(x).all(-1)&(np.linalg.norm(x,axis=-1)>1e-12) for s,x in features.items()}
    mu=features['train'][valid['train']].mean(0);sd=features['train'][valid['train']].std(0)+1e-6
    x={s:torch.as_tensor((v-mu)/sd,device=device) for s,v in features.items()}
    yt=torch.as_tensor(y['train'],device=device);good=torch.as_tensor(np.flatnonzero(valid['train']),device=device)
    torch.manual_seed(42);generator=torch.Generator().manual_seed(4342)
    model=torch.nn.Linear(len(mu),len(labels)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    epoch=0;best=-1.;stale=0
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if ck['identity']!=identity:raise ValueError('Probe source mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        generator.set_state(ck['order_rng']);epoch,best,stale=ck['epoch'],ck['best'],ck['stale']
    @torch.no_grad()
    def score(split):
        pred=torch.cat([model(x[split][i:i+1024]).cpu() for i in range(0,len(x[split]),1024)]).numpy()
        return accuracy(pred,y[split],valid[split])
    while epoch<max_epochs and stale<5:
        order=torch.randperm(len(good),generator=generator,device='cpu').to(device)
        for i in range(0,len(order),512):
            if stop():raise InterruptedError('Probe paused; incomplete epoch restarts from last commit')
            ids=good[order[i:i+512]];opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x['train'][ids]),yt[ids]);loss.backward();opt.step()
        epoch+=1;metric=score('val')
        if metric['instance_top1']>best:
            best=metric['instance_top1'];stale=0
            atomic_torch(out/'best.pt',{'model':model.state_dict(),'epoch':epoch,'validation':metric,'identity':identity})
        else:stale+=1
        atomic_torch(out/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,
            'best':best,'stale':stale,'order_rng':generator.get_state(),'identity':identity})
        atomic_json(out/f'val_{epoch:03}.json',metric)
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'])
    j={'identity':identity,'encoder_frozen':True,'selected_epoch':selected['epoch'],'validation':selected['validation'],
       'test':score('test'),'recipe':'train-only standardization, AdamW linear softmax lr/wd1e-3,batch512,30epochs,validation patience5',
       'class_count':len(labels),'test_selection':False}
    atomic_json(out/'complete.json',j);return j


def run(config,job):
    c=load_config(config);job=Path(job);out=job/'evaluation';out.mkdir(parents=True,exist_ok=True)
    if not (job/'train/complete.json').exists():raise ValueError('Training incomplete; official test not permitted')
    if c['smoke']:raise ValueError('Smoke run cannot produce official full-test result')
    cache,proof=verify(c)
    torch.set_num_threads(4)
    if c['device'].startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.23)
    tr=Trainer(c,initialize=False)
    selected=job/'train/best_validation.pt';ck=torch.load(selected,map_location='cpu',weights_only=False)
    if ck['proof']!=proof or ck['config']!=c:raise ValueError('Selected checkpoint source mismatch')
    tr.model.load_state_dict(ck['model']);identity=sha256(selected)
    if (out/'complete.json').exists():
        j=json.loads((out/'complete.json').read_text())
        if j['checkpoint_sha256']!=identity:raise ValueError('Final evaluation source changed')
        return 0
    stop={'value':False}
    def halt(*_):stop['value']=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,halt)
    def stopped():return stop['value'] or (job/'pause.request').exists()
    def status(phase,**kw):atomic_json(job/'progress.json',{'status':'running','phase':phase,'pid':os.getpid(),'updated_unix':time.time(),**kw})
    try:
        rows={s:split_rows(s) for s in ('train','val','test')}
        features={}
        for split in rows:
            status('final_extract',split=split)
            features[split]=extract(tr.model.teacher,rows[split],cache,out/'clips'/split,identity,c['device'],
                workers=c['extract_workers'],stride=2,stop=stopped,
                notify=lambda _,n,total:status('final_extract',split=split,completed=n,total=total))
            for mode,x in features[split].items():np.save(out/(split+'_'+mode+'.npy'),x)
        report={'selected_step':ck['step'],'checkpoint_sha256':identity,'proof':proof,
                'protocol':'official train gallery40154 -> val10304/test32941;2731glosses',
                'readout':'768D D/N/face;raw window mean then videoL2;64samples;stride2',
                'selection':'raw validation stride32; official test not used for selection','results':{},
                'pretraining':'PL full160 frozen; additional fusion160 epochs; Pose branch scratch'}
        for mode in ('fusion','pl_gate0'):
            status('final_rank',mode=mode)
            result={}
            values={s:features[s][mode] for s in rows}
            for split in ('val','test'):
                metrics,ranks=dictionary(values['train'],values[split],rows['train'],rows[split],c['device'],stopped)
                result['raw_'+split]=metrics
                np.save(out/(mode+'_raw_'+split+'_ranks.npy'),ranks)
            # Whitening on pooled gallery features is a secondary protocol; no query fitting.
            valid=np.isfinite(values['train']).all(-1)&(np.linalg.norm(values['train'],axis=-1)>1e-12)
            sample=values['train'][valid][::3];mu=sample.mean(0)
            with threadpool_limits(limits=4):
                _,singular,axes=np.linalg.svd(sample-mu,full_matrices=False)
                singular=singular/np.sqrt(len(sample));scale=singular[:512]+1e-3*singular[0]
                transformed={s:l2((x-mu)@axes[:512].T/scale) for s,x in values.items()}
            for s,x in transformed.items():x[np.linalg.norm(values[s],axis=-1)<=1e-12]=0
            np.savez(out/(mode+'_whitening.npz'),mean=mu,axes=axes[:512],scale=scale)
            for split in ('val','test'):
                metrics,ranks=dictionary(transformed['train'],transformed[split],rows['train'],rows[split],c['device'],stopped)
                result['white512_'+split]=metrics
                np.save(out/(mode+'_white512_'+split+'_ranks.npy'),ranks)
            status('final_linear_probe',mode=mode)
            result['linear_probe']=linear_probe(values,rows,out/('probe_'+mode),identity,c['device'],stopped)
            report['results'][mode]=result
            atomic_json(out/'report.json',report)
        atomic_json(out/'complete.json',{'status':'complete','checkpoint_sha256':identity,'report':str(out/'report.json'),'finished_unix':time.time()})
        status('complete',report=str(out/'report.json'));return 0
    except InterruptedError:
        status('paused',reason='final evaluation interrupted; cached extraction/probe retained');return 75
    except torch.cuda.OutOfMemoryError:return 76


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--job',required=True);a=p.parse_args()
    try:code=run(a.config,a.job)
    except torch.cuda.OutOfMemoryError:
        print('CUDA_OOM_DURING_EVALUATION_INITIALIZATION: requeue',flush=True);code=76
    raise SystemExit(code)
