"""Resumable full-test raw/whitened retrieval and a validation-selected linear probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from signrepr.data.schema import read_manifest, load_pose, sha256_file
from signrepr.evaluation.dictionary import dictionary_metrics
from signrepr.evaluation.retrieval import normalize_embeddings
from signrepr.export.api import export_checkpoint, SignRepresentationModel
from signrepr.training.checkpoints import atomic_save
from signrepr.training.job_support import atomic_json


def require(value, message):
    if not value: raise ValueError(message)


def write_progress(root, phase, **values):
    record = {'phase':phase,'status':'running','pid':os.getpid(),'updated_unix':time.time(),**values}
    atomic_json(root/'progress.json',record)
    (root/'progress.txt').write_text(f'Pose-JEPA full ASL-Citizen | {phase}\n'+json.dumps(values,ensure_ascii=False)+'\n')


def extract_split(model, rows, split, cache, checkpoint_sha, root, stopped):
    vectors = []
    for index,row in enumerate(rows):
        if stopped(): raise InterruptedError('Final feature extraction paused')
        token = hashlib.sha256(row['sample_id'].encode()).hexdigest(); path = cache/f'{token}.npz'
        valid_cache = False
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as data:
                    valid_cache = str(data['checkpoint_sha256']) == checkpoint_sha and str(data['pose_sha256']) == row['pose_sha256']
                    if valid_cache: vector,status = data['embedding'].copy(),str(data['status'])
            except (OSError,ValueError,EOFError): pass
        if not valid_cache:
            require(sha256_file(row['pose_path']) == row['pose_sha256'],'Evaluation pose checksum mismatch')
            output = model.encode_clip(load_pose(row['pose_path']),row)
            vector,status = output['embedding'].numpy(),output['status']
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,embedding=vector,status=np.array(status),
                                checkpoint_sha256=np.array(checkpoint_sha),pose_sha256=np.array(row['pose_sha256']))
            os.replace(temporary,path)
        if status != 'ok': vector = np.zeros_like(vector)
        vectors.append(vector)
        if index % 100 == 0 or index+1 == len(rows):
            write_progress(root,'evaluate_extract',split=split,completed=index+1,total=len(rows))
    return np.stack(vectors)


def linear_metrics(logits, labels, valid):
    y = np.asarray(labels); valid = np.asarray(valid,bool)
    top = np.argsort(-np.asarray(logits),axis=1,kind='stable')[:, :min(5,logits.shape[1])]
    correct1 = (top[:,0] == y) & valid; correct5 = (top == y[:,None]).any(1) & valid
    return {'instance_top1':float(correct1.mean()),'instance_top5':float(correct5.mean()),
            'macro_top1':float(np.mean([correct1[y == c].mean() for c in np.unique(y)])),
            'macro_top5':float(np.mean([correct5[y == c].mean() for c in np.unique(y)])),
            'num_queries':len(y),'valid_queries':int(valid.sum())}


def train_linear_probe(features, rows, out, root, checkpoint_sha, stopped, device='cuda', max_epochs=30):
    directory = out/'linear_probe'; directory.mkdir(exist_ok=True)
    if (directory/'complete.json').exists():
        result = json.loads((directory/'complete.json').read_text())
        require(result['source_checkpoint_sha256'] == checkpoint_sha,'Linear-probe checkpoint changed')
        return result
    classes = sorted({r['gloss_id'] for r in rows['train']}); lookup = {label:i for i,label in enumerate(classes)}
    require(all(r['gloss_id'] in lookup for split in ('val','test') for r in rows[split]),'Probe requires training vocabulary coverage')
    labels = {s:np.array([lookup[r['gloss_id']] for r in records]) for s,records in rows.items()}
    normalized = {s:normalize_embeddings(x) for s,x in features.items()}
    train,valid_train = normalized['train']; mu = train[valid_train].mean(0); sd = train[valid_train].std(0)+1e-6
    X = {s:torch.from_numpy((values-mu)/sd).to(device) for s,(values,_) in normalized.items()}
    y = {s:torch.from_numpy(values).to(device) for s,values in labels.items()}
    good = torch.from_numpy(np.flatnonzero(valid_train)).to(device)
    torch.manual_seed(42); generator = torch.Generator().manual_seed(4342)
    model = torch.nn.Linear(train.shape[1],len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    epoch,best,stale = 0,None,0
    source_sha = {'checkpoint':checkpoint_sha, 'rows':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
    last = directory/'last.pt'
    def save():
        atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'stale':stale,
                     'order_rng':generator.get_state(),'source_sha':source_sha},last)
    if last.exists():
        state = torch.load(last,map_location='cpu',weights_only=False)
        require(state['source_sha'] == source_sha,'Linear-probe source mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        epoch,best,stale = state['epoch'],state['best'],state['stale']; generator.set_state(state['order_rng'])
    else: save()
    @torch.inference_mode()
    def evaluate(split):
        result = []
        for start in range(0,len(X[split]),1024):
            if stopped(): raise InterruptedError('Linear-probe evaluation paused')
            result.append(model(X[split][start:start+1024]).cpu().numpy())
        return linear_metrics(np.concatenate(result),labels[split],normalized[split][1])
    while epoch < max_epochs and stale < 5:
        permutation = torch.randperm(len(good),generator=generator).to(device)
        model.train()
        for start in range(0,len(permutation),512):
            if stopped(): raise InterruptedError('Linear-probe training paused; resume previous epoch boundary')
            ids = good[permutation[start:start+512]]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(X['train'][ids]),y['train'][ids]); loss.backward(); optimizer.step()
        model.eval(); metrics = evaluate('val'); score = metrics['instance_top1']; epoch += 1
        if best is None or score > best:
            best,stale = score,0
            atomic_save({'model':model.state_dict(),'epoch':epoch,'validation':metrics,'mean':mu,'std':sd,'classes':classes,
                         'source_sha':source_sha},directory/'best.pt')
        else: stale += 1
        save(); atomic_json(directory/f'validation_{epoch:03d}.json',metrics)
        write_progress(root,'evaluate_linear_probe',epoch=epoch,max_epochs=max_epochs,best_validation_top1=best,stale_validations=stale)
    best_state = torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best_state['model']); model.eval()
    result = {'source_checkpoint_sha256':checkpoint_sha,'status':'complete','encoder_frozen':True,
              'recipe':'raw1024, train-only feature standardization, linear softmax head; AdamW lr1e-3 wd1e-3, batch512, max30 epochs, validation patience5',
              'selected_epoch':best_state['epoch'],'validation':best_state['validation'],'test':evaluate('test'),
              'test_used_for_selection':False,'class_count':len(classes)}
    atomic_json(directory/'complete.json',result); return result


def run(root):
    torch.set_num_threads(4); job = json.loads((root/'job.json').read_text())
    out = root/'evaluation'; out.mkdir(exist_ok=True)
    checkpoint = root/'train/best_validation.pt'
    require((root/'train/complete.json').exists() and checkpoint.exists(),'SSL training and checkpoint selection must complete first')
    sha = sha256_file(checkpoint)
    stop = {'value':False}
    def halt(number,frame): stop['value'] = True
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,halt)
    stopped = lambda: stop['value'] or (root/'pause.request').exists()
    try:
        torch.cuda.set_per_process_memory_fraction(0.15)
        bundle = out/'bundle'
        if not (bundle/'metadata.json').exists(): export_checkpoint(checkpoint,bundle)
        model = SignRepresentationModel(bundle,'cuda')
        require(model.metadata['source_checkpoint_sha256'] == sha,'Selected evaluation checkpoint changed')
        rows = {s:read_manifest(root/'data'/f'{s}.jsonl') for s in ('train','val','test')}
        require({s:len(v) for s,v in rows.items()} == {'train':40154,'val':10304,'test':32941},'Full evaluation membership mismatch')
        cache = out/'clips'; cache.mkdir(exist_ok=True)
        features = {s:extract_split(model,records,s,cache,sha,root,stopped) for s,records in rows.items()}
        coverage = {s:{'total':len(rows[s]),'valid':int(normalize_embeddings(x)[1].sum()),
                       'failed_ids':[rows[s][i]['sample_id'] for i in np.flatnonzero(~normalize_embeddings(x)[1])]}
                    for s,x in features.items()}
        atomic_json(out/'coverage.json',coverage)
        results = {}
        def rank(values,split):
            return dictionary_metrics(values['train'],values[split],[r['gloss_id'] for r in rows['train']],
                                      [r['gloss_id'] for r in rows[split]],[r['sample_id'] for r in rows['train']],
                                      [r['sample_id'] for r in rows[split]],device='cuda',should_stop=stopped)
        for split in ('val','test'):
            write_progress(root,'evaluate_raw_retrieval',split=split)
            report = rank(features,split); atomic_json(out/f'raw_{split}_retrieval.json',report); results[f'raw_{split}'] = report['metrics']
        # Label-free whitening uses only valid train/gallery features, for both val and test.
        from threadpoolctl import threadpool_limits
        write_progress(root,'evaluate_whitening',fit_split='train',output_dim=512)
        _,valid_train = normalize_embeddings(features['train'])
        sample = features['train'][valid_train][::3]; mu = sample.mean(0)
        with threadpool_limits(limits=4):
            _,singular,axes = np.linalg.svd(sample-mu,full_matrices=False)
            singular /= np.sqrt(len(sample)); k = min(512,len(axes)); scale = singular[:k]+1e-3*singular[0]
            projected = {s:((x-mu) @ axes[:k].T)/scale for s,x in features.items()}
        for split,x in projected.items(): x[~normalize_embeddings(features[split])[1]] = 0
        np.savez(out/'gallery_whitening.npz',mean=mu,axes=axes[:k],scale=scale)
        for split in ('val','test'):
            write_progress(root,'evaluate_white512_retrieval',split=split)
            report = rank(projected,split); atomic_json(out/f'white512_{split}_retrieval.json',report); results[f'white512_{split}'] = report['metrics']
        # Free SSL inference tensors before fitting the small frozen-feature classifier.
        del model,projected; torch.cuda.empty_cache()
        results['linear_probe'] = train_linear_probe(features,rows,out,root,sha,stopped)
        report = {'model':'PJ14_HYBRID_FULL_ASLC_100E','source_checkpoint_sha256':sha,'results':results,'coverage':coverage,
                  'selection_scope':job['selection_note'],'readout':'whole-clip unique-segment per-part mean concatenation',
                  'whitening':'train-only raw1024 ->512 SVD whitening, fit every third valid gallery feature',
                  'linear_probe_note':'Full-vocabulary validation-selected softmax probe; differs from historical ASLC300 sklearn C-search.',
                  'body_part_probes':'Not scheduled in this full dictionary training job; reports contain retrieval and a frozen linear probe.'}
        atomic_json(out/'report.json',report)
        result = {'status':'complete','report':str(out/'report.json'),'source_checkpoint_sha256':sha,'finished_unix':time.time()}
        atomic_json(out/'complete.json',result); atomic_json(root/'progress.json',{'phase':'complete',**result})
        (root/'progress.txt').write_text(f'Pose-JEPA full ASL-Citizen | COMPLETE\nReport: {out}/report.json\n')
        return result
    except InterruptedError:
        return {'status':'paused'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--job-root',required=True)
    args = parser.parse_args(); print(json.dumps(run(Path(args.job_root).resolve())),flush=True)
