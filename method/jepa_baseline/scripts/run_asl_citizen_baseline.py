"""Configuration-driven fresh ASL-Citizen SignRep baseline and fixed validation retrieval."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from signrep_baseline.augmentation import VideoTransform
from signrep_baseline.cache import sha256, implementation_sha256
from signrep_baseline.cli import held_out_contract, environment_record
from signrep_baseline.data import PairedVideoDataset, StatefulPairSampler, collate_pairs
from signrep_baseline.model import SignRep
from signrep_baseline.preparation import validate_pretrain_audit
from signrep_baseline.retrieval import extract_windows, pool_windows, dictionary_retrieval
from signrep_baseline.training import Pretrainer, seed_everything, atomic_torch_save


def read(path):
    return json.loads(Path(path).read_text())


def status(folder, stage, **fields):
    record = {'stage':stage,'pid':os.getpid(),'unix_time':time.time(),**fields}
    (folder/f'{stage}_status.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


def inference_context(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext()


def evaluate(model, folder, phase, checkpoint_pin, device):
    config=read(folder/'run_config.json')
    gallery_rows, query_rows = held_out_contract(folder/'gallery.jsonl',folder/'queries.jsonl')
    if len(gallery_rows)!=config['train_videos'] or len(query_rows)!=config['val_queries']:
        raise ValueError('Evaluation membership differs from frozen counts')
    target = folder/'retrieval'/phase
    target.mkdir(parents=True,exist_ok=True)
    features, activities, records, total_windows = [], [], [], 0
    model.eval()
    for rows in [gallery_rows,query_rows]:
        part_features, part_activities = [], []
        for row in rows:
            feature_file = target/(row['source_id']+'.npz')
            video_pin = sha256(row['video'])
            if feature_file.is_file():
                with np.load(feature_file,allow_pickle=False) as file:
                    values = {key:file[key].copy() for key in file.files}
                meta = json.loads(str(values['metadata_json']))
                if meta.get('encoder_checkpoint_sha256') != checkpoint_pin or meta['video_sha256'] != video_pin:
                    raise ValueError('Existing frozen feature cache provenance mismatch')
            else:
                with inference_context(device):
                    values = extract_windows(model,row['video'],stride=config['retrieval_stride'],batch_size=config['retrieval_batch_size'],short_policy='reject')
                meta = json.loads(str(values['metadata_json']))
                meta.update(encoder_checkpoint_sha256=checkpoint_pin,
                            inference_compute_dtype='bfloat16' if device=='cuda' else 'float32',
                            inference_preprocessing='immutable public author Transformation; full RGB frame',
                            implementation_sha256=implementation_sha256())
                values['metadata_json'] = np.array(json.dumps(meta))
                temporary=feature_file.with_suffix('.tmp')
                with temporary.open('wb') as file:
                    np.savez_compressed(file,**values)
                os.replace(temporary,feature_file)
            part_features.append(torch.from_numpy(values['features']))
            part_activities.append(torch.from_numpy(values['activity']))
            total_windows += len(values['features'])
            records.append({**row,'features':str(feature_file),'features_sha256':sha256(feature_file),
                            'video_sha256':video_pin,'windows':len(values['features'])})
            if len(records)%32==0 or len(records)==len(gallery_rows)+len(query_rows):
                status(folder,phase,completed_videos=len(records),total_videos=len(gallery_rows)+len(query_rows),
                       total_windows=total_windows)
        features.append(part_features);activities.append(part_activities)
    result = {}
    for mode in ['unweighted','activity_weighted']:
        pooled = [torch.stack([pool_windows(f,a if mode=='activity_weighted' else None)
                               for f,a in zip(fs,acts)]) for fs,acts in zip(features,activities)]
        metrics, ranks = dictionary_retrieval(pooled[0],[row['label'] for row in gallery_rows],
                                             pooled[1],[row['label'] for row in query_rows],
                                             [row['video_id'] for row in gallery_rows],
                                             [row['video_id'] for row in query_rows],split='val')
        truth = torch.tensor([row['label'] for row in query_rows])
        rank = (ranks['class_ranking']==truth[:,None]).long().argmax(1)+1
        metrics.update(classes=len(set(row['label'] for row in gallery_rows)),
                       correct_top1=int((rank<=1).sum()),correct_top5=int((rank<=5).sum()),
                       median_rank=float(np.median(rank.numpy())),mean_rank=float(rank.float().mean()),
                       windows=total_windows,encoder_checkpoint_sha256=checkpoint_pin,
                       pooling=mode,scope=f"{config['classes']}-class fixed baseline; not full ASL-Citizen benchmark")
        result[mode] = metrics
        predictions = []
        for index,row in enumerate(query_rows):
            nearest = gallery_rows[int(ranks['nearest_video_indices'][index])]
            predictions.append({'query_id':row['video_id'],'true_label':row['label'],'true_gloss':row['gloss'],
                                'signer_id':row['signer_id'],'rank':int(rank[index]),
                                'nearest_gallery_id':nearest['video_id'],'nearest_gloss':nearest['gloss'],
                                'top5_labels':ranks['class_ranking'][index,:5].tolist()})
        (target/f'{mode}_predictions.json').write_text(json.dumps(predictions,indent=2))
    for part,rows in [('gallery',records[:len(gallery_rows)]),('queries',records[len(gallery_rows):])]:
        (target/f'{part}_features.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (target/'metrics.json').write_text(json.dumps(result,indent=2))
    status(folder,phase,result='PASS',metrics=result)
    return result


def baseline(args):
    folder=args.output.resolve();config=read(folder/'run_config.json')
    seed_everything(config['seed'])
    model=SignRep()
    model.initialize_backbone(config['kinetics_checkpoint'])
    initial=folder/'initial.model.pt'
    if initial.exists():
        state=torch.load(initial,map_location='cpu',weights_only=True)
        if state['initialization']!='kinetics_mae' or state['seed']!=config['seed'] or state['kinetics_sha256']!=sha256(config['kinetics_checkpoint']):
            raise ValueError('Initial model/config provenance mismatch')
        model.load_state_dict(state['model'],strict=True)
    else:
        atomic_torch_save({'model':model.state_dict(),'initialization':'kinetics_mae','seed':config['seed'],
                           'kinetics_sha256':sha256(config['kinetics_checkpoint']),
                           'implementation_sha256':implementation_sha256()},initial)
    model.to(config['device'])
    evaluate(model,folder,'step0',sha256(initial),config['device'])


def training_contract(folder,config,dataset):
    return {'experiment':config['experiment'],'scope':config['scope'],
            'config':config,'manifest_sha256':dataset.manifest_sha256,
            'audit_sha256':sha256(folder/'corpus/pretrain.jsonl.audit.json'),
            'initial_checkpoint_sha256':sha256(folder/'initial.model.pt'),
            'gallery_sha256':sha256(folder/'gallery.jsonl'),'queries_sha256':sha256(folder/'queries.jsonl'),
            'inventory_sha256':sha256(folder/'inventory.json'),'environment':environment_record()}


def train(args):
    folder=args.output.resolve();config=read(folder/'run_config.json')
    validate_pretrain_audit(folder/'corpus/pretrain.jsonl')
    seed_everything(config['seed'])
    dataset=PairedVideoDataset(folder/'corpus/pretrain.jsonl',VideoTransform(True))
    sampler=StatefulPairSampler(len(dataset),config['batch_views']//2,config['seed'])
    initial=torch.load(folder/'initial.model.pt',map_location='cpu',weights_only=True)
    if initial['initialization']!='kinetics_mae' or initial['seed']!=config['seed'] or initial['kinetics_sha256']!=sha256(config['kinetics_checkpoint']):
        raise ValueError('Fresh initial checkpoint provenance mismatch')
    model=SignRep();model.load_state_dict(initial['model'],strict=True)
    trainer=Pretrainer(model,device=config['device'],total_steps=config['total_steps'],
                       warmup_steps=config['warmup_steps'],max_lr=config['max_lr'],
                       amp=config['amp'],amp_dtype=config['amp_dtype'])
    contract=training_contract(folder,config,dataset)
    train_dir=folder/'train';train_dir.mkdir(parents=True,exist_ok=True)
    metrics_path=train_dir/'metrics.jsonl'
    if not args.resume and metrics_path.exists() and metrics_path.stat().st_size:
        raise ValueError('Existing training metrics require explicit checkpoint resume')
    (train_dir/'run_contract.json').write_text(json.dumps(contract,indent=2))
    if args.resume:
        trainer.resume(args.resume,sampler,contract)
        rows=[json.loads(line) for line in metrics_path.read_text().splitlines() if line] if metrics_path.exists() else []
        rows=[row for row in rows if row['step']<=trainer.step]
        if [row['step'] for row in rows]!=list(range(1,trainer.step+1)):
            raise ValueError('Metrics do not cover the resumed completed steps exactly once')
        metrics_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    started=time.monotonic()
    batch=None  # defined even when resuming at the final step (no further updates)
    while trainer.step<config['total_steps']:
        batch=collate_pairs([dataset[index] for index in sampler.next_indices()])
        start=time.monotonic();metrics=trainer.update(batch)
        metrics['update_seconds']=time.monotonic()-start
        metrics['supervision_valid_elements']={key:int(value.sum()) for key,value in batch['masks'].items()}
        metrics['supervision_total_elements']={key:value.numel() for key,value in batch['masks'].items()}
        with (train_dir/'metrics.jsonl').open('a') as file:
            file.write(json.dumps(metrics)+'\n')
        if trainer.step==1 or trainer.step%25==0:
            status(folder,'training',step=trainer.step,total_steps=config['total_steps'],
                   loss=metrics['loss'],reconstruction=metrics['reconstruction'],
                   elapsed_seconds=time.monotonic()-started,
                   cuda_reserved_MiB=torch.cuda.memory_reserved()/2**20 if config['device']=='cuda' else 0)
        if trainer.step%config['save_interval']==0 or trainer.step==config['total_steps']:
            trainer.save(train_dir/'last.pt',sampler,contract)
    final=folder/'final.model.pt'
    if final.exists():
        saved=torch.load(final,map_location='cpu',weights_only=True)
        if saved['step']!=trainer.step or saved['run_contract']!=contract or any(not torch.equal(value.cpu(),saved['model'][name]) for name,value in trainer.model.state_dict().items()):
            raise ValueError('Existing final checkpoint differs from completed training')
    else:
        atomic_torch_save({'model':trainer.model.state_dict(),'step':trainer.step,'run_contract':contract},final)
    status(folder,'training',result='PASS',step=trainer.step,total_steps=config['total_steps'],
           elapsed_seconds=time.monotonic()-started,final_checkpoint_sha256=sha256(final))
    # Training is complete. Offload optimizer/discriminator before dense inference on a shared GPU.
    encoder=trainer.model
    del trainer,dataset,batch
    gc.collect()
    if config['device']=='cuda':
        torch.cuda.empty_cache()
    evaluate(encoder,folder,f"step{config['total_steps']}",sha256(final),config['device'])
    report(args)


def report(args):
    folder=args.output.resolve();config=read(folder/'run_config.json');inventory=read(folder/'inventory.json')
    phase=f"step{config['total_steps']}"
    initial=read(folder/'retrieval/step0/metrics.json');final=read(folder/'retrieval'/phase/'metrics.json')
    history=[json.loads(line) for line in (folder/'train/metrics.jsonl').read_text().splitlines() if line]
    if [row['step'] for row in history]!=list(range(1,config['total_steps']+1)):
        raise ValueError('Training history does not cover the completed frozen budget')
    coverage={key:sum(row['supervision_valid_elements'][key] for row in history)/sum(row['supervision_total_elements'][key] for row in history) for key in history[0]['supervision_valid_elements']}
    table=[]
    for mode in ['unweighted','activity_weighted']:
        for step,metrics in [(0,initial[mode]),(config['total_steps'],final[mode])]:
            table.append(f"| {mode} | {step} | {metrics['correct_top1']}/{config['val_queries']} ({metrics['per_instance_top1']:.4f}%) | {metrics['correct_top5']}/{config['val_queries']} ({metrics['per_instance_top5']:.4f}%) | {metrics['MRR']:.6f} | {metrics['DCG']:.6f} |")
    lines=[f"# {config['experiment']} — fresh ASL-Citizen SignRep baseline",'',
           '## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run',
           '- Verification Status: UNVERIFIED','- Execution Status: completed; one seed; no independent full training rerun',
           f"- Version Label: {config['experiment']}_v1",'',
           '## Protocol','',
           f"{config['classes']} fixed real classes; {config['train_videos']} official train/gallery videos; {config['val_queries']} signer-disjoint official validation queries. No test evaluation. Gloss metadata does not enter the self-supervised training loss.",
           'Fresh official Kinetics MAE backbone plus random SignRep heads. No public SignRep or previous ASL pilot training checkpoint is loaded.',
           f"SignRep objectives train for {config['total_steps']} updates, {config['batch_views']} views/{config['batch_views']//2} distinct videos per batch,80% masking, {config['warmup_steps']} warmup updates,LR {config['max_lr']}. BF16 compute is explicit. Corpus bone medians use only selected training videos.",
           f"Same gallery/query membership at step0/final. Original full-frame inference preprocessing,16-frame windows,stride {config['retrieval_stride']},inference batch {config['retrieval_batch_size']},BF16 compute/float32 feature caches. Video mean or raw max(LH,RH) activity weighting,then L2/cosine. Distinct classes rank by best individual gallery video.",'',
           '## Retrieval results','',
           '| Pooling | Step | Top1 | Top5 | MRR | DCG |','|---|---:|---:|---:|---:|---:|',*table,'',
           'Instance and macro Top1/Top5 are mathematically equal because each class has five queries. This is a reduced-data/compute single-seed baseline,not a500k-step YT-SL paper-score reproduction. ASLC001 scores use a different64-class dictionary and cannot be compared directly. Step0 activity head is random; supervised activity coverage below must qualify weighted pooling.','',
           '## Observed supervision coverage','',
           'Fraction of valid target elements across all actually sampled batches; repeated views count each time. Missing targets remain masked.','',
           '| Prior | Valid fraction |','|---|---:|',*[f'| {key} | {value:.4%} |' for key,value in coverage.items()], '',
           '## Artifacts','',
           '`run_config.json`, `inventory.json`, official manifests, `corpus/pretrain.jsonl.audit.json`, corpus bone table, `train/metrics.jsonl`, complete `train/last.pt`, lightweight `final.model.pt`, per-phase feature manifests/metrics/predictions. Reviewable final summaries are archived under experiments/<experiment>.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary={'status':'COMPLETED','scope':config['scope'],'initial':initial,'final':final,'config':config,'inventory':inventory,
             'implementation_sha256':implementation_sha256(),'initial_sha256':sha256(folder/'initial.model.pt'),
             'final_sha256':sha256(folder/'final.model.pt'),'supervision_coverage':coverage,
             'report':str(folder/'REPORT.md')}
    (folder/'RESULTS.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'status':'COMPLETED','report':str(folder/'REPORT.md'),'final':final}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['baseline','train','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--memory-fraction',type=float,default=.042)
    args=parser.parse_args()
    config=read(args.output/'run_config.json')
    if args.memory_fraction != config['cuda_memory_fraction']:
        raise ValueError('GPU allocation limit differs from recorded run configuration')
    torch.set_num_threads(2)
    if config['device']=='cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    {'baseline':baseline,'train':train,'report':report}[args.stage](args)


if __name__=='__main__':
    main()
