"""Independent NumPy retrieval arithmetic and checkpoint/data execution audit."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def vector(row, weighted):
    with np.load(row['features'],allow_pickle=False) as file:
        features=file['features'].astype(np.float64)
        if weighted:
            weights=file['activity'].astype(np.float64).max(axis=1)
            value=(features*weights[:,None]).sum(axis=0)/weights.sum()
        else:
            value=features.mean(axis=0)
    return value/np.linalg.norm(value)


def verify_phase(folder,phase):
    target=folder/'retrieval'/phase
    gallery,queries=rows(target/'gallery_features.jsonl'),rows(target/'queries_features.jsonl')
    assert len(gallery)==128 and len(queries)==64
    assert all(row['source_split']=='train' for row in gallery)
    assert all(row['source_split']=='val' for row in queries)
    assert not {row['video_id'] for row in gallery}&{row['video_id'] for row in queries}
    assert not {row['signer_id'] for row in gallery}&{row['signer_id'] for row in queries}
    labels=np.array([row['label'] for row in gallery]);truth=np.array([row['label'] for row in queries])
    original=read(target/'metrics.json');report={}
    for mode in ['unweighted','activity_weighted']:
        g=np.stack([vector(row,mode=='activity_weighted') for row in gallery])
        q=np.stack([vector(row,mode=='activity_weighted') for row in queries])
        scores=q@g.T
        order=np.argsort(-scores,axis=1,kind='stable')
        ranks=[]
        for index,positions in enumerate(order):
            ranked=[]
            for position in positions:
                label=int(labels[position])
                if label not in ranked:
                    ranked.append(label)
            ranks.append(ranked.index(int(truth[index]))+1)
        ranks=np.array(ranks)
        calculated={'per_instance_top1':float((ranks<=1).mean()*100),
                    'per_instance_top5':float((ranks<=5).mean()*100),
                    'MRR':float((1/ranks).mean()),'DCG':float((1/np.log2(ranks+1)).mean())}
        for name,value in calculated.items():
            if abs(value-original[mode][name])>1e-6:
                raise ValueError(f'NumPy64/torch32 numerical ranking disagreement: {phase}/{mode}/{name}')
        predictions=read(target/f'{mode}_predictions.json')
        if [record['rank'] for record in predictions]!=ranks.tolist():
            raise ValueError('Per-query ranks disagree with independent arithmetic')
        report[mode]={'status':'PASS','independent_float64_metrics':calculated,
                      'minimum_winner_cosine_margin':float(np.min(np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2]))}
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-only',action='store_true')
    args=parser.parse_args();folder=args.output.resolve()
    report={'status':'PASS','method':'independent NumPy float64 pooling/cosine/deduplicated-class ranking',
            'step0':verify_phase(folder,'step0')}
    if not args.initial_only:
        import torch
        report['step1000']=verify_phase(folder,'step1000')
        history=rows(folder/'train/metrics.jsonl')
        assert [row['step'] for row in history]==list(range(1,1001))
        assert all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in history)
        initial=torch.load(folder/'initial.model.pt',map_location='cpu',weights_only=True)
        final=torch.load(folder/'final.model.pt',map_location='cpu',weights_only=True)
        assert initial['initialization']=='kinetics_mae' and final['step']==1000
        assert set(initial['model'])==set(final['model'])
        changed=[name for name in initial['model'] if not torch.equal(initial['model'][name],final['model'][name])]
        assert any(name.startswith('backbone.') for name in changed)
        assert any(name.startswith('fc.') for name in changed)
        assert any(name.startswith('cls_head.') for name in changed)
        checkpoint=torch.load(folder/'train/last.pt',map_location='cpu',weights_only=False)
        assert checkpoint['schema']=='signrep_training_v1' and checkpoint['step']==1000
        assert checkpoint['run_contract']['amp_dtype']=='bfloat16'
        assert checkpoint['scheduler']['completed_steps']==1000
        assert checkpoint['d_scheduler']['completed_steps']==1000
        report['training']={'status':'PASS','completed_steps':1000,'all_losses_and_gradients_finite':True,
                            'changed_model_tensors':len(changed),'strict_model_tensor_count':len(final['model']),
                            'complete_checkpoint_scheduler_steps':1000}
    name='INITIAL_VERIFICATION.json' if args.initial_only else 'VERIFICATION.json'
    (folder/name).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
