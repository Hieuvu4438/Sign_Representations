"""NumPy ranking/metric verification, float32 score replay and float64 sensitivity."""
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


def vectors(row, weighted):
    """Replay the specified float32 arithmetic; separately pool in NumPy float64.

    Float32 pooling/cosine uses Torch to preserve its reduction arithmetic. Ranking
    and metrics below are independent NumPy implementations, not production code.
    The float64 path is a sensitivity audit, not an exact-rank equality requirement.
    """
    import torch
    import torch.nn.functional as F
    with np.load(row['features'], allow_pickle=False) as file:
        f64 = file['features'].astype(np.float64)
        f32 = torch.from_numpy(file['features'].copy()).float()
        if weighted:
            w64 = file['activity'].astype(np.float64).max(axis=1)
            w32 = torch.from_numpy(file['activity'].copy()).float().max(dim=1).values
            v64 = (f64 * w64[:, None]).sum(axis=0) / w64.sum()
            v32 = (f32 * w32[:, None]).sum(dim=0) / w32.sum()
        else:
            v64, v32 = f64.mean(axis=0), f32.mean(dim=0)
    if not np.isfinite(v64).all() or np.linalg.norm(v64) == 0:
        raise ValueError('Invalid pooled feature')
    return v64 / np.linalg.norm(v64), F.normalize(v32, dim=0)


def numpy_ranking(scores, labels, truth):
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks, top5 = [], []
    for label, positions in zip(truth, order):
        ranked = list(dict.fromkeys(labels[positions].tolist()))
        ranks.append(ranked.index(int(label)) + 1)
        top5.append(ranked[:5])
    return np.array(ranks), top5, order[:, 0]


def numpy_metrics(ranks):
    return {'per_instance_top1': float((ranks <= 1).mean() * 100),
            'per_instance_top5': float((ranks <= 5).mean() * 100),
            'MRR': float((1 / ranks).mean()),
            'DCG': float((1 / np.log2(ranks + 1)).mean())}


def verify_phase(folder,phase,config):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    target=folder/'retrieval'/phase
    gallery,queries=rows(target/'gallery_features.jsonl'),rows(target/'queries_features.jsonl')
    assert len(gallery)==config['train_videos'] and len(queries)==config['val_queries']
    assert {row['label'] for row in gallery}=={row['label'] for row in queries}
    assert len({row['label'] for row in gallery})==config['classes']
    for part,name in [(gallery,'gallery.jsonl'),(queries,'queries.jsonl')]:
        assert [row['video_id'] for row in part]==[row['video_id'] for row in rows(folder/name)]
    assert all(row['source_split']=='train' for row in gallery)
    assert all(row['source_split']=='val' for row in queries)
    assert not {row['video_id'] for row in gallery}&{row['video_id'] for row in queries}
    assert not {row['signer_id'] for row in gallery}&{row['signer_id'] for row in queries}
    labels=np.array([row['label'] for row in gallery]);truth=np.array([row['label'] for row in queries])
    original=read(target/'metrics.json');report={}
    for mode in ['unweighted','activity_weighted']:
        gv=[vectors(row,mode=='activity_weighted') for row in gallery]
        qv=[vectors(row,mode=='activity_weighted') for row in queries]
        g=np.stack([value[0] for value in gv]);q=np.stack([value[0] for value in qv])
        scores=q@g.T
        # Production dictionary retrieval normalizes the already pooled vectors again.
        g32=F.normalize(torch.stack([value[1] for value in gv]),dim=1)
        q32=F.normalize(torch.stack([value[1] for value in qv]),dim=1)
        scores32=(q32@g32.T).numpy()
        if not np.isfinite(scores).all() or not np.isfinite(scores32).all():
            raise ValueError('Nonfinite retrieval similarities')
        ranks,top5,nearest=numpy_ranking(scores32,labels,truth)
        ranks64,_,_=numpy_ranking(scores,labels,truth)
        calculated=numpy_metrics(ranks)
        for name,value in calculated.items():
            if abs(value-original[mode][name])>(1e-5 if name.startswith('per_instance') else 1e-7):
                raise ValueError(f'Independent ranking/metric disagreement: {phase}/{mode}/{name}')
        predictions=read(target/f'{mode}_predictions.json')
        if [record['rank'] for record in predictions]!=ranks.tolist():
            raise ValueError('Per-query ranks disagree with independent float32 score replay/NumPy ranking')
        for i,(record,row) in enumerate(zip(predictions,queries)):
            assert record['query_id']==row['video_id'] and record['true_label']==row['label']
            assert record['top5_labels']==top5[i]
            assert record['nearest_gallery_id']==gallery[int(nearest[i])]['video_id']
        assert original[mode]['correct_top1']==int((ranks<=1).sum())
        assert original[mode]['correct_top5']==int((ranks<=5).sum())
        score_error=float(np.max(np.abs(scores-scores32.astype(np.float64))))
        classes=np.unique(labels)
        class_scores64=np.stack([scores[:,labels==c].max(axis=1) for c in classes],axis=1)
        # If each video score differs by at most e, ordering can change only for
        # class-score pairs within 2e. Certify every observed rank difference.
        bound=2*score_error+1e-12
        differences=[]
        for i in np.flatnonzero(ranks!=ranks64):
            true_score=class_scores64[i,np.flatnonzero(classes==truth[i])[0]]
            low=1+int((class_scores64[i]>true_score+bound).sum())
            high=int((class_scores64[i]>=true_score-bound).sum())
            if not low<=ranks[i]<=high:
                raise ValueError('Rank discrepancy exceeds measured floating-point score error')
            differences.append({'query_id':queries[i]['video_id'],'recorded_float32_rank':int(ranks[i]),
                                'float64_rank':int(ranks64[i]),'certified_rank_interval':[low,high]})
        report[mode]={'status':'PASS','independent_numpy_rank_metrics':calculated,
                      'every_recorded_rank_top5_and_nearest_video_matches':True,
                      'float32_pooling_and_cosine_backend':'Torch CPU replay; ranking/metrics independently implemented in NumPy',
                      'float64_sensitivity':{'metrics':numpy_metrics(ranks64),
                           'max_absolute_video_cosine_difference':score_error,
                           'pairwise_score_difference_bound':bound,
                           'different_query_ranks':len(differences),'differences':differences},
                      'minimum_float64_winner_cosine_margin':float(np.min(np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2]))}
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-only',action='store_true')
    args=parser.parse_args();folder=args.output.resolve();config=read(folder/'run_config.json')
    report={'status':'PASS','method':'Torch float32 score replay; independent NumPy ranking/metrics; NumPy float64 numerical sensitivity audit',
            'step0':verify_phase(folder,'step0',config)}
    if not args.initial_only:
        import torch
        report[f"step{config['total_steps']}"]=verify_phase(folder,f"step{config['total_steps']}",config)
        history=rows(folder/'train/metrics.jsonl')
        assert [row['step'] for row in history]==list(range(1,config['total_steps']+1))
        assert all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in history)
        initial=torch.load(folder/'initial.model.pt',map_location='cpu',weights_only=True)
        final=torch.load(folder/'final.model.pt',map_location='cpu',weights_only=True)
        assert initial['initialization']=='kinetics_mae' and final['step']==config['total_steps']
        assert set(initial['model'])==set(final['model'])
        changed=[name for name in initial['model'] if not torch.equal(initial['model'][name],final['model'][name])]
        assert any(name.startswith('backbone.') for name in changed)
        assert any(name.startswith('fc.') for name in changed)
        assert any(name.startswith('cls_head.') for name in changed)
        checkpoint=torch.load(folder/'train/last.pt',map_location='cpu',weights_only=False)
        assert checkpoint['schema']=='signrep_training_v1' and checkpoint['step']==config['total_steps']
        assert checkpoint['run_contract']['amp_dtype']=='bfloat16'
        assert checkpoint['scheduler']['completed_steps']==config['total_steps']
        assert checkpoint['d_scheduler']['completed_steps']==config['total_steps']
        report['training']={'status':'PASS','completed_steps':config['total_steps'],'all_losses_and_gradients_finite':True,
                            'changed_model_tensors':len(changed),'strict_model_tensor_count':len(final['model']),
                            'complete_checkpoint_scheduler_steps':config['total_steps']}
    name='INITIAL_VERIFICATION.json' if args.initial_only else 'VERIFICATION.json'
    (folder/name).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
