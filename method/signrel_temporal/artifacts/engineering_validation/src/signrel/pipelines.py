"""Audited CLI artifacts for cache, inference, retrieval and bounded profiling."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from .utils import read_jsonl,write_jsonl,atomic_json,sha256,fingerprint
from .data.cache import read_cache
from .train.loop import validate_training_inputs,choose_device,choose_precision
from .models.encoder import initialized_model
from .data.windows import collate,to_device
from .eval.embed import restore_encoder,embed_arrays
from .eval.retrieval import score_retrieval

EVAL_KEYS = {'schema_version','stage','encoder_checkpoint','cache_manifest','eval_labels','query_ids','gallery_ids',
             'protocol_path','output_dir','pooling','ranking','audit'}
EVAL_SUB = {'pooling':{'window_frames','stride_frames','temporal_bins','normalize_each_block'},
            'ranking':{'level','exemplar_aggregation','ks'},
            'audit':{'require_known_query_gallery_signer_disjoint','disallow_exact_duplicate_videos','disallow_overlapping_recording_segments'}}


def eval_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg)!=EVAL_KEYS or cfg['schema_version']!=1: raise ValueError('Unknown/missing eval config keys')
    for section,keys in EVAL_SUB.items():
        if set(cfg[section])!=keys: raise ValueError(f'Unknown/missing eval keys: {section}')
    if cfg['stage'] not in ('toy','dev','test'): raise ValueError('Invalid evaluation stage')
    if cfg['ranking']['level']!='gloss' or cfg['ranking']['exemplar_aggregation']!='max_cosine': raise ValueError('Invalid gloss ranking')
    if cfg['pooling']['temporal_bins']!=[1,2] or not cfg['pooling']['normalize_each_block']: raise ValueError('Invalid temporal pooling')
    if cfg['stage']=='test':
        lock_path=Path('artifacts/confirmatory_freeze.json')
        if not lock_path.exists(): raise ValueError('Locked test requires config/protocol freeze')
        lock=json.loads(lock_path.read_text())
        if lock.get('protocol_sha256')!=sha256(cfg['protocol_path']) or sha256(path) not in lock.get('eval_config_sha256',[]):
            raise ValueError('Test config/protocol differ from confirmatory freeze')
    return cfg


def audit_cache(cfg,output):
    rows=read_jsonl(cfg['paths']['cache_manifest'])
    results=[]
    lock=json.loads(Path(cfg['paths']['sources_lock']).read_text()) if not cfg['experiment']['synthetic'] else None
    for row in rows:
        expected={'synthetic':cfg['experiment']['synthetic']}
        if lock: expected['teacher_sha256']=lock['teacher']['checkpoint_sha256']
        arrays,meta=read_cache(row['cache_path'],expected)
        if sha256(row['cache_path'])!=row['cache_sha256']: raise ValueError('Manifest/cache bytes differ')
        results.append({'video_id':row['video_id'],'frames':len(arrays['appearance']),
                        'hand_coverage':arrays['app_valid'].mean(0).tolist(),'body_coverage':float(arrays['body_valid'].mean()),
                        'preprocess_fingerprint':meta['preprocess_fingerprint'],'teacher_sha256':meta['teacher_sha256']})
    if not rows: raise ValueError('Empty cache manifest')
    if len(set(r['preprocess_fingerprint'] for r in results))!=1: raise ValueError('Mixed preprocessing fingerprints')
    report={'status':'passed','videos':len(rows),'cache_manifest_sha256':sha256(cfg['paths']['cache_manifest']),'per_video':results,
            'mean_hand_coverage':np.mean([r['hand_coverage'] for r in results],axis=0).tolist(),
            'mean_body_coverage':float(np.mean([r['body_coverage'] for r in results]))}
    atomic_json(output,report)
    return report


def export_embeddings(config_path,device='cpu'):
    cfg=eval_config(config_path)
    torch.set_num_threads(4)
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    model=train_cfg=state=None
    checkpoint=cfg['encoder_checkpoint']
    if checkpoint!='A0': model,train_cfg,state=restore_encoder(checkpoint,device)
    from .data.provenance import inference_expectations
    expected=inference_expectations(state,train_cfg) if state else None
    protocol=json.loads(Path(cfg['protocol_path']).read_text())
    rows=read_jsonl(cfg['cache_manifest'])
    identifiers=set(json.loads(Path(cfg['query_ids']).read_text()))|set(json.loads(Path(cfg['gallery_ids']).read_text()))
    cache_by_id={r['video_id']:r for r in rows}
    if identifiers-set(cache_by_id): raise ValueError('Requested evaluation video has no audited cache; report extraction failure before changing protocol')
    output=Path(cfg['output_dir'])
    output.mkdir(parents=True,exist_ok=True)
    vectors,index=[],[]
    pre_fp=None
    for video_id in sorted(identifiers):
        row=cache_by_id[video_id]
        arrays,meta=read_cache(row['cache_path'],expected)
        if meta['video_id']!=video_id or meta['cache_sha256']!=row['cache_sha256']:
            raise ValueError('Evaluation cache manifest identity/hash mismatch')
        if pre_fp is None: pre_fp=meta['preprocess_fingerprint']
        if pre_fp!=meta['preprocess_fingerprint']: raise ValueError('Evaluation mixes preprocessing fingerprints')
        if state and not meta.get('synthetic'):
            source_lock=json.loads(Path(train_cfg['paths']['sources_lock']).read_text())
            if meta['teacher_sha256']!=source_lock['teacher']['checkpoint_sha256']: raise ValueError('Evaluation teacher differs from training')
        result=embed_arrays(arrays,model,cfg['pooling']['window_frames'],cfg['pooling']['stride_frames'],
                            train_cfg['data']['quality_threshold'] if train_cfg else .5,device)
        position=len(vectors) if result['status']=='ok' else None
        if position is not None: vectors.append(result['embedding'])
        index.append({'video_id':video_id,'row_index':position,'status':result['status'],'coverage':result['coverage'],
                      'preprocess_fingerprint':pre_fp,'cache_sha256':meta['cache_sha256']})
    if not vectors: raise ValueError('No eligible embeddings')
    np.save(output/'embeddings.npy',np.stack(vectors).astype(np.float32),allow_pickle=False)
    write_jsonl(output/'index.jsonl',index)
    if model:
        # Dedicated inference artifact omits decoder weights; restoration is separate.
        from .train.checkpoint import atomic_checkpoint
        atomic_checkpoint(output/'encoder.pt',{'encoder_state':{k:v for k,v in model.state_dict().items() if not k.startswith('heads.')},
                                              'model_config':train_cfg['model'],'config_resolved':train_cfg,
                                              'preprocess_fingerprint':expected['preprocess_fingerprint'],
                                              'teacher_sha256':expected['teacher_sha256'],
                                              'source_checkpoint_sha256':sha256(checkpoint)})
    report={'status':'complete','stage':cfg['stage'],'attempted':len(index),'retained':len(vectors),'dimension':len(vectors[0]),
            'encoder_checkpoint_sha256':sha256(checkpoint) if model else 'A0-frozen-raw-teacher',
            'embeddings_sha256':sha256(output/'embeddings.npy'),'index_sha256':sha256(output/'index.jsonl'),
            'eval_config_sha256':sha256(config_path),'protocol_sha256':sha256(cfg['protocol_path']),
            'cache_manifest_sha256':sha256(cfg['cache_manifest']),'preprocess_fingerprint':pre_fp,
            'synthetic':protocol.get('synthetic',False),'raw_rgb_inference_requires':'MediaPipe + frozen DINOv2 + context encoder',
            'pooling':cfg['pooling']}
    atomic_json(output/'export_report.json',report)
    return report


def evaluate_retrieval(config_path):
    cfg=eval_config(config_path)
    output=Path(cfg['output_dir'])
    features=np.load(output/'embeddings.npy',allow_pickle=False)
    index=read_jsonl(output/'index.jsonl')
    labels={r['video_id']:r for r in read_jsonl(cfg['eval_labels'])}
    export=json.loads((output/'export_report.json').read_text())
    if sha256(output/'embeddings.npy')!=export['embeddings_sha256'] or sha256(output/'index.jsonl')!=export['index_sha256']:
        raise ValueError('Export bytes changed')
    if sha256(config_path)!=export['eval_config_sha256'] or sha256(cfg['protocol_path'])!=export['protocol_sha256']:
        raise ValueError('Evaluation/export protocol mismatch')
    by_id={r['video_id']:r for r in index}
    query_ids=json.loads(Path(cfg['query_ids']).read_text())
    gallery_ids=json.loads(Path(cfg['gallery_ids']).read_text())
    def eligible(ids):
        return [i for i in ids if by_id[i]['status']=='ok']
    qids,gids=eligible(query_ids),eligible(gallery_ids)
    if not qids or not gids: raise ValueError('Empty eligible query/gallery')
    result=score_retrieval(np.stack([features[by_id[i]['row_index']] for i in qids]),[labels[i] for i in qids],
                           np.stack([features[by_id[i]['row_index']] for i in gids]),[labels[i] for i in gids],
                           cfg['audit']['require_known_query_gallery_signer_disjoint'])
    result.update(attempted_query_count=len(query_ids),valid_query_embeddings=len(qids),
                  invalid_query_ids=[i for i in query_ids if i not in qids],invalid_gallery_ids=[i for i in gallery_ids if i not in gids],
                  attempted_set_scored_coverage=result['queries_scored']/len(query_ids),
                  stage=cfg['stage'],synthetic=export['synthetic'],export_report_sha256=sha256(output/'export_report.json'),
                  eval_labels_sha256=sha256(cfg['eval_labels']),protocol_sha256=sha256(cfg['protocol_path']))
    atomic_json(output/'metrics.json',result)
    return result


def profile(cfg,output,device_override=None):
    dataset,_=validate_training_inputs(cfg)
    device=choose_device(cfg,device_override)
    torch.set_num_threads(4)
    model=initialized_model(cfg['model'],17).to(device)
    batch=to_device(collate([dataset.sample(i) for i in range(cfg['train']['microbatch_size'])]),device)
    precision=choose_precision(cfg,device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    from .losses.state import hand_state_loss,body_state_loss
    from .losses.pairs import build_pairs
    from .losses.relation import relation_loss
    from .train.loop import optimizer_for
    from .budget import charge
    optimizer=optimizer_for(model,cfg['train'])
    counter=0
    def update():
        nonlocal counter
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg['train']['gradient_accumulation']):
            samples=[dataset.sample(counter*cfg['train']['microbatch_size']+j) for j in range(cfg['train']['microbatch_size'])]
            counter+=1
            batch=to_device(collate(samples),device)
            with torch.autocast(device.type,dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision!='fp32'):
                pred=model(batch['student'])
            c,s=batch['clean'],batch['student'];r=cfg['loss']['relation']
            loss=hand_state_loss(pred['app_prediction'],c['appearance'],c['app_valid'],c['app_quality'],s['app_ssl_mask'])[0]
            loss+=cfg['loss']['body_weight']*body_state_loss(pred['body_prediction'],c['body'],c['body_valid'],c['body_coord_valid'],c['body_coord_quality'],s['body_ssl_mask'])[0]
            pairs=build_pairs(c['app_valid'],c['app_quality'],s['app_ssl_mask'],batch['mapping']['source_frame_index'],s['time_padding'],r['lags'],r['mode'],r['max_pairs_per_sample'],torch.Generator().manual_seed(counter),cfg['data']['quality_threshold'])
            rel=relation_loss(pred['app_prediction'],c['appearance'],c['app_valid'],c['app_quality'],s['app_ssl_mask'],pairs,r['smooth_l1_beta'],r['mode'],torch.Generator().manual_seed(counter))[0]
            loss+=r['max_weight']*rel if r['enabled'] else 0*rel
            (loss/cfg['train']['gradient_accumulation']).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip_norm'])
        optimizer.step()
    total_start=time.monotonic()
    update()
    if device.type=='cuda': torch.cuda.synchronize()
    start=time.monotonic()
    for _ in range(3): update()
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    result={'status':'measured','precision':precision,'device':str(device),'seconds_per_update':elapsed/3,
            'effective_batch':cfg['train']['microbatch_size']*cfg['train']['gradient_accumulation'],
            'peak_vram_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            'scope':'Sample cached recordings, mask, state and relation losses, accumulated backward, gradient clip and AdamW; excludes checkpoint/report I/O',
            'estimated_1000_update_seconds':elapsed/3*1000}
    charge('profile:'+str(Path(output).resolve()),'profile',gpu_seconds=time.monotonic()-total_start if device.type=='cuda' else 0)
    atomic_json(output,result)
    return result
