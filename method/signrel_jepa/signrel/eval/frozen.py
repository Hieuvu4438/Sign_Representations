"""Periodic validation runs on frozen RGB features, outside SSL batches."""
import contextlib
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from signrel.data.manifest import read_manifest,sha256
from signrel.data.video_dataset import decode_video
from signrel.data.transforms import transform_clip
from signrel.eval.extract import rgb_embedding
from signrel.eval.retrieval import retrieval
from signrel.eval.linear_probe import linear_probe
from signrel.eval.phonology import load_metadata,phonological_probes
from signrel.eval.signer_probe import signer_probe
from signrel.eval.minimal_pairs import minimal_pairs
from signrel.eval.diagnostics import embedding_health,minimal_pair_retrieval


@contextlib.contextmanager
def preserved_rng():
    from signrel.engine.checkpoint import rng_state,restore_rng
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def validate_evaluation(evaluation):
    for key in ('gallery_manifest','query_manifest','config'):
        if not evaluation.get(key):
            raise ValueError(f'Periodic frozen evaluation requires {key}')
    rows = read_manifest(evaluation['query_manifest'])
    if not rows or any(r.get('split') not in {'val','validation','dev'} for r in rows):
        raise ValueError('Periodic checkpoint selection requires a declared validation split, never test/train')
    if evaluation.get('selection_mode','max') not in {'max','min'}:
        raise ValueError('selection_mode must be max or min')
    return [evaluation['gallery_manifest'],evaluation['query_manifest']]


def tensor_hash(encoder):
    digest = hashlib.sha256()
    for name,value in sorted(encoder.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def extract_manifest(encoder,manifest,dc,device,precision='fp32'):
    rows = read_manifest(manifest)
    if not rows:
        raise ValueError('Empty frozen evaluation manifest')
    vectors,retained = [],[]
    for row in rows:
        video,_,_,_ = decode_video(row['video_path'],dc.get('num_frames',16),dc.get('sampling_stride',2),
                                   row.get('start_sec'),row.get('end_sec'))
        x,_,_ = transform_clip(video,size=dc.get('crop_size',224),crop=row.get('signer_crop') or (0,0,1,1))
        with torch.autocast(torch.device(device).type,dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
                            enabled=precision != 'fp32'):
            feature = rgb_embedding(encoder,x[None].to(device)).float().cpu().numpy()[0]
        if not np.isfinite(feature).all():
            raise FloatingPointError('Non-finite periodic frozen embedding')
        vectors.append(feature)
        retained.append({**row,'file_sha256':sha256(row['video_path'])})
    return np.stack(vectors),retained


def evaluate_features(q,qr,g,gr,cfg):
    report = {'config':cfg,'retrieval':retrieval(q,g,qr,gr,exclude_same_signer=cfg.get('exclude_same_signer',False),class_max=cfg.get('class_max',True))}
    if cfg.get('embedding_diagnostics',True):
        report['embedding_health'] = {name:embedding_health(z) if len(z) >= 2 else {'status':'insufficient_samples'} for name,z in [('query',q),('gallery',g)]}
    if cfg.get('linear_probe',True):
        report['linear_probe'] = [linear_probe(g,[r['gloss'] for r in gr],q,[r['gloss'] for r in qr],seed)
                                  for seed in cfg.get('seeds',[42,43,44])]
    if cfg.get('signer_probe',True):
        report['signer_probe'] = signer_probe(np.concatenate((g,q)),gr+qr,seed=cfg.get('seeds',[42])[0])
    if cfg.get('phonology_metadata'):
        metadata = load_metadata(cfg['phonology_metadata'],cfg.get('phonology_encoding','utf-8-sig'))
        report['phonology_metadata_sha256'] = sha256(cfg['phonology_metadata'])
        report['phonology_signer_independent'] = phonological_probes(g,gr,q,qr,metadata,cfg.get('properties'))
        train_gloss = lambda gloss: int(hashlib.sha256(str(gloss).encode()).hexdigest()[:8],16)%3 != 0
        ti = [i for i,r in enumerate(gr) if train_gloss(r['gloss'])]
        ei = [i for i,r in enumerate(qr) if not train_gloss(r['gloss'])]
        report['phonology_gloss_disjoint'] = phonological_probes(g[ti],[gr[i] for i in ti],q[ei],[qr[i] for i in ei],metadata,cfg.get('properties'),require_gloss_disjoint=True)
        report['minimal_pairs'] = minimal_pairs(np.concatenate((g,q)),gr+qr,metadata)
        report['minimal_pair_retrieval'] = minimal_pair_retrieval(q,g,qr,gr,metadata,
            cfg.get('minimal_pair_properties',cfg.get('properties')),exclude_same_signer=cfg.get('exclude_same_signer',False))
    return report


def selection_value(report,path='retrieval.Recall@1'):
    value = report
    for key in path.split('.'):
        value = value[key]
    if not isinstance(value,(int,float)) or not np.isfinite(value):
        raise ValueError('Checkpoint selection metric must be a finite scalar')
    return float(value)


def frozen_validation(encoder,evaluation,dc,device,precision,run_dir,step):
    validate_evaluation(evaluation)
    cfg = yaml.safe_load(Path(evaluation['config']).read_text())
    was_training = encoder.training
    out = Path(run_dir)/'evaluation'/f'step_{step:08d}'
    out.mkdir(parents=True,exist_ok=True)
    try:
        with preserved_rng():
            encoder.eval()
            g,gr = extract_manifest(encoder,evaluation['gallery_manifest'],dc,device,precision)
            q,qr = extract_manifest(encoder,evaluation['query_manifest'],dc,device,precision)
            for name,features,rows in [('gallery',g,gr),('query',q,qr)]:
                np.savez_compressed(out/f'{name}.npz',features=features,rows_json=np.array([json.dumps(r) for r in rows]))
            report = evaluate_features(q,qr,g,gr,cfg)
            report.update(global_step=step,encoder_tensor_sha256=tensor_hash(encoder),
                          gallery_sha256=sha256(out/'gallery.npz'),feature_sha256=sha256(out/'query.npz'),
                          evaluation_config_sha256=sha256(evaluation['config']),
                          manifest_sha256={k:sha256(evaluation[k]) for k in ('gallery_manifest','query_manifest')},
                          upstream_init=encoder.upstream_init,rgb_only=True,
                          pooling='mean tokens, LayerNorm, L2; one fixed center clip',precision=precision)
            value = selection_value(report,evaluation.get('selection_metric','retrieval.Recall@1'))
            (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            return {'global_step':step,'selection_metric':evaluation.get('selection_metric','retrieval.Recall@1'),
                    'selection_value':value,'report':str(out/'report.json'),'report_sha256':sha256(out/'report.json'),
                    'encoder_tensor_sha256':report['encoder_tensor_sha256']}
    finally:
        encoder.train(was_training)
