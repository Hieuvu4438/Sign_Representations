"""Runnable commands for proposal14. Missing resources fail explicitly."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import torch
import torch.distributed as dist
import yaml
from .config import load_config, config_hash, load_protocol_config
from .data.fixture import make_fixture
from .data.schema import sha256_file, read_manifest, load_pose
from .data.splits import validate_manifest, build_protocol
from .data.import_unisign import import_directory
from .data.pose_dataset import PoseDataset
from .data.normalization import canonicalize_sample
from .evaluation.extract import extract_full_sequence
from .evaluation.cache import save_sequence_features
from .evaluation.retrieval import rank_dictionary
from .export.api import export_checkpoint, SignRepresentationModel
from .training.pretrain import SSLTrainer
from .training.stream import SampleStream
from .training.checkpoints import save_training, load_training
from .training.distributed import initialize_distributed, rank, world_size
from .training.validation import frozen_retrieval


from .training.provenance import source_hash, provenance


def profile_config(config, profile):
    cfg = deepcopy(config)
    if profile == 'smoke':
        cfg['data'].update(max_frames=16, num_workers=0)
        cfg['model'].update(encoder_layers=2, predictor_layers=1)
        cfg['optim'].update(per_rank_batch=2, gradient_accumulation_steps=1, max_steps=20, warmup_steps=2, precision='fp32')
    elif profile == 'pilot':
        cfg['optim'].update(max_steps=5000, warmup_steps=500)
    elif profile != 'full':
        raise ValueError('Unknown training profile')
    return cfg


def pretrain(args):
    device = initialize_distributed(args.device)
    torch.set_num_threads(args.cpu_threads)
    config = profile_config(load_config(args.config), args.profile)
    data = PoseDataset(config['data']['train_manifest'], config['data'], sampling=config['data'].get('sampling', 'continuous'))
    metadata = provenance(config, config['data']['train_manifest'])
    trainer = SSLTrainer(config, device)
    stream = SampleStream(data, config['optim']['per_rank_batch'], config['run']['seed']+1, rank(), world_size(),
                          config['data']['num_workers'], config['data']['pin_memory'] and device.type == 'cuda',
                          config['data']['persistent_workers'])
    if args.resume:
        cursor, previous = load_training(trainer, args.resume)
        for key in ('manifest_sha256', 'pose_inventory_sha256', 'code_sha256', 'sources_lock_sha256'):
            if previous.get(key) != metadata[key]:
                raise ValueError(f'Resume provenance mismatch: {key}')
        stream.load_state_dict(cursor)
        metadata = previous
    root = Path(config['run']['output_dir'])
    if rank() == 0:
        root.mkdir(parents=True, exist_ok=True)
        (root/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        (root/'provenance.json').write_text(json.dumps(metadata, indent=2))
    attempts = 0
    best = metadata.get('best_validation_metric')
    protocol_path = config['data'].get('downstream_validation_protocol')
    protocol = None
    if protocol_path:
        protocol = json.loads(Path(protocol_path).read_text())
        for key in ('gallery_manifest', 'query_manifest'):
            protocol[key] = str((Path(protocol_path).parent/protocol[key]).resolve())
    limit = max(100, (config['optim']['max_steps']-trainer.step)*10)
    while trainer.step < config['optim']['max_steps']:
        batches = [stream.next() for _ in range(config['optim']['gradient_accumulation_steps'])]
        report = trainer.update(batches)
        attempts += 1
        if rank() == 0:
            with open(root/'train.jsonl', 'a') as f:
                f.write(json.dumps(report)+'\n')
        if report['successful'] and trainer.step % config['evaluation']['checkpoint_interval_steps'] == 0:
            save_training(trainer, root/'last.pt', stream.state_dict(), metadata)
        if protocol and report['successful'] and (trainer.step % config['evaluation']['frozen_validation_interval_steps'] == 0 or trainer.step == config['optim']['max_steps']):
            score, validation = validate_distributed(trainer.teacher if config['ema']['export_encoder'] == 'ema' else trainer.system.encoder,
                                                      protocol['gallery_manifest'], protocol['query_manifest'], config)
            if rank() == 0: write_json(root/f'validation_{trainer.step:08d}.json', validation)
            if best is None or score > best:
                best = score
                metadata['best_validation_metric'] = best
                save_training(trainer, root/'best_validation.pt', stream.state_dict(), metadata)
        if attempts >= limit:
            save_training(trainer, root/'last.pt', stream.state_dict(), metadata)
            raise RuntimeError('Insufficient supported successful updates; inspect train log and timing/coverage')
    save_training(trainer, root/'last.pt', stream.state_dict(), metadata)
    stream.close()
    return {'checkpoint': str(root/'last.pt'), 'successful_updates': trainer.step, 'profile': args.profile,
            'synthetic': all(r.get('synthetic', False) for r in data.rows),
            'effective_update_batch': config['optim']['per_rank_batch']*config['optim']['gradient_accumulation_steps']*world_size()}


def validate_distributed(encoder, gallery, query, config, metric_head=None):
    result = frozen_retrieval(encoder, gallery, query, config, metric_head) if rank() == 0 else None
    payload = [result]
    if dist.is_initialized(): dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    return result['metrics']['R@1'], result


def train_metric(args):
    from .training.metric import MetricTrainer
    from .data.samplers import PKSampler, gloss_key
    from .data.collate import collate_pose
    device = initialize_distributed(args.device)
    torch.set_num_threads(args.cpu_threads)
    cfg = load_config(args.config)
    dataset = PoseDataset(cfg['data']['train_manifest'], cfg['data'], stage='metric', sampling='isolated')
    vocabulary = sorted({gloss_key(row) for row in dataset.rows})
    trainer = MetricTrainer(cfg, args.init, vocabulary, device)
    sampler = PKSampler(dataset.rows, cfg['metric']['classes_per_batch'], cfg['metric']['clips_per_class'], cfg['run']['seed'], rank(), world_size())
    metadata = provenance(cfg, cfg['data']['train_manifest'])
    metadata.update(stage='metric', supervision='gloss_supervised_metric_adaptation', initial_checkpoint_sha256=sha256_file(args.init),
                    contrastive_batch=sampler.p*sampler.k, positive_policy='distinct_clips_prefer_known_distinct_signers',
                    excluded_singleton_classes=sampler.excluded_classes)
    if args.resume:
        cursor, previous = load_training(trainer, args.resume)
        for key in ('manifest_sha256', 'pose_inventory_sha256', 'code_sha256', 'sources_lock_sha256', 'initial_checkpoint_sha256'):
            if previous.get(key) != metadata[key]: raise ValueError(f'Metric resume mismatch: {key}')
        sampler.load_state_dict(cursor)
        metadata = previous
    root = Path(cfg['run']['output_dir'])
    if rank() == 0: root.mkdir(parents=True, exist_ok=True)
    total = cfg['metric']['head_warmup_steps']+cfg['metric']['joint_steps']
    attempts, best = 0, metadata.get('best_validation_metric')
    while trainer.step < total:
        batches = [collate_pose([dataset[i] for i in sampler.next()]) for _ in range(cfg['optim']['gradient_accumulation_steps'])]
        report = trainer.update(batches); attempts += 1
        if rank() == 0:
            with open(root/'train.jsonl', 'a') as f: f.write(json.dumps(report)+'\n')
        if report['successful'] and (trainer.step % cfg['metric']['validation_interval_steps'] == 0 or trainer.step == total):
            if cfg['data'].get('validation_manifest'):
                score, result = validate_distributed(trainer.system.encoder, cfg['data']['train_manifest'], cfg['data']['validation_manifest'], cfg, trainer.system.metric_head)
                if rank() == 0: write_json(root/f'validation_{trainer.step:08d}.json', result)
                if best is None or score > best:
                    best = score; metadata['best_validation_metric'] = best
                    save_training(trainer, root/'best_validation.pt', sampler.state_dict(), metadata)
            save_training(trainer, root/'last.pt', sampler.state_dict(), metadata)
        if attempts > max(100, total*10): raise RuntimeError('Insufficient valid metric update support')
    save_training(trainer, root/'last.pt', sampler.state_dict(), metadata)
    return {'checkpoint': str(root/'last.pt'), 'steps': trainer.step, 'best_validation': best}


def extract(args):
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        model = SignRepresentationModel(checkpoint, args.device or 'cpu')
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            export_checkpoint(checkpoint, tmp)
            model = SignRepresentationModel(tmp, args.device or 'cpu')
    rows = read_manifest(args.manifest)
    if args.embedding == 'metric' and model.metric_head is None:
        raise ValueError('Metric head has not been trained')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    embeddings, records = [], []
    for i, row in enumerate(rows):
        pose = load_pose(row['pose_path'])
        sequence = model.encode_sequence(pose, row)
        dim = model.config['model']['sequence_dim'] if args.embedding == 'raw' else model.config['model']['latent_dim']
        if sequence['status'] == 'ok':
            vector = sequence['raw_embedding']
            if args.embedding == 'metric':
                with torch.inference_mode(): vector = model.metric_head(vector[None].to(model.device))[0].cpu()
        else: vector = torch.zeros(dim)
        result = {'embedding': vector, 'status': sequence['status']}
        embeddings.append(result['embedding'].numpy())
        if sequence['length']:
            save_sequence_features(root/f'sequence_{i:08d}.npz', sequence)
        records.append({'sample_id': row['sample_id'], 'gloss_key': [row['language'], row['gloss_namespace'], str(row['gloss_id'])] if row.get('gloss_id') is not None and row.get('gloss_namespace') else None,
                        'status': result['status'], 'split': row['split'], 'source_video_id': row['source_video_id'],
                        'pose_sha256': row['pose_sha256'], 'sequence_path': f'sequence_{i:08d}.npz' if sequence['length'] else None})
    if not embeddings:
        raise ValueError('Empty extraction manifest')
    np.save(root/'embeddings.npy', np.stack(embeddings))
    (root/'index.json').write_text(json.dumps(records, indent=2))
    (root/'metadata.json').write_text(json.dumps({**model.metadata, 'embedding': args.embedding,
                                                'manifest_sha256': sha256_file(args.manifest),
                                                'window': model.config['evaluation']['window_frames'],
                                                'stride': model.config['evaluation']['window_stride_frames'],
                                                'stitch_version': 'aligned_unique_segment_mean_v1'}, indent=2))
    return {'samples': len(records), 'valid': sum(r['status'] == 'ok' for r in records)}


def retrieval(args):
    gpath, qpath = Path(args.gallery), Path(args.queries)
    gm, qm = (json.loads((p/'metadata.json').read_text()) for p in (gpath, qpath))
    for key in ('bundle_sha256', 'embedding', 'window', 'stride', 'stitch_version'):
        if gm[key] != qm[key]:
            raise ValueError(f'Feature provenance mismatch: {key}')
    gallery, query = (json.loads((p/'index.json').read_text()) for p in (gpath, qpath))
    if any(r['gloss_key'] is None for r in gallery+query):
        raise ValueError('Dictionary retrieval requires verified namespaced gloss labels')
    # Distinct source crops are screened independently of model embeddings.
    if {r['source_video_id'] for r in gallery} & {r['source_video_id'] for r in query}:
        raise ValueError('Source video contamination between query and gallery')
    result = rank_dictionary(np.load(gpath/'embeddings.npy'), np.load(qpath/'embeddings.npy'),
                             [r['sample_id'] for r in gallery], [r['sample_id'] for r in query],
                             [tuple(r['gloss_key']) for r in gallery], [tuple(r['gloss_key']) for r in query], args.protocol)
    write_json(args.output, result)
    return result['metrics']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('make_fixture'); f.add_argument('--output', required=True); f.add_argument('--samples', type=int, default=16); f.add_argument('--seed', type=int, default=42)
    v = sub.add_parser('validate_data'); v.add_argument('--manifest', required=True); v.add_argument('--report', required=True)
    u = sub.add_parser('import_unisign'); u.add_argument('--input', required=True); u.add_argument('--metadata', required=True); u.add_argument('--output', required=True); u.add_argument('--manifest', required=True)
    b = sub.add_parser('build_protocol'); b.add_argument('--config', required=True); b.add_argument('--output', required=True)
    t = sub.add_parser('pretrain'); t.add_argument('--config', required=True); t.add_argument('--profile', choices=['smoke', 'pilot', 'full'], default='full'); t.add_argument('--resume'); t.add_argument('--device'); t.add_argument('--cpu-threads', type=int, default=4)
    e = sub.add_parser('export'); e.add_argument('--checkpoint', required=True); e.add_argument('--output', required=True); e.add_argument('--encoder', choices=['online', 'ema'])
    m = sub.add_parser('train_metric'); m.add_argument('--config', required=True); m.add_argument('--init', required=True); m.add_argument('--resume'); m.add_argument('--device'); m.add_argument('--cpu-threads', type=int, default=4)
    i = sub.add_parser('train_islr'); i.add_argument('--config', required=True); i.add_argument('--init'); i.add_argument('--device', default='cpu')
    i = sub.add_parser('eval_islr'); i.add_argument('--checkpoint', required=True); i.add_argument('--manifest', required=True); i.add_argument('--output', required=True); i.add_argument('--device', default='cpu')
    for task in ('cslr', 'slt'):
        s = sub.add_parser(f'train_{task}'); s.add_argument('--config', required=True); s.add_argument('--encoder', required=True); s.add_argument('--device', default='cpu')
    x = sub.add_parser('extract_pose'); x.add_argument('--video', required=True); x.add_argument('--metadata', required=True); x.add_argument('--output', required=True); x.add_argument('--device', default='cpu')
    x = sub.add_parser('extract'); x.add_argument('--checkpoint', required=True); x.add_argument('--manifest', required=True); x.add_argument('--output', required=True); x.add_argument('--embedding', choices=['raw', 'metric'], default='raw'); x.add_argument('--device')
    r = sub.add_parser('retrieval'); r.add_argument('--gallery', required=True); r.add_argument('--queries', required=True); r.add_argument('--protocol', choices=['gloss_max_single_relevant', 'video_rank_multiple_relevant'], default='gloss_max_single_relevant'); r.add_argument('--output', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'make_fixture':
        result = {'manifest': str(make_fixture(args.output, args.samples, args.seed))}
    elif args.command == 'validate_data':
        result = validate_manifest(args.manifest)
        write_json(args.report, result)
        if not result['valid']:
            raise SystemExit('Manifest audit failed; see report')
    elif args.command == 'import_unisign':
        result = import_directory(args.input, args.metadata, args.output, args.manifest)
    elif args.command == 'build_protocol':
        policy = load_protocol_config(args.config)
        result = build_protocol(policy, args.output)
    elif args.command == 'pretrain':
        result = pretrain(args)
    elif args.command == 'export':
        result = export_checkpoint(args.checkpoint, args.output, args.encoder)
    elif args.command == 'train_metric':
        result = train_metric(args)
    elif args.command == 'train_islr':
        from .training.islr import train_islr
        result = train_islr(load_config(args.config), args.init, args.device)
    elif args.command == 'eval_islr':
        from .training.islr import eval_islr
        result = eval_islr(args.checkpoint, args.manifest, args.device)
        write_json(args.output, result)
    elif args.command in ('train_cslr', 'train_slt'):
        from .training.sequence_tasks import train_sequence
        result = train_sequence(load_config(args.config), args.encoder, args.command.removeprefix('train_'), args.device)
    elif args.command == 'extract_pose':
        from .data.extract_pose import extract_video
        result = extract_video(args.video, args.output, json.loads(Path(args.metadata).read_text()), device=args.device)
    elif args.command == 'extract':
        result = extract(args)
    elif args.command == 'retrieval':
        result = retrieval(args)
    else:
        raise RuntimeError('Unknown command')
    if rank() == 0:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
