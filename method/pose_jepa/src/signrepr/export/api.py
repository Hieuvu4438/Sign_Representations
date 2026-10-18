from pathlib import Path
import json
import numpy as np
import torch
from ..config import config_hash
from ..models.pose_encoder import PoseEncoder
from ..models.heads import MetricHead
from ..data.normalization import canonicalize_sample
from ..data.schema import PART_ORDER, PART_INDICES, sha256_file
from ..evaluation.extract import extract_full_sequence
from ..evaluation.retrieval import normalize_embeddings
from ..training.checkpoints import atomic_save


def export_checkpoint(checkpoint, output, selected_encoder=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('task') in ('cslr', 'slt') and state.get('encoder') is None:
        source = state['encoder_checkpoint']
        if sha256_file(source) != state['cache_provenance']['encoder_checkpoint_sha256']:
            raise ValueError('Frozen sequence backbone checkpoint checksum mismatch')
        return export_checkpoint(source, output, selected_encoder)
    config = state['config']
    stage = state.get('task') or state.get('metadata', {}).get('stage', config['run']['stage'])
    supervised_encoder = state.get('task') in ('islr', 'cslr')
    step = state.get('step', state.get('optimizer_step'))
    if step is None:
        counters = [float(value['step']) for value in state.get('optimizer', {}).get('state', {}).values() if 'step' in value]
        step = int(max(counters)) if counters else 0
    selection = selected_encoder or ('online' if stage == 'metric' else config['ema']['export_encoder'])
    if supervised_encoder:
        selection = selected_encoder or 'online'
        if selection != 'online': raise ValueError('Supervised encoder checkpoint has no separate EMA bundle')
    metric = state.get('metric_head')
    if metric is not None and step < 1:
        raise ValueError('Cannot export a metric head before a successful training update')
    if metric is not None and selection != 'online':
        raise ValueError('Stage 2 requires matched online encoder and online metric head')
    if selection not in ('online', 'ema'):
        raise ValueError('Unknown encoder selection')
    encoder_state = state['encoder'] if supervised_encoder else state['teacher'] if selection == 'ema' else {k.removeprefix('encoder.'): v for k, v in state['system'].items() if k.startswith('encoder.')}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    bundle = {'schema_version': '1.0', 'encoder': encoder_state, 'metric_head': metric, 'config': config}
    atomic_save(bundle, root/'model.pt')
    active = [key for key in ('latent', 'geometry', 'motion') if config['loss'][key+'_weight'] > 0]
    method = {'geometry': 'pose_masked_geometry', 'motion': 'pose_masked_motion', 'geometry_motion': 'pose_geometry_motion',
              'latent': 'pose_jepa_only', 'latent_geometry': 'pose_jepa_geometry', 'latent_motion': 'pose_jepa_motion',
              'latent_geometry_motion': 'pose_jepa_hybrid_v1'}.get('_'.join(active), 'pose_encoder')
    if config['loss'].get('latent_objective') == 'regression': method = 'pose_jepa_latent_regression'
    if stage == 'metric': method = 'pose_gloss_metric_adaptation'
    if supervised_encoder: method = f'pose_{stage}_supervised_encoder'
    metadata = {**state.get('metadata', {}), 'schema_version': '1.0', 'method': method, 'stage': stage,
                'config_sha256': state.get('metadata', {}).get('config_sha256', config_hash(config)),
                'manifest_sha256': state.get('metadata', {}).get('manifest_sha256', state.get('train_manifest_sha256')),
                'code_commit': state.get('metadata', {}).get('code_commit'),
                'architecture_family': 'pose_jepa_hybrid_v1', 'run_name': config['run']['name'],
                'ssl_objective_weights': {key: config['loss'][key+'_weight'] for key in ('latent', 'geometry', 'motion')},
                'selected_encoder': selection, 'metric_head_trained': metric is not None, 'sequence_dim': config['model']['sequence_dim'],
                'raw_clip_dim': config['model']['sequence_dim'], 'metric_dim': config['model']['latent_dim'] if metric is not None else None,
                'temporal_stride_frames': 2, 'part_order': list(PART_ORDER), 'joint_indices': PART_INDICES,
                'input_coordinate_profile': config['data']['coordinate_profile'], 'supervision': state.get('supervision', config['run']['supervision']),
                'data_protocol': config['run']['protocol'], 'pretrained_external_init': None,
                'optimizer_step': step, 'source_checkpoint_sha256': sha256_file(checkpoint),
                'bundle_sha256': sha256_file(root/'model.pt')}
    (root/'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


class SignRepresentationModel:
    def __init__(self, bundle, device='cpu'):
        path = Path(bundle)
        self.metadata = json.loads((path/'metadata.json').read_text())
        if sha256_file(path/'model.pt') != self.metadata['bundle_sha256']:
            raise ValueError('Inference bundle checksum mismatch')
        state = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
        self.config, self.device = state['config'], torch.device(device)
        self.encoder = PoseEncoder(self.config['model']).to(self.device).eval()
        self.encoder.load_state_dict(state['encoder'], strict=True)
        self.metric_head = None
        if state['metric_head'] is not None:
            self.metric_head = MetricHead(self.config['model']['sequence_dim'], self.config['model']['latent_dim']).to(self.device).eval()
            self.metric_head.load_state_dict(state['metric_head'], strict=True)

    @torch.inference_mode()
    def encode_sequence(self, pose, metadata=None):
        cfg = self.config['data']
        try:
            sample = canonicalize_sample(pose, metadata, cfg['confidence_threshold'], cfg['min_part_valid_fraction'], cfg['coordinate_profile'])
        except (ValueError, KeyError, TypeError) as exc:
            return {'status': 'malformed_input', 'error': str(exc), 'features': torch.empty((0, self.config['model']['sequence_dim'])), 'length': 0}
        ev = self.config['evaluation']
        return extract_full_sequence(self.encoder, sample, ev['window_frames'], ev['window_stride_frames'])

    @torch.inference_mode()
    def encode_clip(self, pose, metadata=None, embedding='raw'):
        if embedding not in ('raw', 'metric'):
            raise ValueError('Embedding must be raw or metric')
        if embedding == 'metric' and self.metric_head is None:
            raise ValueError('Metric head has not been trained')
        out = self.encode_sequence(pose, metadata)
        dim = self.config['model']['sequence_dim'] if embedding == 'raw' else self.config['model']['latent_dim']
        if out['status'] != 'ok':
            return {'embedding': torch.zeros(dim), 'status': out['status']}
        vector = out['raw_embedding']
        if embedding == 'metric':
            vector = self.metric_head(vector[None].to(self.device))[0].cpu()
        return {'embedding': vector, 'status': out['status']}

    def similarity(self, query_embeddings, gallery_embeddings):
        q, qv = normalize_embeddings(query_embeddings)
        g, gv = normalize_embeddings(gallery_embeddings)
        if q.shape[1] != g.shape[1]:
            raise ValueError('Embedding dimensions mismatch')
        return {'scores': np.where(qv[:, None] & gv[None, :], q @ g.T, np.nan), 'query_valid': qv, 'gallery_valid': gv}
