"""Native, pinned SHuBERT encoder; media preprocessing has a separate gate."""
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .io import sha256


def verify_source(root, path, commit):
    source = root / path
    observed = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if observed != commit or changes:
        raise ValueError('Pinned upstream source mismatch or tracked modifications: ' + path)
    return source


class NativeSHuBERT:
    def __init__(self, root, config, device=None):
        self.config = config
        source = verify_source(root, config['upstream_path'], config['upstream_commit'])
        checkpoint = root / config['checkpoint_path']
        if sha256(checkpoint) != config['checkpoint_sha256']:
            raise ValueError('SHuBERT checkpoint checksum mismatch')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source / 'fairseq'))
        # Import fairseq first so examples/__init__.py cannot swallow an import
        # failure and leave a partially populated registration table.
        import fairseq
        from examples.shubert.models.shubert import SHubertConfig, SHubertModel
        self.device = torch.device(device or config['device'])
        torch.set_num_threads(4)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        saved = dict(payload['cfg']['model'])
        known = {field.name for field in dataclasses.fields(SHubertConfig)}
        ignored = sorted(set(saved) - known)
        if set(ignored) - {'discrete'}:
            raise ValueError('Checkpoint config fields absent from pinned architecture: ' + str(ignored))
        native_config = SHubertConfig(**{key: value for key, value in saved.items() if key in known})
        self.model = SHubertModel(native_config)
        result = self.model.load_state_dict(payload['model'], strict=True)
        del payload
        self.model.eval().requires_grad_(False).to(self.device)
        self.audit = {'strict': True, 'missing_keys': list(result.missing_keys),
                      'unexpected_keys': list(result.unexpected_keys), 'checkpoint_sha256': config['checkpoint_sha256'],
                      'upstream_commit': config['upstream_commit'], 'torch': torch.__version__,
                      'fairseq_version': fairseq.__version__, 'device': str(self.device),
                      'checkpoint_model_configuration': {k: v for k, v in saved.items() if k in known},
                      'ignored_checkpoint_config_fields': ignored,
                      'ignored_field_reason': 'discrete is not read by any pinned SHubertModel inference method',
                      'total_parameters': sum(p.numel() for p in self.model.parameters()),
                      'trainable_parameters': 0, 'output_layer': config['output_layer'],
                      'layer_semantics': 'Author features/shubert_inference.py takes layer_results[-1][-1], the last FFN branch before residual addition and final layer normalization. It differs from result[x].'}

    def forward_all_layers(self, streams):
        lengths = set()
        source = {}
        for name, dimension in self.config['stream_dimensions'].items():
            values = np.asarray(streams[name], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
                raise ValueError('Invalid native stream: ' + name)
            lengths.add(len(values))
            source[name] = torch.from_numpy(values).to(self.device)
        if len(lengths) != 1 or not lengths or min(lengths) < 1:
            raise ValueError('Stream time alignment mismatch or empty clip')
        length = next(iter(lengths))
        if length > self.config['max_frames']:
            raise ValueError('Full clip exceeds registered context cap; no silent context truncation')
        for name in self.config['stream_dimensions']:
            source['label_' + name] = torch.zeros((length, 1), device=self.device)
        with torch.inference_mode():
            result = self.model.extract_features([source], padding_mask=None, kmeans_labels=None, mask=False)
            if self.config['output_layer'] != 'last_layer_ffn_branch_upstream_script':
                raise ValueError('Unregistered SHuBERT layer semantics')
            layer_values = [layer[-1] for layer in result['layer_results']]
            if len(layer_values) != self.model.cfg.encoder_layers or any(values.shape != (length, 1, self.config['expected_dimension']) for values in layer_values):
                raise ValueError('Unexpected native layer count or feature axes')
            features = torch.stack([values.squeeze(1) for values in layer_values]).cpu().numpy()
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite native features')
        return features

    def forward(self, streams):
        return self.forward_all_layers(streams)[-1]


def load_native_dino(root, config, asset_id, device='cpu'):
    architecture = verify_source(root, config['dino_path'], config['dino_commit'])
    import json
    asset = json.loads((root / 'provenance/assets.json').read_text())['assets'][asset_id]
    checkpoint = root / asset['local_path']
    if sha256(checkpoint) != asset['expected_sha256']:
        raise ValueError('Author DINO checkpoint mismatch')
    model = torch.hub.load(str(architecture), 'dinov2_vits14_reg', source='local', pretrained=False)
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 384))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    teacher = {key.replace('backbone.', ''): value for key, value in payload['teacher'].items() if 'dino_head' not in key}
    result = model.load_state_dict(teacher, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, {'asset_id': asset_id, 'checkpoint_sha256': asset['expected_sha256'],
                   'architecture_commit': config['dino_commit'], 'strict': True,
                   'missing_keys': list(result.missing_keys), 'unexpected_keys': list(result.unexpected_keys),
                   'generic_pretrained_weights_downloaded': False, 'pos_embed_shape': [1, 257, 384]}
