"""Globally weighted JEPA updates: one denominator per effective optimizer batch."""
from contextlib import nullcontext
from copy import deepcopy
import math
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from ..models.part_tokenizer import token_metadata
from ..models.ema import update_encoder_ema, momentum
from ..masking.part_temporal import MaskGenerator
from ..losses.hybrid import HybridCriterion
from ..data.collate import to_device
from .system import SSLSystem
from .distributed import world_size, rank, global_sum, all_finite, loss_for_backward


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_groups(system, weight_decay):
    no_decay = set()
    for module in system.modules():
        for name, p in module.named_parameters(recurse=False):
            if name.endswith('bias') or p.ndim < 2 or isinstance(module, (nn.LayerNorm, nn.Embedding)):
                no_decay.add(id(p))
    return [{'params': [p for p in system.parameters() if id(p) not in no_decay], 'weight_decay': weight_decay},
            {'params': [p for p in system.parameters() if id(p) in no_decay], 'weight_decay': 0.0}]


class SSLTrainer:
    def __init__(self, config, device='cpu'):
        self.config, self.device = deepcopy(config), torch.device(device)
        seed_everything(config['run']['seed'])
        self.system = SSLSystem(config['model']).to(self.device)
        self.wrapped = self.system
        if world_size() > 1:
            self.wrapped = DistributedDataParallel(self.system, device_ids=[self.device.index] if self.device.type == 'cuda' else None)
        # DDP initialization broadcasts the student before constructing the teacher.
        self.teacher = deepcopy(self.system.encoder).float().eval().requires_grad_(False)
        self.criterion = HybridCriterion({**config['loss'], 'max_motion_gap_sec': config['data']['max_motion_gap_sec']}, config['model']['latent_dim']).to(self.device)
        self.mask_generator = MaskGenerator(config['mask'])
        self.mask_rng = torch.Generator().manual_seed(config['run']['seed'] + 100003*rank())
        opt = config['optim']
        self.optimizer = torch.optim.AdamW(parameter_groups(self.system, opt['weight_decay']), lr=opt['learning_rate'], betas=tuple(opt['betas']))
        precision = opt['precision']
        if precision not in ('fp32', 'fp16', 'bf16', 'bf16_if_supported_else_fp32'):
            raise ValueError('Unsupported precision')
        if precision in ('fp16', 'bf16') and self.device.type != 'cuda':
            raise ValueError('Explicit fp16/bf16 training requires CUDA; CPU uses fp32')
        if precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('The selected CUDA device does not support bf16')
        bf16 = precision == 'bf16' or (precision == 'bf16_if_supported_else_fp32' and self.device.type == 'cuda' and torch.cuda.is_bf16_supported())
        self.autocast_dtype = torch.bfloat16 if bf16 else torch.float16
        self.autocast_enabled = precision == 'fp16' or bf16
        self.scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
        self.step = 0

    def autocast(self):
        return torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled)

    def learning_rate(self):
        cfg = self.config['optim']
        warm = min(cfg['warmup_steps'], cfg['max_steps'])
        if self.step < warm:
            return cfg['learning_rate']*(self.step+1)/max(1, warm)
        progress = (self.step-warm)/max(1, cfg['max_steps']-warm)
        return cfg['min_learning_rate'] + (cfg['learning_rate']-cfg['min_learning_rate'])*(1+math.cos(math.pi*progress))/2

    def update(self, batches, masks=None):
        started = time.perf_counter()
        if not batches:
            raise ValueError('An effective update requires at least one microbatch')
        batches = [to_device(b, self.device) for b in batches]
        metadata = [token_metadata(b) for b in batches]
        masks = masks if masks is not None else [self.mask_generator(m['valid'], self.mask_rng) for m in metadata]
        if len(masks) != len(batches):
            raise ValueError('Microbatch/mask count mismatch')
        masks = [to_device(m, self.device) for m in masks]
        prepared = [self.criterion.targets(b, mask, meta) for b, mask, meta in zip(batches, masks, metadata)]
        weights = {k: self.config['loss'][f'{k}_weight'] for k in ('latent', 'geometry', 'motion')}
        denominators = {k: global_sum(sum(den[k] for _, den in prepared)) for k in weights}
        if not any(weights[k] > 0 and denominators[k].item() > 0 for k in weights):
            return {'successful': False, 'reason': 'zero_support', 'step': self.step}
        lr = self.learning_rate()
        old_rates = [group['lr'] for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.zero_grad(set_to_none=True)
        self.system.train()
        self.teacher.eval()
        center_sum = torch.zeros_like(self.criterion.center)
        raw_sums = {k: self.criterion.center.new_zeros(()) for k in weights}
        entropy_sums = self.criterion.center.new_zeros(4)
        entropy_counts = self.criterion.center.new_zeros(4)
        for index, (batch, mask, meta, (targets, _)) in enumerate(zip(batches, masks, metadata, prepared)):
            sync = self.wrapped.no_sync() if world_size() > 1 and index < len(batches)-1 else nullcontext()
            with sync:
                with self.autocast():
                    # Reconstruction-only controls avoid the teacher forward.
                    with torch.no_grad():
                        target = self.teacher.encode_full(batch)['features'] if weights['latent'] else self.criterion.center.new_zeros((*meta['valid'].shape, self.config['model']['latent_dim']))
                        if weights['latent'] and self.config['loss'].get('shuffle_teacher_targets', False):
                            # Diagnostic control: random same-part temporal/clip
                            # correspondence, with natural validity/counts intact.
                            target = target.clone()
                            for part in range(4):
                                support = mask['target'] & (meta['indices'] % 4 == part)
                                logits = target[support]
                                order = torch.randperm(len(logits), generator=self.mask_rng).to(self.device)
                                target[support] = logits[order]
                    prediction = self.wrapped(batch, mask)
                terms, teacher_sum = self.criterion(prediction, target, targets, mask, meta)
                loss = sum(weights[k]*loss_for_backward(num, denominators[k], world_size()) for k, (num, _) in terms.items())
                self.scaler.scale(loss).backward()
                center_sum += teacher_sum
                if weights['latent']:
                    log_prob = torch.nn.functional.log_softmax((target.detach().float()-self.criterion.center)/self.config['loss']['teacher_temperature'], -1)
                    entropy = -(log_prob.exp()*log_prob).sum(-1)
                    quality = self.criterion.quality(meta)*mask['target']
                    entropy_sums += (entropy*quality).reshape(len(batch['lengths']), -1, 4).sum((0, 1))
                    entropy_counts += quality.reshape(len(batch['lengths']), -1, 4).sum((0, 1))
                for k, (num, _) in terms.items():
                    raw_sums[k] += num.detach()
        self.scaler.unscale_(self.optimizer)
        finite = all_finite(self.system.parameters(), self.device)
        if not finite:
            self.optimizer.zero_grad(set_to_none=True)
            for group, old_rate in zip(self.optimizer.param_groups, old_rates): group['lr'] = old_rate
            if self.scaler.is_enabled():
                self.scaler.update(new_scale=self.scaler.get_scale()/2)
            return {'successful': False, 'reason': 'nonfinite_gradients', 'step': self.step}
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.config['optim']['gradient_clip_norm'], error_if_nonfinite=True)
        except RuntimeError:
            self.optimizer.zero_grad(set_to_none=True)
            for group, old_rate in zip(self.optimizer.param_groups, old_rates): group['lr'] = old_rate
            if self.scaler.is_enabled(): self.scaler.update(new_scale=self.scaler.get_scale()/2)
            return {'successful': False, 'reason': 'nonfinite_gradient_norm', 'step': self.step}
        self.scaler.step(self.optimizer)
        self.scaler.update()
        ema_momentum = momentum(self.step, self.config['optim']['max_steps'], self.config['ema']['initial_momentum']) if weights['latent'] else 0.
        if weights['latent']:
            update_encoder_ema(self.system.encoder, self.teacher, ema_momentum)
            self.criterion.update_center(global_sum(center_sum), denominators['latent'])
        else:
            # Export online encoder for reconstruction-only controls.
            update_encoder_ema(self.system.encoder, self.teacher, 0.)
        self.step += 1
        losses = {k: float(global_sum(raw_sums[k])/denominators[k]) if denominators[k] > 0 else 0. for k in weights}
        ent_num, ent_den = global_sum(entropy_sums), global_sum(entropy_counts)
        return {'successful': True, 'step': self.step, 'raw_losses': losses,
                'weighted_losses': {k: losses[k]*weights[k] for k in weights},
                'denominators': {k: float(v) for k, v in denominators.items()}, 'lr': lr,
                'ema_momentum': ema_momentum, 'grad_norm': float(grad_norm),
                'target_entropy_per_part': [float(a/b) if b > 0 else None for a, b in zip(ent_num, ent_den)],
                'update_duration_sec': time.perf_counter()-started,
                'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
                'mask_ratio_local': float(torch.cat([m['actual_ratio'] for m in masks]).mean()) if all('actual_ratio' in m for m in masks) else None}
