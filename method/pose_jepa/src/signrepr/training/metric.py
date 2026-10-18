from contextlib import nullcontext
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from .system import SSLSystem
from .pretrain import seed_everything, parameter_groups
from .distributed import world_size, rank, global_sum, all_finite, loss_for_backward
from ..models.pose_encoder import pool_clip
from ..models.heads import MetricHead
from ..models.ema import update_encoder_ema, momentum
from ..models.part_tokenizer import token_metadata
from ..masking.part_temporal import MaskGenerator
from ..losses.hybrid import HybridCriterion
from ..losses.supcon import supcon_terms, anchor_support
from ..data.collate import to_device


class MetricSystem(SSLSystem):
    def __init__(self, config, num_classes):
        super().__init__(config)
        self.metric_head = MetricHead(config['sequence_dim'], config['latent_dim'])
        self.classifier = nn.Linear(config['sequence_dim'], num_classes)

    def forward(self, batch, masks, replay):
        with torch.no_grad() if getattr(self, 'head_warmup', False) else nullcontext():
            encoded = self.encoder.encode_full(batch)
        raw, observed = pool_clip(encoded)
        out = {'embedding': self.metric_head(raw), 'classification': self.classifier(raw), 'valid': observed.any(-1)}
        if replay:
            out['replay'] = super().forward(batch, masks)
        return out


class MetricTrainer:
    """Head warmup followed by joint metric/replay updates; teacher stays encoder-only."""
    def __init__(self, config, initial_checkpoint, vocabulary, device='cpu'):
        self.config, self.device, self.vocabulary = deepcopy(config), torch.device(device), vocabulary
        seed_everything(config['run']['seed'])
        self.system = MetricSystem(config['model'], len(vocabulary)).to(self.device)
        initial = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
        if initial['config']['model'] != config['model']:
            raise ValueError('Metric architecture must match the selected Stage 1 checkpoint')
        ssl_state = initial['system']
        for name in ('predictor', 'geometry', 'motion'):
            getattr(self.system, name).load_state_dict({k.removeprefix(name+'.'): v for k, v in ssl_state.items() if k.startswith(name+'.')}, strict=True)
        self.system.encoder.load_state_dict(initial['teacher'], strict=True)
        self.teacher = deepcopy(self.system.encoder).float().eval().requires_grad_(False)
        self.criterion = HybridCriterion({**config['loss'], 'max_motion_gap_sec': config['data']['max_motion_gap_sec']}, config['model']['latent_dim']).to(self.device)
        self.criterion.load_state_dict(initial['criterion'])
        self.mask_rng = torch.Generator().manual_seed(config['run']['seed']+100003*rank())
        self.mask_generator = MaskGenerator(config['mask'])
        # Keep all trainable modules in DDP; warmup sends zero-connected gradients to SSL modules.
        self.wrapped = DistributedDataParallel(self.system, device_ids=[self.device.index] if self.device.type == 'cuda' else None) if world_size() > 1 else self.system
        update_encoder_ema(self.system.encoder, self.teacher, 0.)
        groups = []
        for name, module in [('encoder', self.system.encoder), ('ssl', nn.ModuleList([self.system.predictor, self.system.geometry, self.system.motion])),
                             ('head', nn.ModuleList([self.system.metric_head, self.system.classifier]))]:
            for group in parameter_groups(module, config['metric']['weight_decay']):
                group['role'] = name
                groups.append(group)
        self.optimizer = torch.optim.AdamW(groups, lr=config['metric']['head_lr'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.step = 0

    @property
    def phase(self):
        return 'warmup' if self.step < self.config['metric']['head_warmup_steps'] else 'joint'

    def update(self, batches):
        cfg = self.config['metric']
        warmup = self.phase == 'warmup'
        replay = not warmup and cfg['ssl_replay']
        batches = [to_device(b, self.device) for b in batches]
        metas = [token_metadata(b) for b in batches]
        labels = [torch.tensor([self.vocabulary.index(tuple(label)) for label in b['labels']], device=self.device) for b in batches]
        valid = [meta['valid'].any(-1) for meta in metas]
        masks = [self.mask_generator(meta['valid'], self.mask_rng) for meta in metas] if replay else [None]*len(batches)
        prepared = [self.criterion.targets(b, m, meta) for b, m, meta in zip(batches, masks, metas)] if replay else []
        denominators = {'supcon': global_sum(sum(anchor_support(l, v)[0].sum().float() for l, v in zip(labels, valid))),
                        'class': global_sum(sum(v.sum().float() for v in valid))}
        if not any(d.item() > 0 for d in denominators.values()):
            return {'successful': False, 'reason': 'zero_support', 'step': self.step}
        if replay:
            denominators.update({key: global_sum(sum(den[key] for _, den in prepared)) for key in ('latent', 'geometry', 'motion')})
        phase_step = self.step if warmup else self.step-cfg['head_warmup_steps']
        phase_total = cfg['head_warmup_steps'] if warmup else cfg['joint_steps']
        ramp = min(1., (phase_step+1)/max(1, cfg.get('lr_warmup_steps', 100)))
        factor = ramp*(1+math.cos(math.pi*phase_step/max(1, phase_total)))/2
        old_rates = [group['lr'] for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group['lr'] = (cfg['head_lr'] if group['role'] == 'head' else 0. if warmup else cfg['encoder_lr'])*factor
        self.optimizer.zero_grad(set_to_none=True)
        # Frozen parameters during warmup are represented by zero gradients and LR=0.
        # No encoder or SSL optimizer state evolves: reset grad to None before step.
        self.system.train()
        self.system.head_warmup = warmup
        self.teacher.eval()
        stats = torch.zeros_like(self.criterion.center)
        sums = {k: stats.new_zeros(()) for k in denominators}
        weights = {'supcon': 1., 'class': .25, 'latent': .25, 'geometry': .05, 'motion': .02}
        for i, (batch, meta, label, mask) in enumerate(zip(batches, metas, labels, masks)):
            sync = self.wrapped.no_sync() if world_size() > 1 and i < len(batches)-1 else nullcontext()
            with sync:
                out = self.wrapped(batch, mask, replay)
                sn, sd = supcon_terms(out['embedding'], label, out['valid'], cfg['temperature'])
                cn = nn.functional.cross_entropy(out['classification'], label, reduction='none')
                terms = {'supcon': (sn, sd), 'class': ((cn*out['valid']).sum(), out['valid'].sum())}
                if replay:
                    with torch.no_grad():
                        teacher = self.teacher.encode_full(batch)['features']
                    ssl, stats_i = self.criterion(out['replay'], teacher, prepared[i][0], mask, meta)
                    terms.update(ssl); stats += stats_i
                connected = sum(p.sum()*0 for name, p in self.system.named_parameters() if name.startswith(('encoder.', 'predictor.', 'geometry.', 'motion.')))
                loss = connected+sum(weights[k]*loss_for_backward(num, denominators[k], world_size()) for k, (num, _) in terms.items())
                loss.backward()
                for k, (num, _) in terms.items(): sums[k] += num.detach()
        if not all_finite(self.system.parameters(), self.device):
            self.optimizer.zero_grad(set_to_none=True)
            for group, old_rate in zip(self.optimizer.param_groups, old_rates): group['lr'] = old_rate
            return {'successful': False, 'reason': 'nonfinite_gradients', 'step': self.step}
        for group in self.optimizer.param_groups:
            if (warmup and group['role'] != 'head') or (not replay and group['role'] == 'ssl'):
                # Zero-connected DDP paths must not trigger AdamW decay/state
                # updates on modules frozen by the selected training phase.
                for p in group['params']: p.grad = None
        try:
            torch.nn.utils.clip_grad_norm_(self.system.parameters(), 1., error_if_nonfinite=True)
        except RuntimeError:
            self.optimizer.zero_grad(set_to_none=True)
            for group, old_rate in zip(self.optimizer.param_groups, old_rates): group['lr'] = old_rate
            return {'successful': False, 'reason': 'nonfinite_gradient_norm', 'step': self.step}
        self.optimizer.step()
        if replay and denominators['latent'] > 0:
            update_encoder_ema(self.system.encoder, self.teacher, momentum(self.step-cfg['head_warmup_steps'], cfg['joint_steps']))
            self.criterion.update_center(global_sum(stats), denominators['latent'])
        previous_phase = self.phase
        self.step += 1
        return {'successful': True, 'step': self.step, 'phase': previous_phase, 'ssl_replay': replay,
                'raw_losses': {k: float(global_sum(sums[k])/d) if d > 0 else 0. for k, d in denominators.items()},
                'global_contrastive_batch': len(labels[0])*world_size(), 'effective_update_batch': sum(map(len, labels))*world_size()}
