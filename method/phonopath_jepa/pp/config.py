"""Registered one-factor arms and reproducible run configuration."""
from dataclasses import dataclass, asdict, fields
import json
from pathlib import Path

ARMS = {
 'A0': dict(paths=False, global_mode='none'),
 'A1': {}, 'A2': dict(moment=True),
 'A3': dict(state=False, path_means=True, mask='motion_subtree', global_mode='full', loss='cosine'),
 'A4': dict(relations=False), 'A5': dict(global_mode='full'),
 'A6': dict(global_mode='none'), 'A7': dict(time_channel=True),
 'A8a': dict(smooth=False), 'A8b': dict(asinh=False),
 'A9': dict(mask='mixed_subtree'), 'A10': dict(mask='motion'),
 'A1-pb': dict(p_budget=True),
 'B1': dict(paths=False, global_mode='none', state_relations=True),
 'B2': dict(paths=False, global_mode='none', mask='mixed_subtree'),
 'B3': dict(paths=False, global_mode='none', aligned_warp=True),
}

@dataclass
class Config:
    arm: str = 'A1'
    state: bool = True
    paths: bool = True
    relations: bool = True
    global_mode: str = 'chen'
    smooth: bool = True
    asinh: bool = True
    moment: bool = False
    time_channel: bool = False
    path_means: bool = False
    state_relations: bool = False
    aligned_warp: bool = False
    p_budget: bool = False
    mask: str = 'plcm'
    loss: str = 'cce'
    structural_bias: bool = True
    lambda_p: float = 1.
    d: int = 256
    depth: int = 6
    heads: int = 4
    dp: int = 128
    pdepth: int = 4
    batch: int = 32
    epochs: int = 40
    lr: float = 3e-4
    wd: float = .05
    seed: int = 0
    speed: tuple = (.8,1.25)
    warp: float = .2
    morph: bool = True
    noise: float = .02
    ctx_drop: float = .1
    stats: str = ''
    cache: str = ''
    device: str = 'cuda'
    workers: int = 6
    save_epochs: tuple = (10,20,40)

    @classmethod
    def for_arm(cls, arm, **kw):
        if arm not in ARMS:
            raise ValueError(f'Unknown arm {arm}')
        return cls(arm=arm, **(ARMS[arm] | kw))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        unknown = set(data)-{f.name for f in fields(cls)}
        if unknown: raise ValueError(f'Unknown configuration keys: {unknown}')
        return cls.for_arm(**data)

    def validate(self):
        if self.d % self.heads or (self.d//self.heads)%2 or self.dp%self.heads or (self.dp//self.heads)%2:
            raise ValueError('RoPE requires even head dimension')
        if self.global_mode not in ('chen','full','none') or self.lambda_p < 0:
            raise ValueError('Invalid global mode or lambda_p')
        if self.mask not in ('plcm','mixed_subtree','motion','motion_subtree') or self.loss not in ('cce','cosine'):
            raise ValueError('Unknown mask or latent loss')
        if not (self.state or self.paths) or not 0<=self.ctx_drop<1 or not 0<=self.warp<1 or self.noise<0:
            raise ValueError('Invalid streams or augmentation')
        if self.batch < 1 or self.epochs < 1 or self.workers < 0:
            raise ValueError('Invalid run size')
        return self

    def to_dict(self): return asdict(self)
