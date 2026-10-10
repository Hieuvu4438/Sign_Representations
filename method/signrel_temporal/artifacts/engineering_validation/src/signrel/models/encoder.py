"""Joint time-major contextual encoder; clean targets never enter forward."""
import math
import torch
from torch import nn
from .heads import StateHeads

INPUT_FIELDS = {'appearance','app_valid','body','body_coord_valid','body_valid','app_ssl_mask',
                'body_ssl_mask','time_padding','relative_time_ms'}


def physical_time_embedding(milliseconds,width):
    frequency = torch.exp(torch.arange(0,width,2,device=milliseconds.device,dtype=torch.float32)*(-math.log(10000)/width))
    angle = milliseconds.float()[...,None]/1000*frequency
    out = torch.empty(*milliseconds.shape,width,device=milliseconds.device)
    out[...,0::2],out[...,1::2] = angle.sin(),angle.cos()
    return out


class SignRelEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        d = cfg['hidden_dim']
        self.width = d
        self.hand_projection = nn.ModuleList([nn.Sequential(nn.Linear(384,d),nn.LayerNorm(d)) for _ in range(2)])
        self.body_projection = nn.Sequential(nn.Linear(28,d),nn.LayerNorm(d))
        self.part = nn.Parameter(torch.empty(3,d))
        self.mask_token = nn.Parameter(torch.empty(3,d))
        self.missing_token = nn.Parameter(torch.empty(3,d))
        layer = nn.TransformerEncoderLayer(d,cfg['heads'],cfg['feedforward_dim'],cfg['dropout'],
                                            activation='gelu',batch_first=True,norm_first=cfg['norm_first'])
        self.transformer = nn.TransformerEncoder(layer,cfg['layers'],nn.LayerNorm(d),enable_nested_tensor=False)
        self.heads = StateHeads(d)
        # Reinitialize cloned layers independently rather than inheriting equal copies.
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module,nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module,nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None: nn.init.zeros_(module.in_proj_bias)
        for value in (self.part,self.mask_token,self.missing_token):
            nn.init.normal_(value,std=.02)

    def encode(self,student):
        if set(student) != INPUT_FIELDS:
            raise ValueError(f'Unexpected encoder fields: {set(student)^INPUT_FIELDS}')
        app,body = student['appearance'],student['body']
        b,t,a,d = app.shape
        if (a,d) != (2,384) or body.shape != (b,t,14):
            raise ValueError('Encoder input shape mismatch')
        for key in ('app_valid','app_ssl_mask','body_coord_valid','body_valid','body_ssl_mask','time_padding'):
            if student[key].dtype != torch.bool:
                raise ValueError(f'{key} must be bool')
        if not all(torch.isfinite(student[k]).all() for k in ('appearance','body','relative_time_ms')):
            raise ValueError('Non-finite encoder inputs')
        tokens = torch.stack([proj(app[:,:,i]) for i,proj in enumerate(self.hand_projection)]+
                             [self.body_projection(torch.cat((body,student['body_coord_valid'].float()),dim=-1))],dim=2)
        valid = torch.cat((student['app_valid'],student['body_valid'][...,None]),dim=2)
        masked = torch.cat((student['app_ssl_mask'],student['body_ssl_mask'][...,None]),dim=2)
        tokens = torch.where(valid[...,None],tokens,self.missing_token[None,None])
        tokens = torch.where((masked&valid)[...,None],self.mask_token[None,None],tokens)
        tokens = tokens+self.part[None,None]+physical_time_embedding(student['relative_time_ms'],self.width)[:,:,None]
        key_padding = student['time_padding'][...,None]|~valid
        if key_padding.flatten(1).all(1).any():
            raise ValueError('All attention keys are padding')
        context = self.transformer(tokens.flatten(1,2),src_key_padding_mask=key_padding.flatten(1)).reshape(b,t,3,self.width)
        return context

    def forward(self,student):
        context = self.encode(student)
        app_prediction,body_prediction = self.heads(context)
        return {'context':context,'app_prediction':app_prediction,'body_prediction':body_prediction}


def initialized_model(cfg,seed):
    # Initialization does not consume dropout/window/pair RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return SignRelEncoder(cfg)
