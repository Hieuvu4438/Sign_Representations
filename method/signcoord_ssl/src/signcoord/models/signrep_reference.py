"""Official SignRep T16 latent + fc readout for the external reference."""
import torch
from torch import nn

from .hiera_adapter import HieraAdapter


class SignRepReference(nn.Module):
    def __init__(self, vendor_root, checkpoint):
        super().__init__()
        self.encoder=HieraAdapter(vendor_root,16,checkpoint).eval()
        self.fc=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,768,bias=False))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
        self.fc.load_state_dict({k.removeprefix('fc.'):v for k,v in saved.items() if k.startswith('fc.')},strict=True)
        self.requires_grad_(False)

    def forward(self,rgb):
        if rgb.shape[2:] != (16,224,224): raise ValueError('official reference uses T16 224-square inputs')
        fv=torch.ones(len(rgb),16,device=rgb.device,dtype=torch.bool)
        times=torch.arange(16,device=rgb.device)[None].expand(len(rgb),-1)
        _,_,_,latent=self.encoder.encode_dense(rgb,fv,times)
        return self.fc(latent),latent
