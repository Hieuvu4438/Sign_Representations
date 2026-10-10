"""Token-removal adapters: hidden RGB never enters contextual attention."""
import importlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[2]


def pack_visible(dense, visible):
    if visible is None:
        return dense, torch.zeros(dense.shape[:2],dtype=torch.bool,device=dense.device), None
    if visible.shape != dense.shape[:2] or visible.dtype != torch.bool:
        raise AssertionError("visible_tokens must align with flattened token grid")
    ids = [v.nonzero().flatten() for v in visible]
    if any(len(i)==0 for i in ids):
        raise AssertionError("No visible context")
    tokens = pad_sequence([x[i] for x,i in zip(dense,ids)],batch_first=True)
    padding = torch.arange(tokens.shape[1],device=dense.device)[None] >= torch.tensor(
        [len(i) for i in ids],device=dense.device)[:,None]
    return tokens, padding, ids


def positions(grid, device, dtype):
    axes = [torch.linspace(0,1,n,device=device,dtype=dtype) for n in grid]
    return torch.stack(torch.meshgrid(*axes,indexing="ij"),-1).reshape(-1,3)


class TinyVideoViT(nn.Module):
    """From-scratch debug architecture; never presented as pretrained V-JEPA."""
    def __init__(self, dim=96, depth=2, heads=4, patch=16, tubelet=2, checkpointing=False, **_):
        super().__init__()
        self.dim, self.patch, self.tubelet = dim, patch, tubelet
        self.checkpointing = checkpointing
        self.patch_embed = nn.Conv3d(3,dim,(tubelet,patch,patch),stride=(tubelet,patch,patch))
        self.position = nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim,heads,4*dim,dropout=0,
                            batch_first=True,norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.upstream_init = {"name":"tiny_video_vit", "pretrained":False}

    def grid_shape(self, video):
        _,_,t,h,w = video.shape
        if t % self.tubelet or h % self.patch or w % self.patch:
            raise AssertionError("Video dimensions must be divisible by tubelet/patch")
        return t//self.tubelet,h//self.patch,w//self.patch

    def forward_tokens(self, video, visible_mask=None):
        grid = self.grid_shape(video)
        dense = self.patch_embed(video).flatten(2).transpose(1,2)
        dense = dense + self.position(positions(grid,video.device,dense.dtype))[None]
        x,padding,ids = pack_visible(dense,visible_mask)
        for block in self.blocks:
            if self.checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block,x,src_key_padding_mask=padding,use_reentrant=False)
            else:
                x = block(x,src_key_padding_mask=padding)
        x = self.norm(x).masked_fill(padding[...,None],0)
        global_feature = x.sum(1)/(~padding).sum(1,keepdim=True)
        return {"tokens": x, "padding_mask": padding,"grid_shape":grid,
                "global_feature":global_feature,"visible_indices":ids}


class VJEPAAdapter(nn.Module):
    def __init__(self, repo=None, family="vjepa2_1", architecture="vit_base", checkpoint=None,
                 checkpoint_key="ema_encoder", crop_size=384, num_frames=16,
                 patch=16, tubelet=2, checkpointing=True, expected_sha256=None, **_):
        super().__init__()
        repo = Path(repo or ROOT/"third_party/vjepa2").resolve()
        if not (repo/"LICENSE").exists():
            raise FileNotFoundError("Run scripts/fetch_assets.py before selecting vjepa2")
        sys.path.insert(0,str(repo))
        module_name = "app.vjepa_2_1.models.vision_transformer" if family == "vjepa2_1" else "src.models.vision_transformer"
        module = importlib.import_module(module_name)
        if repo not in Path(module.__file__).resolve().parents:
            raise AssertionError("V-JEPA namespace collision")
        kwargs = dict(img_size=(crop_size,crop_size),num_frames=num_frames,patch_size=patch,
                      tubelet_size=tubelet,use_rope=True,use_sdpa=True,
                      use_activation_checkpointing=checkpointing,uniform_power=False)
        if family == "vjepa2_1":
            kwargs.update(img_temporal_dim_size=1,interpolate_rope=True,n_output_distillation=1)
        self.encoder = getattr(module,architecture)(**kwargs)
        self.dim, self.patch, self.tubelet = self.encoder.embed_dim,patch,tubelet
        import subprocess
        commit = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        dirty = subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True).strip()
        if dirty:
            raise AssertionError("Upstream architecture checkout has modified tracked files")
        self.upstream_init = {"family":family,"architecture":architecture,"pretrained":False,"upstream_commit":commit}
        if checkpoint:
            from signrel.data.manifest import sha256
            digest = sha256(checkpoint)
            if expected_sha256 and digest != expected_sha256:
                raise AssertionError("Upstream checkpoint SHA256 mismatch")
            state = torch.load(checkpoint,map_location="cpu",weights_only=True)
            if checkpoint_key not in state:
                raise KeyError(f"Expected {checkpoint_key}, checkpoint has {list(state)[:20]}")
            cleaned = {k.removeprefix("module.").removeprefix("backbone."):v
                       for k,v in state[checkpoint_key].items()}
            # This architecture uses RoPE. Explicitly record obsolete absolute positions.
            dropped = [k for k in cleaned if k == "pos_embed"]
            for k in dropped:
                del cleaned[k]
            result = self.encoder.load_state_dict(cleaned,strict=True)
            self.upstream_init.update(pretrained=True,path=str(Path(checkpoint).resolve()),sha256=digest,
                                      checkpoint_key=checkpoint_key,missing_keys=result.missing_keys,
                                      unexpected_keys=result.unexpected_keys,dropped_keys=dropped,
                                      normalization="ImageNet RGB mean/std",patch=patch,tubelet=tubelet,
                                      crop_size=crop_size,num_frames=num_frames)

    grid_shape = TinyVideoViT.grid_shape

    def forward_tokens(self, video, visible_mask=None):
        grid = self.grid_shape(video)
        ids = None
        if visible_mask is None:
            x = self.encoder(video)
            padding = torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device)
        else:
            if visible_mask.shape != (video.shape[0],grid[0]*grid[1]*grid[2]):
                raise AssertionError("Upstream visible-token grid mismatch")
            ids = [v.nonzero().flatten() for v in visible_mask]
            if any(len(i)==0 for i in ids):
                raise AssertionError("No visible context")
            # Upstream requires equal sequence lengths. Bucket equal counts; never pad with hidden IDs.
            groups = {}
            for b,i in enumerate(ids):
                groups.setdefault(len(i),[]).append(b)
            outputs = [None]*len(ids)
            for group in groups.values():
                out = self.encoder(video[group],masks=torch.stack([ids[b] for b in group]))
                for b,z in zip(group,out):
                    outputs[b] = z
            x = pad_sequence(outputs,batch_first=True)
            padding = torch.arange(x.shape[1],device=x.device)[None] >= torch.tensor(
                [len(i) for i in ids],device=x.device)[:,None]
        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[-1] != self.dim:
            raise AssertionError("Unsupported upstream token output")
        if visible_mask is None and x.shape[1] != grid[0]*grid[1]*grid[2]:
            raise AssertionError("Upstream output/grid alignment mismatch")
        global_feature = x.masked_fill(padding[...,None],0).sum(1)/(~padding).sum(1,keepdim=True)
        return {"tokens":x,"padding_mask":padding,"grid_shape":grid,
                "global_feature":global_feature,"visible_indices":ids}


def make_backbone(config):
    cfg = dict(config)
    provider = cfg.pop("provider","tiny")
    frozen = cfg.pop("freeze",False)
    if provider == "tiny":
        model = TinyVideoViT(**cfg)
    elif provider == "vjepa2":
        model = VJEPAAdapter(**cfg)
    else:
        raise ValueError(f"Unknown backbone provider: {provider}")
    if frozen:
        model.requires_grad_(False)
    return model
