import torch
from torch import nn
from ..data.schema import PART_ORDER
from .spatial_gcn import SpatialGCN
from .graphs import BODY_EDGES, HAND_EDGES, FACE_EDGES


def token_metadata(batch):
    b, t = batch["real"].shape
    if t % 2:
        raise ValueError("Collated time dimension must be even")
    real = batch["real"].reshape(b, t//2, 2)
    valid = torch.stack([batch["parts"][p]["frame_valid"] & batch["real"] for p in PART_ORDER], -1).reshape(b, t//2, 2, 4).any(2)
    quality = torch.stack([batch["parts"][p]["quality"] for p in PART_ORDER], -1)
    quality = (quality * batch["real"][..., None]).reshape(b, t//2, 2, 4).sum(2) / real.sum(-1).clamp_min(1)[..., None]
    position = (batch["position"].reshape(b, t//2, 2) * real).sum(-1) / real.sum(-1).clamp_min(1)
    n = t//2*4
    return {"valid": valid.reshape(b, n), "quality": quality.reshape(b, n),
            "position": position[:, :, None].expand(b, t//2, 4).reshape(b, n),
            "indices": torch.arange(n, device=real.device).expand(b, n), "segments": t//2}


class PartTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        args = (cfg["spatial_channels"], cfg["encoder_dim"], cfg["anchor_dim"], cfg["joint_aggregation"])
        hand = SpatialGCN(21, HAND_EDGES, *args, use_anchor=cfg.get("global_hand_anchors", True))
        self.frontends = nn.ModuleDict({"body": SpatialGCN(9, BODY_EDGES, *args, use_anchor=False),
                                       "left_hand": hand, "right_hand": hand,
                                       "face": SpatialGCN(18, FACE_EDGES, *args)})
        d = cfg["encoder_dim"]
        self.segment = nn.Sequential(nn.Linear(2*d+2, d), nn.LayerNorm(d))

    def forward(self, batch):
        meta = token_metadata(batch)
        tokens = []
        for p in PART_ORDER:
            stream = batch["parts"][p]
            frame = self.frontends[p](stream)
            b, t, d = frame.shape
            fv = stream["frame_valid"] & batch["real"]
            frame = torch.where(fv[..., None], frame, 0).reshape(b, t//2, 2*d)
            token = self.segment(torch.cat([frame, fv.reshape(b, t//2, 2).to(frame.dtype)], -1))
            tokens.append(token)
        x = torch.stack(tokens, 2).flatten(1, 2)
        return torch.where(meta["valid"][..., None], x, 0), meta
