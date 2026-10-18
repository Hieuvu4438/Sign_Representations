import torch
from torch import nn
from ..data.schema import PART_ORDER, ROOT_LOCAL_INDEX
from .motion import motion_targets
from .geometry import physical_huber_terms
from .latent_ce import latent_ce_terms


class HybridCriterion(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.config = config
        self.register_buffer("center", torch.zeros(dim, dtype=torch.float32))

    def quality(self, meta):
        return meta['quality'] if self.config.get('confidence_weighting', True) else meta['valid'].float()

    def targets(self, batch, masks, meta):
        cfg = self.config
        b, n = masks["target"].shape
        frame_targets = masks["target"].reshape(b, n//4, 4).repeat_interleave(2, dim=1)
        targets = {}
        den = {"latent": (self.quality(meta)*masks["target"]).sum().detach(),
               "geometry": self.center.new_zeros(()), "motion": self.center.new_zeros(())}
        for i, p in enumerate(PART_ORDER):
            stream = batch["parts"][p]
            gv = stream["joint_valid"] & batch["real"][..., None] & frame_targets[:, :, i, None]
            if p in ROOT_LOCAL_INDEX:
                gv = gv.clone()
                gv[:, :, ROOT_LOCAL_INDEX[p]] = False
            mt, mv, mw = motion_targets(batch, stream, cfg["motion_reference_fps"], cfg.get("max_motion_gap_sec", 0.2))
            mv = mv & frame_targets[:, :, i, None]
            factor = cfg["physical_part_weights"][i]
            gw = stream["confidence"] * factor
            mw = mw * factor
            if not cfg.get('confidence_weighting', True):
                gw, mw = torch.full_like(gw, factor), torch.full_like(mw, factor)
            den["geometry"] += (gw*gv).sum().detach()
            den["motion"] += (mw*mv).sum().detach()
            targets[p] = {"geometry": (stream["xy"], gv, gw), "motion": (mt, mv, mw)}
        return targets, den

    def forward(self, prediction, teacher, targets, masks, meta):
        cfg = self.config
        if cfg['latent_weight']:
            latent, den, teacher_sum = latent_ce_terms(prediction["latent"], teacher, masks["target"], self.quality(meta), self.center,
                                                      cfg["student_temperature"], cfg["teacher_temperature"])
        else:
            latent = prediction['latent'].float().sum()*0
            den = (self.quality(meta)*masks['target']).sum().detach()
            teacher_sum = torch.zeros_like(self.center)
        if cfg['latent_weight'] and cfg.get("latent_objective", "ce") == "regression":
            # Per-token LN regression control, without unit-sphere normalization.
            p = torch.nn.functional.layer_norm(torch.where(masks['target'][..., None], prediction['latent'], 0).float(), (teacher.shape[-1],))
            t = torch.nn.functional.layer_norm(torch.where(masks['target'][..., None], teacher.detach(), 0).float(), (teacher.shape[-1],))
            w = self.quality(meta)*masks["target"]
            latent = ((p-t).square().mean(-1)*w).sum()
        terms = {"latent": (latent, den)}
        for kind in ("geometry", "motion"):
            numerator = latent*0
            denominator = den*0
            for p in PART_ORDER:
                target, valid, weights = targets[p][kind]
                num, count = physical_huber_terms(prediction[kind][p], target, weights, valid, cfg[f"{kind}_huber_beta"])
                numerator = numerator+num
                denominator = denominator+count
            terms[kind] = (numerator, denominator)
        return terms, teacher_sum

    @torch.no_grad()
    def update_center(self, teacher_sum, weight):
        if weight.item() > 0:
            m = self.config["center_momentum"]
            self.center.mul_(m).add_(teacher_sum/weight, alpha=1-m)
