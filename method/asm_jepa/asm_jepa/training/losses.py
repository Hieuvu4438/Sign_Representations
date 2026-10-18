"""Losses (§10): grouped centred-CE latent prediction, state anchor, global consistency, coordinate MAE (R1).

All softmax / statistics run in FP32 regardless of autocast.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from asm_jepa.data.schema import PART_ORDER
from asm_jepa.features.tokens import FAMILY_OF_GROUP, N_GROUPS

EPS = 1e-8
FAMILIES = ("state", "relation", "geometry", "timing")


def centered_ce(pred, z_teacher, center, student_temp, teacher_temp):
    """pred, z_teacher, center [N,D] -> per-token CE [N] and teacher entropy [N]."""
    t = F.softmax((z_teacher.detach().float() - center) / teacher_temp, dim=-1)
    logp = F.log_softmax(pred.float() / student_temp, dim=-1)
    ce = -(t * logp).sum(-1)
    ent = -(t * torch.log(t.clamp_min(1e-12))).sum(-1)
    return ce, ent


def jepa_loss(pred, z_teacher, weights, groups, subgroups, centers, student_temp, teacher_temp, family_weights,
              valid=None, n_subgroups: int = 9, group_weights=None):
    """Weighted CE per bucket (sum w*CE / sum w); buckets = groups for state/relation and (family, scale)
    subgroups for geometry/timing. A family averages its buckets that have targets; families mix with
    `family_weights` normalised over families with targets in this batch. Fully vectorised (no host syncs);
    entries with weight 0 (padding) contribute nothing. Returns (loss, logs as detached tensors)."""
    device = pred.device
    if pred.shape[0] == 0:
        return pred.sum() * 0.0, {}
    ce, ent = centered_ce(pred, z_teacher, centers[groups], student_temp, teacher_temp)
    w = weights.float()
    is_motion = groups >= 7
    bucket = groups * n_subgroups + torch.where(is_motion, subgroups, torch.zeros_like(subgroups))
    nb = N_GROUPS * n_subgroups
    num = torch.zeros(nb, device=device).index_add(0, bucket, w * ce)
    den = torch.zeros(nb, device=device).index_add(0, bucket, w)
    has = den > 0
    b_loss = torch.where(has, num / den.clamp_min(EPS), torch.zeros_like(num))
    fam_idx = torch.as_tensor([FAMILIES.index(FAMILY_OF_GROUP[g]) for g in range(N_GROUPS)],
                              device=device).repeat_interleave(n_subgroups)
    # optional per-group weights inside a family (diagnostics: e.g. down-weight body/face state targets)
    gw = (torch.as_tensor([float(x) for x in group_weights], device=device).repeat_interleave(n_subgroups)
          if group_weights is not None else torch.ones(nb, device=device))
    hw = has.float() * gw
    f_num = torch.zeros(len(FAMILIES), device=device).index_add(0, fam_idx, b_loss * hw)
    f_cnt = torch.zeros(len(FAMILIES), device=device).index_add(0, fam_idx, hw)
    f_has = f_cnt > 0
    f_loss = f_num / f_cnt.clamp_min(1.0)
    fw = torch.as_tensor([float(family_weights[f]) for f in FAMILIES], device=device) * f_has
    total = (fw * f_loss).sum() / fw.sum().clamp_min(EPS)
    total = total + pred.sum() * 0.0  # keeps a graph when no family has targets
    nan = torch.full_like(f_loss, float("nan"))
    f_log = torch.where(f_has, f_loss, nan).detach()  # NaN = no targets in this batch (skipped by nanmean)
    logs = {f"loss/{f}": f_log[i] for i, f in enumerate(FAMILIES)}
    m = valid.float() if valid is not None else (w > 0).float()
    logs["teacher_entropy"] = ((ent * m).sum() / m.sum().clamp_min(1.0)).detach()
    g_loss = torch.where(has, b_loss, torch.full_like(b_loss, float("nan"))).reshape(N_GROUPS, n_subgroups)[:7, 0]
    for g in range(7):
        logs[f"loss/group_{g}"] = g_loss[g].detach()
    return total, logs


def anchor_loss(u_state, token_valid_state, usable, heads, batch, n_patches, beta):
    """u_state [B,4*n_p,D] intact online state features. SmoothL1 averaged over observed components per part,
    then over parts that have components. Invalid tokens / padded frames never enter the denominator."""
    B, _, D = u_state.shape
    u = u_state.reshape(B, len(PART_ORDER), n_patches, D)
    tv = token_valid_state.reshape(B, len(PART_ORDER), n_patches) & usable[:, None, None]
    per_part, has, logs = [], [], {}
    for a, part in enumerate(PART_ORDER):
        target = batch[f"anchor_{part}"].float()
        valid = batch[f"anchor_{part}_valid"] & tv[:, a, :, None]
        n = valid.sum()
        pred = heads(part, u[:, a]).float()
        l = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
        lp = (l * valid).sum() / n.clamp_min(1)
        per_part.append(lp)
        has.append(n > 0)
        logs[f"anchor/{part}"] = torch.where(n > 0, lp, torch.full_like(lp, float("nan"))).detach()
    per_part, has = torch.stack(per_part), torch.stack(has).float()
    return (per_part * has).sum() / has.sum().clamp_min(1.0), logs


def mae_loss(pred, slot_idx, slot_valid, layout_sub, layout_patch, heads, batch):
    """R1 control: reconstruct the normalised coordinates of all frames of masked state patches (MSE over
    observed components). pred [B,Lq,D] at target slots slot_idx [B,Lq]."""
    per_part, logs = [], {}
    part_of = layout_sub[slot_idx]
    patch_of = layout_patch[slot_idx]
    for a, part in enumerate(PART_ORDER):
        sel = slot_valid & (part_of == a) & (patch_of >= 0)
        if not bool(sel.any()):
            continue
        b_idx, q_idx = torch.nonzero(sel, as_tuple=True)
        p_idx = patch_of[b_idx, q_idx]
        target = batch[f"anchor_{part}"][b_idx, p_idx].float()
        valid = batch[f"anchor_{part}_valid"][b_idx, p_idx]
        if valid.sum() == 0:
            continue
        out = heads(part, pred[b_idx, q_idx]).float()
        lp = (((out - target) ** 2) * valid).sum() / valid.sum()
        per_part.append(lp)
        logs[f"mae/{part}"] = lp.detach()
    if not per_part:
        return pred.sum() * 0.0, logs
    return torch.stack(per_part).mean(), logs


def global_loss(g_online, g_teacher, std_target=1.0, cov_weight=0.04):
    """MSE(G_online, stopgrad G_teacher) + variance hinge + off-diagonal covariance (unnormalised outputs)."""
    g_online, g_teacher = g_online.float(), g_teacher.detach().float()
    mse = F.mse_loss(g_online, g_teacher)
    std = g_online.std(dim=0, correction=0)
    var = F.relu(std_target - std).mean()
    x = g_online - g_online.mean(0)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    covl = (off**2).sum() / g_online.shape[1]
    return mse + var + cov_weight * covl, {"global/mse": mse.detach(), "global/var": var.detach(),
                                          "global/cov": covl.detach(), "global/std": std.mean().detach()}
