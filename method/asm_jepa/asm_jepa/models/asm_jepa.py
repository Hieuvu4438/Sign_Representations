"""Online backbone (tokenizer + encoder), SSL wrapper (predictor + auxiliary heads) and the EMA teacher."""
from __future__ import annotations

import copy

import torch
from torch import nn

from asm_jepa.data.schema import N_PARTS
from asm_jepa.features.tokens import TokenLayout, layout_from_config
from asm_jepa.models.encoder import Encoder, gather_tokens, pack
from asm_jepa.models.heads import GlobalHead, anchor_heads, mae_heads
from asm_jepa.models.pooling import pool_state
from asm_jepa.models.predictor import Predictor
from asm_jepa.models.tokenizer import Tokenizer


class Backbone(nn.Module):
    """Tokenizer + encoder. `forward_full` attends over all valid tokens; `forward_select` physically gathers
    the selected (context-safe) tokens first, so nothing else enters the encoder input sequence."""

    def __init__(self, cfg: dict, layout: TokenLayout | None = None):
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        self.layout = layout or layout_from_config(cfg)
        self.n_patches = self.layout.n_patches
        self.dim = m["dim"]
        self.tokenizer = Tokenizer(self.layout, m["dim"], d["patch_frames"], m["share_hand_tokenizer"])
        self.encoder = Encoder(m["dim"], m["depth"], m["heads"], m["ffn_dim"], m["dropout"], m["attention_dropout"],
                               m["drop_path"])

    def embed(self, b: dict, suffix: str = ""):
        return self.tokenizer(b["joints" + suffix], b["rel" + suffix], b["motion_geom" + suffix],
                              b["motion_time" + suffix], b["token_time"], b["log_duration"])

    def forward_full(self, b: dict, suffix: str = ""):
        return self.encoder(self.embed(b, suffix), b["token_valid"])

    def context_sequence(self, b: dict, select: torch.Tensor, suffix: str = ""):
        idx, valid = pack(select)
        return gather_tokens(self.embed(b, suffix), idx), idx, valid

    def forward_select(self, b: dict, select: torch.Tensor, suffix: str = ""):
        x, idx, valid = self.context_sequence(b, select, suffix)
        return self.encoder(x, valid), idx, valid

    def pool(self, out: torch.Tensor, b: dict):
        n = N_PARTS * self.n_patches
        feats, present = pool_state(out[:, :n], b["token_valid"][:, :n], self.n_patches)
        return feats, present

    def embed_clip(self, b: dict, suffix: str = ""):
        out = self.forward_full(b, suffix)
        return self.pool(out, b)


class ASMJEPA(nn.Module):
    """Trainable SSL modules. The teacher is a separate EMA copy (see make_teacher) and is never registered here,
    so `model.train()` can never switch the teacher to training mode."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.layout = layout_from_config(cfg)
        m, loss, tr = cfg["model"], cfg["loss"], cfg["train"]
        self.objective = tr["objective"]
        if self.objective not in ("jepa", "mae"):
            raise ValueError(f"train.objective must be jepa or mae, got {self.objective}")
        self.online = Backbone(cfg, self.layout)
        self.predictor = Predictor(self.layout, m["dim"], m["predictor_dim"], m["predictor_depth"],
                                   m["predictor_heads"], m["predictor_ffn_dim"], out_dim=m["dim"])
        self.anchor_enabled = self.objective == "jepa" and loss["lambda_anchor"] > 0
        self.global_enabled = self.objective == "jepa" and loss["lambda_global"] > 0
        self.anchor = anchor_heads(m["dim"], m["share_hand_tokenizer"]) if self.anchor_enabled else None
        self.mae = mae_heads(m["dim"], m["share_hand_tokenizer"]) if self.objective == "mae" else None
        self.global_head = GlobalHead(4 * m["dim"], 512, m["dim"]) if self.global_enabled else None
        A = self.layout.arrays
        for name in ("type", "sub", "patch", "group", "subgroup"):
            self.register_buffer(f"layout_{name}", torch.as_tensor(A[name]), persistent=False)

    def forward(self, b: dict, teacher_out: dict, centers: torch.Tensor, student_temp: float, teacher_temp: float,
                use_global: bool):
        """One SSL microbatch for the online branch. teacher_out = teacher_forward(...) (no grad).

        Online context: only context-safe tokens of the student view (physically gathered). Predictor queries:
        metadata of the target slots. Intact online pass over all valid student tokens feeds the state anchor
        and global terms. Returns (loss, logs, center payload)."""
        from asm_jepa.training import losses as Lo

        cfg_loss = self.cfg["loss"]
        usable = b["usable"]
        ctx_sel = b["context"] & usable[:, None]
        ctx_out, ctx_idx, ctx_valid = self.online.forward_select(b, ctx_sel)
        t_idx, t_valid, z = teacher_out["idx"], teacher_out["valid"], teacher_out["z"]
        pred = self.predictor(ctx_out, ctx_idx, ctx_valid, t_idx, t_valid, b["token_time"], b["log_duration"])
        logs = {}
        D = pred.shape[-1]
        flat_valid = t_valid.reshape(-1)
        groups = self.layout_group[t_idx].reshape(-1)
        weights = (torch.gather(b["token_weight"], 1, t_idx) * t_valid).reshape(-1)  # padded queries weigh 0
        if self.objective == "jepa":
            loss, lj = Lo.jepa_loss(pred.reshape(-1, D), z.reshape(-1, D), weights, groups,
                                    self.layout_subgroup[t_idx].reshape(-1), centers, student_temp, teacher_temp,
                                    cfg_loss["family_weights"], valid=flat_valid,
                                    n_subgroups=self.layout.n_families * self.layout.n_scales,
                                    group_weights=cfg_loss.get("group_weights"))
        else:
            state_q = t_valid & (self.layout_type[t_idx] == 0)
            loss, lj = Lo.mae_loss(pred, t_idx, state_q, self.layout_sub, self.layout_patch, self.mae, b)
        logs.update(lj)
        logs["loss/latent"] = loss.detach()
        if self.anchor_enabled or (self.global_enabled and use_global):
            u = self.online.forward_full(b)
            n_state = 4 * self.layout.n_patches
            if self.anchor_enabled:
                la, lg = Lo.anchor_loss(u[:, :n_state], b["token_valid"][:, :n_state], usable, self.anchor, b,
                                        self.layout.n_patches, cfg_loss["anchor_huber_beta"])
                loss = loss + cfg_loss["lambda_anchor"] * la
                logs.update(lg)
                logs["loss/anchor"] = la.detach()
            if self.global_enabled and use_global:
                from asm_jepa.training.distributed import gather_with_grad

                feats, _ = self.online.pool(u, b)
                g_on = self.global_head(feats["state_part_concat1024_l2"].to(u.dtype))
                keep = gather_with_grad(usable.float()) > 0.5  # equal shapes on every rank, select afterwards
                lgl, lg = Lo.global_loss(gather_with_grad(g_on)[keep], gather_with_grad(teacher_out["g"])[keep].detach(),
                                         cfg_loss["global_std_target"], cfg_loss["global_cov_weight"])
                loss = loss + cfg_loss["lambda_global"] * lgl
                logs.update(lg)
                logs["loss/global"] = lgl.detach()
        logs["loss/total"] = loss.detach()
        payload = None if z is None else {"z": z.reshape(-1, D).detach(), "groups": groups, "weights": weights.detach()}
        return loss, logs, payload


@torch.no_grad()
def teacher_forward(teacher: "Teacher | None", b: dict, need_global: bool) -> dict:
    """Teacher reads every valid token of the clean shared view, contextualises, then gathers target slots.
    Without a teacher (R1 coordinate MAE) only the target slot ids are returned."""
    idx, valid = pack(b["target"] & b["usable"][:, None])
    if teacher is None:
        return {"z": None, "idx": idx, "valid": valid}
    teacher.eval()
    out = teacher.backbone.forward_full(b, suffix="_clean")
    res = {"z": gather_tokens(out, idx), "idx": idx, "valid": valid}
    if need_global and teacher.global_head is not None:
        feats, _ = teacher.backbone.pool(out, b)
        res["g"] = teacher.global_head(feats["state_part_concat1024_l2"].to(out.dtype))
    return res


class Teacher(nn.Module):
    def __init__(self, model: ASMJEPA):
        super().__init__()
        self.backbone = copy.deepcopy(model.online)
        self.global_head = copy.deepcopy(model.global_head) if model.global_head is not None else None
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # the teacher is always in eval mode
        return super().train(False)


def make_teacher(model: ASMJEPA) -> Teacher:
    return Teacher(model)


def ema_pairs(teacher: Teacher, model: ASMJEPA):
    pairs = [(teacher.backbone, model.online)]
    if teacher.global_head is not None:
        pairs.append((teacher.global_head, model.global_head))
    return pairs
