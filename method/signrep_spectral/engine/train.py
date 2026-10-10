"""Single-device trainer for canonical continuation (+ optional auxiliary), and lambda calibration."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import TrainClipDataset, make_schedule
from ..data.manifest import read_jsonl, stable_int
from ..data.priors import HEADS, SPECTRAL_GROUPS
from ..data.scaler import load_scaler
from ..data.video import normalize_frames, worker_init
from ..losses.geometric import geometric_loss_stats
from ..losses.spectral import grouped_term_stats
from ..losses.stats import reduce_loss_stats
from ..losses.temporal import difference_loss_stats
from ..losses.regularizers import covariance_loss, variance_loss
from ..model import (TemporalDecoder, TrainingModel, attention_kernel, encoder_from_vendor_model,
                     make_explicit_mask, state_hash)
from ..vendor import load_vendor_model
from .checkpoint import save_training_checkpoint, rng_state, set_rng_state

REPO_ROOT = Path(__file__).resolve().parents[3]


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def build_training_model(cfg, device) -> Tuple[TrainingModel, dict]:
    base, audit = load_vendor_model(cfg["paths"]["vendor_root"], cfg["paths"]["checkpoint"], cfg["checkpoint_sha256"], "cpu")
    audit.pop("projection_model_class", None)
    enc = encoder_from_vendor_model(base)
    # Original cls_head / active_head are dropped: not forwarded, not optimized.
    del base
    m = cfg["model"]
    dec = TemporalDecoder(cfg["paths"]["vendor_root"], m["heads"], m["decoder_hidden_dim"], m["decoder_latent_dim"], m["total_frames"])
    dec.reset_parameters(m["decoder_seed"])
    model = TrainingModel(enc, dec).to(device)
    audit["initial_encoder_hash"] = state_hash(enc)
    audit["initial_decoder_hash"] = state_hash(dec)
    audit["initial_state_hash"] = state_hash(model)
    return model, audit


def param_groups(model: TrainingModel, cfg) -> List[dict]:
    t = cfg["train"]
    groups = {("enc", True): [], ("enc", False): [], ("dec", True): [], ("dec", False): []}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "enc" if name.startswith("encoder.") else "dec"
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name
        groups[(part, not no_decay)].append(p)
        assert id(p) not in seen
        seen.add(id(p))
    out = []
    for (part, decay), ps in groups.items():
        if ps:
            out.append({"params": ps, "lr": t["lr_encoder"] if part == "enc" else t["lr_decoder"],
                        "weight_decay": t["weight_decay"] if decay else 0.0, "name": f"{part}_{'decay' if decay else 'nodecay'}"})
    n_opt = sum(len(g["params"]) for g in out)
    n_req = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_opt == n_req, "every trainable parameter must appear exactly once"
    return out


def lr_factor(cfg, u: int) -> float:
    t = cfg["train"]
    w = t["lr_warmup_updates"]
    if w and u < w:
        return (u + 1) / w
    if t["scheduler"] == "cosine":
        progress = (u - w) / max(1, t["max_updates"] - w)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return 1.0


class LossComputer:
    def __init__(self, cfg, scaler, device):
        self.cfg = cfg
        self.mu = {h: torch.as_tensor(scaler[h]["mean"], device=device) for h in HEADS}
        self.sc = {h: torch.as_tensor(scaler[h]["scale"], device=device) for h in HEADS}

    def scaled(self, priors, batch):
        out = {}
        for h in HEADS:
            p = (priors[h].float() - self.mu[h]) / self.sc[h]
            t = (batch[f"t_{h}"].float() - self.mu[h]) / self.sc[h]
            out[h] = (p, t, batch[f"m_{h}"])
        return out

    def base(self, sc):
        L = self.cfg["loss"]
        stats = geometric_loss_stats(sc, L["smooth_l1_beta"], L["geometric_head_weights"])
        return reduce_loss_stats(stats, len(HEADS))

    def regularizer(self, features, logs):
        """Same explicit var/cov terms in every arm (part of the base objective when enabled)."""
        L = self.cfg["loss"]
        if not (L["var_weight"] or L["cov_weight"]):
            return features.sum() * 0.0
        z = features.float()
        total = z.sum() * 0.0
        if L["var_weight"]:
            v = variance_loss(z)
            logs["reg/var"] = float(v.detach())
            total = total + L["var_weight"] * v
        if L["cov_weight"]:
            c = covariance_loss(z)
            logs["reg/cov"] = float(c.detach())
            total = total + L["cov_weight"] * c
        return total

    def groups(self, sc):
        return {g: (sc[h][0][..., s], sc[h][1][..., s], sc[h][2][..., s]) for g, (h, s) in SPECTRAL_GROUPS.items()}

    def aux(self, sc, kind: str):
        L = self.cfg["loss"]
        g = self.groups(sc)
        if kind == "spectral_logmag_l1":
            stats = grouped_term_stats(g, L["windows"], "spectral")
            return reduce_loss_stats(stats, len(g) * len(L["windows"]))
        if kind == "window_time_l1":
            stats = grouped_term_stats(g, L["windows"], "window_time")
            return reduce_loss_stats(stats, len(g) * len(L["windows"]))
        if kind in ("velocity_l1", "acceleration_l1"):
            stats = difference_loss_stats(g, 1 if kind == "velocity_l1" else 2)
            return reduce_loss_stats(stats, len(g))
        raise ValueError(kind)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    out["rgb"] = normalize_frames(out.pop("frames"))  # [B,3,T,H,W]
    return out


def batch_mask(cfg, enc, batch_index: int, B: int, seed: int):
    g = torch.Generator().manual_seed(stable_int("mask", seed, batch_index) % (2**63))
    return make_explicit_mask(enc.num_mask_units(), B, cfg["model"]["mask_ratio"], g)


def load_train_records(cfg):
    recs = [r for r in read_jsonl(_abs(cfg["paths"]["manifest"])) if r["eligible"] and r["official_split"] == cfg["data"]["train_split"]]
    if not recs:
        raise RuntimeError("no eligible training records")
    return recs


def make_loader(cfg, recs, schedule):
    ds = TrainClipDataset(recs, schedule, cfg["targets"], cfg["data"]["clip_length"], cfg["data"]["max_relative_dt_deviation"])
    t = cfg["train"]
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"], drop_last=True,
                      pin_memory=t["device"].startswith("cuda"), persistent_workers=False,
                      prefetch_factor=4 if t["num_workers"] else None,
                      worker_init_fn=worker_init if t["num_workers"] else None)


def resolve_lambda(cfg) -> Tuple[str, float]:
    L = cfg["loss"]
    kind = L["auxiliary"]
    if kind == "none":
        return kind, 0.0
    if L["lambda_mode"] == "grad_ratio_calibrated":
        cal = json.load(open(_abs(cfg["paths"]["calibration"])))
        if cal["target_grad_ratio"] != L["target_grad_ratio"]:
            raise ValueError("calibration target ratio mismatch")
        return kind, float(cal["lambdas"][kind])
    return kind, float(L["lambda_spec"] if kind == "spectral_logmag_l1" else L["lambda_aux"])


def train(cfg, ctx) -> None:
    t = cfg["train"]
    device = torch.device(t["device"])
    seed_all(t["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scaler = load_scaler(_abs(cfg["paths"]["scaler"]))
    model, audit = build_training_model(cfg, device)
    aux_kind, lam = resolve_lambda(cfg)
    audit.update(aux_kind=aux_kind, lambda_value=lam, scaler_sha256=scaler["_sha256"])
    ctx.write_json("initial_state_audit.json", audit)
    recs = load_train_records(cfg)
    B = t["batch_size"]
    schedule = make_schedule(recs, t["max_updates"] * B, t["seed"], cfg["data"]["clip_length"])
    np.save(ctx.dir / "schedule.npy", schedule)
    opt = torch.optim.AdamW(param_groups(model, cfg), betas=tuple(t["betas"]), eps=t["eps"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    losses = LossComputer(cfg, scaler, device)
    loader = make_loader(cfg, recs, schedule)
    amp = t["amp"] and device.type == "cuda"
    warm = cfg["loss"]["spec_warmup_updates"]
    u = 0
    consumed = 0
    empty_streak = 0
    log = open(ctx.dir / "train_log.jsonl", "x")
    t0 = time.time()
    t_data = time.time()
    budget = cfg["budget"]["max_wall_seconds"]
    state = "SUCCESS"
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with attention_kernel(cfg["model"]["attention_kernel"]):
        for batch in loader:
            data_s = time.time() - t_data
            bi = consumed
            consumed += 1
            batch = to_device(batch, device)
            mask = batch_mask(cfg, model.encoder, bi, B, t["seed"]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(batch["rgb"], mask=mask)
            sc = losses.scaled(out["priors"], batch)
            base_loss, logs = losses.base(sc)
            base_loss = base_loss + losses.regularizer(out["features"], logs)
            if sum(logs[f"geom/{h}/count"] for h in HEADS) == 0:
                empty_streak += 1
                print(json.dumps({"skip": "empty_geometric_support", "batch": bi}), file=log, flush=True)
                if empty_streak >= 10:
                    raise RuntimeError("10 consecutive batches without geometric support")
                t_data = time.time()
                continue
            empty_streak = 0
            lam_eff = lam * (min(1.0, u / warm) if warm > 0 else 1.0)
            total = base_loss
            aux_val = None
            if lam_eff > 0:  # lambda == 0 -> auxiliary branch is never computed
                aux_loss, aux_logs = losses.aux(sc, aux_kind)
                total = base_loss + lam_eff * aux_loss
                aux_val = float(aux_loss.detach())
                logs.update(aux_logs)
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
            if not torch.isfinite(gnorm):
                raise FloatingPointError(f"non-finite grad norm at update {u}")
            f = lr_factor(cfg, u)
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * f
            opt.step()
            u += 1
            if u % t["log_every_updates"] == 0 or u == t["max_updates"]:
                rec = {"update": u, "batch_index": bi, "base": float(base_loss.detach()), "aux": aux_val,
                       "lambda_eff": lam_eff, "total": float(total.detach()), "grad_norm": float(gnorm),
                       "lr_factor": f, "data_wait_s": data_s, "elapsed_s": time.time() - t0, **logs}
                if device.type == "cuda":
                    rec["peak_mem_bytes"] = torch.cuda.max_memory_allocated(device)
                print(json.dumps(rec), file=log, flush=True)
            if u >= t["max_updates"]:
                break
            if budget and time.time() - t0 > budget:
                state = "BUDGET_STOP"
                break
            t_data = time.time()
    log.close()
    meta = {
        "successful_updates": u,
        "consumed_batches": consumed,
        "wall_seconds": time.time() - t0,
        "peak_mem_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "aux_kind": aux_kind,
        "lambda": lam,
    }
    (ctx.dir / "checkpoints").mkdir()
    save_training_checkpoint(ctx.dir / "checkpoints" / "final.pt", model, opt, cfg, audit, u, consumed, scaler["_sha256"])
    ctx.write_json("train_summary.json", meta)
    ctx.status(state if state != "SUCCESS" else "RUNNING", **meta)
    if state == "BUDGET_STOP":
        ctx.finalize_outputs()


def calibrate(cfg, ctx) -> None:
    """Encoder gradient-norm ratio ||g_aux||/||g_base|| on fixed train-only minibatches at init."""
    t = cfg["train"]
    device = torch.device(t["device"])
    seed_all(t["calibration_seed"])
    scaler = load_scaler(_abs(cfg["paths"]["scaler"]))
    model, audit = build_training_model(cfg, device)
    recs = load_train_records(cfg)
    B = t["batch_size"]
    K = t["calibration_batches"]
    schedule = make_schedule(recs, K * B, t["calibration_seed"], cfg["data"]["clip_length"])
    loader = make_loader(cfg, recs, schedule)
    losses = LossComputer(cfg, scaler, device)
    kinds = ["spectral_logmag_l1", "velocity_l1", "acceleration_l1", "window_time_l1"]
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    ratios = {k: [] for k in kinds}
    raw = []
    amp = t["amp"] and device.type == "cuda"
    model.train()
    with attention_kernel(cfg["model"]["attention_kernel"]):
        for bi, batch in enumerate(loader):
            batch = to_device(batch, device)
            mask = batch_mask(cfg, model.encoder, bi, B, t["calibration_seed"]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(batch["rgb"], mask=mask)
            sc = losses.scaled(out["priors"], batch)
            base_loss, _ = losses.base(sc)
            gb = torch.autograd.grad(base_loss, enc_params, retain_graph=True, allow_unused=True)
            nb = math.sqrt(sum(float(g.float().pow(2).sum()) for g in gb if g is not None))
            row = {"batch": bi, "base_loss": float(base_loss), "base_grad_norm": nb}
            for k in kinds:
                a, _ = losses.aux(sc, k)
                ga = torch.autograd.grad(a, enc_params, retain_graph=True, allow_unused=True)
                na = math.sqrt(sum(float(g.float().pow(2).sum()) for g in ga if g is not None))
                row[f"{k}/loss"] = float(a)
                row[f"{k}/grad_norm"] = na
                if nb > 0 and na > 0 and math.isfinite(na):
                    ratios[k].append(na / nb)
            raw.append(row)
    target = cfg["loss"]["target_grad_ratio"]
    lambdas, notes = {}, {}
    for k in kinds:
        if len(ratios[k]) < max(1, K // 2):
            notes[k] = "insufficient nonzero gradient batches; hold"
            continue
        med = float(np.median(ratios[k]))
        lambdas[k] = float(np.clip(target / med, 1e-4, 1e3))
        notes[k] = {"median_ratio_at_lambda1": med, "num_batches": len(ratios[k]),
                    "clipped": lambdas[k] != target / med}
    res = {"target_grad_ratio": target, "lambdas": lambdas, "notes": notes, "batches": raw,
           "schedule": schedule.tolist(), "initial_state_hash": audit["initial_state_hash"],
           "scaler_sha256": scaler["_sha256"], "rule": "lambda = target / median_b(||g_aux||/||g_base||) on encoder params"}
    ctx.write_json("calibration.json", res)
    print(json.dumps({"lambdas": lambdas, "notes": notes}, indent=2))
