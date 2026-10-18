"""Supervised readouts (§13.3-13.4): linear probe on frozen embeddings and ISLR fine-tuning / supervised scratch."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from asm_jepa.config import arch_diff, load_config, strip_meta
from asm_jepa.data.dataset import LabeledDataset, PoseStoreSource, SampleBuilder, worker_init
from asm_jepa.data.manifest import load_split, load_vocab
from asm_jepa.evaluation.classification import classification_metrics
from asm_jepa.evaluation.embeddings import autocast_ctx, load_embeddings, to_device
from asm_jepa.features.stats import load_stats
from asm_jepa.models.asm_jepa import Backbone
from asm_jepa.models.heads import FinetuneHead
from asm_jepa.training.checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state
from asm_jepa.training.sampler import StepBatchSampler
from asm_jepa.training.schedules import warmup_cosine
from asm_jepa.utils import append_jsonl, git_commit, now, resolve, seed_everything, sha256_file, write_json


# ------------------------------------------------------------------------------------------------ linear probe
def few_shot_subset(class_ids: np.ndarray, usable: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k usable official-train samples per class (all of them when a class has fewer), seeded."""
    rng = np.random.default_rng([seed, k, 2027])
    chosen = []
    for c in np.unique(class_ids):
        idx = np.flatnonzero((class_ids == c) & usable)
        chosen.extend(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())
    return np.sort(np.array(chosen, np.int64))


def train_linear_probe(cfg, train_path, val_path, run_dir: Path, device: str, few_shot_k: int | None = None,
                       few_shot_seed: int = 0):
    seed_everything(cfg["seed"])
    tr = cfg["train"]
    sha_before = {str(p): sha256_file(p) for p in (train_path, val_path)}
    tr_e, va_e = load_embeddings(train_path, cfg["feature_key"]), load_embeddings(val_path, cfg["feature_key"])
    if tr_e["meta"]["checkpoint_sha256"] != va_e["meta"]["checkpoint_sha256"]:
        raise ValueError("train/val embeddings come from different checkpoints")
    if tr_e["meta"]["split"] != "train" or va_e["meta"]["split"] != "val":
        raise ValueError("linear probe: train embeddings must be split=train and val embeddings split=val")
    n_classes = len(load_vocab(resolve(cfg["data"]["processed_root"]))) if cfg["num_classes_from_vocab"] else \
        int(tr_e["class_ids"].max() + 1)
    keep = tr_e["usable"].copy()
    few = None
    if few_shot_k:
        sel = few_shot_subset(tr_e["class_ids"], tr_e["usable"], few_shot_k, few_shot_seed)
        keep = np.zeros_like(keep)
        keep[sel] = True
        counts = np.bincount(tr_e["class_ids"][sel], minlength=n_classes)
        few = {"k_per_class": few_shot_k, "seed": few_shot_seed, "labelled_samples": int(len(sel)),
               "percent_of_train": 100.0 * len(sel) / len(tr_e["class_ids"]),
               "classes_with_fewer_than_k": int(((counts > 0) & (counts < few_shot_k)).sum()),
               "classes_without_samples": int((counts == 0).sum())}
    X = torch.as_tensor(tr_e["features"][keep])
    Y = torch.as_tensor(tr_e["class_ids"][keep])
    Xv = torch.as_tensor(va_e["features"]).to(device)
    head = torch.nn.Linear(X.shape[1], n_classes).to(device)
    if X.shape[1] != cfg["feature_dim"]:
        raise ValueError(f"feature dim {X.shape[1]} != {cfg['feature_dim']}")
    opt = torch.optim.AdamW(head.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    steps_per_epoch = math.ceil(len(X) / tr["batch_size"])
    total = steps_per_epoch * tr["epochs"]
    warm = steps_per_epoch * tr["warmup_epochs"]
    g = torch.Generator().manual_seed(cfg["seed"])
    best, step = None, 0
    run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(tr["epochs"]):
        head.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), tr["batch_size"]):
            idx = perm[i:i + tr["batch_size"]]
            for gr in opt.param_groups:
                gr["lr"] = warmup_cosine(step, total, tr["lr"], tr["min_lr"], warm)
            loss = F.cross_entropy(head(X[idx].to(device)), Y[idx].to(device), label_smoothing=tr["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        head.eval()
        with torch.no_grad():
            logits = head(Xv).float().cpu().numpy()
        m, _ = classification_metrics(logits, va_e["class_ids"], usable=va_e["usable"])
        append_jsonl(run_dir / "metrics.jsonl", {"epoch": epoch + 1, "step": step, "loss": loss.item(),
                                                 "val/cls/top1_instance": m["top1_instance"],
                                                 "val/cls/top5_instance": m["top5_instance"]})
        if best is None or m["top1_instance"] > best["top1_instance"]:
            best = {"epoch": epoch + 1, **{k: v for k, v in m.items() if k != "per_signer"}}
            _save_linear(run_dir / "checkpoints" / "best_val.pt", head, cfg, tr_e, n_classes, epoch + 1, best)
    _save_linear(run_dir / "checkpoints" / "last.pt", head, cfg, tr_e, n_classes, tr["epochs"], best)
    sha_after = {str(p): sha256_file(p) for p in (train_path, val_path)}
    if sha_before != sha_after:
        raise RuntimeError("frozen embeddings changed during probing")
    report = {"created": now(), "mode": "linear_probe", "best_val": best,
              "frozen_inputs_unchanged": True, "embeddings_sha256": sha_before,
              "encoder_checkpoint_sha256": tr_e["meta"]["checkpoint_sha256"], "encoder": tr_e["meta"]["encoder"],
              "train_samples_used": int(len(X)), "train_samples_unusable_excluded": int((~tr_e["usable"]).sum()),
              "few_shot": few,
              "selection_metric": cfg["selection_metric"], "test_exposure": False}
    write_json(run_dir / "reports" / "val.json", report)
    print(f"linear probe best val top1 {best['top1_instance']:.4f} at epoch {best['epoch']}")
    return report


def _save_linear(path, head, cfg, emb, n_classes, epoch, best):
    save_checkpoint(path, {"kind": "linear_probe", "head": head.state_dict(), "config": strip_meta(cfg),
                           "n_classes": n_classes, "feature_dim": head.in_features, "epoch": epoch, "best": best,
                           "encoder_meta": emb["meta"]})


# ------------------------------------------------------------------------------------------------ fine-tuning
class Classifier(torch.nn.Module):
    def __init__(self, model_cfg, n_classes, dropout):
        super().__init__()
        self.backbone = Backbone(model_cfg)
        self.head = FinetuneHead(4 * model_cfg["model"]["dim"], n_classes, dropout)

    def forward(self, b):
        f, present = self.backbone.embed_clip(b)
        return self.head(f["state_part_concat1024_l2"].to(next(self.head.parameters()).dtype)), present


def resolve_model_config(ft_cfg) -> dict:
    src = Path(ft_cfg["_meta"]["source"]) if ft_cfg["_meta"]["source"] else resolve("configs/eval/finetune.yaml")
    return load_config(src.parent / ft_cfg["model_config"], "pretrain")


class FinetuneTrainer:
    def __init__(self, cfg, run_dir: Path, init: str, device: str, overfit: int | None = None,
                 max_steps: int | None = None, resume: str | None = None, limit_train: int | None = None,
                 limit_eval: int | None = None):
        self.cfg, self.run_dir, self.init, self.device = cfg, Path(run_dir), init, torch.device(device)
        self.overfit, self.max_steps = overfit, max_steps
        seed_everything(cfg["seed"])
        mcfg = resolve_model_config(cfg)
        self.init_info = {"init": init}
        if init != "scratch":
            ck = load_checkpoint(init)
            if ck.get("teacher") is None:
                raise ValueError(f"{init} has no EMA teacher (e.g. the R1 MAE control); fine-tuning initialises "
                                 "from teacher weights by default")
            diffs = arch_diff(ck["config"], mcfg)
            if diffs:
                raise ValueError("model_config contradicts the checkpoint architecture (fail-fast): " + "; ".join(diffs))
            self.init_info.update(checkpoint_sha256=sha256_file(init), checkpoint_step=ck["step"],
                                  encoder="teacher", stats_sha256=ck["provenance"]["stats_sha256"])
        if overfit:
            mcfg = copy.deepcopy(mcfg)
            mcfg["model"]["drop_path"] = 0.0
        self.mcfg = mcfg
        root = resolve(cfg["data"]["processed_root"])
        stats_path = root / mcfg["data"]["stats_path"]
        self.stats = load_stats(stats_path, len(mcfg["features"]["interval_widths_sec"]))
        self.stats_sha = sha256_file(stats_path)
        if init != "scratch" and self.init_info["stats_sha256"] != self.stats_sha:
            raise ValueError("feature statistics differ from those used by the SSL checkpoint")
        self.builder = SampleBuilder(mcfg, self.stats)
        self.n_classes = len(load_vocab(root))
        tr_rows = load_split(root, cfg["data"]["train_split"])
        if limit_train:
            tr_rows = tr_rows[:limit_train]
        va_rows = load_split(root, cfg["data"]["val_split"])
        if limit_eval:  # smoke runs only
            va_rows = va_rows[:limit_eval]
        if overfit:
            src = PoseStoreSource(tr_rows, root)
            chosen = []
            for i, r in enumerate(tr_rows):
                if src[i] is not None and self.builder.build(src[i], None, "eval")["usable"]:
                    chosen.append(r)
                if len(chosen) == overfit:
                    break
            tr_rows = va_rows = chosen
        self.tr_rows, self.va_rows = tr_rows, va_rows
        self.train_mode = "eval" if overfit else ("train_view" if cfg["augment"] else "eval")
        self.train_ds = LabeledDataset(PoseStoreSource(tr_rows, root), self.builder, [r["class_id"] for r in tr_rows],
                                       self.train_mode, seed=cfg["seed"])
        self.val_ds = LabeledDataset(PoseStoreSource(va_rows, root), self.builder, [r["class_id"] for r in va_rows],
                                     "eval")
        self.model = Classifier(mcfg, self.n_classes, 0.0 if overfit else cfg["head_dropout"]).to(self.device)
        if init != "scratch":
            state = {k[len("backbone."):]: v for k, v in ck["teacher"].items() if k.startswith("backbone.")}
            self.model.backbone.load_state_dict(state, strict=True)
        tr = cfg["train"]
        self.bs = min(tr["batch_size_per_gpu"], len(tr_rows))
        self.accum = 1 if overfit else tr["accum_steps"]
        self.per_epoch_micro = len(tr_rows) // self.bs
        self.steps_per_epoch = max(self.per_epoch_micro // self.accum, 1)
        self.total_steps = max_steps or self.steps_per_epoch * tr["epochs"]
        self.warmup = 0 if overfit else self.steps_per_epoch * tr["warmup_epochs"]
        from asm_jepa.training.optim import param_groups

        groups = param_groups({"encoder": self.model.backbone, "head": self.model.head}, tr["weight_decay"],
                              {"encoder": tr["encoder_lr"] / tr["head_lr"], "head": 1.0})
        self.opt = torch.optim.AdamW(groups, lr=tr["head_lr"])
        self.amp = autocast_ctx(device, tr["precision"])
        self.step, self.micro, self.epoch, self.best = 0, 0, 0, None
        if resume:
            self._resume(resume)

    def _resume(self, path):
        ck = load_checkpoint(path, map_location="cpu")
        if arch_diff(ck["model_config"], self.mcfg):
            raise ValueError("resume: architecture differs")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.step, self.micro, self.epoch, self.best = ck["step"], ck["micro"], ck["epoch"], ck["best"]
        set_rng_state(ck["rng"])

    def _save(self, name):
        save_checkpoint(self.run_dir / "checkpoints" / name, {
            "kind": "finetune_islr", "model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
            "step": self.step, "micro": self.micro, "epoch": self.epoch, "best": self.best, "rng": rng_state(),
            "config": strip_meta(self.cfg), "model_config": strip_meta(self.mcfg), "n_classes": self.n_classes,
            "init": self.init_info, "stats_sha256": self.stats_sha, "git": git_commit(),
            "overfit_samples": self.overfit})

    @torch.no_grad()
    def evaluate(self, ds, workers=None):
        self.model.eval()
        nw = self.cfg["data"]["num_workers"] if workers is None else workers
        loader = torch.utils.data.DataLoader(ds, batch_size=self.cfg["train"]["eval_batch_size"], num_workers=nw,
                                             worker_init_fn=worker_init if nw else None)
        logits, ys, usable = [], [], []
        for batch in loader:
            b = to_device(batch, self.device)
            with autocast_ctx(self.device, self.cfg["train"]["precision"]):
                out, present = self.model(b)
            logits.append(out.float().cpu().numpy())
            ys.append(batch["class_id"].numpy())
            usable.append((b["usable"] & present.any(1)).cpu().numpy())
        self.model.train()
        return np.concatenate(logits), np.concatenate(ys), np.concatenate(usable)

    def train(self):
        cfg, tr = self.cfg, self.cfg["train"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_dir / "provenance.json", {
            "created": now(), "git": git_commit(), "init": self.init_info, "overfit_samples": self.overfit,
            "n_train": len(self.tr_rows), "n_val": len(self.va_rows), "train_mode": self.train_mode,
            "steps_per_epoch": self.steps_per_epoch, "total_steps": self.total_steps,
            "effective_batch": self.bs * self.accum, "selection_metric": cfg["selection_metric"],
            "test_exposure": False, "pretrain_labels": "none (SSL)" if self.init != "scratch" else "n/a (scratch)"})
        nw = cfg["data"]["num_workers"]
        sampler = StepBatchSampler(len(self.train_ds), self.bs, cfg["seed"], start_micro=self.micro)
        loader = torch.utils.data.DataLoader(self.train_ds, batch_sampler=sampler, num_workers=nw,
                                             pin_memory=cfg["data"]["pin_memory"] and self.device.type == "cuda",
                                             worker_init_fn=worker_init if nw else None,
                                             persistent_workers=bool(nw))
        it = iter(loader)
        self.model.train()
        t0, losses = time.time(), []
        while self.step < self.total_steps:
            lr = warmup_cosine(self.step, self.total_steps, tr["head_lr"], tr["head_lr"] * tr["final_lr_ratio"],
                               self.warmup)
            for g in self.opt.param_groups:
                g["lr"] = lr * g["lr_scale"]
            self.opt.zero_grad(set_to_none=True)
            for _ in range(self.accum):
                b = to_device(next(it), self.device)
                self.micro += 1
                with autocast_ctx(self.device, tr["precision"]):
                    logits, present = self.model(b)
                use = b["usable"] & present.any(1)
                ce = F.cross_entropy(logits.float(), b["class_id"], reduction="none",
                                     label_smoothing=0.0 if self.overfit else tr["label_smoothing"])
                loss = (ce * use).sum() / use.sum().clamp_min(1)
                (loss / self.accum).backward()
                losses.append(float(loss.detach()))
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tr["grad_clip_norm"])
            if not torch.isfinite(gn):
                continue
            self.opt.step()
            self.step += 1
            if self.step % tr["log_every_steps"] == 0:
                append_jsonl(self.run_dir / "metrics.jsonl", {"step": self.step, "loss": float(np.mean(losses)),
                                                              "lr_head": lr, "grad_norm": float(gn),
                                                              "time": time.time() - t0})
                print(f"step {self.step}/{self.total_steps} loss {np.mean(losses):.4f}", flush=True)
                losses = []
            epoch_done = self.step % self.steps_per_epoch == 0 or self.step == self.total_steps
            if epoch_done and not self.overfit:
                self.epoch = math.ceil(self.step / self.steps_per_epoch)
                self._validate()
                self._save("last.pt")
        if self.overfit:
            logits, y, usable = self.evaluate(self.train_ds if self.train_mode == "eval" else self.val_ds, workers=0)
            m, _ = classification_metrics(logits, y, usable=usable)
            res = {"overfit_samples": self.overfit, "steps": self.step, "train_top1": m["top1_instance"],
                   "passed": m["top1_instance"] == 1.0}
            write_json(self.run_dir / "reports" / "overfit.json", res)
            self._save("last.pt")
            print("overfit sanity:", res)
            return res
        return self.best

    def _validate(self):
        logits, y, usable = self.evaluate(self.val_ds)
        signers = [r["signer_id"] for r in self.va_rows]
        m, _ = classification_metrics(logits, y, signers=signers, usable=usable)
        row = {"step": self.step, "epoch": self.epoch, **{f"val/cls/{k}": v for k, v in m.items()
                                                          if not isinstance(v, dict)}}
        append_jsonl(self.run_dir / "metrics.jsonl", row)
        print(f"[val epoch {self.epoch}] top1 {m['top1_instance']:.4f} top5 {m['top5_instance']:.4f}", flush=True)
        if self.best is None or m["top1_instance"] > self.best["top1_instance"]:
            self.best = {"epoch": self.epoch, "step": self.step,
                         **{k: v for k, v in m.items() if not isinstance(v, dict)}}
            self._save("best_val.pt")
            write_json(self.run_dir / "reports" / "best_val.json", {**self.best, "per_signer": m.get("per_signer")})
