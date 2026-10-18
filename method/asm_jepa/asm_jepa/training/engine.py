"""SSL pretraining loop (§12.2) with explicit update order:

  per optimizer step: lr <- schedule(step); zero grads
    per microbatch: teacher (no grad, eval) on the clean shared view -> targets, centre sums
                    online context (student view, safe tokens only) -> predictor -> grouped centred CE
                    intact online pass -> state anchor / global terms; backward(loss / accum)
  unscale (fp16) -> clip -> step; only a *successful* step updates EMA, centres and the step counter.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from asm_jepa.config import arch_diff, config_hash, dump_resolved, strip_meta
from asm_jepa.data.dataset import LabeledDataset, PoseStoreSource, SampleBuilder, SSLDataset, worker_init
from asm_jepa.data.manifest import load_split
from asm_jepa.data.synthetic import SyntheticSource
from asm_jepa.evaluation.diagnostics import embedding_diagnostics
from asm_jepa.evaluation.embeddings import extract_embeddings, to_device
from asm_jepa.evaluation.retrieval import evaluate_retrieval
from asm_jepa.features.stats import identity_stats, load_stats
from asm_jepa.features.tokens import N_GROUPS
from asm_jepa.models.asm_jepa import ASMJEPA, ema_pairs, make_teacher, teacher_forward
from asm_jepa.training import distributed as D
from asm_jepa.training.centers import Centers
from asm_jepa.training.checkpoint import (load_checkpoint, prune_step_checkpoints, rng_state, save_checkpoint,
                                          set_rng_state)
from asm_jepa.training.ema import ema_update
from asm_jepa.training.optim import build_adamw, param_groups, set_lr
from asm_jepa.training.sampler import StepBatchSampler
from asm_jepa.training.schedules import ema_momentum, teacher_temperature, warmup_cosine
from asm_jepa.utils import append_jsonl, git_commit, now, read_jsonl, resolve, seed_everything, sha256_file, write_json

SYNTH_TRAIN, SYNTH_GALLERY, SYNTH_VAL, SYNTH_DIAG = 0, 1, 2, 3


class PretrainEngine:
    def __init__(self, cfg: dict, run_dir: Path, synthetic: bool = False, limit_train: int | None = None,
                 resume: str | None = None, profile_steps: int | None = None, device: str | None = None,
                 limit_eval: int | None = None):
        self.cfg, self.synthetic, self.limit_train, self.profile_steps = cfg, synthetic, limit_train, profile_steps
        self.limit_eval = limit_eval  # smoke runs only: validation on the first N gallery / val clips
        self.rank, self.world, local = D.init_from_env()
        self.device = torch.device(device or (f"cuda:{local}" if torch.cuda.is_available() else "cpu"))
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        tr, d = cfg["train"], cfg["data"]
        if cfg["eval"]["test_during_training"] or "test" in (d["val_split"], d["gallery_split"]):
            raise ValueError("the test split is never used during SSL training/selection (eval.test_during_training, "
                             "data.val_split, data.gallery_split)")
        if d["gallery_split"] != "train":
            raise ValueError("the retrieval gallery is the official train split")
        if resume is None and self.ckpt_dir.exists() and any(self.ckpt_dir.glob("*.pt")):
            raise SystemExit(f"{self.ckpt_dir} already holds checkpoints; pass --resume or use a new --run-dir")
        seed_everything(cfg["seed"] + self.rank)
        self.processed_root = resolve(d["processed_root"])
        self.objective = tr["objective"]
        n_scales = len(cfg["features"]["interval_widths_sec"])
        if synthetic:
            self.stats = identity_stats(n_scales)
            self.stats_sha = "synthetic_identity_fixture"
        else:
            stats_path = self.processed_root / d["stats_path"]
            self.stats = load_stats(stats_path, n_scales)
            self.stats_sha = sha256_file(stats_path)
        self.builder = SampleBuilder(cfg, self.stats, self.objective)
        self.layout = self.builder.layout
        self._setup_data()
        self._setup_model()
        self.step, self.micro_consumed, self.samples_seen, self.overflows = 0, 0, 0, 0
        self.best = None
        self.resumed_from = None
        self._pending_rng = None
        if resume:
            self._resume(resume)
        self._setup_loader()
        if self._pending_rng is not None:  # after the loader iterator, which draws from the global torch RNG
            set_rng_state(self._pending_rng)

    # ----------------------------------------------------------------------------------------------- setup
    def _setup_data(self):
        cfg, d = self.cfg, self.cfg["data"]
        if self.synthetic:
            self.train_source = SyntheticSource(max(512, 4 * cfg["train"]["batch_size_per_gpu"] * self.world), SYNTH_TRAIN)
            self.manifest_sha = "synthetic"
            self.gallery = (SyntheticSource(256, SYNTH_GALLERY), None)
            self.val = (SyntheticSource(128, SYNTH_VAL), None)
            self.n_classes = 10
            return
        manifest = self.processed_root / d["ssl_manifest"]
        rows = read_jsonl(manifest)
        if any("gloss" in r or "class_id" in r for r in rows):
            raise RuntimeError(f"{manifest} contains labels; SSL manifests must be label-free")
        self.manifest_sha = sha256_file(manifest)
        if self.limit_train:
            rows = sorted(rows, key=lambda r: r["sample_id"])[: self.limit_train]
        self.train_source = PoseStoreSource(rows, self.processed_root)
        self.train_ids = [r["sample_id"] for r in rows]
        self.gallery = self.val = None
        if cfg["eval"]["frozen_every_steps"] > 0 or True:
            try:
                g_rows = load_split(self.processed_root, d["gallery_split"])
                v_rows = load_split(self.processed_root, d["val_split"])
                if self.limit_eval:
                    g_rows, v_rows = g_rows[: self.limit_eval], v_rows[: self.limit_eval]
                self.gallery = (PoseStoreSource(g_rows, self.processed_root), g_rows)
                self.val = (PoseStoreSource(v_rows, self.processed_root), v_rows)
                from asm_jepa.data.manifest import load_vocab

                self.n_classes = len(load_vocab(self.processed_root))
            except FileNotFoundError:
                if cfg["eval"]["frozen_every_steps"] > 0:
                    raise

    def _setup_model(self):
        cfg, tr = self.cfg, self.cfg["train"]
        self.model = ASMJEPA(cfg).to(self.device)
        self.teacher = make_teacher(self.model).to(self.device) if self.objective == "jepa" else None
        self.centers = Centers(N_GROUPS, cfg["model"]["dim"], cfg["loss"]["center_momentum"]).to(self.device)
        stat_batch = tr["batch_size_per_gpu"] * self.world
        self.use_global = self.model.global_enabled and stat_batch >= cfg["loss"]["global_min_statistical_batch"]
        self.global_note = None
        if self.model.global_enabled and not self.use_global:
            self.global_note = (f"global term disabled: statistical batch {stat_batch} < "
                                f"{cfg['loss']['global_min_statistical_batch']}")
        if tr["optimizer"] != "adamw":
            raise ValueError("only adamw is implemented")
        modules = {"online": self.model.online, "predictor": self.model.predictor, "anchor": self.model.anchor,
                   "mae": self.model.mae, "global": self.model.global_head if self.use_global else None}
        if self.model.global_head is not None and not self.use_global:
            self.model.global_head.requires_grad_(False)
        self.optimizer = build_adamw(param_groups(modules, tr["weight_decay"]), tr["lr"], tr["betas"], tr["eps"])
        prec = tr["precision"]
        self.amp_dtype = None
        if self.device.type == "cuda" and prec in ("bf16", "fp16"):
            self.amp_dtype = torch.bfloat16 if (prec == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None
        self.precision_used = {torch.bfloat16: "bf16", torch.float16: "fp16", None: "fp32"}[self.amp_dtype]
        self.ddp_model = self.model
        if D.is_dist():
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device.index], find_unused_parameters=True)
        if tr["compile"]:
            raise ValueError("train.compile is intentionally unsupported (debug/resume simplicity)")
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_params_encoder = sum(p.numel() for p in self.model.online.parameters())

    def _setup_loader(self):
        d, tr = self.cfg["data"], self.cfg["train"]
        self.sampler = StepBatchSampler(len(self.train_source), tr["batch_size_per_gpu"], self.cfg["seed"],
                                        start_micro=self.micro_consumed, rank=self.rank, world_size=self.world)
        nw = d["num_workers"]
        kw = dict(batch_sampler=self.sampler, num_workers=nw, pin_memory=d["pin_memory"] and self.device.type == "cuda",
                  worker_init_fn=worker_init if nw else None)
        if nw:
            kw.update(persistent_workers=d["persistent_workers"], prefetch_factor=4)
        self.loader = torch.utils.data.DataLoader(SSLDataset(self.train_source, self.builder), **kw)
        self.loader_it = iter(self.loader)

    def provenance(self) -> dict:
        tr = self.cfg["train"]
        return {"created": now(), "git": git_commit(), "config_sha256": config_hash(self.cfg),
                "manifest_sha256": self.manifest_sha, "stats_sha256": self.stats_sha, "synthetic": self.synthetic,
                "limit_train_samples": self.limit_train,
                "limit_eval_samples": self.limit_eval, "n_train_items": len(self.train_source),
                "world_size": self.world, "physical_batch": tr["batch_size_per_gpu"],
                "statistical_batch": tr["batch_size_per_gpu"] * self.world, "accum_steps": tr["accum_steps"],
                "effective_batch": tr["batch_size_per_gpu"] * self.world * tr["accum_steps"],
                "global_term_active": self.use_global, "global_note": self.global_note,
                "precision": self.precision_used, "params_total": self.n_params,
                "params_encoder": self.n_params_encoder, "layout_tokens": self.layout.n_tokens,
                "objective": self.objective, "selection_uses_val_labels": self.cfg["eval"]["selection_uses_val_labels"],
                "ssl_uses_labels": False, "resumed_from": self.resumed_from, "device": str(self.device),
                "torch": torch.__version__}

    # ----------------------------------------------------------------------------------------------- state
    def state(self) -> dict:
        return {"model": self.model.state_dict(), "teacher": self.teacher.state_dict() if self.teacher else None,
                "centers": self.centers.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler else None, "step": self.step,
                "micro_consumed": self.micro_consumed, "samples_seen": self.samples_seen,
                "overflows": self.overflows, "rng": rng_state(), "config": strip_meta(self.cfg),
                "seed": self.cfg["seed"], "world_size": self.world, "best": self.best,
                "provenance": {"manifest_sha256": self.manifest_sha, "stats_sha256": self.stats_sha,
                               "config_sha256": config_hash(self.cfg), "git": git_commit(),
                               "synthetic": self.synthetic, "limit_train_samples": self.limit_train,
                               "objective": self.objective, "precision": self.precision_used}}

    def save(self, name: str):
        if D.is_main():
            save_checkpoint(self.ckpt_dir / name, self.state())

    def _resume(self, path):
        ck = load_checkpoint(path, map_location="cpu")
        prov = ck["provenance"]
        problems = []
        if prov["manifest_sha256"] != self.manifest_sha:
            problems.append("manifest hash differs")
        if prov["stats_sha256"] != self.stats_sha:
            problems.append("stats hash differs")
        problems += arch_diff(ck["config"], self.cfg)
        for k in ("objective",):
            if prov.get(k) != getattr(self, k):
                problems.append(f"{k} differs")
        if prov.get("limit_train_samples") != self.limit_train:
            problems.append("limit_train_samples differs")
        if problems:
            raise SystemExit("refusing to resume (fail-fast, no silent migration): " + "; ".join(problems))
        self.model.load_state_dict(ck["model"])
        if self.teacher is not None:
            self.teacher.load_state_dict(ck["teacher"])
        self.centers.load_state_dict(ck["centers"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scaler and ck["scaler"]:
            self.scaler.load_state_dict(ck["scaler"])
        self.step, self.micro_consumed = ck["step"], ck["micro_consumed"]
        self.samples_seen, self.overflows, self.best = ck["samples_seen"], ck["overflows"], ck["best"]
        self._pending_rng = ck["rng"]
        self.resumed_from = {"path": str(path), "step": self.step,
                             "changed_statistical_batch": ck["world_size"] != self.world
                             or ck["config"]["train"]["batch_size_per_gpu"] != self.cfg["train"]["batch_size_per_gpu"]}

    # ----------------------------------------------------------------------------------------------- loop
    def _autocast(self):
        if self.amp_dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _micro(self, accum_index: int, accum: int, temps):
        batch = next(self.loader_it)
        self.micro_consumed += 1
        b = to_device(batch, self.device)
        sync_ctx = self.ddp_model.no_sync() if (D.is_dist() and accum_index < accum - 1) else _null()
        with sync_ctx:
            with self._autocast():
                t_out = teacher_forward(self.teacher, b, self.use_global)
                loss, logs, payload = self.ddp_model(b, t_out, self.centers.center, temps[0], temps[1], self.use_global)
            if payload is not None:
                self.centers.accumulate(payload["z"], payload["groups"], payload["weights"])
            if self.scaler:
                self.scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()
        logs.update(self._batch_stats(b))
        self.samples_seen += int(b["usable"].shape[0]) * self.world
        return logs

    def _batch_stats(self, b) -> dict:
        """Mask/closure/target statistics as device tensors (converted once per log interval)."""
        n_state = self.layout.n_state
        tv = b["token_valid"]
        tgt, ctx = b["target"], b["context"]
        f = lambda x: x.float()  # noqa: E731
        out = {"mask/raw_ratio": f(b["mask_units"].sum()) / f(tv[:, :n_state].sum()).clamp_min(1),
               "mask/visible_tokens": f(ctx.sum(1)).mean(), "mask/target_tokens": f(tgt.sum(1)).mean(),
               "mask/fallback_samples": f((b["mask_fallbacks"] > 0).sum()),
               "data/unusable_samples": f((~b["usable"]).sum()),
               "data/anchor_fallback_frames": f(b["anchor_fallback_frames"].sum()),
               "data/saturation_body": f(b["saturation_body"]).mean(),
               "data/saturation_local": f(b["saturation_local"]).mean()}
        if tv.shape[1] > n_state:
            out["mask/closure_induced_ratio"] = f(tgt[:, n_state:].sum()) / f(tv[:, n_state:].sum()).clamp_min(1)
        per_group = torch.zeros(N_GROUPS, device=tgt.device).index_add(
            0, self.model.layout_group, f(tgt.sum(0)))
        for g in self.layout.active_groups():
            out[f"targets/group_{g}"] = per_group[g]
        return out

    def train(self):
        cfg, tr, ev = self.cfg, self.cfg["train"], self.cfg["eval"]
        max_steps = tr["max_steps"]
        stop_at = max_steps if self.profile_steps is None else min(max_steps, self.step + self.profile_steps)
        if D.is_main():
            self.run_dir.mkdir(parents=True, exist_ok=True)
            dump_resolved(cfg, self.run_dir / "config_resolved.yaml")
            write_json(self.run_dir / "provenance.json", self.provenance())
            if self.global_note:
                print("note:", self.global_note, flush=True)
        self.model.train()
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        acc = defaultdict(list)
        t_last, step_times, last_log_samples = time.time(), [], self.samples_seen
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        diag = self._diag_batch() if D.is_main() else None
        while self.step < stop_at:
            t0 = time.time()
            lr = warmup_cosine(self.step, max_steps, tr["lr"], tr["min_lr"], tr["warmup_steps"])
            set_lr(self.optimizer, lr)
            temps = (cfg["loss"]["student_temperature"],
                     teacher_temperature(self.step, cfg["loss"]["teacher_temperature_start"],
                                         cfg["loss"]["teacher_temperature_final"],
                                         cfg["loss"]["teacher_temperature_warmup_steps"]))
            self.optimizer.zero_grad(set_to_none=True)
            self.centers.reset_accumulators()
            for a in range(tr["accum_steps"]):
                for k, v in self._micro(a, tr["accum_steps"], temps).items():
                    acc[k].append(v)
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, tr["grad_clip_norm"])
            ok = bool(torch.isfinite(grad_norm))
            if self.scaler:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                ok = ok and self.scaler.get_scale() >= scale_before
            elif ok:
                self.optimizer.step()
            if not ok:
                self.overflows += 1
                self.centers.reset_accumulators()
                acc["overflow"].append(1.0)
                continue
            m = ema_momentum(self.step, max_steps, tr["ema_start"], tr["ema_end"], tr["ema_schedule"])
            if self.teacher is not None:
                ema_update(ema_pairs(self.teacher, self.model), m)
            self.centers.step()
            self.step += 1
            step_times.append(time.time() - t0)
            acc["grad_norm"].append(grad_norm.detach())
            if D.is_main() and (self.step % tr["log_every_steps"] == 0 or self.step == stop_at):
                dt = time.time() - t_last
                row = {"step": self.step, "time": now(), "lr": lr, "ema_momentum": m, "teacher_temp": temps[1],
                       "samples_seen": self.samples_seen, "micro_consumed": self.micro_consumed,
                       "overflows": self.overflows,
                       "samples_per_s": (self.samples_seen - last_log_samples) / max(dt, 1e-9),
                       "peak_vram_gb": (torch.cuda.max_memory_allocated(self.device) / 2**30
                                        if self.device.type == "cuda" else 0.0)}
                row.update(_reduce_logs(acc))
                if diag is not None:
                    row.update({f"diag/{k}": v for k, v in self._diagnostics(diag).items()})
                append_jsonl(self.run_dir / "metrics.jsonl", row)
                print(f"step {self.step} loss {row.get('loss/total') or float('nan'):.4f} lr {lr:.2e} "
                      f"{row['samples_per_s']:.0f} samples/s vram {row['peak_vram_gb']:.1f}GB", flush=True)
                acc.clear()
                t_last, last_log_samples = time.time(), self.samples_seen
            # validate first so checkpoints written at the same step already contain this step's `best`
            if ev["frozen_every_steps"] and (self.step % ev["frozen_every_steps"] == 0 or self.step == max_steps):
                tv0 = time.time()
                self.validate()
                t_last += time.time() - tv0  # keep validation time out of the training throughput
            if tr["checkpoint_every_steps"] and self.step % tr["checkpoint_every_steps"] == 0:
                self.save(f"step_{self.step:07d}.pt")
                self.save("last.pt")
            if tr["milestone_every_steps"] and self.step % tr["milestone_every_steps"] == 0:
                self.save(f"milestone_{self.step:07d}.pt")
                if D.is_main():
                    prune_step_checkpoints(self.ckpt_dir, tr["keep_step_checkpoints"])
        self.save("last.pt")
        if D.is_main() and self.profile_steps is not None:
            warm = min(tr["profile_warmup_steps"], max(len(step_times) - 1, 0))
            steady = step_times[warm:]
            prof = {"steps_measured": len(steady), "warmup_steps_excluded": warm,
                    "mean_step_s": float(np.mean(steady)) if steady else None,
                    "samples_per_s": (tr["batch_size_per_gpu"] * tr["accum_steps"] * self.world / np.mean(steady))
                    if steady else None,
                    "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0,
                    "note": "throughput benchmark only; not a final checkpoint"}
            write_json(self.run_dir / "reports" / "profile.json", prof)
            print("profile:", prof, flush=True)
        return self.step

    # ----------------------------------------------------------------------------------------------- eval
    def encoder_for_eval(self):
        if self.cfg["eval"]["encoder"] == "teacher":
            if self.teacher is None:
                raise ValueError("eval.encoder=teacher but this objective has no EMA teacher; use online")
            return self.teacher.backbone
        return self.model.online

    def _diag_batch(self):
        n = self.cfg["eval"]["diagnostic_batch_size"]
        if self.synthetic:
            src = SyntheticSource(min(n, 64), SYNTH_DIAG)
        elif self.val is not None:
            rows = self.val[1][:n]
            src = PoseStoreSource(rows, self.processed_root)
        else:
            return None
        ds = LabeledDataset(src, self.builder, None, "eval")
        items = [ds[i] for i in range(len(ds))]
        return torch.utils.data.default_collate(items)

    @torch.no_grad()
    def _diagnostics(self, diag) -> dict:
        enc = self.encoder_for_eval()
        b = to_device(diag, self.device)
        with self._autocast():
            f, _ = enc.embed_clip(b)
        x = f["state_part_concat1024_l2"].float()[b["usable"]]
        return embedding_diagnostics(x.cpu().numpy())

    def _eval_sets(self):
        if self.synthetic:
            g_src, v_src = self.gallery[0], self.val[0]
            return g_src, [r["class_id"] for r in g_src.rows], v_src, [r["class_id"] for r in v_src.rows]
        return (self.gallery[0], [r["class_id"] for r in self.gallery[1]], self.val[0],
                [r["class_id"] for r in self.val[1]])

    @torch.no_grad()
    def validate(self):
        if not D.is_main():
            D.barrier()
            return
        ev, d = self.cfg["eval"], self.cfg["data"]
        g_src, g_lab, v_src, v_lab = self._eval_sets()
        enc = self.encoder_for_eval()
        nw = min(d["num_workers"], 8)
        t0 = time.time()
        g = extract_embeddings(enc, LabeledDataset(g_src, self.builder, g_lab, "eval"), ev["extract_batch_size"], nw,
                               self.device, self.precision_used)
        q = extract_embeddings(enc, LabeledDataset(v_src, self.builder, v_lab, "eval"), ev["extract_batch_size"], nw,
                               self.device, self.precision_used)
        res, ranks, _ = evaluate_retrieval(q["state_part_concat1024_l2"], q["class_ids"], g["state_part_concat1024_l2"],
                                           g["class_ids"], self.n_classes, query_chunk=ev["query_chunk"],
                                           gallery_chunk=ev["gallery_chunk"], device=self.device,
                                           query_usable=q["usable"], gallery_usable=g["usable"])
        bal, _, _ = evaluate_retrieval(q["state_balanced_mean256_l2"], q["class_ids"], g["state_balanced_mean256_l2"],
                                       g["class_ids"], self.n_classes, query_chunk=ev["query_chunk"],
                                       gallery_chunk=ev["gallery_chunk"], device=self.device,
                                       gallery_usable=g["usable"])
        full = res["full"]
        row = {"step": self.step, "time": now(), "kind": "frozen_val", "synthetic": self.synthetic,
               "encoder": ev["encoder"], "seconds": time.time() - t0,
               **{f"val/frozen_gloss_max/{k}": v for k, v in full.items() if not k.endswith("_pct")},
               "val/frozen_gloss_max/coverage": res.get("coverage"),
               "val/frozen_gloss_max_balanced256/r1": bal["full"]["r1"],
               **{f"val/diag/{k}": v for k, v in embedding_diagnostics(q["state_part_concat1024_l2"][:2048]).items()}}
        append_jsonl(self.run_dir / "metrics.jsonl", row)
        print(f"[val step {self.step}] R@1 {full['r1']:.4f} R@5 {full['r5']:.4f} DCG {full['dcg']:.4f} "
              f"coverage {res.get('coverage')}", flush=True)
        key = (full["r1"], full["dcg"])
        if self.best is None or key > (self.best["r1"], self.best["dcg"]):  # strict: earlier checkpoint wins ties
            self.best = {"step": self.step, "r1": full["r1"], "dcg": full["dcg"],
                         "metric": ev["selection_metric"], "selection_uses_val_labels": True}
            self.save("best_val.pt")
            write_json(self.run_dir / "reports" / "best_val.json", {**self.best, "metrics": res})
        self.model.train()
        D.barrier()


def _reduce_logs(acc: dict) -> dict:
    """Mean over the interval, ignoring NaN entries (= no targets for that family/part in a batch)."""
    out = {}
    for k, vals in acc.items():
        t = torch.stack([torch.as_tensor(v, dtype=torch.float32).detach().cpu() for v in vals])
        out[k] = float(torch.nanmean(t)) if not torch.isnan(t).all() else None
    return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
