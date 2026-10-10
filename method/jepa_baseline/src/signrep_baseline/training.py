"""Simultaneous SignRep/adversary updates, atomic complete-state resume and selection."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .schema import FidelityChoices
from .discriminator import (StyleDiscriminator, MatchEMA, gram_style, style_pairs,
                             discriminator_loss, frozen_discriminator)
from .losses import pretraining_loss
from .optim import pretraining_optimizers
from .cache import implementation_sha256


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("Cannot exactly resume CUDA run without CUDA")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Pretrainer:
    def __init__(self, model, choices=None, device="cpu", total_steps=500000, warmup_steps=50000,
                 max_lr=1e-4, amp=False, amp_dtype="float16"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.choices = choices or FidelityChoices()
        self.discriminator = StyleDiscriminator().to(self.device)
        self.ema = MatchEMA(.1).to(self.device)
        (self.optimizer, self.d_optimizer, self.scheduler, self.d_scheduler) = pretraining_optimizers(
            self.model, self.discriminator, max_lr, warmup_steps, total_steps)
        self.amp = bool(amp)
        if amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("AMP dtype must be float16 or bfloat16")
        self.amp_dtype = amp_dtype
        if self.amp and self.device.type != "cuda":
            raise ValueError("AMP recipe supports CUDA only")
        if self.amp and amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support BF16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and amp_dtype == "float16")
        self.step, self.best_validation = 0, -float("inf")

    def update(self, batch):
        self.model.train()
        self.discriminator.train()
        images = batch["images"].to(self.device)
        targets = {k: v.to(self.device) for k, v in batch["targets"].items()}
        masks = {k: v.to(self.device) for k, v in batch["masks"].items()}
        matched_idx, unmatched_idx = style_pairs(batch["video_ids"], device=self.device)
        self.scheduler.before_update(self.step + 1)
        self.d_scheduler.before_update(self.step + 1)
        self.optimizer.zero_grad(set_to_none=True)
        self.d_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=getattr(torch, self.amp_dtype), enabled=self.amp):
            output = self.model(images)
        features = output["features"].float()
        style = gram_style(output["tokens"], self.choices.gram_normalization)
        positive, negative = self.discriminator.paired_logits(features.detach(), style.detach(), matched_idx, unmatched_idx)
        d_loss = discriminator_loss(positive, negative)
        if not torch.isfinite(d_loss):
            raise FloatingPointError("Nonfinite discriminator loss")
        d_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.discriminator.parameters()):
            raise FloatingPointError("Nonfinite discriminator gradients")
        self.d_optimizer.step()
        self.d_optimizer.zero_grad(set_to_none=True)
        with frozen_discriminator(self.discriminator):
            positive, negative = self.discriminator.paired_logits(features, style.detach(), matched_idx, unmatched_idx)
            positive, negative = positive.sigmoid(), negative.sigmoid()
            self.ema.update(positive, negative)
            predicted = {**{k: output["cls_head"][k] for k in targets if k != "activity"},
                         "activity": output["cls_active"]}
            loss, components = pretraining_loss(predicted, targets, masks, features, positive, negative, self.ema, self.choices)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite encoder loss")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.step += 1
        self.scheduler.after_update()
        self.d_scheduler.after_update()
        metrics = {k: float(value.detach()) for k, value in components.items()}
        metrics.update(step=self.step, loss=float(loss.detach()), discriminator=float(d_loss.detach()),
                       gradient_norm=float(norm), lr=self.scheduler.rate(self.step),
                       ema_matched=float(self.ema.matched), ema_unmatched=float(self.ema.unmatched))
        return metrics

    def save(self, path, sampler, run_contract):
        # choices and recipe are checked even if callers omit them from a config.
        contract = {**run_contract, "fidelity_choices": asdict(self.choices), "amp": self.amp, "amp_dtype": self.amp_dtype,
                    "mask_ratio": self.model.mask_ratio, "implementation_sha256": implementation_sha256()}
        atomic_torch_save({"schema": "signrep_training_v1", "model": self.model.state_dict(),
                           "discriminator": self.discriminator.state_dict(), "ema": self.ema.state_dict(),
                           "optimizer": self.optimizer.state_dict(), "d_optimizer": self.d_optimizer.state_dict(),
                           "scheduler": self.scheduler.state_dict(), "d_scheduler": self.d_scheduler.state_dict(),
                           "scaler": self.scaler.state_dict(), "step": self.step,
                           "best_validation": self.best_validation, "sampler": sampler.state_dict(),
                           "random_state": random_state(), "run_contract": contract}, path)
        atomic_torch_save({"model": self.model.state_dict(), "step": self.step,
                           "run_contract": contract}, Path(path).with_suffix(".model.pt"))

    def resume(self, path, sampler, run_contract):
        # Complete local training state includes NumPy/Python RNG (trusted local file).
        state = torch.load(path, map_location="cpu", weights_only=False)
        expected = {**run_contract, "fidelity_choices": asdict(self.choices), "amp": self.amp, "amp_dtype": self.amp_dtype,
                    "mask_ratio": self.model.mask_ratio, "implementation_sha256": implementation_sha256()}
        if state.get("schema") != "signrep_training_v1" or state["run_contract"] != expected:
            raise ValueError("Run/data/fidelity contract changed on resume")
        for key in ["model", "discriminator", "ema"]:
            getattr(self, key).load_state_dict(state[key], strict=True)
        for key in ["optimizer", "d_optimizer", "scheduler", "d_scheduler", "scaler"]:
            getattr(self, key).load_state_dict(state[key])
        self.step, self.best_validation = state["step"], state["best_validation"]
        if self.scheduler.completed_steps != self.step or self.d_scheduler.completed_steps != self.step:
            raise ValueError("Partial-step checkpoint")
        sampler.load_state_dict(state["sampler"])
        restore_random_state(state["random_state"])

    def select_validation(self, result, path, sampler, run_contract, interval=25000):
        if result.get("split") != "val" or result.get("task") != "dictionary_retrieval":
            raise ValueError("Pretraining selection requires validation dictionary retrieval")
        if self.step == 0 or self.step % interval:
            raise ValueError("Checkpoint selection outside configured validation interval")
        score = result["per_instance_top1"]
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation retrieval score")
        if score > self.best_validation:
            self.best_validation = score
            self.save(path, sampler, run_contract)
            return True
        return False
