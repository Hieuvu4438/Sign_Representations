"""Fine-tuning recipe, complete recognition state and validation-only selection."""

from pathlib import Path
import numpy as np
import torch
from .recognition import recognition_loss
from .optim import layerwise_parameter_groups, WarmupCosine
from .training import atomic_torch_save, random_state, restore_random_state
from .retrieval import ranking_metrics
from .cache import implementation_sha256


class RecognitionTrainer:
    def __init__(self, model, distribution, total_steps, warmup_steps=1000, device="cpu", amp=False):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.distribution = distribution.to(self.device)
        self.optimizer = torch.optim.AdamW(layerwise_parameter_groups(model), lr=1e-4, betas=(.9, .95))
        self.scheduler = WarmupCosine(self.optimizer, 1e-4, warmup_steps, total_steps)
        self.amp = amp
        if amp and self.device.type != "cuda":
            raise ValueError("Recognition AMP requires CUDA")
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.step, self.best_validation = 0, -float("inf")

    def update(self, images, labels):
        self.model.train()
        self.scheduler.before_update(self.step + 1)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            logits = self.model(images.to(self.device))
        loss, terms = recognition_loss(logits.float(), labels.to(self.device), self.distribution)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite recognition loss")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        self.scheduler.after_update()
        return {"loss": float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()},
                "step": self.step, "gradient_norm": float(norm), "lr": self.scheduler.rate(self.step)}

    @torch.inference_mode()
    def evaluate(self, dataset, split="val", batch_size=8):
        if split not in {"val", "test"} or any(row.get("split") != split for row in dataset.rows):
            raise ValueError("Recognition evaluation requires a homogeneous held-out split")
        training = self.model.training
        self.model.eval()
        scores, labels = [], []
        try:
            for start in range(0, len(dataset), batch_size):
                batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                logits = self.model(torch.stack([x[0] for x in batch]).to(self.device))
                scores.append(logits.cpu())
                labels.extend(int(x[1]) for x in batch)
        finally:
            self.model.train(training)
        ranking = torch.cat(scores).argsort(dim=1, descending=True, stable=True)
        return {**ranking_metrics(ranking, labels), "split": split, "task": "recognition"}

    def save(self, path, sampler, contract):
        atomic_torch_save({"schema": "signrep_recognition_training_v1", "model": self.model.state_dict(),
                           "distribution": self.distribution, "optimizer": self.optimizer.state_dict(),
                           "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                           "step": self.step, "best_validation": self.best_validation,
                           "sampler": sampler.state_dict(), "random_state": random_state(),
                           "contract": {**contract, "amp": self.amp, "implementation_sha256": implementation_sha256()}}, path)

    def resume(self, path, sampler, contract):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("schema") != "signrep_recognition_training_v1" or state["contract"] != {**contract, "amp": self.amp, "implementation_sha256": implementation_sha256()}:
            raise ValueError("Recognition run contract changed")
        if not torch.equal(state["distribution"].cpu(), self.distribution.cpu()):
            raise ValueError("Class distribution changed")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.step, self.best_validation = state["step"], state["best_validation"]
        if self.step != self.scheduler.completed_steps:
            raise ValueError("Partial recognition checkpoint")
        sampler.load_state_dict(state["sampler"])
        restore_random_state(state["random_state"])

    def select_validation(self, result, path, sampler, contract):
        if result.get("split") != "val" or result.get("task") != "recognition":
            raise ValueError("Recognition selection requires validation scores")
        score = result["per_instance_top1"]
        if not np.isfinite(score):
            raise ValueError("Nonfinite recognition selection score")
        if score > self.best_validation:
            self.best_validation = score
            self.save(path, sampler, contract)
            return True
        return False
