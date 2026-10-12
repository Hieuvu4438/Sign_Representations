from contextlib import nullcontext
from dataclasses import dataclass
import time
import torch
from torch.nn.parallel import DistributedDataParallel

from .distributed import all_true, world_size
from .ema import cosine_momentum, update_teacher
from .schedules import SuccessfulStepScheduler


@dataclass(frozen=True)
class UpdateResult:
    succeeded: bool
    reason: str | None
    successful_step: int
    attempted_batches: int
    loss: float
    logs: tuple[dict, ...]
    gradient_norm: float | None
    ema_momentum: float | None


def choose_precision(requested, device):
    if requested == "bf16_if_supported_else_fp32":
        supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
        return ("bf16", None) if supported else ("fp32", "bf16 auto profile unavailable on this device")
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 GradScaler runtime requires CUDA")
    if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("Explicit bf16 precision is unavailable on this CUDA device")
    return requested, None


class Trainer:
    def __init__(self, model, config, device="cpu"):
        self.config, self.device = config, torch.device(device)
        self.core = model.to(self.device)
        self.precision, self.precision_reason = choose_precision(config["training"]["precision"], self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")
        self.model = self.core
        if world_size() > 1:
            self.model = DistributedDataParallel(self.core,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False, find_unused_parameters=config["training"]["ddp_find_unused_parameters"])
        train = config["training"]
        decay, no_decay = [], []
        for name, parameter in self.core.named_parameters():
            if parameter.requires_grad:
                (no_decay if not train["decay_bias_norm"] and (parameter.ndim <= 1 or name.endswith(".bias")) else decay).append(parameter)
        groups = [{"params": decay, "weight_decay": train["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
        self.optimizer = torch.optim.AdamW(groups, lr=train["learning_rate"], betas=tuple(train["betas"]))
        self.scheduler = SuccessfulStepScheduler(self.optimizer, train["max_successful_steps"], train["learning_rate"],
            train["final_learning_rate"], train["warmup_fraction"])
        self.successful_step, self.attempted_batches, self.attempted_updates = 0, 0, 0
        self.elapsed_runtime = 0.0

    def autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(self.device.type, dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16)

    def update(self, microbatches):
        if len(microbatches) != self.config["training"]["gradient_accumulation"]:
            raise ValueError("Update must provide the configured number of microbatches")
        if self.successful_step >= self.config["training"]["max_successful_steps"]:
            raise ValueError("Successful update budget already exhausted")
        if not all_true(self.elapsed_runtime < self.config["runtime"]["max_runtime_hours"] * 3600, self.device):
            raise TimeoutError("Recorded training runtime budget exhausted")
        started = time.monotonic()
        self.attempted_updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        logs, loss_value, finite_loss, all_targets = [], 0.0, True, True
        for index, batch in enumerate(microbatches):
            if self.config["data"]["pin_memory"] and self.device.type == "cuda":
                batch = batch.pin_memory()
            batch = batch.to(self.device)
            self.attempted_batches += 1
            sync = self.model.no_sync() if world_size() > 1 and index < len(microbatches) - 1 else nullcontext()
            with sync:
                with self.autocast():
                    output = self.model(batch)
                    loss = output.total / len(microbatches)
                finite_loss = finite_loss and bool(torch.isfinite(loss.detach()).all())
                all_targets = all_targets and output.global_semantic_count > 0
                # All ranks always backward, including empty/invalid attempts.
                self.scaler.scale(loss).backward()
            values = output.log_values()
            logs.append(values)
            weights = self.config["loss"]
            loss_value += sum(weights[name + "_weight"] * values[name]["mean"] for name in ("semantic", "geometry", "cluster")) / len(microbatches)
        self.scaler.unscale_(self.optimizer)
        parameters = [p for p in self.core.parameters() if p.requires_grad]
        finite_grad = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
        finite = all_true(finite_loss and finite_grad, self.device)
        have_targets = all_true(all_targets, self.device)
        norm, momentum, reason = None, None, None
        if finite and have_targets:
            measured_norm = torch.nn.utils.clip_grad_norm_(parameters, self.config["training"]["grad_clip_norm"])
            norm = float(measured_norm)
            finite = all_true(torch.isfinite(measured_norm), self.device)
        if not finite:
            reason = "nonfinite_loss_or_gradient"
            if self.scaler.is_enabled():
                # Explicit global scale backoff keeps ranks with local finite gradients aligned.
                self.scaler.update(new_scale=self.scaler.get_scale() * self.scaler.get_backoff_factor())
        elif not have_targets:
            reason = "empty_global_semantic_targets"
            if self.scaler.is_enabled():
                self.scaler.update(new_scale=self.scaler.get_scale())
        else:
            # GradScaler sees only globally finite, unscaled and clipped gradients here.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.successful_step += 1
            self.scheduler.step()
            teacher = self.config["model"]["teacher"]
            momentum = cosine_momentum(self.successful_step, self.config["training"]["max_successful_steps"], teacher["tau_start"], teacher["tau_end"])
            update_teacher(self.core.online, self.core.teacher, momentum)
        self.elapsed_runtime += time.monotonic() - started
        return UpdateResult(reason is None, reason, self.successful_step, self.attempted_batches, loss_value,
                            tuple(logs), norm, momentum)

    def state_dict(self):
        return {"model": self.core.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                "successful_step": self.successful_step, "attempted_batches": self.attempted_batches,
                "attempted_updates": self.attempted_updates, "elapsed_runtime": self.elapsed_runtime,
                "precision": self.precision}

    def load_state_dict(self, state):
        if state["precision"] != self.precision:
            raise ValueError("Resume precision differs from checkpoint; use a new run")
        if state["scheduler"]["successful_steps"] != state["successful_step"]:
            raise ValueError("Checkpoint scheduler/optimizer successful-step counters disagree")
        self.core.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        for key in ("successful_step", "attempted_batches", "attempted_updates", "elapsed_runtime"):
            setattr(self, key, state[key])
        self.core.teacher.requires_grad_(False).eval()
