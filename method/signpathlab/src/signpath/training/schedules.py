import math


class SuccessfulStepScheduler:
    """LR for the next successful update; failed attempts do not advance time."""
    def __init__(self, optimizer, total_steps, initial_lr, final_lr, warmup_fraction):
        self.optimizer, self.total_steps = optimizer, total_steps
        self.initial_lr, self.final_lr = initial_lr, final_lr
        self.warmup = int(total_steps * warmup_fraction)
        self.successful_steps = 0
        self._apply()

    def rate(self):
        step = self.successful_steps
        if self.warmup and step < self.warmup:
            return self.initial_lr * (step + 1) / self.warmup
        progress = min(1.0, (step - self.warmup) / max(1, self.total_steps - self.warmup))
        return self.final_lr + .5 * (self.initial_lr - self.final_lr) * (1 + math.cos(math.pi * progress))

    def _apply(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.rate()

    def step(self):
        self.successful_steps += 1
        self._apply()

    def state_dict(self):
        return {"successful_steps": self.successful_steps}

    def load_state_dict(self, state):
        self.successful_steps = state["successful_steps"]
        self._apply()
