import math
import torch


def freeze(module):
    module.requires_grad_(False)
    module.eval()


@torch.no_grad()
def update_ema(teacher, student, momentum):
    if not 0 <= momentum <= 1:
        raise ValueError("EMA momentum outside [0,1]")
    tp,sp = dict(teacher.named_parameters()),dict(student.named_parameters())
    if tp.keys() != sp.keys():
        raise AssertionError("Teacher/student parameter mismatch")
    for name,t in tp.items():
        if t.requires_grad or t.grad is not None:
            raise AssertionError("Teacher received gradients")
        t.mul_(momentum).add_(sp[name].detach(),alpha=1-momentum)
    tb,sb = dict(teacher.named_buffers()),dict(student.named_buffers())
    if tb.keys() != sb.keys():
        raise AssertionError("Teacher/student buffer mismatch")
    for name,t in tb.items():
        t.copy_(sb[name])


def ema_momentum(step, max_steps, start=0.996, end=1., schedule="cosine"):
    fraction = min(step/max(max_steps-1,1),1)
    if schedule == "cosine":
        fraction = (1-math.cos(math.pi*fraction))/2
    elif schedule != "linear":
        raise ValueError(schedule)
    return start+(end-start)*fraction
