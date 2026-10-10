"""Collective boundaries for equal-sized, optimizer-boundary DDP training."""
from contextlib import nullcontext
from datetime import timedelta
import os
import threading

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class TrainingContext:
    def __init__(self, requested_device):
        self.owns_group = False
        self.device = torch.device(requested_device)
        requested_world = int(os.environ.get("WORLD_SIZE", "1"))
        if self.device.type == "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if requested_world > 1 or dist.is_initialized():
                self.device = torch.device("cuda", local_rank)
            elif self.device.index is None:
                self.device = torch.device("cuda", 0)
            torch.cuda.set_device(self.device)
        if requested_world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                    timeout=timedelta(minutes=5))
            self.owns_group = True
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def all_objects(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def checked(self, stage, function):
        """Call only functions with no collectives inside this boundary."""
        value, error = None, None
        try:
            value = function()
        except Exception as exc:
            error = exc
        self.check_error(stage, error)
        return value

    def check_error(self, stage, error=None):
        if self.world == 1:
            if error is not None:
                raise error
            return
        failed = torch.tensor(int(error is not None), device=self.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            messages = self.all_objects(None if error is None else f"{type(error).__name__}: {error}")
            detail = "; ".join(f"rank {i}: {message}" for i, message in enumerate(messages) if message)
            raise RuntimeError(f"collective training failure at {stage}: {detail}") from error

    def rank_zero(self, stage, function):
        return self.checked(stage, function if self.rank == 0 else lambda: None)

    def rank_zero_long(self, stage, function, heartbeat_seconds=5.):
        """Keep collectives responsive while rank0 performs long frozen evaluation."""
        if self.world==1:
            return self.checked(stage,function)
        done=threading.Event();result={}
        if self.rank==0:
            def work():
                try:result['value']=function()
                except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
                finally:done.set()
            threading.Thread(target=work,daemon=True,name='signcoord_'+stage).start()
        while True:
            message=[None]
            if self.rank==0:
                done.wait(heartbeat_seconds)
                message[0]={'done':done.is_set(),**(result if done.is_set() else {})}
            dist.broadcast_object_list(message,src=0)
            if message[0]['done']:
                if 'error' in message[0]:
                    raise RuntimeError(f'collective training failure at {stage}: rank 0: {message[0]["error"]}')
                return message[0]['value']

    def wrap(self, model):
        if self.world == 1:
            return model
        return DistributedDataParallel(model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False)

    def accumulation_context(self, model, last_microstep):
        if self.world > 1 and not last_microstep:
            return model.no_sync()
        return nullcontext()

    def close(self):
        if self.owns_group:
            dist.destroy_process_group()


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def check_prediction_finite(outputs, context):
    error = None
    if any(not torch.isfinite(value).all() for value in outputs.values()):
        error = ValueError("nonfinite model prediction before loss collectives")
    if context is None:
        if error is not None:
            raise error
    else:
        context.check_error("model_prediction", error)
