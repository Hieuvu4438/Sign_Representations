"""Atomic local checkpoints with config/data identity and rank-local RNG replay."""
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import torch.distributed as dist

from ..cache.hashing import canonical_hash
from .distributed import rank, world_size


def capture_rng(mask_generator=None, data_state=None):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "mask": mask_generator.bit_generator.state if mask_generator is not None else None,
             "data": data_state}
    return state


def restore_rng(state, mask_generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the recorded CUDA device topology")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if state["mask"] is not None:
        if mask_generator is None:
            raise ValueError("Checkpoint mask RNG requires an owned generator")
        mask_generator.bit_generator.state = state["mask"]


def save_checkpoint(path, trainer, config, identities, *, mask_generator=None, data_state=None):
    local_rng = capture_rng(mask_generator, data_state)
    states = [None] * world_size()
    if world_size() > 1:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    error = None
    if rank() == 0:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {"schema_version": 1, "config": config, "config_hash": canonical_hash(config),
                "identities": identities, "world_size": world_size(), "trainer": trainer.state_dict(), "rank_states": states}
            with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    torch.save(checkpoint, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        except Exception as caught:
            error = str(caught)
    if world_size() > 1:
        message = [error]
        dist.broadcast_object_list(message, src=0)
        error = message[0]
    if error:
        raise OSError(f"Checkpoint publication failed: {error}")


def load_checkpoint(path, trainer, config, identities, *, mask_generator=None):
    # Optimizer/RNG state contains trusted local Python objects. Never load untrusted files.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["schema_version"] != 1 or value["config_hash"] != canonical_hash(config) or value["config"] != config:
        raise ValueError("Resume config identity mismatch")
    if value["identities"] != identities or value["world_size"] != world_size():
        raise ValueError("Resume data/protocol/topology identity mismatch")
    trainer.load_state_dict(value["trainer"])
    local = value["rank_states"][rank()]
    restore_rng(local, mask_generator)
    return local["data"]
