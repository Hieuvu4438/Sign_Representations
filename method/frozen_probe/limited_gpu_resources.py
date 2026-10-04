"""Prospective 2GiB torch allocator cap for fixed SignRep batch1 smoke.

User GPU0 grant persists. Does not claim OS VRAM isolation or cancel other jobs.
"""
from resources import normalized_uuid


def configure_gpu_2gib(observation):
    import torch
    if observation['free_mib'] < 3072:
        raise RuntimeError('Need2048MiB allocator plus1024MiB context/workspace margin')
    if not torch.cuda.is_available():
        raise RuntimeError('No visible CUDA device')
    matches = [i for i in range(torch.cuda.device_count())
               if normalized_uuid(torch.cuda.get_device_properties(i).uuid)==normalized_uuid(observation['uuid'])]
    if len(matches)!=1:
        raise RuntimeError('Granted physicalGPU0 not uniquely scheduler-visible')
    selected=matches[0]
    torch.cuda.set_device(selected)
    torch.cuda.set_per_process_memory_fraction(2048/observation['total_mib'],selected)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return f'cuda:{selected}'
