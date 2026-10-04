"""Resource checks are observations, never job cancellation or GPU allocation."""
from pathlib import Path
import shutil
import subprocess
import uuid


def normalized_uuid(value):
    text = str(value)
    if text.lower().startswith('gpu-'): text = text[4:]
    return str(uuid.UUID(text))


def reserve_check(total, free, planned, reserve=.20):
    if total <= 0 or not 0 <= free <= total or planned < 0:
        raise ValueError('Invalid storage observation')
    return free - planned >= total * reserve


def scratch_preflight(directory, planned_bytes, limit_bytes=1 << 30):
    path = Path(directory)
    if path.resolve() != Path('/dev/shm') or not 0 < planned_bytes <= limit_bytes:
        raise ValueError('Only bounded /dev/shm scratch is planned')
    usage = shutil.disk_usage(path)
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable'): memory[name] = int(value.split()[0]) * 1024
    # Reserve additional 2 GiB CPU decoding/preprocessing workspace.
    if (not reserve_check(usage.total, usage.free, planned_bytes)
            or not reserve_check(memory['MemTotal'], memory['MemAvailable'], planned_bytes + (2 << 30))):
        raise RuntimeError('Tmpfs/RAM reserve failed')
    return dict(path=str(path), total_bytes=usage.total, free_bytes=usage.free,
                planned_bytes=planned_bytes, cache_limit_bytes=limit_bytes, reserve_fraction=.20,
                memory_bytes=memory, cpu_workspace_reserve_bytes=2 << 30)


def gpu_preflight(min_free_mib=8192, max_utilization=10):
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
               '--format=csv,noheader,nounits']
    rows = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout.splitlines()
    parsed = [[field.strip() for field in row.split(',')] for row in rows]
    selected = [row for row in parsed if row[0] == '0']
    if len(selected) != 1: raise RuntimeError('Granted GPU0 not found')
    index, uuid, total, free, utilization = selected[0]
    result = dict(physical_index=int(index), uuid=uuid, total_mib=int(total), free_mib=int(free),
                  utilization_percent=int(utilization), min_free_mib=min_free_mib,
                  max_utilization=max_utilization, command=command)
    if result['free_mib'] < min_free_mib or result['utilization_percent'] > max_utilization:
        raise RuntimeError(f'Granted GPU0 busy/insufficient VRAM: {result}')
    return result


def configure_gpu(observation):
    # Import torch only after physical GPU preflight; preserve scheduler visibility.
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('No visible CUDA device')
    matches = []
    for i in range(torch.cuda.device_count()):
        device_uuid = normalized_uuid(torch.cuda.get_device_properties(i).uuid)
        if device_uuid == normalized_uuid(observation['uuid']): matches.append(i)
    if len(matches) != 1: raise RuntimeError('Granted physical GPU0 not uniquely visible to scheduler')
    selected = matches[0]
    torch.cuda.set_device(selected)
    torch.cuda.set_per_process_memory_fraction(8192 / observation['total_mib'], selected)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return f'cuda:{selected}'
