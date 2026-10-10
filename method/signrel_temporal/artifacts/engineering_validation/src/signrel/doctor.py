import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import torch
from .utils import atomic_json


def doctor(output):
    modules = {}
    for name in ('torch','numpy','yaml','pytest','av','PIL','mediapipe'):
        try:
            module = importlib.import_module(name)
            modules[name] = {'status':'available','version':getattr(module,'__version__','unknown'),'path':module.__file__}
        except Exception as exc:
            modules[name] = {'status':'missing_or_broken','error':str(exc)}
    disk = shutil.disk_usage(Path.cwd())
    x = torch.ones(2,2)
    report = {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'modules':modules,
              'disk_free_bytes':disk.free,'cpu_tensor_operation':bool(torch.equal(x@x,torch.full((2,2),2.))),
              'cuda_available':torch.cuda.is_available(),'torch_cuda_runtime':torch.version.cuda,
              'ffmpeg_path':shutil.which('ffmpeg'),'existing_environments_modified':False}
    if torch.cuda.is_available():
        report['gpu'] = [{'name':torch.cuda.get_device_name(i),'total_memory':torch.cuda.get_device_properties(i).total_memory,
                          'free_memory':torch.cuda.mem_get_info(i)[0]} for i in range(torch.cuda.device_count())]
        report['gpu_tensor_operation'] = bool(torch.equal(x.cuda()@x.cuda(),torch.full((2,2),2.,device='cuda')))
        report['nvidia_smi'] = subprocess.run(['nvidia-smi','--query-gpu=driver_version,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    if report['ffmpeg_path']:
        report['ffmpeg_version'] = subprocess.run(['ffmpeg','-version'],capture_output=True,text=True).stdout.splitlines()[0]
    atomic_json(output,report)
    return report
