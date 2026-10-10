"""Resolve actual upstream commits, downloaded byte hashes and runtime evidence."""
import json
import os
import subprocess
import urllib.request
from datetime import datetime,timezone
from pathlib import Path
import yaml
from .utils import sha256,atomic_json


def download(url,path,max_bytes):
    path = Path(path)
    if path.exists(): return 0
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.download')
    total = 0
    try:
        with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as file:
            while block:=response.read(1024*1024):
                total += len(block)
                if total>max_bytes: raise ValueError('Download budget exceeded')
                file.write(block)
        os.replace(temp,path)
    finally:
        if temp.exists(): temp.unlink()
    return total


def resolve_sources(config_path,output,device='cpu'):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if set(cfg) != {'schema_version','teacher','perception','max_download_gb'}:
        raise ValueError('Unknown/missing source config keys')
    teacher = cfg['teacher']
    repo = Path(teacher['repo_dir'])
    if not repo.exists():
        source = teacher.get('local_clone_source') or teacher['repo_url']
        subprocess.run(['git','clone','--no-hardlinks',source,str(repo)],check=True)
        subprocess.run(['git','remote','set-url','origin',teacher['repo_url']],cwd=repo,check=True)
        subprocess.run(['git','checkout',teacher['commit_sha']],cwd=repo,check=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if commit != teacher['commit_sha']: raise ValueError('Source commit differs from explicit pin')
    max_bytes = int(cfg['max_download_gb']*1e9)
    downloads = download(teacher['checkpoint_url'],teacher['checkpoint_path'],max_bytes)
    record = {k:v for k,v in teacher.items() if k!='local_clone_source'}
    record.update(checkpoint_sha256=sha256(teacher['checkpoint_path']),checkpoint_bytes=Path(teacher['checkpoint_path']).stat().st_size,
                  repo_dir=str(repo.resolve()),checkpoint_path=str(Path(teacher['checkpoint_path']).resolve()),runtime_verified=False)
    perception = {}
    for name,asset in cfg['perception'].items():
        downloads += download(asset['url'],asset['path'],max_bytes-downloads)
        perception[name] = {**asset,'path':str(Path(asset['path']).resolve()),'sha256':sha256(asset['path']),
                            'bytes':Path(asset['path']).stat().st_size,'runtime_verified':False}
    lock = {'schema_version':1,'verified_at_utc':datetime.now(timezone.utc).isoformat(),'status':'resolved_not_runtime_verified',
            'teacher':record,'perception':perception,'downloaded_bytes_this_invocation':downloads,
            'licenses':[{'asset':'dinov2 code','license':'Apache-2.0','url':teacher['repo_url']+'/blob/'+commit+'/LICENSE'},
                        {'asset':'DINOv2 model','license':'Apache-2.0','url':teacher['repo_url']+'/blob/'+commit+'/MODEL_CARD.md'},
                        {'asset':'MediaPipe','license':'Apache-2.0','url':'https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE'}]}
    atomic_json(output,lock)
    from .models.teacher import FrozenTeacher
    import torch
    torch.set_num_threads(4)
    model = FrozenTeacher(lock,device)
    crops = torch.linspace(-1,1,2*3*224*224).reshape(2,3,224,224)
    first,second = model(crops),model(crops)
    if not torch.allclose(first,second,atol=1e-6,rtol=1e-5) or first.requires_grad:
        raise ValueError('Teacher repeatability/gradient check failed')
    lock['teacher'].update(runtime_verified=True,output_shape=list(first.shape),repeat_max_error=float((first-second).abs().max()),
                           normalization='ImageNet RGB; 224 square bicubic clean crops; L2 float32 then float16 cache',
                           device=str(device),backend='upstream supported torch attention fallback or installed xFormers')
    import mediapipe as mp
    vision = mp.tasks.vision
    for name,kind in [('hand',vision.HandLandmarker),('pose',vision.PoseLandmarker)]:
        options = (vision.HandLandmarkerOptions if name=='hand' else vision.PoseLandmarkerOptions)(
            base_options=mp.tasks.BaseOptions(model_asset_path=perception[name]['path']),
            running_mode=vision.RunningMode.VIDEO)
        with kind.create_from_options(options) as detector:
            import numpy as np
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.zeros((224,224,3),np.uint8)),0)
        lock['perception'][name]['runtime_verified'] = True
    lock['mediapipe_version'] = mp.__version__
    lock['status'] = 'VERIFIED_RUNTIME'
    atomic_json(output,lock)
    return lock
