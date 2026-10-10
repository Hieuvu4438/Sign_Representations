import copy
import importlib.metadata
from pathlib import Path
import shutil

from signcoord.engine.checkpoint import atomic_checkpoint,load_checkpoint
from signcoord.provenance import atomic_json,execution_stamp,sha256_file


def export_encoder(checkpoint,output):
    path=Path(output); path.mkdir(parents=True,exist_ok=False)
    saved=load_checkpoint(checkpoint)
    cfg=copy.deepcopy(saved["resolved_config"])
    # Keep only architecture/sampler fields; drop all training input paths/labels.
    encoder_config={"model":{k:cfg["model"][k] for k in ("backbone","input_frames","tiny_dim","vendor_root","generic_vendor_root","attention_kernel") if k in cfg["model"]},
                    "data":{"sampler":cfg["data"]["sampler"]}}
    state={k.removeprefix("encoder."):v for k,v in saved["model_state"].items() if k.startswith("encoder.")}
    source=Path(__file__).resolve().parents[1]
    # Ship only deployment modules; inference cannot import the training decoder,
    # projector, data/targets or the pose estimator from this directory.
    deployment_files=['__init__.py','provenance.py','export/inference.py',
                      'models/backbone_adapter.py','models/hiera_adapter.py',
                      'models/generic_hiera_adapter.py','models/attention.py']
    for relative in deployment_files:
        destination=path/'code/signcoord'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/relative,destination)
    backbone=cfg['model']['backbone']
    if backbone=='hiera_base_video_adapter':
        vendor=Path(cfg['model']['generic_vendor_root'])
        for file in sorted((vendor/'hiera').glob('*.py')):
            destination=path/'vendor/hiera/hiera'/file.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,destination)
        for name in ('LICENSE','LICENSE.models'):
            shutil.copy2(vendor/name,path/'vendor/hiera'/name)
        encoder_config['model'].pop('vendor_root',None)
        encoder_config['model']['generic_vendor_root']='vendor/hiera'
    elif backbone=='signrep_hiera_adapter':
        vendor=Path(cfg['model']['vendor_root'])
        for name in ('hiera_clstoken.py','hiera_utils.py','hfhub.py'):
            destination=path/'vendor/signrep/models/hiera'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(vendor/'models/hiera'/name,destination)
        shutil.copy2(vendor/'LICENSE',path/'vendor/signrep/LICENSE')
        encoder_config['model'].pop('generic_vendor_root',None)
        encoder_config['model']['vendor_root']='vendor/signrep'
    else:
        encoder_config['model'].pop('vendor_root',None)
        encoder_config['model'].pop('generic_vendor_root',None)
    versions={}
    for dependency in ('torch','numpy','timm','huggingface-hub'):
        if backbone=='tiny_fixture' and dependency in ('timm','huggingface-hub'):continue
        try:versions[dependency]=importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:versions[dependency]=None
    metadata={**execution_stamp(),"source_checkpoint_sha256":sha256_file(checkpoint),"rgb_only":True,
              "feature_layer":"final_stage_pre_classifier","pooling":"uniform_valid_native_temporal_cells",
              "offline_context":True,"rgb_mean":[.485,.456,.406],"rgb_std":[.229,.224,.225],
              "pose_pseudo_targets_used_during_training":saved.get("optimizer_update",0)>0,"torchscript_onnx":"not_attempted",
              "vendor_source_required":backbone in ("signrep_hiera_adapter","hiera_base_video_adapter"),
              'vendor_source_bundled':backbone in ('signrep_hiera_adapter','hiera_base_video_adapter'),
              'external_vendor_path_required':False,'deployment_dependencies':versions,
              'deployment_source_hashes':{str(p.relative_to(path)):sha256_file(p)
                  for top in ('code','vendor') for p in sorted((path/top).rglob('*')) if p.is_file()}}
    atomic_checkpoint(path/"encoder.pt",{"schema_version":"signcoord_encoder_bundle_v1",
                      "encoder_state":state,"encoder_config":encoder_config,"metadata":metadata})
    atomic_json(path/"model_card.json",metadata)
    atomic_json(path/'dependency_lock.json',{'observed_versions':versions,'python':__import__('sys').version,
                     'policy':'observed_environment_not_an_automatic_installer'})
    (path/"infer.py").write_text('''import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
import argparse
import numpy as np
import torch
from signcoord.export.inference import load_bundle, encode_rgb
p=argparse.ArgumentParser()
p.add_argument('--bundle',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
a=p.parse_args()
encoder,_=load_bundle(a.bundle)
with np.load(a.input,allow_pickle=False) as d:
    rgb=torch.from_numpy(d['rgb']); valid=torch.from_numpy(d['frame_valid']); times=torch.from_numpy(d['times'])
with torch.inference_mode(): out=encode_rgb(encoder,rgb,valid,times)
np.savez_compressed(a.output,**{k:v.numpy() for k,v in out.items()})
''')
    return {"status":"passed","bundle":str(path/"encoder.pt"),"metadata":metadata,
            "encoder_bundle_sha256":sha256_file(path/"encoder.pt")}
