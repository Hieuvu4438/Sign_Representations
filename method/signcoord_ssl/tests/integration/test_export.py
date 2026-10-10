import json
import subprocess
import sys
import shutil
import numpy as np
from pathlib import Path

import torch

from signcoord.engine.checkpoint import atomic_checkpoint
from signcoord.engine.train import make_model
from signcoord.export.bundle import export_encoder
from signcoord.export.inference import encode_rgb,load_bundle
from signcoord.config import load_config


def test_export_parity_and_no_pose_imports(tmp_path):
    project=Path(__file__).resolve().parents[2]
    c=load_config(project/'configs/base.yaml')
    c['model'].update(backbone='tiny_fixture',initialization_mode='random_init')
    c['data']['sampler']['spatial_size']=[32,32]
    c['model']['decoder'].update(width=32,heads=4,blocks=1)
    model=make_model(c).eval()
    checkpoint=tmp_path/'model.pt'
    atomic_checkpoint(checkpoint,{'schema_version':'signcoord_checkpoint_v1','microstep_boundary':0,
                                 'model_state':model.state_dict(),'resolved_config':c})
    report=export_encoder(checkpoint,tmp_path/'export')
    encoder,metadata=load_bundle(report['bundle'])
    rgb=torch.randn(2,3,32,32,32); valid=torch.ones(2,32,dtype=torch.bool); times=torch.arange(32)[None].expand(2,-1)/25
    with torch.inference_mode():
        a=model.encode(rgb,valid,times); b=encode_rgb(encoder,rgb,valid,times)
    for key in a: assert torch.equal(a[key],b[key]),key
    loaded=torch.load(report['bundle'],weights_only=True)
    assert all(not k.startswith(('decoder.','projector.')) for k in loaded['encoder_state'])
    # Fresh process prohibits imports from training targets/data/pose libraries.
    code='''import sys,importlib.abc,torch
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.startswith(('signcoord.targets','signcoord.data','mmpose','mediapipe')): raise RuntimeError('forbidden '+fullname)
sys.meta_path.insert(0,Guard())
from signcoord.export.inference import load_bundle,encode_rgb
encoder,_=load_bundle(sys.argv[1])
with torch.inference_mode():
 out=encode_rgb(encoder,torch.randn(1,3,32,32,32),torch.ones(1,32,dtype=torch.bool),torch.arange(32)[None]/25)
assert out['clip'].shape == (1,32)
'''
    import os
    env=dict(os.environ,PYTHONPATH=str(project/'src'))
    subprocess.run([sys.executable,'-c',code,report['bundle']],env=env,check=True,capture_output=True)
    # Move the complete bundle, remove package PYTHONPATH, and use shipped code.
    moved=tmp_path/'relocated'
    shutil.move(tmp_path/'export',moved)
    input_file=tmp_path/'input.npz';output_file=tmp_path/'output.npz'
    np.savez(input_file,rgb=rgb.numpy(),frame_valid=valid.numpy(),times=times.numpy())
    portable_code='''import sys,importlib.abc,runpy
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.startswith(('signcoord.engine','signcoord.targets','signcoord.data','signcoord.models.decoder','signcoord.models.projector','mmpose','mediapipe')): raise RuntimeError('forbidden '+fullname)
sys.meta_path.insert(0,Guard())
script=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(script,run_name='__main__')
'''
    subprocess.run([sys.executable,'-c',portable_code,str(moved/'infer.py'),'--bundle',str(moved/'encoder.pt'),
                    '--input',str(input_file),'--output',str(output_file)],env=dict(os.environ,PYTHONPATH=''),
                    cwd=tmp_path,check=True,capture_output=True)
    with np.load(output_file) as result:
        for key in b:np.testing.assert_array_equal(result[key],b[key].numpy())
    loaded=torch.load(moved/'encoder.pt',weights_only=True)
    assert 'ssl_train_manifest' not in str(loaded['encoder_config'])
    assert not loaded['metadata']['external_vendor_path_required']
    from pytest import raises
    with (moved/'code/signcoord/models/backbone_adapter.py').open('a') as file:file.write('\n# mutation\n')
    with raises(ValueError,match='source hash mismatch'):load_bundle(moved/'encoder.pt')
