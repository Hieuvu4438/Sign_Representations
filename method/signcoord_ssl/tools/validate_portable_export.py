"""Actual generic Hiera export/relocation parity in a deployment-only process."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from signcoord.engine.checkpoint import load_checkpoint
from signcoord.export.bundle import export_encoder
from signcoord.export.inference import encoder_from_config,encode_rgb
from signcoord.provenance import atomic_json,execution_stamp,sha256_file


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    saved=load_checkpoint(args.checkpoint)
    encoder=encoder_from_config(saved['resolved_config']).eval()
    encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['model_state'].items()
                            if k.startswith('encoder.')},strict=True)
    frames=saved['resolved_config']['model']['input_frames']
    rgb=torch.randn(1,3,frames,224,224,generator=torch.Generator().manual_seed(3407))
    valid=torch.ones(1,frames,dtype=torch.bool);times=torch.arange(frames)[None].double()/25
    valid[:,-3:]=False
    with torch.inference_mode():reference=encode_rgb(encoder,rgb,valid,times)
    np.savez(root/'input.npz',rgb=rgb.numpy(),frame_valid=valid.numpy(),times=times.numpy())
    np.savez(root/'reference_outputs.npz',**{k:v.numpy() for k,v in reference.items()})
    report=export_encoder(args.checkpoint,root/'initial_bundle')
    shutil.move(root/'initial_bundle',root/'relocated_bundle')
    moved=root/'relocated_bundle'
    code='''import sys,importlib.abc,runpy
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.startswith(('signcoord.engine','signcoord.targets','signcoord.data','signcoord.models.decoder','signcoord.models.projector','mmpose','mediapipe')): raise RuntimeError('forbidden '+fullname)
sys.meta_path.insert(0,Guard())
script=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(script,run_name='__main__')
'''
    env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-c',code,str(moved/'infer.py'),'--bundle',str(moved/'encoder.pt'),
                           '--input',str(root/'input.npz'),'--output',str(root/'outputs.npz')],
                           env=env,cwd=root,capture_output=True,text=True)
    (root/'deployment.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('portable deployment failed; inspect deployment.log')
    errors={}
    with np.load(root/'outputs.npz') as output:
        for key,value in reference.items():
            actual=output[key];expected=value.numpy()
            if expected.dtype==bool:
                assert np.array_equal(actual,expected),key
                errors[key]=0.
            else:
                np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
                errors[key]=float(np.max(np.abs(actual-expected)))
    evidence={**execution_stamp(),'status':'passed','checkpoint_sha256':sha256_file(args.checkpoint),
         'bundle':str(moved/'encoder.pt'),'bundle_sha256':sha256_file(moved/'encoder.pt'),
         'backbone':saved['resolved_config']['model']['backbone'],'context':frames,'precision':'fp32',
         'max_absolute_errors':errors,'fresh_process_with_deployment_only_code':True,
         'original_vendor_path_required':False,'forbidden_import_guard_passed':True,
         'scope':'export engineering parity; no downstream efficacy comparison',
         'metadata':report['metadata']}
    atomic_json(root/'portable_export_report.json',evidence)
    atomic_json(PROJECT/'docs/evidence/portable_export_report.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k!='metadata'}))


if __name__=='__main__':main()
