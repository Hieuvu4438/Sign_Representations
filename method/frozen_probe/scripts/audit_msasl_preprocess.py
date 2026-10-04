"""Native stored crop normalization parity with read-only upstream transform."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import albumentations
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import array_spec,verify_store
from resources import scratch_preflight


def main():
    p=argparse.ArgumentParser()
    for n in ('config','run-id','output-dir'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
        resource=scratch_preflight('/dev/shm',256<<20);root=Path(c['native_runtime']);store=verify_store(root)
        spec=importlib.util.spec_from_file_location('msasl_upstream_transform',c['upstream_transform'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        transform=module.Transformation();reports=[]
        means=np.asarray([.485,.456,.406],dtype=np.float64).reshape(1,1,1,3)
        stds=np.asarray([.229,.224,.225],dtype=np.float64).reshape(1,1,1,3)
        fixtures=[]
        for values in ([0,0,0],[255,255,255],[255,0,127]):
            rgb=np.broadcast_to(np.asarray(values,dtype=np.uint8),(224,224,3)).copy()
            actual=transform.aug_video([rgb]).numpy()
            expected=((rgb.astype(np.float64)/255.-means[0])/stds[0]).transpose(2,0,1)[None]
            delta=float(np.max(np.abs(actual-expected)));assert delta<=c['normalization_atol']
            fixtures.append(dict(RGB=values,max_abs_error=delta,passed=True))
        for item in c['samples']:
            sid=item['sample_id'];rgb=np.load(root/f'{sid}_first_native_window.npy',allow_pickle=False)
            assert rgb.shape==(16,224,224,3) and rgb.dtype==np.uint8
            actual=transform.aug_video(list(rgb)).numpy();assert actual.shape==(16,3,224,224) and actual.dtype==np.float32
            expected=((rgb.astype(np.float64)/255.-means)/stds).transpose(0,3,1,2)
            delta=float(np.max(np.abs(actual-expected)));assert np.isfinite(actual).all() and delta<=c['normalization_atol']
            identity_rgb=np.stack([cv2.resize(frame,(224,224),interpolation=cv2.INTER_LINEAR) for frame in rgb]);assert np.array_equal(identity_rgb,rgb)
            run.save_array(f'{sid}_first_normalized_frame.npy',actual[0],array_spec((3,224,224),'<f4'))
            reports.append(dict(sample_id=sid,scalar_values_checked=int(actual.size),shape=list(actual.shape),dtype=actual.dtype.str,
                max_abs_error_against_FP64_manual=delta,manual_atol=c['normalization_atol'],all_values_finite=True,
                upstream_repeat_bitwise=np.array_equal(actual,transform.aug_video(list(rgb)).numpy()),
                normalized_input_SHA=hashlib.sha256(actual.tobytes()).hexdigest(),resize_identity_exact=True))
        assert all(r['upstream_repeat_bitwise'] for r in reports)
        run.save('preprocess_parity.json',dict(passed=True,constant_RGB_oracles=fixtures,samples=reports,
            scalar_values_checked=sum(r['scalar_values_checked'] for r in reports),source_store=store,resource=resource,
            versions=dict(albumentations=albumentations.__version__,cv2=cv2.__version__,numpy=np.__version__,torch=torch.__version__),
            GPU_hours=0,new_encoder_calls=0,
            interpretation='same-agent CPU upstream preprocessing/manual normalization parity on64stored body-cropped nativeframes only. No checkpoint feature parity/full scientific cohort coverage/quality/independent domain confirmation. Pixel bodycrop is adapted MS-ASL policy.'))


if __name__=='__main__':main()
