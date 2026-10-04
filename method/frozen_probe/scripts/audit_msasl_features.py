"""Fixed CPU reference or GPU smoke for four saved native MS-ASL windows."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_gpu_context import BoundedGPUContext
from bounded_artifacts import array_spec,verify_store
from resources import scratch_preflight,gpu_preflight,configure_gpu


def main():
    p=argparse.ArgumentParser()
    for n in ('config','run-id','output-dir'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    initial=json.loads(Path(a.config).read_text());context=BoundedGPUContext if initial['device']=='gpu' else BoundedCPUContext
    with context(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;resource=scratch_preflight('/dev/shm',256<<20)
        from signrep_adapter import load_model,preprocess_rgb,infer_windows
        import torch
        torch.manual_seed(a.seed)
        if c['device']=='gpu':
            sys.path.insert(0,str(Path('method/stream_reliability')))
            from gpu_accounting import inspect_effective_budget
            budget=inspect_effective_budget(c)
            observation=gpu_preflight(c['min_free_mib'],c['max_gpu_utilization'])
            if observation['uuid']!=c['granted_uuid']:raise ValueError('Granted GPU identity differs')
            run.start_gpu(budget);device=configure_gpu(observation)
        else:
            torch.set_num_threads(1);torch.set_num_interop_threads(1);device='cpu';budget=observation=None
        began=time.monotonic();model,params=load_model(c['checkpoint'],device=device)
        if c['device']=='gpu':torch.cuda.synchronize()
        loadseconds=time.monotonic()-began
        native=Path(c['native_runtime']);pre=Path(c['preprocess_runtime']);verify_store(native);verify_store(pre)
        prereport=json.loads((pre/'preprocess_parity.json').read_text());inputSHAs={r['sample_id']:r['normalized_input_SHA'] for r in prereport['samples']}
        results={k:[] for k in ('features','latent','cls_active')};repeats={k:[] for k in results};rows=[]
        transformpath=Path(c['upstream_transform']);spec=importlib.util.spec_from_file_location('feature_smoke_upstream_transform',transformpath)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);transformation=module.Transformation()
        for item in c['samples']:
            sid=item['sample_id'];rgb=np.load(native/f'{sid}_first_native_window.npy',allow_pickle=False)
            pts=np.load(native/f'{sid}_native_pts.npy',allow_pickle=False)[:16].tolist()
            prepared=preprocess_rgb(list(rgb));manual=transformation.aug_video(list(rgb))
            assert torch.equal(prepared,manual) and hashlib.sha256(prepared.numpy().tobytes()).hexdigest()==inputSHAs[sid]
            if c['device']=='gpu':torch.cuda.synchronize()
            warmstart=time.monotonic()
            infer_windows(model,prepared,pts,length=16,stride=2,batch_size=1)
            if c['device']=='gpu':torch.cuda.synchronize()
            warmwall=time.monotonic()-warmstart
            durations=[];direct=[];wrapped=[]
            for repetition in range(3):
                if c['device']=='gpu':torch.cuda.synchronize()
                start=time.monotonic();outputs,plan=infer_windows(model,prepared,pts,length=16,stride=2,batch_size=1)
                if c['device']=='gpu':torch.cuda.synchronize()
                durations.append(time.monotonic()-start);assert len(plan)==1
                wrapped.append({k:v.numpy() for k,v in outputs.items()})
                with torch.inference_mode():ref=model(manual.unsqueeze(0).permute(0,2,1,3,4).contiguous().to(device))
                direct.append({k:ref[k].detach().cpu().numpy() for k in results})
            parity={}
            for k,dimension in (('features',768),('latent',768),('cls_active',2)):
                arrays=[r[k] for r in wrapped];refs=[r[k] for r in direct]
                assert all(x.shape==(1,dimension) and x.dtype==np.float32 and np.isfinite(x).all() for x in arrays+refs)
                assert all(np.array_equal(x,ref) for x,ref in zip(arrays,refs))
                assert all(np.array_equal(x,arrays[0]) for x in arrays[1:])
                results[k].append(arrays[0][0]);repeats[k].extend(x[0] for x in arrays)
                parity[k]=dict(direct_wrapper_bitwise=True,repeat_bitwise=True)
            rows.append(dict(sample_id=sid,input_SHA=inputSHAs[sid],warmup_seconds=warmwall,
                repeated_encoder_with_input_transfer_output_copy_seconds=durations,parity=parity,
                scope='one local16frame storedcrop; fixed3timing repetitions, labelquality not accessed'))
        for k,arrays in results.items():
            matrix=np.stack(arrays);run.save_array(k+'.npy',matrix,array_spec(matrix.shape,matrix.dtype))
            repeat=np.stack(repeats[k]);run.save_array(k+'_repeats.npy',repeat,array_spec(repeat.shape,repeat.dtype))
        cross=[]
        if c['device']=='gpu':
            cpuroot=Path(c['CPU_reference_runtime']);verify_store(cpuroot)
            for k,arrays in results.items():
                cpu=np.load(cpuroot/(k+'.npy'),allow_pickle=False);gpu=np.stack(arrays)
                passed=bool(np.allclose(cpu,gpu,atol=c['feature_atol'],rtol=c['feature_rtol']))
                cross.append(dict(output=k,passed=passed,max_abs_error=float(np.max(np.abs(cpu-gpu))),atol=c['feature_atol'],rtol=c['feature_rtol']))
            memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        else:memory=None
        run.save('feature_smoke.json',dict(device=device,strict_load=True,eval_mode=not model.training,model_params=params,
            checkpoint_SHA=hashlib.sha256(Path(c['checkpoint']).read_bytes()).hexdigest(),samples=rows,CPU_GPU_parity=cross,
            model_load_and_move_seconds=loadseconds,GPU_memory=memory,resource=resource,GPU_observation=observation,
            effective_budget_before_launch=budget,new_encoder_calls=28,
            scope='four technicalnative windows/direct-wrapper/repeat finitefeature checks only; no modelquality/ranking/newmethod/domain confirmation. Timing includes transfer/outputcopy, excludes source decode; warmup and full allocation cost disclosed'))
        if cross and not all(r['passed'] for r in cross):raise ValueError('Fixed CPU/GPU feature tolerance failed')


if __name__=='__main__':main()
