"""Registered dense native extraction cost, no dictionary or target quality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_gpu_context import BoundedGPUContext
from msasl_feature_inputs import collect_native_inputs
from resources import scratch_preflight,gpu_preflight
from limited_gpu_resources import configure_gpu_2gib
from window_budget import pooled_vector
from pooling import activity_mean


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','output-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    initial=json.loads(Path(a.config).read_text());ctx=BoundedCPUContext if initial.get('software_only') else BoundedGPUContext
    with ctx(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config
        if c.get('software_only'):
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
            from test_msasl_feature_inputs import run_tests
            run.save('feature_input_tests.json',run_tests());return
        resource=scratch_preflight('/dev/shm',1<<30)
        import torch,cv2,av
        from signrep_adapter import load_model,preprocess_rgb,infer_windows
        sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
        budget=inspect_effective_budget(c);observation=gpu_preflight(c['min_free_mib'],c['max_gpu_utilization'])
        if observation['uuid']!=c['granted_uuid']:raise ValueError('Granted GPU identity changed')
        run.start_gpu(budget);device=configure_gpu_2gib(observation);torch.manual_seed(a.seed);cv2.setNumThreads(1)
        start=time.monotonic();model,params=load_model(c['checkpoint'],device=device);torch.cuda.synchronize();model_load=time.monotonic()-start
        assert params['mask_ratio']==0.0 and params['out_dim']==768 and not model.training
        items=json.loads(Path(c['extraction_manifest']).read_text());byid={x['sample_id']:x for x in items}
        parents=json.loads(Path(c['source_parents']).read_text());records=[];features=[];activity=[];vectors=[];offset=0;warmup_seconds=0.
        for index,id in enumerate(c['sample_ids']):
            item=byid[id];begin=time.monotonic();stamp=begin
            frames,pts,input_meta=collect_native_inputs(item,c['ffprobe'],c['max_native_frames'],c['max_decoded_frames'],source_identity=parents[item['parent_id']])
            decode_seconds=time.monotonic()-stamp;assert input_meta['source_identity_join']
            stamp=time.monotonic();prepared=preprocess_rgb(frames);preprocess_seconds=time.monotonic()-stamp
            assert prepared.dtype==torch.float32 and tuple(prepared.shape)==(item['selected_native_frames'],3,224,224)
            normalized_sha=hashlib.sha256(prepared.numpy().tobytes()).hexdigest()
            if index==0:
                torch.cuda.synchronize();stamp=time.monotonic()
                infer_windows(model,prepared,pts,selected_indices=[0],length=16,stride=2,batch_size=1)
                torch.cuda.synchronize();warmup_seconds=time.monotonic()-stamp
            torch.cuda.synchronize();stamp=time.monotonic()
            outputs,windows=infer_windows(model,prepared,pts,length=16,stride=2,batch_size=1)
            torch.cuda.synchronize();encoder_seconds=time.monotonic()-stamp
            f,act=outputs['features'].numpy(),outputs['cls_active'].numpy()
            assert f.shape==(item['complete_stride2_windows'],768) and act.shape==(len(f),2)
            assert f.dtype==act.dtype==np.float32 and np.isfinite(f).all() and np.isfinite(act).all()
            assert len(windows)==len(f) and [w['start_frame'] for w in windows]==list(range(0,len(frames)-15,2))
            stamp=time.monotonic();vector=pooled_vector(f,act,'full_activity',list(range(len(f))));_,pool_meta=activity_mean(f,act)
            pooling_seconds=time.monotonic()-stamp
            assert vector.dtype==np.float64 and vector.shape==(768,) and np.isfinite(vector).all() and abs(np.linalg.norm(vector)-1)<1e-12
            records.append(dict(sample_id=id,input=input_meta,normalized_input_sha256=normalized_sha,window_offset=offset,windows=len(f),
                feature_sha256=hashlib.sha256(f.tobytes()).hexdigest(),activity_sha256=hashlib.sha256(act.tobytes()).hexdigest(),
                source_reference_decode_crop_seconds=decode_seconds,preprocess_seconds=preprocess_seconds,encoder_transfer_output_seconds=encoder_seconds,
                pooling_seconds=pooling_seconds,pipeline_seconds=time.monotonic()-begin,pooling_metadata=pool_meta,
                window_start_frames=[w['start_frame'] for w in windows],window_centers_seconds=[w['center_s'] for w in windows]))
            features.append(f.copy());activity.append(act.copy());vectors.append(vector);offset+=len(f)
            run.status.update(progress_clips=index+1,expected_clips=len(c['sample_ids']),encoder_windows=offset)
            run.store.write_json('status.json',run.status)
            print(json.dumps(dict(sample_id=id,progress_clips=index+1,encoder_windows=offset)),flush=True)
            del frames,prepared,outputs,f,act
        assert offset==c['expected_encoder_windows']
        matrices=dict(features=np.concatenate(features),activity=np.concatenate(activity),pooled_vectors=np.stack(vectors))
        for key,array in matrices.items():run.save_array(key+'.npy',array,c['array_specs'][key+'.npy'])
        run.save('feature_cost_records.json',records)
        run.save('feature_cost_summary.json',dict(clips=len(records),encoder_windows=offset,new_encoder_forward_calls=offset+1,
            model_params=params,model_load_and_move_seconds=model_load,warmup_seconds=warmup_seconds,
            summed_clip_pipeline_seconds=sum(x['pipeline_seconds'] for x in records),
            timings={k:sum(x[k] for x in records) for k in ('source_reference_decode_crop_seconds','preprocess_seconds','encoder_transfer_output_seconds','pooling_seconds')},
            GPU_memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()),
            resource=resource,GPU_observation=observation,budget_before_launch=budget,
            versions=dict(torch=torch.__version__,cuda=torch.version.cuda,av=av.__version__,av_libraries=av.library_versions,cv2=cv2.__version__,numpy=np.__version__),
            input_policy='MSASL_ESTCLK_v1; all native16 stride2; source/crop/clock/fullRGB/first16 hash joins',
            pooling='unchanged full_activity: max(clip(raw activity,0,1)), FP64 weighted mean/fallback, FP64 L2',
            measurement='contended GPU pipeline cost, not isolated latency or corpus budget guarantee',
            allocation_policy=dict(torch_allocator_cap_mib=2048,physical_free_min_mib=3072,max_utilization=100),
            quality_access='none; no rankings/similarities/target fit',gallery_reselection='none',
            limitations='Seven fixed count-extreme/median/affected clips; warmup belongs to firstpipeline duration. Sourcehash/context init excluded from perclip timing, included in whole statuswall; source bytes stat-joined and hashes checked by context. Fulltask requires conservative prospective budget.'))


if __name__=='__main__':main()
