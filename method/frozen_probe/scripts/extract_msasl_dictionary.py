"""Full frozen dictionary feature extraction with durable completed-clip shards."""
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
from msasl_feature_shards import reuse_cost_vectors,shard_plan
from msasl_feature_inputs import collect_native_inputs
from resources import scratch_preflight,gpu_preflight
from limited_gpu_resources import configure_gpu_2gib
from pooling import activity_mean
from window_budget import pooled_vector


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','output-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    initial=json.loads(Path(a.config).read_text());ctx=BoundedCPUContext if initial.get('software_only') else BoundedGPUContext
    with ctx(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config
        if c.get('software_only'):
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
            from test_msasl_feature_shards import run_tests
            run.save('feature_shard_tests.json',run_tests());return
        import torch,cv2,av
        from signrep_adapter import load_model,preprocess_rgb,infer_windows
        resource=scratch_preflight('/dev/shm',1<<30);cv2.setNumThreads(1)
        items=json.loads(Path(c['extraction_manifest']).read_text());assert len(items)==c['expected_clips']==1037
        assert [x['sample_id'] for x in items]==c['sample_ids']
        parent=Path(c['cost_runtime']);costrows=json.loads((parent/'feature_cost_records.json').read_text());costvectors=np.load(parent/'pooled_vectors.npy',allow_pickle=False)
        reused=reuse_cost_vectors(costrows,costvectors,items);assert len(reused)==7
        parents=json.loads(Path(c['source_parents']).read_text())
        sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
        budget=inspect_effective_budget(c);observation=gpu_preflight(c['min_free_mib'],c['max_gpu_utilization'])
        if observation['uuid']!=c['granted_uuid']:raise ValueError('Granted GPU identity changed')
        run.start_gpu(budget);device=configure_gpu_2gib(observation);torch.manual_seed(a.seed)
        stamp=time.monotonic();model,params=load_model(c['checkpoint'],device=device);torch.cuda.synchronize();model_seconds=time.monotonic()-stamp
        assert params['mask_ratio']==0 and params['out_dim']==768 and not model.training
        new_windows=0;reused_windows=0;warmup=0.;warmed=False;pool_max_error=0.;timings={k:0. for k in ('decode','preprocess','encoder','pooling')}
        for shard in c['shards']:
            vectors=[];records=[]
            for item in items[shard['start']:shard['stop']]:
                id=item['sample_id']
                if id in reused:
                    vector,prior=reused[id];reused_windows+=item['complete_stride2_windows']
                    record=dict(prior,reused_from=c['cost_runtime'],new_encoder_windows=0)
                else:
                    stamp=time.monotonic();frames,pts,input_meta=collect_native_inputs(item,c['ffprobe'],c['max_native_frames'],c['max_decoded_frames'],source_identity=parents[item['parent_id']]);decode=time.monotonic()-stamp
                    stamp=time.monotonic();prepared=preprocess_rgb(frames);preprocess=time.monotonic()-stamp
                    assert prepared.dtype==torch.float32 and tuple(prepared.shape)==(item['selected_native_frames'],3,224,224)
                    normalized=hashlib.sha256(prepared.numpy().tobytes()).hexdigest()
                    if not warmed:
                        torch.cuda.synchronize();stamp=time.monotonic();infer_windows(model,prepared,pts,selected_indices=[0],batch_size=1)
                        torch.cuda.synchronize();warmup=time.monotonic()-stamp;warmed=True
                    torch.cuda.synchronize();stamp=time.monotonic();outputs,windows=infer_windows(model,prepared,pts,batch_size=1)
                    torch.cuda.synchronize();encoder=time.monotonic()-stamp
                    f,act=outputs['features'].numpy(),outputs['cls_active'].numpy();n=item['complete_stride2_windows']
                    assert f.shape==(n,768) and act.shape==(n,2) and f.dtype==act.dtype==np.float32 and np.isfinite(f).all() and np.isfinite(act).all()
                    assert len(windows)==n and [x['start_frame'] for x in windows]==list(range(0,len(frames)-15,2))
                    stamp=time.monotonic();vector=pooled_vector(f,act,'full_activity',list(range(n)));_,poolmeta=activity_mean(f,act)
                    x=f.astype(np.float64);weights=np.max(np.clip(act.astype(np.float64),0,1),axis=1);total=float(weights.sum())
                    independent=x.mean(axis=0) if total<=1e-12 else np.einsum('i,ij->j',weights,x)/total
                    independent/=np.linalg.norm(independent);error=float(np.max(np.abs(independent-vector)));pool_max_error=max(pool_max_error,error)
                    assert error<=1e-12 and np.isfinite(vector).all() and abs(np.linalg.norm(vector)-1)<=1e-12
                    pooling=time.monotonic()-stamp
                    for key,value in zip(timings,(decode,preprocess,encoder,pooling)):timings[key]+=value
                    record=dict(sample_id=id,input=input_meta,normalized_input_sha256=normalized,windows=n,new_encoder_windows=n,reused_from=None,
                        feature_sha256=hashlib.sha256(f.tobytes()).hexdigest(),activity_sha256=hashlib.sha256(act.tobytes()).hexdigest(),
                        source_reference_decode_crop_seconds=decode,preprocess_seconds=preprocess,encoder_transfer_output_seconds=encoder,pooling_seconds=pooling,
                        pooling_metadata=poolmeta,independent_pool_max_abs_error=error)
                    new_windows+=n;del frames,prepared,outputs,f,act,x
                record['pooled_vector_sha256']=hashlib.sha256(vector.tobytes()).hexdigest();vectors.append(vector);records.append(record)
            run.save_array(shard['array_name'],np.stack(vectors),shard['spec']);run.save(shard['records_name'],records)
            run.status.update(progress_clips=shard['stop'],expected_clips=1037,completed_shards=shard['index']+1,new_encoder_windows=new_windows,reused_windows=reused_windows)
            run.store.write_json('status.json',run.status)
            print(json.dumps(dict(progress_clips=shard['stop'],new_encoder_windows=new_windows,reused_windows=reused_windows)),flush=True)
        assert new_windows==36028 and reused_windows==396 and new_windows+reused_windows==36424
        run.save('extraction_summary.json',dict(clips=1037,shards=len(c['shards']),new_encoder_windows=new_windows,reused_encoder_windows=reused_windows,
            new_encoder_forward_calls=new_windows+1,model_params=params,model_load_and_move_seconds=model_seconds,warmup_seconds=warmup,
            new_clip_timings_seconds=timings,independent_pool_max_abs_error=pool_max_error,resource=resource,GPU_observation=observation,
            GPU_memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()),
            versions=dict(torch=torch.__version__,av=av.__version__,av_libraries=av.library_versions,cv2=cv2.__version__,numpy=np.__version__),
            input_contract='MSASL_ESTCLK_v1',pooling='unchanged FP64 full_activity and L2; allnative16stride2',
            quality_access='none; no dictionary scores/rankings/target fit',durable_window_features='hashes only; seven costclips retain raw arrays separately',
            reuse='Seven costvectors bound by complete parent/source/pipeline/input hashes; no second extraction',
            resume='Completed shards retained on failure; exclusive fresh outputs, any reuse requires a fresh preregistered source/config/checkpoint-consistent run',
            measurement='Contended allocated pipeline; no idlehardware or linguistic/novelmethod inference'))


if __name__=='__main__':main()
