"""Recount raw seven-clip features/activity/pooling and current allocation ledger."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    r=Path('method/research_lab/records');roots={};runs={}
    for name in ('MSFEATTEST001','MSFEATCOST001'):
        reg=json.loads((r/f'experiments/{name}_registration.json').read_text());p=Path(reg['output_dir']);roots[name]=p
        c=json.loads((p/'config.json').read_text());s=json.loads((p/'status.json').read_text())
        assert s['status']=='completed' and s['run_id']==reg['run_id']
        assert sha(reg['config'])==sha(p/'config.json')==reg['config_sha256']==s['config_sha256']
        for f,h in c['input_hashes'].items():assert sha(f)==h
        files={str(x.relative_to(p)):x for x in p.rglob('*') if x.is_file()};m=json.loads((p/'artifact_manifest.json').read_text())
        caps={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        assert set(files)==set(caps) and set(m['artifacts'])==set(files)-{'artifact_manifest.json'} and m['plan']['passed']
        for n,f in files.items():assert f.stat().st_size<=caps[n]
        for n,v in m['artifacts'].items():assert sha(files[n])==v['sha256'] and files[n].stat().st_size==v['bytes']
        for n,h in s['output_hashes'].items():assert sha(p/n)==h
        size=sum(f.stat().st_size for f in files.values());assert size<=c['artifact_ceiling_bytes']
        runs[name]=dict(run_id=s['run_id'],status=s['status'],pid=s['pid'],wall_seconds=s['wall_seconds'],GPU_seconds=s['GPU_hours']*3600,logical_bytes=size)
        if name=='MSFEATCOST001':config=c
    p=roots['MSFEATCOST001'];summary=json.loads((p/'feature_cost_summary.json').read_text());rows=json.loads((p/'feature_cost_records.json').read_text())
    f=np.load(p/'features.npy',allow_pickle=False);a=np.load(p/'activity.npy',allow_pickle=False);v=np.load(p/'pooled_vectors.npy',allow_pickle=False)
    assert f.shape==(396,768) and a.shape==(396,2) and v.shape==(7,768)
    assert f.dtype==a.dtype==np.float32 and v.dtype==np.float64 and all(np.isfinite(x).all() for x in (f,a,v))
    assert [x['sample_id'] for x in rows]==config['sample_ids'] and len(rows)==summary['clips']==7
    extraction={x['sample_id']:x for x in json.loads(Path(config['extraction_manifest']).read_text())};offset=0;max_error=0.
    for i,row in enumerate(rows):
        item=extraction[row['sample_id']];n=item['complete_stride2_windows'];assert row['window_offset']==offset and row['windows']==n
        x=f[offset:offset+n].astype(np.float64);active=a[offset:offset+n].astype(np.float64)
        assert row['feature_sha256']==hashlib.sha256(f[offset:offset+n].tobytes()).hexdigest()
        assert row['activity_sha256']==hashlib.sha256(a[offset:offset+n].tobytes()).hexdigest()
        for k in ('source_sha256','clock_pts_ticks_sha256','all_resized_RGB_sha256','first_native16_RGB_sha256'):assert row['input'][k]==item[k]
        assert row['input']['source_identity_join'] and row['input']['native_frames']==item['selected_native_frames']
        assert row['window_start_frames']==list(range(0,item['selected_native_frames']-15,2))
        weights=[max(min(max(float(y),0.),1.) for y in pair) for pair in active];total=math.fsum(weights)
        if total<=1e-12:mean=np.array([math.fsum(x[:,d])/n for d in range(768)])
        else:mean=np.array([math.fsum(float(x[j,d])*weights[j] for j in range(n))/total for d in range(768)])
        expected=mean/math.sqrt(math.fsum(float(y)**2 for y in mean));err=float(np.max(np.abs(expected-v[i])));max_error=max(max_error,err)
        assert err<=1e-12 and abs(np.linalg.norm(v[i])-1)<=1e-12
        assert row['pooling_metadata']['fallback_mean']==(total<=1e-12) and abs(row['pooling_metadata']['weight_sum']-total)<1e-12
        offset+=n
    assert offset==summary['encoder_windows']==396 and summary['new_encoder_forward_calls']==397
    assert summary['summed_clip_pipeline_seconds']==sum(x['pipeline_seconds'] for x in rows)
    for k,total in summary['timings'].items():assert total==sum(x[k] for x in rows)
    assert summary['GPU_memory']['peak_reserved_bytes']<=2048*(1<<20)
    sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
    budget=inspect_effective_budget(config)
    out=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,runs=runs,encoder_windows=396,forward_calls=397,
        maximum_independent_fsum_pool_error=max_error,raw_arrays_finite_and_input_joins=True,effective_budget=budget,
        captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),charged_GPU_seconds=sum(x['GPU_seconds'] for x in runs.values()),
        logical_bytes=sum(x['logical_bytes'] for x in runs.values()),source_sha256=sha(__file__),
        scope='Same-agent savedfeature/hash/cap/input/count/f64 fsum pooling recount; no newGPU/RGBdecode/taskquality or external scientificreview.')
    with (r/'inventory/MSFEATURE_COST_BINDING_VERIFICATION.json').open('x') as stream:json.dump(out,stream,indent=2)
    print(json.dumps({k:x for k,x in out.items() if k not in ('effective_budget','runs')},indent=2))


if __name__=='__main__':main()
