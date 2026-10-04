"""Verify saved full-cohort input/vector/reuse/store bindings without task scores."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    r=Path('method/research_lab/records');runs={};roots={}
    for key in ('MSFULLTEST001','MSEVALTEST001','MSFULLFEAT001'):
        reg=json.loads((r/f'experiments/{key}_registration.json').read_text());root=Path(reg['output_dir']);roots[key]=root
        c=json.loads((root/'config.json').read_text());s=json.loads((root/'status.json').read_text())
        assert s['status']=='completed' and s['exit_code']==0 and s['run_id']==reg['run_id']
        assert sha(reg['config'])==sha(root/'config.json')==reg['config_sha256']==s['config_sha256']
        for path,digest in c['input_hashes'].items():assert sha(path)==digest,path
        files={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file()}
        manifest=json.loads((root/'artifact_manifest.json').read_text());caps={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        assert set(files)==set(caps) and set(manifest['artifacts'])==set(files)-{'artifact_manifest.json'}
        assert manifest['plan']['file_caps']==caps and manifest['plan']['ceiling_bytes']==c['artifact_ceiling_bytes'] and manifest['plan']['passed']
        for name,p in files.items():assert p.stat().st_size<=caps[name]
        for name,row in manifest['artifacts'].items():assert sha(files[name])==row['sha256'] and files[name].stat().st_size==row['bytes']
        for name,digest in s['output_hashes'].items():assert sha(root/name)==digest,name
        size=sum(p.stat().st_size for p in files.values());assert size<=c['artifact_ceiling_bytes']
        runs[key]=dict(run_id=s['run_id'],pid=s['pid'],wall_seconds=s['wall_seconds'],GPU_seconds=s['GPU_hours']*3600,logical_bytes=size)
    assert json.loads((roots['MSFULLTEST001']/'feature_shard_tests.json').read_text())['passed']
    assert json.loads((roots['MSEVALTEST001']/'geometry_eval_tests.json').read_text())['passed']
    root=roots['MSFULLFEAT001'];c=json.loads((root/'config.json').read_text());items=json.loads(Path(c['extraction_manifest']).read_text())
    assert [x['sample_id'] for x in items]==c['sample_ids'] and len(items)==1037
    cost=Path(c['cost_runtime']);cost_rows=json.loads((cost/'feature_cost_records.json').read_text());cost_values=np.load(cost/'pooled_vectors.npy',allow_pickle=False)
    reused={row['sample_id']:(row,cost_values[i]) for i,row in enumerate(cost_rows)}
    found=[];new_windows=old_windows=0;max_norm_error=max_recorded_pool_error=0.;times={k:[] for k in ('source_reference_decode_crop_seconds','preprocess_seconds','encoder_transfer_output_seconds','pooling_seconds')}
    for index,shard in enumerate(c['shards']):
        assert shard['index']==index and shard['start']==index*25 and shard['stop']==min((index+1)*25,1037)
        rows=json.loads((root/shard['records_name']).read_text());vectors=np.load(root/shard['array_name'],allow_pickle=False)
        part=items[shard['start']:shard['stop']];assert len(rows)==len(part) and vectors.shape==(len(part),768) and vectors.dtype==np.float64
        assert np.isfinite(vectors).all()
        for row,item,vector in zip(rows,part,vectors):
            assert row['sample_id']==item['sample_id'] and row['windows']==item['complete_stride2_windows']
            for field in ('source_sha256','clock_pts_ticks_sha256','all_resized_RGB_sha256','first_native16_RGB_sha256'):assert row['input'][field]==item[field]
            assert row['input']['source_identity_join'] and row['input']['native_frames']==item['selected_native_frames']
            assert row['pooled_vector_sha256']==hashlib.sha256(vector.tobytes()).hexdigest()
            norm=math.sqrt(math.fsum(float(v)**2 for v in vector));max_norm_error=max(max_norm_error,abs(norm-1));assert abs(norm-1)<=1e-12
            found.append(row['sample_id'])
            if row['sample_id'] in reused:
                prior,expected=reused[row['sample_id']];assert np.array_equal(vector,expected)
                assert row['reused_from']==c['cost_runtime'] and row['new_encoder_windows']==0
                for field,value in prior.items():assert row[field]==value
                old_windows+=row['windows']
            else:
                assert row['reused_from'] is None and row['new_encoder_windows']==row['windows']
                assert row['independent_pool_max_abs_error']<=1e-12;max_recorded_pool_error=max(max_recorded_pool_error,row['independent_pool_max_abs_error'])
                for key in times:times[key].append(row[key])
                new_windows+=row['windows']
    assert found==c['sample_ids'] and len(c['shards'])==42 and new_windows==36028 and old_windows==396
    summary=json.loads((root/'extraction_summary.json').read_text())
    assert summary['clips']==1037 and summary['shards']==42 and summary['new_encoder_windows']==new_windows and summary['reused_encoder_windows']==old_windows
    assert summary['new_encoder_forward_calls']==36029 and summary['independent_pool_max_abs_error']==max_recorded_pool_error
    assert summary['GPU_memory']['peak_reserved_bytes']<=2048*(1<<20)
    for short,long in zip(('decode','preprocess','encoder','pooling'),times):assert abs(math.fsum(times[long])-summary['new_clip_timings_seconds'][short])<1e-8
    sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
    budget=inspect_effective_budget(c)
    output=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,runs=runs,
        clips=1037,shards=42,new_windows=new_windows,reused_windows=old_windows,reused_vectors_bitwise_exact=7,
        maximum_independent_fsum_norm_error=max_norm_error,maximum_recorded_new_pool_error=max_recorded_pool_error,
        effective_budget=budget,captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),
        charged_GPU_seconds=sum(x['GPU_seconds'] for x in runs.values()),logical_bytes=sum(x['logical_bytes'] for x in runs.values()),
        source_sha256=sha(__file__),scope='Same-agent savedartifact/currentinputhash/shape/count/source joins and exact sevenvector reuse; independent fsum norm only. New rawwindow arrays retained as hashes, pooling error is extraction-recorded einsum check, no newpixel/encoder/taskquality or external scientificreview.')
    with (r/'inventory/MSFULL_FEATURE_BINDING_VERIFICATION.json').open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps({k:v for k,v in output.items() if k not in ('effective_budget','runs')},indent=2))


if __name__=='__main__':main()
