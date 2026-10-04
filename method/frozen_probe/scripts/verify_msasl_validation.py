"""Close terminal native census/diagnostic/scorer/GPU saved-array contracts.

Same-agent independent arithmetic/hash checks, not a full native re-decode or
external scientific review. All completed source/config/report arrays immutable.
"""
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records')
    names=['DICTTEST001','MSNATIVECOVER001','MSFEATUREGPU002','MSPTSDIAG001'];runs={};roots={}
    for name in names:
        reg=json.loads((records/f'experiments/{name}_registration.json').read_text());p=Path(reg['output_dir']);roots[name]=p
        c=json.loads((p/'config.json').read_text());s=json.loads((p/'status.json').read_text())
        assert s['run_id']==reg['run_id'] and s['config_sha256']==reg['config_sha256']==sha(reg['config'])==sha(p/'config.json')
        assert s['status']==('failed' if name=='MSNATIVECOVER001' else 'completed')
        caps={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        files={str(x.relative_to(p)):x for x in p.rglob('*') if x.is_file()}
        assert set(files)<=set(caps) and all(x.stat().st_size<=caps[n] for n,x in files.items())
        size=sum(x.stat().st_size for x in files.values());assert size<=c['artifact_ceiling_bytes']
        for n,h in s['output_hashes'].items():assert sha(p/n)==h
        if s['status']=='completed':
            m=json.loads((p/'artifact_manifest.json').read_text());assert set(m['artifacts'])==set(files)-{'artifact_manifest.json'}
            for n,item in m['artifacts'].items():assert files[n].stat().st_size==item['bytes'] and sha(files[n])==item['sha256']
            assert sum(files[n].stat().st_size for n in m['artifacts'])==m['bytes_excluding_manifest'] and m['plan']['passed']
        else:
            assert not (p/'artifact_manifest.json').exists() and not s['missing_artifacts']
        runs[name]=dict(run_id=s['run_id'],status=s['status'],pid=s['pid'],wall_seconds=s['wall_seconds'],
                        GPU_seconds=s['GPU_hours']*3600,logical_bytes=size,status_sha256=sha(p/'status.json'),artifact_caps_passed=True)
    native=roots['MSNATIVECOVER001'];rows=json.loads((native/'native_scan_records.json').read_text());summary=json.loads((native/'native_scan_summary.json').read_text())
    c=json.loads((native/'config.json').read_text());candidates=json.loads(Path(c['candidate_records_path']).read_text())['candidates']
    assert [x['sample_id'] for x in rows]==[x['sample_id'] for x in candidates] and len(set(x['sample_id'] for x in rows))==3425
    for a,b in zip(rows,candidates):
        for k in ('sample_id','parent_id','split','class_id','signer_id','annotation_index'):assert a[k]==b[k]
    assert dict(Counter(x['status'] for x in rows))==summary['status_counts']=={'eligible':3384,'incomplete':38,'native_short':3}
    assert dict(Counter(x['reason'] for x in rows if x['reason']))==summary['reason_counts']
    assert sum(x.get('native',{}).get('selected_native_frames',0) for x in rows)==summary['selected_native_frames']==314991
    assert sum(x.get('native',{}).get('complete_stride2_windows',0) for x in rows)==summary['complete_stride2_windows']==133045
    parents=json.loads((native/'source_parents.json').read_text())
    assert len(parents)==1027 and sum(x['bytes'] for x in parents.values())==11666105487
    for r in rows:
        if r['status']!='incomplete':
            n=r['native']['selected_native_frames'];assert r['native']['complete_stride2_windows']==(0 if n<16 else (n-16)//2+1)
            assert r['independent_reference']['selected_native_frames']==n and r['independent_reference']['integer_pts_exact']
            assert r['source_sha256']==parents[r['parent_id']]['sha256']
    for split in ('train','val'):
        counts=Counter(x['class_id'] for x in rows if x['split']==split and x['status']=='eligible')
        assert {str(k):counts[k] for k in range(100)}==summary['splits'][split]['class_counts']
        assert len(counts)==100 and sum(counts.values())==summary['splits'][split]['eligible_rows']
    diagnosis=json.loads((roots['MSPTSDIAG001']/'pts_diagnosis_records.json').read_text())
    assert [x['sample_id'] for x in diagnosis]==[x['sample_id'] for x in rows if x['status']=='incomplete']
    assert len(diagnosis)==38
    for row in diagnosis:
        tb=Fraction(row['time_base']);origin=row['origin_pts']*tb;left,right=map(lambda v:Fraction(str(v)),row['annotation_seconds'])
        ticks=row['decoded_native_ticks'];selected=[t for t in ticks if left<=t*tb-origin<right]
        assert selected==row['selected_native_ticks']==row['independent_reference']['selected_ticks']
        assert ticks==row['independent_reference']['decoded_ticks'][:len(ticks)]
        expected=[(i,a,b) for i,(a,b) in enumerate(zip(ticks,ticks[1:])) if b<=a]
        assert expected==[(x['decoded_indices'][0],*x['integer_pts']) for x in row['nonincreasing_pairs']]
        assert bool(any(b<=a for a,b in zip(selected,selected[1:])))==bool(row['selected_nonincreasing_pairs'])
    assert sum(x['only_preroll_disorder'] for x in diagnosis)==7
    assert sum(bool(x['selected_nonincreasing_pairs']) for x in diagnosis)==30
    assert sum(not x['only_preroll_disorder'] and not x['selected_nonincreasing_pairs'] for x in diagnosis)==1
    assert sum(len(x['nonincreasing_pairs']) for x in diagnosis)==65
    gpu=roots['MSFEATUREGPU002'];gc=json.loads((gpu/'config.json').read_text());report=json.loads((gpu/'feature_smoke.json').read_text());cpu=Path(gc['CPU_reference_runtime'])
    parity={}
    for name in ('features','latent','cls_active'):
        actual=np.load(gpu/(name+'.npy'),allow_pickle=False);ref=np.load(cpu/(name+'.npy'),allow_pickle=False)
        repeat=np.load(gpu/(name+'_repeats.npy'),allow_pickle=False)
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        assert np.array_equal(repeat,np.repeat(actual,3,axis=0))
        assert np.allclose(actual,ref,atol=gc['feature_atol'],rtol=gc['feature_rtol'])
        item=next(x for x in report['CPU_GPU_parity'] if x['output']==name)
        error=float(np.max(np.abs(actual-ref)));assert item['passed'] and error==item['max_abs_error']
        parity[name]=dict(max_abs_error=error,combined_atol=gc['feature_atol'],combined_rtol=gc['feature_rtol'],passed=True)
    assert report['GPU_memory']['peak_reserved_bytes']<=2048*(1<<20)
    sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
    budget=inspect_effective_budget(gc)
    result=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,runs=runs,native_rows=3425,
        native_statuses=summary['status_counts'],native_incomplete_gate_failed=True,diagnosis_rows=38,
        diagnosis_location_counts=dict(selected=30,preroll_only=7,boundary=1),GPU_CPU_parity=parity,
        actual_captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),
        actual_logical_bytes=sum(x['logical_bytes'] for x in runs.values()),
        effective_budget=budget,verifier_source_sha256=sha(__file__),
        scope='Same-agent independent arithmetic/filecaps/hash/rational-selection/GPU saved-array checks; not independent native RGB re-decode/all-source YouTube authentication/physical signer identity/scientific review. Full native failure retained; no cohort/gallery/task promotion.')
    with (records/'inventory/MSASL_VALIDATION_BINDING_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','effective_budget','scope')},indent=2))
    print('Effective GPU seconds:',budget['project_used_seconds'],'H006:',budget['hypothesis_used_seconds'])


if __name__=='__main__':main()
