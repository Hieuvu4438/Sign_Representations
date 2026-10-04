"""Verify exact fullcohort/oldproof/newclock/cap identities without quality access."""
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    r=Path('method/research_lab/records');runs={}
    for name in ('MSESTTEST001','MSESTCOVER001'):
        reg=json.loads((r/f'experiments/{name}_registration.json').read_text());p=Path(reg['output_dir'])
        c=json.loads((p/'config.json').read_text());s=json.loads((p/'status.json').read_text())
        assert s['status']=='completed' and s['GPU_hours']==0 and s['run_id']==reg['run_id']
        assert sha(reg['config'])==sha(p/'config.json')==reg['config_sha256']==s['config_sha256']
        for f,digest in c['input_hashes'].items():assert sha(f)==digest
        files={str(x.relative_to(p)):x for x in p.rglob('*') if x.is_file()};m=json.loads((p/'artifact_manifest.json').read_text())
        caps={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        assert set(files)==set(caps) and set(m['artifacts'])==set(files)-{'artifact_manifest.json'} and m['plan']['passed']
        for n,f in files.items():assert f.stat().st_size<=caps[n]
        for n,v in m['artifacts'].items():assert sha(files[n])==v['sha256'] and files[n].stat().st_size==v['bytes']
        for n,digest in s['output_hashes'].items():assert sha(p/n)==digest
        size=sum(f.stat().st_size for f in files.values());assert size<=c['artifact_ceiling_bytes']
        runs[name]=dict(run_id=s['run_id'],pid=s['pid'],status=s['status'],wall_seconds=s['wall_seconds'],logical_bytes=size)
        if name=='MSESTCOVER001':root=p;config=c
    actual=json.loads((root/'adapted_cohort_records.json').read_text());summary=json.loads((root/'adapted_cohort_summary.json').read_text())
    changed=json.loads((root/'changed_scan_records.json').read_text());ch={x['sample_id']:x for x in changed}
    old=json.loads((Path(config['native_runtime'])/'native_scan_records.json').read_text());parents=json.loads((Path(config['native_runtime'])/'source_parents.json').read_text())
    clocks=json.loads((Path(config['clock_runtime'])/'clock_records.json').read_text());bc={x['sample_id']:x for x in clocks}
    assert len(actual)==len(old)==3425 and len(ch)==38 and [x['sample_id'] for x in actual]==[x['sample_id'] for x in old]
    for x,y in zip(actual,old):
        for k in ('sample_id','parent_id','signer_id','class_id','split','annotation_index'):assert x[k]==y[k]
        assert x['prior_raw_clock_status']==y['status'] and x['input_contract']=='MSASL_ESTCLK_v1'
        assert x['source_sha256']==parents[x['parent_id']]['sha256']
        n=x['native'];count=n['selected_native_frames']
        assert x['status']==('eligible' if count>=16 else 'native_short') and n['native16_eligible']==(count>=16)
        assert n['complete_stride2_windows']==max(0,(count-16)//2+1)
        if y['status']!='incomplete':
            assert all(n[k]==v for k,v in y['native'].items())
            assert n['clock_pts_ticks_sha256']==n['raw_pts_ticks_sha256']==y['native']['native_pts_ticks_sha256']
            assert y['independent_reference']['integer_pts_exact'] and x['independent_reference']==y['independent_reference']
        else:
            assert ch[x['sample_id']]['native']==n and y['reason']=='native_pts_not_increasing'
            b=bc[x['sample_id']];tb=Fraction(b['time_base']);o=b['origin_pts'];left,right=map(Fraction,b['annotation_seconds'])
            selected=[i for i,t in enumerate(b['best_effort_ticks']) if left<=(t-o)*tb<right]
            assert selected==n['selected_decoded_ordinals']==list(range(selected[0],selected[-1]+1))
            assert n['selected_raw_ticks']==[b['raw_ticks'][i] for i in selected]
            assert n['selected_estimated_ticks']==[b['best_effort_ticks'][i] for i in selected]
            assert n['selected_raw_RGB_sha256']==[b['native_rgb_sha256'][i] for i in selected]
            assert all(v<u for v,u in zip(n['selected_estimated_ticks'],n['selected_estimated_ticks'][1:]))
            assert n['raw_prefix_exact'] and n['independent_selected_RGB_exact']
            assert n['decoded_prefix_through_last_selected_frames']==selected[-1]+1
            raw_bytes=b''.join(int(v).to_bytes(8,'little',signed=True) for v in n['selected_raw_ticks'])
            best_bytes=b''.join(int(v).to_bytes(8,'little',signed=True) for v in n['selected_estimated_ticks'])
            assert hashlib.sha256(raw_bytes).hexdigest()==n['raw_pts_ticks_sha256']
            assert hashlib.sha256(best_bytes).hexdigest()==n['clock_pts_ticks_sha256']==n['estimated_pts_ticks_sha256']
            assert count==len(selected) and ch[x['sample_id']]['reference']['raw_ticks']==b['raw_ticks']
            assert ch[x['sample_id']]['reference']['best_ticks']==b['best_effort_ticks']
    assert dict(Counter(x['status'] for x in actual))==summary['status_counts']=={'eligible':3422,'native_short':3}
    for split,count in (('train',2674),('val',748)):
        cs=Counter(x['class_id'] for x in actual if x['split']==split and x['status']=='eligible')
        assert sum(cs.values())==summary['splits'][split]['eligible_rows']==count and set(cs)==set(range(100))
        assert {str(i):cs[i] for i in range(100)}==summary['splits'][split]['class_counts']
        assert min(cs.values())==summary['splits'][split]['min_support']
    assert sum(x['native']['selected_native_frames'] for x in actual)==summary['selected_native_frames']==319858
    assert sum(x['native']['complete_stride2_windows'] for x in actual)==summary['complete_stride2_windows']==135207
    assert summary['short_rows']==[dict(sample_id=k,native_frames=v) for k,v in [('train_10280',13),('train_12438',15),('train_13051',10)]]
    checks=json.loads((root/'source_stat_validation.json').read_text());assert set(checks)==set(parents) and len(checks)==1027
    assert sum(x['rehashed_this_run'] for x in checks.values())==6
    for key,x in checks.items():
        for f,v in parents[key].items():assert x[f]==v
        stat=Path(config['raw_root'],key+'.mp4').stat();assert stat.st_size==x['bytes'] and stat.st_mtime_ns==x['mtime_ns'] and x['stat_unchanged']
    train=[x for x in actual if x['split']=='train'];val=[x for x in actual if x['split']=='val']
    assert not set(x['parent_id'] for x in train)&set(x['parent_id'] for x in val)
    assert not set(x['signer_id'] for x in train)&set(x['signer_id'] for x in val)
    sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
    gc=json.loads(Path(json.loads((r/'experiments/MSFEATUREGPU002_registration.json').read_text())['config']).read_text());budget=inspect_effective_budget(gc)
    out=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,runs=runs,all3425_ids_preserved=True,
             adapted_status_counts=summary['status_counts'],affected_rows=38,inherited_rows=3387,original_rawclock_gate='failed, unchanged',
             selected_frames=319858,complete_stride2_windows=135207,all100classes_each_split=True,sourceparents_stat_unchanged=1027,sourceparents_rehashed=6,
             captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),logical_bytes=sum(x['logical_bytes'] for x in runs.values()),
             effective_budget=budget,verifier_source_sha256=sha(__file__),
             scope='Same-agent savedartifact/rational/ID/hash/count/cap checks; inherited3387 explicitly reused, no newRGB rerun or wholecorpusrehash/independent scientificreview/quality')
    with (r/'inventory/MSEST_COHORT_BINDING_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k not in ('effective_budget','runs')},indent=2))


if __name__=='__main__':main()
