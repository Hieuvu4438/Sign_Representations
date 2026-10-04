"""Independently recount saved temporal/hash evidence, including negative joins."""
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    records=Path('method/research_lab/records');runs={};roots={}
    for name in ('MSCLKTEST001','MSCLKDIAG001','MSCLKTEST002','MSCLKDIAG002'):
        reg=json.loads((records/f'experiments/{name}_registration.json').read_text());root=Path(reg['output_dir']);roots[name]=root
        s=json.loads((root/'status.json').read_text());c=json.loads((root/'config.json').read_text())
        assert s['run_id']==reg['run_id'] and sha(root/'config.json')==sha(reg['config'])==reg['config_sha256']==s['config_sha256']
        assert s['status']==('failed' if name=='MSCLKDIAG001' else 'completed') and s['GPU_hours']==0
        # Source/cfg/protocol identities, including all6 parents; no scientific re-run.
        for path,digest in c['input_hashes'].items():assert sha(path)==digest
        files={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file()}
        caps={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        assert set(files)<=set(caps) and all(p.stat().st_size<=caps[n] for n,p in files.items())
        size=sum(p.stat().st_size for p in files.values());assert size<=c['artifact_ceiling_bytes']
        for n,digest in s['output_hashes'].items():assert sha(root/n)==digest
        if s['status']=='completed':
            m=json.loads((root/'artifact_manifest.json').read_text());assert m['plan']['passed']
            assert set(m['artifacts'])==set(files)-{'artifact_manifest.json'}
            for n,v in m['artifacts'].items():assert v['bytes']==files[n].stat().st_size and v['sha256']==sha(files[n])
            assert sum(v['bytes'] for v in m['artifacts'].values())==m['bytes_excluding_manifest']
        else:
            assert not (root/'artifact_manifest.json').exists()
            assert set(s['missing_artifacts'])=={'clock_records.json','clock_summary.json'}
            assert s['error']=="ValueError('ndarray is not C-contiguous')"
        runs[name]=dict(run_id=s['run_id'],status=s['status'],pid=s['pid'],wall_seconds=s['wall_seconds'],logical_bytes=size,
                        config_sha256=sha(root/'config.json'),status_sha256=sha(root/'status.json'))
    actual=roots['MSCLKDIAG002'];rows=json.loads((actual/'clock_records.json').read_text());summary=json.loads((actual/'clock_summary.json').read_text())
    c=json.loads((actual/'config.json').read_text())
    old=json.loads((Path(c['diagnostic_runtime'])/'pts_diagnosis_records.json').read_text())
    assert [x['sample_id'] for x in rows]==c['sample_ids']==[x['sample_id'] for x in old] and len(rows)==38
    join_counts={k:0 for k in summary['join_pass_counts']};negative=[];membership=[];clock_totals=[];selected_joins=[]
    for x,previous in zip(rows,old):
        for key in ('parent_id','class_id','annotation_index','source_sha256'):assert x[key]==previous[key]
        raw,best,native=x['raw_ticks'],x['best_effort_ticks'],x['native_ticks'];tb=Fraction(x['time_base']);origin=x['origin_pts']
        lo,hi=map(Fraction,x['annotation_seconds']);parts=x['reference_frames']
        assert raw==[int(y['pkt_pts']) for y in parts]==previous['independent_reference']['decoded_ticks']
        assert best==[int(y['best_effort_timestamp']) for y in parts] and len(raw)==len(best)==len(native)==len(x['native_rgb_sha256'])
        def inside(ticks):return [i for i,t in enumerate(ticks) if lo<=(t-origin)*tb<hi]
        a,b=inside(raw),inside(best);z=x['comparison']
        assert z['raw_selected_indices']==a and z['best_selected_indices']==b
        assert z['added_by_best_indices']==[i for i in b if i not in a]
        assert z['removed_by_best_indices']==[i for i in a if i not in b]
        assert z['changed_tick_indices']==[i for i,(v,w) in enumerate(zip(raw,best)) if v!=w]
        assert z['raw_nonincreasing_indices']==[i for i in range(1,len(raw)) if raw[i]<=raw[i-1]]
        assert z['best_nonincreasing_indices']==[i for i in range(1,len(best)) if best[i]<=best[i-1]]
        for ticks,indices,key in ((raw,a,'raw'),(best,b,'best')):
            chosen=[ticks[i] for i in indices]
            assert z[f'selected_{key}_nonincreasing_indices']==[i for i in range(1,len(chosen)) if chosen[i]<=chosen[i-1]]
            assert z[f'{key}_complete_native16_stride2_windows']==max(0,(len(indices)-16)//2+1)
        assert z['selected_indices_identical']==(a==b)
        assert z['maximum_absolute_clock_delta_seconds']==float(max(abs(v-w) for v,w in zip(raw,best))*tb)
        # Reparse CLI text independently, retaining negative clocks rather than fixing them.
        lines=x['ffmpeg']['stdout'].splitlines()
        outtb=Fraction(next(t.split(':',1)[1].strip() for t in lines if t.startswith('#tb 0:')))
        packets=[[v.strip() for v in t.split(',')] for t in lines if t and not t.startswith('#')]
        times=[int(v[2])*outtb for v in packets];hashes=[v[5].lower() for v in packets]
        expected=dict(native_full_raw_exact=native==raw,
                      old_native_prefix_exact=native[:len(previous['decoded_native_ticks'])]==previous['decoded_native_ticks'],
                      ffmpeg_exit_zero=x['ffmpeg']['returncode']==0,ffmpeg_count_exact=len(packets)==len(raw),
                      ffmpeg_rgb_hashes_exact=hashes==x['native_rgb_sha256'],
                      ffmpeg_best_clock_exact=times==[v*tb for v in best],
                      ffmpeg_packet_sizes_exact=all(int(v[4])==x['header']['width']*x['header']['height']*3 for v in packets))
        for key,value in expected.items():assert x['joins'][key]==value;join_counts[key]+=value
        first_last=max(b)
        selected_joins.append(dict(sample_id=x['sample_id'],raw_prefix_through_last_best_selected_exact=native[:first_last+1]==raw[:first_last+1],
            ffmpeg_best_selected_clock_exact=all(times[i]==best[i]*tb for i in b),
            selected_RGB_exact=all(hashes[i]==x['native_rgb_sha256'][i] for i in b)))
        if not all(expected.values()):
            native_bad=[i for i,(v,w) in enumerate(zip(native,raw)) if v!=w]
            ffmpeg_bad=[i for i,(v,w) in enumerate(zip(times,[v*tb for v in best])) if v!=w]
            assert all(i not in b for i in native_bad+ffmpeg_bad)
            negative.append(dict(sample_id=x['sample_id'],native_raw_disagreement_indices=native_bad,
                                 ffmpeg_best_disagreement_indices=ffmpeg_bad,all_disagreements_outside_selected=True))
        if a!=b:membership.append(dict(sample_id=x['sample_id'],raw_count=len(a),best_count=len(b),added=z['added_by_best_indices'],removed=z['removed_by_best_indices']))
        assert not x['corrupt_indices'];clock_totals.append(z)
    assert join_counts==summary['join_pass_counts'] and not summary['all_frame_correspondence_gates_pass']
    assert sum(not z['best_nonincreasing_indices'] for z in clock_totals)==summary['best_clock_monotone_rows']==35
    assert sum(not z['selected_best_nonincreasing_indices'] for z in clock_totals)==summary['best_selected_monotone_rows']==38
    assert sum(len(z['raw_selected_indices']) for z in clock_totals)==summary['selected_raw_frames']==4866
    assert sum(len(z['best_selected_indices']) for z in clock_totals)==summary['selected_best_frames']==4867
    assert len(membership)==summary['changed_membership_rows']==1
    assert summary['added_frames']==1 and summary['removed_frames']==0 and summary['changed_tick_rows']==38
    assert summary['maximum_absolute_clock_delta_seconds']==.5005
    assert membership==[dict(sample_id='train_11136',raw_count=59,best_count=60,added=[6],removed=[])]
    assert all(all(v for k,v in x.items() if k!='sample_id') for x in selected_joins)
    gpu_reg=json.loads((records/'experiments/MSFEATUREGPU002_registration.json').read_text())
    sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
    budget=inspect_effective_budget(json.loads(Path(gpu_reg['config']).read_text()))
    out=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,runs=runs,rows=38,join_pass_counts=join_counts,
             original_whole_prefix_gate_failed=True,negative_joins=negative,membership_changes=membership,
             posthoc_selected_scope_joins=selected_joins,selected_scope_label='Diagnostic localization only; does not relax or rescue registered whole-prefix gate',
             captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),logical_bytes=sum(x['logical_bytes'] for x in runs.values()),
             effective_budget=budget,source_sha256=sha(__file__),
             scope='Same-agent independent saved-file/rational arithmetic/parser/hash recount; no additional RGB decode/scientific review or accepted clock/cohort/quality protocol')
    with (records/'inventory/MSCLK_BINDING_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k not in ('runs','effective_budget','posthoc_selected_scope_joins')},indent=2))


if __name__=='__main__':main()
