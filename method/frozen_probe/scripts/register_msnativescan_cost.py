"""Register target metadata extrema plus separate historical technical controls."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies,put,sha


def main():
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records')
    name='MSNATIVECOST001'
    entry=root/'scripts/audit_msasl_native_scan_v2.py'
    sources=dependencies(entry)+[Path(__file__).resolve().relative_to(Path.cwd()),root/'scripts/register_axis_stage.py']
    snapshots={'source_snapshot/'+str(p.relative_to(root)):str(p) for p in sorted(set(sources))}
    inputs={p:sha(p) for p in snapshots.values()}
    caps={target:Path(p).stat().st_size for target,p in snapshots.items()}
    caps.update({'config.json':65536,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536,
                 'source_parents.json':65536,'native_scan_records.json':262144,'native_scan_summary.json':65536})
    testreg=json.loads((records/'experiments/MSNATIVESCANTEST001_registration.json').read_text())
    test=Path(testreg['output_dir'])
    assert json.loads((test/'status.json').read_text())['status']=='completed'
    assert json.loads((test/'native_scan_tests.json').read_text())['passed']
    for p in test.rglob('*'):
        if p.is_file():inputs[str(p)]=sha(p)
    cr=json.loads((records/'experiments/MSCOHORT003_registration.json').read_text())
    cohort=Path(cr['output_dir']);cp=cohort/'candidate_records.json'
    rows=json.loads(cp.read_text())['candidates']
    oldreg=json.loads((records/'experiments/MSNATIVE002_registration.json').read_text())
    old=Path(oldreg['output_dir']);oldrep=json.loads((old/'native_audit.json').read_text())
    dataset=Path('/home/dongvk/datasets/MS-ASL')
    annotations={s:json.loads((dataset/f'MSASL_{s}.json').read_text()) for s in ('train','val')}
    controls=[]
    for x in oldrep['samples']:
        row=annotations[x['split']][x['annotation_index']]
        assert row['label']>=100
        controls.append(dict(sample_id=x['sample_id'],split=x['split'],annotation_index=x['annotation_index'],
                             parent_id=x['parent_id'],class_id=row['label'],signer_id=row['signer_id'],
                             raw_path=str(dataset/'video'/f"{x['parent_id']}.mp4"),
                             raw_candidate_bytes=(dataset/'video'/f"{x['parent_id']}.mp4").stat().st_size))
    ids=[x['sample_id'] for x in controls];rule={'reference_only_oldfour':ids.copy()}
    for split in ('train','val'):
        subset=[x for x in rows if x['split']==split]
        ordered=sorted(subset,key=lambda x:(x['end_time_seconds']-x['start_time_seconds'],x['sample_id']))
        picks={'max_duration':ordered[-1]['sample_id'],'median_duration':ordered[len(ordered)//2]['sample_id'],
               'max_offset':max(subset,key=lambda x:(x['start_time_seconds'],x['sample_id']))['sample_id'],
               'max_parent_bytes':max(subset,key=lambda x:(x['raw_candidate_bytes'],x['sample_id']))['sample_id']}
        rule[split]=picks
        for key in picks.values():
            if key not in ids:ids.append(key)
    assert set(ids)<={x['sample_id'] for x in rows+controls}
    c=dict(protocol='MSNATIVESCAN001_eligibility_D055_reference_controls',seed=1701,timeout_seconds=1800,
           GPU_allocation_seconds=0,HPO_trials=0,source_snapshots=snapshots,input_hashes=inputs,
           artifact_file_caps=caps,artifact_manifest_cap=65536,artifact_ceiling_bytes=2<<20,
           candidate_records_path=str(cp),dataset_root=str(dataset),selected_sample_ids=ids,selection_rule=rule,
           reference_control_samples=controls,source_byte_limit=512<<20,max_selected_frames=1024,
           max_decoded_frames=4096,reference_timeout_seconds=30,reference_byte_limit=1<<20,progress_every_rows=1,
           prior_native_runtime=str(old),prior_id_mapping={x['sample_id']:x['sample_id'] for x in controls})
    extras=[cohort/'config.json',cohort/'status.json',cp,cohort/'candidate_coverage.json',old/'config.json',old/'status.json',old/'native_audit.json',
            records/'decisions/D054.md',records/'decisions/D055.md',records/'protocols/MSNATIVESCAN001_eligibility.md',
            records/'inventory/MSNATIVECOST001_PRELAUNCH_METADATA_ERROR.json',
            records/'inventory/MSNATIVESCAN_PRELAUNCH_REGISTRATION_ERROR.json',
            records/'inventory/MSNATIVESCANTEST001_PRECONTEXT_IMPORT_FAILURE.json']
    extras += [dataset/f'MSASL_{s}.json' for s in ('train','val')]
    for x in controls:
        extras += [old/(x['sample_id']+suffix) for suffix in ('_first_native_window.npy','_native_pts.npy')]
    extras += [Path(x['raw_path']) for x in rows+controls if x['sample_id'] in ids]
    for p in extras:inputs[str(p)]=sha(p)
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    path=root/'configs'/f'{name}.json';put(path,c);h=sha(path);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(path),config_sha256=h,
             output_dir=str(root/'runtime'/rid),CPU_limit_seconds=1800,GPU_hours=0,durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs)
    put(records/'experiments'/f'{name}_registration.json',reg)
    with (records/'experiments'/f'{name}.md').open('x') as f:f.write(f'# {name} — Registered cost and reference diagnostic\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
