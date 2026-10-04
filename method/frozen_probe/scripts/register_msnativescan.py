"""Exclusive registrations for streaming native tests and metadata-fixed cost."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies, put, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['tests','cost'], required=True)
    stage = parser.parse_args().stage
    root, records = Path('method/frozen_probe'), Path('method/research_lab/records')
    name = 'MSNATIVESCANTEST001' if stage=='tests' else 'MSNATIVECOST001'
    entry = root/'scripts/audit_msasl_native_scan.py'
    sources = dependencies(entry)+[Path(__file__).resolve().relative_to(Path.cwd()), root/'scripts/register_axis_stage.py']
    if stage=='tests':
        sources.append(root/'tests/test_msasl_native_scan.py')
    snapshots = {'source_snapshot/'+str(p.relative_to(root)):str(p) for p in sorted(set(sources))}
    inputs = {source:sha(source) for source in snapshots.values()}
    caps = {target:Path(source).stat().st_size for target,source in snapshots.items()}
    caps.update({'config.json':65536, 'environment.json':32768, 'status.json':65536,
                 'error.txt':65536, 'combined_log.txt':65536})
    for p in (records/'decisions/D054.md', records/'protocols/MSNATIVESCAN001_eligibility.md'):
        inputs[str(p)] = sha(p)
    c = dict(protocol='MSNATIVESCAN001_eligibility',seed=1701,timeout_seconds=120 if stage=='tests' else 1800,
             GPU_allocation_seconds=0,HPO_trials=0,source_snapshots=snapshots,input_hashes=inputs,
             artifact_file_caps=caps,artifact_manifest_cap=65536,artifact_ceiling_bytes=2<<20)
    if stage=='tests':
        c['software_only']=True
        caps['native_scan_tests.json']=16384
    else:
        testreg=json.loads((records/'experiments/MSNATIVESCANTEST001_registration.json').read_text())
        test=Path(testreg['output_dir'])
        assert json.loads((test/'status.json').read_text())['status']=='completed'
        assert json.loads((test/'native_scan_tests.json').read_text())['passed']
        for p in test.rglob('*'):
            if p.is_file():inputs[str(p)]=sha(p)
        cohortreg=json.loads((records/'experiments/MSCOHORT003_registration.json').read_text())
        cohort=Path(cohortreg['output_dir'])
        candidates=cohort/'candidate_records.json'
        rows=json.loads(candidates.read_text())['candidates']
        oldreg=json.loads((records/'experiments/MSNATIVE002_registration.json').read_text())
        old=Path(oldreg['output_dir'])
        oldreport=json.loads((old/'native_audit.json').read_text())
        ids=[x['sample_id'] for x in oldreport['samples']]
        rule={'old_four':ids.copy()}
        for split in ('train','val'):
            subset=[x for x in rows if x['split']==split]
            ordered=sorted(subset,key=lambda x:(x['end_time_seconds']-x['start_time_seconds'],x['sample_id']))
            picks={
                'max_duration':ordered[-1]['sample_id'],
                'median_duration':ordered[len(ordered)//2]['sample_id'],
                'max_offset':max(subset,key=lambda x:(x['start_time_seconds'],x['sample_id']))['sample_id'],
                'max_parent_bytes':max(subset,key=lambda x:(x['raw_candidate_bytes'],x['sample_id']))['sample_id']}
            rule[split]=picks
            for key in picks.values():
                if key not in ids:ids.append(key)
        assert set(ids)<={x['sample_id'] for x in rows}
        c.update(candidate_records_path=str(candidates),dataset_root='/home/dongvk/datasets/MS-ASL',
                 selected_sample_ids=ids,selection_rule=rule,source_byte_limit=512<<20,max_selected_frames=1024,
                 max_decoded_frames=4096,reference_timeout_seconds=30,reference_byte_limit=1<<20,
                 progress_every_rows=1,prior_native_runtime=str(old),prior_id_mapping={x:x for x in rule['old_four']})
        for p in (cohort/'config.json',cohort/'status.json',candidates,cohort/'candidate_coverage.json',
                  old/'config.json',old/'status.json',old/'native_audit.json',
                  records/'inventory/MSASL_NATIVE_SOURCE_SPECIFICATION.json'):
            inputs[str(p)]=sha(p)
        for x in oldreport['samples']:
            for suffix in ('_first_native_window.npy','_native_pts.npy'):
                p=old/(x['sample_id']+suffix);inputs[str(p)]=sha(p)
        for split in ('train','val'):
            p=Path(c['dataset_root'])/f'MSASL_{split}.json';inputs[str(p)]=sha(p)
        for x in rows:
            if x['sample_id'] in ids:
                inputs[x['raw_path']]=sha(x['raw_path'])
        caps.update({'source_parents.json':65536,'native_scan_records.json':262144,'native_scan_summary.json':65536})
    sys.path.insert(0,str(root))
    from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'))
    assert plan['passed']
    config=root/'configs'/f'{name}.json';put(config,c);digest=sha(config)
    rid=f'{name}_20261004_{digest[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(config),config_sha256=digest,
             output_dir=str(root/'runtime'/rid),GPU_hours=0,CPU_limit_seconds=c['timeout_seconds'],
             durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],
             input_hashes=inputs)
    put(records/'experiments'/f'{name}_registration.json',reg)
    with (records/'experiments'/f'{name}.md').open('x') as f:
        f.write(f'# {name} — Registered diagnostic\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':
    main()
