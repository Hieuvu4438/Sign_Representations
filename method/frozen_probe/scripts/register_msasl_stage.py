"""Exclusive prelaunch registration for annotation identity diagnostics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies, sha, put

ROOT=Path('method/frozen_probe'); RECORDS=Path('method/research_lab/records')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['tests','audit'],required=True);a=p.parse_args()
    name='MSMETATEST001' if a.stage=='tests' else 'MSMETA001'
    entry=ROOT/'scripts/audit_msasl_metadata.py'
    sources=dependencies(entry)
    if a.stage=='tests':sources += [ROOT/'tests/test_msasl_metadata.py']
    snapshots={'source_snapshot/'+str(s.relative_to(ROOT)):str(s) for s in sources}
    inputs={str(s):sha(s) for s in sources}
    for s in (Path(__file__),ROOT/'scripts/register_axis_stage.py',
              RECORDS/'decisions/D041.md',RECORDS/'protocols/MSMETA001_release_metadata.md',
              RECORDS/'literature/sources/S40.md'):
        inputs[str(s)]=sha(s)
    caps={target:Path(source).stat().st_size for target,source in snapshots.items()}
    caps.update({'config.json':65536,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol='MSMETA001_release_metadata',seed=1701,timeout_seconds=120,GPU_allocation_seconds=0,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=65536,
           artifact_ceiling_bytes=(2 if a.stage=='tests' else 12)<<20)
    if a.stage=='tests':c['software_only']=True;caps['metadata_tests.json']=16384
    else:
        test=json.loads((RECORDS/'experiments/MSMETATEST001_registration.json').read_text());root=Path(test['output_dir'])
        assert json.loads((root/'status.json').read_text())['status']=='completed'
        assert json.loads((root/'metadata_tests.json').read_text())['passed']
        for path in (root/'status.json',root/'metadata_tests.json',root/'artifact_manifest.json',RECORDS/'inventory/MSASL_METADATA_INTAKE_20261004.json'):
            inputs[str(path)]=sha(path)
        data=Path('/home/dongvk/datasets/MS-ASL')
        for name_ in ('train','val','test','classes','synonym'):
            path=data/('MSASL_'+name_+'.json');inputs[str(path)]=sha(path)
        lineage=json.loads((RECORDS/'experiments/LINEAGE003_registration.json').read_text());parent=Path(lineage['output_dir'])
        pool=parent/'youtube-sl-25-metadata.csv'
        for path in (pool,parent/'status.json',parent/'artifact_manifest.json'):inputs[str(path)]=sha(path)
        c.update(dataset_root=str(data),public_pool_csv=str(pool),archive_download_cap=4<<20,archive_expanded_cap=32<<20,
                 release_url='https://download.microsoft.com/download/3/c/a/3ca92c78-1c4a-4a91-a7ee-6980c1d242ec/MS-ASL.zip')
        caps.update({'official_annotation_archive.bin':4<<20,'annotation_rows.csv':4<<20,'metadata_audit.json':1<<20,
                     'release_provenance.json':262144,'unavailable_source_parent_ids.txt':524288})
    sys.path.insert(0,str(ROOT))
    from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':65536},c['artifact_ceiling_bytes'],('status.json','error.txt'))
    if not plan['passed']:raise ValueError('Complete artifact plan exceeds registered cap')
    config=ROOT/'configs'/f'{name}.json';put(config,c);digest=sha(config)
    rid=f'{name}_20261004_{digest[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(config),config_sha256=digest,
             output_dir=str(ROOT/'runtime'/rid),CPU_limit_seconds=120,GPU_hours=0,
             durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs)
    put(RECORDS/'experiments'/f'{name}_registration.json',reg)
    with (RECORDS/'experiments'/f'{name}.md').open('x') as stream:stream.write(f'# {name} — Registered diagnostic\n\n{rid}; no outcome yet.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
