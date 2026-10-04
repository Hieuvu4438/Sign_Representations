"""Close environment and target dictionary contracts; no encoder or task scores."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    r=Path('method/research_lab/records');roots={};runs={}
    for name in ('MSESTTEST001','MSESTCOVER001','MSESTENV001','MSDICT001'):
        reg=json.loads((r/f'experiments/{name}_registration.json').read_text());p=Path(reg['output_dir']);roots[name]=p
        c=json.loads((p/'config.json').read_text());s=json.loads((p/'status.json').read_text())
        assert s['status']=='completed' and s['GPU_hours']==0 and s['run_id']==reg['run_id']
        assert sha(reg['config'])==sha(p/'config.json')==reg['config_sha256']==s['config_sha256']
        for f,digest in c['input_hashes'].items():assert sha(f)==digest
        files={str(x.relative_to(p)):x for x in p.rglob('*') if x.is_file()};caps={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        m=json.loads((p/'artifact_manifest.json').read_text());assert set(files)==set(caps) and set(m['artifacts'])==set(files)-{'artifact_manifest.json'}
        assert m['plan']['passed'] and all(f.stat().st_size<=caps[n] for n,f in files.items())
        for n,v in m['artifacts'].items():assert sha(files[n])==v['sha256'] and files[n].stat().st_size==v['bytes']
        for n,digest in s['output_hashes'].items():assert sha(p/n)==digest
        size=sum(f.stat().st_size for f in files.values());assert size<=c['artifact_ceiling_bytes']
        assert sum(v['bytes'] for v in m['artifacts'].values())==m['bytes_excluding_manifest']
        runs[name]=dict(run_id=s['run_id'],status=s['status'],pid=s['pid'],wall_seconds=s['wall_seconds'],logical_bytes=size,
                        config_sha256=sha(p/'config.json'),status_sha256=sha(p/'status.json'))
    cohort=json.loads((roots['MSESTCOVER001']/'adapted_cohort_records.json').read_text());byid={x['sample_id']:x for x in cohort}
    env=json.loads((roots['MSESTENV001']/'inference_input_parity.json').read_text());ec=json.loads((roots['MSESTENV001']/'config.json').read_text())
    assert env['passed'] and env['rows']==43 and [x['sample_id'] for x in env['records']]==ec['sample_ids']
    assert env['software_tests']==json.loads((roots['MSESTENV001']/'estimated_clock_tests.json').read_text())
    assert env['software_tests']['passed'] and len(env['software_tests']['groups'])==6
    for row in env['records']:
        assert row['passed'] and all(row['comparisons'].values())
        assert row['source_sha256']==byid[row['sample_id']]['source_sha256']
        assert row['native_frames']==byid[row['sample_id']]['native']['selected_native_frames']
    d=roots['MSDICT001'];dc=json.loads((d/'config.json').read_text());frozen=json.loads((d/'dictionary_manifest.json').read_text())
    summary=json.loads((d/'dictionary_summary.json').read_text());extract=json.loads((d/'extraction_manifest.json').read_text())
    eligible=[x for x in cohort if x['status']=='eligible'];train=[x for x in eligible if x['split']=='train'];val=[x for x in eligible if x['split']=='val']
    assert frozen['class_ids']==list(range(100)) and frozen['support_seeds']==dc['gallery_seeds']==[1701,1702,1703]
    assert frozen['query_ids']==sorted(x['sample_id'] for x in val) and len(val)==748
    assert frozen['numpy_version']==np.__version__
    for seed,ids in frozen['supports'].items():
        assert len(ids)==len(set(ids))==100 and [byid[id]['class_id'] for id in ids]==list(range(100))
        rng=np.random.default_rng(int(seed));expected=[]
        for cls in range(100):
            pool=sorted(x['sample_id'] for x in train if x['class_id']==cls)
            assert len(pool)==frozen['native_train_counts'][str(cls)]
            expected.append(pool[int(rng.integers(0,len(pool)))])
        assert ids==expected
        assert not set(byid[id]['parent_id'] for id in ids)&set(x['parent_id'] for x in val)
        assert not set(byid[id]['signer_id'] for id in ids)&set(x['signer_id'] for x in val)
    supportunion={id for ids in frozen['supports'].values() for id in ids};union=supportunion|set(frozen['query_ids'])
    assert [x['sample_id'] for x in extract]==sorted(union) and len(extract)==1037 and len(supportunion)==289
    for x in extract:
        original=byid[x['sample_id']];n=original['native']
        for k in ('split','class_id','signer_id','parent_id','annotation_index','source_sha256','input_contract','validation_origin'):assert x[k]==original[k]
        for k in ('time_base','origin_pts','selected_native_frames','complete_stride2_windows','clock_pts_ticks_sha256',
                  'all_resized_RGB_sha256','first_native16_RGB_sha256','crop_yxyx_pixels'):assert x[k]==n[k]
        assert x['selected_native_frames']>=16 and x['complete_stride2_windows']==(x['selected_native_frames']-16)//2+1
    assert sum(x['complete_stride2_windows'] for x in extract)==summary['complete_stride2_windows']==36424
    assert sum(x['selected_native_frames'] for x in extract)==summary['selected_native_frames']==87847
    assert summary['extraction_rows']==1037 and summary['unique_support_rows']==289 and summary['query_rows']==748
    assert len({x['signer_id'] for x in val})==summary['query_signers']==19
    assert len({x['parent_id'] for x in val})==summary['query_parents']==171
    assert max(x['selected_native_frames'] for x in extract)==summary['max_native_frames']==327
    assert sum(byid[x['sample_id']]['prior_raw_clock_status']=='incomplete' for x in extract)==summary['affected_extraction_rows']==6
    gc=json.loads(Path(json.loads((r/'experiments/MSFEATUREGPU002_registration.json').read_text())['config']).read_text())
    sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
    budget=inspect_effective_budget(gc)
    out=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,runs=runs,environment_rows=43,
             dictionary_manifest_sha256=sha(d/'dictionary_manifest.json'),extraction_manifest_sha256=sha(d/'extraction_manifest.json'),
             queries=748,unique_support_clips=289,extraction_clips=1037,complete_stride2_windows=36424,
             frozen_seeds=[1701,1702,1703],original_raw_clock_gate='failed, unchanged',
             captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),logical_bytes=sum(x['logical_bytes'] for x in runs.values()),
             effective_budget=budget,source_sha256=sha(__file__),
             scope='Same-agent savedinput/cap/hash/ID/group/rng recount. RNG replay is same algorithm, not independentmanualoracle; priorDICTTEST001 supplies fixtures. No extraRGBdecode/encoder/quality/independentreview.')
    with (r/'inventory/MSDICTIONARY_BINDING_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k not in ('runs','effective_budget')},indent=2))


if __name__=='__main__':main()
