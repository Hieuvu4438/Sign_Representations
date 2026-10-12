"""Audit implementation/setup deliverables, not research performance."""
from pathlib import Path
import sys,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from s1.train import args,sha256
from s1.evaluate import validate_features

checks=[]
def record(requirement,evidence):checks.append({'requirement':requirement,'status':'PASS','evidence':evidence})

for name in ['README.md','SOURCES.md','REVIEW_LOG.md','UPSTREAM.json','pyproject.toml','requirements-tested.txt']:
    assert (ROOT/name).is_file() and (ROOT/name).stat().st_size>0
record('new isolated method, dependency setup, source provenance and staged reviews','README/SOURCES/REVIEW_LOG/UPSTREAM/pyproject/requirements-tested')
upstream=json.loads((ROOT/'UPSTREAM.json').read_text())
repo=ROOT.parents[1]
for r in upstream:assert sha256(repo/r['source'])==r['sha256']
record('reused original baseline source remains unchanged',f'{len(upstream)} original file SHA256 checks')

setup=json.loads((ROOT/'runtime/setup_audit.json').read_text())
current={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'s1').rglob('*.py'))}
assert setup['status']=='PASS' and setup['n_clips']==3000 and setup['source_sha256']==current
assert (setup['evaluation']['val_clips'],setup['evaluation']['gallery_clips'],setup['evaluation']['query_clips'])==(10304,3000,1500)
record('same 3000-video corpus and unchanged full evaluation protocol',setup['evaluation'])

names=['s1_input_4k','s1_full_4k','s1_capacity_4k','s1_shuffle_4k','s1_full_16k','s1_input_16k']
for name in names:
    a=args(['--config',str(ROOT/'configs'/f'{name}.json'),'--out',str(ROOT/'runtime/unused_audit_path')])
    assert a.batch==32 and not a.smoke
record('scratch input/full/capacity/shuffle controls and matched 4k/16k budgets',names)

xml=ROOT/'runtime/tests.xml';tree=ET.parse(xml)
suites=list(tree.getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==22
assert all(int(s.get('failures','0'))==int(s.get('errors','0'))==int(s.get('skipped','0'))==0 for s in suites)
assert xml.stat().st_mtime>=max(p.stat().st_mtime for p in (ROOT/'s1').rglob('*.py'))
record('raw-support/encoder leakage, auxiliary validity, gradients/EMA, resume, worker parity, projection and metric-family tests',{'report':str(xml),'tests':22,'failed':0,'skipped':0})

run=ROOT/'runtime/smoke/final_cuda';manifest=json.loads((run/'manifest.json').read_text())
assert manifest['status']=='complete' and manifest['steps_done']==2 and not manifest['protocol_run']
a=manifest['config'];assert (a['batch'],a['d'],a['depth'],a['heads'],a['dp'],a['pdepth'])==(32,256,6,4,128,4)
assert manifest['provenance']['source_sha256']==current
rows=[json.loads(l) for l in (run/'train_log.jsonl').read_text().splitlines()]
assert len(rows)==2
for row in rows:
    assert np.isfinite(row['loss']) and np.isfinite(row['grad_norm'])
    assert row['near_body_queries']>0 and row['near_relation_queries']>0
ck=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
assert ck['step']==2 and ck['provenance']==manifest['provenance']
record('production architecture/batch32 CUDA-bf16 train and checkpoint execution',{'manifest':str(run/'manifest.json'),'peak_cuda_allocated_bytes':manifest['peak_cuda_allocated_bytes'],'performance_claim':False})

for setname in ['aslc300','val']:
    raw=ROOT/'runtime/feats/S1_VERIFIED_grpraw'/setname
    dest=ROOT/'runtime/feats/S1_VERIFIED_grp'/setname
    assert np.load(raw/'windows.npy').shape[1]==1024
    meta=json.loads((dest/'meta.json').read_text())
    assert meta['checkpoint_sha256']==sha256(run/'final.pt') and meta['extraction_source_sha256']==current
    assert meta['projection']['fit']=='aslc300 gallery windows only' and meta['projection']['dim']==4
    assert not meta['protocol_complete'] and meta['training_config']['smoke']
record('real CLI frozen extraction and gallery-only projection','S1_VERIFIED smoke: real val/aslc300 poses; production 768-dimensional projection separately covered by test_production_projection_dimension')
try:validate_features('S1_VERIFIED_grp','both')
except ValueError as e:assert 'Smoke/partial' in str(e)
else:raise AssertionError('Smoke evaluation was not rejected')
record('partial/smoke outputs cannot masquerade as full-protocol research metrics','validate_features rejects S1_VERIFIED_grp')

result={'scope':'S1 implementation and training/evaluation setup, not full experiment results','status':'PASS','requirements':checks,
        'full_training_launched':False,'full_performance_evaluation_run':False,'source_sha256':current}
(ROOT/'runtime/verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','requirements':len(checks),'tests':22,'output':str(ROOT/'runtime/verification.json')},indent=2))
