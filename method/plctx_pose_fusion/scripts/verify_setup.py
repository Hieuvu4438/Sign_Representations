"""Evidence audit before launch; no training/evaluation is started here."""
from pathlib import Path
import json,sys,hashlib,xml.etree.ElementTree as ET
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fp.utils import sha256,load_config,atomic_json
from fp.paths import REPO
from fp.preflight import verify


def require(condition,message):
    if not condition:raise AssertionError(message)


def main():
    review=ROOT/'runtime/review'
    tests=ET.parse(review/'tests.xml').getroot()
    suites=list(tests.iter('testsuite'));n=sum(int(s.get('tests','0')) for s in suites)
    bad=sum(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped'))
    require(n>=17 and bad==0,'Incomplete test evidence')
    c=load_config(ROOT/'configs/full160.json')
    require(c['steps']==200640 and c['batch']==32 and not c['smoke'],'Full160 budget mismatch')
    require(c['validate_every']==5000 and c['warmup_steps']==2000 and c['save_every']==100,'Validation/checkpoint schedule changed')
    cache,proof=verify(c)
    require(proof['counts']=={'train':40154,'val':10304,'test':32941},'Official splits changed')
    cuda=json.loads((review/'cuda_smoke.json').read_text())
    require(cuda['status']=='PASS' and cuda['exact_cuda_resume'] and cuda['frozen_PL_unchanged'],'CUDA correctness not verified')
    require(cuda['proof']['source_sha256']==proof['source_sha256'],'CUDA evidence is stale')
    require(cuda['gate_and_graph_gradients'] and cuda['teacher_no_grad'],'Gradient ownership invalid')
    ck=torch.load(review/'smoke_verified/train/last.pt',map_location='cpu',weights_only=False)
    require(ck['step']==3 and ck['progress']['validated']==[0,1,2,3],'Final-source CLI/validation smoke incomplete')
    require(ck['proof']['source_sha256']==proof['source_sha256'],'CLI smoke uses another source')
    require(ck['config']['smoke'] and ck['config']['batch']==32,'CLI smoke not production batch32')
    require((review/'smoke_verified/train/complete.json').exists(),'CLI training did not complete')
    for i in [0,1,2,3]:
        value=json.loads((review/f'smoke_verified/train/validation/{i:08}.json').read_text())
        require(not value['protocol_complete'] and set(value['results'])=={'fusion','pl_gate0'},'Smoke misrepresented')
    upstream=json.loads((ROOT/'UPSTREAM.json').read_text())
    for row in upstream:require(sha256(REPO/row['source'])==row['sha256'],'Original input source changed: '+row['source'])
    pose_source=REPO/'method/pose_jepa/src/signrepr/models/spatial_gcn.py'
    result={'status':'PASS','scope':'Implementation/setup, not accuracy verification',
            'tests_passed':n,'test_fail_errors_skips':bad,'test_xml_sha256':sha256(review/'tests.xml'),
            'proof':proof,'original_sources_unchanged':len(upstream),'pose_design_source_sha256':sha256(pose_source),
            'cuda_peak_bytes':cuda['peak_cuda_allocated_bytes'],'exact_cpu_and_cuda_resume':True,
            'real_cli_validated_steps':[0,1,2,3],'required_gpu_free_mib':10240,
            'full_run_total_steps':c['steps'],'full_run_validation_every':c['validate_every'],
            'full_run_save_every':c['save_every'],'checkpoint_initialization':'PLctx full160 frozen; graph scratch'}
    atomic_json(ROOT/'runtime/verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='proof'},indent=2))


if __name__=='__main__':main()
