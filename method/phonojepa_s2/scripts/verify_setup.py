"""Audit final S2 implementation evidence. Does not start research training.

Checks the actual checkpoint/source identity and finite artifact arrays, not just
the presence of files. Run pytest and the documented smoke checks first.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from s2.paths import REPO,RUNTIME
from s2.train import sha256,args
from s2 import evaluate


def require(condition,message):
    if not condition:raise AssertionError(message)


def main():
    source={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'s2').rglob('*.py'))}
    result={'material_passport':{'artifact':'S2 implementation/setup verification',
        'scope':'proposal12 S2; execution correctness, not performance validation'},
        'source_sha256':source,'evidence':{}}
    setup=json.loads((RUNTIME/'setup_audit.json').read_text())
    require(setup['status']=='PASS' and setup['n_clips']==3000,'Corpus preflight missing')
    require(setup['source_sha256']==source,'Stale setup preflight')
    require(setup['evaluation']['val_clips']==10304 and setup['evaluation']['gallery_clips']==3000
            and setup['evaluation']['query_clips']==1500,'Changed evaluation splits')
    evaluator=REPO/'method/obs_jepa/oj/evaluate.py'
    require(sha256(evaluator)==setup['evaluation']['evaluator_sha256'],'Shared evaluator changed')
    result['evidence']['preflight']=setup['evaluation']
    upstream=json.loads((ROOT/'UPSTREAM.json').read_text())
    for row in upstream:require(sha256(REPO/row['source'])==row['sha256'],f'S1 source changed: {row["source"]}')
    result['evidence']['s1_sources_unchanged']=len(upstream)
    suites=ET.parse(RUNTIME/'review/tests.xml').getroot()
    total=sum(int(s.get('tests',0)) for s in suites.iter('testsuite'))
    failures=sum(int(s.get(k,0)) for s in suites.iter('testsuite') for k in ('failures','errors','skipped'))
    require(total>=55 and failures==0,'Incomplete/failing test suite')
    result['evidence']['tests']={'passed':total,'failed_errors_skipped':failures,
        'junit_sha256':sha256(RUNTIME/'review/tests.xml')}
    configs=sorted((ROOT/'configs').glob('*.json'))
    require(len(configs)==16,'Config matrix incomplete')
    for cfg in configs:
        a=args(['--config',str(cfg),'--out','/tmp/unused_s2_config_check'])
        require(not a.smoke and a.stats_clips==640 and a.batch==32,'Nonprotocol config')
    result['evidence']['configs_sha256']={p.name:sha256(p) for p in configs}
    stats=json.loads((RUNTIME/'review/stats640.json').read_text())
    require(stats['status']=='PASS' and stats['n_stats_clips']==640 and stats['n_unique_clips']==640,'Full-size normalization audit missing')
    require(len(set(stats['fit_ids']))==640 and stats['source_provenance']['source_sha256']==source,'Stale normalization evidence')
    from s2.data.cache import ClipCache
    from s2.train import DEFAULT_CACHE
    corpus=ClipCache(DEFAULT_CACHE)
    require(set(stats['fit_ids'])<=set(corpus.ids.tolist()),'Normalization used nontraining clips')
    for key,values in stats['normalization'].items():
        require(len(values['counts_by_path'])==15 and min(values['counts_by_path'].values())>0,f'Uncovered {key} paths')
    result['evidence']['stats640']={'status':'PASS','n_train_clips':640,'seconds':stats['seconds'],
                                 'sha256':sha256(RUNTIME/'review/stats640.json')}
    smokes={}
    for run,mode,sig in [('SMOKE_CUDA_PRIMARY_FINAL','none','plain'),
                         ('SMOKE_CUDA_DESCRIPTOR_FINAL','both','leadlag')]:
        folder=RUNTIME/run;manifest=json.loads((folder/'manifest.json').read_text())
        require(manifest['status']=='complete' and manifest['steps_done']==2 and not manifest['protocol_run'],f'Bad smoke manifest {run}')
        ck=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);c=ck['config']
        require(ck['provenance']['source_sha256']==source,f'Stale checkpoint source {run}')
        require(ck['step']==2 and c['smoke'] and c['descriptor_input']==mode and c['signature']==sig,'Wrong smoke branch')
        require((c['batch'],c['d'],c['depth'],c['heads'],c['dp'],c['pdepth'])==(32,256,6,4,128,4),'Smoke lacks production model/batch')
        for key in ('model','intervals','auxiliary'):
            require(all(torch.isfinite(v).all() for v in ck[key].values()),f'Nonfinite checkpoint {key}')
        for kind in ('motion','signature','level','mae'):
            require((ck['intervals'][kind+'_scale']>0).all(),f'Invalid {kind} normalization')
            require(ck['intervals'][kind+'_count'].sum()>0,f'Unfitted {kind} normalization')
        logs=[json.loads(l) for l in (folder/'train_log.jsonl').read_text().splitlines()]
        require([l['step'] for l in logs]==[1,2],'Wrong smoke step log')
        for row in logs:
            require(all(math.isfinite(v) for v in row.values() if isinstance(v,(float,int))),f'Nonfinite training log {run}')
            require(row['near_interval_queries']>0 and row['far_interval_queries']>0,'Inactive interval supervision')
        require(manifest['peak_cuda_allocated_bytes']>0,'No CUDA execution evidence')
        smokes[run]={'steps':ck['step'],'model_dimensions':[256,6,4,128,4],
                    'batch':32,'checkpoint_sha256':sha256(folder/'final.pt'),
                    'peak_cuda_allocated_bytes':manifest['peak_cuda_allocated_bytes']}
    result['evidence']['cuda_smokes']=smokes
    resume=json.loads((RUNTIME/'review/cuda_resume.json').read_text())
    require(resume['status']=='PASS' and resume['source_sha256']==source and resume['batch']==32,
            'CUDA checkpoint continuation not verified for this source')
    result['evidence']['cuda_resume']=resume
    extraction={}
    for name,run in [('SMOKE_PRIMARY_FINAL','SMOKE_CUDA_PRIMARY_FINAL'),
                     ('SMOKE_DESCRIPTOR_FINAL','SMOKE_CUDA_DESCRIPTOR_FINAL')]:
        original={}
        for split in ('val','aslc300'):
            folder=RUNTIME/'feats'/(name+'_grpraw')/split
            meta=json.loads((folder/'meta.json').read_text())
            require(meta['extraction_source_sha256']==source,'Stale extraction')
            require(meta['checkpoint_sha256']==smokes[run]['checkpoint_sha256'],'Mixed extraction checkpoint')
            require(meta['training_config']['smoke'] and not meta['protocol_complete'],'Smoke mislabeled full')
            x=np.load(folder/'windows.npy');off=np.load(folder/'offsets.npy')
            rows=[json.loads(l) for l in (folder/'clips.jsonl').read_text().splitlines()]
            require(x.ndim==2 and x.shape[1]==2048 and np.isfinite(x).all(),'Bad static/temporal raw features')
            require(len(off)==len(rows)+1 and off[0]==0 and off[-1]==len(x) and np.all(np.diff(off)>0),'Bad extracted offsets')
            original[split]=(x,off,rows)
        projected=RUNTIME/'feats'/(name+'_grp')
        projection=np.load(projected/'projection.npz');mu=projection['mean'];components=projection['components']
        require(components.shape==(16,2048),'Tiny smoke projection must be16D, production768 checked in pytest')
        projection_hash=sha256(projected/'projection.npz')
        for split,(x,off,rows) in original.items():
            z=np.load(projected/split/'windows.npy');meta=json.loads((projected/split/'meta.json').read_text())
            require(z.shape==(len(x),16) and np.isfinite(z).all(),'Bad projected features')
            expected=((x.astype(np.float32)-mu)@components.T).astype(np.float16)
            np.testing.assert_allclose(z,expected,atol=2e-3,rtol=2e-3)
            require(meta['projection']['fit']=='aslc300 gallery windows only' and meta['projection']['sha256']==projection_hash,'Invalid projection provenance')
            gallery=[r['sample_id'] for r in original['aslc300'][2] if r['role']=='gallery']
            import hashlib
            require(meta['projection']['gallery_ids_sha256']==hashlib.sha256(json.dumps(gallery).encode()).hexdigest(),'Projection fit IDs mismatch')
        try:evaluate.validate_features(name+'_grp','both')
        except ValueError as exc:require('Smoke/partial' in str(exc),'Wrong rejection reason')
        else:raise AssertionError('Smoke evaluation accepted as research result')
        extraction[name]={'raw_width':2048,'smoke_projected_width':16,'smoke_evaluation':'rejected',
                          'windows':{s:len(x) for s,(x,_,_) in original.items()}}
    result['evidence']['extraction_projection']=extraction
    result['test_source_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'tests').glob('*.py'))}
    result['status']='PASS'
    (RUNTIME/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'tests':total,'s1_unchanged':len(upstream),
                      'cuda_smokes':list(smokes),'extractions':extraction},indent=2))


if __name__=='__main__':main()
