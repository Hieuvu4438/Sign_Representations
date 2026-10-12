import json
import numpy as np
import torch
import pytest
from threadpoolctl import threadpool_limits
from s2.project import gallery_fit
from s2 import evaluate as wrapper
from s2.extract import Extractor,prepare,rows_for
from s2.train import Trainer
from test_training import config


def test_projection_does_not_fit_queries():
    rng=np.random.default_rng(3);x=rng.normal(size=(120,16)).astype('float32')
    clips=[{'role':'gallery' if i<80 else 'query'} for i in range(120)]
    off=np.arange(121)
    a=gallery_fit(x,off,clips,8)
    x[80:]=1e6
    b=gallery_fit(x,off,clips,8)
    for u,v in zip(a,b):np.testing.assert_array_equal(u,v)


def test_real_extraction_shapes_and_controls(tmp_path):
    torch.set_num_threads(2)
    p=prepare(rows_for('val')[0])
    for mode in ['real','capacity','shuffle']:
        a=config(tmp_path,input=mode,lambda_body=0.,lambda_relations=0.,lambda_motion=0.)
        tr=Trainer(a);path=tmp_path/(mode+'.pt');tr.save(path,0,{})
        ex=Extractor('model',path,device='cpu')
        result=ex.encode(p)
        assert result['_grpraw'].shape==(len(p['spans']),256)
        assert result['_mean'].shape[1]==result['_cls'].shape[1]==32
        assert all(np.isfinite(v).all() for v in result.values())


def test_same_evaluator_all_metric_families(tmp_path,monkeypatch):
    E=wrapper.E
    assert str(E.__file__).endswith('/method/obs_jepa/oj/evaluate.py')
    monkeypatch.setattr(E,'RUNTIME',tmp_path)
    monkeypatch.setattr(E,'EVAL_DIR',tmp_path/'eval')
    rng=np.random.default_rng(2);d=12;classes=6;signers=6
    centers=rng.normal(size=(classes,d))
    for setname in ['val','aslc300']:
        clips=[];wins=[]
        for signer in range(signers):
            for label in range(classes):
                role='gallery' if signer<4 else 'query'
                clips.append({'sample_id':f'{signer}_{label}','gloss':str(label),'label':label,'signer':str(signer),'role':role})
                wins.extend(centers[label]+rng.normal(scale=.2,size=(4,d)))
        out=tmp_path/'feats/toy'/setname;out.mkdir(parents=True)
        np.save(out/'windows.npy',np.array(wins,np.float32));np.save(out/'offsets.npy',np.arange(0,len(wins)+1,4))
        np.save(out/'src_span.npy',np.tile([0,4],(len(wins),1)))
        (out/'clips.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in clips))
    # Synthetic priors verify evaluator wiring/metrics; real pose loading is checked separately below.
    from jt.priors import DIMS
    pri={g:(rng.normal(size=(144,k)).astype('float32'),np.ones((144,k),bool)) for g,k in DIMS.items()}
    monkeypatch.setattr(E,'window_priors',lambda *args:pri)
    with threadpool_limits(limits=2):result=E.evaluate_model('toy','both',True,True)
    assert 'R@1' in result['loso']['white512']
    assert 'DCG' in result['aslc300']['retrieval_white512']
    assert set(result['loso']['probe_R2'])=={'handshape','body','location_relations','non_manual','movement'}
    assert 'signer_probe_acc' in result['loso'] and 'geometry' in result['loso']
    assert 'macro_top1' in result['aslc300']['linear_probe']
    assert (tmp_path/'eval/models/toy.npz').exists()
    paired=E.paired('toy','toy')
    assert set(paired)=={'loso_R@1_white512','aslc300_linear_probe_macro','aslc300_retrieval_R@1'}
    assert all(value['diff']==0 and value['ci95']==[0.,0.] for value in paired.values())


def test_real_prior_loader():
    E=wrapper.E;rows=rows_for('val')[:2]
    pri=E.window_priors(rows,np.array([0,1,2]),np.array([[0,10],[0,10]]))
    assert set(pri)=={'POSE','REL','NM','MOT'}
    for y,m in pri.values():assert y.shape[0]==2 and y.shape==m.shape and np.isfinite(y).all()


def test_reject_smoke_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(wrapper,'RUNTIME',tmp_path)
    d=tmp_path/'feats/smoke/val';d.mkdir(parents=True)
    (d/'meta.json').write_text(json.dumps({'protocol_complete':False}))
    with pytest.raises(ValueError,match='Smoke/partial'):wrapper.validate_features('smoke','loso')


def test_production_projection_dimension():
    x=np.random.default_rng(6).normal(size=(790,2048)).astype('float32')
    clips=[{'role':'gallery' if i<780 else 'query'} for i in range(790)]
    mu,c,idx=gallery_fit(x,np.arange(791),clips)
    assert c.shape==(768,2048) and mu.shape==(2048,) and len(idx)==780


def test_extraction_rejects_changed_source(tmp_path):
    tr=Trainer(config(tmp_path));path=tmp_path/'mixed.pt'
    tr.save(path,0,{'source_sha256':{'s2/train.py':'changed'}})
    with pytest.raises(ValueError,match='source mismatch'):Extractor('model',path,device='cpu')


def test_projection_rejects_mixed_checkpoints_before_writing(tmp_path):
    from s2.project import main
    for split,ckpt in [('aslc300','a'),('val','b')]:
        d=tmp_path/'mixed_grpraw'/split;d.mkdir(parents=True)
        for name in ['windows.npy','offsets.npy','src_span.npy','clips.jsonl']:(d/name).write_bytes(b'')
        (d/'meta.json').write_text(json.dumps({'checkpoint_sha256':ckpt}))
    with pytest.raises(ValueError,match='Mixed extraction'):main(['--root',str(tmp_path),'--name','mixed'])
    assert not (tmp_path/'mixed_grp').exists()
