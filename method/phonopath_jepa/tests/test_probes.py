import json
import numpy as np
import pytest
import torch
from pp.probe import run,PARAMS,minimal_pairs
from pp.signature import RobustStats
from pp.data import PoseSource
from pp.prepare import wlasl
from pp.fit_stats import fit
from pp.config import Config
from test_core import fake_stats


def test_gloss_and_signer_disjoint_probe_protocol():
    rng=np.random.default_rng(2);rows=[];lex={};cmap={}
    for i in range(14):
        lex[str(i)]={col:classes[i%len(classes)] for col,classes in PARAMS.values()}
        for sample in range(3):
            stem=f'g{i}_{sample}';cmap[stem]=str(i)
            rows.append(dict(source_stem=stem,gloss=f'g{i}'))
    gloss=np.array([r['gloss'] for r in rows]);features=rng.normal(size=(len(rows),7))
    tr=dict(gloss=gloss,signer=np.array(['train_signer']*len(rows)),rows=np.array(json.dumps(rows)),grp=features)
    va=dict(gloss=gloss,signer=np.array(['val_signer']*len(rows)),rows=np.array(json.dumps(rows)),grp=features+.1)
    result=run(tr,va,lex,cmap,pca=3,boot=10)
    assert result['train_n']+result['val_n']==len(rows)
    assert all('balanced_accuracy' in p for p in result['probes'].values())
    va['signer']=tr['signer']
    with pytest.raises(ValueError,match='signer-disjoint'):run(tr,va,lex,cmap,pca=3,boot=10)


def test_minimal_pairs_use_independent_annotations():
    z=np.eye(3);g=np.array(['A','B','C'])
    annotations={'A':dict(Movement='straight',Handshape='flat'),'B':dict(Movement='circle',Handshape='flat'),
                 'C':dict(Movement='circle',Handshape='fist')}
    pairs=minimal_pairs(z,g,annotations,'Movement')
    assert len(pairs)==1 and pairs[0]['gloss_a']=='A' and pairs[0]['gloss_b']=='B'


def test_normalization_state_roundtrip_and_train_only_guard(tmp_path):
    cfg=Config();stats=fake_stats(cfg);stats.state['0/1']['median'][0]=2
    stats.save(tmp_path/'stats.pt');loaded=RobustStats.load(tmp_path/'stats.pt')
    assert loaded.transform('0/1',torch.zeros(1,13))[0,0]<0
    stats.provenance['split']='val';stats.save(tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='train only'):RobustStats.load(tmp_path/'bad.pt')


def test_raw_pickle_loader_preserves_missing_and_low_confidence(tmp_path):
    import pickle
    points=np.ones((3,1,133,2),np.float32)*.5
    points[:,:,5]=[.6,.3];points[:,:,6]=[.4,.3]
    score=np.ones((3,1,133),np.float32)*.01;score[1]=0
    with (tmp_path/'pose.pkl').open('wb') as f:pickle.dump(dict(keypoints=list(points),scores=list(score)),f)
    row=dict(sample_id='a',official_split='train',pose_relpath='pose.pkl',width=640,height=480,fps=25.)
    (tmp_path/'train.jsonl').write_text(json.dumps(row)+'\n')
    x,c,fps=PoseSource(tmp_path/'train.jsonl').clip(0)
    assert np.isfinite(x[0]).all() and not np.isfinite(x[1]).any()
    assert c[0].max()<.01 and fps==25
    with pytest.raises(ValueError,match='split'):PoseSource(tmp_path/'train.jsonl','val')
