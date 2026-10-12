import copy
import numpy as np
import pytest
import torch
from test_training import config,batch
from test_data import clip,pair
from s2.train import Trainer,Windows,DEFAULT_CACHE,args
from s2.data.dataset import Cfg,collate
from s2.data.cache import ClipCache
from s2 import features as Fe
from s2.intervals import descriptors,selectors,layout,P
from s2.interval_heads import IntervalHeads,DescriptorTokenizer,balanced
from s2.model import Encoder


@pytest.mark.parametrize('desc,signature,kind',[
    ('none','plain','motion'),('none','leadlag','motion'),
    ('none','plain','latent'),('none','plain','mae'),
    ('motion','plain','motion'),('signature','plain','motion'),
    ('both','leadlag','motion')])
def test_all_arms_finite_gradients_teacher_frozen(tmp_path,desc,signature,kind):
    torch.set_num_threads(2);torch.manual_seed(5)
    tr=Trainer(config(tmp_path,descriptor_input=desc,signature=signature,interval_target=kind,
                      lambda_signature=.1,lambda_body=0.,lambda_relations=0.))
    before=copy.deepcopy(tr.model.teacher.state_dict())
    out=tr.step(batch(),0)
    assert np.isfinite(out['loss']) and out['near_interval_queries']>0
    assert tr.intervals.heads[kind].weight.grad.abs().sum()>0
    assert tr.intervals.heads['signature'].weight.grad.abs().sum()>0
    assert tr.intervals.heads['level'].weight.grad.abs().sum()>0
    assert tr.model.student.tok.mlps[0][0].weight.grad.abs().sum()>0
    if desc!='none':assert tr.model.student.descriptor.net[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in tr.model.teacher.parameters())
    assert any(not torch.equal(x,before[k]) for k,x in tr.model.teacher.state_dict().items())


def test_real_training_fit_stats_descriptor_extraction_checkpoint(tmp_path):
    from s2.extract import Extractor,prepare,rows_for
    tr=Trainer(config(tmp_path,descriptor_input='both',signature='leadlag'))
    ds=Windows(ClipCache(DEFAULT_CACHE),Cfg(),7)
    tr.fit_stats(ds)
    assert tr.intervals.motion_count.sum()>0
    assert tr.intervals.signature_count.sum()>0
    assert torch.isfinite(tr.intervals.motion_scale).all()
    torch.testing.assert_close(tr.model.student.descriptor.center,tr.model.teacher.descriptor.center)
    assert tr.model.student.descriptor.center.abs().sum()>0
    b=collate([ds[(0,j,i)] for j,i in enumerate([0,33])])
    tr.step(b,0)
    path=tmp_path/'test.pt';tr.save(path,1,{'test':True})
    original=tr.step(b,1)
    restored=Trainer(tr.a);assert restored.load(path,{'test':True})==1
    assert restored.step(b,1)==original
    for k,x in tr.intervals.state_dict().items():torch.testing.assert_close(x,restored.intervals.state_dict()[k],rtol=0,atol=0)
    p=prepare(rows_for('val')[0]);ex=Extractor('model',path,device='cpu')
    out=ex.encode(p)
    assert out['_grpraw'].shape==(len(p['spans']),256)
    assert all(np.isfinite(x).all() for x in out.values())


@pytest.mark.parametrize('kind',['near','far'])
def test_full_raw_hidden_perturbation_descriptor_and_encoder(kind):
    torch.set_num_threads(2)
    xy,conf,fps=clip();enc=Encoder(32,2,4,'both').eval()
    for seed in range(4):
        original=pair(xy,conf,fps,kind,seed)
        altered=np.where(original['raw_scrub'][...,None],1e7,xy)
        v2=pair(altered,np.where(original['raw_scrub'],.125,conf),fps,kind,seed)
        def encode(v):
            v=collate([{'near':v,'far':v,'flipped':False}])['near']
            anchor=Fe.anchors(v['s_xy'],v['s_obs'])
            desc=descriptors(v['s_xy'],v['s_obs'],v['s_conf'],v['valid_f'],v['times'],anchor,
                             v['s_aff'],v['forbidden'],signature='leadlag')
            ins,pres=Fe.tokens(Fe.frame_features(v['s_xy'],v['s_obs'],v['s_conf'],v['s_jh'],v['s_aff'],anchor),v['valid_f'])
            return desc,enc(ins,pres,v['visible'],descriptors=desc),v
        a,ha,v=encode(original);b,hb,_=encode(v2)
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.testing.assert_close(ha[1],hb[1],rtol=0,atol=0)
        visible=v['visible'].transpose(1,2).flatten(1)
        torch.testing.assert_close(ha[0][visible],hb[0][visible],rtol=0,atol=0)


def test_invalid_descriptor_tokens_never_attention_keys():
    torch.set_num_threads(2);v=batch()['near'];enc=Encoder(32,2,4,'both').eval()
    desc=descriptors(v['s_xy'],v['s_obs'],v['s_conf'],v['valid_f'],v['times'],
                     forbidden=v['forbidden'])
    ins,p=Fe.tokens(Fe.frame_features(v['s_xy'],v['s_obs'],v['s_conf'],v['s_jh']),v['valid_f'])
    bad=copy.deepcopy(desc)
    bad['input'][~desc['valid']]=1e6
    a=enc(ins,p,v['visible'],descriptors=desc);b=enc(ins,p,v['visible'],descriptors=bad)
    torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)


def test_empty_target_loss_and_per_group_balancing(tmp_path):
    tr=Trainer(config(tmp_path,lambda_body=0.,lambda_relations=0.));v=batch()['near']
    v['target'].zero_();v['forbidden'].zero_()
    loss,out=tr.view_loss(v,0)
    assert loss==0 and out['interval_queries']==0
    loss.backward()
    p,_,_,_=layout();err=torch.ones(1,len(p),2);w=torch.ones_like(err)
    assert balanced(err,w)==1
    # Adding query observations to one path must not drown another path/group.
    err[:,p==0]=3;first=balanced(err,w)
    w[:,p==0]*=.1
    torch.testing.assert_close(balanced(err,w),first)


def test_descriptor_input_capacity_and_inactive_centering():
    modules=[DescriptorTokenizer(32,k) for k in ('motion','signature','both')]
    assert len({sum(p.numel() for p in m.parameters()) for m in modules})==1
    v=batch()['near'];desc=descriptors(v['t_xy'],v['t_obs'],v['t_conf'],v['valid_f'],v['times'])
    mod=modules[0];mod.center.fill_(7)
    before=mod(desc)[0]
    change=copy.deepcopy(desc);change['input'][...,12:22]=1e5
    torch.testing.assert_close(mod(change)[0],before,rtol=0,atol=0)


def test_stats_are_path_specific_robust_and_ignore_invalid():
    v=batch()['near'];desc=descriptors(v['t_xy'],v['t_obs'],v['t_conf'],v['valid_f'],v['times'])
    aux=IntervalHeads(16,32);aux.fit([desc])
    original=copy.deepcopy(aux.state_dict());changed=copy.deepcopy(desc)
    changed['motion'][~desc['motion_valid']]=1e9
    changed['signature'][~desc['signature_valid']]=-1e9
    aux.fit([changed])
    for k,x in aux.state_dict().items():torch.testing.assert_close(x,original[k],rtol=0,atol=0)
    p,_,_,_=layout()
    changed=copy.deepcopy(desc);changed['motion'][:,p==0,0]+=10
    aux.fit([changed])
    torch.testing.assert_close(aux.motion_center[0,0],original['motion_center'][0,0]+10)
    torch.testing.assert_close(aux.motion_center[1:],original['motion_center'][1:])


def test_primary_channels_identical_to_reviewed_s1():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'phonojepa_s1'))
    from s1 import features as S1
    v=batch()['near'];kw=dict(aff=v['s_aff'],anchor=Fe.anchors(v['s_xy'],v['s_obs']))
    a,pa=Fe.tokens(Fe.frame_features(v['s_xy'],v['s_obs'],v['s_conf'],v['s_jh'],**kw),v['valid_f'])
    b,pb=S1.tokens(S1.frame_features(v['s_xy'],v['s_obs'],v['s_conf'],v['s_jh'],**kw),v['valid_f'])
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(pa,pb,rtol=0,atol=0)


def test_config_matrix_and_capacity_matching(tmp_path):
    from pathlib import Path
    configs=sorted((Path(__file__).resolve().parents[1]/'configs').glob('*.json'))
    assert len(configs)==16
    models=[]
    for path in configs:
        a=args(['--out',str(tmp_path),'--config',str(path)])
        assert (a.batch,a.d,a.depth,a.dp,a.pdepth,a.stats_clips)==(32,256,6,128,4,640)
        # Test parameter matching at small dimensions; scientific controls keep full dims.
        a.d=32;a.dp=16;a.depth=2;a.pdepth=1;a.device='cpu'
        tr=Trainer(a)
        models.append((a.descriptor_input,sum(p.numel() for p in tr.model.parameters())+
                       sum(p.numel() for p in tr.intervals.parameters())))
    assert len({n for mode,n in models if mode=='none'})==1
    assert len({n for mode,n in models if mode!='none'})==1


@pytest.mark.parametrize('extra',[
    ['--heads','0'],['--lambda-motion','nan'],['--lambda-motion','-1'],
    ['--lambda-ctx','0','--interval-target','latent']])
def test_invalid_scientific_configs_rejected(tmp_path,extra):
    with pytest.raises(SystemExit):args(['--out',str(tmp_path),*extra])
