import copy
import numpy as np
import pytest
import torch
from torch import nn
from signrel.models.signrel_jepa import SignRelJEPA
from signrel.models.ema import freeze,update_ema
from signrel.models.region_pooler import token_membership,RegionPooler
from signrel.models.temporal_pyramid import temporal_targets
from signrel.models.relation_projector import RelationProjector
from signrel.losses.latent import latent_loss
from signrel.masks.base import SignMaskGenerator
from signrel.data.transforms import TransformRecord
from signrel.data.manifest import ssl_record,assert_no_labels,write_manifest
from signrel.data.leakage import assert_disjoint,audit_and_filter,youtube_id
from signrel.data.sampler import ResumableBatchSampler
from signrel.preprocessing.articulator_boxes import boxes_from_landmarks
from signrel.engine.checkpoint import save_checkpoint,load_checkpoint


def test_labels_rejected_recursively():
    for key in ("signer_id","gloss","class_id","phonology_label","translation"):
        with pytest.raises(AssertionError):
            assert_no_labels({"meta":[{key:"secret"}]})
        with pytest.raises(AssertionError):
            ssl_record({"sample_id":"x","video_path":"v","source_video_id":"s","split":"ssl_train",key:"a"})
    assert "dataset" not in ssl_record({"sample_id":"x","video_path":"v","source_video_id":"s","split":"ssl_train","dataset":"a"})


def test_teacher_grad_and_projector_training(cfg,batch):
    model = SignRelJEPA(cfg).train()
    out = model(batch)
    out["loss"].backward()
    for module in (model.teacher_encoder,model.teacher_relation_projector):
        assert not module.training
        assert all(p.grad is None and not p.requires_grad for p in module.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.student_relation_projector.parameters())
    assert not out["target_relation_0"].requires_grad
    assert torch.isfinite(out["loss"])


def test_ema_numeric():
    s,t = nn.Linear(2,2),nn.Linear(2,2)
    freeze(t)
    old = copy.deepcopy(t.state_dict())
    update_ema(t,s,0.75)
    for k,v in t.state_dict().items():
        torch.testing.assert_close(v,old[k]*0.75+s.state_dict()[k]*0.25)


def test_ema_both_components(cfg):
    m = SignRelJEPA(cfg)
    old = copy.deepcopy(m.teacher_relation_projector.state_dict())
    with torch.no_grad():
        for p in m.student_relation_projector.parameters():
            p.add_(2)
    m.ema_update(0.8)
    for k,v in m.teacher_relation_projector.state_dict().items():
        torch.testing.assert_close(v,old[k]+0.4)


def test_transform_and_semantic_flip():
    boxes = torch.tensor([[[0.2,0.3,0.4,0.5],[0.6,0.4,0.7,0.5],
                            [0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5]]])
    record = TransformRecord((0.1,0.2,0.9,1.0),True)
    out,conf = record.apply_boxes(boxes,torch.ones(1,4))
    expected_old_lh = torch.tensor([0.625,0.125,0.875,0.375])
    torch.testing.assert_close(out[0,1],expected_old_lh)
    points = record.apply_points(boxes[0,0].reshape(2,2))
    torch.testing.assert_close(points.min(0).values,out[0,1,:2])
    torch.testing.assert_close(points.max(0).values,out[0,1,2:])


def test_region_pool():
    tokens = torch.arange(4.).reshape(1,1,2,2,1)
    membership = torch.zeros(1,1,4,2,2)
    membership[0,0,0,0,0]=1
    membership[0,0,1,1,:]=1
    z,v = RegionPooler()(tokens,membership)
    torch.testing.assert_close(z.flatten(),torch.tensor([0.,2.5,0.,0.]))
    assert v.tolist()==[[[True,True,False,False]]]


def test_invalid_boxes_excluded_from_union():
    boxes = torch.zeros(1,2,4,4)
    conf = torch.zeros(1,2,4)
    boxes[0,0,0] = torch.tensor([0.,0.,0.5,0.5])
    boxes[0,1,0] = torch.tensor([0.,0.,1.,1.])
    conf[0,0,0] = 1
    m = token_membership(boxes,conf,(1,2,2))
    assert m[0,0,0].sum()==1


def test_mask_no_endpoint_leak(cfg,batch):
    m = SignRelJEPA(cfg)
    out = m(batch)
    spec = out["mask"]
    assert not (spec.visible_tokens&spec.hidden_tokens).any()
    for b,meta in enumerate(spec.metadata):
        s,e,a = meta["start"],meta["end"],meta["endpoint"]
        membership = out["membership"][b,s:e,a]>0
        visible = spec.visible_tokens[b].reshape(4,4,4)[s:e]
        assert not (membership&visible).any()
        assert spec.visible_tokens[b].float().mean()>=0.25


def test_hidden_pixels_cannot_affect_context(cfg,batch):
    m = SignRelJEPA(cfg).eval()
    grid = m.student_encoder.grid_shape(batch["video"])
    visible = torch.ones(2,np.prod(grid),dtype=torch.bool)
    visible[:,0]=False
    x = batch["video"].clone()
    changed = x.clone()
    changed[:,:,:2,:8,:8] = 1000
    a = m.student_encoder.forward_tokens(x,visible)["tokens"]
    b = m.student_encoder.forward_tokens(changed,visible)["tokens"]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_relation_invalid_endpoint(cfg,batch):
    batch["articulator_meta"]["confidence"][:,:,2] = 0
    m = SignRelJEPA(cfg)
    out = m(batch)
    assert not out["query_valid_0"][:,:,[1,2]].any()
    # Invalid values contribute nothing to loss.
    p,t = torch.randn(2,3),torch.randn(2,3)
    t[1] = float("nan")
    valid = torch.tensor([True,False])
    torch.testing.assert_close(latent_loss(p,t,valid),latent_loss(p[:1],t[:1],valid[:1]))


def test_zero_regions_fails(cfg,batch):
    batch["articulator_meta"]["confidence"].zero_()
    with pytest.raises(AssertionError,match="Zero-valid"):
        SignRelJEPA(cfg)(batch)


def test_temporal_windows_validity():
    z = torch.arange(4.).reshape(1,4,1,1)
    valid = torch.tensor([[[True],[True],[False],[True]]])
    out,v = temporal_targets(z,valid,2)
    assert v.flatten().tolist()==[True,False,False]
    assert out.flatten().tolist()==[0.5,0.,0.]


@pytest.mark.parametrize("variant",["generic","articulator","relation","combined"])
def test_controlled_variants(cfg,batch,variant):
    cfg["loss"] = {"generic_weight":1.,"articulator_weight":float(variant in {"articulator","combined"}),
                   "relation_weight":float(variant in {"relation","combined"})}
    if variant=="generic":
        cfg["masking"]["mixture"]={"random_multiblock":1.}
    m = SignRelJEPA(cfg)
    out=m(batch)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


def test_rgb_only_inference(cfg):
    out = SignRelJEPA(cfg).eval().encode_video(torch.randn(2,3,8,32,32))
    assert out["global_embedding"].shape==(2,32)
    torch.testing.assert_close(out["global_embedding"].norm(dim=-1),torch.ones(2))


def test_leakage_removes_entire_source():
    rows = [{"sample_id":str(i),"source_video_id":s} for i,s in enumerate(["bad","bad","good"])]
    held = [{"sample_id":"h","source_video_id":"bad"}]
    with pytest.raises(AssertionError):
        assert_disjoint(rows,held)
    clean,report = audit_and_filter(rows,held)
    assert len(clean)==1 and report["removed_sources"]==["bad"]
    assert youtube_id("https://youtu.be/abcdefghijk?t=10")=="abcdefghijk"
    assert_disjoint(clean,held)


def test_sampler_resume():
    whole = list(ResumableBatchSampler(13,2,42,total_batches=11,rank=1,world_size=2))
    resumed = list(ResumableBatchSampler(13,2,42,start_batch=4,total_batches=11,rank=1,world_size=2))
    assert whole[4:]==resumed
    rank0 = list(ResumableBatchSampler(13,2,42,total_batches=11,rank=0,world_size=2))
    assert all({i for i,_ in a}.isdisjoint(i for i,_ in b) for a,b in zip(rank0,whole))


def test_box_gap_handling():
    xy = np.tile(np.array([[[0.2,0.2],[0.4,0.4]]],np.float32),(10,1,1))
    confidence = np.ones((10,2),np.float32)
    confidence[1]=0
    confidence[3:8]=0
    boxes,conf = boxes_from_landmarks(xy,confidence,np.array([0,0]),smooth=1.)
    assert conf[1,0]==1
    assert not conf[3:8,0].any()


def test_checkpoint_resume_matches_uninterrupted(cfg,batch,tmp_path):
    torch.manual_seed(10)
    m = SignRelJEPA(cfg)
    o = torch.optim.AdamW(m.parameters(),lr=1e-3)
    s = torch.optim.lr_scheduler.StepLR(o,1,0.9)
    scaler = torch.amp.GradScaler("cuda",enabled=False)
    def step(model,opt,scheduler):
        opt.zero_grad()
        model(batch)["loss"].backward()
        opt.step()
        model.ema_update(0.9)
        scheduler.step()
    step(m,o,s)
    ck = tmp_path/"state.pt"
    save_checkpoint(ck,m,o,s,scaler,1,cfg,"manifest_hash")
    step(m,o,s)
    reference = copy.deepcopy(m.state_dict())
    m2 = SignRelJEPA(cfg)
    o2 = torch.optim.AdamW(m2.parameters(),lr=1e-3)
    s2 = torch.optim.lr_scheduler.StepLR(o2,1,0.9)
    load_checkpoint(ck,m2,o2,s2,scaler,cfg,"manifest_hash")
    step(m2,o2,s2)
    for k,v in reference.items():
        torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(AssertionError,match="manifest"):
        load_checkpoint(ck,m2,o2,s2,scaler,cfg,"different")
    bad=copy.deepcopy(cfg)
    bad["model"]["relation_dim"]=99
    with pytest.raises(AssertionError,match="config"):
        load_checkpoint(ck,m2,o2,s2,scaler,bad,"manifest_hash")
