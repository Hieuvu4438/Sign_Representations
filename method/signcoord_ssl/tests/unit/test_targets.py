import numpy as np
import pytest
import torch

from signcoord.data.sampling import nearest_frame_plan
from signcoord.data.transforms import letterbox,in_frame
from signcoord.losses.masked import masked_huber
from signcoord.targets.geometry import local_hands, normalized_geometry
from signcoord.targets.motion import interval_velocity
from signcoord.targets.reliability import fill_short_gaps
from signcoord.targets.schema import ANCHORS, PACKED_RAW, PART_RAW, RAW_NAMES


def raw_pose():
    x = torch.full((1, 8, 133, 2), 40.)
    x[:, :, 5, 0], x[:, :, 6, 0] = 20, 60
    return x, torch.ones(1, 8, 133), torch.ones(1, 8, 133, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)


def test_schema_identity():
    assert len(set(PACKED_RAW)) == 119
    assert ANCHORS == (115,116,110)
    assert [RAW_NAMES[PACKED_RAW[i]] for i in ANCHORS] == ["left_wrist","right_wrist","nose"]
    assert [len(v) for v in PART_RAW.values()] == [21,21,68,9]
    assert RAW_NAMES[PACKED_RAW[0]] == "left_hand_root"


def test_fixed_origin_preserves_torso_translation():
    x,c,o,f = raw_pose()
    x[..., 0] += torch.arange(8)[None,:,None]
    result = normalized_geometry(x,c,o,f,(100,100))
    assert result["normalization_valid"].all()
    assert torch.allclose(result["global"][0,1:,110,0]-result["global"][0,:-1,110,0],torch.full((7,),1/40))


def test_small_or_missing_shoulders_invalidate_not_clamp():
    x,c,o,f = raw_pose()
    o[:, :, 5] = False
    result = normalized_geometry(x,c,o,f,(100,100))
    assert not result["joint_valid"].any()
    assert not result["reliability"].any()
    assert torch.isfinite(result["global"]).all()


def test_local_translation_rotation_and_root_confidence():
    x = torch.randn(2,8,119,2)
    r = torch.ones(2,8,119)
    a, rel = local_hands(x,r)
    b, _ = local_hands(x+10,r)
    assert a["L"].shape[-2] == 20
    assert torch.allclose(a["L"],b["L"],atol=2e-6)
    rotated = torch.stack((-x[...,1],x[...,0]),-1)
    d,_ = local_hands(rotated,r)
    assert not torch.allclose(a["L"],d["L"])
    r[:,:,0] = 0
    _, rel = local_hands(x,r)
    assert not rel["L"].any() and rel["R"].all()


def test_nan_missing_loss_and_gradient_finite():
    pred = torch.randn(2,3,4,2,requires_grad=True)
    target = torch.zeros_like(pred)
    target[0,0,0] = float("nan")
    r = torch.ones(2,3,4); r[0,0,0] = 0
    loss,_ = masked_huber(pred,target,r)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert (pred.grad[0,0,0] == 0).all()


def test_all_invalid_huber_differentiable_zero():
    pred = torch.randn(2,3,4,2,requires_grad=True)
    loss, log = masked_huber(pred,torch.full_like(pred,float("nan")),torch.zeros(2,3,4))
    loss.backward()
    assert loss.item() == 0 and not log["active"]
    assert pred.grad is not None and not pred.grad.any()


def test_bounded_gap_interpolation_and_track_discontinuity():
    x=torch.arange(7.).view(7,1,1).expand(-1,-1,2).clone()
    c=torch.ones(7,1); o=torch.ones(7,1,dtype=torch.bool)
    o[1:3]=False; o[4:6]=False
    tracks=torch.tensor([0,0,0,0,1,1,1])
    out,conf,valid,interp=fill_short_gaps(x,c,o,torch.arange(7.),tracks)
    assert torch.allclose(out[1:3],x[1:3]) and (conf[1:3] == .5).all()
    assert interp[1:3].all() and not valid[4:6].any()
    o[:]=True; o[1:4]=False
    _,_,valid,_=fill_short_gaps(x,c,o,torch.arange(7.))
    assert not valid[1:4].any()


@pytest.mark.parametrize("dt",[.04,.08])
def test_timestamp_derivative(dt):
    times=torch.arange(32)[None].double()*dt
    coords=(times*3)[...,None,None].expand(-1,-1,3,2)
    v,r=interval_velocity(coords,times,torch.ones(1,32,3))
    assert v.shape == (1,31,3,2) and torch.allclose(v,torch.full_like(v,3))
    assert r.shape[1] == 31
    with pytest.raises(ValueError):
        interval_velocity(coords,times*0,torch.ones(1,32,3))


def test_affine_matches_rgb_marker():
    rgb=torch.zeros(1,3,10,20); rgb[:,:,4,7]=1
    out,xy,a,inv=letterbox(rgb,torch.tensor([[[7.,4.]]]),(20,40))
    # Pixel-center mapping gives x14.5,y8.5, the center of the four peaks.
    assert torch.allclose(xy,torch.tensor([[[14.5,8.5]]]))
    peaks=(out[0,0] == out[0,0].max()).nonzero().float().mean(0).flip(0)
    assert torch.allclose(peaks,xy[0,0])
    assert torch.allclose(inv@a,torch.eye(3),atol=1e-6)


def test_letterbox_padding_does_not_validate_out_of_camera_joints():
    rgb=torch.zeros(1,3,100,200)
    xy=torch.tensor([[[50.,110.],[50.,50.]]])
    original=in_frame(xy,200,100)
    _,transformed,_,_=letterbox(rgb,xy,(224,224))
    assert in_frame(transformed,224,224).all()
    assert original.tolist()==[[False,True]]


def test_sampling_short_clip_and_repeat():
    p=nearest_frame_plan(np.arange(8)/25,0,.32)
    assert p["frame_valid"].sum() == 8
    assert not p["frame_valid"][8:].any() and p["frame_index"][-1] == 7
    assert np.all(np.diff(p["requested_timestamp_s"]) > 0)
    p=nearest_frame_plan(np.arange(32)/12.5,0,2.56)
    assert p["repeat_fraction"] > .4
    with pytest.raises(ValueError):
        nearest_frame_plan([0,0],0,1)


def test_24fps_source_on_25hz_grid_has_valid_nearest_quantization():
    p=nearest_frame_plan(np.arange(64)/24,.173,2.,32,25.,allowed_start_s=0.)
    assert p['frame_valid'].all() and p['timing_valid'].all()
    assert p['repeat_fraction']<.10
    assert np.max(abs(p['actual_timestamp_s']-p['requested_timestamp_s']))<=.5/24
    # Preserve an explicit strict tolerance as a diagnostic policy.
    strict=nearest_frame_plan(np.arange(64)/24,.173,2.,32,25.,pts_tolerance_s=.5/25,allowed_start_s=0.)
    assert not strict['timing_valid'].all()


def test_nearest_crop_frame_can_precede_grid_without_crossing_manifest():
    p=nearest_frame_plan(np.arange(64)/24,.173,2.,allowed_start_s=0.)
    assert p['actual_timestamp_s'][0]<p['requested_timestamp_s'][0]
    assert p['actual_timestamp_s'][0]>=0
    bounded=nearest_frame_plan(np.arange(64)/24,.173,2.)
    assert bounded['actual_timestamp_s'][0]>=.173
