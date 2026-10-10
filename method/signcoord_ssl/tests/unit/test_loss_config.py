import pytest
import torch

from signcoord.losses.geometry import geometry_loss
from signcoord.targets.schema import PART_SLICES


def test_geometry_weights_and_huber_delta_are_applied():
    target=torch.zeros(1,2,119,2)
    pred=target.clone(); pred[...,PART_SLICES['L'],:]=2
    reliability=torch.ones(1,2,119)
    stats={part:{'mean':[0,0],'scale':1} for part in PART_SLICES}
    cfg={'global_part_weights':{'L':1.,'R':0.,'F':0.,'B':0.},
         'local_total_weight':0.,'local_hand_weights':{'L':.5,'R':.5},'huber_delta':1.}
    first,_=geometry_loss(pred,target,reliability,stats,cfg)
    assert first==pytest.approx(1.5)
    cfg['huber_delta']=.5
    second,_=geometry_loss(pred,target,reliability,stats,cfg)
    assert second==pytest.approx(.875)
    cfg['global_part_weights']={'L':0.,'R':0.,'F':0.,'B':1.}
    third,_=geometry_loss(pred,target,reliability,stats,cfg)
    assert third==0
