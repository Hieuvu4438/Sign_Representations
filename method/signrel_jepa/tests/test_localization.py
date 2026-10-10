import torch

from signrel.eval.localization import jitter_boxes,box_iou

def test_zero_jitter_exact_and_invalid_regions_remain_invalid():
    boxes = torch.tensor([[[.2,.2,.4,.4]]*4]*3)
    conf = torch.ones(3,4)
    conf[:,2] = 0
    shifted,c = jitter_boxes(boxes,conf,0)
    assert torch.equal(shifted,boxes)
    assert torch.equal(c,conf)
    assert torch.allclose(box_iou(shifted,boxes),torch.ones_like(conf))
    shifted,c = jitter_boxes(boxes,conf,.2)
    assert (c[:,2] == 0).all()
    assert torch.equal(shifted[0],shifted[2])
    assert torch.equal(shifted,jitter_boxes(boxes,conf,.2)[0])
