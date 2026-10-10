import torch

from signrel.eval.occlusion import occlude_clip

def test_occlusion_is_rgb_only_nested_reproducible_and_exact_zero():
    video = torch.ones(3,8,100,100)
    clean,info = occlude_clip(video,0)
    assert torch.equal(clean,video) and info['actual_area_fraction'] == 0
    small,_ = occlude_clip(video,.04,42,.36)
    large,info = occlude_clip(video,.36,42,.36)
    assert ((small == 0) <= (large == 0)).all()
    assert float((large == 0).float().mean()) == .36 or abs(float((large == 0).float().mean())-.36) < 1e-6
    assert torch.equal(large[:,0],large[:,-1])
    assert torch.equal(large,occlude_clip(video,.36,42,.36)[0])
    assert torch.equal(video,torch.ones_like(video))
