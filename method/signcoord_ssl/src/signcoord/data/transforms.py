import torch
import torch.nn.functional as F


def in_frame(xy_px, width, height):
    return (torch.isfinite(xy_px).all(-1) & (xy_px[...,0]>=0) & (xy_px[...,0]<width)
            & (xy_px[...,1]>=0) & (xy_px[...,1]<height))


def letterbox(frames, xy_px=None, output_size=(224, 224)):
    """Same explicit affine for RGB[T,3,H,W] and raw pose[T,J,2]."""
    _, _, h, w = frames.shape
    oh, ow = output_size
    nominal_scale = min(oh / h, ow / w)
    nh, nw = round(h * nominal_scale), round(w * nominal_scale)
    top, left = (oh - nh) // 2, (ow - nw) // 2
    out = F.interpolate(frames.float(), (nh, nw), mode="bilinear", align_corners=False)
    out = F.pad(out, (left, ow - nw - left, top, oh - nh - top))
    # align_corners=False maps pixel centers with a half-pixel offset.
    sx, sy = nw / w, nh / h
    a = frames.new_tensor([[sx, 0, left + (sx-1)/2], [0, sy, top + (sy-1)/2], [0, 0, 1]], dtype=torch.float32)
    xy = None if xy_px is None else xy_px @ a[:2, :2].T + a[:2, 2]
    return out, xy, a, torch.linalg.inv(a)
