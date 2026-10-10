"""One spatial transform record shared by RGB, points and all region slots."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformRecord:
    crop: tuple  # normalized x1,y1,x2,y2 actually applied after integer rounding
    flip: bool = False

    def apply_points(self, points):
        x1, y1, x2, y2 = self.crop
        q = points.clone()
        q[..., 0] = (q[..., 0] - x1) / (x2 - x1)
        q[..., 1] = (q[..., 1] - y1) / (y2 - y1)
        if self.flip:
            q[..., 0] = 1 - q[..., 0]
        return q

    def apply_boxes(self, boxes, confidence):
        q = self.apply_points(boxes.reshape(*boxes.shape[:-1], 2, 2))
        boxes = torch.cat((q.amin(-2), q.amax(-2)), -1).clamp(0, 1)
        confidence = confidence.clone()
        if self.flip:
            boxes = boxes[..., [1, 0, 2, 3], :]
            confidence = confidence[..., [1, 0, 2, 3]]
        positive = (boxes[..., 2:] > boxes[..., :2]).all(-1)
        return boxes, confidence * positive


def transform_clip(video, boxes=None, confidence=None, *, size=224, augment=False,
                   crop=(0., 0., 1., 1.), force_flip=None):
    """Input video [C,T,H,W] RGB in [0,1]; crop composed with optional signer crop."""
    _, _, h, w = video.shape
    x1, y1, x2, y2 = crop
    if augment:
        scale = float(torch.empty(()).uniform_(0.85, 1.0))
        dx = float(torch.rand(())) * (1 - scale)
        dy = float(torch.rand(())) * (1 - scale)
        cw, ch = x2 - x1, y2 - y1
        x1, y1, x2, y2 = x1 + dx*cw, y1 + dy*ch, x1 + (dx+scale)*cw, y1 + (dy+scale)*ch
    ix1, iy1 = max(0, int(x1*w)), max(0, int(y1*h))
    ix2, iy2 = min(w, max(ix1+1, int(x2*w))), min(h, max(iy1+1, int(y2*h)))
    if ix1 >= w or iy1 >= h:
        raise ValueError("Invalid signer crop")
    flip = (bool(torch.rand(()) < 0.5) if augment else False) if force_flip is None else force_flip
    record = TransformRecord((ix1/w, iy1/h, ix2/w, iy2/h), flip)
    v = F.interpolate(video[:, :, iy1:iy2, ix1:ix2].permute(1,0,2,3),
                      size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    if flip:
        v = v.flip(-1)
    if augment:
        v = (v * float(torch.empty(()).uniform_(0.85,1.15))).clamp(0,1)
        if bool(torch.rand(()) < 0.1):
            v = v.mean(1, keepdim=True).expand_as(v)
        if bool(torch.rand(()) < 0.1):
            # small blur only; preserve hand details
            v = F.avg_pool2d(F.pad(v, (1,1,1,1), mode="reflect"), 3, stride=1)
    v = v.permute(1,0,2,3)
    mean = v.new_tensor([0.485,0.456,0.406])[:,None,None,None]
    std = v.new_tensor([0.229,0.224,0.225])[:,None,None,None]
    meta = None
    if boxes is not None:
        b, c = record.apply_boxes(boxes, confidence)
        meta = {"boxes": b, "confidence": c}
    return (v-mean)/std, meta, record
