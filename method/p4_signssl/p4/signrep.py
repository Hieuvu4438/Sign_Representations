"""SignRep loading and window inference for the proposal4 project (p4_signssl).

The vendored source in third_party/SignRep is imported read-only. The only runtime
change is an optional attention patch that flattens the 5-D q/k/v tensors to 4-D
before scaled_dot_product_attention (numerically the same op, faster kernel); it is
checked against the vendor path in tests/test_signrep_patch.py.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
VENDOR = REPO / "third_party" / "SignRep"
CKPT = REPO / "checkpoints" / "signrep" / "ckpt.pt"
CKPT_SHA256 = "f8be8ca4"  # prefix check; full digest recorded in run manifests

MODEL_PARAMS = {
    "ckpt_dir": None,
    "backbone_params": {"drop_path_rate": 0.1, "num_cls_tokens": 0},
    "out_dim": 768,
    "mask_ratio": 0.0,
    "init_weights": False,
    "cls_head_name": "models.head_models.dict_head_v6_FINAL_thoughtful",
    "cls_head_params": {
        "latents_params": {"hidden_dim": 512, "out_dim": 384},
        "total_frames": 16,
        "projection_params": {
            "full_body_angles": {"hidden_dim": None, "output_dim": 22 * 2},
            "full_right_angles": {"hidden_dim": None, "output_dim": 41 * 2},
            "full_left_angles": {"hidden_dim": None, "output_dim": 41 * 2},
            "body_pose": {"hidden_dim": None, "output_dim": 61 * 3},
            "right_pose": {"hidden_dim": None, "output_dim": 21 * 3},
            "left_pose": {"hidden_dim": None, "output_dim": 21 * 3},
            "body_dist": {"hidden_dim": None, "output_dim": 12 * 22 * 3},
            "right_dist": {"hidden_dim": None, "output_dim": 5 * 11 * 3},
            "left_dist": {"hidden_dim": None, "output_dim": 5 * 11 * 3},
        },
    },
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _import_vendor():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import models.final_models.FINAL_hiera_latent_model_head_v25_active as final_mod
    import models.head_models.dict_head_v6_FINAL_thoughtful as head_mod
    import models.hiera.hiera_clstoken as hiera_mod

    for m in (final_mod, head_mod, hiera_mod):
        assert str(VENDOR) in str(Path(m.__file__).resolve()), m.__file__
    return final_mod, head_mod, hiera_mod


def _fast_attn_forward(self, x):
    B, N, _ = x.shape
    nw = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
    qkv = self.qkv(x).reshape(B, -1, nw, 3, self.heads, self.head_dim).permute(3, 0, 4, 2, 1, 5)
    q, k, v = qkv[0], qkv[1], qkv[2]  # B, heads, nw, n, hd
    if self.q_stride > 1:
        q = q.view(B, self.heads, nw, self.q_stride, -1, self.head_dim).max(dim=3).values
    nq, nk = q.shape[3], k.shape[3]
    o = F.scaled_dot_product_attention(
        q.reshape(B, self.heads * nw, nq, self.head_dim).contiguous(),
        k.reshape(B, self.heads * nw, nk, self.head_dim).contiguous(),
        v.reshape(B, self.heads * nw, nk, self.head_dim).contiguous(),
    ).view(B, self.heads, nw, nq, self.head_dim)
    o = o.transpose(1, 3).reshape(B, -1, self.dim_out)
    return self.proj(o)


def patch_attention(enable=True):
    _, _, hiera_mod = _import_vendor()
    cls = hiera_mod.MaskUnitAttention
    if not hasattr(cls, "_vendor_forward"):
        cls._vendor_forward = cls.forward
    cls.forward = _fast_attn_forward if enable else cls._vendor_forward


def load_signrep(device="cuda", checkpoint=CKPT, fast_attention=True, check_sha=True):
    final_mod, head_mod, _ = _import_vendor()
    model = final_mod.Model(**copy.deepcopy(MODEL_PARAMS))
    if not isinstance(model.cls_head, head_mod.ProjectionModel):
        raise RuntimeError("vendor constructor silently skipped the prior head")
    info = {"checkpoint": str(checkpoint)}
    if check_sha:
        digest = sha256_file(checkpoint)
        if not digest.startswith(CKPT_SHA256):
            raise RuntimeError(f"unexpected checkpoint digest {digest}")
        info["sha256"] = digest
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    res = model.load_state_dict(state["model"], strict=True)
    info["missing"], info["unexpected"] = list(res.missing_keys), list(res.unexpected_keys)
    patch_attention(fast_attention)
    info["fast_attention"] = fast_attention
    return model.to(device).eval(), info


def normalize_uint8(frames_u8, device):
    """frames_u8: [T,224,224,3] uint8 (RGB) -> [T,3,224,224] float normalized on device."""
    x = frames_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def window_starts(n_frames, length=16, stride=2):
    """Start indices of full windows, as in the upstream example (partial tail dropped)."""
    return list(range(0, n_frames - length + 1, stride)) if n_frames >= length else []


@torch.no_grad()
def encode_windows(model, frames, starts, length=16, batch=64, amp=True):
    """frames: [T,3,224,224] normalized (device). Returns dict of [W,...] float32 CPU tensors."""
    outs = {"features": [], "latent": [], "active": []}
    idx = torch.arange(length, device=frames.device)
    for i in range(0, len(starts), batch):
        st = torch.tensor(starts[i : i + batch], device=frames.device)
        clip = frames[st[:, None] + idx[None]]  # w, L, 3, H, W
        clip = clip.permute(0, 2, 1, 3, 4).contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and frames.is_cuda):
            y = model(clip)
        outs["features"].append(y["features"].float().cpu())
        outs["latent"].append(y["latent"].float().cpu())
        outs["active"].append(y["cls_active"].float().cpu())
    return {k: torch.cat(v) if v else torch.zeros(0) for k, v in outs.items()}
