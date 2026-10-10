"""Parity: p4 fast path (cv2 resize, GPU normalize, 4-D SDPA, bf16) vs vendor example path
(albumentations Transformation, vendor attention, fp32). Run:
  .venv-signrep/bin/python method/p4_signssl/tests/test_signrep_parity.py
"""
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4 import signrep as S  # noqa: E402
from p4.data import aslc_rows, decode_resized  # noqa: E402

model, info = S.load_signrep("cuda", fast_attention=False)
print(info)
from augmentation.video.base_video_aug import Transformation  # vendor, on path now

tr = Transformation()
rows = aslc_rows("train")[:3]
for r in rows:
    cap = cv2.VideoCapture(r["path"])
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    ref = tr.aug_video(fr).cuda()
    starts = S.window_starts(ref.shape[0])
    S.patch_attention(False)
    a = S.encode_windows(model, ref, starts, amp=False)
    arr, _ = decode_resized(r["path"])
    x = S.normalize_uint8(torch.from_numpy(arr), "cuda")
    print("max pixel diff", (x - ref).abs().max().item())
    S.patch_attention(True)
    b = S.encode_windows(model, x, starts, amp=False)
    c = S.encode_windows(model, x, starts, amp=True)
    cos = lambda u, v: torch.nn.functional.cosine_similarity(u, v, dim=-1)
    print(r["sample_id"], len(starts), "fp32 fast vs vendor cos min", cos(a["features"], b["features"]).min().item(),
          "bf16 vs vendor cos min", cos(a["features"], c["features"]).min().item(),
          "active max abs diff", (a["active"] - c["active"]).abs().max().item())
    assert cos(a["features"], b["features"]).min() > 0.9999
    assert cos(a["features"], c["features"]).min() > 0.99
print("PARITY OK")
