# Discovery and preflight (2026-10-05)

Repo: `/home/haipd/Sign_Representations`, parent commit `2a238fb68413292384baefed5785d6f87e136061`
(pre-existing untracked `docs/proposal3/` from the user; not modified).
`method/signrep_spectral/` did not exist before this work.

## Runtime (verified)
- Interpreter: `/home/haipd/Sign_Representations/.venv-signrep/bin/python` (Python 3.13.5).
  The README seed path `artifacts/legacy/.venv-signrep` does **not** exist on this machine.
- torch 2.11.0+cu128, numpy 2.5.3, OpenCV 5.0.0, PyAV 17.1.0, timm 1.0.20, albumentations 2.0.8.
- GPU: 1x NVIDIA RTX 5880 Ada (49 GB); used as a single GPU job at a time.
- Disk: ~63 GB free on `/` (99% used) -> per-window features are not persisted.

## Checkpoint (verified)
- Path: `checkpoints/signrep/ckpt.pt` (the seed path `artifacts/legacy/checkpoints/...` does not exist).
- 222,676,746 bytes, SHA-256 `f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b` (matches spec).
- Root keys `['model']`; strict load into the example constructor: no missing/unexpected keys.
- Original `Model.forward` vs adapter `backbone+fc` on a fixed fixture: max abs diff 0.0 (features, latent).
- Head audit: ProjectionModel present with the 9 published heads (no silent `Identity`).

## Vendor interface facts used
- `hiera_nodec_mae.MaskedAutoencoderHiera.forward(x, mask_ratio, mask, eval=False)` ->
  `(latent, maps, cls)`; latent = mean of tokens (num_cls_tokens=0). 392 mask units, mask True=keep.
- `MaskUnitAttention` passes 5-D tensors to `F.scaled_dot_product_attention`, which falls back
  to the math kernel (~60-68 clips/s inference). A scoped shim folding (heads, windows) into one
  dim gives identical attention (fp32 max abs diff 3.8e-6, within the spec 1e-5 tolerance) and
  ~185 clips/s in bf16. Used for every arm (B0 included); vendor files unchanged.
- Upstream preprocessing = cv2 RGB, albumentations Resize(224) INTER_LINEAR on uint8, ImageNet
  normalize. Our cv2.resize + normalize matches the vendored Transformation (test, atol 1e-5).
  Pixels come from PyAV (needed for PTS); cv2 vs PyAV YUV->RGB differences not quantified.

## Data (ASL Citizen local copy, verified by `runtime/data_audit_001`)
- Splits CSV: train 40,154 / val 10,304 / test 32,941 videos; 2,731 glosses in every split;
  participants 35 / 6 / 11, disjoint across splits. Labels from `label_maps.json` (0..2730).
- Every video and keypoint file exists; pose frame count == decoded frame count for all videos.
- Decoded PTS strictly increasing and uniform (<=5% deviation) everywhere.
- Excluded: 1 train + 17 test videos shorter than 16 frames.
- Pose schema: COCO-WholeBody 133 keypoints, 2-D, image-normalized; scores are unnormalized
  backend scores (~0.3..10), not probabilities; frames with 0 or >1 persons occur (~2%).
  Pose backend/model identity is **unknown** (no provenance file in the dataset folder).
- Hand mapping: COCO-WholeBody left hand 91-111, right 112-132 (anatomical labels from the backend;
  mirroring not verifiable). No hand-identity tracking -> `track_valid` = single-person frame.

## Unknowns
- Whether ASL Citizen was part of SignRep pretraining (affects absolute numbers, not B1 vs M1).
- Exact original SignRep target semantics (angles/distances/body) — not used (native extension).
