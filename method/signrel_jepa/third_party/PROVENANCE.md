# External sources and runtime dependencies

Accessed 2026-10-06. All SignRel source is an independent implementation of the
provided specification. No upstream functions have been copied verbatim. Runtime
imports stay in upstream checkouts; the checkouts and weights are ignored by Git.

## V-JEPA 2 / 2.1 — architecture import and frozen initialization

- Repository: https://github.com/facebookresearch/vjepa2
- Commit: `204698b45b3712590f06245fbfba32d3be539812`
- License: MIT, inspected local `third_party/vjepa2/LICENSE`.
- Consulted: `src/hub/backbones.py`, `src/models/vision_transformer.py`,
  `app/vjepa_2_1/models/vision_transformer.py`,
  `app/vjepa_2_1/models/utils/modules.py`, `app/vjepa/train.py`,
  `app/vjepa_2_1/train.py`, `app/vjepa/transforms.py`, README and training config.
- Import: official `vit_base` constructor; no imported upstream predictor.
- Local adapters: `signrel/models/backbone_adapter.py`; variable visible counts
  are bucketed before upstream calls, so padding never inserts hidden token IDs.
- EMA and query masking are locally implemented from equations, not copied code.
- This upstream revision's hub URL points to `http://localhost:8300`. The adapter
  bypasses the hub loader and loads an explicitly hashed official model-zoo asset.
- Official checkpoint:
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
- SHA256: `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`
- Architecture: V-JEPA 2.1 distilled ViT-B, 768 dimensions; patch 16, tubelet 2,
  native resolution 384, ImageNet RGB normalization, checkpoint `ema_encoder`.
- Strict loading and full/masked forward evidence: `reports/backbone_verification.json`.
- Pretraining is continued SSL/domain adaptation, not representation learning from scratch.

## SignRep — external frozen baseline only

- Repository: https://github.com/ryanwongsa/SignRep
- Existing checkout: `../../third_party/SignRep`, used read-only.
- Commit: `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2`
- License: CC BY-NC-SA 4.0, inspected upstream LICENSE.
- Consulted: `example_usage.py`,
  `models/final_models/FINAL_hiera_latent_model_head_v25_active.py`,
  `augmentation/video/base_video_aug.py`.
- Imported class: official `Model`; constructor arguments follow its released API.
- Local destination: `signrel/eval/baselines.py` is an independent import wrapper.
- No functions copied, monkey-patched or vendored. No SignRep reconstruction
  objective or weights initialize the proposed method.
- Existing checkpoint: `../../checkpoints/signrep/ckpt.pt`; full digest and strict
  key-load result are recorded in each extraction report.
- Source: https://github.com/ryanwongsa/SignRep/releases
- Baseline results are external-pretraining comparisons, not equal-data/compute
  causal comparisons. Released checkpoints may carry source-overlap risks not
  eliminated by the local SignRel pretraining audit.

## SignSeek — availability recheck

SignSeek code availability was also searched on 2026-10-06 via the
[author paper](https://arxiv.org/abs/2609.03695) and its linked author/university
records. No official code/checkpoint link was identified in the inspected sources.
This is a dated availability observation, not proof that no release exists.
No SignSeek result is presented as locally reproduced.

## SHuBERT — optional external baseline, not inherited core code

- Repository: https://github.com/ShesterG/SHuBERT
- Existing checkout: `../../third_party/SHuBERT`, read-only.
- Commit: `cc1929326075bfbad7ad73159b2acf84356059bb`
- Consulted: README / preprocessing documentation.
- README says primarily MIT; no root LICENSE was present in this checkout.
  The bundled fairseq tree has its own LICENSE. No SHuBERT code was copied.
- No SHuBERT result is claimed by the current RGB extractor. Its stream-feature
  input and source clock must be validated before importing comparable features.

## MediaPipe Tasks — structural metadata inference

- Official Python API:
  https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
  (also pose/face task pages).
- Installed package: `mediapipe==0.10.35`; use Tasks API, not removed `mp.solutions`.
- Code project: https://github.com/google-ai-edge/mediapipe (Apache-2.0).
- API calls only; no source files or functions copied.
- Official numbered model URLs and SHA256s: `reports/assets.json`.
- Local implementation: `signrel/preprocessing/landmarks.py`.
- Landmark caches record version, model hashes, source byte hash, raw frame
  indices, dimensions and confidence semantics. Hand/face confidence is gated
  detection, not a fabricated per-joint probability. Anatomical hand assignment
  uses sufficiently confident pose wrists; ambiguous assignments remain invalid.

## WLASL metadata — source-level exclusions

- Official repository: https://github.com/dxli94/WLASL
- HEAD inspected: `ac00e6be631c1a2a486621b65f202219f3964d6b`.
- Asset: `start_kit/WLASL_v0.3.json`; downloader pins the URL to this commit.
- Original downloaded metadata byte digest: `reports/assets.json`.
- Metadata only downloaded; no videos or upstream code copied.

## ASL-LEX metadata — evaluation only

- Existing table: `../../method/p4_signssl/runtime/external/asllex/signdata.csv`.
- Read-only data access; no imports of that method's Python code.
- 2,723 rows with unique `Code`; the local CSV requires Latin-1 decoding.
- Byte digest stored with every phonology evaluation report.
- Official source and restrictions: https://asl-lex.org/download.html
- No ASL-LEX reference videos have been accessed, scraped or reused.

## SignSeek — prior-art guardrail

- Primary source rechecked: https://arxiv.org/html/2609.03695v1
- The local method does not reuse gloss-positive, dominant-articulator supervised
  contrastive paths or reported numbers as locally reproduced results.
- Masked articulator prediction is auxiliary; the proposed hypothesis concerns
  five pairwise visual relation targets under EMA and RGB-only inference.
