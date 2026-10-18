# Codebase review for proposal14

Reviewed 2026-10-13. The workspace contains multiple prior methods, active runtime
artifacts and unrelated uncommitted changes. The new implementation is isolated at
`method/pose_jepa`; no other method implementation is changed.

| Existing area | Available implementation | Decision for proposal14 |
|---|---|---|
| `method/jepa_baseline` | RGB SignRep reproduction, pose priors, retrieval/ISLR/translation, job machinery | Separate external RGB baseline; its RGB checkpoint does not initialize this encoder |
| `method/phonojepa_s1`, `phonojepa_s2` | Pose stream SSL, EMA/centering, motion/signature targets, real ASLC artifacts | Useful reference for existing resources; their selected joints, inputs and readouts differ from proposal14 |
| `method/signpathlab` | Pose cache/preprocessing, masking, distributed training, evaluation, service workflows | Existing pipeline uses a different target recipe; preserve it |
| `method/signcoord_ssl` | Coordination targets, geometry/motion, model adapters, distributed tests and export | Different method and modalities; preserve it |
| `method/obs_jepa` | RGB/pose extraction and ASLC metadata/protocol helpers | Read-only source of resource locations and verified cache availability |
| `third_party/Uni-Sign` | Local checkout at `28794107e6bbf47e352d2554eb3dd0c8d022840b` | Differs from guide pin; source audit fetches the exact pinned files separately |
| `third_party/SignRep` | Checkout at guide revision `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2` | External RGB reference, preserved |

Observed environment: Python 3.13.5, torch 2.11.0+cu128, numpy 2.5.3, PyYAML 6.0.3,
pytest 9.0.3. The guide's Python 3.10/3.11 + torch 2.5.1 is an example reproducible
pin, not the environment actually executed here. Record the tested environment
separately; do not claim tests ran on 2.5.1. The GPU is an RTX 5880 Ada with existing
workloads; correctness and initial fixture smoke use CPU.

Real resource availability was inspected: ASL-Citizen split CSVs, full 133-joint
Uni-Sign-style `keypoints`/`scores` pickle caches, source videos and prior per-video
FPS/dimension metadata exist locally. A sampled cache has 61 frames, each shaped
`[1,133,2]`/`[1,133]`. These resources are not automatically treated as compatible
training data: conversion, split/provenance and timing audits precede real runs.

SSL here predicts contextual pose latents from missing part-segments. Physical
2D geometry preserves finger/face layout and global motion targets preserve
trajectory information. Representation quality must be measured with frozen
retrieval, ISLR and sequence transfer; a synthetic smoke loss is insufficient.
