# Third-party code, models and data

No upstream source file is copied into `asm_jepa/`. The package is a fresh implementation; the upstream
repositories below were used as references for conventions or are runtime dependencies only.

| Upstream | Pinned commit (from the guide) | How it is used here | License |
|---|---|---|---|
| microsoft/ASL-citizen-code | 17f0148b00ef87a5957ec103da8d10bd8eb1fa23 | CSV column convention (user, filename, gloss) and class-ranking metric idea, re-implemented | MIT |
| facebookresearch/jepa (V-JEPA) | 51c59d518fc63c08464af6de585f78ac0c7ed4d5 | EMA teacher / predictor / cosine schedules as a *pattern*; no code copied | CC BY-NC 4.0 |
| facebookresearch/ijepa | 52c1ae95d05f743e000e8f10a1f3a79b10cff048 | masked context/target pattern; no code copied | CC BY-NC 4.0 |
| Tau-J/rtmlib | 03a1693e59e4f7cd84582c0fb30459b3bf18ad42 | runtime dependency of `cli/extract_pose.py` (`Wholebody`, `det_model`, `pose_model`); the local env uses the Uni-Sign vendored copy 0.0.13 (see requirements-pose.lock.txt) | Apache-2.0 |
| S-JEPA (ECCV 2024, paper) | – | method reference for centred / sharpened CE targets | paper |

Models: YOLOX-m HumanArt detector and RTMW-x whole-body (COCO-133) ONNX files from download.openmmlab.com,
cached under `data/models/rtmw133/` (not committed); SHA256 in `requirements-pose.lock.txt` and per run in
`pose_extract_summary_<split>.json`.

Data: ASL-Citizen (Microsoft Research License Terms, non-commercial research use, no redistribution of the
data or modifications). Nothing from the dataset, pose caches, embeddings or checkpoints is committed;
`data/` and `runs/` are git-ignored. The MIT license of the baseline code does not cover the dataset.
