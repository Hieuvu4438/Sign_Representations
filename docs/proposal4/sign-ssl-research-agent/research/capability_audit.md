# Capability audit (T00)

Date: 2026-10-05. Checked by the agent on the running machine; every row was observed, not assumed.

| Capability | Observed | Evidence |
| --- | --- | --- |
| Web search / fetch | YES (WebSearch, WebFetch; arXiv HTML readable) | SignRep v1 HTML fetched 2026-10-05 |
| Filesystem write | YES. Project code: `method/p4_signssl/`; research records: this `research/` dir | — |
| GPU | 1× NVIDIA RTX 5880 Ada, 49 GB, **shared** with other users' jobs (≈2.4 GB used and up to 100% util by others at check time) | `nvidia-smi` 15:41 |
| CPU / RAM | 24 cores, 62 GB RAM (≈55 GB available) | `nproc`, `free -g` |
| Disk | Root FS 3.6 TB, **99% full, ≈48 GB free** | `df -h` |
| SignRep runtime | `.venv-signrep` (torch 2.11.0+cu128, cv2 5.0.0, albumentations 2.0.8, numpy 2.5.3) | import check |
| SignRep checkpoint | `checkpoints/signrep/ckpt.pt`, sha256 `f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b`, strict load: 0 missing / 0 unexpected keys; includes backbone, `fc`, prior head (`cls_head`) and `active_head` | `tests/test_signrep_parity.py` output |
| SignRep source | `third_party/SignRep` commit `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2` (2025-11-19) = same commit as context audit | `git log -1` |
| SHuBERT runtime | `.venv-shubert-native`, checkpoints under `checkpoints/shubert_author_hf/models/` (checkpoint_836_400000.pt, dinov2hand/face, mediapipe tasks) — not yet exercised by this project | `ls` |
| Datasets on disk | ASL Citizen (83,399 videos, 47 GB; COCO-WholeBody 2-D keypoint pkl, 19 GB), WLASL2000, MS-ASL (partial), How2Sign, CSL-Daily, Phoenix14T | `docs/datasets/README.md`, `ls` |
| Scheduler / background runner | Plain background processes (`nohup`), monitored by PID/log. No cluster scheduler. Jobs do not survive machine reboot | — |

## Runtime parity check (SignRep)

`method/p4_signssl/tests/test_signrep_parity.py` (3 ASL Citizen train clips): project fast path (cv2 INTER_LINEAR resize → GPU normalization → 4-D SDPA attention patch → bf16 autocast) vs. vendor example path (albumentations `Transformation`, vendor attention, fp32):
- max pixel difference after normalization 7.2e-7;
- fp32 fast-attention vs vendor: min window cosine ≥ 0.99999976;
- bf16 vs vendor fp32: min window cosine ≥ 0.99996; activity output max abs diff ≤ 0.0062.

G1 level after this check: **INFERENCE_VERIFIED** (checkpoint loads and produces vendor-identical outputs). Not yet EVAL_REPRODUCED.

## Constraints that shape the plan

- Disk ≈48 GB: per-window features for ASL Citizen train+val at stride 2 in fp16 ≈ 6 GB; no decoded-frame caches; pretraining checkpoints must be pruned (≤ 3 kept, ≈0.9 GB each with optimizer).
- Shared GPU: throughput measured ≈172 windows/s for inference while other jobs were active; training throughput must be measured, not assumed.
- Test split of ASL Citizen was already scored by earlier, unrelated projects in this repository (records under `method/research_lab/`, `artifacts/legacy/`). For **this** project the test split stays quarantined until a LOCKED confirmatory protocol; the prior exposure is a limitation to disclose (the selection here never reads those results).
