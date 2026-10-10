# Reproducibility checklist (project p4_signssl)

All commands run from the repository root `/home/haipd/Sign_Representations`. Python: `.venv-signrep/bin/python` (torch 2.11.0+cu128, opencv 5.0.0, numpy 2.5.3) unless stated; SHuBERT: `.venv-shubert-native/bin/python`.

## Assets

| Asset | Path | Hash / version |
| --- | --- | --- |
| SignRep code | `third_party/SignRep` | commit 06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2 (read-only) |
| SignRep checkpoint | `checkpoints/signrep/ckpt.pt` | sha256 f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b |
| SHuBERT checkpoint | `checkpoints/shubert_author_hf/models/checkpoint_836_400000.pt` | sha256 9087f47602401c07c44a9c70164ca1387373938ac4642149126a43e61c20cd7d |
| ASL Citizen | `/home/dongvk/datasets/ASL_Citizen` (official `splits/{train,val,test}.csv`) | manifest sha256 per split in each `runtime/feats/*/meta.json` (e.g. train 765922a9e50f…) |
| WLASL2000 | `/home/dongvk/datasets/WLASL2000` (`preprocess/nslt_2000.json` splits) | manifest sha in meta.json |
| ASL-LEX 2.0 | `method/p4_signssl/runtime/external/asllex/signdata.csv` (OSF zpha4, file 9nygd) | sha256 080ecc3de4b307a04dd9b2c2583c22bc623a5c731ab869d6e3905d4a3540fbd3; evaluation-only |

## Code (sha256 prefixes at EX05 lock)

`p4/signrep.py` 6427dcc8…, `p4/data.py` 5c8a1df2…, `p4/retrieval.py` f7a6cb1e…, `scripts/extract_signrep.py` 7bde6717…, `scripts/eval_retrieval.py` f2a111c3…, `scripts/diag_order.py` 578ef5c1… (then changed only for CPU-resident gallery after the OOM; see EX05 deviations). Tests: `tests/test_signrep_parity.py`, `tests/test_retrieval.py`, `tests/test_cip_fixture.py`.

## Commands

```bash
# per-window SignRep features (stride 2, 16 native frames), resumable shards
.venv-signrep/bin/python method/p4_signssl/scripts/extract_signrep.py --dataset aslc --split train --out method/p4_signssl/runtime/feats/aslc_train_s2
.venv-signrep/bin/python method/p4_signssl/scripts/extract_signrep.py --dataset aslc --split val   --out method/p4_signssl/runtime/feats/aslc_val_s2
# test only under a LOCKED protocol:
.venv-signrep/bin/python method/p4_signssl/scripts/extract_signrep.py --dataset aslc --split test --allow-test --out method/p4_signssl/runtime/feats/aslc_test_s2
# readouts
.venv-signrep/bin/python method/p4_signssl/scripts/eval_retrieval.py --gallery <train_dir> --query <query_dir> --out <dir> --variants features:activity   # R0 (paper readout)
.venv-signrep/bin/python method/p4_signssl/scripts/diag_order.py    --gallery <train_dir> --query <query_dir> --out <dir> --modes chamfer             # R1 bins2, R2 chamfer@20
.venv-signrep/bin/python method/p4_signssl/scripts/ex05_analysis.py   # confirmatory statistics
```

Pilots: `scripts/train_xwp.py` (H002), `scripts/cip_build_states.py` + `scripts/cip_gate.py` (H003), `scripts/h004_train.py` (H004), `scripts/eval_pilot.py`, `scripts/explore_*.py` (diagnostics). SHuBERT features: `method/p4_signssl/shubert/extract.py` (see its README).

## Determinism / known numeric tolerance

Inference in bf16 autocast, stored fp16: min window cosine vs fp32 vendor path ≥ 0.99996; re-extraction with the sharded script matched the first extraction at min cosine 0.99996. Retrieval ties count against the query.

## Data/test access log

`method/p4_signssl/runtime/logs/TEST_ACCESS.log` (single locked confirmatory evaluation, EX05-v1).
