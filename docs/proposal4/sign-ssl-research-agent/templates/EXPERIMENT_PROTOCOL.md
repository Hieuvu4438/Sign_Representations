# EXxx-vN — Experiment protocol

## Status và khóa

| Trường | Giá trị |
| --- | --- |
| Status | DRAFT |
| Date / author / hypothesis ID | UNKNOWN |
| Pilot hay confirmatory | UNKNOWN |
| Code commit/content hash | UNKNOWN |
| Config path/hash | UNKNOWN |
| Lock time và artifacts lúc khóa | UNKNOWN |
| Protocol amendments | Chưa có |

Status: DRAFT / LOCKED / RUNNING / COMPLETED / INVALIDATED. Không LOCKED khi critical fields UNKNOWN. Nếu sửa sau kết quả, tạo version mới và đánh dấu exploratory impact.

## Research question và decision

- Claim chính và scope:
- Null/alternative prediction:
- Evidence cần để quyết định:
- Minimum meaningful effect δ hoặc non-inferiority ε:
- Runtime/memory/cost cap:
- Pass / refute / inconclusive rules:
- Vì sao threshold có ý nghĩa, và đã chốt trước outcomes nào:

## Data contract

| Trường | Giá trị |
| --- | --- |
| Dataset/version/terms/access | UNKNOWN |
| Pretraining data và supervision | UNKNOWN |
| Teacher/init provenance | UNKNOWN |
| Train/val/test manifest paths+hash | UNKNOWN |
| Valid samples/classes/signers per split | UNKNOWN |
| FPS/timestamp/frames/resolution | UNKNOWN |
| Duplicates/near-duplicates/source overlap | UNKNOWN |
| Missing clips/targets/short clips policy | UNKNOWN |
| Normalization statistics fit on | Train only, trừ benchmark quy định khác phải ghi rõ |
| Test access log | Chưa có |
| Transductive hoặc contamination status | UNKNOWN |

Với retrieval: gallery source/labels, query split, embedding/pooling, activity weighting, self-match exclusion, tie-breaking, zero-active policy và similarity normalization phải cụ thể.

## Conditions và controls

| Condition ID | Change | Init/data/compute | Purpose |
| --- | --- | --- | --- |
| B0 | Validated baseline | UNKNOWN | Reference |
| M0 | One mechanism change | Matched với B0 hoặc ghi mismatch | Main hypothesis |
| C0 | Simplest alternative | Matched cost | Alternative explanation |
| A0 | Remove new component | UNKNOWN | Ablation |
| N0 | Negative/perturbation control | UNKNOWN | Mechanism diagnostic |

Nếu một dòng không phù hợp, thay bằng điều kiện hợp lệ và giải thích. Không chạy để lấp bảng.

## Training/selection contract

- Full command và config:
- Optimizer/LR/schedule/mask/augmentation/loss normalization:
- Steps/epochs/effective batch/precision/gradient accumulation:
- Seeds, planned trials, budget mỗi condition:
- Checkpoint selection rule chỉ dùng validation:
- Số lần chọn/tuning và search space:
- Freeze/test-release rule:
- Early stop criteria cố định:
- Active job ownership và duplicate-run prevention:

## Metrics và uncertainty

- Primary metric, unit, higher/lower better:
- Official implementation/version hoặc verified implementation:
- Per-instance/per-class/ranking definitions:
- DCG/NDCG/MRR/Recall names đúng benchmark; không thay lẫn:
- Evaluation unit và dependency clusters:
- Training-seed variance procedure:
- Paired sample/cluster CI procedure:
- Class-macro bootstrap handling:
- Multiple comparisons và secondary/exploratory labeling:
- Runtime/device/batch/timing warmup/teacher cost measurement:

## Readiness checks

- [ ] Dataset/split audit và supervision hợp lệ.
- [ ] Metric known-answer checks pass.
- [ ] Baseline smoke/eval hợp lệ.
- [ ] Method dimensions/targets/gradients kiểm tra.
- [ ] Simple control và falsifier có thể chạy.
- [ ] Resource config cho phép, cap rõ.
- [ ] Test quarantine và selection history ghi.
- [ ] Artifacts/hash/lock lưu.

## Sau chạy

Run report IDs, raw logs/results paths, validity checks, deviation log và decision. Kết quả planned không được điền vào phần measured.

