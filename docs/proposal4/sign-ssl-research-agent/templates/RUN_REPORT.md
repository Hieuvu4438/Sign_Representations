# RUNxxx — Run report

Mỗi run một ID; không tái dùng ID khi đổi config. Chưa có số thật thì dùng N/A, không dùng 0.

## Identity/provenance

| Trường | Giá trị |
| --- | --- |
| Status | PLANNED |
| Hypothesis/protocol version/condition | UNKNOWN |
| Code commit/hash / config hash | UNKNOWN |
| Command / cwd / environment lock | UNKNOWN |
| Dataset/manifest/checkpoint hashes | UNKNOWN |
| Seed / device / precision | UNKNOWN |
| Job ID / owner / started/ended | UNKNOWN |
| Budget reserved/used | UNKNOWN |
| Exit code / logs / output path | UNKNOWN |

Status: PLANNED / RUNNING / SUCCEEDED / FAILED / CANCELLED / INVALID. SUCCEEDED là process success, không đồng nghĩa hypothesis supported.

## Validity

- Train/eval modes, masking và preprocessing:
- Missing/valid sample counts:
- Leakage/split checks:
- Checkpoint load keys/shapes:
- Deviations khỏi protocol:
- Artifact tồn tại, job exit và raw output đã kiểm tra:
- Runtime memory/cost đo bằng cách nào:
- Có thể dùng cho scientific conclusion không, vì sao:

## Measured outputs

| Split | Metric + unit | Value | Raw artifact | Evidence type |
| --- | --- | --- | --- | --- |
| N/A | N/A | N/A | N/A | NOT_RUN |

Không điền số từ paper vào bảng này. Baseline comparator run IDs và protocol match phải rõ.

## Analysis

- Observed effect và variance/CI khi có:
- So với prediction và simple control:
- Alternative explanations:
- Failure cases/negative results:
- Decision và scope:
- Next test hoặc stop/pivot:
- Ledger/state updates:

