# Research state

Template: copy to `method/research_lab/records/STATE.md` only when no state exists. Replace `unknown`/`pending` after verification; never fabricate values. File này phải đủ để resume mà không cần lịch sử chat.

## Identity

- Updated at (UTC): pending
- Owner agent/runtime: pending
- Program version / rules commit: pending
- Research repository branch/commit: pending
- Phase: PREPARE
- Machine access: unknown
- Active hypothesis IDs: none
- Active protocol IDs: none
- Latest decision ID: none

## Allocation và tài nguyên

| Field | Value | Nguồn xác minh |
| --- | --- | --- |
| Host / working path | unknown | pending |
| Allocated GPUs / scheduler | unknown | pending |
| CPU/RAM quota | unknown | pending |
| Artifact storage / disk quota / reserve | unknown | pending |
| Checkpoint paths/hashes | unknown | pending |
| Current dataset manifest | unknown | pending |
| Cycle compute ceiling | Default 4 GPU-hours, chưa dùng | 11_BOOTSTRAP.md; allocation còn cần xác minh |
| Compute used / remaining | 0 / 4 GPU-hours cho cycle mới | Cập nhật từ run ledger |
| Download ceiling / used | Default 20 GiB / 0 | 11_BOOTSTRAP.md |
| HPO ceiling / trials used | Default 6 per hypothesis / 0 | 11_BOOTSTRAP.md |
| Overrides và căn cứ | none | pending |

## Jobs và artifacts

| Run ID | H/P IDs | Job/PID/host | Status | Log/checkpoint/output | Resume action |
| --- | --- | --- | --- | --- | --- |
| none | — | — | — | — | — |

Completed artifacts: none verified yet. Invalidated artifacts: none. Không đổi các field này chỉ dựa trên lời nói “đã chạy”; dẫn evidence file/lệnh.

## Blockers/unknowns

| ID | Điều còn thiếu | Tác vụ bị chặn | Việc độc lập vẫn làm được | Cách resolve |
| --- | --- | --- | --- | --- |
| B001 | Máy/data/checkpoint/GPU chưa xác minh trong phiên này | Extraction/training thật | Literature, protocol/fixture design | Inventory trên máy được cấp |

## Queue

| Priority | Action | H/P IDs | Expected information gain | Budget | Prerequisite |
| --- | --- | --- | --- | --- | --- |
| 1 | Xác minh inventory và allocation | pending | Biết setting khả thi | bounded metadata read | Machine access |
| 2 | Literature + novelty audit cho shortlist | pending | Loại trùng/không có nhãn | CPU/browser | Public sources |

## Next action và handoff

- Next action: pending; ghi action cụ thể sau inventory.
- Exact verified command/config: none; không ghi placeholder như command đã chạy.
- Expected output và gate: pending.
- Last finding / evidence IDs: none.
- Stop/resume conditions: pending.
- Handoff summary: pending.
