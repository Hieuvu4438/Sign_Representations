# Research bundle — 11/10/2026

Đọc trước: [Báo cáo và chiến lược tiếng Việt](../PhonoJEPA_Strategies_Deep_Research_VI.md).

| Tài liệu | Nội dung |
|---|---|
| [LITERATURE_AND_CODE.md](LITERATURE_AND_CODE.md) | 21 bài primary, venue/version, nội dung đã kiểm tra, giới hạn, official code permalinks |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | Scope, tìm kiếm, ba vòng phản biện và requirement audit |
| [local_snapshot/P1_summary.csv](local_snapshot/P1_summary.csv) | Toàn bộ P1 models/readouts và references; tỷ lệ ở dạng 0–1 |
| [local_snapshot/P1.json](local_snapshot/P1.json) | Kết quả và paired CI đã đóng băng |
| [local_snapshot/provenance.json](local_snapshot/provenance.json) | Thời điểm snapshot, git HEAD và SHA256 source/eval |
| [local_snapshot/final_status.json](local_snapshot/final_status.json) | Trạng thái scheduler/eval lần kiểm tra cuối, tách khỏi snapshot so sánh |
| [source_downloads.json](source_downloads.json), [source_supplement.json](source_supplement.json) | PDF URLs, version theo footer, hash, retrieval/access failures và preflight |
| [code_downloads.json](code_downloads.json) | 8 official repository snapshots, commit và danh sách files |
| `sources/` | PDF/text/preflight; `eccv5207` là Path-JEPA do người dùng bổ sung |
| `code/` | Code tác giả chỉ để đọc; không thực thi, không tải checkpoint |

Hai link người dùng gửi đều là Path-JEPA: [ECCV poster](https://eccv.ecva.net/virtual/2026/poster/4270) liên kết [PDF 5207](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf). Trang poster text được lưu ở `eccv4270_page.txt`.

Không thay training source, không dừng/restart tmux hay GPU jobs. Các chiến lược chưa chạy; bundle không phải benchmark mới. Không dùng source snapshots làm package dependencies của project.
