# Literature — vòng đầu, 2026-10-03

Scope: ASL pretrained representations, frozen/low-label probes, temporal
aggregation/compute và missing-stream diagnostics. Đây là bounded search,
không phải systematic/comprehensive review. Nguồn public gốc và code được đọc;
không dùng legacy hoặc báo cáo/phương pháp cũ của project.

[Search log](SEARCH_LOG.md), [matrix CSV](matrix.csv), source cards tại `sources/`.
Trạng thái đọc tách abstract/methods/selected evaluation/code. Bảng matrix có
đủ fields của 03_LITERATURE; `unknown` không có nghĩa là work không làm việc đó.
Không đưa reported scores vào results ledger của lab.

## Những collision đã ảnh hưởng quyết định

- H001: layer mixing/LoRA không tự là mới (S01); adapters temporal có closest
  work L02. Chỉ thực hiện failure diagnostic trước khi đề xuất method.
- H002: activity-weighted pooling đã là baseline SignRep (S03); learned
  downsampling có S12. Offline window-cost frontier là câu hỏi analysis,
  không phải streaming hoặc novel pooling mặc định.
- H003: articulator masking/latent prediction đã có L01; SHuBERT đã so sánh
  masking trong pretraining (S01). Reliability cho natural tracking failure
  cần measured failure, masks đáng tin và strong controls.
- Retrieval hướng F giữ ngoài active shortlist: L01 §4.1 đã ghi rõ class
  score max over gallery instances; không đánh đồng video-instance ranking.
- Nhãn phonology hướng C chưa audit local; S11 đã dùng gloss-corrected
  WLASL, khác original 2.000-class setting. L03 có few-shot/phonological lead
  cần kiểm chứng protocol; không suy từ reported score thành reproduced fact.

Full-text của vài closest mechanisms chưa được review hết; REGISTER và promote
còn pending. Search widening phải nhắm PEFT/frozen low-label temporal order,
missing-modality reliability và method ngoài sign/video để audit novelty.
Không dùng số lượng sources để thay cho gate đọc đủ relevant methods.

AI disclosure: hồ sơ do Codex hỗ trợ tìm/đọc và viết; review nội bộ cùng agent,
không phải external/linguistic validation. S2 programmatic lookup không dùng;
dedup bằng paper ID/title/version; DOI/venue chỉ ghi khi nguồn gốc xác minh.

2026-10-04 D020 refresh: [H003 review](NOVELTY_20261004_H003_r1.md) and S21–S25
add annotation/reliability/tracking controls with explicit partial-read status.
Matrix25 rows =23 distinct papers, one scorer and an additional SignRep version;
16 selected-text reviews/seven leads, not16 fully reviewed papers. No exhaustive
novelty or first-study claim. ASL natural tracking-error labels remain a dependency.

D022 latest refresh: [H002/H004 assessment](NOVELTY_20261004_H002_H004_r3.md),
S19 selected author-PDF and S26–S29. Current matrix29 rows/27 distinct papers/
21 selected-text reads/six leads; one scorer/extra SignRep version retained.
This supersedes counts above, not prior read/fetch history. Both actual H004
completion recipes fail practical gate; no new method/contribution promoted.
