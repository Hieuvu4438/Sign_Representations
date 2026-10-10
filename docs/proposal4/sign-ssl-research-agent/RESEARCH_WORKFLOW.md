# Workflow và các gate nghiên cứu

## Một vòng nghiên cứu

1. Đọc state, budgets, evidence và objection chưa giải quyết.
2. Chọn **một câu hỏi có thể làm thay đổi quyết định**.
3. Tìm literature/code hoặc làm diagnostic rẻ nhất.
4. Cập nhật gap map, hypothesis và giải thích thay thế.
5. Reviewer phản biện bằng evidence, không chỉ ý kiến.
6. Thực hiện phép kiểm tra đã cấp quyền; nếu chưa chạy được, tạo protocol và blocker chính xác.
7. Quyết định CONTINUE / REVISE / REJECT / PIVOT / NEED_RESOURCE.
8. Cập nhật state, ledger và next action; tiếp tục vòng kế trong ngân sách.

Loop không quay về đầu toàn bộ mỗi lần. Chỉ cập nhật phần evidence bị kết quả mới tác động. Tránh quên negative evidence sau compaction.

## Gates

| Gate | Điều kiện đạt | Nếu chưa đạt |
| --- | --- | --- |
| G0 — Scope & evidence | Config được grounding; anchor được đọc; search log và nguồn có version; test boundary rõ | Làm T00/T01; unresolved scope phải ghi UNKNOWN |
| G1 — Baseline validity | Repo/data/metrics audited; baseline run thật được xác thực, hoặc các phần chưa chạy được phân loại rõ | Có thể làm spec giả thuyết; không claim đã reproduce và không bắt đầu confirmatory comparison |
| G2 — Idea defensibility | Gap có evidence; closest prior art được đọc; hypothesis có mechanism và falsifier; objection quan trọng có kế hoạch kiểm tra | REVISE/REJECT, tìm thêm prior art |
| G3 — Experiment readiness | Method spec, baseline runner, data audit, metric sanity, protocol và ngân sách sẵn sàng | Chặn job phụ thuộc; tiếp tục research/spec |
| G4 — Pilot signal | Pipeline ổn; validation cho signal theo tiêu chí pilot; control chưa bác bỏ cơ chế | Sửa một nguyên nhân hoặc bỏ hướng; không mở test để cứu |
| G5 — Confirmatory evidence | Protocol khóa; multi-seed evidence; CI và effect/cost đạt điều kiện; ablation và transfer phù hợp claim | NULL/NEGATIVE/INCONCLUSIVE; narrow claim hoặc PIVOT |
| G6 — Paper readiness | Novelty refresh, evidence-to-claim mapping, reproducibility, limitations và reviewer objections xử lý đủ | Viết proposal/negative-result report, chưa viết result claims |

G1 có trạng thái `AUDIT_ONLY`, `INFERENCE_VERIFIED`, `EVAL_REPRODUCED`, `PRETRAIN_REPRODUCED`. Chỉ `EVAL_REPRODUCED` hoặc baseline evaluation hợp lệ do project xây dựng mới hỗ trợ so sánh thực nghiệm. Mức audit thấp hơn vẫn cho phép đề xuất method/protocol.

Nếu baseline không tái lập đúng reported number, tìm nguồn khác biệt và tạo baseline nội bộ hợp lệ. Không cần sửa mọi khác biệt để “match paper”, nhưng phải khai báo adaptation và không gọi nó exact reproduction.

## Thứ tự lựa chọn thí nghiệm

Ưu tiên expected information trên chi phí:
- Metadata/search/code audit để kiểm tra novelty hoặc khả thi.
- Sanity/perturbation trên features có sẵn, chỉ khi đã được cấp data/checkpoint.
- Frozen retrieval và linear probe để kiểm tra representation.
- Continued pretraining/pilot với matched baseline.
- Full training/fine-tuning sau khi cơ chế có signal.
- Confirmatory holdout sau khi lựa chọn đã khóa.

Nếu cải thiện đến từ pooling/head, so embedding với **cùng** pooling/head. Nếu đến từ additional data/compute, dùng matched control. Nếu claim không cần translation, không thêm SLT pipeline chỉ để làm bảng dài hơn.

## Dừng và pivot

Dừng cycle khi một trong các điều sau xảy ra:
- Giả thuyết bị prior art bao phủ mà không có khác biệt có ý nghĩa.
- Diagnostic/control bác bỏ cơ chế.
- Data leakage hoặc invalid metric làm evidence không dùng được.
- Vượt ngân sách/context/deadline.
- Thiếu nguồn/data/GPU không thể tự giải quyết.
- Hai cycle không thêm evidence hoặc không giải quyết objection chính.
- Hướng đủ evidence cho một draft với scope được giới hạn.

Không đánh dấu hypothesis “accepted” vì loss giảm. Dùng `SUPPORTED_WITHIN_SCOPE`, `REFUTED` hoặc `INCONCLUSIVE`. Scientific conclusion không là task completion status.

## Paper-ready nghĩa là gì

Một đóng góp có thể tập trung vào mechanism, hiệu quả tài nguyên, robustness hoặc evaluation finding. Chọn **một** claim chính và xác định baseline tương xứng.

Cần evidence thật cho claim chính, effect/cost meaningful theo config, controls loại lời giải thích thay thế mạnh nhất và ít nhất một đánh giá transfer/robustness khi claim đòi hỏi. Dataset bổ sung không tự chứng minh mechanism.

Nếu ngân sách chỉ đủ ít seeds/small dataset, có thể tạo pilot research proposal, không gọi confirmatory paper-ready. Không áp dụng máy móc “hai dataset là đủ” hay “+1% là đủ”.

## Artifact mỗi cycle

Tạo `research/cycles/Cxxx.md` với:
- Câu hỏi và active hypothesis.
- Evidence mới và nguồn.
- Phản biện và lời giải thích thay thế.
- Test đã chạy hoặc protocol chưa chạy.
- Kết quả/quyết định, mức chắc chắn và lý do.
- Budget, blocker, next action.
- Ledger/notes/protocol/run paths cập nhật.

Không tạo file rỗng rồi đánh dấu task done.

