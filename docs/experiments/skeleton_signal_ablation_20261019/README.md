# Thực nghiệm kiểm tra tín hiệu skeleton cho Sign Video Retrieval

**Trạng thái: kế hoạch v1, chưa triển khai runner mới và chưa train.** Ngày: 19/10/2026. Kế thừa [literature review](../../research/skeleton_signal_retrieval_20261019/README.md).

**Quyết định chính:** vòng đầu train SSL từ scratch trên **ASL Citizen official train: 40.154 video, 35 signer, 2.731 gloss**. Không dùng gloss/text/phonology label trong SSL loss hoặc sampler. Đánh giá bằng frozen video-to-video dictionary retrieval: **gallery = train**, **query = official val: 10.304 video, 6 signer**.

| Câu hỏi | Thực nghiệm / tài liệu |
|---|---|
| Train bộ nào, split nào, đọc pose ở đâu? | [DATA_PROTOCOL_VI.md](DATA_PROTOCOL_VI.md) |
| Train những cấu hình nào, objective/budget ra sao? | [EXPERIMENT_PLAN_VI.md](EXPERIMENT_PLAN_VI.md) |
| Đo những chỉ số gì, tính và diễn giải thế nào? | [METRICS_AND_DECISIONS_VI.md](METRICS_AND_DECISIONS_VI.md) |
| Danh sách run cần chạy | [arms.csv](arms.csv), 42 dòng arm/seed/budget; các candidate chưa chọn có cờ cần binding |
| Các thông số cần giữ cố định | [protocol.json](protocol.json), **specification**, chưa phải config chạy được |
| Bằng chứng dữ liệu local hiện có | [data_snapshot.json](data_snapshot.json) |
| Bảng điền kết quả | [results_template.csv](results_template.csv), metric chưa đo để trống |
| Thời gian nếu chỉ chạy1seed | [TIMING_ESTIMATE_VI.md](TIMING_ESTIMATE_VI.md), phân biệt log cũ và giả định cho code mới |

Trình tự: **audit → smoke/profile → S0–S7 + đối chứng → 3-seed confirmation → đổi objective/temporal module → tổ hợp/leave-one-out → official test và transfer**. Không chạy đồng thời toàn bộ không gian trong literature review.

Initial batch đầy đủ có **14 SSL run ×10k updates**:8 cấu hình chính,4 controls và2 physical-hand arms. Những dòng confirmation/objective/full-budget chỉ mở theo điều kiện trong CSV. Factorial/leave-one-out E4 được đăng ký thành run cụ thể sau khi xác định module sống sót, trước khi chạy chúng;42 dòng hiện tại không phải tổng số run bắt buộc của mọi nhánh.

**Primary metric:** val gloss-level R@1. Secondary: R@5/10, MRR, DCG, macro-gloss/macro-signer R@1; phonology probes, minimal-pair diagnostics, robustness, collapse và chi phí.

**Dataset ngoài:** WLASL2000 chỉ dùng transfer khi đã khóa model; How2Sign là track context/continuous riêng sau vòng isolated; YouTube-ASL/SignVerse chỉ xét scale sau khi có tín hiệu ổn định. Official test ASL Citizen không dùng để chọn arm; lịch sử test đã được sử dụng trong repo phải được công khai.
