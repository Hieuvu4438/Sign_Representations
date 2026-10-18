# proposal17: SA-JEPA (chiến lược JEPA không dùng Path-JEPA)

Mục tiêu: đề xuất hướng áp dụng các chiến lược JEPA khác (không phải Path-JEPA) vào học biểu diễn ký hiệu SSL trên skeleton cho retrieval từ điển đóng băng và ISLR, dựa trên chẩn đoán định lượng nút thắt của PLctx full160 và deep research.

Trạng thái 2026-10-17: **chỉ đề xuất, v1.2** (hai vòng review độc lập: R1 2 Critical + 12 Major; R2 xác minh phát hiện lỗi lan NaN trong perturbation, đã sửa; xem §9 của proposal). Chưa train gì. Chỉ có chẩn đoán CPU trên embedding đã có (không đụng test). Chưa có vòng review thứ ba.

| File | Đọc khi |
|---|---|
| **`SA_JEPA_Proposal_VI.md`** | Bắt đầu ở đây: tóm tắt, chẩn đoán (§2), bản đồ chiến lược (§3), giả thuyết (§4), method (§5), arm/phase/cổng (§6), rủi ro, NO-GO, nhật ký review (§9) |
| `diagnostics/README.md` | Script, kết quả JSON, giới hạn, cách tái lập |
| `reviews/R1_independent_review.md`, `reviews/R2_verification_review.md` | Review độc lập vòng 1 (số liệu, code, suy luận, thiết kế) và vòng xác minh |
| `literature/Q1_jepa_strategy_zoo.md` | JEPA học gì (Littwin, Sobal), predictor có điều kiện (IWM, ContentVec), target multi-depth, SIGReg |
| `literature/Q2_nuisance_disentanglement_and_retrieval_geometry.md` | Tách danh tính speaker/signer, loại bỏ tuyến tính, hình học retrieval, positive xuyên mẫu |
| `literature/Q3_sign_ssl_scale_temporal_and_units.md` | Dữ liệu và quy mô cho sign SSL, mục tiêu thời gian, đơn vị rời rạc, readout hoạt động |
| `archive/SA_JEPA_Proposal_VI_v1.0.md` | Bản v1.0 trước review (giữ để đối chiếu với R1) |

Tài liệu trước đó, không lặp lại: `docs/proposal12` (PLctx), `docs/research` (phễu 15 arm, VAL-DICT), `docs/proposal16` (PhonoPath-JEPA, Path-JEPA).
