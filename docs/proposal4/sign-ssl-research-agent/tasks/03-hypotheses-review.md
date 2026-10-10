# T03 — Sinh giả thuyết, kiểm tra novelty và phản biện

## Mục tiêu

Biến gap có evidence thành 3–5 giả thuyết có khả năng bị bác bỏ; chọn tối đa hai hướng cho pilot. Không chọn trước rồi tìm literature để hợp lý hóa.

## Hành động

1. Đọc gap map, baseline audit, failure cases có thật và SEED_HYPOTHESES.md.
2. Với mỗi gap, viết causal/mechanistic argument ở mức giả thuyết: representation thiếu gì, vì sao objective mới có thể sửa, dưới điều kiện nào nó thất bại?
3. Viết hypothesis cards bằng template, gồm nearest prior art, mechanism delta, prediction, falsifier, simplest control, supervision và chi phí.
4. Search riêng cho từng idea với synonyms ở sign/video/pose SSL. Full-read nearest works; lập bảng overlap và actual difference.
5. Thiết kế phép kiểm tra phân biệt method với thêm compute, parameters, smoothing, augmentation, data quality và downstream head.
6. Review thành lượt riêng: ghi fatal objections trước rồi mới revise idea. Không dùng chính intuition đề xuất làm evidence.
7. Chấm qualitative 0–5: significance, novelty còn lại, falsifiability, baseline fairness, implementation risk và information/cost. Score không là bằng chứng; fatal objection không được bù bằng tổng điểm cao.
8. Chọn một hướng chính và một dự phòng; loại các hướng còn lại với lý do và evidence.
9. Nếu tất cả yếu, quay lại T01/T02 với query/diagnostic mới; không buộc có “winner”.

## Outputs

- `research/hypotheses/Hxxx.md`.
- `research/novelty_map.md`: closest method, overlap, delta, test và remaining uncertainty.
- `research/reviews/Rxxx.md`.
- `research/hypothesis_ranking.md`.
- Decision log và state cập nhật.

## Acceptance

- Mỗi hypothesis có measurable prediction và outcome có thể khiến nó bị bác bỏ.
- Delta lớn hơn tên module mới; không chỉ “SignRep + attention/contrastive”.
- Hypothesis leading có closest prior art FULL_READ và novelty search log.
- Ít nhất một cheap discriminating experiment/control cho mỗi hướng được chọn.
- Không tuyên bố các seed trong bộ là mới hay ưu tiên bắt buộc.
- Review có phản hồi từng fatal objection: resolved, planned test hoặc unresolved.

G2 chỉ đạt khi hướng có thể bảo vệ như **proposal**. G2 không chứng minh method hiệu quả hoặc novel tuyệt đối.

