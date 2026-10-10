# T01 — Literature review có hệ thống và gap map

## Mục tiêu

Hiểu cơ chế, supervision, protocol và failure modes của lĩnh vực; xác định gap có evidence. Không viết danh sách abstract nối tiếp.

## Hành động

1. Bắt đầu SOURCE_REGISTRY.md; full-read anchor và những method gần nhất. Tìm bản cập nhật tới ngày chạy.
2. Lưu search log. Dùng queries theo nhóm:
   - “sign language self-supervised representation”, masked pose/video, sign priors.
   - “hand-model-aware”, coupling/discrete tokenization, spatial temporal consistency.
   - multi-stream cluster prediction, negative-free SSL, predictive latent video representation.
   - signer/domain generalization, nuisance invariance, phonological compositionality, non-manual signals.
   - reliability/uncertainty of pose supervision và multimodal distillation.
   - video-text/sign-text models để biết thêm supervision có thể giải quyết gì.
3. Đọc theo method families, rồi backward/forward snowballing. Xác minh paper mới bằng nguồn gốc; không tin leaderboard dates là publication dates.
4. Viết paper note bằng template. Với paper dẫn đầu, đọc appendix/protocol và audit code nếu có.
5. Điền matrix với pretraining input/target, teacher/data, labels, initialization, objectives, encoder, inference modalities, temporal unit, evaluation, cost, limitations và read level.
6. Viết synthesis theo axes: thông tin giữ/mất; granularity; temporal/context; inductive priors; nuisance; label dependence; generalization; cost.
7. Lập gap map: evidence → unresolved question → giả thuyết khả thi → prior art đe dọa → test rẻ.
8. Phản biện gap: liệu vấn đề đã giải quyết ở method khác? chỉ thiếu compute/data? evaluation không phù hợp? có metric nào tách được cơ chế?

## Coverage kỳ vọng

Ưu tiên đọc sâu khoảng 12–20 paper phù hợp, gồm anchor, nearest alternatives và cơ chế nền. Đây là mục tiêu lập kế hoạch, không phải quota đạt bằng paper không liên quan.

Bắt buộc phủ các nhóm masked reconstruction/prior; contrastive/negative-free/cluster prediction; video SSL; domain/semantic preservation; evaluation/data provenance. Nếu không tìm được nhóm nào, lưu search scope và khoảng trống.

Closest prior art của mỗi idea dẫn đầu phải FULL_READ trước novelty judgment. Dừng review tạm thời khi thêm hai lượt snowballing không thêm cơ chế/nearest neighbor liên quan; lưu limitations và ngày cập nhật.

## Outputs

- `research/search_log.md`.
- `research/papers/Pxxx.md` cho mỗi nguồn thực đã đọc.
- `research/LITERATURE_MATRIX.md` cập nhật.
- `research/literature_synthesis.md`.
- `research/gap_map.md`.
- Claims và decision log cập nhật.

## Acceptance

- Mỗi gap có source locator hoặc được gắn INFERENCE/UNKNOWN.
- Không trộn RGB inference với pose inference; không trộn SSL-only với text-supervised pretraining.
- Nhận diện ít nhất ba nearest competitors cho hướng dẫn đầu, hoặc giải thích bằng search evidence vì sao ít hơn.
- Tổng hợp mâu thuẫn và blind spots, không chỉ mặt tích cực.
- Không dùng reported scores không cùng settings để xếp hạng method.

Nếu web không có, ghi OFFLINE_LIMITED và xây review dựa nguồn đã cung cấp; không tuyên bố review mới nhất hoặc exhaustive.

