# T04 — Method specification và protocol trước thí nghiệm

## Mục tiêu

Đưa hypothesis qua G2 thành thiết kế implement được và experiment có thể quyết định. Research-only vẫn hoàn thành spec; runtime readiness ghi riêng.

## Method specification bắt buộc

1. Problem setting và allowed supervision; training inputs/targets và inference inputs.
2. Ký hiệu, shapes và data flow; temporal/spatial tokens nào được giữ/pool.
3. Objective đầy đủ: từng loss, valid masks, normalization, loss weights/schedules.
4. Target/teacher provenance; stop-gradient/EMA, gradient paths và collapse prevention.
5. Training algorithm có pseudocode; init, sampling, batch assumptions và optimization.
6. Inference algorithm; pooling/retrieval head, padding/short-video/zero-activity behavior.
7. Compute/parameter/memory estimate, assumptions và cách đo thật.
8. Changes so với baseline; các phần giữ cố định.
9. Predictions về mechanism và failure modes có thể kiểm tra.

## Protocol bắt buộc

- Primary claim → primary dataset/metric → δ/cap → acceptance/falsification.
- Data/split manifest, duplication/provenance audit và test quarantine.
- Matched-data/initialization/compute baselines.
- Simple control trước sophisticated version.
- Ablation mỗi thành phần; diagnostic temporal/non-manual/nuisance phù hợp claim.
- Pilot và confirmatory dùng evidence khác; seeds/trials/chọn checkpoint cố định.
- Sample/training uncertainty, cluster resampling và comparison procedure.
- Secondary/transfer evaluation chỉ theo scope.
- Resource plan và stop rule.

## Hành động

Viết method trước code. Reviewer kiểm tra dimensions, target validity, label leakage, what inference needs và chi phí teacher. Đề xuất bản MVP chỉ có một thay đổi cơ chế. Nếu objective đề xuất quá nhiều thành phần, bỏ đến mức thử được hypothesis chính.

Lưu protocol version DRAFT cho pilot; chỉ LOCKED khi config/data/metric/code/compute sẵn sàng. Mọi amend ghi version và rationale trước outcomes liên quan.

## Outputs

- `research/method_spec.md`.
- `research/protocols/EXxx-vN.md` dùng template.
- `research/ablation_matrix.md`.
- `research/implementation_plan.md`.
- `research/resource_request.md` nếu còn blocker.
- Claims/decisions cập nhật.

## Acceptance

Một experimentalist khác có thể implement từ spec mà không cần hỏi author intent. Protocol định nghĩa cả khi thành công và khi bác bỏ. Unknown shapes/data/cost được đánh dấu; không khóa protocol với field critical UNKNOWN.

Trong research-only, có thể DONE phần design nếu assumption có kiểm soát; G3 vẫn NOT_READY khi thiếu runtime baseline/data/budget.

