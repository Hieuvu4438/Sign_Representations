# Khả năng bổ sung giữa PLctx16k và Pose-JEPA30k

## Material Passport

- Ngày:2026-10-16. Phạm vi: phân tích paired ranks đã lưu, không train hoặc chạy fusion mới.
- Verification Status: ANALYZED. Đã tính lại metrics và đối chiếu IDs/thứ tự/nhãn/signers; chưa xác minh kết quả của một model ghép.
- Nguồn: `method/phonojepa/runtime/eval/models/PJ_PLctx_16k_grp.npz` và archive `docs/experiments/pose_jepa_results_20261015/pose_jepa_results_evidence.tar.gz`.
- Members archive: `pose_jepa/runtime/aslc3000_job/evaluation/legacy/eval/models/PJ14_HYBRID_ASLC3000_grp.npz`, `evaluation/raw_features/rows.json`, `evaluation/legacy/report.json` dưới cùng prefix job.
- Pose-JEPA weights/features/environment đã xóa ngày15/10 theo archive receipt; không thể đo fusion50/50 chỉ bằng rank của đúng gloss.

## Bằng chứng thực tế

ASLC300 white512: đúng cùng3.000 gallery và1.500 query, cùng ID/thứ tự/gloss/label/signer/role. Không gộp với full official test hoặc raw retrieval.

| Run | DCG | Rec@1 | Rec@5 |
|---|---:|---:|---:|
| PLctx16k | .686220 |48.9333%|74.0667%|
| Pose-JEPA30k | .514000 |26.8667%|53.9333%|

| Trường hợp top1 | Số query |
|---|---:|
| Cả hai đúng |276|
| Chỉ PLctx đúng |458|
| Chỉ Pose-JEPA đúng |127|
| Cả hai sai |639|

PLctx đúng734/1500; Pose đúng403/1500. Có127 trường hợp Pose đúng trong766 trường hợp PLctx sai. Đây là bằng chứng bổ sung ở mức **dự đoán**, chưa chứng minh127 trường hợp đó được sửa nhờ body/relations.

Một oracle biết nhãn thật và chọn top1 của model đúng sẽ đạt861/1500=**57,40% Rec@1**, tăng8,47 điểm% so với PLctx. Oracle chọn rank thấp hơn của hai model đạt83,20% Rec@5.

**Các điểm oracle không phải kết quả fusion hoặc dự báo mức cải thiện.** Chúng chỉ là giới hạn của một selector biết đáp án và chọn giữa hai dự đoán/readout đã có. Score/feature fusion là một phương pháp khác: có thể cải thiện, làm hỏng câu PLctx đang đúng, hoặc đưa một gloss không đứng top1 ở cả hai model lên top1; vì vậy57,40% không phải upper bound chung cho mọi kiểu fusion.

## Ý nghĩa của probe và hướng kết hợp

R² cho biết prior nào dễ giải mã tuyến tính từ embedding; không tách embedding thành các vector thuần handshape/body/relations. Token hai encoder đã trao đổi context. Không thể quy toàn bộ khác biệt retrieval cho ba probe.

Thiết kế có thể kiểm chứng: giữ PLctx làm nhánh chính; thêm Pose body/relations qua projection/gate; cân bằng mức đóng góp để giữ handshape. Đối chứng ban đầu là score fusion có trọng số cố định, sau đó chọn trọng số trên validation độc lập; structured fusion cần đầu ra nhánh/weights được tái tạo. Giữ cùng chiều cuối và khai báo thêm parameters/data/labels khi học fusion. Nếu dùng nhãn gloss để train adapter thì đó là setting có supervision, cần report riêng với SSL.

Đánh giá cần so PLctx, Pose, fusion và một đối chứng cùng capacity trên cùng gallery/query; đo retrieval và probes để biết có giữ handshape đồng thời cải thiện body/relations hay không. Không chọn fusion weight theo điểm query đang báo rồi coi đó là test độc lập. ASLC300 hiện tại là official validation và đã được xem nhiều lần, nên số này chỉ phục vụ phân tích thăm dò; cần một đánh giá holdout cuối sau khi cố định thiết kế.

## Artifacts và giới hạn

- `complementarity.json`: thống kê, metrics, per-signer rescue counts và giới hạn.
- `paired_ranks.npz`: hai rank arrays cùng query IDs/signers để kiểm tra lại.
- Source/checkpoint/input/readout/seed/budget khác nhau. Phân tích không xác định quan hệ nhân quả hoặc biến thiên nhiều seed.
- Không khôi phục/training lại weights đã xóa; không tuyên bố đã thực hiện fusion50/50.
