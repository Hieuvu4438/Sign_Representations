# 03 — Deep research và novelty audit

## Quy trình tìm nguồn

1. Bắt đầu từ hai paper được cung cấp, official repos và supplementary. Xác minh mapping paper ↔ repo ↔ checkpoint. Đọc methods, evaluation, ablation và limitations; abstract chỉ để sàng lọc.
2. Backward search: các paper làm nền cho mechanism/dataset/evaluator. Forward search: công trình sau đó, issue/release và benchmark cập nhật.
3. Tìm riêng theo problem, task và setting; không chỉ theo tên SHuBERT/SignRep. Bao phủ sign representation SSL, RGB/pose/multi-stream, lexical/semantic vs temporal representation, transfer, retrieval, label efficiency, non-manual cues, OOD và streaming/efficiency nếu có liên quan.
4. Dùng primary sources: paper/official proceedings, tác giả/project page, dataset release và evaluator repo. Blog hoặc aggregator chỉ làm lead; không dùng làm chứng cứ thuật toán.
5. Ghi exact query, nguồn tìm kiếm, ngày search, phạm vi năm, số item screen và lý do include/exclude. Không khẳng định comprehensive review chỉ từ vài query.

Query khởi đầu, thay năm tại thời điểm chạy:

- `"sign language" "representation" self supervised benchmark`
- `"sign language" frozen features low resource transfer signer independent`
- `"sign dictionary retrieval" cross corpus open set`
- `"sign language" temporal grounding spotting localization representation`
- `"sign language" non manual phonological features compositional generalization`
- `"sign language" efficient streaming causal distillation`
- Tên mechanism dự kiến + `sign language`; sau đó tìm mechanism tương tự trong speech/video để kiểm tra liệu đóng góp chỉ là chuyển tên.

Không loại research gần chỉ vì chưa có code; ghi reproducibility gap. Không xem preprint là conference paper đã được nhận. Kiểm tra version và venue trên nguồn gốc.

## Literature matrix bắt buộc

Một dòng một paper/setting, có các field:

`id, title, authors, year, version, venue_status, url, access_date, full_text_status, official_repo, code_commit, checkpoint_id, sign_languages, modalities, representation_unit, temporal_context, pretrain_data, supervision, known_overlap, downstream_tasks, splits, adaptation_regime, evaluator_source, metrics_and_signature, compute, ablations, limitations, closest_to_H, evidence_locations, confidence, follow_up`.

`representation_unit` là frame/window/clip/utterance; `temporal_context` là bidirectional/causal và khả năng nhìn future. `pretrain_data` ghi video/text/pose/teacher nguồn riêng. `known_overlap` có `verified_disjoint`, `overlap`, `unknown`; không đổi unknown thành disjoint.

Với paper shortlisted, tạo [source card](templates/SOURCE_CARD.md). Trích section/table/page và code path cho claim; lưu tóm tắt bằng lời mình, citation hợp lệ. Không fabricate BibTeX hoặc sao chép dài từ nguồn không có quyền.

## Vòng novelty audit

Trước implementation, trước scale và trước viết paper:

1. Liệt kê 3–5 closest works theo problem, mechanism, supervision và protocol.
2. Viết “claim mới của ta” bằng một câu có thể bác bỏ.
3. Đặt cạnh từng work: đã làm gì, ta khác gì, khác biệt nào được thực nghiệm kiểm chứng, điểm nào chưa xác minh.
4. Tìm phản ví dụ trực tiếp: paper đã xử lý task này, ablation đã chứng minh mechanism, hoặc benchmark không đo claim.
5. Nếu chỉ khác dataset hoặc dùng encoder khác, thu hẹp thành empirical analysis hoặc tìm đóng góp sâu hơn; không gọi đó là novel method theo mặc định.

Phonological prediction, dictionary retrieval và signer robustness đều cần kiểm tra work đã có. `10_SOURCE_MAP.md` có các warning từ nguồn đã xem, kể cả lead năm 2026; chúng chưa thay thế review đầy đủ cho một hướng.

## Độ sâu và điều kiện đủ để bắt đầu pilot

Default vòng đầu screen khoảng 20–30 item liên quan, đọc sâu 8–12 nguồn có giá trị quyết định, gồm tất cả closest works tìm được và dataset/evaluator chính. Đây là quota tham khảo để tránh review quá nông, không phải chuẩn chất lượng hay điều kiện dừng cứng. Nếu ít nguồn thật sự liên quan, ghi lý do; nếu có nguồn gần chưa đọc thì chưa được chốt novelty.

Cho phép pilot về failure của baseline trước khi review toàn miền hoàn tất. Nhưng không scale hay kết luận novel khi full text của closest work vẫn chưa truy cập. Nếu chỉ có abstract, field methods/evaluation để `unknown`.

Refresh search ít nhất mỗi chu kỳ promote hướng và trước submission draft; mặc định hàng tuần nếu nghiên cứu đang diễn ra liên tục. Agent chỉ refresh khi được chạy, không tự tạo lịch nền từ file này. Dùng ngày thực, không giới hạn mãi ở mốc 2026-10-03.
