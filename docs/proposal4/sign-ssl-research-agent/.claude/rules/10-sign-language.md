# Quy tắc chuyên ngành Sign Language

## Đối tượng và phạm vi

Ngôn ngữ ký hiệu là ngôn ngữ tự nhiên với biến thiên ngôn ngữ, vùng và người ký. Không coi chúng là các gesture class có ngữ nghĩa giống nhau toàn cầu. ASL, BSL, DGS, CSL và VSL không hoán đổi bằng cách dịch gloss.

Phân biệt isolated recognition, continuous recognition, dictionary retrieval và translation. Một representation tốt ở isolated clip chưa chứng minh hiểu discourse hay cú pháp. Dictionary retrieval thường có train dictionary labels; nó không tự đồng nghĩa zero-shot class recognition.

Gloss là dạng annotation có giới hạn, không phải mô tả toàn bộ ngôn ngữ. Text/gloss/translation pairs là supervision phải khai báo, kể cả khi đến từ teacher hoặc LLM.

## Thông tin phải giữ

Kiểm tra handshape, orientation, location, movement, hai tay tương tác, timing/coarticulation và non-manual cues như mặt, mouth và thân. Một mục tiêu pose-only có thể thiếu thông tin ngoài target pose; đây là câu hỏi cần đo, không phải kết luận tất yếu về RGB encoder.

Không mặc định signer-specific yếu tố nào cũng nuisance: khác biệt cơ thể, tốc độ và phong cách có thể liên quan ngữ âm/ngữ nghĩa. Tách identity/background không có nghĩa xóa location, orientation, face hoặc mouthing.

## Augmentation

Trước khi dùng augmentation trong positive pair, viết semantic preservation argument và audit samples:
- Flip/chirality có thể đổi nghĩa hoặc tính chấp nhận ngôn ngữ.
- Crop có thể mất mặt, vị trí bàn tay hoặc tương tác hai tay.
- Time reversal có thể đổi dấu; không là positive augmentation mặc định.
- Temporal crop/speed jitter có thể phá movement, onset/offset và ranh giới.
- Rotation/color/background edits có thể tạo artefact hoặc xóa chi tiết.

Áp dụng crop/augmentation nhất quán theo thời gian. Ghi rate/severity, kiểm tra failure cases và giữ baseline augmentation giống nhau. Nếu chưa có người am hiểu ngôn ngữ kiểm tra, ghi độ bất định thay vì xác nhận mọi biến đổi là hợp lệ.

## Data và supervision

Audit signer/video/source/session/channel overlap; không chỉ sample IDs. Split theo clip ngẫu nhiên có thể chia cùng recording sang nhiều tập. Lưu exact duplicate, near-duplicate và kết quả kiểm tra mức video/segment.

So sánh train corpora và eval corpora, kể cả unlabeled data. Nếu pretraining dùng eval videos, đó là transductive setting; phải tách và khai báo. Nếu chỉ có checkpoint upstream và không đủ provenance, ghi contamination là UNKNOWN, không khẳng định clean.

Pose teacher có thể dùng labeled pose/sign data. Không gọi toàn pipeline “không supervision nào”. Khai báo teacher version, training provenance nếu biết, confidence, missing joints và chi phí extraction.

Dùng signer metadata để loss adversarial hoặc annotation phonology để chọn pair là thêm supervision. Có thể nghiên cứu nhánh đó, nhưng phải đổi tên setting và so sánh riêng.

Không dùng dữ liệu hạn chế mà chưa kiểm tra điều kiện sử dụng. Không tự reidentify signers từ mặt. Chỉ phân tích demographic subgroups khi metadata hợp lệ, đủ hỗ trợ thống kê và không suy đoán nhãn.

## Claims và đánh giá

Cross-signer không tự đồng nghĩa cross-language. Cross-language transfer cần protocol rõ label spaces và nguồn language supervision. Visually similar không tự đồng nghĩa semantically equivalent.

Primary metric theo benchmark chính thức; metric mới là supplemental. Per-instance và per-class có thể kể hai câu chuyện khác nhau. Kiểm tra worst groups/hard cases khi dữ liệu cho phép; giữ các hạn chế về quyền truy cập và tính đại diện.

Xin phản biện người am hiểu/Deaf researcher khi quyết định ngữ nghĩa cần chuyên môn, nhưng không bịa việc đó đã diễn ra. Agent vẫn tiếp tục audit kỹ thuật trong khả năng của mình.

