# Đề nghị review chuyên gia ASL — chưa phải annotation đã hoàn tất

Mục đích là kiểm tra 30 lỗi diagnostic đã chọn cố định trong
`reports/ncslgr_interval_audit.json`. Đây là mẫu lỗi sau khi đánh giá test,
không đại diện toàn bộ corpus và không được chuyển thành dữ liệu train/validation.
Không có review hoặc agreement nào đã được thực hiện.

Reviewer cần năng lực ASL và phải xác nhận hoặc chỉnh sửa guideline này trước khi
annotation. Không suy diễn chức năng ngữ pháp chỉ từ nét mặt, gloss hoặc bản dịch.

## Quy trình đề nghị

1. Đợt đầu cung cấp video và `review_template.csv`. Giữ nhãn tham chiếu trong
   `source_packets.jsonl`, prediction, metrics và kiểu lỗi khỏi reviewer ở đợt này.
   Form trống không đồng nghĩa phenomenon vắng mặt.
2. Kiểm tra identity, view, decode và mức nhìn rõ các articulator. Hai view có cùng
   frame clock nhưng framing khác nhau; ghi trường hợp không đủ quan sát.
3. Ghi chức năng được quan sát và mức không chắc chắn. Cho phép nhiều chức năng,
   ngoài NEG/WH/YN, hoặc không thể xác định; không buộc chọn nhãn nguồn.
4. Ghi riêng interval của marker, anatomical onset/core/offset nếu có, và đầy đủ
   grammatical scope. Không mở rộng core thành scope bằng duration hoặc bằng quy
   tắc máy. Phase của anatomical event không tự là phase của chức năng ngữ pháp.
5. Chỉ xác nhận absence khi toàn vùng khai báo được review đầy đủ cho phenomenon
   cụ thể. Vùng không rõ, ngoài vùng review hoặc chưa annotated vẫn là unknown.
6. Đề nghị hai người annotation độc lập 10 ID đầu trong `manifest.json`.
   Lưu hai bản gốc và bản adjudicated riêng; người adjudicate giải quyết bất đồng.
   Các ID còn lại cần review; không tự suy diễn agreement từ một người.

## Thời gian và kết quả

Video chuẩn bị là ROI lossless từ nguồn, giữ mọi frame, loại footer; không resample.
Timestamp bắt đầu frame là `frame_index / 30`. Interval đầu ra dùng half-open
`[start_sec, end_sec)`, kiểm tra `0 <= start < end <= duration_sec`. Resolution nguồn
là khoảng 33,33 ms; ghi tolerance mà chuyên gia xác nhận. Tránh đánh giá những đoạn
không có đủ cue nhìn thấy như negative.

Trong `adjudicated.jsonl`, mỗi row cần canonical ID, reviewer/adjudicator IDs,
timestamp review, video SHA256, vùng đã review, function labels, marker intervals,
anatomical phases, grammatical scope, vùng unknown và notes. Lưu nguồn tham chiếu
và mọi sửa nhãn riêng; giữ nguyên annotation lịch sử, protocols và predictions.

Sau khi có review, báo số case, agreement nhãn và độ trùng interval/endpoint với
denominator, tolerance và quy tắc matching rõ ràng. Một agreement cao trên mẫu lỗi
nhỏ không chứng minh gold đầy đủ cho corpus hoặc xác nhận method độc lập.
Nếu sửa gold test, báo audit bằng phiên bản mới; không tune model bằng sửa đổi này.

Hồ sơ ở local workspace. Không gửi video ra ngoài, liên hệ reviewer hoặc đăng ký
dịch vụ tự động. Điều kiện truy cập/redistribution của ASLLRP vẫn áp dụng.
