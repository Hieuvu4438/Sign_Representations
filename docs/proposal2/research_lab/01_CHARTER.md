# 01 — Research charter

## Mục tiêu và definition of success

Tìm một câu hỏi khoa học về Sign Language Representations, xây dựng phương pháp có cơ chế giải thích được, và kiểm chứng utility bằng downstream task/setting phù hợp. Thành công là evidence đáng tin, kể cả khi nó bác bỏ giả thuyết; một paper chỉ được đề xuất khi có đóng góp và kết luận đủ vững. Không tối ưu agent để luôn sinh kết quả tích cực.

Một hướng đủ điều kiện phát triển phải trả lời: ai cần nó, limitation nào đã được đo, vì sao representation hiện tại chưa đáp ứng, phương pháp sửa gì, phép thử nào có thể bác bỏ cơ chế, và claim giới hạn tới đâu.

Phân biệt ba loại đóng góp: **method** (biểu diễn/objective/adapter mới với evidence cơ chế), **evaluation/benchmark** (setting thực tế và protocol có tính tái lập), **analysis** (phát hiện có hệ thống về representation). Chọn một đóng góp chính; công sức engineering không tự tạo novelty. Không dùng tên framework mới cho tổ hợp module quen thuộc nếu thiếu insight.

## Các ràng buộc của chủ đề

- SHuBERT và SignRep là tài nguyên/baseline/teacher có thể tái sử dụng. Không mặc định phải pretrain foundation model từ đầu.
- Downstream ngoài recognition/translation được khuyến khích nếu có ý nghĩa và supervision đúng: transfer, retrieval, localization, ít nhãn, robustness, calibration, efficiency hoặc linguistically grounded analysis.
- Thay dataset/metric đơn thuần không đủ. Nếu xây benchmark mới, phải có vấn đề thực tế, split có lý do, annotation đáng tin, evaluator mở và baseline.
- Không khẳng định general-purpose representation từ một task, một corpus hoặc một ngôn ngữ ký hiệu. Nếu chỉ có một phạm vi hợp lệ, thu hẹp claim.
- Không đánh đồng ASL, DGS, CSL hoặc các sign languages khác; nhãn text cùng nghĩa không chứng minh cùng sign form hay cùng ontology.
- Biểu cảm, mouthings, timing, vị trí tay và phối hợp hai tay có thể mang thông tin ngôn ngữ. Không coi mọi khác biệt signer là nuisance cần xóa.

## Tự chủ được cấp cho các phiên nghiên cứu

Agent có thể tìm/đọc nguồn public; tải checkpoint/features có quyền; kiểm tra tài nguyên hiện có; tạo method mới; chạy smoke/pilot/confirmatory trong budget và cổng chất lượng; sửa lỗi của code mới; ghi report; quyết định continue/pivot/kill. Không cần hỏi mỗi lần chuyển phase hoặc thử hypothesis đã đăng ký.

Chỉ yêu cầu thông tin khi thiếu thực sự ngăn công việc phụ thuộc: đường dẫn không có, dataset cần quyền truy cập, không biết GPU nào được cấp, annotation ontology bất khả xác minh. Trong khi chờ, làm literature, protocol card hoặc fixture không cần dữ liệu. Không tạo vòng xin xác nhận hình thức cho reversible work.

Chỉ dùng tài nguyên được cấp. Không đụng job người khác; không thay driver/global environment; không xóa artifact nguồn; không chi tiền hoặc nộp/chia sẻ ra ngoài. Nếu budget hết thì checkpoint và gửi kế hoạch mở rộng có con số, không tiếp tục vô hạn.

## Evidence và liêm chính

Mọi thông tin có một trong bốn nhãn:

| Nhãn | Ý nghĩa |
| --- | --- |
| `verified` | Có artifact/lệnh/nguồn trực tiếp kiểm tra và vị trí dẫn chứng |
| `hypothesis` | Dự đoán chưa được kiểm chứng |
| `unknown` | Chưa có thông tin đủ để kết luận |
| `blocked` | Cần dependency cụ thể đang thiếu |

Không suy từ README rằng checkpoint dùng đúng preprocessing. Không suy từ load state_dict thành công rằng tất cả trọng số đã nạp. Không suy từ abstract rằng reproduce được method. Không suy từ một p-value rằng đóng góp có giá trị thực tế.

Lưu kết quả âm, lỗi evaluator và drift protocol. Khi phát hiện sai, đánh dấu mọi run/claim bị ảnh hưởng `invalidated`, sửa protocol, chạy lại phần cần thiết. Không lặng lẽ thay số trong bảng.

## Quyền dữ liệu và giá trị sử dụng

Giữ license/terms/provenance với asset. Không phát tán video hoặc feature nếu chưa rõ quyền; embedding không mặc định ẩn danh. Khi claim liên quan non-manual grammar, intelligibility hoặc usefulness cho người dùng Deaf, cần chuyên gia/người ký hiệu đủ năng lực để xác minh nội dung phù hợp; nếu chưa có, chỉ báo cáo phép đo proxy và hạn chế. Không tự thay nhãn chuyên môn bằng gloss/translation hoặc nhãn MLLM rồi gọi đó là ground truth.
