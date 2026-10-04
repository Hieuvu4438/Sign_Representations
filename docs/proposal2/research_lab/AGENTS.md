# Local research operating rules

Rules này nằm tại `docs/proposal2/research_lab/`, cùng các ràng buộc trong `docs/proposal2/AGENTS.md`. Mọi code/config/test/script và implementation mới phải nằm trong một thư mục con `method/<direction_slug>/` của `/home/haipd/Sign_Representations/method/`; không đặt code trực tiếp tại `method/` hoặc trong thư mục tài liệu rules. `AGENTS.md` của mỗi hướng phải dẫn về bộ rules này vì instructions trong `docs/` không tự áp dụng cho code trong `method/`.

## Luôn đọc theo phase

| Phase | Tài liệu phải nạp |
| --- | --- |
| Khởi động/resume | README, 01, 02, 11 và state hiện hành |
| Literature/idea | 03, 04, 10 |
| Dataset/features | 05, 06 và docs/datasets được phép |
| Implement/experiment | 06, 07, protocol card và hypothesis card đang active |
| Analyze/decide | 08, experiment cards và decision log |
| Write/review | 09, claim ledger, protocol cards và results ledger |

Không cần đọc tất cả log trong mỗi phiên. Nạp index, state và artifact liên quan; tìm theo ID. Không dùng trí nhớ hội thoại thay cho evidence file.

## Hợp đồng điều hành

- Mỗi hypothesis có ID `Hnnn`; protocol có `Pnnn_vN`; experiment có ID duy nhất; decision có `Dnnn`.
- Không overwrite completed run. Thay đổi metric, split, preprocessing, budget hoặc code tạo version/run mới.
- `templates/` trong thư mục rules là mẫu, không chứa kết quả. Tạo các hồ sơ sống tại `method/research_lab/records/`, do agent tương lai khởi tạo; đây là nơi lưu hồ sơ chung, code triển khai vẫn nằm trong `method/<direction_slug>/`.
- Tài liệu rules không phải chương trình scheduler. Chỉ chạy khi có phiên agent/runtime đang hoạt động; không giả định agent tiếp tục sau khi phiên kết thúc.
- Không spawn sub-agent nếu người dùng hoặc môi trường chưa cho phép delegation. Nếu có nhiều agent, khóa quyền ghi theo file/run và giới hạn GPU tổng; một agent chịu trách nhiệm tổng hợp.
- Review vai trò Researcher, Engineer, Protocol Auditor, Statistician và Skeptical Reviewer có thể do cùng agent thực hiện tuần tự. Mỗi vai trò ghi kết luận, rủi ro và chứng cứ riêng; không tự gọi review đó là kiểm toán độc lập.
- Protocol Auditor có quyền dừng measurement khi split/metric/feature provenance chưa rõ. Dừng phần không hợp lệ, tiếp tục literature hoặc debugging cần thiết.
- Mọi gợi ý tự sinh phải ghi `hypothesis`, không ghi như discovery đã được chứng minh.
- Khi một hướng bị loại, lưu lý do và bằng chứng. Chỉ reopen khi có dữ liệu, literature hoặc phép kiểm mới có khả năng thay đổi kết luận.

## Thông tin bắt buộc trong báo cáo phiên

Nêu objective; artifacts/runs thực sự hoàn thành; findings có ID; những điều chưa biết; chi phí thực dùng; quyết định `continue/pivot/kill/hold`; next action; blockers. Không chỉ ghi “đã nghiên cứu sâu” hoặc “có tiềm năng”.
