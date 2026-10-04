# 02 — Vòng nghiên cứu, gates và resume

## State machine

| Phase | Việc cần hoàn thành | Điều kiện qua gate |
| --- | --- | --- |
| `PREPARE` | Inventory máy, dataset, checkpoint, budget, trạng thái inference | Các asset cần cho vòng đầu có đường dẫn/provenance; unknown ghi rõ |
| `DISCOVER` | Search log, literature matrix, failure/problem inventory | Nguồn nền và các đối thủ gần nhất đã đọc đủ để chọn câu hỏi |
| `REGISTER` | 2–3 hypothesis cards, scorecard, protocol feasibility | Dự đoán, falsifier, metric, baseline và budget được khóa cho pilot |
| `VALIDATE` | Dataset audit, evaluator parity, feature parity, baseline smoke | Không còn blocker protocol; fixtures và coverage audit đạt |
| `PILOT` | Phép thử rẻ trên train/dev, kiểm tra cơ chế và baseline | Gain/failure giải thích được; có quyết định có chứng cứ |
| `DECIDE` | Continue/pivot/kill/hold, thứ tự queue, novelty refresh | Chọn tối đa 1 hướng chính để scale; giữ hướng dự phòng có lý do |
| `CONFIRM` | Đóng băng plan, multi-seed, test, ablation và transfer cần thiết | Các run có provenance, protocol hợp lệ và uncertainty phù hợp |
| `SYNTHESIZE` | Claim ledger, figures, limitations, paper dossier | Mỗi claim được evidence trực tiếp hỗ trợ; review hoàn tất |
| `PAPER_READY` | Hồ sơ tái lập và draft đủ tiêu chuẩn `09_PAPER.md` | Sẵn sàng để con người review; không đồng nghĩa accepted/submitted |

Chuyển ngược phase là bình thường: lỗi evaluator → `VALIDATE`; novelty bị literature bác bỏ → `DISCOVER`; gain không qua baseline → `REGISTER` hoặc `DECIDE`. Lưu decision cho mọi chuyển làm thay đổi scope.

## Một vòng hoạt động

1. Đọc state và kiểm tra job đang chạy. Resume job hợp lệ trước khi tạo job trùng.
2. Chọn next action có expected information gain cao nhất trong budget còn lại: resolve protocol, đo failure, kiểm tra một falsifier hoặc xác minh novelty.
3. Trước mutation/job, ghi hypothesis/protocol/run ID và output path. Kiểm tra quota và quyền truy cập cần thiết.
4. Thực hiện; kiểm tra artifact có thật và exit status. Lệnh bị lỗi không thành kết quả âm khoa học.
5. Phân tích theo câu hỏi đã đăng ký. Ghi điều đã học, điều chưa biết, alternative explanation và chi phí.
6. Tạo decision nếu thay đổi hướng hoặc phase; cập nhật state, ledger, queue và next action.
7. Nếu phiên còn hoạt động và budget cho phép, tiếp tục action kế. Nếu bị chặn, xử lý nhánh độc lập có ích hoặc checkpoint.

## Chống lặp vô ích

- Mỗi vòng phải resolve một uncertainty hoặc tạo artifact cần cho gate. “Đọc thêm” cần nêu query và quyết định sẽ thay đổi.
- Sau 2 vòng pilot không có evidence mới cho cùng giả thuyết, reviewer đánh giá redesign/kill; đây là default của lab, có thể override bằng decision có lý do.
- Không thử hàng chục module khi chưa xác minh baseline. Không dùng test như feedback để sửa design.
- Tăng compute chỉ khi gain vượt ngưỡng practical đã đăng ký hoặc phép thử lớn thực sự phân biệt các giả thuyết đang còn hợp lý.
- Khi gain chỉ có ở một seed/slice, kiểm tra noise, coverage, preprocessing và head capacity trước khi thêm complexity.
- Nhánh chờ annotation/hardware để `hold`, không để nó khóa toàn bộ portfolio.

## Quyết định và stopping rules

`continue`: có evidence cho problem và mechanism, không có blocker novelty/protocol, thêm phép thử có giá trị. `pivot`: evidence chỉ ra câu hỏi khác cụ thể; tạo hypothesis mới, không sửa ngầm hypothesis cũ. `kill`: vấn đề không tồn tại trong setting, gain do confound, novelty trùng hoặc chi phí không khả thi. `hold`: thiếu dependency nhưng câu hỏi còn hợp lý.

Định nghĩa practical effect trên dev trước chạy (ví dụ mức giảm compute ở quality floor, mức gain macro accuracy/MRR, hoặc coverage ở error target). Không có ngưỡng phần trăm chung cho mọi task. Điều kiện kill phải gắn với giả thuyết và power/budget, không chỉ “p > 0.05”.

## Checkpoint để một agent khác resume

State lưu: phase; active IDs; code/protocol hashes; budget dùng/còn; job IDs/PIDs và ownership; completed/pending artifacts; blockers; last decision; next action với command/config đã xác minh hoặc dependency chưa có. Không để lệnh placeholder trông như executable.

Trước khi đóng phiên, kiểm tra log đã flush, run status đúng và checkpoint resume tồn tại nếu cần. Long-running job chỉ để tiếp tục nếu runtime cho phép; ghi nơi giám sát và không hứa sẽ theo dõi sau khi phiên đã hết.
