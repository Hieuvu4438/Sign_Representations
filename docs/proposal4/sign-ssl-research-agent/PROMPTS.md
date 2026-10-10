# Prompt sử dụng

Các đoạn dưới là prompt để dán vào Claude. Chúng không giả định có custom slash commands.

## Bắt đầu

~~~text
Đọc CLAUDE.md, PROJECT_CONFIG.md, .claude/rules/ và các file state/ledger trong research/.
Mục tiêu là nghiên cứu chuyên sâu Sign Language Representations bằng Self-Supervised Learning,
dùng SignRep làm anchor, rồi phát triển một hướng method có đóng góp kiểm chứng được cho paper.

Bắt đầu T00, tiếp tục các task không bị chặn trong TASKS.md. Thực hiện tối đa số vòng
đã cấu hình cho phiên này. Tìm và đọc nguồn chính thức, cập nhật literature, audit repo,
phân tích gap, tạo hypotheses, tìm prior art gần nhất và phản biện từng hypothesis.
Không chỉ liệt kê paper hoặc module.

Không chạy training hoặc tải dataset/checkpoint khi config chưa cấp ngân sách.
Nếu compute/data thiếu, tiếp tục toàn bộ công việc nghiên cứu và thiết kế có thể làm được.
Kết thúc bằng checkpoint, quyết định có căn cứ và next action cụ thể.
Không báo kết quả thực nghiệm hoặc novelty chưa được kiểm chứng.
~~~

## Tiếp tục

~~~text
Tiếp tục project từ research/STATE.md, MEMORY.md, các ledger và DECISION_LOG.md.
Kiểm tra artifact, active job và config thật trước khi tiếp tục; không khởi chạy trùng run.
Hoàn tất next_action trước, sau đó chọn task sẵn sàng có giá trị thông tin cao nhất.
Nếu nguồn mới hoặc kết quả mới bác bỏ giả thuyết cũ, cập nhật quyết định và hạ claim.
Chạy số vòng trong ngân sách phiên hiện tại và lưu checkpoint trước khi kết thúc.
~~~

## Phản biện hướng dẫn đầu

~~~text
Đọc hypothesis card, closest prior art, protocol và evidence của hướng đang dẫn đầu.
Làm một lượt reviewer độc lập với lượt đề xuất theo nghĩa quy trình: không sửa idea
trước khi ghi review. Nêu rõ đây là review nội bộ bởi AI, không phải người đánh giá độc lập.

Viết lập luận mạnh nhất cho việc reject hướng này: novelty, supervision, leakage,
confounds, khả năng giữ ngữ nghĩa dấu, compute fairness và chất lượng baseline.
Mỗi objection phải có phép kiểm tra hoặc yêu cầu bằng chứng cụ thể.
Đề xuất thí nghiệm phân biệt method với lời giải thích thay thế đơn giản nhất.
Ra quyết định REJECT, REVISE hoặc PILOT; không tối ưu diễn đạt để che thiếu evidence.
~~~

## Cho phép triển khai và pilot khi đã cấu hình

~~~text
Đọc PROJECT_CONFIG.md và protocol đã khóa. Nếu experiment_enabled và tài nguyên,
dataset, split, metric, baseline, acceptance criteria đã sẵn sàng, thực hiện T04 và T05.
Xác thực method spec, xây baseline runner trước, kiểm tra metrics và data pipeline,
chạy smoke test rồi pilot có kiểm soát.
Không mở test để chọn method. Ghi chính xác config, command, seed, commit/hash,
logs, checkpoint, runtime và kết quả validation vào ledger.
Nếu gate không đạt, sửa hoặc bỏ hướng theo protocol và lưu cả negative result.
~~~

## Đánh giá khả năng viết paper

~~~text
Thực hiện T06 với claim ledger và kết quả thật.
Kiểm tra từng claim: novelty cập nhật, baseline công bằng, evidence về cơ chế,
confirmatory uncertainty, generalization tương ứng scope, chi phí và reproducibility.
Xuất paper_readiness.md với quyết định READY_FOR_DRAFT, NEED_MORE_EVIDENCE hoặc PIVOT.
Nếu chưa có runs thật, chỉ viết proposal và experiment plan.
Không viết abstract kiểu “we outperform” hoặc result table dự đoán.
~~~

