# Sign Language SSL — Bộ điều hành nghiên cứu cho Claude

Bộ Markdown này hướng dẫn AI Agent nghiên cứu **Sign Language Representation Learning bằng Self-Supervised Learning**, lấy SignRep làm baseline khởi đầu. Mục tiêu là tìm một đóng góp có cơ chế rõ, kiểm tra độ mới, tái lập baseline và thu thập bằng chứng đủ để quyết định hướng viết paper.

**Đây là bộ thiết lập và tasks; chưa có method được chứng minh, training run hay kết quả mới.** Các ý tưởng trong bộ này là giả thuyết sơ bộ. Nguồn khởi đầu được kiểm tra ngày **2026-10-05**; literature phải được cập nhật khi chạy.

## Dùng với Claude Code

1. Giải nén, giữ nguyên cả thư mục ẩn `.claude/`.
2. Dùng thư mục `sign-ssl-research-agent/` làm project root. Đặt mã baseline trong thư mục riêng như `external/SignRep/`; không trộn các file hướng dẫn vào repo upstream.
3. Mở và chỉnh [PROJECT_CONFIG.md](PROJECT_CONFIG.md): dữ liệu, compute, deadline, phạm vi supervision. Nếu chưa biết, giữ `UNKNOWN`; agent tiếp tục công việc đọc và thiết kế.
4. Mở Claude Code ở project root. Kiểm tra `CLAUDE.md` và rules được nạp bằng công cụ kiểm tra context/memory của phiên bản đang dùng.
5. Dán prompt **Bắt đầu** trong [PROMPTS.md](PROMPTS.md).
6. Phiên tiếp theo dùng prompt **Tiếp tục**; agent phải đọc trạng thái và ledger trước khi hành động.

Theo [tài liệu Claude Code](https://code.claude.com/docs/en/memory), `CLAUDE.md` và `.claude/rules/` cung cấp context cho agent. Chúng hướng dẫn hành vi; chúng không phải cơ chế cưỡng chế ở runtime. Ngân sách thực và quyền chạy công cụ vẫn do môi trường thực thi kiểm soát.

Nếu project đã có `CLAUDE.md` hoặc rules, hợp nhất có chủ đích; không ghi đè file cũ một cách tự động. Giữ một nguồn cấu hình chính thức.

## Dùng với Claude trong chat

Đính kèm **CLAUDE_RESEARCH_PACK.md** và dán prompt bắt đầu. File này gộp toàn bộ Markdown của bộ. Claude trong chat không tự nạp rules trên máy bạn. Nếu không có filesystem, web hoặc môi trường chạy, agent phải báo rõ giới hạn và xuất các bản cập nhật trạng thái dạng văn bản để bạn lưu; không tuyên bố đã ghi file hoặc chạy thí nghiệm.

Một file gộp có thể vượt context của mô hình bạn dùng. Khi đó nạp `CLAUDE.md`, config, rules và task đang làm; cung cấp state/ledger và các nguồn liên quan khi chuyển task.

## Các file cần biết

| File/thư mục | Vai trò |
| --- | --- |
| [CLAUDE.md](CLAUDE.md) | Điểm vào và hợp đồng hành vi của agent |
| [PROJECT_CONFIG.md](PROJECT_CONFIG.md) | Scope, quyền tự chủ, compute, tiêu chí thành công |
| `.claude/rules/` | Kỷ luật bằng chứng, ngôn ngữ ký hiệu, thực nghiệm, tự chủ |
| [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md) | Vòng lặp, các gate và điều kiện dừng |
| [TASKS.md](TASKS.md), `tasks/` | Backlog có dependencies, sản phẩm và tiêu chí hoàn tất |
| [PROMPTS.md](PROMPTS.md) | Prompt bắt đầu, tiếp tục, phản biện, chạy thí nghiệm |
| `context/` | Audit SignRep, nguồn đọc khởi đầu và giả thuyết gợi ý |
| `templates/` | Mẫu paper note, hypothesis, protocol, run, review |
| `research/` | Trạng thái ban đầu và các ledger cần cập nhật |

## Quy trình đầu ra

Research brief → literature matrix → baseline audit → gap map → hypothesis cards → adversarial review → method specification → protocol khóa trước chạy → pilot → confirmatory experiments → paper readiness memo.

Mỗi bước phải tạo hoặc cập nhật artifact có đường dẫn. Số paper đã đọc hoặc số ý tưởng đã viết không thay thế chất lượng bằng chứng. Thành công có thể là loại bỏ một hướng yếu, tìm ra confound, hoặc xác định chưa đủ dữ liệu.

## Giới hạn xuất phát cần chú ý

Audit đã kiểm tra cây repo SignRep tại commit cố định. Repo có model và ví dụ inference, nhưng chưa thấy training/evaluation pipeline đầy đủ trong cây đó. Task tái lập phân biệt rõ inference checkpoint, tái lập đánh giá và tái lập pretraining.

Chi tiết và đường dẫn bằng chứng nằm ở [SIGNREP_AUDIT.md](context/SIGNREP_AUDIT.md). Đừng suy ra thiếu ở snapshot này đồng nghĩa tác giả không có mã ở nơi khác.

## Tự động hóa nhiều phiên

Agent thực hiện nhiều vòng **trong phiên đang hoạt động**, rồi lưu checkpoint khi hết ngân sách hoặc context. File Markdown không tự tạo scheduler, không giữ mô hình chạy sau khi phiên đóng. Runner bên ngoài có thể mở phiên tiếp theo bằng prompt tiếp tục, đọc trạng thái terminal và không khởi chạy trùng thí nghiệm.

Bộ này không yêu cầu cài skill, không có slash command tùy biến và không phụ thuộc hệ thống multi-agent. Các vai researcher/reviewer/experimentalist có thể được thực hiện bằng những lượt đánh giá riêng trong một agent.

