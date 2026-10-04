# Sign Language Representations — Research Lab

Bộ quy tắc để AI agent thực hiện nghiên cứu có kiểm soát từ literature đến paper, bắt đầu từ checkpoints/inference SHuBERT và SignRep đã được người dùng setup. Đây là bộ Markdown điều hành, chưa phải pipeline huấn luyện hay dịch vụ chạy nền. Ngôn ngữ báo cáo: tiếng Việt; thuật ngữ, ID và tên field dùng tiếng Anh để tương thích code.

## Bắt đầu

1. Bộ rules nằm tại `docs/proposal2/research_lab/`; [AGENTS.md cấp proposal](../AGENTS.md) và [AGENTS.md tại đây](AGENTS.md) là điểm vào. Nạp các file này và prompt trong [11_BOOTSTRAP.md](11_BOOTSTRAP.md) một cách tường minh khi làm việc trong `method/`, vì instructions trong `docs/` không tự áp dụng cho thư mục code.
2. Đọc [01_CHARTER.md](01_CHARTER.md), [02_RESEARCH_LOOP.md](02_RESEARCH_LOOP.md), [11_BOOTSTRAP.md](11_BOOTSTRAP.md). Khởi tạo state từ [template](templates/STATE.md) nếu chưa có.
3. Trên máy nghiên cứu, xác minh các đường dẫn dataset và checkpoint sẵn có. Không tải lại corpus hoặc setup lại third-party theo mặc định.
4. Làm literature matrix, inventory và protocol feasibility trước. Chọn 2–3 hypothesis để sàng lọc rẻ; chỉ promote 1 hướng sau pilot có bằng chứng.
5. Đóng băng confirmatory plan trước lần đánh giá test. Chuẩn bị paper dossier chỉ khi evidence đạt gate.

Mục tiêu là chọn và chứng minh một hướng biểu diễn có ích. Không yêu cầu cải thiện leaderboard của cả SHuBERT và SignRep; không hạ chuẩn so sánh khi chuyển sang task khác. Giả định “benchmark bão hòa” phải được kiểm tra bằng evidence, không được xem là fact.

## Các tài liệu

| File | Chức năng |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Quy tắc vận hành theo phase |
| [01_CHARTER.md](01_CHARTER.md) | Phạm vi, tự chủ và tiêu chuẩn khoa học |
| [02_RESEARCH_LOOP.md](02_RESEARCH_LOOP.md) | State machine, gates, vòng lặp và resume |
| [03_LITERATURE.md](03_LITERATURE.md) | Deep research, nguồn và novelty audit |
| [04_IDEA_PORTFOLIO.md](04_IDEA_PORTFOLIO.md) | Các hướng khởi đầu và phép thử bác bỏ |
| [05_DATA_FEATURES.md](05_DATA_FEATURES.md) | Dataset, split, checkpoint và feature contracts |
| [06_EVALUATION.md](06_EVALUATION.md) | Protocol và kiểm tra evaluator |
| [07_EXPERIMENTS.md](07_EXPERIMENTS.md) | Thiết kế, implementation và job control |
| [08_EVIDENCE.md](08_EVIDENCE.md) | Thống kê, claim ledger và quyết định |
| [09_PAPER.md](09_PAPER.md) | Evidence package và review trước paper |
| [10_SOURCE_MAP.md](10_SOURCE_MAP.md) | Nguồn đã kiểm tra và trạng thái xác minh |
| [11_BOOTSTRAP.md](11_BOOTSTRAP.md) | Prompt khởi động, budget và sản phẩm vòng đầu |

## Hồ sơ sống sau khi agent bắt đầu

Các đường dẫn `records/` dưới đây tính từ `method/research_lab/`, tức `/home/haipd/Sign_Representations/method/research_lab/records/`. Thư mục này lưu hồ sơ chung; bộ rules và templates vẫn nằm tại `docs/proposal2/research_lab/`.

| Vị trí mới | Nội dung |
| --- | --- |
| `records/STATE.md` | Phase, budget, active IDs, next action |
| `records/literature/` | Source cards, search log, literature matrix |
| `records/hypotheses/` | [Hypothesis cards](templates/HYPOTHESIS.md) |
| `records/protocols/` | [Protocol cards](templates/PROTOCOL.md), parity evidence |
| `records/experiments/` | [Experiment cards](templates/EXPERIMENT.md), run index |
| `records/decisions/` | [Decision records](templates/DECISION.md) |
| `records/claims/` | Claim ledger và links tới evidence |
| `records/paper/` | [Paper dossier](templates/PAPER_DOSSIER.md) |

Mọi code/config/test/script và implementation mới phải nằm trong một thư mục con `method/<direction_slug>/`, tức `/home/haipd/Sign_Representations/method/<direction_slug>/`. Chọn slug theo hướng nghiên cứu; không đặt code trực tiếp tại `method/` hoặc trong thư mục rules. Mỗi hướng có `AGENTS.md` dẫn về bộ rules này và README riêng.

Không commit video, checkpoint hoặc feature cache. Artifact lớn đặt ở storage riêng được cấp, có manifest tham chiếu. Nếu chưa có storage riêng, dùng thư mục runtime bên trong `method/<direction_slug>/` và loại khỏi Git bằng `.gitignore` tại chính thư mục đó.

## Trạng thái lúc bàn giao

Rules được soạn ngày **2026-10-03** dựa trên nguồn ở `10_SOURCE_MAP.md`. Đã xem cây thư mục và tài liệu dataset của repo; không khảo sát code/phương pháp cũ. Máy nghiên cứu, GPU, dữ liệu thực, checkpoint thực và kết quả parity chưa được truy cập trong phiên soạn rules. **Không có thực nghiệm khoa học nào được báo cáo là đã chạy.**
