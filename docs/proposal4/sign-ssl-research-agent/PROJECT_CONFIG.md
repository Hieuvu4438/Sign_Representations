# Cấu hình nghiên cứu — nguồn chính thức

Sửa file này trước khi cho phép thí nghiệm tốn tài nguyên. Thay đổi trong cuộc trò chuyện phải được ghi lại vào đây và decision log. `UNKNOWN` là chưa biết, không phải quyền ngầm.

## Mục tiêu và phạm vi

| Trường | Giá trị ban đầu |
| --- | --- |
| Domain | Sign Language Representation Learning / Self-Supervised Learning |
| Anchor | SignRep, arXiv:2503.08529v1 và ryanwongsa/SignRep |
| Đóng góp mong muốn | Một cơ chế representation learning có thể kiểm chứng, không chỉ ghép module |
| Primary track | RGB encoder; pose có thể là teacher/target lúc pretraining |
| Supervision chính | Không dùng gloss/text/sign labels trong SSL; downstream train labels được khai báo |
| Primary evaluation đề xuất | Frozen dictionary retrieval + kiểm tra transfer signer/domain |
| Secondary evaluation đề xuất | Linear probe, low-label fine-tuning, full fine-tuning recognition |
| Sign languages | ASL (primary: ASL Citizen; secondary WLASL2000/MS-ASL local copies). Others local (CSL-Daily, Phoenix14T, How2Sign) only if claim needs them |
| Benchmark đề xuất | ASL-Citizen trước nếu tiếp cận được; WLASL2000 bổ sung; NMFs-CSL khi claim non-manual |
| Deadline / venue / độ dài paper | UNKNOWN |
| Baseline repo directory | `third_party/SignRep/` (commit 06f40b5d, read-only); project code `method/p4_signssl/` (user steering 2026-10-05) |
| Ngôn ngữ báo cáo | Tiếng Việt; draft paper có thể bằng tiếng Anh |

Các benchmark này là phương án lập kế hoạch. Task T00 phải kiểm tra khả dụng, quyền sử dụng, split và độ phù hợp rồi quyết định primary dataset. Không mở đồng thời tất cả benchmark nếu ngân sách không đủ.

## Chế độ và tài nguyên

| Trường | Giá trị ban đầu |
| --- | --- |
| execution_mode | `experiment_enabled` (user steering 2026-10-05: data + checkpoints đã có sẵn trên server, được phép code/chạy trong `method/p4_signssl/`) |
| Đọc web/repo công khai, ghi Markdown cục bộ | Được phép trong môi trường có quyền tương ứng |
| Sửa code/thêm script trong project | Được phép, không ghi đè upstream |
| GPU model/count/VRAM | 1× RTX 5880 Ada 49 GB, shared với job khác |
| GPU-hours được cấp | Agent tự đặt cap: ≤ 60 GPU-h cho cycle đầu (feature extraction + pilot); ghi used trong STATE |
| Chi phí dịch vụ trả phí được cấp | 0 |
| Dung lượng tải dataset/checkpoint được cấp | Không cần tải: dùng data/checkpoint local. Ghi local outputs ≤ 25 GB (disk chỉ còn ≈48 GB) |
| CPU/wall-time cho chạy compute | Được phép; ≤ 16 CPU workers để không chặn người dùng khác |
| Số vòng tối đa mỗi phiên | 3 |
| Số giả thuyết được pilot mỗi cycle | Tối đa 2 |
| Pilot seeds đề xuất | 1; chỉ dùng sàng lọc |
| Confirmatory seeds đề xuất | Ít nhất 3 nếu ngân sách đủ; tăng khi variance cao |
| Refresh literature | Đầu cycle mới và trước khi kết luận novelty/paper readiness |
| external_write | Chưa được cấp |
| Hỗ trợ background runner/scheduler | nohup + PID/log trong `method/p4_signssl/runtime/logs/`; không có scheduler |

Muốn chạy thực nghiệm: đặt `execution_mode=experiment_enabled`, ghi compute cap, storage/download cap, data path và dataset usage đã được kiểm tra. Không cần xin phép lại từng job nếu job nằm trong ngân sách và protocol đã cấp.

`research_only` vẫn phải tạo literature notes, audit, hypotheses, review, method/protocol và resource request cụ thể. Không tự tải checkpoint 200 MB hoặc dataset lớn.

## Các điều kiện phải chốt trước confirmatory run

| Trường | Giá trị ban đầu |
| --- | --- |
| Primary dataset / split manifest hash | UNKNOWN |
| Primary metric / đơn vị / hướng tối ưu | UNKNOWN |
| Evaluation unit và resampling unit | UNKNOWN |
| Baselines có thể chạy | UNKNOWN |
| Minimum meaningful effect δ | UNKNOWN |
| Non-inferiority margin ε nếu dùng claim efficiency | UNKNOWN |
| Runtime/memory/FLOP cap cho method | UNKNOWN |
| Max tuning trials và ngân sách mỗi method | UNKNOWN |
| Checkpoint selection rule | UNKNOWN |
| Confirmatory seeds / CI method | UNKNOWN |
| Test access condition | Chỉ sau khi protocol, code, configs và selection rule khóa |
| In-distribution / OOD claim | Chọn cụ thể; không mặc định “zero-shot/unseen dataset” |

Agent đề xuất δ, ε và cap từ nhu cầu ứng dụng, baseline variability và compute; ghi rationale và khóa **trước** confirmatory outcomes. Không lấy con số tăng đã thấy làm threshold.

## Quyền tự chủ

Được tự tìm nguồn, đọc paper, audit repo, tạo note, loại giả thuyết yếu, sửa thiết kế và chọn phép kiểm tra rẻ. Được chạy job phù hợp ngân sách đã cấp.

Phải hỏi đúng điểm bị chặn khi cần thêm tài nguyên, dữ liệu hạn chế, thay đổi mục tiêu đã chốt hoặc viết ra hệ thống bên ngoài. Đề xuất phải kèm artifact reviewable, lựa chọn khả thi và ước lượng chi phí; tiếp tục phần không phụ thuộc.

Không gửi email, tạo issue/PR, push, chia sẻ video, publish hoặc nộp paper khi chưa được cấp. Không tự sửa ngưỡng thành công để “đạt paper-ready”.

