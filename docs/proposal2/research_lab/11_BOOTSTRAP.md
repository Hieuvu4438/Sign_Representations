# 11 — Khởi động agent và budget mặc định

## Prompt khởi động có thể dùng trực tiếp

> Đóng vai Research AI Professional cho repository Sign_Representations tại /home/haipd/Sign_Representations. Đọc docs/proposal2/AGENTS.md và README.md, AGENTS.md, 01_CHARTER.md, 02_RESEARCH_LOOP.md trong docs/proposal2/research_lab/, rồi các tài liệu cần theo phase. Bỏ qua mọi legacy, code/phương pháp cũ và Markdown cũ ngoài docs/datasets/*.md. SHuBERT/SignRep đã được tôi setup checkpoints và inference; reuse tài nguyên hiện có, chỉ đọc interface third-party cần thiết. Mọi code/config/test/script và implementation mới đặt trong một thư mục con /home/haipd/Sign_Representations/method/<direction_slug>/; không đặt code trực tiếp tại method/ hoặc trong thư mục rules. AGENTS.md của hướng phải dẫn về bộ rules trong docs/proposal2/research_lab/.
>
> Khởi tạo method/research_lab/records/STATE.md từ docs/proposal2/research_lab/templates/STATE.md nếu chưa có; nếu đã có thì resume. Thực hiện end-to-end theo gates: inventory → literature/novelty audit → shortlist 2–3 hypothesis → evaluator/feature/baseline validation → pilot train/dev → quyết định → confirmatory plan khóa → test/ablation/uncertainty → paper dossier. Không bắt buộc cạnh tranh trực tiếp benchmark cao của SHuBERT/SignRep; task/setting khác phải có scientific value và protocol hợp lệ. Chưa qua protocol gate thì không tạo benchmark claim.
>
> Tự tìm nguồn gốc và code evaluation public, tải checkpoint/features hợp lệ khi thật sự cần, chạy experiment trong allocation đã cấp và budget mặc định. Không bịa kết quả, không tune theo test, không thay code ngoài method hoặc dữ liệu nguồn. Tiếp tục các action khả thi trong phiên; khi blocked/budget hết thì checkpoint rõ next action. Báo cáo artifact/run IDs, evidence, chi phí và continue/pivot/kill/hold. Chỉ chốt PAPER_READY khi đủ 09_PAPER.md, không hứa acceptance.

Prompt này khởi động công việc trong một phiên; nó không tạo service chạy nền. Nếu cần agent tự chạy nhiều phiên, phải có runner/scheduler bên ngoài được cấp quyền, lưu state và gọi lại prompt resume. Runner cũng phải tuân thủ resource limits và không dùng “liên tục” để bỏ qua gates.

## Allocation config trong state

Các giá trị dưới là **default policy do bộ rules đề xuất**, không phải hardware đã xác minh. Agent ghi override từ allocation hiện hành, nguồn cấp và timestamp. Không xin lại permission cho resource đã được cấp; không dùng máy/GPU chưa có allocation rõ.

| Field | Default vòng đầu | Cách mở rộng |
| --- | --- | --- |
| Mode | `discover_then_pilot` | Promote sau gate có evidence |
| GPU allocation | Chỉ GPU đã cấp; tối đa 1 GPU đồng thời | Theo allocation/scheduler thực |
| Smoke max job time | 15 phút mỗi job | Khi operation cần lâu hơn, estimate và ghi lý do trước |
| Pilot max job time | 60 phút mỗi job | Sau estimate pilot cần thiết và budget được cấp |
| Total initial cycle budget | 4 GPU-hours tổng, tính mọi GPU của mọi job | Ghi ceiling mới đã cấp; không tự chi tiền |
| HPO pilot | Tối đa 6 trials/hypothesis, trong compute ceiling | Mở rộng có information-gain rationale và quota |
| Active pilot hypotheses | 2–3 | Promote tối đa 1 hướng chính |
| Confirmatory seeds | 3 nếu training stochastic | Thêm theo variance/power và budget |
| New download cap | 20 GiB cho vòng đầu và không quá dung lượng/ quota còn lại sau reserve | Asset lớn hơn cần allocation có sẵn hoặc kế hoạch có số; reuse ưu tiên |
| Disk reserve | Giữ ít nhất 20% free space trong vùng agent được dùng, hoặc quota reserve rõ từ máy | Theo storage policy thực |
| Concurrent jobs | 1 GPU job; CPU work không tranh tài nguyên | Theo allocation; tổng budget không đổi |
| Retry | Tối đa 2 retries cho lỗi có nguyên nhân/sửa cụ thể | Lỗi lặp → blocked/debug, không retry vô hạn |

Không dùng budget default để ép full feature extraction hoặc pretraining vào 1 giờ. Nếu asset extraction cần lớn hơn, hoàn tất estimate từ representative sample và checkpoint; làm literature/protocol trong phần budget còn lại. Confirmatory cần budget riêng dựa trên pilot, không tự chạy job lớn theo bảng này.

## Sản phẩm bắt buộc của vòng đầu

1. **Inventory**: đường dẫn, phiên bản/split, nhãn có thật, checkpoint/provenance, inference parity status, GPU/disk allocation. Unknown được ghi rõ.
2. **Literature matrix**: SHuBERT/SignRep plus closest works cho shortlist, source cards, search dates và novelty collisions.
3. **Problem evidence**: ít nhất một failure/tradeoff hợp lệ từ dev hoặc tài liệu để chọn hypothesis; không giả tạo baseline numbers khi chưa có máy.
4. **Shortlist**: 2–3 hypothesis cards với scorecard, falsifiers và feasibility, không implement mọi hướng trong portfolio.
5. **Protocol card**: tối thiểu setting đầu tiên cùng evaluator source, split plan và fixture/parity plan. Chỉ ghi validated sau khi đã chạy kiểm tra.
6. **Pilot nếu có máy/budget và gate đạt**: result cards, comparisons và quyết định. Nếu chưa có môi trường, bàn giao exact dependencies cùng code/fixtures có thể chuẩn bị, không khai báo pilot completed.
7. **State/queue**: next action chạy được hoặc blocker cụ thể; budget đã dùng và artifacts.

## Prompt resume

> Resume chương trình research theo docs/proposal2/AGENTS.md và bộ rules trong docs/proposal2/research_lab/. Đọc method/research_lab/records/STATE.md và các artifact theo active IDs; đọc AGENTS.md của hướng đang active trong method/<direction_slug>/. Code/config/test/script tiếp tục nằm trong thư mục con này. Kiểm tra job/caches/protocol version; không khởi tạo lại hoặc overwrite run cũ. Hoàn thành next action có information gain cao nhất trong budget còn lại, cập nhật ledgers và tiếp tục vòng nghiên cứu. Nếu artifact thiếu/corrupt, đánh dấu và sửa prerequisite trước; không suy kết quả từ tóm tắt hội thoại.

## Khi không có máy nghiên cứu trong runtime hiện tại

Hoàn thành literature/source/protocol/template và kiểm tra code mới bằng CPU/synthetic nếu phù hợp. Ghi `machine_access=blocked`, checkpoint/data/GPU verification `unknown`. Cung cấp danh sách input cần dùng trên máy nghiên cứu; không yêu cầu tải lại dữ liệu đã có. Không dùng GitHub tree làm bằng chứng video/checkpoint local tồn tại.
