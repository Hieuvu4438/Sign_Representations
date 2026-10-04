# 07 — Implementation và thực nghiệm

## Cấu trúc một hướng mới

Tạo một thư mục con `method/<direction_slug>/` bên trong `/home/haipd/Sign_Representations/method/`, với `AGENTS.md`, README riêng, package code, `configs/`, `scripts/`, `tests/` và manifest runtime. Mọi code/config/test/script và implementation mới của hướng phải nằm trong thư mục con này; không đặt code trực tiếp tại `method/` hoặc trong `docs/proposal2/research_lab/`. Slug mô tả câu hỏi, không đặt tên paper trước evidence. `AGENTS.md` của hướng dẫn về `docs/proposal2/AGENTS.md` và bộ rules tại `docs/proposal2/research_lab/`; không copy instructions đã stale. Các đường dẫn này tính từ repository root; link tương đối từ `method/<direction_slug>/AGENTS.md` dùng `../../docs/proposal2/AGENTS.md` và `../../docs/proposal2/research_lab/README.md`.

Interface tối thiểu cho code tương lai: data loader có sample IDs và masks; encoder adapter trả feature cùng temporal metadata; downstream model; train; predict; evaluate wrapper; manifest và status writer. CLI phải có config path, seed, run ID, output directory và resume semantics. Lệnh thực được viết sau khi code tồn tại và đã smoke; không ghi command tưởng tượng vào run ledger.

Không import phương pháp/code cũ của project. Đọc third-party inference cần thiết, giữ nó read-only; wrapper mới quản lý output/config/env. Nếu upstream bug cần patch, tạo patch hoặc bản copy theo license trong namespace mới, lưu diff/source SHA và kiểm tra parity; không sửa global package hay third-party gốc.

## Thiết kế phép thử trước khi chạy

Mỗi experiment card có question, H/P IDs, primary metric, predicted effect, success/kill rule, baselines, changed variable, controlled variables, dataset/cohort, seeds, HPO budget, compute limit và artifacts cần giữ.

Thay một yếu tố đủ để phép thử phân biệt mechanism; factorial design khi interaction quan trọng và budget cho phép. Không thay encoder, preprocessing, head, training data và metric cùng lúc rồi gán gain cho một loss.

Baselines và method có cùng cohort, label budget, split, scorer, comparable inference access và search budget. Nếu native encoder input khác nhau, ghi khác biệt và report cả native pipeline cost; khi cần cô lập proposed module, so sánh trong cùng encoder. Match compute và parameter controls cho claim về efficiency/capacity.

## Các tầng thực nghiệm

| Tầng | Mục đích | Evidence được phép |
| --- | --- | --- |
| CPU/synthetic smoke | Check shape, masking, mapping, scorer, serialization | Software correctness, không claim performance |
| Data smoke | 2–5 train/dev samples; feature parity; tiny overfit | Pipeline hoạt động; không đại diện corpus |
| Pilot | Fixed dev subset đủ phân biệt hypothesis trong budget | Signal thăm dò và quyết định scale |
| Confirmatory | Plan khóa, full selected cohort, seeds và ablations | Claim định lượng với giới hạn đã định |

Tiny overfit không bắt buộc với bài toán retrieval không train head; chọn sanity check tương ứng. Không đòi reproduce toàn bộ paper pretraining trước khi sử dụng checkpoint: kiểm tra interface/provenance và baseline trong setting cần dùng là đủ để bắt đầu khi limitations rõ.

## Search budget và test lock

Đăng ký search space, số trial, max runtime, selection metric và checkpoint rule. Mọi trial kể cả failed phải log. Không chọn best test checkpoint, best seed hoặc layer theo test. Dev có thể overfit qua nhiều vòng; ghi cumulative search và dùng fresh held-out validation khi scope/search lớn yêu cầu.

Trước confirmatory: đóng băng code/config/split/scorer, HPO rule, seeds, primary metric và comparisons. Khóa test predictions khỏi developer feedback; scoring sau selection. Nếu test làm lộ bug, ghi incident, invalidation và rerun rule. Method thay đổi sau khi nhìn test phải được ghi exploratory và cần test/cohort độc lập cho claim confirmatory mới.

## Seed và ablation plan

Default confirmatory: tối thiểu 3 training seeds độc lập cho stochastic methods, thêm khi variance lớn và budget được cấp. Deterministic frozen retrieval không cần lặp extraction giống nhau chỉ để tạo 3 số; dùng sampling/paired uncertainty đúng đơn vị. Few-shot cần nhiều support subsets khi claim về label efficiency; tách subset seed và optimization seed.

Ablations trả lời cơ chế: bỏ proposed component; thay bằng simpler alternative; capacity/compute-matched control; sensitivity với hyperparameter quan trọng; transfer sang encoder/task/domain thứ hai khi claim tổng quát. Không yêu cầu đầy đủ mọi ablation cho pilot nhưng phải plan trước confirmatory.

## Runtime và môi trường

Tạo env riêng theo encoder khi dependency xung đột; không phá env inference hiện có. Record Python/framework/CUDA/dependency versions, GPU model/VRAM, precision, deterministic settings, package lock và upstream SHAs. Không phải mọi GPU op deterministic; ghi limitation thay vì giả vờ bitwise reproduction.

Run ID gồm timestamp UTC, H/P IDs, config hash và seed. Atomic status: `queued → running → completed/failed/interrupted/invalidated`. Record start/end, PID/job ID/host, code SHA hoặc snapshot hash với dirty diff, command, environment, manifest hashes, last checkpoint, exit code và reason.

Giám sát loss finite, learning rate, gradient norm khi thích hợp, valid sample/token counts, missing labels, memory, throughput và walltime. Alert/dừng job do OOM lặp, NaN, dữ liệu rỗng, không tiến triển hoặc budget vượt; không biến crash thành scientific failure. Retry bounded với sửa config có version, không vòng retry vô hạn.

## Compute và disk control

Budget default ở `11_BOOTSTRAP.md`, overridable bằng allocation cụ thể. Preflight ước tính bytes/video/features/layers, GPU-hours và duration từ sample đại diện. Chọn GPU thực được cấp; idle GPU không tự đồng nghĩa có quyền chiếm. Không override `CUDA_VISIBLE_DEVICES` của scheduler và không dùng hết nhiều GPU mặc định.

Giữ checkpoint/log/predictions quan trọng; prune chỉ scratch artifacts do agent tạo, có danh sách và rebuild path. Không xóa checkpoint/dataset gốc. Feature extraction đắt giữ manifest/checkpoint thường xuyên. Disk remaining và quota kiểm tra trước extraction, không đợi disk full.

Không commit outputs lớn. Config, source, protocol, table/plot scripts và lightweight result summaries thuộc Git; raw predictions/manifests có thể ở artifact store được cấp, nhưng phải truy xuất được để review và rebuild. Không đặt duy nhất evidence trong notebook RAM.

## Kiểm tra code cần thiết

Viết tests cho rủi ro thật: split exclusivity, label map, temporal alignment, padding/loss mask, checkpoint coverage, metric oracle, feature cache identity, prediction completeness, train/test independence và resume. Không cần tests hình thức chỉ phản chiếu dòng code đơn giản.

Khi tests/required checks đạt, chuyển sang phép thử scientific kế; không lặp verification vô ích. Regression protocol tests bắt buộc khi đổi evaluator/data adapter. Lưu output kiểm tra, không chỉ ghi “tests passed”.
