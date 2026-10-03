# Sign_Representations

Source tác giả nằm trong `third_party/`. Code, cấu hình, checkpoint local và
các kết quả trước đây nằm trong `artifacts/legacy/`. Các lệnh dưới đây chạy từ
gốc repository và chỉ dùng các chức năng inference đã public.

## Khôi phục runtime local sau khi di chuyển

```bash
python3 artifacts/legacy/scripts/setup_runtime.py
```

Script tạo symlink local `artifacts/legacy/third_party -> ../../third_party`
và sửa activation, shebang, editable/dependency paths trong ba venv đã di chuyển.
Chạy lại script không thay đổi thêm file khi đường dẫn đã đúng. Source tác giả,
checkpoint và manifest đã khóa giữ nguyên nội dung.

Đây là khôi phục môi trường có sẵn trên máy hiện tại, không phải installer cho
máy mới. Hai venv SHuBERT còn dùng base Python/packages của môi trường Conda
`h4wpp`; venv SignRep cũng dùng system packages. Clone Git không chứa các môi
trường này, source ngoài hoặc checkpoint. Trên máy mới cần chuẩn bị chúng theo
`artifacts/legacy/provenance/source_commits.json`, `provenance/assets.json`,
ba file `provenance/env_*.pip-freeze.txt` và hướng dẫn environment của tác giả.
Các file freeze có đường dẫn local, không nên cài nguyên xi trên máy khác.

## SignRep: chạy example gốc trên video

```bash
runtime_dir="$(pwd)/artifacts/runtime/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$runtime_dir"
video_path=/home/dongvk/datasets/ASL_Citizen/videos/5868753228914183-IMPOSSIBLE.mp4

PYTHONDONTWRITEBYTECODE=1 artifacts/legacy/.venv-signrep/bin/python \
  artifacts/legacy/scripts/run_signrep_example.py \
  --video "$video_path" --device cpu --output "$runtime_dir/signrep.npz"
```

Launcher kiểm tra source commit và hash checkpoint, rồi thực thi
`third_party/SignRep/example_usage.py` với đường dẫn thực và thiết bị được chọn.
Chỉ thay hai placeholder và các lời gọi `.cuda()` trong bản code ở bộ nhớ;
không sửa source checkout hoặc thuật toán extraction. Output chứa `features`
và `latent`, dùng nguyên segment 16 frame, stride 2 và preprocessing gốc.
Video dưới 16 frame không có output theo example gốc.
Có thể chọn `--device cuda:0` khi GPU trống; lần kiểm tra này dùng CPU.

## SHuBERT: video → preprocessing → đặc trưng 12 layer

Các runner có sẵn gọi preprocessing và model từ checkout gốc đã pin.
Ví dụ này lấy một clip train từ pilot local; chạy sau block tạo `runtime_dir` ở trên.

```bash
head -n 1 artifacts/legacy/data/manifests/pilot.jsonl > "$runtime_dir/native_one_clip.jsonl"

OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  artifacts/legacy/.venv-shubert-native/bin/python \
  artifacts/legacy/scripts/prepare_shubert_native.py \
  --manifest "$runtime_dir/native_one_clip.jsonl" \
  --output "$runtime_dir/shubert_preprocessing" --max-clips 1

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  artifacts/legacy/.venv-shubert-native/bin/python \
  artifacts/legacy/scripts/extract_shubert_pilot.py \
  --preprocessing-report "$runtime_dir/shubert_preprocessing/report.json" \
  --output "$runtime_dir/shubert_features" --device cpu
```

Manifest cần `sample_id`, `source_root`, `relative_path`, `official_split=train`;
runner pilot nhận tối đa 20 clip. Output NPZ có `layer_embeddings` dạng
`[12, T, 768]` và `embeddings` của layer cuối, cùng thông tin timestamp/mask.
Pipeline giữ chính sách xử lý landmark thiếu của tác giả.
Các launcher Slurm gốc và pretraining config vẫn cần corpus, labels và cấu hình
cluster tương ứng; không huấn luyện lại hoặc bổ sung downstream module trong lần setup này.

## Kiểm tra checkpoint và demo dịch đã có

```bash
PYTHONDONTWRITEBYTECODE=1 artifacts/legacy/.venv-signrep/bin/python \
  artifacts/legacy/scripts/audit_reproduction.py --output "$runtime_dir/inventory.json"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  artifacts/legacy/.venv-shubert-native/bin/python \
  artifacts/legacy/scripts/smoke_shubert_encoder.py \
  --output "$runtime_dir/shubert_encoder" --dino

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  artifacts/legacy/.venv-shubert-translation/bin/python \
  artifacts/legacy/scripts/smoke_shubert_translation.py \
  --output "$runtime_dir/shubert_translation"
```

Demo dịch dùng clip diagnostic đã chuẩn bị và caption local; là kiểm tra numerical
inference của checkpoint demo, không đo độ chính xác benchmark. Resolver xử lý
đường dẫn tuyệt đối cũ của clip mà không sửa manifest/hash lock.
Các runner từ chối ghi đè output cũ; mỗi lần chạy dùng thư mục mới.

## Git và kết quả kiểm tra

`.gitignore` loại `third_party`, symlink source local, `.venv-*`, checkpoint,
feature caches, video, logs và toàn bộ `artifacts/runtime/`. Chỉ code, hướng dẫn
và báo cáo setup nhỏ được commit.

Xem [báo cáo setup](artifacts/legacy/reports/runtime_setup_20261003.md).
Hướng dẫn và bằng chứng thí nghiệm lịch sử nằm trong
[REPRODUCE.md](artifacts/legacy/REPRODUCE.md); các lệnh tương đối trong đó chạy
từ `artifacts/legacy`, và các output/hash lock cũ phải được giữ nguyên.
