# Đánh giá chuyển bộ dữ liệu: ASL Citizen → WLASL2000

Mục tiêu hiện tại: chỉ đánh giá **JH3D, JV3D, SR3D**, dùng nguyên checkpoint đã train trên ASL Citizen, theo cách **no downstream training** của SignRep Table 4. Kết quả SignRep public đã tính được giữ làm tham chiếu, không chạy lại. Không huấn luyện, fine-tune, linear probe, fit PCA/whitening hoặc cập nhật EMA trên WLASL2000. Không chọn checkpoint, pooling hay hyperparameter bằng kết quả test.

## Dữ liệu và checkpoint đã khóa

| Thành phần | Cấu hình |
|---|---|
| Train các method nội bộ trước đây | 3.000 video ASL Citizen official train, 300 gloss × 10; không dùng nhãn gloss để pretrain |
| Baseline public SignRep | Checkpoint tác giả pretrain trên YouTube-SL-25; không train thêm trên ASL Citizen/WLASL2000 |
| Dataset chuyển sang đánh giá | `/home/dongvk/datasets/WLASL2000` |
| Gallery | WLASL2000 **train**, 14.296 video, 2.000 gloss |
| Query | WLASL2000 **test**, 2.879 video, 2.000 gloss |
| Validation | Không dùng trong lượt eval này |
| Chọn checkpoint | Các checkpoint cuối đã dùng trong bảng ASL Citizen; giữ cố định trước khi eval WLASL test |
| Nhãn WLASL2000 | Chỉ dùng ghép gloss của gallery và tính metrics trên query |

Manifest chỉ gồm train/test, SHA256 annotation, danh sách lớp, checkpoint và thống kê chuẩn hóa graph được lưu trong `method/obs_jepa/runtime/wlasl2000_frozen/protocol.json`. Protocol gốc vẫn giữ registry 17 checkpoint để bảo toàn provenance; **`selection.json` giới hạn hàng đợi và báo cáo còn JH3D, JV3D, SR3D**. Không ghi vào video, keypoint hoặc checkpoint gốc.

## Thuật toán evaluation

1. Encoder ở `eval()`, tất cả parameters `requires_grad=False`; chạy bằng `torch.inference_mode()`.
2. RGB: cửa sổ **16 frame, stride 2**, frame gốc (local WLASL là 25 fps). Feature window lấy từ encoder/projection đã train. A/A00 giữ letterbox; Hiera/JV3D/SRVJ giữ squash 224 theo checkpoint. Không masking/augmentation ngẫu nhiên lúc inference.
3. `avg`: trung bình feature window gốc. `weighted`: trọng số window bằng `max(activity_left, activity_right)`, clamp về [0,1], theo Eq.11–12; nếu tổng trọng số <1e-6 thì dùng trung bình đều và ghi số lần fallback. RGB ngắn hơn 16 frame lặp frame cuối; chỉ dùng full windows, không thêm window cuối ngoài stride.
4. Chuẩn hóa L2 một lần ở cấp video sau pooling, rồi tính cosine query/gallery. Không L2 riêng mỗi window trước pooling.
5. Điểm của một gloss là cosine lớn nhất trong các video gallery có gloss đó; xếp hạng **2.000 gloss khác nhau**, không coi các video cùng gloss là nhiều lớp.
6. Với thứ hạng gloss đúng `r`: `DCG=mean(1/log2(1+r))`, `Rec@K=mean(r<=K)`. Kết quả CSV nhân 100. Tie được xử lý bất lợi: `r = 1 + số gloss khác có điểm >= điểm gloss đúng`. Giữ các embedding bằng 0 và ghi số lượng, không loại mẫu khỏi eval.

Nguồn: [SignRep §6, §7.2, Table 4](https://arxiv.org/html/2503.08529v1). Bản video local đã resize/trim; không khẳng định bản phát hành và tiền xử lý video giống hoàn toàn bản tác giả dùng. Đây là đánh giá trên cùng local gallery/query cho mọi checkpoint.

## Method và pooling được báo cáo

| Method | Pooling |
|---|---|
| JH3D, SR3D | `avg` và `weighted` |
| JV3D | `avg`: checkpoint cuối không lưu activity head đã train |
| SignRep_public (kết quả tham chiếu có sẵn) | `avg` và `weighted` |

Các nhánh A, B, SRVJ, baseline cũ và A00 không còn trong hàng đợi. Ba method được chọn đều nhận RGB, 16 frame, stride 2, squash 224; không có adaptation landmark trong lượt chạy hiện tại.

Hàng đợi gồm **3 method**; CSV gồm **5 dòng method/pooling đang chạy + 2 dòng SignRep public đã xong**. Tiến độ tổng chỉ đếm ba method được chọn, không tính reference thành một job mới. Phải đọc cột `Pooling` khi so kết quả; không gộp `avg` với `weighted` hoặc chọn dòng có điểm test cao hơn làm kết quả mặc định.

## Chạy và theo dõi

Từ `/home/haipd/Sign_Representations`:

```bash
# Tạo/kiểm tra protocol đã khóa và bảng CSV (không dùng GPU).
.venv-signrep/bin/python method/obs_jepa/oj/wlasl_retrieval.py prepare

# Tính SignRep public từ cache stride 2 đã có (không dùng GPU).
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv-signrep/bin/python method/obs_jepa/oj/wlasl_retrieval.py public

# Hàng đợi CHỈ ba method: không có lệnh train.
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv-signrep/bin/python -u method/obs_jepa/oj/wlasl_retrieval.py run --names JH3D,JV3D,SR3D --allow-shared-gpu

# Xem trạng thái và file CSV cập nhật.
.venv-signrep/bin/python method/obs_jepa/oj/wlasl_retrieval.py status
```

Session tmux của lượt đánh giá: `signrep_wlasl_eval`. Xem bằng `tmux attach -t signrep_wlasl_eval`.

Theo chỉ thị mới của người dùng, runner dùng `--allow-shared-gpu`: không chờ các GPU job khác của cùng user, chỉ yêu cầu còn ít nhất 8.000 MiB trước khi bắt đầu mỗi method. Ba method vẫn chạy lần lượt **JH3D → JV3D → SR3D**, một worker eval tại một thời điểm. Chế độ này không gửi tag yêu cầu scheduler khác chờ eval; không sửa, dừng hoặc khởi động lại các job train khác. Chính sách chia sẻ GPU được lưu trong `selection.json`, nên restart vẫn giữ lựa chọn này. Có thể trở lại chế độ chờ bằng `--no-allow-shared-gpu`. Job lỗi dừng hàng đợi, không tự retry. Timeout mỗi method 12 giờ. Chạy lại lệnh `run` sẽ dùng lại selection đã lưu, bỏ qua method xong và tiếp tục shard còn thiếu, sau khi xác minh provenance; không tự mở rộng về 17 method.

Lưu compact pooled embedding theo shard 64 video, không lưu toàn bộ window; dự kiến dưới 1 GB cho bộ này. SHA256 checkpoint được kiểm tra trước/sau inference. Provenance mỗi split có SHA256 code trích feature, checkpoint và manifest, ngăn resume lẫn cấu hình/code khác.

## Kết quả và log

- CSV: `method/obs_jepa/runtime/wlasl2000_frozen/comparison.csv`, UTF-8 BOM để mở bằng Excel; các dòng chưa eval có trạng thái `pending`, metrics để trống.
- Kết quả từng method: `results/<method>.json`; per-query ranks: `results/<method>_<pooling>_ranks.npy`.
- Trạng thái tổng: `progress.json`, `runner.json`.
- Log hàng đợi: `logs/runner.log`; log method: `logs/<method>.log`.
- Kiểm tra CPU trước chạy: `preflight.json`, `logs/preflight.log`; không ghi kết quả test từ smoke check.

## Material Passport

Experiment ID: `proposal11-wlasl2000-frozen`. Mode: code execution. Inputs: existing ASL Citizen checkpoints JH3D/JV3D/SR3D, cached public reference, local WLASL2000 train/test videos. Output: frozen dictionary retrieval metrics and per-query ranks. Human data uploaded: none. Selection/tuning on test: none. Scope: three models explicitly selected by the user; ongoing proposal12 training is outside this evaluation suite.
