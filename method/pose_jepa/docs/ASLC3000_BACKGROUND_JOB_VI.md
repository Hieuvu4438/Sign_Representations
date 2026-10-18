# Job train nền Pose-JEPA trên ASLC3000

**Lưu trữ ngày15/10/2026:** runtime, weights, cache, features và environment riêng
đã xóa theo yêu cầu người dùng; service đã gỡ. Results/evidence được giữ trong
[archive đã xác minh](../../../docs/experiments/pose_jepa_results_20261015/REPORT_VI.md).
Các đường dẫn/lệnh dưới đây là hồ sơ setup lịch sử.

Job được quản lý bằng `systemd --user`, service **`pose-jepa-aslc3000.service`**.
Máy hiện bật `Linger=yes`: service không phụ thuộc terminal hay kết nối SSH.
Service được enable để tự chạy sau khi máy khởi động lại; không yêu cầu quyền root.

Từ thư mục project:

```bash
cd /home/haipd/Sign_Representations/method/pose_jepa
runtime/venv/bin/python scripts/aslc3000_job.py status
cat runtime/aslc3000_job/progress.txt
tail -f runtime/aslc3000_job/logs/train.log
```

Các giai đoạn tự động: **prepare → train → evaluate**. Khi chuẩn bị dữ liệu,
theo dõi `progress.txt`/`progress.json`; `logs/train.log` xuất hiện khi train bắt đầu.
Log loss chi tiết ở `runtime/aslc3000_job/train/train.jsonl`. Log service ở
`runtime/aslc3000_job/service.log` hoặc:

```bash
journalctl --user -u pose-jepa-aslc3000.service -f
```

## Dừng và tiếp tục

```bash
# Dừng có lưu tiến độ; lệnh đợi worker/trainer dừng an toàn.
runtime/venv/bin/python scripts/aslc3000_job.py stop

# Tiếp tục cùng job, không tạo run mới và không train lại từ đầu.
runtime/venv/bin/python scripts/aslc3000_job.py resume
```

Chỉ đóng SSH thì **không cần stop**. Stop chủ động tạo `pause.request`, nên reboot
cũng không tự chạy lại job đã chủ động pause; lệnh resume xóa marker đó.
Không xóa/thay `train.yaml`, `code/`, `job.json`, manifests hoặc checkpoints của job
đang chạy. Code được snapshot độc lập để sửa repo không làm đổi phiên train/resume.

Checkpoint đầy đủ sau update đầu và mỗi **100 updates**, cùng lúc validation và
khi nhận SIGTERM/SIGINT. `last.pt` giữ model/student/teacher/predictor/heads,
optimizer, EMA center, scaler, step, RNG và cursor/sampler. `previous.pt` giữ bản
checkpoint trước đó, độc lập với replacement của `last.pt`. File được ghi tạm rồi
atomic rename. Resume tự dùng checkpoint gần nhất; nếu file không đọc được thì
thử bản trước, và ghi `train/resume.json`. Khác code/data/config hash sẽ báo lỗi,
không tự bỏ qua và không bắt đầu lại từ scratch.

Khi mất điện/SIGKILL, phần chưa checkpoint có thể phải chạy lại tối đa khoảng
100 updates. Khi stop có kiểm soát, trainer lưu tại optimizer boundary. Service
đợi tối đa 300 giây cho graceful shutdown; nếu validation chưa dừng kịp, checkpoint
trước validation vẫn còn và validation đang dang dở sẽ được thực hiện lại khi resume.

Lỗi tiến trình khiến systemd thử lại sau 30 giây, tối đa 5 lần trong một giờ để
không lặp vô hạn khi có lỗi dữ liệu. Sau khi sửa nguyên nhân, dùng resume để reset
failed state. Service status và PID được kiểm tra bằng systemd; JSON trạng thái
không được dùng một mình để kết luận tiến trình còn sống.

## Cấu hình được chốt

- Encoder 8 layers/dim256; predictor 4 layers/dim128; pose 2D, T≤128, whole-sign
  subsampling, shared-hand spatial GCN và recipe JEPA/geometry/motion chính.
- Batch32, accumulation1, BF16 nếu GPU hỗ trợ; AdamW LR3e-4, warmup1.000 rồi cosine.
- Tối đa **30.000 optimizer updates**; validation mỗi2.000; early stopping sau
  5 lần validation không cải thiện. Checkpoint selection dùng EMA encoder/R@1.
- Seed42; không dùng gloss/text trong SSL losses. Train scratch, không external init.
- 4 workers chuẩn bị pose, 4 workers DataLoader; tất cả output dưới job directory.
- Không khởi chạy thêm các baseline hoặc Stage2 ngoài job hybrid được yêu cầu.

GPU đang dùng chung với các job khác. Service không dừng hay thay đổi workload
khác; thời gian thực tế có thể dài hơn benchmark. Checkpoint/resume giữ được
tiến độ nếu cần nhường GPU bằng stop/resume.

## Protocol và validation

Nguồn train/gallery đúng **3.000 IDs** từ ASLC002; query đúng **1.500 IDs** và thứ
tự của các method cũ. Thực tế 1.500 query này thuộc **official val**, không phải
official test. Không dùng chúng để chọn checkpoint.

Chọn checkpoint bằng raw frozen LOSO trên **8.804 official val clips còn lại**:
loại cùng signer khỏi gallery và rank gloss bằng max video cosine; failed queries
ở nguyên denominator. Cách này có đủ classes hơn lựa chọn chỉ giữ gloss thuộc 300
train classes (chỉ còn21 clips). Báo rõ đây là target-validation selection.

Đánh giá cuối giữ nguyên ASLC300 (3.000 gallery/1.500 query), và LOSO/probes trên
10.304 official val clips như code cũ. Vì 8.804 clips đã được dùng để selection,
LOSO toàn validation không phải independent test. Không gắn nhãn zero-shot/strict
held-out cho kết quả đó, và không gọi comparison với budget4k/16k là compute-matched.

Job chuyển cache133-joint bằng PTS thật, giữ all-person raw SimCC scores và profile
`q=s/(1+s)` đã khai báo, kiểm tra frame-count/source/hash; không giả định detector
weights đã xác minh. Mỗi clip có receipt riêng; khởi động lại chỉ chuyển tiếp phần
chưa hoàn tất, không decode toàn bộ từ đầu.

## Evaluation sau train

Supervisor tự chọn `train/best_validation.pt`, export bundle và trích feature từng
clip có cache/resume, rồi chạy adapter cho evaluator `method/obs_jepa/oj/evaluate.py`
trong môi trường legacy hiện có: retrieval, true linear probe, LOSO và body-part/
signer probes. Adapter xử lý các clip lỗi rõ ràng; không sửa code evaluator gốc.
Output ở `runtime/aslc3000_job/evaluation/`.

- `raw_retrieval.json`: raw1024-D output của recipe chính.
- `raw_loso.json`: raw1024-D LOSO trên đủ10.304 val clips.
- `coverage.json`: số lượng và IDs encode thất bại theo từng split.
- `legacy/report.json`: historical pooling/whitening/probe recipe và splits của
  methods cũ, có failure adapter; PCA1024→768 chỉ fit trên phần valid của gallery.
- `bundle/`: EMA encoder selected, không random fusion/projector.
- `complete.json`: marker chỉ tạo sau khi evaluation thật exit0.

Readout adapter dùng một whole-clip raw embedding sau unique-segment stitch cho
mỗi clip; span nguồn toàn clip được giữ. Điều này được ghi trong metadata; không
gọi đó là cùng temporal readout với S1/S2. Audit dữ liệu phát hiện16 clip không
canonicalize được:13/3.000 train-gallery,1/1.500 query và3/10.304 val (query nằm
trong val; selection_val có2 lỗi). Train vẫn giữ protocol3.000 IDs; invalid support
không tạo SSL loss. Retrieval loại invalid gallery; invalid/missing-positive query
có rank∞ và vẫn ở mẫu số. Linear probe tính invalid query là sai. PCA/whitening
chỉ fit valid features. Geometry/body-part/signer diagnostics dùng valid features
và báo coverage riêng; do đó adapter không được mô tả là evaluator hoàn toàn giống
methods cũ. 13/1/3 là lỗi normalization đã biết, coverage cuối còn ghi nhận lỗi encode.

Kiểm thử sau setup:58 tests pass, gồm test SIGTERM/resume subprocess thật,
fallback checkpoint hỏng và LOSO missing-positive/tie/failure denominator. Bằng
chứng ở `docs/evidence/tests_managed_job.xml`. Test cuối còn xác nhận validation bị
gián đoạn ở final step được chạy lại khi resume mà không lặp optimizer update.
Adapter legacy đã được chạy tích hợp với evaluator thật trên fixture: đủ mẫu số,
invalid query rank∞ và linear probe sai. Job thật cũng đã stop tại update64 và
resume vượt update64; bằng chứng ở `docs/evidence/aslc3000_job_live_resume.json`.
