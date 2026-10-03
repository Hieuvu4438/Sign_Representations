# Runtime setup sau khi chuyển vào artifacts/legacy — 2026-10-03

Đã khôi phục và kiểm tra các đường inference public trên máy local bằng CPU.
Source checkout và checkpoint giữ nguyên; không triển khai thêm pretraining,
downstream task hoặc benchmark còn thiếu từ tác giả.

## Thay đổi

- `scripts/setup_runtime.py` nối `legacy/third_party` tới source ở gốc, sửa
  activation/shebang và dependency paths của ba venv đã di chuyển. Chạy lần
  thứ hai ghi nhận `repaired_files: []`; pip của cả ba môi trường hoạt động.
- `scripts/run_signrep_example.py` thực thi example gốc với đường video,
  checkpoint local và thiết bị được chọn. Segment/stride/model/transform giữ
  nguyên; output lưu cả features và latent. Các thay đổi placeholder và `.cuda()`
  chỉ thực hiện trong AST ở bộ nhớ, không ghi vào checkout tác giả.
- Resolver của demo SHuBERT tìm clip ở vị trí mới khi manifest giữ đường tuyệt
  đối cũ. Hash-locked manifest và các kết quả trước đây không bị chỉnh sửa.
- `.gitignore` bổ sung symlink source local và toàn bộ `artifacts/runtime/`.

## Kết quả thực chạy

| Kiểm tra | Kết quả |
|---|---|
| Inventory source commit và SHA256/size của assets đăng ký | PASS |
| Unit tests hiện có | 65/65 PASS |
| SignRep example gốc trên clip ASL Citizen train, 114 frame | Load tất cả key thành công; features `[50, 768]`, latent `[50, 768]`; hữu hạn |
| SHuBERT encoder và DINO face/hand | Strict load PASS; deterministic difference 0; stream lệch thời gian bị từ chối |
| SHuBERT preprocessing trên cùng clip ASL Citizen | PASS; 114 frame, 30 FPS; face/hand/body đồng bộ |
| SHuBERT raw-video feature extraction | PASS; layer features `[12, 114, 768]`; cuối `[114, 768]`; repeat difference 0 |
| SHuBERT demo translation, clip NCSLGR local 104 frame | PASS numerical smoke; strict load; mean NLL 1.4742034673690796, 39 token |

Clip ASL Citizen: `5868753228914183-IMPOSSIBLE.mp4`;
SHA256 `ef2f6ce50dbad154e2937ecfdf186e0bd58a8e94e75ab8cdd48c41dfd1e3bb51`.
Output và JSON chi tiết giữ local trong `artifacts/runtime/`, không commit.
SHuBERT preprocessing chạy khoảng 16.41 giây, feature smoke 16.84 giây,
demo translation 46.21 giây. MediaPipe có thể tạo EGL/OpenGL context dù các
model smoke chạy CPU; không chạy PyTorch CUDA inference trong lần này.

## Môi trường và giới hạn

SignRep: Python 3.13.5, PyTorch 2.11.0+cu128, timm 1.0.20,
albumentations 2.0.8. SHuBERT demo: Python 3.10.19, PyTorch 2.1.1+cu121,
transformers 4.30.2, MediaPipe 0.10.14. SHuBERT dùng base Conda `h4wpp` và
dependency path tới venv native; đây chưa phải môi trường độc lập cho máy mới.

Không kiểm tra CUDA vì GPU đang gần đầy. Không chạy lại full dataset, training
hoặc benchmark paper. Native SHuBERT giữ input FPS và chính sách landmark thiếu
của các script đã public; điều này không xác nhận downstream recipe paper.
Demo translation PASS xác nhận inference numerical, không xác nhận chất lượng dịch.
Các checkout SignRep, SHuBERT và DINOv2 không có thay đổi tracked sau kiểm tra.
