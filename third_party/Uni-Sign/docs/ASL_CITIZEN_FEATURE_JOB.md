# ASL Citizen: job xuất Uni-Sign pose features

Setup ngày 2026-10-12 trên server `ptitiec`.

## Phạm vi và cấu hình

- Đọc cache `/home/dongvk/datasets/ASL_Citizen/keypoints/` của 83.399 video.
- Train 40.154, val 10.304, test 32.941; kiểm tra split không trùng và phủ đúng cache.
- Giữ nguyên pose nguồn. Không chạy lại detector/RTMW và không ghi bản pose trùng.
- Chọn một người ký bằng tracker của serving khi cache có nhiều người.
- Chuẩn hóa 4 nhóm pose, lấy mẫu đều tối đa 64 frame, encode bằng ST-GCN Uni-Sign.
- Checkpoint: `pretrained_weight/unisign/wlasl_pose_only_islr.pth`, nạp `strict=True`.
- Lưu float32, tắt TF32/autocast, cuDNN deterministic. Batch các clip cùng số frame,
  tránh padding làm thay đổi biên temporal convolution.
- Batch size 64; 8 CPU worker, 4 writer; CUDA GPU RTX 5880 Ada.
- Không sinh nhãn: đầu ra là embedding sau pose projection, trước mT5.

## Đầu ra

Thư mục:

`out/asl_citizen_unisign_pose_features_t64/`

Mỗi video tương ứng `features/<shard>/<sample_id>.npz` gồm:

| Trường | Nội dung |
| --- | --- |
| `pose_embeddings` | `T × 768`, float32, `T ≤ 64` |
| `pooled_embedding` | `768`, trung bình theo thời gian |
| `sampled_pose_indices` | Chỉ số trong chuỗi pose sau lọc người/frame |
| `source_frame_indices` | Chỉ số frame tương ứng trong cache pose gốc |
| `selected_person_indices` | Người được chọn tại từng frame cache; -1 nếu bỏ frame |
| `source_frames`, `retained_frames` | Số frame nguồn và số frame còn lại |
| `multi_person_frames` | Số frame nguồn chứa nhiều người |
| `source_sha256`, size, mtime | Dấu vết cache nguồn |
| `sample_id`, `run_fingerprint` | Định danh video và cấu hình/code/checkpoint |

`source_frame_indices` tham chiếu cache gốc, không tự khẳng định timestamp RGB
nếu chưa kiểm tra provenance của bộ cache cũ.

Dung lượng features ước tính khoảng 14,5 GB, cộng mapping và metadata. Mỗi NPZ
được ghi file tạm rồi rename atomically và đọc kiểm tra ngay sau khi ghi.

## Chạy ngầm và resume

Session tmux: `unisign_aslc_features`.

```bash
tmux attach -t unisign_aslc_features
```

Log được ghi vào `out/asl_citizen_unisign_pose_features_t64/run.log`; tmux pane
có thể không hiện log vì stdout/stderr được redirect vào file. Detach bằng
`Ctrl-b`, rồi `d`. Ngắt SSH không dừng session.

Trạng thái nằm ở `status.json`; trạng thái wrapper nằm ở `launcher_status.json`.
Job xử lý từng file độc lập và thử lại lỗi đọc/preprocess tối đa 3 lần. Wrapper
tự khởi chạy lại toàn bộ job tối đa 3 lần nếu process thoát lỗi; các đầu ra đã
đúng được kiểm tra và bỏ qua. Không tạo features giả để thay thế file lỗi.

Nếu cần chạy lại sau khi session cũ đã kết thúc, kiểm tra không còn process
đang chạy rồi dùng:

```bash
bash script/asl_citizen_features_job.sh
```

Lệnh tạo session mới:

```bash
bash script/launch_asl_citizen_features.sh
```

Launcher từ chối nếu session cùng tên đã tồn tại. `job.lock` ngăn hai process
cùng ghi output. Fingerprint ngăn trộn features từ checkpoint/code khác.

## Tiêu chí hoàn thành

Job chỉ ghi `stage: complete` khi đã kiểm tra lại toàn bộ tập đầu ra:

- Đủ 83.399 sample id, đúng từng split; không thiếu hoặc thừa NPZ.
- Shape/dtype đúng; embedding/pooled embedding đều finite.
- Chỉ số frame/người nhất quán và pooled embedding khớp mean của chuỗi.
- SHA-256, size và mtime của pose nguồn khớp metadata đầu ra.

Kết quả cuối ở `verification.json`: cần `success: true`, `validated: 83399`,
`verification_errors: []`, `missing_outputs: []`, `extra_outputs: []`.
Lỗi chi tiết được giữ ở `errors.jsonl` và trong báo cáo verification; không
được xem `completed.jsonl` như danh sách duy nhất vì log này append qua các lần resume.

## Kiểm chứng trước khi chạy toàn tập

- Pilot 128 video trải qua cả train/val/test: 128/128 đầu ra đã pass full verification.
- Chạy FP32 batch so với một clip riêng: sai khác tối đa khoảng 1,91e-5 trên
  clip ngắn và 1,05e-5 trên clip nhiều người, trong tolerance kiểm tra.
- Thử làm hỏng một đầu ra pilot rồi resume: file hỏng được tạo lại, file hợp lệ
  bên cạnh giữ nguyên mtime, cuối cùng 128/128 đầu ra pass verification.
- Syntax Python/shell và `git diff --check` pass.

Người dùng yêu cầu dừng polling sau khi xác nhận job hoạt động ổn. Job tiếp tục
chạy ngầm và tự kiểm tra đầu ra; chỉ kiểm tra tiến độ lại khi người dùng yêu cầu.

## Xác nhận bàn giao setup

Job toàn tập đã được khởi chạy trong session `unisign_aslc_features` lúc
2026-10-12 12:26:20 +07:00, PID launcher 961928, PID Python 961935.
Ở lần kiểm tra ổn định cuối lúc khoảng 12:26:58 +07:00:

- `stage: extracting`, đã ghi và đọc kiểm tra 6.406/83.399 đầu ra.
- `failures: 0`, khoảng 180 clip/s tính từ đầu process.
- Session còn sống; GPU đang được sử dụng.
- Kiểm tra độc lập thêm ba NPZ trong thư mục output đều pass, gồm kiểm tra
  SHA-256 cache nguồn. Đã thấy 7.383 NPZ khi đọc danh sách sau snapshot status;
  số file có thể đi trước status vì status cập nhật mỗi 5 giây.

Đã hoàn tất setup và xác nhận hoạt động, chưa tuyên bố hoàn tất toàn dataset.
Không tiếp tục polling sau lần kiểm tra này theo yêu cầu người dùng.

## Kết quả hoàn tất

Kiểm tra lại theo yêu cầu người dùng lúc 2026-10-12 14:13 +07:00:

- Job hoàn tất lúc 12:36:27 +07:00, thời gian xử lý và xác minh khoảng 604 giây.
- Đủ 83.399/83.399 đầu ra, đã kiểm tra toàn bộ; `success: true`.
- Train 40.154, val 10.304, test 32.941.
- 4.980.274 frame features; tổng NPZ 14.690.471.030 byte (14,690 GB).
- Không thiếu/thừa output, không có lỗi xử lý hoặc lỗi xác minh.
- Wrapper hoàn tất ở attempt 1, exit code 0; tmux pane đã kết thúc bình thường.
- Đã đếm lại file thực tế sau khi job hoàn tất: 83.399 NPZ, tổng byte khớp báo cáo.

Báo cáo máy đọc được: `out/asl_citizen_unisign_pose_features_t64/verification.json`.
