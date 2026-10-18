# E04 full ASL-Citizen — service seed 17

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, run inline.
- Origin Date: 2026-10-12.
- Version Label: E04_full_aslc_retrieval_seed17_optimized_service.
- Verification Status: setup/startup đã qua; experiment đang chạy, chưa có kết quả cuối.

## Phạm vi và cấu hình

Người dùng cho phép thêm một tiến trình chạy full và xác nhận đủ dung lượng.
Giữ nguyên E04: D384, encoder 6 layers, predictor 2 layers, BF16,
semantic JEPA + geometry target, teacher EMA, seed 17. Train trên toàn bộ
40.154 official train videos, 2.731 glosses; labels không đi vào SSL sampler/loss.
10.000 successful updates, microbatch 8 × accumulation 4 = effective batch 32,
tương đương khoảng 320.000 clip draws / 7,97 lượt tập train nếu không skip.

Frozen dictionary retrieval: train làm gallery, 10.304 val queries rồi 32.941
official test queries. Báo DCG, Rec@1, Rec@5; cosine + max exemplar per gloss,
pessimistic ties. Val tại budget cuối, test không dùng chọn checkpoint; không train classifier.

Config: `configs/full_aslc/E04_seed17.yaml`, SHA256
`c7e7186db272cf0ce2dd027fdbaa9bd113949ba6e607d2434212a22705453f41`.
Model runtime implementation hash giữ nguyên so với job 3.000 đang chạy:
`0623c9ee11db3bdc18e7802c640c880e527302e972ebb2880d06d517a66e1dc5`.
Không sửa `src/signpath` hoặc config của hai experiment khi setup service này.

## Tốc độ và tài nguyên

Một Python process cho mỗi job: tổng hai experiment processes.
Cache full dùng 2 independent detector threads, DINO batch 16. Benchmark 8 clips
× 2 repeats trước đó cho gần 1,93× cache speedup và visual/nonvisual outputs bitwise
identical so với 1 thread/batch 16. Batch 32/64 tăng chưa tới 1% so với 2/16,
đồng thời đổi CUDA rounding/cache identity nên giữ 16.

Full cache chạy song song với job 3.000. Hai cache jobs vẫn dùng chung GPU cho DINO,
nhưng phần lớn thời gian cache nằm ở CPU. Sau cache/targets/gate, full job chờ
job 3.000 kết thúc và nhả job lock trước GPU profile/SSL. Từ đó full training dùng GPU
một mình trong phạm vi hai jobs này. GPU tasks của người khác vẫn được giữ nguyên.
Nếu job nhỏ dừng vì lỗi/user stop, full tiếp tục sau khi job lock được nhả;
trạng thái kết thúc của dependency được lưu, không giả thành thành công.

Cache hoàn chỉnh từ job nhỏ được hardlink vào full root sau khi kiểm tra source SHA256,
interval/split, extractor identity, FPS, DINO batch và mọi array checksum.
Không copy payload hoặc viết lại features. Bundle full cũ đã publish nhưng chưa index
ở lần dừng trước được kiểm tra và phục hồi entry. Lúc startup tận dụng 700 full caches
(640 indexed + 60 phục hồi) và hardlink 2.432 train caches từ job nhỏ, khoảng 0,87 GiB.
Full job đã tạo thêm cache mới sau reuse; đây không chỉ là trạng thái process-alive.

Cache/targets/manifests/outputs của hai experiments vẫn riêng; full targets fit trên
toàn 40.154 train, không dùng targets của tập 3.000. Job full cũ giữ `stopped_by_user`
trong `artifacts/full_aslc/job_seed17`; job mới dùng `job_seed17_optimized`, tránh bypass
source/resume guard của job cũ. Full cũ chưa bắt đầu optimizer.

Dung lượng kế hoạch 43–48 GiB cho cache train/val/test, artifacts và dự phòng;
không gồm videos/weights/venv đã có. Hardlink giảm phần trùng nhưng không trừ vào
ngân sách dự phòng. Lúc launch còn khoảng 72 GiB. Cache bảo vệ 6 GiB dung lượng trống.
ETA dự trù 3–5 ngày đến train+val, 5–7 ngày đến cả test, tính từ tối 12/10/2026;
tức khoảng 15–17/10 và 17–19/10. Đây là ngoại suy, có thể đổi do shared disk/GPU,
độ dài video, thời gian gates/training và server downtime.

## Service và recovery

Unit: `signpath-e04-full-aslc.service`, enabled với `default.target`.
Linger của user haipd đã bật. Job sống qua SSH disconnect/logout; khi server reboot,
systemd tự khởi động và resume. Khi server tắt, không có tính toán diễn ra.

Job lock ngăn launcher trùng. SIGTERM/SIGINT chờ atomic boundary, lưu trạng thái
và checkpoint rồi exit 75; service tự resume. Restart khi signal/crash, giới hạn
3 starts/30 phút; scientific/config/gate/disk failures exit 2 và không tự retry.
Checkpoint mỗi 1.000 updates và graceful stop; mất điện đột ngột có thể phải replay
phần chưa checkpoint. Config/model source/evaluation/options phải giữ nguyên để resume.
Chưa thử reboot thật; model checkpoint/resume đã được kiểm tra ở preflight trước.

## TXT logs và thao tác

Full job tự ghi `progress.txt` và `status_latest.txt` cho cả hai jobs mỗi 60 giây bằng
một thread trong process full. Timer logger Python cũ của job nhỏ đã disable để không
thêm process thứ ba. Job nhỏ khi reboot cũng không tự mở reporter trùng.
`output.txt` là symlink tới stdout/stderr append `output.log`, cập nhật khi có output.

```bash
tail -f /home/haipd/Sign_Representations/method/signpathlab/artifacts/full_aslc/job_seed17_optimized/progress.txt
systemctl --user status signpath-e04-full-aslc.service --no-pager
# Graceful stop; unit enabled vẫn có thể chạy lại ở reboot sau.
systemctl --user stop signpath-e04-full-aslc.service
# Resume interruption, cùng identity.
systemctl --user restart signpath-e04-full-aslc.service
# Stop và bỏ tự khởi động sau reboot.
systemctl --user disable --now signpath-e04-full-aslc.service
```

Final result: `artifacts/full_aslc/job_seed17_optimized/result.json`.
Checkpoints: `artifacts/full_aslc/E04_seed17/<run_id>/checkpoints`.
Nếu full service dừng trong khi job nhỏ vẫn chạy, log nhỏ vẫn có stdout/stderr nhưng
progress snapshot không cập nhật cho tới khi reporter được khôi phục.

12 focused operational/recovery tests pass; systemd unit verification pass.
Startup evidence: [full_aslc_optimized_service_start.json](evidence/full_aslc_optimized_service_start.json).
Lần kiểm cuối 19:51:28: full cache train 3.190/40.154 (58 mới sau reuse), hai services
active/enabled, không restart lỗi; TXT snapshots của cả hai jobs được cập nhật.
Agent kiểm startup và tiến độ đầu rồi dừng polling theo yêu cầu người dùng.
