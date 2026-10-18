# Job A1 full160

Job production: `/home/haipd/Sign_Representations/method/phonopath_jepa/runtime/full160`.
Service: `phonopath-a1-full160.service` của user `haipd`. Đã enable; user linger bật, nên job độc lập với SSH và được gọi lại sau reboot khi các đường dẫn dữ liệu/môi trường đã sẵn sàng.

Cấu hình giữ nguyên A1: S + P + Chen-visible G; d256, 6 block, predictor128/4 block, batch32 BF16, seed0; 40.154 clip official train, 160 epoch = 200.640 optimizer update. Người dùng yêu cầu chạy trực tiếp A1 full160; job này không tự chạy G0, screening, downstream hoặc official test.

## Quản lý

```bash
cd /home/haipd/Sign_Representations/method/phonopath_jepa
scripts/a1_full160.sh status
scripts/a1_full160.sh logs
scripts/a1_full160.sh checkpoint
scripts/a1_full160.sh pause
scripts/a1_full160.sh resume
```

`pause` ghi cờ bền vững rồi dừng service, lưu tại ranh giới optimizer; `resume` xóa cờ và mở lại service từ checkpoint. Dừng service thủ công không đồng nghĩa hủy enable; dùng `pause` nếu muốn giữ trạng thái dừng qua reboot. Không chạy thêm `pp.train` vào cùng thư mục production.

Có thể dùng systemd trực tiếp:

```bash
systemctl --user status phonopath-a1-full160.service
systemctl --user restart phonopath-a1-full160.service
```

## Phục hồi

- Mã nguồn `code/pp/`, supervisor `ops/` và config được snapshot, khóa SHA256. Job không phụ thuộc những chỉnh sửa tiếp theo ở source method. Hash manifest dữ liệu cũng được kiểm tra khi supervisor khởi động.
- Checkpoint sau mỗi **200 update hoặc 60 giây**, điều kiện nào tới trước. SIGUSR1 yêu cầu checkpoint tại ranh giới update tiếp theo. SIGTERM/SIGINT yêu cầu lưu và thoát sạch.
- Commit ghi file tạm, fsync, atomic rename; giữ `latest.pt`, `previous.pt`, `initial.pt`. Nếu latest hỏng, kiểm tra và tiếp tục từ previous rồi initial; không tự reset run vì lỗi config/dữ liệu hoặc mất mọi checkpoint.
- Resume gồm student, EMA teacher, predictor, optimizer, centers, frozen normalization, RNG, epoch và vị trí trong permutation dữ liệu. Không đổi batch, budget, LR schedule hay lambda để xử lý lỗi.
- Watchdog theo tiến độ optimizer; 10 phút không tăng bước thì yêu cầu lưu/thoát, sau 180 giây chưa thoát thì kill nhóm process và thử phục hồi. Lỗi transient đợi 30 giây rồi retry. 8 lỗi liên tiếp không có tiến độ sẽ ghi cờ pause và `needs_attention`, tránh vòng lặp lỗi vô hạn.
- Supervisor có process lock; khi gặp child còn sống thì kiểm tra PID cùng birth tick để tránh nhân đôi trainer. Systemd tự mở lại supervisor khi crash.
- Khi bắt đầu/restart, chờ VRAM trống ≥14.000 MiB và tối đa 1 job GPU khác của **cùng user haipd**. Process của tài khoản khác được ghi riêng, không bị dừng/chỉnh sửa.
- Disk trống dưới 8 GiB: yêu cầu lưu và chờ, không xóa dữ liệu/job khác. Không có cơ chế nào bảo đảm không mất bước khi mất điện/SIGKILL; có thể phải chạy lại từ checkpoint bền vững gần nhất.

## File trạng thái

| File | Nội dung |
|---|---|
| `status.json` | Supervisor, trainer PID, stage, retry, GPU admission |
| `train/progress.json` | Update/200640, epoch, loss, speed, ETA, checkpoint step |
| `train/train_log.jsonl` | Log chi tiết, closure và sự kiện resume |
| `train/manifest.json` | Cấu hình, fingerprint, trạng thái train |
| `train/latest.pt`, `previous.pt`, `initial.pt` | Checkpoint phục hồi |
| `train/epoch_040.pt`, `_080.pt`, `_120.pt`, `_160.pt` | Checkpoint các mốc đăng ký |
| `train/final.pt` | Model cuối cùng khi đủ 200640 update |
| `stats.pt` | Median/IQR/still threshold fit từ 640 clip train, seed123 |
| `gpu_preflight.json` | Smoke thật CUDA/BF16 batch32, finite gradients và reload checkpoint |
| `preflight.json`, `lock.json` | Audit dữ liệu, cấu hình, snapshot |
| `service.log`, `train_stdout.log` | Log service và stdout của trainer |

Dòng log sau checkpoint có thể được replay nếu có ngắt đột ngột; mỗi lần resume ghi event và step đã khôi phục. Dùng `progress.json`/checkpoint làm nguồn trạng thái hiện tại, không đếm số dòng log để suy ra số update.

Smoke phục hồi độc lập tại `runtime/recovery_cli_test/`, bằng chứng `evidence/job_recovery_cli.json`: SIGUSR1, SIGTERM, checkpoint hỏng, fallback và hoàn tất 80 update. Smoke GPU ở `runtime/full160/gpu_preflight.json` không được dùng để khởi tạo production; production bắt đầu lại từ seed0 đúng recipe.

## Kiểm chứng trên production

Đã kiểm tra restart sạch và resume từ bước74. Sau đó yêu cầu checkpoint tại bước125 rồi mô phỏng supervisor SIGKILL: systemd tự restart (`NRestarts=1`) và trainer mới tăng tiếp từ checkpoint125. Bằng chứng: `evidence/full160_live_recovery.json`. Tốc độ train ban đầu khoảng 0,5 update/s trên GPU dùng chung; nếu giữ nguyên mức này, ETA khoảng 4–5 ngày, không phải 8–12 giờ như ước lượng trước benchmark.
