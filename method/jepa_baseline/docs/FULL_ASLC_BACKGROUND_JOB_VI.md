# SignRep full ASL-Citizen: job nền và resume

**Cập nhật 2026-10-14:** người dùng đồng ý loại đúng một clip train11frames.
Effective SSL train/full gallery còn40.153 videos, đủ2.731 lớp; official manifests
và val/test giữ nguyên. Xem FULL_ASLC_SHORT_CLIP_REPAIR_VI.md và job audits.
Mọi mô tả40.154 bên dưới là membership official trước exception.

Job được setup và khởi chạy ngày 2026-10-13, theo cấu hình người dùng chọn:
Kinetics MAE encoder, heads mới, batch20/BF16, toàn bộ official train, scheduler
100.000 bước. **Dừng ở 50.000 bước để review validation**, không tự chạy test hoặc
chạy tiếp 100.000. Dataset không bị copy. Không dùng checkpoint SignRep pretrained.

## Service và log

- Service: `signrep-full-aslc-kinetics.service` (`systemd --user`).
- `Linger=yes` đã xác nhận; service độc lập với terminal/SSH và được enable.
- Job directory: `/home/haipd/Sign_Representations/method/jepa_baseline/runtime/full_aslc_kinetics`.
- `progress.txt`: log dễ đọc append liên tục, phù hợp `tail -f`.
- `console.log`: stdout/stderr chi tiết của mọi stage/worker.
- `status.txt` / `status.json`: trạng thái mới nhất.
- `train/metrics.jsonl`: loss, LR, coverage, thời gian mỗi update.

```bash
tail -f /home/haipd/Sign_Representations/method/jepa_baseline/runtime/full_aslc_kinetics/progress.txt
tail -f /home/haipd/Sign_Representations/method/jepa_baseline/runtime/full_aslc_kinetics/console.log
systemctl --user status signrep-full-aslc-kinetics.service
```

Không chạy launcher khác cho cùng job; pipeline có file lock để chống chạy trùng.
Không sửa/xóa `runtime/full_aslc_kinetics/code`, config, manifests hoặc checkpoints
trong job đang chạy. Source/config/assets được hash; service dùng code snapshot,
không dùng các thay đổi sau này trong source repo. Môi trường Python vẫn phải giữ.

## Pipeline tự động

1. Official manifests: 40.154 train, 10.304 val, 32.941 test, 2.731 lớp.
2. Reuse 3.000 raw teacher caches đã xác minh từ ASLC002, rồi trích phần còn lại.
   Cache reuse chỉ là preprocessing, không nạp training weights từ ASLC002.
3. Tính median bone statistics full train và dựng standardized caches mới.
4. Audit corpus / coverage; không âm thầm loại video nếu teacher thiếu valid clip.
   Trường hợp này ghi pose_report.json và dừng yêu cầu review dữ liệu.
5. Train với 4 CPU data workers, prefetch2, 2 threads/worker và fixed per-step seeds.
6. Proxy validation đủ 2.731 lớp mỗi10.000 updates, pooled mean primary / weighted riêng.
7. Ở50.000, full validation cho best proxy model và checkpoint50k, chọn bằng validation,
   lưu `review.json`, ghi `REVIEW_REQUIRED` và service kết thúc bình thường.
8. **Test chưa được evaluate.** Xem review trước khi chọn chạy tiếp hoặc chốt final test.

Service giới hạn CPU quota10cores, memory high22GiB/max26GiB; Nice5. CUDA allocator
giới hạn22% GPU; batch20/BF16 đã đo peak khoảng6550MiB. Những giới hạn này chia sẻ
máy với các service đang chạy, không dừng hoặc đổi cấu hình job khác.
Các số đo GPU là technical smoke, không phải benchmark quality/full-training ETA.

## Dừng và tiếp tục

```bash
# Dừng có kiểm soát: chờ update đang chạy xong, rồi lưu checkpoint.
systemctl --user stop signrep-full-aslc-kinetics.service

# Chạy lại: tự reuse pose caches / normalization và resume last.pt.
systemctl --user start signrep-full-aslc-kinetics.service
```

Tắt SSH hoặc đóng terminal không dừng service. Stop có kiểm soát gửi SIGTERM đến
supervisor, rồi child stage. Training hoàn tất update hiện tại, lưu checkpoint full.
Pose đang viết được coi là partial; các files đã hoàn thành được reuse.

Nếu bị SIGKILL/OOM/power loss, không thể lưu trạng thái đang ở RAM: resume checkpoint
đã ghi atomic gần nhất, không về bước0. Save mỗi500updates hoặc300seconds, và khi
dừng có kiểm soát. Mức rollback tối đa giữa các saves phụ thuộc thời gian một update;
không hứa giữ được update chưa commit. `last.previous.pt` là bản trước đó để dự phòng.
Service restart-on-failure, tối đa3 lần trong30phút; lỗi input/contracts hoặc data
review (exit78) không retry vô hạn. Khi power on, user service enabled có thể chạy lại;
CUDA/resources phải sẵn sàng, nếu quá số retries thì dùng start sau khi xử lý lỗi.

Checkpoint full gồm model, discriminator, EMA, hai optimizers/schedulers, AMP scaler,
CPU/CUDA/NumPy/Python RNG, sampler permutation/cursor và source/data contracts.
Loader chỉ commit sampler sau optimizer update hoàn tất. Queue prefetch không cần
lưu: seeds theo update/position dựng lại batch sau resume, không phụ thuộc worker order.
Log metrics bị truncate sau forced interruption được rollback tới checkpoint step.
Checkpoint/model export và cache được ghi filetạm rồi replace atomic; checkpoint fsync.

Không dùng `last.model.pt` để resume optimizer. File full `last.pt` dành cho resume;
`last.model.pt`, `best.model.pt`, `step50000.model.pt` dành cho frozen evaluation.

Sau khi đã review full validation tại50k và quyết định chạy tiếp:

```bash
.venv-signrep/bin/python method/jepa_baseline/scripts/control_full_aslc_job.py continue-100k
```

Lệnh này chỉ đổi stage_limit50k→100k trong control.json, không sửa scheduler/config,
và khởi chạy lại cùng service. **Chưa gọi lệnh này trong setup ban đầu.**
Kiểm tra nhanh bằng `control_full_aslc_job.py status`.

## Kiểm chứng trước launch

- 36tests pass: ordered prefetch/resume, toàn trạng thái optimizer/RNG, metrics-log
  recovery, preprocessing frame reuse, ranking/tie parity và tests baseline trước đó.
- GPU thật, Kinetics-only initialization,20views/BF16: nạp heads mới, train/backward,
  save/resume, nextbatch/loss/model/discriminator/EMA bitwise equal.
  Evidence: `evidence/full_aslc_gpu_resume.json`.
- Native teacher normalization: arrays bitwise bằng cache đã có; restart reuse đúng
  filehash; compact metadata1686chars thay292615chars mà vẫn pin full bone table SHA.
- Raw cache receipts và source/asset pins kiểm tra trước reuse/resume.
- Service startup/stop/start thực tế được kiểm tra riêng và ghi evidence ở job.

Các checks trên không chứng minh score tốt hoặc reproduce paper500k; review quality
vẫn dựa trên validation, full-corpus coverage và style/activity diagnostics.
