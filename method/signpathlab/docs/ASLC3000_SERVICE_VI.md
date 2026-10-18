# E04 ASLC3000 — systemd service và recovery

**Cập nhật 12/10, 19:47:** người dùng đã cho phép thêm job full. Full service tạo cache
song song, chờ job nhỏ kết thúc trước GPU profile/SSL. TXT reporter hiện chạy trong
process full và ghi cho cả hai jobs mỗi 60 giây; timer Python logger bên dưới đã disable
để giữ hai experiment processes. Xem [service full](FULL_ASLC_SERVICE_VI.md).

**Cập nhật tăng tốc:** service hiện dùng2CPUworkerthreads riêng biệt trong1Pythonprocess,
batchDINO16. [Benchmark](ASLC3000_CACHE_OPTIMIZATION_VI.md) cho cache speedup1,93×;
94caches cũ được giữ. Các số131tests/oneCPUthread và ETA16–25giờ bên dưới là hồ sơ
launch trước tối ưu; hiện133tests pass và ETA ngoại suy mới10–16giờ cho wholepipeline.

Job được người dùng cho phép khởi động sau review setup. Service chính:
`signpath-e04-aslc3000.service`, user `haipd`, enabled ở `default.target`.
`loginctl show-user haipd -p Linger` đã xác minh `Linger=yes`, nên user manager/service
không phụ thuộc phiên SSH hoặc login đang mở.

Unit source: `deploy/signpath-e04-aslc3000.service`.
Installed unit: `/home/haipd/.config/systemd/user/signpath-e04-aslc3000.service`.
Python job dùng1process; CPUextraction dùng2threads riêng trong cùng process, không tạo
workerprocess/resource_tracker/DDP. Một số git/doctorasset probes là subprocess ngắn,
cùng coordinator vẫn trong giới hạn2process. Native MediaPipe/PyTorch threads không
phải separate Python processes.

## Scope đã khóa

- E04, seed17, D384/encoder6layers/predictor2layers.
- Same ASLC002300glosspool:3.000officialtrain/gallery +1.500officialval/query.
-4.000successful updates, microbatch8×accumulation4=effective32.
- Frozen dictionary retrieval DCG/Rec@1/Rec@5; không classifier hoặc finalteststage.
- Config `configs/aslc3000/E04_seed17.yaml`; membership [audit](evidence/aslc3000_selection.json).
- Setup/storage chi tiết ở [preview](ASLC3000_PREVIEW_VI.md).

Service chạy pipeline cachetrain→cacheval→train-onlytargets→gate→profile→SSL→retrieval.
Việc service đã active không có nghĩa optimizer đã bắt đầu; status.stage cho biết
stage đang thực hiện. Kết quả cuối chỉ có khi `job_seed17/result.json` được commit.

## Hành vi khi ngắt kết nối hoặc server

| Tình huống | Hành vi |
| --- | --- |
| Đóng terminal/ngắt SSH/logout | Service tiếp tục; không gắn với shell hoặc tmux session |
| Server reboot/tắt rồi mở lại | Service enabled tự khởi động; runner `--auto-resume` đọc stage markers/cache/checkpoint |
| SIGTERM khi đang train | Signal chỉ đặt request; trainer kết thúc update hiện tại, lưu checkpoint ở safe boundary rồi exit75 |
| SIGKILL/power loss đột ngột | Dùng checkpoint đã publish atomically; các updates sau checkpoint phải replay |
| Source/config/data identity thay đổi | Từ chối resume; không trộn implementations |
| Lỗi dữ liệu/hash/near-duplicate gate | Exit2, dừng để review; không automatic retry liên tục |
| Job complete | Exit0, không restart liên tục; reboot sau đó thấy completed thì không train lại |

Server **không thể tính toán lúc tắt**. Recovery bắt đầu khi OS/dataset/weights/GPU
đã có lại. Không reboot máy để test vì đang dùng chung; đã kiểm actual SIGTERM và
service restart. Cấu hình boot/linger/enable đã được verify ở systemd hiện tại.

Checkpoint initialstep0 được lưu trước update1; mỗi1.000successfulsteps lưu cả
retained step snapshot và `last.pt`, cuối run/SIGTERM cũng lưu last. Abrupt power loss
có thể mất tiến độ sau checkpoint gần nhất, tối đa khoảng999updates theo cadence này;
successful-update budget cuối vẫn4.000, không cộng duplicate replay làm extra training.

Khi resume, log rows sau durable attempted-batch counter được archive ở run/recovery
rồi replay sampler/RNG từ checkpoint. Các target/coordinate files chưa có metadata
commit marker được quarantine và regenerate, không silently overwrite published
artifacts. Nếu bị ngắt trong finalretrieval sau khi đủ optimizerupdates, resume thực
hiện lại evaluation, không bỏ qua vì counter đã đạt4.000.

Service dùng `Restart=on-abnormal`, `RestartForceExitStatus=75`, restartdelay30s,
startlimit3lần/30phút. Application scientific failures exit2không auto retry. TimeoutStop
3phút để checkpoint/cleanup; operator `systemctl stop` không tự restart ngay.

## Verification trước khi launch chính

- **131tests pass**, unit +single-processintegration +recovery cases; không chạy
  multi-processDDP tests dưới userlimit2.
- Real operationalpreflight:2train/1valvideos, actualD384cached training.
- Gửi SIGTERM giữa training: durablecheckpoint step1, exit75, statusinterrupted.
- Restart cùng config/source: tiếp tụcstep2, traininglog chỉ[1,2], finalretrievalcomplete,
  serviceexit0. Không trainclassifier.
- Unit được `systemd-analyze --user verify` kiểm; user service installed/enabled và
  linger đã bật. Realreboot chưa thực hiện, không gọi đó là testedserverreboot.
- Evidence: [preflight/recovery](evidence/aslc3000_service_preflight.json).

## Xem trạng thái/log

Log dễ đọc cập nhật mỗi60giây: `artifacts/aslc3000/job_seed17/progress.txt`.
Snapshot mới nhất: `status_latest.txt`. Fullstdout/stderr: `output.txt` là symlink
cùng nội dung với `output.log`, append ngay khi job ghi. Timer
`signpath-e04-aslc3000-log.timer` enabled, vẫn hoạt động sau logout/reboot. Formatter
oneshot chỉ đọc artifacts, không sửa model/source hoặc restart training; khi chạy
có thêm1Pythonprocess ngắn, cùng training tối đa2Pythonprocess. Không có watcher
Python chạy liên tục. Để dừng log timer dùng
`systemctl --user disable --now signpath-e04-aslc3000-log.timer`.

```bash
tail -f /home/haipd/Sign_Representations/method/signpathlab/artifacts/aslc3000/job_seed17/progress.txt
```

```bash
systemctl --user status signpath-e04-aslc3000.service --no-pager
cd /home/haipd/Sign_Representations/method/signpathlab
.venv-locked/bin/python scripts/status_full_aslc.py --job-dir artifacts/aslc3000/job_seed17
tail -n 40 artifacts/aslc3000/job_seed17/output.log
```

Các lệnh chỉ đọc một lần. Agent dừng polling sau verificationstartup; service tiếp tục.
Không chạy `launch_aslc3000.sh` hay secondPythonjob đồng thời với service.

Thư mục outputs:

- `artifacts/aslc3000/job_seed17/status.json`: PID/currentstage/status.
- `job_seed17/cache.{train,val}.json`: cacheprogress.
- `job_seed17/output.log`: logs append qua mọi service starts.
- `artifacts/aslc3000/E04_seed17/<run_id>/checkpoints`: model/optimizer/EMA/RNG/sampler.
- `job_seed17/result.json`: finalDCG/Rec@1/Rec@5 khicompleted.

Controlcommands cho người dùng:

```bash
# Graceful stop; enabled unit sẽ vẫn chạy ở reboot tiếp theo.
systemctl --user stop signpath-e04-aslc3000.service
# Graceful restart/resume cùng identity.
systemctl --user restart signpath-e04-aslc3000.service
# Dừng và không tự khởi động ở boot tiếp theo.
systemctl --user disable --now signpath-e04-aslc3000.service
```

Nếu failed vì gate/data/source, đọc output.log và báo lại để review. Không dùng restart
để bypass; automaticresume từ chối statusfailed. Explicitresume chỉ dùng sau review lỗi.
Giữ source/config frozen trong khi experiment đang chạy.

Ước lượng hiện tại vẫn khoảng16–25giờ cho cả pipeline theo single-processcalibration;
đây là ngoại suy, không là deadline. Tắtserver làm thời gian calendar dài hơn.
