# Pose-JEPA Hybrid trên toàn bộ ASL-Citizen

**Lưu trữ ngày15/10/2026:** training/test/probe đã hoàn tất. Theo yêu cầu người dùng,
runtime, weights, cache, features và environment riêng đã xóa; service đã gỡ.
[Báo cáo và archive đã xác minh](../../../docs/experiments/pose_jepa_results_20261015/REPORT_VI.md).
Các đường dẫn/lệnh bên dưới là hồ sơ setup lịch sử, không còn là job đang chạy.

Service: **`pose-jepa-full-aslc.service`**, quản lý bằng `systemd --user`.
Job: `/home/haipd/Sign_Representations/method/pose_jepa/runtime/full_aslc_job`.
Service được enable và user có `Linger=yes`: đóng SSH không dừng job; máy reboot
thì service được chạy lại. Một file lock ngăn chạy trùng cùng job.

## Theo dõi, dừng và resume

```bash
cd /home/haipd/Sign_Representations/method/pose_jepa
runtime/venv/bin/python scripts/full_aslc_job.py status
cat runtime/full_aslc_job/progress.txt
tail -f runtime/full_aslc_job/logs/train.log
journalctl --user -u pose-jepa-full-aslc.service -f

# Dừng có kiểm soát, lưu progress ở optimizer boundary
runtime/venv/bin/python scripts/full_aslc_job.py stop

# Tiếp tục cùng job từ cache/checkpoint hiện có
runtime/venv/bin/python scripts/full_aslc_job.py resume
```

Khi chuẩn bị dữ liệu, log ở `logs/prepare_train_val.log`; `train.log` xuất hiện
khi trainer bắt đầu. Status kiểm tra cả systemd lẫn progress JSON, tránh kết luận
process còn sống chỉ từ một file trạng thái cũ.

Stop tạo `pause.request`, nên reboot không tự chạy lại job đã chủ động pause.
Resume xóa marker, reset failed state rồi start service. Chỉ đóng SSH không cần stop.
Systemd restart sau lỗi tiến trình: chờ30 giây, tối đa5 lần/giờ. Không chạy launcher
thứ hai và không thay source/config của job đã khóa.

Từ14/10 lúc18h, supervisor chờ GPU có ít nhất6.144MiB VRAM trống trước train và
evaluation, kiểm tra mỗi30giây mà không tạo CUDA context. Status `waiting_for_gpu`
khác `running` của trainer: khi đang chờ thì optimizer không tiến lên. Probe này
không giữ chỗ VRAM; nếu có cạnh tranh sau probe và child lỗi CUDA OOM, supervisor
đợi ít nhất120giây rồi chờ GPU và resume checkpoint đã commit. Chỉ lỗi CUDA OOM
được retry theo cách này; lỗi dữ liệu/code vẫn đi qua failure policy của systemd.
Không dừng hay điều chỉnh job GPU khác. Khi pause trong lúc chờ, supervisor thoát
và marker pause vẫn giữ nguyên. Bằng chứng sửa resource handling:
`docs/evidence/full_aslc_gpu_queue_repair.json`.

## Cấu hình và protocol

| Thuộc tính | Giá trị |
|---|---|
| Method | `pose_jepa_hybrid_v1`, latent CE +0,25geometry +0,10motion |
| Khởi tạo | Scratch; chỉ reuse preprocessing pose, không nạp weights run3k |
| Official train | 40.154 clips, 35 signers, 2.731 glosses |
| Official validation | 10.304 clips, 6 signers |
| Official test | 32.941 clips, 11 signers |
| Encoder / predictor | 8layers/dim256 /4layers/dim128 |
| Sampling | Isolated whole-clip subsampling, tối đa128frames |
| Batch / precision | 32 / BF16 nếu GPU hỗ trợ |
| Optimizer | AdamW LR3e-4, betas0,9/0,95, weight decay0,05 |
| Budget | 125.482 successful optimizer updates, khoảng100 source epochs |
| Scheduler | Warmup5.000, cosine xuống1e-5 |
| Validation | Mỗi5.000 updates và cuối budget; raw dictionary Rec@1 |
| Early stopping | 5 lần validation liên tiếp không cải thiện |
| Checkpoint / export | Mỗi100updates / EMA encoder selected |
| Seed | 42 |

Selection dùng **toàn bộ train làm gallery, official val làm query**. Gloss được
score bằng max cosine qua các videos gallery. Tie dùng pessimistic `>=`, cùng
quy tắc dictionary evaluator lịch sử; raw selection không dùng whitening/PCA.
Các signer và source IDs của ba split phải disjoint. Exact video/pose hash overlap
được audit; nếu phát hiện thì stage báo lỗi, không silently bỏ các clips.

Official test được chuẩn bị/encode sau khi SSL training và checkpoint selection
hoàn tất. Test không vào SSL sampler, SSL loss hoặc checkpoint selection.
Validation là target-validation selection, không được mô tả là zero-shot selection.

## Cache và các stage

Pipeline: **prepare train/val → train/validate → prepare test → final evaluation**.

Môi trường hiện có được reuse; bổ sung `threadpoolctl==3.6.0` cho bước SVD whitening,
đã pin trong requirements lock và kiểm tra `pip check` thành công.

4 process workers chuẩn bị cache, queue tối đa8 clips. Dữ liệu nguồn được đọc tại
chỗ, không copy videos/pickle cache/environment. Existing13.304pose clips từ
`runtime/aslc3000_job/data` được reuse sau khi xác minh source identity, source pose
hash, output hash, raw score hash và coordinate/score/timing/tracking profile.
Manifests mới trỏ trực tiếp tới pose cache cũ khi reuse. **Giữ cache cũ đó**, vì full
job phụ thuộc các files đã chia sẻ; không cần giữ weights của run3k để train full.

Một số metadata subset cũ dùng hashed signer IDs và gloss aliases; full job tạo
mapping signer xác minh theo chính source IDs và đọc signer/gloss authoritative
từ official CSV. Pose bytes được reuse, nhãn/signer không copy mù từ metadata cũ.
Mọi signer trong full manifests dùng official Participant ID. Các con số LOSO/probes
full-val của báo cáo3k cũ sử dụng metadata mixed signer IDs nên cần rerank/re-evaluate
với mapping thống nhất trước khi so với full run. Selection3k trên8.804val clips
ngoài query subset vẫn dùng official IDs; đây không phải đổi checkpoint run3k.

Mỗi clip có receipt riêng. Resume cache kiểm tra và tiếp tục phần chưa chuẩn bị;
không decode phần hoàn tất từ đầu. Cache giữ native video PTS, frame-count parity,
person tracking, raw all-person scores và conversion `q=s/(1+s)` đã khai báo.
Detector weights gốc chưa được xác minh, score là uncalibrated reliability.
Normalization failure được ghi coverage, không silently thay sample hoặc bỏ query.

## Checkpoint, interruption và tài nguyên

Full checkpoint giữ student, teacher, predictor, physical heads, optimizer,
EMA center, scaler, step, RNG/mask RNG, permutation/cursor và metadata provenance.
DataLoader prefetch không làm checkpoint cursor chạy trước dữ liệu đã tiêu thụ.
`last.pt` ghi atomic; `previous.pt` giữ bản trước. Corrupt latest có thể fallback
previous, đồng thời ghi `train/resume.json`; đổi source/data/config thì từ chối resume.

SIGTERM/SIGINT/SIGHUP lưu ở update boundary. Validation kiểm tra stop giữa các clips;
resume chạy lại validation còn dang dở trước update tiếp theo. Hard kill/mất điện
có thể rollback tối đa khoảng100updates chưa checkpoint; không bảo đảm giữ state
chưa commit. Shutdown service có timeout300giây.

Snapshot source dưới `code/`, ops dưới `ops/`; `job_lock.json` pin hashes của source,
ops, config, input index, job metadata và official split CSVs. Không sửa/xóa các
files này khi job đang chạy. Các sửa trong repo không đổi snapshot của run.

Job chia sẻ RTX5880Ada hiện có, không stop/chỉnh các job khác. CUDA allocator giới
hạn15% VRAM (~7,2GiB); GPU smoke batch32/BF16 peak tensor allocation khoảng2,3GiB.
Service Nice5, CPU quota8cores, MemoryHigh12GiB, MemoryMax16GiB. Prepare kiểm tra
reserve6GiBdisk và dừng có lỗi nếu xuống dưới mức đó, giữ receipts/checkpoints.

Ước tính training+validation100epochs:12–24giờ, có thể đổi theo GPU dùng chung.
Cache mới/progress/evaluation cần dự trù20GiB, tận dụng các cache đã có. Thời gian
full final evaluation được theo dõi riêng, không coi estimate là deadline.

## Final evaluation

Final encoder là `train/best_validation.pt`, export thành `evaluation/bundle/`.
Feature extraction có per-clip checkpoint/pose receipt và resume. Reports:

- `raw_val_retrieval.json`, `raw_test_retrieval.json`: raw1024, DCG, MRR,
  Rec@1/5/10, macroRec@1 và ranks/failures từng query.
- `white512_val_retrieval.json`, `white512_test_retrieval.json`: SVD whitening chỉ
  fit valid train/gallery features; query/test không vào fitting.
- `coverage.json`: failed IDs và số features hợp lệ ở từng split.
- `linear_probe/`: encoder cố định, raw1024 standardized bằng train stats; linear
  softmax head AdamW lr1e-3/wd1e-3, batch512, tối đa30epochs, validation patience5.
  Chọn head bằng val; cuối cùng báo test instance/macroTop1/Top5. Probe resume ở
  epoch boundary; epoch dang dở có thể chạy lại. Đây là recipe full vocabulary,
  khác historical ASLC300 sklearn logistic C-search.
- `report.json`, `complete.json`: report tổng và marker chỉ có khi toàn bộ các bước
  đánh giá trên đã thành công. Body-part/signer probes không được schedule trong
  job full dictionary này; không giả định có metrics chưa chạy.

Invalid gallery không tạo candidates, invalid/missing-positive query rank∞ và
giữ trong mẫu số; invalid linear-probe query tính sai. Raw embeddings không có
learned metric head; classification probe không fine-tune encoder.

## Kiểm chứng

- 66tests pass, không warning: `docs/evidence/tests_full_aslc_job.xml`.
- GPU batch32/BF16: save/resume step1 rồi nextbatch, student/teacher và losses
  bitwise equal; CUDA dictionary ties/failure check pass:
  `docs/evidence/full_aslc_gpu_preflight.json`.
- Actual cache reuse và mapping signer pass:
  `docs/evidence/full_aslc_cache_reuse.json`.
- Tests riêng cho chunked full dictionary denominator/ties, signer split overlap,
  pause/resume validation ở final step không lặp optimizer update và frozen linear
  probe interruption/resume với query lỗi.
- Queue GPU có test cho low-memory -> đủ memory, pause trong khi chờ, giữ counter
  checkpoint và phân loại OOM chỉ từ log của attempt hiện tại.
- Service thật được stop khi cache đã hoàn thành4.324 clips rồi resume bằng PID
  mới, tiếp tục vượt6.263 clips. Receipt timestamps đã hoàn thành giữ nguyên:
  `docs/evidence/full_aslc_service_resume.json`. Service startup/linger/source lock:
  `docs/evidence/full_aslc_setup.json`.

Các checks là bằng chứng kỹ thuật cho setup, không phải chất lượng full benchmark.
