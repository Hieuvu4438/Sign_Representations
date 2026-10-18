# PLctx160 — job continuation tuần tự

## Material Passport

- Origin: academic-research-suite / experiment-agent, mode run, thực hiện inline.
- Mục tiêu: kiểm tra continuation của PLctx160 có tăng full-validation dictionary retrieval.
- Baseline: PLctx scratch/seed0,40.154trainclips,200.640optimizerupdates.
- Phạm vi xác minh: setup, native GPU replay và background service; chất lượng continuation đang đo.
- Không upload dữ liệu ra ngoài. Test không dùng chọn LR/nhánh/checkpoint.

## Service và lệnh quản lý

Service **`plctx160-continuation.service`**, `systemd --user`, enabled và `Linger=yes`.
Job ở `runtime/plctx160_continuation`; source/config/ops được snapshot, pin hashes.
Job không phụ thuộc SSH và tự start sau reboot. File lock ngăn hai supervisor
chạy cùng job; chỉ một stage GPU được launch tại một thời điểm trong job này.

```bash
cd /home/haipd/Sign_Representations/method/phonojepa
../../.venv-signrep/bin/python scripts/plctx_continue_job.py status
cat runtime/plctx160_continuation/progress.txt
journalctl --user -u plctx160-continuation.service -f

# Dừng an toàn; có checkpoint và pause marker
../../.venv-signrep/bin/python scripts/plctx_continue_job.py stop

# Tiếp tục cùng job
../../.venv-signrep/bin/python scripts/plctx_continue_job.py resume
```

Stop chủ động giữ `pause.request`, reboot không tự tiếp tục job đã pause.
Đóng SSH không cần stop. Không chạy launcher thứ hai, không sửa/xóa pinned job
files hoặc original base checkpoint/cache khi experiment chưa hoàn tất.

## Trình tự đã setup

1. **Baseline full validation:** original EMA teacher200.640, gallery toàn40.154
   train clips, query toàn10.304 official val. Reuse gallery features nguyên bản
   đã có và extract val bằng native raw/window64/stride2/_grp recipe.
2. **Nhánh lr1e5:** độc lập từ cùng checkpoint200.640, LRpeak1e-5, tối đa50.160
   additionalupdates. Không lấy weights từ nhánh khác.
3. **Nhánh lr3e5:** lại độc lập từ checkpoint200.640, LRpeak3e-5 và cùng budget/sampler.
4. **Selection:** so baseline và best validationRec@1 của hai nhánh; ties giữ
   baseline/ứng viên xuất hiện trước. Khóa checkpoint SHA trước official test.
5. **Official test:** chỉ đánh giá winner trên32.941testqueries/40.154traingallery.
   Nếu baseline vẫn tốt nhất thì reuse test scores/ranks nguyên bản đã có.
6. **Report:** `report.json`, `REPORT_VI.md`, `validation_iterations.csv`,
   `final_test/metrics.json` và ranks. Không chọn winner bằng test.

Các metrics: DCG, MRR, Rec@1/5/10 và macroRec@1; metric values0–1, khi hiển thị
Recall nhân100. Protocol raw/frozen, cosine, classmax qua gallery clips,
pessimistic ties. Cùng temporal readout với baseline, không PCA/whitening/LP mới.
Native handling zero vectors được giữ nguyên: không drop queries, zero query
nhận rank tie hữu hạn theo scorer cũ. Coverage được ghi riêng.

## Cấu hình continuation

| Thuộc tính | Giá trị |
|---|---|
| Base global step |200.640|
| Budget mỗi nhánh |50.160additionalupdates, tương đương40epochs|
| Maximum global step |250.800, tương đương200epochs tổng|
| Batch |32sourceclips, mỗi clip có near/far views|
| LR đầu/cuối phase |3e-6 /3e-6|
| LR peak |1e-5 hoặc3e-5|
| Schedule phase |Linear ramp500updates tới peak, rồi cosine xuống3e-6|
| Model/loss |Giữ PL/targetctx/CCE; glob, intrinsic objective, curriculum không bật mới|
| Optimizer |Giữ toàn bộ AdamW moments/steps từ baseline; wd0,05|
| Teacher |Giữ weights/center; EMA cố định0,9999, temperature0,06 ở endpoint recipe cũ|
| Validation |Mỗi5.000additionalupdates, cộng final phaseupdate|
| Early stopping |5lượt liên tiếp không vượt best validationRec@1, kể cả baseline|
| Best ban đầu mỗi nhánh |Weights/score của baseline, không dùng first/last ngẫu nhiên|
| Sampler |Cùng post-fork seed160200640 cho cả hai nhánh; shuffle/drop_last32|
| Workers |4train,4evaluation; spawn; mỗiworker1torchthread|
| GPU allocator cap |16% VRAM; đợi ít nhất6.144MiBfree trước stage|

Không đơn thuần đổi tổng `--steps` trong trainer gốc: continuation LR có timeline
riêng, mature teacher schedule giữ nguyên. Teacher/predictor/student/heads,
optimizer/centers/balancer được phục hồi, không khởi tạo model từ đầu.

Checkpoint gốc160epoch không lưu NumPy/CUDA RNG và sampler cursor. Vì vậy bước
fork từ baseline khởi đầu sampler continuation mới được khai báo rõ; không tuyên
bố tái dựng chính xác nextbatch của run lịch sử. Từ checkpoint continuation đầu,
resume lưu đầy đủ Python/NumPy/torch/CUDA RNG, sampler order/cursor/viewseeds,
optimizer, center, balancer, phase/global steps và metadata selection.

## Checkpoint, validation interruption và restart

`arms/<arm>/last.pt` ghi sau phaseupdate1, mỗi100updates, trước/sau validation và
graceful stop. File tmp→fsync→rename; `previous.pt` giữ bản trước, không bị overwrite
bằng latest hỏng. Resume thử last rồi previous chỉ với serialization corruption;
khác config/provenance thì báo lỗi. `best.pt` giữ best validation encoder và full
state. Hard kill/power loss có thể rollback tối đa khoảng100updates chưa commit.

Sampler chỉ commit batch trainer đã dùng; worker prefetch không đẩy checkpoint
cursor về tương lai. Interruption trong validation: resume lặp validation trước
optimizerupdate tiếp theo. Các feature shards có receipt/checksum. Nếu checkpoint
bytes thay đổi khi graceful save, validation cache có thể phải re-extract, nhưng
optimizer progress đã lưu không quay về baseline.

SIGTERM/SIGINT/SIGHUP đặt stop flag, save tại optimizer boundary. Shutdown timeout
300giây; nếu CPU scoring chưa thoát kịp thì checkpoint trước validation vẫn còn.
Systemd restart lỗi sau30giây, tối đa5lần/giờ. CUDA OOM do tranh GPU được supervisor
đợi120giây và chờ đủVRAM để resume, không đốt hết finite restart budget. Những lỗi
code/data khác vẫn báo failure. Job không stop hoặc sửa workload GPU khác.

## Cache, disk và tài nguyên

Original `runtime/cache/aslc_train_full.npz` và checkpoint160 giữ nguyên read-only.
Các NPY array bytes được extract nguyên từ NPZ vào `assets/train_arrays/`, không
đổi preprocessing/dtype/values; khoảng3,34GiB phát sinh. Spawn workers mmap các
files này để tránh pickle/copy toàn cache3,4GiB cho mỗi worker.

Cache evaluation chỉ giữ `_grp` pooled clip vectors theo shards256clips. Mỗi nhánh
giữ một feature-cache hiện hành; khi checkpoint thay đổi, cache đó được thay thế.
Metrics/ranks mọi validation đã hoàn tất vẫn giữ. Budget disk mới dự trù5–7GiB,
gồm memmap3,34GiB, checkpoint khoảng0,6GiB cho hai nhánh và eval/provenance/logs.
Service Nice5,CPUQuota8cores,MemoryHigh18GiB/Max24GiB; không copy original videos.

Original checkpoint/cache, stats, gallery-feature shards, officialCSV, source
snapshot và ops/config có SHA256 pins. Video/pickle source inventory giữ size/mtime
để phát hiện thay đổi trước mỗi stage. Test availability được kiểm tra khi setup;
test features/metrics không tham gia training/selection. Giữ independent splits.

## Kiểm chứng và giới hạn

- Tests CPU gồm native core tests và continuation tests: LR endpoint, spawn
  prefetch/sampler replay, full model/optimizer/center/loss equality, corrupt-latest
  fallback/provenance rejection, dictionary tie/denominator và interrupted
  validation replay không lặp optimizer update.
- Tổng47tests pass, không warning với `-W error`; JUnit
  `runtime/plctx_continuation_tests.xml`.
- GPU native batch32/BF16: save200.641, restore rồi nextbatch/model/teacher/center/
  loss bitwise equal. Native gallery features của4clips thực khớp cache cũ,
  max absolute difference0. Evidence ở `runtime/plctx_continuation_preflight/proof.json`.
- Baseline và hai continuation arms cùng seed/history ban đầu; đây chưa phải
  multi-seed comparison. Số epochs thêm có thể bị cắt bởi early stopping.
- Official test baseline đã được xem trước khi đề xuất continuation. Mọi lựa chọn
  mới dùng val, nhưng đây là exploratory improvement, không gọi untouched new test.
- Code giữ content-based handedness canonicalization của baseline, với declared
  one-bit leak trong proposal12. Không thay recipe này giữa các nhánh.

Estimate ban đầu cần gồm hai lầntrain thêm và tới22lượt full validation; validation
không phải proxy subset. Tốc độ thực tế cập nhật từ baseline extraction và train
updates đầu, không lấy thời gian của run3k để bảo đảm deadline full corpus.

## Kết quả lúc khởi động ngày15/10/2026

Baseline full-validation raw/_grp trên train40.154gallery và val10.304query:
DCG0,462072;MRR0,339943;Rec@122,7193%;Rec@545,9433%;Rec@1056,7838%;macroRec@123,0239%.
Đây là full train→val dictionary protocol, không phải LOSO/white512 trên val.

NhánhLR1e-5 đã chạy thực tế từ200.640 lên201.574; stop lưu checkpoint tại phase934
rồi resume tiếp cùng run. `service_resume_verification.json` ghi bước đã phục hồi
và tiến độ sau resume. Tốc độ300updates gần lần đo ban đầu khoảng0,105giây/update.
Hai nhánh tối đa cần khoảng3giờ trainingthuần; tới22fullvalidation và finaltest
chiếm thêm thời gian. Dự trù toànjob **8–12giờ** ở tảiGPU tương tự, có thể sớm hơn
nếu early stopping. Không coi estimate này là deadline hoặc kết quả chất lượng.
