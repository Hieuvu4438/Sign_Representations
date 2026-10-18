# E04 — full ASL Citizen, seed17: setup và khởi động

**Job mới 12/10, 19:47:** đã khởi động full seed 17 qua systemd, 2 CPU threads / DINO
batch 16, dùng `artifacts/full_aslc/job_seed17_optimized`. Xem
[service hiện tại](FULL_ASLC_SERVICE_VI.md). Phần bên dưới là hồ sơ full job cũ đã dừng.

**Cập nhật:** job dưới đây đã dừng theo yêu cầu người dùng ở stagecache_train;
status=`stopped_by_user`. Người dùng chuyển sang3.000trainvideos/max2process và muốn
review trước khi chạy. Xem [setup mới](ASLC3000_PREVIEW_VI.md). Các mô tả RUNNING/timings
bên dưới là hồ sơ khởi động trước yêu cầu dừng, không phải trạng thái hiện tại.
Launchers hiện đã đổi sang1CPUthread, không resume job full cũ.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: run; setup và kiểm tra khởi động theo yêu cầu người dùng.
- Origin Date: 2026-10-12.
- Verification Status: VERIFIED cho setup/preflight; full training/retrieval đang RUNNING.
- Version Label: E04_full_aslc_retrieval_seed17_v1.

## Trạng thái bàn giao

Job nền PID **1411678** đã được kiểm tra sống, đang ở `cache_train`. Lần quan sát ghi
trong `evidence/full_aslc_setup.json` đã tạo ít nhất240/40154train caches. **Chưa bắt đầu
lượt optimizer chính** vì cần hoàn thành full train/val cache, fit targets và gates trước.
Không có final DCG/Rec@1/Rec@5 từ main run tại thời điểm này.

Theo yêu cầu người dùng, agent dừng polling sau khi setup/khởi động được xác minh.
Job `nohup setsid` tiếp tục chạy độc lập khi kết thúc hội thoại; không cần agent chờ.

## Dữ liệu và config đã khóa

| Phần | Giá trị |
| --- | --- |
| Data | ASL Citizen official CSV, toàn bộ nguồn available, không subset theo nhãn |
| Train / gallery | 40.154 videos, 2.731 glosses, 35 signers |
| Validation | 10.304 videos, 2.731 glosses, 6 signers |
| Final test | 32.941 videos, 2.731 glosses, 11 signers |
| Seed | 17 |
| Model | E04 target-only; D384, encoder6layers/6heads, predictor2layers |
| Streams | Global/left/right/face DINOv2 + body/left/right static pose |
| SSL objective | Semantic normalized L1 + geometry SmoothL1, λgeo=.1 |
| Geometry | Logsig degree2 + anchors/timing,11paths,3scales, projected32D |
| Masks | Strict synchronized temporal50%, full source-support closure |
| Sampling | Shuffle mọi source train theo epoch; random contiguous48selected samples/source |
| Budget | 10.000 successful updates, microbatch8 × accumulation4 → effective batch32 |
| Exposure | 320.000 sampled clips nếu không skipped/resampled batch, khoảng7.97source epochs |
| Optimizer | AdamW lr3e−4, betas(.9,.95), weight_decay.05, warmup5%, cosine LR/EMA |
| Precision | BF16 trên RTX5880Ada, geometry tính ngoài low-precision autocast |
| Checkpoints/diagnostics | Mỗi1.000updates;3recent step checkpoints + best/last |
| Validation | Frozen dictionary retrieval tại step10.000; một checkpoint theo fixed budget |
| Classifier | Không train classifier, không fine-tune encoder cho retrieval |

Config: `configs/full_aslc/E04_seed17.yaml`. `data/full_aslc/manifests` có hashes thực.
Exact sample/source/hash/signer audit của cả3splits pass; missing files/hashes/signer IDs=0.
Import đã đọc/hash test assets để kiểm availability/split, nhưng test không dùng trong
SSL fitting, sampler, validation hoặc lựa chọn model. Test feature extraction/eval được
hoãn tới sau training và checkpoint lock.

Runtime implementation hash:
`3b58b893d2ad39dd89395eed5ebee1a7cf5fe6a4535e883a7c8ce0752f9fb346`.
Job lưu source snapshot, environment/lock, resolved config và markers cho từng stage.
Không sửa `src/signpath` trong khi job chạy; runner từ chối source/config identity đổi
để tránh một experiment trộn nhiều implementations.

## Protocol retrieval

Tham chiếu trực tiếp `../obs_jepa/oj/compare_retrieval_avg.py` và SignRep §6/§7.2:
[paper](https://arxiv.org/html/2503.08529v1). Frozen encoder tạo full-video embeddings,
L2 một lần ở clip level, cosine similarity; score mỗi gloss là max qua các video gallery
của gloss đó. Xếp hạng gloss đúng bằng `1 + số gloss khác có score >= score đúng`.

- DCG = mean `1/log2(1+rank)`.
- Rec@1 = mean `rank<=1`.
- Rec@5 = mean `rank<=5`.
- MRR cũng lưu bổ sung. Metrics lưu0..1, nhân100khi trình bày.

Đây là **dictionary/gloss ranking**, khác generic video-level retrieval của pilot trước.
Metric parity với code tham chiếu đã pass cho300glosses/600gallery/31queries, có zero
embedding/tie cases; exact ranks match, metric tolerance1e−12.

SignPathLab dùng timestamp-aligned48sample windows, overlap16, valid-stream/frame mean,
distinct source-frame pooling. Không có SignRep activity-weighted pooling/16-frame
Hiera/stride2 hoặc RGB-only architecture. Vì vậy không gọi exact SignRep reproduction.
Full2.731gloss results cũng không trực tiếp so số với bảng ASLC300subset trước đây.

## Pipeline đang chạy nền

`scripts/run_full_aslc.py` tự chạy theo thứ tự:

1. Verify official manifests và actual availability/source hashes.
2. CPU IMAGE-mode pose/hand/face với4workers; GPU DINOv2; tạo lossless compressed train cache.
3. Tương tự validation cache.
4. Fit geometry statistics/projection trên **toàn bộ train sources**, bounded-memory moments.
5. Exact/near-duplicate/closure gate, rồi profile actual cached training.
6. Train E04 tới10.000successful updates; final frozen validation retrieval; lưu best/last.
7. Tạo test cache sau khi khóa checkpoint.
8. Verify test split/input identity; frozen test dictionary retrieval và result.json.

Không có automatic retry khi error. Nếu fingerprints tạo candidates, gate yêu cầu recorded
review thay vì tự bỏ kiểm tra. Khi đó status ghi failed/error và agent cần đọc evidence
trước khi tiếp tục. Job cũng dừng nếu còn dưới6GiBdisk reserve; các caches đã publish giữ lại.

Đã sửa cho full-scale runtime: supports và feature arrays nén lossless; shared support
sets được intern khi decode; raw intermediates mỗi clip chỉ xóa **sau cache validate**,
giữ checksum/provenance receipts và original dataset. CPU queue bị giới hạn8clips;
không tạo tất cả raw landmark caches trước. Pixel/normalization arrays có thể regenerate
từ original videos và pinned detectors; full-run intermediate arrays không được giữ dài hạn.
Stats fit dùng population mean/covariance streaming, đã kiểm parity với materialized fit.

## Kiểm tra đã chạy

- **133tests pass**, không skip/failure; `artifacts/full_aslc/all_tests.log`.
- `pip check` pass; full config validation/script syntax pass.
- 24realvideos preflight:16train/8val, source split audit, CPU+GPU extraction, compressed
  cache, train-only targets, pipeline gate, actual E04D3843updates, checkpoint và retrieval.
- Preflight loss qua3updates:1.0844→.9100→.8228; không có classifier/head.pt.
- Preflight là **correctness/calibration**, không phải final dataset score. Metrics của
  8gloss subset không được trình bày thành kết quả full ASL Citizen.
- Evidence: `evidence/full_aslc_preflight.json`, `full_aslc_metric_parity.json`,
  `full_aslc_manifest_audit.json`, `full_aslc_setup.json`.

## Ước lượng thời gian và storage

Máy: một RTX5880Ada48GiB,62GiBhost RAM. Đo ban đầu200–240fulltrainclips khoảng
1.70–1.75s/video cho combined extraction/cache. Preflight D3843updates +24clipvalidation
khoảng14.15s; pool nhỏ có cache reuse nhiều hơn full corpus. Ước lượng chưa có full-scale
steady-state profile:

| Stage | Khoảng giờ |
| --- | ---: |
| Train+val preprocess/cache | 22–30 |
| Train-only targets và verification gates | 4–8 |
| SSL10.000updates | 10–20 |
| Frozen validation retrieval | 2–5 |
| Test preprocess/cache | 14–20 |
| Frozen test retrieval | 3–6 |
| **Toàn pipeline** | **Khoảng55–90giờ, tức2–4ngày** |

Đây là ngoại suy, không là deadline hay measured full-run time. Video dài/quality,
cache reads, CPU supports, GPU cạnh tranh và gates có thể đổi runtime. Training có
max_runtime_hours96cho riêng runtime train; không đặt timeout30phút cho full pipeline.

Lossless cache estimate từ24clipreal≈353KB/video, toàn83.399clips≈27.4GiB trước các
overhead khác. Free disk lúc startup≈43.7GiB; hiện reserve6GiB. Nếu clips thật trung bình
lớn hơn preflight hoặc tác vụ khác dùng đĩa, status có thể dừng trước test stage.
Không xóa dữ liệu/code/cache của method khác để lấy dung lượng.

## Xem trạng thái hoặc nhận kết quả

Chạy từ `method/signpathlab`:

```bash
.venv-locked/bin/python scripts/status_full_aslc.py
tail -n 40 artifacts/full_aslc/job_seed17/output.log
```

`status_full_aslc.py` kiểm process alive ngay lúc gọi, đọc progress stage và latest
successful update nếu train đã bắt đầu. Nó không polling liên tục, không restart job.

- Job status: `artifacts/full_aslc/job_seed17/status.json`.
- Stage progress: `cache.train.json`, `cache.val.json`, `cache.test.json` trong job folder.
- SSL run: `artifacts/full_aslc/E04_seed17/<run_id>/`, model checkpoints + logs.jsonl.
- Final result khi complete: `artifacts/full_aslc/job_seed17/result.json`.
- Test embeddings/metrics/per-query ranks: `job_seed17/retrieval_test/`.
- Original launch command: `bash scripts/launch_full_aslc.sh` đã chạy; **không gọi lại**
  khi job đang sống. Launcher từ chối job folder đã có status.

Nếu job stopped/failed, giữ nguyên evidence và kiểm error. Explicit resume khi đã xử lý
nguyên nhân và source/config vẫn đúng:

```bash
nohup setsid env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-locked/bin/python scripts/run_full_aslc.py \
  --config configs/full_aslc/E04_seed17.yaml --device cuda:0 --workers 4 \
  --job-dir artifacts/full_aslc/job_seed17 --resume \
  >> artifacts/full_aslc/job_seed17/output.log 2>&1 < /dev/null &
```

Không chạy resume đồng thời với process hiện tại. Job có exclusive lock. Gate/source
mismatch cần review trước; command resume không bypass validation hoặc tự đổi protocol.
