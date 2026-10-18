# ASLC3000 — benchmark và lựa chọn tăng tốc cache

Theo yêu cầu người dùng, đã thử cả2hướng: independent CPUworkerthreads và DINO batch
32/64. Đã chọn **2CPUthreads, batchDINO16**, vẫn **1Pythonprocess**. Service chính đã
được cập nhật và khởi động lại; không đổi3.000train/1.500val, seed17, D384 hoặc4.000updates.

## Benchmark thực

8officialtrainvideos chọn trước theo seed2718 từ manifest3.000, cùng video/PTS/assets/
sampling cho mọi candidate. Mỗi candidate2lượt, thứ tự đảo ở lượt2. GPUbackbones warm-up
trước timing; per-run detector init/decode/pose/ROI/DINO/serialization/checksum đều nằm
trong measured walltime. Không dùng cache reuse ở benchmark; CPUqueue bounded.

Script: `scripts/benchmark_cache_options.py`.
Full evidence: [benchmark JSON](evidence/aslc3000_cache_benchmark.json).

| CPUthreads | DINO batch | Mean wall seconds /8clips | Speedup so với1thread/batch16 |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 71,27 | 1,00× |
| **2** | **16** | **36,88** | **1,93×** |
| 1 | 32 | 72,29 | 0,99× |
| 1 | 64 | 70,64 | 1,01× |
| 2 | 32 | 36,54 | 1,95× |
| 2 | 64 | 36,66 | 1,94× |

2thread/batch16lượt1=37,43s,lượt2=36,34s; baseline=71,72/70,83s. Cả2lượt đều tăng
tốc rõ. Batch32thêm≈0,93%so với2thread/batch16, batch64thêm≈0,61%; chênh nhỏ hơn
dao động giữa các lượt và không cho thấy lợi ích thực dụng để đổi cache recipe.

Trong baseline đầu, aggregate pose/decode≈70,18s, trong khi main cache work tổng≈7,03s
và visual/ROI≈2,39s; stages overlap nên **không cộng** các aggregate thành walltime.
Số đo này giải thích vì sao CPUconcurrency có tác dụng còn batchGPUhầu như không.

## Kiểm tra đầu ra và quyết định

Tất cả candidate có nonvisual arrays/validity/PTS/sourceIDs/ROItransforms/supports
giống baseline trên8clips ×2lượt. Với2threads/batch16, visualfp16arrays cũng **bitwise
identical**, maxerror0. Mỗi thread có detector instance riêng, không share một IMAGE
detector qua các threads. Không bật VIDEOtracking/smoothing hay giảm framecount.

Batch32/64gây rounding differences của CUDA forward: khoảng0,94%visualcoefficients
đổi và maxabsdifference0,00390625trênfp16cache. Đây là numerical batch effect, không
là score/quality degradation đã đo. Code version batch execution thành cache identity
khác nếu dùng32/64; không trộn những features này với existingbatch16cache.

**Chọn2threads/batch16** vì phần lớn speedup đến từCPU, giữ exactoutputs/cache identity
và reuse94productioncaches đã validate. Không đổi batch chỉ để lấy dưới1%gain chưa ổn
định trong benchmark nhỏ. [Choice và migration](evidence/aslc3000_cache_optimization_choice.json).

## Code review và bàn giao runtime

- `_preprocess_one` nhận optional explicitdetector, giữ legacyglobalfallback cho CLIcũ.
  Fusedpipeline dùng thread-localdetector riêng; detector instances được close trong
  finally sau workerpoolshutdown. Không có multiprocessing/resource_tracker.
- CPUworkers được giới hạn1hoặc2**threads**, không thêm Pythonprocess. Queue tối đa
  2×workers; model/data/source indexing vẫn ở mainthread.
- DINO batch16default giữ originalbackboneprovenance. Batch32/64được ghi trong actual
  extractoridentity, loader từ chối cache root có khác batch recipe.
- `run_full_aslc.py` lưu executionoptions và từ chối resume đổioptions trong cùng job.
- **133unit/single-process tests pass**, gồm independent detectors/cleanup/dino-batch
  provenance/recovery. Không chạy multi-processDDP tests dưới userlimit2.
- Jobcũ mới đến cachetrain, chưa có SSLmodelcheckpoint. Runtime source thay đổi nên
  archive jobprovenance cũ vào `artifacts/aslc3000/job_history/before_cpu_thread_optimization_*`,
  tạo jobstatus/source snapshot mới; **không bypass sourcehashresume guard**.
- Productioncache recipe(batch16/pinnedbackbone/IMAGEpose/normalization) giữ nguyên.
 94caches đã tạo được reuse với checksums; geometrytargets vẫn fit từ toàn3.000train.

Service unit dùng `--workers2 --dino-batch-size16`; job/status path vẫn
`artifacts/aslc3000/job_seed17`. [Systemd và resume](ASLC3000_SERVICE_VI.md).

## Phạm vi của tốc độ và ETA

1,93×là **cache-pipeline throughput** trên8clips, không là whole-experiment speedup
hay claim cho toàncorpus. 4.000optimizerupdates/validation không được benchmark như
một thành phần của speedup này. Từ mean4,61s/clip,4.500clips≈5,76hcache trước reuse,
rồi còn targets/gates/training/retrieval. Ước lượng toànpipeline mới khoảng**10–16giờ**,
vẫn là ngoại suy cần actualstageprofile khi caches hoàn tất. Tắtserver/gateerrors làm
calendar time dài hơn.

Job đã khởi động lại với cấu hình chọn; agent chỉ kiểm startup rồi dừng polling.
Không gọi secondlauncher hoặc sửa source/config trong khi service đang chạy.
