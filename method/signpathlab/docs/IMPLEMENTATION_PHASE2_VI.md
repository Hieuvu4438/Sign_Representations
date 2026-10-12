# Báo cáo implementation và review — vòng 2, 12/10/2026

**Bản lịch sử. Trạng thái mới nhất: [vòng 3](IMPLEMENTATION_PHASE3_VI.md).**

Code nằm trong `method/signpathlab`; không sửa các method khác. Vòng này đã có
**đường MVP kỹ thuật trên clip thật** tới optimizer/checkpoint/embeddings/head prediction.
Toàn bộ proposal vẫn **chưa hoàn thành**: chưa có held-out utility comparison hay research pilot.

## Code mới và bằng chứng

| Phần | Hành vi đã implement | Evidence |
| --- | --- | --- |
| Data batching | Source-uniform sampling; contiguous crop; short-clip invalid padding; seven typed streams; timestamps; source IDs; independent context/target masks | Padding, alias dedup và sampler/RNG replay tests |
| M6 encoder | Per-stream MLP→time/type embeddings→packed Transformer; hidden values không đi vào projector/attention; full/dense/clip outputs | Hidden-value edits không đổi student; padding không đổi real outputs |
| M6 predictor/teacher | Metadata-only decoder queries; semantic/32D geometry heads; exact-copy no-grad/eval EMA projectors+encoder, exclude predictor | Teacher copy/gradient tests; query/prediction counterfactual |
| M7 objectives | Normalized L1/cosine; group/scale SmoothL1; per-stream cluster CE head; global active-type means; differentiable empty losses | Hand-calculated gradients/counts; geometry/head nonzero gradients; teacher skipped for cluster-only forward |
| Training runtime | AdamW; bias/norm no decay; successful-step LR/EMA; accumulation; fp32/bf16/fp16; global finite/empty consensus | Fixed-batch learning; empty/nonfinite skip; actual CUDA fp16 overflow and recovery |
| DDP | Two-process execution; W-corrected global normalization; empty-local-rank graph; no_sync accumulation; collective checkpoint RNG | Two Gloo ranks agree with one-process contributing-sample reference; two-rank CLI smoke |
| Resume | Atomic checkpoint; exact model/optimizer/scheduler/scaler; Python/NumPy/torch/CUDA/mask RNG; sampler state; config/data/target/implementation identity | 5→10 CPU exact resume, completed-run CLI resume, incompatible identity rejection |
| M8 interfaces | Linear/attentive heads; CTC loss/greedy; causal seq2seq loss/greedy; classification metrics/edit-count WER | Toy CTC/SLT nonzero gradients; repeat-label alignment rejection; causal/memory masks; hand-computed metrics |
| True preprocessing leakage gate | Edit hidden raw RGB and rerun MediaPipe IMAGE + strict DINO extraction | 137 retained raw tokens and 22 retained geometry tokens identical, max error0 |

Final automated suite: **85 tests passed, 0 skipped**. Evidence: `evidence/phase2_tests.json`.
CUDA availability was real on this machine, so fp16 overflow test executed rather than skipped.
DDP tests used CPU/Gloo; no NCCL or multi-GPU AMP claim is made.

## Runs đã thực hiện

### CPU synthetic correctness

`signpath smoke --config configs/smoke.yaml --device cpu` chạy 10 successful updates,
ghi resolved config, environment, JSONL attempts/mask/loss counts, checkpoint, dense/clip
embeddings và toy classifier/predictions. Fixture có variable lengths/missing modality;
mọi artifact gắn `synthetic_correctness_only`. Không có download hoặc random pretrained fallback.

Evidence: `evidence/phase2_cpu_smoke.json`. Run identity gồm config, actual target
transforms và implementation hashes. Chạy lặp không overwrite; resume explicit từ checkpoint.

### Hai processes với accumulation

`configs/smoke_ddp.yaml` chạy 3 updates × 2 microbatches trên hai CPU processes.
Checkpoint gather rank-specific RNG/source sampler states. Global geometry targets và
loss/count logs được reduce; một test riêng buộc rank0 không có context/targets.
Evidence: `evidence/phase2_ddp_smoke.json` và test names trong `phase2_tests.json`.

### Real cached minibatch MVP

Hai video thuộc official **train CSV**, APPLE và IMPOSSIBLE. Full preprocessing tạo
71/76 raw frames → 59/63 selected samples; không cắt prefix khi xuất full-clip embeddings.
Train-only geometry fitting bao phủ mọi center phase của hai sources. SSL provider
không nhận label/gloss/text/signer; labels chỉ được tiêu thụ sau training bởi frozen head.

Với `configs/diagnostic_real.yaml` (D48, encoder2layers, predictor1layer, fp32), run đạt:

- 20/20 successful updates; tổng 233 fully-hidden geometry targets qua các updates.
- Composite train loss khoảng 1.0959 → 0.8958; đây là diagnostic, không là utility metric.
- Checkpoint nạp lại khớp chính xác online/predictor/teacher và optimizer state.
- Dense output `[2,63,48]` với real lengths `[59,63]`, validity và original timestamps.
- Frozen linear head fit/predict trên đúng hai **train examples**, có checkpoint/logits/class IDs.

Evidence: `evidence/phase2_real_mvp.json`, `phase2_real_identities.json` và generated run
folder ghi trong report. Không báo train-minibatch predictions thành validation/test accuracy.
Không gọi checkpoint 20steps là representation đã được nghiên cứu hoặc model đạt chất lượng.

### Raw pretrained counterfactual

Trên IMPOSSIBLE, sửa 32 masked source frames thành black; chạy lại toàn bộ detector/DINO.
Fixed one-image extraction batches loại batch-kernel roundoff khỏi comparison này.
So sánh 137 retained raw tokens và 22 retained geometry descriptors: max absolute error0
với atol/rtol1e−6. Public mask/query schedule giữ cố định; hidden detector validity được
phép đổi và không dùng để chọn schedule mới. Unit test riêng kiểm tra student/query/prediction
không nhận hidden feature/geometry values.

Evidence: `evidence/phase2_pretrained_counterfactual.json`. Kết quả chỉ áp dụng strict
temporal same-frame recipe này, không suy ra strict spatial masking hoặc VIDEO tracking.

## Review từng giai đoạn và lỗi đã sửa

Review do cùng coding agent thực hiện qua đọc contracts/code, numerical checks và runtime
tests; không có independent reviewer hoặc sub-agent. Không đợi approval cho reversible fixes.

1. **Packing/padding:** Transformer có all-masked softmax NaN risk khi local rank rỗng.
   Thêm safe sentinel key, giữ `valid` riêng; sentinel không scatter/pool/loss. Hidden
   values zero trước projector, sau đó chỉ pack visible IDs, không encode mask-zero như observed.
2. **Index/time/privacy:** Raw flatten time-major `[T,7]`, per-stream projector dimensions
   384/28/84/84 đúng schema; geometry center/requested bounds dùng cùng segment origin.
   IDs/source/signer/labels không được embedding/query; aliases không nhân đôi targets/pooling.
3. **Teacher/EMA:** Teacher no-grad/eval và exact initial copy. Chỉ EMA online projectors+
   encoder sau success, integer buffers copy, float buffers EMA. Predictor không thuộc teacher.
4. **Global reductions:** Counts và logging sums detached all-reduce; local gradient loss có
   factorW và global active-type denominator. Empty rank vẫn backward. Accumulation là mean
   của microbatch global means, không được quảng cáo là exact update-wide token mean.
5. **Finite/scaler:** Clip sau unscale; consensus trước optimizer; failed update không đổi
   model/EMA/scheduler. Global overflow backoff scaler explicit kể cả rank local finite.
   Nonfinite diagnostics serialize null với skip reason, không repair model tensors.
6. **DDP numerical review:** Lần so sánh AdamW đầu tiên lệch ~4e−5 ở mathematically-zero
   attention bias gradients (~1e−10), do batch-shaped floating reductions bị Adam khuếch đại.
   Không nới tolerance. Reference được tính trên cùng contributing sample shape, bỏ empty
   example vốn có zero contribution; gradients và update comparison pass ở tolerance ban đầu.
7. **Real fitting bug:** Full-clip center0,4,… bỏ sót group/scale valid sau crop offset3.
   Runtime từ chối missing projection; không random fallback/zero target. Sửa fitting
   enumerate mọi center phase train-only, thêm crafted regression. Failed run giữ failure memo
   trong folder `.invalid_phase0_fit`, không dùng cho bất kỳ comparison nào.
8. **Resume identity:** Thêm fingerprint actual projection/stat arrays và source implementation,
   ngoài config/data/topology. Sửa publication errors để broadcast lỗi rank0 trước khi peers
   tiếp tục; runtime budget/data-preparation consensus tránh một rank tự rời loop.
9. **Stage truthfulness:** Runtime reject geometry-input/causal/extra losses, paired hoặc
   cross-modal configs chưa implement. Cluster-only forward không chạy EMA target encoder,
   nhưng k-means fitting/complete E01 training runner vẫn chưa có.
10. **Downstream:** CTC check repeat-label minimum lengths, không `zero_infinity` che lỗi.
    SLT teacher-forcing causal và suffix padding; greedy không emit PAD/BOS. Toy heads có
    finite loss/nonzero gradient, không class trả zeros/TODO thay cho functioning interface.

## Completion audit theo deliverables proposal

| Deliverable | Trạng thái hiện tại |
| --- | --- |
| Installable/locked environment | Editable/dev install chạy; tested dependency snapshot có; clean-room lock verification còn mở |
| CLI | doctor/config/data/assets/smoke có; generic preprocess/cache/targets/verify/train/profile/downstream/evaluate/embed/export/summarize còn mở |
| Real adapters | ASL Citizen/local JSONL có; full actual corpus import/availability/signer/source/near-duplicate audit chưa chạy |
| Preprocessing/cache | Real IMAGE/DINO pipeline + hashes/supports/ROI/counterfactual có; shards/resume ledger/corpus scaling còn mở |
| Baselines | Raw JEPA model có; frozen visual probe và train-only cluster fitting/complete cluster baseline còn mở |
| Geometry/masking | Degree2/control/train projections/strict closure có; cross-modal strategy variants và full train-pool fit còn mở |
| Training | Single-device AMP/accumulation/EMA/resume + Gloo DDP có; generic dataset runner, best validation selection/retention, multi-GPU validation còn mở |
| Downstream | Functioning heads/basic metrics có; full ISLR/phonology/retrieval protocols, real CTC/SLT, prediction provenance/bootstrap còn mở |
| Experiments/research | Matched E03/E04 runner/budgets/seeds/summary/pilot/decision memo chưa chạy |
| Export/compute | Full-clip diagnostic embeddings có; profiling/deployment/export parity chưa hoàn thành |
| Handoff | README/METHOD/data/model cards, phase reviews/failure notes và code snapshots có; complete reproducible commands/utility evidence còn mở |

Không đánh dấu goal complete. Bước tiếp theo: nối manifest→cache index→target fitting→generic
train/evaluate CLI; hoàn thiện E00/E01 và downstream/retrieval protocols; sau đó chạy pilot
E02/E03/E04 cùng data/masks/head/budget, chọn bằng validation và giữ test ngoài tuning.
