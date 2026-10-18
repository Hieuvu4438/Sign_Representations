# IMPLEMENTATION_STATUS — ASM-JEPA (proposal15)

Cập nhật: 2026-10-15. Trạng thái: **done** (code có), **tested** (có test/smoke chạy thật, output ghi dưới),
**not run** (chưa chạy), **blocked** (thiếu tài nguyên). Chưa có commit git cho package này; mọi lệnh chạy từ
`method/asm_jepa` với `PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python`.

## 1. Tổng quan

| Hạng mục (spec) | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| Manifest chính thức, vocab train-only, split audit (§1.1, §5) | done, tested | `prepare_aslc`: 40.154 / 10.304 / 32.941 video, 35 / 6 / 11 signer, 2.731 gloss = v1.0; video & signer disjoint; 0 video lỗi; 45 s |
| Pose cache contract `.npz` (§6.3) | done, tested | `import_pose_cache`: train 40.154 ok, val 10.304 ok (5,1 GB). Test **chưa import** (theo spec: chỉ sau khi khóa pipeline) |
| Trích pose RTMW (§6.1–6.2) | done, tested (smoke) | `extract_pose` 3 clip, provider CUDA cả det/pose; khớp cache có sẵn (lệch trung vị 0,02 px, tương quan score 1,0) ⇒ cache pickle có sẵn chính là RTMW `performance` đã pin |
| Audit pose (§6.4) | done, tested | 100 clip (overlay) + 300 clip; xem mục 3 |
| Schema 133→114, sampling 64/4, normalization, 7 kênh (§7) | done, tested | tests `test_sampling.py`, `test_mask_dependencies.py` |
| fit_stats train-only (§7.4) | done, tested | 40.132 clip dùng, 22 bỏ (unusable); `data/processed/aslc/stats/train_v1.npz` |
| State / relation tokens (ASM-v1, §8.1–8.2) | done, tested | |
| Motion tokens log-signature bậc 2, geometry + timing (ASM-v2, §8.3–8.4) | done, tested | `test_logsig2.py`; smoke thật với `full.yaml` |
| Encoder / predictor / teacher EMA (§8.5) | done, tested | |
| Mask mixture + dependency closure + counterfactual (§9) | done, tested | 12 case donor (LH, RH, face, elbows, boundary, multi-scale; có/không đổi score) + test đối chứng tắt closure ⇒ phát hiện leak |
| Losses: centred CE theo group/family, state anchor, global (§10) | done, tested | vector hóa, không sync host |
| Training loop, accumulation, overflow, EMA/centre 1 lần/step, checkpoint/resume (§12) | done, tested | resume khớp chạy liên tục (loss rtol 1e-4, teacher atol 1e-5) |
| Frozen embeddings + gloss-max retrieval R@1/5/10, MRR, DCG (§13.1–13.2) | done, tested | chunked = dense; tie class-id; thêm `signrep_style` (tie pessimistic như SignRep / các method khác trong repo); per-signer, quality-bin, bootstrap CI |
| Linear probe (§13.3) + few-shot k=1/2/4 (§13.5) | done, tested | hash embedding trước/sau không đổi |
| Fine-tune ISLR / supervised scratch (§13.4) | done, tested (smoke) | overfit-16: train top-1 = 100 % sau 500 step |
| evaluate_classifier, report → results.csv / final_report.md (§18) | done, tested (smoke) | |
| R1 coordinate MAE control (§14.3) | done, tested (smoke) | |
| DDP (§14.2) | done, **not tested** | helper + DDP wrapper + rank-aware sampler; chỉ có 1 GPU |
| Pilot SSL 60k step (R4), R0 scratch, R2/R3/R5/R6 | **not run** | chờ người dùng quyết định chạy (GPU dùng chung) |
| Đánh giá test | **not run** | đúng spec: chỉ sau khi khóa config bằng val |

## 2. Lệnh đã chạy và kết quả thực

```text
pytest (CPU)                       54 passed (CUDA_VISIBLE_DEVICES="" $PY -m pytest -q tests)
pretrain --synthetic state 20 step OK (loss 13.6 -> 10.1)
pretrain --synthetic full 20+10    OK, validation + resume từ last.pt
pretrain real state, 128 clip, 100 step   loss 10.7 -> 4.49, ~360 samples/s (GPU đang bị job khác dùng 99 %)
extract_embeddings train / val     40.154 clip 15 s, 10.304 clip 5 s; 22 train clip không có pose dùng được
retrieval (smoke ckpt, val)        R@1 0.40 %, R@5 1.52 %, DCG 12.23 (chance R@1 ≈ 0.04 %) — chỉ kiểm tra pipeline
train_classifier linear (3 epoch)  OK; evaluate_classifier OK
train_classifier finetune (512 clip, 2 epoch, init SSL smoke) OK
overfit16 scratch 500 step         train top-1 1.0 (passed)
pretrain mae_r1 real 30 step       OK
profile (300 step, 100 warm-up loại) state.yaml 1.158 samples/s, full.yaml 815 samples/s, VRAM đỉnh 0,6 / 0,8 GB
                                   (GPU vẫn bị job khác dùng ~99 %; trước khi vector hóa loss full.yaml chỉ 130 samples/s)
                                   ⇒ 60k step × 128 ≈ 1,9 h (v1) / 2,6 h (v2) nếu giữ tốc độ này
smoke E2E cuối (full.yaml, 1.024 clip, 60 step, validation đầy đủ train-gallery/val) OK;
                                   val R@1 0,46 % (smoke), 22 clip train unusable bị loại khỏi gallery
extract_pose (.venv-unisign) 3 clip CUDAExecutionProvider, ~10 s/clip lúc khởi động
```

Đo builder CPU: v1 3,2 ms/mẫu, v2 4,9 ms/mẫu (gồm đọc npz) ⇒ 8 worker đủ ~1.600 mẫu/s.

## 2b. Code review độc lập (agent) và các fix

Không phát hiện target leakage. Đã sửa:
1. Resume có thể ghi đè `best_val.pt` bằng checkpoint kém hơn (checkpoint lưu trước validation cùng step) → validation
   chạy trước khi lưu checkpoint định kỳ.
2. Global term crash khi multi-GPU (all_gather với shape khác nhau) → gather toàn bộ rồi lọc `usable`; timeout NCCL 2 h.
3. Số interval widths ≠ 3 làm lẫn bucket geometry/timing → `n_subgroups` lấy từ layout (test 4 widths).
4. Clip gallery không có pose (embedding 0) từng tạo điểm giả 0 cho gloss → bị loại khỏi max, báo số bị loại.
5. Guard: SSL không cho `test` làm val/gallery; retrieval CLI chỉ nhận query val/test, gallery train.
6. Fine-tune từ checkpoint không có teacher (MAE) báo lỗi rõ; saturation rate (§7.3) được log.

## 3. Phát hiện từ audit pose (cần người dùng quyết định)

1. **Ngưỡng confidence 0,20 không có tác dụng** với score RTMW SimCC (không chuẩn hóa, ~0,8–9,5): coverage = 100 %
   mọi part kể cả tay ngoài khung (overlay xác nhận). Code hỗ trợ ngưỡng theo part
   (`data.confidence_threshold: {body: .., lh: .., rh: .., face: ..}`) và kênh confidence = clip(score/10, 0, 1).
   Giá trị mặc định **chưa đổi** vì spec yêu cầu chọn sau audit; các project trước trong repo dùng ngưỡng joint
   ≈ 2,9 (body) / 0,85 (hand) / 4,0 (face). Đổi ngưỡng ⇒ phải fit lại stats.
2. **Hand alias**: 14,3 % frame có LH và RH gần như trùng nhau (khoảng cách trung bình < 0,15 vai) — cả hai tay
   bị đặt lên cùng một bàn tay khi một tay ra khỏi khung. Chưa lọc trong cache (spec: không sửa raw cache).
3. LH khớp body-left-wrist COCO ở 90–91 % frame (phần còn lại chủ yếu khi hai tay gần nhau / alias).
4. 4/100 clip có > 20 % frame không hợp lệ; không có NaN trong observed; không có timestamp gap.

## 4. Sai khác có chủ ý so với hướng dẫn

- Môi trường: dùng lại `.venv-signrep` (Python 3.13, torch 2.11 cu128) và `.venv-unisign` (rtmlib 0.0.13 bản
  vendored của Uni-Sign, không phải checkout commit 03a1693) thay vì tạo venv mới; xem `requirements-*.lock.txt`.
- Pose: dùng `import_pose_cache` cho cache pickle có sẵn (đã chứng minh trùng RTMW performance); timestamps của
  cache import là `index / container fps` (`cfr_index_over_container_fps`), extractor mới dùng PTS khi đơn điệu.
- Thêm key config (đều được tiêu thụ): `data.confidence_scale`, `data.persistent_workers`,
  `data.coordinate_saturation`, `mask.temporal_block_{min,max}_patches`, `mask.ratio_fallback_factor`,
  `mask.max_ratio_fallbacks`, `loss.global_min_statistical_batch`, `train.objective`, `train.keep_step_checkpoints`
  (giữ 3 step checkpoint + last + best_val vì đĩa gần đầy), `train.profile_warmup_steps`,
  `eval.extract_batch_size`, `eval.diagnostic_batch_size`.
- Predictor cộng thêm metadata embedding cho context (vị trí/loại token), như V-JEPA.
- Global head nhận vector `state_part_concat1024_l2` (đã L2) của intact pass.
- R1 MAE không có teacher; extraction dùng online encoder (`mae_r1.yaml` đặt `eval.encoder: online`).
- `--profile-steps` lưu `last.pt` để có thể resume; báo cáo ở `reports/profile.json`.

## 6. Config final và job chạy ngầm (2026-10-16)

**Method final** = `configs/pretrain/final.yaml` (= R6 ASM-v2 full + bộ lọc quan sát + thiết lập chạy). Thay đổi so với
`full.yaml`, có bằng chứng từ audit 300–500 clip train:

| Thay đổi | Lý do / bằng chứng |
|---|---|
| `data.in_image_only: true` | ~50 % frame có body wrist ở mép dưới ảnh; tay nghỉ ở đó có trung bình 42 % joint nằm ngoài ảnh nhưng score vẫn ~1–5 ⇒ ngưỡng score không loại được |
| `data.hand_wrist_max_dist: 0.30` (vai) | frame alias: tay đặt sai có cổ tay cách body wrist COCO cùng bên trung vị 0,96 vai; tay đúng ≤ 0,08 vai ở 75 % frame. Quyết định chỉ đọc chính bàn tay + body wrist/vai (không phải model joint) ⇒ không rò rỉ chéo part (test `test_observation_filter.py`) |
| stats riêng `stats/train_v2_obsfilter.npz` | chuẩn hóa phải fit lại trên train với cùng bộ lọc |
| batch vật lý 128 × accum 1 | cùng batch hiệu dụng 128; VRAM 0,8 GB ở batch 32; global term thấy 128 clip thay vì 32 |
| validation mỗi 2.500 step, milestone mỗi 10k step | chọn best_val mịn hơn (~35 s/lần) và đánh giá theo giai đoạn |

Hiệu ứng bộ lọc (500 clip): coverage LH 97,5 → 74,9 %, RH 97,5 → 83,2 %, body 96,1 %, face không đổi; token hợp lệ
170 → 133/clip; context 51 → 46 token. Ngưỡng score giữ 0,20 vì score không tách được tay tốt/xấu (ngưỡng 1,0 đã bỏ
10 % tay trong khung, chủ yếu tay mờ do chuyển động).

Vì bộ lọc chưa được chứng minh tốt hơn, runner chạy **A/B pilot** 5k step (`final.yaml` vs `final_nofilter.yaml`) và
giữ variant có val R@1 cao hơn (chỉ dùng val). Fine-tune/scratch dùng `configs/eval/finetune_final[_nofilter].yaml`.

Runner `scripts/final_job.py` + `scripts/final_job.sh` (README §Chạy job final): import test → fit stats → A/B pilot →
pretrain 60k → đánh giá theo giai đoạn (milestone + best_val, val và test) → linear probe + few-shot → fine-tune →
scratch R0 → report/summary. Resume từ `last.pt`, retry tối đa 3 lỗi/stage, watchdog cron 10 phút, chờ slot GPU
(≤ 1 job GPU khác của user).

## 5. Việc tiếp theo đề xuất (chưa làm)

1. Quyết định ngưỡng confidence theo part (mục 3.1), fit lại stats nếu đổi.
2. Profile 1000 step cho `state.yaml`/`full.yaml` khi GPU rảnh, chọn batch vật lý (48 GB: 64 × accum 2).
3. Chạy R0 (scratch) và R4 (`state.yaml`) seed 0, chọn checkpoint bằng val R@1; sau đó R2/R3/R5/R6.
4. Khóa config → import pose test → frozen retrieval test + classifier test một lần → `report`.
