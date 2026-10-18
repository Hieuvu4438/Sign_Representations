# Báo cáo triển khai proposal14

Ngày bàn giao: 2026-10-14. Code tại `method/pose_jepa`, package `signrepr`.
Những thay đổi sẵn có ở các method khác được giữ nguyên.

## Kết quả implementation

Đã triển khai pipeline Stage 0 → SSL → downstream/export theo guide:

- Schema/cache 133-joint, converter Uni-Sign, tracker nhiều người, streaming decode
  dùng PTS, raw-score audit, split/dedup/protocol và fixture độc lập train/val/test.
- Canonicalization toàn sample trước windowing; bốn stream độc lập; GCN LayerNorm,
  shared-hand weights, ordered aggregation và token 2 frame.
- Student packed visible context, teacher EMA full natural input, predictor full
  lattice; latent CE/center và local geometry/global motion có support đúng.
- AdamW/no-decay groups, AMP, weighted accumulation/DDP, global center, skip overflow
  đồng bộ, atomic checkpoint, RNG/cursor và worker-prefetch resume.
- SupCon P×K theo namespace, global autograd gather/local anchors, classifier phụ,
  head warmup, joint SSL replay và controls không replay.
- ISLR frozen linear probe/full fine-tune; exact gloss/video retrieval; full sequence
  stitch và cache time mapping; frozen/fine-tuned CSLR giữ cả utterance, cache tắt
  khi fine-tune; bridge mT5 frozen pose, pad=-100, không truncate text/sign âm thầm.
- Export raw/metric bundle đúng encoder/head, metadata/checksum, CPU inference và
  lỗi rõ khi layout không hợp lệ hoặc yêu cầu metric head chưa train.

Configs có recipe chính, các task, baseline cùng encoder và ablation ưu tiên.
Mọi lệnh được yêu cầu trong mục 20 đều có parser và executable workflow. Không có
`NotImplementedError`, TODO hoặc placeholder implementation trong production modules.

## Kiểm chứng đã thực hiện

Môi trường riêng: Python 3.10.19, Torch 2.5.1+cu121, RTX 5880 Ada; lockfile thật
`requirements-tested.lock.txt`. Không thay requirements của các môi trường legacy.

Test suite bao phủ mapping/score-depth, reference, hidden support và negative
control, padding/missingness, physical units/root validity, teacher grad/EMA alias,
weighted center/accumulation, DDP zero-support rank, SupCon gather, AMP overflow,
resume, prefetched worker cursor, retrieval failures/ties, CTC repeats/lengths,
full differentiable stitching, mT5 backward/generation/workflow và CLI pipeline.
Kết quả cuối: **53 tests pass**, warnings được coi là lỗi; JUnit, stdout và JSON
được lưu trong `docs/evidence`. Bản wheel cũng được build/cài độc lập và kiểm tra
modules/source lock.

Đã tải/ghi checksum 13/13 file tham khảo đúng revision. mT5-base config/revision
được pin, weights không được tải/nạp. Không dùng checkpoint RGB/Uni-Sign để
initialize encoder mới. Xem `sources.lock.json` và `codebase_review.md`.

## Experiments thật và cách đọc kết quả

Đã audit 32 clips thật (16 train/8 validation/8 test), xem overlay 20 video và
đối chiếu cache với PTS nguồn. Score legacy vượt 1 được chuyển qua profile khai báo
rõ và giữ raw scores; không gọi đó là calibrated probability/extractor đúng paper.

Đã hoàn thành 5 pilot ×5.000 successful updates trên cùng full backbone/data:
geometry, motion, JEPA-only, hybrid và shuffled-target. `pilot_summary.json` lưu
loss/coverage/entropy/rank, checkpoints, exact hashes, VRAM và elapsed GPU-hours.
Đây là comparison theo số updates, không phải compute-matched/wall-clock cô lập.

| Method | Best validation R@1 | Frozen test retrieval R@1 | Frozen linear probe test Top1 |
|---|---:|---:|---:|
| Geometry baseline | 75,0% | 12,5% | 25,0% |
| JEPA hybrid | 62,5% | 25,0% | 37,5% |

Mỗi cột downstream chỉ có 8 clips/4 classes. Chọn geometry baseline bằng validation
và giữ cả hybrid; không đổi lựa chọn sau khi nhìn test. Những số này không chứng minh
method nào tốt hơn ở benchmark đầy đủ. Shuffled-target có R@1 25% và biểu hiện gần
collapse; correspondence control hoạt động đúng theo sanity check.

Đã chạy overfit sanity 16 train clips thật (100% train Top1), frozen probes,
3 metric controls cùng budget 20 updates và CSLR adapter sanity trên isolated glosses.
Stage 2 validation R@1 đều 37,5%; đây không phải kết quả full metric recipe.
mT5/CSLR full-task metrics chưa chạy vì thiếu corpus translation/continuous transcript
được audit cho recipe này. Minimal-pair/non-manual annotations và calibrated detector
profile chưa có; không đưa ra kết luận bỏ auxiliary loss hay claim SOTA.

## Artifacts và chạy lại

Bundle đã kiểm chứng: `runtime/selected_pose_encoder`, `runtime/hybrid_pose_encoder`,
`runtime/metric_pose_encoder`. Encoder có 10.255.872 parameters, sequence/raw dim 1024,
stride 2. CPU load parity pass; example trên clip 53 frame trả `[27,1024]` và `[1024]`.
Latency khoảng 29 ms/cache clip, không bao gồm detector.

```bash
cd method/pose_jepa
runtime/venv/bin/python -m pytest tests -q -W error
runtime/venv/bin/python -m signrepr.cli --help
OMP_NUM_THREADS=2 runtime/venv/bin/python examples/inference_cpu.py \
  --bundle runtime/hybrid_pose_encoder \
  --pose 'runtime/aslc_sanity/poses/train/16135350953831118-APRIL.npz'
```

Xem README cho setup trên máy khác, fixture, real-data converter và tất cả lệnh task.
Nguồn dữ liệu/weights/runs là configurable paths; corpus, môi trường và weights
không được đưa vào Git. Source snapshots cho từng run được giữ ở
`runtime/source_snapshots/<code_sha256>/signrepr`; dùng parent directory đó trong
`PYTHONPATH` khi cần resume đúng source hash. Các thay đổi validation/diagnostic
trong lúc pilot chạy đã được đối chiếu bitwise update-state với source snapshot cũ.

Implementation baseline và các adapters đã hoàn tất. Full-scale benchmark training,
full-budget Stage 2 và performance claims cần một experiment protocol/data budget
riêng; chúng được phân biệt với code runnable/tested trong `IMPLEMENTATION_STATUS.md`.
