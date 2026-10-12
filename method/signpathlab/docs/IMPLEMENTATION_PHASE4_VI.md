# Implementation và code review — bàn giao vòng 4

**Pipeline cốt lõi đã sẵn sàng để review**, trong folder mới `method/signpathlab`.
Đã chạy kiểm chứng cuối bằng môi trường cài sạch: **128 tests pass, 0 skipped**;
**15 trials E00–E04 × 3 seeds completed**; actual raw-video embed và dynamic export
pass. [Acceptance audit](ACCEPTANCE_AUDIT_VI.md) đối chiếu deliverables/checklist/tests/
commands của proposal với code và evidence thực.

Các nhánh M10/alternate backbones/paired stages còn optional unavailable; real
continuous/phonology experiments chưa có configured data/annotations. Không tuyên bố
mọi hướng nghiên cứu đều được implement hoặc đã có benchmark đầy đủ.

## Những gì được hoàn thiện trong vòng này

| Nhóm | Hành vi thực tế |
| --- | --- |
| Embed | Raw RGB decode PTS→IMAGE landmarks→DINO/ROI/static pose→registered full-clip tiling→dense/clip/logits; checks input identity, source/model hashes; atomic no-clobber NPZ |
| Export | Online projectors/encoder + selected linear/attentive head; frozen-visual variant; dynamic B1..16/T1..512; serialize/load/parity; topology/preprocessing/weights metadata; no predictor/teacher |
| Masking | Anatomical pose, stream-temporal và modality-dropout; explicit cross-modal primitives; giữ RGB/ROI public; cùng closure resolver; tests trainability/alias/counterfactual |
| Evaluation | Versioned prediction JSONL, top5/source/signer/model hashes; availability quality buckets; PI/PC grouped bootstrap; paired E04−E03 grouped signer intervals; source-excluded retrieval |
| Continuous interfaces | CTC missing-time compaction/receptive timestamps, impossible alignment checks; real seq2seq gradients/greedy decode; SacreBLEU signature/ROUGE-L/WER; actual toy CLI checks |
| Phonology | Train-only low-capacity probe với verified annotation provenance; unknown=-100 masked; missing labels report `not_run`, không suy từ English gloss |
| Diagnostics | Fixed one-prefix/source online/teacher/per-stream spectra/std/cosine/rank; projector gradient norms; feature_spectrum NPZ; diagnostic RNG/mode restoration |
| Provenance | Per-run source archive/hash, resolved YAML, source registry, lock, load report, manifests/audit/cache/targets, mask/exposure logs, head protocol, decision memo; immutable stage cost records |
| Raw coordinates | Recipe-hashed sidecars lưu pixel xy, frame W/H, shoulder center/scale/validity và timestamps/source IDs; verified reuse cho train32 + val16 |
| Environment | `.venv-locked` không system-site-packages; full main/extraction/continuous lock; editable install thông thường; `pip check` sạch |

## Self-review: vấn đề tìm thấy và sửa

1. Native SDPA/MHA export trên Torch2.11 sinh dynamic shape guard không hợp lệ.
   Deployment dùng explicit eval QKV/attention/out projection và Transformer block
   algebra với weights đã train. Packed eager-vs-dense và serialized dynamic parity
   pass; không ép sequence length cố định để né lỗi.
2. Time origin trước đây lấy slot đầu nên left padding có thể đổi encoding. Graph nay
   lấy first non-padding source observation, giữ cùng segment origin khi missing/masked.
   Test left-padding invariance pass.
3. Resampled aliases đã dedup losses/fitting/pooling nhưng còn lặp attention keys.
   Nay online và teacher encode một key/source observation rồi scatter về đủ dense
   slots. Uneven repeated-frame test xác minh embeddings không đổi. Pilot nguồn30fps→25fps
   không có upsample aliases; vẫn rerun toàn bộ matrix cuối trên frozen current source.
4. Anatomical pose mask không được tuyên bố strict spatial khi RGB/global CLS còn nhìn
   thấy tay. Primitive domain và counterfactual giữ rõ pose observation vs RGB/ROI priors.
   Closure vẫn dùng support intersection, không có bypass để giữ thêm visible tokens.
5. Initial DDP real gate đụng directory của world_size1. Run identity nay gồm world_size;
   completed two-process gate có folder riêng. Failed log được giữ, không overwrite run1.
6. Historical venv dùng system packages có unrelated OpenFace/OpenCV conflicts.
   Tạo venv sạch và lock toàn bộ closure; dùng một `opencv-contrib-python` theo MediaPipe
   thay vì competing cv2 wheels. Rerun15trials, embed/export bằng venv sạch.
7. Supervised head stage nay có source/environment/manifest/load provenance và spectrum,
   không chỉ head.pt. Pipeline audit theo encoder/head pointers, kiểm actual hashes.
8. Metadata review bổ sung prediction schema, class-aware paired CI và reference/source/
   signer alignment checks. Không gộp seed SD vào sampling CI; missing costs giữ null.
9. `.gitignore` cũ có `data/` không anchored, có thể bỏ qua `src/signpath/data` khi add Git.
   Nay chỉ ignore `/data/` runtime corpus; package data modules đã hiện trong git status.
10. CTC nay compact internal missing observations cùng timestamp supports; SLT rejects
    internal BOS/EOS delimiters. Cache pinning/persistent thread setting được thực hiện,
    không giữ config option vô hiệu.

Đây là review của coding agent, không phải independent review. Các báo cáo vòng1–3
giữ nguyên để lưu audit trail; mô tả “pending” trong đó là trạng thái lịch sử.

## Final pilot: protocol và kết quả

Pool vẫn là ASL Citizen **8 lớp, 32 train/16 val**, selection seed137, official
signer-disjoint membership, actual source SHA. Pool được chọn bằng nhãn/availability
trước model scores, nên được ghi **label-curated development subset**. Không truy cập
test CSV/videos trong sweep. SSL sampler/loss không nhận labels/gloss/text/signer.

Config cuối: `configs/verified_pilot/E*_seed*.yaml`, kế thừa pilot đã khóa và chỉ đổi
output root. D96, encoder2layers/predictor1layer, 100 successful SSL updates, batch4,
accumulation1, seeds17/29/43. Same LR trials [.001,.003,.01], max40headepochs/patience10,
same inference48/overlap16. Mọi trial complete; không có failed trial bị giấu trong root này.

| Method | Val PC Top1 mean ± seed SD (%) | Retrieval R@1 mean (%) |
| --- | ---: | ---: |
| E00 frozen visual | 25.00 ± 0.00 | 31.25 |
| E01 cluster CE | 20.83 ± 9.55 | 12.50 |
| E02 raw JEPA | 27.08 ± 3.61 | 16.67 |
| E03 semantic + ordinary motion | 27.08 ± 3.61 | 16.67 |
| E04 semantic + logsig2 | 27.08 ± 3.61 | 16.67 |

E03/E04 cùng PC score ở từng seed:31.25/25.00/25.00%. Điều này không có nghĩa predictions
giống nhau ở mọi seed. Paired signer bootstrap dùng5official signers/16samples và1000draws:
PC difference95%interval là **[-10.71,+8.33]**, **[-27.78,+20.00]**, **[0,0] percentage
points** theo seed17/29/43. Đây là conditional sampling intervals trên fitted models/pool,
không là CI cho toàn ngôn ngữ hay thay thế seed dispersion.

Matched E03/E04:100updates, cùng mask digests, distinct-source-frame exposure
18727/18668/18705, eligible geometry counts13477/12239/12500. Online weights khác nhau
L2≈.15097/.15266/.15137; geometry branch thực sự được train. Descriptor native rank khác
nhau dù output đều32D, được ghi rõ.

Evidence: [CSV](evidence/phase4_comparison.csv), [full results](evidence/phase4_comparison.json),
[artifact audit](evidence/phase4_handoff_audit.json). Raw logs/12SSL checkpoints/15heads
ở `artifacts/verified_pilot`. Historical pilot ở `artifacts/pilot8` không bị overwrite.

**Decision:** chưa có evidence cho signature advantage. Giữ E00 làm retrieval reference,
E02 làm SSL reference đơn giản; giữ E03/E04 như controlled runnable ablations. Không
tăng degree/lift hoặc tuyên bố superiority dựa vào pilot này. Xem thêm
[decision memo](DECISION_FINAL_VI.md).

## Verification và giới hạn thực tế

- 128 tests pass, 0 skip/failure/error; gồm actual CUDA fp16 overflow, two Gloo ranks
  với empty local targets, single-process exact5→10resume, geometry/metamorphic tests,
  raw static retention, mask closure/counterfactual, metrics, cache và dynamic export.
- Two-process real cached corpus strict gate đã chạy3updates; anatomical real gate
  đã chạy3updates. Đây là system correctness gates, không thêm vào research scores.
- Actual APPLE raw-video inference dùng final E02seed17 head: dense[59,96],59valid,
  visual valid counts[59,0,15,59], original PTS; total decode-to-head wall≈6.12s trên máy
  hiện tại. APPLE nằm ngoài8pilot classes, prediction chỉ kiểm API, không là accuracy.
- Final export9serialized parity cases và actual60-sample full-clip tiled eager-vs-export
  pass với atol2e−5/rtol1e−4. Torch phát một upstream non-writable-buffer warning lúc load;
  không suppress warning hoặc có failed parity.
- Instrumented cached profile lưu online/teacher/predictor forwards, residual backward/
  update, loading fraction, token counts, cache size và peak memory. Pilot tổng thể vẫn
  bị giới hạn bởi cache/support preparation; timing có hook/sync overhead và concurrency,
  không là production throughput. Original preprocessing GPU-hours thiếu measurement
  được giữ null, không coi0. Evidence: `phase4_stage_profile.json`, stage ledgers.
- Real continuous/phonology và full ASL Citizen/final test/NCCL chưa chạy. Các status
  này được giữ `not_run`/optional unavailable, không dùng synthetic metric thay thế.

## Lệnh kiểm tra bản bàn giao hiện có

Chạy từ `method/signpathlab`; các artifact tạo mới không overwrite output cũ.

```bash
.venv-locked/bin/python -m pip check
.venv-locked/bin/python -m pytest -q
SIGNPATH_RUN=artifacts/verified_pilot/heads/E02/seed17/9a184850c755
.venv-locked/bin/signpath evaluate --run "$SIGNPATH_RUN" --split val --export-predictions --device cuda:0
.venv-locked/bin/signpath experiments summarize --root artifacts/verified_pilot \
  --output artifacts/verified_pilot/comparison_review.csv
.venv-locked/bin/python scripts/audit_handoff.py --run "$SIGNPATH_RUN" \
  --export-dir artifacts/export_phase4_final_E02 \
  --embed artifacts/phase4_final_demo_embeddings.npz --output artifacts/handoff_review.json
```

Tái chạy raw input/export vào tên mới:

```bash
.venv-locked/bin/signpath embed --run "$SIGNPATH_RUN" \
  --input /home/dongvk/datasets/ASL_Citizen/videos/15890366051589533-APPLE.mp4 \
  --assets-file registries/local_extraction.yaml --device cuda:0 \
  --output artifacts/review_embeddings.npz
.venv-locked/bin/signpath export --run "$SIGNPATH_RUN" --format torch_export \
  --output artifacts/review_export
```

Code mới không thay đổi method khác và chưa được commit/push. README có lệnh cài,
preprocess/cache/fit/train/DDP/downstream và điều kiện để chạy corpus mới. Dataset,
downloaded weights, venv và runtime artifacts được gitignore, cần giữ riêng khi chuyển máy.
