# Acceptance audit — SignPathLab, 12/10/2026

Đối chiếu trực tiếp specification `docs/proposal13/signpathlab_implementation_agent_vi.md`.
Phạm vi bàn giao là pipeline mặc định, minimum comparison E00–E04 và các deliverable
bắt buộc §1.2. §1.1 cho phép ghi rõ thiếu dataset/checkpoint/runtime; §5.1 và checklist
§20.4 cho phép optional unavailable backends có lỗi cụ thể. Các nhánh đó **chưa được
implement** và không được gộp vào tuyên bố pipeline đã kiểm chứng.

Code review là **self-review**, có kiểm tra chạy thực và tests; không phải independent
review. Báo cáo từng vòng: `IMPLEMENTATION_REPORT_VI.md`, `IMPLEMENTATION_PHASE2_VI.md`,
`IMPLEMENTATION_PHASE3_VI.md`, `IMPLEMENTATION_PHASE4_VI.md`.

## Bằng chứng hiện tại

- `evidence/phase4_tests.json`: 128 pass, 0 skipped/errors/failures; JUnit/log gốc ở
  `artifacts/phase4_tests.xml`, `artifacts/phase4_tests.log`.
- `evidence/phase4_handoff_audit.json`: script `scripts/audit_handoff.py` đọc/kiểm tra
  15 completed trials, source hashes, checkpoint/artifact files, predictions, budgets,
  geometry fitting sources và real full-clip eager/export parity. Không suy completion
  từ việc chỉ có tên file hay chỉ import thành công.
- Final runtime implementation hash:
  `18794cff27053ae216af9d9ebf1685306b43f659e9643010f3dfa725ed8756e6`.
  Mọi final SSL checkpoint và selected head dùng hash này. Source snapshot riêng của
  run được kiểm SHA-256; hồ sơ vòng trước có hash riêng, giữ như lịch sử.
- `evidence/phase4_environment.json`, full lock và editable install thực;
  `.venv-locked/pyvenv.cfg` ghi `include-system-site-packages = false`.
  `pip check`: `No broken requirements found.`

## Deliverables bắt buộc §1.2 và milestones M0–M9

| Deliverable / milestone | Code thực hiện | Bằng chứng và giới hạn |
| --- | --- | --- |
| Installable / M0 | `pyproject.toml`, `config.py`, schema, `cli.py` | Normal editable `.[dev]` thành công; lock đầy đủ; unknown keys/dims/inheritance tests |
| CLI / M0 | `cli.py` | Console/module smoke + help/error tests; lệnh cuối trong README/report |
| Data / M1 | `data/adapters/asl_citizen.py`, `local_jsonl.py`, manifest/audit | Actual CSV columns, file availability/hashes, official split membership; real pilot 32/16; full corpus chưa chạy |
| Decode/extraction / M1–M2 | `data/decode.py`, `pose/*`, `preprocessing.py` | VFR container PTS test; IMAGE detectors; raw candidates/category scores; ROI/normalization tests; raw coordinate sidecars train32/val16 |
| Cached features / M2 | `backbones/dino.py`, `cache/*` | Pinned source + actual weights SHA; missing/unexpected keys rỗng; typed arrays/schema/checksum/load guards; real RGB embed |
| Baselines / M3,M7 | `islr.py`, `kmeans.py`, `teacher_student.py` | E00 frozen visual, E01 cluster CE, E02 raw JEPA ×3 seeds; no random fallback |
| Geometry / M4 | `geometry/*`, target fitting/projection | NumPy oracle + Torch gradients; dimensions 3/10/21 và 12/25/42; motion 19/35/51; projected32; train-only all-center-phase fit |
| Masking / M5 | `masking/*`, tokenizer, collate/provider | Strict temporal + anatomical cross-modal; stream/modality variants; support closure/eligibility/alias/counterfactual tests |
| SSL/training / M6–M7 | encoder/predictor/teacher; `training/*` | Real cached optimizer updates; AMP skip; EMA/finite/accumulation/empty-target/exact resume; two Gloo ranks + real cached two-process gate |
| Downstream / M8 | ISLR/retrieval/phonology/CTC/SLT modules | Linear, attentive và temporal full FT hoạt động; full-clip tiling; CTC/seq2seq actual loss/backprop/decode; real phonology/continuous chưa có data |
| Experiments / M9 | `experiments.py`, pilot configs/scripts | 15 final clean trials; 100 updates/SSL trial; seed17/29/43; E03/E04 same masks/exposure/target counts; CSV/JSON + paired signer bootstrap |
| Tests | `tests/unit`, `tests/integration` | 128 pass không skip; mapping chi tiết bên dưới |
| Reproducibility | registries, provenance, stage ledger, records | Per-run source/config/environment/manifest/load/cache/target/diagnostics; checksums; versioned prediction JSONL; immutable stage records |
| Handoff/deployment | README/cards/method/limits/memos; inference modules | Actual raw-video NPZ; dynamic torch.export; metadata/topology/head/weights; 9 serialized parity cases + real60-sample tiled parity |

## Final checklist §20.4

| # | Requirement | Kết quả kiểm chứng |
| --- | --- | --- |
| 1 | Actual pinned assets, load report | DINO commit `7764ea0…`, weights `b938bf1…`, verified detector hashes; strict keys rỗng |
| 2 | Availability/split/train-only fitting | Pilot manifests và current-source audits; target source IDs nằm trong train pool; held-out fitting rejected |
| 3 | PTS/aliases/handedness/quality | VFR tests; actual PTS/source IDs; ambiguous wrists invalid; category scores phân biệt localization; overlay chỉ diagnostic |
| 4 | Static handshape/location retained | 49-joint body-space coordinates, quality/validity bits; static signature collision test vẫn đổi encoder output |
| 5 | Paths/windows/basis/dimensions | Explicit11groups, seconds windows, upper Lie level2 basis; analytical/native/projected dimension tests |
| 6 | Supports/closure/counterfactual | Shoulder/matching/RGB supports; hidden edits không đổi context/query; actual RGB→detector/DINO rerun ở vòng2 |
| 7 | Strict vs cross-modal | Separate primitive domains và regime diagnostics; pose-observation edit giữ RGB/ROI public; không spatial hiding claim |
| 8 | Teacher/EMA/gradients | Exact-copy/no-grad/eval teacher; successful-step EMA; cluster-only không teacher forward; unused branch policy |
| 9 | Padding/missing/empty/AMP/DDP/resume | Actual CUDA overflow test, Gloo empty-rank equivalence, exact single-process resume; NCCL/multi-GPU replay chưa đo |
| 10 | Commands and optional errors | CLI table bên dưới; unsupported optional stages reject; không silently fallback |
| 11 | Real MVP | Vòng2 raw real clips→loss/optimizer/checkpoint/embedding/head; vòng4 real corpus train + selected head→raw embed/export |
| 12 | Matched E03/E04 utility | 3 pairs, same100updates và mask digests; distinct-frame exposures18727/18668/18705; geometry counts13477/12239/12500 per method |
| 13 | Downstream/provenance/seeds/uncertainty | Same LR/head/tiling budgets; fixed official taxonomy; PI/PC/Top5/retrieval; seed SD tách conditional grouped CI |
| 14 | Functional continuous heads/status | Toy CTC/SLT gradients/decode/metrics pass; actual continuous datasets not configured → `not_run` |
| 15 | Best system separated | Không có best-system score; paired/adaptation chưa chạy; không dùng paper score làm score của code này |
| 16 | Code/config/tests/memos/limits | Files bàn giao trong method folder; reports/cards/decision/limits và commands có thật |

## Test requirements §18

| Nhóm assertions | Tests/evidence |
| --- | --- |
| Constant, L-vs-straight, subdivision, translation, reversal, scaling, affine | `test_geometry.py::test_analytical_signature`, `test_dimensions_and_static_anchors` |
| Batched dtype/oracle/gradients, missing windows/nonpositive dt | `test_geometry.py` |
| Normalization, overlapping windows, raw aliases, eligibility | `test_masking.py`, `test_sampling_collate.py`, `test_pose_cache.py` |
| Hidden-value counterfactual and query isolation | `test_model.py`, `test_cross_modal.py`; `evidence/phase2_pretrained_counterfactual.json` |
| Flattened stream indexing | `test_masking.py::test_flattened_stream_indexing`; original SHuBERT adapter unavailable, không claim reproduction |
| Padding/attention/dense alias invariance | `test_model.py`, `test_deployment.py`, `test_torch_export.py` |
| Teacher ownership/EMA/unused branches | `test_model.py`, `test_training.py` |
| Actual fp16 skipped update | `tests/integration/test_amp.py` trên CUDA, không skip |
| Empty targets/context + global rank normalization | `test_model.py`, `test_training.py`, `test_ddp.py` + real cached Gloo gate |
| Exact resume/data-order/RNG | `test_training.py`, `test_sampling_collate.py`, `test_smoke_cli.py`; single process scope |
| Metrics/retrieval/paired PC bootstrap | `test_downstream_heads.py`, `test_probe_retrieval.py`, `test_evaluation_records.py` |
| Cache/schema/index/LRU/tampering | `test_pose_cache.py`, `test_corpus_targets.py`, `test_raw_coordinate_cache.py` |
| Loader/download errors and actual output | `test_assets.py`, `test_extraction.py`; real DINO load report và embed |
| Functioning CTC/SLT | `test_downstream_heads.py`; `evidence/phase4_ctc_interface.json`, `phase4_slt_interface.json` |
| Short SignCL guard | Conditional optional backend **unavailable**; temporal_contrast_weight>0 bị từ chối. Không có SignCL test/score claim |
| End-to-end | Real artifacts vòng2; final clean15trials; `phase4_handoff_audit.json` |

## Commands và artifact contracts §14.4–14.5

| Command | Output/verification |
| --- | --- |
| doctor, config validate | environment JSON, resolved config/schema checks |
| data import/audit | split JSONL, actual availability/hash reports; import không overwrite |
| assets fetch | temp download→size/content/hash check→atomic publish+receipt |
| preprocess/cache build | raw landmarks/coordinate sidecars/ROI provenance; FeatureBundle/index/failures; immutable stage ledger |
| targets fit | Train-only stats/matrices/codebooks/source IDs/hashes; existing artifact không overwrite |
| verify pipeline | Current manifest/cache identity, exact audits, fingerprint candidates/reviews, masks/counters |
| smoke/train/resume | Config/environment/source/step logs, checkpoints, diagnostics, failure memo; resume identity guard |
| profile | Actual cached loading/update/peak memory; `profile_stages.py` tách online/teacher/predictor forward và residual backward/update |
| downstream fit | Actual trainable head, LR trial summaries, encoder/protocol lock, source/config/embeddings/spectrum |
| evaluate | Metrics, conditional CI, quality buckets, retrieval; JSON/JSONL predictions; explicit val/test và locked head |
| experiments run/summarize | Completed/failed trials, compatible comparison groups, CSV/JSON, paired source/signer bootstrap |
| embed/export | Actual raw-video timestamped embeddings/logits; online/head dynamic graph, topology/metadata/hashes, parity |

Scientific SSL training artifacts và supervised head artifacts ở hai stage folders,
liên kết bằng `downstream_selection.json` và encoder/head hashes. Không coi một head
folder là một SSL checkpoint hay ngược lại. Audit theo link đó.

## Những mục không được tuyên bố đã chạy/implement

M10 scaling, E04a/b descriptor ablations, E05 geometry inputs, E06 deep/visible-dense,
E07 paired/adaptation/fusion; alternate SignRep/SHuBERT/SignHiera/V-JEPA/DINOv3 backbones;
VC/SignCL/pose-prior losses; higher degree/lifts/online spatial masks/causal streaming;
beam/LM/pretrained-text decoder, ONNX/quantization đều chưa có runtime đã kiểm chứng.
Các mục này không âm thầm chọn default. Mở rộng degree/lift không được pilot hiện tại hỗ trợ.

Real phonological/minimal-pair/How2Sign/OpenASL transfer cần configured authorized
annotations/manifests; code interfaces/toy checks được giao theo §1.1, §15.3, §16.1.
Full-corpus benchmark, final test và multi-GPU NCCL không có trong evidence hiện tại.
Per-stage profile, actual raw embed và stage ledgers được lưu, nhưng original total
preprocessing GPU-hours/disk throughput chưa có measurement đầy đủ; fields thiếu giữ
null với reason. Không có fixed-GPU-hours, production latency hoặc efficiency claim.

Kết luận audit: deliverables bắt buộc của pipeline mặc định có code và bằng chứng thực;
optional/asset-dependent experiments được ghi rõ trạng thái. Đây là bản bàn giao để
người dùng review, không phải lời khẳng định mọi nhánh nghiên cứu đều đã hoàn tất.
