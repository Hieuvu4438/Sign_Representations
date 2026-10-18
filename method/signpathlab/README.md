# SignPathLab

**Trạng thái 13/10/2026: chỉ giữ source theo yêu cầu người dùng.** Các job đã dừng,
systemd services/timer đã tắt và gỡ khỏi user manager. Đã xóa toàn bộ `data/`,
`artifacts/`, `checkpoints/`, hai môi trường `.venv`/`.venv-locked` và cache chạy.
Code, tests, configs, dependency locks và tài liệu được giữ; dữ liệu ASL-Citizen gốc,
weights dùng chung và các method khác không bị xóa. Các báo cáo bên dưới là hồ sơ
lịch sử; artifacts mà chúng tham chiếu không còn. Muốn chạy lại cần setup từ đầu.

Triển khai độc lập proposal tại `../../docs/proposal13/signpathlab_implementation_agent_vi.md`.
Code mới nằm hoàn toàn trong thư mục method này. Pipeline cốt lõi và các backend hiện có
được bàn giao cùng acceptance audit; những nhánh mở rộng chưa có được ghi rõ bên dưới.

Hiện có config/data/preprocessing/cache, geometry/masking, source/segment batching,
temporal encoder–predictor–EMA teacher, global type/group losses, optimizer/AMP/DDP,
checkpoint/resume và dense/clip extraction. CPU smoke và real minibatch MVP đã chạy;
linear/attentive heads cùng CTC/seq2seq interfaces hoạt động trong correctness tests.
Đã có generic preprocessing/cache/target/train/downstream/evaluate CLI và pilot E00–E04
trên 8 lớp/3 seeds. Raw-video embed, dynamic torch.export, anatomical cross-modal,
collapse diagnostics, versioned predictions và grouped uncertainty đã hoạt động.

Báo cáo bàn giao: [implementation và review vòng 4](docs/IMPLEMENTATION_PHASE4_VI.md).
Đối chiếu proposal: [acceptance audit](docs/ACCEPTANCE_AUDIT_VI.md).
Các báo cáo vòng 1–3 được giữ nguyên như bằng chứng lịch sử.

ASLC3000 seed17 (3.000 train + 1.500 val) đã hoàn thành trước khi cleanup; job full
ASL-Citizen đã được dừng theo yêu cầu người dùng vì kết quả method chưa đủ tốt.
Toàn bộ features/checkpoints/training outputs của cả hai job đã xóa.
Hồ sơ setup: [ASLC3000](docs/ASLC3000_SERVICE_VI.md),
[full ASL-Citizen](docs/FULL_ASLC_SERVICE_VI.md),
[benchmark cache](docs/ASLC3000_CACHE_OPTIMIZATION_VI.md).
Các file `deploy/` chỉ là templates; hiện không có service SignPathLab được cài để tự chạy.

## Cài và kiểm tra

Môi trường đã cài sạch, không dùng system-site-packages: Python 3.13.5,
Torch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.5.3. Full dependency lock:
`requirements-main-extraction.lock.txt`; doctor: `docs/evidence/phase4_environment.json`.
`pip check` pass. `requirements-tested.txt` và environment vòng trước là hồ sơ lịch sử.
Main/extraction/continuous là extras độc lập; profile đã kiểm chứng cài cả ba trong
một venv vì MediaPipe hiện tại tương thích. Legacy Fairseq/SHuBERT chưa được tích hợp.

```bash
cd /home/haipd/Sign_Representations/method/signpathlab
python3.13 -m venv .venv-locked
.venv-locked/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-main-extraction.lock.txt
PIP_CONSTRAINT="$PWD/requirements-main-extraction.lock.txt" \
  .venv-locked/bin/python -m pip install -e '.[dev,extraction,continuous]'
.venv-locked/bin/python -m pip check
.venv-locked/bin/signpath doctor --output artifacts/doctor.json
.venv-locked/bin/signpath config validate --config configs/base.yaml
.venv-locked/bin/signpath smoke --config configs/smoke.yaml --device cpu
.venv-locked/bin/python -m pytest -q
```

Chỉ cài `.[dev]` cũng đã được kiểm tra bằng editable install thông thường.
Tests extraction/continuous cần các extras tương ứng; CUDA AMP cần GPU để không skip.
Các lệnh tạo output mới từ chối overwrite; dùng tên output mới khi chạy lại.

`signpath` và `python -m signpath.cli` có cùng semantics. Config inheritance dùng
`extends`, deep merge mapping, replace list, reject duplicate/unknown keys và cycles.
Paths bên trong config resolve từ project root; `--config` resolve từ cwd của caller.

## Data và assets

```bash
.venv-locked/bin/signpath data import --adapter asl_citizen \
  --root /home/dongvk/datasets/ASL_Citizen --output data/manifests
.venv-locked/bin/signpath data audit --manifest-dir data/manifests --output artifacts/data_audit
.venv-locked/bin/signpath assets fetch --registry registries/checkpoints.yaml --model dinov2_vits14
.venv-locked/bin/signpath assets fetch --registry registries/detectors.yaml --model face_landmarker
```

Import tính hash video thực sự, có thể tốn thời gian trên cả corpus; không đổi official
split hoặc loại missing records âm thầm. Vocabulary fitted từ train. Local JSONL dùng
schema `ManifestRecord`; `--root` nhận JSONL hoặc directory chứa JSONL. Relative video
paths trong JSONL resolve từ directory của file đó. PretrainRecord không chứa label,
gloss, text, signer hoặc arbitrary metadata.

`data audit` kiểm tra sample/source/hash/signer overlap và xác minh file hiện tại.
`verify pipeline` bổ sung temporal perceptual fingerprints và bắt buộc recorded review
cho candidates, ràng buộc fingerprint hash. Pilot không có candidates dưới threshold đã
khai báo; điều đó không chứng minh không có mọi dạng near-duplicate trên toàn corpus.

Assets tải vào file tạm, kiểm tra size/hash/error page rồi publish atomically.
Receipt lưu SHA-256 thực tế; loader chỉ nhận explicit verified SHA-256, không fallback random.
DINO source phải đúng pinned commit và không có tracked patches chưa khai báo.
MediaPipe model paths được truyền explicit vào extraction script.

## Kiểm tra một clip thật

Không cần download lại DINO/hand/pose assets đã có sẵn. Lệnh đã chạy trên **toàn bộ clip**:

```bash
.venv-locked/bin/python scripts/verify_real_clip.py \
  --video /home/dongvk/datasets/ASL_Citizen/videos/15890366051589533-APPLE.mp4 \
  --dino-repo ../signrel_temporal/third_party/dinov2 \
  --dino-weights ../signrel_temporal/weights/dinov2_vits14_pretrain.pth \
  --pose-model ../signrel_temporal/weights/pose_landmarker_lite.task \
  --hand-model ../signrel_temporal/weights/hand_landmarker.task \
  --face-model checkpoints/face_landmarker.task \
  --device cuda:0 --output artifacts/real_clip_new
```

Output gồm raw landmarks (NaN cho missing), raw candidate hands/handedness scores,
timestamps/source IDs, FeatureBundle arrays và metadata, ROI affine/padding/fallback,
strict load report, geometry/mask diagnostics. Không overwrite output cũ.
Detector infer original observations một lần, sau đó select aliases; không chạy lại
detector cho mỗi duplicated selected sample. Cache writer/reader check shape, dtype,
source/weights/extractor hashes, stream order, normalization, supports và content hashes.

Kết quả preprocessing clip đầu tiên: 71 decoded frames, 59 selected samples, valid visual streams
`[59, 0, 15, 59]` theo global/left/right/face; 10 valid geometry windows.
Riêng evidence này chỉ là preprocessing. Vòng 2 đã đi tiếp tới optimizer/checkpoint/
embedding/head trên hai train clips, theo lệnh bên dưới; vẫn không có validation/test score.

```bash
.venv-locked/bin/python scripts/verify_real_training.py \
  --config configs/diagnostic_real.yaml \
  --cache artifacts/real_clip_003/features --cache artifacts/real_clip_004/features \
  --official-train-csv /home/dongvk/datasets/ASL_Citizen/splits/train.csv \
  --video-root /home/dongvk/datasets/ASL_Citizen/videos --device cuda:0
```

Run folder identity gồm config và implementation hash; smoke thêm data/target-transform
identity. Rerun không overwrite; dùng `--resume <run>/checkpoints/last.pt` cho smoke.
Với real script, tạo cache clip thứ hai bằng `verify_real_clip.py` cùng assets, video
`35618482303951104-IMPOSSIBLE.mp4`, output `artifacts/real_clip_004`.

## Geometry và masking

`logsig2_reference` là Chen accumulation float64. `logsig2` là batched differentiable
Torch backend, luôn tính float32/64 ngoài AMP. Basis: level1 rồi upper-triangular
Lie level2, không level0. Giữ nguyên static body/hand coordinates trong raw tokens.
Signature+anchors+timing có native dimensions 12/25/42 cho d=2/4/6; ordinary-motion
control có 19/35/51. Fixed Gaussian projection đưa cả hai về q32 và standardize
train-only cả trước/sau projection. Saved fit artifacts ghi sources/hashes/dimensions.

`geometry_from_features` đưa extraction prerequisites vào support của geometry.
Hiện dùng conservative per-frame union supports cho matching/grouped pose;
strict temporal vẫn đúng, anatomical closure có thể che nhiều hơn support tối thiểu.
Mask ratio dùng raw source frames, aliases cùng bị che. Geometry targets strict chỉ
eligible khi toàn bộ support bị che; missing/padding không trở thành target.

## Trạng thái commands

| Command | Trạng thái |
| --- | --- |
| `doctor`, `config validate`, `data import/audit`, `assets fetch` | Đã implement |
| `scripts/verify_real_clip.py` | Đã chạy thật decode→detector→DINO→cache→geometry→mask |
| `smoke` | 10 CPU optimizer updates, checkpoint, embeddings, toy head/predictions |
| `scripts/verify_real_training.py` | 20 real cached minibatch updates, checkpoint/reload, full-clip embeddings và train head |
| `scripts/verify_pretrained_counterfactual.py` | Đã rerun raw RGB→detector/DINO sau khi sửa hidden source frames |
| `preprocess`, `cache build`, `targets fit`, `verify pipeline` | Đã implement và chạy trên real pilot train/val |
| `train`, `profile` | Đã implement; best selection bằng validation, last/resume và actual cached timing |
| `downstream fit`, `evaluate`, `experiments run/summarize` | Đã implement;15 real trials hoàn tất; val-only pilot |
| `embed`, `export` | Raw RGB→IMAGE landmarks/DINO→dense/clip/logits; dynamic torch.export với short/long/missing/all-invalid parity |
| SignRep/SHuBERT/SignHiera/V-JEPA/adaptation | Optional backends chưa implement |

Backend optional không có implementation sẽ từ chối chạy với nguyên nhân cụ thể.
Không có weights/score giả thay cho data hoặc pretrained assets còn thiếu.

DDP smoke dùng hai CPU processes, accumulation và rank-specific checkpoint RNG:

```bash
.venv-locked/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m signpath.cli smoke --config configs/smoke_ddp.yaml --device cpu
```

Không có multi-GPU NCCL claim từ test Gloo. CUDA fp16 overflow/skip được test thật
trên một GPU. Review vòng 1 lưu ở `docs/IMPLEMENTATION_REPORT_VI.md`, vòng 2 ở
`docs/IMPLEMENTATION_PHASE2_VI.md`; giới hạn hiện tại ở `docs/KNOWN_LIMITATIONS.md`.

## Pilot thực đã khóa protocol

`scripts/prepare_pilot.py --dataset /home/dongvk/datasets/ASL_Citizen` đã chọn8lớp,
32train/16val theo official splits/actual availability, trước khi xem model scores.
Đây là pool có chọn lớp bằng nhãn, được ghi rõ trong data card; không phải full ASL Citizen.
Không truy cập test trong pilot. Configs đã lưu tại `configs/pilot`, plan/manifests ở
`data/pilot8`, scores/source/code evidence ở `docs/evidence/phase3_*`.

```bash
.venv-locked/bin/signpath preprocess --config configs/pilot/base.yaml --split train \
  --stage pose_roi --shards 2 --assets-file registries/local_extraction.yaml
.venv-locked/bin/signpath cache build --config configs/pilot/base.yaml --split train \
  --device cuda:0 --assets-file registries/local_extraction.yaml
# Thực hiện tương tự cho split val.
.venv-locked/bin/signpath targets fit --config configs/pilot/E04_seed17.yaml --kind geometry --split train
.venv-locked/bin/signpath targets fit --config configs/pilot/E03_seed17.yaml --kind geometry --split train
.venv-locked/bin/signpath targets fit --config configs/pilot/E01_seed17.yaml --kind kmeans --split train
.venv-locked/bin/signpath verify pipeline --config configs/pilot/base.yaml --output artifacts/pilot_pipeline_checks.json
.venv-locked/bin/signpath train --config configs/pilot/E04_seed17.yaml --device cuda:0
.venv-locked/bin/signpath downstream fit --config configs/pilot/E00_seed17.yaml --device cuda:0
.venv-locked/bin/signpath experiments summarize --root artifacts/pilot8 --output artifacts/pilot8/comparison.csv
```

Target files và run folders không overwrite; các lệnh trên mô tả các stage đã thực hiện,
không là script reset dữ liệu. Reuse preprocessing/cache có checksum guards. Resume training
dùng explicit `<run>/checkpoints/last.pt` với cùng config/data/source identity. `scripts/run_pilot.sh`
chạy toàn bộ declared configs khi chưa có runs; `scripts/reproduce_summary.py` chỉ đọc results.

Cache index `data/pilot8/cache/index.json` ràng buộc source SHA, khoảng thời gian, split,
metadata SHA, FPS và input identity. Bounded reader LRU giữ64samples đã xác minh, kiểm tra
file-version mỗi lookup và revalidate content khi thay đổi; không tắt checksum để lấy speed.
`num_workers` hiện là cache-reader threads; source/segment/mask RNG thuộc main process,
tránh phụ thuộc thứ tự worker/prefetch khi resume. Research-scale process DataLoader remains optional future work.

Validation PC Top1 mean: E00=25.00%, E01=20.83%, E02/E03/E04=27.08%. E03 và E04 cùng
score ở mọi seed và matched100successful steps/mask schedules. Chưa có evidence cho
signature advantage; xem [decision memo](docs/DECISION_PILOT8_VI.md). Seed SD không là CI.


## Bản kiểm chứng cuối và deployment

`configs/verified_pilot/E*_seed*.yaml` giữ nguyên pool, seed và ngân sách pilot đã khóa,
chỉ tách output để xác nhận code cuối trong môi trường cài sạch. Bản cũ ở
`artifacts/pilot8` có source archive riêng, không trộn với bản xác nhận này.

```bash
scripts/run_pilot.sh --configs configs/verified_pilot/E*_seed*.yaml \
  --device cuda:0 --output artifacts/verified_pilot/all_runs.json
.venv-locked/bin/signpath experiments summarize --root artifacts/verified_pilot \
  --output artifacts/verified_pilot/comparison.csv
```

Chọn đường dẫn head từ trường `head_run` trong `runs_seed17.json` (E02 là raw-JEPA
reference). `--run` cũng nhận training folder có `downstream_selection.json`.

```bash
SIGNPATH_RUN=artifacts/verified_pilot/heads/E02/seed17/REPLACE_WITH_RUN_ID
.venv-locked/bin/signpath evaluate --run "$SIGNPATH_RUN" --split val --export-predictions --device cuda:0
.venv-locked/bin/signpath embed --run "$SIGNPATH_RUN" \
  --input /path/to/video.mp4 --assets-file registries/local_extraction.yaml \
  --output artifacts/my_embeddings.npz --device cuda:0
.venv-locked/bin/signpath export --run "$SIGNPATH_RUN" --format torch_export \
  --output artifacts/my_export
```

NPZ giữ `dense[T,D]`, original PTS, validity, `clip[D]`, source IDs, model/preprocessing
hashes và optional logits. Export gồm encoder/projectors/head, metadata/topology,
weights checksums và parity report; detector/DINO ngoài graph. `ExportedSession.full_clip`
áp dụng đúng tiling đã khóa. Bidirectional offline; chưa có causal streaming hoặc quantization.

`configs/experiments/E02_anatomical_cross_modal.yaml` che pose observations theo anatomy,
trong khi RGB và ROI priors vẫn public. Đây là cross-modal prediction. Strict temporal
vẫn che RGB+pose theo source frame và kiểm tra toàn bộ support. Xem `docs/METHOD.md`.

CTC/seq2seq functioning heads và metrics có lệnh correctness, không cần continuous corpus:

```bash
.venv-locked/bin/python scripts/smoke_continuous.py --config configs/downstream/cslr_ctc.yaml \
  --output artifacts/my_ctc_correctness.json
.venv-locked/bin/python scripts/smoke_continuous.py --config configs/downstream/slt_fixed_decoder.yaml \
  --output artifacts/my_slt_correctness.json
```

Real continuous runs và phonological metrics chưa chạy vì chưa có configured manifests/
annotations tương ứng. E05 geometry input, E06 deep/dense, E07 paired/adaptation,
SignRep/SHuBERT/SignHiera/V-JEPA/DINOv3, SignCL/VC/pose priors, lifts và degree>2 là
optional unavailable backends; configs không âm thầm fallback. E04a/b descriptor ablations
chưa có runtime và không được dùng để claim level-2 contribution. Chỉ E00–E04 là matrix
đã chạy. Full-corpus training, final test, best-system/transfer score cần experiment riêng.
