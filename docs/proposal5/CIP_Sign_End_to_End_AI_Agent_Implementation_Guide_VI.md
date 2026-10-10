# CIP-Sign: đặc tả triển khai end-to-end cho AI Agent

**Phiên bản:** 1.0 — 05/10/2026.  
**Method:** Conditional Innovation Pretraining for Sign Language Representation.  
**Đầu ra:** repository chạy được, có preprocessing, checkpoint loading, cross-fitted forecasters, SSL training, evaluation, export và tests.  
**Trạng thái khoa học:** proposal chưa được thực nghiệm xác nhận. Các engineering defaults trong guide chưa được benchmark; không phải thông số đã chứng minh vượt SignRep.

## 0. Chỉ thị giao việc cho agent

Đọc toàn bộ guide trước khi sửa code. Triển khai method thực sự và hoàn thành các tests/stages phù hợp tài nguyên hiện có. Không dừng ở skeleton, pseudocode, notebook minh họa hoặc TODO trong đường chạy chính.

Người dùng cho phép agent clone repositories, tải dependencies, pretrained weights/checkpoints và dữ liệu từ nguồn chính thức để hoàn thiện implementation. Thực hiện các bước tự động hóa trong project environment. Dùng credentials đã được cung cấp nếu có; không tự tạo account, giả chấp nhận thỏa thuận hoặc gửi email cho tác giả. Khi một nguồn cần access, dùng fallback đã định nghĩa hoặc hoàn thiện phần độc lập rồi ghi blocker chính xác.

Các CLI **cip-sign** trong tài liệu này là **giao diện agent phải xây dựng**. Chúng chưa tồn tại chỉ vì được mô tả ở đây. Git, pip, pytest, torchrun, ffmpeg, curl và upstream download APIs là công cụ bên ngoài.

### 0.1. Ba mức hoàn thành

| Mức | Điều kiện |
|---|---|
| IMPLEMENTATION_COMPLETE | Tất cả module bắt buộc có implementation; CPU tests qua; end-to-end fixture chạy; không còn TODO trong core path |
| RUNTIME_VALIDATED | Real weights/video chạy teacher cache, forecasters, student backward/resume, frozen evaluation/export; GPU/DDP kiểm tra khi có hardware |
| RESEARCH_VALIDATED | Matched-budget ablations, datasets/seeds/statistics; chỉ dùng sau khi có kết quả thực |

Không gọi synthetic smoke là pretrained model hoặc metric benchmark. Nếu GPU/dataset thiếu, hoàn thiện code và báo runtime/research blocker trung thực.

### 0.2. Quyết định bắt buộc

1. Inference chỉ dùng RGB student; bỏ teachers, forecasters, auxiliary decoder.
2. Main track: Hiera-B từ Kinetics MAE, 16 frames, 224×224.
3. Parts có thứ tự cố định: LH=0, RH=1, FACE=2, BODY=3.
4. CIP target là G1(own_past, peer_past)−G0(own_past). Raw residual là ablation.
5. Innovation student chỉ thấy 8 frames quá khứ; tương lai bị loại trước convolution.
6. Cross-fitting theo source groups; forecaster tạo targets chưa train trên group đích.
7. Teachers, forecasters, normalization và gates đóng băng khi train student.
8. Main controls cùng teacher, decoder, data, input/token budget, readout và selection.
9. Captions/gloss/downstream labels không vào SSL inputs, forecaster inputs hoặc calibration.
10. Implementation thành công không tự chứng minh forecast gain hoặc representation gain.

## 1. Executable method specification

Với K=8, horizons δ∈{2,4,8}:

$$
\hat y_0^b=G_0^b(y_{t-K+1:t}^b,\delta),\quad
\hat y_1^{a\to b}=G_1^{a\to b}(y_{t-K+1:t}^b,y_{t-K+1:t}^a,\delta).
$$
$$
I_{t,\delta}^{a\to b}=\operatorname{stopgrad}(\hat y_1^{a\to b}-\hat y_0^b).
$$

Student có dense state reconstruction, past-only increment prediction và variance/covariance regularization. Không diễn giải cạnh a→b là causal influence; shared intent hoặc shared coordinate frame cũng tạo predictive dependence.

Một window: frames 0–15, prefix 0–7, anchor t=7, future targets tại frames 9/11/15. State pass đọc 16 frames có masking. Prefix pass giữ cùng canvas 16 frames, đặt raw normalized RGB frames 8–15 thành zero và drop mọi future tokens.

$$
\mathcal L=\mathcal L_{\mathrm{state,full}}
+0.5\mathcal L_{\mathrm{state,past}}
+\lambda_{\mathrm{aux}}(s)\mathcal L_{\mathrm{aux}}
+0.1\mathcal L_{\mathrm{VC}}.
$$

Aux mode increment là CIP. λaux ramp 0→1 trong 2.000 optimizer steps đầu của pilot. Các controls có aux dùng cùng ramp.

~~~mermaid
flowchart TD
    A["Source RGB + timestamps"] --> B["Framewise landmarks + crop teacher"]
    B --> C["Fixed states + validity"]
    C --> D["Cross-fitted G0, G1, Gsh"]
    D --> E["Increment targets + frozen gates"]
    A --> F["Shared Hiera-B encoder"]
    F --> G["Dense state decoder"]
    F --> H["Past-only auxiliary decoder"]
    C --> G
    E --> H
    G --> I["SSL optimization"]
    H --> I
    F --> J["Export RGB encoder + fixed readout"]
~~~

## 2. Nguồn chính thức và asset registry

Links được kiểm tra khi viết guide; access/APIs có thể thay đổi sau ngày phiên bản. Prefer official sources.

| ID | Asset | URL chính thức | Vai trò |
|---|---|---|---|
| S01 | SignRep paper | https://arxiv.org/html/2503.08529v1 | Baseline/protocol |
| S02 | SignRep code | https://github.com/ryanwongsa/SignRep | Encoder fork/wrapper |
| S03 | SignRep release | https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1 | Reference checkpoint |
| S04 | Hiera code | https://github.com/facebookresearch/hiera | Architecture/reference MAE |
| S05 | Hiera K400 MAE-B | https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth | Student initialization |
| S06 | DINOv3 code | https://github.com/facebookresearch/dinov3 | Appearance teacher |
| S07 | DINOv3 S/16 | https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m | Canonical teacher |
| S08 | DINOv3 B/16 | https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m | Teacher-size ablation |
| S09 | DINOv3 API | https://huggingface.co/docs/transformers/model_doc/dinov3 | Loader/token layout |
| S10 | DINOv2 small | https://huggingface.co/facebook/dinov2-small | Public fallback |
| S11 | DINOv2 code | https://github.com/facebookresearch/dinov2 | Alternate official loader |
| S12 | Hand API | https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python | Hand teacher |
| S13 | Pose API | https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python | Body teacher |
| S14 | Face API | https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python | Face teacher |
| S15 | MediaPipe examples | https://github.com/google-ai-edge/mediapipe-samples | Bundle/source examples |
| S16 | YouTube-SL-25 release | https://github.com/google-research/google-research/tree/master/youtube_sl_25 | Source metadata |
| S17 | Official YT-SL-25 bucket | https://console.cloud.google.com/storage/browser/gresearch/youtube-sl-25 | Actual video-ID objects |
| S18 | YT-SL-25 paper | https://arxiv.org/abs/2407.11144 | Provenance |
| S19 | ASL Citizen | https://www.microsoft.com/en-us/research/project/asl-citizen/ | Signer-disjoint benchmark |
| S20 | ASL Citizen code | https://github.com/microsoft/ASL-citizen-code | Splits/baseline reference |
| S21 | WLASL | https://github.com/dxli94/WLASL | Metadata/preprocessing |
| S22 | WLASL project | https://dxli94.github.io/WLASL/ | Access guidance |
| S23 | NMFs-CSL | https://ustc-slr.github.io/datasets/2020_nmfs_csl/ | Optional benchmark, agreement required |
| S24 | PyTorch versions | https://pytorch.org/get-started/previous-versions/ | CUDA/version pairs |
| S25 | Transformers install | https://huggingface.co/docs/transformers/installation | Dependencies |
| S26 | HF snapshot API | https://huggingface.co/docs/huggingface_hub/guides/download | Revision-pinned weights |
| S27 | yt-dlp | https://github.com/yt-dlp/yt-dlp | Video acquisition |
| S28 | FFmpeg | https://ffmpeg.org/ffmpeg.html | Decode/probe |

### 2.1. Pinning

SignRep commit đã audit:

~~~text
06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2
~~~

Dùng commit này cho adapter đầu tiên. Các repo khác resolve full SHA khi bootstrap. HF: resolve model_info(...).sha, snapshot_download theo SHA, dùng local snapshot về sau. Không để branch main thay đổi giữa cache và train.

### 2.2. SignRep checkpoint xác minh từ release API

~~~text
URL: https://github.com/ryanwongsa/SignRep/releases/download/v0.0.1/ckpt.pt
Size: 222676746 bytes
SHA256: f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b
API: https://api.github.com/repos/ryanwongsa/SignRep/releases/tags/v0.0.1
~~~

Nếu hash không khớp, reject loading; refresh release metadata và ghi thay đổi nếu upstream đã replace asset.

### 2.3. Landmark bundles version 1

~~~text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
~~~

Đối chiếu links với S12–S15 khi bootstrap. Compute/store SHA256 thực tế, không bịa checksum. Không dùng latest trong lock.

### 2.4. Access/fallback

- DINOv3 HF gated: dùng account/token đã được cấp; không ghi token vào YAML/log/Git.
- Thiếu access: chạy tiếp với facebook/dinov2-small; teacher_variant=dinov2_fallback; rebuild PCA/cache; cùng fallback teacher cho mọi matched control.
- Không trộn DINOv2/DINOv3 caches hoặc ghi fallback thành canonical result.
- NMFs-CSL thiếu agreement/data: implement adapter, tiếp tục ASL Citizen/WLASL; ghi DATA_ACCESS_REQUIRED. Không tự gửi email.
- Lưu license links/copies của assets dùng. SignRep release có điều kiện nghiên cứu phi thương mại; tôn trọng điều kiện asset này.

### 2.5. Asset lock

assets/asset_lock.json lưu schema_version, source URL, full commit/revision SHA, local path, size/hash từng downloaded asset, teacher variant và license source. Runtime doctor reject unresolved runtime markers. Không bịa thông tin thiếu.

## 3. Repository và module responsibilities

Tạo project cip-sign/ độc lập; upstream ở third_party/. Vendor Hiera thành namespace cip_sign.vendor.hiera, giữ headers/licenses và đổi absolute imports nội bộ; không sửa upstream.

~~~text
cip-sign/
  pyproject.toml
  requirements-core.txt
  requirements-preprocess.txt
  requirements-lock.txt
  README.md
  IMPLEMENTATION_STATUS.md
  DECISIONS.md
  THIRD_PARTY_NOTICES.md
  configs/
    base.yaml
    profiles/{smoke,pilot,full}.yaml
    experiments/{dense_geometry,dense_state,future_state,self_forecast,
                 joint_forecast,residual,increment,shuffle_increment}.yaml
  assets/{asset_lock.json,checkpoints/,landmark_models/}
  third_party/{SignRep/,hiera/}
  src/cip_sign/
    config.py
    constants.py
    provenance.py
    vendor/hiera/
    data/{manifests,source_import,benchmark_import,decode,sampling,
          splits,cache,augment,datasets,deduplicate}.py
    teachers/{landmarks,association,crops,appearance,geometry,
              calibration,states}.py
    models/{hiera_adapter,signrep_reference,token_coordinates,
            query_decoder,forecaster,student,readout,losses}.py
    training/{forecaster_trainer,gates,target_provider,ssl_trainer,
              distributed,checkpoints,optim}.py
    evaluation/{extract,retrieval,linear_probe,finetune,
                statistics,diagnostics}.py
    cli/{doctor,assets,data,cache,forecasters,train,
         evaluate,export,pipeline}.py
  scripts/{bootstrap_repos.sh,run_smoke.sh,run_pilot.sh,run_ablations.sh}
  tests/{unit/,integration/,fixtures/}
  data/{manifests/,source_videos/,benchmark_videos/,
        teacher_cache/,calibration/,forecasters/}
  runs/
  reports/
~~~

Use src-layout; pip install -e . phải import ngoài cwd. Entry point cip-sign dispatch subcommands §18. Không tạo package models gây collision upstream.

Minimal packaging contract:

~~~toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cip-sign"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
description = "Conditional Innovation Pretraining for sign representations"

[project.scripts]
cip-sign = "cip_sign.cli.main:main"

[tool.setuptools.packages.find]
where = ["src"]
~~~

Add cli/main.py và package __init__.py files. Dependency install từ requirements trong §4; lock file checked in. Include configs/inference metadata trong package hoặc resolve từ project_root explicit; không phụ thuộc cwd khi load export.

Stages atomically write temp→rename, fingerprint input/config/weights/code, progress cursor và resume. Rerun bỏ qua artifacts đã verify, không overwrite incompatible caches.

## 4. Environment/bootstrap

### 4.1. Starting matrix

Linux/Python 3.11. Chọn CUDA wheel theo driver từ S24:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-preprocess.txt
python -m pip install -e .
python -m pip check
python -m pip freeze > requirements-lock.txt
~~~

CPU fixtures dùng cùng versions, index CPU. Không cần torchaudio.

~~~text
# requirements-core.txt
numpy==1.26.4
transformers==4.56.2
huggingface-hub>=0.34,<1
safetensors>=0.4,<1
timm==1.0.20
PyYAML>=6,<7
pydantic>=2.9,<3
Pillow>=10,<12
scipy>=1.11,<2
scikit-learn>=1.4,<1.8
pandas>=2.1,<3
pyarrow>=16,<21
tqdm>=4.66,<5
pytest>=8,<9
tensorboard>=2.16,<3
~~~

~~~text
# requirements-preprocess.txt
mediapipe==0.10.21
opencv-contrib-python==4.11.0.86
av>=12,<15
requests>=2.31,<3
~~~

Starting pins chưa được chạy trong GPU environment của người dùng. Nếu wheel/conflict không phù hợp, resolve cả matrix, ghi DECISIONS.md, chạy pip check/API/parity tests, rồi lock. Có thể tách preprocess/train venv; cache chỉ NumPy/JSON. Không silently upgrade mọi package.

Cài yt-dlp theo official current release trong download env và lock version thực tế. FFmpeg/ffprobe là system dependencies. Core không cần custom CUDA/FlashAttention.

### 4.2. Repositories

~~~bash
mkdir -p third_party assets/checkpoints assets/landmark_models reports
git clone https://github.com/ryanwongsa/SignRep.git third_party/SignRep
git -C third_party/SignRep checkout 06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2
git clone https://github.com/facebookresearch/hiera.git third_party/hiera
git -C third_party/hiera rev-parse HEAD
~~~

Existing directory: inspect origin/commit/working tree; không reset mất thay đổi. Google Research dùng raw official README hoặc sparse clone:

~~~bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-research/google-research.git third_party/google-research
git -C third_party/google-research sparse-checkout set youtube_sl_25
~~~

### 4.3. Weights

~~~bash
curl --fail --location --retry 3 --continue-at - \
  https://github.com/ryanwongsa/SignRep/releases/download/v0.0.1/ckpt.pt \
  --output assets/checkpoints/signrep_v0.0.1.pt
sha256sum assets/checkpoints/signrep_v0.0.1.pt
curl --fail --location --retry 3 --continue-at - \
  https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth \
  --output assets/checkpoints/hiera_mae_k400_base.pt
~~~

Production asset CLI dùng .part, verify size/hash, rename; reject HTML/error payload. Không download real gigabyte weights trong CPU fixture.

~~~python
from huggingface_hub import HfApi, snapshot_download
model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
revision = HfApi().model_info(model_id).sha
snapshot_path = snapshot_download(
    repo_id=model_id, revision=revision,
    local_dir="assets/checkpoints/appearance_teacher",
)
# Persist model_id, revision, local path, per-file hashes.
~~~

Gated error → fallback §2.4. Sau bootstrap, training dùng local assets; không tự truy cập network.

### 4.4. Doctor

Kiểm tra imports/pip check, FFmpeg/video decode, CUDA/BF16/VRAM, disk, weights/hash/state-dict coverage, teacher shape, manifests/splits/decontamination/cache fingerprint, planned steps/effective batch. Đo warmup throughput trước ước lượng thời gian.

Doctor có stage modes: environment (không yêu cầu assets/data đã tồn tại), assets, data, training, evaluation. Pipeline gọi theo dependency order; missing downstream artifact không làm environment bootstrap fail.

Local JSONL/TensorBoard là default. Không tự thuê máy/mua compute hoặc gửi data tới external logging service.

## 5. Data/timeline/split contract

### 5.1. Source acquisition

Main source: available subset YouTube-SL-25. Official release là video IDs; có private/deleted videos. Resolve real metadata object names từ S16/S17, không bịa CSV filename.

Public GCS listing route:

~~~text
GET https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=youtube-sl-25
~~~

Paginate nextPageToken, lấy items[].name thật rồi official object media API/download link. Nếu prefix thay đổi, đọc S16/S17 và log resolved names.

Parse video_id/channel_id/language nếu metadata có; downloader có thể lấy public metadata bổ sung. Missing field là null. Pilot acquisition target=100 unique source hours. Resume/retry hữu hạn; log unavailable; bỏ audio/subtitles khỏi SSL path. Không dùng benchmark test videos làm source fallback.

~~~bash
yt-dlp --ignore-errors --continue --no-playlist \
  --download-archive data/download_archive.txt \
  --batch-file data/manifests/source_urls.txt \
  --format "bestvideo[height<=720]/best[height<=720]" \
  --merge-output-format mp4 \
  --output "data/source_videos/%(id)s.%(ext)s"
~~~

Format flags phải kiểm tra theo actual yt-dlp release. Probe every download; archive không chứng minh decode thành công. Thiếu source: implement local-video importer và fixtures; ghi SOURCE_DATA_UNAVAILABLE.

### 5.2. Source manifest

Một row/video, Parquet/JSONL:

~~~json
{
  "video_uid": "dataset:original-id",
  "dataset": "youtube_sl_25",
  "source_video_id": "real-id",
  "channel_id": null,
  "signer_group_id": null,
  "group_id": "resolved-stable-group",
  "language": "unknown",
  "path": "relative/path/video.mp4",
  "sha256": "real-sha256",
  "duration_sec": 123.4,
  "native_fps": 29.97,
  "decode_status": "ok",
  "partition": "source_train",
  "fold_id": 0,
  "exclude_reason": null
}
~~~

Original metadata lưu riêng. Training Dataset return phải strip text/caption/gloss.

### 5.3. Group partitions

Group priority: known signer→channel→original video. Khi group systems có liên kết, dùng connected components để known same signer/source không qua partitions. Không infer identity bằng face recognition.

Seed 2026, greedy group balancing theo hours/language:

| Partition | Target | Dùng cho |
|---|---:|---|
| calibration | 5% | PCA/fixed normalization |
| forecaster_val | 5% | Forecaster selection |
| gate_val | 5% | Forecast gains/CI/target scales |
| source_train | 85% | Student + 5-fold OOF |

Indivisible groups có thể làm lệch; báo actual hours/groups. Ít data không cho phép đổi sang random-frame split.

source_train chia 5 folds. Set f train forecasters trên bốn folds khác; videos ở fold f nhận targets từ set f. Calibration/forecaster_val/gate_val không thuộc forecaster/student training.

### 5.4. Decontamination

Trước split/cache: canonicalize benchmark original IDs/URLs, exclude khỏi source; exact hashes cho copies; perceptual frame hashes flag reuploads. Log confirmed/not_duplicate/unresolved. Exclude unresolved high-confidence candidates khỏi main source. Lưu excluded/retained hours.

Recommendation audit không phải cáo buộc SignRep leakage.

### 5.5. Timeline

PyAV decode PTS/display order. Common track: 30 Hz; tại grid τj=j/30 chọn last frame with PTS≤τj. Không future interpolation. Lưu source_frame_index/pts_sec/grid_time_sec/repeated flags, xử lý rotation metadata.

PTS invalid → FPS fallback có flag. Không nối qua video hoặc shot-cut boundaries. SSL chỉ dùng valid contiguous windows đủ 16 frames. Evaluation <16 frames có last-frame padding riêng và valid_frame_count. Official SignRep diagnostic giữ native timeline như upstream; không đánh đồng với common 30 Hz protocol.

### 5.6. Clips/sampling

Clip manifest: clip_uid/video_uid/start_grid_index/length=16/anchor=7/horizons=[2,4,8]/fold/partition/history validity. Initial stride=8 frames. Unique hours tính original videos.

Sample language→group→clip; unknown language là bucket; cap channel dominance. Metadata dùng sampling, không làm network input.

### 5.7. Downstream

ASL Citizen archive:

~~~text
https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
~~~

Check S19 before download, actual checksum, member names/path safety before extract, preserve official splits. Map actual metadata columns bằng adapter tests, không đoán tên.

WLASL: S21/S22, official metadata/download/preprocessing. Count missing videos per split/class. Raw videos follow upstream frame_start/end convention; processed clips không trim lại. Preserve version/sample provenance.

NMFs-CSL: import path của data được cấp và official splits.

Benchmark manifest: sample_uid/path/label_id/label_text/split/signer_id/source ID/decode status. Label map từ train vocabulary; test không sửa vocabulary.

## 6. Teacher extraction và state representation

### 6.1. Landmark backend

Core v1 dùng ba MediaPipe Tasks riêng: HandLandmarker, PoseLandmarker, FaceLandmarker, running_mode=IMAGE. Đây là backend framewise, tránh hidden tracking/smoothing qua tương lai. Mỗi worker sở hữu task instances riêng; không share object qua processes.

- Hand: num_hands=2, detection/presence/tracking thresholds 0.5.
- Pose: num_poses=1, thresholds 0.5, segmentation output off.
- Face: num_faces=1, thresholds 0.5, output_face_blendshapes=True, output_facial_transformation_matrixes=True.
- RGB input đúng dtype/layout; normalized landmarks chuyển về native pixel coordinates.
- Save raw detections, validity, actual model options và version.

MediaPipe returned fields khác nhau giữa tasks. Handedness score là độ tin cậy phân loại handedness, **không phải per-joint localization confidence**. Face blendshape scores là expression coefficients, **không phải detector confidence**. Không bịa per-point confidence.

VIDEO mode causal tracking chỉ là extension sau prefix-equivalence test và reset policy; core không phụ thuộc extension đó. Không centered smoothing, backward interpolation, future-based association hoặc full-video median crops.

### 6.2. Person/hand association

Single-person source là main pilot. Match face với pose nose/head region. Match hand wrists với pose wrists indices 15/16 bằng normalized-distance assignment:

1. Tính distance trong native pixel space / shoulder width.
2. Solve 2×2 assignment, reject khi distance >0.75 shoulder widths hoặc hai assignments gần ngang nhau (relative cost margin <0.1).
3. Một hand: assign nearest valid wrist nếu margin hợp lệ.
4. Không có pose wrists: core reject LH/RH assignment; optional handedness fallback phải là variant khai báo riêng.
5. Không sort hands theo image x coordinate. Khi hands cross, x ordering sai.

LH/RH là anatomical assignment do detector/pose convention; validate trên mirrored/unmirrored fixtures. Không suy ra người quay thực sự thuận tay nào. Ambiguous multi-person video → log/skip hoặc explicit manually supplied ROI importer; không tự dùng future trajectory để resolve.

### 6.3. Crops

Từ current-frame landmarks, native-resolution square crop:

- Hand: enclosing box của 21 points; side=max(width,height)×1.6.
- Face: box của facial landmarks; side=max(width,height)×1.3.
- Center=current box center. Side ít nhất 16 pixels, reject tiny/invalid box.
- Out-of-image padding dùng constant RGB [124,116,104].
- Resize crop về 224×224; không mirror hoặc rotate canonicalizing orientation.
- Save actual xyxy box, out-of-image fraction, landmark association status.

Không dùng BODY appearance crop trong core: full upper-body appearance chứa tay/mặt, làm own-history body forecaster có explicit peer appearance. BODY core chỉ dùng coarse geometry. Hand/face crops vẫn có context overlap; không claim perfectly disentangled articulators.

Crop validity: correct part association, finite box, minimum size, in-image area≥50%. Missing crop → appearance validity false, không fake image embedding bằng zero rồi coi là valid.

### 6.4. Appearance teacher

Canonical: DINOv3 S/16 LVD weights. eval(), requires_grad_(False), inference_mode(). Không text-aligned checkpoint. DINOv2 small là declared fallback.

Use deterministic 224×224 transform, float [0,1], ImageNet mean/std. Native square crop đã dựng ở trên; tránh processor tự crop thêm không được ghi trong config.

Load local snapshot:

~~~python
from transformers import AutoModel
model = AutoModel.from_pretrained(local_snapshot, local_files_only=True)
model.eval().requires_grad_(False)

with torch.inference_mode():
    outputs = model(pixel_values=batch_rgb_normalized)
    hidden = outputs.last_hidden_state
    num_regs = int(getattr(model.config, "num_register_tokens", 0))
    patches = hidden[:, 1 + num_regs:, :]
    raw_appearance = patches.float().mean(dim=1)
~~~

Assert sequence length = 1 + num_regs + (H/patch_size)×(W/patch_size), từ actual config/input. DINOv3 S: hidden=384, patch=16; DINOv2 small: hidden=384, patch=14. Không hardcode 201/257 tokens hoặc bỏ register tokens sai. Test fallback loader trên actual snapshot.

Giữ raw mean-patch representation, không L2-normalize từng frame trước forecasting. Fixed PCA output 128D từng stream LH/RH/FACE; BODY appearance block luôn invalid.

### 6.5. Geometry blocks: definitions không được tự thay

Uniform state D=224:

~~~text
state[0:128]   = projected appearance
state[128:224] = geometry, padded to 96 dimensions
~~~

**Torso reference:** midpoint of pose shoulders 11/12, shoulder scale sT=2D distance giữa chúng. Coordinates tính trên pixels với x×W,y×H. sT<5 px hoặc missing shoulders → torso-relative dimensions invalid.

**LH/RH geometry, 66 real dimensions:**

| Range trong 96D geometry block | Definition |
|---|---|
| 0:60 | landmarks j=1..20, mỗi điểm 3D relative wrist, /sH |
| 60:62 | wrist image x/y relative torso origin, /sT |
| 62:65 | palm normal, normalized cross product của wrist→MCP5 và wrist→MCP17 |
| 65:66 | log(sH/sT) |
| 66:96 | padding, permanently invalid |

sH=2D wrist→middle MCP9 distance, minimum 3 px. For hand local coordinates use (x×W,y×H,z×W), subtract wrist, /sH. Hand z là detector-relative depth proxy; không coi là metric 3D hoặc dùng nó làm global wrist depth.

Palm normal khi norm cross product <1e-6 → dimensions invalid. Missing torso chỉ tắt location/scale, không tắt handshape đã valid. Geometry values phải finite; không clamp orientation thành constant.

**FACE geometry, 63 real dimensions:**

| Range | Definition |
|---|---|
| 0:52 | blendshape coefficients theo sorted category names đã lock |
| 52:61 | 3×3 rotation từ facial transformation matrix, flatten row-major |
| 61:63 | face bbox center relative torso, /sT |
| 63:96 | padding invalid |

Extract rotation bằng polar/SVD projection của matrix 3×3 linear component; enforce determinant +1. Không coi raw transformation translation là common-world position. Nếu matrix output không có, mask rotation dims false. Blendshape name count/order phải verify/store; API mismatch là migration cần báo, không truncate/pad categories giả.

**BODY geometry, 17 real dimensions:**

- Pose indices [0,7,8,11,12,23,24], mỗi điểm x/y relative torso origin, /sT: 14D.
- Unit shoulder direction in image: 2D.
- log(sT / image_width): 1D.
- Geometry padding 17:96 invalid.

BODY không đưa wrist/finger positions vào own state. Nó vẫn chứa coarse head/torso information và shared normalization; predictive edges được diễn giải thống kê.

### 6.6. Calibration

Only calibration partition:

1. Reservoir sample ≤200.000 valid frames mỗi appearance stream, seed fixed.
2. Fit PCA 128 components mỗi stream, whiten=False; require ≥1.024 valid frames/stream.
3. Fit per-dimension mean/std cho projected appearance và geometry, bỏ invalid values.
4. std_floor=1e-3; dimensions không có coverage hoặc variance dưới floor được global-mask false.
5. Apply y=(raw−mean)/max(std,floor), set invalid dims to zero.
6. No clip-wise centering, no per-video PCA, no target statistics từ downstream.

Nếu calibration insufficient, report blocker cho real profile. Smoke có miniature fixture state dimension riêng, không gọi là canonical cache.

Persist calibration.npz (allow_pickle=False), feature basis/schema, valid counts, source group hashes và teacher fingerprint. Không normalized residuals/increments thành unit vector.

### 6.7. Quality/validity

Cache per-dimension m và block qualities q=[appearance,geometry]:

- Appearance q=1 nếu crop validity pass, 0 nếu fail. Không giả calibrated probability.
- Hand geometry q=fraction valid trong 66 real dims.
- Face geometry q=fraction valid trong 63 real dims.
- Body geometry q=mean pose visibility của 7 selected points, với invalid points contribution 0.
- Pose per-coordinate validity dùng visibility≥0.5; hands/face validity dùng availability + geometric checks.
- BODY appearance q=0.

Forecaster inputs nhận past q/m. Future validity/quality chỉ dùng loss weighting; không đi vào student/forecaster input.

Mild color-augmentation teacher consistency là extension có thể thêm sau; core dùng explicit quality definitions trên. Không gate bằng speed hoặc motion magnitude.

## 7. Tensor/cache contracts

### 7.1. Shapes

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| rgb | B×3×16×224×224 | float32/BF16 | ImageNet-normalized frames |
| frame_times | B×16 | float32 | Actual relative timestamps |
| states | B×16×4×224 | float32 | Fixed normalized teacher states |
| state_valid | B×16×4×224 | bool | Per-dimension validity |
| qualities | B×16×4×2 | float32 | Appearance/geometry quality |
| roi_xyxy | B×16×3×4 | float32 | Teacher metadata only |
| keep_full / keep_past | B×392 | bool | Hiera 1=keep, 0=drop |
| g0 / g1 / gsh | B×12×3×224 | float32 | Frozen predictions |
| increments | B×12×3×224 | float32 | g1−g0 |
| aux_valid | B×12×3×224 | bool | Target dimensions usable |
| edge_gates | 12×3×2 | float32 | Edge/horizon/block global gates |
| raw_latent | B×768 | float32 | Pooled backbone output |
| embedding | B×768 | float32 | Fixed LN then L2 readout |

Ordered edges: lexicographic list [(a,b) for a in range(4) for b in range(4) if a!=b]. Save it in constants/config/checkpoints. Horizons order [2,4,8]. Flatten aux query index = edge_idx×3+horizon_idx.

Full state predictions B×16×4×224; past-state predictions B×8×4×224; auxiliary predictions B×12×3×224.

### 7.2. Cache layout

Per-video chunks up to 4.096 frames:

~~~text
video_uid_hash/
  metadata.json
  chunks/000000/
    states.npy          # Tchunk×4×224, float16
    validity.npy        # same shape, bool
    qualities.npy       # Tchunk×4×2, float16
    timestamps.npy      # Tchunk, float64
    source_indices.npy  # Tchunk, int64
    roi_boxes.npy       # Tchunk×3×4, float32
~~~

Read mmap with allow_pickle=False; upcast states/qualities to float32 for loss/forecasters. Fixed projected cache; không cache every raw 768D feature của entire corpus nếu không cần.

First extract calibration features to fit PCA; then stream all source videos with fixed projection. Cache identity = SHA256(video hash + decoder version + grid policy + teacher hashes + options + geometry schema + calibration hash).

Atomic chunk write, checksums, completed-frame ranges. Never mix fingerprints in batch.

100h×30fps×3 streams×128×float16≈8.29 GB appearance-only; total cache lớn hơn do geometry/masks/indices. Use actual measurement. Không precompute dense 36×224 aux targets cho mọi overlapping clip mặc định: lazy frozen forecaster inference tiết kiệm storage.

## 8. Hiera adapter, masking và student

### 8.1. Files cần vendor

Từ pinned SignRep:

~~~text
models/hiera/hiera_nodec_mae.py
models/hiera/hiera_clstoken.py
models/hiera/hiera_utils.py
models/hiera/hfhub.py
~~~

Đổi imports models.hiera.* sang cip_sign.vendor.hiera.*. Timm deprecated imports có thể migrate tương đương, nhưng parity tests phải pass.

Instantiate mae_hiera_base_16x224(pretrained=False,num_classes=400,num_cls_tokens=0,drop_path_rate=0.1). Expected base: embed_dim96, stages(2,3,16,3), final_dim768, patch kernel(3,7,7), stride(2,4,4), padding(1,3,3), q_pool2, q_stride(1,2,2).

Load S05 checkpoint model_state. Filter only known MAE decoder/head/norm extras that encoder-only instance không dùng. Strictly match every trainable encoder parameter shape/name; report missing/unexpected/loaded-byte coverage. Generic strict=False không được làm mất patch embed/blocks/position embeddings mà không báo lỗi.

Nếu format khác model_state, inspect actual official checkpoint và implement explicit migration. Không tự gọi unrelated ImageNet checkpoint là K400.

### 8.2. SignRep reference wrapper

Dùng upstream final model và example model_params, init_weights=False, out_dim768, đúng prior head dimensions; load checkpoint["model"] với exact shape validation. Preserve full reference wrapper for parity, then expose backbone-only and upstream projected features riêng.

Static audit: wrapper init_weights=True gọi method không có trong wrapper và broad except có thể silently đổi cls_head thành Identity. Không dùng default này; reference init_weights=False như example. New student không kế thừa broad exception.

Repo không có complete pretraining/fine-tuning trainer. Không tuyên bố tự tái lập exact SignRep training từ repo alone. Published checkpoint reference và common-budget baselines là hai categories riêng.

### 8.3. Adapter API

~~~python
@dataclass
class EncoderOutput:
    raw_latent: torch.Tensor
    stage_maps: list[torch.Tensor]
    stage_coordinates: list[torch.Tensor]
    keep_mask: torch.Tensor

class HieraVideoEncoder(nn.Module):
    def encode(self, rgb, keep_mask) -> EncoderOutput:
        ...

    def encode_unmasked(self, rgb) -> EncoderOutput:
        # Always explicit all-True mask, including inference.
        ...
~~~

Ellipses trong block này mô tả interface; agent phải thay bằng implementation.

Always explicit keep_mask, cả all-True. Upstream masked forward returns token sequence trước pooling khi mask supplied. Không dựa vào unmasked path rồi mean một tensor đã pooled lần nữa.

### 8.4. Mask geometry

Input grid after patch stride: 8×56×56. Mask units: 1×8×8 tokens → mask grid 8×7×7=392 units. Unit ID:

~~~python
grid_t, grid_y, grid_x = mask_grid_shape
plane = grid_y * grid_x
mu_t = unit_id // plane
mu_y = (unit_id % plane) // grid_x
mu_x = unit_id % grid_x
~~~

Full pass giữ 78 units/sample (int(392×0.2)); all samples cùng count. Prefix pass chỉ chọn units mu_t<4, giữ cũng 78/196 eligible units. Overall token count matched; mask fraction trong available prefix khác full pass và phải log.

~~~python
prefix_rgb = rgb.clone()
prefix_rgb[:, :, 8:] = 0.0
keep_past = sample_fixed_count_mask(
    eligible=(unit_time_index < 4), n_keep=78,
)
assert torch.all(keep_past.sum(-1) == 78)
assert not keep_past[:, unit_time_index >= 4].any()
~~~

Zero ở normalized RGB là mean-color mask. Explicit raw zeroing và dropped future units đều bắt buộc. Upstream do_masked_conv mask trước overlapping convolution; không chỉ drop tokens sau embedding.

No frame-padding future, no random full-window crop using future boxes, no feature extraction from full video rồi slice latent. Prefix photometric augmentation parameters chỉ draw từ RNG hoặc prefix, không fit full-video statistics.

### 8.5. Intermediate maps/coordinates

Selected stages indices 1,2,3:

| Stage index | Channels | Within-unit shape khi mask supplied | Memory tokens với 78 units |
|---:|---:|---|---:|
| 1 | 192 | 1×4×4 | 1248 |
| 2 | 384 | 1×2×2 | 312 |
| 3 | 768 | 1×2×2 | 312 |

Total memory 1872 tokens/pass. Verify dynamically on actual pinned model; fail if shapes differ.

Reroll(mask=...) maps B×Nkeep×MU_t×MU_y×MU_x×C. Unit ordering expected ascending kept unit IDs; verify bằng labeled-index unroll/reroll fixture, không assume flat tokens đã ở raster order.

Coordinates mỗi stage token:

~~~text
x = (mu_x*within_width + local_x + 0.5)/(7*within_width)
y = (mu_y*within_height + local_y + 0.5)/(7*within_height)
t = (2*mu_t)/(16-1)
~~~

Memory = Linear(Cs,256)(token)+fixed Fourier xyz projection+learned stage embedding. Fourier: frequencies [1,2,4,8], sin/cos each of x/y/t →24D→Linear(24,256). Không cung cấp teacher ROI hoặc future state vào memory.

Debug helper scatter masked maps into full unit grid, zero+validity cho missing units. Validate all-True maps bằng upstream undo_windowing. Không infer coordinates từ latent value.

### 8.6. Decoder

Shared temporary TransformerDecoder: d_model256, 3 layers, 8 heads, FFN1024, dropout0.1, norm_first=True, batch_first=True.

Queries:

- Full state: frame embedding index0..15 + part embedding0..3 + state-task embedding; 64 queries.
- Prefix: 32 past-state queries +36 aux queries.
- Aux query: ordered-edge embedding + horizon embedding + anchor frame7 embedding + aux-task embedding.

Cross-attention memory từ selected maps. Queries không nhận target state, crop/landmark coordinates, quality future hoặc labels. Decoder self-attention giữa query embeddings được phép; không teacher forcing.

State head và aux head: riêng MLP256→256→224, GELU. Same heads exist in all controls. Outputs reshape theo §7.

Dense supervision là theo part/frame trên spatial-temporal tokens, không qua pooled vector alone. Không cần explicit ROI query ở core.

### 8.7. Fixed inference readout

~~~python
def representation(raw_latent):
    dim = raw_latent.shape[-1]  # canonical=768; smoke=128
    z = F.layer_norm(raw_latent.float(), (dim,), eps=1e-6)
    return F.normalize(z, p=2, dim=-1, eps=1e-8)
~~~

No trainable pooling/projection ở primary common protocol. Decoder gradient tới encoder; VC trực tiếp regularize fixed-LN raw latent trước L2. Export cả raw_latent và embedding khi cần.

SignRep upstream features = learned LN+Linear, latent = backbone pooled vector. Report separate fields; không interchange silently. Common comparisons dùng fixed readout trên latent của tất cả encoders. Published diagnostics dùng upstream projected features theo named protocol.

## 9. Cross-fitted forecasters

### 9.1. Architecture matching

Three model types per fold: G0 own-only, G1 real-peer, Gsh shuffled-peer. Identical parameter count/architecture/init seed/training budget.

Một training example chọn ordered edge a→b và horizon δ uniformly:

- G0 inputs: two slots đều chứa own b history (duplication adds no peer information).
- G1: slot0 own b, slot1 peer a.
- Gsh: own b + a-history từ nonoverlapping other-video sample theo shuffle policy.

G0 slot1 dùng actual part-ID b; G1/Gsh slot1 part-ID a. Same embedding table, same pair/horizon query. G0 có query biết loại cạnh, nhưng không có peer observations.

Input per history token: concat normalized y224 + validity224 + q2 =450D. Missing values zero, mask explicitly supplied. Project450→128; add part/role/time embeddings.

16 tokens (8 times×2 roles), interleaved theo time. TransformerEncoder 2 layers, d128, 4 heads, FFN512, dropout0.1, norm_first. Attention key j được đọc khi time_j≤time_i; same-time roles được thấy nhau.

One forecast query = pair embedding+δ embedding; TransformerDecoder1 layer,d128,4heads,FFN512 đọc known history. MLP128→128→224 output future state of b.

No signer/channel/language IDs thành learned embeddings. Metadata chỉ sampling/shuffling.

### 9.2. Training

Each fold f:

1. Fit G0/G1/Gsh trên source_train folds !=f.
2. Train all same windows/pair/horizon schedule, seed và optimizer settings.
3. Early select bằng forecaster_val MSE; gate_val chưa dùng selection.
4. Save best states và fit-group list/hash.
5. Freeze every parameter; OOF checks before target generation.

Defaults: batch256 selected edge/horizon examples; AdamW lr3e-4, betas(.9,.95), wd.01, cosine; epochs30/patience5 pilot; clip1; FP32 losses. Evaluation all edges/horizons chunked. Mixed precision optional forward; comparisons cùng precision.

Objective: quality-weighted block MSE over valid future target dimensions (§11), same masks/weights cho G0/G1/Gsh. Không train one model longer sau nhìn gate metrics. Trivial persistence y_t baseline phải được đánh giá để đo task difficulty.

### 9.3. Shuffle policy

Choose other video, nonoverlapping original source, same language and same channel where multiple videos exist; otherwise same language/different group. Preserve peer part a and 8-frame temporal order.

Record matched-channel fraction/fallback fraction. Reject peer sample with inadequate validity; no labels/gloss matching. Deterministic schedule theo seed/clip UID.

Gsh trained với shuffled pairing. Main shuffle_increment control distills Gsh−G0; additional inference-only G1 with permuted peer là diagnostic, không thay Gsh control.

Resolve/store shuffle_partner_uid cho mỗi clip. Training partners chỉ từ forecaster training partitions/folds; validation partners từ cùng held-out validation partition. OOF target partners ở cùng held-out fold f với own clip, để cả own và shuffled peer group chưa nằm trong fit groups của set f. Nếu không tìm được partner đủ validity, tắt aux entry đó cho **tất cả aux controls** trên common support; state losses vẫn dùng clip.

Target generation Gsh phải nhận shuffled partner history, không current real peer history. TargetProvider có read-only StateCache để lấy partner hoặc Dataset cung cấp explicit shuffled_past_y/m/q. Không lấy future states của partner. Save partner UID/seed for reproducibility.

### 9.4. OOF target provider

For each clip read fold f and load model set f. Assert own/shuffled partner groups not in fit-group set. Targets computed inference_mode on states[:8] only; Gsh uses partner history theo §9.3. g0/g1/gsh outputs ordered B×12×3×224; future states only used for validity/loss evaluation/raw-residual control.

All small model sets có thể resident on GPU; otherwise fold-homogeneous batches và load cache. Do not accidentally choose set trained on held-out fold.

Target provider API returns predictions, difference targets, target masks/scales/gates and provenance IDs. Output targets detached, requires_grad=False. Cache computed targets only if profiling justifies storage.

## 10. Gates, target scales và failure handling

### 10.1. Forecast error

Evaluate each fold model on gate_val, per edge/horizon/block:

$$
\Delta = \mathrm{MSE}(G_0,Y)-\mathrm{MSE}(G_1,Y),\quad
g_{\mathrm{rel}}=\Delta/\max(\mathrm{MSE}(G_0,Y),10^{-8}).
$$

Use exactly same targets/valid masks/weights cho models. Per-video aggregation trước cluster bootstrap theo group. Report mean, std, absolute errors, relative gain, group count and effective valid examples.

### 10.2. Gate default

Block gate=1 nếu:

- ≥10 independent groups và ≥1000 valid edge/horizon examples trên gate_val;
- relative MSE gain≥5%;
- 95% bootstrap CI lower bound của Δ>0;
- G1 tốt hơn Gsh trên paired errors, CI lower bound>0;
- increment RMS≥0.05 standardized state units.

Else gate=0, kèm reason. Bootstrap2000 resamples, fixed seed. Đây là engineering thresholds, không phải established theorem. Save all raw metrics để sensitivity analysis; đừng chỉ report edges đã pass.

Gates không theo motion speed; holds có thể valid. BODY appearance luôn gate0. Student dense state loss vẫn học cả parts bị tắt increment.

### 10.3. Validity at sample level

Own/peer histories phải có ≥6/8 available frames trong block tương ứng. Future target dimension validity dùng cached m của b ở frame 7+δ. Weight=global gate×past block coverage×future block quality; future quality chỉ loss-side.

Common aux support cũng yêu cầu valid shuffled history theo §9.3; dùng cùng support/quality-weight policy khi đo G0/G1/Gsh và train aux controls. Report lượng coverage mất do yêu cầu này.

For output coordinate d, aux_valid requires target dimension valid and sufficient history coverage. Fixed global masks của calibration cũng apply. Invalid target zero nhưng không contribute loss.

### 10.4. Target scale

Compute per fold/edge/horizon/block scale on gate_val:

$$
\sigma_{\mathrm{mode}}=
\max\left(\sqrt{\mathrm{mean}(T_{\mathrm{mode}}^2)},0.05\right).
$$

Tmode=increment/future state/G0/G1/raw residual tùy experiment. No per-sample vector normalization. Divide block MSE by σ²; apply scale to loss, không làm target gần zero thành large vector. Shuffled control dùng σincrement của real G1−G0 để không amplify near-zero shuffled signal.

Target-scale tuning budget phải tương đương: default λaux1; optional grid [.3,1,3] source-only selection cho tất cả aux modes. Không tune riêng CIP bằng downstream test.

### 10.5. Không có valid edges

Implementation vẫn phải hoàn chỉnh và tests synthetic positive-edge vẫn chạy. Real run emit NO_VALID_EDGES, λaux effective0, thực hiện dense baseline nếu budget cho phép. Không bypass statistical gates bằng hand-selected weights rồi gọi là core CIP success.

Đây là kết quả âm cho pilot configuration, không phải software completion blocker. Store diagnostics để xem lại teacher quality/context length/forecaster capacity.

## 11. Loss implementation chính xác

### 11.1. Block-weighted MSE

Appearance=0:128; geometry=128:224. Với prediction/target/mask và sample-block weight w:

$$
\ell_{i,k}=
\frac{\sum_{d\in k}m_{i,d}(p_{i,d}-y_{i,d})^2}
{\max(\sum_{d\in k}m_{i,d},1)}.
$$

Reduce weighted mean trên valid sample/query/block entries, không flatten thành một mean của toàn bộ 224 dimensions. BODY appearance/padding không contribute. Appearance128D không được chiếm loss chỉ vì geometry ít dimensions hơn.

~~~python
def weighted_block_mse(pred, target, valid, weight, block_slices, scale=None):
    # All arithmetic below in float32.
    numer = pred.float().sum() * 0.0
    denom = pred.new_zeros((), dtype=torch.float32)
    for k, sl in enumerate(block_slices):
        m = valid[..., sl].float()
        count = m.sum(-1)
        err = ((pred[..., sl].float() - target[..., sl].float()) ** 2 * m).sum(-1)
        err = err / count.clamp_min(1.0)
        if scale is not None:
            err = err / scale[..., k].float().square().clamp_min(0.05 ** 2)
        wk = weight[..., k].float() * (count > 0).float()
        numer = numer + (wk * err).sum()
        denom = denom + wk.sum()
    return numer, denom
~~~

Agent implement global normalization helper để function không biased khi ranks có khác số valid targets.

State weights = per-frame q blocks; state valid=m; full/past state scales=None vì teacher states đã fixed-normalized. Aux weights/gates/scales theo §10. Shape broadcasting được test, không implicit broadcast nhầm edge/time.

### 11.2. Aux modes

| Mode | Target | Ghi chú |
|---|---|---|
| none | no aux loss | Dense baseline |
| future_state | true teacher y at 7+δ for b | RGB future prediction control; không gọi là own-only student |
| self_forecast | G0 | Teacher own-history prediction |
| joint_forecast | G1 | Direct joint-teacher distillation |
| residual | y_future−G0 | Raw unpredictable residual control |
| increment | G1−G0 | Core CIP |
| shuffle_increment | Gsh−G0 | Trained shuffled-peer control |

Same ordered query set; future_state/self_forecast repeated across peer edges được giảm qua cùng reduction, không cộng loss gấp ba. Same gates của real CIP apply aux controls để support set không khác nhau. Record target coverage.

None still executes aux head/query path for architecture/forward accounting; connect 0×aux.sum() khi needed để DDP không có unused params. Profiling backward compute vẫn phải báo thực tế; equal number of passes không tự guarantee equal GPU-hours.

### 11.3. VC regularizer

z=LayerNorm(raw_latent,affine=False), trước L2. Compute separately trên full-pass global microbatch và prefix-pass global microbatch, average hai losses.

$$
\mathcal L_{\mathrm{var}}=
\frac1d\sum_j\max(0,0.5-\sqrt{\mathrm{Var}(z_j)+10^{-4}}),
$$
$$
\mathcal L_{\mathrm{cov}}=
\frac1d\sum_{i\ne j}C_{ij}^2,\qquad
\mathcal L_{\mathrm{VC}}=\mathcal L_{\mathrm{var}}+0.04\mathcal L_{\mathrm{cov}}.
$$

Variance/covariance use unbiased estimator N−1; require N≥4 ở real training. γ0.5/loss weights là defaults, không established optimum. FP32, không add invariance term hoặc copy VICReg loss khác mà không đổi config/spec.

DDP dùng differentiable all_gather cho z. Ordinary detached all_gather làm mất regularizer gradient. Test distributed gradient against single-process global-batch reference.

Gradient accumulation không làm N của VC thành effective batch. Log global_microbatch_for_VC và effective_optimizer_batch riêng. Nếu VRAM buộc N<4, set VC weight0 trong explicit low_memory variant cho **mọi controls** và báo deviation; không silently tính covariance invalid.

### 11.4. Distributed weighted reduction

DDP averages parameter gradients. Với local numerator Nr, global detached denominator D=ΣrDr:

~~~text
loss_rank = world_size * local_numerator / max(global_denominator, epsilon)
~~~

Như vậy DDP averaged gradient tương đương global weighted objective. Detach/all_reduce denominator only; local numerator giữ autograd. If global denom=0, return graph-connected zero. All ranks cùng collectives/order, kể cả all-invalid local batch.

No teacher gradients; no optimizer params của forecasters/teachers. Masked coordinate errors có zero gradient. Unit tests bắt buộc cho các invariants này.

## 12. SSL training, DDP, checkpoint/resume

### 12.1. Two-pass update

1. Decode/sample canonical RGB and cached states, timestamps aligned.
2. Shared temporal photometric augmentation on RGB, fixed teacher target giữ canonical state.
3. Full pass with 78 keep units → full state predictions/raw latent.
4. Build prefix by zeroing future; prefix keep mask only mu_t<4 and count78.
5. Prefix pass →32 past states+36 aux outputs/raw latent.
6. OOF frozen target provider on past states, gate/scales; no_grad.
7. Compute state_full+0.5state_past+λauxaux+0.1VC.
8. Backward/accumulate, unscale if FP16, clip norm1, optimizer update, scheduler update.
9. Logging/checkpoint at optimizer-step intervals.

One shared encoder/decoder instance, not two separately initialized networks. Targets computed outside DDP gradient graph.

### 12.2. Augmentation

Core: exact resize to224×224 như SignRep common input; ImageNet normalize. Whole-clip shared ColorJitter with brightness/contrast/saturation0.2,hue0.02; parameters independent of clip content. Optional JPEG/noise nuisance augmentation sau baseline validation.

No horizontal flip, temporal reverse, speed warp, random spatial crop hoặc per-frame independent jitter ở core. Spatial variants phải transform geometry targets/validity consistently hoặc recache teachers; không giữ old coordinates sau crop.

Teacher/student brightness khác được coi nuisance training choice, cần controls cùng recipe. Không assume mọi visual transform bảo toàn nghĩa.

### 12.3. Optimizer

Pilot defaults:

- AdamW, backbone max LR1e-4, decoder LR2e-4.
- betas(.9,.95), weight decay0.05.
- Layer decay0.85: patch/position layer0, block i→i+1, deepest block24. Backbone LR layer ℓ =1e-4×0.85^(24−ℓ).
- Decoder groups LR2e-4 without layer decay.
- Norm parameters, biases, position/part/time/role embeddings: weight decay0.
- Cosine, min LR1e-6, warmup2.000 optimizer updates.
- Max20.000 optimizer updates, checkpoint every1.000, validation every1.000.
- BF16 autocast if supported; losses FP32; gradient clip1.

These practical defaults khác published SignRep recipe, đặc biệt weight decay. Chúng define common experimental track. Không claim equal training budget with original SignRep chỉ vì number of steps giống.

Default single-GPU microbatch5, accumulation4→effective20. DDP derive:

~~~text
effective_batch = microbatch_per_rank * world_size * accumulation_steps
~~~

World4/micro5/accum1→effective20, VC batch20. Single-GPU default VC batch5. If effective20 không chia hết, config error hoặc explicit new effective batch; không silently ceil.

### 12.4. Dataloader/sampler

Deterministic group-balanced sampler yields (clip_uid,occurrence_idx,fold). Aug/mask seed derived hash(run_seed,global_microstep,rank,clip_uid,occurrence_idx). Không phụ thuộc worker-prefetch RNG.

Batch/fold mixture được target provider xử lý đúng. Optional fold-homogeneous batching vẫn preserve group/language frequencies. Drop last or pad with explicit sample weights0 cho distributed size parity.

Decode readers/task objects local workers; không share unsafe OpenCV/PyAV handles across fork. Avoid per-batch downloading.

### 12.5. DDP

torchrun env LOCAL_RANK/RANK/WORLD_SIZE; one GPU/rank. CPU tests use gloo, CUDA use nccl. no_sync applies accumulated forwards/backward consistently; synchronize update-boundary backward. Trainer must work with world_size1.

Weights/meta teachers not DDP wrapped. Student all parameters either receive real gradient or graph-connected zero. All-invalid batch policy: warning; after50 consecutive zero valid-state batches fail with data diagnostic, not infinite training.

Nans: log offending clip UID/stage, synchronize failure across ranks, save debug metadata; do not skip arbitrarily until “train succeeds”.

### 12.6. Checkpoint contract

Full training checkpoint:

~~~text
schema_version, method_version, experiment, optimizer_step, global_microstep
student state_dict, optimizer state, scheduler state, scaler state if FP16
per-rank RNG/sampler cursor
config resolved + hash, code commit + dirty diff hash
asset lock hash, teacher/calibration/geometry fingerprints
source/splits hashes, forecaster sets/group hashes, gates/scale hashes
readout identifier, input spec, selected stages, edges/horizons
validation summary, runtime/profile status
~~~

Save last.pt atomically + immutable step_XXXXXXX.pt. Per-rank RNG gathered with rank mapping. Keep own resume metadata JSON-safe; avoid arbitrary pickled Python objects. Load own trusted tensor/primitives checkpoints using weights_only=True where possible.

Legacy upstream torch payload nếu cần alternate loader phải được verified official asset và explicit documented loader path; không tự disable safe loading cho arbitrary inputs.

Resume reject incompatible calibration/source folds/teacher/schema. Warm-start là command riêng, resets optimizer/scheduler/counters và logs exact loaded keys.

Exact-resume test dùng deterministic CPU mini model/no nondeterministic kernels; real GPU có numerical tolerance. Record world-size change: exact data-order resume only promised for same world size; changed world size là documented approximate continuation.

### 12.7. Selection

Main SSL checkpoint = fixed final step của budget. Source-only held-out dense/forecast errors là diagnostics; không nhìn downstream test để select. Nếu dùng downstream val cho hyperparameter tuning, declare label-informed selection track riêng và dùng cùng budget cho tất cả methods.

## 13. Baselines/ablations phải implement

| ID | Initialization | State objective | Aux | Vai trò |
|---|---|---|---|---|
| B0 | K400 MAE | no extra SSL | none | Pretrained encoder-only reference |
| B2-ref | Released SignRep | released weights | none at eval | Published checkpoint reference, training budget khác |
| B3 | K400 MAE | geometry only | none | Dense geometry contribution |
| B4 | K400 MAE | appearance+geometry | none | Dense state teacher baseline |
| B5 | K400 MAE | full state | future_state | Raw future-state prediction |
| B5a | K400 MAE | full state | self_forecast | Own-history teacher |
| B6 | K400 MAE | full state | joint_forecast | Direct G1 distillation |
| C0 | K400 MAE | full state | residual | Raw residual |
| C1 | K400 MAE | full state | increment | CIP |
| C2 | K400 MAE | full state | shuffle_increment | Shuffled-peer target |

B3 geometry loss masks appearance false, same model architecture. B4/other controls cùng two encoder passes + state_full/state_past. All aux experiments same support gates và target provider set.

B2-ref không được gọi common-budget reproduction. Có thể implement separate SignRep-like trainer từ paper sau, nhưng published geometry61 points/discriminator details không đầy đủ trong current snapshot; approximations phải có tên riêng.

Warm-start all B3…C2 from same released SignRep checkpoint là additional continuation track, không trộn với K400 main track.

Pixel MAE continuation, long-window32/64, VideoMamba, V-JEPA teacher, MoE và translation là follow-up experiments; không chặn core implementation acceptance. Chúng cần config/compute/protocol riêng nếu làm.

Critical comparison: C1 vs B4, B5, B6, C0, C2. Nếu B6 ngang C1, không claim increment decomposition giúp representation. Nếu B4 đã giải thích gain, report dense supervision/teacher transfer.

## 14. Evaluation/export end-to-end

### 14.1. Common frozen feature protocol

- Every model: same decoded30Hz RGB, 16 frames, window stride2, resize/normalize.
- Encoder unmasked/all-True keep mask, eval mode.
- Per-window feature=fixed LN+L2 raw latent.
- Per-video feature=normalized arithmetic mean của valid window features.
- Include final full window once nếu stride grid không tới last valid start; <16 frames last-pad policy.
- No activity weighting/learned pool in primary common table.

Window tail policy phải giống cho mọi model. Upstream SignRep example drops short tails; diagnostic giữ upstream behavior, common protocol policy nêu trên báo riêng.

Save embedding caches with sample IDs/labels/splits/model hash/protocol hash, not just anonymous arrays. Extraction must run without any teacher/forecaster dependency.

### 14.2. Frozen retrieval

Default class-centroid gallery:

1. Gallery=train samples only.
2. Compute normalized mean feature mỗi train class.
3. Query=val/test video features.
4. Cosine scores; rank distinct classes.
5. R@1/R@5/R@10, MRR, per-instance/per-class accuracy, support counts.

Alternative exemplar gallery reports separate protocol; rank distinct labels by maximum exemplar score. Do not confuse top-k videos with top-k classes.

Class absent from available train gallery: report count and mark impossible retrieval; include errors in all-query metric, optionally covered-class metric separate. Never silently drop difficult/missing classes.

ASL Citizen official signer-disjoint splits preserved. WLASL subset/full protocol and actual available sample counts explicit. Published SignRep retrieval values khác aggregation/activity/gallery recipe nên không compare raw numbers as identical protocol.

### 14.3. Linear probe

Freeze encoder, use cached per-video embeddings. Train linear768→C, CE. Grid LR[1e-3,1e-2]×wd[0,1e-4], AdamW, batch256,100epochs. Select best val accuracy per head/run; same grid/budget all methods. No class-distribution KL.

Test only once after val selection. Record best epoch/head config; PI/PC and confusion summary. Seed head training consistently; no test tuning.

### 14.4. Full fine-tuning

Additional track: same encoder/readout, linear classifier, CE; train random contiguous16-frame window/sample; eval mean window logits. AdamW backboneLR1e-5/headLR1e-3,wd.05,100epochs,warmup5epochs,cosine, effective batch32; aug same safe photometric family. Gradient clip1.

Select best validation PI; same budgets/splits/input resolution for all methods. Class-distribution KL off in common track; optional equal KL variant explicit. Never attribute labeled fine-tuning improvements entirely to SSL.

### 14.5. Statistics/diagnostics

Main research comparisons seeds[0,1,2]; report mean/std between training seeds. Paired bootstrap2000 per signer/group where known, otherwise per video; confidence intervals on metric differences. Windows from same video không là independent samples.

Tests: fixed gallery, source decontamination counts, teacher validity by part, gates by edge/horizon, future/past hold cohorts, missing modalities, embedding variance/collapse.

Linguistic minimal contrasts/phonology cohorts chỉ claim khi có annotations verified. Agent không tự tạo giả nhãn “minimal pair” từ English gloss similarity.

### 14.6. Export

export command writes encoder.safetensors + model_config.json + preprocessing.json + provenance.json + README. Decoder/teachers/forecasters không trong export.

Public inference API:

~~~python
encoder = load_export(export_dir, device="cuda")
window_embeddings = encoder.encode_windows(rgb_windows)
video_embedding = encode_video(video_path, encoder, protocol="common_v1")
~~~

Export parity: full student encoder vs exported model same input produce max abs error≤1e-5 FP32; BF16 test tolerance≤1e-3. Project import/export không download teachers when inference only.

## 15. Configuration đầy đủ

Agent implement strict schema + deep merge base→profile→experiment→CLI overrides. Reject unknown keys, noncanonical main dimensions, unsafe unresolved paths, horizons outside available future, n_keep>eligible. Dump resolved.yaml mỗi run.

### 15.1. configs/base.yaml

~~~yaml
method_version: "cip-sign-v1"
seed: 0
profile: pilot
project_root: "."
paths:
  asset_lock: "assets/asset_lock.json"
  data_root: "data"
  source_manifest: "data/manifests/source.parquet"
  clip_manifest: "data/manifests/clips.parquet"
  benchmark_manifest: "data/manifests/benchmarks.parquet"
  calibration: "data/calibration/calibration.npz"
  teacher_cache: "data/teacher_cache"
  forecaster_root: "data/forecasters"
  run_root: "runs"
  report_root: "reports"
data:
  source_dataset: youtube_sl_25
  source_hours_target: 100
  target_fps: 30
  time_sampling: previous_pts
  frames: 16
  context_frames: 8
  horizons: [2, 4, 8]
  clip_stride: 8
  split_seed: 2026
  partitions: {calibration: 0.05, forecaster_val: 0.05, gate_val: 0.05, source_train: 0.85}
  folds: 5
  sampler: language_group_balanced
  num_workers: 4
teacher:
  appearance_model: "facebook/dinov3-vits16-pretrain-lvd1689m"
  fallback_model: "facebook/dinov2-small"
  allow_declared_fallback: true
  resolution: 224
  pooling: mean_patch
  pca_dim: 128
  geometry_dim: 96
  state_dim: 224
  parts: [LH, RH, FACE, BODY]
  body_appearance: false
  landmark_backend: mediapipe_tasks
  landmark_running_mode: IMAGE
  detection_threshold: 0.5
  hand_crop_factor: 1.6
  face_crop_factor: 1.3
  min_crop_pixels: 16
  min_in_image_fraction: 0.5
  calibration_reservoir: 200000
  calibration_min_valid_frames: 1024
  norm_std_floor: 0.001
  cache_chunk_frames: 4096
encoder:
  name: signrep_hiera_base
  initialization: hiera_mae_k400
  frames: 16
  resolution: 224
  output_dim: 768
  selected_stages: [1, 2, 3]
  drop_path_rate: 0.1
  keep_units_full: 78
  keep_units_past: 78
decoder:
  dim: 256
  layers: 3
  heads: 8
  ffn_dim: 1024
  dropout: 0.1
forecaster:
  dim: 128
  encoder_layers: 2
  decoder_layers: 1
  heads: 4
  ffn_dim: 512
  dropout: 0.1
  batch_size: 256
  epochs: 30
  patience: 5
  lr: 0.0003
  weight_decay: 0.01
  betas: [0.9, 0.95]
  clip_grad_norm: 1.0
gates:
  relative_gain_min: 0.05
  increment_rms_min: 0.05
  min_groups: 10
  min_examples: 1000
  bootstrap_resamples: 2000
  confidence_level: 0.95
  require_shuffled_control_gain: true
  min_history_frames: 6
  target_scale_floor: 0.05
experiment:
  name: increment
  aux_mode: increment
  state_blocks: [appearance, geometry]
  gate_support: real_increment
loss:
  full_state_weight: 1.0
  past_state_weight: 0.5
  aux_weight: 1.0
  aux_ramp_steps: 2000
  vc_weight: 0.1
  vc_gamma: 0.5
  vc_epsilon: 0.0001
  vc_cov_weight: 0.04
training:
  max_steps: 20000
  warmup_steps: 2000
  backbone_lr: 0.0001
  decoder_lr: 0.0002
  min_lr: 0.000001
  layer_decay: 0.85
  weight_decay: 0.05
  betas: [0.9, 0.95]
  microbatch_per_rank: 5
  effective_batch: 20
  accumulation_steps: auto
  precision: bf16_if_supported
  clip_grad_norm: 1.0
  checkpoint_every: 1000
  validation_every: 1000
  selection: fixed_final_step
  max_consecutive_invalid_batches: 50
augmentation:
  spatial: resize_only
  flip: false
  temporal_reverse: false
  speed_warp: false
  color_jitter: [0.2, 0.2, 0.2, 0.02]
  shared_across_frames: true
evaluation:
  protocol: common_v1
  fps: 30
  window_frames: 16
  window_stride: 2
  include_last_window: true
  short_clip_padding: repeat_last
  readout: fixed_ln_l2_latent
  video_pool: normalized_mean
  gallery: train_class_centroids
  recall_k: [1, 5, 10]
  seeds: [0, 1, 2]
  bootstrap_resamples: 2000
runtime:
  device: auto
  allow_network_during_training: false
  external_logging: false
  resume: null
~~~

### 15.2. Profiles/experiment overlays

Pilot inherits base. Full: source_hours_target=null (all acquired available source), max_steps500000,warmup50000, aux_ramp50000, forecaster_epochs60; same initial architecture. Actual available hours must be reported. Full does not auto-launch after pilot unless explicit --profile full.

~~~yaml
# configs/profiles/pilot.yaml
profile: pilot
~~~

~~~yaml
# configs/profiles/full.yaml
profile: full
data:
  source_hours_target: null
training:
  max_steps: 500000
  warmup_steps: 50000
loss:
  aux_ramp_steps: 50000
forecaster:
  epochs: 60
~~~

Smoke: mini Hiera (embed16,stages[1,1,1,1],num_heads1,input16×64×64, q_pool2,mask_unit[1,8,8]); mask grid8×2×2, keep6 per pass, output128; decoderdim64/layers1/heads4/FFN256; forecasterdim32/layers1/FFN128; synthetic state224,20steps,FP32,VCweight0, fixture gates. Shapes/coordinates derive dynamically but real canonical assertions remain. Smoke never tagged canonical pretrained.

~~~yaml
# configs/profiles/smoke.yaml
profile: smoke
data:
  source_dataset: synthetic_fixture
  source_hours_target: null
  num_workers: 0
encoder:
  name: mini_hiera_fixture
  initialization: random_fixture
  resolution: 64
  output_dim: 128
  keep_units_full: 6
  keep_units_past: 6
decoder:
  dim: 64
  layers: 1
  heads: 4
  ffn_dim: 256
forecaster:
  dim: 32
  encoder_layers: 1
  heads: 4
  ffn_dim: 128
  batch_size: 16
  epochs: 2
training:
  max_steps: 20
  warmup_steps: 2
  microbatch_per_rank: 2
  effective_batch: 4
  precision: fp32
  checkpoint_every: 10
  validation_every: 10
loss:
  aux_ramp_steps: 2
  vc_weight: 0.0
evaluation:
  readout: fixed_ln_l2_latent
runtime:
  device: cpu
~~~

Smoke fixture loader overrides real-asset/calibration/gate requirements through a separate synthetic backend tagged in provenance. Do not reuse real data_root: use data/smoke/<seed>/ and runs/smoke/<seed>/. Eval dimensions/resolution derive from export/profile, không force canonical768/224 on mini model.

Experiment YAML chỉ override name/aux_mode/state_blocks; all budget/architecture/teacher fields inherited. Dense_geometry state_blocks=[geometry],aux_mode=none; dense_state aux_mode=none; other names map table §11.2.

~~~yaml
# configs/experiments/increment.yaml
experiment:
  name: increment
  aux_mode: increment
  state_blocks: [appearance, geometry]
~~~

~~~yaml
# configs/experiments/dense_state.yaml
experiment:
  name: dense_state
  aux_mode: none
  state_blocks: [appearance, geometry]
~~~

Other overlays generated from table §13, not separate hand-copied trainers. All commands load same core code.

No unmentioned defaults hidden in Python: all important settings serialized.

## 16. Tests bắt buộc và nghiệm thu

Các tests dưới đây kiểm tra lỗi có thể đảo kết luận nghiên cứu hoặc làm pipeline không chạy. Không thay chúng bằng assertion copy implementation.

### 16.1. CPU unit suite

| Test | Setup | Pass condition |
|---|---|---|
| Timestamp causality | irregular PTS, repeated/missing frames | every selected PTS≤grid time; known indices đúng; no future interpolation |
| Window alignment | indexed RGB/states toy | past0..7, targets9/11/15, folds/source IDs đúng |
| Source grouping | same-video/signer/channel links | connected groups không qua partitions/folds |
| Benchmark exclusion | URL variants/reupload exact hashes | same original IDs excluded, unrelated IDs retained |
| Hand association | crossed wrists/mirrored/one-hand | no x-order swap; ambiguous assignment invalid |
| Geometry | translated/scaled synthetic landmarks | local handshape stable; torso location retained; invalid references masked |
| Face state | sustained coefficient, absent matrix | sustained signal retained; missing rotation masked; no clip-centering |
| DINO token layout | actual/mocked configs with regs | CLS/register excluded; patch count dynamically correct |
| PCA/normalization | disjoint calibration/test sentinels | calibration-only fit, stable basis/hash, invalids don't bias mean |
| Block loss | 1 valid coordinate + padded dims | correct average; padded/missing dims have zero gradient |
| Empty loss | all invalid mask | finite graph-connected zero, backward succeeds |
| Mask count | random seeds | equal keep count; only eligible prefix units |
| Token coordinates | token-ID unroll/reroll | stage MU order/coords verified; all-True maps match raster reference |
| Forecaster information | future states corrupted | predictions/targets unchanged from past-only inputs |
| OOF provenance | own group deliberately in fit list | target generation raises explicit error |
| Aux modes | oracle G0/G1/Y values | increment vs residual vs direct targets distinct and correct |
| Scale floor | zero increments | finite non-amplified loss; gates turn off weak edge |
| Fixed readout | zero/nearly constant latent | no NaN, expected LN/L2 behavior |
| Retrieval | known embeddings/classes/missing class | exact ranks/MRR/PI/PC and coverage policy |
| Export | mini encoder save/load | numerical parity and teacher-free inference |
| Resume | interrupted deterministic mini run | matches uninterrupted weights/counters/sampler |

### 16.2. Future leakage tests

Create two clips identical first8 frames, random different future8. Disable dropout, fixed masks:

1. Prefix RGB tensors after masking identical.
2. Prefix stage maps, raw latents, aux predictions identical within FP32 tolerance1e-6.
3. Gradients of aux prediction sum w.r.t. input frames8..15 exactly0.
4. Forecaster targets G1−G0 unchanged when only future teacher values change; cached gate/scale constants held fixed.
5. Re-extract first8 teacher frames from full video vs truncated video; state/crop outputs identical with fixed calibration.

Full state reconstruction được phép nhìn future. Test isolates prefix branch, không sai lầm đòi toàn bộ total loss independent of future.

Temporarily introduce bug “drop future tokens only after conv” trong dedicated regression fixture và bảo đảm leakage detector có thể bắt bug với overlapping kernel. Không mutate production implementation để test.

### 16.3. Synthetic predictive-dependence fixture

Build disjoint 16-frame samples, own/peer histories independent, future b:

~~~text
Y_b(9)  = 0.6*b(7) + 0.8*a(7) + small_noise
Y_b(11) = 0.6*b(7) + 0.6*a(7) + small_noise
Y_b(15) = 0.6*b(7) + 0.4*a(7) + small_noise
~~~

Oracle/ridge test phải chứng minh joint predictor có gain, shuffled predictor mất gain, increment khác noisy residual. Đây là functional ground truth, không dữ liệu sign language.

Neural smoke forecasters fit same finite fixture; verify optimizer updates/loss decrease/nonzero useful increments/shape/provenance. Không dùng stochastic hard accuracy target brittle để che engineering bug; log deterministic seed và convergence trace.

Generate synthetic RGB clip thể hiện state qua moving patches chỉ cho integration. Source labels không tồn tại. Fixture gates phải tagged synthetic, không bootstrap CI giả với insufficient groups.

### 16.4. Real-weight integration

1. Load official Hiera K400, inspect every key/shape and forward [1,3,16,224,224].
2. Load SignRep release full wrapper; same clip compare upstream latent/features vs adapter, FP32 error≤1e-5.
3. Teacher real crop output dimensions/finiteness/register exclusion.
4. One authorized real video decode→landmarks→fixed states→student forward/backward.
5. Train mini forecaster stage real source subset khi đủ data; frozen provider and real gates logs.
6. Student training10 optimizer steps, stop/resume10, validation/extraction/export.

Nếu actual data không đủ gate requirements, step5 báo no reliable edges; không bypass. Tests oracle vẫn chứng minh code path increment hoạt động.

### 16.5. DDP integration

CPU gloo2 processes mini model:

- Unequal valid masks across ranks → loss/gradient match single-process concatenated batch.
- Differentiable gathered VC gradient match single-process VC.
- All-invalid local rank không hang.
- Grad accumulation/update count correct.
- Save rank RNG/cursor and resume.

CUDA2 GPUs khi có: short torchrun smoke, finite loss, no unused-param errors, profiler GPU-hours logged. Nếu chỉ1GPU, CPU DDP coverage + singleGPU validated, CUDA multiGPU marked NOT_EXECUTED.

### 16.6. Acceptance thresholds

- All required CPU tests pass.
- No TODO/NotImplementedError in commands used by core pipeline.
- Real weight mismatch causes actionable failure, no silent fallback to random weights.
- Teacher/forecaster gradients absent; encoder gradient nonzero from state and increment tests.
- Prefix future-gradient0 and future-perturbation invariance.
- Checkpoint resume/export parity pass.
- Complete source/teacher/experiment provenance.
- Frozen retrieval + linear probe CLI runs end-to-end on fixtures and available benchmarks.
- No claim “better than SignRep” without comparable measured evidence.

## 17. Interfaces giữa modules

Define typed dataclasses/Pydantic records; assert shapes at module boundaries in debug mode.

### 17.1. Data APIs

~~~text
decode_video(path, timeline_spec) -> FrameStream
import_source(metadata_paths, videos_root) -> SourceManifest
import_benchmark(dataset_name, dataset_root, release_spec) -> BenchmarkManifest
split_source(manifest, group_rules, seed, fractions, folds) -> SplitManifest
build_clip_manifest(split_manifest, timeline_spec, window_spec) -> ClipManifest
SSLClipDataset.__getitem__(index) -> {
    "rgb", "states", "state_valid", "qualities",
    "frame_times", "clip_uid", "video_uid", "group_id", "fold_id"
}
~~~

ROI boxes available only in cache diagnostics/preprocessing, not student forward batch. Any Dataset return with captions/label_id in SSL path triggers validation error.

### 17.2. Teacher APIs

~~~text
LandmarkExtractor(frame_rgb_uint8) -> FrameDetections
associate_parts(detections, image_size) -> AssociatedParts
build_crops(frame, associated_parts, crop_spec) -> CropBatch
AppearanceTeacher(crops) -> raw_features
build_geometry(associated_parts, geometry_spec) -> (raw_geometry, validity)
Calibration.fit(calibration_stream) -> frozen_basis_and_stats
StateBuilder(frame_record, frozen_calibration) -> (y[4,224], m[4,224], q[4,2])
~~~

Landmark model files/options in extractor object; teacher model/config local. Functions do not invoke downloads implicitly.

### 17.3. Model/training APIs

~~~text
HieraVideoEncoder.encode(rgb, keep_mask) -> EncoderOutput
QueryDecoder(memory, coordinates, query_spec) -> predictions
Forecaster(past_y, past_m, past_q, edge_ids, horizon_ids, mode) -> predicted_y
CrossFittedTargetProvider(batch_metadata, past_states, shuffled_past_states, future_validity) -> AuxTargets
Student.forward_full(rgb, keep_full) -> FullOutputs
Student.forward_prefix(prefix_rgb, keep_past) -> PrefixOutputs
losses.compute(full_outputs, prefix_outputs, states, target_bundle) -> LossBundle
Trainer.fit(config, resume_path=None) -> RunSummary
~~~

TargetProvider does not receive RGB future. Raw residual/future-state control receives future target states through explicit separate loss-target argument, not history encoder.

### 17.4. Evaluation APIs

~~~text
extract_embeddings(model, benchmark_manifest, protocol) -> EmbeddingCache
evaluate_retrieval(cache, gallery_spec, metrics_spec) -> MetricsAndPredictions
train_linear_probe(cache, train_spec, selection_spec) -> ProbeRun
finetune_encoder(export_or_checkpoint, benchmark_manifest, config) -> RecognitionRun
bootstrap_metric_delta(predictions_A, predictions_B, groups, seed) -> ConfidenceInterval
export_encoder(checkpoint, destination) -> ExportBundle
~~~

Metric functions output predictions/sample IDs as well as aggregate values so statistics are reproducible.

## 18. CLI và commands agent phải implement

All subcommands: --help, config validation, deterministic logging, exit codes, useful error messages. Common flags --config/--profile/--experiment/--set key=value; CLI overrides type-checked. No hidden prompts in batch mode.

### 18.1. Bootstrap/data/cache

~~~bash
cip-sign doctor --config configs/base.yaml --profile pilot --stage environment
cip-sign assets bootstrap --config configs/base.yaml --allow-fallback
cip-sign assets verify --config configs/base.yaml

cip-sign data import-source --dataset youtube_sl_25 \
  --metadata-root data/source_metadata --video-root data/source_videos \
  --out data/manifests/source_raw.parquet
cip-sign data import-benchmark --dataset asl_citizen \
  --root data/benchmark_videos/asl_citizen
cip-sign data import-benchmark --dataset wlasl \
  --root data/benchmark_videos/wlasl --metadata third_party/WLASL/start_kit/WLASL_v0.3.json

cip-sign data decontaminate --config configs/base.yaml
cip-sign data split --config configs/base.yaml
cip-sign data build-clips --config configs/base.yaml
cip-sign data validate --config configs/base.yaml

cip-sign cache calibrate --config configs/base.yaml
cip-sign cache build --config configs/base.yaml --resume
cip-sign cache audit --config configs/base.yaml --sample-videos 100
~~~

WLASL metadata filename/path phải resolve từ actual pinned repo nếu đổi; command example không chứng minh file luôn tồn tại. importer error phải nêu paths available, không fabricate metadata.

assets bootstrap tải checkpoints/bundles. data acquire-source/data acquire-benchmark là separate commands:

~~~bash
cip-sign data acquire-source --config configs/base.yaml --hours 100 --resume
cip-sign data acquire-benchmark --dataset asl_citizen --resume
cip-sign data acquire-benchmark --dataset wlasl --resume
~~~

Acquisition prints expected disk/selected source URLs và progresses, obey bounded hours. Agreement/gated access conditions from §2.4 remain.

### 18.2. Forecasters/gates

~~~bash
cip-sign forecasters fit --config configs/base.yaml --folds all \
  --models g0,g1,gsh --resume
cip-sign forecasters evaluate --config configs/base.yaml --split gate_val
cip-sign forecasters build-gates --config configs/base.yaml
cip-sign forecasters audit-oof --config configs/base.yaml
~~~

Audit proves no group intersection for every clip↔forecaster pairing. evaluate writes per-sample errors with group IDs; build-gates creates gate JSON/NPZ plus human-readable table/reasons.

### 18.3. Training

~~~bash
cip-sign train ssl --config configs/base.yaml --profile pilot \
  --experiment dense_state --seed 0
cip-sign train ssl --config configs/base.yaml --profile pilot \
  --experiment increment --seed 0
cip-sign train ssl --config configs/base.yaml --profile pilot \
  --experiment increment --resume runs/increment_seed0/last.pt
~~~

DDP:

~~~bash
torchrun --standalone --nproc_per_node=4 -m cip_sign.cli.train ssl \
  --config configs/base.yaml --profile pilot --experiment increment --seed 0
~~~

Training CLI worker same logic as entry point, không recursive torchrun. Derive accumulation automatically from WORLD_SIZE; report actual batch sizes.

Pilot ablation runner:

~~~bash
for exp in dense_geometry dense_state future_state self_forecast joint_forecast residual increment shuffle_increment
do
  cip-sign train ssl --config configs/base.yaml --profile pilot \
    --experiment "$exp" --seed 0
done
~~~

Research runner repeats seeds0/1/2 with immutable configs, then evaluates same benchmark manifest/protocol. Run names include experiment/seed/config hash để không resume nhầm. Nếu operator intentionally specifies short run, reports marked pilot.

### 18.4. Eval/export

~~~bash
cip-sign evaluate extract --checkpoint runs/increment_seed0/last.pt \
  --config configs/base.yaml --dataset asl_citizen --splits train,val,test
cip-sign evaluate retrieval --run runs/increment_seed0 --dataset asl_citizen
cip-sign evaluate linear-probe --run runs/increment_seed0 --dataset asl_citizen
cip-sign evaluate finetune --run runs/increment_seed0 --dataset asl_citizen
cip-sign evaluate compare --runs runs/dense_state_seed0,runs/increment_seed0 \
  --dataset asl_citizen --bootstrap 2000
cip-sign export --checkpoint runs/increment_seed0/last.pt \
  --out runs/increment_seed0/export
~~~

Repeated extraction reuses matching model/protocol cache. Test extraction can happen before head selection because encoder frozen; test metrics must not feed selection.

### 18.5. Autonomous orchestration

~~~bash
cip-sign pipeline run --config configs/base.yaml --profile smoke
cip-sign pipeline run --config configs/base.yaml --profile pilot --resume
cip-sign pipeline run --config configs/base.yaml --profile full --resume
~~~

pipeline stages dependency graph:

~~~text
doctor(environment) → assets → doctor(assets) → acquisition/import → exclusion/splits/clips
→ calibration → teacher cache/audit
→ forecaster fitting/validation → gates/OOF audit
→ dense baseline training → CIP/controls training
→ frozen evaluation/probes → export/parity → report
~~~

Never launch full profile just because smoke passes. Pipeline explicit profile and budget govern execution; no infinite “continue improving” loop. If gate returns NO_VALID_EDGES, finish implementation/evaluation diagnostics and log negative pilot result.

## 19. Execution order cho AI Agent

### Phase A — code/environment

1. Inspect existing project/AGENTS instructions; create isolated branch/project without deleting user files.
2. Scaffold packaging/config/CLI, doctor/provenance.
3. Vendor Hiera, implement checkpoint adapters/fixed readout.
4. Unit-test masking/token geometry and checkpoint validation before any large data processing.
5. Implement source/benchmark manifests, decode/timeline/splits, dataset returns.

Deliverable: installable package, pinned source registry, runnable unit tests; no training claims.

### Phase B — teachers/cache

1. Landmarks/association/crops/geometry; synthetic geometry tests.
2. Appearance loader/token tests/fallback handling.
3. Calibration and chunk cache/audit/resume.
4. Small real-video inspection: overlay sampled crops/landmarks, validity table and manual visual QA.
5. Finish large source cache only after alignment/hand swap/face quality checks pass.

Deliverable: immutable calibrated teacher states, cache provenance and quality report.

### Phase C — forecasters

1. Implement identical G0/G1/Gsh, losses/shuffle rules.
2. Synthetic oracle + neural integration.
3. Group-disjoint fold fitting/selection.
4. gate_val errors/CI/scales/gates, OOF audit.
5. Freeze/check no gradients.

Deliverable: targets reproducible from cache+fold models; negative edges reported.

### Phase D — student/trainer

1. Dense query decoder/token coordinate adapter.
2. Two-pass student/losses and strict future leakage tests.
3. AMP/optimizer/accumulation/DDP/checkpoint/resume.
4. CPU smoke and available single/multi-GPU smoke.
5. B4 first, then C1 and mandatory aux controls under same settings.

Deliverable: real SSL training runner, backward/resume evidence, budget profile.

### Phase E — evaluation/export

1. B0/B2-ref and trained models common frozen extraction.
2. Retrieval + linear probe; fine-tuning separate.
3. Multi-seed comparisons when compute/data available.
4. Export teacher-free encoder, numerical parity.
5. Generate final implementation status and research report.

Nếu nguồn bị chặn, hoàn thành các phases độc lập và fixture validation; ghi exact blocker. Không thay public checkpoint bằng random weights mà giữ tên asset.

## 20. Reporting và artifacts nghiệm thu

### 20.1. Required outputs

| Artifact | Nội dung |
|---|---|
| README.md | Install, assets/access, local inference, train/eval/resume examples |
| IMPLEMENTATION_STATUS.md | Completion levels, tests executed/not executed, blockers |
| DECISIONS.md | Deviations from spec, reason, effect on fair comparison |
| assets/asset_lock.json | Real source URLs/revisions/hashes |
| reports/data_audit.json | Hours/groups/splits/missing/decontamination counts |
| reports/teacher_audit.json | Crop/geometry coverage, schema, sample QA |
| reports/forecast_metrics.parquet | G0/G1/Gsh/persistence errors per sample |
| reports/gates.json | Per-fold edge/horizon/block gates, CI/scales/reasons |
| reports/oof_audit.json | Source-group exclusion proof |
| reports/tests.json | Tests/commands/status/versions |
| runs/.../resolved.yaml | Exact runtime configuration |
| runs/.../train_metrics.jsonl | Steps/losses/LRs/gradient norms/coverage/time |
| runs/.../last.pt | Resume checkpoint |
| runs/.../export/ | Encoder weights + config/preprocess/provenance |
| reports/evaluation.json | Protocol, sample counts, PI/PC/R@k/MRR |
| reports/comparisons.md | Measured deltas, CI/seed std, limitations |

### 20.2. Compute accounting

Report actual wall time/GPU-hours cho download/decode (CPU time separately), landmark extraction, appearance inference, calibration/cache, all fold forecasters, student runs và evaluation. Report model/decoder params, visible tokens/pass, number of passes, peak VRAM, cache size, actual examples/second.

External foundation pretraining compute không tự biết thì ghi “external pretrained resource, not measured”. Không gọi cached teachers miễn phí.

### 20.3. Status template

~~~text
Method/version:
Code commit:
Implementation status:
Runtime validation:
Research validation:
Teacher variant:
Student initialization:
Datasets / retained hours:
OOF grouping / folds:
Valid predictive edges:
Executed tests:
Not executed tests + reason:
Actual training budget:
Frozen evaluation protocol:
Measured results:
Source-only vs label-informed selection:
Deviations:
Remaining blockers:
Export location:
~~~

Missing measured result ghi NOT_MEASURED, không điền estimate như kết quả. Final report lead implementation outcome, then evidence/limits.

## 21. Troubleshooting và quyết định có thể tự giải

| Failure | Agent action |
|---|---|
| DINOv3 401/403 gated | Use already-approved credentials; otherwise declared DINOv2 fallback, recache all controls |
| SignRep SHA mismatch | Stop that asset loading, refresh official release metadata; no random substitute |
| K400 unexpected keys | Explicit allowlist MAE decoder/head extras; fail critical encoder mismatch |
| Package collision models | Namespaced vendor imports, rerun parity |
| Face/hand detection coverage thấp | Inspect overlays/association/frames; fix bug before scaling; report genuine missing detections |
| Hands swap at crossing | Wrist association/margin, no x sorting; invalid ambiguous frames |
| Future leak test fails | Trace raw zeroing→mask resize→conv→token coordinates; fix before any CIP training |
| No forecast gain | Report negative pilot; inspect teacher/persistence/capacity/context; no forced gate |
| No valid increments | Dense training remains; increment code tested on known-positive fixture |
| GPU OOM | Activation checkpointing, lower microbatch/adjust accumulation, explicit VC policy; same changes for controls |
| All-invalid cache | Fix extractor/schema/normalization; fail after configured invalid-batch budget |
| DDP hang | Equal collectives/order, rank-local empty loss handling, unused-head graph connection |
| Resume mismatch | Refuse incompatible fingerprints; use explicit warm-start rather than pretend resume |
| Missing WLASL videos | Report counts/coverage, preserve splits/classes; do not regenerate easier split |
| Benchmark access agreement | Implement adapter, continue available benchmarks; record access blocker |
| C1 gain disappears vs B6 | Lower novelty claim; prefer direct G1 if equally effective/simpler |
| Gains only in fine-tuning | Report adaptation result; no strong frozen-representation claim |

No unbounded hyperparameter search. Core guide defines a reviewable method; later objective/backbone changes get new experiment names/version notes.

## 22. Scientific boundaries của implementation

Ideal conditional expectation identity motivates increment:

$$
\mathbb E\|Y-\mathbb E[Y|A]\|^2-
\mathbb E\|Y-\mathbb E[Y|A,B]\|^2
=\mathbb E\|\mathbb E[Y|A,B]-\mathbb E[Y|A]\|^2.
$$

Đây là standard identity, không theorem mới; finite neural forecasters, weighted/missing observations và distribution shift không tự satisfy population-optimal assumptions.

The method's testable claim: conditional increment targets improve transferable frozen RGB representations beyond same-teacher dense state/future/direct-joint controls. Do not assert linguistic semantics from forecast gain alone.

Pose/face detectors có external annotated training. Setting chính xác là no sign-task annotations in SSL pretraining, không “toàn bộ hệ thống không có supervision”.

Teacher quality, source coverage, context/horizon length và held-out dependence quyết định usefulness. Novelty/performance must be reassessed against current literature before paper submission.

## 23. Tham khảo cơ chế và related implementation

Sources S01–S28 đủ cho core bootstrap. Các nguồn sau cung cấp motivation/alternatives, không dependencies bắt buộc:

- SHuBERT: https://arxiv.org/abs/2411.16765 — hand/face/pose masked modeling precedent.
- SignDino (Hu et al., 2026 preprint): https://arxiv.org/abs/2609.06296 — DINO feature + temporal-model precedent.
- V-JEPA 2.1: https://arxiv.org/abs/2603.14482 — dense latent supervision.
- Masked Video Distillation: https://arxiv.org/abs/2212.04500 — complementary teacher supervision.
- Neural Relational Inference: https://proceedings.mlr.press/v80/kipf18a.html — relational dynamics learning.
- Generalized Distillation: https://arxiv.org/abs/1511.03643 — privileged training information.
- VICReg: https://arxiv.org/abs/2105.04906 — variance/covariance mechanism.
- VideoMamba: https://arxiv.org/abs/2403.06977 — optional long-sequence backbone.

Phải cite original works khi viết paper/README research attribution. Implementation recipe ở đây là đề xuất riêng; không mô tả nó như code chính thức của các paper này.

## 24. Prompt khởi động agent có thể dùng nguyên văn

> Hãy implement CIP-Sign theo toàn bộ file này. Xây repository chạy được, clone/download assets chính thức trong phạm vi đã cho phép, pin provenance, hoàn thiện preprocessing/cross-fitted forecasters/gates/student trainer/DDP/resume/evaluation/export. Chạy các tests và smoke/pilot phù hợp tài nguyên. Đừng dừng ở plan, skeleton hoặc TODO. Dùng declared fallbacks khi nguồn cần access; hoàn thành phần độc lập và ghi blocker thật. Giữ strict past-only innovation và matched controls. Báo riêng implementation, runtime và research validation. Chỉ kết luận tốt hơn SignRep nếu có measured comparable results.
