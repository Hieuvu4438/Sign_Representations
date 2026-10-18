# ASM-JEPA trên ASL-Citizen: đặc tả triển khai end-to-end cho AI Coding Agent

**Phiên bản:** 1.0 · **Ngày kiểm tra nguồn:** 15/10/2026 · **Ngôn ngữ hướng dẫn:** tiếng Việt.

**Deliverable yêu cầu từ coding agent:** một repository PyTorch có thể chuẩn bị ASL-Citizen, trích/nhập pose, pretrain ASM-JEPA, chạy frozen dictionary retrieval, linear probe và ISLR fine-tuning; có config, tests, checkpoint và báo cáo kết quả tái lập được.

Tài liệu này là **implementation specification** của proposal ASM-JEPA, không phải code đã chạy hay recipe đã được xác nhận accuracy. Những CLI `python -m asm_jepa...` bên dưới là interface mà agent phải xây dựng. Các URL/repository upstream được ghi riêng. Không báo rằng các CLI đã tồn tại trước khi agent hiện thực chúng.

## 0. Đọc trước: nhiệm vụ, giả định và quy tắc thực thi

**Nhiệm vụ đầu tiên:** chỉ dùng **ASL-Citizen official train split** để SSL pretraining. Sau đó đánh giá trên official validation/test theo signer-independent split. Chưa dùng YT-ASL, SL-25, WLASL, gloss-supervised pretraining, text encoder, RGB distillation hay NTU initialization trong run mặc định.

Ba tầng phải triển khai:

1. **Foundation:** manifest, pose pipeline, schema, normalization, masks, encoder và evaluator.
2. **ASM-v1:** EMA masked latent prediction + state anchoring + relations + tùy chọn global consistency. Đây là pilot phải chạy đầu tiên.
3. **ASM-v2:** bổ sung geometric/timed log-signature tokens và dependency closure đầy đủ. Phải có code/config/test cho tầng này; chạy sau khi v1 và baseline ổn định.

V1 là phiên bản triển khai theo giai đoạn của proposal; v2 mới kiểm tra phần state–motion đầy đủ. Không gọi v1 là bản đã kiểm chứng lợi ích của signatures.

**Giả định phần cứng khởi đầu:** Linux, một GPU NVIDIA khoảng 24 GB VRAM, RAM khoảng 32–64 GB và SSD đủ chỗ cho ZIP + RGB giải nén + pose cache + checkpoints. Đây là profile thiết kế, không phải tài nguyên đã được người dùng xác nhận. Agent phải đo VRAM/throughput trước khi chạy dài; có profile 12–16 GB và multi-GPU ở mục 14.

**Trạng thái dữ liệu:** đã xác minh nguồn công khai và code upstream. Không có bằng chứng rằng raw ASL-Citizen hoặc pose cache của người dùng đã được tải trong workspace hiện tại. Agent kiểm tra đường dẫn được cấu hình; nếu đã có thì reuse, nếu chưa có thì dùng nguồn chính thức. Không tìm file qua các conversation/Library khác.

**Quy tắc cho agent:**

- Đọc `AGENTS.md` và cấu trúc repo hiện tại trước khi sửa; tích hợp vào repo có sẵn nếu phù hợp.
- Làm việc đến một pipeline chạy được; không dừng ở việc tạo các class rỗng hoặc YAML không được tiêu thụ.
- Không tự đổi dataset split, số class, input joints hoặc metric để nâng kết quả.
- Không thêm nhãn vào SSL loss. Không ghép positives từ hai video khác nhau dựa trên gloss trong run SSL.
- Không fabricate source links, pretrained checkpoints, performance hoặc GPU-days.
- Nếu resource thật sự thiếu, ghi exact blocker cùng lệnh đã thử và tiếp tục những phần độc lập.
- Các test tính hợp lệ về split, masking, EMA và metric là bắt buộc; không cần test mọi getter/setter.
- Giữ một `IMPLEMENTATION_STATUS.md`: done / tested / not run / blocked, commit và config liên quan.

## 1. Định nghĩa protocol trước khi viết training code

### 1.1. Split và vocabulary

ASL-Citizen v1.0 paper công bố: train **40.154 video / 35 signer**, validation **10.304 / 6**, test **32.941 / 11**, tổng **83.399**, **2.731 glosses**.[S1]

Đây là expected values của v1.0. Kiểm tra file thực tế; nếu phiên bản tải về khác thì xuất `dataset_diff.json`, giữ nguyên official splits và ghi phiên bản. Không hardcode expected count rồi âm thầm bỏ/thêm mẫu để khớp.

Vocab chỉ tạo từ train: `sorted(unique(gloss.strip()))`, ánh xạ `gloss -> class_id`. Reuse đúng mapping cho val/test. Không tự upper-case, thay dấu, gộp synonyms hoặc parse gloss từ filename. Unknown gloss ở val/test phải được báo lỗi kiểm tra protocol, không tự gán class mới.

Giữa các split phải có:

```text
train_video_ids ∩ val_video_ids = ∅
train_video_ids ∩ test_video_ids = ∅
val_video_ids ∩ test_video_ids = ∅
train_signers ∩ val_signers = ∅
train_signers ∩ test_signers = ∅
val_signers ∩ test_signers = ∅
```

### 1.2. Ba readouts chính

| Run/readout | Train gì? | Dữ liệu tối ưu trọng số | Query / gallery |
|---|---|---|---|
| `ssl_frozen_retrieval` | Không fit target metric head | SSL: train videos, không dùng gloss loss | Val hoặc test query; train gallery |
| `linear_probe` | Chỉ một linear classifier | Train embeddings + gloss | Val chọn head; test đánh giá classifier |
| `finetune_islr` | Encoder + classifier | Train poses + gloss | Val chọn model; test đánh giá classifier |

Thêm `supervised_scratch` cùng backbone để đo giá trị pretraining. `finetuned_embedding_retrieval` là metric phụ, phải ghi rõ encoder đã được học có nhãn.

**Model selection mặc định cho pilot:** chọn SSL checkpoint theo `val/frozen_gloss_max/r1`, tie-break DCG rồi checkpoint sớm hơn. Val labels được dùng để chọn model; ghi rõ `selection_uses_val_labels: true`. Điều này không thêm gloss vào SSL loss nhưng không được gọi toàn bộ pipeline là không dùng nhãn.

Không chạy test theo từng checkpoint. Khóa config sau validation; evaluate test cho các cấu hình cuối cùng/seeds đã định trước. Không quay lại chọn hyperparameter bằng test.

### 1.3. Claim được phép

Tên setting: **in-domain ASL-Citizen SSL with signer-independent evaluation**. Train corpus có cùng vocabulary với evaluation; không gọi là cross-corpus/unseen-class transfer hay tái lập trực tiếp setting external pretraining của SignRep.

## 2. Nguồn dữ liệu và code đã xác minh

### 2.1. Dữ liệu chính thức

- Project: <https://www.microsoft.com/en-us/research/project/asl-citizen/>
- ZIP mà trang project cung cấp: <https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip>
- Baseline repo: <https://github.com/microsoft/ASL-citizen-code>

Chỉ dùng link ZIP sau khi kiểm tra HTTP status/content type/dung lượng và hash sau tải. Không chép một checksum chưa được công bố. Đọc license đi kèm dữ liệu; license code MIT không thay thế license dataset.[S2]

Loader chính thức đọc CSV theo thứ tự: cột 0 `user`, cột 1 `filename`, cột 2 `gloss`; có header. Script upstream sử dụng `ASL_Citizen/splits/train.csv` và `splits/test.csv`.[S3] Agent đọc header thật bằng `csv.DictReader`, có adapter vị trí theo schema này khi cần và lưu mapping đã dùng. Tìm official validation CSV trong archive, không tạo random validation split.

### 2.2. Những gì nên kế thừa

Các commit sau đã được đọc từ GitHub API khi viết hướng dẫn. Pin commit, không phụ thuộc `main` trôi. Model weight URLs cần hash riêng sau tải.

| Repo / commit | Đường dẫn đã kiểm tra | Dùng cho | Không kế thừa mặc định |
|---|---|---|---|
| `microsoft/ASL-citizen-code` · `17f0148b00ef87a5957ec103da8d10bd8eb1fa23` | `I3D/aslcitizen_dataset.py`, `I3D/aslcitizen_testing.py`, `ST-GCN/asl_citizen_dataset_pose.py`, `ST-GCN/README.md` | CSV convention, ranking metrics, baseline reference | 27-joint input như representation đầy đủ cho method mới |
| `facebookresearch/jepa` · `51c59d518fc63c08464af6de585f78ac0c7ed4d5` | `app/vjepa/train.py`, `app/vjepa/utils.py`, `src/models/predictor.py`, `src/utils/schedulers.py` | EMA, predictor pattern, optimizer/checkpoint utilities | RGB patch embedding, video masks và pretrained RGB weights |
| `facebookresearch/ijepa` · `52c1ae95d05f743e000e8f10a1f3a79b10cff048` | `src/train.py`, `src/utils/schedulers.py` | Tham khảo masked teacher/context pattern | ImageNet pipeline |
| `Tau-J/rtmlib` · `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` | `rtmlib/tools/solution/wholebody.py`, `README.md` | Whole-body 133-point extraction | OpenPose conversion, 3D output |
| `sakura2233565548/MASA` · `53bbb3b77bab7e98d33d7fcbd230e3e98c832ed1` | Repo tree: `moco/GCN_Transformer_mask.py`, `pretrain.py`, `dataset.py` | Đối chiếu motion SSL baseline nếu cần | Default datasets, semantic positives hoặc phụ thuộc chưa rà soát |
| `ZechengLi19/Uni-Sign` · `eed438bcb49e30405cd6ccdfcccca330c134e830` | `demo/pose_extraction.py`, `docs/DATASET.md` | Tham khảo RTMPose extraction/data adapter | mT5, text supervision và translation stack |

**Khuyến nghị:** viết repo skeleton nhỏ bằng PyTorch, tham khảo/copy có chọn lọc các tiện ích cần thiết. JEPAs có license CC BY-NC 4.0 tại commit đã đọc; rtmlib Apache-2.0; ASL baseline code MIT. Giữ license/attribution trong `THIRD_PARTY.md` khi kế thừa. Không tự gán license cho upstream chưa có file license rõ.

Chưa xác minh được repository skeleton S-JEPA hoặc MS-MAE đủ dùng để pin trong tài liệu này. Đừng clone repo `gioannides/s-jepa`: đó là **speech S-JEPA**, khác bài skeleton. Dùng paper S-JEPA làm đặc tả phương pháp và Meta JEPA làm tham khảo hạ tầng.[S4][S5]

### 2.3. Pose cache: có gì và chưa có gì?

Nguồn MSKA đã kiểm tra phát hành HRNet keypoints cho PHOENIX-2014/2014T và CSL-Daily; Uni-Sign liệt kê cache cho các bộ khác nhưng không xác minh được ASL-Citizen tại các nguồn này.[S6]

Vì vậy đường chạy chuẩn là **RGB ASL-Citizen -> tự trích RTMW/COCO-133 -> cache chuẩn của dự án**. Nếu người dùng đưa cache sẵn có, viết `import_pose_cache` và kiểm tra joint order, dimension, coordinate system, confidence, timestamps và exact video IDs. Không giả định mọi file `.pkl` shape `[T,133,3]` là cùng schema.

## 3. Cấu trúc repository và interface bắt buộc

Dùng package importable `asm_jepa`; cấu trúc dưới đây là danh sách đường dẫn tương đối, không phải yêu cầu tạo thêm framework lớn.

| Nhóm | Files agent phải hiện thực |
|---|---|
| Packaging | `pyproject.toml`, `requirements-train.lock.txt`, `requirements-pose.lock.txt`, `README.md`, `THIRD_PARTY.md` |
| Config | `configs/data/aslc.yaml`, `configs/pose/rtmw133.yaml`, `configs/pretrain/{jepa,state,full}.yaml`, `configs/eval/{linear,finetune,retrieval}.yaml` |
| CLI | `asm_jepa/cli/{doctor,prepare_aslc,extract_pose,import_pose_cache,audit_pose,fit_stats,pretrain,extract_embeddings,retrieval,train_classifier,evaluate_classifier,report}.py` |
| Data | `data/{manifest,pose_store,schema,sampling,normalization,augment,dataset}.py` |
| Features | `features/{state,relations,logsig2,paths,dependencies,tokens}.py` |
| Model | `models/{tokenizer,encoder,predictor,pooling,heads,asm_jepa}.py` |
| Training | `training/{losses,ema,centers,optim,schedules,engine,checkpoint,distributed}.py` |
| Evaluation | `evaluation/{ranking,retrieval,classification,diagnostics,bootstrap}.py` |
| Tests | Split/vocab, padding, schema/aliases, counterfactual masking, signature, EMA/center, gradients, retrieval, resume |
| Outputs | `runs/<run_id>/{config_resolved.yaml,metrics.jsonl,checkpoints,embeddings,reports}` |

Mọi CLI hỗ trợ `--help`, nonzero exit khi lỗi, `--dry-run` khi hợp lý và paths từ config/arguments. Không hardcode `/home/...`, hidden credentials hoặc nguồn dataset riêng của tác giả.

## 4. Môi trường có thể tái lập

### 4.1. Training environment

Default compatibility profile, chưa được chạy trong hướng dẫn:

```bash
python3.10 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.5.2 PyYAML==6.0.2 einops==0.8.0 tqdm==4.67.1 tensorboard==2.18.0 pytest==8.3.4
python -m pip install -e .
python -m asm_jepa.cli.doctor --config configs/pretrain/state.yaml
```

Torch wheel là một profile CUDA cụ thể được official installation matrix hỗ trợ.[S7] Nếu GPU/driver không phù hợp, agent chọn wheel đúng từ nguồn PyTorch và lưu resolved lock; không hạ CUDA driver hoặc sửa toàn bộ dependency vô điều kiện. `doctor` phải in GPU, CUDA, BF16 support, torch/numpy versions, free disk, RAM và config paths; không in secrets.

Baseline dùng `torch.nn.TransformerEncoder`/SDPA; không cần FlashAttention, xFormers, CUDA custom ops, mmcv hay mmdet. `torch.compile` tắt để debug và resume đơn giản.

Package phải lazy-import pose dependencies trong extraction CLI; training/geometry/metrics không import `cv2` hoặc `rtmlib` ở package root. Đặt pose tools trong optional extra/lock riêng để training environment thực sự độc lập.

### 4.2. Pose environment riêng

Để tránh `onnxruntime` CPU/GPU và nhiều bản OpenCV xung đột:

```bash
python3.10 -m venv .venv-pose
source .venv-pose/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4 onnxruntime-gpu==1.20.1 opencv-python-headless==4.10.0.84 tqdm==4.67.1 PyYAML==6.0.2
python -m pip install --no-deps 'git+https://github.com/Tau-J/rtmlib.git@03a1693e59e4f7cd84582c0fb30459b3bf18ad42'
python -m pip install --no-deps -e .
```

`--no-deps` là chủ ý vì rtmlib upstream liệt kê cả CPU ONNXRuntime và nhiều OpenCV packages; không cần cài chồng chúng cho Wholebody API. Agent smoke-test import thực tế và bổ sung dependency thật sự thiếu vào lock. CUDA/cuDNN của ONNXRuntime phải hợp profile; import `torch` trước khi tạo ORT session nếu dùng các CUDA libraries từ PyTorch.[S8] Không chấp nhận silent CPU fallback khi đã yêu cầu CUDA: kiểm tra session providers và log provider thực tế.

CPU extraction là fallback hợp lệ nhưng phải báo rõ throughput. Không hứa thời gian trích pose từ số đo máy khác.

### 4.3. Tái lập nguồn

Agent có thể clone upstream vào `third_party/` để đọc. Checkout full SHA trong bảng mục 2, ghi `upstream_lock.json`. Không bắt buộc cài các repo lớn như dependency runtime. Giữ training package độc lập với working directory của upstream.

## 5. Chuẩn bị raw data và canonical manifest

### 5.1. Tải dữ liệu khi chưa có

```bash
mkdir -p data/downloads data/raw
curl -L --fail --retry 3 --continue-at - \
  'https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip' \
  --output data/downloads/ASL_Citizen.zip
sha256sum data/downloads/ASL_Citizen.zip > data/downloads/ASL_Citizen.zip.sha256
python -m zipfile -t data/downloads/ASL_Citizen.zip
python -m zipfile -e data/downloads/ASL_Citizen.zip data/raw
```

Tìm đúng root có `splits/` và videos từ archive; không giả định ZIP tạo đúng một folder level. Không tải lại khi local ZIP/manifest đã hợp lệ. Không tự rename video files để đơn giản hóa labels.

```bash
python -m asm_jepa.cli.prepare_aslc \
  --raw-root data/raw/ASL_Citizen \
  --output-root data/processed/aslc \
  --expected-version v1.0
```

Root trên là ví dụ sau giải nén, agent điều chỉnh bằng CLI nếu archive khác. Thao tác phải idempotent.

### 5.2. Manifest contracts

Mỗi official split có một `metadata_<split>.jsonl`:

```json
{"sample_id":"stable_id","official_split":"train","signer_id":"official_user_value","video_relpath":"videos/original_filename.mp4","gloss":"official_gloss","class_id":0,"pose_relpath":"poses/stable_id.npz","duration_sec":2.5,"decode_status":"ok"}
```

Ví dụ chỉ minh họa schema; không coi `stable_id`, gloss hoặc duration trên là dữ liệu thật. `sample_id = sha256(official relative filename)[:24]` là một lựa chọn ổn định; kiểm tra collision. Filename/gloss chỉ là metadata, không vào model.

Sinh `ssl_train.jsonl` từ **official train** với `sample_id`, `pose_relpath`, sampling metadata; bỏ `gloss`, `class_id` khỏi batch SSL. Signer chỉ dùng audit/split/report, không dùng làm target.

Artifacts:

- `vocab.json`, `split_audit.json`, `dataset_version.json`.
- `metadata_train.jsonl`, `metadata_val.jsonl`, `metadata_test.jsonl`, `ssl_train.jsonl`.
- `unreadable_videos.jsonl`, manifest SHA256 và official CSV SHA256.

Chưa cần decode toàn bộ RGB trong `prepare_aslc`; metadata phase có thể đọc header nhanh. `extract_pose` hoàn tất decode checks. Missing video phải ghi exact ID. Benchmark evaluation vẫn giữ denominator official: xem fallback ở mục 13. Không âm thầm tạo dataset nhỏ hơn rồi báo tên full benchmark.

## 6. Pose extraction có provenance

### 6.1. Default extractor

Chọn `rtmlib.Wholebody(mode='performance', to_openpose=False, backend='onnxruntime', device='cuda')` tại commit đã pin. `Wholebody` trả raw `keypoints, scores`; phải assert `[N,133,2]` và `[N,133]`.[S9]

```python
import torch  # tải CUDA libraries trước ORT khi dùng profile tương ứng
from rtmlib import Wholebody

estimator = Wholebody(
    mode="performance",
    to_openpose=False,
    backend="onnxruntime",
    device="cuda",
)
keypoints, scores = estimator(frame_bgr)
```

Đây là upstream API đã kiểm tra; video reading/tracking/cache là code agent phải viết. Không gọi `format_result` để suy ra COCO indexing: helper formatting có quy ước OpenPose khác. Không bật `to_openpose=True`.

Model URLs trong `Wholebody.MODE['performance']` tại commit được đọc:

```yaml
detector_url: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip
detector_input_size: [640, 640]
pose_url: https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip
pose_input_size: [288, 384]
```

Agent tải/caches model bằng đường dẫn explicit, lưu SHA256 thực của ZIP/ONNX. Không đổi `performance` thành một model khác mà giữ nguyên run name. Nếu URL không còn tồn tại, ghi blocker và chọn model thay thế qua config/version mới.

### 6.2. Tracking và frame handling

- Decode toàn video ở native resolution và native frame timestamps; không crop thành hình vuông trước pose.
- Chọn signer nhất quán: frame đầu dùng người có box lớn phù hợp vùng trung tâm; các frame sau dùng box/keypoint continuity với người trước, ưu tiên shoulder consistency. Không dùng gloss/classifier để chọn người.
- Wholebody high-level trả keypoints/scores, không trả boxes. Nếu tracker cần boxes, gọi một lần `bboxes = estimator.det_model(frame_bgr)` rồi `keypoints, scores = estimator.pose_model(frame_bgr, bboxes=bboxes)` tại commit pin, giữ thứ tự instances. Không gọi thêm `estimator(frame_bgr)` trên cùng frame khiến detect hai lần. Zero detections phải tạo missing frame, không lấy người từ frame khác làm observation thật.
- Khi nhiều người và tracking ambiguous, ghi flags; không nhảy sang người khác chỉ vì confidence cao hơn.
- Dùng detection mỗi frame trong default pilot để đơn giản hóa validation. Tối ưu detection interval là thí nghiệm throughput sau này.
- Không smooth/interpolate toàn clip trong cache mặc định. Missing observations vẫn missing; mask/target validity xử lý phía sau.
- Ưu tiên decoder có PTS như PyAV/ffprobe nếu VFR; nếu dùng OpenCV thì kiểm tra `CAP_PROP_POS_MSEC`, tính đơn điệu và ghi fallback `frame_index / fps` cho CFR. Không tạo timestamps giả cho VFR mà không đánh dấu.

### 6.3. Cache contract

Mỗi clip `.npz`, `allow_pickle=False`:

| Key | Dtype/shape | Ý nghĩa |
|---|---|---|
| `xy` | float32 `[T,133,2]` | Pixel coordinates trong frame gốc |
| `score` | float32 `[T,133]` | Confidence nguyên bản từ extractor; không coi là calibrated probability |
| `observed` | bool `[T,133]` | Có tọa độ finite và đạt threshold cấu hình |
| `timestamps_sec` | float64 `[T]` | PTS thực/fallback được ghi |
| `frame_valid` | bool `[T]` | Decode và signer tracking hợp lệ |
| `image_wh` | int32 `[T,2]` hoặc `[2]` | Width, height gốc |
| `source_frame_index` | int32 `[T]` | Truy vết frame |

Metadata JSON sidecar hoặc index: schema/version, extractor SHA, model hashes, threshold, coordinate convention, timestamp mode, failed frames và source video hash/size. Ghi file tạm rồi atomic rename; hỗ trợ resume từ IDs đã hoàn thành.

Confidence default threshold `0.20`. Đây là giá trị khởi đầu, audit trước khi giữ; lưu raw scores để không phải extract lại. Pose import không có confidence dùng `confidence_mode: unavailable`, observed từ dữ liệu hợp lệ và weight 1 cho observations hợp lệ, không trình bày các số 1 đó là estimated confidence.

```bash
python -m asm_jepa.cli.extract_pose --config configs/pose/rtmw133.yaml --split train --limit 100
python -m asm_jepa.cli.audit_pose --processed-root data/processed/aslc --split train --sample-count 100 --render-overlays
python -m asm_jepa.cli.extract_pose --config configs/pose/rtmw133.yaml --split train --resume
python -m asm_jepa.cli.extract_pose --config configs/pose/rtmw133.yaml --split val --resume
```

Test pose có thể trích độc lập nhưng không fit statistics/tune quality threshold trên test. Mặc định chỉ xử lý test sau khi khóa pipeline.

### 6.4. Audit bắt buộc

Xem overlay cho ít nhất 100 train clips được lấy mẫu theo signer/duration/quality; ghi rõ đây là bước agent phải làm. Kiểm tra LH/RH theo **giải phẫu người ký**, ngón tay, tay chạm nhau/che mặt, seated framing và low resolution. Xuất coverage per part, tracking switches, NaN/inf, timestamp gaps, bone-length outliers, missing videos. Không chỉ lấy clips confidence đẹp.

## 7. Canonical schema, sampling và normalization

### 7.1. Chốt chính xác joints để tránh alias leakage

COCO-WholeBody raw indexing: body 0–16, feet 17–22, face 23–90, LH 91–111, RH 112–132. Agent kiểm tra schema bằng overlay tại extractor pin; không suy đoán từ tên variable.

**Model default dùng 114 joints chia bốn streams:**

```python
BODY = [5, 6, 7, 8]        # shoulders, elbows; bỏ duplicated body wrists
LH = list(range(91, 112))  # 21, wrist là local index 0
RH = list(range(112, 133)) # 21, wrist là local index 0
FACE = list(range(23, 91)) # 68
PART_ORDER = ["body", "lh", "rh", "face"]
```

Đây là lựa chọn implementation cụ thể hóa proposal 121-point upper-body candidate: bỏ body nose/eyes/ears và body wrists để giảm alias với face/hands. Raw cache vẫn giữ đủ 133 points. Không coi số 114 là topology của paper khác.

Trong face-68 convention, local nose-tip index 30 dùng cho head anchor; mouth 48–67, brows 17–26. Verify mapping trực quan. Nếu cache dùng format khác, adapter phải map semantic landmarks trước, không dùng cùng indices.

### 7.2. Full-clip sampling

Chốt input `max_frames: 64`, patch length `P: 4`. Test 128 frames là ablation sau này.

- Nếu `T_native > 64`, chọn 64 frames theo các bins thời gian đều từ đầu đến cuối, giữ endpoints. Eval dùng deterministic linspace indices. Train có thể jitter trong bin tối đa 15%, giữ thứ tự và không lặp index.
- Nếu `T_native <= 64`, giữ toàn bộ frames và right-pad; `time_valid=False` cho padding. Không coi frame lặp ở padding là observations mới.
- Không crop một đoạn giữa video làm input mặc định cho isolated signs.
- Giữ **timestamps gốc của frames được chọn**. Full-clip resampling thành 64 samples không đồng nghĩa clip có fps 25 hay duration 2,56 giây.
- Đảm bảo `num_frames` chia hết cho 4 sau padding. Một patch hợp lệ có ít nhất 1 real frame và part coverage đạt ngưỡng; loss vẫn xét observed coordinates riêng.

Nhánh path dùng khoảng thời gian thực và các selected observations; không dùng “8 frames = 0,32s” sau variable-duration sampling. Giá trị duration đến từ timestamp span, không từ `num_frames / target_fps`.

### 7.3. Coordinates và missing values

Tính root `r_t = (shoulder_L + shoulder_R) / 2` từ anchors quan sát được; scale `s_B` là median shoulder distance từ valid shoulder frames trong clip. Root missing có thể fallback về median root từ shoulder frames hợp lệ, và phải có `anchor_fallback` flag. Clip không có đủ shoulder evidence được đánh dấu `unusable_pose`, không normalize bằng hằng số tùy ý.

```text
xy_body(t,j) = (xy_pixel(t,j) - root(t)) / s_B
```

Dùng pixel coordinates và một isotropic scale, tránh chia x cho width/y cho height rồi tính khoảng cách bị biến dạng aspect ratio. Không fit scale từ bounding box của cả tay/mặt nếu các vùng đó được mask.

Với từng hand/frame: `s_H(t) = median(||MCP_i - wrist||, i in {5,9,13,17})` trên các MCP hợp lệ, có ít nhất 2 distances. Nếu không đủ thì hand-local channels tại frame đó không hợp lệ. `xy_local = (xy - wrist)/s_H(t)`. **Không lấy median hand scale trên toàn clip:** nó sẽ đưa hidden hand information sang các visible patches khác. Không xoay bàn tay về hướng chuẩn; orientation còn trong local xy.

Mỗi joint có 7 channels:

```text
[body_x, body_y, hand_local_x, hand_local_y, confidence, observed, local_valid]
```

Body/face có hai hand-local channels và local_valid bằng 0. Missing coordinates được zero-fill **sau normalization** và đi kèm flag 0; confidence không được dùng thay cho observed. local_valid phân biệt local coordinates chưa tính được vì thiếu wrist/MCP với một tọa độ thật bằng 0. Confidence và các flags không z-score. Clip coordinates normalized ra ngoài [-4,4] và hand-local ngoài [-4,4] tại input được ghi saturation rate; không sửa raw cache.

Ở baseline, shoulders là anchors **luôn được phép quan sát**. Mask body stream chỉ che elbow observations; không che shoulder anchors rồi vẫn dùng scale/root thật của chúng. Các masks che anchors thuộc một protocol tương lai cần normalization từ visible anchors.

### 7.4. Fit statistics

`fit_stats` chỉ đọc official train. Tính mean/std cho các state descriptor components, relation components và path components từ observed values; std floor `1e-3`, normalized descriptor clipping ±5. Giữ các component vốn hằng bằng zero sau standardization. Lưu counts/means/std/hash manifest/config.

Stats tính trước train là fixed constants; teacher và online dùng cùng stats. Không tính lại mean/std từ hidden values của current clip trong một phép normalization không được khai báo dependency. No BatchNorm trên joint/time dimension trong tokenizer mặc định.

## 8. Tokenizer và architecture contracts

### 8.1. State tokens

Input mỗi part `[B,64,J_a,7]` được patchify thành `[B,16,4*J_a*7]`; linear projection -> GELU -> LayerNorm -> 256 channels. Hai hands **share tokenizer weights**; body và face có weights riêng. Thêm learned part embedding. Không flip tay trong default.

Token order deterministic: part-major theo `PART_ORDER`, rồi patch index. State tokens tối đa `4*16 = 64`. Token validity dựa vào real frames/observed part coverage; default threshold `0.20`.

Time embedding dùng sinusoidal/Fourier encoding của normalized patch-center time và một duration embedding từ log(1+actual clip duration). Không dùng learned table cố định chỉ phù hợp 64 frames; support 128-frame ablation. Times/interval metadata không chứa giá trị pose target.

### 8.2. Relation tokens — bật trong ASM-v1

Ba relations theo từng patch: LH-wrist − RH-wrist, LH-wrist − nose, RH-wrist − nose. Mỗi frame có `[dx,dy,Euclidean_norm,confidence_pair,observed_pair]`, normalize trong body frame. `confidence_pair = min(c_1,c_2)`. Flatten 4 frames -> Linear(20,256) + GELU + LN; shared relation tokenizer với relation-ID embedding.

Tối đa `3*16=48` relation tokens. Đủ temporal coverage thì valid; missing frame features bằng zero và flag đúng. Relative proximity chỉ là cue 2D, không gọi physical contact.

### 8.3. Motion tokens — ASM-v2

Ba path families:

```text
left_wrist_body_xy
right_wrist_body_xy
left_minus_right_wrist_xy
```

Mỗi family tính tại 4 center patches `[2,6,10,14]` khi có 16 patches, cho 3 interval widths `[0.32,0.64,1.28]` seconds, centered rồi clip vào valid duration. Với max_frames khác, centers là quantiles 1/8,3/8,5/8,7/8 trên valid patch timeline.

Hai views:

- Geometry: level-2 log-signature 2D, dimension **3**.
- Timing: level-2 log-signature `[x,y,t_normalized]`, dimension **6**, cộng `log1p(interval_duration_sec)` -> dimension **7**.

Max motion tokens `3 families * 4 centers * 3 scales * 2 views = 72`; v1 max112 tokens, v2 max184. Intervals không có ≥3 distinct valid timestamps hoặc thiếu required path landmarks tại bất kỳ selected sample nào thì invalid ở baseline; không nối qua gap bằng coordinates 0. Log valid-token coverage theo family/scale. Nới gap policy là ablation riêng.

Geometry/timing có projection riêng Linear(3,256)/Linear(7,256), cộng path-family, scale, view và time embeddings. **Không thay thế log-signature bằng mean velocity mà vẫn gọi config là full ASM.**

### 8.4. Level-2 log-signature không cần custom dependency

Triển khai `logsig2.py` bằng NumPy/PyTorch dưới `no_grad` trên raw descriptors; không yêu cầu gradient xuyên pose extraction. Không cần cài `iisignature` cho baseline level 2.

Với increments δ_i và prefix a_i = Σ_{j<i}δ_j:

```python
s1 = delta.sum(dim=0)
prefix = delta.cumsum(dim=0) - delta
s2 = einsum("ni,nj->ij", prefix, delta) + 0.5 * einsum("ni,nj->ij", delta, delta)
area = 0.5 * (s2 - s2.T)
feature = concat(s1, upper_triangle(area, diagonal=1))
```

Đây là convention bậc 2 trong tensor/Lie basis được ghi rõ; package khác có thể scale/order coefficients khác. Nếu thêm `iisignature`, so sánh basis bằng tests trước khi thay backend, pin version/hash.[S10] Level 3 chỉ là extension, không thuộc default.

### 8.5. Online encoder, teacher và predictor

| Component | Default |
|---|---|
| Encoder | Pre-LN Transformer, 6 blocks, d=256, 8 heads, FFN=1024, GELU |
| Dropout / attention dropout | 0 / 0 |
| DropPath | Linear schedule 0→0.10 trong online; teacher eval tắt |
| Final encoder norm | LayerNorm(256) |
| Predictor | 2 TransformerDecoder blocks, d=128, 4 heads, FFN=512 |
| Predictor context input | Linear(256,128) của visible online features |
| Predictor queries | Learned mask query + type/part/family/scale/time embeddings; không target descriptor values |
| Predictor output | Linear(128,256), thứ tự target IDs giữ ổn định |
| Teacher | Deep copy toàn tokenizer + encoder; EMA; no gradients; luôn eval mode |
| Attention | Non-causal trong isolated-sign pilot |

Teacher đọc đầy đủ **valid tokens** của clip và contextualize trước khi gather targets. Online context chỉ nhận safe visible tokens. Invalid/padded tokens không tham gia attention/pooling/loss. Đừng xử lý padding bằng cách chỉ zero coordinates rồi vẫn cho attention nhìn padding.

State/relation/motion features có physical dependencies; token projection có thể tính độc lập rồi gather nếu chứng minh locality. Mọi cross-token transformer/GCN/convolution chỉ chạy **sau** khi context đã loại unsafe tokens.

## 9. Masking và chống target leakage

### 9.1. Raw target masks

Default mask ratio 0,50 trên valid **state part-time units**, tách biệt tỷ lệ tokens bị loại sau closure. Chọn cả missing/invalid units làm targets là lỗi; chúng không cung cấp supervision.

Mixture theo sample:

- 50%: random part-time masks.
- 30%: contiguous blocks trong từng part, block length 2–4 patches, bổ sung random units để đạt ratio.
- 20%: temporal blocks đồng thời các parts, length 2–3 patches, rồi bổ sung để đạt ratio.

Giữ ít nhất 25% valid state patches quan sát được cho mỗi hand khi hand có evidence; không che toàn bộ một tay trên cả clip. Giữ ≥8 visible state tokens và ≥1 context token toàn mẫu. Thử lại sampling tối đa 20 lần rồi giảm ratio và log fallback; không loop vô hạn.

Với body unit, raw M chỉ gồm elbows vì shoulders là public normalization anchors. Với hand/face unit, M gồm toàn bộ landmarks của part tại real frames của patch. Finger-subset masking chưa thuộc v1 để tránh hidden-wrist/local-frame ambiguity.

### 9.2. Dependency closure

Mỗi token k có `raw_dependencies[k]` gồm raw frame/joint IDs cần để tính nó, kèm anchors/statistics policy. Quy tắc:

```text
context_safe[k] = token_valid[k] AND no_intersection(raw_dependencies[k], raw_mask)
target_ids = valid_tokens_removed_by_raw_mask_or_closure
```

Quan trọng: target_ids bao gồm các relation/motion tokens bị loại qua closure; loss cân bằng theo group, không để số motion tokens chi phối state objective.

| Token/operation | Required dependencies |
|---|---|
| Hand state patch | Hand joints trong patch + per-frame wrist/MCP normalization + public shoulder anchors |
| Body state patch | Shoulders/elbows trong patch + public anchors |
| Face state patch | Face joints trong patch + public anchors |
| Relation patch | Các wrist/nose landmarks trong patch + public anchors |
| Path interval | Landmarks của path ở toàn interval support + public anchors + timestamps |
| Future derivatives/smoothing | Mở rộng support theo actual stencil/receptive field |

Không giữ relation/motion token dùng hidden hand vì nghĩ đó là “modality khác”. Không compute clean GCN rồi mask vài output token. Không lấy hand scale từ toàn clip. Không expose target features qua quality gating, auxiliary skip connection hoặc CLS pooling trước mask.

### 9.3. Counterfactual test bắt buộc

Giữ fixed raw mask M, timestamps, observed flags và public anchors. Thay hidden xy bằng donor values trong M; toàn bộ **context token values, metadata và encoder input sequence** phải không đổi, tolerance `atol=1e-6, rtol=1e-5`. Teacher targets được phép thay đổi.

Test riêng LH, RH, face, body elbows, boundary patch và multi-scale intervals; test thêm hidden confidences nếu chúng có thể đi vào retained features. Signature computation/normalization không được đọc hidden values để ảnh hưởng retained context.

Mask policy mặc định random/blocks, không dựa trên hidden motion saliency. Motion-aware curriculum là extension; khi thêm phải đánh giá cả information qua mask pattern.

## 10. Losses được định nghĩa đủ để code

### 10.1. Centered teacher CE

Teacher feature z_k `[256]` từ final encoder LN; predictor output p_k `[256]`. Tính softmax/loss FP32 kể cả khi forward AMP:

```python
teacher_prob = softmax((z_k.detach().float() - center[group]) / teacher_temp, dim=-1)
student_logprob = log_softmax(p_k.float() / student_temp, dim=-1)
ce_k = -(teacher_prob * student_logprob).sum(dim=-1)
```

Groups: 4 state parts + 3 relation families + 1 geometry + 1 timing = tối đa9. Center `[9,256]`; groups chưa bật không update. Center update theo raw teacher z **trước subtraction**, aggregated sum/count trên accumulated microbatches và ranks; EMA beta=0.90, mỗi optimizer step một lần. Không average các means khi counts khác nhau. Center không có gradients.

Weight target k: detached `mean_confidence_observed * observed_fraction` trong support, clipped [0,1]. Invalid targets weight0. Confidence-unavailable cache dùng observed-fraction weighting và ghi rõ mode. Đừng cho model học weight để tự tắt loss.

Trong mỗi group: Σw_k CE_k / max(Σw_k,eps). State loss average trên state groups có targets; relation loss average trên relation groups; geometry/timing average theo family/scale có targets. Mix:

```text
L_JEPA = weighted_mean(
    state: 1.00,
    relation: 0.50,
    geometry: 0.25,
    timing: 0.25
)
```

Denominator là tổng weights của các families **đang bật và có valid targets**. Không chia cho inactive families; missing groups không làm loss tụt giả.

Student temperature0.10. Teacher temperature warm-up0.08→0.06 trong3000 optimizer steps đầu, sau đó0.06. Đây là adaptation recipe đề xuất, chưa được benchmark; S-JEPA là tham chiếu cho centered/sharpened CE.[S4]

### 10.2. State anchor auxiliary — bắt buộc ASM-v1/v2

Một intact **online** forward `u = encoder_online(all_valid_tokens)` được tái dùng cho anchor/global. Không lấy stop-gradient teacher output làm u: như vậy auxiliary không huấn luyện online encoder.

Mỗi state patch u_{a,p} dự đoán descriptors tại offsets `[0, P//2, P-1]`, tức `[0,2,3]` khi P=4:

| Part | Target tại mỗi offset | Tổng dimension |
|---|---|---:|
| LH/RH | 21*2 hand-local xy + 2 wrist body-relative xy | 3*44 =132 mỗi hand |
| Body | 4*2 body-relative xy |24|
| Face | Mouth48–67 + brows17–26, body-relative xy |3*30*2 =180|

Hand orientation còn trong local xy vì không rotate canonicalization. Linear heads `256 -> target_dim`, hands có thể share head. Standardize bằng train-only stats. Weighted SmoothL1/Huber `beta=0.10`, mean trên observed coordinate components, rồi mean parts. Target/frame missing hoặc padded không vào denominator.

Targets lấy từ cùng shared geometric view của intact online branch; không fit tới một hệ tọa độ khác mà quên transformation. `lambda_anchor=1.0` ban đầu; grid hạn chế `[0.1,1.0]` bằng val nếu thật sự cần. Không tuyên bố weight này tối ưu trước thí nghiệm.

### 10.3. Global consistency — có code, ablation rõ

Pooling encoder intact theo mục11 cho u_online và teacher. Head G: Linear(1024,512) -> GELU -> Linear(512,256), không BatchNorm; teacher G là EMA copy.

```text
L_global = mean_squared_error(G_online(e_online), stopgrad(G_teacher(e_teacher)))
           + mean(relu(1 - std(G_online(e_online), correction=0)))
           + 0.04 * sum(offdiag(cov)^2) / 256
```

Teacher và intact online views cùng exact clip times; chia sẻ rotation/scale transforms để state targets thẳng hàng, student có small observation jitter. Không ghép random clips cùng video/signer như positives.

Variance/covariance dùng unnormalized G outputs, chỉ valid samples. Statistical batch là **physical microbatch × world_size**, không phải effective batch do gradient accumulation. Multi-GPU dùng autograd-compatible gather cho online statistics; single GPU tính local. Nếu statistical batch <16, default tắt global term và ghi config khác; không giả vờ tích lũy gradients tạo ra batch statistics lớn.

`lambda_global=0` ở first state pilot; `0.1` ở full config và ablation bật global. Teacher projection head chỉ tồn tại/update khi global bật. State anchor vẫn hoạt động khi global tắt.

### 10.4. Total loss và augmentation

```text
L_total = L_JEPA + lambda_anchor * L_anchor + lambda_global * L_global
```

Shared view: whole-clip translation/scale normalization, optional rotation ±5°, scale0.9–1.1. Student-only joint jitter sigma0.003 shoulder units, clipped3σ, chỉ observed joints. Anchor targets khớp shared view; không jitter missing values thành observed. Default không mirror, reverse, shuffle, random sign cropping, extreme perspective hoặc 3D rotation.

Nếu dùng extra nuisance views cho global sau này, phải giữ exact temporal correspondence và recompute anchor target theo view của online branch. Core configuration phải chạy được với shared view để tránh mâu thuẫn objective ngay ở pilot.

## 11. Embedding/readout cố định

Chỉ pool **state tokens** sau intact encoder; relation/motion tokens hỗ trợ contextualization nhưng không cộng trực tiếp làm tăng trọng số của một family vì nhiều tokens hơn.

Với từng part: valid-time mean của contextual state features `[256]`; L2-normalize mỗi part; absent part trả zero vector. Concatenate `[B,LH,RH,F]` -> `[1024]`, rồi L2 normalize toàn vector. Feature key: **`state_part_concat1024_l2`**.

Không học pooling trên target labels cho frozen retrieval. Export thêm `state_balanced_mean256_l2` như diagnostic, nhưng không chọn feature key bằng test. All-zero/unusable-pose clip có embedding zero và flag; vẫn giữ trong denominator eval.

Teacher encoder là mặc định để frozen extraction/linear probe sau SSL; lưu online variant để diagnostic, không chọn online/teacher bằng test. Fine-tuning initialize từ teacher **weights của tokenizer/encoder**, bỏ predictor/center/auxiliary heads. Chọn `online` phải là run/config riêng.

Linear probe dùng đúng frozen1024 feature, chỉ `Linear(1024,2731)`. Fine-tune dùng encoder + `LayerNorm(1024) -> Dropout(0.2) -> Linear(1024,2731)`, all encoder params trainable, không masked input.

## 12. Training loop, optimizer và resume

### 12.1. Default schedule

| Thuộc tính | Default pilot/full |
|---|---|
| Successful optimizer steps |60.000|
| Physical batch/GPU |32 ở profile24GB|
| Gradient accumulation |4 ở1GPU -> effective128|
| Optimizer |AdamW, betas(0.9,0.95), eps1e-8|
| Peak LR / final LR |3e-4 /1e-6, không tự scale LR với batch|
| Warm-up |3.000 successful optimizer steps|
| Weight decay |0.05; không decay bias, LayerNorm, embeddings/query tokens|
| Grad clip |Global norm1.0 sau unscale|
| Precision |BF16 nếu được hỗ trợ; FP16+GradScaler fallback; losses/statistics FP32|
| EMA momentum |Cosine0.996→0.9999 trên60.000 steps|
| Checkpoint interval |1.000 steps + last + best_val|
| Frozen validation interval |5.000 steps; subset nhanh chỉ diagnostic|
| Seed |0 cho pilot,0/1/2 cho final comparisons|
| Dataloader |8workers, pin_memory=true, persistent_workers=true nếu được hỗ trợ|

60k*128/40.154 xấp xỉ191 lượt sample exposures, không chính xác là191epochs vì replacement/drop-last/invalid-pose policies. Report number of samples actually seen và optimizer steps. Không lẫn microbatch iteration với step của LR/EMA.

### 12.2. Pseudocode có thứ tự update rõ ràng

```python
teacher = deepcopy(online_tokenizer_encoder)
teacher.requires_grad_(False)
teacher.eval()

for step in range(resume_step, max_successful_steps):
    set_optimizer_lr(lr_schedule(step))
    optimizer.zero_grad(set_to_none=True)
    center_sums, center_counts = zeros_for_active_groups()

    for micro in range(accum_steps):
        batch = next_train_batch()  # chỉ official train; no labels in SSL batch
        shared_view, student_view = augment_same_interval(batch)
        mask = sample_raw_masks(shared_view, seeded_by_step_and_id=True)
        token_meta = build_metadata_and_dependencies(shared_view)
        context_ids, target_ids = dependency_closure(token_meta, mask)

        with no_grad():
            teacher.eval()
            z_teacher = teacher(clean_valid_tokens(shared_view))
            targets = gather(z_teacher, target_ids)
            accumulate_center_sums_counts(targets, confidence_weights)

        context = online(safe_context_tokens(student_view, context_ids))
        predictions = predictor(context, target_queries(token_meta, target_ids))
        loss_jepa = weighted_group_ce(predictions, targets, old_centers)

        if anchor_enabled or global_enabled:
            intact_online = online(all_valid_tokens(student_view))
        loss_anchor = state_anchor_loss(intact_online, aligned_state_descriptors)
        loss_global = global_loss_if_enabled(intact_online, z_teacher)
        loss = loss_jepa + lambda_anchor * loss_anchor + lambda_global * loss_global
        backward(loss / accum_steps)

    unscale_if_fp16()
    clip_grad_norm_(trainable_parameters, 1.0)
    did_update = optimizer_step_with_overflow_check()
    if did_update:
        update_teacher_ema_once(momentum_schedule(step))
        update_centers_once(all_reduce_sum(center_sums, center_counts))
        advance_successful_step()
    checkpoint_or_validate_when_due()
```

Pseudocode tên hàm là contracts, agent hiện thực đầy đủ. Chốt view alignment: nếu student-only jitter làm descriptor observations thay đổi, anchor targets có thể dùng clean shared-view descriptors để denoise; raw state target validity vẫn lấy từ observations gốc. Dùng một policy nhất quán `anchor_target: shared_clean_view`, không để target theo hệ tọa độ chưa transform. Small jitter chỉ là corruption, không đổi sampling times.

Teacher forward không tham gia autograd. EMA update toàn tokenizer/encoder và G head đang bật; predictor/anchor heads không nằm trong teacher encoder. `model.train()` có thể bật lại teacher training mode qua recursion: sau đó phải set `teacher.eval()`.

Mọi optimizer groups chỉ chứa trainable active modules. Khi disable feature/head, không để unused trainable parameters gây DDP errors. Nếu thiếu target một group trong batch thì loss zero có graph phù hợp, hoặc module được loại khỏi graph theo strategy DDP đã test.

GradScaler overflow: không tăng successful-step counter, không update EMA/center/schedule như một optimizer step thành công. Log overflows và consumed samples. Không lưu checkpoint ghi step sai.

### 12.3. Logging bắt buộc

Mỗi100 optimizer steps log loss theo family, anchor per part, global terms, LR, EMA, grad_norm, teacher entropy, target counts, raw mask ratio và closure-induced ratio, visible counts, fallback counts, embedding std/effective rank, samples/s và peak VRAM.

Tính effective rank từ singular/eigenvalue spectrum trên một batch validation cố định, có eps và clip numeric; dùng như diagnostic, không thay retrieval metric. Low pretext loss không đủ để tuyên bố representation tốt.

### 12.4. Checkpoint/resume contract

Lưu: online state_dict, teacher state_dict, predictor, anchor/global heads, centers, optimizer, scaler nếu có, successful step, LR/EMA schedules, RNG Python/NumPy/Torch/CUDA, sampler state/position, seed, resolved config, manifest/stat/extractor hashes, code commit và best validation metric.

Sampling/augmentation dùng seed xác định từ base_seed + successful step + microbatch position + sample_id, hoặc lưu generator states tương đương. Resume test cùng world_size/config phải tiếp tục đúng batch IDs và schedules. Đổi world_size/microbatch phải ghi resumed-with-changed-statistical-batch, không hứa bitwise equivalence.

Checkpoint atomic write; không ghi đè best model bằng một run khác. Nếu checksum manifest/stats không khớp checkpoint, fail-fast trừ khi user chọn explicit migration run.

## 13. Evaluation implementation

### 13.1. Frozen embeddings

Extraction deterministic: no augmentation, full-clip sampling, teacher eval, no masking; cùng checkpoint/config cho train gallery và queries. Save NPZ/NPY memmap float32 `[N,1024]`, sample IDs, class IDs và usability flags. Không lưu mỗi embedding bằng torch pickle cần thiết phải load GPU.

Metadata export có checkpoint SHA, preprocessing/stats hash, schema, feature key, sampler, split và number of fallbacks. Class labels chỉ vào evaluator, không vào feature encoder.

### 13.2. Dictionary retrieval: gloss-level max-instance

Gallery = official train; queries = official val hoặc test. Với mỗi gloss g:

```text
score(query,g) = max cosine(query_embedding, gallery_embeddings_with_gloss_g)
```

Rank tất cả2.731 glosses, không rank individual videos rồi coi duplicates là2.731 classes. Train gallery đã được SSL encoder nhìn trong pretraining: đây là declared in-domain protocol, không phải unseen-corpus setting.

Không materialize ma trận32941×40154 nếu gây tốn RAM/VRAM. Default query chunks256, gallery chunks8192; scores FP32. Duy trì `[query_chunk,2731]` class scores bằng segmented/scatter max, khởi tạo `-inf`, combine chunks bằng maximum. Gallery class counts được audit.

Tie-break deterministic class_id tăng dần. Không nhận y_true khi computing scores hoặc breaking ties. Với rank r bắt đầu1:

```text
R@k = mean(r <= k)               # k=1,5,10
MRR = mean(1 / r)
DCG = mean(1 / log2(r+1))
```

Lưu cả scale0..1 và bản percentage rõ tên keys; không nhân100 hai lần. DCG ở đây chỉ một relevant gloss/query; không đổi thành một định nghĩa NDCG/multi-positive khác mà không đổi protocol name.

**Classifier ranking khác embedding retrieval:** evaluator upstream xếp class scores từ classifier; có thể dùng chung rank→metrics nhưng không gọi đó là nearest-neighbour frozen retrieval.[S3]

### 13.3. Linear probe

Train embeddings frozen trên full train, no input augmentation; `Linear(1024,2731)` có bias. AdamW lr1e-3, weight_decay0, batch256,100epochs, warm-up5epochs, cosine final1e-5. CrossEntropy, label_smoothing0.0. Select checkpoint bằng val Top-1. Không backprop tới feature encoder; kiểm tra hash encoder trước/sau.

### 13.4. ISLR fine-tuning

Initialize tokenizer/encoder từ teacher. No masks; relations/path flags giữ đúng architecture của checkpoint. Head như mục11. Default:

```yaml
epochs: 100
batch_size_per_gpu: 32
accum_steps: 4
optimizer: adamw
encoder_lr: 5.0e-5
head_lr: 3.0e-4
weight_decay: 1.0e-3
warmup_epochs: 10
final_lr_ratio: 0.01
label_smoothing: 0.10
grad_clip_norm: 1.0
selection_metric: val/cls/top1_instance
```

Report Top-1/5 instance và class-macro accuracy, DCG, MRR từ class ranks; class-macro = mean accuracy trên từng class có queries. Thêm per-signer và quality-bin results. Full training set có imbalance thì baseline giữ random sample shuffle; class-balanced sampling là ablation, không mặc định đổi evaluation.

Supervised-scratch dùng cùng model input/head/augmentation/fine-tune schedule; không load SSL weights. Nếu scratch chưa converge sau100epochs, báo learning curves và matched-budget extended run, không cố giữ một baseline undertrained.

### 13.5. Few-shot và test failures

ASL-Citizen nhiều classes, ít train examples/class. Dùng `k=1,2,4 samples/class`, sample từ official train, seeds0/1/2; report actual label counts/percentage. Không gọi nominal1% nếu rule giữ ít nhất1/class làm budget thực lớn hơn1%.

Unreadable/no-usable-pose query: giữ sample ID và denominator. Frozen embedding zero tạo ranking deterministic; classifier fallback uniform logits có deterministic tie handling; report `failure_count`, `coverage`, full-set metrics và usable-only diagnostic. Không dùng majority-class oracle hoặc xóa sample để nâng điểm. Nếu pipeline mất quá nhiều clips, sửa extraction trước khi diễn giải model.

### 13.6. So sánh với literature

Các mốc để tham khảo, không làm target nghiệm thu và không phải cùng setting pretraining:

- Official ASL-Citizen paper: I3D63,10% và ST-GCN59,52% Top-1.[S1]
- MS-MAE: ASL-Citizen75,72% sau external YT-ASL pretraining.[S11]
- SignRep:81,37% trong bảng recognition, khác input/pretraining.[S12]

Bảng kết quả phải ghi `pretrain_source`, `modality`, `extra_labels`, `pose_extractor`, `split_version`, `readout`, `selection_policy` và number of seeds. Pilot train-only không được báo là tái lập trực tiếp các điểm số trên.

## 14. Config defaults, profiles và ablations

### 14.1. YAML merge contract

`load_config` hỗ trợ `inherits` tương đối với file hiện tại; merge recursive, CLI `--set key=value` lặp lại, parse value bằng YAML, reject unknown keys. Lưu resolved YAML không còn inherits. Mọi key trong config phải được tiêu thụ hoặc validation báo lỗi; không để agent tạo flags trang trí không có tác dụng.

Tạo `configs/pretrain/base.yaml`:

```yaml
experiment: aslc_jepa_base
seed: 0
run_dir: runs/aslc_jepa_base_s0

data:
  processed_root: data/processed/aslc
  ssl_manifest: ssl_train.jsonl
  gallery_split: train
  val_split: val
  schema: coco133_to_aslc114_v1
  input_channels: 7
  max_frames: 64
  patch_frames: 4
  sampling: full_clip_bins
  temporal_jitter_fraction: 0.15
  confidence_threshold: 0.20
  part_coverage_threshold: 0.20
  stats_path: stats/train_v1.npz
  num_workers: 8
  pin_memory: true

features:
  relations: false
  geometry_paths: false
  timing_paths: false
  path_families: [left_wrist, right_wrist, left_minus_right_wrist]
  interval_widths_sec: [0.32, 0.64, 1.28]
  centers_per_clip: 4
  logsignature_level: 2
  path_min_points: 3
  path_gap_policy: strict_all_observed

model:
  dim: 256
  depth: 6
  heads: 8
  ffn_dim: 1024
  dropout: 0.0
  attention_dropout: 0.0
  drop_path: 0.10
  share_hand_tokenizer: true
  predictor_dim: 128
  predictor_depth: 2
  predictor_heads: 4
  predictor_ffn_dim: 512
  feature_key: state_part_concat1024_l2

mask:
  ratio: 0.50
  mixture_random: 0.50
  mixture_part_blocks: 0.30
  mixture_temporal_blocks: 0.20
  block_min_patches: 2
  block_max_patches: 4
  hand_visible_fraction_min: 0.25
  min_visible_state_tokens: 8
  max_resample_attempts: 20
  dependency_closure: true
  shoulders_public: true

augment:
  rotation_degrees: 5.0
  scale_range: [0.9, 1.1]
  student_jitter_sigma: 0.003
  mirror_probability: 0.0
  time_reverse_probability: 0.0

loss:
  latent: centered_ce
  student_temperature: 0.10
  teacher_temperature_start: 0.08
  teacher_temperature_final: 0.06
  teacher_temperature_warmup_steps: 3000
  center_momentum: 0.90
  family_weights: {state: 1.0, relation: 0.5, geometry: 0.25, timing: 0.25}
  lambda_anchor: 0.0
  anchor_huber_beta: 0.10
  anchor_target: shared_clean_view
  lambda_global: 0.0
  global_std_target: 1.0
  global_cov_weight: 0.04

train:
  objective: jepa
  max_steps: 60000
  batch_size_per_gpu: 32
  accum_steps: 4
  optimizer: adamw
  lr: 3.0e-4
  min_lr: 1.0e-6
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.05
  warmup_steps: 3000
  grad_clip_norm: 1.0
  precision: bf16
  ema_start: 0.996
  ema_end: 0.9999
  ema_schedule: cosine
  compile: false
  deterministic: false
  checkpoint_every_steps: 1000
  log_every_steps: 100

eval:
  frozen_every_steps: 5000
  selection_metric: val/frozen_gloss_max/r1
  selection_uses_val_labels: true
  encoder: teacher
  query_chunk: 256
  gallery_chunk: 8192
  tie_break: class_id_ascending
  test_during_training: false
```

`configs/pretrain/state.yaml`:

```yaml
inherits: [base.yaml]
experiment: aslc_asm_state
run_dir: runs/aslc_asm_state_s0
features:
  relations: true
loss:
  lambda_anchor: 1.0
```

`configs/pretrain/full.yaml`:

```yaml
inherits: [state.yaml]
experiment: aslc_asm_full
run_dir: runs/aslc_asm_full_s0
features:
  geometry_paths: true
  timing_paths: true
loss:
  lambda_global: 0.1
```

Các YAML là starting defaults đề xuất. Agent không được ghi “paper official hyperparameters” cho các giá trị này.

### 14.2. Hardware profiles

| Profile | Physical batch/rank | World size | Accumulation | Effective batch |
|---|---:|---:|---:|---:|
|1×24GB default|32|1|4|128|
|1×12–16GB|16|1|8|128|
|1×48GB thử sau measurement|64|1|2|128|
|4GPU|32|4|1|128|

Nếu OOM, giảm physical batch và tăng accumulation; giữ tokenization, total updates, effective batch và LR trước. Global statistical batch sẽ thay đổi khi đổi physical batch, phải report; không thể coi mọi profile tương đương tuyệt đối.

DDP: DistributedSampler/step-aware batch sampler, all-reduce centers/counts, rank0 logging/checkpoint, gather embeddings theo sample ID để loại sampler padding duplicates. Không average rank metrics khi số query khác nhau; merge counts/ranks đúng denominator.

### 14.3. Run matrix theo ngân sách

| ID | Architecture/objective | Mục tiêu |
|---|---|---|
|R0|Supervised scratch, state+relations architecture|Baseline bắt buộc cho pilot|
|R1|Coordinate MAE, state+relations architecture|Đối chiếu reconstruction; implement và smoke-test|
|R2|JEPA base, state tokens; anchor/global off|Đối chiếu latent-only|
|R3|R2 + state anchor; relations vẫn off|Tách đóng góp state anchoring|
|R4|ASM-v1 state.yaml: anchor+relations|Pilot chính|
|R5|ASM-v2: anchor+relations+geometry+timing, global off|Kiểm tra motion target|
|R6|full.yaml: R5 + global consistency|Method đầy đủ|

Thứ tự chạy thực tế: smoke tất cả; chạyR0 vàR4 seed0 trước; nếu pipeline ổn mớiR2/R3/R5/R6. R1 có thể chạy khi cần phân xử reconstruction vs JEPA. Không tự khởi động7run60ksteps đồng thời khi chưa đo chi phí.

R1 coordinate target: reconstruct masked raw normalized coordinates của cả4frames trong mỗi state patch; hands gồm local21×2 và global wrist2, body4×2, face68×2, với observed-coordinate loss. Decoder/head riêng; không dùng EMA targets. Đừng gọi reconstruction của3offset anchor descriptor là full coordinate MAE. Input/tokenizer/evaluator giữ đúng architecture đối chứng và report extra pass/FLOPs khác nhau.

Để kết luận novelty, cần state-only vs state+anchor, velocity/raw temporal vs signatures, geometry-only vs timing-added, và matched-data/compute comparisons. Cùng optimizer steps không đồng nghĩa cùng compute khi có thêm tokens/intact online pass.

### 14.4. Pose và readout config files

`configs/pose/rtmw133.yaml` tối thiểu; model URLs theo mục6 hoặc exact local ONNX paths đã hash:

```yaml
data:
  raw_root: data/raw/ASL_Citizen
  processed_root: data/processed/aslc
pose:
  library: rtmlib
  revision: 03a1693e59e4f7cd84582c0fb30459b3bf18ad42
  solution: Wholebody
  mode: performance
  to_openpose: false
  backend: onnxruntime
  device: cuda
  require_cuda_provider: true
  model_cache: data/models/rtmw133
  native_resolution: true
  native_timestamps: true
  cfr_fallback_timestamps: true
  detection_interval_frames: 1
  select_one_signer: true
  confidence_threshold: 0.20
  interpolate: false
  smooth: false
  cache_format: npz
  overwrite_valid_cache: false
```

`configs/eval/linear.yaml`:

```yaml
mode: linear_probe
feature_key: state_part_concat1024_l2
feature_dim: 1024
num_classes_from_vocab: true
head: linear
data:
  processed_root: data/processed/aslc
train:
  optimizer: adamw
  lr: 1.0e-3
  weight_decay: 0.0
  batch_size: 256
  epochs: 100
  warmup_epochs: 5
  min_lr: 1.0e-5
  label_smoothing: 0.0
selection_metric: val/cls/top1_instance
test_during_training: false
```

`configs/eval/finetune.yaml`:

```yaml
mode: finetune_islr
model_config: ../pretrain/state.yaml
feature_key: state_part_concat1024_l2
num_classes_from_vocab: true
head: layernorm_dropout_linear
head_dropout: 0.20
data:
  processed_root: data/processed/aslc
  train_split: train
  val_split: val
train:
  optimizer: adamw
  encoder_lr: 5.0e-5
  head_lr: 3.0e-4
  weight_decay: 1.0e-3
  batch_size_per_gpu: 32
  accum_steps: 4
  epochs: 100
  warmup_epochs: 10
  final_lr_ratio: 0.01
  label_smoothing: 0.10
  grad_clip_norm: 1.0
  precision: bf16
selection_metric: val/cls/top1_instance
test_during_training: false
```

`model_config` là tham chiếu architecture/data normalization, không kích hoạt SSL loss trong classifier stage. CLI `--init` quyết định scratch hoặc SSL initialization. Với full checkpoint, tạo `finetune_full.yaml` đổi model_config sang full.yaml. Không tự drop unknown state_dict keys để load architecture khác.

`configs/eval/retrieval.yaml` lưu protocol `gloss_max`, teacher selection, feature key1024, chunks256/8192, tie policy, gallery=train, full-set denominator; evaluator cũng nhận explicit embeddings paths từ CLI như mục15. Tách schema validation theo CLI/stage: keys hợp lệ nhưng stage-inactive được nhận biết, unknown keys phải lỗi.

## 15. Lệnh chạy end-to-end và điều kiện qua từng bước

Mọi lệnh bên dưới là CLI contract của repo agent tạo; `--set` chỉ override keys đã có trong config. Cấu hình classifier/retrieval cần được agent tạo theo thông số mục13. Không có số liệu benchmark nào được sinh bằng cách chép các mốc literature vào output.

### BướcA — xác minh environment/data

```bash
python -m asm_jepa.cli.doctor --config configs/pretrain/state.yaml
python -m asm_jepa.cli.prepare_aslc --raw-root data/raw/ASL_Citizen --output-root data/processed/aslc --expected-version v1.0
```

Qua bước khi: official splits/vocab được audit, không signer overlap, raw paths resolve, class IDs dùng một mapping. Nếu raw root khác thì sửa argument, không hardcode lại trong source.

### BướcB — 100clips pose rồi mở rộng

```bash
python -m asm_jepa.cli.extract_pose --config configs/pose/rtmw133.yaml --split train --limit 100
python -m asm_jepa.cli.audit_pose --processed-root data/processed/aslc --split train --sample-count 100 --render-overlays
```

Qua bước khi:133raw joints đúngindex, LH/RH đúnggiải phẫu, source timestamps/observed flags hợp lệ, overlays đã xem, provider GPU được xác nhận hoặc fallback được công bố. Không đưa một cache 27joints vào adapter133.

Sau đó chạy extract train/val với `--resume` như mục6. Không fit stats bằng test.

### BướcC — fit feature stats và tests

```bash
python -m asm_jepa.cli.fit_stats --config configs/pretrain/full.yaml --split train --output data/processed/aslc/stats/train_v1.npz
python -m pytest -q tests/test_splits.py tests/test_sampling.py tests/test_mask_dependencies.py tests/test_logsig2.py tests/test_ema_and_centers.py tests/test_ranking.py
```

`fit_stats` tính descriptor families của full config để file stats dùng chung cho base/state/full. Không instantiate training loss hay yêu cầu stats file đã tồn tại khi chính nó đang fit. Input standardized stats phải chỉ từ train, deterministic sampler; log số observations/group. Extension level3/schema mới phải tạo stats version mới.

### BướcD — synthetic và real smoke tests

```bash
python -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --synthetic --run-dir runs/smoke_synthetic --set train.max_steps=20 --set train.warmup_steps=2 --set train.batch_size_per_gpu=4 --set train.accum_steps=1 --set data.num_workers=0 --set eval.frozen_every_steps=0 --set train.checkpoint_every_steps=10
python -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --run-dir runs/smoke_real --limit-train-samples 128 --set train.max_steps=100 --set train.warmup_steps=10 --set eval.frozen_every_steps=0 --set train.checkpoint_every_steps=50
```

Synthetic mode tạo in-memory133-point poses, times, missing cases và label-free SSL batch; không cần dữ liệu thật/stats đã fit, dùng synthetic fixture statistics. Run name và report phải ghi synthetic, không coi metrics synthetic là ASL-Citizen.

Qua bước khi: shapes đúng, loss/grad finite, teacher không có gradients, online/predictor/anchor thay đổi, teacher chỉ thay sau EMA, target masks không leak, save/resume works. `--limit-train-samples` lấy deterministic train subset và đổi smoke-run provenance; không đổi official val/test.

### BướcE — overfit sanity và throughput

```bash
python -m asm_jepa.cli.train_classifier --config configs/eval/finetune.yaml --init scratch --overfit-samples 16 --run-dir runs/sanity_overfit16 --max-steps 500
python -m asm_jepa.cli.pretrain --config configs/pretrain/full.yaml --run-dir runs/profile_full --profile-steps 1000 --set eval.frozen_every_steps=0
```

Overfit16: dùng16train clips có labels hợp lệ, tắt augmentation/dropout và kiểm tra model có thể ghi nhớ; nếu không thì debug loader/label/head/optimizer. Đây chỉ là correctness check. Profile full đo steady-state sau warm-up100steps, VRAM và preprocessing/feature overhead. Không lấy thời gian download/model warm-up để ước lượng training throughput.

`--profile-steps` thực hiện benchmark riêng, không coi1000steps đó là một checkpoint final; có resume nếu user chọn tiếp cùngrun. Không hứa accuracy từ throughput.

### BướcF — SSL pilot chính

```bash
python -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --run-dir runs/aslc_asm_state_s0
```

Automatic validation dùng train gallery/val queries, không test. Output checkpoints `last.pt`, `best_val.pt`, step checkpoints, logs và resolved config. Run tiếp từ checkpoint:

```bash
python -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --run-dir runs/aslc_asm_state_s0 --resume runs/aslc_asm_state_s0/checkpoints/last.pt
```

Không tự restart từ step0 nếu optimizer state load lỗi; sửa compatibility hoặc ghi rõ warm-start run mới.

### BướcG — frozen retrieval thủ công

```bash
python -m asm_jepa.cli.extract_embeddings --config configs/pretrain/state.yaml --checkpoint runs/aslc_asm_state_s0/checkpoints/best_val.pt --split train --output runs/aslc_asm_state_s0/embeddings/train.npz
python -m asm_jepa.cli.extract_embeddings --config configs/pretrain/state.yaml --checkpoint runs/aslc_asm_state_s0/checkpoints/best_val.pt --split val --output runs/aslc_asm_state_s0/embeddings/val.npz
python -m asm_jepa.cli.retrieval --gallery runs/aslc_asm_state_s0/embeddings/train.npz --queries runs/aslc_asm_state_s0/embeddings/val.npz --protocol gloss_max --query-chunk 256 --gallery-chunk 8192 --output runs/aslc_asm_state_s0/reports/frozen_val.json
```

Embedding files có labels trong evaluation metadata; encoder không được nhận labels. Evaluator phải kiểm tra cùng checkpoint/preprocessing/feature key giữa gallery và query.

### BướcH — probe và fine-tune

```bash
python -m asm_jepa.cli.train_classifier --config configs/eval/linear.yaml --train-embeddings runs/aslc_asm_state_s0/embeddings/train.npz --val-embeddings runs/aslc_asm_state_s0/embeddings/val.npz --run-dir runs/aslc_linear_state_s0
python -m asm_jepa.cli.train_classifier --config configs/eval/finetune.yaml --init runs/aslc_asm_state_s0/checkpoints/best_val.pt --run-dir runs/aslc_ft_state_s0
python -m asm_jepa.cli.train_classifier --config configs/eval/finetune.yaml --init scratch --run-dir runs/aslc_scratch_s0
```

`finetune.yaml` chứa architecture tương ứng state.yaml; khi init từ full checkpoint phải dùng full architecture config được checkpoint lưu, không bỏ motion modules im lặng. Nếu CLI config mâu thuẫn checkpoint thì fail-fast và báo keys khác nhau.

### BướcI — full method và ablations

```bash
python -m asm_jepa.cli.pretrain --config configs/pretrain/base.yaml --run-dir runs/aslc_jepa_base_s0
python -m asm_jepa.cli.pretrain --config configs/pretrain/base.yaml --run-dir runs/aslc_anchor_only_s0 --set loss.lambda_anchor=1.0
python -m asm_jepa.cli.pretrain --config configs/pretrain/full.yaml --run-dir runs/aslc_full_no_global_s0 --set loss.lambda_global=0.0
python -m asm_jepa.cli.pretrain --config configs/pretrain/full.yaml --run-dir runs/aslc_full_s0
```

Những lệnh này tuần tự theo kế hoạch sau validation; không bắt agent phải chạy mọi long run trước khi báo pilot đầu tiên. Seed1/2 cho các cấu hình được chốt, có cùng data manifest/stats.

### BướcJ — final test sau khi khóa config

```bash
python -m asm_jepa.cli.extract_pose --config configs/pose/rtmw133.yaml --split test --resume
python -m asm_jepa.cli.extract_embeddings --config configs/pretrain/state.yaml --checkpoint runs/aslc_asm_state_s0/checkpoints/best_val.pt --split test --output runs/aslc_asm_state_s0/embeddings/test.npz
python -m asm_jepa.cli.retrieval --gallery runs/aslc_asm_state_s0/embeddings/train.npz --queries runs/aslc_asm_state_s0/embeddings/test.npz --protocol gloss_max --output runs/aslc_asm_state_s0/reports/frozen_test.json
python -m asm_jepa.cli.evaluate_classifier --checkpoint runs/aslc_ft_state_s0/checkpoints/best_val.pt --split test --output runs/aslc_ft_state_s0/reports/test.json
python -m asm_jepa.cli.report --runs runs/aslc_asm_state_s0 runs/aslc_ft_state_s0 runs/aslc_scratch_s0 --output runs/final_report.md
```

Test labels chỉ đọc tại evaluator. Những config khác có output folder riêng. Nếu chưa đủ tài nguyên để test, report `not_run`, không thay bằng validation score.

## 16. Tests quyết định tính đúng của method

| Test | Assertion bắt buộc |
|---|---|
|Splits/vocab|Signer/video disjoint; one train-only mapping; unknown class bị phát hiện|
|Schema|133raw ->114model; hands21/21, face68, body4; không duplicated body wrist/face aliases|
|Sampling/padding|Giữ endpoints/thứ tự; timestamps thật; padded frames không vào attention/loss/mean|
|Missing hand|Local-valid0 khi thiếu wrist/MCP; zeros không trở thành observed|
|Normalization leakage|Hidden hand thay đổi không làm scale của visible hand patches thay đổi; public shoulder anchors fixed|
|Closure|LHmask loại LH-dependent relations, overlapping path windows; face mask loại hand–nose relations|
|Counterfactual|Retained context invariant khi hidden xy thay; teacher targets có thể đổi|
|Logsig2|Constant paths→zero positive levels; translation invariance; line out-and-back triệt tiêu; thêm time phân biệt ví dụ tổng hợp; basis/order đúng|
|Teacher gradients|`teacher.grad is None`; predictor/online/anchor grads xuất hiện khi enabled|
|EMA|Exact update vớiµ biết trước; không update mỗi microbatch; teacher eval|
|Centers|Count-weighted all-reduce; group không target không đổi; FP32; resume đúng|
|Loss masking|All-invalid coordinates contribution0, denominator không NaN; family averages không phụ thuộc số padded tokens|
|Retrieval|Toy gallery có multiple instances/gloss; max aggregation đúng; chunked=dense; ties deterministic không dùng y_true|
|Metrics|Ranks[1,2,5]→R@1=1/3,R@5=1,MRR=(1+1/2+1/5)/3,DCG đúng log2(r+1)|
|Frozen readout|Encoder hash trước/sau linear probe không đổi|
|Resume|Step20→save→resume cùng batch IDs/EMA/LR; loss trajectory trong tolerance của precision được khai báo|
|Failure accounting|Unusable query vẫn trong full denominator; usable-only được gắn tên diagnostic|

Code test nhỏ với fixtures đủ chứng minh correctness; không mirror từng implementation line. Một test phát hiện leakage phải có adversarial donor values đủ khác để không pass tình cờ.

Tiny-data training check là bổ sung, không là unit test bắt buộc chạy trên mọi PR. CI tối thiểu CPU cho geometry/masks/metrics; GPU smoke dành cho AMP/EMA training path.

## 17. Chẩn đoán trước khi tăng compute

| Hiện tượng | Điều tra đầu tiên | Cách xử lý có kiểm soát |
|---|---|---|
|Loss NaN|All-padded attention; scale0; invalid logits; FP16 overflow|Guard validity/eps; CEFP32; BF16; không thay NaN bằng0 vô điều kiện|
|Loss thấp, retrieval gần ngẫu nhiên|Collapse/channel domination; wrong gloss map; feature extractor mismatch|Entropy/std/effective rank; toy evaluator; overlay; thử anchor/variance term|
|Frozen yếu nhưng FT khá|Global metric chưa tốt; pooling/state nuisance|So balanced mean vs concat bằngval; global consistency ablation; không kết luận SSL vô ích chỉ từ một readout|
|FT kém scratch|LR quá lớn/nhỏ; teacher/online load nhầm; preprocessing drift|Check hashes; tune LR trong grid nhỏ; learning curves|
|Signature gain mất sau closure|Baseline có leakage hoặc context quá ít|Fix leakage; match effective visible ratio; không giữ leaked result làm headline|
|V2 chậm hơn nhiều|CPU logsignature per item/window; quá nhiều invalid tokens|Vectorized level2, cache deterministic eval features, profile token builder; không bỏ timestamp handling|
|Validation tăng, test drop|Signer/framing shift, selection overfit|Per-signer/quality report; seeds; không tune tiếp bằngtest|
|CUDA pose chạy nhưCPU|ORT provider load failed|Kiểm tra session providers/library versions; fail-fast hoặc explicit CPU run|

Không có absolute accuracy target làm điều kiện “code đúng”. Chấp nhận pilot khoa học khi pipeline/reproducibility hợp lệ, baselines hội tụ, kết quả/failure cases được báo trung thực. Nếu model chưa tốt, ghi nguyên nhân đã kiểm tra và thí nghiệm phân xử tiếp theo.

## 18. Definition of done và handoff

Coding agent phải giao:

1. Repo import/run được; một README quickstart và config schema.
2. Data/pose source provenance, exact official split/vocab/hash, cache schema và audit overlays/report.
3. Masking/normalization/signature/EMA/evaluator tests pass; ghi command và output thực.
4. Synthetic20steps, real100steps, overfit16 và throughput profile đã chạy, hoặc exact resource blocker.
5. Ít nhất một ASM-v1 SSL pilot + frozen validation, linear probe/FT và scratch khi training resources cho phép; không bỏ claim `not_run` nếu chưa chạy.
6. V2 code/config/tests đầy đủ; trạng thái long run riêng, không trộn vớiv1.
7. Best/last checkpoints với resume metadata; locked config; embeddings/retrieval outputs có hashes.
8. `results.csv`, `final_report.md`, `IMPLEMENTATION_STATUS.md`, `THIRD_PARTY.md`.

`results.csv` tối thiểu các cột:

```text
run_id,seed,method_version,pretrain_source,pretrain_labels,
split_version,pose_extractor,pose_model_hash,manifest_hash,
encoder_variant,feature_key,readout,selection_policy,
test_exposure,checkpoint_step,params_m,training_hours,
peak_vram_gb,frozen_r1,frozen_r5,frozen_r10,frozen_mrr,frozen_dcg,
cls_top1_instance,cls_top5_instance,cls_top1_class,coverage,status
```

Chưa chạy metric nào để null/empty và status rõ; không dùng0 khiến người đọc hiểu là kết quả đo. Không để external-pretraining literature scores ở cùng row với local run như thể đã tái lập.

Trong report, tách finding theo mức bằng chứng: verified correctness, measured validation/test, exploratory hypothesis và not-run. Báo compute extraction/backbone/end-to-end riêng.

### Prompt khởi động có thể đưa trực tiếp cho AI coding agent

> Hãy triển khai ASM-JEPA trên ASL-Citizen theo toàn bộ tài liệu này. Trước tiên đọc AGENTS.md/repo/environment và lập danh sách dữ liệu đã có. Hiện thực pipeline nhỏ chạy được theo thứ tự data→pose→schema→tokens/dependency masks→EMA JEPA/state anchor→retrieval→probe/fine-tune. Dùng cấu hình24GB làm giả định khởi đầu, đo lại bằngdoctor/profile. Giữ official splits, train-only SSL, no test tuning; pin nguồn và model hashes. Viết các tests có khả năng phát hiện leakage/EMA/metric sai. CLI trong tài liệu là contract cần xây dựng. Chạy synthetic/real smoke trước long run; hoàn thành ASM-v1 pilot khi tài nguyên cho phép, và implement ASM-v2 với level2 geometry/timing targets cùng ablations. Không giả lập kết quả và không tự nhận các defaults đề xuất là hyperparameters chính thức. Ghi mọi phần đã/chưa chạy và exact blocker; tiếp tục các phần độc lập đến khi handoff cụ thể, review được.

## 19. Nguồn tham khảo đã sử dụng

**[S1]** Desai et al., *ASL Citizen*, NeurIPS2023 Datasets and Benchmarks. [Paper](https://papers.nips.cc/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf). Counts/split và supervised baseline context.

**[S2]** [Microsoft ASL-Citizen project/download](https://www.microsoft.com/en-us/research/project/asl-citizen/). ZIP URL lấy từ project; HTTP HEAD kiểm tra ngày15/10/2026 trả200, content length45.924.134.223bytes. Chỉ kiểm tra header, không tải toàn bộ ZIP trong quá trình viết hướng dẫn.

**[S3]** [Official ASL-citizen-code](https://github.com/microsoft/ASL-citizen-code), commit trong bảng mục2. Đã đọc CSV loader, ST-GCN dataset và I3D ranking evaluator. Upstream code có cách sampling/input khác; implementation mới không tự gọi là exact reproduction.

**[S4]** Abdelfattah, Alahi, [S-JEPA, ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf); [project](https://sjepa.github.io/). Tham khảo latent prediction, EMA và centered CE; defaults trong tài liệu là đề xuất cho ASL-Citizen.

**[S5]** [V-JEPA official code](https://github.com/facebookresearch/jepa); [I-JEPA official code](https://github.com/facebookresearch/ijepa). Đã kiểm tra các đường dẫn training/predictor/scheduler và license ở commit pin.

**[S6]** [MSKA](https://github.com/sutwangyan/MSKA); [Uni-Sign dataset documentation](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/docs/DATASET.md). Dùng để xác minh phạm vi cache công khai, không suy ra đã có ASL-Citizen poses.

**[S7]** [PyTorch official previous-version install matrix](https://pytorch.org/get-started/previous-versions/). CUDA wheel phải phù hợp GPU/driver của máy triển khai.

**[S8]** [ONNXRuntime CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). Tài liệu dependency/provider, không phải benchmark throughput.

**[S9]** [rtmlib Wholebody source](https://github.com/Tau-J/rtmlib/blob/03a1693e59e4f7cd84582c0fb30459b3bf18ad42/rtmlib/tools/solution/wholebody.py); [README](https://github.com/Tau-J/rtmlib). API/mode URLs đã đọc; COCO133 raw arrays và OpenPose formatter phải phân biệt.

**[S10]** [iisignature](https://github.com/bottler/iisignature), optional reference; default level2 ở đây viết bằng tensor operations. [Signature primer](https://arxiv.org/pdf/1603.03788) cho invariance và path properties. Chưa kế thừa full Path-JEPA implementation; [publisher/project](https://singh17ashu.github.io/Path-JEPA/) chỉ là tham chiếu phương pháp ở giai đoạn này.

**[S11]** Mo, Vo, Nakayama, [MS-MAE, IJCNLP-AACL2025](https://aclanthology.org/2025.ijcnlp-long.66/). Dùng làm literature baseline; chưa xác minh repo chính thức để kế thừa trong guide.

**[S12]** Wong et al., [SignRep, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html); [code](https://github.com/ryanwongsa/SignRep). External RGB representation reference, không dùng weights trong default pilot.




