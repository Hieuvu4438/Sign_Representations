---
title: ASC-JEPA — End-to-End AI Agent Implementation Specification
spec_version: "1.0"
source_snapshot: "2026-10-06"
language: vi
method_status: proposed_and_unvalidated
document_status: implementation_contract_not_executed_training
primary_objective: frozen_sign_representation_and_label_efficiency
---

# ASC-JEPA: đặc tả implementation end-to-end cho AI Agent

## 0. Đọc trước khi bắt đầu

Bạn là AI Research Engineer thực hiện **ASC-JEPA — Articulatory State-and-Change JEPA**. Hãy xây dựng code chạy được, tái lập baseline và kiểm tra hypothesis; không chỉ tạo skeleton, không bịa API, checkpoint hay kết quả. Tài liệu này đứng độc lập, không yêu cầu đọc lịch sử hội thoại.

Method là một proposal: RGB video encoder học latent prediction, local articulatory state và temporal change; local supervision được điều tiết theo lượng context còn quan sát được. Pose/ROI chỉ tạo training targets, không đi vào predictor và không bắt buộc ở inference. Chưa có experiment chứng minh proposal vượt SignRep.

### 0.1 Quy ước bắt buộc

- **MUST:** contract khoa học hoặc kỹ thuật phải giữ.
- **SHOULD:** lựa chọn mặc định; thay đổi phải ghi Decision Record và áp dụng công bằng cho baseline liên quan.
- **PILOT:** hyperparameter/design đề xuất, chưa được validated.
- **VERIFIED:** thông tin đã đối chiếu nguồn công khai trong snapshot này; không có nghĩa code đã chạy local.
- **AUDIT:** Agent phải đọc source được clone/pin và tạo bằng chứng trước khi dùng.
- **ADAPTED:** component có tiền lệ nhưng được thay đổi cho Hiera/sign task; không gọi là faithful reproduction.
- **NEW-CANDIDATE:** phần của proposal cần kiểm chứng novelty và thực nghiệm; không claim “first”.

### 0.2 Research contracts

1. Core SSL không dùng gloss, captions, sentence embeddings, class labels hay pseudo-gloss làm target. Metadata language/signer chỉ dùng cho split/audit/sampling đã khai báo, không đưa vào encoder.
2. Teacher đọc full RGB; online encoder đọc masked context. Teacher features không đi vào predictor input.
3. Không truyền target keypoints, pose trajectories, target bbox hay detector confidence vào predictor. ROI chỉ áp **sau prediction** để tính loss.
4. Dùng random masks độc lập với pose trong MVP. Mask theo tay/trajectory là một experiment khác có nguy cơ side-channel.
5. Low-quality pose không làm clip biến mất khỏi generic RGB loss. Missing targets không được biến thành tọa độ zero ground truth.
6. Original SignRep checkpoint, controlled objective comparisons và foundation-transfer checkpoints là ba nhóm khác nhau trong bảng kết quả.
7. Không dùng test cho hyperparameter/checkpoint/readout selection. Mọi thay đổi data, initialization, resolution, visible budget, batch statistics và tuning trials đều phải log.
8. Code, data manifest, checkpoint checksum và feature readout phải được pin; `main`/`latest` chỉ dùng để khám phá lần đầu.
9. Code đã implement, code đã smoke-test, training đã chạy và empirical improvement là các trạng thái riêng.

### 0.3 Kết quả cuối Agent phải giao

- Package `asc_jepa` cài được; CLI dữ liệu, preprocessing, audit, train, extract, evaluate, ablate, export.
- Baseline extraction từ checkpoint SignRep nếu checkpoint/data khả dụng; báo chính xác phần chưa tái lập.
- Hiera masked/unmasked adapters với tensor/grid contracts được test.
- B0–B4 và controls có config, manifest, initialization, seed schedule và evaluation chung.
- Checkpoint đầy đủ để resume, inference-only encoder export và model card.
- `sources.lock.json`, `checkpoints.lock.json`, `data_manifest.lock.json`, environment lock, run registry, metrics thật.
- `IMPLEMENTATION_STATUS.md` với evidence cho từng milestone, limitations và quyết định tiếp theo.

Không cần triển khai translation decoder hoặc mọi SSL family trước khi B0–B4 ổn định. Giai đoạn transfer là milestone riêng; chưa chạy thì ghi rõ chưa chạy.

## 1. Nguồn bắt buộc và cách kế thừa

### 1.1 Source matrix

| ID | Nguồn gốc | Kế thừa gì | Không được suy diễn |
|---|---|---|---|
| R01 | SignRep, ICCV 2025 | RGB encoder + sign geometry priors; checkpoint/readout reference | Repo example không chứng minh toàn bộ pretraining code có sẵn |
| R02 | Official SignRep repo | Published model, transform, example, release | Không suy dims của tensor chỉ từ tên `latent` |
| R03 | Hiera, ICML 2023 + official repo | Backbone, MAE-K400 initialization, sparse masking, Roll/Unroll | Token sequence không mặc định ở raster order |
| R04 | V-JEPA / official `jepa` repo | EMA teacher, masked latent prediction và predictor precedent | Không gọi Hiera adaptation là exact V-JEPA architecture |
| R05 | V-JEPA 2.1 + official `vjepa2` repo | Dense/context loss và deep supervision precedent; transfer checkpoint | Hiera recipe không tái lập internet-scale training của paper |
| R06 | VICReg + official repo | Variance floor, covariance penalty và distributed statistics precedent | Loss weights không portable nếu reduction/dimension thay đổi |
| R07 | LeJEPA + official repo | Optional SIGReg branch | Thay regularizer không biến EMA model thành faithful LeJEPA |
| R08 | Pose lifting được SignRep dẫn | 3D pose/angles và bone normalization reference | MediaPipe pseudo-depth không tương đương pose lifting này |
| R09 | MediaPipe official docs/source | Practical offline pose/ROI extractor cho adaptation | Handedness score không phải per-joint localization confidence |
| R10 | YouTube-SL-25 | Pretraining video manifest/source IDs | Có captions không có nghĩa core phải dùng captions |
| R11 | ASL-Citizen + ASL-LEX | Signer-independent split và phonological metadata | Annotation-neighbor không tự là linguistic minimal pair |
| R12 | WLASL | Secondary recognition/retrieval benchmark | Paper sample count không đảm bảo URLs hiện còn đủ |
| R13 | SHuBERT | Contextual representation baseline/literature | Không so khác data/readout như objective-only improvement |
| R14 | SignDINO | Local/global sign distillation precedent | Local hand/face supervision không phải invention mới |
| R15 | SignMAE 2026 | Articulator/motion-aware masking precedent | Không nhầm với SignMAE trong UniGloR/WACV 2025 |
| R16 | SignSeek | Articulator latent prediction và updated retrieval comparator | Framework dùng gloss supervision, không là pure label-free comparator |
| R17 | PyTorch official docs/source | DDP, AMP, optimizer, differentiable collectives | Plain `dist.all_gather` không tự bảo đảm gradient contract cần dùng |
| R18 | Gradient Cache paper + author repo | Two-pass embedding-gradient cache cho memory-limited statistics | Cần RNG/DDP gradient-equivalence tests; không phải drop-in sign method |

### 1.2 R01–R02: SignRep, code entrypoints và checkpoint

**Paper:** https://arxiv.org/html/2503.08529v1

**Proceedings:** https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html

**Code:** https://github.com/ryanwongsa/SignRep

**Example đã đọc:** https://github.com/ryanwongsa/SignRep/blob/main/example_usage.py

**Requirements:** https://github.com/ryanwongsa/SignRep/blob/main/requirements.txt

**Release:** https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1

Reading targets: §§3–6, §7.1, Appendix A1–A4 và A6. Cần đọc code Model/head/transform sau clone.

Thông tin paper để audit replication: Hiera-B; MAE-Kinetics initialization; 16 frames; mask 80%; 500K updates; batch 20; Adam, peak LR 1e-4, warmup 50K, cosine, layer decay 0.85. Đây là **reference recipe**, không phải hyperparameters mặc định của pilot ASC-JEPA.

Example công bố import paths:

- `models.final_models.FINAL_hiera_latent_model_head_v25_active.Model`
- `augmentation.video.base_video_aug.Transformation`
- `models.head_models.dict_head_v6_FINAL_thoughtful` là head name được config tham chiếu.

Example load checkpoint field `model`, input `[B,3,16,224,224]`, sliding windows stride 2, outputs `features` và `latent`. Những đường dẫn này cần được tìm lại tại pinned commit; không copy nguyên config khi chưa kiểm tra model/head.

**AUDIT URLs cho code sâu:**

- https://github.com/ryanwongsa/SignRep/blob/main/models/final_models/FINAL_hiera_latent_model_head_v25_active.py
- https://github.com/ryanwongsa/SignRep/blob/main/augmentation/video/base_video_aug.py
- Head: tìm bằng import path ở trên trong clone; không giả định README path `head_models/` là root thực.

Trong phiên xây spec, full nội dung các file sâu không fetch được. Agent MUST đọc chúng local; không có API sâu nào trong spec được tuyên bố đã chạy.

Paper retrieval (§6) dùng activity-weighted window aggregation. Controlled protocol trong spec dùng uniform aggregation cho mọi encoder; do đó khác original readout. Giữ cả hai protocol, không đối chiếu số published như cùng protocol nếu chưa match.

### 1.3 R03: Hiera backbone và MAE checkpoint

- Paper: https://proceedings.mlr.press/v202/ryali23a.html
- Official code/model zoo: https://github.com/facebookresearch/hiera
- Model reference: **`hiera_base_16x224`**, checkpoint key **`mae_k400`**.
- Không chọn `mae_k400_ft_k400` cho controlled SSL mặc định: key đó đã có supervised fine-tuning.
- Code audit: `hiera/hiera.py`, `hiera/hiera_utils.py`, `hiera/hiera_mae.py`, `hubconf.py`, `examples/`.
- `hiera_utils.py` và `return_intermediates=True` được README dùng để giải thích cách lấy feature maps sau token reordering.
- Native MAE convention được README ghi: mask `True` là visible. Adapter của project cũng dùng `visible=True`; test conversion tại một nơi.

Repository root xác nhận model/key và các semantics trên. Full source adapter vẫn là AUDIT. Không hard-code constructor kwargs, stage indices hoặc grid dimensions chưa được probe.

### 1.4 R04–R05: JEPA nguồn chính

**V-JEPA paper:** https://arxiv.org/abs/2404.08471

**V-JEPA code:** https://github.com/facebookresearch/jepa

**V-JEPA 2.1 paper:** https://arxiv.org/html/2603.14482v3

**Code/checkpoints:** https://github.com/facebookresearch/vjepa2

**Hub entrypoints:** https://github.com/facebookresearch/vjepa2/blob/main/hubconf.py

Reading targets V-JEPA 2.1: §2.3.1 dense loss, §2.3.2 deep self-supervision, model distillation, Appendix A/B. Repo README xác nhận các thư mục:

- `app/vjepa_2_1/`: training loop.
- `configs/train_2_1/`: recipe và cooldown configs.
- `src/models/`, `src/masks/`, `src/utils/`: model, mask và shared utilities.
- `evals/video_classification_frozen/`: frozen evaluation precedent.

Agent phải `rg` trong pinned repo để xác định **actual file, symbol, target normalization, visible/context weight, teacher layer tap và EMA schedule**. Không invent các filenames còn lại.

Foundation-transfer candidate đã có trong model zoo: **`vjepa2_1_vit_base_384`**, ViT-B/16 distilled, 384 resolution. Có thêm `vjepa2_1_vit_large_384`. Resolve checkpoint/config URL từ catalog tại pinned source. Không đoán filename `.pt`; không đổi model thành 224/16-frame mà vẫn gọi đó là original transfer recipe.

Core của spec lấy ý tưởng dense/multi-layer supervision, nhưng dùng Hiera, predictor mới và data nhỏ hơn: tên kết quả phải là **V-JEPA-2.1-inspired Hiera control**.

### 1.5 R06–R07: anti-collapse losses

**VICReg paper:** https://arxiv.org/abs/2105.04906

**Code:** https://github.com/facebookresearch/vicreg

**Loss implementation reference:** https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

Tìm symbols `std_loss`, `cov_loss`, `off_diagonal`, gather function. Kế thừa variance/covariance mechanism, không lấy ResNet weights. Định nghĩa normalized reduction mới ở §10.6 là project adaptation.

**LeJEPA paper:** https://arxiv.org/html/2511.08544v3

**Official code:** https://github.com/galilai-group/lejepa

README hiện có API `lejepa.univariate.EppsPulley` và `lejepa.multivariate.SlicingUnivariateTest`. Agent pin version/commit rồi audit signatures; dùng SIGReg thay variance/covariance ở optional C6, không cộng cả hai. Một EMA-Hiera + SIGReg experiment phải mang tên đó, không mang tên “LeJEPA reproduction”.

### 1.6 R08–R09: geometry source và adaptation boundary

**Pose lifting paper:** https://arxiv.org/abs/2308.09525 — *Improving 3D Pose Estimation for Sign Language*, Ivashechkin, Mendez, Bowden.

SignRep Appendix A1 sử dụng pose/angle estimation từ hướng này và canonical bone lengths. Muốn faithful target replication phải có actual extractor, skeleton order, FK conventions và train-derived bone statistics. Trong snapshot chưa xác nhận được official extractor repo/checkpoint đầy đủ; ghi blocked cho **exact 3D replication**, không giả định code có trong SignRep.

**Practical extraction branch R09:**

- Current Tasks Holistic Python guide: https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker/python
- Legacy source reference: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/holistic.py
- Legacy docs: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
- Pinned legacy distribution candidate: https://pypi.org/project/mediapipe/0.10.14/
- Known hand ROI issue paper: https://arxiv.org/abs/2405.03545

MVP ưu tiên Tasks Holistic `VIDEO` mode nếu package/model bundle tương thích. Fallback là **isolated preprocessing env** dùng legacy version đã import/smoke-test; không silently đổi detector giữa experiments. Record package version, graph/model hash và returned landmark schema. Legacy face và Tasks face có thể khác landmark counts; adapter MUST dùng schema, không ép reshape.

MVP geometry dưới đây là **2D adaptation do spec đề xuất**. Không gọi output `z` của landmark detector là ground-truth 3D, không load 2D targets vào original 3D SignRep head.

### 1.7 R10–R12: data

- YouTube-SL-25 paper: https://arxiv.org/html/2407.11144v1
- Released IDs navigation target: https://github.com/google-research/google-research/tree/master/youtube_sl_25
- Agent xác nhận URL qua footnote release trong paper và clone; subtree chưa đọc đầy đủ trong snapshot.
- ASL-Citizen official description: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/
- ASL-LEX metadata: https://asl-lex.org/about.html
- ASL-LEX paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685/
- WLASL official data/code: https://github.com/dxli94/WLASL

ASL-Citizen giữ official participant splits và ASL-LEX Code mapping. WLASL giữ actual available sample IDs. Không tự tìm mirror không rõ provenance để bù missing videos.

### 1.8 R13–R16: nearest competitors và optional integrations

- SHuBERT: https://aclanthology.org/2025.acl-long.1397/ ; full method https://arxiv.org/html/2411.16765v3
- SignDINO: https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html
- SignMAE 2026: https://arxiv.org/html/2605.02094v1
- SignSeek: https://arxiv.org/html/2609.03695v1

Đây là nguồn để thiết kế comparator và tránh claim trùng. Không có repo/checkpoint path nào của bốn công trình được giả định trong spec. Agent chỉ thêm integration sau khi xác minh link code của tác giả, checkpoint, preprocessing và protocol; nếu chưa khả dụng, ghi `not_available`, không tạo fake wrapper thay rồi gắn tên paper.

SignSeek có gloss-supervised component. Kết quả của nó nằm trong nhóm **label-supervised pretraining**, được báo nguồn nhãn riêng.

### 1.9 R17: runtime technical references

- PyTorch install/version matrix: https://pytorch.org/get-started/locally/
- AMP: https://docs.pytorch.org/docs/stable/amp.html
- DDP: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- Differentiable collectives source: https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py
- Checkpoint loading: https://docs.pytorch.org/docs/stable/generated/torch.load.html

Runtime installed version quyết định API thực. Test DDP gradients against single-process full-batch reference trước khi dùng distributed regularizer; không suy correctness chỉ từ tên function.

## 2. Source audit, pinning và provenance

### 2.1 Bootstrap repositories

Các lệnh sau là bootstrap targets, chưa phải chứng nhận dependencies đã cài. Chạy từ project root sau kiểm tra root có writable và không chứa work cần giữ.

```bash
mkdir -p external artifacts/checkpoints artifacts/audits manifests configs reports runs
git clone https://github.com/ryanwongsa/SignRep.git external/SignRep
git clone https://github.com/facebookresearch/hiera.git external/hiera
git clone https://github.com/facebookresearch/jepa.git external/jepa
git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
git clone https://github.com/facebookresearch/vicreg.git external/vicreg
git clone https://github.com/galilai-group/lejepa.git external/lejepa
```

Nếu destination tồn tại: reuse đúng repo, kiểm tra `git status`; không overwrite. Optional repos chưa dùng không cần cài package.

Sau clone cho mỗi repo:

```bash
git -C external/hiera rev-parse HEAD
git -C external/hiera status --short
rg -n 'class Hiera|def forward|return_intermediates|mask_unit|Roll|Unroll' external/hiera
rg -n 'FINAL_hiera|dict_head|features|latent|load_state_dict' external/SignRep
rg -n 'layer_norm|ema|momentum|context|loss' external/vjepa2/app/vjepa_2_1
rg -n 'std_loss|cov_loss|off_diagonal|Gather' external/vicreg
```

Pin actual full commit SHA bằng detached checkout/submodule/vendor metadata. Agent tạo permalinks `https://github.com/OWNER/REPO/blob/FULL_SHA/path`; không ghi SHA do suy đoán. `v0.0.1` chỉ là checkpoint release reference, không chứng minh latest source cùng commit.

### 2.2 File `sources.lock.json`

Mỗi source component có:

```json
{
  "component": "hiera_backbone",
  "paper_id": "R03",
  "repository_url": "https://github.com/facebookresearch/hiera",
  "commit_sha": null,
  "local_files": [],
  "symbols_used": [],
  "permalinks": [],
  "license_files": [],
  "changes_from_upstream": [],
  "audit_status": "pending"
}
```

Các `null`/empty ở đây là template. M0 phải điền actual values cho component dùng; unresolved checkpoint/source không được coi ready. Giữ LICENSE/NOTICE của code được vendor; record terms của weights/dataset riêng vì có thể khác code.

### 2.3 Audit evidence cần tạo

`artifacts/audits/upstream_contracts.json` MUST có:

- Hiera actual model factory signature, native visibility convention, stage taps, dimension/grid probe.
- Native masked return layout và mapping inverse sau Roll/Unroll; shape khác unmasked được ghi riêng.
- SignRep model/head/transform availability, constructor kwargs, output shapes, state-dict key conventions.
- V-JEPA loss/normalization/EMA logic nào thực sự dùng; code mới khác ở đâu.
- Detector output counts, coordinate units, visibility/confidence fields thực sự có.
- Source snippet location bằng file/symbol/line; không cần copy cả third-party file vào report.

Nếu upstream không có training loop cần thiết, reimplement từ paper và gắn `paper_reimplementation`, kèm differences. Không gọi là official code execution.

### 2.4 Checkpoint registry

`checkpoints.lock.json` mỗi checkpoint ghi: name, source paper/repo/release, actual download URL, architecture/config hash, SHA256, size bytes, state-dict root, missing/unexpected keys, initialization provenance, pretraining data/supervision known, load verification status.

| Checkpoint | Cách resolve | Dùng ở đâu |
|---|---|---|
| Hiera-B MAE-K400 | Model zoo R03, key `mae_k400`, factory `hiera_base_16x224` | Shared initialization B0–B4 |
| SignRep published | Release R02 `v0.0.1`; enumerate actual assets | Published frozen baseline; optional warm-start branch riêng |
| V-JEPA 2.1 distilled B | R05 model zoo + associated config/hub entrypoint | Foundation transfer comparator |

Resolve SignRep assets bằng public GitHub release API hoặc:

```bash
gh release view v0.0.1 --repo ryanwongsa/SignRep --json assets,tagName,targetCommitish,url
```

Không đoán filename hoặc dùng generic pattern rồi load nhầm asset. Chọn model asset theo config/state-dict compatibility. Nếu asset list có nhiều model, inspect từng metadata trên CPU và ghi lựa chọn. Không cài GitHub app/login chỉ để đọc public release nếu HTTPS/API dùng được.

Hiera initialization có thể dùng local pinned repository qua Torch Hub factory với `source="local"`, `pretrained=True`, `checkpoint="mae_k400"`. AUDIT signature trước. Record actual fetched URL và cached bytes hash. Không dùng remote mutable Hub để tự lấy latest ở mỗi run.

Checkpoint load policy:

- Generic Hiera encoder: chỉ cho phép missing/unexpected classifier/decoder keys đã xác minh; mọi backbone shape mismatch là error.
- SignRep published: ưu tiên strict load đúng original Model; patch format phải có conversion report.
- Không dùng `strict=False` rồi bỏ qua log.
- Random initialization chỉ dùng synthetic/shape tests, không lặng lẽ thay missing pretrained checkpoint trong experiment.

## 3. Project layout và execution model

### 3.1 Files/modules Agent phải tạo

| Path | Nhiệm vụ |
|---|---|
| `pyproject.toml` | Package và CLI entrypoints; extras train/preprocess/eval |
| `src/asc_jepa/config.py` | Typed config, schema validation, resolved config hash |
| `src/asc_jepa/data/manifest.py` | Versioned schema, split/provenance validation |
| `src/asc_jepa/data/video.py` | Timestamp-aware decode/sampling, RGB conversion |
| `src/asc_jepa/data/transforms.py` | Shared geometry + separate mild photo views |
| `src/asc_jepa/data/dataset.py` | Training item không trả labels/text |
| `src/asc_jepa/preprocess/holistic.py` | Offline extractor adapters |
| `src/asc_jepa/preprocess/regions.py` | ROI, quality, interpolation validity policy |
| `src/asc_jepa/preprocess/geometry.py` | 2D targets/stats; optional exact 3D adapter riêng |
| `src/asc_jepa/models/hiera_adapter.py` | Native sparse masks, inverse token mapping, StageMeta |
| `src/asc_jepa/models/predictor.py` | Context memory + grid/time queries + per-stage outputs |
| `src/asc_jepa/models/heads.py` | Geometry head, regularizer projector |
| `src/asc_jepa/models/ema.py` | Encoder EMA and buffers policy |
| `src/asc_jepa/losses/latent.py` | Masked/context normalized losses |
| `src/asc_jepa/losses/articulatory.py` | ROI pool, observability, state/change reduction |
| `src/asc_jepa/losses/geometry.py` | Valid/quality-weighted target groups |
| `src/asc_jepa/losses/regularization.py` | VICReg-style / optional SIGReg; distributed gather |
| `src/asc_jepa/train/engine.py` | AMP, accumulation, optimizer/EMA/RMS update ordering |
| `src/asc_jepa/train/checkpoint.py` | Atomic save, strict resume, RNG/sampler states |
| `src/asc_jepa/train/gradient_cache.py` | Optional exact reg-gradient cache và replay contracts |
| `src/asc_jepa/eval/extract.py` | Frozen encoder features and cache provenance |
| `src/asc_jepa/eval/retrieval.py` | Gallery/query isolation, streaming class scores |
| `src/asc_jepa/eval/probe.py` | Low-label linear/attentive probe |
| `src/asc_jepa/eval/diagnostics.py` | Phonology, observability, shortcuts, dynamics |
| `src/asc_jepa/automation/runner.py` | Dependency gates, resumable task/run registry |
| `src/asc_jepa/cli.py` | Documented commands in §15 |
| `tests/` | Contract, gradient, protocol and resume checks |
| `configs/` | Pilot, B0–B4, controls, published/controlled eval |
| `reports/` | Evidence, metrics and decisions; không dùng làm fake result store |

Third-party code ở `external/` read-only sau pin. Modifications cần patch/version riêng, không sửa âm thầm source baseline.

### 3.2 Dependencies

Train env SHOULD dùng Python 3.10/3.11, PyTorch+torchvision matching CUDA/driver, NumPy, PyYAML, einops, timm và upstream Hiera dependencies. Decode MVP dùng OpenCV; alternative decoder phải vượt timestamp fixture trước khi thay.

Preprocess env tách riêng nếu MediaPipe dependencies xung đột với training. Tasks bundle/package version được audit. Legacy fallback candidate Python 3.10 + MediaPipe 0.10.14 phải kiểm tra `mp.solutions.holistic`; không giả định latest package còn legacy API.

Agent tạo env, cài theo official compatibility rồi ghi exact resolved versions vào lock. Đừng bịa “tested environment” từ version wishlist. Không bulk-install requirements của mọi upstream repo vào một env.

Runtime audit: OS, Python, torch/torchvision/CUDA/cuDNN, GPU model/memory, CPU cores, decoder backend, detector version, optional Flash/SDPA use. CPU-only mode phải chạy synthetic and data/loss unit tests; real training cần actual throughput/memory probe.

### 3.3 Task state machine

`pending → running → passed | failed | blocked | not_run`. `passed` cần artifact evidence và config/source/data hash; không chỉ stdout “done”. Nếu hash đổi, invalidate dependent tasks.

| Milestone | Phụ thuộc | Output/gate |
|---|---|---|
| M0 Source/env/checkpoints | Không | Sources/checkpoints/env locks; load tests |
| M1 Manifest/protocol | M0 data sources | Exclusion audit, train/eval split locks |
| M2 Preprocessing | M1 | Sidecars, target stats, 200-clip audit plan/results |
| M3 Backbone/predictor contracts | M0 | Grid/mask/gradient tests and architecture probe |
| M4 Loss/step correctness | M2–M3 | Synthetic gradient, reduction, DDP checks |
| M5 Frozen baselines | M1, checkpoint load | Published/control extraction and validation metrics |
| M6 Sanity/throughput | M4 | 1–2K update cap; finite losses, no evident collapse |
| M7 B0–B4 pilot | M5–M6 | Matched 10–20K runs and validation comparison |
| M8 Controls/confirmation | M7 | Mechanism controls; selected three-seed runs |
| M9 Retrieval/low-label test | Frozen selections M8 | Locked protocol test metrics + CI |
| M10 Transfer/scale | M8 and data/compute | Optional continuous task; separate status |
| M11 Export/report | Completed applicable milestones | RGB-only model/export, model card, honest report |

Ngăn lỗi kỹ thuật ở một task; tiếp tục công việc độc lập. Ví dụ missing SignRep checkpoint có thể block M5 published baseline nhưng không block M3 synthetic adapter tests. Không đổi `blocked` thành `passed` để chạy research claim.

## 4. Data contracts và chống leakage

### 4.1 Manifest schema

Một JSONL row biểu diễn một available video/clip; IDs phải ổn định:

```json
{
  "schema_version": 1,
  "sample_id": "dataset:release:stable_id",
  "dataset": "youtube_sl_25",
  "dataset_release": "resolved_release_or_commit",
  "source_video_id": "actual_source_id",
  "source_group_id": "same_raw_video_or_duplicate_group",
  "local_video_path": "data/videos/actual_file.mp4",
  "video_sha256": "actual_hash",
  "split": "pretrain_train",
  "language": "ase",
  "signer_id": null,
  "clip_start_ms": 0,
  "clip_end_ms": 0,
  "duration_ms": 0,
  "fps_num": 0,
  "fps_den": 1,
  "width": 0,
  "height": 0,
  "label_namespace": null,
  "class_id": null,
  "asl_lex_code": null,
  "pose_sidecar": null,
  "acquisition_status": "pending"
}
```

Template zero/null chưa hợp lệ để train. M1 chỉ xuất `available` rows có positive duration, decodable timestamps và actual hashes. Với pretrain manifest, strip text/labels/class_id khỏi training view; keep audit metadata riêng ngoài batch loader.

### 4.2 Default data scope

- Pretraining pilot: subset YT-SL-25 available, train-only, khoảng 50–200 giờ là PILOT budget; actual hours ghi rõ.
- Nếu chưa có YT-SL RGB, dùng unlabelled **downstream train-only** cho engineering pilot và đặt tên `in_domain_train_only_pilot`. Không claim cùng regime public SignRep hoặc unseen-dataset transfer.
- Primary downstream: ASL-Citizen official signer-independent split.
- Secondary: WLASL actual available official split.
- NMFs-CSL optional sau khi xác minh data/source/protocol; không tự tạo path hoặc số lượng.
- Generic pretrained checkpoint có unknown large-scale provenance: chỉ claim “unseen during sign adaptation” khi có đủ manifest, không claim never-seen globally.

### 4.3 Exclusion pipeline

1. Chuẩn hóa source IDs trước split; keep aliases URL/video ID.
2. Remove pretraining rows có raw source ID trùng **mọi downstream split** cho transfer regime.
3. Group crops/time spans từ cùng raw video vào cùng source group.
4. Check near-duplicates bằng hashes/perceptual signatures ở sampled frames nếu khả thi; report unresolved risk, không fabricate perfect deduplication.
5. Respect ASL-Citizen participant split; không random-resplit clips để tăng score.
6. WLASL class ID/split đọc actual annotation; giữ missing-video report và sample-ID intersection giữa methods.
7. Channel ID chỉ là source-group proxy, không thay signer ID khi claim signer independence.
8. Tạo exclusion CSV: reason, source ID, affected sample IDs, method and audit date.

Pretrain subset selection deterministic trên source groups; không chọn sample bằng downstream test difficulty. Target normalization, pose bone stats, quality thresholds calibrated chỉ từ pretrain_train/predefined validation.

### 4.4 Acquisition behavior

Ưu tiên user-provided/official available videos. Chỉ acquire các IDs đã thuộc manifest; downloader không mở rộng tìm kiếm ngoài corpus. Retry network/decode failures hữu hạn, log missing; không lặp vô hạn hoặc bypass access wall. Một benchmark subset hợp lệ phải báo sample count và missing classes trước tính metrics.

Core training không đọc captions/audio. Nếu video có chữ/gloss burned-in, ghi data audit; crop/intervention policy phải chung giữa methods, không xử lý chỉ proposal.

### 4.5 Data version output

`data_manifest.lock.json` chứa manifest SHA256, available sample IDs hash, source exclusions hash, clip exposure policy, aggregate hours/languages, splitter version, target-stat hashes, geometry schema hash và gallery/query protocol hash.

## 5. Decode, sampling và augmentation

### 5.1 Temporal policy

MVP: `T=16`. Hai policy phải tách tên:

- `native_consecutive`: 16 consecutive decoded frames và actual timestamps; phù hợp replication reference.
- `fixed_rate`: target 25 FPS là PILOT choice, nearest timestamp sampling; không giả định input nào cũng 25 FPS.

Controlled B0–B4 dùng cùng policy. Lưu `frame_indices`, `timestamps_ms`, `frame_valid`, `is_repeat` mỗi sample. Không tính temporal change giữa repeated source frames hoặc invalid timestamps. Video short hơn T có thể repeat-pad để generic encoding, nhưng padded positions không đóng góp change/geometry target loss; report short-video count.

Không gọi sliding-window **stride 2** của example SignRep là temporal subsampling stride 2: đó là bước dịch cửa sổ, không tự là khoảng giữa frames trong một clip. Eval window indices và frame sampling policy là hai config khác nhau.

### 5.2 Spatial policy và RGB-only inference

MVP controlled path: resize giữ aspect ratio rồi deterministic center/full-frame crop đủ upper body khi phù hợp. Resize/crop transform cần output affine matrix từ original pixels sang model pixels. RGB→float32 [0,1]→normalization theo actual pretrained model metadata đã audit.

Training augmentation nhẹ dùng cùng geometric transform cho teacher/student và sidecars; khác photometric views. Crop không cố tình bỏ hands/face. Nếu dùng pose để **tạo cached torso crop** như preprocessing reference, record `pose_assisted_crop=true`; inference API phải có explicit RGB-only fallback/person detector policy. Không tuyên bố cả pipeline RGB-only khi inference vẫn gọi pose cropper.

Để claim simple RGB-only MVP, mặc định full-frame geometry không dựa pose. Bbox sidecars chỉ dùng loss. ROI disappearance do crop làm target invalid ở frame đó.

### 5.3 Augmentations mặc định

- Shared mild resize/crop và translations; exact affine applied to ROI/keypoints.
- Independent mild color jitter/brightness/contrast; disable strong blur ở pilot để giữ fingers.
- Disable horizontal flip, time reversal, strong temporal warp và aggressive cutout mặc định.
- Disable different temporal crops between teacher/student.
- Không dùng label/pose-dependent photo augmentation.

Các augmentation từ SignRep/LeJEPA không được copy nguyên mà không kiểm tra sign invariance. Ablation dùng chúng cần config và data quality audit riêng.

### 5.4 Decoder checks

MUST có fixtures cho variable FPS, short clip, BGR/RGB, non-square resolution, corrupt frame và repeat timestamps. Pose extraction và train decode dùng cùng timestamp correspondence. OpenCV timestamp metadata có thể thiếu; ưu tiên container PTS/ffprobe audit và chỉ dùng `frame_index/fps` fallback cho confirmed constant-rate stream, ghi rõ fallback.

Training batch tensor: RGB `[B,3,T,H,W]`; timestamps `[B,T]` float64/float32 ms; validity bool. Không gửi decoded frames hoặc pose worker tensor lên GPU hai lần không cần thiết. CPU cache/shard là engineering choice phải đo I/O throughput và hash.

## 6. Offline pose, ROI và geometry

### 6.1 Extractor adapter

Implement `extract_video(video, timestamps) -> PoseSidecar`. Tasks Holistic VIDEO mode dùng monotonic timestamps; reset instance theo raw video, không carry tracking state từ signer/video trước. Resolve exact model asset từ official docs và hash. Nếu fallback legacy: detector/graph/version khác phải mang new preprocessing version.

Expected semantic parts: left_hand, right_hand, face, upper_body. Adapter phải map anatomical left/right; không suy từ screen x hoặc tin handedness classification trên non-mirrored input mà không audit. Multi-person ambiguous clip được quality flag, generic loss vẫn có thể dùng.

Sidecar lưu:

```text
timestamps_ms: [F]
landmarks_xy: schema-defined named arrays
landmarks_aux_z: optional, never silently treated as metric 3D
joint_valid: matching bool arrays
pose_visibility/presence: only fields actually returned
roi_xyxy: [F,4,4], normalized original-image coordinates
part_valid: [F,4]
part_quality: [F,4]
quality_provenance: per-part rule/field names
extractor_id, extractor_version, model_sha256, schema_sha256
```

### 6.2 Confidence policy

MVP `q=1` cho valid part, `q=0` cho missing/invalid; dùng pose visibility chỉ khi field semantics có nguồn và sanity check. Optional jitter score là heuristic quality, đặt tên rõ.

Không tạo per-joint confidence cho hand/face nếu extractor không trả nó; không lấy option `min_tracking_confidence` hoặc handedness probability làm localization accuracy. Score không là calibrated probability.

Interpolation mặc định off. Nếu interpolate ngắn: keep original valid flag riêng; interpolated targets không được count như ground truth trong main local/change loss. Out-of-frame/clipped ROI phải giảm actual support; không clamp keypoint rồi coi prediction đúng.

### 6.3 ROI construction

- Hands: bbox của valid hand xy points; PILOT margin `max(0.15*bbox_extent, 0.02*image_extent)` mỗi chiều, clip vào image.
- Face: valid face landmark bbox + margin; fallback body nose/eyes chỉ là fallback ROI type, không giả làm face geometry target.
- Upper body: shoulders/elbows/hips/head box theo schema; crop-boundary clipping có log.
- Không dùng bbox ở thời điểm missing để tạo valid target bằng carry-forward. Tracking có thể dùng cho association nhưng loss validity không đổi.
- Hands overlap nhau hoặc face/body: cho phép một token thuộc nhiều parts; part-balanced reduction xử lý thay vì exclusive segmentation.

ROI chỉ chuyển sang augmented coordinates sau shared affine. Không thay bbox theo teacher photometric view. Track part association giữa frame trước temporal differences; flagged swap làm change pair invalid.

### 6.4 2D geometry targets do project đề xuất

Tạo `geometry_schema.json`, tất cả indices dưới đây thuộc **MediaPipe hand-local order**, không thuộc SignRep 61-joint skeleton.

Hand chain reference: wrist 0; fingers `[1,2,3,4]`, `[5,6,7,8]`, `[9,10,11,12]`, `[13,14,15,16]`, `[17,18,19,20]`. Agent xác minh source schema R09.

| Group | Định nghĩa | Dim/hand | Validity |
|---|---|---:|---|
| `hand_shape_xy` | 21 joints relative to wrist, scale bằng palm span | 42 | Wrist/palm scale + joint valid |
| `hand_location_xy` | Wrist relative to shoulder midpoint, scale shoulder width | 2 | Body anchors và wrist valid |
| `hand_orientation_xy` | Unit vector wrist→middle MCP (index 9) | 2 | Nonzero vector |
| `hand_joint_angles` | 3 interior joint angles trên mỗi wrist+finger chain; encode cos/sin | 30 | Valid triplet và nonzero bones |
| `hand_tip_distances` | 10 unordered pairwise distances giữa tips 4,8,12,16,20, normalized palm scale | 10 | Valid endpoint pair |

Body target dùng named upper-body joints từ detector schema, shoulder-relative xy; không cần regress toàn bộ face mesh trong MVP. Face được học bằng local latent state; chưa claim có facial grammar supervision.

Coordinates cho geometry dùng x-right/y-up nhất quán, conversion từ image y-down được test. Palm scale là distance MCP 5↔17; shoulder scale là left↔right shoulder. Scale quá nhỏ => group invalid; threshold lấy từ train quantiles/pilot, record. Không rotation-canonicalize cả hand: giữ orientation/location trong các group riêng.

Angles tính qua normalized dot và signed 2D cross; encoding cos/sin tránh discontinuity ±π. Khoảng cách không được gọi là metric 3D. Train-only robust normalization stats cho từng group; keep group masks và continuous quality. Không convert left/right thành cùng orientation bằng silent mirror.

### 6.5 Exact SignRep target branch

Chỉ bật khi Agent có extractor/3D conventions đủ để match R01 Appendix A1 và code head. Branch có riêng `geometry_schema_id=signrep_3d_exact`, bone stats, projection dimensions và original loss reductions. Thiếu component => `blocked_exact_replication`.

2D geometry branch vẫn có thể nghiên cứu ASC-JEPA nhưng tên baseline là **SignRep-inspired geometry control**, không “reproduced SignRep pretraining”. Published checkpoint evaluation không bị thay thế bởi branch này.

### 6.6 Preprocessing gates

Audit stratified khoảng 200 available training clips theo quality/motion/source. Tạo overlays và CSV số valid hands/face, left-right swaps, outside-image joints, missing spans, ROI occupancy và timestamp offsets. Agent kiểm tra trực quan overlays có sẵn; không ghi đã có native signer review nếu chưa có.

Lỗi correspondence, target schema hoặc scale phải sửa trước training. Nếu detector missing nhiều, vẫn triển khai generic branch và đo quality; không gate toàn dataset để chỉ còn easy clips.

## 7. Backbone adapter: tensor, grid và mask contracts

### 7.1 Canonical internal layout

Không dùng classification logits làm representation. Adapter lấy feature maps thực của encoder và khai báo:

```python
@dataclass
class StageMeta:
    name: str
    channels: int
    grid_thw: tuple[int, int, int]
    flat_order: str                 # canonical t-major, then h, then w
    spatial_support_xyxy: Tensor    # [N,4], normalized augmented coordinates
    temporal_support: Tensor       # [T_stage,T_input], detached support weights
    mask_unit_to_flat: Tensor       # audit-derived native→canonical mapping

@dataclass
class StageContext:
    tokens: Tensor                  # [B,N_visible,D_stage]
    flat_indices: Tensor            # [B,N_visible], canonical positions
    token_valid: Tensor             # [B,N_visible], handles padding
    visible_grid: Tensor            # [B,T_stage,H_stage,W_stage], True=visible
    meta: StageMeta

@dataclass
class StageTarget:
    grid: Tensor                    # [B,T_stage,H_stage,W_stage,D_stage]
    time_ms: Tensor                 # [B,T_stage]
    time_valid: Tensor              # [B,T_stage]
    meta: StageMeta
```

`spatial_support` ở đây là tile footprint để tính ROI coverage, không là exact effective receptive field/conditional observability. Convolution/attention có receptive field rộng; proxy này cần diagnostic và không được viết thành theorem.

`temporal_support` cũng là structural patch/pooling footprint dùng để gán timestamps và sidecar coverage; nó không mô tả toàn bộ temporal receptive field sau attention. Agent ghi rõ stride/kernel/pooling convention và audit boundary effects.

Teacher forward returns `dict[str, StageTarget]`; student forward returns `dict[str, StageContext]`. Predictor returns full canonical grid cho các stages requested. Features ngoài stage tensor phải được khai báo riêng, không overload `latent`.

### 7.2 Stage selection

MVP request một stage giữa và final stage. Chọn stage giữa có hand ROI occupancy đủ và manageable cost; thường candidate là stage trước final trong native Hiera hierarchy, nhưng actual grid/channel phải probe.

`architecture_probe.json` MUST ghi:

- Model name, SHA/checkpoint, input T/H/W.
- Ordered stage names, channels, full/masked output shape.
- Native mask-unit count, pooling stride, per-stage canonical grid mapping.
- Spatial/temporal support và effective time spacing ở actual sampling policy.
- Number of visible tokens tại 60%/80% masks.
- ROI occupancy histogram trên sample training clips.
- Predictor query/context counts, peak memory, measured seconds/step.

Không cố định số 768/384 cho adapter chỉ vì SignRep head dùng các dim đó. Constructor/config resolved bằng probe; fail nếu dimensions không match checkpoint.

Nếu final stage hand occupancy thường dưới một tile, dùng mid-stage local loss và giữ final cho global representation. Nếu mid cũng quá thô, thử earlier stage/resolution như **separate resolution ablation** trước khi kết luận loss thất bại.

### 7.3 Native sparse masking

MVP MUST dùng native Hiera masked path hoặc adapter tương đương đã chứng minh context semantics. Native mask groups và pooling làm thay đổi packing; không lấy full unmasked features rồi xóa vài tokens ở output để giả masked encoder.

Procedure:

1. Generate native mask-unit visibility với fixed count per batch item.
2. Feed visibility vào native masked path đúng vị trí source yêu cầu.
3. Recover stage token coordinates qua native Roll/Unroll/mask mapping.
4. Return only actual visible tokens cùng canonical indices.
5. Reconstruct `visible_grid` từ indices; hidden slots không trở thành “visible zero features”.

Mask generator dựa clip seed và dimensions, không nhìn ROI/pose/quality/class. `mask_ratio` là fraction hidden; canonical visibility bool dùng `True=visible`.

Native group sizes khiến realized ratio khác requested ratio; log cả hai. Đừng gọi pixel zeroing + full compute là sparse masking nếu không thực sự sparse. Dense-masked fallback cần tên/config/cost riêng và cùng fallback cho mọi control; không silent fallback vì adapter khó.

### 7.4 Mask recipe

PILOT `hidden_ratio=0.8`; `0.6` ở sensitivity branch. Khởi đầu bằng uniform random native mask units; thêm spatial blocks/tubes sau khi inverse mapping đúng. Seed schedule giống giữa B0–B4.

Mỗi mask MUST giữ ít nhất một context unit và một hidden target unit; batch drop_last cho equal-size DDP nếu regularizer cần. Không giữ tay bằng pose-conditioned masks trong MVP.

Native patch embedding có thể có overlapping kernels. Audit masking trước/sau stem và spatial receptive fields. Test masked-input perturbations **ngoài support của visible stem tokens**; thay đổi pixels trong overlapping support có thể đổi visible token một cách hợp lệ. Record mask semantics, không yêu cầu invariance bất khả thi đối với mọi hidden pixel khi kernels overlap.

### 7.5 Adapter tests bắt buộc

- `Roll/Unroll` round-trip coordinate sentinel, không chỉ shape equality.
- Native packed visible indices unique/in-range; recovered visibility count match.
- Same clip+mask+eval mode → deterministic context.
- No-mask path matches native unmasked feature extraction trong tolerance.
- All-visible/all-hidden edge cases xử lý theo explicit policy; all-hidden rejected.
- Teacher `requires_grad=False`, stop-gradient targets; teacher never in optimizer.
- No-full-target leakage: context không lấy teacher outputs hoặc unmasked encoder outputs làm memory.
- `T/H/W` thay đổi được xử lý bằng new probe/validity hoặc reject rõ ràng; không guessed reshape.

## 8. Predictor và auxiliary heads

### 8.1 Architecture được đề xuất, không copy nguyên V-JEPA

MVP dùng **shared context-memory transformer + per-stage dense query decoders**. Đây là Hiera adaptation inspired by R04/R05; NEW-CANDIDATE contribution không nằm ở cross-attention architecture này.

1. Project visible tokens mỗi stage `D_l → D_pred=256` bằng Linear riêng.
2. Add canonical spatial/time positional encoding và stage embedding.
3. Concatenate requested stage memories, giữ padding attention mask.
4. Context transformer 2 blocks, 8 heads, MLP ratio 4, LayerNorm; dropout 0 cho initial correctness, later ablate if needed.
5. Full-grid positional queries cho mỗi target stage đọc context bằng cross-attention + FFN; output Linear `256 → D_l`.
6. Output full grid theo canonical order, không cần per-part inputs.

Queries chỉ có `(t,h,w)`, stage ID và learned shared query embedding. Không có bbox position, joint positions, part ID, detector score hoặc teacher latent. Local parts được pool **sau** full-grid prediction.

### 8.2 Positional encoding

Spatial x/y từ canonical tile centers. Time từ actual token timestamps relative to clip start, seconds; dùng shared fixed-frequency sinusoidal features rồi linear projection. Chọn frequency range/config trước experiment, không suy nó từ sign labels.

Context và query dùng cùng coordinates/clock. Không dùng packed index làm time/spatial position. Native backbone positional encoding giữ đúng source; additional predictor encoding không thay checkpoint pos embeddings âm thầm.

### 8.3 Query chunking

Dense self-attention trên toàn bộ early-stage queries có thể đắt. MVP query decoder không có query-to-query self-attention; each query cross-attends same memory. Vì vậy có thể chunk `Q` theo 128/256/512 queries và concat exact outputs.

Chunking phải giữ dropout 0 hoặc RNG equivalence và pass output/gradient equivalence test. Không concat chunks rồi tính loss riêng không đúng normalization. Log full query count, chunk size, cross-attention memory và measured cost. Context memory vẫn có temporal mixing.

Nếu đổi sang full query self-attention hoặc official V-JEPA predictor, coi là predictor ablation; không giữ giả định chunk-equivalent này.

### 8.4 Target normalization và readout phân biệt

- `raw_teacher_grid`: encoder stage features cho audit/readout.
- `target_grid = feature_LayerNorm(raw_teacher_grid.detach())`, không affine, normalization chỉ trên D.
- `predicted_grid`: predictor output raw, so với normalized target trong generic loss.
- Local states dùng LayerNorm của ROI-pooled target/prediction (§10).
- Regularizer dùng projector của online visible features, không teacher features.

Không normalize cả `[T,H,W,D]` thành một block. Không detach online features hoặc predictor khi tính training losses.

### 8.5 Geometry head

MVP head nhận **global pooled visible online context** và fixed time query encoding cho từng input frame. MLP `D_final + D_time → 512 → sum(target_group_dims)` hoặc separately typed heads. Không đưa actual ROI vào head input. Pose target là output supervision.

Geometry head dùng cùng architecture/reduction cho B0–B4. Nếu thử dense temporal geometry decoder, chạy dedicated head-only control; không quy gain cho routing. Original SignRep head là integration riêng để published checkpoint load, không reuse weights vào incompatible 2D head.

### 8.6 Regularizer projector

Visible online final-stage tokens valid-mean pool thành `[B,D_final]`; MLP `D_final→512→128`, GELU, optional feature LayerNorm ở **hidden layer**, không BN và không final LayerNorm/L2 normalization. Projector initialization và dimensions chung giữa controls.

Không dùng nhiều tokens của cùng clip làm “independent samples”. Actual regularizer sample là clip/source group; batch sampler SHOULD tránh duplicate source group trong một optimizer batch khi feasible. Hai photometric views của cùng clip không tự tính là hai independent clips.

### 8.7 Inference/export

Default representation là **online encoder final-stage unmasked valid mean**, L2 normalize sau window/video aggregation. Remove predictor, geometry head, regularizer projector và EMA copy khỏi deployment.

EMA encoder có thể báo additional row như consistent comparator; default không chọn giữa online/EMA theo test. Tensor API inference nhận `[B,3,16,H,W]` RGB normalized, trả raw embedding `[B,D_final]` và optional selected stage features.

Inference import không khởi tạo MediaPipe, không tải training targets và không yêu cầu sidecar. Export test phải chạy khi pose/sidecar directories không hiện diện.

## 9. ROI pooling, time correspondence và routing

### 9.1 ROI weights

Với spatial tile `i` và ROI `p` ở stage time `t`, `pi[p,t,i]` là intersection-area / tile-area, trong [0,1], detached. ROI time correspondence dùng StageMeta temporal support: tính weighted coverage trên input-frame ROIs trong support, không mix arbitrary nearest frames giữa teacher/student.

`pi` tensor `[B,P,T_l,H_l,W_l]`; `P=4`. Apply same augmented geometry. Frame/part invalid => corresponding weights zero; track effective valid support riêng để không confound missing với actual occlusion.

`part_valid_l` đòi ROI nonempty, sufficient valid temporal support và positive tile coverage. PILOT min weighted occupancy 1.0 tile; run sensitivity/earlier stage khi quá nhiều hands invalid. Đây là engineering proxy, không ngưỡng linguistic.

### 9.2 State tensors

```text
target_states[l]: [B,P,T_l,D_l]
predicted_states[l]: same shape
state_valid[l]: [B,P,T_l]
state_quality[l]: [B,P,T_l]
observability[l]: [B,P,T_l]
routing_gate[l]: [B,P,T_l]
```

State pooling:

$$
u^l_{bpt}=\operatorname{LN}_D\left(\frac{\sum_i\pi^l_{bpti}y^l_{bti}}{\sum_i\pi^l_{bpti}+\epsilon}\right),\quad
\hat u^l_{bpt}=\operatorname{LN}_D\left(\frac{\sum_i\pi^l_{bpti}\hat y^l_{bti}}{\sum_i\pi^l_{bpti}+\epsilon}\right).
$$

Invalid states được zero-filled chỉ để storage an toàn; loss validity mask MUST loại chúng khỏi numerator/denominator. Teacher state detach. `epsilon=1e-6` FP32 PILOT.

### 9.3 Observability proxy

Window `w(t)` mặc định timestamps nằm trong ±160 ms quanh target token; window truncated theo clip boundary. Đây là bidirectional context, không causal future forecasting.

$$
A^l_{bpt}=\frac{\sum_{\tau\in w(t)}\sum_i\pi^l_{bp\tau i}v^l_{b\tau i}}{\sum_{\tau\in w(t)}\sum_i\pi^l_{bp\tau i}+\epsilon}.
$$

`v=True` là actual visible context ở stage. Exclude invalid part support khỏi cả hai tổng. Gate `o=1[A>=0.25]` là PILOT; threshold 0.1/0.25/0.4 sensitivity sau main baseline. Route score detached; không có gradient qua pose, mask hoặc gate.

No-routing B2/B3 dùng `o=1` cho valid state. B4 dùng gate. Generic loss không dùng gate. Local target vẫn cần `q` và validity, không chỉ A.

Coverage theo token tile không xác nhận handshape “recoverable”. Diagnostic phải kiểm tra hypothesis; không nói A=0.25 là đủ semantic information. Wider window có thể chứa một articulatory state khác; kiểm tra ±80/160/320 ms sau pilot nếu routing có gain.

### 9.4 Timestamp pair construction cho change

For each desired horizon δ ∈ {80,160} ms và valid stage time t:

1. Tìm `j>t` gần `time[t]+δ` nhất.
2. Accept khi `abs((time[j]-time[t])-δ) <= max(10 ms,0.25*δ)`.
3. Both states, tracked part identity, source frame support và timestamps phải valid; loại pairs chứa chỉ repeat padding/same source frames.
4. Deduplicate identical `(t,j)` nếu nhiều horizons map cùng pair; giữ desired horizon gần actual gap hơn.
5. Log actual lag histogram và eligible-pair count. Horizon không có pairs bị inactive, không force fake adjacency.

Không hard-code `δ=1/2 tokens`; stages có thể có temporal pooling khác. Local dynamics không được tính trên hai vị trí spatial cố định khi tay di chuyển: use same tracked part's pooled states.

## 10. Loss specification có reduction rõ ràng

### 10.1 Shared helpers

`weighted_valid_mean(e,w)` = sum(e*w)/sum(w), chỉ trả value khi positive weight. Nếu không có valid target: differentiable zero `prediction.sum()*0`, active=false và count=0. Không replace NaN bằng zero rồi coi run healthy.

`balanced_part_mean`:

1. Mean feature dimension D của per-state errors.
2. Within mỗi `(b,p,l)`, weighted mean trên valid t/pairs.
3. Mean giữa active parts trong từng sample/layer.
4. Mean giữa active samples trong layer.
5. Sum enabled layer means với weights renormalized trên active layers.

Part/body có nhiều tokens không được tự có weight lớn hơn. Samples no local targets đóng góp 0 local loss và vẫn generic loss; report local active fraction. DDP reducer thống nhất counts khi per-rank active fractions khác (§11.5).

### 10.2 Generic JEPA loss

Feature-wise **L1** là initial choice lấy precedent R04/R05; Smooth-L1 generic là separate sensitivity. Với normalized teacher grid `y`, prediction `yhat` và visibility v:

$$
L^l_{mask}=\operatorname{Mean}_{b,t,i:v=0}\operatorname{Mean}_D|\hat y^l_{bti}-\operatorname{sg}(y^l_{bti})|.
$$

$$
L^l_{context}=\operatorname{Mean}_{b,t,i:v=1}\operatorname{Mean}_D|\hat y^l_{bti}-\operatorname{sg}(y^l_{bti})|.
$$

$$
L_J=\sum_l b_l\left(L^l_{mask}+\kappa L^l_{context}\right).
$$

Each mean normalized separately; changing mask ratio không chỉ đổi loss magnitude vì số hidden/visible tokens. Target time validity loại padded supports. B0 final stage only, κ=0. B1 enabled mid+final, normalized b=(0.5,0.5), κ=0.05 PILOT. Add B1a context-only-final and B1b multi-layer-mask-only controls nếu cần tách hai effects.

**ADAPTED:** κ/layer weights này không được tuyên bố là exact V-JEPA 2.1 hyperparameters. Faithful official recipe phải đọc actual pinned config và chạy separate row.

### 10.3 Local state loss

Use Smooth-L1 `rho`, beta=1.0, reduction none then mean D. Weight:

```text
w_state = state_valid * state_quality * routing_gate
```

`L_S = balanced_part_mean(rho(predicted_states - detached_target_states), w_state)` theo §10.1. Quality không chứa hand-speed semantic weighting; holds không bị zero loss chỉ vì motion thấp.

### 10.4 Change loss và scale state

Teacher difference `Δu = u[j]-u[t]`. Maintain scalar RMS per `(part,layer,horizon)`:

$$
s^2_{pl\delta}=\operatorname{EMA}_{0.99}\left(E_{valid\ pairs}\operatorname{Mean}_D(\Delta u)^2\right).
$$

`s=max(sqrt(bias_corrected_s2),0.05)` PILOT. Valid teacher pairs đóng góp stats **không phụ thuộc routing gate** để tránh scale confound. Maintain count and EMA update count riêng cho từng active cell. Không update cell khi count=0; FP32, distributed sum/count, all stop-gradient.

$$
d_{t,j}=\frac{u_j-u_t}{\operatorname{sg}(s)+\epsilon},\qquad
\hat d_{t,j}=\frac{\hat u_j-\hat u_t}{\operatorname{sg}(s)+\epsilon}.
$$

Pair weight:

```text
w_pair = pair_valid * q[t] * q[j] * gate[t] * gate[j]
```

Require both endpoints' local coverage gates, an explicit refinement of the initial routing sketch. `L_D` uses balanced part/sample/layer reduction, then mean across active horizons. No pairs => differentiable zero and counters; không divide by fixed |H| khi một horizon inactive.

RMS denominator dùng state **trước update** của optimizer step. Collect teacher sums/counts ở microbatches, apply stats update một lần sau successful optimizer step, giống EMA clock. First step chưa có RMS: use floor hoặc measured detached batch RMS fallback declared in config; default batch RMS for initialization, then EMA. Report difference noise/scale; nếu normalization floor chi phối mọi sample, investigate teacher/lag rather than increase beta.

Change là biến đổi từ state targets, không independent extra ground truth. Loss có thể chỉ reweight errors hoặc amplify jitter; controls §14 là bắt buộc cho contribution claim.

### 10.5 Geometry loss

`L_G` lấy inspiration R01 nhưng targets 2D adapted §6.4. For each target group g, standardize train stats nếu applicable; compute Smooth-L1, mean valid dims/time trong sample rồi balanced mean active groups/samples. Joint/group masks apply cả numerator/denominator.

Do not copy original SignRep target weights/scales sang normalized 2D groups. Initial equal group weights; `gamma=0.1` PILOT sau inspection target and gradient scale. Pose scores chỉ weight group loss, không predict missing targets hoặc feed vào encoder.

Không có adversarial style discriminator trong MVP. Published SignRep giữ original model; nếu reimplement style objective, follow R01 §5.2/A4 với separate optimizer, pair sampler và stability gating, rồi ghi distinct baseline. “Không style loss” là declared adaptation, không faithful SignRep.

### 10.6 VICReg-style variance/covariance branch

Regularizer embeddings `Z=[N_independent_clips,128]`, raw projector outputs FP32, no final normalization. `N>=2`; prefer actual global N>=16. Population/biased variance `unbiased=False` chosen consistently; covariance uses `(N-1)` as declared below. This is project-normalized variant, not byte-for-byte upstream.

$$
L_{var}=\frac1D\sum_j\max\left(0,1-\sqrt{\operatorname{Var}_N(Z_{:,j})+10^{-4}}\right).
$$

$$
C=\frac{(Z-\bar Z)^\top(Z-\bar Z)}{N-1},\qquad
L_{cov}=\frac1D\sum_{i\ne j}C_{ij}^2.
$$

`L_R=L_var+0.04*L_cov`; total eta=0.1 PILOT, later choose from predeclared tiny search/gradient ratios and hold shared. Invariance part của original VICReg không được thêm riêng vì generic JEPA đã là main alignment objective; đặt tên **VICReg-style var/cov regularizer**.

Covariance loss sum off-diagonal rồi chia D, không vô tình chia D² mà giữ weight cũ. N<D rank limit được log; std/rank thấp có thể chỉ batch rank bound, không kết luận collapse từ một batch.

Gradient accumulation thông thường không làm estimator thấy nhiều clips. Với microbatch nhỏ, dùng actual DDP gather hoặc exact reg-gradient cache (§11.6). Không ghi regularizer N=optimizer effective batch nếu chưa thực sự tính trên batch đó.

### 10.7 Optional SIGReg

Replace var/cov bằng official pinned R07 loss, initial API choices theo audited README/class signature; retain same raw projector outputs. Record random slices, integration/test grid, sample batch và seeds. Don't assume SIGReg scale equals L_R; match a declared gradient-scale calibration budget for both variants.

README API reference để Agent tìm đúng implementation: `EppsPulley(num_points=17)` truyền vào `SlicingUnivariateTest(univariate_test=..., num_slices=1024)`. Đây là starting integration reference đã thấy trong README, không là validated sign hyperparameters; check pinned signatures và dtype/device behavior trước gọi.

EMA+SIGReg hybrid là regularizer ablation. Faithful LeJEPA baseline cần own objective/views và no-EMA recipe theo source, được chạy riêng khi có resources. Không claim theoretical guarantees của LeJEPA apply nguyên vẹn cho hybrid.

### 10.8 Total objective và weights

$$
L=L_J+\alpha L_S+\beta L_D+\gamma L_G+\eta L_R.
$$

PILOT starting values: alpha=0.5, beta=0.1, gamma=0.1, eta=0.1. State/change warmup 0→full trong 10% đầu pilot; generic/context/geometry/regularizer settings shared trong controlled chain. Không gọi các numbers này là recommended optimum từ papers.

Log raw và weighted losses, active counts, gradient norms per component trên diagnostic steps. Nếu local term bị scale quá lớn, sửa normalization/target validity trước khi tuning weight. Không tune từng method bằng số trials không bằng nhau.

## 11. Training engine, memory và resume

### 11.1 Optimizer groups

PILOT AdamW: backbone LR 1e-4, new predictor/heads LR 3e-4, weight decay 0.05; biases/norm parameters no decay. AdamW là project choice, khác Adam recipe reference SignRep. Initial comparison không cần layer-wise LR decay; nếu thêm, apply cùng mapping/factor cho controls và log actual param groups.

Warmup 10% successful optimizer updates rồi cosine đến 1e-6 backbone/3e-6 heads. Schedule clock theo optimizer updates, không microbatch count. Không linearly scale LR theo effective batch mà không validation budget.

Gradient clip norm 1.0 sau AMP unscale. BF16 nếu GPU support; otherwise FP32 hoặc verified FP16+GradScaler. LayerNorm targets, covariance, RMS/gather stats luôn FP32. Debug path dropout/drop-path=0; pilot drop-path 0.1 only after source/native behavior audited. Teacher always eval, no stochastic drop-path.

### 11.2 EMA teacher

Deepcopy online encoder sau checkpoint load; initial weights identical. Teacher không có predictor/head/projector. `requires_grad_(False)`, `eval()`, no optimizer group.

PILOT cosine momentum từ 0.996 đến 0.9999 trên declared training horizon:

$$
m(k)=m_{end}-(m_{end}-m_{start})\frac{1+\cos(\pi k/K)}2.
$$

Sau successful optimizer step: `theta_teacher = m*theta_teacher + (1-m)*theta_online`. Named parameter mapping asserted. Floating running buffers policy explicit: copy online buffers hoặc EMA như audited source; integer counters copy. Hiera LayerNorm không cần BN running stats, nhưng adapter không được giả mọi backbone đều vậy.

Nếu AMP skip optimizer step do overflow: không advance LR/EMA/RMS/global optimizer step; log attempted/skipped count, excessive skip => fail numerical gate. Resume teacher và momentum schedule, không recreate teacher từ student để giả continuing.

### 11.3 One optimizer update: correct order

```python
# Algorithm contract, not runnable upstream code.
optimizer.zero_grad(set_to_none=True)
step_stats = empty_detached_teacher_difference_stats()

for micro in optimizer_batch:
    rgb_student, rgb_teacher, sidecars = paired_views(micro)
    visibility = mask_from_stateless_seed(micro.ids, step, micro.index)

    with no_grad():
        targets = teacher.encode_full(rgb_teacher, selected_stages)
        target_grid = normalize_targets(targets)

    contexts = online.encode_masked(rgb_student, visibility)
    predicted = predictor(contexts, target_grid_metadata_only(targets))
    # metadata contains only canonical grids and timestamps, never target features
    local = build_loss_sidecars(sidecars, contexts.stage_meta, visibility)

    loss_rgb_local = compute_J_S_D_G(predicted, target_grid, contexts, local)
    z = reg_projector(pool_visible(contexts.final_stage))
    loss = normalized_micro_objective(loss_rgb_local, z, reg_batch_mode)
    backward_with_amp(loss)
    step_stats += detached_teacher_stats(target_grid, local)

unscale_if_needed()
assert_finite_gradients()
clip_grad_norm(online_and_heads, 1.0)
stepped = optimizer_step_with_overflow_status()
if stepped:
    scheduler.advance()
    update_encoder_ema()
    update_change_rms_once(step_stats)
    global_optimizer_step += 1
```

`target_grid_metadata_only` là typed interface chỉ grid/timestamp metadata; test cấm tensor feature accidental passthrough. Loss implementation độc lập khỏi teacher/predictor class để dễ control.

Geometry/state/change weights warmup tính theo same global step. Validation không update optimizer, teacher hoặc RMS. Initial RMS bootstrap record separately.

### 11.4 Three batching quantities phải log

1. `micro_batch_per_rank`: clip count của một forward.
2. `optimizer_batch`: micro_batch × accumulation × world_size.
3. `regularizer_batch`: actual unique clip/source embeddings mà one statistics evaluation nhìn thấy.

Plain accumulation mode tính L_R từng microbatch => regularizer batch là micro_batch×world_size, dù optimizer batch lớn hơn. Không dùng temporal tokens/frame count để tăng N giả.

### 11.5 DDP normalization và collectives

Teacher replicated outside trainable DDP wrapper; all ranks start identical. Actual masks/clip indices deterministic theo rank/step; DDP sampler avoid unintended padding duplicate samples trong main train statistics.

Generic/local/geometry losses có số valid targets khác nhau giữa ranks. Để DDP average gradients đúng global normalized objective, dùng local differentiable numerator `S_r` và detached global denominator `N=sum_r N_r`:

```text
loss_for_backward_on_rank_r = world_size * S_r / N
reported_global_loss = sum_r detached(S_r) / N
```

Apply ở reduction level thực của từng term: generic valid tokens; local sample/part balanced sums; geometry active samples. Enabled layer/horizon weights renormalize bằng global active flags. N=0 => graph-connected zero trên mọi rank. Khi accumulation có unequal valid counts, denominator tính cho **whole optimizer batch** nếu muốn exact full-batch mean; precompute masks/valid weights, hoặc declare micro-mean approximation và dùng chung. Default exact normalization.

Regularizer dùng differentiable gather của embeddings, không plain detached gather; all ranks evaluate same full-stat objective. Gradient scale của gather backward/standard DDP averaging phải test. Không thêm world-size division/multiplication máy móc vào loss global đã gathered.

Tests: single-process full batch vs 2-rank Gloo gradients trên tiny model với unequal valid counts, rồi GPU smoke. `DDP.no_sync` cho accumulated forwards khi thích hợp; deadlocks/final sync/unused branches phải test. Inactive heads trong B0–B4 cần explicit unused-parameter policy, không bỏ grad warnings.

### 11.6 Optional exact regularizer gradient cache

**Engineering precedent R18:** Gao et al., *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup*:

- Paper: https://arxiv.org/abs/2101.06983
- Code: https://github.com/luyug/GradCache

Kế thừa kỹ thuật decouple embedding-loss gradient; adaptation ở đây áp dụng cho var/cov hoặc SIGReg, không claim contribution. Không assume package GradCache drop-in cho multi-loss Hiera; implement small audited adapter hoặc original two-pass routine.

Khi microbatch nhỏ nhưng muốn regularizer thấy N=16+ independent clips:

1. Fix toàn bộ optimizer-batch samples, views, masks và RNG states.
2. Pass A: online encoder + projector **no-grad**, thu `z_cache` cho mỗi microbatch; không EMA/RMS/optimizer update.
3. Concatenate detached cache as requires-grad leaf; compute actual full-batch `eta*L_R`; obtain `g_z=d(eta*L_R)/dz_cache`.
4. Pass B: recompute từng microbatch với đúng input, mask, dropout/drop-path RNG và unchanged weights; tính ordinary J/S/D/G gradients và surrogate `sum(z_recomputed*g_z_detached)`.
5. Ordinary batch-local losses dùng exact batch normalization; surrogate **không chia thêm accumulation factor**, vì g_z đã có full-batch reduction và eta.
6. One optimizer step sau mọi replay; EMA/RMS update một lần; restore RNG advance state để không tiêu thụ dropout stream hai lần một cách âm thầm.

Không BN running stats trong projector; nếu backbone có stateful modules phải freeze/handle buffer update equivalence. First/replay outputs phải identical trong declared precision tolerance. Không thay augmentations ở replay. Cache không giữ encoder autograd graphs của Pass A.

DDP: full-cache loss/gather gradient contract như §11.5; surrogate scale phải test against true full-batch gradient, không đoán. Cache gradients tính FP32; second-pass AMP surrogate xử lý casting/GradScaler nhất quán.

MUST tests trên tiny model: full-batch regularizer backward vs cache gradient relative error <=1e-5 FP32, dropout replay, two-rank equivalence, interrupted-step rollback. Nếu không pass, dùng direct gather/smaller actual batch và khai báo limitation; không pretend exact regularizer batch.

Trade-off: thêm encoder forward và RNG orchestration; report measured cost. Không claim “free large batch”. Gradient cache optional sau M4 direct-loss correctness, không che lỗi trong core algorithm.

### 11.7 Checkpoint state

Full checkpoint fields MUST include:

```text
schema_version
online_encoder, ema_encoder, predictor, geometry_head, regularizer_projector
optimizer, scheduler, scaler_if_used
global_optimizer_step, attempted_steps, clip_exposures
change_rms_state/counts/bias_correction
python/numpy/torch_cpu/torch_cuda_rng_states
sampler_state_or_stateless_batch_cursor
resolved_config, config_hash, source_lock_hash, data_lock_hash
model_stage_metadata, detector/geometry_schema_ids
best_validation_state_and_selection_rule
environment_and_world_size
```

Atomic write temp→rename, last checkpoint không corrupt nếu interrupted. Save only complete optimizer-step boundary; cache microstep resume unsupported thì rollback step with same seeds. Keep rotating last/best plus periodic checkpoints theo disk budget.

Strict resume requires matching data/source/config/world-size clocks; allowed path relocation không đổi data identities. Nếu thay world size/batch/objective, gọi **warm-start/forked run**, không exact resume. Optimizer state được restore lên actual device; teacher/RMS không reset.

### 11.8 Resource measurement và bounded execution

M6 benchmark 20 warmup + 200 timed optimizer updates, same precision/batch/cache mode. Measure GPU peak allocated/reserved memory, seconds/update, clips/sec, data wait, cumulative exposures, GPU-hours và preprocessing time riêng.

Use existing available hardware; không tự provision paid cloud instances. `max_updates`, `max_gpu_hours`, `disk_budget_gb` và `max_concurrent_runs` là config budgets; stop at boundary, save checkpoint, emit `budget_exhausted` có evidence. Một example budget không là claim hardware feasibility.

OOM policy: checkpointing/query chunking trước; reduce microbatch + validated cache/accumulation sau; rồi reconsider selected stage/resolution. Mọi config change tạo new run và đồng bộ baseline settings; không âm thầm giảm resolution chỉ proposal. Không tự tăng model size khi chưa có mechanism gain.

## 12. Frozen feature extraction và retrieval

### 12.1 Two evaluation protocols

**`published_signrep_protocol`:** audit R01 §6/§7.1 và original Model/transform; use correct activity head/readout, windowing, gallery/query split. Đây là replication attempt của published checkpoint.

**`controlled_uniform_protocol`:** all compatible encoders dùng deterministic RGB preprocessing và common video-level aggregation. Dùng cho B0–B4, Hiera init và SignRep checkpoint re-evaluation. Protocol differences với published numbers phải có column riêng.

Never change uniform vs activity-weighted aggregation chỉ cho proposal. Different encoder native frame length/resolution được báo trong foundation-transfer group, không coi objective-only comparison.

### 12.2 Default controlled extraction

1. Encoder `eval`, all params frozen, no-grad.
2. Deterministic preprocessing; sample declared temporal policy.
3. Sliding windows length 16, window step 2 sampled frames; short-video padding policy explicit.
4. Per-window raw embedding = mean unmasked final-stage valid tokens, before L2.
5. Video embedding = uniform mean raw window embeddings, then L2 normalize once.
6. Zero-norm embedding => error/flag, không arbitrary unit vector.

This chooses raw-mean-then-L2; mean of individually normalized windows is different protocol and must not be mixed. Holds/resting windows remain included in uniform control. Published activity-weighted readout handled separately.

Main encoder selection online; EMA output optional secondary row locked before test. Geometry/predictor heads not needed for controlled inference.

### 12.3 Feature cache identity

Cache rows: sample_id, embedding float32, valid/extract status; sidecar metadata gồm encoder checkpoint hash, online/EMA, readout, preprocessing, frame/window policy, source/data/config hashes, feature dim, extractor version.

Cache key không chỉ filename/experiment name. Hash change => invalidate, không reuse old features because file exists. Extraction deterministic shard ordering; write sample IDs để tránh labels bị lệch khi join.

### 12.4 Gallery/query policy

Default controlled gallery = labelled **train split**, queries = val/test split; this is a declared proposal protocol. Validate original benchmark exact gallery policy trước reproduction; không presume paper dùng default này.

Exclude same sample/source duplicate from query-gallery where appropriate; report signer overlap policy. ASL-Citizen official train/test signer disjoint must persist. Query class absent in gallery ghi separately; report both all-query metric (missing class counted fail) và covered-class diagnostic, không chỉ bỏ query rồi tăng score.

Gallery labels dùng scoring, không update encoder; wording chính xác: “no downstream encoder training”, không “system uses no labels anywhere”. Class namespaces gồm language/dataset; không merge gloss strings khác sign language.

### 12.5 Scoring và metrics

Cosine scores `S=Q @ G.T` với L2 normalized vectors. Compute streaming query/gallery chunks, never allocate toàn huge score matrix nếu memory không đủ.

Controlled dictionary class score:

$$
s(q,c)=\max_{g:\operatorname{label}(g)=c}\langle q,g\rangle.
$$

Rank **unique classes**, stable tie-break theo class ID. R@K = fraction queries whose true class in top-K unique classes. Top-K videos có duplicate classes là metric khác; phải đặt tên khác nếu báo.

MRR = mean reciprocal rank of true class; absent true class rank infinite => 0. DCG with binary class relevance chỉ báo khi định nghĩa/mapping match official benchmark; otherwise omit để tránh gọi sai metric. Record macro-per-class và micro-query metrics riêng.

Published reproduction có thể dùng khác voting/aggregation; audit rồi separate config. Do not choose max vs class-mean scoring on test.

### 12.6 Retrieval tests

- Toy gallery có duplicated classes: top-K unique-class result đúng.
- Identical normalized vectors/self-match exclusion/tie ordering.
- Streaming scores identical dense tiny reference.
- Missing-class queries không bị drop âm thầm.
- Feature↔label join bằng sample ID, không array position chưa audit.
- No train/val/test source intersection under declared regime.

## 13. Recognition ít nhãn, diagnostics và transfer

### 13.1 Label-efficient frozen probe

Predeclare train fractions 1%, 5%, 10%, 100%; stratified clip selection theo class từ official train split, shared indices giữa methods và three subset seeds khi resources cho phép. `min_one_per_class` có thể khiến actual fraction > requested ở dataset nhỏ: record actual clips/class coverage, không gọi exact 1% khi không đạt.

Main probe linear classifier trên frozen video embeddings. Normalize/standardize bằng train subset stats, không val/test. CE class-balanced hoặc unweighted policy fixed all methods; report both per-instance và per-class Top-1/Top-5.

PILOT probe tuning: LR {1e-3,1e-2}, WD {0,1e-4}, max 100 epochs, patience 10 trên validation. Same trial budget/seeds/early stopping cho mọi encoder. Separate probe tuning from SSL checkpoint selection; labels used for model selection must be counted in supervision disclosure.

Attentive temporal probe và full fine-tuning là separate rows, same architecture/data budgets. Không gọi attentive probe “linear evaluation”. Không dùng fine-tuning gain đơn độc chứng minh frozen representation mechanism.

### 13.2 Phonological diagnostics

Join ASL-Citizen bằng ASL-LEX Code, not English gloss fuzzy matching. Construct annotation-neighbor pairs trước method comparison; split diagnostic construction/threshold selection train/val vs final test.

Evaluate retrieval confusion/discrimination theo handshape, location và movement similarities. Pairs gần nhau theo annotation chỉ gọi **phonological neighbors**; claim “minimal pairs” cần native signer/linguistic verification thực sự, report annotation procedure nếu có.

Hypothesis evidence: routing/state/change giảm confusion đúng thuộc tính, không chỉ average R@1. Ngược lại, global gains without local diagnostic gains làm yếu mechanism claim.

### 13.3 Observability/quality/resolution buckets

Fixed buckets từ validation/train-defined thresholds: token-visible ROI fraction, detector validity, hand pixel occupancy, low/high motion và hold. Thresholds chung mọi encoder, query perturbation masks/common seed nếu so masked diagnostics.

For plain RGB inference test, observability analysis có thể dùng predefined artificial occlusion hoặc training diagnostic clips; đừng áp mask training lên published unmasked retrieval rồi coi same main metric. Separate masked diagnostic và normal test tables.

Low A không xác nhận semantic uncertainty; log confound quality/motion/ROI size. Method cần giữ hold và face-sensitive groups, không chỉ high-motion hands.

### 13.4 Shortcut và collapse checks

- Foreground/background intervention với same segmentation/crop rule cho mọi method; detector inaccuracies report.
- Identity/background supervised probes trên train/val với encoder frozen, diagnostic-only; identity predictability thấp không tự là good semantic representation.
- Mask-pattern-only probe: dùng visibility/coverage metadata mà không RGB; nếu predictor inputs/masks leak sign, investigate. Evaluation probe labels không đi vào SSL.
- Effective rank, per-feature std và cosine similarity trên **fixed multi-source held-out unlabelled bank**; per-part pooled representation kiểm tra riêng. Batch rank limit N<D không tự là collapse.
- Local loss giảm nhưng per-part states constant across clips => branch collapse; global regularizer có thể không ngăn local collapse.
- Time-order corruption và horizon analysis; không áp time-reversal invariance nếu sign meaning có thể thay đổi.

### 13.5 Continuous transfer, optional M10

Chọn một continuous recognition hoặc translation benchmark có available data/protocol. Encoder frozen và temporal readout/decoder giống giữa methods. WER cho recognition; BLEU/ROUGE/chRF theo task/protocol thật cho translation. Record labels/text used **downstream**, không rewrite core SSL supervision claim.

SHuBERT là related contextual comparator; chỉ integrate official code/config khi available. Long-context SSL hoặc VL-JEPA text alignment là extension, không thêm vào MVP để cứu pilot bằng confounded supervision.

Translation decoder có thể che lỗi encoder: add video corruption/visual-sensitivity diagnostic, same decoder capacity and tuning. Missing M10 does not block code export, nhưng paper không được claim continuous transfer evidence khi M10=`not_run`.

### 13.6 Statistical reporting

Main comparisons: at least three training seeds for selected candidates, mean/variation và effect sizes. Paired bootstrap cùng query IDs; signer/source-cluster bootstrap khi appropriate. Query bootstrap không thay training-seed variability.

Predeclare main endpoint và report all chosen metrics. CI/gain phải từ actual predictions, not simulated target values. Một seed pilot chỉ là screening evidence. Khi effect nhỏ, thêm seed/control trước scale backbone.

## 14. Experiment matrix và vòng lặp research

### 14.1 Baseline groups

| Group | Runs | Mục đích |
|---|---|---|
| Published checkpoints | SignRep; available author checkpoints | External quality anchors; protocol/source rõ |
| Controlled Hiera | Same MAE-K400 init + same sign data, B0–B4 | Objective/mechanism effects |
| Dense recipe controls | B1a/B1b; optional official V-JEPA-inspired adaptation | Tách dense/deep gains khỏi proposed local terms |
| Foundation transfer | V-JEPA 2.1 B/L at native config; optional image SSL | Current foundation feature quality; khác data/compute |
| Label-supervised pretraining | SignSeek/SignCLIP nếu integrate được | Quality comparator với labels/text disclosed |

Frozen Hiera MAE init là mandatory cheap anchor. Continuing pixel-MAE là strong control khi actual upstream video-MAE factory/trainer khả dụng; audit native output/mask/targets, không giả target type. Không thay missing pixel-MAE implementation bằng dummy loss để gọi baseline.

### 14.2 Main controlled chain

Same initialization bytes, manifest/exposures, sampler/mask seeds, augmentations, batch statistics, predictor/geometry head capacity, optimizer schedule, readout và trial budget.

| Run | Generic loss | Local state | Change | Routing | Geometry/regularizer |
|---|---|---|---|---|---|
| B0 | Final masked only | 0 | 0 | Off | Shared G/R |
| B1 | Mid+final masked + context | 0 | 0 | Off | Shared G/R |
| B1a | Final masked + context | 0 | 0 | Off | Shared G/R |
| B1b | Mid+final masked only | 0 | 0 | Off | Shared G/R |
| B2 | B1 | On | 0 | Off | Shared G/R |
| B3 | B1 | On | On | Off | Shared G/R |
| B4 | B1 | On | On | Coverage gate | Shared G/R |

Keep the same registered modules across chain, deactivate losses via config; log active vs total parameter counts. Backbone context memories shared across runs. Inactive head parameters không cần train, nhưng allocated capacity/readout khác biệt phải disclosed. B0–B4 không được tự chọn detector/resolution khác nhau.

### 14.3 Mechanism controls

| ID | Implementation | Điều loại trừ |
|---|---|---|
| C1 | B2 với alpha tăng theo predeclared gradient/weight budget, no change | Gain do thêm prediction pressure |
| C2 | B2 state errors weighted `1 + lambda*clipped_teacher_change_magnitude`, detached, giữ weight floor cho hold | Change chỉ là temporal reweighting |
| C3 | B3 with teacher difference targets shuffled across valid examples within same part/layer/horizon; same marginal scale | Gain không cần actual temporal pairing |
| C4a | B4 replace state/pair gates bằng random retention matching valid-target counts per sample/part/layer và quality bucket | Gain do ít target/easier optimizer budget |
| C4b | Shared independent masks + compare/reweight fixed coverage buckets, không sửa mask dựa pose | Gain do context coverage distribution khác |
| C5a | B4 gamma=0 | Dependence on geometry anchor |
| C5b | B4 valid flags only, no quality weights | Detector score/selection effect |
| C6 | B4 var/cov ↔ SIGReg with shared tuning budget | Regularizer là nguồn gain chính |
| C7 | Geometry pooled head ↔ dense temporal head, all local losses off | Decoder/head bottleneck |
| C8 | 16↔32 frames hoặc 224↔larger, applied to strong baseline and proposal | Gain do temporal/spatial resolution |

C4a state-point budget và change-pair budget match riêng; independent random gates là budget control, không giả cùng endpoint-gating mechanism. C4b không dùng pose-conditioned mask generator để tránh side-channel. All main runs share fixed random-mask schedule, nên coverage counts có thể được kiểm tra chính xác theo sample IDs.

Không chạy toàn combinatorial grid. Initial one-seed B0–B4; nếu tín hiệu tốt thì B1/B2/B4 three-seed + C1/C2/C4. Add other controls theo failure mode và available budget; critical missing control được ghi là unresolved claim.

### 14.4 Research loop và go/no-go

1. **Data/target loop:** fix swaps, timestamps, scales, ROI coverage. No method tuning để che preprocessing lỗi.
2. **Baseline loop:** extract/check readout; controlled B0/B1 stable. Missing exact SignRep recipe => adaptation label, không superiority to original training.
3. **Mechanism loop:** screen state/change/routing trên validation và diagnostic; simplify components không vượt controls.
4. **Confirmation loop:** selected three-seed runs, low-label và cross-dataset; freeze selections trước final test.
5. **Scale/transfer loop:** chỉ scale còn gain và clear mechanism, hoặc pivot về efficiency nếu quality tương đương với lower compute.

PILOT project criterion có thể là ≥2 percentage points frozen R@1 trên hai validation benchmarks, hoặc smaller stable gain với clear local diagnostic/efficiency evidence. Đây là decision threshold đề xuất, không performance forecast/publication guarantee.

Stop contribution claim khi:

- C1/C2 match change gain: remove/de-emphasize L_D.
- C4 match routing gain: mechanism chưa được chứng minh, lower claim.
- Gain chỉ ở one seed hoặc do data/init/readout khác: cần control/confirmation.
- Local branch collapse hoặc teacher không encode fingers: inspect resolution/geometry trước tăng loss weight.
- Routing chỉ loại hard/low-quality examples, hard-group metrics tệ hơn: redesign gate hoặc bỏ routing.
- Proposal không vượt strong baseline: report negative result; không fabricate superiority.

Sau khi final test đã đọc, không dùng nó để chọn new beta/gate/readout. A new exploratory iteration phải được ghi exploratory và có validation/holdout strategy mới trước confirmatory claim.

## 15. Config và CLI contract để tự động hóa

### 15.1 `configs/base_pilot.yaml` template

Agent MUST tạo schema và validate config. `null` source/checkpoint/stage fields được resolver điền ở M0; real training reject nếu còn unresolved. Dưới đây là PILOT design, không tested runtime recipe.

```yaml
schema_version: 1
experiment:
  name: asc_jepa_B4_pilot
  seed: 17
  stage: sanity
  research_regime: controlled_label_free_2d_adaptation
  source_snapshot: "2026-10-06"
paths:
  source_lock: artifacts/sources.lock.json
  checkpoint_lock: artifacts/checkpoints.lock.json
  data_lock: artifacts/data_manifest.lock.json
  train_manifest: manifests/pretrain_train.jsonl
  ssl_val_manifest: manifests/pretrain_val.jsonl
  run_root: runs
  cache_root: artifacts/features
data:
  clip_frames: 16
  sampling_policy: fixed_rate
  target_fps: 25.0
  image_size: 224
  read_captions: false
  read_class_labels: false
  photometric_policy: mild_independent
  geometric_policy: shared_rgb_fullframe
  horizontal_flip: false
  time_reverse: false
  interpolated_pose_targets: false
model:
  backbone: hiera_base_16x224
  checkpoint_key: mae_k400
  checkpoint_path: null
  selected_stages:
    mid: null
    final: null
  stage_metadata_path: artifacts/audits/architecture_probe.json
  mask_mode: sparse_native
  hidden_ratio: 0.8
  pose_conditioned_mask: false
  drop_path: 0.1
predictor:
  architecture: shared_memory_cross_attention
  hidden_dim: 256
  memory_blocks: 2
  attention_heads: 8
  mlp_ratio: 4.0
  query_chunk_size: 256
  dropout: 0.0
  inputs: [visible_rgb_tokens, grid_positions, timestamps, stage_id]
geometry:
  schema: mediapipe_2d_adapted_v1
  schema_path: artifacts/geometry_schema.json
  normalization_stats_path: artifacts/geometry_train_stats.json
  head: pooled_context_time_mlp
  hidden_dim: 512
  use_pseudo_depth_as_3d: false
loss:
  generic_distance: l1
  target_feature_layer_norm: true
  generic_layers: [mid, final]
  generic_layer_weights: [0.5, 0.5]
  context_weight: 0.05
  state_weight: 0.5
  change_weight: 0.1
  geometry_weight: 0.1
  regularizer_weight: 0.1
  local_smooth_l1_beta: 1.0
  local_weight_warmup_fraction: 0.1
  local_reduce: sample_part_layer_balanced_valid_mean
routing:
  mode: hard_coverage
  threshold: 0.25
  window_radius_ms: 160.0
  min_roi_weighted_tiles: 1.0
  confidence_policy: valid_flags
  generic_loss_on_all_clips: true
change:
  horizons_ms: [80.0, 160.0]
  lag_tolerance_fraction: 0.25
  lag_min_tolerance_ms: 10.0
  pair_gate: both_endpoints
  rms_momentum: 0.99
  rms_floor: 0.05
  rms_bootstrap: valid_batch_rms
  rms_update_per_optimizer_step: true
regularizer:
  kind: vicreg_style_var_cov
  projector_hidden_dim: 512
  projector_output_dim: 128
  final_normalization: none
  variance_floor: 1.0
  variance_epsilon: 0.0001
  covariance_weight: 0.04
  batch_mode: grad_cache
  require_gradient_equivalence_test: true
train:
  optimizer: adamw
  backbone_lr: 0.0001
  heads_lr: 0.0003
  weight_decay: 0.05
  warmup_fraction: 0.1
  schedule: cosine
  min_backbone_lr: 0.000001
  micro_batch_per_rank: 4
  accumulation_steps: 4
  max_updates: 2000
  precision: bf16_if_supported
  gradient_clip_norm: 1.0
  exact_valid_count_normalization: true
  ema_start: 0.996
  ema_end: 0.9999
  teacher_mode: eval
  checkpoint_every_updates: 500
  validate_every_updates: 500
  log_every_updates: 20
evaluation:
  encoder: online
  readout: raw_uniform_window_mean_then_l2
  window_frames: 16
  window_step: 2
  gallery_split: train
  query_split: val
  class_score: max_gallery_cosine
  primary_metric: frozen_micro_recall_at_1
  checkpoint_selection: fixed_schedule
  test_selection: forbidden
budget:
  max_gpu_hours: 24.0
  disk_budget_gb: 100.0
  max_concurrent_runs: 1
  use_existing_hardware_only: true
```

`max_gpu_hours`/disk values là example caps, override theo actual resources. 24 giờ không đảm bảo hoàn thành all experiments. `sanity` config max_updates=2K; pilot/confirmation create own schedule horizon 20K/50–100K trước run, không extend K âm thầm khi EMA/LR schedule đã kết thúc.

`checkpoint_selection=fixed_schedule` là default label-free selection. Alternate downstream-validation selection có same tuning budget all methods và supervision disclosure, đặt separate protocol. Primary validation endpoint dùng screening; final test không có vai trò chọn checkpoint.

### 15.2 Config overlays

Agent implement overlays bằng typed deep-merge, reject unknown keys. Resolved config stored immutable mỗi run; hash includes all inherited values.

```yaml
# B0 overlay
experiment:
  name: B0_final_masked
loss:
  generic_layers: [final]
  generic_layer_weights: [1.0]
  context_weight: 0.0
  state_weight: 0.0
  change_weight: 0.0
routing:
  mode: off
```

```yaml
# B1 overlay
experiment:
  name: B1_dense_deep
loss:
  state_weight: 0.0
  change_weight: 0.0
routing:
  mode: off
```

```yaml
# B2 overlay
experiment:
  name: B2_local_state
loss:
  change_weight: 0.0
routing:
  mode: off
```

```yaml
# B3 overlay
experiment:
  name: B3_state_change
routing:
  mode: off
```

B4 dùng base local settings. Pilot config overlays stage=pilot/max_updates=20K; confirmation uses declared seeds/schedule. B1a/B1b/controls generate explicit configs, không patch Python code giữa runs.

### 15.3 CLI yêu cầu Agent implement

Các lệnh `asc_jepa` dưới đây là **project interface specification**, không phải lệnh hiện có trong upstream repositories. Agent phải implement và test `--help`/dry-run trước sử dụng.

```bash
python -m asc_jepa.cli audit-sources --source-lock artifacts/sources.lock.json
python -m asc_jepa.cli resolve-checkpoints --registry artifacts/checkpoints.lock.json
python -m asc_jepa.cli build-manifest --config configs/data.yaml
python -m asc_jepa.cli validate-manifest --manifest manifests/pretrain_train.jsonl
python -m asc_jepa.cli audit-leakage --config configs/protocol.yaml
python -m asc_jepa.cli preprocess-pose --config configs/preprocess.yaml
python -m asc_jepa.cli fit-geometry-stats --split pretrain_train --config configs/preprocess.yaml
python -m asc_jepa.cli inspect-targets --config configs/preprocess.yaml --sample-count 200
python -m asc_jepa.cli probe-architecture --config configs/base_pilot.yaml
python -m asc_jepa.cli resolve-config --config configs/base_pilot.yaml --output artifacts/resolved_sanity.yaml
python -m asc_jepa.cli benchmark --config artifacts/resolved_sanity.yaml --warmup-updates 20 --timed-updates 200
python -m asc_jepa.cli train --config artifacts/resolved_sanity.yaml
python -m asc_jepa.cli extract --config configs/eval_controlled.yaml --split val
python -m asc_jepa.cli retrieval --config configs/eval_controlled.yaml --split val
python -m asc_jepa.cli probe --config configs/low_label.yaml
python -m asc_jepa.cli diagnostics --config configs/diagnostics.yaml
python -m asc_jepa.cli run-study --study configs/study.yaml --until M8
python -m asc_jepa.cli freeze-selection --study configs/study.yaml
python -m asc_jepa.cli evaluate-locked-test --study configs/study.yaml
python -m asc_jepa.cli export --config configs/export.yaml
python -m asc_jepa.cli report --study configs/study.yaml
```

`train --resume PATH` strict resume policy §11.7. `run-study --dry-run` phải liệt kê dependencies, unresolved data/checkpoints, estimated compute và exact planned commands; dry-run không đánh dấu task done hoặc tạo scores.

Implement DDP launch với `torchrun` vào project train entrypoint theo installed PyTorch args; không gọi official `app.main` cho Hiera config của project. Official V-JEPA commands chỉ dùng khi chạy native official experiment/config riêng.

### 15.4 `configs/study.yaml` và run registry

```yaml
schema_version: 1
study_id: asc_jepa_controlled_v1
shared:
  base_config: configs/base_pilot.yaml
  source_lock: artifacts/sources.lock.json
  data_lock: artifacts/data_manifest.lock.json
  checkpoint_lock: artifacts/checkpoints.lock.json
screening:
  seeds: [17]
  updates: 20000
  runs: [B0, B1, B2, B3, B4]
confirmation:
  seeds: [17, 29, 43]
  updates: 50000
  candidates: [B1, B2, B4]
  controls: [C1, C2, C4a, C4b]
  launch_only_if_screening_passes: true
selection:
  use_test: false
  primary_validation_endpoint: frozen_micro_recall_at_1
  fixed_checkpoint_schedule: true
  require_matched_protocol: true
runtime:
  max_concurrent_runs: 1
  stop_on_budget_exhaustion: true
```

Candidate list trong template là planned, không kết luận chúng tốt. Runner chỉ launch dependencies đã pass và remaining budget đủ theo measured estimate. Identical config/data/checkpoint/seed đã có complete run thì reuse; incomplete run chỉ strict resume nếu compatible. Never overwrite completed run artifacts.

`runs/registry.jsonl` per-run fields: run_id, config/source/data/init hash, seed, stage, status, hardware, actual steps/exposures, start/end, checkpoints, metrics paths, selection rule, compute, failure/block reason, evidence paths.

Metrics files bắt buộc có sample/prediction provenance. Planned result table placeholders dùng `null`/`not_run`, không numeric zeros có thể bị đọc như actual result.

## 16. Verification suite và acceptance gates

### 16.1 Tests có ý nghĩa cho algorithm này

| Test | Fixture/expected property | Gate |
|---|---|---|
| Grid mapping | Coordinate sentinel và inverse packing | M3 |
| Mask semantics | Fixed ratio/count; True visible; no teacher passthrough | M3 |
| Predictor query chunking | Full vs chunked output+gradient match | M3 |
| ROI affine | Known crop/resize transforms map bbox/joints correctly | M2/M4 |
| Weighted reduction | Duplicate spatial tiles không tăng part weight; missing groups excluded | M4 |
| All-invalid local | Graph-connected 0, finite backward, generic gradients remain | M4 |
| No pose input leakage | Same RGB/mask but different ROI → identical predicted grid, possibly different loss | M4 |
| Stop-gradient | Teacher/ROI/gate/RMS no grads; online+predictor have grads | M4 |
| Change time mapping | Irregular FPS/repeats/invalid lags, no fake adjacency | M4 |
| Hold preservation | Zero-motion valid part still has state supervision | M4 |
| RMS clock | One update per successful optimizer step, resume identical | M4/M6 |
| Regularizer covariance | Off-diagonal reduction matches tiny analytic/reference matrix | M4 |
| DDP normalization | Unequal valid-count full-batch gradient reference | M4 |
| Gradient cache | Direct vs replay gradient, stochastic RNG and world-size scaling | M4 optional mode |
| Resume | N uninterrupted steps vs K+resume+(N-K) with same seeds/config | M6 |
| Gallery/query protocol | Missing class, duplicated gallery class, tie/self-match and streaming | M5 |
| Label namespace | Same gloss from different languages not silently merged | M1/M5 |
| Feature cache invalidation | Encoder/readout/manifest hash changes invalidate cache | M5 |
| RGB-only export | Runs without pose package/sidecars and matches eval encoder | M11 |

FP32 small fixtures SHOULD use atol/rtol 1e-5; stochastic/AMP/native kernels choose realistic documented tolerance. Equality outside expected precision is a bug investigation, không lấp bằng huge tolerance.

### 16.2 Sanity training M6

Actual unlabelled available sample subset; debugging overfit small set không là downstream gain. Check finite losses/gradients, teacher-online drift, local active fractions, ROI occupancy, change scale và fixed-bank representation diversity. Loss giảm không đủ pass scientific improvement gate.

M6 pass technical gate khi:

- No schema/timestamp/gradient leaks hoặc unexplained NaN.
- Correct successful-step/EMA/RMS/resume clocks.
- Generic/local branches có active supervision và measurable gradients where enabled.
- No evident branch collapse trong diagnostics; nếu nghi ngờ thì mark unresolved và investigate.
- Real memory/throughput report và planned experiment cost.

Không đặt universal loss target hoặc bắt pretext loss →0; teacher mục tiêu cùng học và ambiguity không cho phép suy như supervised regression cố định.

### 16.3 Failure handling table

| Failure | Agent action |
|---|---|
| Missing source/checkpoint | Mark dependent task blocked; continue independent code/tests; no random substitute in paper table |
| Incompatible state dict | Audit actual constructor/config; explicit conversion report; no silent strict=False |
| Sparse token mapping wrong | Stop training, fix inverse mapping/sentinel tests |
| ROI too coarse | Inspect stage/resolution; same change for relevant controls |
| Many no-pair horizons | Fix FPS/support policy or disable unsupported horizon with declared config |
| Detector hand swap/noisy pose | Flag validity/association; generic branch remains; audit quality |
| AMP overflow | Skip clocks together; persistent => lower precision/LR after numerical diagnosis |
| OOM | Chunk/checkpoint/cache then bounded config fork; keep matched comparisons |
| Corrupt train video mid-run | Log and repair/version manifest; don't silently replace only one run's samples |
| Baseline differs from published | Check sample IDs/protocol/readout; call subset/adaptation replication honestly |
| No gain after controls | Simplify method, pivot scope or negative result; no invented gain |
| Compute/disk cap | Save full checkpoint/status; stop further runs with clear pending tasks |

## 17. Research artifacts, claims và final handoff

### 17.1 Report tables Agent phải xuất từ actual files

- Source/component provenance table: paper, code SHA, target/head change, checkpoint supervision.
- Data protocol table: splits, available counts/classes/signers, exclusions, train hours, eval gallery policy.
- Baselines table: init/data/labels/resolution/clip length/readout/compute/trials.
- B0–B4+controls with seed means/variation, R@1/R@5 and low-label results.
- Diagnostic table: phonology, hold/motion, face, coverage/quality, rank, failure cases.
- Efficiency table: actual GPU-hours, memory, clips/sec, preprocessing, cache overhead.
- Missing/not-run table; no compare incompatible rows as same protocol.

`results.csv` MUST be generated from run metrics JSON, not hand-entered paper numbers mixed with actual experimental scores. Published numbers có separate `source_type=paper_reference` và protocol ID.

### 17.2 Candidate contribution and reviewer questions

Inherited: latent teacher prediction, EMA, geometry priors, var/cov/SIGReg, dense loss, multi-layer supervision, articulator locality và gradient cache. Cite sources; không liệt kê thành inventions.

Candidate contribution: a measured sign-specific ambiguity/observability analysis và **routing local state/change supervision** that beats matched alternatives. Need evidence mechanistically stronger than “V-JEPA + weighted pose loss”. Novelty chưa được xác nhận exhaustively; update primary-source search trước submission.

| Reviewer objection | Required evidence |
|---|---|
| SignSeek/SignDINO đã local articulator learning | Clear prior-work overlap, supervision regime, no first claim |
| Just dense V-JEPA recipe | B1/B1a/B1b and B2–B4 comparisons |
| Change = state reweighting | C1/C2/C3 and timestamp/noise diagnostic |
| Routing just drops hard targets | C4 retention/coverage controls and hard-group outcomes |
| Teacher lacks handshape | Phonological probes + stage/ROI occupancy + geometry ablation |
| Hidden supervision | Detector/init/paired labels provenance, 2D adaptation disclosed |
| Gain from bigger backbone/data/readout | Controlled chain plus separate foundation-transfer rows |
| Translation only language prior | Same decoder/frozen encoder and visual sensitivity |

### 17.3 Export package

`encoder_only` artifact chứa online backbone state, exact model config/stage metadata, RGB normalization, temporal/crop/readout policy và checkpoint provenance. Không yêu cầu detector/teacher/predictor để load. Optional EMA export mang filename/model-card field khác.

Verify exported features match training-eval encoder cùng input tolerance. Document deployment input clip length and video aggregation. Không đưa pseudo-depth, keypoint schema hoặc training loss knobs vào inference API nếu người dùng inference không cần.

### 17.4 Agent final report format

Khi kết thúc một automation run, báo:

1. Milestones đã hoàn thành với file/evidence links.
2. Exact commands/config hashes/source/data/init versions đã chạy.
3. Actual metrics và compute; chưa chạy thì ghi `not_run`.
4. Remaining blocked/pending tasks, mỗi task có reason và smallest next action.
5. Method components retained/removed và evidence supporting decision.
6. Claims hiện được phép viết; claims chưa đủ evidence.

Không nói “implementation complete” khi chỉ có scaffolding; không nói “method tốt” chỉ từ successful training hoặc declining pretext loss. Trong code handoff, tách implementation correctness khỏi research validity.

## 18. Prompt khởi động có thể đưa trực tiếp cho AI Agent

> Hãy thực hiện đặc tả ASC-JEPA trong file này như một AI Research Engineer. Đọc §0–§3 và các nguồn R01–R09 trước; audit actual pinned code thay vì suy API. Xây package và task runner theo các tensor/loss/protocol contracts. Bắt đầu M0, tiếp tục M1–M6 khi dependencies khả dụng; hoàn thành các việc độc lập nếu một nguồn/data task bị block. Giữ source/checkpoint/data/config provenance và resumable status. Chạy tests và smoke phù hợp, không tạo fake checkpoint/metrics hoặc gọi adaptation là faithful reproduction. Khi budgets/data/hardware đã có, runner tiến tới controlled pilots và required controls; chỉ scale candidates có evidence. Handoff code, configs, actual evidence và remaining tasks. Mọi phần kế thừa phải có source IDs/permalinks và differences; mọi proposed hyperparameter phải được ghi là pilot choice.

## 19. Completion checklist

- [ ] Sources/commits/checkpoints/environment resolved cho components thực dùng.
- [ ] Exact SignRep replication vs 2D adaptation được đặt tên đúng.
- [ ] Data splits/source overlap/available manifests verified.
- [ ] RGB timestamps và offline sidecars aligned.
- [ ] Hiera native mask/packing/grid contract tested.
- [ ] Predictor không nhận target features/pose; queries chunk-equivalent.
- [ ] State/change/geometry reductions và validity đúng.
- [ ] Regularizer actual batch, DDP/cache gradients verified.
- [ ] Successful-step/EMA/RMS/resume clocks đúng.
- [ ] Frozen published/control readouts và gallery policy tách rõ.
- [ ] B0–B4/controls shared settings và actual run records.
- [ ] Validation selections frozen before confirmatory test.
- [ ] Positive/negative evidence, compute và limitations report honestly.
- [ ] Export chạy RGB-only và match encoder features.

Checklist biểu thị completion criteria cho implementation tương lai. **Trong phiên tạo tài liệu này chưa clone/run các training repos, chưa tải model weights/datasets và chưa chạy ASC-JEPA training.** Những API sâu cần source audit và mọi claimed gain cần experiment.
