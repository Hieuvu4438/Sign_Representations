# SignPathLab — đặc tả triển khai end-to-end cho AI Coding Agent

**Phiên bản:** 1.0 — 12/10/2026.  
**Ngôn ngữ tài liệu:** tiếng Việt; tên module, schema và CLI dùng tiếng Anh.  
**Mục đích:** triển khai, kiểm chứng và nghiên cứu self-supervised sign language representation, với nhánh geometry lấy cảm hứng từ Path-JEPA.  
**Trạng thái:** implementation specification; chưa có repository SignPathLab đã được viết hoặc mô hình đề xuất đã được huấn luyện trong tài liệu này.

Tài liệu này độc lập, có thể đưa trực tiếp cho AI Coding Agent. Những tên package, class, CLI và config thuộc SignPathLab là **hợp đồng cần implement**, không phải công cụ đã tồn tại. Đoạn mã ghi “reference implementation” là mã tham chiếu cho thuật toán; đoạn ghi “pseudocode” cần được Agent chuyển thành chương trình và kiểm thử. Đường dẫn tới repository tác giả là source thật, được phân biệt với file dự kiến của dự án.

Quy ước bằng chứng:

- **[E] Evidence:** paper, source hoặc model metadata đã được đối chiếu.
- **[D] Design:** lựa chọn triển khai của dự án này; không gán cho paper gốc.
- **[H] Hypothesis:** giá trị hoặc chiến lược cần so sánh bằng thí nghiệm.
- **[C] Contract:** hành vi phần mềm phải đáp ứng để kết quả có thể kiểm chứng.

Các số epoch, learning rate, mask ratio và architecture của recipe đề xuất đều là [H], trừ khi ghi rõ là thông số tái lập paper. Không xem chúng là hyperparameter tối ưu.

## Mục lục

1. [Lệnh giao việc và Definition of Done](#1-lệnh-giao-việc-và-definition-of-done)
2. [Phạm vi, giả định và quyết định kiến trúc](#2-phạm-vi-giả-định-và-quyết-định-kiến-trúc)
3. [Paper → thành phần triển khai](#3-paper--thành-phần-triển-khai)
4. [Source code chính thức và pretrained registry](#4-source-code-chính-thức-và-pretrained-registry)
5. [Repository, môi trường và interface](#5-repository-môi-trường-và-interface)
6. [Dataset, manifest và split](#6-dataset-manifest-và-split)
7. [Decode video, pose, ROI và coordinate systems](#7-decode-video-pose-roi-và-coordinate-systems)
8. [Cache đặc trưng và tokenizer](#8-cache-đặc-trưng-và-tokenizer)
9. [Geometry: paths, anchors và level-2 logsignature](#9-geometry-paths-anchors-và-level-2-logsignature)
10. [Masking và dependency closure](#10-masking-và-dependency-closure)
11. [Encoder, predictor và teacher](#11-encoder-predictor-và-teacher)
12. [Objectives và chống collapse](#12-objectives-và-chống-collapse)
13. [Training runtime và distributed correctness](#13-training-runtime-và-distributed-correctness)
14. [Config hoàn chỉnh và command contracts](#14-config-hoàn-chỉnh-và-command-contracts)
15. [Downstream evaluation](#15-downstream-evaluation)
16. [Lộ trình implementation và các vòng lab](#16-lộ-trình-implementation-và-các-vòng-lab)
17. [Ablations và quyết định chọn mô hình](#17-ablations-và-quyết-định-chọn-mô-hình)
18. [Tests, numerical oracle và acceptance](#18-tests-numerical-oracle-và-acceptance)
19. [Compute, profiling và deployment](#19-compute-profiling-và-deployment)
20. [Diagnostics, reproducibility và bàn giao](#20-diagnostics-reproducibility-và-bàn-giao)
21. [Tài liệu gốc](#21-tài-liệu-gốc)

## 1. Lệnh giao việc và Definition of Done

### 1.1. Prompt thực thi cho Agent

> Implement một research codebase tên SignPathLab theo toàn bộ specification này. Ưu tiên một pipeline hoàn chỉnh và baseline mạnh, sau đó thêm geometry bằng những thay đổi có thể ablate. Implement data contracts, preprocessing, cache, baseline, masked JEPA, deterministic geometry targets, evaluation và tests trước khi tăng model/data scale. Đọc các file source được chỉ định trước khi port; pin commit và ghi provenance của checkpoints. Nếu đang có project, đọc AGENTS.md và giữ cấu trúc hợp lý của project đó. Không tự viết lại toàn bộ Fairseq, Hiera hoặc foundation backbone.
>
> Tối ưu chất lượng biểu diễn và hệ thống. Giữ pretrained models, reconstruction, cluster CE, contrastive learning, pose priors, supervised downstream fine-tuning và fusion nếu validation cho thấy hữu ích. Phân biệt sign-only SSL, paired supervision và best-system results. Không tạo score, model weights hoặc dữ liệu giả để thay cho một experiment thật.
>
> Làm smoke run và integration tests trên dữ liệu tổng hợp nhỏ. Khi có dataset và GPU, chạy pilot có ngân sách được ghi trong config. Nếu thiếu dữ liệu, checkpoint hoặc runtime, hoàn thành phần code không phụ thuộc chúng, ghi blocker cụ thể và cung cấp lệnh để chạy tiếp. Không claim reproduction hoặc training thành công chỉ vì import không lỗi.

### 1.2. Deliverables bắt buộc

| Deliverable | Điều kiện hoàn thành |
| --- | --- |
| Package installable | `pip install -e ".[dev]"` thành công trong môi trường đã lock |
| CLI | Có help, validation config, error có nguyên nhân; CPU smoke không cần tải weights |
| Data adapter | Ít nhất một adapter thật: ASL Citizen; adapter local JSONL dùng cho dữ liệu khác |
| Preprocessing | Decode timestamps; landmark/ROI extraction; validity và support provenance |
| Cached features | DINOv2 default; feature schema/version/hash được kiểm tra khi load |
| Baselines | Visual-only frozen probe; cluster-prediction; raw RGB+pose JEPA |
| Geometry | Degree-2 backend thuần NumPy/PyTorch; anchors; ordinary-motion controls |
| Masking | Synchronized temporal và anatomical cross-modal; closure, target eligibility, counters |
| Training | AMP, accumulation, EMA, resume, finite checks; single GPU và DDP |
| Downstream | ISLR linear/attentive/full FT; retrieval; interface CTC và SLT |
| Experimental runner | Cấu hình ablations, fair budgets, seeds, summary CSV/JSON |
| Tests | Geometry, leakage, indexing, padding, EMA, empty targets, metrics và resume |
| Reproducibility | source/checkpoint registry; manifests, environment, config, logs, predictions |
| Handoff | README, model/data cards, failure memo và lệnh train/eval/export |

**MVP Definition of Done [C]:** một clip thật đi từ raw video → timestamps/pose/ROI → cached visual/pose features → tokenizer → masking → teacher/student loss → optimizer step → checkpoint → unmasked clip/dense embedding → downstream predictions. Ngoài ra, các phép kiểm tra leakage và static-pose retention phải pass.

**Research Definition of Done [C]:** baseline và geometry/control được so sánh trên manifest, backbone, downstream head và ngân sách phù hợp; có nhiều seeds khi chạy kết quả chính; kết luận được giới hạn theo evidence thực tế. MVP chạy đúng chưa đủ để kết luận idea tốt.

## 2. Phạm vi, giả định và quyết định kiến trúc

### 2.1. Defaults khi chưa có thông tin khác

| Quyết định | Default [D/H] | Vì sao |
| --- | --- | --- |
| Ngôn ngữ kiểm chứng | ASL | Có ASL Citizen, Sem-Lex và continuous benchmarks; không suy ra ngôn ngữ từ tiếng Việt của yêu cầu |
| Primary pilot | ASL Citizen, official signer-disjoint split | Kiểm chứng khả năng chuyển sang signer mới |
| Metric chính | Per-class Top-1; đồng báo cáo per-instance Top-1 | Chọn metric trước experiment; không đổi theo kết quả |
| Supervision pretraining | Sign video/pose/features, không đưa gloss/text vào loss/sampler | Đánh giá utility của sign-only SSL |
| Pretrained image encoder | DINOv2 ViT-S/14, frozen để cache | Khởi đầu dễ kiểm chứng, 384 chiều; không chặn mở rộng model mạnh hơn |
| Visual streams | Global signer, hand-left, hand-right, face | Giữ thông tin visual và non-manual |
| Static pose streams | Upper body, left hand, right hand | Giữ handshape, orientation 2D và body-relative location |
| Temporal encoder | Transformer 6 layers, dimension 384, 6 heads | Có thể train trên cached features, điều chỉnh theo profiling |
| Geometry version đầu | Deterministic target-only, degree 2, không lead-lag | Ít confound và dễ kiểm tra tính đúng |
| Windows geometry | 0.16 / 0.32 / 0.64 s | Giả thuyết khởi đầu; lưu bounds thực tế theo timestamps |
| Mask MVP | Synchronized temporal, 0.50 requested frame ratio | Cho phép strict temporal information hiding ở cached mode |
| Deployment ban đầu | Offline RGB + pose | Hệ đa modality cần detector thật khi inference |
| Nâng cấp performance | SignRep, SHuBERT, DINOv3/SignDINO-style, V-JEPA 2.1 | So sánh để tìm backbone/recipe tốt nhất; không mặc định model mới thắng |

Không cắt clip ký hiệu để khớp 1.92 giây khi đánh giá ISLR. Fixed-length segments chỉ là đơn vị pretraining; downstream cần cả clip hoặc nhiều cửa sổ và aggregation được khai báo.

### 2.2. Ba tuyến model phải tách rõ

1. **Representation baseline:** visual/pose encoder và downstream head giống nhau giữa raw JEPA, motion targets và signature targets.
2. **Path-inspired research:** geometry làm auxiliary target trước; sau đó thêm geometry input, multi-scale relational paths hoặc learned geometry teacher nếu có evidence.
3. **Best system:** cho phép backbone lớn hơn, sign-adapted features, paired/text pretraining, supervised fine-tuning, fusion hoặc ensemble. Báo cáo bổ sung tài nguyên/supervision của từng thay đổi.

Không dùng best-system gains để kết luận riêng về signature. Không loại một technique có ích chỉ vì nó làm thiết kế bớt “JEPA thuần”.

### 2.3. Ranh giới claim

- [E] Path-JEPA đánh giá skeleton action recognition, không chứng minh performance trên sign language [1].
- [E] Temporal signature của đường đứng yên không phân biệt những vị trí/handshape đứng yên khác nhau [28].
- [D] Vì vậy hệ thống giữ visual/static pose và anchors; signature chỉ thêm motion geometry.
- [E] Invariance với time reparameterization có điều kiện về cùng geometric path. Fixed-delay lift, time-channel, sampling mất đỉnh hoặc cửa sổ theo giây không tự bảo toàn claim đó.
- [C] Không ghi “first”, “state of the art” hoặc một mức improvement khi chưa có search/experiment phù hợp.

## 3. Paper → thành phần triển khai

Bảng sau định hướng code, không phải khẳng định mỗi thành phần sẽ tăng score. Các tên file ở cột cuối là file cần tạo trong dự án.

| Nguồn | Evidence / ý tưởng đáng mượn | Adaptation [D/H] | Module cần tạo |
| --- | --- | --- | --- |
| Path-JEPA [1] | Multi-scale joint/edge/chain path features; teacher latent prediction; propagation masks | Explicit path definition, static anchors, support closure; degree-2 trước | geometry/paths.py, geometry/logsig2.py, masking/closure.py |
| SignRep [2] | RGB encoder học sign priors; pose chỉ cần lúc pretraining | Frozen RGB baseline; auxiliary hand/body angles/distance khi hữu ích | backbones/signrep.py, losses/sign_priors.py |
| SHuBERT [3] | Multi-stream cluster targets và temporal contextual encoder | Reproduce original adapter; cluster CE baseline; layer mixture | backbones/shubert.py, targets/kmeans.py, losses/cluster_ce.py |
| SignBERT / SignBERT+ [4,5] | Pose/hand priors; anatomical masking | Confidence-aware pose reconstruction; giữ static shape/location | pose/topology.py, losses/pose_recon.py |
| BEST [6] | Discrete pose tokenization | Option coupling tokenizer; không gọi cluster là gloss | targets/discrete_pose.py |
| Spatial-temporal consistency [7] | Joint/motion, hand/trunk relationships | Velocity/ordinary-motion control, relation objective | geometry/motion_controls.py |
| SSVP-SLT [8] | SignHiera MAE; separate language-supervised stage | Hiera adapter; MAE/LSP như baseline độc lập | backbones/signhiera.py, stages/mae.py |
| SignDINO [9] | Sign-aware local/global self-distillation | DINOv3 sign adaptation, giữ kích thước canvas local mask | stages/sign_dino_adapt.py |
| SignMAE [10] | Region-focused reconstruction và heatmap/RGB fusion | Hand-weighted reconstruction hoặc heatmap target | masking/regions.py, losses/region_recon.py |
| SignCL [11] | Temporal discrimination | Guard empty pairs; sampling theo giây cho sign-only SSL | losses/temporal_contrast.py |
| SignCLIP / Uni-Sign [12,13] | Paired alignment, confidence-aware pose/RGB fusion | Optional paired stage; khai báo text supervision | stages/paired.py, fusion/confidence_gate.py |
| GFP [14] | Lightweight hierarchical target generator, VC | Alternative geometry target projector có gradient thật | targets/learned_geometry.py |
| S-JEPA [15] | Raw-skeleton latent prediction, CE/centering | Raw-pose JEPA baseline; không gán cosine loss của Path cho S-JEPA | stages/raw_pose_jepa.py |
| MAMP [16] | Motion targets | Displacement/velocity matched control | geometry/motion_controls.py |
| I-JEPA / V-JEPA [17,18] | Masked latent prediction | Context/target separation, EMA, consistent probing | models/teacher_student.py |
| V-JEPA 2 / 2.1 [19,20] | Video initialization; dense/deep targets | Backbone control; context loss và layer targets riêng | backbones/vjepa.py, losses/dense_latent.py |
| VideoMAE / 2023 sign transformer study [21,39] | Reconstruction/HOG/hierarchical video models | Giữ MAE/MaskFeat comparison nếu ngân sách cho phép | stages/reconstruction.py |
| VICReg [22] | Variance/covariance regularization | Counter-collapse ablation với sample independence | losses/vicreg.py |
| LeJEPA [23] | SIGReg và recipe không EMA | Experimental recipe riêng; không trộn tùy tiện vào EMA variant | stages/lejepa.py |
| data2vec / HuBERT [24,25] | Multi-layer continuous target / hidden units | Layer target mixture và cluster CE | targets/layer_targets.py |
| DINOv2 / v3 [26,27] | Foundation visual features | ROI/global cache hoặc online adaptation | backbones/dino.py |
| Signature theory / Signatory [28,29] | Algebra, dimensions, transforms | Numerical oracle; backend optional, không ép extension vào MVP | geometry/logsig2.py |
| Fair SLT / signer dependence [30,31] | Decoder/metric/split confounds | Fixed evaluation contracts và overlap audit | evaluation/protocols.py, data/audit.py |

## 4. Source code chính thức và pretrained registry

### 4.1. Commit cần pin và file cần đọc

Các commit dưới đây là snapshot đã đối chiếu. Chúng không bảo đảm tương thích mọi CUDA/PyTorch hiện hành. Agent phải lưu commit thực sự dùng và patch riêng.

| Repository | Pinned commit | File trọng tâm / cách reuse |
| --- | --- | --- |
| [SignRep](https://github.com/ryanwongsa/SignRep) | `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2` | [model](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/models/final_models/FINAL_hiera_latent_model_head_v25_active.py), [heads](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/models/head_models/dict_head_v6_FINAL_thoughtful.py), example_usage.py, augmentation/video/base_video_aug.py |
| [SHuBERT](https://github.com/ShesterG/SHuBERT) | `cc1929326075bfbad7ad73159b2acf84356059bb` | [model](https://github.com/ShesterG/SHuBERT/blob/cc1929326075bfbad7ad73159b2acf84356059bb/fairseq/examples/shubert/models/shubert.py), [config](https://github.com/ShesterG/SHuBERT/blob/cc1929326075bfbad7ad73159b2acf84356059bb/fairseq/examples/shubert/config/base_random.yaml), QUICKSTART.md, features/shubert_inference.py |
| [SSVP-SLT](https://github.com/facebookresearch/ssvp_slt) | `7ed332b0db35884a36b73019ca29bf0755537def` | [MAE](https://github.com/facebookresearch/ssvp_slt/blob/7ed332b0db35884a36b73019ca29bf0755537def/src/ssvp_slt/modeling/sign_hiera_mae.py), pretraining/README.md, translation/README.md, launch_feature_extraction.py |
| [V-JEPA 2 / 2.1](https://github.com/facebookresearch/vjepa2) | `204698b45b3712590f06245fbfba32d3be539812` | [2.1 train](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py), src/hub/backbones.py, hubconf.py, evals/hub/preprocessor.py |
| [SignCL](https://github.com/JinhuiYE/SignCL) | `fba1a1d96c310e7ee335bb55d591511c9f46ec68` | [standalone loss](https://github.com/JinhuiYE/SignCL/blob/fba1a1d96c310e7ee335bb55d591511c9f46ec68/SignCL.py), examples/GFSLT-VLP/signcl.py |
| [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) | `eed438bcb49e30405cd6ccdfcccca330c134e830` | [models](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/models.py), datasets.py, docs/DATASET.md |
| [DINOv2](https://github.com/facebookresearch/dinov2) | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | [hub backbones](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/backbones.py), models/vision_transformer.py, data/transforms.py, hubconf.py |

DINOv2 đã kiểm tra thêm hub/forward/normalization source cho tài liệu triển khai này; sáu repo khác kế thừa audit của review và đối chiếu lại phần loader liên quan. Đây không phải một đợt chạy training của các repo.

**Path-JEPA:** [project](https://singh17ashu.github.io/Path-JEPA/) có nút Code trỏ tới `#` ở snapshot kiểm tra. Chưa xác minh implementation công khai khác trong phạm vi nguồn được kiểm tra. Bản [5207.pdf](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf) không kèm Appendix A/B mà main text dẫn tới. Agent implement specification độc lập, đặt tên “Path-inspired”; chưa gọi là exact reproduction.

**SignDINO, SignMAE, S-JEPA, GFP, SignBERT+:** phương pháp được đọc trong paper; không tự gán source repo/checkpoint đã audit. Nếu dùng để reproduce, tra official author link và cập nhật registry sau khi kiểm tra.

### 4.2. Checkpoint matrix có thể hành động

| Model | Nguồn weights đã xác minh ở metadata/source | Vai trò | Điều kiện |
| --- | --- | --- | --- |
| DINOv2 ViT-S/14 | [official backbone weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth); hub entry `dinov2_vits14` | Default frozen image encoder, 384D | Tải file và tính SHA-256 thực tế; chưa tải weights trong quá trình viết tài liệu |
| DINOv2 ViT-B/14 | [official weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth); `dinov2_vitb14` | 768D higher-capacity comparison | Đổi feature schema; không reuse cache 384D |
| SignRep | [release v0.0.1](https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1), asset [ckpt.pt](https://github.com/ryanwongsa/SignRep/releases/download/v0.0.1/ckpt.pt) | Sign-specific RGB clip baseline | API release metadata: 222,676,746 bytes. Model weights CC BY-NC-SA 4.0 kèm notice trên release |
| SHuBERT + hand/face DINO | [folder do README tác giả cung cấp](https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV?usp=sharing) | Original multi-stream checkpoint baseline | Chưa kiểm tra bytes/tên file trong folder. Không đoán file ID; weights generic DINO không thay thế silently adapted DINO |
| V-JEPA 2.1 ViT-B/16 distilled | [official weights](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt); `vjepa2_1_vit_base_384` | Modern video backbone control | 384px; loader trả encoder và predictor; xử lý lỗi base URL ở `4.4 |
| V-JEPA 2 ViT-L/16 | [official weights](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt); HF [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | Video understanding backbone control | Dùng processor và sampling theo model; không dùng robotics AC checkpoint cho default |
| DINOv3 ViT-S/16 | HF [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | Stronger image initialization / sign adaptation | Metadata revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`; manual gated. Dùng access đã được cấp, không giả định tải được |
| Uni-Sign | [author model collection](https://huggingface.co/ZechengLi19/Uni-Sign) | Paired/supervised best-system comparison | Chọn đúng dataset/task checkpoint; không ghi là sign-only SSL |
| SSVP-SLT / SignHiera | [official code/instructions](https://github.com/facebookresearch/ssvp_slt) | MAE baseline hoặc Hiera architecture | Chưa xác minh release weight URL cụ thể trong audit. Dùng checkpoint người dùng cung cấp hoặc train theo official instructions; không bịa URL |
| MediaPipe landmarks | [Hand](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), [Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker), [Face](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) model bundles | Preprocessing external prior | Resolve file từ bảng model chính thức, lưu version/hash; detector không phải ground-truth |

Trạng thái “official weights được liên kết” nghĩa là nguồn đã được xác minh, không có nghĩa đã chạy inference, tải toàn bộ bytes hoặc reproduce metric. Mọi checksum weights phải được tính sau download, không copy checksum code commit làm checksum weights.

### 4.3. Registry và checkpoint download contract

Tạo `registries/checkpoints.yaml`:

~~~yaml
schema_version: 1
models:
  dinov2_vits14:
    code_repository: https://github.com/facebookresearch/dinov2
    code_commit: 7764ea0f912e53c92e82eb78a2a1631e92725fc8
    hub_entry: dinov2_vits14
    weights_url: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
    local_path: checkpoints/dinov2_vits14_pretrain.pth
    sha256: null
    output_dim: 384
    pretrained_domain: generic_images
    sign_text_supervision: false
  signrep_v001:
    code_repository: https://github.com/ryanwongsa/SignRep
    code_commit: 06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2
    release_tag: v0.0.1
    weights_url: https://github.com/ryanwongsa/SignRep/releases/download/v0.0.1/ckpt.pt
    local_path: checkpoints/signrep_v001.pt
    expected_size_bytes: 222676746
    sha256: null
    input_frames: 16
    input_resolution: 224
  vjepa21_vitb:
    code_repository: https://github.com/facebookresearch/vjepa2
    code_commit: 204698b45b3712590f06245fbfba32d3be539812
    hub_entry: vjepa2_1_vit_base_384
    weights_url: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
    local_path: checkpoints/vjepa21_vitb.pt
    sha256: null
    input_resolution: 384
    encoder_state_key: target_encoder
~~~

`sha256: null` là chưa download/verify; scientific run không được giữ null cho weights đã dùng. Downloader phải stream vào `.part`, kiểm tra file/size rồi rename atomic; không nhận HTML lỗi/login page làm checkpoint. Không auto fallback từ pretrained sang random initialization. Nếu chưa có access, ghi missing asset và dùng baseline khác dưới run ID khác.

Provenance thêm vào mỗi run:

~~~json
{
  "registry_key": "dinov2_vits14",
  "code_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
  "weights_sha256": "COMPUTED_AFTER_DOWNLOAD",
  "load_report_path": "artifacts/checkpoint_load_report.json",
  "preprocessing_hash": "COMPUTED_FROM_CANONICAL_CONFIG",
  "initialization": "official_pretrained"
}
~~~

Giá trị dạng `COMPUTED_...` trong ví dụ là placeholder của schema; validator reject nếu xuất hiện trong scientific run thật.

### 4.4. Loader contracts và các bẫy source thật

**DINOv2 [E/D].**

- Clone full official repo, checkout commit đã pin. Hub load từ local repo để tránh code drift.
- Instantiate `pretrained=False`, load official state dict `strict=True`. Với ViT-S/14 không register, dimension 384.
- Input `[B,3,224,224]` RGB float32 với ImageNet mean/std. Không chuẩn hóa hai lần.
- `forward_features` có `x_norm_clstoken` và `x_norm_patchtokens`. MVP lưu CLS features. Patch/ROI pooling là ablation riêng.
- Full signer/hand/face crops đều dùng cùng generic backbone ở default; sign-adapted hand/face weights của SHuBERT là backend khác.

~~~python
# Adapter contract; deps/repo phải có sẵn. Không phải smoke fixture.
import torch

model = torch.hub.load(
    str(repo_dir), "dinov2_vits14", source="local", pretrained=False
)
state = torch.load(weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval().to(device)
with torch.inference_mode():
    output = model.forward_features(images)
    features = output["x_norm_clstoken"]  # [B,384]
~~~

**SignRep [E/D].**

- Wrapper constructor `ckpt_dir` load `model_state` cho initialization backbone, nhưng official example load released full model bằng checkpoint `["model"]`. Adapter phải inspect keys và chọn loại load; không hoán đổi hai path.
- Khởi tạo đúng prior-head config của `example_usage.py`; input `[B,3,16,224,224]`, normalization như official augmentation.
- Official wrapper `features` là projected latent, `latent` là backbone latent; export cả hai trong pilot adapter để chọn bằng validation.
- Eval tắt mask, nhưng wrapper vẫn tính heads; `extract_features=True` chưa là fast path thật trong snapshot.
- Trước khi bỏ heads cho inference, so feature output của full wrapper và fast adapter trên cùng tensor. Chỉ chấp nhận tương đương số trong tolerance.
- Đây là pooled 16-frame output; không coi một vector là dense frame features. Sliding windows stride 2 là một cách lấy sequence; support của một window là cả 16 input frames.

**SHuBERT [E/D].**

- Original input: face, hand-left, hand-right, mỗi stream 384D, body 14D; checkpoint backbone 768D. Không đưa static-hand 84D của SignPathLab vào original checkpoint.
- Run extraction trong environment legacy riêng; export `[L,T,768]` kèm timestamps/layer IDs. New package chỉ đọc cache, không import hai Fairseq versions cùng process.
- Official inference script dùng `strict=False`; adapter phải báo cáo missing/unexpected keys và reject thiếu backbone quan trọng.
- Original random mask flatten `[B,T*C]` trước rồi reshape: span length 3 có thể băng qua streams. Có hai config tên riêng: `shubert_code_random` và `shubert_stream_temporal`. Không sửa mà vẫn gọi faithful reproduction.
- Original collater ở audited path crop về shortest sample; modern padded implementation là adaptation, cần test.
- README downstream fine-tuning còn TODO; viết probe/FT adapter riêng và ghi đó là implementation của dự án.

**V-JEPA 2.1 [E/D].**

- Trong pinned `src/hub/backbones.py`, `VJEPA_BASE_URL = "http://localhost:8300"`, còn official URL bị comment. Không copy README hub call rồi mặc định download sẽ hoạt động.
- Giải pháp: explicit official weights download; local hub `pretrained=False`; adapter load `target_encoder` theo key normalization của source.
- Hub trả `(encoder, predictor)`; encoder ViT-B có interface riêng, không xem tuple là nn.Module.
- Inspect function signature của entry để tránh truyền trùng `img_size` với giá trị hard-coded.
- Match architecture và checkpoint, rồi strict load hoặc narrowly allow-listed exceptions được source giải thích. V-JEPA 2.1 adapter không tự bỏ errors hàng loạt.
- Teacher output/layer distillation khác predictor output; verify tensor shapes trước pooling.
- Original training config generic-video fps thấp không phải recommendation cho signing. Backbone feature extraction phải khai báo timestamps và receptive field thật.

**SignCL [E/D].** Standalone default `seq_len=16, margin=20` có empty negative set. Port phải guard; source dùng L1/softplus và normalization cần kiểm tra. Không copy paper Euclidean/hinge rồi gán source giống nhau.

## 5. Repository, môi trường và interface

### 5.1. Layout cần implement

Các file optional đánh dấu trong README bằng milestone, không tạo stub im lặng trả zero loss.

~~~text
signpathlab/
  pyproject.toml
  README.md
  configs/
    base.yaml
    smoke.yaml
    experiments/
      E00_frozen_visual.yaml
      E01_cluster.yaml
      E02_raw_jepa.yaml
      E03_motion_target.yaml
      E04_logsig_target.yaml
      E05_logsig_input.yaml
      E06_dense_deep.yaml
      E07_best_system.yaml
    downstream/
      islr_linear.yaml
      islr_attentive.yaml
      islr_finetune.yaml
      cslr_ctc.yaml
      slt_fixed_decoder.yaml
  registries/
    checkpoints.yaml
    sources.yaml
  src/signpath/
    cli.py
    config.py
    contracts.py
    data/{manifest,audit,decode,dataset,collate,sampling}.py
    data/adapters/{local_jsonl,asl_citizen,wlasl,how2sign,openasl}.py
    pose/{extract,topology,coordinates,quality,roi}.py
    backbones/{base,dino,signrep,shubert,signhiera,vjepa}.py
    cache/{schema,writer,reader,hashing}.py
    geometry/{paths,logsig2,anchors,motion_controls,supports}.py
    targets/{kmeans,layer_targets,learned_geometry}.py
    masking/{samplers,closure,eligibility,diagnostics}.py
    models/{tokenizer,encoder,predictor,teacher_student,pooling,heads}.py
    losses/{latent,geometry,cluster_ce,vicreg,pose_recon,temporal_contrast}.py
    training/{engine,ema,schedules,distributed,checkpoint,monitor}.py
    evaluation/{islr,retrieval,phonology,ctc,slt,bootstrap,protocols}.py
    inference/{extract,aggregate,export}.py
  tests/
    unit/
    integration/
    fixtures/
  scripts/
    doctor.sh
    run_pilot.sh
    reproduce_summary.py
  docs/{METHOD,DATA_CARD,MODEL_CARD,KNOWN_LIMITATIONS}.md
  artifacts/                         # generated, gitignored
  checkpoints/                       # downloaded, gitignored
  third_party/                       # pinned deps/submodules, license retained
~~~

Không cần tạo mọi optional stage trước MVP. Mỗi unavailable backend phải ném error “not implemented at milestone X”, không chọn backend khác âm thầm.

### 5.2. Environments

[D] Tách 3 môi trường:

| Environment | Nội dung | Lý do |
| --- | --- | --- |
| main | PyTorch, NumPy, PyYAML/OmegaConf, scikit-learn, pytest, metric libs | Temporal SSL, tests và evaluation |
| extraction | PyAV/FFmpeg, OpenCV, MediaPipe, DINO adapter | CPU/GPU detector và backbone cache |
| legacy_shubert | Chính environment files của official SHuBERT | Fairseq/Python compatibility riêng |

Profile khởi đầu có thể kiểm tra: Python 3.11, PyTorch 2.5.1 + torchvision 0.20.1 cho main/DINOv2; **đây là compatibility candidate, chưa được chạy trong tài liệu**. Chọn CUDA wheel phù hợp driver bằng [official install matrix](https://pytorch.org/get-started/previous-versions/). DINOv3/V-JEPA có thể cần profile khác; lock chúng sau adapter smoke. Không ép dependency cũ của Fairseq vào main.

Các dependency bắt buộc có version lock sau khi `doctor` và tests pass:

- `torch`, `torchvision`, `numpy`, `scikit-learn`.
- `PyYAML` hoặc `omegaconf`; `typer` hoặc `argparse`, chọn một.
- `av`, `opencv-python-headless`, `mediapipe` trong extraction.
- `pytest`; `sacrebleu`/`jiwer` cho continuous evaluation.
- `transformers` và tokenizer chỉ ở backend/stage có dùng.
- `signatory` là optional. MVP degree-2 không phụ thuộc C++ extension; PyTorch ABI của Signatory phải kiểm tra [29,45].

`doctor` ghi Python, package versions, OS, GPU model, driver, CUDA toolkit/runtime, bf16 availability, git hashes; không sửa config silently khi không có CUDA.

### 5.3. Core APIs

~~~python
class BackboneAdapter:
    def load(self, registry_entry, device): ...
    def extract(self, frames, timestamps, rois) -> "FeatureBundle": ...
    def provenance(self) -> dict: ...

class SignTokenizer:
    def build(self, feature_bundle, geometry_bundle=None) -> "TokenBundle": ...

class MaskSampler:
    def sample(self, sample, generator) -> "RawMask": ...

class ClosureResolver:
    def resolve(self, token_bundle, raw_mask) -> "MaskPlan": ...

class ResearchModel:
    def encode_full(self, batch, return_layers=False) -> "EncoderOutput": ...
    def training_loss(self, batch, mask_plan) -> "LossOutput": ...
    def extract_embeddings(self, batch) -> "EmbeddingOutput": ...
~~~

Data classes phải lưu shape, dtype, validity, token IDs, timestamps và support. Không dùng dictionary không có schema ở ranh giới cache hoặc checkpoint.

`LossOutput` gồm scalar total loss, per-type sums/counts, eligible targets, skip reasons, mask ratios và numerical diagnostics. `EmbeddingOutput` gồm dense timestamps/validity; clip pooled embedding là field phụ.

## 6. Dataset, manifest và split

### 6.1. Data priority và quyền dùng nhãn

| Dữ liệu | Nguồn | Vai trò | Nhãn được phép dùng |
| --- | --- | --- | --- |
| ASL Citizen | [official description](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/) [32] | Primary pilot ISLR/retrieval | Pretrain không đọc label; downstream train đọc label; validation chọn checkpoint |
| WLASL | [author repo](https://github.com/dxli94/WLASL) | Secondary ISLR comparison | Khóa version và available-video manifest; không sửa taxonomy im lặng |
| Sem-Lex | [author repo](https://github.com/leekezar/SemLex) [33] | Phonological probes | Labels chỉ cho probe |
| How2Sign | [official project](https://how2sign.github.io/) [34] | Continuous ASL transfer | Text chỉ ở downstream/paired stage |
| OpenASL | [official repo](https://github.com/chevalierNoir/OpenASL) [35] | Domain-diverse SLT | Source overlap audit trước pretraining |
| YouTube-ASL / SL-25 | [official YT-ASL instructions](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md) [36,37] | Optional SSL/paired scale | Nếu strict SSL, không đưa subtitle vào loss hoặc sampler |
| FLEURS-ASL | [paper](https://arxiv.org/abs/2408.13585) [38] | Held-out transfer | Không đưa vào model-selection loop rồi gọi zero-shot |

Agent không cần download toàn bộ corpus để code MVP. Dùng authorized/local data khi có; synthetic fixture phục vụ correctness, không phục vụ research metrics. Data card lưu điều kiện sử dụng thực tế của corpus và weights đã dùng.

### 6.2. Manifest format

Một dòng JSONL = một clip/sequence đã định danh:

~~~json
{
  "schema_version": 1,
  "sample_id": "asl_citizen_example_0001",
  "dataset": "asl_citizen",
  "language": "ase",
  "split": "train",
  "source_group_id": "source_video_example_0001",
  "video_path": "data/raw/example_0001.mp4",
  "source_sha256": "COMPUTED_FOR_LOCAL_VIDEO",
  "signer_id": "official_signer_id",
  "start_s": 0.0,
  "end_s": 3.20,
  "nominal_fps": 30.0,
  "label_id": 42,
  "gloss": "example_label",
  "text": null,
  "official_split_version": "recorded_source_version",
  "availability": "available"
}
~~~

Ví dụ `ase` là ISO language tag của ASL; nếu dữ liệu khác thì đổi đúng language. Nhãn `42` là ví dụ, không phải nhãn thật của một sign.

**Separate views [C]:**

- Pretraining loader chỉ expose video/features, validity, timestamps và sampler metadata không chứa target labels.
- Downstream loader expose nhãn của train split.
- Validation/test labels chỉ trong evaluation module.
- `source_group_id` dùng chống chia hai đoạn cùng source vào train/test. Nó không được nhét vào model embedding.
- `signer_id` dùng audit/evaluation, không là input feature.
- Missing signer ID ghi null; không tự nhận diện danh tính người để lấp schema.

### 6.3. Split audit và fitting

1. Import official split trước khi tạo sliding clips/crops.
2. Kiểm tra sample/source IDs giao nhau và exact hash duplicates giữa train/val/test.
3. Ghi near-duplicate checks với video perceptual fingerprints; manual review các candidate thực sự ảnh hưởng evaluation.
4. Với YouTube/source clip, exclude held-out nguồn từ train; audit theo full source chứ không chỉ basename đoạn.
5. Fit coordinate statistics, PCA/random projection configuration, k-means và vocabulary bằng train.
6. Validation dùng model selection phải được công bố. Không gọi pipeline “target-domain label-free selection” nếu chọn bằng validation labels.
7. Test chỉ chạy khi config/model-selection recipe đã khóa.

ASL Citizen split được source mô tả signer independent [32]. Nếu đổi split vì thiếu video, lưu cả official metadata và manifest thực tế. Với WLASL, báo cáo available counts và class counts; 79.02% trên corrected 1535-class setup không so trực tiếp original WLASL2000 [39].

### 6.4. Sampler

[D] Pretraining default uniform source → uniform valid segment, không class-balanced theo gloss. Segment length 48 samples ở target 25fps là khoảng 1.92s; segment thực tế dùng timestamps và không giả định mọi sample đạt độ dài này.

- Short clip: pad, giữ real length; không repeat padded frames thành target thật.
- Long clip: random contiguous segment trong train; deterministic tiling trong inference.
- Có option source-length weighting; log distribution để biết model gặp các corpus theo tỷ lệ nào.
- Không reweight theo held-out frequency.
- Supervised downstream class-balanced sampler/loss là option độc lập; báo cáo tác động riêng.

## 7. Decode video, pose, ROI và coordinate systems

### 7.1. Video decode

[C] Dùng PyAV/FFmpeg PTS, convert to seconds, sort/check monotonicity. `nominal_fps` chỉ dùng metadata/fallback có flag. Video variable-frame-rate phải được kiểm tra bằng timestamps thực tế.

[D] Resample mục tiêu 25fps bằng nearest observed frame:

- Không bịa intermediate RGB frames.
- Store `source_frame_index` và `actual_timestamp_s`; giữ duplicate selection flags khi nguồn fps thấp.
- Không claim temporal resolution tăng khi upsample 15fps thành 25fps.
- Geometry dùng original/selected timestamps; velocity chia actual dt, loại dt=0.
- Strict mask support tính theo raw source frame, không theo index sau resample. Hai resampled frames trỏ cùng raw frame phải cùng bị che.
- Clustering và SSL không nhân đôi contribution của cùng source frame vô điều kiện.

Có config `target_fps: 15` để so SHuBERT original, và higher fps nếu source thật có. Fingerspelling cần experiment riêng về temporal resolution.

RGB extraction dùng PyAV rgb24 hoặc OpenCV BGR→RGB explicit. Scale uint8/255 một lần trước ImageNet normalization; box coordinates theo original frame dimensions. Resize interpolation và letterbox fill value ghi trong extractor hash.

### 7.2. Pose extractor và topology

[D] Default 2D prototype: 49 joints, gồm 7 upper-body + 21 mỗi tay.

| SignPathLab IDs | Primitive |
| --- | --- |
| 0 | nose |
| 1,2 | left/right shoulder |
| 3,4 | left/right elbow |
| 5,6 | left/right body wrist |
| 7..27 | left hand MediaPipe hand order 0..20 |
| 28..48 | right hand MediaPipe hand order 0..20 |

MediaPipe Pose source indices cho upper7: nose0, left_shoulder11, right_shoulder12, left_elbow13, right_elbow14, left_wrist15, right_wrist16. Map explicit sang SignPathLab IDs0..6; không lấy seven first landmarks.

Giữ body wrist và hand wrist là hai measurements khác nhau, không overwrite. Face landmarks chỉ dùng ROI trong MVP; face geometry là nhánh sau. Pose raw cache lưu all extracted landmarks để có thể sửa mapping mà không chạy detector lại.

MediaPipe hand order: wrist 0; thumb 1..4; index 5..8; middle 9..12; ring 13..16; pinky 17..20. Topology file phải có ordered list, edges và semantic labels; hash topology đi vào cache.

**Handedness / mirror [C].**

- Raw video orientation và mirrored metadata phải được ghi. Detector “Left/Right” cần đối chiếu camera convention.
- Verify overlay trên ít nhất một clip có tay trái/phải rõ; không dùng x-position duy nhất để assign vì tay có thể bắt chéo.
- MVP strict cache dùng per-frame detection, không smoothing/tracking qua thời gian. Nếu cần matching hai tay, dùng wrist proximity với same-frame pose và lưu ambiguity; uncertain hand ID → invalid.
- VIDEO-mode tracking hoặc sequence smoothing có temporal dependencies; bật chúng thì phải mở rộng support hoặc đánh dấu approximate masking. Không coi cached video-mode detector là strict same-frame extraction [40].

### 7.3. Quality không đồng nghĩa ground-truth confidence

Hand Landmarker thường trả coordinates và handedness category score; category score không phải độ tin cậy định vị riêng từng joint. Không copy handedness score vào 21 joints rồi ghi “per-joint detection probability”.

Schema raw:

- `observed[T,J]`: landmark có kết quả hữu hạn.
- `quality[T,J]`: quality/proxy score, cùng `quality_kind`.
- Body có visibility/presence nếu API cung cấp, ghi ý nghĩa của từng field.
- Hand quality có thể là binary observed hoặc proxy từ box size/kinematic plausibility; metadata phải ghi rõ.
- Missing coordinates lưu NaN trong raw cache; model tensor thay zero **kèm** observed=0 và quality=0.
- Không nhận (0,0) là missing theo giá trị vì tọa độ đó có thể hợp lệ.

[D/H] Geometry targets MVP yêu cầu mọi raw joint trong window observed và quality qua threshold đã định nghĩa. Điều này có thể làm ít targets; log và sửa preprocessing/sampling nếu thiếu, không tự convert missing sang đường đứng yên.

### 7.4. Coordinate systems

Raw normalized image x/y có aspect-ratio distortion nếu dùng trực tiếp euclidean distance. Convert trước:

$$p_{t,j}=(x_{t,j}W_t,\ y_{t,j}H_t).$$

Body-space per frame:

$$c_t=\tfrac12(p_{t,Lshoulder}+p_{t,Rshoulder}),\quad
s_t=\max(\|p_{t,Lshoulder}-p_{t,Rshoulder}\|_2,\epsilon),\quad
b_{t,j}=(p_{t,j}-c_t)/s_t.$$

[D] Dùng isotropic scale, không rotate mặc định. Nó giữ vị trí tay trong body coordinate và orientation với camera. Nếu shoulders quá gần/invalid, frame normalization invalid; không suy shoulder scale từ toàn clip trong strict MVP.

Hand-local shape:

$$h_{t,j}=(p_{t,j}-p_{t,wrist})/
\max(\|p_{t,middleMCP}-p_{t,wrist}\|_2,\epsilon).$$

Giữ đồng thời:

- Hand-local coordinates để giữ shape.
- Body-space wrist/palm anchors để giữ location.
- Orientation vector/palm basis nếu feature đó được bật.
- Scale/quality metadata ở downstream nếu đã đăng ký, không expose data-dependent hidden values trong prediction query.

MVP pose tokens dùng body-space joint coordinates, nên handshape vẫn được giữ. Hand-local + orientation có thể thêm như ablation; thay dimension schema rõ ràng.

**Validity propagation [C]:** raw observed flags giữ trong landmark cache. FeatureBundle lưu thêm pose_coordinate_valid = raw_observed AND every normalization prerequisite valid/finite. Model bit thứ tư là coordinate_valid; quality được zero ở invalid coordinates, còn raw quality giữ trong cache. Geometry eligibility yêu cầu coordinate_valid cho mọi sample/joint. Nếu shoulders invalid hoặc scale≤epsilon, không normalize hands thành values lớn; model coordinates zero với validity0. Epsilon khởi đầu1 pixel, lưu config.

**Normalization dependency [C]:** body-space coordinate của một hand joint phụ thuộc raw joint đó **và cả hai shoulder joints cùng frame**. Closure phải đưa shoulders vào support. Hand-local phụ thuộc wrist và scale joints. Không mask một shoulder rồi giữ normalized hands như không liên quan.

Giữ image-space coordinates và c_t/s_t trong raw cache để thử body-motion/absolute-location branch khi cần. Body normalization có thể loại một số movement của toàn torso; không giả định mọi body translation là nuisance. Main static/visual streams và phonological probes giúp kiểm tra tradeoff này.

### 7.5. ROI

[D] DINO default:

- Global signer ROI từ same-frame body/hand envelope có margin; nếu không xác định được signer, toàn frame là fallback có flag.
- Hand ROI từ hand landmarks envelope, square padding, margin khởi đầu 1.5×.
- Face ROI từ face landmarks hoặc face box, giữ vùng mắt/mày/miệng.
- Resize letterbox isotropic về 224×224; lưu affine transform và padding fractions.
- Không center-crop làm mất ngón/miệng; không tự zoom khác nhau giữa train/eval.
- Nếu box quá nhỏ/invalid: missing stream, không lấy hand crop thành background mà đánh dấu valid.
- Per-frame boxes ở strict cached mode. Smoothing là ablation, cần record support halo.

SignRep official preprocessing resize về 224×224; preserve nó trong checkpoint baseline để không gán preprocessing improvement cho representation. Letterbox là default của DINO-based proposed pipeline, không là faithful SignRep reproduction.

### 7.6. Augmentation

| Strategy | MVP | Mở rộng / điều kiện |
| --- | --- | --- |
| Brightness/color nhẹ | Offline cache variant hoặc online extraction; paired spatial view thống nhất | Không dùng augmentation RGB khi chỉ có một immutable cache |
| Spatial translation/scale | Giới hạn để giữ cả tay/mặt; transform pose/ROI cùng cách | Đo robustness và minimal pairs |
| Horizontal flip | Off | Chỉ bật khi mapping handedness, semantics và dataset được kiểm chứng |
| Temporal reversal | Off | Có thể đổi nghĩa/movement |
| Frame dropping/time warp | Off trong geometric invariance claim | Nếu bật, target cần recompute và log dt/corners |
| Pose jitter | Off ở phase đầu | Confidence-aware noise ablation; không làm teacher target noisy mà không khai báo |
| Modality dropout | Optional sau raw JEPA | So inference với missing streams, giữ face evidence |
| Random erasing hand/face | Off | Sign-aware masks có objective riêng, tránh mất semantics vô kiểm soát |

Không có hai augmentation views “thật” nếu chỉ copy một cached feature tensor hai lần. Mọi experiment augmentation phải ghi view generation và cache versions.

## 8. Cache đặc trưng và tokenizer

### 8.1. FeatureBundle schema

Per-sample directory; dùng .npy memory-map cho arrays lớn, JSON cho metadata. Không pickle arbitrary objects để lưu simple arrays.

~~~text
cache/<extractor_hash>/<sample_id>/
  meta.json
  timestamps.npy            float64 [T]
  source_frame_index.npy    int64   [T]
  visual.npy                float16 [T,4,384]
  visual_valid.npy          bool    [T,4]
  pose_xy_body.npy          float32 [T,49,2]
  pose_observed.npy         bool    [T,49]
  pose_quality.npy          float32 [T,49]
  pose_coordinate_valid.npy bool    [T,49]
  raw_supports.npz          typed dependency encoding
~~~

Visual stream order cố định: `global, left_hand, right_hand, face`. Body/hand pose projection được tạo lúc batch load; không trộn quality vào visual feature dimension.

`meta.json` tối thiểu:

~~~json
{
  "schema_version": 1,
  "sample_id": "example",
  "extractor_id": "dinov2_vits14_generic",
  "extractor_hash": "COMPUTED",
  "weights_sha256": "COMPUTED",
  "feature_dim": 384,
  "visual_stream_order": ["global", "left_hand", "right_hand", "face"],
  "topology_hash": "COMPUTED",
  "detector_mode": "IMAGE",
  "temporal_dependency_policy": "same_frame",
  "normalization": "per_frame_shoulder_center_isotropic_scale",
  "timestamp_unit": "seconds",
  "source_sha256": "COMPUTED",
  "quality_kind": "observed_binary_hand_visibility_body"
}
~~~

[C] Reader reject cache sai dimension, stream order, weights hash hoặc coordinate definition. Không reuse features chỉ vì filename giống nhau.

### 8.2. Stream dimensions của proposed pipeline

| Stream | Shape trước projector | Shape sau projector |
| --- | --- | --- |
| Global/left/right/face | Mỗi stream `[B,T,384]` | Mỗi stream `[B,T,384]` |
| Body static | 7 joints × (x,y,quality,coordinate_valid) = 28 | `[B,T,384]` |
| Left-hand static | 21 × 4 = 84 | `[B,T,384]` |
| Right-hand static | 21 × 4 = 84 | `[B,T,384]` |

Timestamps trong cache giữ source seconds. Model/query dùng relative seconds từ start của segment/view đã chọn; teacher và student cùng origin, không reset theo first visible frame. Export embeddings giữ original timestamps.

Raw token count = `7*T`; mỗi token có stream ID và timestamp. Với T=48 là 336 raw tokens. Missing visual stream không có target và không là observed token. Pose token có partial missing joints phải giữ per-joint indicators; target của cả token chỉ eligible theo policy đã khai báo.

Projector mỗi type: `Linear(input_dim,384) → GELU → Linear(384,384) → LayerNorm`. Không dùng một linear chung nhận dimensions khác nhau bằng zero padding mà không type identity.

### 8.3. Cluster targets baseline

[D/H] K=256 per stream; MiniBatchKMeans hoặc k-means thường nếu dữ liệu nhỏ. Fit train only, sample frame pool theo source để hạn chế long-video dominance.

- Four visual streams và three pose streams có codebooks riêng.
- Pose feature space chuẩn hóa bằng train statistics; missing indicators không bị coi là hand coordinate.
- Cluster fit chỉ lấy valid vectors theo policy. Store centroids, seed, fitting source IDs, sample counts, feature hash.
- Cache cluster IDs optional; -100 = invalid/padded và ignore trong CE.
- Không gọi codebook được train trên proposed 7 streams là original SHuBERT. Original adapter có 4 streams/body14D.
- Cluster baseline và continuous JEPA dùng cùng temporal backbone, train sources và effective frame budget.
- Cluster count/silhouette/occupancy là diagnostics, không chứng minh pseudo-labels là ngữ nghĩa sign.

### 8.4. SignRep/SHuBERT/video-backbone cache adapters

- SignRep: output window vector + window start/end + raw 16-frame support; inference tiled windows, không gán mỗi window vector thành frame feature mà không mapping.
- SHuBERT: giữ layer axis hoặc layer subset; last-layer và learned mixture là hai downstream settings.
- V-JEPA/SignHiera: output token grid và original input sampling; cache pooled window feature nếu chỉ làm baseline, dense extraction nếu đã xác minh temporal stride.
- Masked pretraining với **cached contextual video features** không strict temporal hiding nếu features thấy cả vùng bị mask. Chỉ dùng chúng cho downstream backbone comparison, hoặc mask từ raw video và recompute backbone online.
- DINOv2 per-frame cache phù hợp strict temporal MVP khi extraction/ROI/normalization không có hidden temporal dependencies.

## 9. Geometry: paths, anchors và level-2 logsignature

### 9.1. Path families được định nghĩa độc lập

[D] Path lấy coordinates body-space từ `7.4. Chain ở đây là **concatenation theo feature channels tại cùng timestamp**, không phải đi dọc spatial joints thành một đường riêng. Không khẳng định đây là convention của unreleased Path-JEPA source.

Default topology 11 groups:

| Group | Joint IDs từ `7.2 | Path dimension d |
| --- | --- | --- |
| left_wrist / right_wrist | 7 / 28 | 2 |
| left_index_tip / right_index_tip | 15 / 36 | 2 |
| left_wrist_to_index | b[15] − b[7] | 2 |
| right_wrist_to_index | b[36] − b[28] | 2 |
| left_finger_chain | concat(b[11], b[15], b[27]) | 6 |
| right_finger_chain | concat(b[32], b[36], b[48]) | 6 |
| left_arm_chain | concat(b[1], b[3], b[5]) | 6 |
| right_arm_chain | concat(b[2], b[4], b[6]) | 6 |
| bimanual_wrist_chain | concat(b[7], b[28]) | 4 |

Edge là path của relative vector, không phải signature trên hai endpoints như hai samples của temporal path. Có thể mở rộng joints/fingers sau pilot; không mặc định tokenize 49 joints × mọi scale ngay từ đầu.

### 9.2. Windows và sampling

[D/H] Centers mỗi 4 selected samples. Window w∈{0.16,0.32,0.64}s chọn observations trong half-open interval `[center-w/2,center+w/2)` theo timestamps.

- Không cố định point count nếu variable fps.
- Window vượt đầu/cuối clip: MVP đánh dấu incomplete và không làm target.
- Cần≥2 distinct source observations, finite coordinates và valid supports.
- Store bounds, requested duration, actual sample span, point count và distinct raw-frame count.
- Không pad path bằng zero.
- ID=(sample,center,scale,group) ổn định, không phụ thuộc batch order.

T=48, stride4 → tối đa 12 centers. 11 groups×3 scales → tối đa396 geometry tokens trước boundary/validity filtering. Input-enabled model có tối đa732 tokens gồm336 raw tokens; target-only chỉ encode336.

Signature tính trên piecewise-linear path qua observations đã chọn. Sampling bỏ corner có thể đổi path, không được viện dẫn reparameterization invariance để biện minh arbitrary frame drop.

### 9.3. Degree-2 coordinates

Với increments Δx, Chen accumulation [28,29]:

$$a_{new}=a+\Delta x,\qquad
B_{new}=B+a\otimes\Delta x+\tfrac12\Delta x\otimes\Delta x.$$

Logsignature:

$$L_1=a,\quad L_2=B-\tfrac12a\otimes a,\qquad
\ell_2(x)=[a_1,\ldots,a_d,(L_2)_{ij}\text{ với }i<j].$$

L2 antisymmetric trong exact arithmetic. Ordered output dimension `d+d(d−1)/2`: d2→3, d4→10, d6→21. Library basis ordering/sign cần đối chiếu trước khi dùng chung cache.

Đoạn thẳng có level2 Lie coordinates bằng0. Đường L (0,0)→(1,0)→(1,1) có first level=(1,1), L2_01=0.5; đoạn chéo cùng endpoints có L2_01=0.

Không thêm scalar level0=1 vào `logsig`. Path-JEPA có một dimension không khớp convention đơn giản ở trường hợp d6; chưa có code xác định scalar/metadata/basis. Không ép dimension22 bằng giá trị tùy ý rồi gọi reproduction.

### 9.4. Reference implementation: NumPy float64

Mã độc lập dùng làm offline oracle. Port Torch bằng batched tensor operations; MVP target cache không cần gradient qua detector.

~~~python
import numpy as np

def logsig2_reference(points):
    """Ordered [level1, upper-triangular Lie level2], no level0."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need >=2 finite observations [T,D].")
    d = x.shape[1]
    a = np.zeros(d, dtype=np.float64)
    b = np.zeros((d, d), dtype=np.float64)
    for delta in np.diff(x, axis=0):
        b = b + np.outer(a, delta) + 0.5 * np.outer(delta, delta)
        a = a + delta
    lie2 = b - 0.5 * np.outer(a, a)
    upper = np.triu_indices(d, k=1)
    return np.concatenate([a, lie2[upper]])
~~~

[C] Geometry coefficients tính float32/64 ngoài fp16 autocast. Degree2 scale quadratically theo path scale; coordinate normalization khác descriptor normalization.

### 9.5. Static anchors và timing

[D] Descriptor mặc định:

~~~text
logsig2:           d + d(d-1)/2
anchors:          x_start, mean(x), x_end                 # 3*d
timing/motion:     actual_duration, arc_length, rms_speed # 3
native total dim:  4*d + d(d-1)/2 + 3
~~~

Total d2→12,d4→25,d6→42. Arc length=sum||Δx||; rms_speed=sqrt(mean||Δx/Δt||²), trên distinct observations dt>0. Duration theo seconds; dt=0 không eligible.

- Anchors giữ absolute body-space path values; edge anchors giữ relative shape.
- Raw hand pose vẫn giữ full21joints; chain3tips không thay thế handshape.
- Có anchors/timing thì full descriptor không pure translation/time-reparameterization invariant. Claim theo từng subfield.
- Ablate logsig-only / +anchors / +timing; RGB/static pose luôn giữ ở main model.
- Holds/tốc độ/nhịp có thể là linguistic information; không loại chỉ vì invariance nghe hấp dẫn.

### 9.6. Ordinary-motion controls và matched dimensions

[D] Same paths/windows, control M:

~~~text
x_start, x_mean, x_end, std(x),
delta_total, mean(velocity), rms_per_channel(velocity),
sum(abs(increments)), duration, arc_length, rms_speed
~~~

Native dimension `8*d+3`. Standardize train-only, fixed Gaussian projection về q32 cho **cả** M và signature+anchor+timing:

$$u=\operatorname{Standardize}_{train}(descriptor),\quad
g=uR,\quad R\in\mathbb{R}^{F\times32}.$$

- Seed R cố định, variance1/F, non-trainable; output standardized bằng train stats.
- Store matrices/stats/hashes; không fit val/test.
- Same decoder32D, query count, quality policy và λ ở E03/E04.
- Matching output/head dimensions không làm native information content/rank giống nhau; báo cáo native dimensions.
- Native-dim sensitivity sau pilot nếu compression quyết định winner.
- Level1-only vs level1+level2 kiểm signed-area terms vượt displacement không.
- Control M có anchors/timing, không gọi là velocity-only. Có thêm displacement/velocity-only riêng nếu cần attribution static anchors.

### 9.7. Lift và higher degrees

Off trong MVP.

- Discrete lead-lag: (x0,x0),(x1,x0),(x1,x1),…,(xn,xn), dimension2d.
- Fixed-delay: (x(t),x(t−δ)) cần interpolation và endpoint supports; không đồng nhất discrete lead-lag, không commute với nonlinear time warp nói chung.
- Time-channel: append t−window_start; physical time giữ pace nhưng đổi invariance.
- Degree3+: chỉ sau degree2 hơn motion controls; verify Witt dimension, basis và memory.
- Signatory optional: check extension/ABI compatibility [29,45]. Không thay main stack bằng version quá cũ vì optional feature.

## 10. Masking và dependency closure

### 10.1. Raw primitives / supports

[C] Raw M dùng source-frame RGB primitives hoặc (raw frame,joint). Online spatial mode cần pixel/patch primitives. Derived support D(v) bao gồm mọi preprocessing dependency:

~~~text
body-space wrist(t) -> wrist(t), both shoulders(t)
velocity(t)         -> both endpoint coords + normalization supports
interpolated point  -> both observed interpolation endpoints
logsignature window -> all path points + normalization/lift supports
SignRep window      -> all16 raw backbone input frames
~~~

Resampled aliases cùng raw frame là cùng primitive. ROI smoothing/tracking có temporal supports. Type/window schedule là public metadata; anchors/quality/box center từ hidden observations không được lén đưa vào query.

### 10.2. Closure

$$visible(v)\iff valid(v)\land(D(v)\cap M=\varnothing).$$

~~~python
# Pseudocode; set-based first, optimize bitsets/intervals after tests.
def close_masks(tokens, masked_raw_cells):
    visible, hidden = [], []
    for token in tokens:
        touched = bool(token.raw_support.intersection(masked_raw_cells))
        if token.valid and not touched:
            visible.append(token.id)
        elif token.valid:
            hidden.append(token.id)
    return visible, hidden
~~~

Không zero joints rồi recompute signature như observed token: tạo motion giả. Precompute descriptor/support rồi exclude nếu giao M. Recompute một visible segment được định nghĩa lại chỉ hợp lệ khi có ID/support khác, không giả là full window.

### 10.3. Eligibility khác visibility

- Semantic targets: valid raw-stream tokens thuộc declared hide mask.
- Geometry target-only strict temporal: **D(g)⊆M**, complete valid window; dự đoán geometry ẩn hoàn toàn.
- Geometry input-enabled: context chỉ có D(g)∩M=∅; target eligibility “any overlap” hoặc “fully hidden” là config riêng.
- Missing detector observations không là target chỉ vì bị mask.
- Padding không vào context/loss.
- Teacher nhận full **valid** observations; không biến missing thành truth.

Teacher thấy target observations là JEPA design. Leakage cần ngăn là hidden observations vào student context hoặc query.

### 10.4. Strategies và regimes

| Strategy | Raw hide policy | Protocol |
| --- | --- | --- |
| Synchronized temporal | RGB + all pose cùng timestamps/source frames | Strict cached temporal MVP |
| Anatomical pose | Finger/hand/arm joints | Cross-modal nếu RGB còn thấy region |
| Stream-temporal | Một visual stream span | Multi-stream prediction |
| Modality dropout | Toàn stream | Missing-input robustness |
| Motion-aware | Spans/regions theo score | Data-dependent choice, matched controls |
| Online RGB region | Che pixels/patches rồi run backbone | Strict spatial chỉ khi receptive-field handling đúng |
| Random flattened stream | Original SHuBERT code behavior | Faithful code baseline, khác stream-temporal |

[D/H] Temporal spans0.16–0.96s, requested fraction0.50. Merge spans, map raw aliases rồi tính union fraction; tối đa8 retries nếu không có semantic target/context.

- Min visible raw tokens32; short-clip exception có policy riêng.
- Empty geometry: differentiable zero, count0; vẫn có thể train semantic.
- Empty semantic: all ranks resample/skip update nhất quán.
- Log `mask_ratio_requested`, `raw_source_mask_ratio_actual`, `token_mask_ratio_after_closure` và counts per stream/family/scale.
- Overlapping geometry windows có thể hầu hết hidden dù raw mask50%.
- Với grouped hand token, một joint support hidden → exclude cả token.
- Later compare ratios0.30/0.50/0.70; không copy80%patch thành80%joints.
- Mixture strategies là ablation riêng; không cùng lúc đổi objective và mask.

### 10.5. Cache limitations

1. Cached global CLS thấy whole frame. Hide hand pose mà giữ RGB là cross-modal; không gọi strict spatial hiding.
2. Zero hand-ROI stream không xóa hand evidence khỏi global embedding.
3. Contextual SHuBERT/SignRep/video cached features có thể thấy target time. Dùng cho downstream backbone comparison; strict SSL phải mask raw/recompute online.
4. MVP per-frame detector/ROI processing. VIDEO tracking history chưa biết đầy đủ → conservative support, recompute hoặc approximate protocol [40].
5. Clip-wide mean/scale/crop khiến support toàn clip. Nếu closure che hết context, đổi preprocessing/recipe; không bỏ dependency để giữ score.

### 10.6. Counterfactual test

[C] Hai raw samples giống ngoài M, khác lớn trong M:

- Retained student values và query metadata phải giống nhau trong tolerance.
- Teacher targets được phép khác.
- Cross-modal test chỉ thay primitives thật sự hidden, không claim strict spatial.
- Set-intersection test chưa đủ nếu preprocessing dependency chưa được khai báo.

## 11. Encoder, predictor và teacher

### 11.1. Raw JEPA baseline

[D/H] Per-stream MLP projectors `8.2, timestamp encoding từ relative physical seconds, learnable stream/type embeddings; Transformer6blocks,D384,heads6,MLP ratio4,pre-LN. Dropout/stochastic depth0 ở initial recipe. Final LN; return layers2/4/6 optional. Bidirectional offline.

$$e_v=P_{type}(value_v)+E_{type}+E_{time}(t_v)+E_{path}+E_{scale}.$$

Raw tokens không có path/scale identity. Target-only teacher không nhận geometry input.

~~~text
full raw:        [B,N_raw,D]
visible IDs:     [B,N_visible_max]       # -1 padded
student input:   [B,N_visible_max,D]
student padding:[B,N_visible_max]       # True ignore in nn.Transformer
teacher full:   [B,N_full_max,D]
query metadata: [B,N_target_max,D]
prediction:     [B,N_target_max,D]
~~~

Pack visible tokens; không đưa zero hidden observation vào encoder như observed. Mask-token architecture có thể dùng dưới variant riêng.

### 11.2. Predictor

[D/H] 2 TransformerDecoder blocks,D384,heads6; self-attention queries + cross-attention visible context.

Query chỉ gồm type/path/scale/window times và learned mask token, không pose/anchor/quality/ROI/teacher value.

- Semantic head outputs384D.
- Geometry head outputs32D projected target; IDs riêng.
- Native dimensions dùng typed heads và parameter-count comparison.
- Query self-attention không có ground truth, nên hợp lệ.
- Predictor bỏ khi inference.

### 11.3. EMA teacher

[C] Deepcopy online tokenizer/projectors+encoder; exclude predictor. Initialize same weights; requires_grad=False, eval,no_grad. Learnable time/type embeddings cũng EMA.

$$\theta_t\leftarrow\tau\theta_t+(1-\tau)\theta_s.$$

[H] τ cosine0.996→1 theo **successful optimizer steps**.

- Floating buffers update phù hợp, integer buffers copy; LN giảm BN ambiguity.
- Teacher absent-stream không bịa observation.
- Không EMA một online branch không nhận gradient.
- Teacher parameters không trong optimizer.
- Không EMA khi AMP step skip.

### 11.4. Dense / clip outputs

Unmasked encoder → scatter raw slots [B,T,7,D] → pool valid streams mỗi frame → dense z[B,T,D]+timestamps+validity → optional mean/attentive clip pool.

Geometry-input variant ảnh hưởng context qua attention; default dense pooling vẫn7raw slots, không xem geometry centers là additional frames.

Layer mixture fitted downstream train, normalized scalar weights; report trainable params. Last-layer-only control; same mixture policy giữa methods.

### 11.5. Geometry variants

**D0 target-only first:** raw7streams; semantic EMA teacher; deterministic normalized geometry prediction. Inference không cần signature nhưng còn pose cho static inputs.

**D1 input-enabled:** append full teacher geometry; online geometry chỉ retained theo closure. Online projectors nhận gradients qua visible geometry. Inference tính/cached descriptors thật.

**D2 learned target-only:** φ(g) train jointly với prediction+VC (GFP-inspired adaptation). Nếu không stop-grad target, ghi gradient flow. Nếu dùng EMA φ_t, online φ_s phải có objective nhận gradient riêng. Không update EMA từ projector chưa bao giờ học. Keep deterministic anchor/geometry objective như safeguard nếu hữu ích; named method riêng.

**D3 RGB-only distillation:** teacher multimodal/studentRGB; train/eval separate. Bỏ pose input miễn phí ở deployment là distribution shift.

## 12. Objectives và chống collapse

### 12.1. Semantic targets

[D/H] Normalized L1 default, cosine alternative; Path-JEPA cosine không là loss của mọi JEPA [1,18].

$$y_v=\operatorname{stopgrad}(\operatorname{LayerNorm}(E_t(full)_v)),\qquad
L_{sem,k}=\frac1{|M_k|}\sum_{v\in M_k}\frac{\|\hat y_v-y_v\|_1}{D}.$$

Average across active stream types; empty type excluded, log count. Cosine=1−dot/max(norm product,epsilon), norms/loss fp32. Normalization/epsilon không tự giải quyết collapse.

### 12.2. Geometry

[D/H] D0 g32D fixed, standardized, no-grad:

$$L_{geo}=\operatorname{mean}_{active(group,scale)}
\frac{\sum_{eligible}q_v\,\operatorname{SmoothL1}(\hat g_v,g_v)}
{\max(\sum_{eligible}q_v,\epsilon)}.$$

MVP q=1 valid/0invalid, SmoothL1 beta1. λ_geo sweep0/0.1/0.3; initial0.1. Same weights/group-scale means cho ordinary controls. Log projected/native variance, target counts và scales. Empty=zero/count0, không “solved”.

### 12.3. Objective composition

~~~text
L = L_sem + lambda_geo*L_geo + lambda_cluster*L_cluster
  + lambda_var*L_variance + lambda_cov*L_covariance
  + lambda_pose*L_pose_prior + lambda_contrast*L_temporal_contrast
  + lambda_dense*L_visible_dense
~~~

E02/E04 initial sem1,geo0/0.1, extras0. Nếu collapse, registered VC variant bật cho **cả controls**. Không bật nhiều losses đồng thời rồi không biết attribution.

VC samples một embedding/clip hoặc independent source-balanced samples. Không hàng nghìn adjacent tokens của một video làm independent batch. Starting var0.01/cov0.001 là [H], không là official full VICReg recipe. SIGReg là LeJEPA alternative, không phải path signature.

### 12.4. Cluster CE, dense/deep và extras

**Cluster:** per-stream256 logits, labels−100 invalid, average streams; codebooks train-only. Proposed cluster control cùng mask/backbone JEPA; original SHuBERT reproduction khác config.

**Deep:** normalize layers2/4/6 individually; average target hoặc separate heads là named variants. Visible-context loss λ_dense0/0.1, normalized independently không lấn masked targets do counts. V-JEPA2.1 distance-weighting/deep recipe cần code trace nếu gọi reproduction [20].

**Priors:** coordinates/angles/distances/activity; angle sin/cos; confidence-validity. Style discriminator/gradient reversal là separate stage nếu implement; incomplete SignRep source không đủ claim exact loss reproduction.

**Reconstruction:** RGB/hand heatmap/HOG cần corresponding raw data; cached features chỉ feature reconstruction. Giữ reconstruction nếu tốt.

**Temporal contrast:** sampling theo seconds, real lengths; distant frames có thể cùng sign/hold. Empty pairs skip term/count; actual-pair normalization, không double-divide batch mean. Strict SSL không dùng text length margin.

**Paired:** video-text alignment/translation cho best-system; data/supervision manifest riêng.

### 12.5. Monitors

Mỗi500 successful steps [H] trên fixed train diagnostic set:

- Feature per-dim std, covariance spectrum/effective rank.
- Off-diagonal cosine, student/teacher norm.
- Per-stream/group/scale loss/count/gradients.
- Static/low-motion clips, phonological probes/minimal pairs nếu có labels.
- Pose-quality/missing-stream buckets.
- Train-only kNN/retrieval sanity; labels không vào SSL optimizer.

Loss giảm mà rank/cosine/probes collapse: không chọn “best” bằng loss thấp. Sửa mask/normalization/regularization rồi chạy lại controls.

## 13. Training runtime và distributed correctness

### 13.1. Phases và optimizer

[D/H] CPU synthetic10steps,D48/2layers → real small overfit100–500steps → pilot10K → confirm100K candidate,≥3seeds. Đây không tái lập400KSHuBERT/500KSignRep.

AdamW lr3e−4,betas(0.9,0.95),wd0.05; bias/norm no-wd; warmup5%,cosine tới3e−6,grad clip1. Frozen extractor ngoài optimizer. Online backbone lower lr1e−5 candidate; LoRA/fullFT riêng. Log trainable/total params.

### 13.2. AMP và update order

bf16 nếu support, fp32 fallback ghi rõ; fp16 GradScaler. Geometry/teacher normalization/denominators fp32.

~~~python
# Pseudocode, actual AMP/distributed implementation must be verified.
for update in planned_optimizer_updates:
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accumulation_steps):
        batch = next_batch()
        masks = sample_and_close(batch)
        with no_grad_teacher():
            teacher_targets = teacher_forward_full(batch)
        with autocast_student():
            student = online_forward_visible(batch, masks)
            prediction = predictor(student, masks.query_metadata)
            loss = globally_normalized_loss(
                prediction, teacher_targets, batch, masks
            )
        backward(loss / accumulation_steps, no_sync_if_not_last_micro)
    unscale_if_fp16()
    step_allowed = global_finite_consensus()
    if step_allowed:
        clip_grad_norm()
        step_succeeded = optimizer_or_scaler_step_with_consensus()
    else:
        update_scaler_on_global_overflow_if_fp16()
        step_succeeded = False
    if step_succeeded:
        scheduler.step()
        ema_update()
        successful_step += 1
    save_periodic_checkpoint_if_needed()
~~~

Không double optimizer.step/scaler.step. Consensus trước step; all ranks cùng successful-step count. Finite pre-backward loss không đủ xác định optimizer success.

### 13.3. DDP normalization

Per-type local sum N_r/count C_r, world size W:

$$L_r=W\,N_r/\sum_q C_q.$$

DDP average gradients; factorW cho gradient global mean. All-reduce detached counts; logging separately all-reduce sums. Global active-type count dùng làm type-average denominator.

Default accumulation average K microbatch global-type means. Không tương đương exact total-token mean nếu counts khác. Nếu cần exact update-wide mean, prefetch counts của K và denominator chung. Controls cùng semantics.

- All ranks collectives same order kể cả local count0.
- Empty geo không làm một rank skip backward riêng.
- Dynamic unused streams: find_unused_parameters=True initial hoặc connected zero losses; optimize sau correctness.
- Sampler set_epoch; worker/rank/mask RNG saved.
- Deduplicate sampler padding IDs khi evaluate.
- Test deterministic one-GPU vs DDP gradient update.

### 13.4. Padding / API semantics

Ba masks riêng: invalid/padding, pretext hide, target eligibility. MVP nn.Transformer batch_first=True; boolean padding True=ignore. SDPA boolean attn_mask True=**allow**, ngược MHA [41]; wrapper convert rõ và test.

Không [T,T] mask cho flat [T*7,T*7]. Padding queries không pool/loss. Không nan_to_num toàn model để giấu bug.

### 13.5. Resume

~~~text
online tokenizer/encoder/predictor/heads; teacher tokenizer/encoder
optimizer/scheduler/scaler
successful_step/attempted_batches/epoch/sampler state
Python/NumPy/Torch CPU/CUDA/mask RNG states
resolved config + hash
manifest/cache/topology/source/checkpoint/statistics identifiers
best-selection metric/protocol
~~~

Atomic temp→rename. Resume reject incompatible dims/topology/protocol; explicit fine-tune reset là newrun. Multiworker exact data-order replay cần saved sampler plan; nếu chỉ approximate resume thì ghi đúng. Tiny interrupted/uninterrupted test compare online+teacher weights. Save last và best registered validation metric, không chọn bằng test. Failed/budget-stopped run vẫn lưu memo.

## 14. Config hoàn chỉnh và command contracts

### 14.1. Initial E04 target-only config

[D/H] Đây là proposed recipe có đủ fields để Agent triển khai validator. Không là published optimal setup. Relative paths resolve từ project root, không phụ thuộc cwd tùy ý.

~~~yaml
schema_version: 1
experiment:
  id: E04_logsig_target
  seed: 17
  supervision_regime: sign_only_ssl
  output_dir: artifacts/E04_logsig_target/seed17
  primary_validation_metric: islr_per_class_top1
  evaluation_protocol: asl_citizen_official_signer_disjoint_v1

data:
  adapter: asl_citizen
  root: data/asl_citizen
  train_manifest: data/manifests/asl_citizen.train.jsonl
  val_manifest: data/manifests/asl_citizen.val.jsonl
  test_manifest: data/manifests/asl_citizen.test.jsonl
  pretrain_split: train
  expose_labels_to_pretrain: false
  sampler: uniform_source_then_segment
  segment_samples: 48
  target_fps: 25.0
  short_clip_policy: pad_invalid
  num_workers: 4
  pin_memory: true
  persistent_workers: true

cache:
  kind: per_frame_dino
  root: data/cache
  extractor_registry_key: dinov2_vits14
  validate_hashes: true
  visual_dim: 384
  visual_streams: [global, left_hand, right_hand, face]
  detector_mode: IMAGE
  temporal_dependency_policy: same_frame
  roi_policy: per_frame_letterbox
  tensor_dtype_on_disk: float16
  timestamps_dtype: float64

pose:
  topology: upper7_hands42_v1
  coordinate_system: per_frame_body_isotropic
  rotation_canonicalization: false
  include_hand_local: false
  quality_policy: observed_binary_hand_visibility_body
  min_body_visibility: 0.5
  geometry_requires_all_joints_observed: true
  interpolation: none
  smoothing: none

geometry:
  enabled: true
  role: target_only
  descriptor: logsig2_anchor_timing
  degree: 2
  basis: level1_then_upper_lie2
  include_level0: false
  path_groups: default_11_groups_v1
  window_seconds: [0.16, 0.32, 0.64]
  center_stride_samples: 4
  incomplete_boundary_window: invalid
  min_distinct_observations: 2
  lift: none
  append_time_channel: false
  compute_dtype: float32
  projection:
    kind: fixed_gaussian
    output_dim: 32
    seed: 991
    fit_statistics_split: train
    statistics_path: data/targets/logsig2_train_stats.npz
    matrices_path: data/targets/logsig2_projection.npz

masking:
  strategy: synchronized_temporal
  regime: strict_temporal
  requested_frame_ratio: 0.5
  span_seconds: [0.16, 0.96]
  map_resampled_aliases: true
  closure: full_declared_support
  geometry_target_eligibility: fully_hidden
  min_visible_raw_tokens: 32
  short_clip_min_context: quarter_of_valid_tokens
  retries: 8
  empty_semantic_policy: resample_then_global_skip
  empty_geometry_policy: zero_term_with_count

model:
  raw_streams: [global, left_hand, right_hand, face, body, left_pose, right_pose]
  raw_dimensions: [384, 384, 384, 384, 28, 84, 84]
  embed_dim: 384
  projection_hidden_dim: 384
  time_encoding: physical_seconds_sinusoidal
  type_embeddings: true
  encoder:
    layers: 6
    heads: 6
    mlp_ratio: 4
    norm_first: true
    dropout: 0.0
    stochastic_depth: 0.0
    causal: false
    return_layers: [2, 4, 6]
  predictor:
    layers: 2
    heads: 6
    embed_dim: 384
    geometry_output_dim: 32
    query_values: metadata_only
  teacher:
    kind: ema
    initialization: exact_online_copy
    tau_start: 0.996
    tau_end: 1.0
    tau_schedule: cosine_successful_steps
    eval_mode: true
  pooling:
    dense: valid_stream_mean
    clip: valid_frame_mean

loss:
  semantic: normalized_l1
  semantic_weight: 1.0
  semantic_normalization: global_active_type_mean
  geometry: smooth_l1
  geometry_beta: 1.0
  geometry_weight: 0.1
  geometry_normalization: global_active_group_scale_mean
  cluster_weight: 0.0
  variance_weight: 0.0
  covariance_weight: 0.0
  pose_prior_weight: 0.0
  temporal_contrast_weight: 0.0
  visible_dense_weight: 0.0
  accumulation_normalization: microbatch_global_means

training:
  max_successful_steps: 10000
  microbatch_per_gpu: 8
  gradient_accumulation: 4
  precision: bf16_if_supported_else_fp32
  optimizer: adamw
  learning_rate: 0.0003
  final_learning_rate: 0.000003
  betas: [0.9, 0.95]
  weight_decay: 0.05
  decay_bias_norm: false
  warmup_fraction: 0.05
  grad_clip_norm: 1.0
  ddp_find_unused_parameters: true
  finite_policy: globally_skip_and_record
  log_interval: 50
  diagnostic_interval: 500
  checkpoint_interval: 1000
  validation_interval: 2000
  keep_last: 3
  resume: null

evaluation:
  split_during_development: val
  probe_protocol: fixed_linear_train_labels
  probe_config: configs/downstream/islr_linear.yaml
  checkpoint_selection: registered_validation_probe
  test_access_during_sweep: false
  report_metrics: [per_instance_top1, per_class_top1, top5]
  prediction_export: true

runtime:
  deterministic_smoke: false
  allow_random_pretrained_fallback: false
  allow_missing_provenance: false
  max_runtime_hours: 8.0
~~~

`max_runtime_hours` là budget guard, không hứa 10K steps hoàn thành trong8giờ. Trước run, profile và estimate; budget stopped checkpoint được ghi. Nếu máy khác/budget khác, đổi config trước experiment và ghi lý do.

### 14.2. Required derived fields và validators

- Config include unknown keys → error, không ignore.
- `embed_dim % heads == 0` cho encoder và predictor.
- Raw dims match cached extractor và coordinate schema.
- Geometry matrices/stats được fit train và match descriptor/hash.
- `target_only` không append geometry encoder tokens.
- `strict_temporal` + contextual/video-mode cache chưa có correct supports → reject.
- Same-frame cache + temporal tracking/smoothing flag → reject unless support policy covers dependencies.
- `sign_only_ssl` không đọc label/text sampler fields.
- Geometry degree2 với declared basis; degree3 chưa implement → error.
- Scientific run không synthetic cache, unresolved hash hoặc fake model asset.
- Non-CPU backbone cần available device; không silent random fallback.
- Duplicate config key → error.
- Data labels/source IDs/mirror metadata không được model tokenize.
- Strict temporal masked set mở rộng tất cả resampled aliases.

Resolved config lưu full values, source overrides và hash canonical JSON; không chỉ filename base.yaml.

Config validation có hai stages: structural validation không yêu cầu cache/weights/stats đã tồn tại; command-specific validation của verify/train/evaluate enforce assets thật. targets fit ghi statistics/matrices ở output paths mà không yêu cầu chúng đã được tạo. Cache build không load geometry statistics nếu stage chỉ extraction. Smoke dùng fixture provider, không cố import ASL Citizen từ nonexistent data root.

### 14.3. Experiment overrides

Lưu full config ở `14.1 thành configs/base.yaml. Agent implement composition bằng field extends: resolve path từ directory của file config, detect cycles, deep-merge dictionaries và replace lists. Sau resolve, bỏ extends khỏi runtime schema nhưng lưu config chain trong provenance. Mỗi experiment gồm config đầy đủ đã resolve; ví dụ dưới đây là delta có inheritance tường minh.


~~~yaml
# E02_raw_jepa.yaml: same data/mask/backbone/runtime as E04.
extends: ../base.yaml
experiment:
  id: E02_raw_jepa
  output_dir: artifacts/E02_raw_jepa/seed17
geometry:
  enabled: false
loss:
  geometry_weight: 0.0
~~~

~~~yaml
# E03_motion_target.yaml: same query architecture, q=32 and target policy.
extends: ../base.yaml
experiment:
  id: E03_motion_target
  output_dir: artifacts/E03_motion_target/seed17
geometry:
  descriptor: ordinary_motion_anchor_timing
  projection:
    statistics_path: data/targets/motion_train_stats.npz
    matrices_path: data/targets/motion_projection.npz
loss:
  geometry_weight: 0.1
~~~

Override merge phải deep merge dicts, replace lists; sau merge validate và update output/run hash. Mỗi experiment phải có `id` và output riêng. Source stats cho motion và signature khác nhau nhưng same fitting pool.

~~~yaml
# smoke.yaml: synthetic_fixture backend, không là scientific result.
extends: base.yaml
experiment:
  id: smoke
  output_dir: artifacts/smoke
cache:
  kind: synthetic_fixture
model:
  embed_dim: 48
  projection_hidden_dim: 48
  encoder:
    layers: 2
    heads: 4
    return_layers: [1, 2]
  predictor:
    layers: 1
    heads: 4
    embed_dim: 48
training:
  max_successful_steps: 10
  microbatch_per_gpu: 2
  gradient_accumulation: 1
  precision: fp32
  validation_interval: 10
  checkpoint_interval: 10
runtime:
  deterministic_smoke: true
  allow_missing_provenance: true
~~~

[C] Smoke command bypass external downloads và dataset requirements bằng generated fixture metadata, không bật `allow_random_pretrained_fallback` cho scientific config. Fixture output phải gắn flag `synthetic_correctness_only`.

### 14.4. CLI contracts cần implement

Agent cần tạo E04 config extends ../base.yaml (giữ default), E01 đổi semantic_weight=0/cluster_weight=1/geometry disabled, và E00/downstream configs theo protocol `15. Teacher chỉ chạy khi objective cần semantic latent targets; cluster-only không cần EMA forward. Record cost đúng từng stage.

Đây là commands của **package sẽ được Agent viết**. Chưa có executable `signpath` trong deliverable Markdown này.

~~~bash
# 0. Build/inspect environment
python -m pip install -e ".[dev]"
signpath doctor --output artifacts/doctor.json
signpath config validate --config configs/base.yaml

# 1. No-network CPU correctness smoke
signpath smoke --config configs/smoke.yaml --device cpu
pytest -q tests/unit tests/integration

# 2. Ingest authorized/local data, audit before crops
signpath data import --adapter asl_citizen --root data/asl_citizen \
  --output data/manifests
signpath data audit --manifest-dir data/manifests \
  --output artifacts/data_audit

# 3. Resolve and verify weights (not part of CPU smoke)
signpath assets fetch --registry registries/checkpoints.yaml \
  --model dinov2_vits14

# 4. Preprocessing/cache, split name logged in every output
signpath preprocess --config configs/base.yaml --split train \
  --stage pose_roi --shards 8
signpath cache build --config configs/base.yaml --split train
signpath cache build --config configs/base.yaml --split val
signpath cache build --config configs/base.yaml --split test

# 5. Train-only target fitting
signpath targets fit --config configs/base.yaml --kind geometry \
  --split train
signpath targets fit --config configs/experiments/E03_motion_target.yaml \
  --kind geometry --split train
signpath targets fit --config configs/experiments/E01_cluster.yaml \
  --kind kmeans --split train

# 6. Required leakage/data validation gate
signpath verify pipeline --config configs/base.yaml \
  --output artifacts/pipeline_checks.json
signpath profile --config configs/base.yaml --steps 20 \
  --output artifacts/profile.json

# 7. Training and resume
signpath train --config configs/experiments/E02_raw_jepa.yaml
signpath train --config configs/experiments/E04_logsig_target.yaml
signpath train --config configs/experiments/E04_logsig_target.yaml \
  --resume artifacts/E04_logsig_target/seed17/checkpoints/last.pt

# 8. Freeze protocol and evaluate
signpath downstream fit --config configs/downstream/islr_linear.yaml \
  --encoder artifacts/E04_logsig_target/seed17/checkpoints/best.pt
signpath evaluate --run artifacts/E04_logsig_target/seed17 \
  --split val --export-predictions
signpath experiments summarize --root artifacts \
  --output artifacts/comparison.csv

# 9. Final test after selected config/head/metric has been frozen
signpath evaluate --run artifacts/selected_run \
  --split test --export-predictions

# 10. Reusable embeddings/deployment smoke
signpath embed --run artifacts/selected_run --input data/local/demo.mp4 \
  --output artifacts/demo_embeddings.npz
signpath export --run artifacts/selected_run --format torch_export \
  --output artifacts/exported_encoder
~~~

Test cache build ở step4 được phép xử lý unlabeled video để sẵn evaluation, nhưng mọi fitting ở step5 chỉ train. Để giảm development-test exposure, có thể hoãn test cache tới step9; metadata/labels không dùng trong tuning.

DDP entry contract:

~~~bash
torchrun --standalone --nproc_per_node=2 -m signpath.cli train \
  --config configs/experiments/E04_logsig_target.yaml
~~~

Agent phải implement module execution entry và init_process_group khi launcher cung cấp rank env. `signpath` console script và `python -m signpath.cli` phải cùng semantics.

### 14.5. Artifact contract mỗi command

| Command | Output |
| --- | --- |
| data import/audit | Split JSONL, actual availability counts, audit report, manifest hashes |
| preprocess | Raw landmarks, ROI transforms, timestamps, failure list |
| cache build | FeatureBundle arrays/meta; extraction provenance; resume ledger |
| targets fit | Train stats/projections/codebooks + fitting-source IDs/hashes |
| verify pipeline | Assertions, counterexamples, mask counts; failing sample IDs |
| train | resolved config, logs JSONL, checkpoints, diagnostics, step/budget counters |
| downstream fit | head checkpoint, encoder identity, selection metric, decoder settings |
| evaluate | metrics JSON, predictions, protocol signature, CI if requested |
| embed | dense and clip embeddings with timestamps/validity |
| export | model plus topology/preprocessing/schema/weights metadata |

Không overwrite một run khác chỉ vì config id trùng. Run folder identity gồm experiment/seed/config hash hoặc explicit run UUID.

## 15. Downstream evaluation

### 15.1. ISLR: bắt đầu bằng frozen linear probe

[D/H] Protocol:

1. Freeze foundation extractor và temporal encoder; unmasked deterministic embeddings.
2. Mean pool real frames, optionally L2 normalize theo fixed config.
3. Linear softmax classifier fit bằng train labels.
4. LR sweep {1e−3,3e−3,1e−2}; max100epochs, same trials cho mọi method; early stopping validation primary metric.
5. Report PI Top1, PC Top1, Top5 và class counts.
6. Cùng clip/window schedule và feature layer policy giữa methods.

Frozen attentive probe thêm trainable context/head; không gọi linear probe. Temporal encoder fine-tuning khác foundation backbone fine-tuning. Với immutable DINO cache, “full fine-tune” chỉ có nghĩa mọi trainable temporal module; raw visual encoder vẫn frozen. Báo cáo từng module frozen/LoRA/full.

PC Top1 = mean per-class accuracy với class support>0 theo fixed taxonomy; báo absent classes. Không gộp std across seeds thành confidence interval rồi quên split sampling uncertainty.

[D/H] Finetune candidate lr encoder1e−4, head1e−3; focal/class-balanced/class-distribution losses là options có ablation. Nếu dùng SignRep auxiliary supervised loss, tách gain khỏi SSL checkpoint.

ISLR full-clip inference:

- Long clips tile length48, overlap16sample default, aggregate clip evidence theo preregistered mean/logit pooling.
- Short clips pad validity, không normalize loss bằng padded length.
- Compare uniform fixed-frame baseline khi benchmark official recipe yêu cầu.
- Do not train random segment rồi test chỉ prefix đầu clip.

### 15.2. Retrieval

[C] Định nghĩa gallery/query:

- Query val/test, gallery train hoặc official dictionary split, không chứa same-source duplicate.
- Relevance theo official sign label hoặc annotated semantic relation; declare.
- L2 normalized vectors, cosine retrieval; ties stable.
- R@1/5/10, MRR; PC retrieval nếu imbalance.
- Không tính same clip như relevant hit.
- Multilingual text-sign retrieval có paired model khác; không lẫn sign-to-sign metric.

Use FAISS optional; NumPy exact cosine đủ pilot. Check exact-vs-index topK trên subset khi dùng ANN.

### 15.3. Phonological probes / minimal pairs

[D] Sem-Lex hoặc label source đáng tin [33]:

- Train probes riêng cho handshape, location, orientation, movement, non-manual khi annotations có.
- Linear/low-capacity heads, same data/trials across methods.
- Macro-F1/accuracy per feature, unknown labels masked.
- Không suy phonology từ English gloss bằng heuristics rồi gọi ground truth.
- Minimal pairs cùng movement khác shape/location; static hold subset; two-hand coordination.
- If no labels, report missing experiment, dùng deterministic input retention tests; không gán một semantic score giả.

### 15.4. CSLR / CTC

[D/H] Dense z[B,T,D] → temporal head/linear vocab+blank. CTC uses lengths sau actual temporal subsampling; labels là gloss sequence.

- Store temporal stride và timestamp receptive fields.
- Target length phải≤usable input length; check repeated labels cần enough blank positions.
- Empty/invalid examples report; không zero_infinity để silently hide mọi error.
- WER components substitution/deletion/insertion, same tokenizer/case.
- Greedy và beam search là separate evaluation settings.
- Language model/gloss supervision declare.
- Local-window geometry thích hợp hơn một pooled global signature duy nhất cho sequence tasks.

### 15.5. SLT: encoder comparison với decoder cố định

[D] First continuous ASL task How2Sign; OpenASL transfer/scale sau overlap audit. Keep decoder initialization, vocabulary/tokenizer, training sources, optimization budget và search identical giữa representations.

Hai đường implementation:

1. Export dense embeddings về interface của SSVP-SLT translation code [8], adapter timestamps/lengths đúng.
2. New Transformer seq2seq decoder với cross-attention dense z. Hoặc T5 pretrained decoder route, nhưng initialization text supervision phải khai báo.

Không claim exact SSVP-SLT reproduction nếu replacement features, decoder hoặc preprocessing khác. Original translation instructions có T5 precision caveat; chạy numerical smoke, không mặc định fp16 cho mọi pretrained text model.

Metric contract:

- SacreBLEU corpus BLEU, export full signature/version/tokenizer/case.
- ROUGE-L implementation/tokenization fixed; word-level WER khi task appropriate.
- BLEURT optional nếu checkpoint và language phù hợp; pinned model/version.
- Beam size, max decode length, length penalty fixed/preregistered.
- Save one prediction per sample ID; ID/reference alignment check before metrics.
- Sentence overlap và signer overlap report separate [30,31].
- FLEURS-ASL zero-shot only nếu chưa tune/model-select trên nó.

RGB encoder low-cost init + strong text decoder có thể tăng SLT trong khi SSL representation gain nhỏ. Bảng encoder utility và best-system performance riêng.

### 15.6. Seeds và uncertainty

[D/H] Pilot1seed chỉ screening. Main ≥3seeds:17,29,43; same seeds/head search budget cho controls.

- Mean±seed std.
- Paired bootstrap metric differences grouped by signer hoặc source; không iid bootstrap hàng nghìn clips tương quan.
- Signer ID thiếu: group source, report limitation.
- CI of difference, sample counts; không chỉ một p-value.
- Frozen head cùngseed và split giúp paired comparison.
- Model selection dùng val; khóa chosen method trước test. Mọi failed pilot logs giữ lại để biết selection process.

## 16. Lộ trình implementation và các vòng lab

### 16.1. Milestones theo dependency, không theo lịch cố định

| Milestone | Work items cụ thể | Acceptance / output |
| --- | --- | --- |
| M0: contracts | Package/CLI/config/schema, source registry, fixtures, doctor | Help/validation chạy; schema rejects wrong dims/unknown keys |
| M1: data | ASL Citizen/local adapters, source split audit, PTS decode, sampler | Manifest/hashes; fixture VFR và duplicate aliases pass |
| M2: extraction | Pose topology/quality, per-frame coordinates/ROI, DINO loader/cache | Real clip overlay đúng handedness; state dict load report; cache roundtrip |
| M3: strong baseline | Frozen visual mean/linear; SHuBERT/SignRep adapters khi assets có | Val probe baseline + provenance; không random fallback |
| M4: geometry | Degree2, anchors/timing, windows/supports, motion control, train stats | Analytical/metamorphic tests pass; native dims3/10/21, projected32 |
| M5: masking | Raw M, closure/eligibility, regimes, counterfactual tests | No retained alias; actual ratios/counters |
| M6: raw SSL | Projectors, encoder, queries, predictor, teacher, optimizer/resume | CPU synthetic and real minibatch end-to-end; teacher no grads |
| M7: target SSL | Deterministic geo/motion loss, cluster baseline, monitor | E01–E04 same runtime contracts; empty-target/DDP normalization pass |
| M8: downstream | ISLR/retrieval/phonology; functioning CTC and seq2seq heads | Fit/eval/export pipeline; toy CTC/SLT integration, real val when data exists |
| M9: research pilot | Frozen protocol, fair trials, seeds, diagnostics/failure buckets | Decision memo; choose/drop geometry by evidence |
| M10: scaling | Input-geometry/deep/dense/adaptation, continuous transfer, best-system | Matched utility results + best-system table; compute/deployment report |

M8 phải có heads/metrics hoạt động; không đủ tạo tên class trả tensor zeros hoặc để CTC/SLT thành TODO. Với thiếu continuous dataset, implement và test trên toy sequence, ghi rõ chưa có real continuous experiment.

### 16.2. Một vòng lab phải để lại gì?

1. **Hypothesis:** ví dụ level2 tăng movement probe khi anchors giữ location.
2. **Alternatives:** chỉ displacement, thêm context, regularization hoặc detector quality cũng có thể giải thích gain.
3. **Minimal discriminating experiment:** E03 vs E04, same masks/data/head.
4. **Pre-run checks:** source hashes, split audit, leakage, target availability và finite gradients.
5. **Run:** config locked, budget/frame exposure và attempts logged.
6. **Analysis:** primary metric, seed dispersion, phenotype/error buckets, cost.
7. **Decision:** keep/revise/stop; thay đổi và experiment kế tiếp.
8. **Re-run requirement:** nếu sửa bug/protocol, đánh dấu previous runs invalid/diagnostic, không dùng scores cũ cho comparison mới.

Agent không cần đợi một “review” thủ công cho mỗi reversible code change. Nhưng phải giữ audit trail của quyết định nghiên cứu và không dùng test set làm tuning oracle.

### 16.3. Decision memo template

~~~markdown
# Decision memo: iteration ID
- Research question:
- Hypothesis and competing explanations:
- Data/model/protocol/config hashes:
- Baselines and changed factors:
- Training budget; unique frames; GPU/preprocessing time:
- Correctness gates:
- Validation results by seed:
- Phonology/robustness/missing-modality effects:
- Failure buckets and example IDs:
- Claim supported:
- Claim not yet supported:
- Decision: keep / revise / stop
- Next experiment and why it distinguishes alternatives:
~~~

## 17. Ablations và quyết định chọn mô hình

### 17.1. Minimum comparison matrix

| ID | Same foundation data/features? | Input | SSL targets / change | Câu hỏi |
| --- | --- | --- | --- | --- |
| E00 | Yes | Visual only | Frozen extractor + supervised probe | SSL temporal model vượt image features chưa? |
| E01 | Yes | Raw visual+pose | Cluster CE | Continuous JEPA tốt hơn hidden-unit control? |
| E02 | Yes | Raw visual+pose | Semantic latent | Baseline trước geometry |
| E03 | Yes | Raw visual+pose | Semantic + ordinary motion/anchors/timing | Motion-statistics target đã đủ chưa? |
| E04 | Yes | Raw visual+pose | Semantic + logsig2/anchors/timing | Signature vượt matched motion control? |
| E04a | Yes | Same | Level1-only, same anchors/timing | Level2 còn thông tin vượt displacement? |
| E04b | Yes | Same | Logsig2 without anchors/timing | Gain phụ thuộc static/timing targets thế nào? |
| E05a | Yes | Raw + geometry tokens | Semantic, λ_geo=0 | Geometry input tự hữu ích không? |
| E05b | Yes | Raw + geometry tokens | Semantic + geometry | Input và auxiliary target bổ sung nhau? |
| E06 | Yes | Winning inputs | ±deep / ±visible dense, each factor separately | Objective/context layer cải thiện gì? |
| E07 | Declared changes | Winning system | Sign adaptation/paired/FT/fusion | Performance tốt nhất trong tài nguyên đã dùng |

E00 không cùng input/capacity với E02; nó là floor/reference, không isolate motion. Main signature attribution dùng E03/E04 matched controls.

### 17.2. Fairness contract

**Primary utility comparison:**

- Same train sources/availability, feature extractor weights, raw streams và windows.
- Same masked raw region schedule seeds hoặc paired mask logs.
- Same temporal encoder/predictor, geometry output32D, query count and eligibility.
- Same successful steps, effective batch, frame exposure và optimizer/head search budgets.
- Same downstream head, layer mixture, finetune protocol và metric.
- Target stats fitted cùng train pool; descriptor-dependent stats khác là unavoidable và disclosed.
- Equalization by fixed projection không làm native rank giống nhau; native sensitivity reported.
- Record preprocessing/teacher/geometry costs và params.

Không luôn match cả wall-clock và step count vì methods cost khác. Chọn **matched updates/frame exposure** cho main utility; thêm **fixed GPU-hours** comparison khi muốn efficiency claim. Không ghi “same compute” chỉ vì same student FLOPs mà bỏ teacher/detector.

**Best system:** cho phép data/backbone/head/supervision thay đổi, nhưng mỗi thay đổi có row và provenance. RGB-only và multimodal inference có input-cost comparison.

### 17.3. Focused experiments sau minimum matrix

| Factor | Pilot values [H] | Confound cần giữ |
| --- | --- | --- |
| λ_geo | 0 / 0.1 / 0.3 | Same type/group normalization |
| Mask temporal fraction | 0.3 / 0.5 / 0.7 | Effective closure counts |
| Window scales | .32 only / .16+.32+.64 | Target count/decoder compute |
| Spatial families | joint / +edge / +chain | Native dimension/capacity |
| Static cues | ±anchors, ±raw hand pose, ±face | Hình dạng/location/non-manual retention |
| Lift/time | none / discrete lead-lag / physical time | Không claim cùng invariance |
| Feature backbone | DINOv2S / B / adapted / SignRep / V-JEPA | Data/supervision/cost khác được declare |
| SSL objective | cluster / L1 latent / cosine | Same mask/head budget |
| VC | off / on | Same policy across controls |
| Layer output | last / learned mixture | Same probe capacity |
| Inference inputs | all / pose dropped / distillRGB | Missing-input training |
| Detector quality | high/medium/low buckets | Labels/supports consistent |

Không full Cartesian product. Tune coarse trên val, confirm một số candidates; giữ failed/negative runs. Motion-aware masking cùng geometry teacher có hai thay đổi; tách chúng khi attribution.

### 17.4. Decision criteria

[H] Chọn primary minimum useful effect trước chạy, ví dụ+0.5 percentage point PC Top1; giá trị này là lab decision, không standard ngành. Điều chỉnh theo estimated variance/budget trước experiment, không sau nhìn scores.

**Keep geometry:** hơn ordinary-motion controls qua seeds/CI phù hợp, giữ static phonology, không leakage, cost hợp lý, có transfer evidence.

**Revise:** movement tăng nhưng shape/location/face giảm; giảm λ, giữ visual/static inputs, sửa windows/quality/normalization hoặc đổi target role.

**Stop signature branch:** gain mất khi matched target dimensions/masks, chỉ đến từ thêm data/backbone/decoder, hoặc detector noise làm geometry thất thường. Giữ encoder/preprocessing/regularization có ích, đầu tư sign adaptation, context/data quality hoặc fusion tốt hơn.

**No claim yet:** chỉ1seed, synthetic data, test-dependent selection, novel detector/split, hoặc chưa reproduce baseline. Report đúng mức.

## 18. Tests, numerical oracle và acceptance

### 18.1. Tests bắt buộc, có ý nghĩa khoa học

| Test | Assertion | Vì sao cần |
| --- | --- | --- |
| Constant trajectories | Logsig constant paths zero; anchors khác | Static collision được xử lý có chủ đích |
| Straight vs L path | Same endpoints, level2 khác0 vs0.5 | Code có geometry vượt displacement |
| Collinear subdivision | Same logsig trong tolerance | Piecewise-linear consistency |
| Translation | Logsig unchanged; anchors translated | Không nhầm location invariance của whole model |
| Path reversal | log(signature(reverse)) = −log(signature) at degree2 | Ordered convention/algebra |
| Scaling | L1 scalesα, L2 scalesα² | Normalization/numerical sanity |
| 2D affine transform | Area scales determinant(A) | Coordinate geometry được giữ đúng |
| Missing observations | Invalid window excluded, no fake zero path | Detector failure không thành target |
| Window overlap closure | Neighboring derived tokens hidden nếu support intersects | Ngăn target alias leakage |
| Normalization supports | Mask shoulder invalidates normalized hand tokens | Preprocessing dependency thật |
| Raw resampling alias | Same source observation all hidden | VFR/low-fps resample leakage |
| Counterfactual | Visible values unchanged under hidden-value edit | Supports không bị thiếu |
| Stream/time indexing | Original flatten vs stream-temporal khác đúng expected IDs | SHuBERT adaptation rõ |
| Padding | Adding padding không đổi real outputs/metrics | Variable lengths đúng |
| Attention masks | Hidden/padded key edits không ảnh hưởng allowed query | MHA/SDPA semantics đúng |
| Teacher | No grads; same initial params; EMA đúng successful steps | Latent target implementation |
| AMP skipped update | Online/teacher/scheduler step states đồng bộ | Resume/EMA tránh lệch |
| Empty targets | Finite differentiable loss0, counts0 | Không NaN/false progress |
| Short SignCL | Empty-negative guard, actual-pair normalization | Known source edge case |
| DDP normalization | One GPU vs2ranks update close | Global mean/empty local rank |
| Resume | Tiny uninterrupted/resumed parameters close | Reproducibility |
| Metrics | Hand-check PI/PC/retrieval/edit-distance examples | Score không sai denominator |
| Cache | Roundtrip dtype/shape/hash; wrong schema rejected | Không trộn feature versions |
| Loader | Expected outputs; strict/allowlist load report | Không random partial weights |
| End-to-end | Real clip→features→loss→checkpoint→embeddings→head | Package thực sự hoạt động |

Reference checks `9.4 và `18.2 đã được chạy bằng NumPy khi kiểm tra tài liệu; điều đó chỉ xác nhận algebra/indexing examples. PyTorch model, detector inference và GPU/DDP training chưa được chạy.

Counterfactual value-change test giữ validity và mask schedule cố định; missingness changes có eligibility tests riêng. Query selection có thể expose declared availability, nhưng không hidden coordinate/anchor values.

### 18.2. Numerical verification suite không cần Torch

Đoạn sau dùng `logsig2_reference` ở `9.4; là independent analytical/metamorphic checks, không mô phỏng accuracy/BLEU.

~~~python
def run_geometry_checks():
    import numpy as np

    l_path = np.array([[0., 0.], [1., 0.], [1., 1.]])
    diagonal = np.array([[0., 0.], [1., 1.]])
    refined = np.array([[0., 0.], [.5, 0.], [1., 0.],
                        [1., .5], [1., 1.]])
    np.testing.assert_allclose(logsig2_reference(l_path), [1., 1., .5])
    np.testing.assert_allclose(logsig2_reference(diagonal), [1., 1., 0.])
    np.testing.assert_allclose(
        logsig2_reference(refined), logsig2_reference(l_path)
    )
    np.testing.assert_allclose(
        logsig2_reference(l_path + [8., -3.]), logsig2_reference(l_path)
    )
    np.testing.assert_allclose(
        logsig2_reference(l_path[::-1]), -logsig2_reference(l_path)
    )
    scaled = logsig2_reference(3. * l_path)
    np.testing.assert_allclose(scaled, [3., 3., 4.5])
    transform = np.array([[2., 1.], [0., 3.]])
    affine = logsig2_reference(l_path @ transform.T)
    np.testing.assert_allclose(affine[-1], np.linalg.det(transform) * .5)

    a = np.tile([.2, .4], (4, 1))
    b = np.tile([.7, .9], (4, 1))
    np.testing.assert_allclose(logsig2_reference(a), [0., 0., 0.])
    np.testing.assert_allclose(logsig2_reference(b), [0., 0., 0.])
    assert np.linalg.norm(a[0] - b[0]) > 0.5
    for d, expected in [(2, 3), (4, 10), (6, 21)]:
        x = np.stack([np.zeros(d), np.ones(d)])
        assert logsig2_reference(x).shape == (expected,)

    # Token supports: g0 uses raw0..2; g1 uses raw2..4; g2 uses raw5.
    supports = [{0, 1, 2}, {2, 3, 4}, {5}]
    hidden = {2}
    visible = [i for i, support in enumerate(supports)
               if not support.intersection(hidden)]
    assert visible == [2]

    # Original SHuBERT flattened span crosses channels.
    c = 4
    indices = list(range(3, 6))
    assert [(i // c, i % c) for i in indices] == [(0, 3), (1, 0), (1, 1)]
    print("Geometry/mask analytical checks passed.")
~~~

Float64 oracle tolerance1e−10; float32 batched backend initial atol1e−5/rtol1e−4, scale-aware tests và finite inputs. Set tolerance theo measured numerical behavior, không inflate tới mức bỏ qua orientation/sign errors.

Nếu dùng Signatory, compare mapped degree2 basis với oracle; higher-degree backend có Chen/reversal/subdivision property tests riêng.

### 18.3. Tiny training tests

- Synthetic features có clear time-dependent pattern + two classes, missing modality và variable lengths.
- Controlled one-batch learning: prediction loss giảm; teacher no grads; online gradients finite/nonzero.
- Force no geometry targets: semantic vẫn train; geometry0/count0.
- Force one rank no local eligible targets: collective không deadlock, global update match reference.
- Resume at step5 of10, deterministic seed/data-order; compare all states.
- Dense embeddings shape matches real length; classifier và toy CTC/SLT heads produce finite losses.
- Không dùng synthetic scores để kết luận transfer hoặc sign-language performance.

### 18.4. Test order

Geometry/schema/indexing → preprocessing counterfactual → raw SSL step → resume/padding → DDP → small real clip → pilot. Tránh train hàng nghìn steps rồi mới kiểm label/source split hoặc closure.

## 19. Compute, profiling và deployment

### 19.1. Profile trước scale

[C] Đo riêng:

- Decode/landmarks/ROI seconds mỗi video-hour.
- Visual feature extraction GPU-hours và storage.
- Data loading wait %, disk throughput/cache hit.
- Student/teacher/predictor peak memory/latency, raw/geometry token counts.
- Targets fit/projection/geometry preprocessing.
- Downstream head/decoder training.
- Inference end-to-end, gồm detector, không chỉ transformer forward.

Hai chế độ cost reporting: total pipeline cost và reused-cache incremental cost. Không chỉ báo fast geometry kernel để bỏ qua expensive detector/foundation encoder.

### 19.2. Cache estimate có công thức

Default4visual streams×384features×2bytes×25fps:

$$\text{GiB/hour}=
\frac{4\cdot384\cdot2\cdot25\cdot3600}{2^{30}}
\approx0.2575.$$

Khoảng257.5GiB/1000hours cho visual features chưa compression. Pose arrays raw body xy49×2×4bytes, quality49×4bytes, raw observed49bytes, coordinate_valid49bytes →686bytes/frame ≈0.0575GiB/hour. Timestamps/source indices thêm overhead; supports/meta và raw videos tính riêng.

Maximum geometry32Dfp16,11groups×3scales×(25/4)centers/s ≈0.0443GiB/hour nếu cache every complete token. Actual lower do boundaries/invalidity; support metadata có thể lớn hơn values nếu representation kém.

Không promise GPU wall-clock khi chưa profile. Compression/mmap/shard size chỉ tối ưu sau measurement.

### 19.3. Practical compute tiers

| Tier | Recipe | Việc cần đo |
| --- | --- | --- |
| CPU development | Synthetic small encoder + numerical/data tests | Functional correctness, không performance claim |
| 1 GPU≈24GB | Cached DINO features; D384/6layers, microbatch8 candidate | Peak VRAM có fit thật không; reduce batch/accumulate nếu cần |
| 4–8 GPUs | Sign adaptation hoặc video backbone, dense/deep targets | Teacher/online costs, total compute, tokenizer/window changes |
| Large scale | Extra unlabeled/paired corpora, larger backbones | Overlap/supervision attribution, sample efficiency |

Nếu OOM, giảm microbatch, activate checkpointing hoặc shorten **pretraining** segment dưới new config. Không tự giảm inference duration/signal quality rồi giữ same protocol. Flash/SDPA chỉ bật sau mask tests.

### 19.4. Inference modes

**Offline multimodal:**

1. Decode raw video timestamps.
2. Extract pose/ROIs và correct input normalization.
3. DINO/selected backbone inference.
4. Geometry compute nếu model role=input-enabled.
5. Unmasked temporal encoder.
6. Dense embeddings và optional clip pooling/head.
7. Report missing streams/quality and inference provenance.

**Sliding continuous inference:** default window96samples after a dedicated long-context config/FT, overlap32candidate. Center outputs hoặc weighted overlap average; không average frame representations có mismatch timestamps. Window-edge receptive fields khác full-sequence inference; evaluate boundary artifacts.

**Streaming:** bidirectional default không causal. Streaming needs causal attention, past-only geometry windows/ROI processing và training protocol mới; latency measured including lookahead. Không export offline model rồi gọi realtime causal.

**RGB-only:** distilled student hoặc separately trained modality-robust model; compare against RGB-only pretrained baseline, report teacher training cost. Geometry target-only không cần geometry inference, nhưng static-pose encoder vẫn cần detector.

### 19.5. Export

- Export unmasked online encoder + projectors + selected downstream head, không predictor/EMA teacher.
- Preprocessing/pose extractor có thể ngoài graph; artifact package phải gồm topology, coordinate/ROI config và codebook/projection nếu cần.
- Torch export/ONNX optional according backend compatibility; parity test PyTorch vs exported output trên short/long/missing-modality inputs.
- Dynamic lengths/masks và timestamp encodings được giữ đúng.
- Quantization là separate evaluation; không claim accuracy giữ nguyên khi chưa đo.
- Offline embedding NPZ gồm `dense[T,D],timestamps[T],valid[T],clip[D],model_hash,preprocessing_hash`.

## 20. Diagnostics, reproducibility và bàn giao

### 20.1. Failure table

| Triệu chứng | Cần kiểm tra đầu tiên | Hướng sửa |
| --- | --- | --- |
| Loss gần0 ngay đầu, frozen probes yếu | Collapse, teacher/student values bị shared, hidden value leakage | Verify independence/closure; normalize; VC matched comparison |
| Geometry loss luôn0 | Target count0; boundary/missing/fully-hidden policy | Log eligibility; span windows; improve detector, không bịa target |
| Geometry dominate | Target scales/projection/group averaging | Train-only standardize, lower λ, same controls |
| Moving clips tốt, holds tệ | Static path collision, shape/location loss | Retain raw hand pose/RGB/anchors; phonology probes |
| Handshape tệ | Crop truncation/tiny ROI, smoothing, few fingertips | Full hand joints/crop quality, fps/context appropriate |
| Face/non-manual tệ | Face removed/blurred or modality collapse | Retain visual face; quality buckets |
| Train tốt, unseen signer tệ | Source overlap/style background/label split | Official disjoint protocol, robust visuals/pose, held-out analysis |
| Mask requested50%, visible geometry≈0 | Overlapping supports/long windows | Target-only recipe, contiguous spans, local windows; report change |
| Strict masked baseline rất mạnh bất thường | Contextual cache or global preprocessing aliases | Raw support counterfactual; recompute online |
| NaN on short clips | Empty pair/target, dt0, all-padding attention | Guard validity; aliases; not broad nan_to_num |
| Resume scores drift | RNG/sampler/EMA/scaler/scheduler missing | Tiny replay test, exact or disclosed approximate resume |
| DDP khác single GPU | Global counts/factorW/unused branches | Deterministic distributed test |
| V-JEPA download connection refused | Localhost URL in pinned loader | Explicit official download and pretrained=False local load |
| SHuBERT loaded nhưng features sai | Generic vs adaptedDINO, bodydims, missing keys/layer indexing | Original environment/feature spec, strict load report |
| SLT BLEU khác lớn | ID alignment, tokenizer/case/decoder/data difference | Metric signature, saved refs/hyps, fixed decoder |
| Runtime faster nhưng score giảm | Inference changed sampling/streams | Compare full pipeline and registered settings |

### 20.2. Source/runtime recording

`registries/sources.yaml` ghi paper URLs, versions, code commits, local patch IDs; save environments và manifests. Không gọi arbitrary community fork là official implementation.

Mỗi scientific run lưu:

~~~text
resolved_config.yaml
environment.json + pip/conda lock
source_registry.json + local_patch.diff
checkpoint_load_report.json
manifest_hashes.json + split_audit.json
cache_and_target_provenance.json
train_log.jsonl
mask_diagnostics.jsonl
feature_spectrum.npz
checkpoints/
downstream_head_config.yaml
predictions.<split>.jsonl
metrics.<split>.json
decision_memo.md
~~~

Predictions records:

~~~json
{
  "sample_id": "example",
  "split": "val",
  "reference_label_id": 42,
  "prediction_label_id": 42,
  "top5_label_ids": [42, 5, 7, 9, 11],
  "source_group_id": "example_source",
  "signer_id": "official_id_or_null",
  "quality_bucket": "high",
  "model_hash": "COMPUTED"
}
~~~

SLT records replace labels bằng reference/hypothesis strings; classification head logits optional. Không publish signer identifying fields khi dataset terms không cho phép; evaluation grouping có thể giữ pseudonymous IDs.

### 20.3. Results table không đánh lừa

~~~text
run_id | seed | task | dataset_manifest | backbone_weights
SSL_data | SSL_supervision | downstream_supervision
input_streams | geometry_role | decoder/head | trainable/total_params
PI_top1 | PC_top1 | R@1 | phonology_metrics | WER | BLEU_signature
GPU_hours_preprocess | GPU_hours_train | inference_latency
test_used_for_selection | mask_regime | valid_counts | status
~~~

Không fill missing metrics0; dùng null + reason. Distinguish “not run”, “failed”, “invalid protocol”, “completed”. Published paper metric không ghi vào proposed model row.

### 20.4. Final Agent acceptance checklist

- [ ] Source/checkpoint assets thật, pinned, load report kiểm tra; không random fallback.
- [ ] Manifest có real availability, split/source overlap audit và train-only target fitting.
- [ ] Raw timestamps/aliases/handedness và quality semantics đúng.
- [ ] Static handshape/location tồn tại trong raw tokens; signature không thay tất cả modalities.
- [ ] Window/path/basis/native/projected dimensions match config.
- [ ] Geometry/input/normalization supports explicit; closure và counterfactual pass.
- [ ] Strict temporal và cross-modal protocols không bị lẫn.
- [ ] Teacher no-grad/EMA update theo successful steps; unused branch có gradient policy.
- [ ] Padding/missing/empty sets có handling đúng; AMP/DDP/resume tests pass.
- [ ] All CLI commands documented, implemented hoặc clearly optional unavailable backend.
- [ ] Real MVP end-to-end chạy khi có assets/data; synthetic results chỉ correctness.
- [ ] Main E03/E04 matched utility comparison; giữ methods có kết quả tốt.
- [ ] Downstream head/metrics/provenance và seeds/uncertainty cố định.
- [ ] Continuous model interfaces có functioning heads/tests; real runs ghi status đúng.
- [ ] Best-system score tách khỏi signature contribution.
- [ ] Deliver code/config/tests/memos và known limitations; không claim score chưa đo.

### 20.5. Ngắn gọn kiến trúc cho implementer

~~~mermaid
flowchart TD
    A["Raw frames, pose and timestamps"] --> B["Features and raw supports"]
    B --> C["Full valid teacher input"]
    B --> D["Raw mask and support closure"]
    D --> E["Visible student input"]
    E --> F["Encoder and query predictor"]
    C --> G["EMA semantic targets"]
    B --> H["Deterministic geometry targets"]
    F --> I["Masked semantic and geometry losses"]
    G --> I
    H --> I
~~~

Teacher input không có deterministic geometry tokens ở D0. H chỉ vào loss, không predictor query. EMA từ online encoder được định nghĩa ở `11.3; không EMA predictor. Diagram mô tả training D0, không inference.


## 21. Tài liệu gốc

Các paper [1]–[39] là nguồn của literature review và lựa chọn hypotheses. Mức đọc D=paper lõi đọc sâu, M=methods/experiments liên quan, S=screening metadata/abstract; code audit riêng ở `4. Không suy source availability hoặc checkpoint compatibility chỉ từ paper. Số liệu paper chỉ tham khảo, không làm target score tự sinh cho model mới.

- [1] Ashutosh Singh, Ashish Singh. **Path-JEPA: Path Signature Based Predictive Learning for Skeleton Action Recognition.** ECCV 2026. D. [PDF do người dùng cung cấp](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf); [Springer publication record](https://link.springer.com/chapter/10.1007/978-3-032-37258-1_7); [project](https://singh17ashu.github.io/Path-JEPA/). Đọc `3 tokenization/loss/masking, `4 tables và References; appendix không có trong bản 5207.
- [2] Ryan Wong, Necati Cihan Camgoz, Richard Bowden. **SignRep: Enhancing Self-Supervised Sign Representations.** ICCV 2025. D. [Accepted paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.pdf); [arXiv với appendix](https://arxiv.org/html/2503.08529v1). Quan trọng: `3-5 priors/objectives, Tables 1-8 và Appendix A1-A8.
- [3] Shester Gueuwou et al. **SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction.** ACL 2025. D. [Authoritative ACL PDF](https://aclanthology.org/2025.acl-long.1397.pdf); [record](https://aclanthology.org/2025.acl-long.1397/). Dùng bản ACL cho số liệu; `3-4 và Appendix A-E, gồm phonological probes và translation examples.
- [4] Hezhen Hu et al. **SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition.** ICCV 2021. M. [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.pdf); [arXiv](https://arxiv.org/abs/2110.05382). Chú ý hand prior và fusion RGB.
- [5] Hezhen Hu, Weichao Zhao, Wengang Zhou, Houqiang Li. **SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding.** TPAMI 2023. M. [Paper](https://arxiv.org/abs/2305.04868); [official project](https://signbert-zoo.github.io/). Đọc `3 MANO/embedding/multilevel masks và confidence-weighted reconstruction.
- [6] Weichao Zhao et al. **BEST: BERT Pre-training for Sign Language Recognition with Coupling Tokenization.** AAAI 2023. S. [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/25470). Trong lượt này không tải thành công PDF từ endpoint AAAI; mechanism được nêu theo abstract chính thức, không làm cơ sở cho recipe chi tiết.
- [7] Weichao Zhao et al. **Self-Supervised Representation Learning with Spatial-Temporal Consistency for Sign Language Recognition.** TIP 2024. M. [Full text](https://arxiv.org/abs/2406.10501). Đọc joint/motion, hand/trunk consistency và knowledge-transfer objective; link code trong PDF chưa được xác minh như implementation chính thức khả dụng.
- [8] Phillip Rust et al. **Towards Privacy-Aware Sign Language Translation at Scale.** ACL 2024; SSVP-SLT. M. [Paper record](https://aclanthology.org/2024.acl-long.467/); [full text](https://arxiv.org/html/2402.09611v2). Đọc MAE-only/LSP stages, datasets và translation comparisons.
- [9] Shiwei Gan et al. **Learning Effective Sign Features without Text for Gloss-free Sign Language Translation.** CVPR 2026; SignDINO. M. [Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.pdf). Đọc sign-aware local/global views, initialization, ablations và dataset-wise results.
- [10] Kunyuan Xie, Zhixi Cai, Kalin Stefanov. **SignMAE: Segmentation-Driven Self-Supervised Learning for Sign Language Recognition.** Theo arXiv metadata: accepted ICPR 2026. M. [Paper](https://arxiv.org/abs/2605.02094). Đọc `3 masking/fusion, Tables 1-4 và discussion của encoder comparison scope.
- [11] Jinhui Ye et al. **Improving Gloss-free Sign Language Translation by Reducing Representation Density.** NeurIPS 2024; SignCL. M. [Proceedings](https://proceedings.nips.cc/paper_files/paper/2024/hash/c225136cfe52a8fd66658bbcf9d894ab-Abstract-Conference.html); [paper](https://arxiv.org/abs/2405.14312). Đọc `4 sampling/loss và stage integration; đối chiếu code riêng.
- [12] Zifan Jiang et al. **SignCLIP: Connecting Text and Sign Language by Contrastive Learning.** EMNLP 2024. S. [Authoritative record](https://aclanthology.org/2024.emnlp-main.518/). Dùng để đặt paired multilingual alignment trong taxonomy; không audit implementation.
- [13] Zecheng Li et al. **Uni-Sign: Toward Unified Sign Language Understanding at Scale.** ICLR 2025. S + code audit. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html). Full PDF có trong corpus; các claim ở đây theo abstract chính thức và code fusion đã đọc.
- [14] Shengkai Sun et al. **Towards Efficient General Feature Prediction in Masked Skeleton Modeling.** 2025. M. [Full text](https://arxiv.org/html/2509.03609v1). Hierarchical targets, jointly optimized TGN và VC; speed claims chỉ áp dụng setup paper.
- [15] Mohamed Abdelfattah, Alexandre Alahi. **S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition.** ECCV 2024. M. [ECVA paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf); [project](https://sjepa.github.io/). Latent CE/centering khác cosine recipe của Path-JEPA.
- [16] Yunyao Mao et al. **Masked Motion Predictors are Strong 3D Action Representation Learners.** ICCV 2023. S. [Paper](https://arxiv.org/abs/2308.07092); [official code](https://github.com/maoyunyao/MAMP). Dùng làm motion-target control; code chưa audit trong lượt này.
- [17] Mahmoud Assran et al. **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.** CVPR 2023. M. [Paper](https://arxiv.org/abs/2301.08243). Đọc context/target blocks, overlap removal và learned-target framework.
- [18] Adrien Bardes et al. **Revisiting Feature Prediction for Learning Visual Representations from Video.** 2024; V-JEPA. M. [Paper](https://arxiv.org/abs/2404.08471). Đọc feature-vs-pixel objective, mask and probe ablations; attentive probing là biến cần kiểm soát.
- [19] Mahmoud Assran et al. **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.** 2025. S + shared official-repo context. [Paper](https://arxiv.org/abs/2506.09985). Không lấy robotics planning results làm evidence cho sign recognition.
- [20] Lorenzo Mur-Labadia et al. **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.** 2026. M + code audit. [Full text](https://arxiv.org/html/2603.14482v1). Đọc `2 dense/deep self-supervision và Table 1; nguồn cho loss/context/multi-layer choices.
- [21] Zhan Tong et al. **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.** NeurIPS 2022. S; các adaptation experiments theo [39]. [Paper](https://arxiv.org/abs/2203.12602).
- [22] Adrien Bardes, Jean Ponce, Yann LeCun. **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.** ICLR 2022. S; objective liên quan được đọc trong [2,14]. [Paper](https://arxiv.org/abs/2105.04906).
- [23] Randall Balestriero, Yann LeCun. **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.** 2025. M. [Full text](https://arxiv.org/html/2511.08544v1). Đọc SIGReg definition/recipe/limitations của distribution regularization; không suy rộng theorem thành guarantee cho ngôn ngữ ký hiệu.
- [24] Alexei Baevski et al. **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language.** ICML 2022. S. [Paper](https://arxiv.org/abs/2202.03555).
- [25] Wei-Ning Hsu et al. **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.** 2021. S. [Paper](https://arxiv.org/abs/2106.07447). Hướng chuyển sang signing được đọc sâu ở [3].
- [26] Maxime Oquab et al. **DINOv2: Learning Robust Visual Features without Supervision.** 2023. S; ứng dụng ROI theo [3]. [Paper](https://arxiv.org/abs/2304.07193).
- [27] Oriane Siméoni et al. **DINOv3.** 2025. S; sign adaptation theo [9]. [Paper](https://arxiv.org/abs/2508.10104). Dense features/Gram anchoring không phải sign-specific experimental proof.
- [28] Ilya Chevyrev, Andrey Kormilitzin. **A Primer on the Signature Method in Machine Learning.** 2016, bản cập nhật 2025. M. [Paper](https://arxiv.org/abs/1603.03788). Đọc signature definitions, time reparameterization, translation invariance, basepoint augmentation và limitations.
- [29] Patrick Kidger, Terry Lyons. **Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU.** 2020. M. [Paper](https://arxiv.org/abs/2001.00706). Đọc definitions/Chen identity/Witt dimension; không benchmark thư viện trong lượt này.
- [30] Ozge Mercanoglu Sincan et al. **Gloss-Free Sign Language Translation: An Unbiased Evaluation of Progress in the Field.** CVIU, DOI 10.1016/j.cviu.2025.104498. M. [Full text](https://arxiv.org/html/2603.13240v1); [authors' unified codebase](https://github.com/ozgemercanoglu/sltbaselines). Đọc standardized setup, metric differences và matched reproductions. Không coi reimplementation là reproduction hoàn toàn giống original training recipe.
- [31] Keren Artiaga et al. **Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation.** Findings EMNLP 2025; arXiv version xuất hiện sau. M. [Authoritative record](https://aclanthology.org/2025.findings-emnlp.997/); [PDF](https://aclanthology.org/2025.findings-emnlp.997.pdf). Dùng cho protocol signer/sentence overlap; không dùng extreme score drops như prediction cho mọi model.
- [32] Aashaka Desai et al. **ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition.** NeurIPS 2023 Datasets and Benchmarks. S. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f29cf8f8b4996a4a453ef366cf496354-Abstract-Datasets_and_Benchmarks.html); [official split description](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/).
- [33] Lee Kezar et al. **The Sem-Lex Benchmark: Modeling ASL Signs and Their Phonemes.** 2023. S. [Paper](https://arxiv.org/abs/2310.00196); [data repository](https://github.com/leekezar/SemLex). Phonological feature usage được đọc thêm ở [3,39].
- [34] Amanda Duarte et al. **How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language.** CVPR 2021. S. [Official project](https://how2sign.github.io/); [paper record](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html).
- [35] Bowen Shi et al. **Open-Domain Sign Language Translation Learned from Online Video.** EMNLP 2022; OpenASL. S. [Paper](https://aclanthology.org/2022.emnlp-main.427/); [official repository](https://github.com/chevalierNoir/OpenASL).
- [36] Dave Uthus, Garrett Tanzer, Manfred Georg. **YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus.** NeurIPS 2023 Datasets and Benchmarks. S. [Proceedings](https://papers.neurips.cc/paper_files/paper/2023/hash/5c61452daca5f0c260e683b317d13a3f-Abstract-Datasets_and_Benchmarks.html); [official dataset instructions](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md).
- [37] Garrett Tanzer, Biao Zhang. **YouTube-SL-25: A Large-Scale, Open-Domain Multilingual Sign Language Parallel Corpus.** ICLR 2025. S. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cbbb65dc108e8ac2f82cba25bc5992fc-Abstract-Conference.html).
- [38] Garrett Tanzer. **FLEURS-ASL: Including American Sign Language in Massively Multilingual Multitask Evaluation.** 2024. S. [Paper](https://arxiv.org/abs/2408.13585).
- [39] Marcelo Sandoval-Castañeda et al. **Self-Supervised Video Transformers for Isolated Sign Language Recognition.** 2023. M. [Full text](https://arxiv.org/html/2309.02450v1). Đọc methods, phonological probes và Appendix A: corrected gloss labels khác original WLASL.

### 21.1. Tài liệu kỹ thuật phục vụ implementation

- [40] **Google MediaPipe official documentation.** [Hand Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python), [Hand overview/models](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), [Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker), [Face](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker), [Holistic result fields](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HolisticLandmarkerResult). API schema, IMAGE/VIDEO modes và quality semantics; version actual phải lock.
- [41] **PyTorch official documentation.** [Transformer boolean masks](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html), [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention), [DDP design](https://docs.pytorch.org/docs/stable/notes/ddp.html), [previous version install matrix](https://pytorch.org/get-started/previous-versions/). Mask semantics/global gradient averaging; verify chosen version runtime.
- [42] **DINO official repositories.** [DINOv2](https://github.com/facebookresearch/dinov2), [model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md), [DINOv3](https://github.com/facebookresearch/dinov3). Model names/weights links; DINOv2 loader/forward/normalization audited at pinned commit `4.
- [43] **Author model distribution metadata.** [SignRep release](https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1), [SignRep release API](https://api.github.com/repos/ryanwongsa/SignRep/releases), [DINOv3 ViT-S/16 model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), [Uni-Sign model collection](https://huggingface.co/ZechengLi19/Uni-Sign). Asset names/access/supervision phải được kiểm tra trước download và scientific run.
- [44] **WLASL official repository.** [dxli94/WLASL](https://github.com/dxli94/WLASL). Data/annotations/versioning; actual availability manifest quyết định corpus đã đánh giá.
- [45] **Signatory official implementation/installation.** [repository](https://github.com/patrick-kidger/signatory), [installation source](https://github.com/patrick-kidger/signatory/blob/master/docs/pages/usage/installation.rst), [compatibility FAQ](https://github.com/patrick-kidger/signatory/blob/master/docs/pages/miscellaneous/faq.rst). Optional backend; version/ABI smoke required.
- [46] **scikit-learn MiniBatchKMeans official API.** [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html). Set random_state/n_init/batch_size explicitly and save version.
- [47] **SacreBLEU official repository.** [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu). Corpus BLEU, reproducibility signature và tokenizer settings.
- [48] **JiWER official repository.** [jitsi/jiwer](https://github.com/jitsi/jiwer). WER/CER transforms, edit counts; lock preprocessing/version.

### 21.2. Bằng chứng source và giới hạn của deliverable này

- Pretrained links/model metadata và loader definitions được đối chiếu; chưa chạy foundation-model inference hoặc proposed training.
- Degree-2 numerical checks có thể chạy độc lập bằng NumPy; full PyTorch/CUDA/DDP correctness còn phải do Agent implement và execute.
- Folder weights SHuBERT do tác giả liên kết nhưng file bytes chưa được đọc. SignDINO/SignMAE/Path-JEPA không được gán checkpoint/source release giả.
- Default ASL pipeline là lựa chọn triển khai, không xác nhận mục tiêu cuối của người dùng là ASL. Dataset/language/GPU/deployment thay đổi qua explicit config.
- Không có score mới, novelty claim, GPU timing guarantee hay published-result reproduction trong specification.

**Thứ tự thực thi ưu tiên:** M0–M2 đúng dữ liệu → M3 strong visual baseline → M4–M7 raw/geometry/control SSL → M8 downstream → M9 evidence → M10 performance scaling. Chọn mô hình tốt nhất theo kết quả, không theo độ thuần của câu chuyện novelty.
