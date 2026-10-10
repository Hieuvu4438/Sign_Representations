# SIGNREL — End-to-End Implementation Specification for an AI Agent

**Phiên bản:** 1.0  
**Ngày đối chiếu nguồn:** 06/10/2026  
**Ngôn ngữ mô tả:** tiếng Việt; tên module, API, cấu hình và log dùng tiếng Anh.  
**Trạng thái:** đặc tả triển khai một method đề xuất; chưa có model được huấn luyện hoặc kết quả benchmark mới.  
**Tên research làm việc:** Learning Cross-Articulator Temporal Relations for Self-Supervised Sign Representations.  
**Tên package nội bộ:** signrel. Đây là tên làm việc, chưa kiểm tra trùng tên package/paper.

Tài liệu này tự chứa: Agent có thể bắt đầu từ đây mà không cần đọc hội thoại trước. Tài liệu nghiên cứu đi kèm, nếu có, là Sign_Language_SSL_Research_Proposal_VI.md. Khi có khác biệt về chi tiết kỹ thuật, dùng đặc tả này cho MVP và ghi thay đổi vào docs/DECISIONS.md.

## 0. Lệnh giao việc cho Agent

> Hãy xây dựng một repository PyTorch cho SIGNREL theo đặc tả này. Thực hiện lần lượt các phase ở mục 17; đọc nguồn chính thức được chỉ định trước khi dùng code/checkpoint. Triển khai CLI, pipeline dữ liệu, encoder, objective, kiểm thử và evaluation thật. Chạy kiểm thử và smoke test trong môi trường có sẵn; chỉ chạy dữ liệu thật khi manifest, quyền sử dụng, checkpoint và ngân sách đã hợp lệ. Ghi trạng thái và bằng chứng sau mỗi phase. Nếu thiếu GPU, video hoặc annotation, tiếp tục các phần độc lập và ghi phần bị chặn. Không tạo kết quả giả, không coi synthetic test là kết quả nghiên cứu, không thay checkpoint âm thầm, không mở rộng sang cloud trả phí hoặc full-scale training khi chưa có cấu hình ngân sách cho bước đó. Kết thúc bằng code chạy được, RUNBOOK, source lock, test report, run manifest và bảng kết quả có trạng thái rõ ràng.

### 0.1 Các từ bắt buộc hiểu đúng

- **MUST:** yêu cầu nghiệm thu; vi phạm thì phase không đạt.
- **DEFAULT:** lựa chọn kỹ thuật của đề xuất này; không phải hyperparameter được tác giả paper bảo đảm.
- **OPTIONAL:** triển khai sau MVP, có config riêng; thiếu nó không chặn baseline chính.
- **INHERITED:** ý tưởng, code hoặc checkpoint có nguồn được chỉ ra.
- **PROPOSED:** thiết kế mới của project; chưa chứng minh hiệu quả hoặc novelty.
- **VERIFIED_DOC:** đã đối chiếu paper/README/source hiển thị công khai; chưa đồng nghĩa tải và chạy được.
- **VERIFIED_RUNTIME:** Agent đã tải đúng artifact, kiểm hash, load và chạy kiểm tra; chỉ Agent triển khai mới có thể gán trạng thái này.
- **BLOCKED_EXTERNAL:** cần dữ liệu, quyền truy cập, annotation hoặc tài nguyên ngoài môi trường.
- **REPRODUCTION:** chỉ dùng khi khớp đủ recipe upstream. Biến thể khác preprocessing/data/backbone phải gọi là adaptation.

Các lệnh python -m signrel trong tài liệu là **CLI mà Agent phải xây dựng**, chưa phải lệnh tồn tại sẵn trong repo của paper. Các snippet được ghi là reference code của đề xuất này, không phải code lấy từ tác giả.

## 1. Mục tiêu, giả thuyết và phạm vi MVP

### 1.1 Bài toán

Pretrain encoder bằng video sign language không dùng gloss, translation text, class ID hoặc signer ID làm training target. Encoder nhận hai stream appearance của tay và một stream pose thân trên, học từ một window bị che để dự đoán:

1. Trạng thái feature sạch tại những token bị che.
2. Quan hệ giữa tay trái ở thời điểm t và tay phải ở thời điểm u, với u có thể trước, cùng hoặc sau t.

Đánh giá chính: **frozen dictionary retrieval theo gloss với query/gallery khác signer**. Đánh giá bổ sung: natural minimal-pair discrimination nếu có annotation được kiểm chứng.

Giả thuyết H1: dưới cùng dữ liệu, initialization, backbone, masks và ngân sách, relation objective cải thiện representation so với state-only.

Giả thuyết H2: cross-stream signed-lag relations hữu ích hơn chỉ intra-stream hoặc zero-lag relations.

Giả thuyết H3: cải thiện giữ được khi kiểm soát identity, nguồn ghi hình và teacher initialization.

H1–H3 là giả thuyết cần bác bỏ được. Nếu state prediction hoàn hảo thì relation của các state đó cũng đúng; relation loss không tạo thêm thông tin target về mặt đại số. Đóng góp tiềm năng là inductive bias/optimization trong điều kiện capacity và compute hữu hạn.

### 1.2 Phạm vi cố định

| Thành phần | MVP | Mở rộng sau MVP |
| --- | --- | --- |
| Ngôn ngữ | ASL nếu dùng các nguồn dữ liệu bên dưới | Ngôn ngữ khác với split riêng; không gộp gloss khác ngôn ngữ theo English label |
| Input | LH, RH, BODY | FACE; hand landmarks chi tiết; RGB backbone trainable |
| Teacher appearance | Frozen original DINOv2 ViT-S/14, 384-D | SHuBERT hand checkpoint hoặc teacher khác, chạy tất cả controls lại |
| Context encoder | Joint Transformer nhỏ, train từ đầu | GCN/Conformer, factorized attention |
| Targets | Continuous clean features + 2-D upper-body pose | Cluster/codebook chỉ trong experiment riêng |
| Objective | State + cross-hand relation | Nonmanual relation nếu teacher đủ nhạy |
| Pretraining labels | Không dùng | Text alignment là một setting có supervision khác |
| Evaluation | Frozen retrieval; diagnostic minimal pairs | Linear probe, fingerspelling, translation |
| Scale | Synthetic → vài chục video → bounded pilot | 30–100 giờ rồi full scale có ngân sách riêng |

Không thêm EMA teacher, InfoNCE, signer adversary, VICReg regularization, text decoder hoặc learned pooling vào MVP. Mỗi thay đổi sau này cần một ablation độc lập.

### 1.3 Những giới hạn phải giữ trong báo cáo

DINOv2 là pretrained image SSL teacher, còn detector/pose estimator dùng supervision ngoài sign dataset. Vì vậy mô tả hệ thống là **sign-label-free representation pretraining với pretrained visual/perception priors**; không nói toàn bộ pipeline không có bất kỳ supervision nào.

Hand crops có thể mất vị trí so với thân. BODY giữ vị trí cổ tay, khuỷu tay, vai và mũi; đây là phần cần thiết, không phải metadata phụ.

MVP chưa xử lý đầy đủ biểu cảm mặt, mouthing, head/body grammatical cues. Không suy ra kết quả hai tay là representation đầy đủ của ngôn ngữ ký hiệu.

Cosine giữa hai DINO crop là quan hệ trong không gian thị giác của teacher, chưa phải tương quan ngữ pháp hoặc phonological unit.

## 2. Nguồn kế thừa và quy tắc attribution

### 2.1 Ma trận kế thừa

| Thành phần trong SIGNREL | Nguồn | Kế thừa cụ thể | Cách dùng trong project |
| --- | --- | --- | --- |
| Tách stream tay/body và contextual encoding | SHuBERT [S02] | Bài toán multi-stream masked learning | Adapt ý tưởng; không copy cluster target hoặc toàn bộ fairseq |
| Frozen appearance teacher | DINOv2 [S01] | Backbone/checkpoint/API trích feature | Dùng code và weights upstream có pin/hash |
| Masked latent prediction | I-JEPA [S05] | Dự đoán feature bị che từ context | Ý tưởng; không dùng checkpoint I-JEPA, không dùng EMA recipe của nó |
| Frozen target để giảm training compute | SALT [S06] | Tách target cố định khỏi student training | Inspiration; teacher của SIGNREL không phải teacher được SALT train bằng pixel reconstruction |
| Sign-specific priors và masking | SignRep [S03] | Motivation về cấu trúc ký hiệu | Related work + external baseline; không kế thừa Hiera weights cho model chính |
| Articulator-aware retrieval | SignSeek [S04] | Motivation và setting retrieval | So sánh setting supervision; không dùng GAC/MAC supervised contrastive loss trong SSL chính |
| Temporal relations/self-distillation gần đề tài | SignDino của Hu et al. [S07] | Prior art cần đối chiếu novelty | Không tự nhận đầu tiên dùng temporal/Gram relations |
| Collapse diagnostics | VICReg [S08] | Theo dõi variance/covariance | Metric tự viết; không thêm VICReg loss mặc định |
| Upper-body/hand perception | MediaPipe [S09,S10] | Models và Tasks API | Dependency có asset/version/hash |
| Query/gloss/signer evaluation | ASL Citizen [S11] | Dữ liệu/official split nếu được phát hành | Adapter + audit; custom gallery phải có protocol riêng |
| Minimal-pair diagnostic | Phonological Perception [S13] | Motivation, thiết kế kiểm tra | Adapter cho annotation thật; không giả định stimuli đã public đầy đủ |
| Encoder, relation loss, masking schedule, pooling | **PROPOSED** | Thiết kế được chốt trong tài liệu này | Tự triển khai và ghi citation nền tảng phù hợp |

### 2.2 Registry nguồn chính thức

**[S01] DINOv2 — dependency bắt buộc cho teacher mặc định**

- Paper: https://arxiv.org/abs/2304.07193
- Repo: https://github.com/facebookresearch/dinov2
- Hub factory: https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py
- URL/model naming: https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/utils.py
- Output API và ViT-S dimension: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
- Normalization constants: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
- Model identifier: dinov2_vits14; original model không register tokens; ViT-S/14, feature dim 384.
- Checkpoint URL được suy ra từ factory và base URL chính thức: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
- Trạng thái: VERIFIED_DOC cho factory, key x_norm_clstoken, dimension và normalization. Chưa tải binary, chưa có SHA-256, chưa load runtime trong lần viết tài liệu này.
- Agent phải pin commit và tải asset rồi kiểm tra trước khi gán VERIFIED_RUNTIME. Không thay bằng dinov2_vits14_reg, DINOv3, CellDINO, XRay-DINO hoặc random initialization mà giữ nguyên tên experiment.
- Repo/license và checkpoint/model card phải được ghi riêng vào third_party/NOTICE.md.

**[S02] SHuBERT — ACL 2025; nguồn multi-stream và baseline tùy chọn**

- Paper: https://aclanthology.org/2025.acl-long.1397/
- PDF: https://aclanthology.org/2025.acl-long.1397.pdf
- Project: https://shubert.pals.ttic.edu/
- Repo chính thức theo link project: https://github.com/ShesterG/SHuBERT
- Pipeline: https://github.com/ShesterG/SHuBERT/blob/main/QUICKSTART.md
- Feature guide: https://github.com/ShesterG/SHuBERT/blob/main/FEATURES.md
- Dataset guide: https://github.com/ShesterG/SHuBERT/blob/main/DATASETS.md
- Checkpoint folder theo README: https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV?usp=sharing
- Các entry point được QUICKSTART chỉ ra: dataset/clips_crop.sh, dataset/kpe_mediapipe.sh, dataset/crop_hands.sh, dataset/crop_face.sh, features/body_features.sh, features/hands_features.sh, features/face_features.sh, features/shubert_inference.sh.
- Environment upstream tách riêng: environment_feature_extraction.yml, environment_dino.yml, environment_shubert.yml; mô hình dùng fork fairseq trong repo.
- Đã xác minh hướng dẫn dùng **DINO đã fine-tune cho face/hand**. Original DINOv2 của SIGNREL không được gọi là feature extractor giống hệt SHuBERT.
- Chưa xác minh tên file/ID từng checkpoint trong Drive hoặc khả năng tải hiện tại. Agent phải lấy file list, config, SHA-256 và load test; không đặt tên asset tưởng tượng.
- Không cần chạy SLURM scripts nguyên trạng. Nếu xây adapter local, giữ đúng feature convention và ghi patch. Không coi downstream TODO trong repo là phần đã tái lập xong.

**[S03] SignRep — ICCV 2025; baseline tùy chọn**

- Bản user cung cấp: https://arxiv.org/html/2503.08529v1
- Bản conference: https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html
- Repo: https://github.com/ryanwongsa/SignRep
- Inference example: https://github.com/ryanwongsa/SignRep/blob/main/example_usage.py
- Release: https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1
- Entry class được example chỉ ra: models.final_models.FINAL_hiera_latent_model_head_v25_active.Model.
- Transform: augmentation.video.base_video_aug.Transformation.
- Example load payload key model và output features / latent; input là B×3×16×224×224.
- Release tag v0.0.1 có thông báo checkpoint; chưa xác minh tên binary asset và hash. Agent dùng release API để lấy asset thật, không suy đoán download URL.
- Release ghi CC BY-NC-SA 4.0 và điều kiện sử dụng checkpoint. Ghi điều khoản cụ thể theo phiên bản tải; không mặc định checkpoint mang license như code.
- Baseline pretrained dùng data/init khác chỉ là external comparison, không phải ablation nhân quả của relation loss.

**[S04] SignSeek — preprint tháng 9/2026; khác SHuBERT**

- https://arxiv.org/abs/2609.03695
- https://arxiv.org/html/2609.03695v1
- Bài user đưa ở link thứ hai là SignSeek. GAC/MAC sử dụng gloss supervision; không đưa vào setting “không dùng gloss” bằng cách đổi tên loss.
- Chưa xác minh được official public repository/checkpoint trong lần rà soát này.
- Chỉ đọc method/evaluation để đối chiếu. Nếu muốn adapter, cần nguồn code chính thức hoặc đánh dấu own reimplementation, kèm recipe và test riêng.

**[S05] I-JEPA — CVPR 2023; inheritance mức ý tưởng**

- Paper: https://arxiv.org/abs/2301.08243
- Official repo: https://github.com/facebookresearch/ijepa
- Nên đọc src/train.py, src/helper.py, src/masks/ và src/models/ từ commit đã pin.
- Repo đã archive; hữu ích để hiểu predictor, stop-gradient và masking, không là dependency của MVP.
- Không gọi SIGNREL là exact I-JEPA reproduction: target, tokenization, teacher update và kiến trúc đều khác.

**[S06] SALT — Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers**

- Version dùng để đối chiếu method: https://arxiv.org/html/2509.24317v1
- Proceedings ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/e6d1d6195f6f3e32a930643e0ef46332-Abstract-Conference.html
- Kế thừa motivation frozen teacher; SIGNREL dùng DINO image features, không tái tạo hai-stage recipe SALT.
- Chưa xác minh repo chính thức và checkpoint trong lần rà soát. Không dùng một URL repo suy đoán làm dependency.

**[S07] SignDino của Hu et al. — prior art tháng 9/2026**

- https://arxiv.org/abs/2609.06296
- https://arxiv.org/pdf/2609.06296
- Project được paper chỉ ra: https://junyi2005.github.io/signdino/
- Paper có temporal-axis self-distillation và temporal Gram matching; phải so sánh với cross-articulator relations của đề xuất.
- Project không truy cập được bằng công cụ rà soát này; chưa xác minh repo/checkpoint. Đặt OPTIONAL / UNVERIFIED_ASSET.
- Có công trình khác tên gần giống SignDINO tại CVPR 2026; không gộp citation hay checkpoint của hai paper.

**[S08] VICReg — ICLR 2022**

- https://arxiv.org/abs/2105.04906
- https://github.com/facebookresearch/vicreg
- File tham khảo: main_vicreg.py cho variance/covariance computations.
- Repo đã archive. MVP chỉ dùng diagnostics, không lấy loss/checkpoint của VICReg.

**[S09] MediaPipe Hand Landmarker**

- Python guide: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
- Model guide: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- Tasks API: HandLandmarker, HandLandmarkerOptions, RunningMode.VIDEO, detect_for_video(image, timestamp_ms).
- Output có 21 landmarks/hand và handedness; handedness score không phải xác suất landmark chính xác.
- Agent lấy model asset từ model guide tại thời điểm triển khai, ghi URL/file/hash. Tài liệu này không khóa một URL .task chưa load-test.

**[S10] MediaPipe Pose Landmarker**

- https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- Tasks API: PoseLandmarker và VIDEO mode; output normalized landmarks có visibility/presence.
- Agent chọn model variant rồi khóa asset/hash. MVP dùng x,y; không gọi normalized z là metric depth.

**[S11] ASL Citizen — downstream dataset chính**

- Project/dataset: https://www.microsoft.com/en-us/research/project/asl-citizen/
- Code: https://github.com/microsoft/ASL-citizen-code
- Baseline weights link từ README: https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1
- Repo đã archive. Các thư mục I3D, ST-GCN có README riêng.
- Data license, schema metadata và splits phải đọc từ bản dataset tải thật. Code MIT không đồng nghĩa video có cùng license.
- Không cần I3D/ST-GCN checkpoint để chạy SIGNREL retrieval; chúng là baseline có supervision riêng.

**[S12] YouTube-ASL — pretraining corpus tùy theo quyền truy cập**

- Paper: https://arxiv.org/abs/2306.15162
- Proceedings: https://proceedings.neurips.cc/paper_files/paper/2023/hash/5c61452daca5f0c260e683b317d13a3f-Abstract-Datasets_and_Benchmarks.html
- Official README: https://github.com/google-research/google-research/blob/master/youtube_asl/README.md
- Official folder: https://github.com/google-research/google-research/tree/master/youtube_asl
- README phát hành video IDs, không bảo đảm mọi video luôn tải được.
- Có captions trong corpus nhưng SSL adapter phải bỏ chúng khỏi input/target. Không lấy caption boundaries làm segment supervision mặc định.
- Ưu tiên video đã có hợp lệ trong máy. Không tự clone toàn bộ google-research hoặc tải toàn corpus trước khi biết budget.

**[S13] Phonological Perception of Sign Language Models**

- Paper: https://arxiv.org/html/2606.28667v1
- Code: https://github.com/kayoyin/sign-phonology
- Đọc README, analysis/ và scripts/ trước khi xây adapter.
- README hiện ghi curated stimuli minimal_pairs_asl_citizen.csv và minimal_pairs_semlex.csv còn TODO về URL phát hành. Không giả định các CSV có trong clone.
- Code phân tích và annotation là hai dependency khác nhau. Thiếu curated CSV thì diagnostic này BLOCKED_EXTERNAL; retrieval vẫn có thể hoàn thành.
- Không dùng từ “minimal pair” cho hai English gloss khác nhau nếu chưa có annotation phonological phù hợp.

**[S14] Prior art về spatial-temporal consistency**

- https://arxiv.org/html/2406.10501v1
- https://doi.org/10.1109/TIP.2024.3416881
- Cần đọc trước khi viết novelty claim về phối hợp hand/body; không dùng code hoặc checkpoint chưa xác minh.

**[S15] V-JEPA 2 / 2.1 — tham khảo mở rộng**

- https://github.com/facebookresearch/vjepa2
- https://arxiv.org/abs/2603.14482
- Dùng để đối chiếu dense video SSL; không tải backbone lớn hoặc copy launch recipe vào MVP.

**[S16] PyTorch — nguồn kỹ thuật của Transformer**

- Source: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
- Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- Đọc docs/source đúng release đã cài. Kiểm tra batch_first, norm_first, key padding mask và initialization của các cloned layers.
- Không dùng main branch làm căn cứ version compatibility của môi trường đã lock.

### 2.3 Phải tạo nguồn khóa trước real extraction

Agent tạo artifacts/sources.lock.json. Ví dụ schema dưới đây là **template chưa resolved**, không được dùng trực tiếp cho publication run:

~~~json
{
  "schema_version": 1,
  "verified_at_utc": null,
  "teacher": {
    "paper_id": "2304.07193",
    "repo_url": "https://github.com/facebookresearch/dinov2",
    "commit_sha": null,
    "entrypoint": "dinov2_vits14",
    "checkpoint_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "checkpoint_sha256": null,
    "feature_key": "x_norm_clstoken",
    "feature_dim": 384,
    "runtime_verified": false
  },
  "perception": {
    "mediapipe_version": null,
    "hand_asset_url": null,
    "hand_asset_sha256": null,
    "pose_asset_url": null,
    "pose_asset_sha256": null
  },
  "optional_baselines": [],
  "licenses": [],
  "unresolved": ["commit_sha", "checkpoint_sha256", "perception assets"]
}
~~~

Rules:

1. Lấy full commit SHA thực tế; không tự viết SHA giả từ ngày hoặc tag.
2. Hash binary sau khi tải; không gọi filename hash là SHA-256.
3. Lưu paper version, repo commit, checkpoint URL, original filename, downloaded bytes, SHA-256, config, license URL và load-test result.
4. Lock preprocessing cùng nguồn weights; checkpoint đúng nhưng preprocessing sai vẫn là experiment khác.
5. final-run command MUST fail nếu dependency bắt buộc chưa resolved. Optional baseline bị chặn không làm train A1/A4 thất bại.
6. Nếu patch upstream, lưu patch dưới third_party/patches/ và hash bản source thực tế sau patch.

## 3. Research protocol phải đăng ký trước

Agent tạo artifacts/protocol.json trước khi xem locked-test metrics, gồm:

- Primary endpoint: macro gloss Recall@1 của frozen retrieval.
- Secondary: micro R@1/R@5, macro R@5, MRR, coverage.
- Primary contrast: A4 − A1, cùng seed, cùng cache, cùng eval protocol.
- Diagnostic contrast: A4 − A3; A4 − A2; shuffled relation control.
- Checkpoint rule: last completed fixed-budget checkpoint; không chọn theo composite training loss khác nhau giữa objective.
- Validation dùng riêng để debug/tune; locked test chỉ chạy sau khi configs đã khóa.
- Pilot là exploratory; confirmatory run có 3 seeds và config freeze.
- Không lấy best seed; báo từng seed và mean ± sample standard deviation.
- Nếu lấy labels từ dev retrieval để chọn lambda/mask, khai rõ labeled model selection. Nó không biến pretraining target thành gloss supervision nhưng là supervision trong selection.
- Nếu pretraining continuous corpus không có lexical annotations, không claim strict unseen-gloss pretraining. Claim unseen video/source group hoặc signer chỉ khi audit xác nhận được.
- Corpus, signer overlap unknown phải được ghi unknown, không tự suy luận từ channel ID.

**Novelty gate trước viết paper:** đọc [S02,S03,S04,S07,S14] và cập nhật literature mới. Tạo bảng so sánh target type, stream fusion stage, temporal direction, supervision, evaluation và compute. Không claim “first multi-stream SSL”, “first JEPA for sign” hoặc “first temporal relation loss” từ việc pipeline chạy được.

## 4. Môi trường và dependency strategy

### 4.1 Hai môi trường tách biệt

| Environment | Nhiệm vụ | Dependency chính |
| --- | --- | --- |
| signrel-core | Đọc cache, train, inference, retrieval, tests | PyTorch, NumPy, PyYAML, pytest; CLI dùng argparse |
| signrel-perception | Decode video, MediaPipe, frozen DINO extraction | MediaPipe, PyAV/FFmpeg, Pillow, PyTorch; torchvision chỉ nếu thực sự cần |

DEFAULT dùng Python 3.11 cho môi trường mới vì nhiều perception wheels hỗ trợ; đây là lựa chọn của project, không phải version đã được kiểm chứng trên mọi hệ điều hành. Nếu môi trường hiện hữu hoạt động tốt, giữ nó và lock thay vì thay toàn bộ.

Agent MUST thực hiện:

1. Kiểm tra Python, OS, GPU model/VRAM, driver, CUDA runtime, disk và FFmpeg.
2. Chọn bản PyTorch phù hợp với GPU/driver từ nguồn cài đặt chính thức. Không ghi một CUDA wheel ngẫu nhiên vào lock.
3. Cài minimum dependencies, chạy import test và tensor operation CPU/GPU.
4. Giải quyết compatibility MediaPipe/NumPy/protobuf trong environment perception.
5. Xuất requirements-core.lock.txt và requirements-perception.lock.txt với exact versions; ghi wheel/source index và container digest nếu dùng container.
6. Optional SHuBERT/SignRep baseline chạy environment riêng từ upstream; không kéo fairseq hoặc dependency legacy vào core nếu không cần.

Không coi file lock là hợp lệ chỉ vì pip freeze thành công: còn cần decoder, perception, teacher forward và core train smoke test.

### 4.2 Pin DINO source và load weights

Clone repo [S01] vào third_party/dinov2, resolve commit, checkout full SHA, ghi vào source lock. Real experiments load source local, không dùng floating torch.hub remote mỗi lần.

Reference code tự viết cho teacher loader; Agent thêm hash check, path resolution và error handling:

~~~python
from pathlib import Path
import torch

def load_frozen_teacher(repo_dir: str, checkpoint_path: str):
    checkpoint_path = str(Path(checkpoint_path).resolve())
    model = torch.hub.load(
        repo_dir,
        "dinov2_vits14",
        source="local",
        pretrained=False,
    )
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model
~~~

DINO checkpoint mặc định được factory load như state_dict trực tiếp. Nếu bản tải thật có wrapper khác, loader phải báo cấu trúc và đối chiếu upstream; không bỏ strict hoặc loại key hàng loạt để “load được”.

Test teacher trên 2 ảnh/crops hợp lệ: output shape [2,384], finite, không cần grad, lặp cùng input cho output ổn định trong tolerance của dtype/device. Không yêu cầu hash bitwise của feature GPU giữa mọi GPU.

Nếu dependency optional như xFormers không có, chỉ dùng fallback được source pin hỗ trợ; log backend và benchmark nó. Không sửa kiến trúc để né dependency mà giữ nguyên tên teacher.

## 5. Hợp đồng dữ liệu và split

### 5.1 Nguồn dữ liệu theo thứ tự

1. Video local do người dùng/project có quyền nghiên cứu.
2. Subset YouTube-ASL [S12] nếu IDs/video truy cập được và phù hợp quyền sử dụng.
3. ASL Citizen training split [S11] làm **prototype SSL corpus riêng**, bỏ labels khỏi train loader; dùng tên setting ssl_aslcitizen_train. Không gọi setting này là replication 1,000 giờ của SHuBERT.

Agent không được dừng toàn bộ implementation chỉ vì YouTube download bị chặn. Có thể hoàn thành core + synthetic tests + local-data adapter; report phần real-data còn thiếu.

### 5.2 SSL manifest: một dòng JSON/video hoặc recording segment

~~~json
{
  "schema_version": 1,
  "video_id": "local_000001",
  "relative_path": "videos/local_000001.mp4",
  "source_dataset": "authorized_local",
  "source_recording_id": "recording_00001",
  "source_group_id": "recording_00001",
  "language": "ASL",
  "start_ms": 0,
  "end_ms": 12000,
  "signer_group_id": null,
  "signer_group_status": "unknown",
  "sha256": null,
  "license_record_id": "local_permission_record",
  "split": "ssl_train"
}
~~~

- relative_path được resolve dưới data root; không dùng absolute path trong manifest portable.
- video_id duy nhất, ổn định; source_recording_id dùng để nhóm các cut cùng recording.
- signer_group_id chỉ phục vụ split/audit/sampler group nếu được xác minh, không đưa vào model input hoặc loss.
- Không có gloss, translation, English caption, phonological label hay class index trong SSL manifest.
- Known signer ID không được tạo pseudo target, contrastive positives hoặc adversarial supervision trong MVP.
- SHA-256 của video bắt buộc trước extraction hoàn tất; null ở ví dụ chỉ là trạng thái import ban đầu.
- video duration và actual decoded metadata nằm trong audit result, không lấy FPS nominal làm sự thật tuyệt đối.
- Caption-derived segments chỉ được dùng trong một setting riêng có khai báo segmentation prior. Default là random windows trong recording.

### 5.3 Evaluation manifest tách riêng

~~~json
{
  "schema_version": 1,
  "video_id": "eval_000001",
  "relative_path": "videos/eval_000001.mp4",
  "source_dataset": "ASL_Citizen",
  "source_recording_id": "eval_recording_000001",
  "signer_id": "verified_signer_001",
  "language": "ASL",
  "gloss_id": "dataset_native_gloss_id",
  "official_split": "test",
  "eval_role": "query",
  "annotation_version": "resolved_dataset_version",
  "sha256": null
}
~~~

Tên signer/gloss trong ví dụ là placeholder schema, không phải ID từ ASL Citizen. Adapter phải đọc metadata thật, lưu mapping raw column → canonical field trong artifacts/data_adapter.json.

SSL Dataset class MUST chỉ đọc SSL manifest + numeric cache. Evaluation labels dùng sau export embeddings hoặc cho probe riêng. Cấm nhét labels vào .npz cùng teacher features.

### 5.4 Split algorithm

1. Giữ official downstream train/dev/test nếu bản dataset có split rõ ràng.
2. Tạo SSL train/dev bằng group split ở source_group_id; default hash-based 95/5 với split_seed=2026.
3. Các cut cùng recording bắt buộc cùng group. Không split random frame/window.
4. Nếu có signer mapping đáng tin cậy, dùng signer group split cho SSL dev; nếu không, gọi là source-group-disjoint, không gọi signer-disjoint.
5. Loại exact duplicate video hash và duplicated segment ranges xuyên train/query/test.
6. Loại recording/group của query dev và locked test khỏi SSL train. Audit source IDs, temporal intervals và hashes.
7. Signer-disjoint claim cần audit identity xuyên SSL và evaluation; unknown identity không đạt claim này, dù downstream official splits khác signer.
8. Nếu pretrain trên ASL Citizen train, query dev/test phải không nằm trong corpus. Gallery có thể lấy từ train: đây là seen-gallery setting và phải khai rõ.
9. Strict unseen-gloss setting cần lexical annotation phục vụ split exclusion; không giả định corpus continuous đã đạt điều kiện đó.

Fingerprint split = SHA-256 của canonical sorted manifest và config split. Mọi run/eval phải lưu fingerprint này.

### 5.5 Dedup và lỗi dữ liệu

- File hash phát hiện exact duplicates; near-duplicate clip/time overlap dùng source metadata và frame fingerprints trên audit subset.
- Không tự gộp mọi video “trông giống” thành cùng signer.
- Failed download/decode phải ghi error row, stage, retry count; không biến failure thành valid sample.
- Retry tối đa 2 lần với backoff ngắn. Sau đó đánh dấu missing và tiếp tục phần độc lập.
- Không tự dịch/merge gloss giữa ASL và ngôn ngữ khác. Namespace label = language + dataset label schema.
- Báo planned videos/hours, available, successfully decoded, kept, excluded và lý do; denominator evaluation phải rõ.

## 6. Preprocessing từ video đến cache

### 6.1 Timestamp và decode

DEFAULT decode bằng PyAV/FFmpeg, lấy presentation timestamp (PTS) và time base. Lưu actual time_ms và original frame index.

Tạo canonical grid 25 Hz trong đoạn video. Mỗi grid time chọn decoded frame gần nhất, tolerance tối đa 40 ms. Lưu cả grid_time_ms và actual_time_ms; nếu không có frame trong tolerance thì frame_valid=false.

Không dùng frame_number / nominal_fps cho video variable frame rate mà không kiểm tra. Không đọc toàn video hàng giờ vào RAM.

Nếu nhiều grid vị trí ánh xạ cùng decoded frame, lưu source_frame_index để loại duplicate relation constraints khi cần. Với video thấp FPS, vẫn ghi coverage và unique-frame ratio; không coi các bản lặp là quan sát độc lập.

MediaPipe VIDEO mode phải nhận timestamp tăng nghiêm ngặt. Khi canonical grid lặp một source frame, chạy perception một lần tại actual timestamp rồi reuse output. Nếu rounding ms gây trùng, xử lý unique PTS và ghi policy; không dùng LIVE_STREAM mode có thể drop frame.

### 6.2 Chọn signer

MVP dùng single-signer crop/full frame có bounding box rõ. Có thể reuse cách tổ chức crop trong [S02], nhưng detector weights phải được khóa riêng nếu thêm YOLO.

- Nếu frame có nhiều người, giữ signer track đã xác minh bằng source metadata/preview; không chọn arbitrary person mỗi frame.
- Không có signer track đáng tin cậy thì excluded_reason=ambiguous_signer.
- Detector không bắt buộc với dataset đã crop single signer.
- Lưu crop transform để đưa landmarks/crop boxes về tọa độ frame gốc nhất quán.

### 6.3 Hand assignment và crop

LH/RH nghĩa là **tay giải phẫu của signer**, không phải bên trái/phải trên màn hình.

1. Dùng hand landmarks + pose wrists + temporal track association.
2. Kiểm tra handedness convention của version/model đang dùng; nếu video mirrored, ghi mirrored_status và association rule.
3. Không tự horizontal flip video để “chuẩn hóa tay thuận”.
4. Nếu label detector mâu thuẫn với wrist association hoặc crossing không rõ, đánh dấu ambiguous; không swap stream âm thầm.
5. Preview tối thiểu 20 clips có crossing, signing near face, occlusion và tay ra khỏi frame.

Crop policy của project:

- Bounding square phủ 21 hand landmarks, center theo box; side = 1.5 × max(width,height).
- Side tối thiểu 24 pixel; crop quá nhỏ/không đủ landmarks → invalid.
- Padding bằng RGB mean color tương ứng normalization, không ép crop kéo giãn méo hình.
- Resize square đến 224×224 bicubic; padding boundary policy được ghi vào cache fingerprint.
- teacher input RGB float [0,1], channel-first; mean [0.485,0.456,0.406], std [0.229,0.224,0.225] theo [S01].
- Không random resized crop, blur hoặc flip clean teacher crops.
- Tỷ lệ crop clipped tại border được log. Default clipped fraction >0.5 thì invalid; threshold là lựa chọn project cần kiểm bằng preview.

Crops teacher không giữ tuyệt đối vị trí tay; vị trí được giữ trong BODY.

### 6.4 Confidence/validity: định nghĩa thực tế

MVP dùng **quality proxy**, không giả định detector có per-landmark calibrated confidence.

- BODY point proxy: min(visibility,presence) của pose landmark, nếu API version có đủ trường; nếu thiếu presence, dùng visibility và log policy.
- Hand proxy: quality của pose wrist tương ứng × crop_in_bounds_fraction × association_valid.
- Hand output handedness score dùng cho association diagnostic, không gọi là hand detection confidence.
- Nếu pose wrist thiếu nhưng hand crop rõ: DEFAULT không dùng sample hand đó làm target; OPTIONAL binary-quality fallback phải là config riêng và sensitivity analysis.
- Quality threshold default 0.5. Đây là hyperparameter project.
- Validity không được tạo từ gloss/label.
- Target weight lấy từ **clean** sample; không lấy từ prediction hoặc corrupted quality.
- Teacher features invalid lưu zero + valid=false; không lưu NaN.
- Interpolated detector outputs không tự thành clean observed targets. MVP không interpolate thiếu appearance features.

Báo coverage theo signer/source, tốc độ chuyển động và occlusion khi metadata cho phép. Quality threshold có thể loại ký hiệu khó; báo both attempted-set coverage và metrics trên processed set.

### 6.5 Upper-body pose 14-D

Dùng 7 landmark MediaPipe Pose: nose=0, shoulders=11,12, elbows=13,14, wrists=15,16. Agent đối chiếu index map đúng model guide [S10].

Từ normalized x,y, đổi về pixel x×W,y×H trước khi lấy khoảng cách để tránh distortion do aspect ratio.

Center = midpoint hai shoulders. Scale = khoảng cách shoulder pixels. Cần cả hai shoulders valid và scale ≥20 pixel. Không đạt thì BODY frame invalid.

Với landmark j:

\[
b_{t,j} = ((x_{t,j}-c_{t,x})/s_t,\;(y_{t,j}-c_{t,y})/s_t).
\]

- Preserve signed left/right coordinates; không rotate thân về một orientation chung.
- Không clip tọa độ về [-1,1]: tay xa thân có thể mang nội dung.
- Nếu point invalid, zero riêng coordinate đó và coord_valid=false.
- BODY token valid khi center/scale hợp lệ và tối thiểu 4/7 points đạt quality threshold.
- Chỉ tính body target loss trên coordinates valid.
- Quality BODY per frame = median proxy của các valid points; lưu point-level proxy cho body loss.
- 14-D body không nằm cùng metric space với DINO 384-D; không cosine body vector với hand appearance.

### 6.6 Teacher extraction

teacher.eval(), requires_grad_(False), torch.inference_mode(). Lấy forward_features(input)["x_norm_clstoken"], float32 rồi L2 normalize từng feature trước khi lưu float16.

- Feature norm sau load/renormalize phải gần 1 trên valid observations.
- Cache chứa normalized original teacher vectors; không PCA/whiten trong MVP.
- LH/RH dùng **cùng một checkpoint và preprocessing**, nên có thể dot product trong cùng feature space.
- Nếu mở rộng dùng hand teacher và face teacher khác nhau, không cosine trực tiếp hai spaces; phải đổi relation design và ablation.
- Extract crop microbatch theo GPU profile; không thay source FPS khi gặp OOM.
- Teacher không được update qua student train.
- Deterministic clean cache được reuse nguyên vẹn cho tất cả controlled baselines.

### 6.7 Cache schema

Mỗi recording segment có một .npz không object pickle, kèm .json metadata:

| Key | Shape | Dtype | Ý nghĩa |
| --- | --- | --- | --- |
| appearance | [T,2,384] | float16 | unit clean features, thứ tự LH,RH |
| app_valid | [T,2] | bool | quan sát hợp lệ |
| app_quality | [T,2] | float32 | clean proxy trong [0,1] |
| body | [T,14] | float32 | normalized 2-D coordinates |
| body_coord_valid | [T,14] | bool | validity mỗi coordinate |
| body_coord_quality | [T,14] | float32 | point proxy lặp cho x,y |
| body_valid | [T] | bool | BODY token validity |
| body_quality | [T] | float32 | BODY proxy |
| frame_valid | [T] | bool | timestamp/decode validity |
| grid_time_ms | [T] | float64 | canonical grid |
| actual_time_ms | [T] | float64 | actual decoded timestamps |
| source_frame_index | [T] | int64 | correspondence về decoded frame |
| crop_boxes | [T,2,4] | float32 | pixel coordinates, audit only |

JSON metadata gồm video_id, video_sha256, cache_schema_version, teacher_sha256, source_commit, perception hashes, preprocessing config hash, split fingerprint và coverage. Không có labels.

Cache loader dùng allow_pickle=False, kiểm shapes/dtypes/finiteness/feature dimensions. Chỉ normalize valid features. Missing data không được “fix” bằng feature trung bình rồi set valid=true.

Atomic write: .tmp → validate → rename. Cache key gồm video hash + preprocessing/teacher/perception hashes; đổi một thành phần thì cache mới. Chạy lại phải skip artifact đúng hash, recompute artifact hỏng, không xóa cache cũ.

Ước lượng riêng cho 100 giờ, 25 fps, **2** hand streams, 384-D float16: khoảng 13.8 GB appearance chưa kể pose, metadata, crops và overhead. Với thêm FACE thành 3 appearance streams: khoảng 20.7 GB. Đây là phép tính storage, không phải benchmark thời gian extraction.

## 7. Window sampling, corruption và tensor contract

### 7.1 Train window

DEFAULT window length T=64 tại grid 25 Hz, khoảng 2.56 giây; không giả định window là một sign hoàn chỉnh.

- Random start trong SSL recording, không dùng gloss/caption boundary.
- Cần tối thiểu 32 canonical frames và đủ observed hand/body tokens.
- Video 32–63 frames: pad đến 64, padded tokens không có targets.
- Video <32 frames: skip MVP và ghi coverage; không lặp frame để biến clip rất ngắn thành sample độc lập.
- Sampling cân bằng theo recording trước, rồi chọn start trong recording; tránh video dài chi phối hoàn toàn. Không cân bằng theo gloss.
- Có thể group signer cho split, nhưng sampler mặc định không cần signer labels.
- Sample được reject nếu không có eligible hand state target hoặc không còn ≥4 visible valid tokens sau corruption. Resample tối đa 10 lần; hết quota thì skip có log.
- A1 và A4 phải nhận cùng accepted windows và masks. Đừng reject A4 riêng vì relation count thấp rồi làm dataset hai bên khác nhau: dùng common eligibility hoặc loss relation=0 trên window đó.

### 7.2 Ba mask khác nhau

| Mask | Ý nghĩa | Tính loss? | Là attention padding? |
| --- | --- | --- | --- |
| natural_valid | Quan sát sạch có thật và đủ quality | Có thể, khi bị SSL mask | Token natural-missing có thể bị ignore làm key |
| ssl_mask | Agent cố ý che một quan sát hợp lệ | Có | **Không**; giữ learned mask token làm query/key |
| time_padding | Ngoài video/window | Không | Có |

Dùng bool; True/False semantics phải có unit test. Không dùng chung biến mask cho cả ba.

Appearance SSL mask áp dụng lên toàn bộ feature vector và mọi field appearance input có thể tiết lộ feature đó. Clean proxy chỉ dùng cho weighting, không feed vào hidden token dưới dạng confidence “sạch”.

BODY SSL mask che toàn bộ 14-D token, coord-valid pattern có thể giữ như observation pattern nếu policy được lock; DEFAULT replace full input bằng learned body mask token để đơn giản. Không che BODY làm label mất validity của target.

Natural-missing token dùng learned missing embedding; không tính state/relations trên nó. Key padding mask = time_padding OR natural_invalid. Reject sample có tất cả keys padding.

### 7.3 Mask schedule của project

DEFAULT trên observed targets:

- Hand mask ratio 0.5 cho mỗi hand.
- BODY mask ratio 0.25.
- 50% samples dùng shared temporal spans: block thời gian che cả hai tay; BODY che một subset cùng thời gian.
- 50% dùng independent stream spans để có trường hợp dùng tay kia làm context.
- Span length ngẫu nhiên 4–12 canonical positions. Thêm/bớt positions cuối để đạt ratio trong ±1 token.
- Giữ ít nhất 4 visible valid tokens tổng cộng; giữ target count trên mỗi hand nếu có đủ observations.
- Không dùng random contiguous crop của hai đoạn khác nhau làm semantic positive.
- Whole-hand dropout là OPTIONAL ablation, không trộn vào config chính chưa kiểm chứng.
- Tất cả mask ratios/span choices là PROPOSED defaults; không được ghi là recipe của SHuBERT hay SignRep.

### 7.4 Correspondence và optional time resampling

MVP không đổi tốc độ: g(t)=canonical index của window. Vẫn triển khai explicit mapping để tránh lỗi khi mở rộng.

Nếu thêm global speed resampling:

1. Cùng một monotonic map g cho LH, RH, BODY; không shift riêng một tay.
2. Gather clean target, validity và quality theo g.
3. Student position encoding dùng actual_time_ms[g(t)] − actual_time_ms[g(0)].
4. Relation target tại student pair (t,u) là q_clean[g(t)]·q_clean[g(u)].
5. Không dùng g(t)+delta để thay g(u).
6. Nếu hai view positions trỏ cùng source frame, không coi pair đó là lagged observation mới.
7. Không gọi thay đổi tốc độ là bất biến ngữ nghĩa phổ quát: tốc độ/hold có thể mang chức năng ngôn ngữ.
8. Rate set, duplicate handling, physical-lag distribution phải có experiment ID riêng.

Default disabled: horizontal flip, temporal reverse, hand-specific shifts, erasing holds, facial blur, random signer swaps. Chúng không phải augmentation mặc nhiên bảo toàn ký hiệu.

### 7.5 Batch interface

~~~text
clean.appearance       Float[B,T,2,384]    # gathered, renormalized, detached
clean.app_valid        Bool[B,T,2]
clean.app_quality      Float[B,T,2]
clean.body             Float[B,T,14]
clean.body_coord_valid Bool[B,T,14]
clean.body_coord_quality Float[B,T,14]
clean.body_valid       Bool[B,T]
clean.body_quality     Float[B,T]
student.app_ssl_mask   Bool[B,T,2]
student.body_ssl_mask  Bool[B,T]
student.time_padding   Bool[B,T]
student.relative_time_ms Float[B,T]
mapping.g              Long[B,T]          # cache indices, audit/target side only
mapping.source_frame_index Long[B,T]
~~~

Video ID/source ID nằm trong metadata để trace; không embed vào model. Bounding boxes, source frame indices và clean targets không được encoder đọc qua hidden side channel.

## 8. Kiến trúc encoder chính

### 8.1 Projection và tokenization

Streams theo thứ tự [LH,RH,BODY], S=3. Hidden dimension d=256.

- LH/RH: riêng Linear(384,256) + LayerNorm(256). Hai projection weights tách nhau nhưng initialized cùng quy tắc.
- BODY: Linear(28,256) + LayerNorm(256). Input 28-D = 14 normalized coordinates + 14 coord_valid flags; missing coordinates là zero.
- Không feed clean quality vào model mặc định.
- Learned part embedding [3,256].
- Learned SSL mask token riêng mỗi stream [3,256].
- Learned missing token riêng mỗi stream [3,256].
- Sinusoidal physical-time embedding 256-D từ relative_time_ms / 1000.
- Stack [B,T,S,256], flatten **time-major**, index = t×S + stream.
- Joint bidirectional self-attention giữa các stream/time. Đây không phải causal forecasting model.

Sinusoidal embedding dùng frequency 10000^(−2i/d), sin/cos của thời gian tính bằng giây. Không thêm absolute recording timestamp hoặc source embedding.

### 8.2 Transformer

DEFAULT:

- 4 layers; 8 attention heads; feed-forward 1024.
- GELU; dropout 0.1; batch_first=True; norm_first=True.
- Final LayerNorm(256).
- PyTorch TransformerEncoder reference [S16]; enable_nested_tensor=False cho MVP để semantics padding dễ test.
- Sau khi clone layers, reinitialize từng layer độc lập với seed stream cố định. Không để các layers khởi tạo tensor giống nhau do deep-copy upstream.
- Bias zero, Linear/attention matrices Xavier uniform; LayerNorm weight=1,bias=0; special embeddings normal std=0.02.
- Không dùng pretraining weights SHuBERT cho Transformer này.
- Same initialization checkpoint được dùng cho A1/A2/A3/A4/A6 của cùng seed.

Attention key padding mask [B,T×S] là True tại padded hoặc natural-invalid tokens. SSL-masked valid tokens vẫn được attention thấy qua learned mask embedding. Có thể ignore missing keys nhưng giữ outputs để reshape; missing outputs không vào loss/pool.

### 8.3 Heads

Context outputs h:[B,T,3,256].

- Hand decoder mỗi stream: Linear(256,512) → GELU → Linear(512,384).
- Body decoder: Linear(256,128) → GELU → Linear(128,14).
- Hand predictions p:[B,T,2,384] là raw vectors.
- Relation dùng normalize(p) theo feature dimension.
- Không có relation head riêng trong MVP: relation được dựng từ các predicted states. Điều này giữ objective đơn giản và làm redundancy hypothesis rõ.
- Không có learned retrieval head hoặc attentive pooling chưa được train.

Encoder forward signature:

~~~text
forward(student_batch) -> {
  "context": Float[B,T,3,256],
  "app_prediction": Float[B,T,2,384],
  "body_prediction": Float[B,T,14]
}
~~~

Clean targets truyền vào loss function riêng, không truyền vào forward encoder.

### 8.4 Leakage test bắt buộc

Giữ visible student inputs, masks và timestamps cố định; thay hidden clean features bằng noise **chỉ ở target side**. Model forward phải cho output y như trước trong eval mode.

Ngược lại, target loss phải thay đổi. Test này phát hiện loader/model đọc nhầm clean target để reconstruct.

Student được dùng các visible adjacent hand/body observations của cùng window. Đây là context hợp lệ, không phải label leakage. Tuy vậy interpolation shortcut phải được kiểm bằng diagnostic ở mục 16.

## 9. Objective: state và temporal relation

### 9.1 Notation

- q^a_t: L2-normalized clean DINO target cho hand a.
- p^a_t: raw hand decoder output; qhat^a_t = normalize(p^a_t).
- m^a_t: SSL mask; v^a_t: clean valid; c^a_t: clean quality.
- b_t,j và bhat_t,j: body coordinate target/prediction.
- All clean targets và weights stop-gradient; loss reduction FP32.

### 9.2 Hand state loss

Với a thuộc {LH,RH}:

\[
w^a_t=m^a_t v^a_t c^a_t,\qquad
\mathcal L^a_{\rm hand}
=\frac{\sum_{i,t}w^a_{i,t}\|p^a_{i,t}-q^a_{i,t}\|_2^2}
       {\sum_{i,t}w^a_{i,t}}.
\]

L_hand là mean của các hand streams có positive denominator.

**Scale convention:** dùng squared L2 sum trên vector 384-D. Tương đương MSE mean trên feature dimension nhân 384. Đây là normalization cố định của đề xuất, giúp loss không tự nhỏ đi 384 lần so với scalar relation loss. Nếu đổi convention thì lambda cũng phải đổi và experiment name phải mới.

State prediction dùng raw p để norm được ràng buộc; relation dùng normalized p. Không L2 normalize BODY và không coi pose như DINO token.

### 9.3 Body state loss

\[
w^{B}_{i,t,j}=m^B_{i,t}v^{B}_{i,t}
v^{\mathrm{coord}}_{i,t,j}c^{\mathrm{coord}}_{i,t,j},
\]
\[
\mathcal L_{\rm body}
=\frac{\sum_{i,t,j}w^B_{i,t,j}\,
  {\rm Huber}_{\beta_B}(\hat b_{i,t,j}-b_{i,t,j})}
 {\sum_{i,t,j}w^B_{i,t,j}}.
\]

Huber ở đây là **Smooth L1 convention**: 0.5 e²/β nếu |e|<β, ngược lại |e|−0.5β. DEFAULT β_B=1.0.

\[
\mathcal L_{\rm state} = \mathcal L_{\rm hand}
+\lambda_B\mathcal L_{\rm body},\quad \lambda_B=1.0.
\]

Body coefficient và averaging khác hand được khai rõ; không để dimension/missingness tự quyết định trọng số giữa modalities.

### 9.4 Cross-hand relation

Canonical orientation chỉ **LH → RH**. Student pair (t,u) giữ signed delta=u−t:

\[
R_{t,u}=\langle q^{LH}_{g(t)},q^{RH}_{g(u)}\rangle,\qquad
\hat R_{t,u}=\langle \hat q^{LH}_t,\hat q^{RH}_u\rangle.
\]

DEFAULT lag set:

\[
\Delta=\{-8,-4,-2,-1,0,1,2,4,8\}.
\]

Ở identity mapping và 25 Hz, |delta|=8 tương đương 320 ms; actual lag vẫn phải log từ timestamps.

Pair eligible khi:

- 0≤t,u<T, cùng sample, không wrap/cross-video.
- Both endpoint observations valid và quality≥threshold.
- Ít nhất một endpoint SSL-masked.
- Không phải lag≠0 nhưng source frame index hai endpoint trùng nhau.
- Không padded.
- Mapping g hợp lệ cho cả hai endpoints.

Weight w_tu = min(c_LH_t,c_RH_u). Loss:

\[
\mathcal L_{\rm rel}
=\frac{\sum_{(i,t,u)\in E}w_{i,t,u}
{\rm SmoothL1}_{\beta_R}(\hat R_{i,t,u}-R_{i,t,u})}
{\sum_{(i,t,u)\in E}w_{i,t,u}},\quad \beta_R=0.1.
\]

R nằm [-1,1]. Tính target từ clean normalized features sau gather; tuyệt đối không tạo target từ masked student input.

Dot product đối xứng khi đồng thời swap cả stream và thời điểm: R_LR(t,u)=R_RL(u,t). Vì vậy không thêm RH→LH như một bản sao loss. Với **stream orientation cố định**, R_LR(t,t+k) và R_LR(t+k,t) nói chung khác nhau, nên signed lag giữ hai constraints khác nhau. Bản thân scalar cosine không đủ để chứng minh đã học linguistic direction.

### 9.5 Pair budget

Default relation cap=32 scalar constraints/sample. Lấy candidates bằng generator riêng, không ảnh hưởng augmentation/dropout RNG.

- A4: stratify theo signed lag, round-robin lấy các bins không rỗng, rồi fill tới cap.
- A3: zero-lag candidates, cùng cap.
- A2: intra-stream candidates cùng cap; chỉ positive nonzero lags để tránh symmetric duplicates và self-diagonal=1.
- Nếu ít candidates: dùng toàn bộ, không duplicate để đủ cap.
- Control study lưu count/weight/lag distribution. Thêm matched-pair-budget sensitivity nếu counts chênh đáng kể.
- Không nhân loss lên chỉ vì có nhiều pairs; denominator là tổng weight đã chọn.
- Không dựng dense [B,T,T] trong MVP; pair table sparse [N,5] giảm chi phí và dễ audit.

### 9.6 Total loss và warmup

\[
\mathcal L=\mathcal L_{\rm hand}
+\lambda_B\mathcal L_{\rm body}
+\lambda_R(s)\mathcal L_{\rm rel}.
\]

DEFAULT λ_R,max=0.1; ramp tuyến tính từ 0 đến 0.1 trong first 100 successful optimizer updates. Pilot thử λ∈{0.1,1.0} nếu còn budget, dùng dev selection đã khai báo; không tune trên test.

λ không được xem là universal hoặc lấy từ SignRep/SHuBERT. Log component losses **trước** và sau coefficient, raw prediction norm, gradient norm theo module và target relation distribution.

Zero eligible targets: loss bằng differentiable zero. Nếu entire accumulated step không có hand/body targets, skip optimizer/scheduler update; không để weight decay đổi model ở empty step.

### 9.7 Reference PyTorch loss code

Code này là implementation skeleton của đề xuất. Pair builder và validation phải viết riêng. Pair table columns: [batch_index, time_1, stream_1, time_2, stream_2]; streams 0=LH,1=RH.

~~~python
import torch
import torch.nn.functional as F

def weighted_mean(value, weight, zero_reference):
    value = value.float()
    weight = weight.detach().float()
    denominator = weight.sum()
    numerator = (value * weight).sum()
    # Positive fractional weights must not be clamped up to 1.
    loss = numerator / denominator.clamp_min(1e-12)
    zero = zero_reference.float().sum() * 0.0
    return torch.where(denominator > 0, loss, zero), denominator

def hand_state_loss(pred, target, valid, quality, ssl_mask):
    # pred/target [B,T,A,D]; target already gathered by g.
    target = target.detach().float()
    error = (pred.float() - target).square().sum(dim=-1)
    weight = valid.float() * ssl_mask.float() * quality.detach().float()
    terms = []
    counts = []
    for a in range(pred.shape[2]):
        loss_a, denominator = weighted_mean(
            error[:, :, a], weight[:, :, a], pred
        )
        terms.append(loss_a)
        counts.append(denominator)
    active = torch.stack([x > 0 for x in counts]).float()
    terms = torch.stack(terms)
    loss = (terms * active).sum() / active.sum().clamp_min(1.0)
    return loss, torch.stack(counts)

def body_state_loss(pred, target, body_valid, coord_valid,
                    coord_quality, ssl_mask, beta=1.0):
    error = F.smooth_l1_loss(
        pred.float(), target.detach().float(),
        reduction="none", beta=beta
    )
    weight = (
        ssl_mask[:, :, None].float()
        * body_valid[:, :, None].float()
        * coord_valid.float()
        * coord_quality.detach().float()
    )
    return weighted_mean(error, weight, pred)

def relation_loss(pred, target, valid, quality, ssl_mask,
                  pairs, beta=0.1):
    # pairs must already pass time boundaries / source-frame checks.
    if pairs.numel() == 0:
        return pred.float().sum() * 0.0, pred.new_zeros(())
    i, t, a, u, b = pairs.long().unbind(dim=-1)
    p = F.normalize(pred.float(), dim=-1, eps=1e-6)
    q = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
    predicted = (p[i, t, a] * p[i, u, b]).sum(dim=-1)
    clean = (q[i, t, a] * q[i, u, b]).sum(dim=-1)
    eligible = valid[i, t, a] & valid[i, u, b]
    eligible = eligible & (ssl_mask[i, t, a] | ssl_mask[i, u, b])
    weight = eligible.float() * torch.minimum(
        quality[i, t, a], quality[i, u, b]
    ).detach().float()
    error = F.smooth_l1_loss(
        predicted, clean, reduction="none", beta=beta
    )
    return weighted_mean(error, weight, pred)
~~~

Caller MUST kiểm input finite và apply quality threshold vào valid trước loss; zero×NaN không giải quyết dữ liệu NaN. Quality có miền [0,1]; pair indices bounds/uniqueness phải được validate. Clean targets không có gradients. Shape errors phải raise, không broadcast im lặng.

Snippet là single-process reference, **không** tự xử lý global DDP reduction hoặc averaging khác nhau do gradient accumulation.

## 10. Optimizer, loop, logging và resume

### 10.1 Default pilot

| Setting | Giá trị khởi đầu của project |
| --- | --- |
| Seed đầu tiên | 17 |
| Microbatch | 8 windows |
| Gradient accumulation | 4 microbatches |
| Effective batch, 1 GPU | 32 windows/update |
| Optimizer | AdamW |
| Peak learning rate | 3e-4 |
| Weight decay | 0.05, trừ bias/LayerNorm/special embeddings |
| Adam betas | [0.9,0.95] |
| Gradient clipping | global norm 1.0 |
| LR warmup | 100 successful updates |
| LR sau warmup | cosine đến 3e-5 |
| Pilot updates | 1000 maximum, bị giới hạn thêm bởi wall time |
| Checkpoint interval | 100 successful updates và trước graceful stop |
| Precision | BF16 nếu GPU hỗ trợ; FP32 losses |
| Development device | CPU được phép cho smoke; real pilot ưu tiên GPU |

Đây là defaults để bắt đầu profile, không là hyperparameters từ paper. Full config phải lấy throughput/learning curve của pilot để chốt, không mặc định 1000 steps đủ cho paper.

### 10.2 Loop contract

1. Lấy effective-batch plan từ deterministic sampler.
2. Gather clean targets/validity/quality theo g; clone student inputs.
3. Apply masks trước forward.
4. Encoder + decoders trong autocast khi phù hợp.
5. Tính losses FP32; relation targets detached.
6. Scale loss theo accumulation convention, backward.
7. Nếu có FP16 GradScaler: unscale trước clipping; BF16 không mặc định cần scaler.
8. Reject non-finite losses/gradients, lưu diagnostic sample IDs + config; không tiếp tục một run lỗi mãi.
9. Optimizer step, scheduler step chỉ khi optimizer thực sự update.
10. Update step counter, log components và checkpoint theo interval.

Không dùng average loss sau detach để backward. Gradient của relation MUST đi tới hand decoders và context encoder, không đi vào teacher/cache.

### 10.3 Accumulation reduction phải được khai rõ

Default MVP: mỗi microbatch tính weighted mean riêng; accumulated objective là **mean của các microbatch means**. Đây không hoàn toàn bằng pooled weighted mean nếu denominator khác nhau. A1/A4 phải dùng cùng microbatch partition, masks và convention.

Nếu muốn pooled effective-batch weighted mean, cần denominator của cả effective batch trước backward và stream-wise reduction đúng; triển khai riêng, test với một big-batch reference. Không đổi convention giữa baselines.

DEFAULT DDP disabled. Khi mở rộng:

- Phải xử lý global numerator/denominator đúng theo stream.
- Nếu dùng local numerator với global denominator, cần tính cả world-size factor vì DDP average gradients.
- Kiểm tra one-process big batch vs two-process cùng effective batch, cả trường hợp một rank không có valid target.
- Không chỉ average local weighted means khi các ranks có coverage khác nhau mà gọi đó là exact global objective.
- Sampler/augmentation seed phải độc lập theo rank nhưng tái lập được.

### 10.4 Determinism

Seed riêng cho initialization, data/window sampling, masking, relation sampling, dropout và evaluation gallery.

Không dùng Python hash() cho stable sampling. Dùng SHA-256-derived integer hoặc deterministic PRNG.

Cùng seed controlled baselines phải reuse:

- initialization state,
- ordered window plan,
- mask plan,
- extraction cache,
- train/dev/test fingerprints.

Relation sampling generator không được làm thay đổi random stream của masking/dropout. Log determinism flags, precision/backend; numerical tolerance thay cho cam kết bitwise trên mọi GPU.

### 10.5 Checkpoint payload

~~~text
schema_version
model_state
optimizer_state
scheduler_state
scaler_state_or_null
successful_update
consumed_microbatch_index
sampler_state
rng_states_python_numpy_torch_cpu_cuda
config_resolved
config_sha256
sources_lock_sha256
cache_manifest_sha256
split_fingerprint
git_commit_and_dirty_patch_hash
metric_history_path
elapsed_seconds
budget_consumed
~~~

Checkpoint local resume có thể chứa RNG objects; chỉ load checkpoint do project tạo theo policy tin cậy. Public upstream weights ưu tiên weights_only loader nếu compatible. Resume MUST kiểm config/source/cache hashes; mismatch không được resume âm thầm.

Exact-resume MVP lưu ở boundary optimizer update. Nếu dừng giữa accumulation, bỏ partial gradients rồi quay lại đầu effective batch từ checkpoint đã hoàn tất.

Atomic checkpoint write + rename; giữ last và checkpoint intervals, không overwrite file tốt bằng file đang write. Không “resume” chỉ bằng model_state rồi quên optimizer/scheduler.

### 10.6 Logs

Dùng local JSONL/CSV/TensorBoard; external trackers không bắt buộc. Log tối thiểu:

- successful_update, elapsed_time, examples, LR, grad_norm, GPU peak memory;
- L_hand_LH, L_hand_RH, L_body, L_relation, scaled components;
- state/coordinate/pair denominators và counts;
- masked ratios, natural coverage, accepted/rejected windows;
- predicted feature norm, contextual variance, rank diagnostics;
- mean/std relation target và prediction theo lag;
- teacher/source/cache/config/split fingerprints.

Không chỉ log total loss: total A1 và A4 khác objective nên không so sánh trực tiếp để chọn model.

## 11. Frozen embedding export và inference

### 11.1 Inference pipeline

Raw video → cùng timestamp/perception/crop policy → frozen DINO hand features + BODY → context encoder ở eval mode → deterministic pooling.

- Tắt SSL masking/dropout.
- Bỏ hand/body decoders khi export encoder.
- DINO và perception vẫn cần ở inference từ RGB; **không** nói model chỉ cần một Transformer nếu chưa có cached features.
- Input cached features có cùng source fingerprint được hỗ trợ để đánh giá nhanh.
- Không đưa gloss/text/signer metadata vào embed function.

Public interface:

~~~text
embed_cached(recording_cache, encoder_checkpoint) -> {
  "embedding": Float[2304],
  "video_id": String,
  "coverage": Dict,
  "encoder_sha256": String,
  "preprocess_fingerprint": String,
  "status": "ok" | "insufficient_observation"
}
~~~

2304 = 3 streams × 3 temporal bins × 256 dimensions.

### 11.2 Long clips và overlap

- Inference window length 64, stride 32.
- Short clip pad, không loop/repeat để lấp window.
- Với overlapping windows, gom contextual output ở cùng canonical (frame,stream) bằng arithmetic mean trên các windows phủ vị trí đó.
- Natural-invalid positions không tham gia mean.
- Không mean vectors đã final L2 normalize của từng window thay cho canonical aggregation.
- Tail window cover last valid frame; không bỏ phần cuối do T không chia hết stride.
- A1/A2/A3/A4 và external adapted eval dùng policy deterministic được khai rõ.

### 11.3 Fixed temporal pyramid

Toàn clip chia 3 bins: whole clip, first half, second half theo **relative physical time**, không chia theo số frames padded.

Với mỗi stream và mỗi bin:

1. Mean h trên naturally valid observations.
2. L2 normalize pooled vector; empty bin → zero vector.
3. Concatenate theo thứ tự [LH bins; RH bins; BODY bins].
4. L2 normalize final vector.
5. Không có learned attention/projection/pooler.

Nếu toàn clip không có valid hand observations hoặc final norm=0: status=insufficient_observation, không insert zero vector vào gallery như một item bình thường. Coverage và attempted-denominator được báo riêng.

Các controls learned encoder có cùng 2304-D và pooling. A0 frozen raw-feature baseline dùng cùng binning/normalization nhưng raw dimensions: 3×(384+384+14)=2346-D. Dimension difference này phải ghi rõ; A0 là diagnostic, còn A4−A1 là primary causal contrast.

OPTIONAL A0-hands-only dùng 2304-D raw feature pyramid không BODY; báo riêng, không chọn phiên bản kém hơn để làm đẹp gain.

Fixed mean pooling có thể mất fine temporal order. Nếu muốn DTW/token retrieval hoặc learned pooling, dùng separate evaluation variant áp dụng cho tất cả controls; không thêm cho A4 riêng.

### 11.4 Export artifacts

- embeddings.npy [N,D] float32;
- index.jsonl: video_id, row_index, status, fingerprints, coverage;
- evaluation labels giữ ở labels.jsonl riêng;
- embeddings không order theo gloss/signer để “nhìn giống cluster”; index phải join bằng video_id.
- encoder.pt chỉ chứa inference weights/config/hash; resume checkpoint đầy đủ ở file khác.
- MODEL_CARD.md ghi input assumptions, languages, teacher, training data, nonmanual limitations và measured metrics.

## 12. Retrieval protocol và metrics

### 12.1 Main setting: seen gallery, held-out queries

Khi ASL Citizen version có official train/dev/test:

- Gallery candidates: official train videos.
- Dev queries: official dev.
- Locked test queries: official test.
- SSL dùng pretraining corpus đã audit; nếu ASL Citizen train là corpus, gallery videos có thể đã nằm trong SSL train và phải ghi điều đó.
- Query signer không có trong gallery cho signer-independent dictionary retrieval.
- SSL signer overlap cần audit riêng; chưa rõ thì chỉ claim downstream query/gallery signer disjoint.

Nếu official splits không thỏa signer disjoint hoặc metadata không rõ, build explicit group-disjoint custom protocol và gọi là **custom SIGNREL retrieval**, không gọi exact official benchmark.

Primary gallery K=1/video per gloss, chọn bằng deterministic hash với gallery_seed=2026. K=5 là secondary nếu đủ candidate videos; ghi số exemplars thật/gloss. Freeze gallery once và reuse across models.

Queries có gloss không có gallery: đưa vào excluded_no_gallery count; closed-set R@k chỉ trên covered queries. Báo coverage rate và optional open-set result riêng.

Mọi video/query/gallery cùng duplicate recording hoặc same signer theo protocol phải loại, và audit lưu lý do.

### 12.2 Rank theo gloss

Similarity = cosine final normalized embeddings. Nếu K>1:

\[
s(q,g)=\max_{v\in\mathrm{gallery}(g)} \langle z_q,z_v\rangle.
\]

Rank gloss IDs duy nhất theo s(q,g), không rank 5 exemplar của cùng gloss thành 5 predicted classes. Ties resolve stable bằng gloss_id; MRR tie policy phải ghi rõ.

- Micro R@k: mean query indicator(correct gloss in top k).
- Macro gloss R@k: mean per-gloss query accuracy, mỗi gloss cùng weight.
- MRR: mean 1/rank(correct gloss).
- Primary k=1; secondary k=5 hoặc min(5,number_of_glosses).
- Report dataset-wide attempted query count, valid embeddings count, gallery-covered count và final scored count.

Không thêm supervised nearest-centroid classifier mà vẫn gọi frozen dictionary retrieval nếu centroid/weights được fit bằng evaluation query labels.

### 12.3 Baselines

| ID | Encoder/init | Objective | Vai trò |
| --- | --- | --- | --- |
| A0 | Raw frozen DINO + pose | Không train | Teacher-feature sanity baseline |
| A1 | Same new encoder | State only | Primary baseline |
| A2 | Same new encoder | State + intra-hand lag relations | Kiểm tra temporal vs cross-stream |
| A3 | Same new encoder | State + cross-hand delta=0 | Kiểm tra lag contribution |
| A4 | Same new encoder | State + cross-hand signed lag | Proposed main |
| A5 | Same new encoder | State + symmetrized cross-hand temporal relations | Optional direction diagnostic |
| A6 | Same new encoder | State + shuffled relation targets | Negative control |
| A7 | State-only larger/longer | State only | Compute-matched sensitivity |
| EXT-SignRep | Public pretrained SignRep | Upstream | Different init/data; contextual comparison |
| EXT-SHuBERT | Public pretrained SHuBERT | Upstream | Different init/data; contextual comparison |
| EXT-SignSeek | Supervised method | Upstream if usable | Different label budget; separate table |

A2 chỉ a=b, delta∈{1,2,4,8}; cả LH/RH, cap chung 32, không self-diagonal. A3 dùng canonical LH→RH delta=0. A4 dùng lag set ở mục 9.

A5 nếu làm: với t<u, target = 0.5×[R_LR(t,u)+R_LR(u,t)], prediction cùng symmetrization. Cần đủ valid endpoints của cả hai edges; scalar-constraint và underlying-edge counts phải báo, matched-budget experiment riêng. State loss vẫn có temporal position nên A5 chỉ loại direction khỏi **relation objective**, không loại mọi thông tin order khỏi encoder.

A6 shuffle clean relation scalars giữa different windows trong batch với permutation không identity, giữ lag bucket khi có thể. Nếu batch/bucket chỉ có 1 window, skip relation control group và log. Không shuffle state targets. Control này có thể gây optimization noise; nó không tự tách mọi confound.

A7 chạy thêm budget state-only theo **measured training compute/time** của A4, hoặc tăng depth có parameter count được report. Chọn biến thể trước confirmatory run. Không gọi same updates là compute matched nếu relation có overhead.

### 12.4 Statistical reporting

- Confirmatory: seeds [17,29,43], paired initialization/data plan theo seed.
- Report từng seed, mean ± sample SD, paired A4−A1.
- Bootstrap 1000 replicates theo gloss clusters; OPTIONAL signer-cluster hoặc hierarchical signer/gloss sensitivity.
- Paired bootstrap dùng cùng query IDs/cluster draws cho hai models.
- Mean/CI của subset không được gọi population-wide nếu coverage thấp.
- Pre-register primary contrast; các subgroup/multiple ablations exploratory trừ khi đã đăng ký.
- Không dùng p-value của hàng nghìn correlated video frames như independent samples.

## 13. Minimal-pair và các downstream diagnostics

### 13.1 Điều kiện dữ liệu

Ưu tiên curated natural pairs từ [S13] khi thật sự có annotation. Repo code không đủ: README còn TODO đường dẫn stimuli.

Nếu annotation chưa có, Agent:

1. Hoàn thành API/schema/evaluator với synthetic fixture.
2. Ghi minimal_pair_real=BLOCKED_EXTERNAL.
3. Có thể tạo candidate list từ metadata phong phú, nhưng gọi là **candidate phonological neighbors**, không tự gán verified minimal pair.
4. Cần người có chuyên môn ASL/linguistics xác nhận property và pairs trước primary phonological claim.
5. Không dùng English gloss spelling/edit distance làm phonological distance.

### 13.2 Triplet schema và scorer

~~~json
{
  "triplet_id": "curated_triplet_0001",
  "anchor_video_id": "video_a",
  "positive_video_id": "video_p",
  "minimal_pair_video_id": "video_n",
  "anchor_gloss_id": "gloss_a",
  "positive_gloss_id": "gloss_a",
  "negative_gloss_id": "gloss_b",
  "property": "handshape",
  "verified_by": "annotation_record_id",
  "annotation_version": "resolved_version",
  "signer_control": "cross_signer_matched",
  "source_control": "matched_recording_conditions"
}
~~~

Scorer:

\[
m_i=\cos(z_{a_i},z_{p_i})-\cos(z_{a_i},z_{n_i}).
\]

Report mean margin, accuracy(m>0), ties, paired bootstrap theo sign pair/property và eligible triplet coverage.

- Positive là cùng sign, signer khác nếu có; không chọn positive cùng recording còn negative khác lighting làm shortcut.
- Negative khác sign và được xác nhận khác đúng property cần đo.
- Cân bằng source/signer khi có thể, báo residual confounds.
- Không giả định mouthing/orientation/nonmanual annotation đủ chỉ vì có handshape annotations.
- Frame/part perturbations là stress test, không ground-truth grammatical judgment.

### 13.3 Optional probes

Frozen linear gloss probe:

- Fit trên downstream train labels; tune dev; test locked.
- Encoder frozen; khai label budget và trainable head parameters.
- Không gộp probe result vào “zero-label retrieval”.
- Same label subset/seeds/optimizer head cho A1/A4.

Signer leakage probe:

- Frozen representation, signer classification trên groups được biết.
- Dùng diagnostic để phát hiện identity retention; không tự nói lower signer accuracy chứng minh semantic invariance.
- Control semantic/gloss distribution vì signer data có thể không cân bằng.

Temporal shift stress:

- Shift riêng RH ±1/±2/±4 canonical positions chỉ lúc diagnostic.
- So sánh feature/loss sensitivity; không huấn luyện shift như positive.
- Không gán shifted clip là một sign khác có label thật.

Translation/ASL-MTP không thuộc MVP: cần decoder/scorer và protocol có text supervision riêng; không cosine trực tiếp video vector với một câu English để thay comprehension evaluation.

## 14. Repository layout và trách nhiệm module

Các paths dưới đây là paths Agent phải tạo trong repo implementation, không phải source paths đã tồn tại trong workspace này.

| Path | Trách nhiệm |
| --- | --- |
| pyproject.toml | Package signrel, minimal dependencies, optional extras |
| src/signrel/__main__.py | CLI entry; dispatch subcommands |
| src/signrel/config.py | Typed config, unknown-key rejection, validation |
| src/signrel/sources.py | Resolve/hash/pin upstream assets; source lock |
| src/signrel/data/manifest.py | Schemas, canonicalization, label-free enforcement |
| src/signrel/data/adapters.py | ASL Citizen/local/YouTube-ASL adapters |
| src/signrel/data/splits.py | Group split, duplicate/query exclusion, fingerprints |
| src/signrel/data/decode.py | PTS decoding, canonical grid, VFR tests |
| src/signrel/data/perception.py | MediaPipe assets/API, identity/hand association |
| src/signrel/data/crops.py | Hand bounding square, padding, resize |
| src/signrel/data/cache.py | Schema, atomic write/read, audit/hash |
| src/signrel/data/windows.py | Deterministic sampler, short/padded windows |
| src/signrel/data/corruption.py | Distinct mask types, g mapping, no leakage |
| src/signrel/models/teacher.py | Frozen DINO loader/extractor |
| src/signrel/models/encoder.py | Joint 3-stream Transformer and position embedding |
| src/signrel/models/heads.py | State decoders; no learned retrieval pooler |
| src/signrel/losses/state.py | Hand/body weighted reductions |
| src/signrel/losses/pairs.py | Sparse canonical pair builder and budget |
| src/signrel/losses/relation.py | Relation loss + controls |
| src/signrel/train/loop.py | Optimizer, accumulation, precision, budget checks |
| src/signrel/train/checkpoint.py | Atomic checkpoint, hash validation, exact resume |
| src/signrel/train/diagnostics.py | Collapse/shortcut/coverage/target variance |
| src/signrel/eval/embed.py | Deterministic long-clip export and pyramid |
| src/signrel/eval/retrieval.py | Query/gallery audit, gloss ranks, metrics |
| src/signrel/eval/minimal_pairs.py | Annotation-driven triplets, margin |
| src/signrel/eval/bootstrap.py | Paired clustered confidence intervals |
| src/signrel/eval/probe.py | Optional frozen linear probe |
| src/signrel/baselines/ | Separate adapters for optional upstream models |
| configs/ | smoke, pilot, full template, A1–A7, eval configs |
| tests/ | Meaningful algebra/data/leakage/resume/eval tests |
| scripts/ | Thin documented wrappers, no hidden hyperparameters |
| artifacts/ | Source/data/config locks and protocol, JSON/CSV reports |
| docs/RUNBOOK.md | Install/import/extract/train/eval/resume commands |
| docs/STATE.md | Current phase, evidence, blockers, next concrete action |
| docs/DECISIONS.md | Changes from this spec and rationale |
| docs/RELATED_WORK.md | Source-grounded comparison and unresolved novelty |
| third_party/NOTICE.md | Code/model/data attribution and exact license sources |

Không vendor toàn repo upstream vào source package nếu chỉ cần teacher inference. Nếu reuse code, giữ license header và provenance cụ thể. Local data/cache/checkpoints không đưa vào git.

### 14.1 CLI phải triển khai

| Command | Input → output | Acceptance |
| --- | --- | --- |
| doctor | env → environment.json | CPU/GPU/import/FFmpeg/disk findings thực tế |
| resolve-sources | source config → sources.lock.json | Pin/hash và load tests hoặc explicit blockers |
| import-data | metadata/local roots → canonical manifests | Không fake dataset IDs |
| audit-splits | manifests → split_audit.json | Recording/hash/query overlap kiểm đúng |
| extract-cache | video manifest → .npz + cache manifest | Atomic, idempotent, traced teacher |
| audit-cache | cache manifest → cache_audit.json | Shapes, masks, finite, hashes, coverage |
| make-synthetic | seed/shape → synthetic cache | Rõ synthetic; không đưa vào real metrics |
| profile | config/cache → profile.json | Measured throughput/memory/cost |
| train | resolved config → checkpoint + logs | Loss/backward/resume/budget thật |
| export-embeddings | caches/checkpoint → embedding/index files | Correct inference fingerprint/shape |
| eval-retrieval | embedding/index/labels/protocol → metrics | Gloss-level rank, overlap fail |
| eval-minimal-pairs | triplets/embeddings → metrics | Annotation/coverage enforcement |
| aggregate | run manifests → results CSV/Markdown | Missing runs là missing, không là zero |
| build-report | artifacts → research_report.md | Facts, uncertainty và exploratory status rõ |

Common flags: --config, --output, --dry-run, --limit-videos, --resume, --device. Mọi command có --help, return code rõ, structured error report.

Train --dry-run validate paths/shapes/config, không tạo placeholder checkpoint. Eval không “pass” nếu thiếu real labels/embeddings.

### 14.2 State tracking

STATE.md gồm:

~~~text
phase:
status: pending | running | passed | failed | blocked_external
last_successful_command:
artifact_paths:
artifact_hashes:
verification_evidence:
blocked_by:
next_action:
remaining_budget:
~~~

Sau mỗi phase update bằng output thật. Agent không tick checklist chỉ vì đã tạo file skeleton.

## 15. Config đầy đủ cho bounded pilot

File đề xuất: configs/pilot_a4.yaml. Các values dưới đây là project defaults; đường dẫn cần được resolve theo môi trường implementation.

~~~yaml
schema_version: 1
experiment:
  name: pilot_a4
  variant: A4
  seed: 17
  output_dir: runs/pilot_a4_seed17
  synthetic: false
paths:
  data_root: data
  ssl_manifest: data/manifests/ssl_train.jsonl
  ssl_dev_manifest: data/manifests/ssl_dev.jsonl
  cache_manifest: data/cache/cache_manifest.jsonl
  sources_lock: artifacts/sources.lock.json
  protocol: artifacts/protocol.json
data:
  language: ASL
  fps: 25
  window_frames: 64
  min_frames: 32
  sampling: recording_uniform
  quality_threshold: 0.5
  split_seed: 2026
  max_resample_attempts: 10
  allow_caption_boundaries: false
teacher:
  name: dinov2_vits14
  feature_dim: 384
  feature_key: x_norm_clstoken
  repo_dir: third_party/dinov2
  checkpoint_path: weights/dinov2_vits14_pretrain.pth
  frozen: true
  normalization: l2
preprocess:
  crop_size: 224
  hand_margin: 1.5
  hand_min_side_px: 24
  max_clipped_fraction: 0.5
  shoulder_min_distance_px: 20
  body_landmark_indices: [0, 11, 12, 13, 14, 15, 16]
  hand_assignment: pose_wrist_temporal
model:
  streams: [LH, RH, BODY]
  hidden_dim: 256
  layers: 4
  heads: 8
  feedforward_dim: 1024
  dropout: 0.1
  norm_first: true
  time_embedding: sinusoidal_seconds
  nested_tensor: false
corruption:
  hand_mask_ratio: 0.5
  body_mask_ratio: 0.25
  shared_span_probability: 0.5
  span_min: 4
  span_max: 12
  minimum_visible_tokens: 4
  whole_hand_dropout: 0.0
  time_resample: false
  horizontal_flip: false
  temporal_reverse: false
loss:
  hand_reduction: squared_l2_sum_then_stream_mean
  body_weight: 1.0
  body_smooth_l1_beta: 1.0
  relation:
    enabled: true
    mode: cross_signed
    stream_pairs: [[LH, RH]]
    lags: [-8, -4, -2, -1, 0, 1, 2, 4, 8]
    max_pairs_per_sample: 32
    sampling: lag_stratified
    require_masked_endpoint: true
    smooth_l1_beta: 0.1
    max_weight: 0.1
    ramp_updates: 100
    reject_duplicate_source_frame_for_nonzero_lag: true
train:
  device: auto
  precision: bf16_if_supported_else_fp32
  microbatch_size: 8
  gradient_accumulation: 4
  accumulation_reduction: mean_of_microbatch_weighted_means
  optimizer: adamw
  learning_rate: 0.0003
  min_learning_rate: 0.00003
  weight_decay: 0.05
  betas: [0.9, 0.95]
  warmup_updates: 100
  max_updates: 1000
  grad_clip_norm: 1.0
  checkpoint_every: 100
  log_every: 10
  workers: 2
  ddp: false
eval:
  window_frames: 64
  stride_frames: 32
  pooling: physical_time_pyramid_mean
  temporal_bins: [1, 2]
  normalize_each_block: true
  gallery_seed: 2026
  gallery_k: 1
  gloss_aggregation: max_exemplar_cosine
  primary_metric: macro_gloss_recall_at_1
  bootstrap_replicates: 1000
budget:
  phase: pilot
  max_wall_minutes: 120
  max_gpu_hours: 2.0
  max_download_gb: 5.0
  max_new_cache_gb: 10.0
  allow_paid_cloud: false
  allow_full_scale: false
~~~

**Những ngân sách trên là cap đề xuất cho một pilot run**, không là authorization mở cloud account hay tải mọi dataset. Agent tạo global budget ledger cho tổng các runs; không reset cap 2 giờ cho hàng trăm trials tự phát.

A1 config phải kế thừa cùng config resolved: đổi experiment.variant/name, experiment.output_dir sang run directory riêng và loss.relation.enabled=false. Seed khác cũng phải có output_dir riêng. Không thay dropout, mask ratio, crop, sampler hoặc initialization. Configs A2/A3 thay mode/pairs/lags có log; toàn bộ diff phải được lưu. Train MUST từ chối ghi đè run directory có config hash khác nếu không có migration/resume policy rõ ràng.

### 15.1 Config validation

MUST reject:

- hidden_dim không chia hết cho heads;
- feature_dim khác source lock;
- teacher.frozen=false trong setting chính;
- stream order khác cache schema chưa có adapter;
- relation.cosine với BODY 14-D;
- lags không integer hoặc vượt T−1;
- β≤0, quality ngoài [0,1], negative loss weights;
- ddp=true khi global reduction chưa test;
- train data contains label fields;
- test query leakage hoặc nguồn cache mismatch;
- full-scale phase khi budget.allow_full_scale=false.

Unknown keys MUST lỗi; không silently ignore typo. Ghi resolved config sau inheritance, không chỉ file parent.

## 16. Diagnostics và quyết định có nên mở rộng training

### 16.1 Teacher suitability gate

Trước train hàng giờ, đo trên ≥20 clips thật nếu có:

- crop/hand assignment quality qua preview;
- hand valid coverage, body valid coverage, unique source frame ratio;
- teacher norm, per-dimension variation;
- R_clean mean/std/histogram theo lag;
- appearance distance trong cùng clip và giữa clip;
- A0 retrieval trên **dev** nếu có labels, không trên locked test.

Nếu relation target std <0.01 ở đa số lag buckets, tạo cảnh báo low_target_variation và kiểm tra crop/normalization/teacher. Threshold 0.01 là diagnostic heuristic, không là định lý “không học được”.

Nếu constants/lag-only predictor gần như bằng learned relation prediction, giảm kỳ vọng relation supervision. Không scale lambda lớn để che target không có thông tin.

Không tự thay checkpoint hoặc crop policy sau khi đã train A1 mà giữ kết quả làm control. Chọn teacher/preprocessing mới thì rebuild cache và chạy controls lại.

### 16.2 Collapse gate

Trên một cố định held-out SSL dev subset, log cho context pooled/raw features:

- mean per-dimension std;
- average cosine giữa khác videos;
- covariance spectrum và effective rank r_eff=exp(−Σ p_j log p_j), p_j là eigenvalues dương normalized;
- cùng metrics của teacher baseline để có scale reference;
- representation norm và decoder raw prediction norm.

N phải đủ để estimate spectrum; covariance rank bị giới hạn bởi min(N−1,D). Không nói rank thấp do N nhỏ là collapse. Dùng sample subset/downproject deterministic cho diagnostics nếu D quá lớn, ghi method.

Fixed teacher target không tự bảo đảm context encoder không collapse: decoders có thể học conditional mean hoặc exploit shortcuts.

### 16.3 Shortcut controls

So sánh state/relation errors trên dev với:

1. Constant teacher-target mean predictor fit **SSL train only**.
2. Nearest-visible-frame interpolation predictor, không labels.
3. Lag-only mean relation predictor fit SSL train.
4. State-only encoder A1.

Nếu A4 giảm training relation loss nhưng frozen retrieval không hơn A1, kết luận optimization target tốt hơn chưa đủ cho representation utility.

Nếu advantage biến mất khi thay matched compute/masks/gallery, ghi confound. Không chọn subset thuận lợi để viết headline.

### 16.4 Diagnostic-driven changes

| Kết quả | Hành động được phép |
| --- | --- |
| Hand swap/crop lỗi | Sửa perception, tăng schema/cache version, rerun extraction và controls |
| Target relation gần constant | Xem teacher/crop; giữ result negative; thiết kế teacher khác ở experiment mới |
| State reconstruction chỉ interpolation | Tăng span length trong ablation có same masks cho A1/A4 |
| Hidden-feature leakage | Dừng real training, sửa loader/model, tests, invalidate affected runs |
| OOM | Giảm microbatch, tăng accumulation để giữ effective batch nếu có thể; same change all controls |
| GPU không hỗ trợ BF16 | FP32 hoặc FP16 đã test; log dtype và run equivalence |
| DINO feature không phân biệt phonological cues | Không claim phonology; teacher adaptation/new target là research branch |
| Retrieval improved, signer probe strong | Kiểm source/signer controlled eval; không suy ra invariance |
| Relation không giúp qua 3 seeds | Báo negative result; không tự thêm 5 losses để cứu giả thuyết |

## 17. Các phase triển khai và nghiệm thu

Agent chạy tuần tự. Các bước có thể bị chặn bởi dữ liệu/GPU không chặn phần code/test độc lập.

### Phase 0 — Discovery và protocol

**Input:** tài liệu này, workspace, các dữ liệu local được phép.

**Tasks:**

1. Inventory repo hiện hữu bằng rg; đọc AGENTS.md nếu có.
2. Không overwrite project hoặc report cũ; tạo branch/worktree/project folder phù hợp.
3. Chạy doctor, ghi environment.json và actual limits.
4. Đọc source cards [S01,S02,S03,S09,S10,S11,S12,S13].
5. Tạo RELATED_WORK, DECISIONS, source manifest unresolved, data inventory.
6. Khóa research setting, primary endpoint, query/gallery rule và budget ledger.

**Artifacts:** docs/STATE.md, environment.json, protocol.json, sources.lock.json (có trạng thái unresolved nếu cần).

**Pass:** biết task boundaries, real-data availability và compute; chưa cần GPU hoặc full dataset.

### Phase 1 — Core package và synthetic fixtures

**Tasks:**

1. Tạo package/typed config/CLI.
2. Viết synthetic cache generator có correlated LH/RH temporal states, missing patterns và body coords.
3. Viết manifest/cache schemas, window sampler, masks/g mapping.
4. Viết encoder, heads, state/relation losses, sparse pair builder.
5. Viết unit tests cho algebra, boundary, leakage và invalid data.
6. Train synthetic 20–50 updates trên CPU/GPU nhỏ.
7. Verify loss finite, encoder gradients, target no-grad, checkpoint resume.

**Pass:** pytest core passed, smoke output có synthetic=true, no upstream weights required. Không coi loss synthetic giảm là evidence H1.

### Phase 2 — Resolve upstream và perception adapter

**Tasks:**

1. Pin DINO source; tải checkpoint đúng identifier và hash.
2. Resolve MediaPipe assets và hashes từ official docs.
3. Teacher forward test; verify dim/key/preprocessing.
4. Adapter decoder VFR, canonical timestamps.
5. Hand/pose perception, anatomical assignment, crop/quality policy.
6. Nếu optional upstream environment lỗi, isolate và ghi blocker; không làm hỏng core.

**Pass:** required teacher/perception có VERIFIED_RUNTIME và preview hợp lý, hoặc các phần bị chặn được nêu chính xác. Chưa được đánh dấu real pipeline pass nếu weights chưa load.

### Phase 3 — Manifest và split audit

**Tasks:**

1. Import một subset 20–50 local videos nếu có; labels sang eval manifest riêng.
2. Resolve source/group IDs, available video hashes và licenses.
3. Build SSL train/dev; giữ downstream query/gallery logic.
4. Audit exact duplicates, same recording cuts và known signer overlaps.
5. Lưu coverage và unknown identity.

**Pass:** label-free manifest, fingerprints, split_audit passed; không có known query leakage.

### Phase 4 — Real cache pilot

**Tasks:**

1. Extract 20–50 videos trước.
2. Audit .npz, norms, per-stream coverage, g/PTS/source indices.
3. Xuất contact sheets/crop overlays phục vụ QA; không cần upload video ra dịch vụ ngoài.
4. Teacher suitability gate và target variation report.
5. Time/memory/disk profile; dự đoán cost từ measured samples.
6. Chỉ mở rộng subset khi budget cho phép.

**Pass:** cache chạy lại idempotent, exact source hashes, no NaN, QA samples traceable. Nếu quality thấp, sửa trước train.

### Phase 5 — Real training smoke và pilot A1/A4

**Tasks:**

1. Lưu same-seed init checkpoint.
2. Train A1 và A4 trên identical windows/masks/cache.
3. Overfit 1–4 fixed windows riêng để debug target fitting.
4. Main bounded pilot ≤1000 updates/run và cap ngân sách global.
5. Log collapse/shortcut diagnostics.
6. Resume từ checkpoint và so sánh với uninterrupted trace.
7. Export frozen embeddings, eval dev retrieval nếu có.
8. Ghi measured throughput/peak VRAM và total cost.

**Pass:** training/inference hoạt động, losses/gradients đúng, tests và resume pass. H1 chưa cần đúng để phase implementation đạt; negative result là output hợp lệ.

### Phase 6 — Evaluation pipeline thật

**Tasks:**

1. Freeze embedding/export policy và gallery.
2. Hand-worked toy test cho rank theo gloss.
3. Real dev retrieval + coverage audit.
4. Minimal pairs chỉ chạy khi annotation verified; nếu thiếu, evaluator tests pass và real result blocked.
5. Clustered paired bootstrap và report generator.
6. Optional frozen probe để đo utility với label budget riêng.

**Pass:** metrics tái lập được từ saved embeddings/protocol; no test tuning.

### Phase 7 — Controlled ablations

**Prerequisite:** pilot stable, có ngân sách mới cụ thể cho matrix.

**Tasks:**

1. Lock A1/A2/A3/A4/A6 configs; optional A5/A7 nếu resources cho phép.
2. Chạy 3 seeds cho confirmatory contrast A4−A1.
3. Match data, mask, init, pooling, teacher và model selection rule.
4. Báo pair counts/compute differences, không che costs.
5. External pretrained baselines chạy separate table và source/data disclosure.

**Pass:** đủ run manifests và stats, hoặc explicit incomplete rows. Không tick passed nếu chỉ có một seed.

### Phase 8 — Research package và paper-readiness

**Tasks:**

1. Re-audit recent prior art/novelty.
2. Tạo main table, ablations, coverage/compute table và limitations.
3. Viết method theo implementation cuối; bỏ những branches không chạy.
4. Ghi public/restricted assets, code/weights provenance.
5. Chuẩn bị README/RUNBOOK, inference model card và reproducibility manifest.
6. Tất cả số liệu dẫn về metrics file + checkpoint + source/cache/split hashes.

**Pass:** có kết luận có bằng chứng, kể cả bác bỏ hypothesis. Paper-ready implementation không đồng nghĩa guaranteed publication.

## 18. Kiểm thử bắt buộc

Các tests dưới đây bảo vệ semantics và research validity, không chỉ kiểm tra implementation gọi chính nó.

| Test | Fixture độc lập | Điều cần chứng minh |
| --- | --- | --- |
| Weighted denominator | Errors [1,3], weights [0.1,0.2] | Mean=7/3; không clamp denominator lên 1 |
| Zero eligible | All weights 0, finite predictions | Differentiable zero; no NaN |
| Target detach | target.requires_grad=True | backward không tạo target gradient |
| Stream balance | LH có 10 tokens, RH có 1 với known errors | Mean theo stream đúng, không global-count dominate |
| Body coordinates | Một coord invalid, very large error | Invalid coord không đóng góp |
| Mask OR | Visible-visible vs one-masked edge | Chỉ edges có masked endpoint eligible |
| Pair direction | Known LH/RH vectors | +lag và −lag có targets khác khi data khác |
| Pair boundaries | T=4, lags [-3,0,3] | Không wrap như torch.roll |
| Canonical pairs | Both directions available | Không double-count transpose duplicates |
| Source duplicates | Same decoded index ở lag≠0 | Pair bị loại theo policy |
| g correspondence | Nonidentity g=[0,2,3,5] | Target lấy g(u), không g(t)+delta |
| Hidden-feature leakage | Change clean hidden target only | Encoder output không đổi |
| Missing vs masked | Real missing hand + SSL mask | Missing không có target; masked vẫn có query token |
| VFR decode | Known irregular PTS video | Mapping nearest/tolerance/duplicates đúng |
| Hand crossing | Labeled QA clip | Stream anatomical không swap im lặng |
| Cache fingerprint | Change teacher/crop config | Không reuse incompatible cache |
| Gallery aggregation | 2 exemplar cùng gloss | Rank gloss unique, max policy đúng |
| Leakage audit | Same recording/query signer | Eval fail hoặc excludes đúng protocol |
| Long-clip tail | T=65/95/129 | Last valid frames không bị bỏ |
| Pooling padding | Same clip + extra padding | Embedding không đổi trong tolerance |
| Resume equivalence | Fixed synthetic plan/dropout seeds | Continuous vs resume cùng trajectory |
| Budget stop | Very small wall/update cap | Save checkpoint và stop reason hợp lệ |

### 18.1 NumPy oracle độc lập cho relation

Snippet tự viết này chạy được không cần PyTorch; dùng nó đối chiếu formula và pair semantics. Test PyTorch implementation với cùng arrays rồi compare loss trong tolerance.

~~~python
import numpy as np

def relation_oracle(pred, target, valid, quality, masked, pairs, beta=0.1):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pn = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-6)
    qn = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
    numerator = 0.0
    denominator = 0.0
    for i, t, a, u, b in np.asarray(pairs, dtype=np.int64).reshape(-1, 5):
        if not (valid[i, t, a] and valid[i, u, b]):
            continue
        if not (masked[i, t, a] or masked[i, u, b]):
            continue
        weight = min(quality[i, t, a], quality[i, u, b])
        error = float(pn[i, t, a] @ pn[i, u, b]
                      - qn[i, t, a] @ qn[i, u, b])
        absolute = abs(error)
        penalty = (0.5 * error * error / beta if absolute < beta
                   else absolute - 0.5 * beta)
        numerator += weight * penalty
        denominator += weight
    return numerator / denominator if denominator > 0 else 0.0

def test_relation_oracle():
    target = np.array([[[[1., 0.], [1., 0.]],
                        [[0., 1.], [1., 0.]]]])
    valid = np.ones((1, 2, 2), dtype=bool)
    quality = np.full((1, 2, 2), 0.1)
    masked = np.zeros((1, 2, 2), dtype=bool)
    masked[0, 0, 0] = True
    pairs = np.array([[0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 1]])
    assert relation_oracle(target, target, valid, quality, masked, pairs) == 0.0
    predicted = target.copy()
    predicted[0, 0, 0] = [0., 1.]
    assert np.isclose(
        relation_oracle(predicted, target, valid, quality, masked, pairs),
        0.95,
    )
    masked[:] = False
    assert relation_oracle(predicted, target, valid, quality, masked, pairs) == 0.0
    assert relation_oracle(predicted, target, valid, quality, masked, []) == 0.0
    errors = np.array([1., 3.])
    weights = np.array([0.1, 0.2])
    assert np.isclose((errors * weights).sum() / weights.sum(), 7.0 / 3.0)

if __name__ == "__main__":
    test_relation_oracle()
~~~

Oracle không thay validation/pair builder và không chứng minh model cải thiện. Nó kiểm một phần algebra được xác định rõ.

### 18.2 Reference PyTorch gradient checks

Agent port thành tests/test_losses.py, chạy trong core environment:

~~~python
def test_reference_gradients_and_empty():
    torch.manual_seed(7)
    pred = torch.randn(1, 4, 2, 3, requires_grad=True)
    target = torch.randn(1, 4, 2, 3, requires_grad=True)
    valid = torch.ones(1, 4, 2, dtype=torch.bool)
    quality = torch.full((1, 4, 2), 0.2)
    masked = torch.ones(1, 4, 2, dtype=torch.bool)
    pairs = torch.tensor([[0, 0, 0, 1, 1], [0, 2, 0, 3, 1]])
    loss, denominator = relation_loss(
        pred, target, valid, quality, masked, pairs
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert denominator > 0
    assert target.grad is None
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0

    pred.grad = None
    empty_loss, denominator = relation_loss(
        pred, target, valid, quality, torch.zeros_like(masked), pairs
    )
    empty_loss.backward()
    assert empty_loss.item() == 0.0
    assert denominator.item() == 0.0
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() == 0.0
~~~

Phải thêm hand/body tests và encoder leakage test; snippet này chỉ là một test tối thiểu, không là full suite.

### 18.3 Integration tests

- Synthetic end-to-end: make cache → train A1/A4 → resume → export → toy eval → aggregate.
- Real tiny subset: decode/perception/teacher/cache → one forward/backward → frozen dev retrieval.
- CPU smoke luôn có thể chạy core nếu PyTorch CPU đã cài; CUDA tests conditional.
- Optional upstream adapter phải có inference shape/output test riêng trước real eval.
- Không viết golden numeric constants của GPU forward phụ thuộc backend; dùng independent invariants/tolerances.

## 19. Runbook mà Agent phải làm cho chạy được

**Tất cả commands ở mục này là giao diện dự kiến của package signrel mới. Agent phải triển khai và smoke-test chúng; không paste vào upstream repo rồi kết luận upstream hỏng.** Các paths là tương đối với implementation repo root.

### 19.1 Setup và core smoke

~~~bash
python -m pip install -e '.[dev]'
python -m signrel doctor --output artifacts/environment.json
python -m signrel make-synthetic --output data/synthetic --seed 17
python -m pytest tests -q
python -m signrel train --config configs/smoke_a1.yaml
python -m signrel train --config configs/smoke_a4.yaml
~~~

configs/smoke_*.yaml dùng synthetic=true, cache dưới data/synthetic, max_updates=20, microbatch=2, accumulation=1, CPU/auto, no external downloads. Synthetic generator tạo own labels cho **toy eval fixture** riêng, không lẫn vào SSL cache.

doctor phải có thể report thiếu dependency thay vì crash không có thông tin. Không dùng smoke config làm full paper config.

### 19.2 Required sources và data

~~~bash
python -m signrel resolve-sources --config configs/sources.yaml --output artifacts/sources.lock.json
python -m signrel import-data --config configs/data_local.yaml --output data/manifests
python -m signrel audit-splits --config configs/data_local.yaml --output artifacts/split_audit.json
python -m signrel extract-cache --config configs/pilot_a4.yaml --limit-videos 20
python -m signrel audit-cache --config configs/pilot_a4.yaml --output artifacts/cache_audit.json
python -m signrel profile --config configs/pilot_a4.yaml --output artifacts/profile.json
~~~

configs/sources.yaml chứa official URLs từ registry và local paths; adapter resolve-sources lấy SHA thực tế, không dùng null lock làm pass.

configs/data_local.yaml chỉ dùng authorized local roots/metadata. ASL Citizen/YouTube-ASL có adapter configs riêng, không đoán dataset schema từ file example trong tài liệu này.

### 19.3 Controlled pilot và resume

~~~bash
python -m signrel train --config configs/pilot_a1.yaml
python -m signrel train --config configs/pilot_a4.yaml
python -m signrel train --config configs/pilot_a4.yaml --resume runs/pilot_a4_seed17/checkpoints/last.pt
~~~

Resume run đã đạt max_updates phải no-op có status=already_complete, không train thêm tự động. Thêm budget/updates cần config version mới ghi trong run history.

Training command tự graceful-stop khi budget hết, save checkpoint và exit status=budget_stopped. Scheduler/update counters không advance nếu optimizer không update.

### 19.4 Export và dev evaluation

~~~bash
python -m signrel export-embeddings --config configs/eval_dev_a1.yaml
python -m signrel export-embeddings --config configs/eval_dev_a4.yaml
python -m signrel eval-retrieval --config configs/eval_dev_a1.yaml
python -m signrel eval-retrieval --config configs/eval_dev_a4.yaml
python -m signrel aggregate --config configs/aggregate_pilot.yaml --output results/pilot_results.csv
~~~

Evaluation config phải có:

~~~yaml
schema_version: 1
stage: dev
encoder_checkpoint: runs/pilot_a4_seed17/checkpoints/last.pt
cache_manifest: data/cache/eval_cache_manifest.jsonl
eval_labels: data/manifests/eval_labels.jsonl
query_ids: data/splits/dev_query_ids.json
gallery_ids: data/splits/frozen_gallery_ids.json
protocol_path: artifacts/protocol.json
output_dir: results/pilot_a4_seed17_dev
pooling:
  window_frames: 64
  stride_frames: 32
  temporal_bins: [1, 2]
  normalize_each_block: true
ranking:
  level: gloss
  exemplar_aggregation: max_cosine
  ks: [1, 5]
audit:
  require_known_query_gallery_signer_disjoint: true
  disallow_exact_duplicate_videos: true
  disallow_overlapping_recording_segments: true
~~~

Nếu signer mapping chưa biết, command phải báo requirement không đạt. Có thể tạo **protocol khác** require=false, nhưng tên report và claim phải đổi; không downgrade điều kiện im lặng.

### 19.5 Locked test / confirmatory matrix

Chỉ tạo configs/test_*.yaml sau protocol/config freeze. Tạo matrix manifest có variant, seed, config hash, sources/cache hashes, planned compute và state. Agent có thể chạy sequential trên một GPU.

~~~bash
python -m signrel aggregate --config configs/aggregate_confirmatory.yaml --output results/confirmatory_results.csv
python -m signrel build-report --config configs/report.yaml --output results/research_report.md
~~~

aggregate không tự khởi chạy thêm 100 experiments. Nó kiểm completeness và tạo table từ artifact tồn tại.

## 20. External baseline adapters: triển khai có kiểm chứng

### 20.1 SignRep adapter

1. Clone [S03], resolve full commit phù hợp release; lưu environment lock riêng.
2. Dùng release v0.0.1/API để enumerate assets. Nếu API/tải bị chặn: explicit BLOCKED_EXTERNAL.
3. Ghi original asset name, URL, hash, license và config thực tế.
4. Đọc example_usage.py và model/transform files được nó import.
5. Load payload với key model nếu đúng binary; strict state matching. Báo missing/unexpected keys; không filter-away weights.
6. Convert RGB thành [B,3,16,224,224] theo transform upstream; kiểm xem Transformation.aug_video có randomness/training augmentation hay không.
7. Nếu cần deterministic eval adaptation, document policy; không gọi exact reproduction khi đổi recipe.
8. Introspect output features và latent shapes. Chọn output cho retrieval bằng dev protocol trước test; report lựa chọn.
9. Handle clips <16 frames và tail theo actual model constraints, ghi coverage; không dùng target SIGNREL length64 để ép input sai.
10. Export embeddings với inference adapter đã test. External dimension/pooling khác được khai riêng.

Không import class SignRep vào encoder chính chỉ để mượn model name. Baseline không được random init nếu weights thiếu.

### 20.2 SHuBERT adapter

1. Clone [S02], pin commit và fork fairseq.
2. Tạo các environments theo environment_*.yml; không cài cùng core nếu conflicts.
3. Tải checkpoints thật từ official folder, resolve tên/IDs/hash. Cần cả upstream DINO hand/face và SHuBERT weights nếu chạy raw-video full pipeline.
4. Đọc QUICKSTART/FEATURES + scripts referenced trước khi sửa path.
5. Giữ upstream hand/face/body features, timestamp/FPS/normalization/dimensions và model config.
6. Thay scheduler wrappers SLURM bằng local invocation nếu cần, lưu patch/call graph.
7. Đừng cấp original generic DINO cache vào pretrained SHuBERT rồi gọi upstream baseline đúng recipe.
8. Chạy tiny upstream feature/inference pipeline; inspect actual contextual output shape.
9. Export feature adapter và frozen retrieval theo protocol common, ghi pooling adaptation nếu khác paper.
10. Downstream fine-tuning repo TODO không được Agent đánh dấu reproduced chỉ vì inference chạy.

SHuBERT feature pretraining corpus/checkpoints khác main controls: báo dataset/init và contamination unknown. Kết quả external chỉ cho context, không thay A1.

### 20.3 SignSeek / SignDino / SALT

Nếu chưa có official public code/weights xác minh được:

- Giữ related-work metadata, không tạo pretrained adapter trả random vectors.
- Own reimplementation chỉ khi user/project chọn branch đó với resource plan; gọi đúng tên reimplementation và ghi missing recipe.
- Không đoán GitHub usernames từ author name.
- Không lấy code của SignDINO khác paper thay cho SignDino [S07].
- Không gọi original DINOv2 frozen teacher là SALT pretrained checkpoint.

## 21. Budget, automation, idempotency và blocker handling

### 21.1 Budget estimation

Agent ghi measured:

- decoded frames/second;
- perception frames/second;
- teacher crops/second;
- cache bytes/hour;
- train seconds/update, peak VRAM;
- export seconds/video và retrieval query cost.

Ước tính total bằng actual retained videos/frames, không chỉ nominal dataset size. Feature extraction/perception cost phải được cộng vào main method cost, dù tất cả A1/A4 reuse cache.

Full-scale plan gợi ý 30–100 giờ và 50,000 successful updates chỉ là **planning template**, không là đủ compute được bảo đảm hay paper default. Phải profile và chọn budget mới trước khi bật full-scale.

### 21.2 Automation logic

~~~text
discover
  -> build_core_and_tests
  -> resolve_required_sources
  -> import_and_audit_data
  -> tiny_cache_and_quality_gate
  -> profile_and_check_budget
  -> paired_pilot_A1_A4
  -> export_and_dev_eval
  -> freeze_configs_if_ready
  -> bounded_confirmatory_matrix_if_budgeted
  -> report_evidence_and_blockers
~~~

Đây là task ordering, không phải một shell script giả đang hoạt động. Agent nên triển khai phase runner với state machine thay vì “run all” không kiểm artifact.

Mỗi phase có dependencies, hashes và status. Nếu artifact đúng fingerprint + pass report tồn tại, reuse. Nếu config thay, invalidate dependent phases theo DAG; không tự xóa dữ liệu upstream.

### 21.3 Stop/retry rules

- Missing GPU: CPU tests/core implementation tiếp tục; real GPU benchmark blocked.
- Missing data: implement adapters/synthetic fixtures; request exact local path/authorized dataset access nếu cần, không tự suy luận file ở nơi khác.
- Missing minimal-pair CSV: retrieval tiếp tục; diagnostic real blocked.
- Missing optional checkpoint: skip external baseline với status, không block main method.
- Required DINO checkpoint không tải được: source-dependent real extraction blocked; core tests vẫn tiếp tục.
- Auth-required public asset: ghi route/URL; không scrape credentials hoặc thay asset không rõ nguồn.
- Two repeated identical failures ở một stage: diagnose/log, đừng infinite retry.
- Budget near cap: graceful-stop và save resume state.
- Dependency fix thay outputs/cache: increment version, rerun affected tests/controls.
- Kết quả không hỗ trợ H1: hoàn thành report negative; implementation không tự đổi research question để giấu kết quả.

### 21.4 Experiment result row schema

~~~json
{
  "run_id": "pilot_a4_seed17",
  "variant": "A4",
  "seed": 17,
  "status": "planned",
  "synthetic": false,
  "successful_updates": 0,
  "checkpoint_sha256": null,
  "config_sha256": null,
  "teacher_sha256": null,
  "cache_manifest_sha256": null,
  "split_fingerprint": null,
  "eval_protocol_sha256": null,
  "macro_gloss_r1": null,
  "micro_r1": null,
  "r5": null,
  "mrr": null,
  "minimal_pair_margin": null,
  "query_coverage": null,
  "train_gpu_hours": null,
  "extraction_gpu_hours": null,
  "notes": "No experiment has been executed in this specification."
}
~~~

Null có nghĩa chưa đo/không có, không đổi thành 0. Planned row không được tính vào mean. Unavailable minimal pairs không có “0 accuracy”.

## 22. Definition of done

### 22.1 Implementation MVP hoàn tất khi

- [ ] Package/CLI có thể install và commands trong RUNBOOK đã chạy smoke.
- [ ] Configs reject unknown keys, label leakage và cache mismatches.
- [ ] Data/cache/target/position/pair contracts ở mục 5–9 được triển khai.
- [ ] State/relation code có independent oracle + gradient/leakage tests.
- [ ] Required upstream assets resolved/load-tested cho real run, hoặc explicit blocker.
- [ ] Synthetic e2e và resume tests pass.
- [ ] Real tiny pipeline có QC nếu video/weights có sẵn.
- [ ] A1/A4 dùng same cache/init/window/mask/pooling.
- [ ] Retrieval ranks gloss đúng, coverage/overlap được audit.
- [ ] Source lock, run manifest, docs và model card có provenance thật.
- [ ] Agent không ghi “trained successfully” khi chỉ tạo checkpoint skeleton.

### 22.2 Research experiments hoàn tất khi

- [ ] Primary protocol/config locked trước test.
- [ ] A4/A1 có 3 seeds hoặc ghi rõ exploratory/incomplete.
- [ ] Ablations/compute controls cần cho claims đã thực sự chạy.
- [ ] Natural minimal-pair claims chỉ dùng annotations verified.
- [ ] All numbers traceable về immutable metrics/checkpoint/source/data versions.
- [ ] External baselines tách supervision/init/data budget.
- [ ] Limitations và negative/confounded outcomes được trình bày.
- [ ] Novelty audit gần thời điểm submission không chỉ dựa vào tài liệu ngày 06/10/2026.

### 22.3 Agent phải bàn giao

1. Code implementation và tests.
2. RUNBOOK với commands đã kiểm chứng.
3. Sources/data/config locks + upstream attribution.
4. Cache/training/inference/evaluation pipeline.
5. Checkpoints/embeddings/metrics **nếu đã có real execution**, kèm hashes.
6. Experiment matrix với passed/failed/blocked/planned rows rõ.
7. Report: điều đã implement, đã chạy, đã đo, chưa chạy và lý do.
8. Next concrete action tương ứng blocker/budget; không chỉ một câu “có thể tiếp tục”.

## 23. Prompt ngắn để khởi động Agent từ file này

> Đọc toàn bộ SIGNREL_IMPLEMENTATION_SPEC.md. Trước hết kiểm tra workspace/AGENTS.md, môi trường và dữ liệu có sẵn, rồi tạo docs/STATE.md. Thực hiện Phase 0 và Phase 1, triển khai actual code và tests trước khi mở rộng dữ liệu. Sau khi core chạy được, resolve/pin/load-test nguồn required từ registry, làm tiny real cache và QC nếu inputs có sẵn. Tiếp tục bounded paired A1/A4 pilot và evaluation trong ngân sách đã cấu hình; nếu chưa có budget cho real training, hoàn thành code/profile và ghi blocker cụ thể. Giữ deterministic controls, label-free SSL input và citation/provenance. Không tự đổi method, loss, teacher hoặc eval protocol mà không ghi config diff/rationale. Các result chưa đo phải là null. Không dừng ở plan hoặc skeleton khi các bước triển khai còn làm được.

## 24. Ghi chú về kiểm chứng tài liệu này

Trong lần soạn đặc tả, đã đối chiếu primary papers, official repository/docs và các entry point nêu rõ trong source registry. Chưa clone/load các model weights, chưa chạy teacher/perception/encoder training hoặc benchmark thật.

Các source SHA, binary SHA và environment versions intentionally chưa được điền giả. Agent triển khai phải resolve chúng trước real experiments.

Reference code và config trong file là thiết kế để implementation, không là một repo đã hoàn thiện. Tình trạng kiểm tra syntax/schema và NumPy oracle được ghi ở phụ lục validation bên dưới.

### 24.1 Validation khi soạn đặc tả

- Python: 4 code blocks đã qua AST parse. NumPy relation oracle đã thực thi và các assertions trong test_relation_oracle() đều pass.
- Config/schema examples: 5 JSON blocks và 2 YAML blocks parse thành công; code fences và 16 source registry IDs được kiểm tra nhất quán.
- Bash: các block lệnh được kiểm tra shell syntax; đây vẫn là CLI contract cần Agent triển khai trước khi chạy thực sự.
- PyTorch không có trong môi trường soạn tài liệu: chưa chạy gradient tests, teacher loader, perception hoặc training. Các bước này thuộc phase triển khai và không được coi là đã pass.
- Không có kết quả thực nghiệm mới về SIGNREL trong file này.

