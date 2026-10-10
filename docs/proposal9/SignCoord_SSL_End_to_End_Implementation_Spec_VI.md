# SignCoord-SSL: End-to-End Implementation Specification for an AI Agent

**Ngôn ngữ:** tiếng Việt; tên API, biến và file bằng tiếng Anh.  
**Phiên bản:** 1.0, ngày 07/10/2026.  
**Mục đích:** giao tài liệu này cho một AI coding/research agent để xây dựng, kiểm thử và vận hành đầy đủ pipeline self-supervised sign language representation.  
**Ngân sách mục tiêu:** 1-4 GPU, chưa biết model GPU/VRAM.  
**Đầu ra nghiên cứu:** RGB-only sequence embeddings và clip embeddings dùng cho retrieval, recognition và translation.  
**Trạng thái:** phương pháp đề xuất; chưa có kết quả huấn luyện SignCoord-SSL trên dữ liệu ký hiệu thật. Các hyperparameter bên dưới là recipe khởi đầu, không phải cấu hình đã chứng minh tối ưu.

Tài liệu này tự chứa những quyết định cần cho implementation. Không yêu cầu Agent phải có hội thoại trước đó. Nguồn `[Rxx]` là paper/dataset; `[Cxx]` là code/documentation. Danh mục đầy đủ nằm cuối file. Phân biệt rõ: **REPORTED** = thông tin từ nguồn, **PROPOSED** = thiết kế của dự án, **TO_VERIFY** = việc Agent phải xác minh trên code/data/hardware thực tế.

## 0. Instructions ưu tiên cho AI Agent

Bạn là implementation owner và research engineer. Hãy hoàn thiện pipeline, không dừng ở skeleton repo, danh sách TODO, hoặc notebook demo chỉ chạy forward.

1. Đọc toàn bộ tài liệu; lập `docs/implementation_plan.md` với task, dependency, acceptance criteria và trạng thái.
2. Đọc repo hiện có cùng `AGENTS.md` nếu có. Tôn trọng codebase; không thay framework/backbone tùy tiện.
3. Triển khai các hợp đồng dữ liệu và kiểm thử số học trước khi chạy training dài.
4. Hoàn thành baseline, dense/velocity controls và lag control trước khi kết luận phase hữu ích.
5. Không tự nhận đã reproduce SignRep nếu đổi targets, sampler, pretrained weights hoặc training budget.
6. Không dùng gloss, class, caption, translation hay metadata signer làm input/objective SSL mặc định. Metadata signer chỉ phục vụ split/audit; supervised downstream được tách riêng.
7. Không bịa checkpoint URL, commit, landmark index, data availability hoặc metric. Resolve các input ngoài môi trường thành cấu hình, kiểm tra và ghi rõ nguồn.
8. Mọi run phải có config đã resolve, manifest hashes, checkpoint hashes, seed, code revision, environment và compute ledger.
9. Nếu thiếu real data hoặc GPU: hoàn thiện phần code độc lập, chạy fixture/synthetic tests, ghi rõ phần nào chưa thực thi. Không điền kết quả synthetic vào benchmark thật.
10. Những lựa chọn chưa có bằng chứng được thực hiện theo default của tài liệu, ghi vào decision log; không coi mọi lựa chọn implementation là lý do phải chờ người dùng.

**Definition of complete** gồm hai tầng:

- **Engineering complete:** CLI/data/model/loss/train/eval/export hoạt động; tests có ý nghĩa; có reproducible smoke run và hướng dẫn chạy real data.
- **Research complete:** chạy đủ matched baselines/ablations trên real data, nhiều seed, kiểm tra leakage và đánh giá frozen transfer. Engineering complete không đồng nghĩa Research complete.

## 1. Bài toán và giả thuyết cần kiểm chứng

### 1.1. Điều được kế thừa

SignRep [R01] học encoder RGB bằng masked reconstruction của các sign priors từ skeleton, có regularization và style objective. Skeleton là tín hiệu pretraining, không bắt buộc khi inference. PGF-SLR [R02] là CSLR có giám sát; cơ chế part-wise/Fourier và forecasting của bài không tự chứng minh tác dụng trong SSL.

Chuyển giao ý tưởng của dự án này là: **RGB encoder học geometry, motion và cross-articulator timing thông qua target giải tích có reliability**, thay vì copy hệ thống CTC của PGF-SLR.

### 1.2. Research questions

- **RQ1:** dense geometric + velocity targets có cải thiện frozen representation so với baseline cùng nguồn khởi tạo không?
- **RQ2:** gated relative-phase loss có thêm giá trị vượt dense + velocity không?
- **RQ3:** phase có vượt temporal lag/correlation control với cùng tín hiệu, context và budget không?
- **RQ4:** gain có giữ được handshape, orientation, holds và nonmanual, và transfer qua signer/dataset không?

Không giả định câu trả lời là có. Nếu lag tốt hơn, chọn lag và thu hẹp claim. Nếu dense/velocity giải thích toàn bộ gain, bỏ claim Fourier.

### 1.3. MVP được chốt

**Student input:** masked RGB, shape `[B,3,T,H,W]`.  
**Training targets:** pose 2D, confidence/validity, normalized geometry và các quan hệ thời gian.  
**Backbone:** Hiera video, cùng checkpoint/adapter giữa các arms được so sánh.  
**Decoder:** part-time query decoder nhẹ, dự đoán tọa độ có timestamp.  
**Loss:** balanced geometry + velocity + lựa chọn lag hoặc gated phase + variance/covariance regularization.  
**Inference:** encoder nhận RGB; bỏ pose extractor, decoder và projector regularization; trả `h_t` và `z`.

MVP không thêm đồng thời learned Fourier teacher, DINO distillation, forecasting, InfoNCE, prototype clustering hoặc appearance teacher. Các extension có milestone riêng ở mục 22.

### 1.4. Những phân biệt bắt buộc

- Temporal FFT khác anatomical Graph Fourier Transform. Eigenmode Laplacian không phải Hz [R24].
- FFT đầy đủ + orthonormal MSE tương đương temporal MSE theo Parseval. Chỉ đổi basis không tạo thông tin mới.
- Amplitude không xác định relative timing; phase không mô tả mọi coordination.
- Target từ pose extractor có prior/supervision của extractor. Gọi setting là label-free sign pretraining với pose pseudo-targets; không tuyên bố không có bất kỳ supervision upstream nào.
- Frozen retrieval, linear probe, full fine-tuning và supervised CSLR là các protocol khác nhau.

## 2. Quyết định triển khai và provenance

| ID | Quyết định | Loại | Cơ sở và kiểm chứng |
|---|---|---|---|
| D01 | Giữ RGB-only inference | PROPOSED, dựa trên SignRep | [R01]; export test không gọi pose pipeline |
| D02 | Dense part-time decoder thay pooled-only reconstruction trong candidate | PROPOSED | So A1/A2, tất cả dense arms cùng decoder/taps |
| D03 | Pose confidence tách khỏi x/y | PROPOSED | Không FFT confidence như coordinate/motion |
| D04 | Hai hệ target: global signing space và local handshape | PROPOSED | Không xóa location/orientation bằng canonical rotation |
| D05 | Phase từ predicted trajectories, không dùng direct phase head trong MVP | PROPOSED | Giữ consistency giữa geometry và relational loss |
| D06 | Gate chỉ tính từ target | PROPOSED | Student không được tự tắt target khó |
| D07 | Lag control cùng anchors/signals/windows | PROPOSED | Fourier/correlation gần nhau; cần matched control |
| D08 | Uniform pooling là primary | PROPOSED | Không cần pose/activity tại inference |
| D09 | T=32 là experimental adapter, T=16 là checkpoint compatibility test | PROPOSED | [C02]; kiểm tra positional embeddings, Unroll/Reroll |
| D10 | Strict inductive data setting là primary | PROPOSED | Pretrain loại downstream dev/test và source overlap |
| D11 | Causal claims không thuộc MVP | PROPOSED | Encoder/decoder được nhìn cả clip; ghi là offline |
| D12 | Không dùng coherence gating mặc định | PROPOSED | K=1 coherence scalar gần 1 không phải confidence [C11] |

Viết mỗi thay đổi so với bảng này vào `docs/decision_log.md`, gồm lý do, experiment IDs và tác động đến attribution. Không âm thầm biến phase proposal thành một phương pháp khác rồi giữ cùng tên run.

## 3. Repo và executable interfaces

### 3.1. Cấu trúc cần tạo

```text
signcoord-ssl/
  pyproject.toml
  requirements.lock
  README.md
  configs/
    base.yaml
    sources.yaml
    pose_schema.yaml
    arms.yaml
    eval_retrieval.yaml
    eval_ctc.yaml
  src/signcoord/
    cli.py
    config.py
    data/{manifest,video_io,sampling,transforms,pose_cache,dataset,collate}.py
    targets/{schema,geometry,reliability,motion,spectral,lag,statistics}.py
    models/{backbone_adapter,hiera_adapter,decoder,projector,representation}.py
    losses/{masked,geometry,motion,phase,lag,regularization,distributed_reduce}.py
    engine/{train,checkpoint,distributed,logging,profile}.py
    eval/{features,retrieval,probes,ctc,translation,bootstrap,slices}.py
    export/{bundle,inference}.py
  tests/
    unit/
    integration/
    fixtures/
  tools/
    make_manifest.py
    audit_overlap.py
    extract_pose.py
    audit_targets.py
    generate_synthetic_fixture.py
    compare_runs.py
  docs/
    implementation_plan.md
    source_lock.json
    decision_log.md
    data_card.md
    reproduction_delta.md
    evaluation_protocol.md
  runs/                      # không commit tensor/video lớn
```

Đây là **interface yêu cầu Agent triển khai**, không phải tuyên bố repo này đã tồn tại. Dùng package `signcoord`, console entrypoint `signcoord = signcoord.cli:main`; mọi command ở mục 19 phải hoạt động sau khi implementation hoàn tất.

### 3.2. CLI commands bắt buộc

| Command | Input chính | Output chính |
|---|---|---|
| `signcoord doctor` | config/environment | dependency, CUDA, inputs và adapter checks |
| `signcoord manifest` | dataset adapter, paths | normalized JSONL + audit |
| `signcoord audit-data` | manifests | overlap/split report, frozen hashes |
| `signcoord extract-pose` | approved manifest, extractor | sharded pose cache + coverage |
| `signcoord fit-target-stats` | SSL train only | stats JSON/NPZ + hash |
| `signcoord audit-targets` | pose/stats/sampler | validity/phase diagnostics |
| `signcoord smoke` | config, fixture/real subset | finite forward/backward/checkpoint |
| `signcoord profile` | config, fixed sample plan | throughput/VRAM/compute ledger |
| `signcoord train` | resolved config + arm + seed | checkpoints, curves, provenance |
| `signcoord extract-features` | frozen encoder, eval manifest | sequence/clip feature cache |
| `signcoord eval-retrieval` | query/gallery caches | per-query results + aggregate |
| `signcoord eval-probe` | frozen features, train/dev labels | head metrics + split provenance |
| `signcoord eval-ctc` | frozen features, gloss train/dev | WER + decoding logs |
| `signcoord eval-translation` | frozen features, text data | metrics + predictions + tokenizer |
| `signcoord compare` | completed run IDs | paired comparison and confidence intervals |
| `signcoord export` | checkpoint + deployment config | encoder bundle + example inference |

Exit code nonzero khi schema/input invalid. Không tự tải dữ liệu ở `train`; data acquisition là bước riêng. Resume không ghi đè experiment khác. JSON logs phải parse được; không chỉ in console.

## 4. Environment và source locking

### 4.1. Dependencies

Đề xuất Python 3.10 hoặc 3.11, PyTorch 2.x tương thích CUDA thực tế; torchvision tương ứng, numpy, scipy, PyYAML hoặc OmegaConf, pytest, video decoder, tqdm và plotting. Dùng `av`/PyAV làm decoder timestamp-aware mặc định; có thể chọn decoder khác sau benchmark, vẫn phải giữ timestamp contract.

Pose extraction chạy trong environment riêng nếu MMPose/MMCV/CUDA xung đột với training. Cache format không phụ thuộc thư viện extractor. Không hard-pin ngẫu nhiên một CUDA wheel; Agent phải chọn bản tương thích driver, kiểm tra tiny GPU backward và lưu lock thực tế.

Tài liệu API dùng PyTorch 2.9 làm mốc có version [C05-C09], **không yêu cầu cài 2.9 bất kể hardware**. Nếu dùng version khác, kiểm tra semantics và ghi vào lock. Không suy API từ documentation `stable` mà không biết version.

### 4.2. Code đã đối chiếu khi viết đặc tả

| Repo | Commit đã đọc | Những file liên quan |
|---|---|---|
| `ryanwongsa/SignRep` | `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2` | `example_usage.py`, final model wrapper, minimal requirements [C01] |
| `facebookresearch/hiera` | `b12b842542ee5c757fcfec8c41f6b56fcbe89b65` | `hiera.py`, `hiera_utils.py`, `hiera_mae.py` [C02] |

Agent có thể pin những commit này hoặc resolve revision mới, nhưng phải ghi diff. Đọc license của code và weights khi lấy checkpoint; đây là phần provenance của asset, không tự gán cùng license cho cả hai.

`docs/source_lock.json` phải chứa `url`, `revision`, `retrieved_at`, `local_sha256`, `role`, `license_ref` cho code/checkpoint/schema. Không dùng `main` làm provenance cuối cùng cho run.

### 4.3. Checkpoint và baseline

Các chế độ khởi tạo phải tách rõ:

- `generic_hiera_mae`: Hiera video MAE K400 checkpoint, chưa fine-tune classifier. Primary để so SignRep-style/dense/coordination cùng khởi tạo nếu tái lập pretraining.
- `signrep_pretrained`: checkpoint SignRep đã huấn luyện. Đây là **continued pretraining/adaptation**; matched control cũng bắt đầu từ cùng checkpoint đó.
- `random_init`: debug/control riêng, không thay thế pretrained baseline.

Không dùng K400 supervised fine-tuned weights rồi gọi đó là pure MAE init. Khi thiếu official SignRep training pipeline/priors/data, đặt baseline là `signrep_style_reimplementation` và ghi `docs/reproduction_delta.md`; không hứa reproduce số công bố.

Official SignRep example là feature extraction example; sự tồn tại của example/checkpoint không bảo đảm mọi phần training, dữ liệu và normalization đã được public [C01]. Agent phải inventory các phần thực sự có trước khi lập reproduction claim.

## 5. Dữ liệu, split và manifest

### 5.1. Chọn nguồn

MVP dùng 50-200 giờ video sign hợp lệ sẵn có để pilot; số giờ là gợi ý ngân sách. Có thể dùng nguồn How2Sign/YouTube-SL-25 hoặc corpus người dùng cung cấp [R31,R35]. Không coi việc paper giới thiệu corpus là bảo đảm toàn bộ media đang tải được.

Primary downstream: ASL-Citizen frozen retrieval [R32], WLASL trên subset thực sự có sẵn [R33]; task sequence bổ sung khi có data. Corpus multilingual cần sampling theo nguồn/ngôn ngữ có ghi nhận; không đoán sign language từ ngôn ngữ caption.

**Strict inductive:** loại khỏi SSL các downstream dev/test videos, intervals, reuploads và signer overlap biết được. Nếu identity SSL không đủ để xác minh signer-disjoint, ghi `signer_disjoint_verified=false`; không biến thiếu metadata thành bằng chứng không overlap. Pretraining có dùng upstream train của downstream là một setting khác và phải khai báo.

### 5.2. Manifest record

Ví dụ schema, các ID/path dưới đây là minh họa cấu trúc:

```json
{
  "schema_version": "video_manifest_v1",
  "sample_id": "corpusA:source001:1000:4000",
  "dataset_id": "corpusA",
  "source_video_id": "source001",
  "source_group_id": "recording_session001",
  "media_path": "data/videos/source001.mp4",
  "media_sha256": "sha256-of-real-file",
  "start_ms": 1000,
  "end_ms": 4000,
  "signer_id": null,
  "sign_language": null,
  "split": "ssl_train",
  "nominal_fps": 25.0,
  "duration_ms": 120000,
  "width": 1280,
  "height": 720,
  "pose_cache_key": "derived-after-extraction",
  "duplicate_group_id": null,
  "label_ref": null,
  "access_status": "available",
  "license_ref": "dataset-access-document"
}
```

Timestamp là thời gian trên video gốc; `[start_ms,end_ms)` half-open. `source_group_id` dùng grouping cho split/bootstrap. `label_ref` downstream trỏ sang bảng labels riêng; `SSLVideoDataset.__getitem__` không trả label/caption/text/signer identity cho model.

Fields thêm: `decode_status`, `pose_status`, `signer_track_id`, `extractor_id`, `manual_review_flag`. Một corpus có nhiều người cần track người ký ổn định; không chọn người có motion lớn nhất trong từng frame vì có thể đổi identity.

### 5.3. Audit overlap

1. Exact match `media_sha256`, source IDs và intersecting intervals.
2. Reupload/near-duplicate: image/video perceptual matching trên keyframes; xác minh candidate, lưu quyết định và threshold.
3. Group theo recording/session trước split; không chia overlapping clips của cùng source qua train/dev/test.
4. So SSL manifest với mọi downstream dev/test. Những label/class chung không tự là leakage; media/source/signers overlap mới phải khai báo theo protocol.
5. Loại lỗi tải/decode bằng manifest có version; lưu counts lớp/signer còn lại. Mọi arm dùng cùng subset.

Report `available/declared` counts; hash từng ordered sample list. Không im lặng thay video hỏng bằng video khác rồi giữ cùng manifest hash.

## 6. Video sampling, timestamp và augmentation

### 6.1. Sampling contract

MVP: `T=32`, spatial `224x224`, sampling grid `25 Hz` khi nguồn cho phép. Đây là sampling grid dự án, không bảo đảm camera FPS là 25. Source PTS/time base được đọc từ decoder.

Với `t_i=t_0+i/f_s`, chọn frame gần nhất trong source, lưu `requested_timestamp_s`, `actual_timestamp_s`, `frame_index`, `frame_valid` và `is_repeat`. Pose/RGB phải đọc cùng source frame hoặc interpolation mapping hợp lệ.

- Khoảng thời gian giữa sample RGB cuối và đầu là `(T-1)/f_s`, không phải chỉ số T tự nó. T=32 tại 25 Hz có span 1.24 s; không tuyên bố dài hơn mọi 16-frame setting vốn có thể stride lớn.
- Derivative dùng timeline đã thống nhất. FFT dùng grid đều; sai lệch actual PTS vượt tolerance hoặc lặp frame nhiều phải báo audit. Không FFT trực tiếp series có khoảng cách mẫu không đều rồi gọi bin là Hz.
- Nếu source FPS thấp, common grid có thể lặp frame. Đặt `max_repeat_fraction=0.10` làm threshold pilot cho spectral eligibility; sensitivity trên dev. Geometry vẫn có thể dùng sample đó với flag rõ.
- Short clip: pad bằng last frame cho tensor, `frame_valid=false` tại padding; không cho padding vào loss/phase/pooling. Không circular-wrap cuối clip về đầu.
- Với continuous source, chọn khoảng t0 ngẫu nhiên trong allowed interval. Với isolated sign, deterministic eval phủ toàn clip, không luôn lấy center rồi bỏ onset/offset.

Nếu common temporal speed augmentation được bật sau MVP, tạo timeline mới và tính lại geometry/velocity/phase theo mapping. Independent hand-time shift chỉ dùng diagnostic corruption; không phải positive augmentation bảo toàn nghĩa.

### 6.2. Spatial transforms

MVP dùng resize giữ aspect ratio + letterbox vào 224x224, cùng affine cho toàn clip. RGB và pose coordinates áp cùng ma trận `A` trong pixel convention. Track signer ROI nếu dữ liệu nhiều người, nhưng ROI phải ổn định theo clip.

Training photometric jitter nhẹ, cùng tham số trong toàn clip để tránh tạo flicker giả. Tắt mặc định horizontal flip, left/right swap, time reversal và crop cắt tay/mặt. Sau này muốn dùng phải có semantic assessment và ablation.

Geometry targets được tính trong **hệ tọa độ đầu vào đã biến đổi**, sau affine; không dùng pose trước crop để supervise trajectory sau crop. Lưu `A`, `inverse_A`, output geometry và visibility thay đổi sau crop.

Không dùng pose để crop/mask student trong primary baseline khi candidate mới có lợi thế này. Nếu dùng pose-guided ROI/masking, tất cả matched arms nhận cùng policy và ghi preprocessing privilege.

### 6.3. Missing data và interpolation

Cache phân biệt: observed, low-confidence, occluded/unobserved, out-of-frame, absent part, padded frame và interpolated.

MVP chỉ linear-interpolate gap tối đa 2 source-grid frames, có hai endpoints đáng tin, không qua tracking discontinuity; reliability interpolation = `0.5 * min(endpoint_confidence)`. Gap dài invalid. Giá trị không hợp lệ được fill finite 0 sau khi lưu validity; không dùng NaN trong computation graph.

Không biến `NaN * 0` thành masked reduction. Với causal extension, interpolation từ endpoint tương lai bị cấm; MVP là offline và phải ghi rõ.

## 7. Pose schema, normalization và targets

### 7.1. Canonical 2D schema của MVP

Chọn COCO-WholeBody 133 như một cache schema được xác minh từ dataset metadata MMPose [C03,C04]. Extractor cụ thể được chọn theo model zoo, benchmark quality/throughput, pin config và weights. Không quy định checkpoint URL chưa xác minh.

Cache: `xy_px[F,133,2]`, `confidence[F,133]`, `observed[F,133]`, `status[F,133]`, `pts_s[F]`. Confidence không là chiều z. Nếu đổi sang MediaPipe/3D, viết adapter riêng; không giữ cùng schema name.

| Raw COCO-WholeBody indices | Ý nghĩa | MVP sử dụng |
|---|---|---|
| 0..16 | COCO body | Giữ torso/body subset dưới đây |
| 17..22 | Foot | Cache, không supervise MVP |
| 23..90 | Face 68 | Face geometry target |
| 91..111 | Left hand 21 | Local/global hand geometry |
| 112..132 | Right hand 21 | Local/global hand geometry |

Body subset, ordered raw indices: `[0,5,6,7,8,9,10,11,12]` = nose, L/R shoulders, L/R elbows, L/R wrists, L/R hips. Đây là **lựa chọn target dự án**, không khẳng định giống danh sách 77 joints của PGF-SLR.

Packed target parts theo thứ tự cố định:

- `L`: raw 91..111, packed 0..20.
- `R`: raw 112..132, packed 21..41.
- `F`: raw 23..90, packed 42..109.
- `B`: body subset trên, packed 110..118.

Tổng `J_target=119`; coordinates 2D. Local finger landmark order phải đọc metadata; không suy từ index của schema khác. Unit test semantic names trước khi extract hàng loạt.

**Coordination anchors:** L = body left_wrist raw 9 / packed 115; R = body right_wrist raw 10 / packed 116; H = body nose raw 0 / packed 110. Dùng body wrists là default có namespace rõ; hand roots 91/112 vẫn có reconstruction riêng. Không fallback đổi anchor qua từng frame nếu chưa có policy audited. Nose trajectory là head proxy, không đại diện đầy đủ facial grammar.

### 7.2. Fixed clip origin và scale

Trong transformed pixel space, với cả hai shoulders hợp lệ:

\[
o=\operatorname{median}_t\tfrac12(x_{Lsh,t}+x_{Rsh,t}),\qquad
s=\operatorname{median}_t\|x_{Lsh,t}-x_{Rsh,t}\|_2.
\]

Median áp theo component cho origin; scale là median scalar. Dùng reliability-thresholded valid frames của clip, tối thiểu 4 trong pilot. `s` phải lớn hơn `0.02 * min(H,W)` trong input pixels; đây là guard heuristic cần audit. Không silently clamp scale nhỏ để tạo target rất lớn.

Global coordinate `G=(x-o)/s`, x sang phải ảnh, y xuống dưới ảnh. Cùng origin/scale cho L,R,F,B. Fixed origin giữ torso translation; per-frame shoulder-centering có thể triệt body trajectory.

Nếu shoulders không đủ: sample vẫn dùng cho independent debug data, nhưng geometry target normalized bị invalid trong MVP. Extension dùng bbox scale phải là normalization version mới và matched ablation, không fallback ngầm.

### 7.3. Local handshape giữ orientation

Với từng tay, hand root là landmark đầu của hand schema đã xác minh:

\[
H_{p,t,j}=(x_{p,t,j}-x_{p,t,root})/s.
\]

Không xoay palm về canonical orientation. MVP derive `H_hat` từ predicted global coordinates, không cần head độc lập. Root local target luôn 0 nên **exclude root** khỏi local loss. Giữ global target của root để học location.

Angle/distance priors là compatibility extension cho SignRep-style reproduction. Nếu thêm, define joint tuples, sin/cos convention, validity từ mọi joints cần thiết và normalization trong config; không tự invent 41/22 angle lists từ paper prose.

### 7.4. Signed relational geometry

Derive từ global predictions và target:

- L-R wrist displacement, signed `[dx,dy]`.
- L-H và R-H signed displacement.
- Shoulder-center và head relation khi đủ validity.

MVP primary geometry gồm global/local; signed-relational loss có flag nhưng mặc định weight 0 để A2/A3 attribution đơn giản. Có thể report relation reconstruction diagnostic ngay cả khi không thêm loss. Khi bật, tạo arm riêng; mọi candidate/lag controls dùng cùng setting.

### 7.5. Reliability và standardization

Binary validity = finite coordinates + observed/interpolation policy + in-frame + frame valid + normalization valid. Confidence là quality score, không khẳng định calibrated probability.

Global scalar reliability `r_tj = validity * clipped_confidence`. Local hand reliability = min(confidence joint, confidence root) nhân tất cả validity. Velocity reliability = min hai endpoints. Anchor pair reliability xét mọi anchors/endpoints cần thiết.

Fit train-only target statistics: robust location/scale hoặc mean/std có clip outlier audit, riêng từng group. Default supervised reconstruction z-score theo **group**, cùng scale x/y và mọi joints trong part, floor 0.05 trong shoulder-width units. Không transform output thành per-joint frame tùy ý.

Quan trọng: decoder output standardized `Y_hat`; invert bằng frozen stats để lấy physical normalized `G_hat` trước derivative/phase/lag. Phase anchors dùng cùng physical coordinates, không lấy các axis/parts với z-score khác nhau rồi dot-product.

Geometry stats, energy thresholds và observed gap statistics chỉ fit SSL train. Dev chọn hyperparameters; test không dùng để fit threshold, standardizer hoặc checkpoint.

## 8. Tensor contracts và adapter boundary

| Tensor | Shape | Dtype/semantics |
|---|---|---|
| `rgb` | `[B,3,T,224,224]` | float, normalized theo checkpoint recipe |
| `times` | `[B,T]` | float64 cache / float32 compute, seconds |
| `frame_valid` | `[B,T]` | bool; padding false |
| `target_global` | `[B,T,119,2]` | float32, finite physical normalized coordinates |
| `joint_valid` | `[B,T,119]` | bool; tách basic validity khỏi confidence score |
| `joint_reliability` | `[B,T,119]` | float32 [0,1] |
| `encoder_memory` | `[B,N_visible,D_e]` | visible final-stage tokens, có coordinates |
| `memory_coords` | `[B,N_visible,3]` | relative time,y,x; không chứa pose |
| `query_hidden` | `[B,T,4,D_d]` | part-time decoder hidden |
| `pred_standardized` | `[B,T,119,2]` | linear heads, không sigmoid/tanh mặc định |
| `anchor_velocity` | `[B,T-1,3,2]` | physical shoulder-width/s; L,R,H |
| `phase_target` | `[B,3,F_keep]` | complex64; pair order LR,LH,RH |
| `phase_gate` | `[B,3,F_keep]` | float32; target-derived, detached |
| `h` | `[B,T_enc,D_e]` | encoder-only sequence, timestamps riêng |
| `z_raw` / `z` | `[B,D_e]` | uniform pool / L2 normalized |

`T_enc` là temporal output của backbone, không tự bằng T. Nếu dense targets ở T, decoder dùng T queries; encoder representation vẫn giữ native temporal resolution. Upsample h cho downstream không tạo thông tin mới và phải ghi rõ.

`SSLBatch` được tách hai phần: `student_inputs` chứa RGB, RGB mask, frame/time metadata; `targets` chứa pose/geometry/reliability. `EncoderAdapter.forward` không nhận `targets`. Unit test monkeypatch target branch unavailable vẫn chạy `encode`.

Packed indices và part/head output dimensions derive từ schema, không duplicate số magic trong nhiều modules. Assert `sum(part_sizes)==119`, canonical names/hashes match cache và stats.

## 9. Backbone adapter và masking

### 9.1. API cần cung cấp

```python
class EncoderAdapter(nn.Module):
    def forward_visible(self, rgb, keep_mask, frame_valid, times):
        # return memory[B,N,D], coords[B,N,3], token_valid[B,N], pooled[B,D]
        ...

    def encode_dense(self, rgb, frame_valid, times):
        # return h[B,T_enc,D], h_times[B,T_enc], h_valid[B,T_enc], z_raw[B,D]
        ...
```

Ellipsis trong đoạn này là signature contract cho Agent hiện thực; kernel ở mục 12 cung cấp phép toán cụ thể. Viết hai backend nếu cần: official SignRep checkpoint compatibility và upstream Hiera experimental adapter. Không trộn state dict tùy tiện giữa hai backend.

### 9.2. Hiera chi tiết cần xử lý

Code upstream đã đọc [C02] có video factory mặc định input 16 frames; mask `True` nghĩa là **keep**. Sau masking, token order/shape khác dense grid; `return_intermediates` trả spatial reroll khi unmasked và retained mask-unit structure khi masked. Khi không mask, output chính có thể đã qua pooled classification head.

Implementation requirements của dự án:

1. Lấy final-stage feature map trước classifier làm encoder features. Không gọi classifier probabilities là representation.
2. Map retained tokens về `(time,y,x)` bằng cùng Unroll/Reroll/index transform như backbone. Không reshape visible token list thành full grid.
3. Tạo synthetic index grid và apply cùng indexing để kiểm thử metadata mapping. Mask IDs của mỗi batch khác nhau; mapping phải dùng đúng keep mask.
4. Khi dense inference, spatial-average final map theo từng temporal cell; normalize bằng encoder feature norm đã chọn. Uniform pool hợp lệ cho `z_raw`; L2 normalization cho `z` chỉ ở evaluation/export, không trước variance loss.
5. Chỉ final-stage taps trong MVP. Nếu thêm earlier taps để giữ detail, tạo setting khác và cho A2/A3/A6/A7 cùng access; include inference cost nếu export sử dụng chúng.
6. Padding frame validity propagate sang encoder temporal cells theo receptive-field policy đã ghi. MVP lấy cell center nằm trong valid interval; không gọi attention trên padded frames là padding-invariant nếu chưa chứng minh. Eval primary ưu tiên full valid windows; short clips report riêng.

### 9.3. Từ 16 sang 32/64 frames

Không chỉ sửa `total_frames` của decoder. Rebuild encoder với input geometry phù hợp: temporal patch stride, mask-unit geometry, positional embeddings, Unroll/Reroll sizes và index metadata phải cùng T.

Upstream factory đã hardcode `input_size=(16,224,224)`; truyền thêm keyword `input_size` có thể gây duplicate argument. Tạo controlled factory dùng Hiera core constructor và architecture config đã pin. Video patch stride ở implementation upstream là `(2,4,4)`, pooling temporal stride là 1; vẫn assert từ instantiated model, không dựa vào tên model.

Load spatial weights không đổi. Với separable temporal positional embedding:

```python
old_t = state_dict["pos_embed_temporal"]       # [1,T_old_tokens,D]
new_t = F.interpolate(
    old_t.transpose(1, 2),
    size=T_new_tokens,
    mode="linear",
    align_corners=False,
).transpose(1, 2)
```

Đây là adaptation policy đề xuất, không phải claim official SignRep recipe. Với combined position embedding, interpolate theo declared 3D grid; không interpolate flattened N như một axis khi spatial order bị trộn.

`load_state_dict(strict=False)` phải đi kèm allowlist missing/unexpected keys. Missing decoder/classifier được phép theo init mode; missing encoder blocks/patch embed hoặc shape mismatch ngoài positional adaptation là lỗi. Lưu load report.

Test T=16 cho compatibility, T=32 cho candidate, T=64 cho context ablation. Mọi matched dense arms cùng positional adaptation, T, physical span và sampling distribution.

### 9.4. Masking policy

Random mask-unit masking mặc định ratio 0.80; đo actual ratio sau integer rounding. Num keep phải bằng nhau giữa samples nếu backbone yêu cầu. Ít nhất 1 valid mask unit giữ lại; config validator reject zero-visible.

Không pose-guided masking trong primary experiment. Nếu mask-unit chứa temporal patch 3-frame receptive field, apply backbone masked-convolution policy; không để pixel từ masked unit tràn vào visible patch qua một patch embed khác mà không audit.

Dense decoder supervise tất cả valid frame targets trong MVP, vì part query không có mapping một-một với RGB patches. Đặt `target_supervision=all_valid`. Không gọi loss này là chỉ tính trên masked joints. Một setting masked-only sau này cần mapping ROI/target rõ và matched control.

## 10. Dense part-time decoder

### 10.1. Kiến trúc mặc định

- D_e lấy từ encoder, ví dụ Hiera-B final features 768 khi architecture xác nhận.
- `D_d=384`, `n_heads=6`, `n_blocks=2`, `MLP_ratio=4`, dropout 0.0.
- Memory projection `Linear(D_e,D_d)`.
- Query cho part p tại time t: learned `part_embedding[p]` + continuous sinusoidal time embedding + learned query base. Không dùng pose coordinates, sample ID, signer ID hoặc class.
- 2 blocks pre-norm: query self-attention, cross-attention vào visible RGB memory, feed-forward. Có memory positional encoding từ RGB token coordinates, không từ target.
- Head mỗi part: `Linear(D_d,2*J_p)`; output ở từng frame query. Ghép đúng L,R,F,B order thành `[B,T,119,2]`.

Không teacher forcing bằng ground-truth previous pose. Không predict reliability để dùng làm gate trong MVP. Confidence head nếu thêm sau này chỉ diagnostic; gate scientific vẫn target-derived.

### 10.2. Shared architecture giữa arms

A2/A3/A5/A6/A7/A8/A9 dùng **cùng decoder params, memory taps, mask và output heads**. Phase/lag tính từ output geometry, không thêm phase-specific decoder capacity. Disable loss bằng config, không thay encoder size cùng lúc.

Regularization projector chỉ train-time; downstream `z` lấy encoder features, không decoder features hoặc pose-informed queries. Decoder h đẹp không bảo đảm encoder h tốt, vì vậy frozen encoder metrics là primary.

### 10.3. Representation interface

```python
result = model.encode(rgb, frame_valid=frame_valid, times=times)
# result = {"sequence": h, "sequence_times": h_times,
#           "sequence_valid": h_valid, "clip_raw": z_raw, "clip": z}
```

`encode` không nhận target/cache path. Eval `.eval()` và `inference_mode`; frozen modules `requires_grad=False`. Training `.train()` không tự bật train-time modules trong exported bundle.

## 11. Geometry, velocity và regularization losses

### 11.1. Safe masked reduction

Coordinate Huber áp elementwise, sau đó mean theo coordinate dimension. Group reliability shape `[B,T,J_group]`.

\[
L_g=\frac{\sum r\,\operatorname{mean}_{d=1,2}\operatorname{Huber}_{\delta}(\hat Y-Y)}{\max(\sum r,\epsilon)}.
\]

MVP `delta=1.0` trong standardized group units. `global_L/global_R/global_F/global_B` có weight 0.25 mỗi group; local_L/local_R có weight 0.5 mỗi group trong local term. Tổng:

\[
L_{geom}=\sum_{p}0.25L_{global,p}+0.5\,L_{local},\quad
L_{local}=0.5L_{local,L}+0.5L_{local,R}.
\]

Đây là loss balancing khởi đầu đề xuất. Số face joints không được tự làm face weight lớn hơn hand. Group hoàn toàn invalid cho loss 0 và report `active=false`; không renormalize giữa groups tùy sample một cách không khai báo.

Targets invalid đã fill finite; cũng dùng `torch.where(valid, error, 0)` trước reduction. Denominator 0 trả differentiable zero từ prediction graph; không trả `float(0)` làm mất backward toàn loss trong edge case.

### 11.2. Velocity

\[
V_t=(G_{t+1}-G_t)/(\tau_{t+1}-\tau_t),\quad t=0,\dots,T-2.
\]

Không prepend zero và gọi zero đó là velocity observed. Output signal dài `T-1`, time là midpoint của interval. `dt<=0` là lỗi; padding/repeat eligibility có policy riêng.

MVP supervise global motion theo bốn groups, reliability `min(r_t,r_t+1)`; normalize velocity bằng train-only scalar scale chung cho mỗi geometry group rồi Huber và balanced average. Phase/lag dùng **physical normalized velocity trước standardization**, từ anchors L,R,H, để common vector coordinates có nghĩa.

MVP `lambda_vel=0.10`. Tune `{0.05,0.10,0.20}` trong dev pilot với cùng điều kiện. Relative λ không có ý nghĩa nếu implementations dùng denominator khác; lock reduction trước tune.

### 11.3. Variance/covariance regularization

Lấy pooled visible encoder feature qua train-only projector `D_e -> 768 -> 256`, GELU, không L2 normalize output. MVP dùng các term variance/covariance lấy cảm hứng VICReg [R25], không gọi đây là toàn bộ VICReg vì không có default two-view invariance term.

Với Z `[N_stats,256]`:

\[
L_{var}=\frac1D\sum_i\max(0,1-\sqrt{\operatorname{Var}(Z_i)+10^{-4}}),
\]
\[
C=\frac{(Z-\bar Z)^\top(Z-\bar Z)}{N_{stats}-1},\qquad
L_{cov}=\frac1D\sum_{i\ne j}C_{ij}^2.
\]

`N_stats>=2`; covariance unbiased denominator thống nhất. Initial `lambda_var=1.0`, `lambda_cov=0.01`, phải monitor weighted gradient norms vì đây chưa phải optimum. Không thay regularizer giữa baseline và candidate.

Statistics batch = local microbatch trên 1 GPU, hoặc differentiable all-gather của **cùng microstep** trên DDP. Gradient accumulation không tự tạo stats batch lớn. Detached queues không tương đương differentiable large-batch stats.

Cross-rank gather dùng autograd-enabled implementation được kiểm thử cho runtime [C08]. Mọi rank gọi collectives cùng thứ tự; không skip gather khi một rank có invalid sample. Test gradients so centralized concatenation để kiểm tra scaling.

### 11.4. Style objective

Primary matched dense setting `style_loss=off` để triển khai nhỏ và kiểm tra attribution. Official SignRep reference vẫn có protocol riêng. Nếu reproduce adversarial style objective, derive đúng target/head/optimization từ paper/code [R01], không đoán style label bằng signer ID.

Giữ style policy cố định giữa arms trong cùng comparison. Tắt chỉ ở baseline hoặc bật chỉ ở candidate làm comparison không còn trả lời RQ2.

### 11.5. Distributed masked reductions

DDP averaging local masked means sai weighting khi valid count khác giữa ranks. Với differentiable local numerator n_r và detached global denominator D:

\[
L_{backward,r}=world\_size\times n_r/\max(D,\epsilon).
\]

DDP average gradients sẽ cho đúng global weighted-mean gradient. `D=all_reduce(sum local denominator)`, numerator all-reduce chỉ cho logging dùng detached tensors. Áp riêng từng group rồi weighted sum.

Với accumulation A: default contract là mean của A **microstep global weighted means**; không tuyên bố bằng single weighted mean trên mọi valid target của effective batch khi denominators khác nhau. Nếu đổi sang exact accumulation weighted denominator, viết implementation/test riêng và match mọi arms.

## 12. Gated relative-phase target và reference kernel

### 12.1. Fixed signal definition

Anchor order `[L,R,H]`, pair order `[(L,R),(L,H),(R,H)]`. Tín hiệu là velocity `[B,T-1,3,2]` trong common normalized camera axes. Không dot-product hand-angle vector với face-landmark vector không tương thích.

MVP T=32 -> velocity length 31; `W=16`, `hop=8`, starts `[0,8]`, **M=2 windows**, không phải 3. Nếu T=64, W=16 -> M=6; W=32/hop16 -> M=2. Config validator tính M từ signal length thực tế.

MVP `window_tail_policy=drop_incomplete`: T32 dùng 24/31 velocity intervals cho phase, 7 intervals cuối vẫn được geometry/velocity supervise. Log spectral coverage. Nếu thêm tail-aligned window, phải sửa cùng window planner cho phase/lag, số windows và calibration version; không tự append rồi giữ config/hash cũ.

Require ít nhất 2 target-valid windows mỗi pair cho phase supervision. Windows overlap không độc lập; M không phải effective independent sample count. W=32 với T=32 không đủ velocity window và phải bị reject, không zero-pad rồi coi là observed.

Window: mean-center từng coordinate trong mỗi window; periodic Hann; `rfft(norm="ortho")` theo time dimension. Giữ non-DC, non-Nyquist bins với W chẵn: `k=1..W/2-1`. Frequencies `f_k=k*f_s/W`; W=16 và fs25 cho 7 bins, spacing 1.5625 Hz. Bins rất thô trên clip ngắn; không suy điều này bao phủ mọi cấu trúc ngôn ngữ.

Không áp frequency-band weighting mặc định; không mặc định bỏ high-frequency vì nghĩ mọi high-frequency là noise. Đánh giá W/context resolution riêng.

### 12.2. Valid windows và estimates

Một pair-window hợp lệ nếu mọi interval của cả hai anchors có reliability `>=0.30`, frame/timing normalization hợp lệ, và không qua discontinuity. `a_j` = mean của `min(r_p,r_q)` trong window, nhân binary eligibility. Threshold 0.30 là pilot policy cần sensitivity, không calibrated confidence.

Với U complex vector, dùng cùng target weights a_j cho estimates target và prediction:

\[
S_{pq}(k)=\frac{\sum_j a_j\sum_d U_{p,j,d}(k)\overline{U_{q,j,d}(k)}}{\max(\sum_j a_j,\epsilon)},
\]
\[
S_{pp}(k)=\frac{\sum_j a_j\sum_d|U_{p,j,d}(k)|^2}{\max(\sum_j a_j,\epsilon)},
\quad q_{pq}=S_{pq}/\max(|S_{pq}|,\epsilon).
\]

**Convention:** `U_p * conj(U_q)`. SciPy `csd(p,q)` thường dùng `conj(U_p) * U_q`, nên khi cross-check phải conjugate để cùng dấu [C10]. Với pure periodic sinusoid và q(t)=p(t-d), phase bằng `omega*d` modulo 2π. Principal phase chỉ dương khi nằm trong (0,π); một delay dương có thể wrap thành phase âm. Unit test khóa công thức modulo, không suy lead/lag chỉ từ dấu principal phase. Một frequency đơn không xác định delay duy nhất.

Không dùng phase angle MSE trực tiếp do wrap ±π. So squared real/imaginary distance. Vector channel cancellation/orthogonal motion có thể khiến S nhỏ; phải gate cross-energy và giữ geometry objectives.

### 12.3. Train-only energy thresholds và gate

Fit train-only thresholds `tau_pp[p,k,W]` và `tau_cross[pair,k,W]`: percentile 10 của **positive finite** spectra trên eligible windows/pairs, floor 1e-6 trong current normalization. Lưu sample counts, quantiles, fps/W/schema/stats hashes. Không fit riêng threshold theo test signer/language.

Nếu quá ít samples để fit một threshold, báo calibration failure cho pair/config hoặc disable pair có ghi nhận; không lấy test để bù. Percentile/floor là default engineering policy, phải sensitivity trên dev.

\[
g=I(M_{valid}\ge2)\cdot\bar a\cdot
\min(1,S_{pp}/\tau_{pp})\cdot
\min(1,S_{qq}/\tau_{qq})\cdot
\min(1,|S_{pq}|/\tau_{cross}).
\]

Gate hoàn toàn detached, target-derived. Require teacher cross-energy >1e-6 cho eligible bin. Static/hold intervals vẫn học geometry/local targets; phase gate=0 không có nghĩa sample bị loại toàn bộ khỏi pretraining.

Coherence-gating optional: chỉ sau averaging và report effective samples/sensitivity. Không dùng K=1 scalar coherence như reliability. Khi bật, tạo A7_coherence riêng, không đổi mặc định.

### 12.4. Loss và numerical stability

\[
L_{phase}=\frac{\sum_{b,pair,k}g\,|\hat q-q|^2}{\max(\sum g,\epsilon)}.
\]

MVP initial λphase=0.05; warmup từ 0 trong 2,000 updates đầu, linear ramp đến full tại update 5,000. Geometry/velocity train trước giúp predicted trajectories không nằm ở trạng thái hằng, nơi cross-spectrum bilinear có thể có gradient kém. Tune 0.02/0.05/0.10 nếu cần.

FFT và complex loss float32/complex64, autocast disabled cho block này [C05,C06]. Dùng denominator floor target-calibrated/fixed, không adaptive student gate. Monitor finite gradients, weighted grad norms và sample coverage; không chỉ loss value.

### 12.5. Reference code cho PyTorch

Đây là kernel tham chiếu **do proposal định nghĩa**, không copy implementation PGF-SLR. Agent hiện thực module production tương đương, thêm checks, DDP reduction, calibration và tests. Code chưa được chạy bằng PyTorch trong lúc viết tài liệu; phải chạy gradcheck/integration trên môi trường implementation.

```python
import torch

PAIRS = ((0, 1), (0, 2), (1, 2))

def interval_velocity(coords, times, reliability):
    # coords [B,T,P,2]; times [B,T]; reliability [B,T,P]
    dt = times[:, 1:] - times[:, :-1]
    if not torch.all(dt > 0):
        raise ValueError("timestamps must increase")
    vel = (coords[:, 1:] - coords[:, :-1]) / dt[..., None, None]
    rel = torch.minimum(reliability[:, 1:], reliability[:, :-1])
    return vel, rel

def window_fft(signal, W=16, hop=8):
    # signal [B,N,P,C]. unfold result [B,M,P,C,W].
    if signal.ndim != 4 or signal.shape[1] < W:
        raise ValueError("insufficient observed signal length")
    x = signal.unfold(1, W, hop)
    x = x - x.mean(dim=-1, keepdim=True)
    win = torch.hann_window(W, periodic=True, device=x.device, dtype=x.dtype)
    return torch.fft.rfft(x * win, dim=-1, norm="ortho")

def pair_window_weights(reliability, W=16, hop=8, min_conf=0.30):
    # reliability [B,N,P]; unfold -> [B,M,P,W]
    r = reliability.unfold(1, W, hop)
    weights = []
    for p, q in PAIRS:
        common = torch.minimum(r[:, :, p], r[:, :, q])
        valid = (common >= min_conf).all(dim=-1)
        weights.append(common.mean(dim=-1) * valid)
    return torch.stack(weights, dim=-1)  # [B,M,Q]

def cross_spectra(U, weights, eps=1e-6):
    # U [B,M,P,C,F], weights [B,M,Q]
    cross, power_p, power_q = [], [], []
    for qi, (p, q) in enumerate(PAIRS):
        a = weights[:, :, qi]
        den = a.sum(dim=1).clamp_min(eps)[:, None]
        Spq = (U[:, :, p] * U[:, :, q].conj()).sum(dim=2)
        Spp = U[:, :, p].abs().square().sum(dim=2)
        Sqq = U[:, :, q].abs().square().sum(dim=2)
        cross.append((Spq * a[..., None]).sum(dim=1) / den)
        power_p.append((Spp * a[..., None]).sum(dim=1) / den)
        power_q.append((Sqq * a[..., None]).sum(dim=1) / den)
    return (torch.stack(cross, dim=1),
            torch.stack(power_p, dim=1),
            torch.stack(power_q, dim=1))

def phase_loss_terms(pred_signal, target_signal, target_rel,
                     tau_p, tau_q, tau_cross, W=16, hop=8, eps=1e-6):
    # tau_* broadcast to [B,Q,F_keep], positive train-calibrated tensors.
    # timing/padding eligibility has already zeroed target_rel.
    with torch.autocast(device_type=pred_signal.device.type, enabled=False):
        # Preserve float64 only for numerical/gradcheck debug tests.
        math_dtype = torch.float64 if pred_signal.dtype == torch.float64 else torch.float32
        pred = pred_signal.to(math_dtype)
        target = target_signal.detach().to(math_dtype)
        rel = target_rel.detach().to(math_dtype)
        if W % 2 != 0:
            raise ValueError("MVP uses even W")
        weights = pair_window_weights(rel, W, hop)
        if weights.shape[1] < 2:
            raise ValueError("need at least two planned windows")
        keep = slice(1, W // 2)  # omit DC and even-W Nyquist
        with torch.no_grad():
            U_t = window_fft(target, W, hop)
            S_t, P_t, Q_t = cross_spectra(U_t, weights, eps)
            S_t, P_t, Q_t = S_t[..., keep], P_t[..., keep], Q_t[..., keep]
            q_t = S_t / S_t.abs().clamp_min(eps)
            count = (weights > 0).sum(dim=1)                  # [B,Q]
            mean_rel = weights.sum(dim=1) / count.clamp_min(1)
            eligible = (count >= 2)[..., None] & (S_t.abs() > eps)
            gate = eligible * mean_rel[..., None]
            gate = gate * (P_t / tau_p).clamp(0, 1)
            gate = gate * (Q_t / tau_q).clamp(0, 1)
            gate = gate * (S_t.abs() / tau_cross).clamp(0, 1)
        U_s = window_fft(pred, W, hop)
        S_s, _, _ = cross_spectra(U_s, weights, eps)
        S_s = S_s[..., keep]
        q_s = S_s / S_s.abs().clamp_min(eps)
        err = (q_s.real - q_t.real).square() + (q_s.imag - q_t.imag).square()
        numerator = (gate * err).sum()
        denominator = gate.sum()
        # Production caller uses global weighted reduction on DDP.
        return numerator, denominator, {"gate": gate, "q_target": q_t,
                                        "valid_window_count": count}
```

Production module không giữ diagnostic tensors có graph trong logger. Timing eligibility/threshold fit phải hiện thực đầy đủ, không hiểu kernel là toàn bộ pipeline. Khi toàn gate=0, local numerator vẫn liên kết prediction; reduction trả finite differentiable zero.

## 13. Temporal lag baseline và magnitude/Parseval controls

### 13.1. Lag objective

Dùng cùng physical anchor velocities, cùng detrend/windows và target window weights. Lag set `ell=-4..4`, không circular shift. Với ell>0, compare `p[t]` với `q[t+ell]`; nếu q delayed, correlation peak nằm ở ell>0. Boundaries chỉ lấy overlapping observed intervals.

\[
R^{(j)}_{pq}(\ell)=
\frac{\sum_t\langle u_{p,j}(t),u_{q,j}(t+\ell)\rangle}
{\sqrt{\sum_t\|u_{p,j}(t)\|^2\sum_t\|u_{q,j}(t+\ell)\|^2}+\epsilon}.
\]

Aggregate theo a_j, require >=2 eligible windows, target energy-gated. Compute predicted correlation từ predicted trajectories với cùng target weights. Huber hoặc squared loss trên R, lock choice trước run. Không predict lag bằng head mới trong control chính, vì phase arm không có extra head.

Lag là matched information/control, không mathematically identical objective với normalized phase khi lag truncation/window/normalization khác. Report các khác biệt; không tuyên bố hai objectives độc lập hoàn toàn vì correlation và cross-spectrum liên hệ Fourier.

### 13.2. Magnitude control

A5 dùng magnitude/auto-spectrum targets từ cùng anchor velocities; normalized log power với floor và train scale đã chốt. Không âm thầm dùng phase-cross energy cả trong objective và claim magnitude-only. Gate reliability/auto-energy, no relative phase.

Tune λ theo predefined small grid/gradient balance giống candidate; một λ bất lợi không phải control mạnh. Report waveform/coordination diagnostics và frozen metrics.

### 13.3. Parseval sanity control

Với full complex FFT `norm="ortho"`, squared residual energy phải bằng temporal squared energy. Dùng double precision CPU reference, relative tolerance khoảng 1e-10; float32 khoảng 1e-5 tùy scale.

Nếu dùng rFFT, weight DC=1, Nyquist=1 cho W chẵn, remaining bins=2; mẫu odd không có Nyquist riêng. Không so naive one-sided `.mean()` với temporal MSE rồi tuyên bố FFT loss khác information.

A4 là equivalence test và optional replacing-control: cùng MSE weights, chỉ thay computational domain; không thêm một bản copy loss vào A3 rồi tính gain là contribution. Không cần tốn full training run để xác nhận đẳng thức đã có numerical test, trừ khi pipeline masking/reduction làm sai equivalence.

## 14. Tổng loss và ablation registry

\[
L=L_{geom}+\lambda_vL_{vel}+\lambda_cL_{relation}
+\lambda_{var}L_{var}+\lambda_{cov}L_{cov}.
\]

`L_relation` là phase **hoặc** lag trong MVP, không cả hai mặc định. Warmup λrelation như mục 12.4. Shared geometry/local/regularizer policies cố định.

| Arm | Setting | Câu hỏi | Priority |
|---|---|---|---|
| A0 | Generic Hiera frozen, no sign SSL | Starting representation | Required |
| A1_ref | Official SignRep checkpoint/protocol | External reference, không matched continued training | Required nếu checkpoint có |
| A1 | SignRep-style pooled prior reconstruction | Baseline cùng init/data; label reproduction delta | Required |
| A2 | Dense geometry decoder | Dense targets/taps giải thích gain? | Required |
| A3 | A2 + velocity | Motion explains gain? | Required |
| A4 | Full-FFT MSE equivalence | Implementation sanity | Required test |
| A5 | A3 + magnitude objective | Amplitude sufficient? | Pilot control |
| A6 | A3 + gated lag | Timing without spectral objective | Required |
| A7 | A3 + gated relative phase | Main candidate | Required |
| A8 | A7 với no energy/reliability quality gate; vẫn giữ basic validity | Quality gate có ích? | Required confirmatory |
| A9 | A7 với shuffled pairing target | Gain do real coordination? | Required confirmatory |

**A9 definition:** dùng deterministic seed để permute target relational signal/sample trong batch, không self-pair; decoder geometry/velocity targets vẫn đúng. Gate phải gắn với shuffled relational target tương ứng; ghi permutation strategy. Không dùng test để chọn shuffle khó. Đây là negative control, có thể làm optimization khó; interpret cùng valid count/gradient norms, không chỉ score thấp.

**A8 exact policy:** giữ finite coordinates, observed/padding/timing/normalization validity và `|S_target|>eps` để phase được định nghĩa; dùng pair-window weights = 1 cho basic-valid windows, không min-confidence threshold hoặc confidence weighting, không tau-based soft energy gate. Require >=2 basic-valid windows. Đây là bỏ **quality gating**, không supervise undefined phase của zero signal. Cần `joint_valid` riêng; không thể khôi phục basic validity chỉ từ confidence đã nhân mask.

**A9 compatibility:** shuffle trong nhóm có cùng fs/W/timing và basic-valid support, ưu tiên full-valid clips; nếu batch không đủ để derange, skip relational negative-control term có synchronized/logged policy. Không lấy target valid window của sample khác để supervise padded prediction. Log how many pairs thực sự bị đổi và eligible denominator; so coverage với A7.

A1_ref vs candidate khác pretraining history nên report external reference riêng. `signrep_pretrained` mode cần A2/A3/A6/A7 tiếp tục từ cùng checkpoint và cùng budget; không so tiếp tục 100k steps với baseline không được tiếp tục rồi claim objective win.

Nếu exact SignRep pooled priors chưa có đủ code/schema, triển khai A1-inspired cùng 2D target schema làm additional control; nhãn phải thể hiện inspired. Paper-faithful reproduction và shared-schema mechanism study là hai comparison khác nhau.

## 15. Training engine, DDP, checkpoints và compute budget

### 15.1. Training sequence

1. Validate config, hashes, splits, schema, stats/calibration compatibility.
2. Instantiate backbone/decoder/projector; load init with allowlist report.
3. Seed Python/numpy/torch/CUDA/DataLoader workers; distributed sampler `set_epoch` đúng. Lưu sample-plan seed và mask seed, dùng paired plans cho matched arms.
4. Build optimizer theo named groups; exclude bias/norm/positional embeddings khỏi weight decay theo policy đã chốt.
5. Sample RGB + pose trên cùng timeline, synchronous spatial transform, build finite target tensors/reliability.
6. Sample keep mask; student forward chỉ đọc RGB/mask/timing metadata.
7. Decode standardized coords, invert target stats, derive local/velocity/relations.
8. Geometry/velocity/regularizer + scheduled relation loss. All-gather/reduce theo contract, FP32 spectral block.
9. Backward/accumulate, unscale nếu fp16, clip grad norm, optimizer step, scheduler step **theo successful optimizer update**.
10. Log metrics/coverage/compute; save atomic checkpoint; evaluate frozen dev theo cadence định trước.

### 15.2. Optimizer recipe khởi đầu

AdamW: LR 1e-4 encoder và 3e-4 new decoder/projector, betas (0.9,0.95), weight decay 0.05; linear warmup 1,000 optimizer updates và cosine decay. Nếu reproduce original recipe có nguồn rõ, dùng original trong setting riêng; không gọi những con số này là SignRep published defaults.

Clip grad norm 1.0; log pre-clip norm và fraction clipped. AMP BF16 nếu hardware hỗ trợ, FP16+GradScaler nếu cần, FP32 CPU fixtures. Mọi arm cùng optimizer/batch/precision. Layer-wise LR decay optional arm riêng, không silently bật chỉ cho candidate.

Pilot 10k-20k updates, full confirmatory 50k-100k updates hoặc exposure budget chốt trước; chưa phải paper-faithful 500k reproduction. Checkpoint eval cadence 2,000 updates, save cadence 2,000; select primary checkpoint bằng dev endpoint đã predeclare.

### 15.3. Batch và accumulation

Khởi đầu microbatch 4/GPU, accumulation 4 -> effective optimizer batch `4 * world_size * 4`. Stats batch chỉ `4 * world_size` nếu gather mỗi microstep. Những giá trị này phụ thuộc VRAM; `doctor/profile` quyết định feasibility.

Nếu OOM: giảm microbatch trước, tăng accumulation để giữ optimizer batch; **stats batch sẽ đổi** nên record và match baseline. Gradient checkpointing nếu model hỗ trợ có kiểm thử; không tự giảm resolution/T chỉ ở candidate. Muốn đổi context hoặc freeze encoder phải là setting riêng.

Có thể nâng microbatch trên 1 GPU cho var/cov ổn định hoặc all-gather trên 4 GPU. Không report “batch 64 for VICReg” khi optimizer accumulation 64 nhưng actual stats batch 4.

### 15.4. DDP correctness

- Một process/GPU; NCCL CUDA, Gloo CPU fixtures [C07].
- All ranks cùng iteration count/collective order, `drop_last=True` cho training nếu cần same gather shape.
- Microsteps chưa optimizer step dùng `no_sync` cho DDP parameter gradient khi thích hợp; điều đó không bỏ differentiable stats collectives.
- Global masked denominator theo mục 11.5; không dùng `all_reduce` in-place trên tensor có graph mà chưa hiểu autograd.
- Không rank-local `continue` khi loss invalid. Đồng bộ failure flag, dump diagnostic và kết thúc/resume rõ.
- Unused-head handling phải kiểm tra arm registry; default all geometry heads active. DDP `find_unused_parameters` có thể bật debug, sau audit optimize.
- Log split seed, rank RNG/sampler state. Resume đúng world size nếu muốn claim exact continuation; đổi world size là run có note riêng.

### 15.5. Checkpoint contents

```text
schema_version
model_state / optimizer_state / scheduler_state / scaler_state
optimizer_update / microstep_boundary / epoch / sampler_state
rng_state_python_numpy_torch_cuda_per_rank
resolved_config / config_hash / code_revision
initialization_checkpoint_hash / load_report
ssl_manifest_hash / eval_manifest_hashes
pose_schema_hash / extractor_weights_hash / target_stats_hash
phase_calibration_hash / pooling_protocol_hash
precision / world_size / batch_contract
best_dev_metric / selection_rule / compute_ledger
```

Atomic write temp file -> fsync/rename trong cùng filesystem. Save tại optimizer boundary để không cần giữ partially accumulated grads. Nếu stream loader không restore exact position, record replay/skip semantics; không claim bitwise resume.

`last` phục vụ resume, `best_dev` phục vụ selected eval, `final` phục vụ fixed-budget comparison. Giữ cả fixed-step và dev-selected metrics để không giấu checkpoint selection advantage.

### 15.6. Logging và diagnostics

Mỗi update: total/unweighted/weighted losses, λschedule, LR, grad norm, valid denominators theo group, phase active sample/pair/bin counts, signal-energy quantiles, average gate, input repeat fraction, normalized-scale failures.

Mỗi eval/profile interval: feature per-dimension std/effective rank, mean cosine between random clips, projector stats, target/predicted trajectory examples, phase circular error trên eligible bins, frozen dev retrieval, GPU memory/time, data loading time.

Không log raw labeled test samples trong pretrain tuning dashboard. Visual diagnostics lấy SSL train/dev được phép theo protocol; report source IDs có khả năng đối chiếu.

### 15.7. Compute ledger

Measure 500-1,000 stable updates sau warmup, không lấy first iteration compilation/loading. Ghi model/driver, VRAM, batch, T, H/W, precision, params, elapsed CUDA-synchronized time, clips/s, peak allocated/reserved memory.

Track riêng pose extraction hours, CPU decoding, cache size, training GPU-hours, inference cost. Extrapolate runtime với uncertainty; không hứa một số ngày tuyệt đối từ “1-4 GPU”. Candidate budget tolerance +25% so A3 là **target design** cần đo, không kết quả đã đạt.

## 16. Config chuẩn và validation

### 16.1. `configs/base.yaml`

```yaml
schema_version: signcoord_config_v1
project: signcoord_ssl
arm: A7
seed: 3407
run_dir: runs/A7_seed3407

data:
  ssl_train_manifest: null
  ssl_dev_manifest: null
  pose_cache_root: null
  target_stats_path: null
  phase_calibration_path: null
  manifest_schema: video_manifest_v1
  pose_schema: coco_wholebody133_to_signcoord119_v1
  strict_inductive: true
  clips_per_source_cap: 200
  sampler:
    frames: 32
    grid_fps: 25.0
    spatial_size: [224, 224]
    resize: letterbox
    max_repeat_fraction_spectral: 0.10
    short_clip: repeat_last_with_invalid_padding
    temporal_speed_aug: false
  augmentation:
    color_jitter: 0.10
    same_photometric_params_over_time: true
    horizontal_flip: false
    time_reverse: false
    independent_part_time_shift: false
    pose_guided_masking: false
  normalization:
    origin: clip_median_shoulder_center
    scale: clip_median_shoulder_width
    min_valid_shoulder_frames: 4
    min_scale_fraction: 0.02
    gap_interpolation_max_frames: 2
    interpolated_reliability_factor: 0.5

model:
  backbone: hiera_base_video_adapter
  initialization_mode: generic_hiera_mae
  initialization_checkpoint: null
  source_lock_path: docs/source_lock.json
  input_frames: 32
  temporal_position_adaptation: linear_separable
  encoder_taps: [final]
  mask_ratio: 0.80
  mask_semantics: true_means_keep
  decoder:
    width: 384
    blocks: 2
    heads: 6
    mlp_ratio: 4
    dropout: 0.0
    part_order: [L, R, F, B]
    target_supervision: all_valid
  regularization_projector:
    hidden: 768
    out_dim: 256

loss:
  geometry:
    global_part_weights: {L: 0.25, R: 0.25, F: 0.25, B: 0.25}
    local_total_weight: 0.5
    local_hand_weights: {L: 0.5, R: 0.5}
    huber_delta: 1.0
  velocity_weight: 0.10
  signed_relation_weight: 0.0
  relation:
    kind: phase
    weight: 0.05
    warmup_zero_updates: 2000
    ramp_end_update: 5000
    anchors_packed: {L: 115, R: 116, H: 110}
    pairs: [[L, R], [L, H], [R, H]]
    window: 16
    hop: 8
    window_tail_policy: drop_incomplete
    min_valid_windows: 2
    min_confidence: 0.30
    gate: target_validity_auto_and_cross_energy
    coherence_gate: false
    calibration_percentile: 10
    energy_floor: 1.0e-6
    fft_norm: ortho
    hann_periodic: true
    remove_window_mean: true
    exclude_dc: true
    exclude_nyquist: true
    lag_min: -4
    lag_max: 4
  variance_weight: 1.0
  covariance_weight: 0.01
  variance_target_std: 1.0
  stats_batch: differentiable_gather_per_microstep
  style_policy: "off"
  masked_ddp_reduction: global_weighted_mean_per_microstep

train:
  device: cuda
  precision: bf16_if_supported_else_fp16
  microbatch_per_gpu: 4
  accumulation_steps: 4
  optimizer: adamw
  encoder_lr: 1.0e-4
  new_modules_lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.05
  warmup_updates: 1000
  scheduler: cosine
  max_updates: 20000
  grad_clip_norm: 1.0
  eval_every_updates: 2000
  save_every_updates: 2000
  save_at_optimizer_boundary: true
  dataloader_workers: 4
  paired_sample_plan: true

evaluation:
  primary_dataset: asl_citizen
  selection_split: dev
  selection_metric: retrieval_rec1
  pooling: uniform_timestamp_coverage
  gallery_split: train
  query_split: dev
  prototype_gallery: false
  test_evaluation: final_only

export:
  rgb_only: true
  include_decoder: false
  include_projector: false
  include_pose_extractor: false
  sequence_and_clip: true
```

`null` ở input paths là **required runtime input**, không phải placeholder Agent được bỏ qua. `doctor` phải chỉ rõ missing input và refuse real training; fixture smoke dùng config fixture riêng có paths thật. Không fill bằng guessed asset URL.

Yaml parser phải disable/handle YAML 1.1 boolean ambiguity; serialize `style_policy` thành string `"off"` nếu parser coi `off` là false. Config implementation dùng typed dataclass/Pydantic hoặc schema validation, unknown keys là lỗi.

### 16.2. `configs/arms.yaml` semantics

Arm override không đổi data/backbone/optimizer/pooling. A2 velocity=0, relation kind=none; A3 velocity=0.10, relation=none; A6 relation=lag; A7 relation=phase; A8 quality gate off nhưng basic validity vẫn bắt buộc; A9 pair shuffle true. Required unused relation inputs được nới theo arm, ví dụ A2 không cần phase calibration file.

Override `arm=A6` phải thực sự switch criterion, không chỉ đổi tên folder. Save resolved config cùng diff so base. Hash scientific config, không include timestamp/log path vào hash dùng matched check.

### 16.3. Config validator phải reject

- Label/caption input bật trong SSL primary.
- `input_frames != sampler.frames`, decoder width không chia hết heads.
- W > T-1, M < min_valid_windows, odd W trong MVP.
- `grid_fps<=0`, hop<=0, mask ratio ngoài range hoặc zero-visible.
- Calibration/schema/stats hash không khớp current config.
- Infer `rgb_only=true` nhưng activity weights yêu cầu pose runtime.
- Gallery/query sai split hoặc selected checkpoint dựa test.
- Invalid energy floor/negative loss weight.
- Claim exact resume khi source manifest/code/world-size bị đổi.

## 17. Frozen evaluation end-to-end

### 17.1. Feature extraction

Mỗi checkpoint eval `.eval()`, weights frozen, no augmentation ngẫu nhiên. ASL-Citizen/WLASL dùng deterministic overlapping windows trên toàn clip, stride half clip span; thêm tail window để không bỏ offset. Mỗi window có times/validity và native `h_times`.

Overlapping temporal cells không được tự đếm nhiều lần ở clip pooling. Splat features theo timestamp vào common bins, average features tại bin từ các windows phủ nó; sau đó uniform average các observed bins. Tolerance/stride/bin policy pin trong `pooling_protocol.json` và apply identically mọi model. Cuối cùng L2 normalize z với eps; zero norm là error được report.

Một comparison với official SignRep pooling phải chạy protocol adapter riêng [R01,C01]. Primary custom uniform pooling không được đặt cạnh published activity-weighted number rồi claim hơn nếu không match.

Cache features: sample/source IDs, sequence timestamps/validity, `z_raw/z`, checkpoint hash, preprocessing hash, manifest hash, pooling hash. Dedupe các samples lặp do distributed extraction padding. Không cache feature của mutable checkpoint dưới cùng key.

### 17.2. Retrieval protocol

Primary gallery = official downstream train clips; query dev cho tuning, test cho final. Không gallery dev/test, không self-match/duplicate/reupload source. Labels chỉ để xác định relevance trong evaluation, không được đọc khi extract encoder features.

Primary instance retrieval cosine similarity trên normalized embeddings, sort deterministic theo `(score descending, gallery_id)` khi ties. Chunk matrix để không OOM; results phải bằng dense tiny reference.

Rec@K = query có ít nhất một gallery instance cùng class trong top K. MRR = reciprocal rank của first relevant gallery. Report micro và macro theo class/signer, Rec@1/5/10, number queries, gallery size, classes with no gallery match. Missing-class queries giữ trong primary nếu official protocol yêu cầu; report open/closed-set split rõ, không loại âm thầm.

Class-prototype retrieval là secondary: prototype = mean train class features rồi L2 normalize; không trộn ranking instances với ranking classes. ASL-Citizen official signer-independent split dùng như supplied [R32]; strict SSL signer separation vẫn cần audit upstream.

Bootstrap per-signer/source group trên paired query results; nhiều seeds report mean±SD. Nếu ít signers, CI có hạn chế phải ghi; không coi thousands overlapping clips là thousands independent people.

### 17.3. Linear/kNN probes và label efficiency

Frozen z/h chỉ; train linear logistic head trên downstream train labels, choose regularization trên dev. Không tune encoder hoặc BatchNorm stats trên eval. kNN k/similarity chọn dev. Label fractions 1/5/10/100% theo class-aware seeded subsets, same subsets mọi encoder.

Fine-tuning full encoder là secondary protocol có optimizer riêng; report tách frozen. Class-distribution-based inference/postprocessing dùng labels cần tách và match; không tính vào SSL representation gain.

### 17.4. Linguistic và robustness slices

Nếu có verified annotations: handshape, orientation, location, movement, nonmanual và lexical mapping [R36,R37]. ASL-LEX mapping theo verified lexeme/variant identity; không join gloss string tùy tiện, vì homonym/variant và numbering khác.

Không có nonmanual ground truth thì dùng NMFs-CSL hoặc annotations hợp lệ khi available [R34], hoặc đánh dấu chưa đánh giá; pose confidence không là nhãn linguistic.

Slice tối thiểu: low/high motion, holds, bilateral coordination, one-hand, occlusion/pose quality, short clips, repeat-frame rate, signer/source/language. Bins thresholds fit train/dev; report sample counts. Motion-derived slices chỉ diagnostic, không bằng chứng semantic category.

Perturbations không dùng để kết luận bảo toàn ngữ nghĩa: reverse, hand swap, independent temporal shifts. Dùng đo sensitivity và kiểm tra target behavior; linguistic minimal pairs cần người/annotation xác minh.

### 17.5. Statistical comparison và checkpoint policy

Predeclare primary endpoint ASL-Citizen frozen Rec@1, secondary MRR và independent-dataset endpoint. Tune candidate trên dev cùng tuning budget cho A3/A6. Test chỉ final evaluation theo frozen protocol.

Pilot chỉ screening; confirmatory ≥3 seeds, identical exposure budgets. Paired difference A7-A3 và A7-A6; CI từ paired signer/source bootstrap, seed variation report riêng. Nếu combine seed/bootstrap, viết hierarchical procedure rõ; không gộp run best seed thành score chính.

Nếu +1 điểm phần trăm là practical threshold, chốt trước confirmatory và ghi đó là design threshold. Không claim statistically significant chỉ vì best seed hơn 0.1 điểm. Một data leakage report tốt không sửa được tuning test đã xảy ra.

## 18. Sequence transfer, translation và export

### 18.1. Frozen CSLR/CTC

Cache offline sequence h từ frozen RGB encoder; train cùng lightweight temporal head cho mọi arms, ví dụ 2-layer temporal Conv1D/Transformer + linear vocabulary head. Đây là supervised downstream, không gọi pretraining CTC-free là label-free toàn pipeline.

CTC: logits `[T_out,B,V]`, `log_softmax` float32, input lengths đúng observed output resolution, targets không chứa blank; blank ID pin [C09]. Với repeated consecutive glosses, feasibility yêu cầu input length đủ label length cộng adjacent repeats. Không dùng `zero_infinity=True` để âm thầm che toàn bộ invalid alignments; log invalid count/rate và repair sampling/head resolution.

Greedy decoding collapse repeats rồi remove blank đúng thứ tự; unit fixtures. Beam decoding/LM nếu dùng là secondary cùng weights/lexicon cho mọi arms. WER=(substitution+deletion+insertion)/reference_tokens; report S/D/I, text normalization và dev checkpoint selection.

Offline whole-clip context có thể nhìn future; không report online latency/causal recognition. Head BiLSTM hoặc encoder bidirectional càng không causal.

### 18.2. Frozen SLT transfer

Implement feature-to-text head cùng architecture/params/tokenizer/training budget trên all selected encoder arms. Cached h frozen; alignment/projection trainable. Có thể dùng small seq2seq Transformer để budget feasible; pre-trained LM là separate disclosed setting vì upstream text supervision khác.

Language/tokenizer vocab fit train; unknown handling, padding, BOS/EOS thống nhất. Eval dev checkpoint bằng preregistered metric; output predictions/references per sample. BLEU implementation/signature và chrF tokenizer options pin; không lấy một BLEU number làm bằng chứng linguistic competence. PHOENIX audit [R38] cần được đọc khi dùng benchmark này.

Translation optional execution nếu data/budget không có, nhưng module/CLI/tests phải hoàn thiện cho engineering scope. Không báo metric placeholder. Nếu chỉ có ASL-Citizen, kết luận multi-task phải ghi sequence transfer chưa thực thi.

### 18.3. Encoder export

Bundle gồm encoder weights, architecture/config, RGB normalization, temporal/spatial sampler, feature-layer identity, pooling policy, input/output schema, provenance và runnable `infer.py`.

Không include pose/decoder/projector trong default deployment dependency. Import/export test trong environment thiếu pose library phải pass. Compare exported output với training `encode` ở same RGB, tolerance theo precision.

TorchScript/ONNX optional chỉ khi backend hỗ trợ; không đổi output semantics để export rồi giữ cùng metric label. Primary deliverable có thể là native PyTorch bundle kèm dependency lock.

Inference profiling gồm decoding/preprocessing/encoder/pooling và riêng kernel time; không chỉ đo cached features rồi gọi end-to-end FPS. Model card nêu offline temporal context, nguồn pretraining, pose pseudo-targets, limitations và measured performance.

## 19. Runbook executable từ repo trống tới real experiments

Mọi command dưới đây là **contract cho CLI cần được implement**. Agent phải chạy và lưu output thực tế; tài liệu này không khẳng định các commands đã tồn tại hoặc được thực thi. CLI global options: `--config PATH`, repeatable `--set dotted.key=value` với typed parsing, `--arm`, `--seed`; reject unknown keys.

### 19.1. Bootstrap

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
signcoord doctor --config configs/base.yaml
pytest -q tests/unit
```

`doctor` với base chưa điền data/checkpoint có thể report inputs missing và nonzero; environment tests vẫn chạy được. `pip install` phải theo lock đã chọn; bootstrap không tự đổi dependency trong repo người dùng.

### 19.2. Synthetic fixture end-to-end

Agent viết generator tạo:

- 32-64 short videos nhỏ với known RGB/pose timing, static hold, sinusoidal motion, lead/lag, missing spans, repeated frames và signer/source grouping giả.
- Canonical133 pose cache, manifest train/dev, labels chỉ cho downstream test.
- `configs/fixture_smoke.yaml` chọn tiny test encoder implement đúng adapter interface, FP32 CPU, small decoder, T=32, W16/hop8. Tiny encoder là fixture, không scientific baseline.
- Fixtures phải có đủ positive-energy valid pairs để phase calibration tests có nghĩa; include intentionally invalid cases riêng.

```bash
python tools/generate_synthetic_fixture.py --output data/fixture --seed 7
signcoord audit-data --config configs/fixture_smoke.yaml
signcoord fit-target-stats --config configs/fixture_smoke.yaml \
  --output data/fixture/target_stats.json \
  --phase-output data/fixture/phase_calibration.npz
signcoord audit-targets --config configs/fixture_smoke.yaml
signcoord smoke --config configs/fixture_smoke.yaml --steps 20
signcoord train --config configs/fixture_smoke.yaml --arm A3 --seed 7 \
  --set train.max_updates=100 --set run_dir=runs/fixture_A3
signcoord train --config configs/fixture_smoke.yaml --arm A7 --seed 7 \
  --set train.max_updates=100 --set run_dir=runs/fixture_A7
pytest -q tests/integration
```

Fixture config set warmup/ramp ngắn phù hợp smoke để phase backward được thực sự exercised; default 2k warmup sẽ không activate trong 100 updates. Save config diff; không sửa scientific warmup vì fixture.

### 19.3. Real data preparation

1. Resolve authorized/local dataset paths, dataset adapters, source manifests/splits; viết `configs/sources.yaml`.
2. `manifest` cần implement `--adapter`, `--root`, `--output`; không download lại media đã có.
3. Audit overlap với downstream splits trước extract pose hàng loạt.
4. `extract-pose` supports sharding, resume, failed record ledger, fixed signer tracking và cache schema. Write temp chunk -> verify -> atomic rename; content key gồm video/source interval/extractor/schema hashes.
5. Fit train stats/calibration trên deterministic sample plan, có coverage theo nguồn/part. Audit visualize overlays trước batch lớn.

```bash
signcoord manifest --config configs/base.yaml --adapter configured_source \
  --root data/videos --output data/manifests/ssl_train.jsonl
signcoord audit-data --config configs/base.yaml
signcoord extract-pose --config configs/base.yaml --shard-index 0 --num-shards 1
signcoord fit-target-stats --config configs/base.yaml \
  --output data/stats/target_stats.json \
  --phase-output data/stats/phase_calibration.npz
signcoord audit-targets --config configs/base.yaml
signcoord smoke --config configs/base.yaml --steps 50
signcoord profile --config configs/base.yaml --warmup-steps 50 --measure-steps 500
```

Trước những command real này, phải điền input paths/checkpoint thật trong config; các path `data/...` là convention của repo, không khẳng định đã có dataset. `configured_source` route dùng adapter được định nghĩa trong `sources.yaml`, không giả định corpus có schema thống nhất.

### 19.4. Matched pilot

```bash
signcoord train --config configs/base.yaml --arm A2 --seed 3407 \
  --set run_dir=runs/A2_seed3407
signcoord train --config configs/base.yaml --arm A3 --seed 3407 \
  --set run_dir=runs/A3_seed3407
signcoord train --config configs/base.yaml --arm A6 --seed 3407 \
  --set run_dir=runs/A6_seed3407
signcoord train --config configs/base.yaml --arm A7 --seed 3407 \
  --set run_dir=runs/A7_seed3407
```

Không bắt đầu A7-only để tiết kiệm rồi claim phase; A3/A6 là required. A0 extraction và A1 reference/reimplementation chạy cùng evaluation suite. Budget 1 GPU có thể sequential; 4 GPU dùng DDP hoặc independent matched runs theo measured VRAM, không tự spawn agents vì tài liệu này.

DDP entrypoint khi đủ GPU:

```bash
torchrun --standalone --nproc-per-node=4 -m signcoord.cli train \
  --config configs/base.yaml --arm A7 --seed 3407 \
  --set run_dir=runs/A7_seed3407_ddp4
```

Primary comparison không trộn 1-GPU stats batch với 4-GPU stats batch mà bỏ qua confound. Chọn world size cố định hoặc run matched baseline cho setting mới.

### 19.5. Frozen eval và export

Feature CLI supports `--checkpoint`, `--manifest`, `--output`; eval retrieval supports `--gallery`, `--queries`, `--output`. Cache chứa labels riêng hoặc explicit evaluation label file, không làm encoder đọc labels.

```bash
signcoord extract-features --config configs/eval_retrieval.yaml \
  --checkpoint runs/A7_seed3407/checkpoints/best_dev.pt \
  --manifest data/manifests/asl_citizen_train.jsonl \
  --output runs/A7_seed3407/features/aslc_train.npz
signcoord extract-features --config configs/eval_retrieval.yaml \
  --checkpoint runs/A7_seed3407/checkpoints/best_dev.pt \
  --manifest data/manifests/asl_citizen_dev.jsonl \
  --output runs/A7_seed3407/features/aslc_dev.npz
signcoord eval-retrieval --config configs/eval_retrieval.yaml \
  --gallery runs/A7_seed3407/features/aslc_train.npz \
  --queries runs/A7_seed3407/features/aslc_dev.npz \
  --output runs/A7_seed3407/eval/aslc_dev.json
signcoord compare --run-ids A3_seed3407 A6_seed3407 A7_seed3407 \
  --output runs/comparisons/pilot.json
signcoord export --config configs/base.yaml \
  --checkpoint runs/A7_seed3407/checkpoints/best_dev.pt \
  --output runs/A7_seed3407/export
```

NPZ là small-fixture convention; production large feature datasets dùng sharded files/memmap, giữ same API/hash/provenance. Test queries được extract/eval sau protocol/checkpoint khóa; không sửa target thresholds sau khi xem test.

### 19.6. Confirmatory experiments

Seeds mặc định `[3407,2027,42]`, fixed exposure và world-size setting. A3/A6/A7 tối thiểu; A5/A8/A9 để mechanism; T64 và pose-extractor sensitivity là trục riêng. Không grid search mọi combinations trước khi pilot cho thấy signal.

Agent tạo `experiments/preregistered_plan.yaml`: arms, seeds, budgets, selection rule, primary/secondary endpoints, stop rules, allowed hyperparameter grids. Sau run, compare script phải verify matched scientific config tự động trước aggregate.

## 20. Tests và acceptance criteria

### 20.1. Unit tests bắt buộc

| Test | Input/expected behavior | Lỗi cần bắt |
|---|---|---|
| Schema identity | Raw133 semantic names -> packed119 đúng, L/R/H anchors 115/116/110 | Hand-root/body-wrist hoặc namespace nhầm |
| Affine alignment | Known point transformed cùng RGB marker | Pose trước crop supervise RGB sau crop |
| Fixed origin | Torso dịch chuyển trong clip giữ trajectory khác 0 | Per-frame centering xóa body motion |
| Local roots | Wrist-relative root excluded, rigid translation không đổi local hand | Constant-zero targets inflate loss stats |
| Local orientation | Rotation đổi local orientation được giữ | Canonicalization xóa cue |
| Confidence propagation | Invalid finger/root/endpoints làm corresponding target invalid | Wrist confidence hợp thức hóa mọi finger |
| Missing finite | NaN raw -> fill + invalid; loss/grad finite | NaN*0 masking |
| Short gap | Gap<=2 đúng reliability, gap dài không interpolate | Lấp bất kỳ missing segment |
| Timestamp derivative | Same physical speed sampled khác dt cho same velocity | Pixels/frame confound |
| Velocity length | T32 ->31 intervals, W16/hop8 ->2 windows | Off-by-one/padded derivative |
| FFT axis | Independent parts/channels, sinusoid peak đúng k | FFT qua joint/channel/flatten order |
| Phase sign | Periodic delay d known, convention U_p conj(U_q) đúng sign | Conjugate convention đảo |
| Phase swap | Swapping p,q conjugates S/q | Pair semantic bug |
| Amplitude ambiguity | Circular delay giữ magnitude, đổi relative phase | Treat magnitude as timing |
| Parseval | Full FFT/rFFT weighted equality | Invalid spectral novelty/sanity |
| DC/Nyquist | Even W keep bins1..W/2-1, odd rejected MVP | Wrong number frequencies |
| Target gate detach | Gate/threshold no grad; student prediction không đổi gate | Student opts out |
| Static hold | All phase gate0 nhưng geom loss valid và backward finite | Drop static samples toàn pipeline |
| Orthogonal vector | Small cross energy gated, auto energy có thể lớn | Treat all energetic pairs reliable |
| Invalid window | One missing interval rejects window đúng policy | FFT zero-filled missing như observed |
| Warmup | λphase0 đến2k, ramp đến5k; fixture schedule khác explicit | Phase never activated/activated too early |
| Phase backward | Nonconstant perturbed prediction có finite nonzero gradient | Detach prediction hoặc bilinear-zero trap |
| Gradcheck | Small float64 numerical/analytic comparison ngoài thresholds | Complex autograd/reduction bug |
| Lag boundary/sign | Noncircular delayed signals peak positive lag | wrap-around/correlation reversed |
| Weighted reduction | Same result centralized vs partition valid counts | Mean of rank means |
| Retrieval ranking | Handcrafted nearest neighbors, ties, no relevant cases | Metric/index bug |
| Overlap exclusion | Exact/interval duplicates removed đúng protocol | Inflated frozen retrieval |
| CTC fixture | Adjacent repeats/blank collapse và infeasible lengths | Silent alignment invalid |
| Export isolation | Missing pose library/targets vẫn infer | Training-only dependency leak |

Dùng rectangular/no-window pure sinusoid cho exact phase-sign test; production Hann/window crop là separate tolerance test. Không lấy exact shift invariance của periodic signal làm expected assertion cho finite cropped Hann windows.

Gradcheck chạy float64/complex128 debug path của kernel, away from gate thresholds/clamp boundaries. Production spectral path float32/complex64. Nếu dùng reference code có dtype override, preserve float64 cho test; không cast float64 về float32 rồi yêu cầu numerical gradcheck chính xác.

### 20.2. Integration tests

1. Tiny fixture: manifest -> targets/stats/calibration -> A3/A7 train -> checkpoint resume -> frozen features -> retrieval -> export.
2. Hiera adapter T16/T32: correct tensor shapes, memory coords, keep-mask semantics, unmasked h timestamps, allowable checkpoint keys. T64 khi enable context arm.
3. Target-independent inference: delete/move pose cache và remove pose package; exported encoder vẫn hoạt động.
4. Frozen eval hash check: encoder weights trước/sau probe/eval giống hash; gradient only downstream head.
5. CPU two-process Gloo weighted reduction/gather test; GPU two-rank NCCL smoke khi environment có GPU.
6. Resume optimizer boundary: same next sample/mask/RNG and close/equal update theo determinism mode. Không đòi bitwise nếu kernels nondeterministic, nhưng report tolerance và causes.
7. All-invalid batch/pair: không NaN, collectives không deadlock, skip-policy synchronized; không có silent optimizer step trên completely invalid training batch nếu policy reject.

### 20.3. Real-data gates trước training dài

- Overlay kiểm tra ít nhất 50 clips từ nhiều source/quality bins; hand identity/face/tracking không nhảy bất thường.
- Report tỷ lệ normalization valid, hand/face geometry coverage, pair spectral eligibility và repeat FPS; đủ eligible samples để thử H1. Không đặt threshold coverage giả là universal optimum; nếu phase coverage gần 0 phải sửa input/target policy trước train.
- Run 50 updates finite backward, một small-set overfit diagnostic cho geometry/trajectory. Mục tiêu là chứng minh gradient path hoạt động; overfit không phải representation metric.
- Profile cùng config cho A3/A7; verify actual context, memory và λ activation.
- Frozen baseline evaluation working và leakage audit closed trước confirmatory test.

### 20.4. Acceptance evidence bundle

Lưu `tests_report.json`, `smoke_report.json`, `adapter_report.json`, `target_audit.json`, `data_overlap_report.json`, `profile.json`, `checkpoint_load_report.json`. Mỗi file có code/config/data hashes và thực thi timestamp; `status=not_run` nếu chưa chạy. Không dùng static checklist ticked thay evidence.

## 21. Task breakdown cho Agent và Definition of Done

### Milestone M0: Inventory, nguồn và config

**Tasks:** inspect repo, source lock, dependency environment, data/checkpoint availability, typed configs và CLI contracts.  
**Deliverables:** plan, lock, doctor, decision log, reproduction delta.  
**Done:** tiny environment test chạy; source/checkpoint/schema references không bịa; missing real inputs được xác định cụ thể.

### Milestone M1: Data và targets

**Tasks:** dataset adapters, normalized manifest, overlap audit, timestamp decoder, pose cache schema/reader, transforms, global/local targets, reliability và train-only stats.  
**Deliverables:** reproducible fixture, manifest/cache validation, overlays, target stats.  
**Done:** unit alignment/normalization/reliability tests pass; train/dev/test target calibration boundaries được enforce.

### Milestone M2: RGB model và losses cơ bản

**Tasks:** encoder adapters, checkpoint loading, T adaptation, keep-mask indexing, dense decoder, balanced geometry/velocity, projector regularizer.  
**Deliverables:** A0/A1 adapters, A2/A3 configs, smoke/backward/profiling.  
**Done:** encoder-only inference works; dense target decoder không nhận pose; gradients finite; DDP reductions/gather validated.

### Milestone M3: Coordination objectives

**Tasks:** spectrum kernels, calibration/gate, lag/magnitude controls, warmup, A4/A6/A7/A8/A9 registry.  
**Deliverables:** mathematical diagnostics, reference-vs-production tests, coverage dashboards.  
**Done:** phase/lag sign, Parseval, invalid windows, gradcheck và target-only gate tests pass; controls share information/model capacity.

### Milestone M4: Trainer và resume

**Tasks:** optimizer/AMP/DDP/accumulation, checkpoint boundaries/RNG, deterministic sample plans, logging và compute ledger.  
**Deliverables:** reproducible fixture end-to-end, real-data smoke/profile khi inputs có.  
**Done:** resume test pass, one/2-rank tests không deadlock, selected/final checkpoints và provenance đầy đủ.

### Milestone M5: Evaluation và deployment

**Tasks:** frozen feature extraction, pooling/dedupe, retrieval/probes/bootstrap/slices, CTC/SLT head interface, export.  
**Deliverables:** per-query/per-sample predictions, metric signature/protocol, encoder bundle và inference example.  
**Done:** fixture metrics đúng handcrafted expected; frozen weights unchanged; export không cần pose; real evaluation status ghi rõ.

### Milestone M6: Scientific experiments

**Tasks:** matched pilot; confirmatory >=3 seeds khi viable; two-dataset transfer, mechanism/quality controls, downstream sequence setting nếu available.  
**Deliverables:** complete experiment table, CI/seed stats, compute table, failure cases, limitations và recommendation.  
**Done:** mọi scientific claim liên kết run IDs/results/protocol; negative result được giữ; không best-seed cherry-picking.

### Final handoff checklist

- [ ] Repo có install instructions, lock, typed config và full CLI.
- [ ] Data/cache/schema/calibration có version/hash và split integrity.
- [ ] RGB input exclusivity và RGB-only export có tests.
- [ ] A3/A6/A7 implementation và matched comparison hoạt động.
- [ ] Meaningful unit/integration tests và execution reports được lưu.
- [ ] Baseline reproduction deviations và chưa chạy thực nghiệm nào được ghi rõ.
- [ ] Frozen metrics, fine-tuned metrics, synthetic metrics phân biệt.
- [ ] Checkpoints/configs/sampler/compute/protocol đủ để chạy lại.
- [ ] Report cuối nêu method đã implement, kết quả thật, phần bị block và bước tiếp theo.

Không tick checklist bằng suy luận; tick sau evidence. Không dừng ở “can implement”. Nếu real resource thiếu, engineering deliverables vẫn cần hoàn tất trong phạm vi có thể thực thi, research items ghi pending input/run rõ.

## 22. Failure handling, stop rules và extensions

### 22.1. Debugging matrix

| Symptom | Kiểm tra trước | Hành động |
|---|---|---|
| Pose coverage tốt nhưng phase gate gần0 | Cross energy, physical coordinates, window validity, calibration units | Audit vectors/orthogonality, timing; không hạ mọi threshold vô điều kiện |
| Phase loss giảm, frozen metrics không tăng | Decoder shortcut, encoder std/rank, dense/lag controls | Không tăng teacher complexity; test attribution |
| Phase gradients0 | Pred trajectories hằng, accidental detach, λwarmup | Warmup geometry, gradcheck nonconstant fixture |
| NaN/huge gradients | Raw NaNs, dt0, scale nhỏ, phase denominator, FP16 FFT | Repair contract; save failing sample/config |
| Good one seed only | Sample/mask plans, checkpoint selection, stats batch | More matched seeds; report variance |
| Retrieval quá cao bất thường | Gallery/query overlap, source reuploads, labels in pooling | Fix audit/protocol trước công bố |
| Handshape/nonmanual giảm | λbalance, pose noise, RGB resolution, motion sampling | Rebalance/dev ablation; appearance extension riêng |
| T32 load fail | Temporal pos shape, fixed factory args, Unroll/Reroll | Rebuild adapter; không ignore encoder mismatches |
| DDP diverges from1GPU | Loss denominator scaling, gather gradients, stats batch | Centralized-reference tests; match batch contract |
| Only pose extractor change improves | Same selected raw frames/coords, confidence calibration | Separate preprocessing contribution |

### 22.2. Stop/continue scientific rules

1. A7 không hơn A3/A6 trên frozen dev trong matched pilot: sửa lỗi/coverage trước; nếu implementation đúng và result bền, không mở rộng spectral teacher.
2. A2/A3 giải thích gain: contribution là dense/motion pretraining, không relative-phase supremacy.
3. A6>=A7 với compute thấp hơn: chọn lag hoặc claim coordination; không cố giữ Fourier vì tên đẹp.
4. Gain chỉ full fine-tune, không frozen: chỉ claim downstream adaptation trong setting đo được.
5. Nonmanual/hold giảm đáng kể: không gọi representation đa tác vụ tốt hơn chỉ vì micro retrieval tăng.
6. Test protocol/threshold thay sau xem test: report exploratory; cần new held-out confirmation cho stronger claims.
7. Pipeline engineering correct nhưng no real metrics: report implementation ready, không claim paper-level success.

### 22.3. Optional extension E1: appearance targets

Nếu geometry không đủ handshape/nonmanual, thêm train-only hand/face visual teacher có nguồn weights rõ [R12,R13,R29]. Same RGB crop timing, EMA/frozen status và supervision provenance. Teacher RGB không được dùng source future ngoài declared context.

Compare appearance-only vs geometry+appearance vs geometry+coordination+appearance. Extra targets phải có matched decoder/capacity/budget. SignDINO CVPR2026 [R12] và SignDino Sep2026 [R13] khác nhau; đọc cả hai trước novelty claim.

### 22.4. Optional extension E2: learned spectral teacher

Chỉ khi analytical phase/lag đã có bằng chứng. Teacher part-wise Fourier module và matched temporal Transformer/TCN teacher có cùng input, target dimension, params và compute.

Nếu gain amplitude enhancement, dùng bounded positive gain `A'=(1+alpha*tanh(g))*A`, `0<alpha<1`, phase input tại bước gain được giữ. Encoder trước gain có thể làm đổi phase; không tuyên bố original physical phase preserved end-to-end chỉ từ công thức này [R02,R23]. Không coi Fourier diagonal multiplication là arbitrary graph operator trừ assumptions được chứng minh.

### 22.5. Optional extension E3: longer context và hierarchy

T64/T128 khi budget/data cho phép; all controls cùng context/timestamp grid. Window W32/64 phải đủ observed velocity samples và subwindows. Compare length riêng trước spectral novelty. Relative phase trên long clip chứa nhiều signs có thể average mất local timing; local contexts/hierarchy cần annotation-free segmentation policy audited [R11].

### 22.6. Optional extension E4: forecasting/causal

Separate past-only encoder/head, no future interpolation, no bidirectional attention/teacher leakage. Evaluation latency tính acquisition window + pose preprocessing nếu inference có + model. Offline MVP không chuyển thành causal bằng cách đổi tên metric.

## 23. Cách đọc nguồn để Agent không hiểu sai method

### 23.1. Reading order

**Trước code model/loss:** [R01] main + supplementary, [R02] methods/ablation, [R23] assumptions/appendix, [R25] loss/statistics, [C01,C02] relevant implementation, [C05,C10,C11] complex spectral convention.

**Trước novelty/architecture extension:** [R08,R10,R11,R12,R13,R18,R20,R22,R30]. Phân biệt representation SSL, text-supervised pretraining, downstream gloss supervision và paper/preprint status. Không gộp hai bài tên gần SignDINO/SignDino.

**Trước data/eval:** [R31-R39], [C03,C04,C09]. Đọc actual split metadata và media availability của version dùng; official signer-independent downstream không bảo đảm upstream SSL đã signer-disjoint.

### 23.2. Source-to-module mapping

| Implementation module | Nguồn nền tảng | Phần dự án tự định nghĩa |
|---|---|---|
| RGB sign-prior pretraining | R01 | Dense trajectory decoder và shared-schema controls |
| Part-wise temporal motivation | R02 | Chuyển sang unlabeled coordination target, không CTC pretrain |
| Hiera adapter | R26/C02, C01 compatibility | T adaptation, representation export/index tests |
| Geometry/local hand | R01/R03/R04, C03 | 119-node 2D schema, local/global balancing |
| Velocity objective | R18 | Derivative/timing/reliability trên RGB prediction |
| Cross-spectrum/phase | C05/C10/C11 | Target definitions, gates, warmup, matched lag |
| Var/cov | R25 | Single-stream regularization policy và gather contract |
| Dense context motivation | R20/R30 | Lightweight analytic targets/budget |
| Frozen retrieval | R01/R32/R33 | Primary uniform coverage pooling, matched gallery |
| Linguistic slices | R34/R36/R37 | Probes hợp lệ khi có verified mapping |
| Benchmark limitations | R38/R39 | Leakage/preprocessing/protocol controls |

Không cite nguồn như thể tác giả nguồn đã đề xuất SignCoord-SSL. Mọi thresholds, hyperparameters và module ghép của spec là PROPOSED trừ khi explicitly ghi REPORTED.

### 23.3. Rules viết kết quả sau implementation

Result record phải có `experiment_id`, `claim`, `evidence_paths`, `comparison_ids`, `metric`, `split`, `uncertainty`, `compute` và `limitations`. Source paper numbers chỉ external context, không thay measured numbers.

Không dùng các câu “first”, “SOTA”, “A* guaranteed”, “semantic understanding proven”, “phase preserving end-to-end” hoặc “causal” nếu evidence không đủ. A* paper quality đến từ câu hỏi rõ, closest baselines, experiment controls và kết luận có thể bị bác bỏ.

## 24. Prompt giao việc có thể copy cho implementation Agent

> Hãy đọc toàn bộ `SignCoord_SSL_End_to_End_Implementation_Spec_VI.md` và thực hiện các milestones M0-M6 theo dependency. Ưu tiên hoàn thiện data contracts, Hiera adapter, dense geometry/velocity baseline, lag control, gated relative-phase loss, training/DDP/resume, frozen evaluation và RGB-only export. Không dừng ở scaffold hoặc pseudocode. Dùng nguồn cuối file, pin code/checkpoint/schema; mọi thay đổi ghi decision log. Chạy meaningful unit/integration tests và synthetic smoke; chạy real-data experiments khi inputs/hardware có. Không bịa data/checkpoint/results; tách engineering complete khỏi research complete. Giữ baselines cùng initialization, data exposure, context, decoder, regularization và pooling. Report cuối gồm code đã implement, tests đã chạy, result thật với experiment IDs, deviations, blockers cụ thể và recommendation dựa bằng chứng. Không claim phase tốt hơn cho đến khi vượt dense+velocity và matched lag trên frozen transfer.

## 25. Tài liệu tham khảo đầy đủ

Danh sách R01-R40 giữ mapping với research report trước, nhưng file này không cần report đó để triển khai. Những nguồn background/abstract không thay thế việc đọc method khi Agent dùng chúng làm baseline/extension. Links là nguồn sơ cấp; ưu tiên proceedings và official code khi có.


**[R01]** Wong, Camgoz, Bowden. *SignRep: Enhancing Self-Supervised Sign Representations*. ICCV 2025. [Paper](https://arxiv.org/abs/2503.08529); [bản ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.pdf); [phụ lục](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wong_SignRep_Enhancing_Self-Supervised_ICCV_2025_supplemental.pdf); [code](https://github.com/ryanwongsa/SignRep). Đối chiếu phương pháp, bảng kết quả và phụ lục.

**[R02]** Wei, Hu, Ma. *Part-Wise Graph Fourier Learning for Skeleton-Based Continuous Sign Language Recognition*. Journal of Imaging 11(8):286, 2025. [DOI/article](https://www.mdpi.com/2313-433X/11/8/286); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12387829/). Đối chiếu mục 3-4, công thức, hình và ablation.

**[R03]** Hu et al. *SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition*. ICCV 2021. [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html). Phương pháp.

**[R04]** Hu et al. *SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding*. TPAMI 2023. [Paper](https://arxiv.org/abs/2305.04868). Phương pháp và spectral GCN tokenization.

**[R05]** Zhao et al. *BEST: BERT Pre-training for Sign Language Recognition with Coupling Tokenization*. AAAI 2023. [Article/PDF](https://ojs.aaai.org/index.php/AAAI/article/view/25470). Phương pháp.

**[R06]** Jiang et al. *Skeletor: Skeletal Transformers for Robust Body-Pose Estimation*. CVPR Workshops 2021. [Paper](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeletor_Skeletal_Transformers_for_Robust_Body-Pose_Estimation_CVPRW_2021_paper.html). Abstract và mô tả phương pháp.

**[R07]** Sandoval-Castaneda et al. *Self-Supervised Video Transformers for Isolated Sign Language Recognition*. 2023. [Paper](https://arxiv.org/abs/2309.02450). Phương pháp, protocol, phonological probing, bảng kết quả.

**[R08]** Gueuwou et al. *SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction*. ACL 2025. [Paper](https://aclanthology.org/2025.acl-long.1397/). Phương pháp, data, protocol và kết quả.

**[R09]** *SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale*. Findings of ACL 2025. [Paper](https://aclanthology.org/2025.findings-acl.1157/). Các phần representation và kiến trúc.

**[R10]** Madjoukeng et al. *SSL-SLR: Self-Supervised Representation Learning for Sign Language Recognition*. TMLR 2026; preprint 2025, updated 2026. [Paper](https://arxiv.org/abs/2509.05188); [OpenReview](https://openreview.net/forum?id=buTZkTXijy). Phương pháp negative-free và augmentation; kiểm tra publication metadata.

**[R11]** Xie, Cai, Stefanov. *SignMAE: Segmentation-Driven Self-Supervised Learning for Sign Language Recognition*. arXiv 2026. [Paper](https://arxiv.org/abs/2605.02094). Phương pháp và ablation; dùng status preprint.

**[R12]** Gan et al. *Learning Effective Sign Features without Text for Gloss-free Sign Language Translation*. CVPR 2026. [Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.pdf); [author repository copy](https://wrap.warwick.ac.uk/id/eprint/197870/). Phương pháp, ablation và budget.

**[R13]** Hu et al. *SignDino: Self-Supervised Sign Language Representation Learning via Temporal-Axis Self-Distillation*. arXiv 09/2026. [Paper](https://arxiv.org/html/2609.06296). Phương pháp, ablation, limitations; dùng status preprint.

**[R14]** Jiang et al. *SignCLIP: Connecting Text and Sign Language by Contrastive Learning*. EMNLP 2024. [Paper](https://aclanthology.org/2024.emnlp-main.518/). Supervision và representation setting.

**[R15]** Pu et al. *Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding*. arXiv 2025. [Paper](https://arxiv.org/abs/2509.21223). Early fusion và paired-supervision objectives.

**[R16]** Li et al. *Uni-Sign: Toward Unified Sign Language Understanding at Scale*. 2025. [Paper](https://arxiv.org/abs/2501.15187). Unified pretraining, CSL-News và supervision.

**[R17]** *Phonological Representation Learning for Isolated Signs Improves Out-of-Vocabulary Generalization*. arXiv 2025. [Paper](https://arxiv.org/abs/2509.04745). Phương pháp, bảng ablation, reconstruction/recognition trade-off.

**[R18]** Mao et al. *Masked Motion Predictors are Strong 3D Action Representation Learners*. ICCV 2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html). Motion target và masking.

**[R19]** Yan et al. *SkeletonMAE: Graph-based Masked Autoencoder for Skeleton Sequence Pre-training*. ICCV 2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_SkeletonMAE_Graph-based_Masked_Autoencoder_for_Skeleton_Sequence_Pre-training_ICCV_2023_paper.html). Graph/joint/edge reconstruction. Không nhầm với bài khác cũng tên SkeletonMAE.

**[R20]** Xu et al. *Skeleton2vec: A Self-supervised Learning Framework with Contextualized Target Representations for Skeleton Sequence*. 2024. [Paper](https://arxiv.org/abs/2401.00921). Teacher và latent target.

**[R21]** Zhu et al. *MotionBERT: A Unified Perspective on Learning Human Motion Representations*. ICCV 2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html). Corrupted 2D-to-3D pretraining và kiến trúc.

**[R22]** Sun et al. *Exploring Adaptive Masked Reconstruction for Self-Supervised Skeleton-Based Action Recognition*. CVPR 2026. [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Exploring_Adaptive_Masked_Reconstruction_for_Self-Supervised_Skeleton-Based_Action_Recognition_CVPR_2026_paper.html); [arXiv](https://arxiv.org/abs/2606.11450). Decoder/focal reconstruction.

**[R23]** Yi et al. *FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective*. NeurIPS 2023. [Paper](https://arxiv.org/abs/2311.06190). Kernel assumptions, FGO và phụ lục n-invariant.

**[R24]** Shuman et al. *The Emerging Field of Signal Processing on Graphs*. IEEE Signal Processing Magazine, 2013. [Paper](https://arxiv.org/abs/1211.0053). GFT, Laplacian frequency và filtering background.

**[R25]** Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR 2022. [Paper](https://openreview.net/pdf?id=iWpcWZ8phD). Loss/statistics background.

**[R26]** Ryali et al. *Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles*. ICML 2023. [Paper](https://proceedings.mlr.press/v202/ryali23a.html). Backbone và MAE setting.

**[R27]** Tong et al. *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*. NeurIPS 2022. [Paper](https://arxiv.org/abs/2203.12602). Masking/pretraining findings.

**[R28]** Wang et al. *VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking*. CVPR 2023. [Paper](https://arxiv.org/abs/2303.16727). Abstract và architecture/recipe background.

**[R29]** Siméoni et al. *DINOv3*. 2025. [Paper](https://arxiv.org/abs/2508.10104). Metadata/abstract và vai trò trong hai SignDINO; không tái lập pretraining.

**[R30]** *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*. 2026. [Paper](https://arxiv.org/abs/2603.14482). Dense/context prediction objective; không tái lập scaling.

**[R31]** Tanzer, Zhang. *YouTube-SL-25: A Large-Scale, Open-Domain Multilingual Sign Language Parallel Corpus*. ICLR 2025. [Paper](https://arxiv.org/abs/2407.11144). Corpus và distribution.

**[R32]** Desai et al. *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*. NeurIPS Datasets and Benchmarks 2023. [Paper](https://arxiv.org/abs/2304.05934); [official split description](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/). Dataset/protocol.

**[R33]** Li et al. *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison*. WACV 2020. [Paper](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.html). Dataset và protocol background.

**[R34]** *Global-local Enhancement Network for NMFs-aware Sign Language Recognition*. 2020. [Paper](https://arxiv.org/abs/2008.10428). Dataset/abstract và vai trò non-manual evaluation.

**[R35]** Duarte et al. *How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language*. CVPR 2021. [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html). Dataset/metadata.

**[R36]** Sehyr et al. *The ASL-LEX 2.0 Project: A Database of Lexical and Phonological Properties for 2,723 Signs in American Sign Language*. Journal of Deaf Studies and Deaf Education, 2021. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685/); [official data](https://asl-lex.org/download.html). Feature/probe background.

**[R37]** Sandler. *Prosody and Syntax in Sign Languages*. Transactions of the Philological Society, 2010. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3250094/). Non-manual scope và prosody background.

**[R38]** Czehmann et al. *“A Sacred Bird Called the Phoenix”. Auditing the most-used Parallel Corpus for German Sign Language Recognition and Translation*. LREC SignLang 2026. [Paper](https://aclanthology.org/2026.signlang-1.9/). Audit/evaluation findings.

**[R39]** O’Brien et al. *Evaluation of Pose Estimation Systems for Sign Language Translation*. LREC SignLang 2026. [Paper](https://aclanthology.org/2026.signlang-1.39/); [arXiv](https://arxiv.org/abs/2604.24609). Abstract/protocol findings, không tái lập estimator comparison.

**[R40]** Chen et al. *Two-Stream Network for Sign Language Recognition and Translation*. NeurIPS 2022. [Paper](https://arxiv.org/abs/2211.01367). Modality/supervision background.

### 25.1. Official code và implementation documentation

**[C01] SignRep official repository.** [Repository](https://github.com/ryanwongsa/SignRep); [revision đã đọc](https://github.com/ryanwongsa/SignRep/tree/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2); [feature extraction example](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/example_usage.py); [final model wrapper](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/models/final_models/FINAL_hiera_latent_model_head_v25_active.py). Vai trò: checkpoint compatibility, feature layer, augmentation entrypoint và inventory những gì repo public thực sự cung cấp. Đọc các imported modules liên quan trước reproduction.

**[C02] Hiera official repository.** [Repository](https://github.com/facebookresearch/hiera); [revision đã đọc](https://github.com/facebookresearch/hiera/tree/b12b842542ee5c757fcfec8c41f6b56fcbe89b65); [Hiera core](https://github.com/facebookresearch/hiera/blob/b12b842542ee5c757fcfec8c41f6b56fcbe89b65/hiera/hiera.py); [Unroll/Reroll utilities](https://github.com/facebookresearch/hiera/blob/b12b842542ee5c757fcfec8c41f6b56fcbe89b65/hiera/hiera_utils.py); [MAE implementation](https://github.com/facebookresearch/hiera/blob/b12b842542ee5c757fcfec8c41f6b56fcbe89b65/hiera/hiera_mae.py). Vai trò: architecture, temporal/spatial indexing, mask semantics, positional embedding adaptation và code/model license. Không dùng ImageNet classifier checkpoint thay video MAE cùng tên backbone.

**[C03] COCO-WholeBody schema trong MMPose.** [Canonical dataset metadata](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py). Vai trò: semantic names, 133 landmark indices, hand roots và L/R identity. Agent phải pin MMPose commit/config metadata hash tại extraction; URL `main` ở đây là entrypoint, không đủ provenance cho run.

**[C04] MMPose whole-body model zoo.** [Official documentation](https://mmpose.readthedocs.io/en/latest/model_zoo/wholebody_2d_keypoint.html). Vai trò: chọn extractor có config/weights/schema phù hợp, thử quality và throughput, đọc dependencies. Không mặc định mọi model trong zoo có cùng landmark order hoặc confidence calibration.

**[C05] PyTorch `torch.fft.rfft`, API mốc 2.9.** [Versioned documentation](https://docs.pytorch.org/docs/2.9/generated/torch.fft.rfft.html). Vai trò: FFT axis, normalization, one-sided frequency output và precision constraints. Production block dự án dùng float32/complex64; debug numerical tests có float64 path.

**[C06] PyTorch Automatic Mixed Precision, API mốc 2.9.** [Versioned documentation](https://docs.pytorch.org/docs/2.9/amp.html). Vai trò: autocast/scaler; disable autocast tại sensitive spectral math. Kiểm tra signature đúng runtime được install.

**[C07] PyTorch distributed communication, API mốc 2.9.** [Versioned documentation](https://docs.pytorch.org/docs/2.9/distributed.html); [DistributedDataParallel](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html). Vai trò: process/rank/backend, parameter-gradient synchronization và collectives. DDP không tự sửa global validity weighting của custom loss.

**[C08] PyTorch autograd-enabled distributed functions.** [Official source, tag v2.9.0](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/distributed/nn/functional.py). Vai trò: kiểm tra differentiable gather/backward phù hợp version. Nếu API deprecated/unsupported trong runtime chọn, dùng supported equivalent và centralized-gradient test; không fallback detached gather rồi giữ cùng regularization claim.

**[C09] PyTorch CTCLoss, API mốc 2.9.** [Versioned documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CTCLoss.html). Vai trò: input layout/log probabilities/lengths/blank và infeasible alignments. Đây là downstream head, không default SSL pretraining loss.

**[C10] SciPy `signal.csd`.** [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html). Vai trò: Welch averaging/windowing/scaling và conjugate convention. Dự án dùng `U_p * conj(U_q)`; khi cross-check CSD cùng input order phải xử lý khác convention.

**[C11] SciPy `signal.coherence`.** [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html). Vai trò: coherence definition và windowed spectral estimation background. Không sử dụng scalar single-segment coherence làm confidence tự động.

### 25.2. Phạm vi xác minh của chính tài liệu

- Đã đối chiếu proposal nghiên cứu, code chính thức tại hai commit nêu trên và các documentation/nguồn sơ cấp liên quan khi viết spec.
- Đã kiểm tra cú pháp các Python snippets, parse JSON/YAML, consistency của source IDs/config, và NumPy/SciPy reference cho window counts, phase convention, orthogonal-channel cancellation và Parseval. Đây không phải PyTorch execution/gradient verification.
- Các chẩn đoán Parseval, magnitude/phase và arbitrary-vs-circulant FFT ở báo cáo nghiên cứu trước là numerical synthetic diagnostics.
- File này là **implementation specification**, chưa tạo training repo hoàn chỉnh, chưa chạy kernel PyTorch/gradient tests hoặc benchmarks SignCoord-SSL trên real data. Agent phải thực hiện các tests/runbook và lưu evidence theo milestones.
- Tất cả filenames/commands của repo tương lai là contracts cần hiện thực; checkpoint/data paths `null` phải được resolve từ môi trường thực tế.

**Kết quả triển khai mong muốn:** một pipeline RGB-only có thể tái lập, đo được contribution của dense geometry, velocity, lag và gated relative phase; lựa chọn method cuối cùng dựa trên frozen transfer và compute thực tế, kể cả khi phương án tốt nhất không dùng Fourier.
