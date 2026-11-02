# Kế hoạch train để kiểm tra tín hiệu skeleton

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: plan; ngày 2026-10-19; version `signal_ablation_v1`.
- Verification Status: **PLAN_ONLY**. Split counts, manifest paths, pose-path presence và annotation schema đã kiểm tra; recipe mới chưa implement/train.
- Research question: tín hiệu nào cải thiện **frozen sign dictionary retrieval** trong cùng data/encoder/objective/budget, và module nào cần thiết để dùng tín hiệu đó?
- Main input: ASL Citizen official train; main development endpoint: official val. Chi tiết ở [data protocol](DATA_PROTOCOL_VI.md).

## 1. Các tầng thực nghiệm

| Đợt | Việc cần làm | Dataset train | Evaluation | Kết quả cần có |
|---|---|---|---|---|
| E0 | Preflight, no-training baselines, profile | Không SSL | ASLC train gallery / val query | Contract hợp lệ, floor, thời gian thực |
| E1 | S0–S7: tách feature signals | ASLC train, bỏ nhãn khi SSL | Full ASLC val retrieval + diagnostics | Bản đồ sufficiency/incremental usefulness |
| E1b | Physical LH/RH và matched controls | ASLC train | Như E1 | Bỏ confound capacity, side và head reference |
| E2 | Xác nhận baseline/candidate | ASLC train, train mới từ scratch | Val, 3 paired seeds | Gain có ổn định không? |
| E3 | Objective và temporal mechanism | ASLC train | Val + movement/order slices | Signal yếu vì objective/readout hay vì observation? |
| E4 | Kết hợp và retrained leave-one-out | ASLC train | Val | Module có cộng dồn hay dư thừa? |
| E5 | Full-budget confirmation và test | ASLC train | ASLC test + WLASL transfer | Kết quả cuối, cùng contract |
| E6 | Context/scale/composition riêng | Dataset theo track | Target eval đã khóa | Giới hạn generalization, không trộn vào E1 |

**Không có training mới được khởi chạy trong lúc viết kế hoạch.** Các dòng ở [arms.csv](arms.csv) đều là `PLANNED_NOT_RUN`; không có metric dự kiến được điền thành kết quả.

## 2. E0: điều kiện trước khi train

1. Khớp IDs giữa CSV official và JSONL; kiểm tra split/signer/gloss, pose schema, timestamps, NaN/out-of-frame, confidence scale, track switches, wrist mismatch. Giữ query denominator official.
2. Ghi hashes manifest, code, config, normalization và data adapter. ASL-LEX/test labels không đi vào SSL sampler/loss.
3. No-training baselines: normalized H/L descriptors với mean+variance; sequence/DTW reference báo riêng cost; random-init encoder cùng readout; confidence/missingness/duration-only controls.
4. Kiểm tra encoder nhận input hợp lệ, masked feature không nhìn target qua derivative/anchor, feature bị bỏ không quay lại teacher/predictor. Chạy smoke chỉ trên train/dev; không score test để debug.
5. Profile 200–500 updates sau warmup và extraction 1k train clips. Ghi peak VRAM, sec/update, clips/s, bytes/video. Batch32 là effective batch; nếu phải gradient accumulate, giữ data draw/update và report microbatch.

Chỉ tiếp tục khi baseline train không collapse và evaluator vượt sanity checks. Nếu có lỗi, sửa và chạy lại đối chứng bị ảnh hưởng; không diễn giải run lỗi như evidence tín hiệu vô ích.

## 3. Recipe cố định cho E1/E2

Các giá trị dưới là **quyết định thiết kế ban đầu**, không phải kết quả tuning hoặc tuyên bố optimum. Lấy kích thước/optimizer/latent objective từ recipe PLctx để có anchor kỹ thuật, nhưng **feature adapter mới** để tách H/L/M/B/F/G. S2 mới không đồng nhất checkpoint PLctx cũ.

| Thành phần | Giá trị khóa cho screening |
|---|---|
| Initialization | Scratch cho mọi arm; EMA teacher khởi tạo từ student cùng arm |
| Encoder | Temporal/part Transformer: dim256, 6 blocks, 4 heads |
| Predictor | Dim128, 4 blocks; chỉ dùng trong pretraining |
| Time input | 64 samples ở `min(source_fps,25)`; 2 samples/bin, 32 temporal bins |
| Spatial feature projection | MLP cùng thiết kế; mỗi enabled feature group → dim256 |
| Part slots | 8 slots ×32 bins, có mask cho inactive slots; không attention/pool/loss inactive slots |
| Positional information | Temporal position + physical-side/group IDs; cùng recipe giữa arm |
| Objective | Masked contextual EMA latent prediction, centered CE trên feature dimension; **không** intrinsic geometry/motion regression ở E1 |
| Teacher target | LayerNorm trung bình 4 layer cuối, stop-gradient; cùng enabled inputs với student arm trước masking |
| Student/teacher temperatures | 0,1 / warmup 0,04→0,06 trong 10% updates đầu |
| EMA / center | Teacher momentum cosine 0,996→0,9999; center EMA0,9 |
| Mask | Khóa một sampler random temporal blocks: khoảng 50% valid bins, block4 bins; cùng source-mask RNG giữa arm |
| Mask accounting | Ghi target/visible fraction thực sau dependency closure; không ép mask vào padding/missing data |
| Loss normalization | Mean valid target tokens từng enabled group rồi mean các group; không để face nhiều joints có trọng số lớn tự động |
| Confidence policy | E1 không input confidence và không confidence-weight loss; valid targets weight1, confidence chỉ audit |
| Batch / optimizer | Effective batch32; AdamW lr3e-4, betas0,9/0,95, wd0,05 |
| LR / stability | Warmup10%, cosine đến 1% peak; grad clip1,0; bf16 nếu thiết bị hỗ trợ |
| Window sampling | Uniform theo video, random valid start, speed1,0 ở E1; short clip pad+valid mask |
| Augmentation | E1 không mirror, không temporal warp, không morphology hoặc noise tăng cường; mask là corruption chính |
| Embedding chính | EMA encoder, group-balanced mean contextual tokens theo time/group, L2, **256 chiều** |
| Inference | Dense windows, stride2 trên resampled grid, thêm last valid window để phủ tail; một window có padding cho clip ngắn |
| Clip pooling | Mean raw window vectors → L2, như nhau giữa arm; không activity weighting/PCA trong primary |

Source recipe để đối chiếu: [PLctx launch config](../../../method/phonojepa/runtime/launch/PJ_PLctx_FULL_160ep.config.json), [training](../../../method/phonojepa/pj/train.py), [centered loss](../../../method/phonojepa/pj/losses.py). Đây là tham chiếu, **chưa có CLI hiện tại chạy đúng toàn bộ bảng trên**. Không gọi script cũ với tên arm mới rồi coi như đã triển khai ablation.

H/L dùng 4 feature groups: LH configuration, RH configuration, LH location, RH location. Các feature bổ sung có slot riêng, mapping phải nằm trong config. Tối đa ba module sống sót được ghép ở E4 nên 8 slots đủ cho kế hoạch này. Slots giống nhau không bảo đảm active FLOPs/capacity giống nhau; phải ghi active groups/params và có matched controls.

**Giới hạn estimand:** teacher của S0 cũng chỉ thấy H; teacher S1 chỉ thấy L. Vì target latent thay theo enabled input, E1 đo “tín hiệu hữu ích dưới cùng SSL recipe”, không cô lập toàn bộ ảnh hưởng của input khỏi target distribution. Một teacher chung nhìn đủ mọi stream sẽ là distillation track khác, không được gọi S0 là H-only SSL. Khi cần kiểm tra fixed target, dùng supervised reference hoặc frozen-teacher track được khai báo riêng.

## 4. E1: tám cấu hình chính

Mọi arm train trên **40.154 ASLC train clips**, 10.000 optimizer updates × batch32, seed0; checkpoint chính là **đúng update10.000**. Val tại5.000 chỉ theo dõi learning curve; không chọn checkpoint khác nhau cho arm screening.

| ID | Input | So sánh chính | Giả thuyết/diagnostic |
|---|---|---|---|
| S0 | H | S0 vs S2 | Local hand shape có đủ? Location pairs có yếu hơn? |
| S1 | L | S1 vs S2 | Wrist/arm path có đủ khi thiếu ngón? Handshape pairs? |
| S2 | H+L | Baseline | Shape/location kết hợp; encoder256 và raw mean readout |
| S3 | H+L+M | S3−S2, rồi S3−C_M | Explicit velocity giúp movement và R@1 ngoài capacity? |
| S4 | H+L+B | S4−S2, S4−C_B | Relative bimanual encoding giúp hơn một nhánh position? |
| S5 | H+L+Fmouth | S5−S2, S5 vs C_HEAD | Mouth expression gain hay head reference effect? |
| S6 | H+L+Fother | S6−S2, S6 vs C_HEAD | Brows/eyes/head bundle có ích ở nhóm nào? |
| S7 | H+L+G | S7−S2, S7−C_G | Bone/angle encoding giúp ngoài tọa độ và extra branch? |

Với batch32, mỗi run10k dùng 320.000 clip presentations, khoảng7,97 lượt qua corpus; hai teacher/student views không phải hai video độc lập. Tám run tương ứng80.000 updates =2.560.000 clip presentations. Không gọi 10k là10k epochs hoặc10k distinct clips.

### E1b: controls và physical hand streams

| ID | Cấu hình | Tại sao cần? |
|---|---|---|
| C_M | S2 + nhánh L duplicate, cùng dimension/missing-mask của M | Loại hiệu ứng extra capacity/token và quality-mask shortcut |
| C_B | S2 + nhánh position duplicate, cùng dimension/mask của B | Đối chứng learned relation so với explicit relation |
| C_G | S2 + nhánh H duplicate, cùng dimension/mask của G | Loại lợi ích extra hand encoder capacity |
| C_HEAD | S2 + rigid head geometry, không expression | Kiểm tra S5/S6 có thêm thông tin định vị head thay vì facial deformation |
| P_LH | H_LH + L_LH + static anchors | Physical left-hand-only sufficiency |
| P_RH | H_RH + L_RH + static anchors | Physical right-hand-only sufficiency |

Các run này cùng10k/seed0. C_M/C_B/C_G có projection/params tương ứng candidate và sample cùng validity pattern, nhưng không được mang giá trị M/B/G vào input. Chúng là matched controls, không phải phương pháp cuối. C_HEAD là semantic control, không phải capacity-only control; nếu face candidate còn thắng, thêm duplicate-position control ở confirmation.

Tổng initial training batch đầy đủ: **14 run ×10k =140.000 updates**, chưa tính random-init/no-training baselines. P_LH/P_RH không được gọi dominant/non-dominant. Nếu muốn theo tay trội, làm một track canonicalization riêng với heuristic error audit.

## 5. E2: xác nhận nhiều seed và kiểm tra tín hiệu thực sự

Chọn S2 và **tối đa hai candidate từ S3–S7**, giữ ít nhất một family khác nếu CI screening chưa rõ. Train lại từ scratch **40.000 updates**, seeds0/1/2. Chấm endpoint40k cho mọi arm. Không resume run10k đã dùng cosine schedule10k thành một run40k rồi so với scratch40k.

- Core confirmation: 3 cấu hình ×3 seeds =9 run, tổng360.000 updates.
- Candidate có gain phải xác nhận thêm matched control tương ứng với cùng40k/3seeds trước claim gain riêng từ tín hiệu.
- Không chọn “best seed”. Báo từng seed, mean±SD và paired effect trên cùng query.
- Lưu checkpoint10k/20k/40k để kiểm tra thứ hạng có đảo theo budget; endpoint40k vẫn là primary của E2.

Frozen interventions trên val: drop part, reverse, shuffle temporal blocks, static-pose repeat, LH/RH desync, FPS/jitter/gap. Ghi primary clean gallery/corrupted query và secondary both-sides corruption; luôn recompute derived features.

Nếu cần biết liệu input có dùng được dưới target cố định, chạy **SUP_S2** và **SUP_CANDIDATE** cùng architecture/input, gloss CE reference,20k updates/seeds0,1; discard head rồi frozen retrieval. Đây là **supervised reference**, không vào bảng SSL. Nó không phải information-theoretic upper bound. Gain supervised mà SSL không có gợi ý kiểm tra objective/optimization, chưa chứng minh tín hiệu không tồn tại.

## 6. E3: tín hiệu yếu do objective hay do temporal structure?

Chỉ mở sau E2; khóa một input recipe, data và readout. Không đổi feature/architecture cùng lúc khi so objective.

| ID | Thay đổi duy nhất so O_LAT | Đối chứng |
|---|---|---|
| O_LAT | Latent centered CE như E1 | Baseline objective |
| O_POS | Masked coordinate/feature reconstruction | Cùng source mask, normalized target, same predictor budget |
| O_MOT | Masked motion-residual prediction | Cùng input/mask; chỉ target đổi; denominator và derivative validity báo rõ |
| O_UNIT | Train-only multi-stream cluster targets | Codebook chỉ fit train; codebook usage và extra cost |
| O_CLIP | Clip-level negative-free alignment | Views giữ lexical identity; không chọn positives bằng gloss |

Screen objective10k/seed0, xác nhận top recipe và O_LAT40k/3seeds. Nếu chưa có implementation parity cho một objective, ghi `NOT_IMPLEMENTED`, không dùng run code khác family làm matched arm.

Temporal follow-up, không trộn vào objective sweep:

1. Cùng64 samples: local vs global temporal receptive field, capacity-matched; xem movement/order/repetition diagnostics.
2. Cùng checkpoint: mean vs ordered bins2 vs token matching, fit mọi transform trên train. Đây là **readout gain**, không gán thành encoder gain.
3. Nếu global không giúp, chưa kết luận long-context vô ích: window có thể không phủ đủ sign. Đo distribution duration/coverage trước khi thử128 samples.
4. Hierarchical joint→part→segment objective chỉ thêm khi diagnostic chỉ ra vấn đề và có control cùng capacity. Composition cần lexical/combination holdout riêng.

## 7. E4/E5: ghép module, full budget và test

E4 chọn tối đa3 module có bằng chứng, chạy tối đa2³=8 cấu hình,40k/seed0; đo tương tác. Sau đó retrain full candidate và leave-one-out từng module với3seeds. Nếu hai module thay thế nhau, chọn theo gain/cost; không bắt mọi module phải có main effect dương ở mọi tổ hợp.

E5 đề xuất budget neo **200.640 updates** cho S2 và final candidate,3seeds/cấu hình. Đây là budget tương đương số update PLctx full160, **không** đảm bảo wall time hay quality tương đương. Chỉ thực hiện khi profiler/budget cho phép và learning curves40k còn phù hợp. Nếu đổi budget, khóa trước run và đổi cả baseline/candidate.

Val checkpoint grid:40k,80k,120k,160k,200.640; chọn best val R@1 theo cùng quy tắc, tie ưu tiên update sớm hơn. Test mọi seed đã đăng ký của baseline/candidate được chọn bằng val; báo mean±SD, không ensemble ngầm, không chọn theo test.

Final batch:

- ASLC gallery train40.154 / query test32.941; primary cùng readout/scorer E1.
- WLASL2000 gallery train / query test, không fine-tune, normalization giữ ASLC-train fit. Availability/provenance và overlap báo riêng; chưa khẳng định n local khi chưa audit.
- Frozen probe/robustness/minimal pairs cuối; các cặp từ test chỉ dùng ở đây.
- Test đã được dùng ở nghiên cứu trước trong repo: đây là locked new experiment, không tuyên bố test hoàn toàn chưa từng được nhìn.

## 8. Theo dõi và artifact khi chạy thật

Implementation mới nên nằm trong một method directory riêng, ví dụ `method/skeleton_signal_ablation/`, **chưa được tạo/implement trong đợt tài liệu này**. Không ghi vào runtime/weights cũ. Mỗi run cần:

```text
<run>/config.json                # arm, seed, full observation/target/readout contract
<run>/provenance.json            # code/data/schema hashes, supervision and source paths
<run>/train.jsonl                # consumed update, losses, gradients, timing, memory
<run>/checkpoints/               # student, EMA, predictor, optimizer, centers, RNG, cursor
<run>/val/step_<n>/metrics.json
<run>/val/step_<n>/query_ranks.csv
<run>/diagnostics/               # probes, perturbations, collapse, slices and coverage
```

Resume chỉ khi config/data/code tương thích; không dùng checkpoint path tồn tại để suy ra job đang chạy. Log ít nhất mỗi50 updates, save mỗi500 updates và ở endpoint. Chạy lần lượt trên GPU dùng chung; không giả định có GPU trống. Không ghi số GPU-hour dự đoán trước profiler.

Tổng train time tính từ `sum(updates × measured_sec_per_update)/3600`, cộng extraction/scoring/probes. Timeout từng run được đặt sau profiler với headroom và lưu trong run config; kế hoạch này chưa áp hard timeout.

## 9. Khi nào được chọn module?

Xem [metric/decision contract](METRICS_AND_DECISIONS_VI.md). Một candidate có lợi khi gain R@1 lặp lại qua seed, không chỉ do extra branch/readout/coverage, diagnostic phù hợp và cost chấp nhận được. “Không significant” là chưa kết luận, không phải tín hiệu vô ích. Nếu chỉ proxy probe tăng mà retrieval không tăng, giữ như cơ chế cần nghiên cứu; chưa ghép thành final method.
