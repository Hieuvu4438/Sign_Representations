# Trạng thái thực thi nghiên cứu Sign Representations

## Material Passport

Artifact: `reports/material_passport.json`. Mode: run. Raw datasets: read-only. Runtime: Codex, một agent.

**IN_PROGRESS — chưa có method novelty được kiểm chứng; goal vẫn active.**

Báo cáo này phản ánh bằng chứng hiện có, không phải final synthesis hoàn tất kế hoạch.

Cập nhật UTC: 2026-10-01T07:52:17Z

G0: **COMMON_PLUS_INTERNAL_RECORDED_FUNCTION_DIAGNOSTIC** trong phạm vi nhãn đã kiểm tra. Baseline lexical và diagnostic nội bộ đã có kết quả; xem phạm vi bên dưới.

## Dữ liệu và leakage

| Dataset | Annotation rows | File hiện có | Train/val/test |
|---|---:|---:|---|
| asl_citizen | 83399 | 83399 | {'train': 40154, 'val': 10304, 'test': 32941} |
| wlasl2000 | 21095 | 21095 | {'train': 14296, 'val': 3920, 'test': 2879} |
| ms_asl | 25513 | 18855 | {'train': 16054, 'val': 5287, 'test': 4172} |
| how2sign | 35041 | 34917 | {'train': 31165, 'val': 1527, 'test': 2349} |
| csl_daily | 20654 | 20654 | {'train': 18401, 'val': 1077, 'test': 1176} |
| csl_daily_sentence_crop | 20654 | 20654 | {'train': 18401, 'val': 1077, 'test': 1176} |
| phoenix14t | 8257 | 8257 | {'train': 7096, 'val': 519, 'test': 642} |

Nguồn: `reports/manifest_coverage.json`, `reports/manifest_validation.json`, `reports/split_audit*.json`.

ASL Citizen có 52 participant IDs trong CSV chính thức và không có signer overlap giữa các split.
MS-ASL có 5 nguồn YouTube xuất hiện ở nhiều split. PHOENIX có 494 recording-name group candidates xuất hiện ở nhiều split; đây là grouping theo tên nguồn, chưa phải xác nhận video trùng.
WLASL có 21.095 source recording groups chưa resolve; How2Sign test có 2.349 groups chưa resolve. Các điều này giới hạn transfer claims.
Decode coverage toàn bộ corpus và pretraining overlap chưa được chứng minh.

## Backbone và pilot

SignRep upstream `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2`, W01 SHA256 `f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b`.
Strict load: không có missing/unexpected keys; pretrained projection head hiện diện; 55.644.329 tổng tham số.
8 mẫu train smoke đều PASS, feature dimension 768, RGB/transform/timestamps/masks hợp lệ, sai khác eval lặp tối đa 0.
Nguồn: `features/signrep_pilot/extraction_report.json`, `features/signrep_pilot/load_audit.json`, `logs/signrep_smoke_attempt4.log`.
Pilot single pass: 8.194 giây cho 328 windows; peak allocation 1243020800 bytes.
Cohort forecast 1.34 giờ; full ASL Citizen forecast 23.73 giờ. Đây là extrapolation từ 8 train clips, chưa phải đo full-duration distribution.
SHuBERT dùng project venv Python 3.10; strict load encoder và hai DINO checkpoints đã PASS trên CPU. Raw-video pilot xem trạng thái bên dưới. Public translator đã có CPU numerical smoke; xem report bên dưới, không gọi là paper checkpoint.

## Protocol và runs

Protocol common_v1 đã khóa trước test: 200 lớp chọn từ train, 2.400 train / 726 val / 1.600 test; primary metric macro recall.
Baseline này phục vụ kiểm tra pipeline/lexical retention; không phải full benchmark reproduction hoặc kết quả grammar.
B02 linear và B03 shallow temporal head chỉ học readout. Các head báo tham số thực tế; khác biệt head không được diễn giải thành novelty của encoder.

| Run | Status | Metrics | Failure reason |
|---|---|---|---|
| signrep_smoke_attempt1 | FAIL | NA — chưa có | Upstream constructor contains integer products; initial literal parser rejected them |
| signrep_smoke_attempt2 | FAIL | NA — chưa có | Inherited timm 0.4.5 lacks required Mlp; installed timm 1.0.20 in project venv |
| signrep_smoke_attempt3 | FAIL | NA — chưa có | Untracked Python bytecode falsely triggered source-change guard; tracked source remains unchanged |
| signrep_smoke_attempt4 | PASS | features/signrep_pilot/extraction_report.json |  |
| common_v1_extraction | PASS | features/signrep_common_v1/extraction_report.json |  |
| ncslgr_signrep_smoke_v1 | PASS | features/signrep_ncslgr_smoke_v1/extraction_report.json |  |
| ncslgr_signrep_extraction_v1 | PASS | features/signrep_ncslgr_diagnostic_v1/extraction_report.json |  |
| ncslgr_frozen_readout_suite_v1 | PASS_INTERNAL_DIAGNOSTIC | runs/ncslgr_diagnostic_v1/summary.json |  |
| common_v1_B00_signrep_seed0 | PASS | runs/common_v1_B00_signrep_seed0/metrics.json |  |
| common_v1_B01_signrep_seed0 | PASS | runs/common_v1_B01_signrep_seed0/metrics.json |  |
| common_v1_B02_signrep_seed0 | PASS | runs/common_v1_B02_signrep_seed0/metrics.json |  |
| common_v1_B02_signrep_seed1 | PASS | runs/common_v1_B02_signrep_seed1/metrics.json |  |
| common_v1_B02_signrep_seed2 | PASS | runs/common_v1_B02_signrep_seed2/metrics.json |  |
| common_v1_B03_signrep_seed0 | PASS | runs/common_v1_B03_signrep_seed0/metrics.json |  |
| common_v1_B03_signrep_seed1 | PASS | runs/common_v1_B03_signrep_seed1/metrics.json |  |
| common_v1_B03_signrep_seed2 | PASS | runs/common_v1_B03_signrep_seed2/metrics.json |  |
| shubert_encoder_smoke_attempt1 | PASS | features/shubert_encoder_smoke_attempt1/report.json |  |
| shubert_encoder_smoke_attempt2 | PASS | features/shubert_encoder_smoke_attempt2/report.json |  |
| shubert_native_preprocessing_attempt1 | FAIL | features/shubert_native_preprocessing_attempt1/report.json |  |
| shubert_native_preprocessing_attempt2 | PASS | features/shubert_native_preprocessing_attempt2/report.json |  |
| shubert_native_pilot_attempt1 | PASS | features/shubert_native_pilot_attempt1/report.json |  |
| shubert_native_pilot_gpu_attempt1 | FAIL | NA — chưa có | SIGSEGV in native GPU smoke; no terminal Python exception; CPU native raw-video smoke passed. CUDA runtime isolation in progress. |
| shubert_native_pilot_gpu_attempt2 | FAIL | NA — chưa có | SIGSEGV in torch.cuda._lazy_init after decord import, before model transfer; faulthandler trace preserved. CUDA-first import-order smoke passed. |
| shubert_native_pilot_gpu_attempt3 | PASS | features/shubert_native_pilot_gpu_attempt3/report.json |  |
| shubert_native_pilot_gpu_attempt4 | PASS | features/shubert_native_pilot_gpu_attempt4/report.json |  |
| shubert_native_common_v1_preprocessing | RUNNING | features/shubert_native_common_v1_preprocessing/status.json |  |
| shubert_native_common_v1 | FAIL | features/shubert_native_common_v1/status.json | ValueError: Native feature worker reached reserved disk limit |
| shubert_native_ncslgr_v1_preprocessing | PASS_NATIVE_PREPROCESSING | features/shubert_native_ncslgr_v1_preprocessing/report.json |  |
| shubert_native_ncslgr_v1 | PASS | features/shubert_native_ncslgr_v1/extraction_report.json |  |
| fair_ncslgr_v1_suite | PASS_EXPLORATORY_EQUAL_BUDGET_NATIVE_FUNCTION_CONTROLS | runs/fair_ncslgr_v1/summary.json |  |
| ncslgr_author_pool_v1_suite | PASS_EXPLORATORY_AUTHOR_IMPUTED_FUNCTION_POOLING | runs/ncslgr_author_pool_v1/summary.json |  |
| shubert_translation_smoke_attempt1 | PASS_LOCAL_PUBLIC_DEMO_NUMERICAL_SMOKE | features/shubert_translation_smoke_attempt1/report.json |  |
| ncslgr_recorded_interval_suite_v1 | PASS_EXPLORATORY_RECORDED_INTERVAL_DIAGNOSTIC | runs/ncslgr_intervals_v1/summary.json |  |
| ncslgr_categorical_calibration_v1 | PASS_EXPLORATORY_VALIDATION_TEMPERATURE_SENSITIVITY | runs/ncslgr_calibration_v1/summary.json |  |
| fair_common_v1_suite | FAIL | runs/fair_common_v1/status.json | ValueError: Native extraction is terminal without full PASS: FAIL |
| semantic_pairs_readiness_v1 | BLOCKED_GOLD_OR_ACCESS | reports/semantic_pairs_readiness_v1/status.json | ValueError: External semantic protocol remains blocked or unlocked: BLOCKED_ACCESS |
| semantic_pair_backend_smoke_attempt1 | PASS_SEMANTIC_BACKEND_NUMERICAL_ONLY | features/semantic_pair_backend_smoke_attempt1/report.json |  |
| grounding_readiness_v1 | BLOCKED_GOLD_OR_MAPPING | reports/grounding_readiness_v1/status.json | ValueError: Grounding protocol blocked or unlocked: BLOCKED_DATA |
| native_common_disk_recovery_v1 | CPU_PREPROCESSING_RECOVERY | state/native_common_disk_recovery_v1/status.json |  |
| ncslgr_branch_ablation_import_failure_v1 | FAIL_IMPORT_BEFORE_MODEL_LOAD | evidence/ncslgr_branch_ablation_import_failure_v1/manifest.json | ModuleNotFoundError defusedxml from eager ncslgr XML import in extraction runtime |
| ncslgr_branch_ablation_smoke_v2 | PASS_TRAIN_ONLY_EIGHT_CONDITION_NUMERICAL_SMOKE | reports/ncslgr_branch_ablation_smoke_v2_audit.json |  |
| ncslgr_branch_ablation_v2_features | FAIL | features/ncslgr_branch_ablation_v2/status.json | ValueError: Recomputed all-stream original features differ beyond registered tolerance: 4.7206878662109375e-05 |
| ncslgr_branch_ablation_v2_pipeline | FAIL | state/ncslgr_branch_ablation_v2/status.json | ValueError: Child RUNNING_CPU_INTERVENTION_EXTRACTION exited 1; see logs/ncslgr_branch_ablation_extraction_v2.log. No automatic retry. |
| ncslgr_branch_ablation_smoke_v3 | PASS_TWO_TRAIN_CASES_WITHIN_RUN_CUE_ABLATION | reports/ncslgr_branch_ablation_smoke_v3_audit.json |  |
| ncslgr_branch_ablation_v3_features | RUNNING | features/ncslgr_branch_ablation_v3/status.json |  |
| ncslgr_branch_ablation_v3_pipeline | RUNNING_CPU_INTERVENTION_EXTRACTION | state/ncslgr_branch_ablation_v3/status.json |  |
| ncslgr_branch_reproduction_diagnostic_v1 | PASS_NUMERICAL_DIAGNOSIS_NO_SCIENTIFIC_METRICS | reports/ncslgr_branch_reproduction_diagnostic_v1.json |  |

Native lexical preprocessing: RUNNING, 1585/4726; GPU0 features: FAIL, 1235/4726 (26.1%).
Equal-grid lexical suite: FAIL. Số clip xử lý chưa phải số clip đủ điều kiện pooling; coverage cuối và class support còn chờ audit.
Tiến độ là snapshot từ status files; không suy diễn metric native từ cache một phần.

SHuBERT raw-video GPU pilot: PASS_NATIVE_RAW_VIDEO_SMOKE
Native author output là nhánh FFN cuối (`layer_results[-1][-1]`), khác encoder output sau residual. Raw-video scripts giữ FPS đầu vào; paper pretraining bỏ mỗi frame thứ hai và downstream recipe còn TODO, nên không claim exact paper reproduction.
Cả encoder và DINO sử dụng weights tác giả đã khớp SHA256, không có missing/unexpected keys. Cache cuối lưu [12,T,768], last-layer và layer-average; mask loại frame thiếu stream khỏi pooling/loss.
7/8 native pilot clips có frame đủ bốn streams để pooling; clip PRETTY có zero valid pooling frames và phải exclude ở future probes. 8/8 extraction PASS không có nghĩa 8/8 usable readouts. Nguồn: `reports/shubert_native_missingness_audit.json`.
Hai GPU smoke attempts SIGSEGV được giữ lại. Import-order audit tái hiện lỗi decord-before-CUDA; CUDA-first workaround và full raw-video forward đã PASS. Nguồn: `reports/shubert_cuda_import_audit.json`.

## Local lexical results

| Baseline | Macro recall mean | Recording bootstrap 95% CI | Training seed SD | Seeds |
|---|---:|---|---:|---|
| B00 | 0.0050 | [0.0050, 0.0050] | NA — deterministic | ['0'] |
| B01 | 0.6525 | [0.6291, 0.6750] | NA — deterministic | ['0'] |
| B02 | 0.7969 | [0.7774, 0.8155] | 0.0016535945694153339 | ['0', '1', '2'] |
| B03 | 0.8117 | [0.7953, 0.8271] | 0.002194927865177665 | ['0', '1', '2'] |

Primary delta B03−B02: 0.0148, paired recording CI [0.0015, 0.0284].

CIs conditional trên heads đã train và cohort 200 lớp; không bao gồm training randomness hoặc independent signer-population uncertainty. Không chọn best seed làm headline.
B03 có 6 learning-rate/width candidates; B02 có 3 learning-rate candidates trong protocol đã khóa. Head parameters được match trong 0.1%, nhưng tuning search khác nhau là confound; delta chỉ là diagnostic, không chứng minh superiority của encoder/method.
Active GPU compute hours chưa đo trực tiếp. Registry lưu wall_hours và một-GPU allocation wall-hours bao gồm decode/loading/I/O; không gọi chúng là GPU compute-hours.
Error audit dùng 30 IDs cố định từ union lỗi B03, kiểm tra kỹ thuật tự động. Review lexical correctness/ambiguity và video identity bởi ASL expert còn PENDING.


## NCSLGR recorded-function diagnostic

Historical public XML mirrors có functional gold tiers; 222 utterances / 444 paired views. Train107, val27, test88; 2/1/1 source signers tách biệt.
Target: loại functional NEG/WH/YN được ghi dương trong nguồn. Không suy diễn affirmative/absence, không dùng gold core intervals làm input pooling.
Mỗi head linear768→3 có2.307 tham số, cùng100epochs và3LR candidates; val chọn head, test không chọn cấu hình. Trung bình seeds0/1/2.

| Readout | Macro recall mean | Conditional sequence bootstrap95%CI | Training seed SD |
|---|---:|---|---:|
| body_mean_linear | 0.6094 | [0.5161732803927058, 0.7024694184315894] | 0.0342 |
| face_mean_linear | 0.5769 | [0.4822420634920634, 0.6695795673391459] | 0.0310 |
| paired_normalized_mean_linear | 0.5598 | [0.4738806249813851, 0.6472505210838544] | 0.0222 |

Predefined paired-minus-body delta: -0.0496, CI [-0.15079448701575138, 0.0409574230678882]; chưa thấy improvement được hỗ trợ. Prior macro recall0.3333.
Val chỉ2YN/4NEG, test chỉ một signer; camera color/resolution khác nhau. Không diễn giải view comparison thành causal facial-cue isolation hoặc encoder mất ngữ pháp.
Source sequence và exact decoded-video overlap bằng0; bốn XML aggregate groups vẫn cross-split. Historical release chưa xác nhận byte-identical với bản2011.
Nguồn: `reports/ncslgr_annotation_audit/report.json`, `reports/ncslgr_media_audit.json`, `reports/ncslgr_input_preparation.json`, `runs/ncslgr_diagnostic_v1/summary.json`.
Post-hoc sensitivity bỏ test XML groups có mặt ngoài test: 27 utterances, class counts {'NEG': 8, 'WH': 5, 'YN': 14}, macro recall {'body_mean_linear': 0.6849206349206348, 'face_mean_linear': 0.4932539682539683, 'paired_normalized_mean_linear': 0.6261904761904762}. Giữ nguyên trained heads; không retune hoặc thay primary test.
Audit prediction coverage/gold/metadata đã PASS; 30 fixed error IDs chưa có expert linguistic review. Nguồn: `reports/ncslgr_diagnostic_audit.json`.

## Recorded functional-core interval diagnostic

Protocol mới khóa trước interval metrics, sau khi đã xem classification test cùng cohort: exploratory, chưa phải xác nhận độc lập.
222 utterances dùng classification; 179 single-event utterances dùng endpoint loss/evaluation (89train/25val/65test). 43multi-event utterances không bị coi là absent.
Bốn readouts body/face × global/temporal, mỗi head6.915activeparameters; cùng3LR ×100epochs ×3seeds. Functional gold là supervision, không vào forward inputs.

| Readout | Macro class-aware interval IoU | Conditional95%CI |
|---|---:|---|
| body_global | 0.3239 | [0.2509479254109823, 0.3989185506352308] |
| body_temporal | 0.3406 | [0.27060364930742603, 0.4110553415802012] |
| face_global | 0.2358 | [0.1772228989690463, 0.2971162210211715] |
| face_temporal | 0.2580 | [0.1950295436702359, 0.31815413175131535] |

Primary body temporal−global: 0.0167, CI [-0.022397251418940715, 0.05262067346314116]; chưa có bằng chứng improvement được hỗ trợ.
Boundary position distributions lưu theo timestamp nguồn; không phải xác suất presence/absence và chưa calibrated. Source functional-core khác complete grammatical scope.
Nguồn: `configs/protocol_ncslgr_intervals_v1.json`, `runs/ncslgr_intervals_v1/summary.json`.
Machine audit: 528boundaryfiles và12predictiontables khớp originalclock/targets; zeroASLexpertreviews.
Post-hoc same-classifier duration-prior sensitivity: priorIoU 0.4224, temporalIoU 0.3406, delta -0.0818, CI [-0.11449766340166903, -0.05234426638201324]. Prior endpoints chỉ fit từ train; không thay primary hoặc chọn head bằng sensitivity này.
Head hiện tại chưa vượt prior đơn giản trong sensitivity; không suy diễn encoder thiếu ngữ pháp hoặc thêm attention là novelty.

## Public-demo translator local smoke

SHuBERT-author-demo-11625: PASS_LOCAL_PUBLIC_DEMO_NUMERICAL_SMOKE; CPU, strictload không missing/unexpectedkeys.
Nguồn caption: Has mother finished reading that book?; meanNLL 1.474203, sumNLL 57.493935, 39ByT5tokens cóEOS.
Padding=-100 giữ logits token hợp lệ trong maxdelta 1.43e-06. Generation49token budget: The beat is starting to fold.
Generation khác caption nguồn. Đây là numerical smoke một clip train-signer, không phải semantic pair accuracy, external test hoặc grammar comprehension.
Embedded demo encoder khác213/226W02tensors; không dùng kết quả decoder để kết luận W02 mất thông tin. Runtime torch2.1.1/transformers4.30.2 là tested compatibility, không bitwise author reproduction.
Nguồn: `features/shubert_translation_smoke_attempt1/report.json`, `reports/shubert_demo_source_audit.json`, `provenance/W08.json`.

## Categorical probability sensitivity

Temperature chọn bằng validation NLL trên sáu giá trị khóa trước; dùng lại chín head và normalization train đã lưu. Test predictions/probabilities gốc được tái tạo và đối chiếu.
Body mean NLL: 13.197945 → 1.901478; delta -11.296467, conditional95%CI [-15.858035037185479, -7.11493534986416].
Cả chín head chọn temperature8, ở biên trên grid; đây là sensitivity trong grid hữu hạn, chưa chứng minh xác suất đã calibrated. Class argmax không thay đổi.
ECE dùng năm equal-width reliability bins với counts lưu đầy đủ. Validation chỉ27utterances/một signer và đã dùng chọn LR trước đó; test đã được quan sát. Không calibration boundary/presence hoặc method novelty.
Nguồn: `configs/protocol_ncslgr_calibration_v1.json`, `runs/ncslgr_calibration_v1/summary.json`.

## Native NCSLGR controls

CPU preprocessing: PASS_NATIVE_PREPROCESSING, 222/222; native CPU features: PASS, 222/222.
Equal-budget function suite: PASS_EXPLORATORY_EQUAL_BUDGET_NATIVE_FUNCTION_CONTROLS. SignRep, native last FFN và fixed12-layer average; 18heads cùng6915parameters,3seeds,3LR,100epochs.
Đối chứng duration prior chỉ học interval fractions từ train. Chung eligible IDs; ghi mọi exclusions và dừng nếu lớp mất support. Không đánh giá partial native test hoặc thêm GPU worker.
Protocol: `configs/protocol_fair_ncslgr_v1.json`; pilot/cache provenance và timestamp erratum: `reports/native_ncslgr_cpu_pilot_audit.json`.

Strict common population: 175/222utterances; 47exclusions. Train65/val27/test83; interval test61. Giữ mọi lớp nhưng population khác diagnostic222gốc.
| Readout | Interval IoU | Train-duration prior IoU | Classification macro recall |
|---|---:|---:|---:|
| signrep_global | 0.2336 | 0.3156 | 0.5192 |
| signrep_temporal | 0.2180 | 0.3107 | 0.5107 |
| shubert_last_global | 0.2099 | 0.2974 | 0.4036 |
| shubert_last_temporal | 0.1444 | 0.2874 | 0.4028 |
| shubert_average_global | 0.1838 | 0.2892 | 0.4462 |
| shubert_average_temporal | 0.2007 | 0.3046 | 0.4786 |
Primary SHuBERT last temporal−global: -0.0655, conditional95%CI [-0.1284533547851308, -0.011053615775708271]. Temporal readout không vượt global; cả sáu endpoint readouts thua prior học từ train.
Native cues/context khác SignRep; một test signer và selection validation nhỏ. Kết quả không chứng minh encoder thiếu grammar hoặc một adaptation mới.

## Author-imputed pooling sensitivity

Protocol riêng giữ toàn bộ native contextual tokens trong utterance support, kể cả carry-forward/black crop inputs. Mask quan sát không đổi; không gọi imputed là observed. Full222utterances107train/27val/88test,9heads2307parameters.
Đã khóa sau khi strict native result files tồn tại; lựa chọn dựa trên train missingness. Không claim blind/independent evaluation.
| Frozen backbone | Macro recall | Conditional95%CI |
|---|---:|---|
| signrep | 0.6094 | [0.5161732803927058, 0.7024694184315894] |
| shubert_last | 0.4122 | [0.33547744430366383, 0.48568288103771967] |
| shubert_average | 0.5777 | [0.5151664116110336, 0.6389375736772761] |
Primary last−SignRep: -0.1972, CI [-0.3184143872914198, -0.07052498314140487]. Fixed12-layer average là secondary; không chọn best-test-layer làm primary.
Audit `reports/native_ncslgr_controls_audit.json`:222source clocks/caches,27savedheads và747boundaryfiles đã đối chiếu; zeroexpertreviews.

Native lexical extraction terminal FAIL do reserved disk limit tại1235/4726features,1237preparedclips. Equal-grid lexical waiter cũng terminal FAIL trước fitting; không có native lexical metrics.
Lần dừng và logs được giữ ở `evidence/native_common_disk_stop_v1/`. Reviewed recovery cần khoảng60GiBfree; giữ21GiBguard và caches, không giảm reserve hoặc xóa dữ liệu khác.

Reviewed recovery hiện tại: CPU_PREPROCESSING_RECOVERY. CPU preprocessing phải hoàn tất trước soleGPU0feature consumer, rồi same locked equal-grid suite trong `runs/fair_common_v1_recovery1`.
Các terminal failures ở trên thuộc attempt trước; recovery giữ nguyên attempts/logs, cache identities và disk guard.

Semantic-pair CLI readiness: BLOCKED_GOLD_OR_ACCESS. `scripts/run_semantic_pairs.py` đã triển khai; candidate ordering/ties, recording donors và utterance aggregation đã unit-test.
Dry-run dừng trước model loading vì thiếu reviewed canonical gold/media/cue caches. Không có external semantic accuracy hoặc full-model pair integration đã được kiểm chứng.

Semantic backend numerical integration: PASS_SEMANTIC_BACKEND_NUMERICAL_ONLY; 104 frame, 13.01 giây CPU.
Tái tạo NLL caption nguồn 1,474203467/token và tie credit 0,5 khi hai candidate giống nhau. Blank streams chạy hợp lệ. Một clip từ signer train; chưa có distinct semantic-pair gold hoặc external accuracy.

Q02 CLI readiness: BLOCKED_GOLD_OR_MAPPING. Subsequence-DTW, timestamp gốc, NMS, khớp interval một-một, macro/pooled AP, false positives/minute, boundary error và recording bootstrap đã có kiểm thử tổng hợp.
Cấu hình thật dừng trước search vì chưa có mapping/occurrence/absence/grouping xác minh. Fixture tổng hợp không cung cấp gold hoặc metric grounding thực tế. Xem `reports/track_requirements_audit_v1.md`.

A03 cue-intervention extraction: RUNNING, 26/175 utterance trong tập chung đã khóa; CPU-only, tám điều kiện.
Giữ nguyên video/crop/context/observation mask và head budget. Zeroing là distribution-shift diagnostic, không causal branch importance. V2 dừng ở absolute cross-cache guard; v3 dùng baseline all tính lại trong cùng lượt và kiểm tra repeat với ngưỡng1e-5. Không chứng nhận tương đương số học với cache cũ.
Các lỗi/import, smoke và numerical diagnosis được lưu trong evidence và registry. Chưa có metric ablation khi extraction chưa đầy đủ.

## Grammar, grounding và novelty

Người dùng xác nhận chưa có annotation gold bổ sung ngoài datasets đã liệt kê.
ASL-MTP CSV có 1.275 rows, 48 raw video IDs và 938 URLs; raw video IDs không phải canonical utterance IDs.
Pilot 3 public DAI pages đã resolve metadata cho 5 pairs, nhưng video/linguistic review chưa thực hiện; chức năng download yêu cầu login.
ASL-MTP vẫn là external test; không train/tune trên gold candidates hoặc published NLL/BLEURT tables.
A-G1: PASS_FOR_INTERNAL_RECORDED_FUNCTION_DIAGNOSTIC_WITH_LIMITS; A-G2: PARTIAL_RECORDED_SINGLE_INTERVAL_DIAGNOSTIC_COMPLETE.
Adaptation: PENDING_MECHANISM_AND_INDEPENDENT_CONFIRMATION. External A-G0 và grounding B-G0/B-G1/B-G2 vẫn gated.
Nguồn: `reports/gate_decisions.json`, `reports/asl_mtp_metadata_audit.json`, `reports/asl_mtp_resolution_pilot.json`, `reports/annotation_request.csv`.
P35/P36/P37 đã nhận dạng chức năng ngữ pháp, temporal relations, core/onset/offset và timing rules. P26 dùng frozen motion encoder cùng readout cho non-manual lexical distinctions. Prior art cũng đã bao gồm non-manual cue analysis, phonological subspaces/minimal pairs, headshake detection, query-by-example spotting, DTW, retrieval và QA. Không claim first từ các task/component này.
Câu hỏi còn kiểm chứng: encoder information-access so với decoder utilization, và functional scope-sensitive adaptation có lexical retention. Local recorded-function diagnostic đã có; chưa có method gain hoặc external semantic confirmation.
Nguồn: `evidence/literature_matrix.csv`, `reports/novelty_matrix.csv`.

## Kiểm tra và giới hạn

63 integrity tests: PASS. Coverage gồm grouping/view leakage, path escape, timestamps, padding/masks, matched-capacity heads, cache invalidation/corruption, ambiguous DAI mapping và incomplete PDF detection.
P04 explicit-version PDF ban đầu chỉ trả 4 MiB; đã quarantine. Canonical PDF có đúng v3 trong tài liệu nhưng checksum khác ETag của explicit-version HEAD; giữ FAIL, không đổi expected hash để bỏ lỗi. Claims P04 dùng HTML v3 chính thức, không dùng page anchors của PDF lỗi.
37 nguồn đã có mức đọc khai báo: 17 core, 17 expansion trong kế hoạch và ba prior functional non-manual cũ. Mức đọc gồm phần liên quan của bài hoặc trang tài nguyên chính thức; chưa phải toàn văn mọi paper hoặc acquisition/PDF integrity hoàn tất.
Môi trường SignRep dùng project venv kế thừa các package đã có; đã khóa phiên bản quan sát và không cài vào conda base. Không claim bitwise reproduction.

## Task states

| Task | Status | Lý do/phần còn lại |
|---|---|---|
| T00 | PASS | Read-only bounded inventory complete; GPU 0 explicitly granted; disk reserve now passes. |
| T01 | PARTIAL_REQUIRED_SOURCE_SCOPING_COMPLETE | 37matrixentries:17core +17listed expansion +3older closest functional priors. All listed sources have relevant declared reading scope, including How2Sign arXivv2dataset methods/grouping and supplement reading. Full source/PDF acquisition and remaining integrity/venue checks are not certified complete; no exhaustive/full-read claim. |
| T02 | PASS | Seven available dataset formats normalized and all 214613 rows schema-validated. Missing videos, unknown recording IDs and unverified CSL crop lineage remain explicit coverage limitations, not gold claims. |
| T03 | PASS | Gate decisions updated after public historicalNCSLGRsource discovery; limited internal recorded-function protocol locked. External semantic/B query/negative gates remain blocked; source/pretraining/external overlap limitations explicit. |
| T04 | PASS_LOCAL_TRANSLATOR_NUMERICAL_SMOKE | C04 source/configs and W08checksum verified. Custom ByT5CPU strict load, real104frame C04cue pipeline,39token source-caption NLL/padding and boundedgeneration PASS. W08encoder differs213/226W02tensors; no exact paper model/semantic-pair accuracy claim. Separate env freeze saved. |
| T05 | IN_PROGRESS_REVIEWED_NATIVE_LEXICAL_RECOVERY | NativeNCS222fullcache/auditcomplete. Originalnativelexical diskguard failurespreserved1235features/1237prepared. Capacityfreed; reviewedrecovery nowCPUpreprocessing RUNNING, then soleGPU0consumer onfullpreparation. Samehashes/reserve; noextraGPU. |
| T06 | WAITING_FOR_REVIEWED_NATIVE_LEXICAL_RECOVERY | Original8SignRep runscomplete. Oldfair_commonwaiter pre-fittingFAIL retained. Same locked equal-grid controls scheduled inruns/fair_common_v1_recovery1 only afterfullnativePASS. |
| T07A | PARTIAL_INTERNAL_NATIVE_FUNCTION_CONTROLS_COMPLETE | Native222CPUcache PASS. Strict18equal-budget controls completed on175eligibleutterances65train/27val/83test;61intervaltest. Native temporal minus globalIoU-.06551CI[-.12845,-.01105]; allsixlearnedreadouts belowtrain-durationprior. Separate9head author-imputed sensitivity full222done; masks audited. Nofullscope/absence/independentsemanticgold. ExternalsemanticCLIimplemented; scoring/accountingtestspass, realgold/media readinessdryrunblocked; noA01accuracyclaim. New semantic CLI backend numerical integration PASS on one 104-frame training-signer source caption, repeated text tie and blank streams; no distinct semantic-pair/external accuracy. A03 eight same-source interventions now extracting CPU on exact175common cohort after two training-only numerical smoke cases. Original-cache equivalence unestablished; v1 import and v2 guard failures archived. No ablation classification metric until full extraction then24matchedheads. |
| T07B | BLOCKED_DATA | Q02 standard CPU subsequence-DTW and metrics implemented; eight synthetic tests including complete CLI integration PASS. Actual readiness stops before search with BLOCKED_GOLD_OR_MAPPING. Verified ASL lexeme mapping, exhaustive occurrences/absence and resolved query/target signer/recording groups remain missing; Q00/Q01/Q03/Q04/Q05 not evaluated. |
| T08 | PENDING_MECHANISM_AND_INDEPENDENT_CONFIRMATION | Nativefunctioncontrols complete without supportedtemporalbenefit; allsixendpointreadouts belowpredeclaredtrainprior. Author-imputedpooling sensitivity doesnotestablishcausalencoderadvantage. Noadaptationchosen; independentmechanism/goldconfirmation and lexicalretentioncontrols required. |
| T09 | PENDING | SignRep common paired recording bootstrap, historical NCSLGR classification/recorded-core interval bootstraps, boundary integrity and train-duration-prior sensitivity complete. Fixed error cases have zero expert reviews. Native functioncontrols completed; lexical recovery live in CPU preprocessing; external confirmation and ASL expert review pending. Validation-temperature categorical sensitivity completed on9immutable heads; all9choices at upper grid boundary, original predictions preserved. Concrete30case interval review packet prepared; proposed fixed10dual-review subset; all expert fields still pending, no external contact. Native222cache,27savedheads and747boundaryfiles auditPASS; strict/imputedpopulationdifferences retained. |
| T10 | PENDING | Interim reporting only; final synthesis, claim ledger completion and requirement-by-requirement audit remain. |

## Expert review và quyết định method

Đã chuẩn bị hồ sơ 30 trường hợp; expert reviews hoàn thành: 0. Form và hướng dẫn đề xuất: `reports/ncslgr_expert_review_v1/`.
Quyết định interim: `reports/method_decision_v1.md`. Chưa có representation adaptation hoặc novelty được kiểm chứng.

## Tiếp tục và tái lập

Xem `REPRODUCE.md`, `state/tasks.json`, `state/common_suite.json` và `runs/experiment_registry.csv`.
Revalidate PID/job handles trước khi tiếp tục; không restart extraction đang live.
Việc tiếp theo: reviewed disk recovery cho native lexical cohort và equal-grid controls trong output riêng; native NCSLGR và function controls đã hoàn tất. Còn semantic pairs độc lập, adjudication cho đầy đủ grammatical scope/absence và kiểm chứng cơ chế trước method adaptation.
Claim ledger: `reports/claim_ledger.csv`; chi phí native: `reports/shubert_native_pilot_cost.json`; resource plan: `reports/resource_plan.md`.
Completion audit chưa đạt. Không task/claim blocked nào được chuyển thành PASS chỉ để báo hoàn thành.
