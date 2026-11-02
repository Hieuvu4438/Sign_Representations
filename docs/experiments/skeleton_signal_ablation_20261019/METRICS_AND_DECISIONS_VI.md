# Đo chỉ số gì và dùng kết quả để quyết định thế nào?

## 1. Primary: frozen gloss-level dictionary retrieval

Với query video q, gallery clip g, embedding clip z được L2-normalize. Gallery là ASLC official train, query phát triển là official val. Không train retrieval head bằng gloss trong primary evaluation.

```text
clip_similarity(q,g) = cosine(z(q), z(g))
gloss_score(q,c) = max clip_similarity(q,g), với g thuộc gloss c
rank(q) = 1 + số gloss khác có score >= score của gloss đúng
```

Rank theo **distinct gloss**, không lấy top5 video rồi đếm trùng gloss. Ties pessimistic là contract mới; ghi rõ khi so code cũ có gallery-order ties. Không fit normalizer/PCA bằng val/test. Primary dùng raw256 + L2, không whitening/activity weighting.

| Metric | Tính trên query hợp lệ có rank r | Ý nghĩa |
|---|---|---|
| **R@1** | `mean(r <= 1)` | Primary để so/chọn arm |
| R@5, R@10 | `mean(r <= K)` | Correct gloss có trong shortlist không? |
| MRR | `mean(1/r)` | Phạt lỗi xếp hạng gần đầu |
| DCG | `mean(1/log2(1+r))` | Thứ hạng gloss đúng theo contract một relevant entry |
| Macro-gloss R@1 | Mean R@1 từng gloss có query | Tránh gloss nhiều query chi phối |
| Macro-signer R@1 | Mean R@1 từng signer có query | Độ đồng đều giữa signer |
| Per-signer/per-gloss | Metric và query count từng group | Tìm gain chỉ đến từ vài group |

R@K/DCG/MRR trong JSON lưu **0–1**. Bảng đọc dùng R@K×100, DCG×100; MRR giữ0–1. Delta R@1 báo **điểm phần trăm**:0,17−0,16 =+1,0pp, không phải+1% relative.

**Failure contract:** giữ denominator 10.304 val /32.941 test. Query không trích được embedding hữu hạn/khác zero: lưu `failed=true`, rank để trống, đóng góp **0** cho tất cả retrieval metrics. Không ép rank giả rồi tính DCG khác0. Gallery lỗi không dùng cosine; ghi nominal/effective gallery và class coverage. Query có gloss không còn valid gallery cũng là failure với contribution0. Không chỉ report metric trên successful queries; metric conditional-on-success nếu có là phụ.

Lưu cho mỗi query: `sample_id`, split, signer, gloss, valid annotation code nếu có, failed/reason, rank, top10 gloss/scores, true-gloss score, top competitor, valid fraction từng stream. Giữ full ID list để paired comparison.

Không gọi trained classification accuracy là training-free retrieval. Không trộn Chamfer/DTW nhiều token với256-dim vector score mà không ghi scorer/cost. Gallery cân bằng số clip/gloss là sensitivity analysis riêng, không thay official gallery sau khi thấy kết quả.

## 2. Diagnostic theo từng loại tín hiệu

| Tín hiệu | Diagnostic độc lập nên đo | Metadata/probe có sẵn | Cách diễn giải |
|---|---|---|---|
| H: handshape | Handshape/flexion/selected-fingers probe; verified contrast pairs | ASL-LEX categorical fields | Probe tốt là thông tin đọc được, chưa bảo đảm cosine dùng đúng |
| L: location | Major/minor location probe; contrast pairs | ASL-LEX location | Kiểm tra H-only vs H+L và hand-location perturbation |
| M: motion | Movement/repetition probe; speed/reversal/block-order diagnostics | ASL-LEX movement, repeated movement | Raw derivative R² chỉ là measurement fidelity, không là semantic ground truth |
| B: bimanual | SignType/non-dominant handshape probe; desync degradation curve | ASL-LEX sign type/handshape | Tách bimanual relation khỏi explicit velocity |
| Fmouth/Fother | Expression perturbation; expert-verified non-manual subset nếu có | **Chưa có gold non-manual đầy đủ được xác minh** | Không suy ra face vô ích từ toàn bộ val average |
| O: orientation | UlnarRotation probe; direction-sensitive contrast nếu được kiểm chứng | UlnarRotation chỉ là proxy hẹp | Không claim đầy đủ orientation3D từ2D proxy |
| Temporal | Correct-order vs reordered triplet preference, ΔR@1 theo block size/lag | Human-verified order-sensitive subset nếu có | Không yêu cầu mọi sign đều nhạy reversal |
| Composition | Held-out code/feature-combination retrieval | Code, compound và attributes sau audit | Strict unseen-lexeme cần encoder chưa pretrain thấy các lexeme đó |

### Probe protocol

- Frozen encoder; classifier linear logistic/ridge trên **train embeddings**, đánh giá val embeddings; không fine-tune encoder.
- Categorical label: balanced accuracy và macro-F1, majority/chance baseline, label support, per-class counts. Không hồi quy scalar lên numeric category ID.
- Primary phonology probe tách lexical code80/20 bằng hash cố định trước xem kết quả: fit trên official train clips thuộc80% code; eval trên official val clips thuộc20% code. Hyperparameters chọn bằng grouped CV **bên trong probe train**. Không đưa `NA` thành code thật; loại khỏi probe, giữ trong retrieval.
- Encoder E1 vẫn đã thấy **unlabeled** train clips của toàn vocabulary: lexical-heldout **probe** kiểm tra readout generalization, chưa phải strict unseen-lexeme representation learning. Track strict phải pretrain lại sau khi loại code/variant families.
- Fit standardization/PCA trên probe train; một nonlinear MLP nhỏ, cùng budget giữa encoder, là diagnostic phụ nếu linear probe yếu. Không dùng nonlinear accuracy để thay primary retrieval.
- Annotated contrast pairs: report cặp, gloss, signer counts và CI. Hai lexical entries khác một database attribute chỉ là **attribute contrast** cho đến khi variant/video được người biết ASL kiểm tra; không mặc định là linguistic minimal pair.
- Các259/178 minimal pairs thuộc evaluation trong literature không được dùng tune S0–S7 rồi gọi final independent test. Xem [nguồn và giới hạn](../../research/skeleton_signal_retrieval_20261019/LITERATURE_MATRIX_VI.md).

## 3. Perturbation: đo model dựa vào gì?

Đánh giá trên **val**, checkpoint frozen; clean gallery + corrupted query là chính. Both-sides corruption là phụ. Đăng ký trước severity; các giá trị dưới là grid ban đầu để triển khai, không phải benchmark chuẩn đã xác nhận semantic validity.

| Corruption | Grid đề xuất | Metric |
|---|---|---|
| Independent part missing | Drop LH/RH/body/face hoặc feature group | ΔR@1 + per-group coverage |
| Frame dropout | 10/30% observed samples | R@1 và failure rate |
| Coordinate jitter | σ0,005/0,01 shoulder widths | R@1 theo σ, handshape probe stability |
| FPS | Native→25/15/10 khi source hỗ trợ | R@1, latency; không upsample lại để giả native fps |
| Temporal reversal | Toàn clip | Movement/order slice preference; overall ΔR@1 là phụ |
| Block shuffle | Block0,16/0,32/0,64 giây | R@1 theo scale, input-validity audit |
| LH/RH desync | ±0,08/0,16/0,32 giây | ΔR@1 ở one/two-hand slices; mask edge padding |
| Mild time warp | Speed0,8/1,0/1,2 | Retrieval stability; không mặc định bảo toàn mọi aspect |
| Static-repeat | Mean observed pose lặp theo thời gian | Sensitivity, đối chiếu static-trained reference |
| Face neutralization | Neutral từ train stats, giữ rigid head | Face sensitivity, matched C_HEAD |

Recompute M/B/G sau biến đổi; recompute timestamps/validity. Mask joint phải lan sang derived feature phụ thuộc joint, cả smoothing/interpolation support. Không giữ velocity nguyên gốc sau shuffle/reverse.

Corruption là **model sensitivity dưới distribution shift**, không phải causal importance của một articulator trong ngôn ngữ. Đối chiếu retrained removal và matched capacity controls trước kết luận.

## 4. Collapse, nuisance và chi phí

| Nhóm | Chỉ số cần log | Vai trò |
|---|---|---|
| Training | SSL loss từng term, gradient norm, lr, teacher entropy, center norm, valid target count | Debug optimization; không dùng loss thấp để chọn best retrieval |
| Embedding | Effective rank, feature variance, random-pair cosine distribution, zero/duplicate rate | Phát hiện collapse; không xếp hạng model chỉ bằng rank |
| Nuisance | Signer classifier; score-only/duration-only retrieval | Tìm collection/style shortcuts |
| Quality | Missing fraction/longest gap theo part; wrist disagreement; gain theo quality quantiles | Tín hiệu yếu vì detector hay vì objective? |
| Cost | Total/active params, dense/active token count, peak VRAM, sec/update, total GPUh | Khớp budget và chi phí thực |
| Inference | Pose-cost policy, sec/video, bytes/video, gallery search latency | Lợi ích thực tế so với thêm module |

Signer probe phải chia clips sao cho signer classes xuất hiện cả hai phía, nhưng group source/recording và cân bằng gloss khi có thể. Main train-gallery/val-query không có cùng signer, nên **same-signer neighbour rate ở đó luôn không trả lời nuisance question**.

Pose preprocessing cost báo riêng; cached pose inference không đại diện end-to-end camera latency. Benchmark timing trên cùng thiết bị/batch/coverage. RankMe nếu dùng: singular values σ của centered embeddings, `p=σ/sum(σ)`, `exp(-sum(p log p))`, ghi N/D và sample IDs.

## 5. So sánh và luật quyết định

**Screening:**1 seed, fixed endpoint10k; chọn candidate khám phá, không kết luận có ý nghĩa từ1run. **Confirmation:** ít nhất3 paired seeds, fixed40k endpoint và matched controls; báo raw từng seed.

So sánh chính đăng ký trước: S2−S0, S2−S1, S3−S2, S4−S2, S5−S2, S6−S2, S7−S2. Capacity contrasts được report riêng trong family matched-controls. Mọi screening comparison đều exploratory; không lấy CI hẹp qua query để che training variance.

CI95%: paired query outputs, cluster theo signer và gloss; thêm sensitivity theo valid code/variant groups. Với crossed cluster bootstrap có thể resample signer/gloss độc lập và nhân trọng số query, renormalize metric. Báo seed variance riêng; seed-resampled CI với3seeds chỉ là ước lượng yếu. Holm correction trong confirmation comparison family đã đăng ký.

Val có6signers, P13 chỉ2clips: báo macro gồm đủ6 và sensitivity trên5signers ≥100queries, không âm thầm bỏ P13. Leave-one-signer-out effect và grouped holdout từ train là kiểm tra stability bổ sung, không nhập vào official val metric.

Practical margin ban đầu: **δ=1,0 điểm R@1**, khóa trước confirmation; có thể thay sau noise pilot nhưng phải trước xem candidate confirmation. Giữ δ và cost criterion như nhau giữa candidate.

| Evidence | Quyết định |
|---|---|
| Mean gain≥δ, CI hỗ trợ dương, stable seeds, thắng matched control, cost chấp nhận được | PROMOTE để full-budget confirmation; chưa claim lower-bound>δ nếu CI chưa vượtδ |
| CI lower bound>δ | Bằng chứng mạnh hơn cho gain thực dụng vượtδ trong phạm vi evaluation |
| Probe/perturbation đúng cơ chế nhưng retrieval gain chưa rõ | MECHANISM_ONLY; chưa chọn vào final model |
| CI rộng, seed/signer đảo thứ hạng | INCONCLUSIVE; thêm replication hoặc sửa measurement |
| Equivalence interval nằm trong ±δ | Không có gain thực dụng trong recipe/budget đang xét |
| Có hại ổn định, hoặc extra branch giải thích hết gain | DROP phiên bản module; không loại cả loại tín hiệu nói chung |

Chấp nhận extra cost ở v1: candidate phải báo gain và cost ratio, không áp một ngưỡng cost giả định trước profiler. Trước confirmation, ghi trần latency/storage theo profiler và nhu cầu thực; metric chưa đo phải để trống.

## 6. Bảng kết quả cần xuất

Mỗi arm/seed/update: status, input signals, objective, train source/split, gallery/query counts, embedding/readout, R@1/5/10, MRR, DCG, macro metrics, failure rate, checkpoint hash, GPUh, latency/storage. Mỗi contrast: delta, CI, seed mean±SD, per-signer effects, matched control, diagnostic coverage, decision và lý do.

Tổng hợp cuối là **signal × only × add × retrain-remove × perturb × phonology × cost**, không chỉ một leaderboard. [results_template.csv](results_template.csv) giữ metric trống cho đến khi runner sinh số đo thật.
