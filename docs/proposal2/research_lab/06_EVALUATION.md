# 06 — Protocol đánh giá: gate bắt buộc

## Nguyên tắc

Không chạy phép đo nghiên cứu chính nếu chưa có protocol card và evaluator được xác minh. Code public là reference cần audit, không phải bảo chứng không lỗi. Ưu tiên evaluator chính thức cho đúng dataset/task/version; nếu không có, dùng implementation tác giả hoặc widely used reference khớp thuật toán. Chỉ implement metric mới khi cần thật và có specification cùng oracle fixtures.

Tìm bằng official project/paper/code links, rồi GitHub search theo dataset + metric + evaluation. Ghi URL, repository commit, file path/symbol, dependencies, license và local wrapper hash. Không dùng evaluator repo khác chỉ vì trùng tên metric.

## Protocol card phải khóa gì

Tạo [card](templates/PROTOCOL.md) cho mỗi dataset × task × regime × cohort. Ghi task/input/output, allowed supervision, pretraining overlap, split/label ontology, cohort hash, missing samples, preprocessing, timestamp convention, inference crop/ensemble/beam, prediction schema, primary/secondary metrics, scorer options, aggregation, tuning/selection rule, seed plan, uncertainty unit và comparability class.

Comparability class:

- `official-comparable`: cùng release/cohort, split, supervision/regime, metric, preprocessing/scoring và inference conditions cần thiết; differences disclosed.
- `adapted-protocol`: có thay đổi đã version; so sánh nội bộ fair nhưng không gọi hơn official benchmark.
- `diagnostic`: smoke, custom corruption hoặc exploratory slice; không dùng làm evidence leaderboard.

Số tham khảo từ paper có nguồn ghi `reported`; baseline chạy trong lab ghi `reproduced`. Không gộp hai loại thành một ranking khi khác protocol/resources.

## Evaluator validation workflow

1. Đọc specification và code thực tính metric, không chỉ README/tên hàm. Audit input ordering, mapping, handling missing/empty predictions, normalization, tie break, aggregation và denominator.
2. Tạo predictions fixture biết trước, gồm perfect, all-wrong, ties, missing/duplicate IDs, empty output và corner cases đúng task.
3. Chạy upstream reference và wrapper trên cùng fixture. Exact match cho integer counts; tolerance đăng ký cho float. Wrapper không được âm thầm thay metric.
4. Một phép tính độc lập bằng tay hoặc implementation nhỏ khác xác minh fixture phân biệt metric variants. Parity với cùng code bị copy không phải oracle độc lập.
5. Nếu có released predictions/checkpoint với môi trường tương thích, replay scorer/reproduce dev reference. Thiếu artifact đó không tự ngăn synthetic oracle, nhưng phải ghi rằng reproduction score chưa được xác minh.
6. Tạo parity report: fixture IDs, expected/observed, version/hashes, tolerance, pass/fail và caveats. Protocol Auditor ghi `VALIDATED` hoặc `BLOCKED`.
7. Run khoa học có assertions: prediction IDs đúng cohort, uniqueness/ordering đúng schema, labels hợp lệ, values finite và coverage hợp lý. Lỗi không được nuốt để tạo điểm.

Không lấy test predictions của dự án để calibrate metric, chọn normalization hoặc threshold. Dùng fixtures hoặc train/dev samples. Khi đổi scorer version/options, tăng protocol version và invalidate phép so sánh bị ảnh hưởng.

## Checklist theo task

### Isolated sign recognition

- Ghim subset/vocabulary (WLASL100/300/1000/2000 hay local cohort) và label map; không renumber class khác nhau giữa train/test.
- Top-k per-instance và mean-per-class là hai phép đo khác nhau. Macro denominator/classes with zero support phải đúng protocol; không dùng `nanmean` để che cohort thiếu lớp.
- Define k khi số lớp < k, ties và multi-view prediction aggregation. Accuracy classification không tự đổi thành retrieval recall.
- Report N test, class support, top-1/top-5 per-instance và macro khi protocol yêu cầu; không chọn metric tốt nhất để kể chuyện.
- Fixture phân biệt macro/micro: class A có 9 mẫu đúng hết, B có 1 mẫu sai → micro top-1 = 0.9, macro top-1 = 0.5 trên hai lớp có support. Dùng fixture này để bắt aggregation sai, không thay rule official về lớp thiếu.

Reference đã kiểm tra sự tồn tại: [WLASL evaluator](https://github.com/dxli94/WLASL/blob/master/code/I3D/test_i3d.py). Phải ghim commit và audit lại trước dùng; downstream head/inference của nó không được áp nguyên cho encoder khác.

### Dictionary/video retrieval

- Khóa query/gallery IDs, gallery train-only hay external dictionary, class/lexeme relevance, seen/unseen classes, cross-signer/corpus và self-match exclusion.
- Ranking các video instances khác ranking các unique classes/prototypes. Collapse multiple same-label entries, aggregation scores và gallery duplicates phải đúng reference, không tự đổi để tăng Top-k.
- Recall@K với multiple relevant items, hit@K và classification top-k có thể khác nhau. Chỉ gọi cùng tên khi specification trùng.
- MRR xét first relevant rank; DCG phải rõ relevance gain, discount, cutoff và có normalize hay không. **DCG không được thay bằng nDCG** chỉ vì library có sẵn.
- Fixture một relevant item ở rank 3: MRR = 1/3; DCG variant dùng binary gain và `1/log2(rank+1)` cho giá trị 1/2. Đây là ví dụ của variant, không áp mặc định cho mọi benchmark. Chọn expected value từ official scorer.
- Tie rule deterministic; report gallery size, relevance coverage và query không có relevant item xử lý ra sao. Approximate nearest neighbor cần recall/parity kiểm chứng với exact search nếu ảnh hưởng metric.
- Nếu class mapping giữa corpus chưa đáng tin, không chạy cross-corpus metric. English gloss overlap chưa đủ xác nhận lexeme.

Reference: [ASL Citizen testing code](https://github.com/microsoft/ASL-citizen-code/blob/17f0148b00ef87a5957ec103da8d10bd8eb1fa23/I3D/aslcitizen_testing.py). File này có classification ranking evaluation; dùng để truy đúng metric, **không xem là complete dictionary retrieval protocol**. Agent phải fetch hàm `eval_metrics` và chuẩn gallery khi chọn task.

### Continuous recognition

- Gloss sequence WER: `(S + D + I) / N_ref`, với tokenization, blank/duplicate collapse, lexicon và normalization đúng dataset. Corpus WER tính tổng edit counts/tổng reference tokens, không lấy mean sentence WER nếu reference không làm vậy.
- Ghi decode greedy/beam, LM/lexicon supervision, output length constraints. Không đưa written-language translation tokens vào gloss evaluator.
- Perfect sequence cho 0 WER; toàn deletion cho 1 khi reference không rỗng; insertions có thể làm WER > 1, không clamp. Empty reference cần rule chính thức.
- PHOENIX-2014 và PHOENIX-2014-T phải phân biệt release/split/annotations. Nếu official preprocessing/scorer yêu cầu riêng thì giữ nó, không đổi sang library chung rồi tuyên bố comparable.

### Sign language translation

- Corpus BLEU, chrF, ROUGE hoặc BLEURT dùng đúng implementation/version/checkpoint/options của benchmark. SacreBLEU là nguồn public hữu ích cho signature, không tự làm điểm tương đương mọi paper cũ.
- Ghi tokenization (đặc biệt tiếng Trung), detokenization, case, punctuation, smoothing, effective order, number of references, scaling (0–1 hay 0–100), max generation length và beam.
- Sentence-mean BLEU không phải corpus BLEU. Không sửa reference/test text để nâng điểm.
- BLEURT/learned scorer: ghim checkpoint/version, input language applicability và giới hạn. Không coi semantic metric là đánh giá sign intelligibility trực tiếp.
- Perfect đủ dài vs empty output làm fixtures; kiểm tra ID order tránh predictions/reference lệch hàng. Nếu reference implementation lịch sử khác SacreBLEU, báo cáo hai tên rõ thay vì đổi metric ngầm.

Reference: [SacreBLEU](https://github.com/mjpost/sacrebleu), ghi signature và package version trong mỗi result.

### Temporal spotting, grounding hoặc detection

- Xác định annotation là sign, fingerspelling, sentence hay event; không dùng sentence boundary làm sign ground truth.
- Timestamp unit (s/frame), interval closed/half-open, offset sau crop/downsample, duration và matching class/event phải rõ.
- tIoU thresholds, score ranking, NMS, one-to-one matching, AP interpolation, per-class/micro aggregation phải khớp evaluator task. Frame accuracy/F1 không thay cho temporal mAP.
- Fingerspelling sequence/detection có metric riêng; không thay bằng generic action mAP nếu benchmark dùng protocol khác.
- Fixtures: exact match, boundary offset biết trước, two duplicate predictions cho one GT, false positive, no GT/no prediction. Khóa threshold/NMS trên dev.

Chưa có official evaluator được xác minh cho setting temporal mới trong bộ rules này; agent phải tìm và validate trước measurement. “Có code chạy” không vượt qua gate.

### Phonological/non-manual, few-shot, robustness và efficiency

- Linguistic labels: missing vs `none` category phân biệt; annotate/join provenance; mask missing labels trong metric và ghi per-feature denominator. Leave-lexeme-out nếu claim vượt memorization lexeme; không suy hiểu grammar từ lexeme-level annotation.
- Few-shot: episodes/subsets cố định, train/dev/test pools và sampling units đúng; reuse cùng sampled support cho mọi method. Report variability giữa label subset và training seed riêng.
- Robustness: clean và corrupted results; corruption severity/schema/seed khóa từ dev. Không mặc định corrupted video vẫn giữ original label; case phá linguistic signal chỉ là stress test có limitation.
- Calibration: confidence, ECE binning, NLL/Brier, threshold selection và OOD definition ghi rõ; calibrator/threshold fit trên dev hoặc train, không test. Selective risk/coverage so sánh ở operating point đã đăng ký.
- Efficiency: hardware/dtype/batch/length, warmup, GPU synchronization, p50/p95 và preprocessing/encoder/head costs tách rõ. Cache-only throughput không gọi end-to-end throughput.
- Streaming: prediction tại thời điểm t chỉ dùng thông tin tới t + allowed lookahead. Feature từ bidirectional encoder trên full clip gây future leakage; sửa feature extraction hoặc thu hẹp claim thành offline compression.

## Protocol bị chặn

Ghi nguyên nhân, nguồn đã thử, thông tin còn thiếu và cách kiểm tra kế. Tiếp tục literature/synthetic fixture/feature debugging được phép. Không chạy hàng loạt benchmark để “lấy số trước rồi sửa sau”.
