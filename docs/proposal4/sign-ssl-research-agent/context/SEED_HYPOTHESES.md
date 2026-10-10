# Seed hypotheses — đề xuất chưa kiểm chứng

Đây là **HYPOTHESIS**, chưa có novelty judgment hoặc empirical result. Dùng làm câu hỏi nghiên cứu, không dùng làm shortlist cố định. T03 có thể loại tất cả sau khi đọc prior art.

## H-seed-A — Giữ động học cục bộ trong representation

**Câu hỏi:** Một mục tiêu predictive theo thời gian có giúp biểu diễn phân biệt dấu khác movement/timing tốt hơn một mục tiêu chỉ khớp hình học tĩnh?

**MVP đề xuất:** Giữ temporal tokens ở pretraining và dự đoán target vận động ở horizon ngắn: relative hand/body positions, velocity và inter-hand interaction changes. Chuẩn hóa theo timestamp/FPS; mask missing targets. Chỉ một prediction family ở thử nghiệm đầu.

**Prediction:** Nếu mechanism đúng, gains phải xuất hiện ở frozen representation và tập hard cases liên quan movement với labels/annotations hợp lệ; không chỉ sau fine-tuning.

**Falsifier/control:** So với cùng tokens/decoder/compute nhưng target position-only; thêm control smoothing và velocity-only baseline. Nếu temporal shuffling không làm giảm lợi ích hoặc extra-capacity control đạt ngang, cơ chế temporal chưa được hỗ trợ.

**Prior-art search:** Spatial-temporal consistency, masked motion prediction, sign cluster prediction, video latent prediction. Không coi thêm velocity loss tự động là mới.

**Rủi ro:** Pose derivatives khuếch đại jitter; horizons có thể không đúng boundaries; learned timing có thể học FPS/source shortcut. Time reversal dùng diagnostic riêng, không positive augmentation mặc định.

## H-seed-B — Reliability-aware supervision từ pose teacher

**Câu hỏi:** Calibrating độ tin cậy target có giúp RGB encoder học từ nguồn teacher nhiễu mà không chỉ bỏ vùng khó?

**MVP đề xuất:** Học với per-target weights/uncertainty có giới hạn và valid masks; chuẩn hóa theo valid weight mass. Không dùng teacher confidence như ground truth uncertainty khi chưa kiểm tra calibration.

**Prediction:** Gains tập trung ở nhóm thiếu/occluded targets có audit hợp lệ, đồng thời không giảm mạnh nhóm rare handshapes hoặc fast motion.

**Falsifier/control:** Hard threshold, soft confidence weighting và temporal smoothing là controls bắt buộc. Nếu learned uncertainty chỉ downweight examples khó hoặc simple weighting đạt ngang, contribution cần giảm/đổi.

**Prior-art search:** Confidence weighting, heteroscedastic regression, robust losses, pose distillation, teacher ensembles ở video/sign/pose SSL. Xác định explicit delta với baseline audit.

**Rủi ro:** Confident teacher vẫn sai; missingness correlated với groups; target-quality gains không chắc chuyển thành semantic representation.

## H-seed-C — Nuisance robustness giữ nguyên nội dung ngôn ngữ

**Câu hỏi:** Có thể giảm background/appearance shortcuts mà vẫn giữ location, orientation và non-manual signals không?

**MVP đề xuất:** Controlled nuisance interventions hoặc bounded consistency target; tách background manipulation khỏi articulator geometry. Chỉ dùng intervention sau semantic/sample audit.

**Prediction:** Cải thiện trên signer/domain shift đã định nghĩa; semantic hard cases giữ performance, không chỉ giảm identity probe accuracy.

**Falsifier/control:** Same-compute augmentation baseline, existing style regularization control, background-only shortcut probe và content retention diagnostics. Nếu invariance làm mất cues cần cho dấu, reject.

**Prior-art search:** Domain-adversarial representation, nuisance disentanglement, semantic-preserving augmentation và style losses. Signer-ID loss dùng metadata là nhánh supervision khác, không mặc định SSL-only.

**Rủi ro:** Probe identity kém không chứng minh thông tin bị xóa; body differences không luôn là nuisance; synthetic backgrounds có artefacts.

## H-seed-D — Non-manual information bổ sung có thể đo được

**Câu hỏi:** Mục tiêu facial/mouth dynamics có cải thiện representation ở dấu cần non-manual cues vượt lợi ích từ thêm crop/compute?

**MVP đề xuất:** Thêm một self-supervised facial dynamics target ở training; giữ RGB-only inference. Chọn target/teacher/data khả dụng, không dùng transcript hay lip-reading labels rồi gọi SSL-only.

**Prediction:** Gain trên benchmark/hard groups có non-manual relevance được xác thực, với same-head frozen eval và inference cost được đo.

**Falsifier/control:** Extra-parameters control, face-region masking diagnostic, generic visual target và teacher-quality audit. Nếu không có grouping/annotation đáng tin, chỉ claim overall benchmark gain; không gán nguyên nhân linguistic.

**Prior-art search:** Face/mouth/body multi-stream prediction, sign recognition non-manual cues và visual speech teachers. Đọc SHuBERT cùng nearest alternatives trước.

**Rủi ro:** Learning identity/mouthing shortcuts; teacher phát sinh supervision; ethical/data constraints; face crop không bảo đảm giữ ngữ nghĩa.

## Chọn hướng

So giá trị thông tin, scope, nearest-prior-art overlap, khả năng falsification và compute. Một MVP nên chỉ kiểm tra **một** cơ chế; không ghép A+B+C+D thành method ngay.

