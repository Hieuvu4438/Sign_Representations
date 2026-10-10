# ASLC002 — truy vết mức cải thiện retrieval thấp

Material Passport: **ANALYZED**. Phân tích hồi cứu ngày 2026-10-08; không train thêm, không sửa core, cấu hình, dữ liệu hoặc checkpoint của ASLC002. Các dấu hiệu dưới đây đã được đo; mức đóng góp nhân quả của từng yếu tố cần ablation training riêng.

Kết luận chính: pilot có tín hiệu học pose yếu so với regularization, discriminator mất phụ thuộc vào style, và activity thiếu supervision. Ngân sách nhỏ là yếu tố bổ sung; tăng epoch với nguyên cấu hình chưa có cơ sở bảo đảm khắc phục các vấn đề này. ASLC002 hiện chưa phải baseline đủ mạnh để kết luận module mới tốt hơn SignRep.

## Kết quả và phạm vi

300 lớp, 3.000 video train/gallery, 1.500 validation queries signer-disjoint; không dùng test. Hoàn tất 5.000 updates. Khởi tạo backbone bằng Kinetics MAE chính thức, projection/prior/activity heads mới; không dùng pretrained SignRep hoặc ASLC001. Step0 cũng có projection mới, vì vậy không phải phép đánh giá trực tiếp backbone Kinetics nguyên bản.

| Pooling | Top1 trước → sau | Top5 trước → sau | MRR trước → sau |
|---|---:|---:|---:|
| Mean | 1,600% → 1,667% | 6,000% → 6,600% | 0,048479 → 0,055724 |
| Activity | 1,467% → 1,800% | 5,733% → 6,133% | 0,047280 → 0,055187 |

Đây là metrics chính thức từ [RESULTS.json](../RESULTS.json). Bootstrap ghép cặp theo 300 class, 4.000 lần, giữ năm queries/class: CI95% cho mức tăng Top5 mean là **[−1,4; +2,6] điểm phần trăm** và MRR là **[−0,003935; +0,018547]**. Chưa đủ bằng chứng cho cải thiện ổn định. Đây là uncertainty có điều kiện trên dictionary, signer cohort và một training seed; không đo biến thiên giữa các lần train. Xem [paired_class_uncertainty.json](paired_class_uncertainty.json).

## 1. Regularization lấn át tín hiệu pose

Ở 1.000 updates cuối, trung bình reconstruction = **0,999**, variance = **270,767**, covariance sau nhân 0,004 = **126,914**. Reconstruction chiếm **0,251%** tổng scalar loss; gradient norm trước clipping khoảng **1.591**, trong khi clip norm = 1.

Scalar loss riêng chưa chứng minh gradient dominance. Bốn CPU probes với checkpoint cuối, input theo training, augmentation/drop-path/mask 80%, đo gradient tại embedding cho thấy:

| Probe | Norm gradient regularization / reconstruction | Cosine gradient reconstruction với tổng gradient |
|---|---:|---:|
| 1 | 365,75× | 0,0202 |
| 2 | 184,82× | −0,0336 |
| 3 | 147,19× | 0,0240 |
| 4 | 124,46× | −0,0132 |

Tín hiệu pose đóng góp rất ít vào hướng gradient embedding trong các probes này. Phép đo ở CPU float32, không replay chính xác minibatches GPU BF16 và không đo gradient toàn bộ backbone.

Thiết kế pilot dùng **4 views từ 2 video** để giảm VRAM, nhưng giữ hệ số/reduction regularization của recipe lớn. Đây là hạn chế của cấu hình pilot mình đã chọn. Với embedding 768 chiều, covariance của batch4 có rank tối đa 3; batch20 có rank tối đa 19. Yêu cầu std gần 1 đồng thời giảm toàn bộ off-diagonal khó thỏa mãn khi batch rất nhỏ. Ví dụ, nếu mọi chiều có sample std = 1, bất đẳng thức trace/rank cho weighted off-diagonal loss tối thiểu 440,64 ở batch4, so với 109,294 ở batch20. Đây là minh họa có điều kiện, không phải minimum của toàn objective.

Code: [losses.py:26](../../../src/signrep_baseline/losses.py#L26), [losses.py:52](../../../src/signrep_baseline/losses.py#L52), [training.py:110](../../../src/signrep_baseline/training.py#L110). Dữ liệu: [optimization_trace.json](optimization_trace.json), [gradient_signal_trace.json](gradient_signal_trace.json).

## 2. Loss pose thấp nhưng phụ thuộc yếu vào nội dung video

Trên bốn batches trên, reconstruction của model trung bình **1,05985**; predictor hằng theo thời gian, chỉ dùng trung bình priors từ train, đạt **1,09335**. Model chỉ giảm loss khoảng **3,06%** so với predictor không đọc RGB. Trung bình priors được tính trên toàn 3.000 video train, có trọng số theo phân phối uniform-video/uniform-window; không được khẳng định là nghiệm tối ưu SmoothL1.

Đổi embedding giữa hai video khác nhau nhưng giữ targets/masks làm loss trung bình tăng chỉ **4,40%**; một batch còn giảm nhẹ. Phép thử cho thấy dependence yếu vào đúng video trong bốn batches này, không chứng minh toàn corpus đều như vậy. Pose reconstruction giảm từ 7,89 xuống khoảng 1 chưa đủ chứng minh embedding phân biệt tốt 300 ký hiệu.

Xem [corpus_prior_means_scope.json](corpus_prior_means_scope.json), [conditional_prior_probe.json](conditional_prior_probe.json).

## 3. Nhánh style discriminator mất tác dụng

Hai ma trận đầu của style projection có max absolute weight khoảng **2,10e−42**; L2 float64 khoảng 5,56e−40 và 3,42e−40. Đây là số subnormal, không phải mọi phần tử bằng 0 về toán học. Trong cả bốn probes, gradient output discriminator theo style input bằng **0** ở float32; matched và unmatched probabilities trùng nhau theo từng anchor, quanh 0,5. Gradient theo content vẫn khác 0.

Training log xác nhận discriminator loss quanh **0,693147**, EMA matched = unmatched và **adversarial loss = 0 trên toàn bộ 1.000 updates cuối**. EMA gate tắt khi không có chênh lệch matched/unmatched. Trạng thái này không chứng minh embedding đã loại bỏ style thành công.

Gram normalization chia cho K×D là lựa chọn triển khai U04 chưa được xác nhận từ code training tác giả. Adam của discriminator có coupled weight decay 1e−3. Probe discriminator mới khởi tạo, không cập nhật optimizer, trên features/styles đã lưu: gradient weight decay / gradient data khoảng **0,50–0,87**; scale style tăng 768 lần giảm tỷ số xuống **0,16–0,19**. Đây là bằng chứng về conditioning và một hướng ablation, chưa chứng minh weight decay hoặc normalization là nguyên nhân duy nhất, cũng chưa xác nhận scale mới là recipe tác giả.

Code: [discriminator.py:10](../../../src/signrep_baseline/discriminator.py#L10), [optim.py:80](../../../src/signrep_baseline/optim.py#L80), [losses.py:38](../../../src/signrep_baseline/losses.py#L38). Xem [discriminator_style_path_trace.json](discriminator_style_path_trace.json), [discriminator_initial_gradient_trace.json](discriminator_initial_gradient_trace.json).

## 4. Embedding còn thiên về người ký và có ít hướng biến thiên

Trong gallery, nearest other video cùng signer giảm từ **99,7% xuống 46,87%**, nhưng vẫn cao so với **5,96%** theo tỷ lệ signer trong gallery. Đã giảm bias một phần; bias vẫn mạnh và gây khó cho validation có signer khác train.

Covariance của pooled, L2-normalized gallery embeddings có entropy effective rank giảm **7,41 → 2,48**, PC1 chiếm **54,99% → 71,90%** variance. Đây là concentration của biến thiên trên gallery inference, không phải rank của minibatch training, cũng không chứng minh tất cả features hằng nhau. Between-class variance fraction vẫn tăng **13,16% → 17,10%**, nên có một phần tín hiệu class được cải thiện.

Cosine trung bình query/gallery đúng class sau train là **0,9594**, sai class **0,9529**, với các phân phối chồng lấn mạnh. Centering bằng gallery mean cho Top5 7,0% và MRR 0,06018, chỉ cải thiện vừa phải. Đây là thử nghiệm hồi cứu trên validation đã dùng, chưa là kết quả độc lập của method mới.

Xem [cached_retrieval_trace.json](cached_retrieval_trace.json).

## 5. Activity, coverage và ngân sách

- Coverage activity trong 5.000 updates chỉ khoảng **0,19%**. Audit toàn bộ cửa sổ train có 563 labels LH + 675 RH hợp lệ; **tất cả active, không có inactive**. Điều kiện đủ cả 16 frames và torso references bị ảnh hưởng bởi độ tin cậy hips rất thấp. Activity head có nối gradient, nhưng không có dữ liệu đủ để học phân biệt active/inactive. Cosine giữa vector mean và activity pooling sau train trung bình 0,99793. Vấn đề này chủ yếu giải thích activity pooling chưa giúp ổn định, không giải thích toàn bộ kết quả mean pooling.
- Hand supervision chỉ phủ khoảng **45% RH**, **35% LH**, interaction/body distances khoảng **28%**. Audit hình học/FK/masks đã PASS; điều này không chứng minh teacher chính xác tuyệt đối với pose thật trong video.
- 5.000 updates × 2 video = 10.000 selections, khoảng **3,33 video-pass equivalents**, 20.000 augmented views. Ngân sách còn nhỏ. Layer decay 0,85 cũng làm early backbone thích nghi chậm: patch embedding thay đổi tương đối 0,305%, block0 0,188%, block23 11,95%. **327/327 tensors đã đổi**, không có bằng chứng encoder bị freeze nhầm.
- Training crop signer rồi random crop; inference dùng full frame. 128 train videos kiểm tra có median signer bbox area 74,66% frame. Có distribution mismatch; chưa chạy crop-consistent retrieval nên chưa đo đóng góp nhân quả.

Xem [POSE_PRIORS_AUDIT.md](../POSE_PRIORS_AUDIT.md), [supervision_trace.json](supervision_trace.json), [crop_domain_trace.json](crop_domain_trace.json), [optimization_trace.json](optimization_trace.json).

## Thứ tự thí nghiệm tiếp theo

1. Đảm bảo batch tính variance/covariance đủ lớn; log gradient từng loss và kiểm tra cân bằng ngay từ đầu. **Gradient accumulation thông thường trên các loss batch4 riêng không làm covariance được tính trên batch lớn**; cần tính statistics trên representations của batch lớn có gradient, hoặc gather phù hợp. Pilot batch20×1.000 updates có cùng 20.000 views với batch4×5.000, nhưng số optimizer updates khác: chỉ là so sánh matched exposure, không phải ablation batch-only hoàn toàn được kiểm soát.
2. Kiểm tra discriminator riêng: style gradient khác 0, matched/unmatched có khả năng phân biệt, EMA gate hoạt động. Ablate Gram normalization và decay groups từng yếu tố; ghi rõ mọi thay đổi recipe. Cần khởi tạo mới cho run sửa, giữ ASLC002 làm evidence.
3. Xử lý missingness/coverage activity bằng policy rõ ràng, có cả nhãn inactive; không gán missing thành inactive. Dùng mean pooling làm phép đo chính trong lúc activity chưa đạt điều kiện học.
4. Kiểm tra preprocessing crop nhất quán và baseline raw Kinetics; sau khi signal hoạt động mới tăng budget và chạy nhiều training seeds. Mọi phương án đã xem validation hiện tại là exploratory; cần validation/protocol cố định cho so sánh tiếp theo.

Không có ablation training mới trong báo cáo này. Các chi tiết thiếu từ code tác giả vẫn ghi ở [REPRODUCTION_SPEC.md](../../../docs/REPRODUCTION_SPEC.md); chưa khẳng định tái tạo chính xác toàn bộ training SignRep gốc.

## Artifacts và provenance

[Biểu đồ trace PNG](trace_summary.png), [PDF](trace_summary.pdf). Scripts: [trace_cached_retrieval.py](../../../scripts/trace_cached_retrieval.py), [trace_training_signal.py](../../../scripts/trace_training_signal.py). Các probes D initialization, feature swap, bootstrap và chart là diagnostics supplemental được thực hiện trong phiên; không tuyên bố các scripts trên bao gồm tất cả supplemental commands.

[trace_provenance.json](trace_provenance.json) pin checkpoint, training log, core implementation và các artifacts mới; kiểm tra lại archive/runtime hashes. Ngoại lệ launch log là phần final COMPLETED được append sau archive, đã có certificate prefix khớp hash ở [post_completion_check.json](../post_completion_check.json). Không sửa evidence cũ và không rerun độc lập toàn bộ training.
