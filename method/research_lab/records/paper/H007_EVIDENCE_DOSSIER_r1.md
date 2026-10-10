# H007 — hồ sơ diagnostic retrieval và rejection

2026-10-05, [D085](../decisions/D085.md), H007_v1/P007_v1. Exploratory dev evidence;
chưa có method mới, contribution/confirmatory gates còn mở, không PAPER_READY.

## 1. Câu hỏi và use case

Video một sign được tra cứu trong từ điển có một video mẫu mỗi lớp. Khi sign
không có trong từ điển, hệ thống cần từ chối thay vì ép gán một lớp. Câu hỏi hiện
tại: điểm confidence và ngưỡng được calibration ở corpus nguồn còn giữ độ phủ
sign hợp lệ và tỷ lệ nhận nhầm sign ngoài từ điển khi đổi corpus/kích thước không?
Sign ngoài từ điển vẫn có thể là sign hợp lệ; không đo invalid posture, ngữ pháp,
dịch hay nhận diện video ký hiệu liên tục. Contribution chưa được xác lập.

## 2. Closest works và data access

Nguồn selected-scope: [NAC S47](../literature/sources/S47_CODE_r1.md),
[FR-Disc S48](../literature/sources/S48_CODE_r1.md),
[original OSCR S50](../literature/sources/S50.md), các source cards lịch sử và
literature matrix50 rows. NAC author commit971f6767, utils/license5648bytes,
30 CPU author-function cases pass. FR-Disc pinned OSCR implementation bị loại
qua fixture, không kết luận paper results sai. Trained residual comparator cần
audit supervision và loss riêng trước triển khai. Không claim novelty từ việc
đổi backbone/corpus hoặc ghép score/calibration có sẵn.

## 3. Hệ thống thực sự đã đo

Checkpoint SignRep đóng băng, RGB native-window features đã có từ các run
trước; mean-pooling805 base vectors768D FP64 và reuse1037 MS feature vectors.
Ba seed1701/1702/1703, cosine với gallery one-example/lớp; dictionary10/25/50.
Giữ ranking và ties theo global class ID. Existing scores: max cosine,
top1-minus-top2 margin, cosine softmax scale1, NAC top-min(16,N) softmax scale1.
NAC clamp với N10 là adaptation được khai báo; chưa train adapter/head/encoder.

## 4. Protocol và khác biệt benchmark

[P007_v1](../protocols/P007_v1_rejection_transfer.md) khóa trước measurement.
Base train calibration286clips/8 signer IDs và pool1027/24 đều100classes;
base dev378/5, MS-ASL validation748/19. Dictionary examples ít mẫu nhưng có thêm
nhãn calibration; không gọi toàn bộ supervision là one-shot. FROZEN50 chọn
ngưỡng source-known ở dictionary50 rồi dùng cho mọi size/corpus;
MATCHED_SOURCE_SIZE chọn ngưỡng source-known theo từng size. Chấp nhận inclusive
score>=threshold, target ít nhất95% source-known acceptance; không dùng unknown
hoặc target/dev labels để fit threshold. Không bảo đảm unknown FAR hay CCR95%.
Adapted joint FAR/CCR giữ grouped ties/counts/endpoint CCR=closed accuracy;
scalar grouped-trapezoid area chỉ diagnostic, không official OSCR score parity.

## 5. Kết quả thực và uncertainty

Mean ngang ba fixed galleries, size50/FROZEN50; đơn vị phần trăm.

| Existing score | Base FAR | MS FAR | Base CCR | MS CCR | Base known acceptance | MS known acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| maxcosine | 68.80 | 49.83 | 66.25 | 46.66 | 91.29 | 74.36 |
| margin | 86.69 | 87.94 | 65.90 | 50.93 | 93.96 | 91.89 |
| maxsoftmax | 74.04 | 75.86 | 65.90 | 49.51 | 92.54 | 85.43 |
| NAC16 | 76.68 | 79.43 | 66.25 | 50.14 | 92.72 | 86.99 |

Primary maxcos FAR MS−base −18.9696341pp; conditional signer bootstrap2000
replicates95% interval[−29.4086582,−9.5705893]pp, invalid0. Per-seed differences
−19.6951395/−8.5244598/−28.6893029pp. Registered absolute.05/same-sign/nonzero-CI
screen passes. Chiều LOWER target FAR cần đọc cùng độ phủ known giảm. Các score
khác chỉ descriptive; không kết luận winner/equivalence hoặc method gain.

Tất cả216 perseed rows và48 mean rows ở
[per_seed.csv](../../../frozen_probe/runtime/REJECTREPORT001_20261005_dbfdc329_s1701/per_seed.csv)/
[mean.csv](../../../frozen_probe/runtime/REJECTREPORT001_20261005_dbfdc329_s1701/mean.csv).
[PNG](../../../frozen_probe/runtime/REJECTREPORT001_20261005_dbfdc329_s1701/joint_curves.png)/
[PDF](../../../frozen_probe/runtime/REJECTREPORT001_20261005_dbfdc329_s1701/joint_curves.pdf):
24 individual-gallery curves,8 mean operating points, không averaged curve/bands.

## 6. Controls, lỗi và boundary

Closed cosine Top1 tại50 không đổi giữa scores: base67.4925%, MS52.2412%.
Margin/MSP/NAC thu hẹp claim universal confidence drift. Persistent-last50 absent
và common-first10 known controls được lưu cho size changes; không coi membership
hay task difficulty tương đương. Raw FROZEN50 FAR10→50 tăng26.82→68.80% trên
base,19.46→49.83% trên MS; matched-source-size10 chấp nhận nhiều known hơn nhưng
cũng nhận nhầm nhiều unknown hơn. Đây là secondary mô tả, không inferential gate.

Posthoc seed1701 chọn tối đa2 mẫu/bucket/corpus/seed,30 buckets: known correct
accepted/rejected, known wrong accepted, unknown accepted/rejected; giữ empty
buckets, bỏ known wrong rejected theo protocol. Metadata/IDs/ranks/accept flags
chỉ là numerical cases, chưa xem video/human/linguistic validation.

Own verifier đầu thất bại vì producer NumPy log2(int16) dùng FP32: secondary
DCG precision claim gốc không đạt FP64 oracle. Original fields giữ nguyên và
replay exact; [erratum](../../../frozen_probe/runtime/REJECTVERIFY002_20261005_2bf03591_s1701/dcg_precision_erratum.json)
version P007_v1.secondary_DCG_FP64_r1 sửa216 rows riêng, maximum8.4999630778e−8.
CSV có cả originalFP32/correctedFP64; primary FAR/thresholds/predictions không đổi.

## 7. Reproduction và resources

Six config/registration/command/source snapshots/statuses trong experiment cards
REJECTSCORETEST001/REJECTANALTEST001/REJECTEVAL001/REJECTVERIFY001/
REJECTVERIFY002/REJECTREPORT001; không restart terminal runs. Separate same-agent
verifier validates pooling805/predictions12708/calibration36/metrics216/curves108/
curvepoints50940/bootstrap2000. [Current closure](../inventory/H007_QUALITY_CYCLE_BINDING_VERIFICATION_r1.json)
checks2652bindings/allcaps/exports/case selection. Five complete stores và failed
partial verifier giữ nguyên. Runtime11,440,051logicalbytes, capturedwall16.0442215s,
GPU0/newencoder0; publicNACcapture5648bytes riêng, other workflow cost uninstrumented.
All-root effectiveGPU7523.0830524/14400s; full previous extraction cost không bị
coi là miễn phí. Numerical parity trước semantic recount dùng saved values;
không cross-environment bitwise/full raw-video/upstream task reproduction promise.

## 8. Hạn chế và bước tiếp

Cả hai corpus đều ASL; không cross-language/causal domain/general-purpose claim.
Official signer IDs không xác minh physical identity/session; encoder pretraining
overlap chưa rõ, preprocessing basefullframe/MSbodycrop khác. MS validation đã
exposed, không fresh confirmation. Target cached partition infeasible, HOLD.
Năm source-dev signer groups ít; CI conditional fixed support/calibration,
không population guarantee. Không phát tán video/features hay nộp paper.

Tiếp theo audit closest-work và supervision của stronger rejection comparator,
chọn falsifiable mechanism khi có gap còn sống qua controls, và audit fresh
confirmation feasibility trước train. Không retune current MSval để cứu method.
