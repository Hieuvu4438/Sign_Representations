# Hồ sơ bằng chứng H006 — chuyển giao hướng covariance giữa hai corpus ASL

Ngày 2026-10-05. **Bản nháp bằng chứng, chưa PAPER_READY**. Nguồn quyết định:
[D072](../decisions/D072.md), [protocol đã khóa](../protocols/H006_v3_MSASL_quality_v1.md)
và [audit của vòng thực nghiệm](../inventory/MSASL_QUALITY_CYCLE_BINDING_VERIFICATION.json).
Các vai trò được thực hiện tuần tự bởi cùng agent; chưa có review khoa học bên ngoài.

## 1. Câu hỏi và giới hạn đóng góp

Hướng covariance của biến thiên trong cùng lớp, học từ ASL Citizen, có giữ được
lợi thế so với hướng covariance tổng khi chuyển nguyên tham số sang MS-ASL,
với cùng phổ eigenvalue không?

Phép so sánh đã đăng ký cho kết quả dương. Tuy nhiên, residual whitening vẫn
thấp hơn raw cosine và centered cosine. Vì vậy đây là bằng chứng về một phép
so sánh hình học cụ thể; chưa chứng minh một method downstream tốt hơn.
Đóng góp phân tích đủ giá trị cho conference cũng chưa được xác nhận.
Tác vụ là tra cứu video ký hiệu đơn lẻ, offline, theo nhãn lớp. Chưa đo ích lợi
với người dùng từ điển, tính dễ hiểu của ký hiệu, nhận diện liên tục hoặc dịch.

## 2. Các công trình gần nhất

[S35](../literature/sources/S35.md) dùng khoảng cách phụ thuộc covariance trong
few-shot; [S37](../literature/sources/S37.md) dùng whitening theo lớp trên frozen
features; [S43](../literature/sources/S43.md) nghiên cứu phép biến đổi feature học
trên train và chuyển giữa dataset; [S44](../literature/sources/S44.md) dùng scatter
của các cặp matching để whitening cho retrieval. Đây là các tiền lệ, chưa được
reproduce trong setting của lab.

[S41](../literature/sources/S41.md) và [S42](../literature/sources/S42.md) mới xác minh
được tiêu đề/tác giả trong danh sách accepted của WSLP 2026. Chưa đọc toàn văn,
thuật toán, code hay kết quả. [Novelty audit r2](../literature/NOVELTY_20261004_H006_r2.md)
giữ cổng contribution mở; không gọi whitening là thuật toán mới hoặc tuyên bố
đã rà soát toàn bộ literature.

## 3. Representation và phép biến đổi cố định

SignRep giữ nguyên checkpoint SHA
`f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b`,
upstream `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2`. Input MS-ASL dùng body crop đã
khai báo, khoảng annotation half-open trên estimated clock, cửa sổ 16 frame gốc
stride 2, resize 224 và ImageNet normalization. Encoder chạy FP32, batch 1, eval,
mask ratio 0, no-grad; truy cập toàn bộ cửa sổ hoàn chỉnh.

Mỗi video được pooling FP64 trên feature 768D. Trọng số là giá trị lớn nhất của
hai output activity sau khi clamp vào [0,1]; nếu tổng trọng số không quá 1e-12,
dùng mean không trọng số. Vector clip được L2-normalize.

Geometry dùng đúng mean, R/T bases và R/T spectra đã lưu trong AXISTRANSFER001,
học từ 1.313 clip ASL Citizen có nhãn thuộc 100 lớp. R là residual covariance có
shrinkage; T là total covariance có shrinkage. Với mode DS:

`L2((z - mean) @ U_D @ diag(1/sqrt(lambda_S)) @ U_D.T)`.

Bốn mode là RR/TR/RT/TT; hai control là raw cosine và trừ base mean rồi L2.
Không fit target mean/covariance/PCA, tính lại eigenbasis, hoặc chọn rank/lambda/
layer/seed mới. Điểm query–gallery tính FP64; tie theo class ID tăng dần.

## 4. Protocol và quyền truy cập dữ liệu

MSASL_ESTCLK_v1 giữ nguyên danh sách 3.425 candidate: 2.674 train và 748 val đủ
input; ba train clip ngắn 13/15/10 frame được ghi nhận từ trước khi chấm chất lượng.
Các lỗi của raw-clock gate cũ vẫn được lưu. Không padding, loại query theo điểm
hoặc đổi cohort sau kết quả.

Ba gallery seed 1701/1702/1703 có một train video cho mỗi lớp 0–99. Tất cả method
chia sẻ đúng gallery và đủ 748 query. Hợp của 289 support khác nhau và 748 query
là 1.037 vector. Query gồm 171 source parent, nằm trong 19 official signer ID;
source và signer ID không giao với gallery. Danh tính vật lý chưa được chứng nhận.

Supervision là 100 nhãn gallery mỗi seed **cộng 1.313 clip base có nhãn**, không
phải learner chỉ dùng 100 nhãn. Raw không cần nhãn geometry bổ sung. Nhãn val chỉ
dùng đánh giá, không fit hay chọn method. Đây là validation trên collection ASL
khác, không phải official MS-ASL classifier/test benchmark. Pretraining overlap
và overlap về nghĩa ký hiệu chưa rõ; không claim encoder-unseen hoặc cross-language.
Base ASL Citizen dùng full frame còn MS-ASL dùng body crop, nên không quy mọi
khác biệt giữa corpus cho một nguyên nhân domain shift riêng lẻ.

## 5. Kết quả và uncertainty

Trung bình ba gallery cố định. Top1/Top5 là fraction; DCG là binary gain với
`1/log2(rank+1)`. Macro tính đủ 100 lớp có query support.

| Mode | Micro DCG | Micro MRR | Micro Top1 | Micro Top5 | Macro DCG |
| --- | ---: | ---: | ---: | ---: | ---: |
| RR | .657898 | .567690 | .460339 | .692513 | .661862 |
| TR | .618588 | .522658 | .417112 | .642602 | .622406 |
| RT | .646872 | .554664 | .448307 | .674688 | .651443 |
| TT | .601844 | .503037 | .397059 | .618538 | .606200 |
| raw | .679797 | .594405 | .476827 | .746435 | .679117 |
| centered | .681641 | .596586 | .480392 | .737968 | .682589 |

Primary RR−TR là **+0,03930933496 DCG**, từng gallery +0,03690092885 /
+0,03461964582 / +0,04640743020. Gate đã đăng ký ≥0,005 và không gallery nào âm
đều pass. Bootstrap 2.000 lần, seed 1701, theo 19 signer ID và giữ toàn bộ clip
của mỗi nhóm được lấy mẫu: conditional 95% interval **[0,02372283639; 0,06270928809]**.
Interval có điều kiện trên gallery, base fit, encoder và collections cố định;
không đo variance của gallery sampling hoặc bảo đảm kết quả trên population.

RR−raw là −0,02189946740 và RR−centered −0,02374365223 DCG. Cả hai control đơn giản
đều hơn RR ở mọi gallery. Đây là secondary descriptive effects; không thêm
significance/equivalence test, đổi primary hoặc chọn một secondary winner làm method.
[Bảng CSV đủ precision](../../../frozen_probe/runtime/MSREPORT001_20261005_0ca4b580_s1701/metrics.csv).

![So sánh toàn bộ method và gallery](../../../frozen_probe/runtime/MSREPORT001_20261005_0ca4b580_s1701/geometry_comparison.png)

## 6. Kết quả âm và cases

Lợi thế của hướng residual so với total không kéo theo superiority so với cosine
đơn giản. Giữ primary PASS nhưng chưa promote whitening; không tune lại trên
MS-ASL val này để cứu một mean/rank/shrinkage recipe.

Gallery 1701 có 313 query RR và raw cùng đúng Top1, 328 cùng sai, 46 chỉ RR đúng,
61 chỉ raw đúng. Chọn tối đa ba case mỗi nhóm bằng seed 1701, sau khi biết kết quả,
và hiển thị rank của tất cả mode/gallery trong
[cases](../../../frozen_probe/runtime/MSREPORT001_20261005_0ca4b580_s1701/descriptive_cases.json).
Các case chỉ có metadata/rank; chưa xem video hay đánh giá lỗi handshape/ngôn ngữ.
Không dùng chúng làm suy luận chọn lọc, chứng minh signer invariance hay fairness.

## 7. Reproducibility và compute

Registration, lệnh, config SHA, source snapshot, environment, prediction, hash và
status nằm trong các run độc lập: MSFULLFEAT001, MSEVALTEST001, MSEVAL001,
MSEVALVERIFY001 và MSREPORT001. Không overwrite/restart. Các tên MSEVAL có stem
20261004 theo helper bất biến; actual UTC launch là 2026-10-05 và đúng identity.

Extraction dùng 1.777,68446072 giây GPU: 1.030 clip mới/36.028 cửa sổ + một warmup;
bảy vector cost đã xác minh được tái sử dụng cho 396 cửa sổ. Store 9.536.905 B,
dưới cap 20 MiB. Encoder cost chia sẻ giữa mọi mode, chưa đo efficiency riêng từng
method. Evaluation CPU 0,82646 giây và replay CPU 0,79923 giây gồm hash/input/output
overhead; không gọi đây là latency end-to-end khi triển khai.

Replay bằng quadratic metric trong không gian gốc khớp **13.464 full ranking**,
sai số score tối đa 9,88e-15 dưới tolerance cố định 1e-10. Row metrics và aggregates
khớp chính xác; macro, gate và bootstrap interval đã kiểm tra. Không có exact tie;
near-degenerate bases vẫn hạn chế portability. Quality store 14.515.798 B <20 MiB;
tám run đều trong cap và 817 current input bindings pass. Cùng agent tính lại bằng
công thức khác, không phải external scientific review.

## 8. Readiness và bước nghiên cứu tiếp

Chưa PAPER_READY: đóng góp đủ khác biệt và hữu ích, đối chiếu toàn văn S41/S42,
validation ngôn ngữ/use case và generality vẫn chưa hoàn thành. Pretraining,
lexeme overlap, physical identity và tính sẵn có của corpus giới hạn diễn giải.
Video, feature và checkpoint gốc được giữ nguyên; chưa upload, publish hoặc submit.

Kết luận nội bộ: retain bằng chứng về orientation transfer; chưa scale residual
whitening như method tốt hơn. Tiếp theo đối chiếu nearest works công khai và chọn
một câu hỏi contribution mới có thể bác bỏ trước khi cấp thêm compute. Diagnostics
trên kết quả hiện có phải ghi rõ post-hoc, không chọn method mới trên validation
đã đọc. H003 annotation intake còn thiếu nhưng không chặn toàn chương trình.
Mục tiêu proposal2 vẫn active.
