# Kỷ luật thực nghiệm và claims

## Trước khi chạy

Mỗi run liên kết hypothesis ID và protocol ID. Protocol xác định dataset/split hash, metric implementation, preprocessing, resolution/frames/FPS/stride, training/eval mode, baseline, checkpoint selection, seeds, resource cap và pass/fail.

Không tuning trên test, không dùng test labels cho class-distribution matrix, pair mining, normalization hay teacher adaptation. Dictionary labels chỉ lấy từ gallery/train split theo protocol. Validation tuning phải được ghi, kể cả lựa chọn checkpoint upstream có dùng validation.

Khóa protocol, code revision và config hash trước confirmatory run. Nếu code chưa ở git, tạo content hash cho code và configs. Sau khóa, thay đổi tạo protocol version mới; không sửa lịch sử.

## Công bằng

Giữ data, initialization, architecture size, input shape, augmentations, training budget và downstream recipe như nhau để đo thay đổi SSL objective. Không so “method có nhiều dữ liệu hơn” với baseline ít dữ liệu rồi quy công cho loss.

Tách external reported results khỏi internally rerun results. Khi không thể matched-compute, ghi mismatch và thu hẹp claim. Báo pretraining cost, pose teacher extraction cost, downstream cost và inference cost riêng.

Frozen retrieval, frozen linear probe, partial/full fine-tuning và supervised training là các settings riêng. Nếu method chỉ sửa pooling/retrieval head, cần kiểm tra embedding encoder ở cùng head để phân biệt representation với post-processing.

Checkpoint extension phải so cả baseline được tiếp tục train với cùng steps/data. Không gọi continued pretraining từ SignRep checkpoint là tái lập toàn bộ pretraining.

## Kiểm tra pipeline có ý nghĩa

Trước pilot: xác thực decoding, timestamp/FPS, tensor layout, normalization, missing targets, finite loss/gradients, checkpoint keys và output dimensions. Kiểm tra metric trên ví dụ nhỏ có đáp án biết trước và self-match/tie/zero-active edge cases.

Kiểm tra không học shortcut qua cache, labels, filenames, source/signer overlap hoặc dictionary duplicates. BatchNorm/dropout, eval/train flags, data order và masking phải đúng.

Loss được chuẩn hóa theo valid target counts; all-missing samples không tạo NaN hay division by zero. Claim temporal mechanism cần kiểm tra timestamp, variable FPS và pose jitter.

Smoke test chỉ chứng minh pipeline chạy; pilot chỉ sàng lọc. Không coi chúng là confirmatory evidence.

## Thiết kế cho cơ chế

Mỗi thành phần mới có:
- Ablation bỏ riêng thành phần.
- Control đơn giản nhất với compute tương đương.
- Negative control hoặc perturbation có dự đoán rõ.
- Diagnostic phân biệt cơ chế với data/recipe/parameter-count confound.

Không thêm nhiều module để cứu score trước khi hiểu failure. Hạn chế thí nghiệm không trả lời câu hỏi khoa học.

## Thống kê

Dùng nhiều seeds cho confirmatory comparison; một seed không đủ cho claim ổn định. Số seeds nhỏ phải công khai uncertainty, không hứa precision.

Phân biệt variance giữa training seeds với uncertainty trên eval samples. Paired bootstrap dùng cùng query/clip cho hai method; chọn signer/video/recording clusters nếu samples phụ thuộc. Class-macro metrics phải tính lại trong mỗi bootstrap, không bootstrap các clip như IID một cách mù quáng.

Báo mọi seed, mean/spread và CI theo protocol. Không xem 3 seeds như 3,000 độc lập từ các clips. Nếu claim thắng nhiều datasets, không chọn riêng dataset thuận lợi; khai báo primary/secondary và multiple comparisons.

Pilot chọn idea bằng validation. Confirmatory runs là tập evidence mới; khóa số trials và tiêu chí dừng, tránh optional stopping. Với test đã xem, mọi adapt tiếp theo là exploratory; muốn xác nhận mới cần holdout chưa dùng hoặc protocol xác nhận mới.

## Quyết định

Minimum meaningful effect δ, regression tolerance và resource cap phải chốt trước confirmatory outcomes. Nếu không đạt, ghi NULL/NEGATIVE/INCONCLUSIVE. Không đổi metric primary hoặc bỏ seeds để đạt gate.

Cải thiện accuracy, efficiency hoặc robustness có thể đều là đóng góp. Claim efficiency phải dùng non-inferiority margin khóa trước và đo cost end-to-end tương xứng. Claim robustness phải chỉ rõ perturbation hoặc domain shift.

