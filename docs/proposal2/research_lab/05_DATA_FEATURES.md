# 05 — Dataset, checkpoint và feature contracts

## Inventory tại máy nghiên cứu

Người dùng cho biết đã tải dataset và setup inference. Xem đó là trạng thái do người dùng cung cấp; không yêu cầu tải lại. Đọc `docs/datasets/*.md` để tìm đường dẫn, rồi xác minh tài nguyên cần cho hypothesis trên máy thực. Không quét code/phương pháp cũ.

Inventory ghi dataset/version, path, sign language, modality, annotation types, label ontology, original/re-aligned annotation, official split IDs, local split names, expected/observed counts, missing/corrupt list, provenance và license. Chỉ chạm phần dataset cần cho phase; không hash toàn corpus ngay nếu quá tốn I/O. Có thể inventory metadata trước, hash subset smoke và xác minh coverage toàn bộ selected cohort khi lên gate.

- `How2Sign/eval` local cần xác minh tương ứng official validation; không tự xem là test hoặc gộp với train.
- Hai bản CSLDaily có thể trùng video nhưng khác preprocessing. Không đếm là hai dataset độc lập và không đưa hai phiên bản vào hai split.
- WLASL/MS-ASL có thể thiếu video tải được; giữ availability manifest, failed IDs và class support. Không lặng lẽ drop sample rồi so với số đầy đủ trong paper.
- Không giả định phonological labels, non-manual labels, framewise gloss boundaries hoặc signer IDs có sẵn chỉ vì có video/keypoint/text.
- Không lấy thư mục `subset_2000` local làm benchmark chính thức.

## Split và leakage audit

Trước pilot: đóng băng selected IDs, train/dev/test, class map và exclusion reason. Ghi đơn vị độc lập là clip, source video, signer, recording session hoặc lexeme theo claim.

Kiểm tra duplicate exact/near, cùng source YouTube ID/timestamp, overlap session/signer, clip từ cùng utterance, augmentation cache reuse, class map fit trên test và feature preprocessing fit toàn corpus. Không random split frame/window từ một video qua train và test. Multi-view cùng recording giữ cùng group.

Audit pretraining theo checkpoint provenance: corpus, videos/IDs và known exclusions. Không thể xác minh → `overlap_unknown`; báo cáo giới hạn và tìm cohort disjoint có thể xác minh khi claim cần. Không khẳng định zero-shot/unseen-data chỉ vì không dùng nhãn target: pretrained encoder có thể đã gặp video hoặc class.

Tách **official split reproduction** khỏi **new research protocol**. Official split không được âm thầm chỉnh để tạo signer independence. Nếu cần group split mới, tạo version, lý do, manifest và báo cáo riêng. Với cross-corpus transfer: chuẩn hóa ontology chỉ bằng thông tin được setting cho phép; xác minh synonym/polysemy/sign variants và duplicate recording. Mapping test để scoring có thể tồn tại trong evaluator nhưng không được truyền vào training.

## Các chế độ dữ liệu

| Regime | Dữ liệu target được dùng khi học/chọn model |
| --- | --- |
| Inductive frozen probe | Train labels cho head; dev cho selection; encoder cố định |
| Fine-tune/adapter | Train theo setting; dev chọn; test chỉ đánh giá đã khóa |
| Few/low-shot | Subset train cố định theo seed/budget; hỗ trợ per-class được ghi rõ |
| Unsupervised domain adaptation | Unlabeled target train được phép; dev/test access khai báo riêng |
| Transductive | Target evaluation inputs được dùng theo protocol đã nêu; không gọi là inductive |
| Zero-shot | Định nghĩa tường minh zero-shot về task/class/language/corpus; ghi pretraining và gallery supervision |

Chỉ inference độc lập bằng encoder đã khóa trên từng test input để cache feature là hợp lệ trong setting offline. Nhưng không fit PCA, whitening, standardization corpus-level, k-means, thresholds, layer weights hoặc adapters trên test nếu protocol không cho phép. Per-sample normalization phải khớp preprocessing đã đăng ký.

## Checkpoint loading audit

Ghi checkpoint path, nguồn/version, SHA-256, upstream commit, architecture/config, modality và trained/fine-tuned status. Kiểm tra missing/unexpected keys, tensor shapes và parameter coverage. `strict=False` chỉ dùng khi mismatch có danh sách giải thích được; không coi cảnh báo bị bỏ qua là load thành công. Xác minh model ở eval mode, dropout/masking tắt khi frozen extraction, no-grad và dtype/device đúng.

Upstream inference dùng như reference để kiểm tra parity trên 2–5 sample train/dev. So sánh input preprocessing, intermediate shape, output layer, dimension, finite values, sequence length và numerical tolerance theo dtype. Nếu checkpoint/commit khác reference, ghi rõ và không claim exact parity với paper.

## Feature contract mỗi encoder

Trước extraction toàn cohort, viết manifest field:

`sample_id, source_id, split, signer_group_if_available, language, input_path_or_id, source_hash, source_duration_s, timestamps_or_pts, source_fps, sampling_rule, crop_rule, normalization, channels_and_order, handedness, missingness_policy, preprocessing_code_sha, checkpoint_hash, output_layer, feature_unit, shape, dtype, stride_s, receptive_field, valid_mask, context_mode, cache_version, status, error_reason`.

Thời gian dùng presentation timestamps khi video VFR; không giả định mọi clip có FPS giống nhau. Cửa sổ, stride, padding, overlap và offset phải quy đổi được về giây/frame gốc. Không align hai encoder chỉ bằng cắt sequence tới chiều dài ngắn hơn. Resampling hoặc interpolation cần mapping timestamp và decision về tín hiệu mất đi.

Khác biệt representation frame/window/context phải giữ trong feature contract. Mean pooling là một design choice cần đăng ký, không phải thao tác vô hại. Layer selection/scalar mixing học hoặc chọn trên train/dev. Features all-layers chỉ cache nếu cần và nằm trong disk budget.

Multi-stream: thứ tự face/left/right/body, shape, missing mask, confidence, left/right anatomical vs image coordinates, crop validity và synchronization phải rõ. Không mặc định zero vector là missing nếu nó có thể là giá trị hợp lệ. Không coi interpolated landmarks là quan sát ground truth.

Không tự flip video hoặc swap hands mà không xác minh bảo toàn nhãn. Speed jitter, temporal crop, face blur và occlusion có thể làm mất fingerspelling/non-manual marker/temporal scope. Đánh giá augmentation trên train/dev với annotation liên quan và ghi hạn chế.

## Cache và coverage

Cache key gồm source identity/hash, preprocessing, checkpoint, layer và context/sampling config. Existing filename không đủ để skip extraction. Mỗi shard ghi manifest checksum và completion marker sau write atomic; resume chỉ reuse shard khớp toàn key.

Training không được đưa sample lỗi vào bằng feature zeros mà không khai báo. Bảng coverage cho từng split/encoder và cohort chung; nếu một encoder thiếu sample, xác định cả full-availability results lẫn matched-cohort comparisons khi cần. Việc chọn cohort không được phụ thuộc accuracy/test difficulty.

Không commit data/checkpoint/features lớn. Download mới chỉ khi inventory cho thấy dependency thật: record nguồn, bytes estimate, quyền sử dụng, checksum và destination; reuse/resume trước. Không chạy remote install script không hiểu hoặc sửa môi trường dùng chung.
