# Quyết định method hiện tại: DIAGNOSTIC_ONLY, cần kiểm chứng cơ chế độc lập

Mục tiêu cuối vẫn là method novelty Sign Representations có bằng chứng. Pipeline,
head mới, kết quả calibration hoặc task grammar có sẵn không thay thế mục tiêu đó.
Quyết định này là interim; không đóng goal và không kết luận encoder mất grammar.

## Bằng chứng có thể dùng

| Quan sát local | Phạm vi suy luận |
|---|---|
| SignRep common B03 cao hơn B02 khoảng0,01479 macro recall, conditionalCI dương | Grid cũ B03 có6choices/B02có3; chưa đủ làm bằng chứng cho cơ chế temporal. Bộ equal-grid mới đang chờ native. |
| NCSLGR paired-minus-body macro recall −0,04956, CI[−0,15079;0,04096] | Chưa có lợi ích paired mean được hỗ trợ. Camera/framing khác nhau nên không suy luận nhân quả về facial information. |
| Body temporal-minus-global intervalIoU +0,01666, CI[−0,02240;0,05262] | Chưa có lợi ích temporal được hỗ trợ trên recorded single-core diagnostic. Không phải scope gold hoàn chỉnh hoặc absent detector. |
| Đối chứng posthoc train-duration prior tốt hơn body temporal khoảng0,08179IoU | Giữ classifier đã chọn, không refit. Chỉ là sensitivity sau primary; không chọn lại headline hoặc suy ra encoder không có grammar. |
| Temperature được chọn bằng validation giảm categoricalNLL/Brier; cả9head chạm biên trên8 | Xác suất raw có vấn đề trong cohort này. Không đổi class predictions, không calibration boundary và chưa chứng minh calibrated uncertainty. |
| Native NCSLGR có 42/222 clip không có frame đủ bốn stream; strict readout giữ175utterances và loại47 | Coverage conditioning đổi population. Strict train65/val27/test83 vẫn đủ ba lớp; single-interval test61. Không so trực tiếp với diagnostic222gốc như cùng population. |
| Native last temporal−global intervalIoU −0,06551, CI[−0,12845;−0,01105] | Native temporal readout không vượt global trong đối chứng đã khóa. Cả sáu endpoint readouts thua duration prior học từ train; chưa hỗ trợ cơ chế temporal adaptation được đề xuất. |
| Author-imputed pooling giữ222utterances; last/12-layer average/SignRep macro recall0,4122/0,5777/0,6094 | Primary last−SignRep−0,19717, CI[−0,31841;−0,07052]. Fixed average là secondary với CI delta chứa0. Context/cues và token rates khác nhau; không chọn layer bằng test hoặc kết luận mất thông tin grammar. |

Các số lấy từ immutable summaries và audits, không từ paper benchmark:
`reports/common_v1_primary_bootstrap.json`, `runs/ncslgr_diagnostic_v1/summary.json`,
`runs/ncslgr_intervals_v1/summary.json`, `reports/ncslgr_interval_audit.json`,
`runs/ncslgr_calibration_v1/summary.json`, `runs/fair_ncslgr_v1/summary.json`,
`runs/ncslgr_author_pool_v1/summary.json`. Audit native222clip,27savedheads và747
boundaryfiles đã PASS ở `reports/native_ncslgr_controls_audit.json`.
Author-pooling protocol được khóa sau khi strict result files đã tồn tại, trước
khi đọc metrics trong workflow này; không có certified blind/independent test.
Native lexical đã dừng do disk reserve ở1235/4726features; không suy diễn accuracy
từ cache lexical một phần. Lần dừng được giữ ở `evidence/native_common_disk_stop_v1`.

## Hướng không đủ làm novelty

Functional-marker classification, phân chia onset/core/offset và temporal relations
đã có trong nearest priors. [Liu et al.,2014](https://www.sciencedirect.com/science/article/pii/S0262885614000468)
mô tả hệ phân cấp CRF cho anatomical events/phases và grammatical markers.
Mức đọc P35 hiện là primary excerpts; full-author-PDF fetch chưa thành công.
Không suy diễn exact split hoặc superiority từ đoạn trích.

Face branch, attention, temporal convolution, missingness-aware fusion hoặc VQ
không đủ novelty khi đứng riêng. Nearest-prior matrix còn gồm reliability fusion,
asynchronous cue fusion, segment tokens và phonological supervision. Một adapter
phải sửa failure cụ thể vượt đối chứng mạnh, không chỉ thêm capacity/labels/trials.
Danh sách và phạm vi đọc nằm trong `reports/novelty_matrix.csv` và
`evidence/literature_matrix.csv`.

## Hành động quyết định tiếp theo

1. Resume native lexical sau khi đủ dung lượng. Existing1237preprocessing và1235
   feature cache hashes/source identities đã kiểm tra; giữ attempt/logs cũ, disk
   reserve và một GPU worker. Không xóa dữ liệu hoặc giảm guard để chạy tiếp.
2. Chạy equal-grid lexical protocol đã khóa sau full terminalPASS. Giữ failed
   waiter trước fitting và dùng output attempt riêng; giữ class-support guard.
3. Giữ strict175 và author-imputed222 function controls thành hai population/
   protocol riêng. Không gọi imputed là observed hoặc đổi primary theo kết quả.
4. Dựa vào đối chứng và audit, chọn tối đa một cơ chế adaptation với hypothesis,
   cùng capacity, labels, tuning và lexical retention. Chưa chọn cơ chế ở thời
   điểm này vì các native controls chưa hỗ trợ cơ chế đã đề xuất và chưa có
   independent mechanism confirmation. Chênh lệch layer readout chỉ là diagnostic.
5. Để xác nhận scope/semantic claim, cần gold độc lập và review ASL. Hồ sơ30case
   đã chuẩn bị ở `reports/ncslgr_expert_review_v1`; các form vẫn trống. Mẫu lỗi test
   không được tái sử dụng để train/tune adaptation hoặc gọi là independent test.

Gate hiện tại giữ `PENDING_MECHANISM_AND_INDEPENDENT_CONFIRMATION`.
Không mở pretraining/full fine-tune chỉ để tìm improvement. Một kết quả âm là
hợp lệ trong execution plan, nhưng chưa có nghĩa mục tiêu method novelty đã đạt.
