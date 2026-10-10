# Open-set intake — D073, 2026-10-05

Trạng thái: DISCOVER; hồ sơ chọn phép đo failure, chưa phải kết luận method.
H006 đã đóng vòng quality theo D072. Không dùng MS-ASLval để sửa whitening.
Các review dưới đây do cùng agent thực hiện tuần tự; không phải external review.

## Nguồn và nghĩa của task

L01 được đọc lại đúng §4.1: protocol xếp hạng lớp đúng trong dictionary và
báo cáo rank metrics. Trong phần đã đọc không có phép đo query vắng lớp đúng.
Đây là phân biệt protocol của ta; không suy tác giả chưa từng xử lý rejection
ở supplementary/code chưa audit. Không tuyên bố task rejection chưa có nghiên cứu.
S46 là invalid Thai fingerspelling posture rejection, S47 là few-shot gallery
face rejection, S48 là known/unknown few-shot video-action recognition.
S45 là publisher-abstract lead về sign-domain conformal expert early exit.
S49 cung cấp điều kiện của conformal risk control, chưa có mapping hợp lệ sang
task đang dự kiến. Xem các source cards để biết scope thực sự đã đọc.

## Ba câu hỏi được sàng lọc

| Câu hỏi | Scientific value dự kiến | Prior-art/falsifier | Quyết định |
| --- | --- | --- | --- |
| A: Ngưỡng cosine rejection cố định có giữ operating point khi dictionary và signer thay đổi? | Có thể xác định failure của frozen representation/score trên task dictionary chưa đầy đủ | S47/S48 đã có rejection; nếu strong simple controls giữ coverage/FAR thì nhu cầu cơ chế mới mất đi | Ưu tiên metadata feasibility rồi phép đo failure train/dev; H007 draft |
| B: MLP trên query-minus-support có cải thiện rejection? | Có thể là baseline useful | S48 FR-Disc trực tiếp đã có cơ chế này; encoder/dataset đổi không đủ novelty | Không chọn làm method mới; chỉ baseline nếu nhánh A cần supervised comparison |
| C: Conformal calibration trên known queries có bảo đảm từ chối unknown signs qua corpus? | Guarantee hấp dẫn nhưng thiếu assumption/negative population | S49 không cho guarantee này từ known-only calibration; S45 là sign calibration lead cần đọc full methods | Bác bỏ claim guarantee hiện tại; không implement dưới tên mới |

FINER cho A (nội bộ, 1–5): Feasible3/Interesting3/Novel2/Ethical4/Relevant3,
mean3.0. Novel2 chỉ là câu hỏi empirical chưa resolve, không chứng minh mới.
User benefit chưa đo; không khẳng định Deaf-user usability hay safety deployment.
Theo weights04, H007 scores3/1/3/4/3 =2.65/5; hard registration gates vẫn thắng.

## Estimands cần tách trước implementation

Cho dictionary G, nhãn absent được định nghĩa chính xác bởi class_id(y)∉G.
Đó là missing *dataset class*, không tự là lexeme/encoder-pretraining unseen,
invalid sign, non-sign posture hay cross-language generalization.
Đầu ra gồm rank trong G và accept/reject. Cần báo cáo riêng:

- Unknown FAR: tỷ lệ query absent bị accept.
- Known coverage: tỷ lệ query present bị accept.
- CCR: tỷ lệ query present vừa accept vừa dự đoán đúng Top1, mẫu số toàn bộ known.
- Accepted-known error: incorrect / accepted-known; undefined nếu không accept ai.
- Known-only DCG và joint OSCR/AUROC với orientation/ties/endpoints khóa rõ.

Không tính absent là rank cuối rồi gọi giảm DCG là failure. Không tối ưu unknown
AUROC mà bỏ known classification. Không gộp wrong-known và absent thành một
ground-truth linguistic error. Reject-all có FAR0 nhưng CCR0.

Counterexample thuần logic (không pilot/scientific measurement): mọi known và
absent query đều có score1; threshold≤1 accept cả hai. Known miscoverage0 vẫn
đi cùng unknown FAR1. Guarantee known-only không suy ra guarantee unknown; cần
negative population/assumption tương ứng. Ngưỡng score này là minh họa, không
config được chọn cho dữ liệu.

## Cheapest next action và fairness

Metadata-only census các feature stores đã terminal, chia train/dev/test, source,
signer, class, loại feature và quyền sử dụng. Chưa đọc thêm quality hoặc fit.
Đặc biệt phải xác định nguồn train-only calibration queries ngoài gallery,
khả năng tách signer/source và giữ tất cả cohort; tránh một vector/lớp không đủ
calibration bị gọi là few-shot target support. Thêm nhãn calibration phải tính.
MSFULLFEAT0011037 clip vectors gồm289train supports/748known MS-ASLval queries;
mọi MS-ASLval outcome mới từ chúng chỉ exploratory sau D072, không fresh test.

Sau census mới khóa protocol/runs và evaluator parity. Strong frozen scoring
controls: max cosine, top1–top2 margin, maximum softmax/NAC và energy-family
score nếu định nghĩa scale/temperature/normalization đúng; chưa gọi cosine
score là trained-classifier logit. Equal input/gallery/query/calibration labels,
search budget, every seed, no encoder geometry tuning. Gallery-relative
leave-class-out calibration là standard control candidate, không novelty claim.
FR-Disc nếu cần phải ghi supervised label/training cost và matched simpler head.

Gallery-size comparison phải có nested galleries và common query estimands;
classes được thêm làm distractor không thể bị coi là cùng unknown cohort sau
khi trở thành present. Không cherry-pick lớp/query theo score. Metadata census
sẽ quyết định exact cardinalities/signer splits; protocol hiện chưa khóa chúng.

New candidate can be killed nếu baseline failure nhỏ/không ổn định hoặc standard
control giải thích toàn improvement. Không GPU allocation/scale cho H007 hiện tại.
Fresh confirmatory cohort, official/public evaluator code và significance plan
vẫn unknown. H003 annotation hold không khóa công việc metadata này.
