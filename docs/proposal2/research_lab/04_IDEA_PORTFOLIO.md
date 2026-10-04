# 04 — Portfolio ý tưởng và hypothesis design

## Từ problem đến hypothesis

Không bắt đầu bằng “thêm attention/contrastive loss/LoRA”. Đầu tiên đo một failure hoặc một tradeoff có ích trên dev, kiểm tra đó có phải lỗi protocol hay không, rồi tìm cơ chế sửa. Một hypothesis gồm problem, mechanism, prediction, falsifier và cheapest discriminating experiment. Viết trước khi chạy; dùng [template](templates/HYPOTHESIS.md).

Ví dụ dạng câu: “Trong setting S, biểu diễn E mất thông tin X vì cơ chế Y; phương pháp M khôi phục X mà giữ Z, nên primary metric Q cải thiện ít nhất ngưỡng δ đã đăng ký dưới budget B; nếu control C cho gain tương đương hoặc probe X không đổi thì cơ chế bị bác bỏ.” Điền giá trị từ inventory/dev, không tự chọn δ theo kết quả test.

## Các hướng khởi đầu, chưa phải kết luận novelty

| Hướng | Câu hỏi/cơ chế để khảo sát | Tài nguyên có thể dùng | Pilot rẻ và falsifier | Điều kiện/đối thủ cần kiểm tra |
| --- | --- | --- | --- | --- |
| A. Ít nhãn và chuyển miền trong cùng sign language | Adapter biểu diễn temporal hoặc structured alignment có giảm số nhãn target cần dùng so với frozen probe và adapter thường? | ASL Citizen, WLASL, MS-ASL; checkpoints hiện có | Feature audit → probe trên các budget nhãn train giống nhau → transfer curve; bác bỏ nếu gain biến mất khi cùng head/search budget | Đối chiếu ontology ASL, signer overlap, target-label use và prior SSL/PEFT work; không chuyển class ID thẳng giữa corpus |
| B. Temporal representation cho spotting/grounding | Multi-scale temporal unit có giữ boundary/coarticulation tốt hơn mean pooling hoặc temporal head thông thường? | How2Sign hoặc corpus khác chỉ khi có annotation temporal đúng task | Kiểm tra annotation trước; small dev localization + temporal perturbation; bác bỏ nếu tăng head capacity giải thích toàn gain | Sentence timestamps không phải sign boundaries; review spotting/alignment gần đây; SHuBERT context cần fair control |
| C. Biểu diễn non-manual/phonological có tính tổ hợp | Có giữ sự khác biệt ngôn ngữ trong khi giảm nuisance signer? Có generalize sang tổ hợp/lexeme chưa thấy? | Phonological/non-manual metadata sau khi xác minh và join đúng | Leave-lexeme-out probe, head capacity control và per-feature missingness audit; bác bỏ nếu metadata/lexeme identity đủ dự đoán | Phonological classification đã có baseline; cần supervision đáng tin, linguistic review và ontology chuẩn |
| D. Robustness khi mất stream hoặc tracking lỗi | Reliability-aware adaptation có giữ linguistic cues tốt hơn zero fill, stream dropout và imputation? | Multi-stream SHuBERT; RGB SignRep làm comparator phù hợp | Predeclare corruptions trên dev, giữ clean-quality floor; kiểm tra natural-failure slice | Robustness synthetic không tự suy ra thực tế; xóa mặt/tay có thể xóa chính nhãn; dropout/imputation là strong controls |
| E. Biểu diễn gọn/causal cho latency có ràng buộc | Distillation/context compression có đạt quality–latency frontier tốt hơn pooling/downsampling hoặc student thường? | Feature/checkpoint sẵn có, task với timestamp và latency use case | Feature-based compression pilot; tách encoder-only và end-to-end cost; bác bỏ nếu preprocessing chiếm toàn lợi ích | Frozen bidirectional features không dùng để chứng minh online; teacher/student access và training cost phải công khai |
| F. Cross-corpus/open-set dictionary retrieval | Representation giữ sign-form discriminability qua signer/domain mà không cần fine-tune target? | ASL Citizen, WLASL, MS-ASL sau ontology/overlap audit | Frozen nearest-neighbor/prototype controls trên fixed gallery; class- vs instance-ranking kiểm tra riêng | SignRep đã retrieval; SignSeek là closest-work lead cần đọc; một retrieval head mới chưa chắc novel |

Các hướng là menu để agent khám phá, không phải yêu cầu implement tất cả. Cross-language transfer là nhánh riêng có thể xem xét nhưng không mặc định khả thi từ các corpus ASL/CSL/DGS: phải xác định task chung, ontology, supervision và chứng cứ chuyển được representation.

## Scorecard để chọn hướng

Chấm 0–5 có justification và evidence IDs:

| Tiêu chí | Trọng số default |
| --- | --- |
| Giá trị vấn đề/use case và failure đã đo | 25% |
| Novelty có thể kiểm chứng so với closest works | 25% |
| Dataset/nhãn/protocol khả thi | 20% |
| Chi phí phù hợp tài nguyên và pilot nhanh | 15% |
| Khả năng phân biệt mechanism khỏi confound | 15% |

Tổng điểm chỉ sắp thứ tự tìm hiểu. Hard blockers thắng score: không có nhãn đúng task, không tách train/test được, không rõ quyền dữ liệu, evaluator không thể xác minh, hoặc contribution trùng closest work chưa giải quyết. Không cho 5/5 vì “có vẻ mới”.

## Baseline và đối chứng theo cơ chế

Mỗi hướng tối thiểu có: frozen encoder + head đơn giản, strong standard method cho task/setting, phương pháp mới dưới cùng budget, và ablation trực tiếp của mechanism. SHuBERT/SignRep dùng khi interface/task phù hợp; không ép cả hai vào mọi task. Nếu chỉ chạy một encoder, hạn chế claim phụ thuộc encoder và nêu lý do.

Các control thường cần: mean/max pooling, trainable scalar mix, linear vs MLP vs temporal head có capacity gần nhau, adapter thường, random/no-pretraining encoder khi tính toán hợp lý, teacher đơn, concatenation/ensemble đơn giản nếu đề xuất fusion, training lâu hơn dưới matched compute. Không dùng baseline cố tình yếu hoặc hyperparameters chưa tune trong khi method được tune nhiều.

Không diễn giải fusion thắng một encoder là representation mechanism mới nếu simple concat với cùng parameters đã thắng. Không coi LoRA thắng linear probe là bằng chứng tốt hơn fine-tuning khi chưa kiểm soát regime.

## Chốt hướng

Sau pilot, decision memo phải có câu hỏi paper, evidence failure, closest works, contribution còn sống, result dev có uncertainty sơ bộ, confounds đã loại, compute confirmatory và fallback. Chỉ promote khi reviewer có thể nói phép thử kế sẽ phân biệt điều gì. Không chốt chỉ vì một task cho điểm cao.
