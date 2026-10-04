# 09 — Từ evidence tới paper conference

## Tiêu chuẩn chốt hướng

Một direction memo được chốt khi có: câu hỏi quan trọng và giới hạn rõ; closest-work comparison có nguồn; failure baseline được đo; cơ chế đề xuất; dev pilot vượt confounds chủ yếu; protocol khả thi; confirmatory plan có budget; và risk/alternative explanation còn lại. Đây là **hướng để viết paper**, chưa phải kết quả đủ để nộp.

Không quyết định venue trước evidence nếu việc đó ép claim. Có thể xem venue mục tiêu để hiểu loại đóng góp; kiểm tra current CFP, deadlines/timezone, anonymity, reproducibility, supplementary, ethics và AI-use policy từ official website tại lúc viết/nộp. Không suy deadline từ năm trước.

## PAPER_READY gate

- [ ] Research question và contribution một câu; novelty audit mới nhất có ngày và closest works.
- [ ] Task/data có ý nghĩa, annotation đúng claim; protocol cards được validate.
- [ ] Main results confirmatory với strong baselines và điều kiện so sánh rõ.
- [ ] Seeds/uncertainty phù hợp; all trials và negative results không bị giấu.
- [ ] Ablations kiểm tra proposed mechanism, capacity/compute và simpler alternatives.
- [ ] Nếu claim representation dùng rộng: có ít nhất hai setting/task/domain độc lập phù hợp, hoặc evidence tương đương có lý do; nếu chưa có, claim thu hẹp tới setting đã đo.
- [ ] Error/slice analysis, qualitative cases và failure modes.
- [ ] Pretraining overlap, modality/teacher/text supervision và compute công khai.
- [ ] Code/config/environment/predictions/scorer provenance đủ tái lập; tables/figures rebuild được.
- [ ] Mỗi claim traceable; không claim từ unsupported/invalidated runs.
- [ ] Limitations và data/license/privacy considerations đúng scope.
- [ ] Skeptical review có unresolved concerns, fixes hoặc explicit qualification.

Checklist là default quality bar nội bộ, không guarantee acceptance hay thay tiêu chuẩn venue.

## Cấu trúc paper dossier

Tạo từ [template](templates/PAPER_DOSSIER.md), với:

1. Problem statement/use case, observed limitation và contribution.
2. Related-work table về problem, mechanism, supervision, representation và protocol.
3. Method specification: input/output shapes, temporal context, losses, data access, training/inference, algorithm và complexity. Không chỉ mô tả module bằng tên.
4. Evaluation specification và protocol equivalence/differences.
5. Main result table; label/compute/transfer curves nếu claim cần; ablation table; uncertainty.
6. Mechanism/error analysis, negative cases, sensitivity và boundary of validity.
7. Reproducibility index: Git/config/checkpoint/scorer/split hashes và commands đã chạy.
8. Limitations, license, appropriate human/linguistic validation và release plan.

Draft có thể viết trước confirmatory để làm rõ story, nhưng mọi ô số chưa có phải ghi `pending`/`TBD`, không tạo số minh họa như thật. Không nộp paper hoặc phát tán artifact hạn chế bằng quyền chạy experiment.

## Skeptical review trước khi khóa draft

Reviewer ghi trả lời có evidence IDs:

- Contribution có thực mới so với nearest works, hay chỉ đổi backbone/task/dataset?
- Task mới có use case và annotation đúng không? Benchmark selection có cherry-pick không?
- Baseline có cùng data, head/search budget, labels và future context?
- Gain đến từ representation hay decoder/teacher/additional supervision?
- Evaluator có đúng algorithm/denominator? Bản local thiếu dữ liệu có được disclosed?
- Conclusion có sống qua seeds, ablation, alternative explanations và setting thứ hai khi claim đòi hỏi?
- Tradeoff/compute có đo end-to-end và training cost không?
- Có claim về sign language grammar/Deaf users mà chỉ đo text similarity/proxy không?
- Người ngoài có rebuild con số headline từ manifest/code/predictions không?

Nếu reviewer tự thực hiện bởi cùng agent, ghi rõ internal review. Với final scientific/linguistic judgment, khuyến nghị người nghiên cứu/chuyên gia phù hợp review; không dùng đồng thuận tự sinh như external validation.

## Chốt kết luận có giới hạn

Viết “trên corpus/setting X dưới protocol P, M cải thiện Q so với B với uncertainty U và cost C”. Chỉ mở rộng tới sign languages/tasks đã có evidence. Nếu không đủ phương pháp paper, có thể chuyển sang analysis/benchmark contribution khi đáp ứng tiêu chuẩn tương ứng; không cố kể một story method từ kết quả âm.
