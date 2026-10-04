# 08 — Phân tích, thống kê và evidence ledger

## Phân biệt evidence

Literature claim, software check, exploratory dev result và confirmatory test result có nhãn riêng. Một smoke loss giảm không chứng minh biểu diễn tốt. Baseline reference số từ paper không phải baseline đã reproduce. Kết quả invalidated không được đưa vào bảng performance như valid.

Result ledger mỗi run:

`run_id, H_id, P_id_version, status, regime, cohort_hash, encoder_checkpoint_hash, config_hash, code_sha_or_snapshot, seed, label_subset_seed, selection_rule, scorer_hash_and_signature, split, N_expected, N_scored, metrics_with_units, runtime, GPU_hours, peak_memory, artifact_paths, validity, caveats`.

Claim ledger mỗi claim:

`claim_id, exact_statement, scope, evidence_type, supporting_run_ids_or_sources, strongest_baseline, effect_size, uncertainty, alternative_explanations, falsifying_evidence, limitations, status`.

Claim status: `hypothesis`, `exploratory`, `supported`, `qualified`, `refuted`, `invalidated`. Không dùng supported nếu run hoặc protocol còn blocked. “Không thấy khác biệt” không chứng minh equivalence; equivalence/non-inferiority cần margin và design riêng.

## Thống kê đúng đơn vị

- Report từng seed, mean và SD khi training ngẫu nhiên. CI theo query/clip không thay cho variance training.
- So sánh paired trên cùng evaluation samples/cohort và matched seeds/subsets khi hợp lý. Ghi effect size bằng đơn vị metric thật (percentage points, MRR, seconds...), không lẫn relative percent.
- Bootstrap/resampling cần theo đơn vị độc lập phù hợp claim: signer/source video/session/utterance/lexeme, không mặc định IID frames. Clip bootstrap có thể quá lạc quan khi cùng signer/video xuất hiện nhiều.
- Corpus BLEU/temporal AP phải recompute metric trên resampled units, không bootstrap average sentence BLEU/AP sai nghĩa. Dùng scorer hỗ trợ hoặc implementation đã validate.
- Default 95% CI và khoảng 2,000 bootstrap replicates là lựa chọn ban đầu của lab; đăng ký unit, seed, method và đủ precision theo cost. Không coi bootstrap với rất ít signers là evidence population rộng.
- Với nested training seeds và evaluation groups, mô tả rõ nguồn uncertainty; không pool toàn predictions across seeds như thêm independent test samples. Có thể report seed variability cùng paired group bootstrap cho từng seed, hoặc thiết kế phân cấp hợp lệ.
- Nhiều task/slice/variant: predeclare primary comparison. Điều chỉnh multiple testing nếu đưa inferential significance cho nhiều comparisons; secondary analyses ghi exploratory.
- Không báo best seed làm headline. Không bỏ seed âm; seed thất bại kỹ thuật ghi failed với nguyên nhân, không tự đổi seed để lấy gain.

## Ý nghĩa thực tiễn và representation mechanism

Một result đáng theo đuổi cần phù hợp practical threshold đăng ký và có cơ chế hợp lý, không chỉ CI không chứa 0. Với efficiency: quality floor/margin, end-to-end cost và training amortization. Với ít nhãn: learning curve theo số labels thực, cùng subsets và confidence bands; không chỉ chọn điểm budget tốt nhất.

Các probe kiểm tra representation phải kiểm soát probe capacity, label availability và sample count. Probe accuracy cao không chứng minh factor causal hoặc disentanglement. Saliency/t-SNE/UMAP là diagnostics, không bằng chứng duy nhất cho ngôn ngữ hay mechanism. Ghi seed, projection settings và cách chọn sample.

Identity/nuisance probing: classifier yếu không dự đoán signer không chứng minh representation vô danh/invariant. Invariance không được làm mất lexeme, spatial reference hoặc non-manual marker có nghĩa; cần task-sensitive controls. Causal claim đòi design phù hợp; dùng “consistent with” nếu chỉ có observational ablation.

## Error và slice analysis

Chọn slices từ train/dev metadata trước test: frequency, length, signer/domain, frame quality, missing stream, motion/handshape complexity khi metadata đáng tin. Ghi N/support cho mỗi slice; không kể chuyện từ slice có quá ít sample.

Qualitative examples gồm successes, failures và baseline disagreements; sampling rule có seed, không chỉ chọn video đẹp. Sửa lỗi grounding/timeline trước diễn giải nội dung. Language/non-manual interpretations cần người có chuyên môn khi claim phụ thuộc chúng.

Negative results được phân loại: hypothesis refuted, budget/power insufficient, protocol invalid, implementation failed hoặc domain mismatch. Các loại này dẫn tới quyết định khác nhau. Không kết luận method không hiệu quả từ một job crash hoặc baseline bị lỗi.

## Rebuild tables và figures

Mỗi bảng/figure sinh từ valid ledger/predictions bằng script versioned trong `method/<direction_slug>/scripts/`. Lưu figure data, units, cohort/protocol IDs, aggregation và uncertainty definition. Không nhập số thủ công mà không link nguồn.

Tách reported literature numbers bằng chú thích; không tô đậm “best” trên metrics incomparable. Ghi encoder/pretrain/supervision/regime/data/compute khác biệt. Bảng thiếu denominator hoặc provenance phải sửa trước paper dossier.

## Quyết định có evidence

Dùng [decision template](templates/DECISION.md). Reviewer hỏi: gain có thể do data leakage, stronger head, search budget, preprocessing, missing cohort, future context, teacher supervision hay noise không? Cơ chế vẫn sống sau control nào? Phép thử kế có information gain cao hơn việc đổi idea không?

Chỉ scale khi protocol valid, baseline hợp lý và evidence dev đủ để đáng chi compute; claim publishable chỉ được chốt sau confirmatory evidence. Nếu kết luận không vững, thu hẹp claim, thiết kế phép thử khác hoặc kill hướng, không kéo dài training để tìm số đẹp.
