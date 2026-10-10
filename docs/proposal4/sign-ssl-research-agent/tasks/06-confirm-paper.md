# T06 — Confirmatory evaluation và quyết định hướng paper

## Điều kiện vào

Pilot qua G4; baseline hợp lệ; confirmatory protocol LOCKED trước run; budget/data/test access rõ. Có thể chuẩn bị paper outline trước gate, nhưng không viết empirical claims.

## Hành động thực nghiệm

1. Freeze hypothesis, code/config, primary metric, δ/ε, seeds, tuning cap, checkpoint rule, splits và CI procedure.
2. Chạy baseline/method với multi-seed planned comparison; log mọi seed và failure.
3. Đánh giá test một lượt theo locked protocol sau validation selection. Ghi lịch sử truy cập test.
4. Chạy ablations/negative controls/transfer phù hợp claim; không mở rộng claim từ kết quả exploratory chưa xác nhận.
5. Tính effect, uncertainty và cost. Tách seed variance với sample uncertainty; resample đúng dependencies.
6. Kiểm tra contamination/provenance; uncertainty upstream chưa giải quyết phải vào limitations.
7. Refresh prior art trước novelty conclusion. Nếu có work gần mới, cập nhật delta hoặc giảm claim.
8. Reviewer kiểm tra strongest alternative explanation còn lại, reproducibility và failure cases.

## Quyết định

- READY_FOR_DRAFT: claims chính có evidence thật đúng scope, novelty vẫn defensible, baseline công bằng, controls và uncertainty đủ.
- NEED_MORE_EVIDENCE: ghi cụ thể evidence còn thiếu và experiment có thể thay đổi quyết định.
- PIVOT: hướng không meaningful/không distinct hoặc cơ chế bị bác bỏ.
- Nếu thiếu compute: PROPOSAL_ONLY, không dựng bảng kết quả giả.

Paper-worthy không đồng nghĩa SOTA mọi benchmark. Có thể là cơ chế rõ với giới hạn tốt, efficiency có non-inferiority, hoặc evaluation finding được chứng minh. Không hứa acceptance.

## Outputs

- `research/confirmatory_analysis.md`.
- `research/paper_readiness.md`: decision + evidence-to-claim table.
- `research/paper_outline.md`: problem, contributions, method, experiments, limitations.
- `research/paper/draft.md` chỉ khi READY_FOR_DRAFT; nếu chưa, `research/paper/proposal.md`.
- `research/reproducibility_checklist.md`: exact commands/configs/data hashes/checkpoints/metric code.
- `research/open_questions.md`.
- Ledgers/state/decisions cập nhật.

## Acceptance

- Mọi kết quả dùng trong kết luận có run IDs, raw artifacts và protocol hợp lệ.
- Claim chính liên kết evidence, uncertainty, controls và scope; novelty có ngày refresh.
- Paper readiness memo chọn một quyết định rõ và nêu evidence còn thiếu nếu có.
- Quyết định negative/pivot được xem là đầu ra nghiên cứu hợp lệ, không bị viết lại thành thành công.
- Chưa có confirmatory runs thì task chỉ PARTIAL/BLOCKED; proposal không thay thế empirical validation.

## Draft requirements

Mỗi contribution có claim ID và supporting runs/controls. Tách reported/reproduced/exploratory. Mọi result table lấy từ machine-readable logs và run ledger; không nhập số bằng intuition.

Viết limitations thực: nguồn teacher, data diversity, languages/domains, signer variation, compute, preprocessing, protocol và expert validation chưa có. Không bịa Deaf-community feedback.

Không tự publish/push/submit. Khi user muốn xuất bản, hoàn thành draft reviewable trước hành động bên ngoài.
