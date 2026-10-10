# T00 — Grounding và thiết lập scope

## Mục tiêu

Xác định domain, tài nguyên, supervision và đầu ra có thể làm thật. Không bắt đầu bằng việc chọn một method đẹp.

## Inputs

CLAUDE.md, PROJECT_CONFIG.md, context/SIGNREP_AUDIT.md, context/SOURCE_REGISTRY.md và state hiện tại.

## Hành động

1. Kiểm tra khả năng đọc web/paper/repo, filesystem, code runtime và persistence. Ghi capability thật, không mặc định có GPU hoặc browser.
2. Đọc paper SignRep v1 và appendix; kiểm tra bản mới/final proceedings. Ghi version dùng. Nếu nguồn không truy cập được, đánh dấu mức đọc đúng.
3. Recheck repo commit/tree/releases bằng API hoặc checkout read-only. So snapshot với audit khởi đầu; không tự tải checkpoint khi chưa cấp storage.
4. Chọn một research question ban đầu, ví dụ: representation nào giữ thông tin dấu và generalize qua signer/domain với RGB inference?
5. Chọn primary track và một benchmark khả thi. Lập phương án ít tài nguyên trước phương án large-scale pretraining.
6. Kiểm tra dataset availability/terms/official splits; ghi thiếu dữ liệu, provenance hoặc metadata.
7. Lập kế hoạch ngân sách và search. Không hỏi dồn nhiều câu; chỉ hỏi missing choice có tác động trực tiếp và không suy ra được.
8. Cập nhật config, state, tasks. Tiếp tục T01/T02 audit nếu compute chưa có.

## Outputs

- `research/research_brief.md`: problem, motivation, main question, scope, supervision, primary/secondary evaluation, deliverables.
- `research/capability_audit.md`: tool/runtime/data/GPU/storage có thật.
- `research/resource_plan.md`: research-only, small pilot, confirmatory options với assumptions và estimates.
- Config/state cập nhật.

## Acceptance

- Main question cụ thể, không chỉ “beat SignRep”.
- Source version/commit và scope được ghi.
- UNKNOWN chưa được ngụy trang thành giả định đã xác nhận.
- Ngân sách download/training rõ; defaults giữ research-only.
- Next task có inputs và hành động cụ thể.

## Blockers

Nếu không đọc được paper, vẫn audit metadata/repo và chuẩn bị search queries; không gọi G0 đạt khi anchor chưa full-read. Nếu không có compute, T00 vẫn hoàn thành với giới hạn rõ.

