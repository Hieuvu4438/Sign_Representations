# Backlog nghiên cứu

Trạng thái ban đầu: mọi task chưa được agent thực thi. Audit và source seeds của bộ là inputs, không thay thế việc agent kiểm tra nguồn/config lúc chạy.

Status hợp lệ: TODO / IN_PROGRESS / PARTIAL / DONE / BLOCKED / SKIPPED_WITH_REASON. Mỗi cập nhật có ngày, artifact và acceptance checks.

| ID | Task | Dependency | Status | Artifact/checkpoint |
| --- | --- | --- | --- | --- |
| T00 | [Grounding, scope, tài nguyên](tasks/00-bootstrap.md) | Không | DONE (2026-10-05) | research/capability_audit.md, PROJECT_CONFIG.md, D001 |
| T01 | [Literature review, taxonomy, gap map](tasks/01-literature.md) | T00 scope sơ bộ | DONE (provisional coverage) | papers/P1xx–P4xx (58 notes), search_log_A–E, literature_synthesis.md, gap_map.md |
| T02 | [Baseline và data/evaluation audit](tasks/02-baseline.md) | T00 | DONE — G1 EVAL_REPRODUCED on test (C011) | capability_audit.md, EX01-v2, C008/C011, runtime/eval/* |
| T03 | [Hypotheses, novelty, phản biện](tasks/03-hypotheses-review.md) | T01 coverage + T02 audit | DONE | hypotheses/H002–H004, hypothesis_ranking.md, reviews/R001.md |
| T04 | [Method specification và protocol](tasks/04-method-protocol.md) | T03 qua G2 | DONE (per hypothesis) | protocols/EX02–EX06 |
| T05 | [Implementation, pilot, ablations](tasks/05-implementation-pilot.md) | T02 baseline hợp lệ + T04 + ngân sách | DONE — method pilots negative | pilot_analysis.md, EXPERIMENT_LEDGER.md |
| T06 | [Confirmatory evidence và paper memo](tasks/06-confirm-paper.md) | T05 qua G4 + protocol khóa | DONE for evaluation claim (EX05 test + EX06b SHuBERT val); method track PROPOSAL_ONLY | confirmatory_analysis.md, paper_readiness.md, paper_outline.md, paper/draft.md, reproducibility_checklist.md, open_questions.md |
| LOOP | [Research cycles và resume](RESEARCH_WORKFLOW.md) | Chạy sau mỗi evidence/decision | TODO | Chưa có |

## Quy tắc điều phối

Có thể làm T01 và phần audit T02 xen kẽ sau T00. Thiết kế T04 có thể tiến hành khi G1 mới AUDIT_ONLY, nhưng phải ghi assumption và chưa đủ quyền chạy T05.

Trong research_only, T02/T05/T06 có thể PARTIAL/BLOCKED ở phần runtime; T01, T03 và T04 vẫn phải tiến hành đến mức nguồn cho phép. Không tự skip baseline check hoặc confirmatory evidence để báo thành công.

Khi T03 bị reject, quay lại T01 bằng query mới hoặc chọn hypothesis khác. Khi T05 thất bại, diagnose một nguyên nhân trước; không tự thêm module. Khi T06 cần thêm evidence, ghi đúng experiment phân biệt còn thiếu.

Agent được tạo task con có ID ổn định, acceptance criteria và dependency. Chỉ thêm task nếu nó giải quyết một uncertainty/objection cụ thể.

