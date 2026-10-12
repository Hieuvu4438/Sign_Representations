# Decision memo — final clean-environment pilot

- Research question: logsig2 targets có thêm utility so với ordinary motion/anchors/
  timing dưới cùng data, model, masks và downstream budgets không?
- Hypothesis: signed-area level2 có thể bổ sung motion shape. Alternatives: static/RGB
  evidence, ordinary motion, regularization, detector quality hoặc thêm context giải thích
  score mà không cần signature.
- Discriminating comparison: E03/E04 × seeds17/29/43; output32D, same raw inputs,
  train-only fitting pool,100successful updates, batch4 và registered linear probe.
- Actual evidence: `evidence/phase4_comparison.json`, `phase4_handoff_audit.json`;
  configs `configs/verified_pilot`; runtime source hash `18794cff…`; source archives perrun.
- Correctness: 128tests pass; actual MVP/raw embed/export; closure/counterfactual;
  matched masks/source-frame exposures/target counts; geometry gradients trained.
- Result: E03/E04 cùng meanPC Top1=27.08±3.61% seedSD; paired difference0point trên
 16val samples. Signer-grouped sampling intervals rộng ởseed17/29; no positive evidence
  of a stable useful effect. E00 retrieval R@1=31.25%, temporal controls=16.67% mean.
- Quality:12low/4medium/0high availability-proxy val samples ở E02seed17 report.
  Missing hands/geometry là plausible bottleneck, không được sửa bằng fabricated zeros.
- Phonology/static: deterministic raw static-retention tests pass. No trusted annotation
  pool → real phonology/minimal-pair results not_run. Không suy ra linguistic retention
  từ correctness test hoặc English gloss.
- Costs: actual run/head timing và instrumented stage profile recorded; original total
  preprocessing GPU-hours unknown. Không có equal-GPU-hours/efficiency comparison.
- Decision: giữ E00 reference và E02 raw SSL baseline; retain E03/E04 runnable, không
  nâng signature branch thành default có performance claim. Không scale degree/lift.
- Next useful experiment, nếu tiếp tục nghiên cứu: frozen broader train/val protocol,
  stronger baseline budget, audited detector quality/static labels và matched E03/E04.
  Đây là future experiment, không là điều kiện để gọi optional backend đã implement.
- Claim supported: pipeline và controlled pilot hoạt động, no evidence yet for signature
  advantage in this small measured setting.
- Claim unsupported: full ASL Citizen benchmark, exact paper reproduction, SOTA,
  superiority, transfer, causal streaming hoặc continuous translation performance.
