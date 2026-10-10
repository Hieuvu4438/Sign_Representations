# Working research memory

Chưa có kết luận nghiên cứu do agent. Không lưu phỏng đoán như fact.

## Invariants

- Primary domain: Sign Language Representation Learning bằng SSL.
- SignRep là anchor, không phải giới hạn literature.
- No experimental runs trong bộ ban đầu.
- Config mặc định research-only.
- Audit code khác reproduction; reported numbers khác measured numbers.

## Evidence-backed findings

- C011: SignRep frozen ASL Citizen test reproduced (0.4902 vs 0.4995 R@1).
- C012 (confirmatory, test once): ordered bins2 +5.9 R@1, chamfer +9.4 over the paper's averaged readout on the same frozen windows; WLASL +2.0/+1.2.
- C014: same direction for SHuBERT residual stream (val): +3.5/+4.0; author FFN-branch features are at floor for kNN.
- Chamfer gain = every query window finds a counterpart anywhere in the exemplar (q→g one-directional ≥ symmetric > partial).
- Same-signer gallery clips act as distractors (WLASL: 0.304 → 0.328 when removed).

## Rejected ideas / negative results

- H002 XWP cross-window latent prediction continuation: = matched continuation (D003).
- H003 CIP-Sign (user proposal5): 0/36 forecast gates on ASL Citizen geometry (D004).
- H004 label-free set-to-vector distillation: below training-free bins2 (D006).
- Rest-window subtraction readout: hurts. Learned aggregators from scratch under-fit.

## Current uncertainty / next checks

Capabilities, data availability, prior-art coverage, baseline validity và compute. Chi tiết ở STATE.md và config.

## Memory update rule

Chỉ giữ facts/decisions quan trọng, liên kết evidence. Không chép toàn paper vào memory. Khi finding bị bác bỏ, giữ ghi chú superseded và link decision; không xóa lịch sử.

