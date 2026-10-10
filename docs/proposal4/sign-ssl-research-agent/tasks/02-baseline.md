# T02 — Baseline, dữ liệu và evaluation audit

## Mục tiêu

Có baseline đánh giá được và hiểu khoảng cách paper–code. Không biến repo thiếu pipeline thành một reproduction tưởng tượng.

## Hành động audit không cần training

1. Pin commit, liệt kê file thực và release assets. Map paper components → source files → phần chưa cung cấp.
2. Kiểm tra environment/dependencies/imports/version constraints. Requirements không pin không đủ chứng minh reproducibility.
3. Xác định checkpoint architecture, key names, head, input shape, normalization và feature dùng downstream.
4. Audit example_usage.py: loading, CUDA assumptions, short/empty video, final partial window, stride, output pooling, activity scores và normalization.
5. Phân biệt code inference hiện có với preprocessing/sign-prior extraction, loss/discriminator, train loop, data loader, downstream eval và configs còn thiếu.
6. Đọc protocol retrieval/recognition/translation đúng paper version. Xác định train dictionary, query split, metric và selection history.
7. Tạo data audit: valid counts per split/class/signer, missing web clips, duplicates, provenance và contamination unknown.
8. Lập baseline grid: pretrained generic video encoder; SignRep checkpoint; simple objective/control; nearest sign SSL nếu khả thi.

## Nếu experiment_enabled

1. Tạo environment cô lập và lock dependencies sau khi xác minh import.
2. Xác minh checksum checkpoint/data được phép tải; log provenance. Không execute untrusted downloaded code ngoài scope.
3. Smoke inference: output shape/finite values/repeated eval determinism; lưu command/log và device.
4. Xây metric runner có known-answer checks. Gallery chỉ train; exclusion/self-match/ties/short-video/zero-activity policy rõ.
5. Reproduce frozen retrieval ở primary dataset trên validation trước. Đối chiếu activity-weighted với plain averaging và paper settings.
6. Rerun hoặc implement downstream baseline khi scope đòi hỏi. Ghi mọi adaptation và missing original component.
7. Đo baseline variability, runtime/memory và cost. Không mở test chỉ để kiểm tra score có giống paper.

## Outputs

- `research/baseline_audit.md`: source map, code gaps, version diff và mức tái lập.
- `research/DATA_AUDIT.md` và local split manifests khi dữ liệu có thật.
- `research/baseline_protocol.md`.
- `research/baseline_report.md`: số REPORTED riêng, số measured riêng; missing dùng N/A.
- Runner/code/config/environment lock khi đã triển khai.
- Run records thật nếu có.

## Acceptance

Audit có thể DONE với `AUDIT_ONLY`; phần runtime BLOCKED nếu không có resources. Toàn task ghi PARTIAL khi baseline evaluation còn thiếu.

Baseline hợp lệ cần command chạy lại được, logs, metrics đã sanity check, data fingerprint và provenance. Mọi discrepancy được giải thích hoặc giữ UNKNOWN. “Chạy được checkpoint” chưa là “reproduced paper”.

Nếu không tái lập original pretraining: ghi `continued-pretraining experiment`, `adapted baseline` hoặc `checkpoint evaluation` đúng thực chất. Không tạo train.py giả định tồn tại ở upstream.

