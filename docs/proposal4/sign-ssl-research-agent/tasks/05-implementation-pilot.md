# T05 — Implementation, pilot và phép kiểm tra cơ chế

## Điều kiện vào

experiment_enabled, protocol pilot đã chốt, dataset/checkpoint được phép dùng, budgets rõ và baseline runner hợp lệ. Nếu thiếu, ghi BLOCKED chính xác; không chạy job để “xem thử”.

## Hành động

1. Tạo implementation MVP trong project, giữ upstream riêng. Gắn code revision/hash và config cho baseline/method.
2. Kiểm tra có ý nghĩa: tensor/layout, loss masks/normalization, finite gradients, stop-gradient/EMA, metric known answers và leakage boundary.
3. Chạy smoke test nhỏ có log. Không dùng synthetic outputs làm accuracy evidence.
4. Chạy matched baseline rồi method với cùng data/init/steps/augmentation/compute cap.
5. Đo frozen retrieval/linear probe trên validation để xem encoder có lợi ích. Với pooling mới, chạy cùng pooling cho cả encoder.
6. Chạy simple control và ablation tối thiểu để phân biệt mechanism với extra parameters/compute/smoothing.
7. Audit failure cases theo group đã định nghĩa; chỉ dùng annotation hợp lệ. T-SNE/attention chỉ supplemental.
8. Ghi mọi run, bao gồm OOM, NaN, early stop, bad config, negative signal.
9. Reviewer ra CONTINUE/REVISE/REJECT/INCONCLUSIVE dựa protocol và uncertainty.

## Quy tắc điều chỉnh

Mỗi lần revise chỉ sửa một giả định chính, tạo config/run ID mới và lưu lý do. Không tuning test. Không giữ riêng best seed. Không so một baseline undertrained với method đã tuning nhiều.

Pilot một seed có thể dùng quyết định chi phí thấp; chưa hỗ trợ claim ổn định. Nếu control đơn giản đạt ngang method, chọn control hoặc sửa claim; không che kết quả.

## Outputs

- Implementation/code/config và environment lock.
- `research/runs/RUNxxx.md` bằng RUN_REPORT template.
- `research/EXPERIMENT_LEDGER.md` cập nhật.
- `research/pilot_analysis.md`: delta, cost, controls, uncertainty và next test.
- Cycle/review/decision records.

## Acceptance

Metrics có raw outputs và logs thật. Dữ liệu, code, configs, seed và protocol version liên kết được. G4 đạt chỉ khi pilot tiêu chí và mechanism-control không bác bỏ hướng.

Nếu chỉ smoke test chạy được, status PARTIAL. Nếu baseline bất hợp lệ, hủy scientific interpretation của comparison và quay T02.

