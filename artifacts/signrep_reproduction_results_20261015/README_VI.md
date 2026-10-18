# Kết quả SignRep reproduction đã lưu

Đã lưu ngày 2026-10-15 và dọn runtime. Run hoàn thành **40.000 bước**, trong kế hoạch review 50.000, với **40.153 video train / 2.731 lớp**. Encoder khởi tạo từ Hiera-B MAE Kinetics, các SignRep heads khởi tạo mới.

| Bước | Proxy Top1 % | Proxy Top5 % | MRR |
|---:|---:|---:|---:|
| 10000 | 1.8674 | 4.7235 | 0.037758 |
| 20000 | 0.1098 | 0.4394 | 0.004973 |
| 30000 | 0.0732 | 0.2197 | 0.003892 |
| 40000 | 0.0000 | 0.1098 | 0.002732 |

Checkpoint tốt nhất ở bước **10.000**, giữ tại `best_step10000.model.pt`. Proxy gallery gồm 5.462 train videos, queries gồm 2.731 validation videos; mean pooling là phép đo chính. **Chưa có full validation/test; đây không phải paper benchmark reproduction.**

Run dừng do môi trường NumPy thay đổi khiến resume contract không khớp. Audit ghi nhận chất lượng suy giảm sau 10.000 bước, regularization áp đảo tín hiệu pose, style discriminator mất tác dụng và embedding giảm effective rank. Báo cáo chi tiết nằm trong `records.tar.gz`, đường dẫn `experiments/FULLASLC_AUDIT_20261014/REPORT_VI.md`.

## Bản lưu

- `RESULTS.json`: toàn bộ proxy metrics, ngân sách và phạm vi kết luận.
- `best_step10000.model.pt`: model inference tốt nhất, đã load kiểm tra 327 state entries.
- `best_proxy_features.tar.gz`: pooled features tại bước 10.000 để tính lại retrieval.
- `records.tar.gz`: source, configs, docs, evidence, manifests, experiment records, training metrics và audits.
- `run_logs.tar.gz`: console và progress logs.
- `RECORD_FILES.json`, `SHA256SUMS.json`, `ARCHIVE_VERIFIED.json`: hashes và kết quả kiểm chứng archive.
- `CLEANUP_REPORT.json`: những đường dẫn đã xóa và dung lượng thu hồi.

## Sau khi dọn

Đã xóa `method/jepa_baseline/runtime`, các checkpoint phục vụ audit và systemd service của job này. **Không còn optimizer checkpoint để resume run 40.000 bước.** Model tốt nhất vẫn dùng được cho inference/đánh giá hoặc làm điểm khởi tạo thí nghiệm mới. Dataset gốc, môi trường dùng chung và author checkpoint nằm ngoài folder reproduce được giữ nguyên. Source, tài liệu và experiment records nhỏ trong repo được giữ.

Dung lượng thu hồi ròng sau khi trừ bản lưu: khoảng **26,70 GB**. Bản lưu khoảng **352 MB**, phần source/records còn lại khoảng **169 MB**.
