# PhonoPath-JEPA A1 — archived 19/10/2026

Đã hoàn tất train160epoch/200.640updates và official ASL Citizen test. Theo yêu cầu người dùng, đã lưu kết quả rồi xóa checkpoint, optimizer, feature cache và runtime lớn;4service systemd đã được gỡ.

- Bản lưu kết quả: [artifacts/phonopath_jepa_results_20261019](../../artifacts/phonopath_jepa_results_20261019/README_VI.md).
- So sánh các method trên official ASL Citizen: [báo cáo19/10](../../docs/experiments/method_comparison_20261019/REPORT_VI.md).
- A1 full160: readout chính grp+P R@1=6,928%; rawgrp11,511%; PLctxfull160 vẫn cao hơn với16,678%.
- Đã giữ đầy đủ metrics, ranks từng query, config/source hashes, log train nén, loss curve và bằng chứng kiểm chứng.
- Model weights và optimizer không được giữ; không thể resume hoặc inference run cũ. Code/config/tests và pose dùng chung được giữ.

Các hướng dẫn/service/runtime path cũ trong tài liệu job là lịch sử. Muốn chạy lại cần tạo job/train mới; không bật lại unit cũ trỏ vào runtime đã xóa.
