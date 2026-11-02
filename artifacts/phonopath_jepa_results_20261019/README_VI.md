# PhonoPath-JEPA A1 — kết quả đã lưu, 19/10/2026

A1 đã hoàn tất160epoch/200.640updates trên40.154 clip train. Official ASL Citizen test:40.154gallery/32.941query,2.731gloss, EMA teacher, cosine class-max/ties bi quan, PCA chỉ fit trên gallery train.

| Readout | Epoch40 test R@1 (%) | Epoch160 test R@1 (%) | Epoch160 R@5 (%) | Epoch160 R@10 (%) |
|---|---:|---:|---:|---:|
| grp | 11.305 | 11.511 | 28.005 | 37.100 |
| **grp+P (chính)** | **8.567** | **6.928** | **18.305** | **25.640** |
| mw | 11.059 | 9.872 | 25.078 | 34.298 |
| ord2 | 9.365 | 7.340 | 19.259 | 26.769 |

Kết quả run này không vượt PLctx full160 (`grp`16.678%); loss giảm không tương ứng lợi ích retrieval. Không chọn model/readout theo test. Đây là một seed, không có đối chứng A0/A2 cùng code và budget để quy nguyên nhân riêng cho signature.

## Nội dung bản lưu

- `results/epoch040_val`, `epoch040_test`, `epoch160_test`: JSON gốc, báo cáo và đủ ranks/ID/gloss/signer từng query cho4readout. NPZ ranks được nén lossless và đã đối chiếu lại từng array.
- `records.tar.gz`: code/config/snapshot, manifests, protocol/ledger, toàn bộ train log200.640update, console logs và evidence. Không chứa model weights, embedding/shard hoặc PCA matrices lớn. Một số normalization stats nhỏ được giữ.
- `loss_by_1000_steps.csv`: đường loss đã gom theo1.000unique optimizer steps; log gốc vẫn có trong tar.
- `metadata/source_inventory.json`: kích thước, inode và SHA256 tất cả file trước cleanup.
- `metadata/recomputed_metrics.json`: metrics tính lại từ ranks, khớp report.
- `services/`:4unit systemd và trạng thái trước khi gỡ.
- `SHA256SUMS.json`, `VERIFIED.json`: checksum và kiểm chứng toàn bộ archive.
- `CLEANUP_REPORT.json`: chỉ xuất hiện sau khi xóa hoàn tất, ghi chính xác dung lượng thu hồi.

**Theo yêu cầu giải phóng dung lượng, checkpoint/optimizer, các feature cache và runtime lớn được xóa sau khi archive đã kiểm chứng. Bản lưu kết quả không dùng để resume hoặc inference model cũ; cần train lại để tạo weights.** Code/config/tests trong `method/phonopath_jepa` được giữ. Pose cache, môi trường và các method khác được giữ nguyên.

Có thể xem lại text record bằng `tar -tzf records.tar.gz`; trích file cần dùng vào thư mục scratch riêng. Chỉ metrics từ ranks là tái tính được từ bản lưu này; không tuyên bố tái chạy encoder khi weights đã xóa.
