# Pose-JEPA Hybrid — kết quả đã lưu trước khi dọn runtime

Đã hoàn tất125.482updates,26lượt validation. EMA checkpoint115.000 được chọn bằng raw validation Rec@1.
Official train/gallery40.154; validation10.304; official test32.941;2.731glosses.

**Weights, pose caches, feature matrices và environment riêng được xóa sau archive verification. Không thể resume hoặc inference từ archive này; cần train lại/tạo weights mới. Code nguồn được giữ trong repo và source snapshots nằm trong archive.**

## Official test retrieval

| Readout | DCG | MRR | Rec@1 (%) | Rec@5 (%) | Rec@10 (%) | MacroRec@1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.2486 | 0.1132 | 5.74 | 15.28 | 21.94 | 5.78 |
| white512 | 0.3103 | 0.1725 | 9.32 | 24.05 | 33.36 | 9.38 |

Whitening fit valid train/gallery only; query lỗi giữ trong mẫu số.

## Frozen linear probe

| Split | InstanceTop1 (%) | InstanceTop5 (%) | MacroTop1 (%) | MacroTop5 (%) |
|---|---:|---:|---:|---:|
| validation | 34.78 | 62.74 | 35.36 | 63.41 |
| test | 22.90 | 46.36 | 23.08 | 46.55 |

Head selected epoch17 bằng validation; encoder frozen; test không dùng chọn head.

## Validation theo iterations

| Iteration | DCG | MRR | Rec@1 (%) | Rec@5 (%) | Rec@10 (%) | MacroRec@1 (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 0.2258 | 0.0895 | 4.16 | 11.89 | 17.74 | 4.15 |
| 10000 | 0.2500 | 0.1118 | 5.51 | 14.90 | 21.79 | 5.48 |
| 15000 | 0.2649 | 0.1260 | 6.48 | 16.98 | 24.72 | 6.48 |
| 20000 | 0.2779 | 0.1386 | 7.08 | 18.96 | 27.00 | 7.07 |
| 25000 | 0.2836 | 0.1444 | 7.52 | 19.67 | 28.07 | 7.51 |
| 30000 | 0.2890 | 0.1503 | 7.95 | 20.87 | 28.90 | 7.96 |
| 35000 | 0.2910 | 0.1520 | 7.99 | 20.98 | 29.39 | 7.97 |
| 40000 | 0.2931 | 0.1543 | 8.15 | 21.19 | 29.77 | 8.15 |
| 45000 | 0.2967 | 0.1585 | 8.59 | 21.73 | 30.03 | 8.64 |
| 50000 | 0.2966 | 0.1581 | 8.44 | 21.85 | 30.10 | 8.47 |
| 55000 | 0.2970 | 0.1582 | 8.38 | 21.94 | 30.38 | 8.42 |
| 60000 | 0.2987 | 0.1600 | 8.44 | 22.28 | 30.83 | 8.46 |
| 65000 | 0.3012 | 0.1632 | 8.97 | 22.13 | 30.81 | 9.00 |
| 70000 | 0.3031 | 0.1651 | 9.01 | 22.58 | 31.48 | 8.99 |
| 75000 | 0.3055 | 0.1675 | 9.13 | 22.97 | 31.88 | 9.14 |
| 80000 | 0.3068 | 0.1685 | 9.03 | 23.35 | 31.94 | 9.02 |
| 85000 | 0.3098 | 0.1716 | 9.36 | 23.64 | 32.64 | 9.33 |
| 90000 | 0.3112 | 0.1732 | 9.45 | 24.08 | 32.71 | 9.41 |
| 95000 | 0.3124 | 0.1747 | 9.70 | 24.25 | 32.83 | 9.65 |
| 100000 | 0.3130 | 0.1751 | 9.68 | 24.19 | 32.88 | 9.65 |
| 105000 | 0.3136 | 0.1758 | 9.75 | 24.01 | 32.95 | 9.73 |
| 110000 | 0.3135 | 0.1757 | 9.77 | 24.05 | 33.03 | 9.76 |
| 115000 | 0.3140 | 0.1762 | 9.85 | 24.11 | 33.20 | 9.85 |
| 120000 | 0.3140 | 0.1763 | 9.84 | 24.12 | 33.18 | 9.83 |
| 125000 | 0.3140 | 0.1762 | 9.83 | 24.15 | 33.15 | 9.83 |
| 125482 | 0.3140 | 0.1762 | 9.83 | 24.15 | 33.15 | 9.83 |

## Files và phạm vi archive

- `official_test_report.json`: report gốc đầy đủ, coverage và recipe.
- `validation/`, `validation_iterations.csv`:26lượt validation và ranks/failures từng query.
- `coverage.json`, `linear_probe_report.json`: coverage/failed IDs và classifier result.
- `comparison_plctx160_official_test.csv`: kết quả so sánh raw với PLctx160.
- `pose_jepa_results_evidence.tar.gz`: toàn bộ text evidence/runtime metadata, code/config snapshots, train logs, receipts, manifests, pilot/subset results và legacy rank artifacts.
- `archive_file_inventory.json`, `SHA256SUMS`, `archive_verification.json`: kiểm tra toàn bộ file đã lưu.
- `excluded_runtime_binaries.json`: các binary đã chủ động không giữ, gồm weights hashes.
- `cleanup_receipt.json`: phạm vi và dung lượng thực tế giải phóng, tạo sau cleanup.

Một seed42; không có multi-seed uncertainty/significance claims. Các thử nghiệm pilot nhỏ là evidence kỹ thuật, không thay thế full official test.
