## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-13
- Verification Status: ANALYZED (một seed; đã kiểm tra checkpoint và provenance)
- Version Label: jh3d-no-prior-report-vi-v1

## Kết quả ablation JH3D bỏ prior loss

**Bỏ prior loss làm chất lượng retrieval giảm mạnh trong cấu hình thí nghiệm này.** Train và pipeline đánh giá đã hoàn tất thành công.

## Thiết lập

- Bỏ toàn bộ chín prior hình học và activity loss, giữ JEPA latent loss (`prior_w=0`, `jepa_w=1`).
- Train từ cùng Kinetics-MAE initialization của ASLC002; đối chứng là checkpoint JH3D gốc, không train lại đối chứng.
- Giữ 3.000 video train, 4.000 bước, batch 16 video / 32 views, seed 20261008, 6 workers, mask 80%, predictor, AdamW/schedule và EMA 0.99925.
- Đánh giá trên cùng 10.304 video ASL Citizen validation bằng leave-one-signer-out; không đọc test.
- Metric chính đã khóa trước run: R@1 với whitening 512 chiều và mean pooling. Whitening fit trên validation theo protocol exploratory EX16 gốc.
- Train mất khoảng 78,6 phút; toàn bộ train + extraction + đánh giá hoàn tất khoảng 12:59 ngày 13/10/2026 (+07).

## Retrieval

| Embedding / metric | JH3D gốc | Bỏ prior | Chênh lệch (điểm %) |
|---|---:|---:|---:|
| Pipeline, white512 + mean, R@1 | 39.15% | 5.63% | -33.52 |
| Pipeline, white512 + mean, R@5 | 65.25% | 13.04% | -52.20 |
| Pipeline, white512 + mean, R@10 | 73.73% | 17.08% | -56.65 |
| Backbone trước fc, white512 + mean, R@1 | 38.05% | 5.59% | -32.46 |
| Pipeline, mean chưa whitening, R@1 | 12.50% | 0.25% | -12.25 |

Chênh lệch R@1 chính = -33.52 điểm phần trăm; signer-cluster bootstrap 95% CI [-36.67, -25.77]. Có 5/6 signer giảm, 1/6 bằng nhau, không signer nào tăng. CI của đối chứng trước fc là [-35.42, -24.94] điểm phần trăm. Bootstrap 10.000 lần trên 6 signer, không đo biến thiên giữa training seeds.

Khi bỏ prior, fc không nhận gradient. Đối chứng trước fc vẫn giảm 32,46 điểm phần trăm, nên mức giảm xuất hiện cả ở backbone; projection không được train không giải thích hết kết quả. Không suy ra mức đóng góp độc lập của từng prior hay của activity vì đã bỏ cả nhóm cùng lúc.

## Probe và hình học embedding

Các probe dưới đây dùng embedding pipeline, cùng nested ridge selection của evaluator gốc; R² càng cao càng tốt.

| Probe R² | JH3D gốc | Bỏ prior |
|---|---:|---:|
| Hình dạng bàn tay | 0.213 | -0.009 |
| Cơ thể | 0.163 | 0.029 |
| Vị trí / quan hệ | 0.584 | 0.257 |
| Chuyển động | 0.284 | 0.203 |
| Non-manual | -0.228 | -0.193 |

Signer probe accuracy tăng 92.36% → 97.90%. Mean cosine giữa clip embeddings tăng 0.607 → 0.972, cho thấy vector tập trung cùng hướng hơn. Đây không đủ để kết luận collapse hoàn toàn; effective rank centered là 121.3 → 150.2.

Latent training loss cuối giảm từ khoảng 0,332 ở JH3D gốc xuống 0,254 khi bỏ prior, trong khi retrieval giảm. Loss latent thấp hơn không đồng nghĩa representation hữu ích hơn cho phân biệt ký hiệu.

## Kiểm tra và giới hạn

- Checkpoint đủ 4.000 bước; mọi train log có reconstruction = 0 và total loss = latent loss. Backbone được cập nhật; fc/prior/activity heads giữ nguyên initialization.
- Hash toàn bộ input/source đã lưu khớp protocol. Baseline extraction đầy đủ khớp chính xác cache gốc (max absolute error = 0).
- Pipeline xác minh clip order, offsets và window starts giống nhau giữa hai nhánh.
- Đây là một training seed và 3.000 video; chưa thể khái quát cho JEPA nói chung, full dataset, nhiều seed, hoặc recipe latent-only đã được tuning riêng.
- Chỉ báo cáo validation, không diễn giải là test performance. Không chọn checkpoint hoặc pooling theo kết quả run.
- Rà soát 11/11 nhóm sai lầm thống kê của ARS: không đảo dấu khi chia signer; không suy luận cấp cá thể từ trung bình; dùng cùng cohort lọc của đối chứng, chưa khái quát ngoài cohort; không điều chỉnh collider hậu nghiệm; không bỏ qua nền xác suất; không thiết kế pre/post chọn cực trị; đủ cả 10.304 mẫu đánh giá; báo cả pooling/probes, không chỉ chọn outcome tốt; metric chính khóa trước run; can thiệp được kiểm soát nhưng kết luận giới hạn trong recipe; không đảo chiều can thiệp/kết quả. Giới hạn trọng yếu còn lại là một seed, chỉ 6 signer và whitening fit trên validation.

## Artifacts

- `evaluation.json`: toàn bộ retrieval, probes, geometry.
- `comparison.json`: delta và CI cho pipeline / backbone trước fc.
- `protocol.json`: điều kiện, versions và input hashes.
- `status.json`: trạng thái completed.
- `../../JH3D_NOPRIOR_20261013/last.pt`: checkpoint ablation.
