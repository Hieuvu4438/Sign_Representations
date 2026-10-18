# proposal17: chẩn đoán "thông tin so với hình học" của PLctx full160

Mọi chẩn đoán chỉ dùng CPU và **không train model nào**. Chúng đọc embedding `_grp` 768-d đã cache của checkpoint gốc `method/phonojepa/runtime/runs/PJ_PLctx_FULL_160ep/final.pt` (gallery: `eval/PJ_PLctx_FULL_160ep_dictionary/aslc/train`, 40.154 clip train, 35 signer; query: `plctx160_continuation/baseline/feature_cache/val`, 10.304 clip val, 6 signer). **Split test không được tải ở bất kỳ script nào.** Nhãn gloss và signer chỉ dùng để chấm điểm hoặc ở các arm oracle đã khai báo; mọi biến đổi được fit trên train. `common.py` tái lập đúng baseline: **R@1 22,7193 / R@5 45,9433 / R@10 56,7838**.

Một số script cần embedding trích lại (`emb_variants.py`, ~35 phút CPU, nằm ở scratchpad `.../scratchpad/emb`, vài GB, không lưu vào repo): teacher / student / encoder ngẫu nhiên trên tập con (train mỗi clip thứ hai: 20.077; val mỗi clip thứ hai: 5.152; stride 16, cos 0,9975 với embedding stride 2 đã cache), cùng token đầu vào (`inp_features.py`) và một lượt perturbation hình thái trên 1.500 clip val.

| Script | Kết quả | Nội dung | Kết luận chính (xem proposal §2) |
|---|---|---|---|
| `common.py`, `sanity.py` | stdout | tải embedding, giao thức class-max cosine, bootstrap signer×gloss | khớp baseline |
| `geo_headroom.py` | `geo_headroom.json` | bậc thang thô (center, whiten, trung bình signer oracle, LEACE, LDA), đường cong gallery theo số signer và số clip/gloss | whiten +6,7; LDA ×2; gallery +3,5/lần gấp đôi |
| `geo_ladder.py` | `geo_ladder.json` | bậc thang đầy đủ có CI signer×gloss, theo signer, LDA lưới mở rộng | CORAL signer +9,1 so với whiten; LDA-96 50,9 |
| `ladder_cv.py` | `ladder_cv.json` | lập lại trên 5 tập 6 signer train giữ lại | bậc theo signer lặp lại; **whiten chỉ +1,5** |
| `nuisance_decomp.py` | `nuisance_decomp.json` | phương sai trong-gloss, η² theo signer/thời lượng/fps, hạng Sw/Sb, láng giềng cùng signer | 75 % trong-gloss; láng giềng cùng signer 85,4 % |
| `signer_norm.py` | `signer_norm.json` | CMN/CMVN/CORAL oracle; precision láng giềng xuyên signer | hiệp phương sai theo signer là phần chủ chốt |
| `emb_signer_cluster.py`, `eclus_control.py` | `emb_signer_cluster.json`, `eclus_control.json` | CORAL theo cụm không nhãn trên embedding `grp`; nhóm ngẫu nhiên đối chứng | cụm không nhãn +5…+8 so với whiten; ngẫu nhiên không |
| `cls_cluster.py` | `cls_cluster.json` | CLS làm mã signer: AMI, purity, CORAL theo cụm CLS | cụm CLS = oracle signer |
| `pseudo_signer.py` | `pseudo_signer.json` | pseudo-signer từ biến phong cách | thất bại (AMI val 0,15) |
| `local_norm.py` | `local_norm.json` | trừ trung bình láng giềng, CSLS (quy nạp) | 30,4 (k=200); CSLS ≈ 0 |
| `coral_head.py`, `coral_head2.py` | `coral_head*.json` | ánh xạ tuyến tính quy nạp với loss căn chỉnh cụm (signer thật, cụm CLS, ablation mean/cov) | head quy nạp +3…+6 so với whiten; cov quan trọng. `coral_head.py` chỉ hoàn tất cấu hình đầu (35,5), các cấu hình còn lại bị dừng vì chậm và được thay bằng `coral_head2.py` trên tập con |
| `nn_head.py` | `nn_head.json` | head InfoNCE trên positive láng giềng (không hạn chế, loại cụm, loại signer, oracle gloss), 3 vòng | **hại** ở precision hiện tại |
| `style_vars.py`, `nuis_pc.py` | `style_vars.json`, `nuis_pc.json` | ~25 biến phong cách; nhận diện signer; R² trên signer giữ lại cho phương sai trong-gloss | nhận diện 90 % nhưng ≈ 0 % phương sai giải thích |
| `emb_variants.py` | `extract.log` + scratchpad | trích embedding nhiều biến thể (teacher/student/ngẫu nhiên, mọi tầng, theo thứ tự, theo kênh, CLS), perturbation hình thái | — |
| `morph_effect.py` | `morph_effect.json` | độ nhạy với perturbation hình thái, năng lượng dịch trong không gian con signer | encoder gần bất biến; dịch nhỏ nhưng cùng hướng |
| `analyze_emb.py`, `ceilings.py`, `channel_analysis.py` | `emb_analysis.json`, `ceilings.json`, `channel_analysis.json` | tầng, student/teacher, ngẫu nhiên, đầu vào, theo kênh | training không khuếch đại signer; mặt mang signer; CLS = chữ ký |
| `inp_features.py` | `inp_features.log` + scratchpad | trung bình token đầu vào (không encoder) | trần đầu vào LDA 12,9 |
| `lex_probe_emb.py` | `lex_probe_emb.json` | probe 9 tham số ASL-LEX, gloss-rời + signer-rời, so với input và encoder ngẫu nhiên | encoder hơn input +5…+18 |
| `error_attrib.py` | `error_attrib.json` | lift tham số ASL-LEX của lỗi theo hình học | lỗi bám handshape/vị trí, không bám chuyển động |
| `retention_scalefree.py` | `retention_scalefree.json` | thống kê không phụ thuộc thang (participation ratio, η² trong toạ độ whiten, giải mã signer tuyến tính chuẩn hoá, láng giềng) cho encoder ngẫu nhiên / student / teacher | **rút lại** "JEPA không khuếch đại signer" của v1.0: giải mã 89,1 → 94,6 % |
| `ladder_extra.py` | `ladder_extra.json` | whitening riêng từng signer, đối chứng gallery signer nặng, LDA fit trên gloss rời (819 gloss giữ lại), lưới k LDA | LDA gloss-rời 66,0 so với raw 35,9; whitening theo signer 41,5 |
| `nn_precision_fullpool.py` | `nn_precision_fullpool.json` | precision láng giềng xuyên signer trên pool đầy đủ 40k; pool tập con với cụm CLS; head với positive từ hình học CORAL | sửa bảng pool 12k của v1.0; positive căn chỉnh: head 32,0–32,4 so với whiten 29,4 |
| `nn_head_cls.py` | `nn_head_cls.json` | head không nhãn: positive từ hình học CLS-CORAL, loại trừ cụm CLS | +0,8 so với whiten (tập con), trong nhiễu |
| `kfree_cluster_corral.py` | `kfree_cluster_corral.json` | CORAL theo cụm với K chọn bằng silhouette và trung bình trên K; CV signer | silhouette chọn K tệ ở val (29,45); trung bình trên K 33,4 / 31,6 (cụm CLS) / 24,7 (CV) |
| `routed_coral.py` | `routed_coral.json` | **N: chuẩn hoá theo cụm CLS định tuyến quy nạp**, thống kê từ gallery; đối chứng định tuyến ngẫu nhiên | 28,6 so với whiten 24,9; ngẫu nhiên 20,8–22,2 |
| `coral_head_mb.py` | `coral_head_mb.json` | mô phỏng mini-batch của module R (EMA thống kê theo cụm, sampler cân bằng cụm, cụm xáo trộn) | thất bại (12,3–21,2 < 24,9) |
| `cls_early.py` | `cls_early.json` | mã signer CLS ở checkpoint 4k/16k/160 epoch, ổn định chẵn/lẻ cửa sổ, NMI vs AMI | AMI 0,45/0,58/0,72; ARI 0,92–0,97; NMI vô dụng, AMI đúng |
| `morph_dose.py` | `morph_dose.json` | perturbation hình thái theo liều ±15/30/50 %, tập cân bằng 300 clip × 5 signer, biến thể không đồng nhất | R@1 không đổi, dịch ≤ 0,42 |
| `routed_controls.py` | `routed_controls.json` | đối chứng của N: K = 1, định tuyến theo cụm `grp`, 3 seed k-means, CV signer giữ lại trong nửa train | K=1 24,8 ≈ whiten; CLS 28,3–28,9 ± 0,1–0,3; `grp` 23,5–27,9; CV +2,3 (5/5 dương) |
| `morph_unittest.py` | stdout | unit test cho `morph()` NaN-safe (ở hệ số ≈ 1 không đổi cờ quan sát nào) | sửa lỗi R2-N1 |
| `subset_check.py` | `subset_check.json` | kiểm tra phụ thuộc kích thước gallery và stride | stride không ảnh hưởng raw/whiten; k LDA quan trọng |

## Giới hạn

- Val chỉ có 5 cụm signer hiệu dụng (P13 có 2 clip; ba signer giữ 87 % query). CI rất rộng; kết luận dựa trên Δ có cặp và nhất quán theo signer.
- Các phép chuẩn hoá oracle và theo cụm query là **transductive**; LDA dùng nhãn gloss train. Không có số nào ở đây tương đương với số raw của giao thức test.
- Các bảng trên tập con dùng nửa gallery: số tuyệt đối thấp hơn tập đầy đủ (R@1 18,3 so với 22,7 raw; LDA-96 40,1 so với 50,9). Chỉ so sánh trong cùng bảng.
- LDA: `k` và co rút chọn trên nhóm signer giữ lại trong train (`geo_ladder.py`: k=96, co rút 0,05). `analyze_emb.py` dùng k=256 cố định (thấp hơn tối ưu); `ceilings.py` dùng k=96.
- `ladder_cv.py` dùng clip train mà encoder đã thấy lúc SSL không nhãn (signer đã thấy), nên phép thử trộn "signer đã thấy" với "chưa thấy"; chỉ nhìn Δ giữa các bậc, không dùng để phán quyết độ bền của whitening.
- Chưa đo: `mw` (readout theo hoạt động), chamfer, SignRep với cùng stack, lớp N trên gallery đầy đủ 40k.
- Các script v1.1 sau review R1 (`retention_scalefree`, `ladder_extra`, `nn_precision_fullpool`, `nn_head_cls`, `kfree_cluster_corral`, `routed_coral`, `coral_head_mb`, `cls_early`, `morph_dose`) chạy qua `run_r1_fixes.sh` / `run_r1_fixes2.sh` (log `results/r1_fixes.log`). Sau review R2 (lỗi lan NaN trong `morph()`, ARI tầm thường), `run_r2_fixes.sh` chạy lại `morph_dose`, `morph_effect`, `cls_early` (ARI trên clip ≥ 4 cửa sổ) và `routed_controls` (log `results/r2_fixes.log`). Kết quả `morph_*` và `cls_early` cũ (v1.0/v1.1) đã bị thay thế.
