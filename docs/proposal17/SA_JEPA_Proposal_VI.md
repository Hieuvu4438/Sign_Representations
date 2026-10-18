# Proposal 17 — SA-JEPA: chiến lược JEPA cho biểu diễn ký hiệu từ skeleton, xây trên chẩn đoán "thông tin so với hình học"

## Không dùng Path-JEPA. Một lớp hình học không nhãn, cộng các thay đổi huấn luyện JEPA có kiểm soát

**Trạng thái (2026-10-17): chỉ đề xuất, v1.2 sau hai vòng review độc lập (R1: 2 Critical, 12 Major; R2 xác minh: 1 Critical mới là lỗi lan NaN trong perturbation, đã sửa; xem §9).** Chưa train model nào. Mọi con số [D] đến từ chẩn đoán CPU trên embedding PLctx full160 đã có, **chỉ dùng train (gallery) và val (query), không đụng test**. Mọi dự báo về SA-JEPA là [I], chưa kiểm chứng bằng huấn luyện. Bản v1.0 lưu ở `archive/`.

Ký hiệu nguồn: **[D]** chẩn đoán trong `diagnostics/` · **[V]** số đã đối chiếu HTML thô của paper (`literature/Q1–Q3`) · **[I]** suy luận / dự báo · **[P]** số trong paper chưa kiểm lại · **[R]** số nội bộ của repo (report cũ).

---

## 0. Tóm tắt điều hành

**Câu hỏi.** Nếu bỏ hướng Path-JEPA, nên áp dụng chiến lược JEPA nào cho SSL biểu diễn ký hiệu trên skeleton (downstream: retrieval từ điển đóng băng kiểu SignRep, ISLR), và nên xây method thế nào?

**Cách làm.** Đo trước nút thắt của PLctx full160 (test R@1 16,68; val 22,7) bằng chẩn đoán CPU trên chính embedding của nó, rồi mới chọn chiến lược. Ba báo cáo literature (`literature/Q1–Q3`) cung cấp bản đồ chiến lược. Một review độc lập (R1) đã tái chạy nhiều chẩn đoán và buộc tôi rút lại hoặc làm yếu 3 kết luận của v1.0.

**Kết quả chính (val; gallery 40.154 clip train / 35 signer; 10.304 query / 6 signer, thực chất 5 cụm; tái lập chính xác R@1 22,7193):**

1. **Khoảng cách phần lớn là hình học, và nó tổng quát hoá qua gloss.** Một ánh xạ tuyến tính LDA fit **chỉ trên các gloss khác** nâng R@1 trên 819 gloss giữ lại từ 35,9 (raw) / 44,4 (whiten) lên **66,0**; fit trong mẫu trên toàn val: 22,7 → 50,9 [CI 45,7–54,2]. Encoder đã học nhiều: probe gloss-rời + signer-rời trên 9 tham số ASL-LEX hơn input +5…+18 điểm và hơn encoder ngẫu nhiên +5…+13 (LDA 12,9 → 21,2 → 40,1).
2. **Nhiễu chi phối là signer, ở bậc hai (trung bình + hiệp phương sai).** 75 % phương sai nằm trong cùng gloss; láng giềng top-1 trong train cùng signer **85,4 %** (ngẫu nhiên 5,9 %), cùng gloss 5,5 %. Chuẩn hoá theo signer (oracle): trừ trung bình +2,2 so với whiten; CORAL +9,1; **whitening riêng từng signer +12,1** (41,5). Kết luận được ủng hộ là "chuẩn hoá bậc hai theo miền loại phần lớn nhiễu", **không** phải "signer là một ánh xạ tuyến tính" (§2.4).
3. **Token CLS là mã signer có sẵn, không nhãn.** AMI(signer) 0,45 / 0,58 / 0,71 ở checkpoint 4k bước / 16k bước / 160 epoch; AMI(gloss) ≈ 0 (độ ổn định giữa cửa sổ: xem §2.5). Hai cách dùng không nhãn, đều đã đo:
   - **CORAL theo cụm CLS ở thời điểm truy vấn (transductive)**: tập con, whiten 24,9 → trung bình trên mọi K thử 31,6 [30,2–32,6] (CORAL signer thật: 31,7);
   - **định tuyến quy nạp theo CLS (mỗi query chỉ dùng CLS của nó, thống kê cụm lấy từ gallery)**: **28,3 ở K = 35 (+3,4 so với whiten; cao nhất 28,6 ở K = 16; trung bình 5 giá trị K: +3,0; K = 70 chỉ 26,7)**; reviewer tái tạo +3,4…+3,7 qua 3 seed k-means, cả 5 signer val cùng cải thiện, SE ghép cặp 0,55; định tuyến **ngẫu nhiên** cho 20,8–22,2 (tụt dưới whiten). Đây là tăng quy nạp, không nhãn, không dùng thống kê batch query, nhưng mới đo trên một checkpoint, nửa gallery và 5 signer val.
4. **Căn chỉnh *trong* encoder chưa được kiểm chứng.** Head tuyến tính với thống kê toàn batch cho +3,6…+6,1 so với whiten, nhưng **bộ ước lượng mini-batch của thiết kế v1.0 (32 clip, EMA thống kê theo cụm) làm tụt xuống dưới điểm khởi tạo** (17,8–21,2 so với 24,9; batch 256: 12,3). Đối chứng của reviewer cho thấy lỗi nằm ở bộ ước lượng (thống kê trễ so với W đang đổi, hiệp phương sai hạng 1), không phải ở độ dài huấn luyện: cùng trọng số và lr với thống kê toàn batch cho 28,4, còn cụm xáo trộn 22,2–23,1. Đây **không** chứng minh R thất bại trong encoder; nó chứng minh thiết kế v1.0 không dùng được. R không vào GPU trước khi qua cổng G-R (§6).
5. **Các giả thuyết bị chẩn đoán bác bỏ hoặc làm yếu:**
   - Nhiễu signer không được tái tạo bởi biến đổi hình thái giải tích clip-hằng: arm/hand/không đồng nhất theo từng ngón, hệ số ±15/30/50 %, trên 1.500 clip cân bằng 5 signer (hàm perturbation đã sửa lỗi lan NaN do review R2), dịch embedding **tăng theo liều nhưng nhỏ**: 0,08 → 0,15 → 0,21–0,23 (cánh tay, tất cả, không đồng nhất; riêng bàn tay 0,02–0,04) so với 0,90 giữa cùng-gloss-khác-signer; R@1 đổi ≤ 1,2 điểm (cánh tay ±30–50 %: −1,1…−1,2; còn lại trong nhiễu). 30 biến phong cách nhận diện signer 90,4 % (hình thái riêng 72,6 %) nhưng giải thích ≈ 0 % phương sai trong-gloss ở signer giữ lại (kiểm định yếu về công suất, §2.7).
   - **"JEPA không khuếch đại signer" của v1.0 bị rút lại.** Trên số đo không phụ thuộc thang: giải mã signer tuyến tính (chuẩn hoá) 89,1 % (ngẫu nhiên) → 94,6 % (đã train); láng giềng cùng signer 71,9 → 84,4 %. η² có trọng số phương sai giảm (0,27 → 0,14) chỉ vì encoder ngẫu nhiên rất dị hướng (participation ratio 7,5 so với 83,5).
   - NN-contrastive tuning: positive mined trong hình học raw/whiten **làm hại**; positive mined trong hình học đã căn chỉnh chỉ **cải thiện nhẹ** (head: +3,0 so với whiten với CORAL signer thật trên pool đầy đủ; +0,8 với cụm CLS không nhãn trên tập con). Precision tương hỗ trên pool đầy đủ (raw 25,9 / whiten 33,4 / CORAL signer 44,8 %) **đã sửa** so với v1.0 (bảng cũ dùng pool 12k).
   - Chọn tầng, student/teacher, readout có thứ tự: không giúp; CLS không dùng làm readout (R@1 0,25).
6. **Độ bền.** Val chỉ có 5 cụm signer (Kish ≈ 3,8). Trên 5 lần tách 6 signer train giữ lại, các bậc theo signer lặp lại (CORAL +9,2; CORAL không nhãn +7,6; LDA +25,8 so với raw), còn whitening toàn cục chỉ +1,4 (so với +6,7 trên val), nhưng phép thử đó trộn "signer đã thấy" với "chưa thấy" nên **không phán quyết được** whitening có mong manh hay không.

**Đề xuất: SA-JEPA v1.2, gồm ba lớp:**

| Lớp | Nội dung | Trạng thái bằng chứng |
|---|---|---|
| **G. Hình học (không train)** | readout chỉ tay, bỏ CLS; PCA-whiten không nhãn; **chuẩn hoá theo cụm CLS định tuyến quy nạp (N)**; báo cáo riêng bản transductive | đã kiểm trên CPU (một checkpoint, tập con) |
| **J. Thay đổi huấn luyện JEPA** | **T** center target theo cụm CLS cố định; **DYN** (multi-depth, multi-span, hiệu thời gian, khối tương lai); **H** chỉ tay; **trục dữ liệu đa signer** | giả thuyết, cần Phase 2 |
| **Có điều kiện** | **R** căn chỉnh hiệp phương sai trong encoder (qua cổng G-R); **NN-tuning** ở mức encoder (qua cổng G-NN) | chưa đạt cổng |

**Bị loại bằng chứng:** retarget hình thái giải tích; NN-tuning với positive raw/whiten; SIGReg/VICReg làm mục tiêu chính; conditioning predictor gây equivariance; readout có thứ tự đơn giản; CLS alignment; dùng NMI(gloss) làm luật dừng.

**Dự báo thực tế [I]:** lớp G một mình có thể nâng val raw từ 22,7 lên khoảng 26–33 (cận dưới nếu whitening chỉ cho +1,4; test khoảng 20–24, vì tỉ lệ test/val 0,73 chỉ đo trên raw), chưa kiểm trên tập đầy đủ; lớp J cộng thêm 0–4 điểm, không có bằng chứng huấn luyện nào. Khoảng cách tới SignRep raw (49,0 đo local) **vẫn lớn**; LDA 66,0 trên gloss giữ lại dùng nhãn nên không so được.

**Chương trình:** Phase 0 chẩn đoán CPU (xong) → Phase 1 chuẩn hoá lớp G trên tập đầy đủ + `mw`/chamfer → Phase 2 screening 40 epoch (A0×3 + ~15 arm, 1 seed) → Phase 3 xác nhận 3 seed → Phase 4 trục dữ liệu + full160 + test một lần → Phase 5 downstream. **Ngân sách ≈ 62–82 GPU-h** (≈ 37 bản tiết kiệm), tuần tự. Cần bạn duyệt trước khi dùng GPU.

**Điều bạn cần quyết định:** (i) cho phép lớp G (chuẩn hoá theo cụm CLS từ gallery) được báo cáo như một hàng riêng, khai báo "inference-time, không nhãn"; (ii) cho phép signer id chỉ ở các arm chẩn đoán `m`; (iii) mở trục dữ liệu MS-ASL/WLASL (222/119 signer); (iv) lịch GPU khi job PhonoPath full160 đang chạy.

---

## 1. Bối cảnh và vị trí

- **Đã có:** PLctx full160 (token 7 kênh × 32 bin, mask PLCM-lite, centered-CE EMA, readout `grp`) test R@1 16,68, val 22,7; tiếp tục train đến khoảng 164–180 epoch cho val +0,23 / test −0,05 [R] (bão hoà). `docs/research` đã có phễu 15 arm; `docs/proposal16` (PhonoPath-JEPA) đang chạy ở `method/phonopath_jepa`. **Chưa có** ở tài liệu nào: phép đo định lượng nút thắt của PLctx, và một chiến lược xuất phát từ phép đo đó.
- **Khác ở đâu:** đo trước, loại arm đã bị bác bỏ (retarget V1, NN-tuning O8 với positive thô, isotropy objective), thêm họ chiến lược mới (căn chỉnh theo miền signer, qua mã CLS không nhãn) mà `docs/research` chưa có, và tách rõ phần *không cần train* (lớp G) khỏi phần cần GPU.
- **Ràng buộc giữ nguyên:** skeleton RTMW-x COCO-WholeBody 2D; method chính không dùng nhãn gloss; test ASL Citizen đã dùng một lần; một job GPU mỗi lúc, GPU dùng chung.

---

## 2. Chẩn đoán

**Thiết lập và toàn vẹn.** Embedding `_grp` 768-d của checkpoint gốc `PJ_PLctx_FULL_160ep/final.pt` (EMA teacher), đã cache: gallery 40.154 clip train (35 signer), query 10.304 clip val (6 signer; P13 chỉ có 2 clip; ba signer P21/P26/P39 giữ 87 % query). `common.py` tái lập **R@1 22,7193 / R@5 45,9433 / R@10 56,7838** (khớp `validation_iterations.csv`) và review độc lập đã kiểm bằng vòng lặp brute-force. Mọi biến đổi fit chỉ trên train; siêu tham số chọn trên nhóm signer giữ lại trong train (chọn K, k và co rút nêu rõ ở §2.11). Test không bao giờ được tải. Bảng script ↔ kết quả ở `diagnostics/README.md`.

### 2.1 Bậc thang retrieval (val, `geo_ladder.py`, `ladder_extra.py`)

| Bậc | Điều kiện | Val R@1 | Δ so với raw (CI 95 % bootstrap signer→gloss) | Ghi chú |
|---|---|---:|---|---|
| 0 | raw (giao thức) | **22,72** [16,8–27,7] | — | |
| 1 | PCA-whiten (k = 512, fit train, không nhãn) | 29,44 | +6,7 (+5,2…+8,2) | quy nạp; độ bền chưa phán quyết (§2.11) |
| 2 | trừ trung bình láng giềng k = 200 | 30,41 | +7,7; so với whiten [−0,2…+2,3] | quy nạp, không nhãn; k chọn trên val (12 cấu hình) |
| 3 | LEACE signer + whiten | 31,66 | +8,9 | signer id (tín hiệu yếu) |
| 4 | CMN theo signer + whiten | 31,66 | +8,9 | oracle |
| 5 | CORAL theo cụm trong `grp` + whiten | 34,58 | +11,9; so với whiten [+2,2…+8,5] | transductive; **K = số signer val (oracle)**; không oracle: §2.5 |
| 6 | CORAL theo signer thật + whiten | 38,53 | +15,8; so với whiten [+7,6…+11,2] | oracle, transductive |
| 7 | **whitening riêng từng signer** + whiten | **41,51** | +18,8 | oracle, transductive |
| 8 | LDA có giám sát gloss (k = 96) | **50,88** [45,7–54,2] | +28,2 | trần tuyến tính **trong mẫu** (§2.2) |

Theo signer (R@1 raw → CORAL oracle → LDA): P12 14,6 → 33,3 → 43,8; P21 20,2 → 36,8 → 52,9; P26 21,5 → 33,9 → 47,2; P39 30,3 → 47,6 → 55,3; P5 13,6 → 31,4 → 45,3. Mọi signer cải thiện ở mọi bậc; bản thân R@1 raw dao động 13,6–30,3 theo signer. Khoảng tin cậy signer-cấp (t, df = 4; do reviewer) cho CORAL − whiten +10,0 [6,9; 13,2], whiten − raw +6,5 [4,6; 8,4]; cả hai loại trừ 0.

### 2.2 Trần LDA không chỉ là ghi nhớ gloss (`ladder_extra.py`)

LDA ở bậc 8 fit và chấm trên cùng 2.731 gloss, nên tự nó chưa chứng minh "hình học". Thử nghiêm hơn: chia gloss 70/30 (seed 0, trùng phân hoạch của probe; 819 gloss giữ lại); gallery = clip train của gloss giữ lại, query = clip val của chúng; ánh xạ fit **chỉ trên clip train của 70 % gloss kia**.

| Điều kiện (819 gloss giữ lại) | R@1 |
|---|---:|
| raw | 35,9 |
| PCA-whiten fit trên gloss khác | 44,4 |
| LDA fit trên **gloss rời** (k = 32 / 48 / 64 / 96 / 128) | 64,3 / 64,7 / **66,0** / 65,4 / 65,3 |
| LDA fit trong mẫu trên chính 819 gloss (k = 64 / 96) | 67,3 / 68,0 |

Một ánh xạ tuyến tính học từ gloss khác tái tạo 96–98 % trần trong mẫu, nên khoảng cách này **tổng quát hoá qua gloss**: nó là hình học, không phải ghi nhớ nhãn. (Hệ số tăng LDA/raw lớn hơn với encoder ngẫu nhiên: 5,0× so với 2,2×, nên "dư địa" tự nó là đặc tính chung của readout có giám sát so với không giám sát.) Lưới k của LDA trên toàn val: 32 → 48,6; 48 → 50,0; 64 → 50,3; 96 → 50,9.

### 2.3 Cấu trúc nhiễu (`nuisance_decomp.py`, train 40.154 clip)

| Đại lượng | Giá trị |
|---|---|
| phương sai trong cùng gloss | 75,4 % (η²_gloss thô 24,6 %; giá trị null do xáo gloss 6,8 % ở N = 40k, 13,6 % ở N = 20k) |
| η²_signer của tổng / của phương sai trong-gloss | 13,8 % / 17,5 % |
| nhân tố kỹ thuật trong phương sai trong-gloss | thập phân vị thời lượng 2,9 %; fps 0,5 %; độ phân giải 0,03 % |
| participation ratio của Sw (trong-gloss) / Sb (giữa-gloss) | 84,4 / 26,9 |
| phần phương sai Sb nằm trong 100 hướng đầu của Sw | 80,9 % |
| không gian con signer 34 chiều chứa | 39,1 % vết Sw; 20,0 % vết Sb |
| láng giềng top-1 trong train: cùng signer / cùng gloss | **85,4 %** (ngẫu nhiên 5,9 %) / 5,5 % |
| láng giềng top-1 buộc khác signer: cùng gloss | 14,4 % |

**Đường cong gallery (raw, val)**, đây là hiệu ứng của **gallery lúc truy vấn**, không phải của dữ liệu tiền huấn luyện: số clip mỗi gloss 1→2→4→8→16: 8,2 → 11,1 → 14,4 → 18,8 → 22,7 (≈ +3,5 mỗi lần gấp đôi); số signer trong gallery 4→8→16→24→35: 9,8 → 14,9 → 18,9 → 20,5 → 22,7 (≈ +3,4 mỗi lần gấp đôi: 8→16 +4,0; 16→35 +3,8 cho 1,13 lần gấp đôi). Các tập con theo signer cũng đổi số clip mỗi gloss.

### 2.4 Chuẩn hoá theo signer: cái gì thực sự có tác dụng? (`signer_norm.py`, `ladder_extra.py`)

| (val R@1, oracle signer, transductive) | không whiten | + whiten |
|---|---:|---:|
| CMN (trừ trung bình) | 28,4 | 31,7 |
| CMVN (trung bình + phương sai theo chiều) | 28,8 | 31,2 |
| CORAL (căn chỉnh hiệp phương sai về toàn cục) | 30,0 | 38,5 |
| **whitening riêng từng signer, không tô màu lại** | **42,7** | **41,5** |

- CORAL + whiten gần như đại số trùng với whitening riêng từng signer; hơn nữa whitening riêng từng signer thắng CORAL. Thao tác hiệu quả vì vậy là **whitening theo miền**, sửa dị hướng và hubness *riêng từng signer*; điều này **không** chứng minh "signer là một ánh xạ tuyến tính của cùng một biến ẩn". Mô hình chỉ cần: thống kê bậc hai theo signer khác nhau và cần cân bằng.
- **Đối chứng thành phần gallery.** 12 signer có ≥ 1.500 gloss khác nhau giữ 79,7 % clip train (32.015). Gallery chỉ gồm họ: raw 21,05; whiten 27,96; CMN+whiten 30,18; CORAL+whiten 36,24; whitening riêng từng signer 39,57. CORAL − whiten co từ +9,1 xuống +8,3 (CORAL − CMN từ +6,9 xuống +6,1). Ngưỡng ≥ 1.500 gloss được chọn sau khi xem dữ liệu; với ngưỡng chặt hơn của reviewer R1 (≥ 2.400 gloss, 9 signer, 66 % clip) CORAL − whiten co còn +6,5 (khoảng 28 % so với +9,1). Hiệu ứng signer còn nguyên nhưng **khoảng một phần tư đến một phần ba của con số tiêu đề phụ thuộc thành phần gallery**.
- **Đối chứng nhóm ngẫu nhiên:** CORAL gallery theo signer thật với nhóm query ngẫu nhiên (K = 6): 32,7; ngẫu nhiên cả hai phía: 27,5 (dưới whiten 29,4). Cụm thật đóng góp +4,8 so với nhóm ngẫu nhiên phía query.

### 2.5 CLS là mã signer, và chuẩn hoá theo cụm CLS không cần nhãn (`cls_cluster.py`, `kfree_cluster_corral.py`, `routed_coral.py`, `cls_early.py`)

**CLS.** Trên tập con (gallery 20.077, query 5.152): η²_signer của CLS 0,44 (so với 0,14 của `grp`), láng giềng cùng signer 96,6 %, R@1 raw của CLS 0,25. K-means CLS: AMI với signer thật 0,71 / 0,70 / 0,66 trên train (K = 35 / 70 / 140), 0,44 / 0,455 trên val (K = 6 / 12). **Mã này có từ rất sớm và ổn định** (`cls_early.py`, tập 10.039 clip train, 2.576 clip val):

| Checkpoint | AMI signer train (K=35) | AMI signer val (K=6) | AMI gloss train | ARI chẵn/lẻ cửa sổ, clip ≥ 4 cửa sổ (train / val) | giải mã signer tuyến tính (chuẩn hoá) |
|---|---:|---:|---:|---:|---:|
| pilot 4k bước | 0,45 | 0,31 | 0,009 | 0,83 / 0,55 | 85,9 % |
| pilot 16k bước | 0,58 | 0,47 | −0,006 | 0,86 / 0,41 | 88,7 % |
| 160 epoch | 0,71 | 0,49 | −0,013 | 0,94 / 0,63 | 94,6 % |

(**ARI chỉ tính trên clip có ≥ 4 cửa sổ stride 16**: 1.164 clip train và 283 clip val; 74 % clip train và 67 % clip val chỉ có một cửa sổ nên chẵn = lẻ và ARI 0,97 / 0,80 của bản trước là tầm thường, theo review R2.) Các checkpoint pilot là pilot 3.000 clip (khoảng 3–13 epoch dữ liệu thật), proxy cho "sớm", không phải epoch 10/40 của run đầy đủ, và cả ba được huấn luyện **không** có T/R. **Kết luận:** mã signer trong CLS có từ sớm và AMI(gloss) ≈ 0, nhưng độ ổn định giữa cửa sổ ở signer chưa thấy chỉ vừa phải (ARI val 0,41–0,63), nên việc dùng CLS của một clip để định tuyến là nhiễu ở mức clip; N gom nhiều cửa sổ nên đỡ hơn.

Luật dừng NMI(cụm, gloss) ≤ 0,2 của v1.0 **không dùng được**: NMI gloss của cụm CLS 0,332 so với 0,358 của cụm ngẫu nhiên (ngân hàng 16.384 clip, K = 48, 2.731 gloss; chỉ là sai lệch ước lượng); AMI gloss −0,016 (cụm CLS) và −0,0005 (cụm ngẫu nhiên) đều ≈ 0, nên AMI là thước đo đúng (không thiên lệch theo số cụm). Dùng AMI, chỉ để ghi nhận.

**Transductive, K không oracle** (CORAL theo cụm của tập query và của gallery, + whiten):

| Thiết lập | whiten | K chọn bằng silhouette | trung bình trên K ∈ {4,6,8,12,16} × K_g ∈ {30,40,50,70} [min–max] |
|---|---:|---:|---:|
| A. val, cụm trong `grp` (tập đầy đủ) | 29,44 | 29,45 (K_q = 24, K_g = 70) | **33,41** [29,26–36,24] |
| C. tập con, **cụm CLS** | 24,94 | 29,44 (K_q = 2, K_g = 50) | **31,56** [30,24–32,59] (CORAL signer thật 31,72) |
| B. 5 lần CV signer giữ lại, cụm trong `grp` | 17,65 | 25,35 | 24,69 |

Silhouette chọn K tệ ở thiết lập A (29,45, không hơn whiten) nhưng trung bình trên K cho +4,0; cụm CLS bền hơn nhiều theo K (khoảng 2,4 điểm). Kết quả v1.0 "34,6" dùng K_q = 6 trùng số signer val, nên là **cận trên lạc quan**.

**Quy nạp (N).** Cụm CLS trên gallery (K cụm); mỗi cụm có biến đổi CORAL học từ gallery; mỗi **query chỉ dùng CLS của chính nó** để chọn cụm gần nhất, rồi whiten fit trên gallery đã biến đổi. Không dùng thống kê batch query, không nhãn.

| K | định tuyến theo CLS (quy nạp) | định tuyến ngẫu nhiên | transductive (K_q = 8–12) |
|---|---:|---:|---:|
| 8 | 27,85 | 21,91 | 30,57 |
| 16 | **28,61** | 22,19 | 31,17 |
| 35 | 28,32 | 20,75 | 32,30 |
| 48 | 27,99 | 20,75 | 31,74 |
| 70 | 26,73 | 21,10 | 31,72 |

(tập con: raw 18,28; whiten 24,94). Định tuyến quy nạp cho +1,8…+3,7 so với whiten (K = 35 mặc định: +3,4; trung bình K ∈ {8,16,35,48,70}: +3,0; cao nhất là giá trị chọn sau khi xem kết quả) và +8,5…+10,3 so với raw; đối chứng định tuyến ngẫu nhiên tụt 2,7–4,2 điểm **dưới whiten**, nên đây là tín hiệu thật (chọn cụm đúng quan trọng). **Đối chứng (`routed_controls.py`):** K = 1 (toàn cục) 24,77 ≈ whiten 24,94 (như dự đoán); cụm CLS qua 3 seed k-means: K = 16: 28,90 ± 0,21; K = 35: 28,45 ± 0,14; K = 48: 28,34 ± 0,27; **định tuyến bằng cụm `grp` thay cho CLS**: K = 16: 27,93; K = 35: 24,20; K = 48: 23,47 (không bền theo K, nên CLS đặc biệt hơn `grp`); **CV signer giữ lại trong nửa train** (5 lần × 6 signer, thống kê chỉ fit trên signer còn lại, encoder đã thấy các clip lúc SSL): N − whiten = +3,73 / +1,36 / +0,50 / +1,74 / +4,30 (trung bình **+2,33**, cả 5 lần dương). Reviewer R2 độc lập tái tạo +3,4…+3,7 qua seed, 5/5 signer val cải thiện, SE ghép cặp 0,55. Chưa có: tập đầy đủ, nhiều seed của chính mô hình, kNN-local whitening làm baseline quy nạp không dùng CLS. Đây là bằng chứng trực tiếp cho lớp G, với phạm vi một checkpoint.

### 2.6 Căn chỉnh học được: head tuyến tính (`coral_head.py`, `coral_head2.py`, `coral_head_mb.py`)

Ma trận W (768 → 512) khởi tạo ở whitening, huấn luyện trên embedding train bằng `‖Cov(z) − I‖² + Σ_c w_c(a_μ‖μ_c‖² + a_Σ‖Cov_c − Cov‖²)`; một W duy nhất áp cho mọi clip khi test.

| Nhóm | Thống kê **toàn batch** | Tập đầy đủ (whiten 29,4) | Tập con (whiten 24,9) |
|---|---|---:|---:|
| signer thật (mean + cov) | toàn train | **35,5** | 28,1 |
| cụm CLS K = 35 / 70 | toàn train | — | **28,5** / 28,2 |
| cụm `grp` K = 70 | toàn train | — | 26,0 |
| signer thật chỉ mean / chỉ cov | toàn train | — | 25,8 / 29,3 |

Nhưng đây **không** phải cơ chế huấn luyện thật. Mô phỏng bộ ước lượng mini-batch của **thiết kế v1.0** (module R: batch 32 clip, thống kê EMA theo cụm trộn α = 0,25 với ước lượng batch, bỏ cụm < 20 mẫu, K = 48 cụm CLS; tập con):

| Cấu hình | R@1 cuối | Ghi chú |
|---|---:|---|
| khởi tạo (whiten) | 24,94 | |
| batch 32 ngẫu nhiên, trọng số đề xuất (a_μ=0,25; a_Σ=1; a_iso=0,25) | **17,78** | L_align 35 ở bước 250 → 0,69 |
| batch 32 cân bằng cụm (4 cụm × 8 clip) | **20,01** | |
| batch 256 ngẫu nhiên | **12,27** | |
| batch 32 ngẫu nhiên, trọng số (1,1,1) | 15,55 | |
| batch 32 cân bằng cụm, (1,1,1) | 21,20 | |
| **cụm xáo trộn** (cân bằng cụm, đề xuất) | **19,41** | gần bằng cụm thật (20,01) |

Tất cả dưới điểm khởi tạo. **Phân tích của reviewer R2:** (i) bộ ước lượng có khuyết tật riêng: thống kê EMA của Z = XW trễ so với W đang đổi, EMA hiệp phương sai khởi tạo từ một batch (hạng ≤ 31 trong 512 chiều), số hạng hiệp phương sai của batch có hạng 1 khi ≈ 1,4 clip mỗi cụm và phạt ‖z − μ‖⁴ (co độ lệch trong-cụm, vốn chứa tín hiệu gloss); (ii) **đối chứng thống kê toàn batch với cùng trọng số, lr 1e-3, 250/1.500 bước, cụm CLS thật: 27,8 / 28,4 (không sụp, trên whiten 24,9); cụm xáo trộn: 23,1 / 22,2 (dưới whiten)**, nên độ dài huấn luyện và việc thiếu số hạng neo không phải nguyên nhân, và cụm xáo trộn cũng hại ngay cả với thống kê đúng, nên so sánh "thật 20,0 so với xáo trộn 19,4" trong mô phỏng mini-batch **không đo được tính đặc thù**. Kết luận đúng: **ước lượng v1.0 bất ổn; thiết kế v1.1 (thống kê từ teacher, hạng thấp, cân bằng cụm) chưa có mô phỏng nào đạt**; đó là việc của cổng G-R. Hệ quả: R chuyển sang "có điều kiện" với cổng G-R (§6.4).

### 2.7 Phần nhiễu không giải thích được bằng biến đơn giản (`style_vars.py`, `nuis_pc.py`, `morph_dose.py`)

- **Nhận diện signer bằng biến diễn giải được:** 30 biến clip-level (hình thái, tư thế, hành vi, camera) cho 35-way **90,4 %**; riêng hình thái 72,6 %, tư thế 48,7 %, hành vi 44,4 %, camera 20,0 % (cờ độ phân giải là rò rỉ giao thức). Embedding chuẩn hoá cùng cách: **94,2 %** (con số 90,7 % của v1.0 dùng đặc trưng chưa chuẩn hoá nên thiên về ngang bằng).
- Trên signer giữ lại, các biến này giải thích **≈ 0 %** phương sai trong-gloss (R² nhiều đầu ra −1,7 %; riêng hành vi +0,16 %), nhưng PC1 R² 45 % và PC3 37 % (chủ yếu `eye_dist`, `rest_height_D`, `brow_width`) còn PC4 −55 %. **Hạn chế công suất:** hồi quy 768 chiều từ 30 biến với 28 signer huấn luyện mỗi fold, một alpha chung cho mọi hướng; kết quả "≈ 0" là giới hạn của kiểm định, không phải bằng chứng mạnh rằng biến phong cách không liên quan.
- **Perturbation hình thái theo liều** trên **1.500 clip cân bằng** (300 clip mỗi signer P12/P21/P26/P39/P5), clean R@1 14,2; hệ số log-đều ±15 / 30 / 50 %. **Đây là bản chạy lại sau khi reviewer R2 phát hiện lỗi:** `morph()` ban đầu để keypoint cổ tay NaN (không quan sát) lan thành NaN toàn bộ bàn tay (đổi 5,4 % cờ quan sát ngay ở hệ số ≈ 1), làm bảng v1.1 đầu sai (dịch "0,29 không đổi theo liều"). Hàm nay NaN-safe, có unit test (`morph_unittest.py`: ở hệ số ≈ 1 không đổi cờ nào, |Δxy| < 2e-4 px).

| Chế độ | cos so với clean (±15 / 30 / 50 %) | chuẩn dịch | ΔR@1 (±15 / 30 / 50 %) |
|---|---|---|---|
| cánh tay | 0,982 / 0,966 / 0,950 | 0,085 / 0,150 / 0,213 | −0,5 / **−1,2 / −1,1** |
| bàn tay | 1,000 / 0,999 / 0,998 | 0,016 / 0,029 / 0,044 | 0,0 / +0,2 / +0,5 |
| cả ba (mặt + tay + cánh tay) | 0,984 / 0,966 / 0,945 | 0,084 / 0,158 / 0,232 | −0,5 / −0,2 / −0,2 |
| **không đồng nhất** (tay trái/phải, từng ngón độc lập) | 0,981 / 0,965 / 0,944 | 0,090 / 0,160 / 0,234 | −0,1 / −0,5 / 0,0 |

  Tham chiếu: cos cùng-gloss-khác-signer 0,589, chuẩn dịch cùng-gloss-khác-signer 0,897. Phần dịch nằm trong không gian con signer (18–23 % năng lượng so với 4,1–4,4 % ngẫu nhiên). Cỡ mặt: encoder gần như bất biến (cos 1,000, dịch 0,001). **Đọc đúng:** encoder nhạy với độ dài cánh tay một cách khiêm tốn (R@1 −1,1…−1,2, khoảng 2 SE ghép cặp 0,6), gần như bất biến với cỡ bàn tay và độ dài ngón; ở ±50 % dịch chỉ bằng ≈ 25 % cự ly cùng-gloss-khác-signer. Perturbation clip-hằng vì vậy **quá yếu để mô phỏng chênh lệch giữa các signer**, nhưng điều đó không chứng minh huấn luyện bất biến với chúng là vô ích. Giới hạn: chỉ họ biến đổi hình thái clip-hằng; biên độ không gian ký hiệu, tốc độ, tư thế nghỉ chưa thử.

### 2.8 Encoder đã học gì? (`emb_variants.py`, `analyze_emb.py`, `ceilings.py`, `retention_scalefree.py`, `lex_probe_emb.py`, `channel_analysis.py`)

Trích embedding teacher / student / **encoder ngẫu nhiên** (cùng kiến trúc, khởi tạo mặc định, một seed) và token đầu vào (không encoder) trên tập con (gallery 20.077, query 5.152; stride 16; cos 0,998 với embedding stride 2 đã cache).

| Biểu diễn | R@1 raw | whiten | LDA-96 | giải mã signer tuyến tính | láng giềng cùng signer |
|---|---:|---:|---:|---:|---:|
| trung bình token đầu vào (không encoder) | 3,4 | 6,2 | 12,9 | — | 58,3 % |
| encoder **ngẫu nhiên** (`grp`) | 4,2 | 11,0 | 21,2 | **89,1 %** | 71,9 % |
| **teacher** (`grp`) | **18,3** | **24,9** | **40,1** | **94,6 %** | 84,4 % |
| student (`grp`) | 18,4 | 24,9 | — | 94,4 % | 84,4 % |
| teacher, chỉ tay (kênh 0–3) | 20,7 | 25,5 | 41,9 | — | 82,5 % |
| teacher, chỉ CONF (kênh 0, 2) | **21,7** | **26,6** | 39,1 | — | 78,7 % |
| teacher, chỉ mặt (kênh 4–6) | 9,4 | 16,2 | 35,1 | — | 83,5 % |
| teacher CLS | 0,25 | 0,5 | 0,8 | — | 96,6 % |

Đọc [D]:
- **Training làm tăng khả năng giải mã signer** (89,1 → 94,6 %; láng giềng 71,9 → 84,4 %) và không giảm thông tin signer. η² có trọng số phương sai giảm (0,27 → 0,14) vì participation ratio của encoder ngẫu nhiên là 7,5 so với 83,5; trong toạ độ whiten η² 0,055 → 0,031 cũng không có nghĩa vì whiten khuếch đại chiều nhiễu của encoder ngẫu nhiên. Giả thuyết "JEPA ưu tiên đặc trưng dự đoán được (Littwin, Sobal [V])" **không bị bác bỏ**; nhưng phần lớn thông tin signer đã có sẵn ở input (token trung bình: 58,3 % láng giềng cùng signer).
- **Học thật sự diễn ra:** LDA 12,9 (input) → 21,2 (ngẫu nhiên) → 40,1 (đã train).
- **Mặt mang signer, ít mang gloss:** F_HD η²_signer 0,37, η²_gloss 0,14; bỏ mặt khỏi readout nâng raw R@1 +2,4 (chỉ tay) đến +3,4 (chỉ CONF) nhưng chỉ +0,6…+1,7 sau whiten; LDA không giảm. Chọn trong 11 tổ hợp kênh trên val nên là **chọn trên val**. Bỏ CLS khỏi mọi readout.
- **Tầng, student/teacher, thứ tự:** LDA-96 theo tầng 1–6 và tầng cuối 37,3–40,9 (tầng 2: 40,9; cuối: 40,1); student ≈ teacher; readout có thứ tự 2 / 4 đoạn **kém hơn** `grp` (whiten 18,9 / 15,9 so với 24,9; LDA-256 29,9 / 30,6 so với 36,1).

**Probe ASL-LEX 2.0 (gloss-rời, signer-rời, balanced accuracy; cùng giao thức D-LEX v3 của proposal16: cùng phân hoạch 819 gloss giữ lại, PCA 256, LR C = 0,05; query là nửa val: 1.527 thay vì 3.086 clip; bootstrap theo gloss 300 lần):**

| Mô tả | SignType | Movement | Repeated | MajorLoc | Contact | FlexChange | WristTwist | SelFingers | Flexion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PLctx teacher** `grp_fin` | **76,9** | **35,7** | **67,3** | **68,2** | **71,8** | **62,8** | **64,0** | **68,1** | **47,6** |
| encoder ngẫu nhiên | 63,8 | 27,6 | 55,0 | 58,4 | 67,1 | 55,2 | 55,2 | 55,5 | 42,1 |
| token input trung bình | 58,4 | 25,0 | 53,7 | 54,3 | 66,4 | 54,5 | 51,9 | 50,8 | 39,9 |
| descriptor thô tốt nhất (D-LEX v3; tốt nhất trong 9 descriptor, nên baseline hơi lạc quan) | 56,9 | 30,9 | 56,7 | 51,1 | 66,0 | 56,9 | 54,1 | 48,9 | 35,4 |
| hơn input / hơn ngẫu nhiên | +18,5/+13,1 | +10,7/+8,1 | +13,6/+12,3 | +13,9/+9,8 | +5,4/+4,7 | +8,3/+7,6 | +12,1/+8,8 | +17,3/+12,6 | +7,7/+5,5 |

CI 95 % của teacher rộng (ví dụ Movement 35,7 [30,4–41,7] chồng với encoder ngẫu nhiên 27,6 [23,1–32,8]; so sánh chưa ghép đôi): với tham số chuyển động coi là **gợi ý**, không phải kết luận. Các tham số handshape, vị trí, loại ký hiệu có khoảng cách rõ.

### 2.9 Lỗi retrieval theo tham số ASL-LEX (`error_attrib.py`)

Lift = tỉ lệ khớp tham số giữa gloss đúng và gloss được trả về (trong query sai) chia tỉ lệ khớp của hai gloss ngẫu nhiên. Vì lift tối đa bị chặn bởi tỉ lệ nền, dưới đây thêm cột **phần dư địa đã đạt** (khớp − nền) / (1 − nền) trên hình học raw:

| Tham số | lift raw | lift whiten | lift LDA | phần dư địa đạt (raw) |
|---|---:|---:|---:|---:|
| Handshape (đầy đủ) | 4,95 | 5,20 | 7,31 | — |
| SignType | 2,46 | 2,49 | 2,66 | 0,642 |
| MajorLoc | 2,16 | 2,20 | 2,52 | 0,443 |
| SelFingers | 1,98 | 1,95 | 2,32 | — |
| Flexion | 1,65 | 1,69 | 2,03 | — |
| Contact | 1,24 | 1,26 | 1,36 | 0,270 |
| FlexChange | 1,12 | 1,13 | 1,19 | 0,167 |
| Repeated | 1,13 | 1,13 | 1,17 | 0,128 |
| WristTwist | 1,04 | 1,07 | 1,10 | 0,106 |
| Movement | 1,18 | 1,26 | 1,40 | 0,073 |

(cột cuối lấy từ phép tính của reviewer R1 trên cùng dữ liệu; thứ tự khớp với lift.) Lỗi bám handshape, vị trí, loại ký hiệu; ít bám các tham số chuyển động. Ghép với probe (§2.8): thông tin chuyển động có trong embedding nhưng chiếm phần nhỏ của hình học cosine. Khoảng cách Hamming trung bình (9 tham số) của lỗi: 3,71 (raw), 3,13 (LDA) so với 5,26 ngẫu nhiên.

### 2.10 Positive láng giềng xuyên signer (`nn_precision_fullpool.py`, `nn_head.py`, `nn_head_cls.py`)

**Precision trên pool đầy đủ** (40.154 clip; láng giềng top-1 thuộc signer khác; "tương hỗ" = láng giềng của láng giềng là chính nó). Bảng v1.0 dùng pool 12k và đã sai (precision thấp hơn thật):

| Hình học | top-1 | tỉ lệ tương hỗ | precision tương hỗ |
|---|---:|---:|---:|
| raw | 14,3 % | 19,0 % | 25,9 % |
| whiten | 22,9 % | 25,2 % | 33,4 % |
| CMN + whiten | 24,1 % | 26,2 % | 34,8 % |
| LEACE + whiten | 24,6 % | 26,2 % | 35,4 % |
| CORAL signer + whiten | 32,4 % | 28,9 % | **44,8 %** |
| LDA-96 (fit trên chính train, lạc quan) | 52,8 % | 25,3 % | 71,1 % |

Trên pool tập con (20.077 clip, cụm CLS-48 làm tiêu chí loại trừ, không nhãn): raw 8,7 / 14,6 %; whiten 14,7 / 21,8 %; **CORAL cụm CLS + whiten 26,4 / 36,6 %** (loại trừ signer thật: 28,2 / 41,3 %).

**Head InfoNCE trên đầu vào whiten (quy nạp), positive mined ở hình học khác nhau:**

| Positive mined từ | precision | val R@1 (whiten tham chiếu) |
|---|---:|---:|
| không hạn chế (hình học whiten) — full | 6,8 % | 13,3 (29,4) |
| loại trừ signer thật, hình học whiten — full | 22,9 % | 25,2 (29,4) |
| loại trừ signer thật, hình học whiten, tương hỗ — full | 33,4 % | 26,1 (29,4) |
| **loại trừ signer thật, hình học CORAL signer — full** | 32,4 % | **32,0** (29,4) |
| **idem, tương hỗ — full** | 44,8 % | **32,4** (29,4) |
| loại trừ cụm CLS, hình học whiten — tập con | 14,7 % | 16,8 (24,9) |
| **loại trừ cụm CLS, hình học CORAL-CLS — tập con (không nhãn)** | 26,4 % | **25,7** (24,9) |
| idem, tương hỗ — tập con | 36,6 % | 24,9 (24,9) |
| positive cùng gloss (oracle) — full / tập con | — | 38,8 / 34,0 |

Positive thô làm hại; positive từ hình học đã căn chỉnh giúp nhẹ khi dùng signer thật (+2,5…+3,0 trên pool đầy đủ) và gần như không giúp khi không nhãn trên tập con (+0,8, trong nhiễu). Ngưỡng "precision tương hỗ ≥ 40 %" của v1.0 là **[I]** (gán nhãn nhầm [V] ở v1.0) và không phải đại lượng quyết định; cổng G-NN nay là một phép thử vận hành (§6.4). Đây chỉ là head MLP trên đặc trưng đóng băng, **không** chứng minh gì về giai đoạn NN-tuning ở mức encoder (STARS/MAE-CT).

### 2.11 Độ tin cậy, chọn lọc trên val, và các phép "không nhãn"

- **Val chỉ có 5 cụm signer (Kish ≈ 3,8).** Bootstrap 1.000 lần (lấy lại signer rồi gloss) cho CI gần như trùng với bản nhân trọng số (reviewer kiểm), nhưng với 5 cụm là phản bảo thủ; đã thêm CI signer-cấp (t, df = 4).
- **Lập lại trên signer khác (`ladder_cv.py`, 5 lần tách 6 signer train giữ lại, 29 gallery; các phép fit đều lại trên gallery):**

| Bậc | R@1 (trung bình ± SD) | Δ so với raw (trung bình; khoảng) |
|---|---:|---|
| raw | 16,2 ± 2,9 | — |
| whiten | 17,7 ± 3,5 | +1,41 (−0,95…+3,07) |
| CMN + whiten | 21,7 ± 3,3 | +5,43 |
| LEACE + whiten | 19,1 ± 3,6 | +2,9 |
| CORAL theo cụm không nhãn, K_q = 6 (= số signer, oracle) + whiten (`ladder_cv.py`) | 23,9 ± 3,4 | +7,61 |
| CORAL theo cụm không nhãn, K chọn bằng silhouette (`kfree_cluster_corral.py`) | 25,35 | +9,1 (trung bình trên K: 24,69, +8,4) |
| CORAL signer thật + whiten | 25,5 ± 2,0 | +9,23 |
| LDA-96 | 42,0 ± 4,3 | +25,79 |

  Các bậc theo signer và trần LDA lặp lại. Whitening riêng chỉ +1,4 so với +6,7 trên val. **Phép thử này trộn hai yếu tố:** query là signer encoder đã thấy lúc SSL (không nhãn), khác với signer val chưa thấy; nên **không phán quyết được** rằng whitening mong manh. Cách đúng là huấn luyện một mô hình SSL bỏ ra vài signer (GPU), hoặc coi lợi ích whitening là chưa chắc.
- **Các phép được gọi là "không nhãn" và nơi tín hiệu val/nhãn len vào** (theo reviewer, 8 chỗ):
  1. K của CORAL theo cụm: K_q = 6 trùng số signer val; đã thay bằng trung bình trên K và silhouette (§2.5).
  2. K = 35 / 48 của cụm CLS chọn theo số signer train và AMI có nhãn; kết quả bền theo K = 8–48 (§2.5).
  3. AMI = 0,71 của CLS đo trên clip encoder đã thấy; AMI val 0,44–0,49 (signer chưa thấy) mới là con số tổng quát.
  4. Whitening (k = 512, lũy thừa 1) chọn bằng R@1 trên nhóm signer giữ lại **có nhãn** trong train; nằm ở biên lưới.
  5. Trừ trung bình láng giềng k = 200 chọn trên val (12 cấu hình); không khác whiten theo CI.
  6. Readout chỉ tay chọn trong 11 tổ hợp kênh trên val.
  7. LDA là giám sát gloss (trần).
  8. Các arm `m` (signer id) chỉ là chẩn đoán; bản tiết kiệm của v1.0 đã bỏ chúng nên nay được giữ lại (§6.1).
- **Giới hạn khác:** 17 trong 20.077 vector `grp_fin` stride-16 của train bằng 0 (gallery stride-2 chính thức có 39); η²_gloss thô chứa mức null (0,136 ở N = 20k) nên chỉ so sánh chênh lệch giữa các hàng; LDA ở §2.1 dùng k = 96, ở `analyze_emb.py` dùng k = 256; tất cả chọn lọc (K, k, tổ hợp kênh, cấu hình trừ trung bình) làm lạc quan khi lặp lại trên val; **chưa đo:** `mw` (SignRep +12,5 [V]), chamfer, SignRep với cùng stack; val không đại diện cho test (22,7 so với 16,7); mọi dự báo về SA-JEPA là [I].

---

## 3. Bản đồ chiến lược JEPA: literature gặp chẩn đoán

GO = vào chương trình; CONDITIONAL = chỉ chạy khi cổng đạt; NO-GO = bị bằng chứng bác bỏ hoặc chưa đủ điều kiện. Số trong cột literature đã đối chiếu HTML thô [V] (chi tiết `literature/Q1–Q3`).

### 3.1 Target (cái được dự đoán)

| ID | Chiến lược | Literature | Nội bộ [D] | Phán quyết |
|---|---|---|---|---|
| **T** | **Center target theo cụm CLS cố định** (CMN trong không gian target) | S-JEPA centering sẵn có trong PLctx; ContentVec: bỏ teacher đã tách speaker, ABX across 6,32 → 6,92 [V] | CMN theo signer +2,2 so với whiten trên val (+4,0 trên CV) | **GO** (arm T1f) nhưng kỳ vọng nhỏ |
| **D1** | Target **multi-depth nối** (khối 2, 4, 6), z-score từng tầng | Bootleg: nối khối hơn trung bình kiểu data2vec (+2 ở Bảng 3, nhưng "mỗi khối thứ 4" −1: gần với {2,4,6} hơn là {1,4,8,12}); target chỉ tầng cuối thoái hoá 61,0 → 43,8 từ 100 → 1.200 epoch [V]; EEG-JEPA +3,66; V-JEPA 2.1 SSv2 62,5 → 72,1 [V] | LDA bão hoà từ khối 2; PLctx dùng *trung bình* 4 tầng cuối | **GO** (D1), rẻ, kỳ vọng nhỏ |
| **D2** | Target **multi-span gộp** (4/8/16 bin + toàn clip) | GFP: truy hồi không train NTU-60 62,0 → 70,9 [V] | tham số chuyển động yếu trong hình học | **GO** |
| **D3** | Target **hiệu thời gian** (khoảng ≥ 3 bin) | MAMP +10,1 linear; MASA k=3 49,06 so với k=1 47,64 [V]; MotionJEPA chống triệt tiêu đặc trưng nhanh [V] | idem | **GO**, cảnh báo jitter 2D |
| D4 | Dự đoán **khối tương lai** phụ | OpenHands DPC 91,2 → 94,7; Human-JEPA: block-mask hại re-ID 17 mAP [V]; V-JEPA từ đầu thích hai chiều [V] | — | CONDITIONAL |
| T-dis | Target rời rạc (HuBERT-style) | PNMI 0,28 → 0,64 → 0,70 [V]; SHuBERT chỉ tham chiếu | không đo purity | CONDITIONAL thấp nhất |
| T-iso | **Đẳng hướng** SIGReg/VICReg làm mục tiêu chính | LeJEPA cần nhiều view toàn cục, batch ≥ 256, regulariser trên chính embedding được căn chỉnh [V]; pilot nội bộ −13,8 [R] (SIGReg ở head phụ, đọc `pj/train.py`) | whitening +6,7 val nhưng +1,4 CV (chưa phán quyết; chỉ cho biết lợi ích post-hoc có thể thay thế mục tiêu isotropy) | **NO-GO** làm mục tiêu chính |

### 3.2 Predictor và teacher

| ID | Chiến lược | Literature | Nội bộ [D] | Phán quyết |
|---|---|---|---|---|
| C-pred | Predictor có điều kiện theo mã signer (ContentVec) | bỏ conditioning: ABX within 5,13 → 5,76, across 6,32 → 7,08 [V]; seq-JEPA: conditioning trên nhân tố không liên quan lớp 87,4 → 79,3 [V] | CLS **đã** đi vào predictor mà token vẫn mang signer | CONDITIONAL (arm C1: mã cụm làm điều kiện AdaLN; sau T) |
| P-IWM | Conditioning theo tham số augmentation | IWM: bất biến thắng linear (74,5 so với 67,5) [V] | retrieval cosine cần bất biến | **NO-GO** |
| E-ema | EMA hay online | CF-JEPA online +3,9 phân loại [V] | student ≈ teacher (18,44 so với 18,28 raw) | không phải đòn bẩy |
| E-anc | Teacher neo cho train dài | Human-JEPA: tiếp tục ngây thơ sụp đổ, neo ngăn [V] | tiếp tục 160 → ~164–180 epoch phẳng [R] | CONDITIONAL (Phase 4) |
| E-canon | Teacher nhận input chuẩn hoá hình thái giải tích | ContentVec: bộ chuyển đổi yếu tệ hơn không có (7,78 so với 6,92) [V]; AUTSL: ẩn danh thô −16 điểm [V] | perturbation ±15…50 % (đã sửa lỗi NaN): dịch tối đa 0,23 so với 0,90 giữa signer; R@1 đổi ≤ 1,2 | **Ưu tiên thấp, không phải bị bác bỏ** (họ clip-hằng quá yếu; biên độ, tốc độ, tư thế nghỉ chưa thử) |

### 3.3 Regularizer, positive chéo mẫu, readout/hình học, dữ liệu

| ID | Chiến lược | Literature | Nội bộ [D] | Phán quyết |
|---|---|---|---|---|
| **G/N** | **Chuẩn hoá theo cụm định tuyến bằng CLS** (inference-time, thống kê từ gallery) | AdaBN (1603.04779), Deep CORAL (1607.01719) [id đã kiểm]; nén không gian con speaker −5…15 % ABX trên speaker chưa thấy [V]; hubness/CSLS [V] | quy nạp 28,6 so với whiten 24,9 (tập con); ngẫu nhiên 20,8–22,2 | **GO** (lớp G) |
| R | Căn chỉnh phân phối theo nhóm **trong** encoder | DG: CORAL, DICA, DANN (adversarial thất bại ở láng giềng [V]) | head toàn batch +3,6…+6,1; mô phỏng mini-batch **thất bại** | **CONDITIONAL**, cổng G-R |
| X1 | NN-contrastive tuning (STARS/MAE-CT/NNCLR) | STARS kNN 63,1 → 79,9; bỏ NN lookup còn 40,5; positive chung nhiễu thất bại [V] | positive raw/whiten hại; hình học căn chỉnh +0,8 (không nhãn) đến +3,0 (signer thật) | CONDITIONAL thấp, cổng G-NN vận hành |
| X2 | Contrast trong clip (ContentVec Eq. 2; WTN) | WTN theo kịp supervised dưới nhiễu cố định [V] | không đo | CONDITIONAL thấp |
| X3 | Adversarial signer (GRL) | SRM −1,7 WER; EEG 88,6 → 73,0 vẫn rất cao; attacker 56,0 vs adversary 49,0 [V] | — | NO-GO làm method |
| Z1 | Readout chỉ tay, bỏ CLS | — | raw +2,4…+3,4 (chọn trên val, +0,6…+1,7 sau whiten) | **GO**, nêu rõ chọn trên val |
| Z2 | Whitening không nhãn | all-but-the-top, hubness [V] | +6,7 val, +1,4 CV | báo cáo, độ bền chưa rõ |
| Z3 | CORAL theo cụm CLS transductive | TTA | 31,6 [30,2–32,6] tập con | báo cáo riêng |
| Z4 | Readout có thứ tự / chamfer / `mw` | SignRep ordered +5,9, chamfer +9,4 trên đặc trưng SignRep [R]; activity weighting +12,48 [V] | ordered 2/4 đoạn **kém hơn**; chamfer, `mw` chưa đo | ordered NO-GO; chamfer, `mw` đo ở Phase 1 |
| Y1 | **Dữ liệu đa signer** (MS-ASL 222, WLASL 119 signer) | scale sign: không có đường cong frozen; BEST/STC lõm +4…+14 fine-tune; SHuBERT 10× dữ liệu +2,3 BLEURT [V] | gallery lúc truy vấn +3,4 mỗi lần gấp đôi signer (khác thao tác với tiền huấn luyện) | **GO** (phép đo độ dốc) |
| Y2 | Corpus lớn (SignVerse-2M 3.207 h DWPose) | OpenHands 1.129 h pose không nhãn [V] | — | CONDITIONAL theo Y1 |

**Nhận xét.** Hai điều dịch chuyển so với v1.0: (i) phần có bằng chứng mạnh nhất nằm ở **lớp hình học không train** (G/N), không phải ở mục tiêu huấn luyện; (ii) mô-đun huấn luyện R từ "chính" thành "có điều kiện". Các chiến lược kinh điển của JEPA (T, D1–D3) được giữ nhưng với kỳ vọng thấp và gắn với những chế độ lỗi cụ thể (độ lệch target theo signer; thông tin chuyển động yếu).

---

## 4. Giả thuyết đăng ký trước

Metric chính **VAL-DICT raw**: query val (10.304), gallery train (40.154), cosine, class-max, tie bi quan. Mọi arm báo ba hàng: raw; + lớp G quy nạp (hands, whiten, N); + TTA transductive (khai báo). **Hiệu ứng:** τ = max(1,5; 2,5 × SD_seed(A0)) (SD từ 3 seed A0 có sai số tương đối ≈ 50 %, nên τ chỉ là hướng dẫn). **CI:** bootstrap signer→gloss **và** CI t cấp signer (df = 4); với 5 cụm signer hiệu dụng (Kish 3,8) KEEP đòi cả hai CI loại trừ 0 và cả 5 signer không tệ đi (xác suất 4/5 cải thiện dưới null là 0,19, nên "≥ 4/5" không đủ). **Tuyên bố "hơn stack"** chỉ khi Δ sau lớp G ≥ τ. Mọi lựa chọn (K, tổ hợp kênh, cấu hình) đã tinh chỉnh trên val; kết quả val là lạc quan, ghi rõ.

| ID | Giả thuyết | Dự đoán định lượng [I] | Arm | Bác bỏ / kết luận | Control |
|---|---|---|---|---|---|
| **H1** | Center target theo cụm CLS cố định làm encoder bớt mang độ lệch signer | CONFIRM: Δ raw ≥ τ, láng giềng cùng signer −5 điểm (84 % → ≤ 79 %), LDA-96 không giảm quá 2. REFUTE: Δ raw < 0,5 và cận trên CI < τ. Khoảng giữa: INCONCLUSIVE | T1f vs A0 | như nêu | **T1-shuf** (nhãn cụm xáo trộn) ≈ A0 |
| **H2** (có điều kiện G-R) | Căn chỉnh hiệp phương sai theo cụm trong encoder tăng retrieval raw | Δ raw ≥ max(τ; 3) | R1f vs A0, vs R0 | R1f ≤ R0 hoặc ≤ R1-shuf | **R0** (chỉ đẳng hướng), **R1-shuf** |
| **H3** | Cụm CLS không nhãn không kém signer thật | R@1(CLS) ≥ R@1(meta) − 1,5 (không-kém, một phía; 1,5 = sàn của τ) | T1f vs T1m | R@1(CLS) < R@1(meta) − τ | — |
| **H4** | Gói động học làm tăng thông tin chuyển động | trung bình 4 tham số {Movement, Repeated, FlexChange, WristTwist} của probe +3 (CI ghép cặp signer×gloss loại trừ 0); raw ≥ A0 + 1,5 | DYN vs A0 | Δ probe ≤ 0 | A0 + cùng số tham số đầu thừa |
| **H5** | Target multi-depth không làm xấu đi khi train dài | VAL-DICT của D1 ở 160 epoch ≥ A0 ở 160 epoch − 0,5 (A0 đã phẳng nên đây là kiểm không-hại, không phải kiểm lợi ích) | D1 Phase 4 | thoái hoá ≥ 1 điểm | A0 train dài (đã có) |
| **H6** | Mặt mang signer, ít mang gloss | loss + readout chỉ tay: raw +1,5, LDA-96 không giảm (đã thấy +2,4 trên readout, chọn trên val) | H-only | không tăng | — |
| **H7** | Thêm signer vào corpus tiền huấn luyện tăng retrieval | quy tắc độ dốc **thiết kế** (không phải dự đoán): ≥ +1,5 R@1 mỗi lần gấp đôi → mua dữ liệu; < +0,5 → dừng; 25/50/100 % signer của ASLC rồi + WLASL + MS-ASL | Phase 4 | độ dốc < 0,5 | kiểm soát số bước |
| **H8** | T và DYN cộng được | tương tác trong ±1,5 (khoảng nhiễu) | T+DYN vs tổng | \|tương tác\| > 1,5 | — |
| **H9** (thăm dò) | Sau huấn luyện, lớp G còn cần | Δ của N trên arm ≥ 1,5 (so với ≈ 3,7 trên A0 tập con) | N trên mọi arm | Δ < 1,5: căn chỉnh đã được nội hoá | — |
| **H10** | Lớp G cộng được trên cấu hình cuối | N + whiten + hands thêm ≥ +2 inductive trên cấu hình cuối | Phase 4 | thêm < 2 | — |

---

## 5. Method: SA-JEPA v1.2

### 5.1 Nguyên tắc

- **Fork từ `method/phonojepa`** (token 7 kênh × 32 bin, d = 256, 6 khối, predictor d = 128 × 4, mask PLCM-lite, EMA 0,996 → 0,9999, centered-CE τ_t 0,04 → 0,06, τ_s 0,1, AdamW 3e-4). Mỗi arm là một diff nhỏ so với A0, cùng optimizer, lịch, evaluator.
- **Chỉ thêm thứ chẩn đoán chỉ ra.** Không thêm tham số cho nhiễu mà encoder đã bất biến (cỡ mặt, tỉ lệ hình thái).
- **Method chính không dùng nhãn gloss và không dùng signer id.** Signer id chỉ ở arm `m` (diagnostic).
- **Tách hai loại đóng góp:** lớp G (không train, inference-time, thống kê từ gallery) và lớp J (thay đổi huấn luyện). Mọi báo cáo phân biệt rõ.

### 5.2 Lớp G: readout và chuẩn hoá theo cụm định tuyến (N)

Index (một lần cho mỗi gallery, trong ASL Citizen gallery là train; trong WLASL là train WLASL):
1. Với mỗi clip gallery, teacher cho `grp` (hoặc `hands`) và CLS (trung bình trên cửa sổ).
2. K-means cosine trên CLS đã L2-chuẩn hoá, K = 35 mặc định (K = 8–48 cho kết quả phẳng trên checkpoint đã đo).
3. Mỗi cụm: trung bình μ_k và hiệp phương sai co rút (λ = 0,2); biến đổi `T_k = Σ_k^{-1/2} Σ^{1/2}` (về hiệp phương sai toàn cục); cụm < 200 clip chỉ trừ trung bình.
4. Biến đổi gallery theo cụm của chính nó; fit PCA-whiten (k = 512) trên gallery đã biến đổi.

Truy vấn: tính CLS và `grp` của query; chọn cụm gần nhất theo cosine với tâm; áp `T_k`, rồi whiten; retrieval cosine như cũ. **Không dùng thống kê batch query, không nhãn.** Đã đo trên tập con: 28,6 so với whiten 24,9 và raw 18,3; định tuyến ngẫu nhiên 20,8–22,2 (kiểm soát). Chưa đo: tập đầy đủ, nhiều checkpoint, signer test (test có 11 signer).

Báo cáo song song: (a) raw, (b) `hands` + whiten, (c) + N (quy nạp), (d) TTA transductive (CORAL theo cụm của tập query), (e) trần LDA-96 để theo dõi khoảng cách "thông tin so với hình học".

### 5.3 Mã signer từ CLS (C)

- **C-fixed (mặc định cho screening):** một lượt teacher của A0-160 trên mọi clip train cho CLS; k-means K = 48; ánh xạ clip → cụm cố định suốt huấn luyện. Chi phí một lượt trích embedding (≈ 20 phút CPU cho nửa train còn thiếu CLS). Loại bỏ vòng phản hồi và giải thích sạch hơn; đổi lại phụ thuộc một checkpoint không nhãn sẵn có (A0-160).
- **C-online (arm riêng T1o):** CLS của teacher đang train, ngân hàng 16.384 clip, làm mới mỗi epoch với tâm khởi tạo từ lần trước; id theo clip (trung bình CLS teacher của hai view). Rủi ro "tan rã cụm": khi T/R bỏ tín hiệu signer khỏi token, CLS có thể thôi mã hoá signer, cụm trở nên ngẫu nhiên. Mã CLS đã thấy từ 4k bước (AMI 0,45) và ổn định giữa cửa sổ (ARI 0,92–0,97), nhưng **chưa đo dưới T/R**. Cảnh báo: AMI(cụm, signer) giảm quá 0,15 so với đỉnh (đo có nhãn, chỉ ghi, không đổi hành vi).
- **Giám sát (log-only, dùng nhãn chỉ để đo):** AMI(cụm, signer) và AMI(cụm, gloss) kèm baseline hoán vị; **không** dùng làm luật dừng hay tiêu chí KEEP (NMI(gloss) của cụm ngẫu nhiên đã cao hơn của cụm CLS).

### 5.4 Mô-đun T: center target theo cụm

PLctx: `p_t = softmax((t − c) / τ_t)`, `c` một vector D chiều (EMA β = 0,9). SA-JEPA: bảng `C[K, D]`; clip thuộc cụm k dùng `c_eff,k = λ_k · C[k] + (1 − λ_k) · c_global`, `λ_k = n_k / (n_k + 64)` (co về toàn cục khi cụm thưa); EMA β = 0,9 chỉ cho cụm có mặt trong batch. K = 1 trở về PLctx bit-for-bit. Arm T1f-cb thêm **sampler cân bằng cụm** (4 cụm × 8 clip mỗi batch) để mỗi bước có ≥ 8 clip cho mỗi center (với sampler ngẫu nhiên, 32 clip trên 48 cụm cho khoảng 1,4 clip mỗi cụm có mặt; reviewer mô phỏng).
Căn cứ và giới hạn: bằng chứng là CMN +2,2 (val) / +4,0 (CV) so với whiten ở mức embedding gộp với signer oracle; chưa có chẩn đoán nào kiểm CMN theo cụm CLS ở mức token-target. Hiệu ứng kỳ vọng nhỏ.

### 5.5 Mô-đun D (DYN): ép động học

| ID | Nội dung | Chi tiết |
|---|---|---|
| D1 | multi-depth | target teacher = nối đầu ra khối {2, 4, 6}, LayerNorm riêng từng tầng; predictor ra 3 đầu (mỗi đầu CE); loss trung bình. Thay cho trung bình 4 tầng cuối hiện dùng. |
| D2 | multi-span | mỗi khối mask (4–8 bin) thêm một query dự đoán **trung bình teacher** trên khối; thêm query trung bình toàn clip theo kênh; spans 4/8/16 bin + toàn cục. |
| D3 | hiệu thời gian | dự đoán `LN(h_{t+k}) − LN(h_t)` của teacher từ ngữ cảnh student tại (t, t+k), k = 3 bin (0,24 s ở 25 fps), chỉ bin hợp lệ cả hai đầu, trọng số 0,5. |
| D4 | khối tương lai | xác suất 0,3 mỗi view thay mask gần bằng "che 25 % bin cuối". Chỉ khi D1–D3 dương. |

Mọi target chỉ phụ thuộc teacher (thấy toàn cảnh); không rò rỉ. Test rò rỉ 1e6 của PLctx áp lại cho các đầu mới. Căn cứ: literature (Bootleg, GFP, MASA); chẩn đoán nội bộ chỉ cho thấy thông tin chuyển động có nhưng yếu trong hình học (§2.8–2.9), không chứng minh các target này sẽ tăng nó.

### 5.6 Mô-đun R (có điều kiện G-R): căn chỉnh trong encoder, thiết kế lại sau review

Thiết kế v1.0 bị mô phỏng mini-batch bác bỏ (§2.6). Phiên bản v1.1, **chỉ chạy GPU sau khi vượt cổng G-R**:
- **View:** view xa (khoảng 88 % token nhìn thấy) để gần với embedding retrieval; mean-pool group trên token nhìn thấy; không dùng view gần (che 50 %).
- **Thống kê:** từ **teacher** (thay đổi chậm, nhất quán theo thời gian) trên ngân hàng clip đã pool đầy đủ, **không** từ student (EMA thống kê của student bị trôi khi W đổi); cụm có ít hơn 64 mẫu trong ngân hàng bị bỏ qua.
- **Dạng alignment:** trung bình + **đường chéo + hạng thấp (r = 32)** của hiệp phương sai thay vì ma trận 768² đầy đủ (chống ồn khi ≈ 1,4–8 clip mỗi cụm mỗi batch); sampler cân bằng cụm bắt buộc.
- **Cân bằng loss:** bộ chuẩn hoá di chuyển (EMA β = 0,999, không đóng băng sau warm-up) cho L_R, trọng số hiệu dụng ≤ trần để L_R chiếm ≤ 15 % CE.
- **Đẳng hướng:** trên embedding đã chuẩn hoá theo chiều (EMA std), đích là ma trận tương quan = I (thay vì I trong thang tuyệt đối của pooled LayerNorm).
- **Cổng G-R (CPU, trước GPU):** mô phỏng `coral_head_mb.py` với **số hạng neo** (`‖W x − W₀ x‖²`, trọng số chọn trên nhóm signer giữ lại) và dạng alignment ở trên; yêu cầu val R@1 cuối ≥ whiten-init + 2, đối chứng cụm xáo trộn ≤ whiten-init + 0,5, trên 3 seed. Hiện chưa đạt (cấu hình v1.0: 12,3–21,2).

### 5.7 Giai đoạn NN-tuning (có điều kiện G-NN)

Chỉ khi head thử trên đặc trưng đóng băng của chính cấu hình cuối đạt: positive mined trong hình học đã căn chỉnh (CLS-CORAL, loại trừ cụm CLS, không nhãn) cho head val R@1 ≥ stack quy nạp + 2 (G-NN vận hành; hiện +0,8). Nếu đạt, mới thiết kế giai đoạn encoder kiểu STARS (top nửa encoder, layer-decay, khoảng 10 % ngân sách).

### 5.8 Hiện thực và chi phí

- Package `method/sajepa` (fork `phonojepa`); móc: `losses.CenteredCE` (center theo cụm), `train.py` (ngân hàng CLS, cụm cố định, sampler cân bằng cụm), `model.py` (đầu đa tầng, query pooled).
- Unit test: (1) K = 1 ⇒ trùng PLctx từng bit; (2) cụm giống nhau ⇒ L_R = 0; (3) gradient chỉ qua phần batch; (4) id cụm cố định tái lập; (5) target D1/D2/D3 đúng hình dạng và chỉ bin hợp lệ; (6) test rò rỉ 1e6 sau toàn bộ pipeline; (7) **L_R giảm trên embedding A0 thật** (tránh hỏng như v1.0).
- Chi phí: T và D thêm vài query token; ước tính 1,1–1,3× PLctx. Số liệu đo: 40 epoch PLctx = 1,01 h (14.580 s cho 160 epoch / 4), nên 1,3–1,6 GPU-h mỗi arm là thực tế. **Chưa tính** trong ngân sách: trích embedding stride-2 gallery đầy đủ cho mỗi checkpoint (≈ 20–40 phút), trích pose cho MS-ASL/WLASL (Phase 4), CPU cho lớp G.

---

## 6. Arm và chương trình thực nghiệm

### 6.1 Arm (screening 40 epoch, 1 seed trừ A0)

| Arm | Mô tả | Giả thuyết | Vai trò |
|---|---|---|---|
| **A0** | PLctx REF, 40 epoch, **3 seed** | — | anchor, SD seed |
| **B\*** | A0 + lớp G (hands, whiten, N), không train | H9, H10 | đối thủ phải thắng ở hàng "hơn stack" |
| **T1f** | T, cụm CLS cố định (từ A0-160), K = 48 | H1 | chính |
| **T1f-cb** | T1f + sampler cân bằng cụm | H1 | kiểm soát sampler |
| **T1-shuf** | T1f với nhãn cụm xáo trộn | H1 | control |
| **T1m** | T với signer id thật | H3 | cận trên (khai báo), **giữ trong bản tiết kiệm** |
| **T1o** | T, cụm CLS online | H1 | rủi ro tan rã cụm |
| **H-only** | loss + readout chỉ tay | H6 | |
| **D1, D2, D3** | từng mô-đun động học | H4, H5 | |
| **DYN** | D1 + D2 + D3 | H4 | |
| **T+DYN** | T1f + DYN | H8 | |
| (R1f, R0, R1-shuf) | chỉ chạy nếu G-R đạt | H2 | R1f, đẳng hướng riêng, cụm xáo trộn |
| Phụ | K ∈ {24, 96}; D4; C1 | | nếu còn ngân sách |

Số run: bản đầy đủ A0×3 + 11 arm (14 run) + 3 (R-path nếu G-R đạt); bản tiết kiệm A0×3 + T1f, T1-shuf, T1m, H-only, DYN, T+DYN (9 run).

### 6.2 Phase

| Phase | Nội dung | GPU-h |
|---|---|---:|
| 0 | Chẩn đoán CPU (xong, §2) | 0 |
| 1a | **Cổng G-G, lớp G trên tập đầy đủ:** trích CLS cho cả train, đo N trên gallery 40k (stride 2) và `mw`, chamfer; kèm đối chứng K = 1, định tuyến theo cụm `grp`, nhiều seed k-means, CV signer giữ lại; **trích đặc trưng SignRep 40k clip và áp cùng stack là bắt buộc nếu muốn tuyên bố "hơn SignRep"** (nếu không, chỉ được so với PLctx) | 3 (+6–10) |
| 1b | **Cổng G-T, pilot huấn luyện ≤ 1–2 GPU-h** (công thức pilot 4k bước trên ASLC002: A0, T1f, T1-shuf): T có làm đổi giữ lại signer (kNN cùng signer, giải mã) và raw R@1 theo hướng dự đoán không; nếu T không làm gì ở quy mô này thì lớp J không có hỗ trợ và proposal thu hẹp về lớp G + trục dữ liệu | 1–2 |
| 2 | A0×3 và ~11 arm ở §6.1; VAL-DICT raw + 3 hàng + probe ASL-LEX + metric nhiễu ở epoch 10/20/40 | 24–32 |
| 3 | 1–2 cấu hình tốt nhất ×3 seed + A0; tổ hợp 2×2 nếu hai họ cùng KEEP | 12–15 |
| 4 | Trục dữ liệu (ASLC 25/50/100 % signer; + WLASL + MS-ASL, lọc nhiễm chéo theo D1); full160 cấu hình cuối; **test một lần** (ASL Citizen đóng băng + WLASL2000) | 14–18 |
| 5 | Downstream: ISLR (linear, fine-tune, few-shot), WLASL, MS-ASL local, độ bền fps/frame drop (P16 §9.7) | 6–10 |
| **Tổng** | | **≈ 62–82** (≈ 37 bản tiết kiệm) |

### 6.3 Luật quyết định (kế thừa `docs/research` §6.6)

- **KEEP:** Δ raw ≥ τ; CI bootstrap signer→gloss **và** CI t cấp signer loại trừ 0; các signer lớn (≥ 2.000 query: P21, P26, P39) không tệ đi (signer ≈ 600–700 query có SE nhị thức ≈ 1,4 điểm nên chỉ báo cáo, không dùng làm điều kiện); LDA-96 không giảm quá 2; kiểm soát tương ứng (T1-shuf, R0/R1-shuf) thua arm ≥ 1 điểm.
- **KEEP-MECH:** Δ raw trong ±τ nhưng cơ chế cải thiện (láng giềng cùng signer −5; probe chuyển động +3; Δ của N co lại). Giữ để tổ hợp.
- **DROP:** Δ ≤ −1,5 hoặc TOST tương đương ±1,5. (Biên ±1 của v1.0 nhỏ hơn nhiễu của hiệu hai run một seed, nên không dùng.)
- **Đối chiếu bắt buộc:** so với B\* ở hàng "hơn stack"; với T/R báo cả hàng cụm CLS và hàng signer id; Holm trên họ so sánh retrieval {H1, H2, H4, H6}.
- 1 seed ở Phase 2 chỉ cho ứng viên; kết luận ở Phase 3 (3 seed). Kendall τ giữa epoch 40 và epoch cuối khi có điểm trung gian; nếu không có thì ghi giả định.

### 6.4 Cổng và kế hoạch dự phòng

| Cổng | Điều kiện | Nếu đạt | Nếu không |
|---|---|---|---|
| **G-G** (Phase 1a) | N quy nạp ≥ whiten + 2 trên val tập đầy đủ, ≥ K = 1 và định tuyến theo `grp` + 2, dương ở ≥ 4/5 lần CV signer giữ lại | lớp G là một phần của hệ thống cuối | bỏ N; giữ hands + whiten + TTA khai báo |
| **G-T** (Phase 1b) | pilot: T1f khác T1-shuf theo hướng dự đoán (kNN cùng signer giảm, raw R@1 không giảm) | mở Phase 2 | thu hẹp về lớp G + dữ liệu |
| **G-R** (CPU) | mô phỏng mini-batch có neo: ≥ whiten-init + 2, shuffled ≤ + 0,5, 3 seed | chạy R1f/R0/R1-shuf | R bị loại, ghi nhận |
| G-SA (sau T1f, T1-shuf, T1m, T1o) | T1f ≥ A0 + τ và ≥ T1-shuf + 1 | giữ T; Phase 3 | bỏ T; Plan B |
| G-DYN | DYN: probe chuyển động +3 và raw ≥ +1,5 | T+DYN | bỏ DYN; giữ D1 nếu H5 đạt |
| G-NN | head trên đặc trưng cấu hình cuối: positive hình học căn chỉnh không nhãn ≥ stack + 2 | thiết kế NN-tuning encoder | giữ NO-GO |
| G-DATA (H7) | độ dốc ≥ +1,5 / lần gấp đôi signer | mua thêm dữ liệu (Y2) | dừng mở rộng |

**Plan B (nếu mọi họ J thất bại):** hệ thống cuối = A0 + lớp G (khai báo là inference-time, không nhãn); theo đuổi trục dữ liệu; arm cận trên có signer id (adversarial/DANN) chỉ để đo; quay lại `docs/research` O4 (teacher RGB đóng băng) nếu mọi nhánh không nhãn < +τ.

### 6.5 Kỳ vọng thực tế và cách báo cáo

- **Giá trị tham chiếu (val raw):** A0 22,7; lớp G trên tập con: raw 18,3 → whiten 24,9 → N quy nạp 28,6 (+10,3 so với raw). **Dự báo [I]:** lớp G = whiten (+1,4…+6,7, chưa phán quyết) + N (+3,0…+3,4 trên một checkpoint/nửa gallery): val raw khoảng **26–33** (nếu lợi ích whitening chỉ +1,4 thì cận dưới; 33 chỉ khi cả whitening +6,7 và N giữ nguyên trên gallery đầy đủ). Tỉ lệ test/val 0,73 đo trên raw, chưa có bằng chứng áp dụng cho lợi ích hình học, nên test chỉ nêu khoảng **20–24**. Lớp J cộng 0–4 điểm: T có nền CMN +2,2 (val)/+4,0 (CV) ở mức embedding gộp; D1–D3 không có chẩn đoán dự báo; **không đưa dự báo đầu-cuối** cho SA-JEPA vì chưa có bằng chứng huấn luyện.
- **Vẫn thấp xa SignRep raw (49,0 đo local);** LDA 66,0 trên gloss giữ lại dùng nhãn nên không so được; không có chẩn đoán nào ở đây cho thấy khoảng cách đó đóng được chỉ bằng JEPA skeleton không nhãn.
- **Cách báo cáo:** mỗi arm ba hàng (raw; + lớp G quy nạp; + TTA transductive khai báo), kèm LDA-96, láng giềng cùng signer, probe 9 tham số. Test chấm một lần cho cấu hình cuối; WLASL2000 cùng checkpoint.
- **Điều có thể thay đổi kết luận:** G-G không đạt; G-SA không đạt; độ dốc dữ liệu (H7) lớn; SignRep + cùng stack thu hẹp khoảng cách (Phase 1a, bắt buộc cho tuyên bố so sánh).

---

## 7. Rủi ro

| Rủi ro | Mức | Cách giảm |
|---|---|---|
| Lớp G chỉ đúng trên checkpoint/tập con đã đo; signer test chưa thấy; test có 11 signer | cao | G-G ở Phase 1 trên tập đầy đủ; báo cả raw và G; N phụ thuộc cụm CLS của gallery |
| Cụm CLS gom theo gloss, alignment xoá từ vựng | thấp–trung bình | AMI gloss ≈ 0 đã đo; LDA-96 làm bảo vệ; T1-shuf |
| Tan rã cụm CLS dưới T/R (arm online) | trung bình | C-fixed mặc định; AMI log-only; T1o là arm riêng |
| R mất ổn định mini-batch (đã thấy trong mô phỏng) | cao | cổng G-R; thiết kế lại (teacher, hạng thấp, cân bằng cụm) |
| Val có 5 cụm signer, tinh chỉnh nhiều trên val | cao | CI signer-cấp; CV; Phase 3 ba seed; ghi rõ lạc quan |
| Lợi ích whitening chưa rõ bền (CV +1,4 so với val +6,7; CV trộn signer đã thấy/chưa thấy) | trung bình | báo cả hai; không dựa vào whitening để biện minh cho method |
| T hiệu ứng nhỏ | cao | khai báo kỳ vọng thấp; T1m cận trên |
| D3 nhiễu do jitter 2D | trung bình | k ≥ 3 bin, trọng số 0,5, làm mượt phía teacher |
| Xếp hạng 40 epoch không dự báo 160 | trung bình | điểm trung gian, Phase 3 |
| Cạnh tranh GPU với job PhonoPath full160 | thực tế | chỉ chạy khi bạn duyệt và GPU còn dư; một job mỗi lúc |

---

## 8. Không nên làm

1. **Ưu tiên cho retarget hình thái giải tích / teacher chuẩn hoá hình thái: thấp.** Họ clip-hằng đã thử (sau khi sửa lỗi NaN) dịch embedding tối đa 0,23 (±50 %) so với 0,90 giữa signer và R@1 đổi ≤ 1,2; encoder gần bất biến với cỡ bàn tay và mặt (§2.7). Không phải "đã bác bỏ": biên độ không gian ký hiệu, tốc độ, tư thế nghỉ chưa thử.
2. **NN-tuning với positive thô** (raw/whiten): hại (13–26 so với whiten 29,4). Chỉ xem xét lại qua G-NN.
3. **Isotropy/SIGReg làm mục tiêu chính.**
4. **Alignment hoặc readout trên CLS:** CLS là chữ ký signer (R@1 0,25). CLS chỉ dùng làm mã định tuyến.
5. **Readout có thứ tự đơn giản:** kém hơn `grp`. (Chamfer và `mw` chưa đo.)
6. **Conditioning predictor trên nhân tố gây equivariance** (IWM, seq-JEPA).
7. **R với thiết kế v1.0** (EMA thống kê student, batch 32 ngẫu nhiên, ma trận đầy đủ).
8. **Dùng NMI(cụm, gloss) làm luật dừng hoặc KEEP.**
9. **Tuyên bố chắc chắn rằng whitening là mong manh hoặc bền;** và rằng "signer là ánh xạ tuyến tính" (chỉ có thể nói chuẩn hoá bậc hai theo miền loại phần lớn nhiễu).

---

## 9. Nhật ký rà soát

**R1 (độc lập, 2026-10-17, `reviews/R1_independent_review.md`):** 2 Critical, 12 Major, 1 nhóm Minor. Không có rò rỉ test hay nhãn val vào phép fit quy nạp; retrieval, CORAL, head tuyến tính, bảng §2 tái lập. Bảng phản hồi:

| ID | Mức | Vấn đề | Xử lý trong v1.1 |
|---|---|---|---|
| F1 | Critical | "JEPA không khuếch đại signer" là artifact dị hướng của η² có trọng số phương sai | **Rút lại.** Số đo không phụ thuộc thang (giải mã tuyến tính chuẩn hoá 89,1 → 94,6 %; láng giềng 71,9 → 84,4 %; PR 7,5 → 83,5): §2.8, `retention_scalefree.py` |
| F2 | Critical | Bảng precision NN dùng pool 12k; ở pool đầy đủ cổng 40 % đã đạt; ngưỡng gán nhầm [V] | Tính lại pool đầy đủ (§2.10, `nn_precision_fullpool.py`); ngưỡng thành [I]; head với positive hình học căn chỉnh (+3,0 signer thật, +0,8 không nhãn); G-NN thành phép thử vận hành |
| F3 | Major | NMI(cụm, gloss) ≤ 0,2 bắn trên cụm ngẫu nhiên; luật dùng nhãn | AMI log-only (§5.3, `cls_early.py`) |
| F4 | Major | `ladder_cv` trộn signer đã thấy / chưa thấy; "lạc quan" sai chiều | Câu chữ sửa, nêu CV không phán quyết được; dùng delta JSON (§2.11) |
| F5 | Major | "signer là ánh xạ tuyến tính" quá mạnh; gloss-coverage; whitening riêng từng signer 41,5 | Đổi diễn đạt, thêm đối chứng thành phần và whitening theo signer (§2.4, `ladder_extra.py`) |
| F6 | Major | "không nhãn" che K = số signer, chọn trên val, nhãn trong luật dừng (8 chỗ) | K không oracle (silhouette, trung bình trên K; §2.5), bảng 8 chỗ (§2.11) |
| F7 | Major | NO-GO retarget dựa trên kiểm định công suất thấp, perturbation mặt vô hiệu, tập không cân bằng | Liều ±15/30/50 %, tập cân bằng, biến thể không đồng nhất; giải mã chuẩn hoá 94,2 %; ghi giới hạn. **R2 phát hiện lỗi lan NaN trong `morph()`** → sửa, thêm unit test, chạy lại; NO-GO hạ thành "ưu tiên thấp" (§2.7, §8.1) |
| F8 | Major | Dự báo 30–34 đếm trùng whitening | Dự báo viết lại theo lớp G và lớp J (§6.5) |
| F9 | Major | R chưa implement được: view bị che, ~1,4 clip/cụm, bộ cân bằng đóng băng | Mô phỏng mini-batch (§2.6: thất bại); R thành có điều kiện, thiết kế lại, cổng G-R (§5.6) |
| F10 | Major | Tan rã cụm; thiếu T1-shuf/R1-shuf; bản tiết kiệm bỏ T1m/R1m | C-fixed mặc định; arm T1-shuf, T1m trong bản tiết kiệm, T1o riêng (§5.3, §6.1) |
| F11 | Major | H1/H3/H5 không kiểm chứng được; ngưỡng ±1 dưới nhiễu | Viết lại §4; không-kém thay tương đương ±1; H5 thành kiểm không-hại; CI signer-cấp |
| F12 | Major | Đường cong gallery trích sai (+2 thay vì ≈ 3,4) và dùng sai cho H7 | Sửa §2.3, §3.3; H7 là quy tắc thiết kế từ literature |
| F13 | Major | Trần LDA trong mẫu không chứng minh hình học | Thêm LDA fit trên gloss rời: 66,0 so với raw 35,9 (§2.2, `ladder_extra.py`) |
| F14 | Major | ContentVec within/across, Bootleg sắc thái, ngưỡng [I]/[V], so sánh probe | Sửa §3 và §2.8 |
| F15 | Minor | η²_gloss chứa null; delta làm tròn; "~25/30 biến"; k LDA lẫn; lift theo nền; vector 0; NaN trong JSON | Sửa trong §2.3, §2.8, §2.9, §2.11; JSON không còn NaN |

**R2 (xác minh, cùng reviewer, `reviews/R2_verification_review.md`):** F1–F4, F10–F14 Fixed; F5, F6, F8, F9, F15 Partly; F7 Not fixed vì lỗi mới N1. Xử lý:

| ID | Mức | Vấn đề | Xử lý |
|---|---|---|---|
| N1 | Critical | `morph()` lan NaN từ cổ tay không quan sát sang cả bàn tay (đổi 5,4 % cờ quan sát ở hệ số ≈ 1): bảng liều v1.1 đầu sai; R1 từng xác nhận nhầm hàm này đúng | Hàm NaN-safe + `morph_unittest.py`, chạy lại toàn bộ (§2.7: dịch tăng theo liều 0,08 → 0,23; cánh tay ±30–50 % làm R@1 −1,1…−1,2); E-canon từ "NO-GO" thành "ưu tiên thấp" |
| N2 | Major | ARI chẵn/lẻ tầm thường (74 %/67 % clip chỉ có một cửa sổ) | Tính lại trên clip ≥ 4 cửa sổ: 0,94 / 0,63 (final), ARI val 0,41–0,63 (§2.5) |
| N3 | Major | Hàng CV trong §2.11 gán nhãn "silhouette" nhưng số là K_q = 6 | Tách hai hàng, ghi đúng nhãn |
| N4 | Major | N thiếu đối chứng và chứng cứ signer giữ lại | Thêm K = 1, định tuyến `grp`, 3 seed, CV signer giữ lại (§2.5); G-G đòi các đối chứng này trên tập đầy đủ |
| N5 | Major | Thất bại mini-batch bị quy cho method trong khi bộ ước lượng có lỗi; so sánh thật/xáo trộn không đo tính đặc thù | Viết lại §2.6 và §0.4 theo phân tích R2 (đối chứng thống kê toàn batch 28,4; xáo trộn 22,2–23,1) |
| N6 | Minor | Ngưỡng ≥ 1.500 gloss chọn sau; KEEP "≤ 1 điểm" dưới nhiễu; H3 chưa có số; "+3,7" là best-of-K; vài sai số | Nêu cả hai ngưỡng (+8,3 / +6,5); KEEP chỉ ràng buộc signer lớn; H3 ±1,5; +3,4 mặc định, +3,0 trung bình K; "0,71", "+2,5…+3,0", lưu ý SE ghép cặp 0,6 |
| Thiếu | — | Thứ quan trọng nhất trước khi duyệt GPU: chưa có chẩn đoán hỗ trợ lớp J (T, D1–D3), toàn bằng chứng dương là cho lớp G không train | Thêm cổng G-T (pilot ≤ 1–2 GPU-h: A0, T1f, T1-shuf) và G-G đầy đủ với SignRep + cùng stack bắt buộc nếu tuyên bố so sánh (§6.2, §6.4) |

Chưa có vòng review thứ ba; nếu §5.6 (R) hoặc N trên tập đầy đủ được hiện thực, cần review lại trước khi chạy GPU.

---

## 10. Nguồn

**Chẩn đoán nội bộ (`diagnostics/`):** `common.py`, `geo_headroom.py`, `geo_ladder.py`, `ladder_cv.py`, `ladder_extra.py`, `nuisance_decomp.py`, `signer_norm.py`, `emb_signer_cluster.py`, `eclus_control.py`, `cls_cluster.py`, `kfree_cluster_corral.py`, `routed_coral.py`, `cls_early.py`, `pseudo_signer.py`, `local_norm.py`, `coral_head.py`, `coral_head2.py`, `coral_head_mb.py`, `nn_head.py`, `nn_head_cls.py`, `nn_precision_fullpool.py`, `style_vars.py`, `nuis_pc.py`, `morph_effect.py`, `morph_dose.py`, `emb_variants.py`, `analyze_emb.py`, `ceilings.py`, `retention_scalefree.py`, `channel_analysis.py`, `inp_features.py`, `lex_probe_emb.py`, `error_attrib.py`, `subset_check.py`; kết quả JSON ở `diagnostics/results/`; review `reviews/R1_independent_review.md`.

**Literature (verified HTML thô):** `literature/Q1`, `Q2`, `Q3`. arXiv chính: Littwin 2407.03475; Sobal 2211.10831; Gulati–Nemenman 2606.07770; MotionJEPA 2609.23881; ContentVec 2204.09224; Komatsu 2409.10103; Spin 2305.11072; IWM 2403.00504; seq-JEPA 2505.03176; Bootleg 2603.15553; V-JEPA 2.1 2603.14482; EEG-JEPA 2608.00114; WavJEPA 2509.23238; CF-JEPA 2606.07031; Human-JEPA 2608.21160; LeJEPA 2511.08544; LeWM 2603.19312; MAMP 2308.07092; GFP 2509.03609; MASA 2405.20666; STARS 2407.10935; MAE-CT 2304.10520; NNCLR 2104.14548; CAP 2012.10674; SSPS 2505.14561; Liu et al. 2305.12464; LEACE 2306.03819; Gonen–Goldberg 1903.03862; SignRep 2503.08529; SignSeek 2609.03695; ASL Citizen 2304.05934; OpenHands 2110.05877; SSL-SLR 2509.05188; BEST 2302.05075; SHuBERT 2411.16765 (tham chiếu). Miền: Deep CORAL 1607.01719; AdaBN 1603.04779; MixStyle 2104.02008 (id kiểm bằng curl; không trích số).

**Tài liệu nội bộ trước đó:** `docs/proposal12` (PLctx), `docs/research` (phễu, VAL-DICT, luật quyết định), `docs/proposal16` (PhonoPath-JEPA, D-LEX, audit dữ liệu D1).

## Phụ lục A: tái lập

```bash
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
cd /home/haipd/Sign_Representations/docs/proposal17/diagnostics
export OMP_NUM_THREADS=4
$PY geo_headroom.py results/geo_headroom.json && $PY geo_ladder.py && $PY ladder_cv.py && $PY ladder_extra.py
$PY nuisance_decomp.py && $PY signer_norm.py && $PY emb_signer_cluster.py && $PY eclus_control.py
./run_extract.sh                                  # ~35 phút: teacher/student/ngẫu nhiên, mọi tầng, tập con, perturbation
$PY analyze_emb.py && $PY ceilings.py && $PY channel_analysis.py && $PY inp_features.py && $PY lex_probe_emb.py
$PY cls_cluster.py && $PY kfree_cluster_corral.py && $PY routed_coral.py
$PY coral_head.py && $PY coral_head2.py && $PY coral_head_mb.py
$PY nn_head.py && $PY nn_precision_fullpool.py && $PY nn_head_cls.py
$PY style_vars.py results/style_vars.json && $PY nuis_pc.py && $PY morph_effect.py && $PY error_attrib.py
./run_r1_fixes.sh                                 # retention_scalefree, ladder_extra, ..., morph_dose, cls_early (chạy tuần tự)
```

Embedding trích lại nằm ở scratchpad (`.../scratchpad/emb`), không lưu vào repo (vài GB). ASL-LEX: `docs/proposal16/data/` (CC BY-NC, không commit).
