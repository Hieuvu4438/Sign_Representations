# Proposal 17 — SA-JEPA: chiến lược JEPA căn chỉnh signer và ép động học cho biểu diễn ký hiệu từ skeleton

## Từ chẩn đoán "thông tin so với hình học" của PLctx full160 đến một method JEPA không dùng Path-JEPA

**Trạng thái (2026-10-17): chỉ đề xuất, v1.0, chưa qua review độc lập.** Chưa train model nào. Mọi con số [D] đến từ chẩn đoán CPU chạy trên embedding PLctx full160 đã có sẵn, **chỉ dùng train (gallery) và val (query), không đụng test**. Mọi con số dự báo hiệu quả của SA-JEPA là [I] (suy luận), chưa được kiểm chứng bằng huấn luyện.

Ký hiệu nguồn: **[D]** chẩn đoán trong `docs/proposal17/diagnostics/` · **[V]** số đã đối chiếu với HTML thô của paper (xem `literature/Q1–Q3`) · **[I]** suy luận / dự báo · **[P]** số trong paper chưa kiểm lại · **[R]** số nội bộ của repo (report cũ).

---

## 0. Tóm tắt điều hành

**Câu hỏi.** Nếu bỏ hướng Path-JEPA, nên áp dụng chiến lược JEPA nào cho SSL biểu diễn ký hiệu trên skeleton (downstream: retrieval từ điển đóng băng kiểu SignRep, ISLR), và method nên được xây thế nào?

**Cách làm.** Thay vì chọn chiến lược theo trực giác, tôi đo trước xem PLctx full160 (test R@1 16,68; val 22,7) đang thiếu gì, bằng chẩn đoán CPU trên chính embedding của nó, rồi mới chọn chiến lược. Ba báo cáo literature mới (`literature/Q1–Q3`, ~1.150 dòng, số liệu đối chiếu HTML thô) cung cấp bản đồ chiến lược và các điều kiện cổng.

**Kết quả chẩn đoán (val, gallery = 40.154 clip train, 10.304 query; tái lập đúng R@1 22,7193 của baseline):**

1. **Thông tin có sẵn, hình học thì sai.** Một ánh xạ tuyến tính có giám sát gloss (LDA) trên chính embedding này đưa R@1 từ 22,7 lên **50,9** [CI 45,7–54,2]. Encoder đã học được nhiều: probe gloss-rời + signer-rời trên 9 tham số ASL-LEX cho thấy encoder hơn input của nó +5…+18 điểm balanced-accuracy và hơn encoder khởi tạo ngẫu nhiên +5…+13. Khoảng cách 22,7 → 50,9 không phải thiếu thông tin tuyến tính mà là **hình học cosine bị nhiễu chi phối**.
2. **Nhiễu chi phối là signer, và nó là một ánh xạ tuyến tính theo signer, không phải độ lệch cộng.** 75 % phương sai embedding nằm trong cùng gloss; láng giềng gần nhất trong train cùng signer **85,4 %** (ngẫu nhiên 5,9 %), cùng gloss chỉ 5,5 %. Trừ trung bình theo signer cho +2,2 so với whitening; căn chỉnh **hiệp phương sai theo signer (CORAL)** cho **+9,1** [CI 7,6–11,2].
3. **Token CLS là chữ ký signer gần như hoàn hảo** (η²_signer 0,44; láng giềng cùng signer 96,6 %; R@1 chỉ 0,25). Phân cụm CLS cho pseudo-signer **không nhãn** đủ tốt để CORAL đạt mức oracle: 32,3 so với 31,7 (oracle signer thật) trên tập con.
4. **Một ánh xạ tuyến tính quy nạp (không dùng thống kê val) huấn luyện bằng mục tiêu căn chỉnh cụm** nâng val R@1 từ 29,4 (whiten) lên **35,5** khi dùng signer id, và trên tập con cụm CLS không nhãn cho 28,5 so với 28,1 (signer thật) so với whiten 24,9. Hiệp phương sai là phần quan trọng (cov-only 29,3 > mean-only 25,8).
5. **Những giả thuyết bị chẩn đoán bác bỏ:**
   - Nhiễu *không* phải biến đổi phân tích đơn giản: 30 biến phong cách giải thích ≈ 0 % phương sai trong-gloss ở signer giữ lại; perturbation hình thái ±15 % (tay, cánh tay) dịch embedding rất nhỏ (cos 0,93; so với cùng-gloss-khác-signer 0,60) và **không làm đổi R@1**.
   - Training JEPA **không khuếch đại** phương sai signer so với encoder ngẫu nhiên (η²_signer 0,27 → 0,14), dù cụm cục bộ cùng-signer mạnh hơn (72 % → 84 %).
   - Tinh chỉnh bằng positive láng giềng xuyên signer **làm hại** ở độ chính xác hiện tại (precision 10–36 % < cổng 40 %): head train trên positive không hạn chế cho val R@1 13,3; loại trừ signer thật 25,2–26,1; đều dưới whitening 29,4.
   - Chọn tầng, student/teacher, readout có thứ tự thời gian (bins 2/4) **không giúp**; token CLS không dùng được làm readout.
6. **Cảnh báo độ tin cậy.** Val chỉ có 5 cụm signer hiệu dụng. Lập lại thang trên 5 lần tách signer khác nhau (6 signer train giữ lại, 29 gallery): các bậc theo signer **lặp lại** (CORAL +9,3; CORAL không nhãn +7,7; LDA +25,8), nhưng **whitening một mình chỉ +1,5** (khoảng −1,0…+3,1 qua 5 lần; so với +6,7 trên val). Kết luận: căn chỉnh theo signer là đòn bẩy bền; whitening toàn cục là mong manh.

**Method đề xuất — SA-JEPA (Signer-Aligned JEPA), có mô-đun động học tuỳ chọn +DYN:**

| Mô-đun | Nội dung | Căn cứ chính |
|---|---|---|
| **T** | Center mục tiêu của teacher **theo pseudo-signer** (EMA center theo cụm CLS) thay vì một center toàn cục | CMN +2,2; JEPA-native; không thêm tham số |
| **R** | Regularizer căn chỉnh **hiệp phương sai và trung bình theo cụm** trên embedding gộp `grp`, cộng điều khoản đẳng hướng | CORAL +9; head quy nạp +3…+6 |
| **C** | Pseudo-signer = k-means trên **CLS** (làm mới mỗi epoch); signer id chỉ cho arm chẩn đoán | CLS-CORAL = oracle; AMI 0,71 |
| **D** | Mục tiêu ép động học: multi-depth, multi-span, hiệu thời gian, khối tương lai | Bootleg, GFP, MASA, MotionJEPA; probe chuyển động yếu |
| **H/S** | Readout chỉ tay, bỏ CLS, whitening fit không nhãn; CORAL theo cụm CLS báo riêng (transductive) | hands-only +2,4…+3,4 raw |

**Bị loại bằng chứng (NO-GO):** retarget hình thái giải tích; positive láng giềng xuyên signer (đến khi cổng 40 % đạt); SIGReg/VICReg làm mục tiêu chính; conditioning predictor trên nhân tố tạo equivariance; readout có thứ tự đơn giản; CLS alignment.

**Chương trình:** Phase 0 chẩn đoán CPU (đã xong) → Phase 1 chuẩn hoá baseline B* = A0 + stack, đo `mw` (CPU/suy luận) → Phase 2 screening 40 epoch (A0×3 + ~11 arm, 1 seed) → Phase 3 xác nhận 3 seed → Phase 4 trục dữ liệu đa signer + full160 + test một lần → Phase 5 downstream. **Ngân sách ≈ 58–75 GPU-h** (≈ 30 GPU-h bản tiết kiệm), tuần tự, một job GPU mỗi lúc. Cần bạn duyệt trước khi dùng GPU.

**Điều bạn cần quyết định:** (i) cho phép dùng signer id của ASL Citizen làm tín hiệu yếu *chỉ ở arm chẩn đoán* (đã khai báo), còn method chính không nhãn; (ii) có cho phép báo cáo thêm biến thể transductive (CORAL theo cụm query) như một hàng riêng không; (iii) có mở trục dữ liệu MS-ASL/WLASL (222/119 signer) ở Phase 4 không; (iv) lịch GPU khi job PhonoPath full160 đang chạy.

---

## 1. Bối cảnh và vị trí của proposal này

- **Đã có trong repo:** PLctx full160 (token 7 kênh × 32 bin, mask PLCM-lite, centered-CE EMA, readout `grp`) test R@1 16,68, val 22,7; tiếp tục train 160 → ~200 epoch cho val +0,23 / test −0,05 [R] (bão hoà). `docs/research` đã thiết kế phễu 15 arm O/K/V/I; `docs/proposal16` (PhonoPath-JEPA) thêm token signature và đang được chạy ở `method/phonopath_jepa`. Hai điều **chưa có** trong tài liệu nào: (a) một phép đo định lượng nút thắt của PLctx nằm ở đâu, (b) một chiến lược JEPA bắt nguồn từ phép đo đó.
- **Proposal này khác ở đâu:** không thêm arm theo trực giác. Nó (1) đo nút thắt trên embedding đã có, (2) dùng đo đạc để loại các arm đã đề xuất nhưng bị chẩn đoán bác bỏ (retarget giải tích V1, NN-tuning O8, isotropy objective), (3) đề xuất một họ chiến lược mới (căn chỉnh theo signer trong target và biểu diễn) mà `docs/research` chưa có.
- **Ràng buộc giữ nguyên:** skeleton RTMW-x COCO-WholeBody 2D; method chính không dùng nhãn gloss; test ASL Citizen đã dùng một lần (không dùng để chọn); một job GPU mỗi lúc, GPU dùng chung (hiện job PhonoPath full160 của bạn đang chạy).

---

## 2. Chẩn đoán: PLctx full160 đang thiếu gì?

**Thiết lập và kiểm tra tính toàn vẹn.** Embedding là `_grp` 768-d của checkpoint gốc `PJ_PLctx_FULL_160ep/final.pt` (EMA teacher), đã cache sẵn: gallery 40.154 clip train (35 signer), query 10.304 clip val (6 signer, trong đó P13 chỉ có 2 clip, nên thực chất 5 cụm). `common.py` tái lập chính xác baseline: **R@1 22,7193 / R@5 45,9433 / R@10 56,7838** (khớp `validation_iterations.csv`). Giao thức: cosine, class = max trên clip gallery, tie bi quan (SignRep §6). Mọi biến đổi được fit **chỉ trên train**; siêu tham số chọn trên một nhóm signer giữ lại bên trong train. Test không bao giờ được tải. Mọi bootstrap lấy lại mẫu signer rồi gloss.

Bảng script ↔ kết quả đầy đủ ở `diagnostics/README.md`.

### 2.1 Bậc thang retrieval (val, `geo_ladder.py`)

| Bậc | Điều kiện | Val R@1 | Δ so với raw (CI 95 %) | Ghi chú |
|---|---|---:|---|---|
| 0 | raw (giao thức) | **22,72** [16,8–27,7] | — | |
| 1 | PCA-whiten (k=512, fit train, không nhãn) | 29,44 | +6,7 (+5,2…+8,2) | quy nạp, **mong manh** (§2.9) |
| 2 | trừ trung bình láng giềng (k=200) | 30,41 | +7,7 | quy nạp, không nhãn; Δ so với whiten [−0,2…+2,3] |
| 3 | LEACE signer + whiten | 31,66 | +8,9 | cần signer id (tín hiệu yếu) |
| 4 | CMN theo signer + whiten | 31,66 | +8,9 | oracle |
| 5 | **CORAL theo cụm không nhãn + whiten** | **34,58** | +11,9 (+8,1…+15,5) | transductive: 6 cụm query, 70 cụm gallery trong embedding |
| 6 | **CORAL theo signer thật + whiten** | **38,53** | +15,8 (+13,5…+18,1) | oracle, transductive |
| 7 | LDA có giám sát gloss (k=96) | **50,88** [45,7–54,2] | +28,2 (+25,2…+32,3) | trần tuyến tính |

Bậc 5 và 6 so với bậc 1: Δ CI [2,2…8,5] và [7,6…11,2]. Theo signer (R@1 raw → CORAL oracle → LDA): P12 14,6 → 33,3 → 43,8; P21 20,2 → 36,8 → 52,9; P26 21,5 → 33,9 → 47,2; P39 30,3 → 47,6 → 55,3; P5 13,6 → 31,4 → 45,3. **Cả 5 signer cải thiện ở mọi bậc.** R@1 raw dao động mạnh theo signer (13,6–30,3) nên bản thân signer là nguồn phương sai lớn của chỉ số.

### 2.2 Cấu trúc nhiễu (`nuisance_decomp.py`, train 40.154 clip)

| Đại lượng | Giá trị |
|---|---|
| phương sai nằm trong cùng gloss | **75,4 %** (η²_gloss = 24,6 %) |
| η²_signer của tổng phương sai / của phương sai trong-gloss | 13,8 % / 17,5 % |
| các nhân tố kỹ thuật trong phương sai trong-gloss | thập phân vị thời lượng 2,9 %; fps 0,5 %; độ phân giải 0,03 % |
| tính "hạng" (participation ratio) của hiệp phương sai trong-gloss Sw / giữa-gloss Sb | **84,4 / 26,9** |
| phần phương sai Sb nằm trong 100 hướng đầu của Sw | **80,9 %** (tín hiệu gloss và nhiễu dùng chung không gian con) |
| không gian con signer 34 chiều chứa | 39,1 % vết Sw; 20,0 % vết Sb |
| láng giềng top-1 trong train: cùng signer / cùng gloss | **85,4 %** (ngẫu nhiên 5,9 %) / 5,5 % |
| láng giềng top-1 buộc khác signer: cùng gloss | 14,4 % |

**Đường cong gallery (raw, val):** số clip mỗi gloss 1→2→4→8→16: 8,2 → 11,1 → 14,4 → 18,8 → 22,7 (**khoảng +3,5 mỗi lần gấp đôi**); số signer trong gallery 4→8→16→24→35: 9,8 → 14,9 → 18,9 → 20,5 → 22,7. Retrieval phụ thuộc mạnh vào độ phủ cross-signer của gallery và chưa bão hoà.

### 2.3 Signer là một ánh xạ tuyến tính, và CLS là mã signer (`signer_norm.py`, `cls_cluster.py`, `emb_signer_cluster.py`)

- Chuẩn hoá theo signer (oracle, transductive), val R@1 không whiten → có whiten: CMN 28,4 → 31,7; CMVN (trung bình + phương sai) 28,8 → 31,2; **CORAL (căn chỉnh hiệp phương sai) 30,0 → 38,5**. Phần hơn nằm ở **hiệp phương sai theo signer**, không ở độ lệch cộng hay thang theo chiều.
- Nhóm đối chứng: nhóm query ngẫu nhiên (K=6) với CORAL gallery theo signer thật cho 32,7; ngẫu nhiên cả hai phía cho 27,5 (dưới whiten 29,4): các cụm *thật* đóng góp (+4,8 so với ngẫu nhiên phía query), không phải artifact của whiten.
- **Cụm trong embedding `grp`** (không nhãn, transductive): Kq=6 → 37,5 (gallery signer thật) / 34,6 (gallery 70 cụm); Kq=12 → 34,4 / 32,6; Kq=24 → 31,3 / 29,5; Kq=48 → 32,3 / 30,0. Ít cụm lớn tốt hơn nhiều cụm nhỏ. **Lưu ý chọn K:** Kq=6 trùng số signer val (kiến thức oracle) và là giá trị tốt nhất trong lưới; trên test (11 signer) K phải ước lượng không nhãn, nên con số 34,6 là *cận trên lạc quan* của biến thể không nhãn.
- **CLS = chữ ký signer.** Trên tập con (gallery 20.077, query 5.152): η²_signer của CLS 0,44 so với 0,14 của `grp`; láng giềng cùng signer 96,6 %; R@1 raw của CLS 0,25. K-means CLS: AMI với signer thật 0,71 / 0,70 / 0,66 (K=35/70/140 trên train; purity 0,79/0,85/0,88); trên val AMI 0,44 / 0,455 (K=6/12; purity 0,73/0,80). Các cụm *biến phong cách* (hình thái, tư thế) chỉ cho AMI val 0,15 và CORAL +1,0…1,4 (không dùng được).
- **CORAL theo cụm CLS (không nhãn, transductive), tập con:** raw 18,3; whiten 24,9; oracle signer thật 31,7; cụm CLS 35/6 → 31,6; 35/12 → **32,3**; 70/12 → 31,7; 140/12 → 29,4. Tức **cụm CLS không nhãn tái tạo được toàn bộ lợi ích oracle**.

### 2.4 Có thể học quy nạp phần căn chỉnh này không? (`coral_head.py`, `coral_head2.py`)

Một ma trận W (768 → 512), khởi tạo ở whitening, huấn luyện chỉ trên embedding train bằng loss `‖Cov(z) − I‖² + Σ_c w_c(‖μ_c‖² + ‖Cov_c − Cov‖²)` với cụm c. Một W duy nhất áp cho mọi clip khi test (không thống kê val):

| Nhóm | Tập đầy đủ (val R@1; whiten 29,4) | Tập con (whiten 24,9) |
|---|---:|---:|
| signer thật (trung bình + hiệp phương sai) | **35,5** | 28,1 |
| cụm CLS K=35 (không nhãn) | — | **28,5** |
| cụm CLS K=70 (không nhãn) | — | 28,2 |
| cụm `grp` K=70 (không nhãn) | — | 26,0 |
| signer thật, chỉ trung bình | — | 25,8 |
| signer thật, chỉ hiệp phương sai | — | **29,3** |

Kết luận [D]: căn chỉnh theo cụm có thể được học quy nạp ngay cả khi chỉ là một ánh xạ **tuyến tính** (+3…+6 so với whiten), bằng nhóm không nhãn lấy từ CLS, và hiệp phương sai là phần chủ chốt. Đây là bằng chứng trực tiếp ủng hộ một regularizer căn chỉnh *bên trong* encoder (phi tuyến, có thể hơn nữa).

### 2.5 Phần nhiễu KHÔNG giải thích được bằng biến đơn giản (`style_vars.py`, `nuis_pc.py`, `morph_effect.py`)

- **Nhận diện signer bằng biến diễn giải được:** từ ~25 biến clip-level (hình thái, tư thế, hành vi, camera) cho độ chính xác 35-way **90,4 %** (embedding: 90,7 %; baseline lớp đa số 7,5 %). Riêng hình thái 72,6 %; tư thế 48,7 %; hành vi 44,4 %; camera 20,0 % (cờ độ phân giải là rò rỉ của giao thức ghi).
- Nhưng **trên signer giữ lại**, các biến này giải thích **≈ 0 %** phương sai trong-gloss của embedding (R² nhiều đầu ra: −1,7 %; riêng hành vi +0,16 %; phần còn lại sau trừ trung bình signer: −1,0 %). Chỉ vài hướng chủ đạo của nhiễu có thể diễn giải: PC1 R² 45 %, PC3 37 %, chủ yếu `eye_dist`, `rest_height_D`, `brow_width` (khung hình/tư thế nghỉ).
- **Perturbation hình thái giải tích** (cắt kích thước mặt, độ dài cánh tay, cỡ tay, hệ số log-đều [0,85; 1,15], không đổi theo clip, áp lên keypoint thô trước canonicalise) trên 1.500 clip val: cỡ mặt cos 1,000 (dịch 0,0007: encoder đã bất biến nhờ khung mắt trừ neutral); cánh tay 0,929; bàn tay 0,930; cả ba 0,928 (dịch 0,29). Tham chiếu: cùng-gloss-khác-signer 0,601; cùng-signer-khác-gloss 0,471; cặp ngẫu nhiên 0,440; chuẩn dịch cặp cùng-gloss-khác-signer 0,883. **R@1 không giảm**: giảm 0,07 điểm (mặt), tăng khoảng 1,1 điểm (cánh tay, bàn tay, cả ba; nằm trong nhiễu của tập con 1.500 clip). Phần dịch nằm trong không gian con signer: 23–27 % năng lượng so với 4,4 % ngẫu nhiên, tức *cùng hướng nhưng quá nhỏ*.
- Hệ quả: nhiễu signer là **đặc thù, nhiều chiều và không phải phép biến hình thái đơn giản**. Retarget giải tích (V1 của `docs/research`) có giá trị kỳ vọng thấp, khớp với G3morph ≈ 0 [R]. Nhưng nó vẫn là nhiễu *thống kê* (hai mô-men đầu theo nhóm) nên căn chỉnh phân phối xử lý được.

### 2.6 Encoder đã học gì? (`emb_variants.py`, `analyze_emb.py`, `ceilings.py`, `lex_probe_emb.py`, `channel_analysis.py`)

Trích embedding teacher / student / **khởi tạo ngẫu nhiên** (cùng kiến trúc) và token đầu vào (không encoder) trên tập con (gallery 20.077, query 5.152; stride 16, đã kiểm: cos 0,9975 với embedding stride 2 đã cache).

| Biểu diễn | R@1 raw | whiten | LDA-96 | η²_gloss | η²_signer | láng giềng cùng signer |
|---|---:|---:|---:|---:|---:|---:|
| trung bình token đầu vào (không encoder) | 3,4 | 6,2 | 12,9 | 0,24 | 0,15 | 58,3 % |
| encoder **ngẫu nhiên** (`grp`) | 4,2 | 11,0 | 21,2 | 0,25 | **0,27** | 72,4 % |
| **teacher đã train** (`grp`) | **18,3** | **24,9** | **40,1** | 0,30 | **0,14** | 84,0 % |
| student (`grp`) | 18,4 | 24,9 | (36,2 ở k=256) | 0,30 | 0,14 | 83,9 % |
| teacher, chỉ tay (kênh 0–3) | 20,7 | 25,5 | 41,9 | 0,30 | 0,12 | 82,5 % |
| teacher, chỉ CONF (kênh 0,2) | **21,7** | **26,6** | 39,1 | 0,32 | 0,11 | 78,7 % |
| teacher, chỉ mặt (kênh 4–6) | 9,4 | 16,2 | 35,1 | 0,28 | 0,18 | 83,5 % |
| teacher CLS | 0,25 | 0,5 | 0,8 | 0,13 | **0,44** | 96,6 % |

Đọc [D]:
- **Training không khuếch đại phương sai signer** (η² 0,27 → 0,14) nhưng làm cụm cục bộ cùng-signer chặt hơn (72 → 84 %). Giả thuyết "JEPA ưu tiên đặc trưng dự đoán được nên khuếch đại signer" (Littwin [V], Sobal [V]) **không được ủng hộ ở mức phương sai, được ủng hộ yếu ở mức láng giềng**. Phần lớn nhiễu signer **thừa hưởng từ input**: encoder ngẫu nhiên đã có 72 % láng giềng cùng signer, token input trung bình 58 %.
- **Học thật sự diễn ra:** LDA 12,9 (input) → 21,2 (ngẫu nhiên) → 40,1 (đã train); R@1 raw 3,4 → 4,2 → 18,3.
- **Mặt là nơi mang signer:** kênh F_HD η²_signer 0,37 và η²_gloss 0,14; bỏ mặt khỏi readout nâng raw R@1 **+2,4** (chỉ tay) đến **+3,4** (chỉ CONF), LDA không giảm (41,9). Bỏ CLS khỏi mọi readout.
- **Tầng, student/teacher, thứ tự:** tất cả tầng 1–6 và tầng cuối cho LDA 33–41 (tầng 2: 40,9, tầng 1: 37,3, cuối: 40,1): thông tin tuyến tính bão hoà sớm. Student ≈ teacher (R@1 18,44 vs 18,28). Readout có thứ tự (2 và 4 đoạn) **kém hơn** `grp` (whiten 18,9 / 15,9 so với 24,9; LDA-256 29,9 / 30,6 so với 36,1), có thể vì cửa sổ trượt không căn theo thời điểm bắt đầu dấu.

**Probe tham số âm vị học ASL-LEX 2.0 (gloss-rời, signer-rời, balanced accuracy; cùng giao thức D-LEX v3 của proposal16, tách 819 gloss giữ lại trùng khớp):**

| Mô tả | SignType | Movement | Repeated | MajorLoc | Contact | FlexChange | WristTwist | SelFingers | Flexion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PLctx teacher** `grp_fin` | **76,9** | **35,7** | **67,3** | **68,2** | **71,8** | **62,8** | **64,0** | **68,1** | **47,6** |
| PLctx chỉ tay | 76,3 | 36,7 | 66,8 | 66,9 | 71,3 | 62,6 | 62,0 | 67,5 | 47,8 |
| encoder ngẫu nhiên | 63,8 | 27,6 | 55,0 | 58,4 | 67,1 | 55,2 | 55,2 | 55,5 | 42,1 |
| token input trung bình | 58,4 | 25,0 | 53,7 | 54,3 | 66,4 | 54,5 | 51,9 | 50,8 | 39,9 |
| descriptor thô tốt nhất (D-LEX v3) | 56,9 | 30,9 | 56,7 | 51,1 | 66,0 | 56,9 | 54,1 | 48,9 | 35,4 |
| **hơn input / hơn ngẫu nhiên** | +18,5/+13,1 | +10,7/+8,1 | +13,6/+12,3 | +13,9/+9,8 | +5,4/+4,7 | +8,3/+7,6 | +12,1/+8,8 | +17,3/+12,6 | +7,7/+5,5 |

Probe cho thấy **cả các tham số chuyển động đều giải mã tuyến tính được hơn input và hơn encoder ngẫu nhiên** (Repeated +13,6, WristTwist +12,1, Movement +10,7). Điều này **điều chỉnh** nhận định cũ "movement không được mã hoá" (R² < 0 [R]): thông tin có mặt nhưng yếu (Movement 35,7 % cho 5 lớp, CI 30,4–41,7).

### 2.7 Lỗi retrieval theo tham số ASL-LEX (`error_attrib.py`)

Với mỗi query sai top-1, so tham số của gloss đúng và gloss được trả về; lift = tỉ lệ khớp trong lỗi chia tỉ lệ khớp của hai gloss ngẫu nhiên (mức lift tối đa bị chặn bởi tỉ lệ nền với tham số nhị phân):

| Tham số | raw | whiten | LDA |
|---|---:|---:|---:|
| Handshape (đầy đủ) | **4,95** | 5,20 | 7,31 |
| SignType | 2,46 | 2,49 | 2,66 |
| MinorLoc | 2,37 | 2,41 | 2,88 |
| MajorLoc | 2,16 | 2,20 | 2,52 |
| SelFingers | 1,98 | 1,95 | 2,32 |
| Flexion | 1,65 | 1,69 | 2,03 |
| Contact | 1,24 | 1,26 | 1,36 |
| Movement | **1,18** | 1,26 | 1,40 |
| Repeated | 1,13 | 1,13 | 1,17 |
| FlexChange | 1,12 | 1,13 | 1,19 |
| WristTwist | **1,04** | 1,07 | 1,10 |

Khoảng cách Hamming trung bình (9 tham số lõi) của lỗi: 3,71 (raw) so với 5,26 ngẫu nhiên; LDA 3,13. **Lỗi bám vào handshape, vị trí, loại ký hiệu, nhưng không bám vào tham số chuyển động ngay cả khi LDA có nhãn.** Ghép với probe (§2.6): thông tin chuyển động có trong embedding nhưng chiếm phần phương sai nhỏ, nên hình học cosine hầu như không dùng nó.

### 2.8 Positive láng giềng xuyên signer có đủ sạch để tinh chỉnh không? (`signer_norm.py`, `nn_head.py`)

Độ chính xác (cùng gloss) của láng giềng top-1 buộc khác signer trong train:

| Hình học | precision top-1 | tỉ lệ cặp tương hỗ | precision cặp tương hỗ |
|---|---:|---:|---:|
| raw | 9,8 % | 21,3 % | 19,3 % |
| whiten | 16,5 % | 30,2 % | 25,9 % |
| LEACE + whiten | 17,7 % | 31,2 % | 28,2 % |
| CORAL theo signer + whiten | 24,0 % | 34,2 % | 35,9 % |
| LDA (fit trên chính train, lạc quan) | 39,4 % | 29,5 % | 58,8 % |

Tinh chỉnh một head MLP trên đặc trưng whiten đóng băng bằng InfoNCE với các positive này (val R@1; tham chiếu whiten 29,4): không hạn chế 13,3; loại trừ cụm 13,1; cụm + tương hỗ 15,6; loại trừ signer thật 25,2; signer thật + tương hỗ 26,1; **positive cùng gloss (oracle) 38,8**. Các vòng tự huấn luyện thứ 2–3 **giảm thêm**. **Cổng của literature (precision tương hỗ ≥ 40 % [V]) chưa đạt ở bất kỳ hình học không giám sát nào**, nên NN-tuning (O8, STARS/MAE-CT) là **NO-GO** cho đến khi căn chỉnh nâng được precision.

### 2.9 Độ tin cậy và giới hạn

- **Val chỉ có 5 cụm signer.** CI bootstrap (signer rồi gloss) rất rộng (raw 16,8–27,7). Kết luận quan trọng dựa vào Δ có cặp và tính nhất quán theo từng signer, không dựa vào giá trị tuyệt đối.
- **Lập lại thang trên signer khác (`ladder_cv.py`):** 5 lần tách ngẫu nhiên 6 signer train làm query, 29 signer làm gallery, mọi phép fit lại chỉ trên gallery (lưu ý: encoder đã thấy các clip này lúc SSL không nhãn, nên R@1 tuyệt đối lạc quan; chỉ nhìn Δ):

| Bậc | R@1 (trung bình ± SD qua 5 lần) | Δ so với raw |
|---|---:|---:|
| raw | 16,2 ± 2,9 | — |
| whiten | 17,7 ± 3,5 | **+1,5** (khoảng −1,0…+3,1) |
| CMN theo signer + whiten | 21,7 ± 3,3 | +5,5 |
| LEACE + whiten | 19,1 ± 3,6 | +2,9 |
| **CORAL không nhãn + whiten** | 23,9 ± 3,4 | **+7,7** |
| **CORAL theo signer + whiten** | 25,5 ± 2,0 | **+9,3** |
| LDA-96 | 42,0 ± 4,3 | +25,8 |

  Các bậc theo signer (CMN, CORAL, CORAL không nhãn) và trần LDA **lặp lại**; riêng whiten toàn cục chỉ +1,5 thay vì +6,7: **không nên coi whitening là lợi ích bền**.
- **Tất cả bậc ở §2.1 là post-hoc**; bậc 5–6 là **transductive** (dùng thống kê của tập query) và phải được báo cáo như hàng riêng, không so trực tiếp với số raw của SignRep. Bậc LDA dùng nhãn gloss của train.
- **Chưa kiểm:** (a) SignRep có hưởng cùng stack không (cần trích đặc trưng 40k clip, ước tính 6–10 GPU-h); (b) readout hoạt động `mw` (SignRep +12,5 [V]) và chamfer trên PLctx; (c) val không đại diện cho test (22,7 so với 16,7); (d) mọi con số về hiệu quả của SA-JEPA là dự báo.

---

## 3. Bản đồ chiến lược JEPA: literature gặp chẩn đoán

Mỗi dòng gắn chiến lược với một **chế độ lỗi đo được** (§2) và một phán quyết. GO = vào chương trình thực nghiệm; CONDITIONAL = chỉ chạy khi cổng đạt; NO-GO = bị bằng chứng bác bỏ hoặc chưa đủ điều kiện. Số trong cột literature đã đối chiếu HTML thô [V] (chi tiết ở `literature/Q1–Q3`).

### 3.1 Target (cái được dự đoán)

| ID | Chiến lược | Literature | Nội bộ [D] | Phán quyết |
|---|---|---|---|---|
| **T** | **Center target theo pseudo-signer** (CMN trong không gian target) | S-JEPA centering là cơ chế sẵn có của PLctx; ContentVec huỷ thông tin speaker khỏi target teacher (bỏ thành phần: ABX across 6,32 → 6,92) [V] | CMN theo signer +2,2 so với whiten; CLS-cụm = oracle | **GO** (arm T1) |
| **D1** | Target **multi-depth nối** (khối 2, 4, 6), z-score từng tầng | Bootleg: nối 4 khối hơn trung bình kiểu data2vec, và target chỉ tầng cuối thoái hoá khi train dài (61,0 → 43,8 từ 100 → 1.200 epoch) [V]; EEG-JEPA +3,66 [V]; V-JEPA 2.1 SSv2 62,5 → 72,1 [V] | LDA bão hoà từ khối 2 (40,9 so với 40,1 cuối); PLctx đang dùng *trung bình* 4 tầng cuối | **GO** (D1), kỳ vọng nhỏ, rẻ |
| **D2** | Target **multi-span gộp** (4/8/16 bin + toàn clip) | GFP: truy hồi không train NTU-60 62,0 (MAMP) → 70,9 [V] | chuyển động: probe yếu, lỗi không bám tham số chuyển động | **GO** (D2) |
| **D3** | Target **hiệu thời gian** (dự đoán embedding hiệu của teacher, khoảng ≥ 3 bin) | MAMP +10,1 linear; MASA k=3 tốt nhất 49,06, k=1 47,64 (jitter) [V]; MotionJEPA/DISReg chống triệt tiêu đặc trưng nhanh [V] | cùng trên | **GO** (D3), cảnh báo jitter 2D |
| D4 | Dự đoán **khối tương lai** phụ (hỗn hợp mask) | OpenHands DPC 91,2 → 94,7; Human-JEPA block-mask làm hỏng re-ID 17 mAP, trộn hai họ mask cho đánh đổi đơn điệu [V]; V-JEPA từ đầu thích hai chiều (SSv2 67,4 so với 49,8 chỉ-nhân-quả) [V] | — | CONDITIONAL (chạy khi D1–D3 dương) |
| T-dis | Target rời rạc / cụm (HuBERT-style) | HuBERT PNMI 0,28 → 0,64 → 0,70 [V]; SHuBERT là tham chiếu, không đầu tư theo ghi chú của bạn | không đo purity ASL-LEX | CONDITIONAL, thấp nhất |
| T-iso | **Đẳng hướng** (SIGReg/VICReg) làm mục tiêu chính | LeJEPA cần nhiều view toàn cục, batch ≥ 256, SIGReg trên cùng embedding được căn chỉnh [V]; pilot nội bộ L1+SIGReg −13,8 [R] vì SIGReg đặt ở head phụ trên token thị giác (đọc `pj/train.py`) | whitening post-hoc bất ổn (+1,5…+6,7) | **NO-GO** làm mục tiêu chính; chỉ giữ như điều khoản đẳng hướng nhỏ trong R |

### 3.2 Predictor và teacher

| ID | Chiến lược | Literature | Nội bộ [D] | Phán quyết |
|---|---|---|---|---|
| **C-pred** | Predictor **có điều kiện theo mã signer** để encoder thôi mang signer (ContentVec) | bỏ conditioning: ABX 5,13 → 5,76 [V]; seq-JEPA: conditioning trên nhân tố không liên quan lớp làm giảm 87,4 → 79,3 [V] (dấu ngược: thêm thông tin vào encoder) | CLS **đã** đi vào predictor và token vẫn mang signer (láng giềng cùng signer 84 %) | CONDITIONAL (arm C1: mã cụm làm điều kiện AdaLN, chạy sau T/R) |
| P-IWM | Conditioning theo tham số augmentation (equivariance) | IWM: bất biến thắng linear (74,5 so với 67,5) [V] | retrieval của ta là cosine, cần bất biến | **NO-GO** |
| E-ema | Teacher EMA | CF-JEPA: encoder online hơn EMA +3,9 cho phân loại [V] | student ≈ teacher (18,44 so với 18,28) | giữ EMA; **không** phải đòn bẩy |
| E-anc | Teacher neo (bản sao đóng băng) cho giai đoạn tiếp tục | Human-JEPA: tiếp tục train ngây thơ sụp đổ trong khi loss giảm, neo ngăn sụp đổ [V] | tiếp tục 160→200 epoch phẳng (+0,23 / −0,05) [R] | CONDITIONAL (chỉ cho Phase 4 train dài) |
| E-canon | Teacher nhận input **chuẩn hoá hình thái giải tích**, student nhận input nhiễu | ContentVec cần voice converter thật; bộ chuyển đổi yếu **tệ hơn không có** (7,78 so với 6,92) [V]; AUTSL: ẩn danh thô mất ~16 điểm [V] | encoder đã gần bất biến với tỉ lệ ±15 %; R@1 không đổi | **NO-GO** |

### 3.3 Regularizer, positive chéo mẫu, readout, dữ liệu

| ID | Chiến lược | Literature | Nội bộ [D] | Phán quyết |
|---|---|---|---|---|
| **R** | **Căn chỉnh phân phối theo nhóm signer** trên embedding gộp (Deep CORAL, DICA, AdaBN: căn chỉnh thống kê bậc hai giữa miền) | Deep CORAL (1607.01719), AdaBN (1603.04779) [id đã kiểm]; nén không gian con speaker giảm ABX across 5–15 % tương đối trên speaker chưa thấy [V] | CORAL theo signer +9,1; không nhãn bằng cụm CLS; head tuyến tính quy nạp +3…+6 | **GO** (arm R1) |
| X1 | Tinh chỉnh **NN-contrastive** (STARS/MAE-CT/NNCLR) | STARS kNN 63,1 → 79,9; bỏ NN lookup còn 40,5; điều kiện xuất phát kNN ≥ 51–61 %; positive chung nhiễu thất bại [V] | precision tương hỗ 19–36 % < 40 %; head train làm giảm R@1 xuống 13–26 | **NO-GO** đến khi cổng H10 đạt |
| X2 | Contrast **trong clip** (negative cùng clip, ContentVec Eq. 2; Gulati WTN) | WTN theo kịp supervised dưới nhiễu cố định [V], nhưng đặc trưng giữ lâu (handshape) cũng gần bất biến theo clip | không đo | CONDITIONAL, thấp |
| X3 | **Adversarial** signer (GRL) | SRM −1,7 WER; subject-invariant EEG 88,6 → 73,0 vẫn còn rất cao; attacker hậu nghiệm 56,0 so với adversary 49,0 [V] | — | NO-GO làm method; chỉ chẩn đoán cận trên |
| Z1 | Readout **chỉ tay, bỏ CLS** | — | raw +2,4…+3,4; LDA không giảm | **GO** (miễn phí) |
| Z2 | **Whitening** nhãn-không | all-but-the-top, hubness [V] | +6,7 (val) nhưng +1,5 (CV signer khác) | báo cáo, **không coi là bền** |
| Z3 | CORAL theo cụm CLS ở **thời điểm truy vấn** | TTA | = oracle; transductive | báo cáo riêng, khai báo |
| Z4 | Readout **có thứ tự** (bins) / chamfer | SignRep ordered bins +5,9, chamfer +9,4 trên đặc trưng SignRep [R] | ordered 2/4 đoạn **kém hơn** | ordered NO-GO; chamfer/`mw` **chưa đo** (Phase 1) |
| Y1 | **Dữ liệu đa signer** (MS-ASL 222, WLASL 119 signer) | scale: không có đường cong frozen cho sign; BEST/STC fine-tune lõm +4…+14; SHuBERT 10× dữ liệu +2,3 BLEURT [V] | R@1 +≈2 mỗi lần gấp đôi số signer trong gallery | **GO** (phép đo độ dốc) |
| Y2 | Corpus lớn (SignVerse-2M, 3.207 h DWPose) | OpenHands 1.129 h pose không nhãn [V] | — | CONDITIONAL theo Y1 |

**Nhận xét tổng hợp.** Chiến lược mới chính (T, R) không xuất phát từ cuộc chạy đua "thêm arm" mà từ hai quan sát liên kết: (i) signer là biến đổi bậc hai (mean + cov) trong embedding, nên chuẩn hoá bậc hai theo nhóm là phép sửa đúng dạng; (ii) CLS đã mã hoá signer nên có nhóm không nhãn miễn phí. Các chiến lược "kinh điển" của họ JEPA (multi-depth, multi-span, hiệu thời gian) được giữ lại nhưng với kỳ vọng thấp hơn và gắn với một chế độ lỗi khác (thông tin chuyển động yếu), không phải với khoảng cách signer.

---

## 4. Giả thuyết đăng ký trước

Metric chính **VAL-DICT raw**: query val (10.304), gallery train (40.154), cosine, class-max, tie bi quan (đúng giao thức test). Kèm hàng "sau cùng một stack post-hoc" cho mọi arm. Ngưỡng hiệu ứng τ = max(1,5; 2,5 × SD_seed(A0)); CI bootstrap signer × gloss; nhất quán ≥ 4/5 signer; Holm trên họ so sánh retrieval. **Phải thắng A0 trên raw; hàng "hơn cả stack" chỉ được tuyên bố nếu Δ sau stack ≥ τ.**

| ID | Giả thuyết | Dự đoán định lượng [I] | Arm | Bác bỏ nếu | Control |
|---|---|---|---|---|---|
| **H1** | Center target theo pseudo-signer làm encoder bớt mang signer | tỉ lệ láng giềng cùng signer giảm ≥ 5 điểm (84 % → ≤ 79 %) và raw R@1 ≥ A0 + τ; LDA-96 không giảm quá 2 | T1 vs A0 | Δ raw < 0,5 hoặc CI chứa 0 | T1-shuffle (nhãn cụm xáo trộn): phải ≈ A0 |
| **H2** | Regularizer căn chỉnh hiệp phương sai theo cụm tăng retrieval raw | raw R@1 ≥ A0 + max(τ; 3) (head tuyến tính quy nạp đã cho +3…+6 so với whiten); LDA-96 không giảm quá 2 | R1 vs A0; R1 vs R0 | R1 ≤ R0 (lợi ích chỉ do đẳng hướng) | **R0** (chỉ đẳng hướng); R1-shuffle |
| **H3** | Cụm CLS không nhãn tương đương signer thật | \|R@1(CLS) − R@1(meta)\| ≤ 1 cho T và R | T1 vs T1m; R1 vs R1m | gap > 2 | — |
| **H4** | Gói động học (D1+D2+D3) làm tăng thông tin chuyển động | trung bình 4 tham số {Movement, Repeated, FlexChange, WristTwist} của probe +3 điểm (CI loại trừ 0); raw R@1 ≥ A0 + 1,5 | DYN vs A0 | Δ probe ≤ 0 | A0 + cùng số tham số (head thừa) |
| **H5** | Target multi-depth ngăn thoái hoá khi train dài | VAL-DICT ở 160 epoch ≥ ở 40 epoch − 0,3; không thoái hoá rank | D1 ở Phase 4 | thoái hoá ≥ 1 điểm | A0 train dài (đã có: phẳng) |
| **H6** | Mặt mang signer, ít mang gloss | readout chỉ tay: raw +1,5 và LDA-96 không giảm; đã thấy +2,4 trên checkpoint hiện có | H1-only (loss chỉ tay) | không tăng | — |
| **H7** | Thêm signer vào corpus tiền huấn luyện tăng retrieval | độ dốc ≥ +1,5 R@1 mỗi lần gấp đôi số signer (25/50/100 % signer của ASLC; rồi + WLASL + MS-ASL) | Phase 4 | độ dốc < 0,5 | kiểm soát số bước |
| **H8** | SA (T+R) và DYN cộng được | tương tác trong ±1 | SA+DYN vs tổng | \|tương tác\| > 1,5 | — |
| **H9** | Sau SA-JEPA, post-hoc ít còn tác dụng | Δ whiten sau SA < 2; Δ CORAL-CLS (TTA) sau SA < 3 | stack trên mọi arm | Δ TTA ≥ 3 (alignment chưa được nội hoá) | — |
| **H10** | SA-JEPA nâng precision NN xuyên signer qua cổng | precision tương hỗ (cụm CLS loại trừ) ≥ 40 % | đo trên TR | < 40 % | — |

---

## 5. Method: SA-JEPA

### 5.1 Nguyên tắc

- **Fork từ `method/phonojepa`** (cùng token 7 kênh × 32 bin, d = 256, 6 khối, predictor d = 128 × 4, mask PLCM-lite, EMA 0,996 → 0,9999, centered-CE với τ_t 0,04 → 0,06, τ_s 0,1, AdamW 3e-4). Mỗi arm là một diff nhỏ so với A0, cùng optimizer, cùng lịch, cùng evaluator, cùng readout.
- **Chỉ thêm thứ mà chẩn đoán chỉ ra là thiếu.** Không thêm tham số cho nhiễu mà chẩn đoán cho thấy encoder đã bất biến (kích thước mặt, tỉ lệ ±15 %).
- **Method chính không dùng nhãn gloss và không dùng signer id.** Signer id chỉ ở các arm có hậu tố `m` (diagnostic) để đo khoảng cách tới nhóm không nhãn.

### 5.2 Mô-đun C: pseudo-signer từ CLS

- Nguồn: CLS của **teacher** (L2-chuẩn hoá, 256-d), ổn định hơn student. Giữ ngân hàng bộ nhớ 16.384 clip gần nhất (mỗi clip một CLS từ view gần).
- Cứ mỗi epoch: k-means cosine K = 48 trên ngân hàng, **khởi tạo từ tâm cụm lần trước** để id ổn định giữa các lần làm mới; gán id theo tâm gần nhất, **tính theo clip** (trung bình CLS teacher của hai view gần và xa của clip đó) để id không đổi giữa các cửa sổ. K = 48 vì K = 35 cho AMI 0,71 / purity 0,79 trên train và CORAL tốt nhất ở K nhóm gần số signer (140 làm giảm: 29,4 so với 32,3); quét K ∈ {24, 96} ở arm phụ.
- Giám sát (dùng nhãn **chỉ để đo**, không vào loss): AMI(cụm, signer) và NMI(cụm, gloss). Cảnh báo dừng nếu NMI(cụm, gloss) > 0,2 hoặc > NMI(cụm, signer): cụm đang gom theo gloss, alignment sẽ xoá thông tin từ vựng.
- Rủi ro vòng phản hồi (cụm ảnh hưởng loss, loss ảnh hưởng CLS): dùng teacher, stop-gradient, làm mới chậm, và arm kiểm soát nhãn cụm xáo trộn.

### 5.3 Mô-đun T: center target theo cụm

PLctx: `p_t = softmax((t − c) / τ_t)` với `c` là một vector D chiều (EMA β = 0,9 của trung bình teacher). SA-JEPA: bảng `C[K, D]`, với clip thuộc cụm k dùng `c_eff,k = λ_k · C[k] + (1 − λ_k) · c_global`, `λ_k = n_k / (n_k + 64)` (n_k = số mẫu đã thấy trong cụm, co về center toàn cục khi cụm thưa). Cập nhật EMA β = 0,9 chỉ cho cụm có mặt trong batch. Khi K = 1 trở về PLctx đúng. Đây là CMN trong không gian target: trừ độ lệch trung bình theo nhóm khỏi chính cái mà student phải đoán, nên encoder không được thưởng khi mang độ lệch signer.

### 5.4 Mô-đun R: căn chỉnh phân phối theo cụm trên embedding gộp

- **Biểu diễn được căn chỉnh:** `u = concat(group-mean của đầu ra student trên token nhìn thấy)` theo ba nhóm (tay trội, tay phụ, mặt) → 768-d, đúng đối tượng retrieval dùng (hoặc 512-d chỉ tay khi arm H bật). Mask độc lập nội dung nên thống kê giữa các cụm vẫn so sánh được.
- **Thống kê:** giữ EMA theo cụm của trung bình μ_k và hiệp phương sai Σ_k (768² × 48 ≈ 28 triệu số thực), cộng toàn cục μ, Σ. Trong batch dùng ước lượng hỗn hợp `(1 − α)·EMA(stop-grad) + α·batch(có gradient)`, α = 0,25. Batch chỉ có 32 clip × 2 view nên **bắt buộc dùng ngân hàng thống kê**.
- **Loss:**
  `L_R = Σ_k w_k [ a_μ ‖μ̂_k − μ̂‖² + a_Σ ‖Σ̂_k − Σ̂‖_F² / d ] + a_iso ‖Σ̂ − I‖_F² / d`,
  `w_k ∝ n_k`. Mặc định a_Σ = 1, a_μ = 0,25 (T đã lo trung bình), a_iso = 0,25. Chẩn đoán tuyến tính cho thấy hiệp phương sai là phần chính (cov-only 29,3 > mean+cov 28,1 > mean-only 25,8). Trọng số tổng λ_R đi qua bộ cân bằng của PLctx (mỗi term chia cho trung bình chạy của nó, đóng băng sau warm-up) và khởi động tuyến tính trong 10 % đầu, mục tiêu L_R chiếm 10–20 % loss.
- **Điều khoản đẳng hướng** nhỏ vì chỉ cần chống suy biến do alignment (không phải để thay whitening). Nó đặt trên chính embedding được căn chỉnh, đúng điều kiện LeJEPA (cùng embedding), khác pilot SIGReg cũ.
- **Không dùng CLS làm đối tượng alignment** (G1 −5,7 [R]; CLS là chữ ký signer): CLS chỉ cung cấp nhóm.

### 5.5 Mô-đun D: ép động học

| ID | Nội dung | Chi tiết |
|---|---|---|
| D1 | multi-depth | Target teacher = nối đầu ra khối {2, 4, 6} sau LayerNorm riêng từng tầng; predictor ra 3 đầu (mỗi đầu CE như cũ), loss trung bình. Thay cho trung bình 4 tầng cuối đang dùng. |
| D2 | multi-span | Với mỗi khối mask (4–8 bin), thêm một query token cho (kênh, khối) dự đoán **trung bình teacher** trên khối; thêm query cho trung bình toàn clip theo kênh. Spans 4/8/16 bin + toàn cục. |
| D3 | hiệu thời gian | Dự đoán `LN(h_{t+k}) − LN(h_t)` của teacher từ ngữ cảnh student tại (t, t+k), k = 3 bin (0,24 s ở 25 fps; MASA: k=3 > k=1 vì jitter), chỉ tại bin hợp lệ cả hai đầu; trọng số 0,5. |
| D4 | khối tương lai | Xác suất 0,3 mỗi view thay mask gần bằng "che 25 % bin cuối". Chỉ chạy khi D1–D3 dương. |

Mọi target chỉ phụ thuộc teacher (thấy toàn cảnh), student chỉ thấy ngữ cảnh: không rò rỉ. Test rò rỉ 1e6 của PLctx áp lại cho mọi đầu mới.

### 5.6 Readout và stack (cho mọi arm, như nhau)

- `hands`: trung bình theo kênh của bốn kênh tay (1.024-d), **không dùng mặt và CLS**. Báo cáo song song `grp` (baseline giao thức).
- `+whiten`: PCA-512 fit trên embedding **của corpus tiền huấn luyện** (không nhãn, không dùng gallery nhãn); báo cáo nhưng không coi là bền.
- `+TTA-CLS-CORAL`: CORAL theo cụm k-means của CLS trên query (K ≈ số signer ước lượng) và trên gallery; **transductive, báo cáo ở hàng riêng**, không so trực tiếp với SignRep raw.
- Hàng tham chiếu LDA-96 (trần tuyến tính, dùng nhãn train) để theo dõi "khoảng cách thông tin so với hình học" cho mọi arm.

### 5.7 Hiện thực và chi phí

- Package mới `method/sajepa` (fork `phonojepa`); móc: `losses.CenteredCE` (center theo cụm), `train.py` (ngân hàng CLS, làm mới cụm, ngân hàng thống kê), `model.py` (đầu đa tầng, query pooled).
- Unit test: (1) K = 1 ⇒ trùng PLctx từng bit; (2) cụm giống nhau ⇒ L_R = 0; (3) gradient chỉ qua phần ước lượng batch; (4) id cụm ổn định khi làm mới với tâm khởi tạo; (5) target D1/D2/D3 khớp hình dạng và chỉ dùng bin hợp lệ; (6) test rò rỉ 1e6 sau toàn bộ pipeline.
- Chi phí: ngân hàng CLS và thống kê nhỏ; L_R là phép nhân 768 × 768 theo cụm xuất hiện trong batch (≤ 32); D1–D3 thêm vài query token. Ước tính 1,1–1,3× PLctx (1,3–1,6 GPU-h cho 40 epoch).

---

## 6. Các arm và chương trình thực nghiệm

### 6.1 Arm (screening 40 epoch, 1 seed trừ A0)

| Arm | Mô tả | Giả thuyết | Vai trò |
|---|---|---|---|
| **A0** | PLctx REF, 40 epoch, **3 seed** | — | Anchor, SD seed |
| **B\*** | A0 + readout `hands` + stack (không train) | — | Đối thủ bắt buộc phải thắng ở hàng "hơn stack" |
| **T1** | T với cụm CLS K = 48 | H1 | chính |
| **R1** | R (cụm CLS, a_Σ = 1, a_μ = 0,25, a_iso = 0,25) | H2 | chính |
| **R0** | chỉ đẳng hướng (a_Σ = a_μ = 0, a_iso = 0,25) | H2 control | kiểm soát |
| **TR** | T1 + R1 = **SA-JEPA** | H1, H2 | chính |
| **TR-shuf** | TR với nhãn cụm xáo trộn mỗi epoch | control | tách "alignment thật" khỏi "thêm regularizer" |
| **T1m, R1m** | T và R với signer id thật | H3 | cận trên (khai báo) |
| **H-only** | loss + readout chỉ trên kênh tay | H6 | |
| **D1, D2, D3** | từng mô-đun động học | H4 | |
| **DYN** | D1 + D2 + D3 | H4 | |
| **SA+DYN** | TR + DYN | H8 | |
| Phụ | TR với K = 24, 96; D4; C1 (predictor có điều kiện mã cụm) | | nếu còn ngân sách |

### 6.2 Phase

| Phase | Nội dung | GPU-h |
|---|---|---:|
| **0** | Chẩn đoán CPU (đã xong, §2) | 0 |
| **1** | Chuẩn hoá B\*: trích embedding A0 cho stack; đo `mw` (SignRep +12,5) và chamfer trên PLctx; (tuỳ chọn công bằng) trích đặc trưng SignRep 40k clip để áp cùng stack | 2 (+6–10) |
| **2** | A0 ×3 và ~13 arm ở §6.1; chấm VAL-DICT raw + stack + probe ASL-LEX + metric nhiễu ở epoch 10/20/40 | 24–30 |
| **3** | 1–2 cấu hình tốt nhất ×3 seed + A0 (đã có); tổ hợp 2×2 nếu hai họ cùng KEEP | 12–15 |
| **4** | Trục dữ liệu: ASLC 25/50/100 % signer, rồi + WLASL + MS-ASL (lọc nhiễm chéo theo D1); full160 cấu hình cuối; **test một lần** (ASL Citizen đóng băng + WLASL2000) | 14–18 |
| **5** | Downstream: ISLR (linear, fine-tune, few-shot), WLASL, MS-ASL local, độ bền fps/frame drop (như P16 §9.7) | 6–10 |
| **Tổng** | | **≈ 58–75** (+6–10 nếu trích đặc trưng SignRep để áp cùng stack) |

Bản tiết kiệm (bỏ D4, K phụ, R1m, T1m, H-only, C1; 1 seed ở Phase 3; bỏ trục dữ liệu mở rộng): ≈ 30 GPU-h.

### 6.3 Luật quyết định (kế thừa `docs/research` §6.6, sửa cho proposal này)

- **KEEP:** Δ raw ≥ τ; CI bootstrap signer × gloss loại trừ 0; ≥ 4/5 signer cải thiện; LDA-96 không giảm quá 2 điểm; NMI(cụm, gloss) ≤ 0,2.
- **KEEP-MECH:** Δ raw trong ±τ nhưng cơ chế cải thiện (láng giềng cùng signer −5; probe chuyển động +3; khoảng whiten−raw co lại). Giữ để tổ hợp.
- **DROP:** Δ ≤ −1,5 hoặc TOST tương đương ±1.
- **Bắt buộc đối chiếu:** (i) R1 phải hơn R0 và TR phải hơn TR-shuf ít nhất 1 điểm, nếu không lợi ích là thêm regularizer chứ không phải alignment; (ii) so với B\* (hàng "hơn stack"); (iii) với T/R phải báo cả hàng cụm CLS và hàng signer id.
- 1 seed ở Phase 2 chỉ cho ứng viên; chỉ Phase 3 (3 seed) cho kết luận. Holm trên họ so sánh retrieval {H1, H2, H4, H6}.
- **Kiểm tra proxy:** Kendall τ giữa xếp hạng arm ở epoch 40 và epoch cuối khi có điểm trung gian; nếu không đo được thì ghi rõ giả định.

### 6.4 Cổng và kế hoạch dự phòng

| Cổng | Điều kiện | Nếu đạt | Nếu không |
|---|---|---|---|
| G-SA (sau T1, R1, R0, TR, TR-shuf) | TR ≥ A0 + τ và TR ≥ TR-shuf + 1 | giữ họ SA; Phase 3 | **Plan B** |
| G-DYN | DYN: probe chuyển động +3 và raw ≥ +1,5 | tổ hợp SA+DYN | bỏ DYN; giữ D1 nếu H5 đạt |
| G-NN (H10) | precision tương hỗ ≥ 40 % trên TR | mở O8 (NN-tuning) ở Phase 4 | giữ NO-GO |
| G-DATA (H7) | độ dốc ≥ +1,5 / lần gấp đôi signer | mua thêm dữ liệu (Y2) | dừng mở rộng |

**Plan B (nếu SA thất bại):** (1) hệ thống cuối = A0 + `hands` + stack, khai báo là method post-hoc; (2) theo đuổi DYN và trục dữ liệu; (3) arm cận trên có signer id (adversarial/DANN) chỉ để đo; (4) quay lại `docs/research` O4 (teacher RGB đóng băng) nếu mọi nhánh label-free < +τ.

---

### 6.5 Kỳ vọng thực tế và cách báo cáo

- **Giá trị tham chiếu (val, raw):** A0 22,7; oracle thang: whiten 29,4 (mong manh), CORAL cụm không nhãn 34,6 (transductive), CORAL signer thật 38,5, LDA-96 50,9 (dùng nhãn). Tỉ lệ test/val của A0 là 16,68/22,72 ≈ 0,73.
- **Dự báo [I], chưa kiểm:** nếu T+R nội hoá được phân nửa đến hai phần ba lợi ích của head tuyến tính quy nạp (+6 so với whiten trên val), SA-JEPA đạt khoảng val 30–34 trên raw; ngoại suy sang test khoảng 22–25. DYN cộng thêm khoảng 1–2. Đây vẫn **thấp xa** số SignRep raw (49,0 đo local), và không có chẩn đoán nào ở đây cho thấy khoảng cách đó đóng được chỉ bằng JEPA skeleton không nhãn; LDA 50,9 trên val dùng nhãn gloss nên không so được.
- **Cách báo cáo:** mỗi arm báo ba hàng: raw (protocol), + stack quy nạp (hands, whiten), + TTA-CLS-CORAL (transductive, khai báo); kèm LDA-96, tỉ lệ láng giềng cùng signer, probe 9 tham số. Test chỉ chấm một lần cho cấu hình cuối; WLASL2000 cùng checkpoint.
- **Điều có thể thay đổi kết luận:** G-SA không đạt (khi đó SA bị loại và Plan B); độ dốc dữ liệu (H7) lớn hơn dự kiến (khi đó trục dữ liệu quan trọng hơn mọi mô-đun mục tiêu); hoặc SignRep + cùng stack thu hẹp khoảng cách (Phase 1 tuỳ chọn).

---

## 7. Rủi ro và cách giảm

| Rủi ro | Mức | Cách giảm |
|---|---|---|
| Cụm CLS gom theo gloss, alignment xoá thông tin từ vựng | cao | NMI(cụm, gloss) giám sát; LDA-96 làm bảo vệ; TR-shuf; K nhỏ, làm mới chậm |
| Vòng phản hồi CLS ↔ cụm ↔ loss | trung bình | dùng teacher, stop-grad, tâm khởi tạo từ lần trước |
| Thống kê theo cụm nhiễu vì batch chỉ 32 clip | trung bình | ngân hàng EMA + co về toàn cục (λ_k) |
| Regularizer hoặc điều khoản đẳng hướng làm suy biến như pilot SIGReg cũ (−13,8) | trung bình | điều khoản nhỏ (0,25), đặt trên embedding được căn chỉnh, R0 làm control, theo dõi rank |
| Val có 5 cụm signer; CI rộng | cao | chỉ tin Δ có cặp, nhất quán theo signer, CV signer khác; Phase 3 ba seed |
| Lợi ích chỉ xuất hiện ở hàng transductive | trung bình | báo cáo raw và TTA riêng, quyết định KEEP theo raw |
| D3 nhiễu do jitter 2D | trung bình | k ≥ 3 bin, trọng số 0,5, làm mượt phía teacher |
| Xếp hạng 40 epoch không dự báo 160 epoch | trung bình | điểm trung gian, Kendall τ, Phase 3 xác nhận |
| 1 bit tay trội (canonical) rò rỉ như PLctx | thấp | đã khai báo ở PLctx; giữ nguyên |
| Cạnh tranh GPU với job PhonoPath full160 | thực tế | chỉ chạy khi bạn duyệt và GPU còn dư; một job mỗi lúc |

---

## 8. Danh sách không nên làm (bị chẩn đoán bác bỏ hoặc chưa đủ điều kiện)

1. **Retarget hình thái giải tích / teacher chuẩn hoá hình thái** (V1, E-canon): encoder đã bất biến với tỉ lệ ±15 %, R@1 không đổi (§2.5); literature cũng cảnh báo bộ biến đổi yếu tệ hơn không có.
2. **NN-contrastive tuning** (O8, STARS): precision 10–36 % < 40 %, head train làm R@1 xuống 13–26 (§2.8).
3. **Isotropy/SIGReg làm mục tiêu chính:** whitening post-hoc cho +6,7 trên val nhưng +1,5 trên signer khác; lợi ích bền nằm ở alignment theo signer.
4. **Alignment trên CLS hoặc dùng CLS làm readout:** CLS là chữ ký signer (R@1 0,25).
5. **Readout có thứ tự đơn giản:** kém hơn `grp` (§2.6). (Chamfer và `mw` chưa đo, không bác bỏ.)
6. **Conditioning predictor trên nhân tố gây equivariance** (IWM, seq-JEPA).
7. **Mở rộng tầng/teacher/student:** không phải đòn bẩy (§2.6).
8. **Kết luận rằng "JEPA khuếch đại signer":** không được ủng hộ ở mức phương sai (0,27 → 0,14).

---

## 9. Nhật ký rà soát

(Chưa có. Cần một vòng review độc lập trước khi triển khai, tập trung vào: (a) tính đúng của các script chẩn đoán và các phép fit không rò rỉ nhãn val; (b) các bước suy luận từ chẩn đoán sang chiến lược; (c) tính khả thi và rò rỉ của R và T; (d) con số trong tài liệu so với JSON.)

---

## 10. Nguồn

**Chẩn đoán nội bộ (`docs/proposal17/diagnostics/`):** `common.py` (tải embedding, giao thức, bootstrap), `geo_headroom.py`, `geo_ladder.py`, `ladder_cv.py`, `nuisance_decomp.py`, `signer_norm.py`, `emb_signer_cluster.py`, `eclus_control.py`, `cls_cluster.py`, `pseudo_signer.py`, `local_norm.py`, `coral_head.py`, `coral_head2.py`, `nn_head.py`, `style_vars.py`, `nuis_pc.py`, `morph_effect.py`, `emb_variants.py`, `analyze_emb.py`, `ceilings.py`, `channel_analysis.py`, `inp_features.py`, `lex_probe_emb.py`, `error_attrib.py`, `subset_check.py`; kết quả JSON ở `diagnostics/results/`.

**Literature (verified HTML thô):** `literature/Q1_jepa_strategy_zoo.md`, `Q2_nuisance_disentanglement_and_retrieval_geometry.md`, `Q3_sign_ssl_scale_temporal_and_units.md`. arXiv chính: Littwin 2407.03475; Sobal 2211.10831; Gulati–Nemenman 2606.07770; MotionJEPA 2609.23881; ContentVec 2204.09224; Komatsu 2409.10103; Spin 2305.11072; IWM 2403.00504; seq-JEPA 2505.03176; Bootleg 2603.15553; V-JEPA 2.1 2603.14482; EEG-JEPA 2608.00114; WavJEPA 2509.23238; CF-JEPA 2606.07031; Human-JEPA 2608.21160; LeJEPA 2511.08544; LeWM 2603.19312; MAMP 2308.07092; GFP 2509.03609; MASA 2405.20666; STARS 2407.10935; MAE-CT 2304.10520; NNCLR 2104.14548; CAP 2012.10674; SSPS 2505.14561; Liu et al. 2305.12464; LEACE 2306.03819; Gonen–Goldberg 1903.03862; SignRep 2503.08529; SignSeek 2609.03695; ASL Citizen 2304.05934; OpenHands 2110.05877; SSL-SLR 2509.05188; BEST 2302.05075; SHuBERT 2411.16765 (tham chiếu). Miền: Deep CORAL 1607.01719; AdaBN 1603.04779; MixStyle 2104.02008 (id đã kiểm bằng curl; không trích số).

**Tài liệu nội bộ trước đó:** `docs/proposal12` (PLctx), `docs/research` (phễu, VAL-DICT, luật quyết định), `docs/proposal16` (PhonoPath-JEPA, chẩn đoán D-LEX, audit dữ liệu D1).

## Phụ lục A: tái lập

```bash
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
cd /home/haipd/Sign_Representations/docs/proposal17/diagnostics
OMP_NUM_THREADS=4 $PY geo_headroom.py results/geo_headroom.json       # thang cơ bản, đường cong gallery
OMP_NUM_THREADS=4 $PY geo_ladder.py                                   # bậc thang có CI và theo signer
OMP_NUM_THREADS=4 $PY ladder_cv.py                                    # lập lại trên 5 tập signer khác
./run_extract.sh                                                      # ~35 phút: teacher/student/ngẫu nhiên, tập con
$PY analyze_emb.py && $PY ceilings.py && $PY channel_analysis.py
$PY lex_probe_emb.py && $PY morph_effect.py && $PY inp_features.py
$PY signer_norm.py && $PY cls_cluster.py && $PY coral_head.py && $PY coral_head2.py
$PY nuisance_decomp.py && $PY style_vars.py && $PY nuis_pc.py && $PY error_attrib.py && $PY nn_head.py
```

Embedding trích lại nằm ở scratchpad (`.../scratchpad/emb`), không lưu vào repo (vài GB). ASL-LEX: `docs/proposal16/data/` (CC BY-NC, không commit).
