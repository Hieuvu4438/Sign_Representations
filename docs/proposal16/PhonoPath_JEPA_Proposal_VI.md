# Proposal 16 — PhonoPath-JEPA

## Học biểu diễn ngôn ngữ ký hiệu tự giám sát từ skeleton, với token path-signature theo kênh âm vị học

Ngày: 2026-10-16. Trạng thái: **đề xuất method, chưa triển khai, chưa train**. Chỉ có các chẩn đoán CPU không cần train (§5), chạy trên dữ liệu có sẵn trên server.

Tài liệu đi kèm trong `docs/proposal16/`:

| File | Nội dung |
|---|---|
| `literature/P0_path_jepa_close_reading.md` | Đọc kỹ toàn văn Path-JEPA (ECCV 2026): cái gì thật sự là target, ablation, điểm mơ hồ, những gì mang sang sign được |
| `literature/P1_path_signatures.md` | Toán signature cho thiết kế (Chen, lead-lag, nhiễu, chuẩn hoá); signature trong gesture/sign; phần mềm |
| `literature/P2_jepa_skeleton_update.md` | Cập nhật JEPA và skeleton SSL 2024–10/2026 (delta so với L1–L4, A–C); 12 nguyên tắc N1–N12 |
| `literature/P3_sign_domain_signals.md` | Âm vị học ASL → tín hiệu skeleton → mức signature; ASL-LEX 2.0; thẻ dataset; xếp hạng tín hiệu |
| `literature/D1_server_dataset_audit.md` | Audit dataset trên server: định dạng keypoint, kích thước, signer, nhiễm chéo |
| `diagnostics/` | Script và kết quả chẩn đoán không cần train (D-TEMPO, D-SIG, D-LEX), xem `diagnostics/README.md` |
| `data/signdata*.csv` | ASL-LEX 2.0, tải từ OSF (dự án `zpha4`) để probe âm vị học. CC BY-NC; không đưa vào git |

Quy ước nhãn bằng chứng:
- **[P]**: paper nói rõ.
- **[I]**: kết quả nội bộ đã có trong repo.
- **[D]**: chẩn đoán mới chạy cho proposal này (§5).
- **[S]**: suy luận hoặc thiết kế của ta.

---

## 0. Tóm tắt điều hành

**Bài toán.** Học encoder skeleton (COCO-WholeBody 2D) không dùng nhãn gloss. Downstream:
- dictionary retrieval đóng băng trên ASL Citizen (đúng protocol SignRep), cross-dataset WLASL2000;
- ISLR: linear probe, fine-tune, few-shot trên ASL Citizen, WLASL, MS-ASL.

**Hiện trạng [I].** Model skeleton tốt nhất của ta là PLctx full160 (PhonoJEPA): test ASL Citizen R@1 = **16.68**. SignRep (RGB, pretrain YouTube-SL-25) đạt 49.02 khi đo local (activity-weighted). Mọi model skeleton của ta đều có ba điểm mù:
- **movement** không giải mã tuyến tính được (probe R² < 0);
- non-manual cũng không;
- danh tính signer đọc được tới 93.7 %.

**Vì sao Path-JEPA.** Path-JEPA (ECCV 2026) đưa skeleton qua **token path-signature đa thang** trên joint / edge / chain. Sau đó nó học bằng **JEPA target latent** (teacher EMA), với **masking đóng theo phụ thuộc** [P]. Path signature là cách mô tả chuyển động có cơ sở toán học:
- level 1 = độ dời;
- level 2 = diện tích Lévy, mã hoá độ cong, chiều quay và lead/lag giữa các kênh;
- bất biến với reparameterization, tức bất biến tempo.

Movement đúng là chỗ ta thiếu. Nó cũng là trục đối lập từ vựng lớn nhất của ASL: **7.510 minimal pair chỉ khác movement**, so với 3.048 cặp khác handshape và 1.886 cặp khác location (đếm trên ASL-LEX 2.0, P3 §1.2).

**Điều ta chưa từng thử.**
- Các lần dùng signature trước đây (S2, SignPathLab) đều dùng nó làm **target tường minh** và đều âm hoặc vô hiệu [I]. Path-JEPA *không* làm vậy: target của nó là latent của teacher; signature chỉ là **tokenizer** (P0 §1.2).
- Dạng "signature làm token + target latent + closure masking" **chưa bao giờ chạy** trong repo.
- Trong phạm vi tìm kiếm của P1 §2.3, chưa thấy công trình nào dùng signature cho SSL trên sign language.

**Phát hiện mới từ chẩn đoán không cần train (§5) [D].**
1. **Tempo tuyệt đối là hỗn hợp, trong đó nhiễu signer trội hơn.**
   - Signer giải thích **73 %** phương sai log thời lượng clip, và 34 % thời lượng ký thật (bỏ frame nghỉ).
   - Gloss vẫn giải thích khoảng 10–17 % phương sai *trong cùng signer* (§5.2, đã sửa sau review).
   → Bỏ thời gian **tuyệt đối**, giữ thời gian **tương đối**: cửa sổ theo giây, vị trí trong đoạn hoạt động.
2. **Gần một nửa frame là nghỉ.** Frame "hoạt động" trung vị chỉ chiếm 53 % clip. Signature của đoạn đứng yên bằng 0, nên token signature của đoạn nghỉ không mang nội dung, đúng như ta muốn. Activity weighting đóng góp +12.5 R@1 cho SignRep [P]. Nhưng động tác nâng/hạ tay vào/ra tư thế nghỉ lại là path *lớn nhất* clip. Vì vậy readout có trọng số phải dựa vào **độ cao tay so với mức nghỉ**, không dựa vào năng lượng quỹ đạo (§7.8).
3. **Signature không phải "feature tốt hơn sẵn có" (kết quả âm, báo cáo đầy đủ).**
   - Làm descriptor không học, nó *không* vượt toạ độ theo bin: VAL-DICT R@1 3.41 so với 4.02.
   - Probe ASL-LEX **rời gloss và rời signer** (D-LEX v3): signature thuần thua mọi baseline. Ghép vào vị trí trung bình chỉ giúp Contact (+3.7) và WristTwist (+2.2), còn làm hại handshape do pha loãng.
   - Feature chuyển động tuyến tính mạnh nhất hoá ra là **độ lệch chuẩn vị trí** (biên độ), tức moment đối xứng bậc hai. Nó *không* phải Lévy area.
   - Lợi thế "ít signer" **không** đứng vững khi so ở cùng mức hữu ích (§5.3, đã sửa sau review). Signature cần khử nhiễu: làm mượt cho +26 % R@1 tương đối.
   → Ủng hộ thiết kế **lai**: giữ token trạng thái của PLctx làm nền. Coi luồng signature là **inductive bias để học movement và quan hệ**, *không* phải nguồn thông tin mới. Control A2 dùng moment đối xứng cùng chiều. Kiểm tra rẻ trước bằng cổng có giám sát G0 (§9.1) rồi mới tốn GPU cho SSL; có Plan B (§9.9) nếu thất bại.

**Đề xuất: PhonoPath-JEPA** = PLctx (anchor đã kiểm chứng) + **luồng token path-signature theo kênh âm vị học**, với sáu thay đổi so với Path-JEPA gốc để hợp với sign:

| # | Thành phần | Khác Path-JEPA ở đâu |
|---|---|---|
| C1 | **Họ path âm vị học.** Mỗi tay: quỹ đạo cổ tay, chain cánh tay, chain ngón (frame bàn tay), vector hướng bàn tay (wrist twist), aperture. **Kênh REL** gom mọi path khác kênh: tay–mặt, tay–tay, hai tay tổng/hiệu sau khi gương hoá. Mặt là tuỳ chọn | Path-JEPA chỉ có joint/edge/chain trên cây động học |
| C2 | **Token lai trạng thái + chuyển động.** Giữ 7 × 32 token tĩnh của PLctx; thêm token signature ở 2 thang (0.32 s, 0.64 s), có basepoint và độ tin cậy | Path-JEPA chỉ có token signature |
| C3 | **Global view nhất quán Chen**: student chỉ tính global từ các đoạn nhìn thấy | Global của Path-JEPA rò rỉ nếu có mask thời gian (P1 §4.7) |
| C4 | **Closure trên support (khớp × mẫu)**: mask thời gian + kênh (PLCM-lite) + articulator subtree | Path-JEPA chỉ che theo khớp, ở mọi thời điểm |
| C5 | **Chống nhiễu cho level 2**: làm mượt tay, chuẩn hoá robust + asinh, gating theo observation filter | Path-JEPA trên 3D NTU sạch; tay 2D của ta có score thấp ở 42–48 % frame (D1) |
| C6 | **Readout có trọng số hoạt động** theo độ cao tay so với mức nghỉ, cộng năng lượng CONF; cùng readout theo nhóm của PLctx | Path-JEPA dùng linear probe, không retrieval |

**Thực nghiệm** (§9): phễu 5 phase có cổng.
- Trước hết là **cổng G0**: pre-screen có giám sát khoảng 1–2 GPU-h, xem token signature có giúp ngay cả khi mô hình được dùng nhãn hay không. Nếu không, chuyển sang Plan B (§9.9).
- Thiết kế screening **một-yếu-tố-mỗi-lần** quanh A1 (không phải factorial đầy đủ): khoảng 10 arm, 1 seed, toàn bộ 40k clip train, khoảng 40 epoch. A0 chạy 3 seed để biết nhiễu seed (§9.3). Mỗi arm gắn với một giả thuyết đăng ký trước (H1–H9, §6), có control khớp token: moment bậc hai đối xứng thay cho Lévy area.
- Sau đó xác nhận 3 seed, chạy budget đầy đủ, chấm test **một lần**.
- Ngân sách khoảng **55–75 GPU-h** trên một GPU dùng chung, chạy tuần tự.

**Tiêu chí thành công.**

| Mức | Điều kiện |
|---|---|
| Tối thiểu | Mechanism: probe ASL-LEX rời gloss + rời signer cho repeated / wrist-twist / contact tăng có ý nghĩa so với REF, và retrieval không giảm |
| Chính | VAL-DICT R@1 ≥ REF + τ, với τ = max(1.5, 2.5 × SD seed); CI bootstrap signer × gloss loại trừ 0. Trên test, vượt 16.68 của PLctx full160 với budget đầy đủ |
| Mục tiêu xa | Đóng một phần khoảng cách tới SignRep. Khoảng cách này còn phụ thuộc scale dữ liệu (Phase 4) |

---

## 1. Bài toán, phạm vi, tiêu chí

### 1.1 Định nghĩa

- **Input.** Chuỗi keypoint COCO-WholeBody 133 điểm, 2D. Đây là output rtmlib RTMW-x, có sẵn cho mọi dataset trên server (D1 §3).
- **Pretrain.** Không dùng nhãn gloss/text. Được dùng signer ID? Không: signer ID chỉ dùng để chia split và để đánh giá.
- **Encoder output.** Embedding cấp clip, cộng với chuỗi token cấp thời gian cho tác vụ dày (CSLR/SLT là hướng mở rộng, không đánh giá chính).

### 1.2 Downstream và metric

| Tác vụ | Dataset | Protocol | Metric |
|---|---|---|---|
| Sign video retrieval (chính) | ASL Citizen | Frozen dictionary: gallery = train 40.154, query = test 32.941, 2.731 gloss, cosine, class-max, ties bi quan (giống `phonojepa/pj/frozen_dictionary.py`) | R@1/5/10, MRR, DCG, macro R@1 |
| Retrieval dev | ASL Citizen | **VAL-DICT**: gallery train, query val 10.304. Mô phỏng đúng protocol test (thiết kế ở `docs/research`) | R@1 + CI bootstrap theo signer |
| Retrieval chéo dataset | WLASL2000 | Gallery train 14.296 / query test 2.879. Signer chồng giữa các split (D1), nên ghi rõ | R@1/5, DCG |
| ISLR | ASL Citizen; WLASL100/300/2000; MS-ASL1000 local (74 % clip) | Linear probe (đóng băng) và fine-tune; few-shot k ∈ {1, 5} | top-1/top-5, macro |
| Robustness | ASL Citizen val | Đổi fps thật (10/15/25, **không** nội suy lại), bỏ frame, jitter tay σ | ΔR@1 |
| Cơ chế | ASL Citizen val + ASL-LEX 2.0 | Probe tuyến tính theo tham số âm vị học; minimal pair; signer-NN; độ nhạy đảo thời gian | balanced acc, %, cos |

### 1.3 Ràng buộc thực tế

- Một RTX 5880 Ada 49 GB dùng chung. Tối đa 2 job đồng thời, chạy tuần tự (quy tắc của người dùng).
- Disk còn khoảng 94 GB (98 % đã dùng), nên giữ cache FP16 và chỉ lưu checkpoint cần thiết.
- Python 3.13 + torch 2.11 (`.venv-signrep`). signatory và keras_sig không tương thích. **Dùng công thức đóng level 2 trong torch; pySigLib chỉ làm oracle cho test** (P1 §3).

---

## 2. Path-JEPA: điều mang sang và điều phải sửa

Chi tiết ở `literature/P0_path_jepa_close_reading.md`. Tóm lại:

**Method [P].**
- Lead–lag trễ δ trên từng path. Log-signature bậc 2 cho mọi path joint / edge / chain.
- Mỗi token = [global log-sig trên [0,T] ‖ local log-sig của cửa sổ]. Hai thang: 25 frame stride 12, và 50 frame.
- Projection riêng theo loại path; embedding cấu trúc theo (thang, độ sâu động học).
- Mask: chọn 20–25 % khớp (seed theo chuyển động, mở rộng thành subtree), che mọi token có path chạm khớp bị che, **ở mọi thời điểm**.
- Student chỉ thấy token context; predictor cross-attention; teacher EMA thấy toàn bộ. Loss cosine trên feature teacher đã centering.

**Kết quả [P].** NTU60 X-Sub linear 87.6, hơn GFP 1.7 và S-JEPA 2.3.

**Đọc kỹ ablation [P + S]:**

| Câu hỏi | Bằng chứng | Hệ quả cho ta |
|---|---|---|
| Lợi ích đến từ đâu? | Joint + global only = 84.4 < S-JEPA 85.3. Phần tăng đến từ local (+1.4), edge (+0.6), chain (+1.2), subtree mask (+1.3) | Lợi ích chủ yếu từ **token quan hệ và đa thang**, chưa chắc từ "signature" thuần. → Bắt buộc có control **token tăng lượng (increment)** khớp chiều (arm A2) |
| Global view cần không? | Local only 83.3 < global + local 85.8 | Có, nhưng phải sửa rò rỉ khi có mask thời gian (C3) |
| Tỉ lệ mask | 20–25 % khớp là tối ưu; 10 % và 40 % đều kém | Đơn vị là khớp × mọi thời điểm, không so được với 50 % cell của PLctx |
| Bền với fps | 87.1–88.1 trên dải 15–60 fps | Protocol nội suy lại độ dài gốc, nên là kiểm tra nhẹ. Ta sẽ đo bằng đổi fps thật |

**Bốn điểm phải sửa khi sang sign [S]:**
1. **Static bị mất.** Path đứng yên có signature = 0, và signature bất biến tịnh tiến. Handshape và location chủ yếu là tĩnh (P3 §1.3). → Token lai (C2), basepoint.
2. **Rò rỉ global** khi thêm mask thời gian. Dùng định lý Chen thì khôi phục *chính xác* được cửa sổ bị che từ global và các cửa sổ còn lại; sai số 1e-15 [P1 §4.7, đã kiểm bằng numpy]. → C3.
3. **Path tree-like.** Chuyển động đi-về thẳng lặp lại có log-signature = 0 chính xác (P1 §1.7). Lévy area chéo hai tay trên cả chu kỳ bằng 0 *cả* khi cùng pha lẫn khi luân phiên (P3 §1.4, sửa nhận định trong P0). → Signature cửa sổ con (local), kênh tổng/hiệu hai tay, lead–lag là arm.
4. **Nhiễu tay 2D.** Độ lệch chuẩn của diện tích Lévy do nhiễu thuần tăng như $0.70\,\sigma^2\sqrt N$, và QV do nhiễu như $4(N-1)\sigma^2$ [P1 §1.8, mô phỏng]. → Làm mượt, hạ 25 fps (đã có ở PLctx), chuẩn hoá robust, gating (C5).

---

## 3. SignRep và vị trí của hướng skeleton

Chi tiết ở `docs/research/Skeleton_SSL_Sign_Research_Design_VI.md` §2. Ba điểm liên quan trực tiếp đến proposal này:

1. **Sức mạnh của SignRep nằm ở target mang thêm thông tin mà input không có.**
   - Encoder RGB nhưng phải dự đoán hình học 3D đã chuẩn hoá danh tính. Ablation của SignRep: prior quan hệ (dist, angle) hơn keypoint tuyệt đối 12–14 DCG [P].
   - Với skeleton, prior hình học là hàm tất định của chính input. Target tường minh vì thế không thêm thông tin (S1 và S2 đều âm [I]).
   - → Với ta, signature thuộc về **phía input/tokenizer**. Nó là một **inductive bias** về chuyển động, không phải nguồn thông tin mới. Thông tin mới vẫn phải đến từ target latent theo ngữ cảnh (PLctx, Path-JEPA).
2. **Readout activity-weighted cho SignRep +12.5 R@1 trên ASL Citizen [P].** Token signature cho ta một tín hiệu hoạt động miễn phí, không cần head riêng (C6).
3. **SignRep mù thứ tự:** cos(clip, clip đảo ngược) = 0.959 [I]. Movement mã hoá yếu nhất (lift 1.6× so với handshape 8.8×) [I]. Signature đổi dấu khi đảo thời gian (P3 §1.4.5), nên đây là một lợi thế đo được.

---

## 4. Bài học từ codebase nội bộ

Toàn bộ trên ASL Citizen. Số liệu lấy từ các report trong repo.

| Method (thư mục) | Ý tưởng chính | Kết quả | Bài học cho proposal16 |
|---|---|---|---|
| **PLctx full160** (`phonojepa`) | 7 kênh âm vị học × 32 bin; PLCM-lite (near: block thời gian; far: một kênh tay); target ctx EMA centered-CE; readout `grp` | Test R@1 **16.68**, WLASL 9.62; LOSO white 29.5; LP macro 68.9; movement R² −0.03; signer 93.7 % | **Anchor (REF)**: giữ nguyên token trạng thái, mask, loss, readout. Movement là lỗ hổng cần lấp |
| PLint / PLboth | Thêm target descriptor tường minh | Kém PLctx khoảng 5 (ASLC300) | Target tường minh không giúp |
| S1 / **S2** (`phonojepa_s1/_s2`) | S2 = ctx + target motion + **log-signature bậc 2 làm target** | S2 ASLC300 15.7, kém S1 7.6 và kém PLctx16k 33.2 | Signature làm *target* thì âm. Nhưng so sánh bị lẫn biến: base S1 đã kém PLctx, readout khác. Arm "signature làm input" (`signature_input_4k`) **chưa chạy** |
| **SignPathLab** (`signpathlab`, proposal13) | DINO RGB + pose stream; logsig2 làm geometry target | Pilot 8 lớp, 16 val: E03 = E04, không có bằng chứng lợi ích | Pilot quá nhỏ, không kết luận được về signature. Pipeline phức tạp (RGB + DINO), bị dừng |
| **Pose-JEPA** (`pose_jepa`, proposal14) | Hybrid body/relations, 125k bước | Test raw 5.74, white 9.32 | Token per-joint/graph yếu hơn token kênh âm vị học nhiều. Nhưng bổ sung được: 127/766 query PLctx sai mà Pose đúng (`fusion_plctx_pose_rank_audit_20261016`) → quan hệ tay–thân mang thông tin riêng |
| **ASM-JEPA** (`asm_jepa`, proposal15) | Per-joint, articulator masking | Test R@1 1.07; fine-tune 55.4 so với scratch 53.8 | Gần như không học được feature đóng băng. Tốt nhất ở bước 5k rồi giảm |
| Graph-JEPA B10–B13 | Per-joint 67 landmark | ASLC300 17–21 | Như trên |
| SignRep tái tạo (`jepa_baseline`) | Recipe SignRep trên ASL Citizen | Sụp: effective rank 63 → 6 | Var/cov + adversarial không chuyển được |
| R0 / R4 masks | Random tube / articulator span ở cả hai view | ASLC300: PL 43.2 > R0 37.9 > R4 35.3 | **Che cả articulator ở cả hai view thì kém.** Kiểu subtree của Path-JEPA chỉ nên là *một thành phần* của view far |
| Global CLS alignment (G1) | Căn CLS giữa hai view | −5.7 | Không thêm loss cấp clip kiểu này làm mặc định. N11 của P2 (luôn có mục tiêu cấp clip) mâu thuẫn với bằng chứng nội bộ, nên chỉ là arm phụ |

**Năm bài học thiết kế:**
1. **Đơn vị token theo kênh âm vị học đã thắng mọi kiểu per-joint/graph** trong cùng ngân sách: PLctx 48.9 so với B13 21.3 trên ASLC300 [I]. → Token signature phải gom **theo kênh** (D-CONF, D-LOC, N-CONF, N-LOC, REL), không theo từng khớp như Path-JEPA.
2. **Target latent theo ngữ cảnh là thứ duy nhất có tín hiệu rõ** [I]. → Giữ centered-CE + EMA. Loss cosine của Path-JEPA chỉ là arm phụ.
3. **So sánh phải cùng codebase, cùng readout, cùng budget.** S2 và SignPathLab không kết luận được vì lẫn biến. → Mọi arm fork từ `phonojepa`, chung evaluator, chung lịch học.
4. **Proxy pilot đánh giá quá cao.** ASLC300 47.7 so với test 16.7. → Dev metric là VAL-DICT. Corpus screening là toàn bộ 40k clip train.
5. **Vệ sinh tay.** Quan sát tay phải lọc theo khoảng cách tới cổ tay thân và theo việc nằm trong ảnh; score SimCC không tách được tay tốt và xấu (`asm_jepa/data/observation.py`). Khoảng 14–17 % frame bị alias trái/phải (D1).

---

## 5. Tín hiệu trong ngôn ngữ ký hiệu: phân tích và chẩn đoán

### 5.1 Từ âm vị học đến tín hiệu skeleton và mức signature

Tóm tắt từ P3 §1.3. Phân bố lấy trên 2.723 ký hiệu ASL-LEX 2.0.

| Tham số | Tĩnh / động | Mang bởi | Mức signature mã hoá được | Ghi chú thiết kế |
|---|---|---|---|---|
| Handshape (selected fingers, flexion) | Tĩnh | 21 điểm tay, frame bàn tay | Không (path tĩnh = 0) | → **token trạng thái** CONF của PLctx |
| Handshape change (19.5 %) | Động, local | Aperture, góc ngón | Level 1 của aperture (mở/đóng); level 2 aperture × path | Path aperture 2D trong D-CONF |
| Wrist twist (16.5 %) | Động, local | Vector MCP trỏ → MCP út; dấu lòng bàn tay; độ co ngắn | **Sửa sau review:** area của vector quay so với dây cung là $\tfrac12L^2(\theta-\sin\theta)\sim\theta^3$, *không* ≈ θ. Hơn nữa, sấp/ngửa trong ảnh 2D chính diện chủ yếu là quay ngoài mặt phẳng: vector co ngắn chứ không quay | Path 2D [góc trong mặt phẳng đã unwrap, độ co ngắn có dấu lòng bàn tay (theo CONF của PLctx)]. Level 1 = góc quay ròng và thay đổi co ngắn. 3D (HaMeR/WiLoR trên tập con) là hướng tương lai |
| Path movement (84.7 % có) | Động | Quỹ đạo cổ tay so với thân | Level 1: hướng. Level 2: thẳng ≈ 0, cung vừa, tròn lớn, dấu = chiều quay | Path cổ tay + chain cánh tay trong D-LOC |
| Repeated (44.6 %) | Động, tuần hoàn | Như trên | Vòng kín: area × k. **Đi-về: tree-like = 0** | Token local ở thang 0.32 s; lead–lag là arm |
| Location (Head 26.7 %, Hand 25 %) | Tĩnh | Tay so với mặt / thân / tay N | Chỉ khi có basepoint hoặc path tương đối | Path tay − mũi, tay − tay N, có basepoint; LOC trạng thái |
| Contact (62.2 %) | Sự kiện | min khoảng cách | Level 1 của khoảng cách: tiến rồi lùi | Path tương đối tay − mặt |
| Sign type / Battison (60.8 % hai tay) | Động | Năng lượng tay N, quan hệ D–N | Cross area trên cả chu kỳ = 0 cả khi cùng pha lẫn luân phiên. **Sửa sau review:** đổi sang tổng/hiệu $[\sigma,\delta]$ là phép tuyến tính khả nghịch của $[w_D,\mathcal M(w_N)]$, nên path tree-like vẫn tree-like; tự nó *không* sửa được. Cái phân biệt được: (a) **level 1 theo cửa sổ con** của $\sigma$, $\delta$ (luân phiên thì $\sigma$ ≈ hằng, cùng pha thì $\delta$ ≈ hằng trong cửa sổ nửa chu kỳ); (b) cross-QV của lead–lag | Kênh REL (§7.3): $[\sigma,\delta]$ ở thang 0.32 s, dùng vì nó *tách* năng lượng cùng pha và luân phiên ra hai trục. Arm phụ lead–lag cross-QV |
| Non-manual | Động | Miệng, mày | Level 1–2 | Kênh phụ: ISLR ASL chỉ +0.56 top-1 (NLA-SLR, P3 §2.1) |
| Nhiễu: signer, tempo, morphology, framing, handedness | — | — | Bất biến reparam → tempo. Scale → chuẩn hoá vai. Mirror → canonical + giữ dấu | §5.2 đo trực tiếp |

### 5.2 D-TEMPO: tempo là tín hiệu từ vựng hay nhiễu theo signer?

Script: `diagnostics/tempo_anova.py` và `diagnostics/active_tempo.py`.
- Hồi quy hiệu ứng cố định cộng tính gloss + signer trên train ASL Citizen.
- Frame "hoạt động" = cổ tay trội cao hơn mức nghỉ của clip (phân vị 10 % + 0.25 độ rộng vai) **hoặc** đang chuyển động nhanh hơn 1 độ rộng vai/s.

R² "gloss only" có bias dương lớn vì nhiều nhóm, ít mẫu. Kỳ vọng dưới giả thuyết null ≈ $(k-1)/(n-1)$: 0.068 trên full train và 0.136 trên nửa train. Cột "gloss (hiệu chỉnh)" trừ bias đó: $(R^2-b)/(1-b)$.

| Biến (log) | Tập | R² gloss | Gloss (hiệu chỉnh) | R² signer | R² gloss + signer |
|---|---|---:|---:|---:|---:|
| Thời lượng clip | Full train, 40.154 | 0.065 | **≈ 0.00** | **0.732** | 0.777 |
| Thời lượng clip | Nửa train, 20.056 | 0.132 | ≈ 0.00 | 0.730 | 0.794 |
| Thời lượng hoạt động | Nửa train | 0.238 | ≈ 0.12 | **0.336** | 0.540 |
| Tốc độ cổ tay trung vị khi hoạt động | Nửa train | 0.234 | ≈ 0.11 | 0.128 | 0.354 |
| Độ dài đường đi cổ tay khi hoạt động | Nửa train | 0.250 | ≈ 0.13 | 0.216 | 0.442 |
| Tỉ lệ frame hoạt động | Nửa train | 0.237 | ≈ 0.12 | 0.156 | 0.374 |

Thời lượng clip: trung vị 2.53 s, p5–p95 là 1.54–5.57 s. Thời lượng hoạt động: trung vị 1.34 s, p5–p95 là 0.60–2.64 s. Frame hoạt động chiếm trung vị 53 % clip. Hệ số tempo theo signer, $\exp(\text{signer effect})$, dao động 0.67–3.27 trên thời lượng clip.

**Diễn giải [D + S]**, sửa sau review độc lập:
- Thời lượng clip **chủ yếu** là thuộc tính của signer và cách quay. ASL Citizen tự quay qua webcam, có đoạn đầu và cuối.
- Nhưng gloss **vẫn** mang thông tin tempo khi đã biết signer. $R^2$ riêng phần của gloss, $(R^2_{g+s}-R^2_s)/(1-R^2_s)$:
  - thời lượng clip: 0.17, so với khoảng 0.07 kỳ vọng dưới null;
  - thời lượng hoạt động: 0.31, so với khoảng 0.14 dưới null.
  Tức gloss giải thích khoảng 10–17 % phương sai *trong cùng một signer*.
- Heuristic "hoạt động" dùng tốc độ thô chưa làm mượt, nên jitter riêng của từng signer có thể thổi phồng phần của signer.
- → Kết luận đúng: tempo tuyệt đối là **hỗn hợp nhiễu signer và tín hiệu từ vựng**, với nhiễu signer lớn hơn. Thiết kế phù hợp là **bỏ thời gian tuyệt đối nhưng giữ thời gian tương đối**: cửa sổ tính theo giây sau khi resample cố định 25 fps, kèm vị trí tương đối trong đoạn hoạt động. Không phải bỏ hẳn thông tin thời gian.
- Đây là bằng chứng định lượng đầu tiên trong repo cho giả thuyết H5, ở dạng đã điều chỉnh: bất biến với tốc độ *trong* cửa sổ và với tempo toàn cục của signer.

### 5.3 D-SIG: descriptor không cần train, so từng họ tín hiệu

Script: `diagnostics/prep_pose.py` và `diagnostics/sig_diagnostics.py`. Kết quả JSON: `diagnostics/results/diag_results.json`.

**Tiền xử lý.**
- Observation filter: trong ảnh, và cổ tay tay ≤ 0.30 độ rộng vai so với cổ tay thân.
- Khung thân: gốc ở giữa vai, đơn vị là độ rộng vai trung vị, đơn vị pixel nên đã đúng tỉ lệ khung.
- Canonical tay trội: tay có đường đi cổ tay dài hơn; gương hoá nếu tay trái trội.
- Nội suy khoảng trống.

**Descriptor.**
- Mỗi họ path cho một block. z-score theo chiều, fit trên gallery. Mỗi block có trọng số bằng nhau. Sau đó L2 rồi cosine.
- *glob* = log-sig trên cả clip; *loc4* = 4 cửa sổ bằng nhau theo thời gian; *base* = điểm đầu và trung bình (thông tin tĩnh); *LL2* = lead–lag trễ 2 frame; *time* = kênh thời gian; *smooth5* = trung bình trượt 5 frame trước khi tính.
- "man" = joint + edge + chain + relation; "all" = thêm mặt.

**Metric.**
- VAL-DICT R@1/5: gallery = nửa train (20.056 clip, mọi gloss), query = toàn bộ val 10.304.
- LOSO R@1: trong val, query với gallery là các signer val khác.
- sNN: % neighbour gần nhất *khác gloss* nhưng cùng signer. Càng thấp thì signer leakage càng ít; mức ngẫu nhiên là 26.2 %.
- tempo cos: cosine giữa descriptor của clip gốc và clip resample ×0.67.
- ε² gloss / signer: phần phương sai hiệu chỉnh của các chiều descriptor giải thích bởi gloss / signer.

| Descriptor | D | VAL-DICT R@1 | R@5 | LOSO R@1 | sNN % ↓ | tempo cos | ε² gloss | ε² signer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static(all): trung bình + độ lệch chuẩn vị trí | 616 | 3.09 | 8.14 | 2.70 | 92.0 | 0.998 | 0.091 | 0.126 |
| bins8(all): vị trí trung bình 8 bin thời gian | 2464 | **4.02** | **9.58** | **3.23** | 94.9 | 0.998 | 0.072 | 0.126 |
| vel8(all): vận tốc, \|vận tốc\| 8 bin | 4928 | 2.61 | 6.79 | 1.76 | 85.3 | 0.911 | 0.024 | 0.064 |
| sig glob L1 (man) | 274 | 0.01 | 0.16 | 0.04 | 62.0 | 1.000 | −0.002 | 0.010 |
| sig glob chỉ L2 (man) | 389 | 1.13 | 3.77 | 1.09 | **41.3** | 0.943 | 0.012 | 0.002 |
| sig glob L1+L2 (man) | 663 | 0.75 | 2.44 | 0.52 | 56.4 | 0.975 | 0.006 | 0.005 |
| sig glob+loc4 (man) | 3315 | 2.54 | 7.07 | 1.73 | 75.4 | 0.956 | 0.019 | 0.022 |
| sig glob+loc4+base (man) | 3863 | 2.71 | 7.19 | 2.13 | 85.2 | 0.965 | 0.024 | 0.035 |
| + lead–lag trễ 2 | 12438 | 3.33 | 8.30 | 2.31 | 82.5 | 0.915 | 0.020 | 0.025 |
| + kênh thời gian | 5753 | 2.58 | 6.75 | 1.89 | 86.5 | 0.964 | 0.020 | **0.098** |
| + smooth5 | 3863 | 3.41 | 8.49 | 2.36 | 87.6 | 0.979 | 0.026 | 0.039 |
| sig glob+loc4+base (all, thêm mặt) | 4186 | 2.80 | 7.62 | 2.08 | 88.2 | 0.965 | 0.023 | 0.040 |
| sig chỉ loc4 + base (man) | 3200 | 2.97 | 7.28 | 2.10 | 85.3 | 0.963 | 0.028 | 0.042 |
| họ joint | 798 | 1.96 | 5.18 | 1.38 | 82.1 | 0.974 | 0.046 | 0.067 |
| họ edge | 836 | 2.47 | 6.37 | 1.81 | 86.5 | 0.964 | 0.026 | 0.036 |
| họ chain | 2076 | 2.18 | 5.67 | 1.51 | 84.7 | 0.951 | 0.013 | 0.016 |
| họ relation (tay–mặt, tay–tay, 2 tay) | 153 | 1.66 | 5.04 | 1.14 | 77.7 | 0.975 | 0.057 | 0.131 |
| họ relation, không basepoint | 125 | 1.31 | 4.12 | 0.93 | 68.4 | 0.968 | 0.060 | 0.082 |
| họ relation, chỉ vị trí tĩnh | 28 | 1.53 | 5.31 | 1.33 | 74.7 | 0.999 | 0.182 | 0.228 |
| họ joint, chỉ vị trí tĩnh | 168 | 2.66 | 7.02 | 2.02 | 80.1 | 0.998 | 0.136 | 0.106 |
| họ face | 323 | 0.38 | 1.17 | 0.35 | 89.4 | 0.974 | 0.007 | 0.101 |
| bins8 + sig (all) | 6650 | 4.12 | 9.38 | 2.98 | 93.4 | — | 0.041 | 0.072 |

**Đọc kết quả [D]:**
1. **Signature toàn cục trên cả clip vô dụng làm descriptor.**
   - Level 1 trên clip có R@1 = 0.01. Clip đi từ nghỉ về nghỉ, nên độ dời ròng ≈ 0.
   - Chỉ level 2 lên được 1.13.
   - Signature phải theo cửa sổ: loc4 cho 2.5–3.4.
   - Hệ quả: global view kiểu Path-JEPA chủ yếu mang chuyển động vào/ra tư thế nghỉ, và token local mới là chính. Đây là lý do C3 dùng G chỉ làm token phụ, và arm A6 bỏ hẳn G.
2. **Là descriptor không học, signature không vượt toạ độ theo bin.**
   - bins8 = 4.02 so với signature tốt nhất 3.41.
   - Ghép hai loại chỉ được 4.12, tăng +0.1, trong mức nhiễu.
   - Hệ quả: lợi ích của signature, nếu có, phải đến từ việc **encoder học được cách dùng** token signature. Nó không đến từ thông tin tuyến tính sẵn có. Đây là rủi ro lớn nhất của proposal, và được kiểm tra rẻ bằng cổng G0 (§9.1d).
   - **Giới hạn (review):** mốc ở đây yếu. Descriptor tốt nhất chỉ đạt 4.02, so với B0 = 13.46 (descriptor CONF khung bàn tay của proposal12, gallery đầy đủ). Gallery chỉ là nửa train. Mỗi block có trọng số bằng nhau, nên các area nhiễu của chain ngón làm loãng descriptor. Cửa sổ là phân số clip, không phải 0.32/0.64 s như method. Phase 0(e) sẽ đo lại B0 so với B0 + signature với cửa sổ đúng thiết kế.
3. **Signer leakage: lợi thế của signature *chưa* được chứng minh khi so ở cùng mức hữu ích** (đã sửa sau review).
   - sNN thấp nhất (41 %, level 2 toàn cục) thuộc về descriptor gần vô dụng (R@1 1.13). Descriptor yếu thì tự nhiên tiến về mức ngẫu nhiên 26 %.
   - Cấu hình signature tốt nhất (smooth5) có sNN 87.6 %, chỉ thấp hơn toạ độ (92–95 %) một chút.
   - Tỉ số ε² gloss / ε² signer: smooth5 0.67, static 0.72, bins8 0.57, LL2 0.80, signature không basepoint 0.86. Không có lợi thế rõ.
   - Điểm đứng vững: ε² signer của vị trí tĩnh tay–mặt (0.228) cao hơn ε² gloss (0.182). Vị trí tương đối mang hình thể signer, và bỏ basepoint làm giảm cả hai.
   - → Lý do giữ luồng P là **inductive bias cho movement** (cần G0 kiểm tra). Ít-signer chỉ là giả thuyết H5, đo bằng probe signer-ID / gloss ở cùng mức retrieval, không còn là lý do chính.
4. **Nhiễu tay là thật.**
   - Làm mượt 5 frame: R@1 từ 2.71 lên 3.41 (+26 % tương đối), tempo cos từ 0.965 lên 0.979.
   - Tempo cos của signature (0.96) *thấp hơn* của toạ độ (0.998), dù lý thuyết nói signature bất biến reparam. Lý do: resample làm thay đổi phần area do jitter, vì phần này tỉ lệ $\sigma^2\sqrt N$ (P1 §1.8).
   - → Bất biến tempo của signature **chỉ có khi đã khử nhiễu**. Ủng hộ C5 và H6.
5. **Kênh thời gian đưa danh tính signer vào:** ε² signer tăng từ 0.035 lên 0.098, R@1 từ 2.71 xuống 2.58.
   - **Giới hạn (review):** ở đây kênh thời gian tính bằng giây tuyệt đối trên cửa sổ là *phân số của clip*, nên nó chủ yếu mã hoá thời lượng clip, vốn do signer quyết định (D-TEMPO).
   - Kết quả này **không** chuyển thẳng sang A7, nơi cửa sổ có độ dài cố định theo giây. A7 vẫn là thí nghiệm mở.
6. **Lead–lag trễ 2:** +0.6 R@1 với 3.2 lần số chiều. Tempo cos giảm còn 0.915, vì QV nhạy với lưới lấy mẫu (P1 §1.5). → Chỉ là arm phụ.
7. **Mặt** gần như không có thông tin gloss (0.38) nhưng nhiều thông tin signer (ε² 0.101). Khớp với PLctx D8. → Kênh mặt P chỉ là tuỳ chọn.

### 5.4 D-LEX: họ tín hiệu nào lộ ra tham số âm vị học nào?

Script: `diagnostics/asllex_probe.py` (v1), `asllex_probe_v2.py` (v2), `asllex_probe_v3.py` (v3).
- Probe = StandardScaler → PCA → logistic regression cân bằng lớp. Metric là balanced accuracy.
- Nhãn từ ASL-LEX 2.0, nối qua cột `ASL-LEX Code`, phủ 99.8 % clip.

**Sửa sau review (Critical 1).** Nhãn ASL-LEX gắn theo *gloss*. Ở v1/v2, probe-train (nửa train) và val dùng chung mọi gloss, nên probe có thể thắng nhờ *nhận ra gloss* chứ không nhờ mã hoá âm vị học. Hơn nữa, descriptor "static" của v1/v2 gồm trung bình **và độ lệch chuẩn** vị trí, mà độ lệch chuẩn chính là biên độ chuyển động.

v3 sửa cả hai:
- Chia ngẫu nhiên 30 % gloss (819 gloss) làm gloss test. Fit trên 14.000 clip nửa-train của các gloss còn lại; chấm trên 3.086 clip val của gloss test. Như vậy probe **rời gloss và rời signer**.
- Cùng PCA 256 cho mọi descriptor.
- Thêm descriptor *chỉ trung bình*.
- CI 95 % bootstrap theo gloss test.

Kết quả JSON: `diagnostics/results/probe_v3_results.json`. Một lần chạy.

| Descriptor | SignType | Movement | Repeated | MajorLoc | Contact | FlexChange | WristTwist | SelFingers | Flexion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *ngẫu nhiên* | 25 | 20 | 50 | 20 | 50 | 50 | 50 | 17 | 20 |
| static chỉ trung bình | 51.6 [48–54] | 25.6 [22–29] | 54.8 [53–57] | 43.0 [40–46] | 60.0 [58–62] | 55.0 [52–58] | 50.9 [48–54] | **48.7** [45–52] | 32.0 [30–35] |
| static trung bình + std | 55.9 [53–59] | **30.9** [27–34] | **56.7** [55–59] | 50.2 [46–54] | 63.9 [62–66] | **56.9** [54–60] | 53.1 [50–56] | **48.9** [46–52] | 33.2 [30–36] |
| bins8 | **56.9** [54–60] | 27.6 [24–31] | 54.4 [52–57] | **51.1** [47–55] | **66.0** [64–68] | 55.7 [53–59] | **54.1** [51–57] | 48.4 [45–52] | **35.4** [33–38] |
| vel8 | 51.1 [49–54] | 23.4 [21–26] | 55.7 [54–58] | 43.9 [41–47] | 63.4 [61–66] | 53.3 [51–56] | 53.9 [51–57] | 30.6 [28–33] | 27.1 [25–29] |
| sig + base, smooth5 | 50.2 [48–53] | 21.9 [19–25] | 51.9 [50–54] | 43.4 [40–47] | 63.3 [61–65] | 52.3 [50–55] | 53.7 [51–56] | 29.6 [27–32] | 28.9 [27–31] |
| sig + base, LL2 | 50.8 [48–54] | 24.6 [21–29] | 55.4 [53–57] | 46.3 [43–50] | 62.7 [60–65] | 54.5 [52–57] | 51.7 [49–55] | 29.5 [27–32] | 27.0 [25–29] |
| sig không basepoint | 42.2 [40–45] | 21.3 [18–25] | 51.6 [50–54] | 37.8 [35–41] | 58.6 [56–61] | 54.4 [51–57] | 51.1 [48–54] | 25.5 [23–28] | 24.5 [23–27] |
| trung bình + sig smooth5 | 51.2 [49–54] | 22.7 [20–26] | 53.0 [51–55] | 44.9 [42–48] | 63.7 [61–66] | 53.5 [51–56] | 53.1 [50–56] | 34.1 [31–37] | 30.5 [28–33] |
| trung bình + std + sig smooth5 | 52.9 [50–56] | 23.3 [20–27] | 54.7 [53–57] | 47.4 [44–51] | 63.8 [61–66] | 52.7 [50–56] | 53.9 [51–57] | 37.3 [34–40] | 30.2 [28–32] |

v1/v2, *không rời gloss*, ở `results/probe_results.json` và `probe_v2_results.json`: cùng thứ hạng, số cao hơn 1–3 điểm. Chỉ dùng làm tham khảo.

**Đọc kết quả [D]:**
1. **Cái mang thông tin movement trong "static" là độ lệch chuẩn**, tức biên độ / độ trải của chuyển động.
   - Từ chỉ trung bình lên trung bình + std: Movement 25.6 → 30.9, MajorLoc 43.0 → 50.2, SignType 51.6 → 55.9. Các CI tách nhau hoặc gần tách.
   - Thống kê bậc hai của vị trí đơn giản là feature chuyển động rất mạnh. Nó cũng là thành phần đối xứng của level 2 (covariance của path), trong khi signature giữ thành phần *phản đối xứng* (Lévy area).
   - → Gợi ý trực tiếp cho control A2 (§8): thay area bằng moment bậc hai đối xứng cùng chiều.
2. **Signature thuần thua mọi baseline**, đặc biệt ở handshape (SelFingers khoảng 30 so với 49). Handshape là tĩnh, và chain ngón theo thời gian không mang nó.
3. **Ghép signature vào trung bình**, cùng PCA 256:
   - giúp Contact (+3.7) và WristTwist (+2.2), MajorLoc +1.9, CI chồng nhau;
   - làm hại Movement (−2.9) và SelFingers (−14.6). Đây là pha loãng: PCA 256 trên khoảng 4.000 chiều bị các area nhiễu chiếm chỗ.
   → Thông tin bổ sung có ở tham số quan hệ và quay (contact, twist). Kết hợp tuyến tính đơn giản thì không khai thác được.
4. **Movement khó với mọi descriptor:** tối đa 31 %, ngẫu nhiên 20 %. Nhãn ASL-LEX Movement là phạm trù citation thô, và 2D mất trục sâu. Probe này nhiễu, chỉ là báo cáo phụ.
5. **Hàm ý cho thiết kế:**
   - Signature **không** phải feature tốt hơn sẵn có. Lợi ích chỉ có thể đến từ việc encoder học dùng nó, cộng với một phần thông tin quan hệ/quay. Cổng G0 kiểm tra điều này.
   - Probe cơ chế của H1 phải **rời gloss và rời signer**, như v3.

### 5.5 Tổng hợp: tín hiệu nên ép mô hình học và tín hiệu nên bất biến

Kết hợp P3 §4 (lý thuyết và literature) với D-TEMPO, D-SIG, D-LEX:

**Nên ép mô hình học** (thứ tự ưu tiên):

| # | Tín hiệu | Bằng chứng | Đưa vào method bằng |
|---|---|---|---|
| 1 | Handshape tay trội (tĩnh, khung bàn tay) | 3.048 minimal pair; probe rời gloss SelFingers 49 % chỉ từ vị trí trung bình, Flexion 32–35 % (D-LEX v3); PLctx handshape R² 0.32 [I] | Luồng S (CONF) — giữ nguyên |
| 2 | Quan hệ tay–mặt / thân / tay N (location, contact) | MajorLoc 50–51 %, Contact 64–66 % từ thống kê vị trí (D-LEX v3); thêm signature vào trung bình +3.7 Contact; Pose-JEPA cứu 127/766 lỗi của PLctx nhờ body/relations [I] | Luồng S (LOC) + path quan hệ P **có** basepoint |
| 3 | Path movement: hướng, độ cong, chiều quay, theo cửa sổ con | Trục minimal pair lớn nhất (7.510); mọi model skeleton có R² < 0 [I]; descriptor tuyến tính đều yếu (D-LEX) | Token P cửa sổ 0.32 / 0.64 s; cổng G0 kiểm tra trước |
| 4 | Local movement: wrist twist, aperture change | Một trong số ít tham số mà signature thêm được vào trung bình vị trí (WristTwist +2.2, D-LEX v3); path twist đã sửa (§7.3) | Path hướng bàn tay + aperture trong D-CONF |
| 5 | Cấu trúc hai tay (Battison) | SignType 56–57 % từ thống kê vị trí (D-LEX v3); 60.8 % ký hiệu là hai tay | Kênh REL: $[\sigma,\delta]$ + level 1 theo cửa sổ con |
| 6 | Thứ tự thời gian trong ký hiệu | SignRep mù thứ tự (cos 0.959) [I]; signature đổi dấu khi đảo | Token P + RoPE; đo bằng H10 |
| 7 | Miệng (kênh phụ) | ISLR +0.56 (NLA-SLR); D-SIG face R@1 0.38 | Tuỳ chọn, trọng số thấp |

**Nên bất biến** (nhiễu):

| Nhiễu | Bằng chứng định lượng | Cách xử lý |
|---|---|---|
| Tempo (thời lượng, tốc độ) | Signer giải thích 73 % thời lượng clip và 34 % thời lượng hoạt động; gloss ≈ 0 / 12 % (D-TEMPO). Kênh thời gian làm ε² signer ×2.8 (D-SIG) | Không có kênh thời gian; time-warp augmentation; signature đã khử nhiễu |
| Frame nghỉ, nâng/hạ tay | 47 % frame không hoạt động; level 1 toàn cục ≈ 0; nâng/hạ là path lớn nhất | Token local; readout trọng số theo độ cao tay |
| Hình thể và vị trí tương đối theo signer | ε² signer của vị trí tĩnh tay–mặt 0.228 > ε² gloss 0.182; sNN của toạ độ 92–95 % | Chuẩn hoá vai; morphology augmentation; luồng P không basepoint có sNN thấp hơn |
| Jitter tay | Smoothing +26 % R@1 descriptor; tempo cos 0.965 → 0.979 | SG5, asinh, jitter-aug, gating |
| Handedness | Gương hoá đổi dấu area | Canonical + giữ dấu |
| Mặt / danh tính khuôn mặt | Face ε² signer 0.101 so với gloss 0.007 | Kênh mặt phụ, neutral-face (PLctx) |

---

## 6. Giả thuyết đăng ký trước

Mỗi giả thuyết có: dự đoán định lượng, arm kiểm định (§8), tiêu chí bác bỏ, và control cho giải thích thay thế. Luật quyết định chung ở §9.4.

| ID | Giả thuyết | Dự đoán | Arm | Bác bỏ nếu | Control / giải thích thay thế |
|---|---|---|---|---|---|
| **H1** | Token signature giúp encoder mã hoá movement | Probe ASL-LEX **rời gloss + rời signer** (protocol D-LEX v3) cho Repeated / WristTwist / FlexChange / Contact tăng ≥ 2 điểm so với REF, CI bootstrap theo gloss loại trừ 0. Minimal pair theo movement (P3 §1.2) tăng. ASL-LEX Movement chỉ báo phụ, vì nó nhiễu (D-LEX). **Đã bỏ** probe tự gán nhãn cw/ccw từ quỹ đạo: nhãn đó gần như chính là dấu Lévy area mà A1 nhận làm input, nên vòng tròn (review Critical 2). Chỉ dùng lại nếu nhãn đến từ nguồn độc lập, ví dụ gán tay khoảng 300 clip val | A1 vs A0 | Δ ≤ 0 hoặc CI chứa 0 ở mọi probe | A2 tách "có thêm token chuyển động" khỏi "là Lévy area" |
| **H2** | Luồng signature cải thiện retrieval | VAL-DICT R@1 ≥ REF + max(1.5, 2.5 × SD seed của A0) (§9.4) | A1 vs A0 | Δ < 0.5 với TOST tương đương ±1 | Khớp token bằng A2; so sánh chính là A1-`grp` với A0-`grp`, cùng xử lý centering/PCA (§9.4) |
| **H3** | Lévy area (phần phản đối xứng của level 2) mang thông tin vượt moment bậc hai đối xứng cùng chiều | A1 > A2 về retrieval và probe WristTwist / Contact / Repeated | A1 vs A2 | A2 ≥ A1 | A2 (định nghĩa lại sau review, §8): cùng số token, cửa sổ, chiều; thay $A_{ij}$ bằng moment đối xứng (QV / covariance của increment) — đây là thông tin mà D-LEX v3 cho thấy rất mạnh (std vị trí) |
| **H4** | Global [0,T] gây rò rỉ khi có mask thời gian; global Chen-visible sửa được | Loss P-token của A5 thấp bất thường (< 50 % của A1) nhưng retrieval kém hơn; A6 ≥ A1 | A5, A6 vs A1 | A5 không có dấu hiệu rò rỉ và không kém | Test rò rỉ đơn vị: khôi phục cửa sổ che bằng Chen |
| **H5** | Bất biến với tốc độ *trong* cửa sổ và tempo toàn cục của signer giúp retrieval độc lập signer (sửa sau review) | Tỉ số acc probe signer-ID / acc probe gloss trên embedding giảm so với REF, đo ở cùng mức retrieval; ΔR@1 khi đổi fps thật (10/15/25) nhỏ hơn REF | A1 vs A0; A7 (thêm kênh thời gian tương đối trong cửa sổ) | Không giảm ở cả hai chỉ số | D-TEMPO: tempo tuyệt đối là hỗn hợp nhiễu signer (lớn hơn) và tín hiệu gloss (10–17 % trong cùng signer) |
| **H6** | Nhiễu tay chi phối level 2; làm mượt + chuẩn hoá robust là cần thiết | A1 (SG5 + asinh) > A1-raw | A1 vs A8a, A8b | A8a ≥ A1 và A8b ≥ A1 | Jitter tổng hợp σ ∈ {0.01, 0.02, 0.04} trên val |
| **H7** | Path quan hệ tay–mặt và tay–tay mang location / contact / sign-type | Bỏ họ R: probe MajorLoc / Contact / SignType giảm, retrieval giảm | A4 vs A1 | Không giảm | Cùng số token (họ R thay bằng 0, đánh dấu absent) |
| **H8** | Mask articulator subtree (kiểu Path-JEPA), làm *một thành phần* của view far, có ích | A9 ≥ A1 + 1 | A9 vs A1 | A9 < A1 | R4 trong cả hai view đã âm [I]; ở đây chỉ trộn 30 % vào view far |
| **H9** | Readout trọng số chuyển động (không nhãn) giúp mọi model | R@1(mw) > R@1(grp) cho REF và A1 | Chỉ readout | Δ ≤ 0 | Chạy trên checkpoint đã có, không train |

Giả thuyết phụ, chỉ đo, không có arm riêng:
- **H10.** Signature làm embedding nhạy thứ tự: cos(clip, clip đảo) giảm so với REF.
- **H11.** Token signature của cửa sổ nghỉ có năng lượng gần 0, tức chúng tự phân đoạn được. Đo bằng tương quan giữa năng lượng P-token và nhãn frame hoạt động của D-TEMPO.

---

## 7. Method: PhonoPath-JEPA

### 7.1 Tổng quan

```
keypoints (COCO-WB 133, 2D)
  → observation filter + canonical body frame + canonical tay trội                  [như PLctx]
  → resample 25 fps, cửa sổ 64 mẫu (2.56 s), near/far views                         [như PLctx]
  ├─ Luồng S (state):  7 kênh × 32 bin = 224 token tĩnh                             [PLctx, không đổi]
  └─ Luồng P (path):   5 kênh × (15 cửa sổ 8 mẫu + 7 cửa sổ 16 mẫu) = 110 token logsig2
                       + 5 token global (tính nhất quán Chen)                        [mới]
  → mask PLCM-lite trên cell (kênh × bin) → closure lên support (khớp × mẫu) → ẩn S, P, G
  → student Transformer chung (S ∪ P ∪ G visible; RoPE theo thời gian liên tục; bias cấu trúc)
  → predictor (query metadata) → dự đoán feature teacher tại các S, P bị ẩn
  → teacher EMA (thấy toàn bộ, global thật) → target = LN mean 4 layer cuối
  → loss centered-CE (S) + λ_P centered-CE (P)
Inference: teacher encoder, readout grp(S) ‖ grp(P), pooling có trọng số chuyển động
```

### 7.2 Tiền xử lý (C5, thừa kế PLctx)

1. **Observation.** `in_image_only` + `hand_wrist_max_dist = 0.30`, như `asm_jepa/data/observation.py`. Score SimCC chỉ dùng làm feature liên tục (độ tin cậy), không làm ngưỡng.
2. **Khung thân.** Như PLctx: xoay theo góc vai trung vị, gốc ở giữa vai, đơn vị là độ rộng vai. Toạ độ pixel được hiệu chỉnh tỉ lệ khung.
3. **Canonical tay trội.** Như PLctx (rò rỉ 1 bit, đã khai báo). Gương hoá kèm hoán vị iBUG cho mặt, và **giữ dấu Lévy area** sau gương hoá (P3 §1.4.4).
4. **Làm mượt riêng cho luồng P.** Savitzky–Golay cửa sổ 5, bậc 2 (ở 25 fps) trên điểm tay và cổ tay **trước khi** tính signature. Luồng S giữ nguyên toạ độ như PLctx. Arm A8a bỏ bước này. Làm mượt mở rộng support của token P thêm ±2 mẫu; closure phải tính đến điều này (§7.5).
5. **Khoảng trống.** Nội suy tuyến tính tối đa 2 mẫu. Một token P là *absent* nếu bất kỳ khớp support nào thiếu quá ngưỡng đó. Không bao giờ zero-fill (N12 của P2: SLiM 87.2 so với 49.6). Cầu nội suy dài hơn là chuyển động bịa ra (P1 §1.4).

### 7.3 Họ path theo kênh (C1)

Ký hiệu:
- $w$ = cổ tay; $e$ = khuỷu; $s$ = vai;
- $h_k$ = điểm tay thứ $k$ (0 = gốc, 4/8/12/16/20 = đầu ngón);
- $n$ = đầu mũi; $c$ = cằm.

Chỉ số D/N là tay trội / không trội. Path được biểu diễn trong khung thân, riêng chain ngón dùng **khung bàn tay**: gốc $h_0$, trục $h_0\to h_9$, đơn vị là độ dài lòng bàn tay có floor, như CONF của PLctx.

| Kênh | Path | d | dim log-sig2 | Tham số âm vị học nhắm tới |
|---|---|---:|---:|---|
| **D-LOC** | $w_D$ (khung thân) | 2 | 3 | path movement: hướng, độ cong, chiều quay |
| | chain cánh tay $[e_D-s_D,\ w_D-e_D]$ | 4 | 10 | phối hợp vai–khuỷu (tầm, cung) |
| **D-CONF** | 5 chain ngón, mỗi chain 4 cạnh tương đối trong khung bàn tay | 8 mỗi chain | 36 mỗi chain | handshape change, finger wiggle |
| | [góc trong mặt phẳng của $h_{D,5}-h_{D,17}$ đã unwrap, độ co ngắn có dấu lòng bàn tay] | 2 | 3 | **wrist twist** (§5.1, đã sửa sau review) |
| | aperture $[\lVert h_{D,4}-h_{D,8}\rVert,\ \lVert h_{D,4}-h_{D,12}\rVert]$ | 2 | 3 | mở/đóng bàn tay |
| **N-LOC, N-CONF** | Như D | | | tay không trội |
| **REL** (quan hệ, khác kênh) | $w_D - n$, $h_{D,8}-n$, $w_N-n$ | 2 mỗi path | 3 mỗi path | location / contact vùng mặt |
| | $[\sigma,\delta]$ với $\sigma=w_D+\mathcal M(w_N)$, $\delta=w_D-\mathcal M(w_N)$; $\mathcal M$ = gương qua trục giữa thân | 4 | 10 | tách năng lượng cùng pha / luân phiên ra hai trục; level 1 theo cửa sổ con (không sửa tree-like, xem §5.1) |
| | $w_D-w_N$ | 2 | 3 | contact hai tay, location = Hand |
| **Mặt** (tuỳ chọn, arm phụ) | [độ mở miệng, độ rộng môi] | 2 | 3 | mouthing; trọng số thấp |

**Sửa sau review (Major 5).** Bản trước đặt $w_D-n$ và $h_{D,8}-n$ vào D-LOC, nên support của D-LOC lẫn khớp D-CONF (đầu ngón trỏ) và khớp mặt (mũi). Mô phỏng với sampler thật của PLctx cho thấy:
- view far chỉ che 13 % cell S nhưng ẩn 62–71 % token P D-LOC;
- luật target ≥ 0.5 khi đó phụ thuộc số khớp, không phụ thuộc thời gian.

Bản này giữ mỗi token P D-LOC/D-CONF/N-LOC/N-CONF có support **nằm trọn trong khớp của một kênh S**, cộng anchor vai công khai. Mọi path chạm nhiều kênh dồn vào kênh **REL**. REL bị ẩn khi *bất kỳ* kênh nào nó chạm bị che ở các mẫu đó, và đây là hành vi mong muốn: dự đoán quan hệ từ ngữ cảnh.

**Vì sao như vậy [S]:**
- Mọi path đều ít chiều. Không ghép phẳng cả 21 điểm tay thành một path (N2 của P2: 903 chiều trở lên).
- Path chain ngón dùng cạnh tương đối, giống edge/chain của Path-JEPA.
- Path quan hệ **không** đi theo cây động học. Location và contact của sign là quan hệ giữa các bộ phận không kề nhau. Đây là khác biệt cốt lõi so với action recognition.

**Chiều của token P** (log-sig2 + basepoint + cờ có mặt theo path + độ tin cậy):

| Kênh | Thành phần | Tổng |
|---|---|---:|
| D-LOC | 3 + 10 = 13; basepoint $w_D$ (2); cờ (2); conf, coverage (2) | 19 |
| D-CONF | 5 × 36 + 3 + 3 = 186; cờ (7); conf (2) | 195 |
| REL | 3 × 3 + 10 + 3 = 22; basepoint của 3 path tay–mũi và $w_D-w_N$ (8); cờ (5); conf (2) | 37 |

Năm kênh P (D-LOC, D-CONF, N-LOC, N-CONF, REL) cho cùng số token như trước: 5 × 22 = 110.

### 7.4 Token (C2) và global view nhất quán Chen (C3)

**Thang thời gian.** Mọi clip được resample về **đúng 25 fps**: upsample tuyến tính cho clip 15 fps, không phải min(fps, 25) như PLctx. Có hai lý do. Upsample tuyến tính không đổi signature. Và nhờ đó cửa sổ có độ dài cố định **theo giây**: với min(fps, 25), một cửa sổ 8 mẫu ở clip 15 fps dài 0.53 s, không phải 0.32 s (review Major 8). Thang trên cửa sổ 64 mẫu:
- ℓ = 1: cửa sổ $W_1 = 8$ mẫu (0.32 s), stride 4 → 15 cửa sổ;
- ℓ = 2: cửa sổ $W_2 = 16$ mẫu (0.64 s), stride 8 → 7 cửa sổ.

Lý do: thời lượng ký hoạt động trung vị 1.34 s (§5.2). Một movement segment khoảng 0.2–0.5 s. Thang 0.32 s đủ ngắn để đi-về lặp lại không bị tree-like triệt tiêu, và đủ dài (8 điểm) để level 2 có nghĩa. Bin 2 mẫu của PLctx thì quá ngắn: diện tích của 2 điểm luôn bằng 0. Path-JEPA dùng 25/50 frame ở 30 fps, xấp xỉ độ dài cả sign (N4 của P2).

**Token P.** Với kênh $c$, thang ℓ, cửa sổ $k$:
$$
p_{c,\ell,k} = \Big[\ \phi\big(\mathrm{logsig}_2(x_p|_{[t_{\ell,k}, t_{\ell,k}+W_\ell]})\big)_{p\in\mathcal P_c}\ \Big\|\ b_{c,\ell,k}\ \Big\|\ m_{c,\ell,k}\ \Big\|\ \bar q_{c,\ell,k}\ \Big]
$$
- **Basepoint $b_{c,\ell,k}$** (sửa sau review, Critical 3): giá trị của path tại **mẫu đầu tiên của chính cửa sổ đó**, $x_p(t_{\ell,k})$. Nó chỉ lấy từ support của token. **Không** dùng trung bình toàn clip như descriptor D-SIG: trung bình đó đi qua mẫu bị che, sẽ rò rỉ và làm test ghi đè 1e6 thất bại. Có unit test riêng.
- $m_{c,\ell,k}$ = cờ có mặt theo từng path.
- Công thức đóng cho path tuyến tính từng đoạn $x_0..x_N$:
  - level 1 = $x_N-x_0$;
  - $A_{ij}=\tfrac12\sum_k[(x_k-x_0)_i\Delta_{k,j}-(x_k-x_0)_j\Delta_{k,i}]$.
  Hai `einsum`, khớp pySigLib tới 1e-14 [P1 §1.3].
- $\phi$ = robust z-score theo (kênh, thang, chiều) dùng median/IQR fit trên train, rồi $\mathrm{asinh}$ để hãm đuôi nặng của level 2.
- **Gating cửa sổ đứng yên** (review Minor 22): z-score sẽ khuếch đại nhiễu của cửa sổ nghỉ. Vì vậy, nếu năng lượng thô $\lVert\Delta x\rVert$ của path trong cửa sổ dưới ngưỡng ε (fit trên train: phân vị 20 % năng lượng của cửa sổ nghỉ), phần log-sig của path được đặt bằng 0 **sau** $\phi$ và gắn cờ "still". Nhờ vậy "signature của đoạn nghỉ = 0" còn đúng sau chuẩn hoá.
- $\bar q$ = độ tin cậy trung bình và tỉ lệ mẫu được quan sát.
- Embedding:
  $$\tilde p = \mathrm{MLP}_{type(c)}(p) + e_{chan}(c) + e_{scale}(\ell) + e_{stream}(P)$$
  với $\mathrm{MLP}$ riêng theo loại kênh (LOC / CONF / REL), như Eq. 5 của Path-JEPA.
- Vị trí thời gian cho RoPE = tâm cửa sổ tính theo đơn vị bin của luồng S, nên S và P nằm chung một trục thời gian.

**Token global G (một token mỗi kênh, 5 token):**
- Teacher: $\log S$ trên toàn bộ 64 mẫu.
- Student: $\log S$ của **chuỗi các đoạn nhìn thấy, bỏ khoảng trống** (gap-skipping):
  $$S_{vis}=\bigotimes_{k\ \text{visible}} S(\text{seg}_k)$$
  tính trên các đoạn 4 mẫu **dùng chung đầu mút**: đoạn $k$ là $[4k, 4k+4]$. Một đoạn là visible chỉ khi cả 5 mẫu của nó, kể cả hai đầu mút, nằm ngoài $\mathcal H$ đã giãn.
  - Review Minor 16: với đoạn không chồng nhau, increment *giữa* hai đoạn liên tiếp bị bỏ, nên ngay cả khi không che gì $G_{student}\neq G_{teacher}$.
  - **Teacher dùng đúng cấu trúc đó**, với mọi đoạn visible. Khi không che, $G_{student}=G_{teacher}$ chính xác.
  - Ở log-space bậc 2:
  - increment cộng lại;
  - area $A=\sum_k A_k+\tfrac12\sum_{k<l}a_k\wedge a_l$.
  Tính bằng cumsum, $O(K)$.
- Không chứa thông tin của đoạn bị che theo cấu trúc, **với điều kiện** "visible" được xét trên $\mathcal H$ đã giãn theo bán kính làm mượt (§7.5). Unit test: nhiễu toàn bộ mẫu bị che thì $G_{student}$ giữ nguyên bit-for-bit.
- Đây là cách sửa (b) trong N5 của P2. Arm A5 dùng global thật (rò rỉ) làm control dương cho H4. Arm A6 bỏ G hoàn toàn.

**Tổng số token:** 224 (S) + 110 (P) + 5 (G) + CLS = **340**, so với 225 của PLctx.
- Ở d = 256, phần tuyến tính chiếm ưu thế: attention/linear ≈ N/6d ≈ 0.22. Chi phí GPU vì vậy khoảng **1.5–1.6×**, không phải 2.3× như bản trước (review Minor 17).
- PLctx full160 chạy 13.8 bước/s, nên ước tính 8–9 bước/s với batch 32.
- **Rủi ro thật là CPU dataloader:** closure theo từng token và signature. Phải vector hoá trên GPU trong `features.py`, như PLctx, và profile ở Phase 1.

### 7.5 Masking và closure (C4)

**Support.**
- Cell S $(c,b)$: khớp sở hữu của kênh $c$ × mẫu $\{2b, 2b+1\}$.
- Token P $(c,\ell,k)$: các khớp dùng trong $\mathcal P_c$ × mẫu $[t_{\ell,k}, t_{\ell,k}+W_\ell)$.
- Path quan hệ dùng mũi/cằm, tức khớp của kênh mặt. Kênh REL chạm nhiều kênh S (tay D, tay N, mặt).
- **Support tính trong không gian frame nguồn, có tính bán kính của mọi phép trộn thời gian** (phát hiện khi tự rà soát). Mọi phép xử lý trộn các frame lân cận đều mở rộng support:
  - Savitzky–Golay 5 mẫu: ±2 mẫu;
  - nội suy khoảng trống: các frame quan sát dùng làm đầu mút nội suy;
  - resample do augmentation tốc độ: hai frame nguồn kề nhau.

  Nếu không làm vậy, một token P visible ngay cạnh block bị che sẽ mang thông tin của 1–2 mẫu bị che qua bộ lọc làm mượt. Đây là rò rỉ thật. Cách thực hiện: tính $\mathcal H$ ở frame nguồn, như PLctx đã propagate qua resample, rồi giãn (dilate) $\mathcal H$ thêm 2 mẫu **chỉ khi xét token P**. Phương án thay thế rẻ hơn là làm mượt *trong* từng cửa sổ (padding phản xạ biên cửa sổ), khi đó support đúng bằng cửa sổ. Cả hai phương án đều có unit test.

Anchor công khai: vai, như PLctx. Mũi và cằm **không** là anchor công khai: đầu mũi thuộc F_HD, cằm thuộc F_MO (`pj/joints.py`).

**Sinh mask** (thừa kế PLCM-lite, content-independent):
- View near: R2, block thời gian 4–8 bin trên mọi kênh, khoảng 50 % cell.
- View far: R3, một kênh tay trên span dài, khoảng 11 % cell.
- Arm A9: với xác suất 0.3, view far thay R3 bằng **articulator subtree trên toàn cửa sổ**: cả tay D (CONF + LOC), hoặc tay N, hoặc mặt. Đây là phiên bản kinematic subtree của Path-JEPA.

**Closure (Eq. 7 của Path-JEPA, tổng quát hoá).** Gọi $\mathcal H$ là tập (khớp, mẫu) bị che: hợp các support của cell S bị che, cộng closure mặt F_HD ⇒ F_UP, F_MO như PLctx.
- Token P bị **ẩn khỏi student** nếu $\mathrm{supp}\cap\mathcal H\neq\emptyset$.
- Nó là **target** nếu ít nhất 50 % **số mẫu** của cửa sổ có support bị che. Tỉ lệ tính theo thời gian, không theo số khớp: sau khi tách kênh REL, mỗi token đơn kênh có support thời gian rõ ràng. Với REL, "bị che" ở một mẫu nghĩa là bất kỳ kênh nào nó chạm bị che ở mẫu đó.
- Nếu không thoả, token chỉ bị *drop*: không vào student, không làm target. Đây là quy tắc chồng lấn **mới**. Nó khác "drop" của PLctx, vốn là drop ngẫu nhiên 10 % ngữ cảnh cộng propagation.
- Token G của student luôn nhìn thấy, nhưng chỉ chứa đoạn không thuộc $\mathcal H$ (§7.4).

**Ngân sách.** Tỉ lệ che tính trên cell S, như PLctx, để so sánh công bằng với REF.

Mô phỏng của reviewer với `pj.masks.sample_view` thật (32 bin hợp lệ, trước khi tách REL) cho thấy view near chỉ còn khoảng **26 % token ℓ1 và 13 % token ℓ2 visible**, tức khoảng một cửa sổ ℓ2 mỗi kênh. Hệ quả: trong view near, ℓ2 chủ yếu là target, và ngữ cảnh P đến từ ℓ1. Ta chấp nhận điều này và đăng ký trước. Hai việc đi kèm:
- (a) **Trước Phase 2**, đo lại tỉ lệ ẩn/target/drop theo kênh và thang sau khi tách REL, và ghi vào run card.
- (b) Nếu tỉ lệ token P visible trong view far dưới 70 %, hoặc trong view near dưới 15 % (ℓ1), thì dùng **ngân sách P riêng**: closure P chỉ áp các block thời gian có độ dài ≤ 4 bin của mask S. Đây là arm A1-pb, được báo cáo riêng, không thay A1.

Path-JEPA tối ưu ở 20–25 % khớp bị che. Số liệu tham khảo cho mức ẩn của P là 20–40 %.

**Cấm** (N6 của P2):
- che mọi kênh tại cùng một cửa sổ ngắn ở cả hai view (bias-inflation);
- mask dựa vào nội dung trong arm mặc định.

**Test rò rỉ bắt buộc** (thừa kế PLctx): ghi đè mọi giá trị trong $\mathcal H$ bằng 1e6 hoặc nhiễu → toàn bộ input student (S, P, G) phải giống bit-for-bit. Test phải chạy **sau** toàn bộ pipeline: smoothing, nội suy, resample tốc độ, gương hoá, chuẩn hoá. Không chỉ ở bước tạo token.

**Các rò rỉ đã xét và chấp nhận** (khai báo, như PLctx):
- (i) Tay trội canonical quyết định trên toàn clip: 1 bit.
- (ii) Vị trí mask độc lập nội dung, nên vị trí khoảng trống trong G không mang thông tin.
- (iii) Token P visible chồng frame với token P bị ẩn, ở các frame *không* thuộc $\mathcal H$. Đây là ngữ cảnh một phần hợp lệ, không phải rò rỉ.
- (iv) Thống kê chuẩn hoá $\phi$ fit trên train và đóng băng. Không có thống kê per-clip nào tính qua frame bị che. Nếu dùng thống kê per-clip (ví dụ độ rộng vai trung vị), nó chỉ được lấy từ anchor công khai.

### 7.6 Encoder, predictor, teacher, loss

**Encoder.** Thừa kế PLctx: d = 256, 6 block, 4 head, RoPE theo thời gian liên tục, CLS. Thêm **bias attention cấu trúc** (Eq. 6 của Path-JEPA, rẻ):
$$\alpha_{ij} \mathrel{+}= \beta_t\,\kappa(|t_i-t_j|) + \beta_{cc'}[c_i,c_j] + \beta_{\ell\ell'}[\ell_i,\ell_j]$$
- $\kappa$ = bucket khoảng cách thời gian.
- Ma trận học được 7+5 kênh × 7+5 kênh, và 3 × 3 cho thang (S, ℓ1, ℓ2).
- Bắt đầu tại 0, nên ban đầu tương đương PLctx.

**Predictor.** d = 128, 4 block, như PLctx. Query cho mỗi target:
- cell S: embedding (kênh, bin), như PLctx;
- token P: (luồng P, kênh, thang, tâm cửa sổ qua RoPE, $W_\ell$).

Không có giá trị target nào trong query.

**Teacher.** EMA của student, momentum 0.996 → 0.9999 (cosine). Thấy toàn bộ S, P và G thật. Target = trung bình LayerNorm của 4 layer cuối, như PLctx. Nối z-score nhiều block (N10) là arm phụ.

**Loss.**
$$\mathcal L = \mathrm{CE}_c(\hat z_S, z^*_S) + \lambda_P\,\mathrm{CE}_c(\hat z_P, z^*_P)$$
- Centered cross-entropy theo chiều feature, kiểu S-JEPA, với temperature và center như PLctx.
- Mặc định $\lambda_P = 1$. Tổng trên view near và far.
- Không có target signature tường minh (bài học S2). Không có CLS alignment (bài học G1).

**Optimizer.** Giống PLctx: AdamW 3e-4, wd 0.05, warmup 10 %, cosine, clip 1.0, bf16, batch 32.

### 7.7 Augmentation

Thừa kế PLctx: đổi tốc độ, affine camera, near/far lấy cửa sổ độc lập. Thêm:
- **Time-warp đơn điệu phi tuyến** (spline 4 nút, ±20 %), sửa sau review (Critical 4):
  - Áp **cùng một warp cho teacher và student của một view**, giống cách PLctx dùng chung `window_positions` trong `build_pair`. Nếu chỉ warp student, bin $b$ sẽ chỉ hai thời điểm khác nhau ở hai nhánh và target bị lệch.
  - Bản trước nói "P-token gần như không đổi". Điều đó sai: chỉ signature của *toàn* path mới bất biến reparam. Cửa sổ có độ dài mẫu cố định sẽ chứa đoạn khác nhau sau warp.
  - Muốn học bất biến tempo thì đó là mục tiêu riêng có căn chỉnh lại theo thời gian: Plan B3.
- **Morphology augmentation:** scale độ dài xương ±10 % theo từng clip. Trên PLctx trước đây chưa kết luận (+0.5) [I], nên giữ ở mức nhẹ.
- **Jitter tay:** dùng lại `keypoint_noise` của PLctx (0.02 × (1 − conf)), áp **trước** khi tính luồng P. Không thêm nguồn jitter mới (review Minor 20). Jitter tổng hợp σ ∈ {0.01, 0.02, 0.04} chỉ dùng ở đánh giá robustness (H6).

### 7.8 Readout (C6) và đánh giá

| Readout | Định nghĩa | Vai trò |
|---|---|---|
| `grp` | PLctx: mean token S theo nhóm (D, N, mặt), nối lại, 768 chiều | So sánh trực tiếp với REF |
| `grp+P` | `grp` ‖ mean token P theo nhóm (D, N, REL), 1536 → PCA 768 (fit trên gallery train, không nhãn) | Chính của PhonoPath |
| `mw` | Như `grp+P`, nhưng mỗi bin có trọng số $w_b \propto a_b\cdot(1+E^{CONF}_b)$, trong đó: <br>• $a_b=\mathrm{sigmoid}\big((y_{w_D,b}-y_{rest})/0.1\big)$ = cổ tay trội cao hơn mức nghỉ của clip (phân vị 10 % độ cao + 0.25 độ rộng vai, như D-TEMPO); <br>• $E^{CONF}_b$ = năng lượng token P của D-CONF (thay đổi handshape/hướng). <br>**Không** dùng năng lượng D-LOC: động tác nâng/hạ tay vào/ra tư thế nghỉ là path lớn nhất trong clip (P3 §4.2), nên sẽ được trọng số sai. Các hằng số cố định trước | Activity weighting không nhãn (H9) |
| `ord2` | `grp+P` trên hai nửa của đoạn hoạt động, nối lại | Giữ thứ tự (H10) |

Clip có nhiều cửa sổ 64 mẫu (stride 2 khi test, như PLctx): lấy trung bình embedding các cửa sổ rồi L2. Readout chính được **đăng ký trước** là `grp+P`. `mw` và `ord2` báo cáo phụ; chọn trên VAL-DICT, không bao giờ chọn trên test.

### 7.9 Độ phức tạp và rủi ro số học

- **Tính signature:** 110 token × khoảng 30 path × ≤ 16 điểm. Không đáng kể trên GPU (P1: 0.137 s CPU cho 9.600 path). Tính on-the-fly trong `features.py`, fp32, autocast tắt, như PLctx.
- **Level 2 với path 8D (chain ngón):** 28 hệ số mỗi path. Chuẩn hoá theo từng chiều.
- **Rủi ro chính là nhiễu:** xem H6 và §11.

---

## 8. Các arm và baseline

Mọi arm fork từ `phonojepa`: cùng dữ liệu, cùng mask sampler cho luồng S, cùng optimizer và lịch học, cùng evaluator. Screening: 1 seed, toàn bộ 40.154 clip train, **40 epoch** (khoảng 50k bước × 32).

| Arm | Mô tả | Giả thuyết | Khác A1 ở đâu |
|---|---|---|---|
| **A0 = REF** | PLctx, cùng budget 40 epoch | Anchor | Không có luồng P/G |
| **A1** | PhonoPath-JEPA đầy đủ (§7), global Chen-visible | H1, H2 | — |
| **A2** | Như A1, nhưng thay phần **Lévy area** (phản đối xứng) của mỗi path bằng **moment bậc hai đối xứng cùng chiều** trong cửa sổ: covariance của increment (tương đương QV), lấy $d(d+1)/2$ hệ số rồi cắt hoặc pad về đúng $d(d-1)/2$. Level 1 và basepoint giữ nguyên. Bản cũ (vận tốc trung bình) gần như trùng với level 1, nên thực chất có ít chiều hiệu dụng hơn (review Major 7) | H3 (control khớp token và chiều) | Thay area bằng moment đối xứng |
| **A3** | **Path-JEPA trung thành nhất có thể**: chỉ token P (bỏ luồng S), mỗi token có basepoint + trung bình vị trí trong cửa sổ; mask kiểu Path-JEPA: seed theo chuyển động + mở rộng subtree khớp, mọi thời điểm, khoảng 25 % khớp (rò rỉ của seed theo nội dung được khai báo, như A10); global [0,T] đầy đủ, vì không có mask thời gian nên không rò rỉ | Tham chiếu "port thẳng" | Không có S; mask khác |
| **A4** | A1 bỏ họ path quan hệ (kênh REL): token absent, cùng số token | H7 | — |
| **A5** | A1 với global thật [0,T] cho student (rò rỉ) | H4 (control dương) | G không Chen-visible |
| **A6** | A1 bỏ token G | H4, N5(a) | Không G |
| **A7** | A1 thêm kênh thời gian vào mọi path P | H5 | +time |
| **A8a** | A1 không làm mượt (giữ asinh) | H6 | Raw smoothing |
| **A8b** | A1 không asinh (giữ làm mượt) | H6 | Raw scaling |
| **A9** | A1 + articulator subtree trong view far (p = 0.3) | H8 | Mask |
| **A10** | A1 + **seed mask theo chuyển động** kiểu Path-JEPA/S-JEPA: chọn kênh và khoảng thời gian bị che với xác suất tỉ lệ năng lượng chuyển động. Đây là thành phần mask có lợi lớn nhất trong Path-JEPA (84.1 → 86.8, Table 5). **Có rò rỉ nhẹ, đã khai báo**: vị trí mask phụ thuộc nội dung | H8 (mở rộng) | Mask phụ thuộc nội dung |

**Arm phụ** (chỉ chạy nếu còn ngân sách sau Phase 2):
- lead–lag chuẩn trên path cổ tay (cho repeated);
- **projection tuyến tính học được trước signature** (LogSig-RNN; Cheng et al. TMM 2024): path $x\mapsto Wx$ với $W$ học được, rồi mới lấy logsig2. Cho phép mô hình tự chọn tổ hợp khớp, tức "learnable path";
- loss cosine của Path-JEPA thay cho centered-CE;
- bias cấu trúc tắt;
- kênh mặt P;
- $\lambda_P\in\{0.5, 2\}$;
- target nối nhiều block (N10);
- KoLeo 0.1 trên embedding clip (N11; chú ý G1 âm).

**Baseline ngoài, đã có số [I]:**
- PLctx full160 (test 16.68);
- Pose-JEPA (5.74 raw);
- ASM-JEPA (1.07);
- descriptor không cần train (B0 = 13.46 OFF-val; D-SIG §5.3);
- SignRep public (49.02 weighted, 37.47 avg);
- MASA khoảng 28, SignSeek 57.66 (pose, *có* nhãn gloss).

---

## 9. Chương trình thực nghiệm

### 9.1 Phase 0 — chẩn đoán không cần train (đã làm một phần)

- **Đã có:** D-TEMPO, D-SIG, D-LEX (§5).
- **Còn làm trước Phase 2** (CPU, khoảng 2 giờ, trừ mục d):
  - (a) **SNR level 2 theo họ path.** So sánh area trên clip gốc với clip thêm jitter σ, và với smoothing SG5/không; chọn σ của augmentation.
  - (b) **Tỉ lệ token P absent** sau observation filter — **đã chạy** (`diagnostics/absent_rate.py`, `results/absent_rate.json`, val).
    - Quy tắc "mọi khớp support được quan sát, trừ tối đa 2 frame".
    - D-LOC (khớp thân + mũi) có mặt ở 96.7 % cửa sổ 8 mẫu và 94.3 % cửa sổ 16 mẫu.
    - D-CONF (21 khớp tay) chỉ có mặt ở 67.0 % và 55.3 % trên mọi cửa sổ. Nhưng trên cửa sổ *hoạt động* (cổ tay trên mức nghỉ ít nhất nửa cửa sổ; chỉ khoảng 26–27 % số cửa sổ) thì lên 91.6 % và 84.1 %.
    - Phần thiếu chủ yếu là tay nằm ngoài ảnh khi nghỉ.
    - → Presence phải tính **theo từng path trong token** (cờ riêng + giá trị 0 sau chuẩn hoá), không loại cả token khi một path tay thiếu. Ví dụ path $h_{D,8}-n$ của REL phụ thuộc tay. Đa số cửa sổ là nghỉ, nên token P nghỉ sẽ chiếm đa số; đây là lý do loss P nên được chuẩn hoá theo số target.
  - (c) **H9 trên checkpoint có sẵn:** readout `mw` cho PLctx full160 (`phonojepa/runtime/...`) trên VAL-DICT, không train.
  - (e) **Làm lại D-SIG với mốc mạnh** (review Major 13): B0 của proposal12 so với B0 + signature, cửa sổ 0.32/0.64 s ở 25 fps, gallery đầy đủ 40k, trọng số block theo phương sai giải thích. Mục đích là biết signature có thêm gì vào descriptor training-free mạnh nhất hiện có hay không.
  - (f) **Thứ hạng 40 epoch có dự báo 160 epoch không?** Nếu PLctx full160 có checkpoint trung gian (hoặc sau Phase 2 train lại có mốc), đo Kendall τ giữa VAL-DICT ở epoch 40 và 160. Nếu không có, coi screening 40 epoch là giả định đã khai báo.
  - (d) **Cổng G0 — pre-screen có giám sát** (khoảng 1–2 GPU-h). Đây là bước quyết định có tốn GPU cho SSL hay không.
    - Train một encoder nhỏ (d = 256, 4 block), **có giám sát** ISLR, chọn bằng val, 3 input:
      - (i) chỉ token S (PLctx);
      - (ii) S + P signature;
      - (iii) S + P moment đối xứng (A2).
    - **Chế độ nhãn** (sửa sau review, Major 10): 10 % nhãn chỉ là khoảng 1.5 clip/gloss, quá thoái hoá. Thay bằng:
      - (α) **40 % nhãn** của toàn bộ 2.731 gloss, khoảng 6 clip/gloss;
      - (β) **k-shot k = 5 trên tập cố định 300 gloss**, chọn ngẫu nhiên trước, không xem kết quả.
      - Mỗi chế độ ≥ 2 seed. Ngoài ra chạy 100 % nhãn để tham khảo.
    - Cùng số bước, cùng augmentation.
    - Câu hỏi: khi mô hình được phép tự do khai thác bất kỳ tín hiệu nào, token signature có giúp không?
    - **Vai trò của G0 là veto, không phải bằng chứng.** Luật (trung bình seed, chế độ α và β, bootstrap theo signer × gloss):
      - nếu (ii) và (iii) đều **thấp hơn** (i) quá 2 lần SD seed ở cả α và β: luồng P gây hại rõ ngay cả khi có nhãn. **Không chạy luồng P trong SSL**, chuyển sang Plan B (§9.9);
      - nếu (iii) ≥ (ii) rõ rệt: Phase 2 ưu tiên A2 (rẻ hơn), A1 thành arm phụ;
      - mọi trường hợp khác: mở Phase 2 với A1 làm arm chính.
    - G0 không kiểm tra được phía target: token P làm target latent. Đó là việc của Phase 2.
    - Nhãn chỉ dùng cho chẩn đoán này. Không model SSL nào được khởi tạo từ checkpoint của nó.
    - **Hạn chế của G0** (tự rà soát): lợi ích có giám sát và lợi ích SSL không trùng nhau.
      - Với khoảng 15 clip/gloss có nhãn, mô hình S có thể tự học movement. Inductive bias của signature sẽ ít lộ ra.
      - → Vì vậy dùng các chế độ ít nhãn α và β ở trên; lợi ích của inductive bias thường rõ nhất khi ít nhãn.
      - G0 là heuristic tiết kiệm GPU, không phải bằng chứng. Nếu G0 âm nhưng người dùng vẫn muốn thử, A1/A2/A0 vẫn chạy được như một mini-Phase 2 (khoảng 6 GPU-h).

### 9.2 Phase 1 — hạ tầng (khoảng 1–2 ngày code, không cần GPU dài)

1. `method/phonopath_jepa/` fork `phonojepa/pj`. Các module mới:
   - `pp/signature.py`: logsig2, Chen gap-skipping, lead–lag, asinh-robust;
   - `pp/paths.py`: họ path §7.3;
   - `pp/ptokens.py`: cửa sổ, support, absent;
   - `pp/closure.py`: (khớp × mẫu) → ẩn/target/drop;
   - `pp/model.py`: luồng P/G, bias cấu trúc;
   - `pp/extract.py`: các readout §7.8.
2. **Unit test bắt buộc:**
   - logsig2 so với oracle pySigLib (cài trong venv tạm, không đưa vào venv chính);
   - bất biến reparameterization (upsample ×4 cho cùng kết quả);
   - Chen gap-skipping khớp oracle;
   - **test rò rỉ** (§7.5) cho S, P, G;
   - gương hoá đổi dấu area và được hoán đổi D/N đúng;
   - closure: không token nào visible có support giao $\mathcal H$ (đã giãn);
   - basepoint chỉ lấy từ support của token;
   - $G_{student}=G_{teacher}$ chính xác khi không che gì;
   - cùng time-warp cho teacher và student của một view.
3. Profile tốc độ trên GPU khi có slot trống. Mục tiêu ≥ 5 bước/s batch 32.
4. Evaluator: dùng lại `pj/frozen_dictionary.py` (VAL-DICT và test) cùng probe ASL-LEX của `diagnostics/asllex_probe.py`, chuyển sang embedding.

### 9.3 Phase 2 — screening (1 seed, 40 epoch, khoảng 10 arm)

- **A0 chạy 3 seed trước** (khoảng 4–5 GPU-h) để biết SD seed ở 40 epoch, trước mọi so sánh (review Major 9).
- Thứ tự chạy theo ưu tiên: **A0 × 3, A1, A2** → A5, A6 → A4, A9 → A3, A7, A8a, A8b, A10.
- Ước tính khoảng 1.3–2 GPU-h mỗi arm (1.5–1.6× PLctx), cộng khoảng 0.3 GPU-h đánh giá mỗi arm (VAL-DICT + probe ở 3 mốc). Tổng **22–30 GPU-h**.
- Mỗi run lưu checkpoint EMA ở 10/20/40 epoch, và chấm VAL-DICT (R@1, R@5, DCG) + probe ASL-LEX ở mỗi mốc.
- **Cổng G1 (sau A0, A1, A2):**
  - Nếu A1 < A0 − 1 và A2 < A0 − 1: luồng P làm hại (có thể do nhiễu hoặc capacity). Chạy A8a/A8b và arm $\lambda_P = 0.5$ trước khi kết luận.
  - Nếu A1 ≈ A2 > A0: lợi ích đến từ token chuyển động, không phải signature. Ghi nhận H3 bị bác bỏ; tiếp tục với dạng rẻ hơn.

### 9.4 Luật quyết định (thừa kế `docs/research`, điều chỉnh)

So sánh cặp trên cùng query val (10.304). Checkpoint (40 epoch) và readout được **cố định trước**.

Sửa sau review (Major 9):
- Val có 6 signer với số query [2, 609, 697, 2997, 2999, 3000]. Thực chất là **5 cụm**, và ba signer giữ 87 % query.
- Permutation test ở mức query bỏ qua cấu trúc cụm, nên quá lạc quan.

Vì vậy:
- **Ngưỡng hiệu ứng** $\tau=\max(1.5,\ 2.5\times \mathrm{SD}_{seed}(A0))$, với SD đo từ 3 seed A0 ở Phase 2.
- **Kiểm định:** bootstrap phân tầng theo **signer × gloss**, 2.000 lần. CI 95 % của Δ phải loại trừ 0.
- **KEEP:** Δ R@1 ≥ τ, CI bootstrap loại trừ 0, và ≥ 4/5 signer có ≥ 600 query cải thiện.
- **KEEP-MECH:** Δ R@1 trong ±1.5 nhưng probe cơ chế đăng ký trước tăng (ví dụ H1). Giữ lại để tổ hợp.
- **INCONCLUSIVE:** không thoả hai mức trên, và cũng không thoả DROP.
- **Nhiều phép so sánh:** Phase 2 có khoảng 9 so sánh với A1 hoặc A0.
  - Áp Holm–Bonferroni trên họ so sánh *retrieval* đã đăng ký (H2, H3, H4, H7, H8), α = 0.05 cho cả họ.
  - Probe cơ chế báo cáo không hiệu chỉnh nhưng gắn nhãn "thăm dò".
  - Với 1 seed, mọi KEEP ở Phase 2 chỉ là ứng viên. Chỉ Phase 3 (3 seed) mới cho kết luận.
- **Readout cùng chiều:** mọi so sánh retrieval chính dùng embedding 768 chiều.
  - **So sánh chính** (review Major 6): A1-`grp` với A0-`grp`, cùng 768 chiều, không PCA. Nó trả lời "luồng P có làm token S tốt hơn qua attention và target không".
  - **So sánh phụ:** A1-`grp+P` với A0-`grp`. Cả hai đi qua **cùng một pipeline** centering + PCA 768 fit trên gallery train. A0 cũng PCA (768 → 768, chỉ xoay và center), vì centering riêng nó đã đổi cosine R@1.
- **DROP:** Δ ≤ −1.5 (có hại), hoặc TOST tương đương trong ±1 với p < 0.05.

### 9.5 Phase 3 — xác nhận và tổ hợp

- Lấy 1–2 cấu hình tốt nhất cộng A0. **3 seed ở 40 epoch** (khoảng 15 GPU-h).
- Nếu hai thành phần độc lập cùng KEEP (ví dụ H7 và H8), chạy tổ hợp 2×2.
- Kết quả: cấu hình cuối, đóng băng trước Phase 4.

### 9.6 Phase 4 — budget đầy đủ và scale

1. **PhonoPath full160:** 160 epoch, 1 seed, khoảng 8–10 GPU-h, cùng budget với PLctx full160. Chấm VAL-DICT theo mốc 40/80/120/160.
2. **Test một lần:** ASL Citizen frozen dictionary + WLASL2000. Báo đủ ba readout, readout chính đăng ký trước.
3. **Scale corpus** (tuỳ chọn, cần người dùng duyệt): ASL Citizen + WLASL train + MS-ASL train + How2Sign train, khoảng 105 h (D1 §6).
   - Lọc nhiễm chéo: bỏ clip MS-ASL trùng nguồn YouTube với test WLASL, và ghi rõ 56 % test MS-ASL local có nguồn trong YouTube-ASL (D1).
   - Cache FP16 khoảng 4.5 GB.

### 9.7 Phase 5 — downstream đầy đủ

- **ISLR ASL Citizen:**
  - linear probe đóng băng;
  - fine-tune 30k bước, cùng recipe với `asm_jepa/finetune` để so với ASM-JEPA 55.4 và scratch 53.8;
  - few-shot k = 1, 5.
- WLASL100/300/2000 fine-tune. Ghi rõ đây là signer-dependent.
- MS-ASL1000 local. Ghi rõ chỉ có 74 % clip, không so trực tiếp với số trong paper.
- **Robustness:**
  - fps thật 10/15/25: đổi bước resample, không nội suy lại;
  - bỏ ngẫu nhiên 10/30 % frame;
  - jitter tay σ.
  So A1 với A0. Đây là phiên bản sign của Table 4 Path-JEPA.

### 9.8 Ngân sách

| Phase | GPU-h | Ghi chú |
|---|---:|---|
| 0 | 3–5 | CPU (a–c, e) + cổng G0 có giám sát (d): 3 input × 2 chế độ × 2 seed + 100 % nhãn |
| 1 | khoảng 1 | profile, smoke |
| 2 | 22–30 | A0 × 3 seed + khoảng 10 arm, gồm cả đánh giá |
| 3 | 12–18 | 3 seed × 2 cấu hình (+ tổ hợp) |
| 4 | 9–12 (+10–15 nếu scale corpus) | |
| 5 | 6–10 | fine-tune, robustness |
| **Tổng** | **khoảng 55–75** (không tính scale corpus) | |

Bản tiết kiệm: bỏ A3, A7, A8a, A8b, A10 và chỉ 1 seed ở Phase 3, còn khoảng 30 GPU-h.

### 9.9 Plan B: dùng ý tưởng Path-JEPA mà không có luồng signature

Áp dụng nếu G0 (hoặc G1 ở Phase 2) cho thấy token P không giúp. Ba thành phần của Path-JEPA không phụ thuộc signature. Chúng đều có hậu thuẫn: ablation của Path-JEPA (P0 §1.3) và nguyên tắc N5–N7 của P2.

| Arm Plan B | Ý tưởng | Kiểm định |
|---|---|---|
| B1 | **Token quan hệ không signature** cho luồng S: thêm vào kênh LOC các khoảng cách / vector tay–mặt, tay–tay *theo bin*. Path-JEPA cho thấy phần tăng chủ yếu từ edge/chain (+1.8); Pose-JEPA bổ sung PLctx ở body/relations [I] | so với A0, H7 |
| B2 | **Closure + articulator subtree** trong view far của PLctx (A9 không có luồng P) | H8 |
| B3 | **Bất biến tempo bằng mục tiêu:** cặp view của cùng clip với time-warp mạnh khác nhau; teacher thấy bản gốc, student thấy bản warp, dự đoán latent theo thời gian đã căn lại | H5 trên luồng S |
| B4 | Readout `mw` / `ord2` trên PLctx (không train) | H9, H10 |

Plan B dùng lại toàn bộ hạ tầng của Phase 1 (closure, readout, evaluator, probe) và chỉ tốn khoảng 10–15 GPU-h.

---

## 10. Đóng góp dự kiến (nếu giả thuyết đứng vững)

1. **JEPA path-signature đầu tiên cho sign language**, và là skeleton SSL đầu tiên dùng path quan hệ ngoài cây động học (tay–mặt, hai tay tổng/hiệu).
2. **Global view nhất quán Chen:** cách sửa rò rỉ có cơ sở toán học khi kết hợp token signature global với temporal masking. Áp dụng được cho mọi token signature.
3. **Token lai trạng thái + chuyển động** có bằng chứng tách bạch: control increment khớp compute (H3), bỏ họ quan hệ (H7).
4. **Phân tích định lượng tempo là nhiễu theo signer** (D-TEMPO) và readout trọng số chuyển động không nhãn.
5. **Probe âm vị học ASL-LEX theo tham số**, đánh giá cơ chế cho mọi encoder sign. Đây là công cụ tái sử dụng được.

---

## 11. Rủi ro và cách giảm

| Rủi ro | Xác suất | Ảnh hưởng | Giảm thiểu / kế hoạch B |
|---|---|---|---|
| Nhiễu tay làm level 2 vô nghĩa | Cao | Luồng P chỉ thêm nhiễu | Smoothing, asinh, gating, jitter-aug (H6). Nếu A1 ≈ A2: giữ dạng increment, rẻ hơn |
| Luồng P lấn át luồng S (shortcut) | Vừa | Handshape giảm | Theo dõi probe handshape; $\lambda_P = 0.5$; bias cấu trúc khởi tạo 0 |
| Compute 1.5–1.6× và dataloader CPU làm chậm screening | Vừa | Ít arm hơn | Ưu tiên A0 × 3, A1, A2, A5, A6; tính signature/closure vector hoá trên GPU; batch 32 giữ nguyên; SDPA |
| Lợi ích chỉ có ở pilot | Vừa | Không chuyển sang test | VAL-DICT trên full gallery; test chấm một lần |
| Lẫn compute và capacity | Vừa | Kết luận sai về signature | A2 khớp token, cửa sổ, chiều |
| Chen gap-skipping vẫn mất thông tin hữu ích (bước nhảy qua gap) | Thấp | G yếu | So sánh A6 (không G). **Không** thêm feature "bước nhảy qua gap": level 1 của bước nhảy giữa hai đầu mút nhìn thấy chính là level 1 của cửa sổ bị che, tức một shortcut trực tiếp tới target (review Minor 16) |
| Không vượt PLctx full160 trên test dù screening tốt | Vừa | Không có claim chính | Vẫn có claim cơ chế (H1, H5, H10). Phase 4 scale corpus |
| Path-JEPA phát hành code và khác cách hiểu của ta | Thấp | Cần sửa A3 | A3 ghi rõ là "port theo PDF". Cập nhật khi có code |

---

## 12. Kế hoạch triển khai (khi người dùng duyệt)

- **Thư mục:** `method/phonopath_jepa/`, package `pp`. Chạy bằng `.venv-signrep`, tách runtime khỏi `phonojepa`.
- **Dữ liệu:** dùng lại cache full-train của PLctx (`phonojepa/scripts/build_full_train_cache.py`). Nếu cần observation filter mới, dùng cache của `asm_jepa/data/processed/aslc/poses`, có sẵn train / val / test.
- **Runner:** tuần tự, có watchdog kiểu `phonojepa/scripts/schedule.py`; chờ GPU có ≤ 1 job khác của người dùng; `setsid nohup`; mỗi arm có `manifest.json`, `train_log.jsonl` và eval JSON.
- **Test split:** chỉ chấm ở Phase 4, một lần, ghi vào ledger các lần dùng test trước đó.
- **Code review:** hai vòng độc lập trước khi chạy Phase 2, như proposal12. Trọng tâm: rò rỉ, closure, chuẩn hoá chỉ fit trên train.

---

## 12b. Nhật ký rà soát (v1.1, 2026-10-16)

**Vòng 1: tự rà soát.**
- Rò rỉ qua làm mượt / nội suy / resample.
- Test rò rỉ phải chạy sau toàn bộ pipeline.
- Holm cho nhiều phép so sánh.
- Readout `mw` trọng số sai vào đoạn nâng/hạ tay.
- Hai nhận định sai trong P0: Lévy area chéo hai tay, và lead–lag.

**Vòng 2: review độc lập.** Một agent reviewer đọc proposal, literature, script và JSON, chạy mô phỏng mask bằng `pj.masks.sample_view` thật. Tìm ra 4 Critical, 11 Major, 8 Minor. Reviewer xác nhận đúng: mọi số Path-JEPA ở §2, mọi số D-SIG/D-LEX so với JSON, các số PLctx, số token/cửa sổ, chiều token P và tổng ngân sách.

| # | Vấn đề | Đã xử lý ở |
|---|---|---|
| C1 | D-LEX v1/v2 không rời gloss, nên probe có thể thắng nhờ nhận ra gloss; "static" chứa std (biên độ chuyển động) | **Chạy lại D-LEX v3** rời gloss + rời signer, cùng PCA, có static chỉ trung bình, CI bootstrap theo gloss (§5.4). Kết luận âm giữ nguyên; thêm phát hiện std vị trí là feature movement mạnh nhất |
| C2 | Probe tự gán nhãn cw/ccw là vòng tròn với input của A1 | Bỏ khỏi tiêu chí H1; chỉ dùng lại với nhãn độc lập (§6) |
| C3 | Basepoint chưa định nghĩa; nếu là trung bình clip thì rò rỉ | Basepoint = mẫu đầu của chính cửa sổ; có unit test (§7.4) |
| C4 | Time-warp chỉ ở student làm lệch target; P-token không bất biến với warp | Cùng warp cho teacher và student của một view; bất biến tempo chuyển sang Plan B3 (§7.7) |
| M5 | D-LOC lẫn khớp tay và mặt, nên view far ẩn 62–71 % token P; near view còn rất ít ℓ2 | Tách kênh REL; target theo tỉ lệ thời gian; đo tỉ lệ ẩn trước Phase 2; arm A1-pb với ngân sách P riêng (§7.3, §7.5) |
| M6 | Readout A0 với A1 lệch do PCA/centering | So sánh chính A1-`grp` với A0-`grp`; so sánh phụ cùng pipeline PCA (§9.4) |
| M7 | A2 cũ (vận tốc) trùng với level 1 | A2 = moment đối xứng bậc hai cùng chiều (§8, H3) |
| M8 | Cửa sổ không cố định theo giây; kết quả "kênh thời gian" của D-SIG không chuyển được sang A7 | Resample cố định 25 fps; H5 viết lại; giới hạn ghi ở §5.3.5 |
| M9 | Thống kê yếu: 5 cụm signer thực, permutation ở mức query, SD seed chưa biết | A0 × 3 seed; τ = max(1.5, 2.5 SD); bootstrap signer × gloss (§9.3, §9.4) |
| M10 | G0 10 % nhãn quá thoái hoá | 40 % nhãn và 5-shot trên 300 gloss, ≥ 2 seed; G0 chỉ là veto (§9.1d) |
| M11 | Lợi thế "ít signer" chọn lọc | Viết lại: chưa chứng minh ở cùng mức hữu ích (§0, §5.3) |
| M12 | D-TEMPO nói quá ("hoàn toàn") | Thêm R² riêng phần của gloss trong cùng signer: 10–17 % (§5.2) |
| M13 | D-SIG có mốc yếu (4.02 so với B0 13.46) | Ghi giới hạn; thêm Phase 0(e) với B0 + signature (§5.3, §9.1) |
| M14 | Wrist twist: area ~θ³; ở 2D chủ yếu là co ngắn | Path [góc unwrap, độ co ngắn có dấu]; 3D là hướng tương lai (§5.1, §7.3) |
| M15 | Tổng/hiệu không sửa được tree-like | Lý giải lại: tách trục năng lượng + level 1 theo cửa sổ con; lead–lag cross-QV là arm phụ (§5.1, §7.3) |
| m16 | Đoạn Chen không chồng nhau bỏ increment giữa đoạn; feature bước nhảy là shortcut | Đoạn dùng chung đầu mút, teacher cùng cấu trúc; bỏ feature bước nhảy (§7.4, §11) |
| m17 | Chi phí 2.3× là quá cao; chưa tính dataloader và đánh giá | 1.5–1.6×; thêm chi phí đánh giá vào ngân sách (§7.4, §9.3, §9.8) |
| m18 | Thiết kế không phải factorial; A8 đổi hai yếu tố | Gọi đúng là một-yếu-tố-mỗi-lần; tách A8a/A8b (§0, §8) |
| m19 | Cằm thuộc F_MO; "drop" khác PLctx; C6 mâu thuẫn §7.8 | Đã sửa (§7.5, §0) |
| m20 | Jitter trùng `keypoint_noise` | Dùng lại `keypoint_noise` (§7.7) |
| m21 | Thiếu seed mask theo chuyển động (+2.7 trong Path-JEPA) | Arm A10 (rò rỉ khai báo), và đưa vào A3 (§8) |
| m22 | Chuẩn hoá khuếch đại nhiễu cửa sổ nghỉ | Gating "still" trước khi dùng (§7.4) |
| m23 | Claim "chưa có công trình" không kiểm chứng được | Làm mềm (§0) |
| Ý tưởng thiếu | Projection học được trước signature; hướng tay 3D; thứ hạng 40 so với 160 epoch | Arm phụ (§8); §5.1; Phase 0(f) |

Trước khi chạy Phase 2, cần thêm một vòng review code tập trung vào closure, rò rỉ và chuẩn hoá chỉ fit trên train (§12).

## 13. Tài liệu tham khảo chính

Danh sách đầy đủ, có ID đã kiểm tra, nằm trong P0–P3.

- Singh & Singh. Path-JEPA: Path Signature Based Predictive Learning for Skeleton Action Recognition. ECCV 2026, DOI 10.1007/978-3-032-37258-1_7.
- Wong, Camgoz, Bowden. SignRep: Enhancing Self-Supervised Sign Representations. ICCV 2025, arXiv 2503.08529.
- Abdelfattah & Alahi. S-JEPA. ECCV 2024.
- Assran et al. I-JEPA (CVPR 2023); Bardes et al. V-JEPA (2024); V-JEPA 2 (2025).
- Mao et al. MAMP (ICCV 2023). Sun et al. GFP (ICCV 2025).
- Lyons & McLeod. Signature methods in machine learning, arXiv 2206.14674.
- Morrill, Fermanian, Kidger, Lyons. A generalised signature method for multivariate time series feature extraction, arXiv 2006.00873.
- Chevyrev & Oberhauser. Signature moments to characterize laws of stochastic processes. JMLR 2022.
- Liao et al. LogSig-RNN (BMVC 2021). Jiang et al. GCN-DevLSTM (2024). Moreno-Pino et al. Rough Transformers (NeurIPS 2024). Cheng et al. Learnable paths + signature (IEEE TMM 2024). Li et al. AAAI 2019.
- Desai et al. ASL Citizen (NeurIPS 2023 D&B). Li et al. WLASL (WACV 2020). Joze & Koller MS-ASL (BMVC 2019). Duarte et al. How2Sign (CVPR 2021).
- Sehyr et al. ASL-LEX 2.0 (2021). Battison (1978). Brentari (1998). Liddell & Johnson (1989).
- Nội bộ: `docs/proposal12/PhonoJEPA_Method_Proposal_VI.md`; `docs/research/PLctx_full160_Method_and_Results_VI.md`; `docs/research/Skeleton_SSL_Sign_Research_Design_VI.md`; `method/phonojepa_s2/runtime/eval/S2_FULL_4k_REPORT_VI.md`; `method/signpathlab/docs/DECISION_FINAL_VI.md`; `docs/experiments/pose_jepa_results_20261015/REPORT_VI.md`; `method/asm_jepa/runs/final_s0/final_report.md`.
