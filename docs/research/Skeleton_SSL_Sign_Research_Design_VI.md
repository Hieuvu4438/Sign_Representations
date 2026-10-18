# Đề xuất hướng nghiên cứu: Học biểu diễn ngôn ngữ ký hiệu tự giám sát từ skeleton
## Từ SignRep (RGB) đến một phương pháp skeleton-SSL / JEPA có kiểm chứng từng tín hiệu

| Mục | Nội dung |
|---|---|
| Ngày | 2026-10-14 |
| Trạng thái | **Đề xuất thiết kế** (chưa implement, chưa chạy) |
| Dữ liệu dev | ASL Citizen (train/val); test chỉ dùng một lần ở cuối |
| Đầu vào | Báo cáo literature L1–L4 (`docs/proposal12/literature/`), ba báo cáo literature mới (`docs/research/literature/`), toàn bộ kết quả nội bộ đến 2026-10-14 |
| Quy ước tag | **[V]** số liệu đã grep từ văn bản gốc của paper; **[V\*]** số lấy từ bảng của một paper khác; **[S]** chỉ có từ tóm tắt hoặc tìm kiếm; **[I]** đo nội bộ trong repo này; **[K]** kiến thức nền chưa kiểm tra lại |

**Mục lục và cách đọc**

| Mục | Nội dung | Đọc khi |
|---|---|---|
| 0 | Tóm tắt điều hành | luôn đọc |
| 1 | Bài toán, ràng buộc, tiêu chí thành công L0–L3 | luôn đọc |
| 2 | SignRep đọc kỹ: phương pháp, ablation, giới hạn, bài học cho skeleton | khi cần định vị so với SignRep |
| 3 | Literature review: sign SSL, skeleton SSL, JEPA, masking, chẩn đoán; 12 nguyên tắc thiết kế | khi cần căn cứ |
| 4 | Bằng chứng nội bộ: kế thừa / bỏ / chưa kết luận, các khoảng trống, không gian chưa thử | trước khi đề xuất arm mới |
| **5** | **Hệ thống giả thuyết H0, H1–H12**: dự đoán, tiêu chí bác bỏ, control, ma trận giả thuyết × thực nghiệm | **lõi khoa học** |
| 6 | Khung đo lường: VAL-DICT, metric cơ chế M1–M9, pre-screen, luật KEEP/DROP | khi phân tích kết quả |
| 7 | Không gian thiết kế: các họ O/K/V/I/R/S và bảng ưu tiên | khi chọn arm |
| 8 | Năm luận điểm method T-A…T-E | khi viết paper |
| **9** | **Chương trình thực nghiệm Phase A–F**, có cổng, kèm budget | **khi chạy** |
| 10 | Cây quyết định, ledger, quy tắc vận hành, anti-pattern | khi chạy |
| 11 | Tài liệu tham khảo; Phụ lục A–C: công thức, run card, ánh xạ arm cũ | tra cứu |

## 0. Tóm tắt điều hành

**Mục tiêu.** Xây một encoder **skeleton** tự giám sát, không dùng nhãn gloss, cho biểu diễn ngôn ngữ ký hiệu.

- Đánh giá frozen theo đúng protocol của SignRep (dictionary retrieval trên ASL Citizen).
- Đóng góp phải đo được theo từng module, không chỉ là một con số tổng.

**Hiện trạng** [I]:

| Model | ASL Citizen test R@1 (frozen) | Ghi chú |
|---|---:|---|
| PLctx full160 (model skeleton tốt nhất của ta) | 16.68 | |
| MASA (pose SSL label-free tốt nhất đã báo cáo) | ≈ 28 [V\*] | |
| SignRep (RGB) | 37.47 / 49.95 | avg / activity-weighted |
| SignSeek | 57.66 | pose, *có* nhãn gloss |

Khoảng cách lớn này gợi ý — và ta đăng ký thành giả thuyết **H0** để kiểm chứng ở Phase A — rằng **input skeleton không phải nút thắt chính**. Nút thắt là ba thứ: (a) tín hiệu tự giám sát, (b) readout và cách đo, (c) scale.

**Năm phát hiện từ dữ liệu nội bộ định hình thiết kế** [I]:

1. **Pilot đánh giá quá cao.** Cùng một model (PLctx full160): ASLC300-white (300 gloss, có whitening) đạt 47.7 trong khi full test chỉ 16.7. Cần dev metric mô phỏng đúng test (**VAL-DICT**) và corpus screening phủ đủ 2.731 gloss.
2. **Chỉ số hình học chỉ dùng để phát hiện collapse, không dùng để xếp hạng.** Effective rank có τ ≈ 0.14 trên các họ model; signer probe có τ ≈ 0. Probe handshape có τ = 0.64 trong họ skeleton.
3. **Đây là ba lỗ hổng chung của mọi model skeleton:** movement và non-manual hầu như không được mã hoá (probe R² < 0); dị hướng lớn (white so với raw); signer leakage cao.
4. **Trong không gian đã thử, chỉ một tổ hợp có tín hiệu rõ:** latent contextual target (EMA, centered-CE) + phonological-channel masking + readout theo nhóm. Target tường minh (descriptor, motion, log-signature), alignment toàn cục và regularizer phân phối đều âm.
5. **Recipe của SignRep sụp đổ khi tái tạo trên ASL Citizen** (effective rank 63→6), nên không thể coi các module của nó là đã được chứng minh trong setting của ta.

**Đề xuất.** Không xoay quanh các module cũ. Chỉ giữ chúng làm **anchor (REF)**, và mở một **phễu khám phá tín hiệu** gồm 15 arm thuộc 4 họ **chưa từng thử** trong cùng protocol, cộng 3 control:

- **Họ O, paradigm/target:** masked cluster prediction kiểu HuBERT, contrastive cấp clip, hybrid, teacher RGB frozen, dự báo latent nhân quả, target đa tầng nối lại, tinh chỉnh NNCLR ngắn kiểu STARS (có biến thể positive khác signer).
- **Họ K, masking:** motion-aware có ngẫu nhiên, semantic tube khoảng 0.5 giây.
- **Họ V, bất biến signer và view:** retarget khung xương chuẩn, GRL làm cận trên, view mạnh chỉ cho student + mask token cho khớp thiếu.
- **Họ I, input:** đa luồng joint/bone/motion, encoder tay dùng chung, token quan hệ tay–mặt/thân.

Mỗi arm được sinh ra từ một **giả thuyết domain đã đăng ký trước** (mục 5: H0 và H1–H12, mỗi giả thuyết có dự đoán định lượng, tiêu chí bác bỏ và control cho các giải thích thay thế).

Mỗi arm có **chữ ký cơ chế đăng ký trước**, đo bằng các metric sau, và được xét bằng **luật quyết định bốn nhánh** (KEEP / KEEP-MECH / INCONCLUSIVE / DROP; DROP chỉ khi có kiểm định tương đương hoặc có hại):

- minimal-pair accuracy theo từng tham số âm vị học (ASL-LEX);
- signer-neighbour rate;
- độ nhạy với thứ tự thời gian;
- anisotropy gap;
- group-drop.

**Chương trình** gồm 6 phase có cổng, khoảng 150–170 GPU-h (bản tiết kiệm khoảng 90), tương đương 4–6 tuần lịch trên GPU dùng chung:

| Phase | Nội dung |
|---|---|
| A | Đo lường và neo: evaluator v2, chấm lại mọi checkpoint, seed noise, cận trên có giám sát |
| B | Chốt readout và input |
| C | Phễu khám phá: pre-screen không cần train → screening 1 seed → xác nhận 3 seed |
| D | Fractional factorial để tổ hợp và đo tương tác |
| E | Scale dữ liệu, bước và capacity |
| F | Leave-one-out, 3 seed, test một lần |

**Kết quả mong đợi.** Ít nhất là một bản đồ tín hiệu: họ nào giúp tham số nào, họ nào vô ích trong domain này. Đây là tri thức để đề xuất method về sau. Tốt nhất là một trong năm luận điểm method ở mục 8 đạt mức L1 trở lên.

---

## 1. Bài toán, phạm vi và tiêu chí thành công

### 1.1 Định nghĩa bài toán

Học một encoder $f_\theta$ ánh xạ chuỗi keypoint 2-D (COCO-WholeBody 133 điểm, ASL Citizen) thành embedding. Encoder có thể ra một vector cho cả clip, hoặc một chuỗi token theo thời gian/bộ phận cơ thể. Có hai ràng buộc:

- **Pretraining không dùng nhãn gloss** (label-free). Metadata signer chỉ dùng để đánh giá. Nếu một nhánh dùng signer ID khi train (mục 7.3, V3), nhánh đó phải khai báo rõ là *weak supervision* và không được dùng cho claim "label-free".
- **Đánh giá với encoder frozen**, theo đúng tinh thần SignRep:
  - (i) dictionary retrieval không cần train;
  - (ii) linear probe;
  - (iii) fine-tune ISLR, chỉ làm ở giai đoạn cuối.

Đầu vào lúc inference chỉ là skeleton; RGB không xuất hiện ở test time.

### 1.2 Vì sao skeleton — lợi thế và rủi ro thật

| Lợi thế | Rủi ro / giới hạn |
|---|---|
| Rẻ, khoảng 5M tham số; 200k bước train mất 4 giờ trên một GPU. Thử nghiệm nhanh hơn RGB 1–2 bậc. | Hand keypoint là mắt xích yếu nhất: COCO-WholeBody hand AP thấp hơn body/face rõ rệt, và lỗi tập trung ở các frame tay chạm tay hoặc tay chạm mặt (§3.7). |
| Ít nhiễu ngoại hình (quần áo, nền, ánh sáng). Dễ chuẩn hoá theo cơ thể, dễ mirror, dễ retarget. | Signer vẫn rò qua động học: tỉ lệ xương, nhịp, phong cách. Signer probe trên mọi model nội bộ đạt khoảng 84–94%. |
| Cấu trúc bộ phận rõ ràng (tay trội, tay không trội, mặt, thân), khớp với các tham số âm vị học: handshape, location, movement, orientation, non-manual. | Mất thông tin mà RGB có: mouthing, độ sâu, chi tiết ngón khi bị che khuất. |
| Có cận trên đo được bằng nhãn: pose ISLR có giám sát đạt rất cao trên ASL Citizen (§3.2). | Chưa có model skeleton label-free nào được công bố vượt khoảng 28 R@1 trên protocol frozen retrieval ASL Citizen (§3.2). |

**Luận điểm định vị** (giả thuyết làm việc H0, §5.2; kiểm chứng bằng cận trên có giám sát ở A6). Bản thân input skeleton không phải nút thắt chính. Có hai bằng chứng gián tiếp: pose có giám sát vượt SignRep trên cùng protocol, và model tốt nhất của ta (16.7 R@1 trên test) thấp hơn nhiều so với những gì skeleton có thể mang. Nút thắt nằm ở ba chỗ:

- **(a) Tín hiệu tự giám sát:** objective, target, view và masking.
- **(b) Readout và cách đo.**
- **(c) Quy mô dữ liệu và độ dài train.**

Chương trình thực nghiệm ở mục 9 được thiết kế để tách ba nguồn này ra khỏi nhau.

### 1.3 Ràng buộc thực tế

- **Dữ liệu.** ASL Citizen gồm 2.731 gloss, mỗi split đều phủ đủ toàn bộ gloss:
  - train 40.154 clip, 35 signer;
  - val 10.304 clip, 6 signer. Val rất lệch: P39, P26 và P21 mỗi người khoảng 3.000 clip; P12 có 697, P5 có 609, P13 chỉ có 2. Thực chất chỉ có 5 cụm signer, điều này ảnh hưởng đến thống kê ở mục 6.6;
  - test 32.941 clip, 11 signer.

  File split có sẵn cột `ASL-LEX Code` (2.723 mã). File `signdata.csv` của ASL-LEX 2.0 **không còn trong máy** và phải tải lại. File này chỉ dùng cho đánh giá.
- **Holdout.** Test của ASL Citizen, WLASL và MS-ASL đều đã bị truy cập:
  - project p4 dùng ở EX05 và EX07;
  - PhonoJEPA full160 đã đánh giá trên test của ASL Citizen và WLASL.

  Vì vậy mọi quyết định trong chương trình này **chỉ dùng val**. Test chỉ dùng **một lần ở Phase F**, và phải công khai lịch sử truy cập trước đó.
- **Compute.** Một GPU RTX 5880 dùng chung, tối đa 2 tiến trình, ổ đĩa đã đầy khoảng 97%. Mọi phase được xếp hàng tuần tự. Ngân sách ước tính ở §9.8.
- **Phạm vi tài liệu.** Đây là **đề xuất thiết kế**: chưa implement, chưa chạy. Mỗi thực nghiệm đều có giả thuyết, nhánh (arm), control, metric và luật quyết định, để khi chạy có thể ra quyết định giữ hay bỏ một cách máy móc.

### 1.4 Tiêu chí thành công

Đo trên ASL Citizen test (frozen dictionary retrieval, một lần duy nhất ở cuối). Phải so cùng loại readout: SignRep có hai số là *avg* và *activity-weighted*.

| Mức | Ngưỡng R@1 | Ý nghĩa |
|---|---|---|
| L0 (hiện tại) | 16.68 (PLctx full160, avg, raw) | điểm xuất phát |
| L1 | > 27.96 | vượt model pose SSL label-free tốt nhất đã báo cáo (MASA, theo bảng của SignSeek) |
| L2 | ≥ 37.47 | ngang SignRep khi dùng readout avg |
| L3 | ≥ 49.95 | ngang SignRep khi dùng activity-weighted: skeleton bằng RGB SSL |

Ngoài con số tuyệt đối, có bốn **trục đóng góp trực giao**. Mỗi trục có metric riêng (§6.4) và có thể tạo thành claim ngay cả khi chưa đạt L3:

1. **Hiệu quả:** tham số, GPU-h, độ trễ inference.
2. **Bền với signer:** signer-neighbour rate thấp hơn; khoảng cách cùng signer / khác signer nhỏ hơn.
3. **Trung thực âm vị học:** độ chính xác minimal-pair theo từng tham số, đặc biệt là *movement*, vì mọi model skeleton nội bộ hiện đều có probe R² < 0 cho movement.
4. **Bền với nhiễu keypoint.**

---

## 2. SignRep: đọc kỹ và bài học cho hướng skeleton

*Wong, Camgoz, Bowden. "SignRep: Enhancing Self-Supervised Sign Representations", ICCV 2025, arXiv 2503.08529.* Chỉ có bản v1, và mọi con số trong bản camera-ready ICCV (main + supplement) đều trùng với v1 [V]. Chi tiết đầy đủ ở `docs/research/literature/A_signrep_and_sign_ssl.md` §1–2.

### 2.1 Phương pháp

| Thành phần | Chi tiết |
|---|---|
| Backbone | Hiera-B video, **khởi tạo từ Hiera MAE trên Kinetics** (không có ablation nào về việc này), 51.5M tham số [V\*]. Input 16 frame, crop phần thân trên, 224². Token grid 8×56×56; mask unit = 2 frame × 32×32 px. |
| Bỏ pixel decoder | Lấy trung bình token → LN → FC → **z^avg ∈ R^768** (đây là representation). Sau đó ConvTranspose "mở" z^avg thành 16 vector theo frame, mỗi vector 384-d, rồi đi qua các head tuyến tính cho từng prior. **Không có target theo token, không có token theo bộ phận.** |
| Teacher pose | Bộ ước lượng 3D chuyên cho ký hiệu (Ivashechkin et al.) cho ra **góc khớp**. Keypoint 3D được dựng lại từ góc với **độ dài xương median trên toàn YT-SL**, nên mọi người có cùng khung xương: target đã được chuẩn hoá danh tính. Bỏ keypoint có confidence < 0.5 [V]. |
| Prior (smooth-L1 theo frame) | Keypoint tay lấy cổ tay làm gốc (21×3), keypoint toàn thân (61×3), góc tay (41, sin/cos), góc thân (22), **vector dịch chuyển** đầu ngón (5×11×3), **dịch chuyển tương tác tay–thân/mặt** (12×22×3), hoạt động tay trái/phải (heuristic). Trọng số: 10/10/10/10/20/20/0.2 [V]. |
| Regularizer | Variance hinge (w = 1.0) + covariance (w = 0.004) trên z^avg; **adversarial style** (w = 2.0): Gram matrix của token ghép với z^avg, discriminator phân biệt cặp cùng video hay khác video, chỉ kích hoạt khi E_M > E_U [V]. |
| Masking | Ngẫu nhiên 80% theo mask unit. Token bị che không được tái tạo; masking hoạt động như *information dropout* [V]. |
| Dữ liệu và compute | YouTube-SL-25 (khoảng 3.000 giờ theo các paper trích dẫn), đã loại ID video của WLASL. 500K vòng lặp, batch 20, một RTX 3090. **Checkpoint được chọn bằng retrieval trên WLASL val** [V]. |

### 2.2 Ablation: điều gì thật sự tạo ra kết quả

Table 5 đo WLASL DCG, stride 8, mỗi cấu hình 1 run [V]:

| Cấu hình | DCG |
|---|---:|
| angle | 45.1 |
| kpt | 32.9 |
| dist | 47.2 |
| angle + kpt + dist | 48.5 |
| không mask | 46.3 |
| + var/cov | 49.9 |
| + adversarial | 50.7 |

Table 4 đo ASL Citizen frozen dictionary retrieval, R@1 [V]:

| Feature | R@1 |
|---|---:|
| HieraMAE-YTSL (pixel MAE trên cùng dữ liệu) | 0.39 |
| Góc khớp tay, lấy trung bình (không cần train) | 8.81 |
| SignRep avg | 37.47 |
| SignRep activity-weighted | **49.95** |

Xếp hạng theo bằng chứng:

1. **Target đặc thù cho ký hiệu thay cho pixel MAE: khoảng +37 R@1.** Pixel MAE trên cùng dữ liệu còn tệ hơn descriptor góc tay không cần train.
2. **Readout activity-weighted: +12.5 R@1** trên ASL Citizen. Khoảng một phần tư con số 49.95 là do readout. Nội bộ còn đo được thêm +5.9 với bins2 và +9.4 với chamfer [I, C012].
3. **Target quan hệ (dist, angle) ≫ keypoint tuyệt đối**, chênh 12–14 DCG; gộp tất cả chỉ thêm +1.3.
4. Dữ liệu in-domain: khoảng +5.7 khi fine-tune (MAE-Kinetics 51.5 → MAE-YTSL 57.2).
5. Masking +2.2; var/cov +1.4; adversarial +0.8 (trong mức nhiễu, 1 run).

### 2.3 Giới hạn

- **Đã nêu trong paper:** chỉ đặc trưng tay ("solely manual"), cửa sổ 16 frame, phụ thuộc chất lượng teacher, bias của YT-SL.
- **Thấy được khi phân tích:**
  - **Không có thứ tự ở cấp sign:** clip là trung bình của các cửa sổ; cos(clip, clip đảo ngược) = 0.959 [I].
  - **Không có cấu trúc bộ phận.**
  - **Không canonical hoá tay trội** và không dùng flip.
  - **Bất biến signer chỉ ở mức ngoại hình:** neighbour cùng signer 48.7% > cùng gloss 43.8% [I, H010].
  - **Movement mã hoá yếu nhất** (lift 1.6× so với handshape 8.8×) [I].
  - **Vệ sinh đánh giá:** chọn checkpoint trên WLASL val; 1 run, không seed; ASL Citizen trùng từ vựng giữa train và test.
  - **Chi phí inference:** Hiera-B trên RGB 224², cửa sổ trượt stride 2.

### 2.4 Bài học cho hướng skeleton

1. **Sức mạnh của SignRep nằm ở *thông tin mà target mang thêm so với input*.** Encoder thấy pixel nhưng phải dự đoán hình học 3D đã chuẩn hoá danh tính. Đây là một dạng tín hiệu cross-modal.

   Với skeleton thì ngược lại: các prior đó là hàm tất định của *chính input*. Nội bộ đã thử tương đương skeleton của chúng: intrinsic target và explicit relations (S1), cả hai đều không giúp. Ở đây shufint ≈ PLboth.

   → Với skeleton, cái tương đương với prior của SignRep là **target chứa thông tin mà skeleton không có**:
   - (a) latent theo ngữ cảnh: tích hợp thông tin từ phần còn thấy (REF);
   - (b) đơn vị rời rạc học từ phân bố của toàn bộ dữ liệu (O1);
   - (c) teacher RGB (O4).

   Đây là lý do họ O có ưu tiên cao nhất.
2. **Readout phải được kiểm soát riêng.** Mọi so sánh với SignRep phải ghi rõ avg hay weighted. Readout activity-weighted cho skeleton có thể lấy trọng số hoạt động từ chuyển động và độ cao của tay mà không cần nhãn (R4).
3. **Chuẩn hoá danh tính** đáng mang sang input skeleton (V1). SignRep chỉ áp nó cho target.
4. **Recipe regularizer của SignRep không chuyển được:** var/cov và adversarial sụp đổ khi tái tạo trên ASL Citizen [I], và đóng góp gốc của chúng nằm trong mức nhiễu (+1.4, +0.8).
5. **Chưa có khoảng trống nào bị "đóng" vì skeleton kém hơn.** Có ba bằng chứng:
   - model skeleton label-free tốt nhất (MASA, 27.96) chỉ pretrain trên khoảng 146K clip ISLR, so với khoảng 3.000 giờ của SignRep;
   - pose có giám sát đạt 57.3–57.7 trên cùng protocol;
   - trên zero-shot BSL SignBank, **MASA (pose) 6.80 > SignRep 4.30** [V\*].

   Tồn tại **SignVerse-2M** (DWPose cho khoảng 2M clip của YT-SL-25), cho phép một so sánh **khớp dữ liệu** giữa skeleton và SignRep (Phase E4).

---

## 3. Literature review: phạm vi rộng, tập trung vào bằng chứng ablation

Mục này tổng hợp **7 báo cáo**:

- **L1–L4** (`docs/proposal12/literature/`, 2026-10-11): JEPA, masking, skeleton SSL, sign SSL.
- **A, B, C** (`docs/research/literature/`, 2026-10-14): A = SignRep và sign SSL cập nhật; B = skeleton SSL và masking; C = JEPA, chỉ số chẩn đoán, độ chặt chẽ thống kê.

Các con số được trích ở đây đều đã grep từ văn bản gốc [V], hoặc ghi rõ nguồn thứ cấp [V\*] hay chỉ có từ tóm tắt [S]. Chi tiết từng paper xem trong báo cáo gốc.

### 3.1 Bản đồ lĩnh vực

| Trục | Các lựa chọn chính trong literature | Vị trí của ta |
|---|---|---|
| Input | RGB (SignRep, SHuBERT crop, SignDino) / pose 2-D, 3-D (MASA, SignBERT+, BEST, Sigma) / hybrid | pose 2-D COCO-WholeBody |
| Supervision | label-free (SignRep, MASA, SignBERT+, BEST) / text (Uni-Sign, Sigma, SignCLIP, LA-Sign) / gloss (SignSeek, SignMatch, OneSign) | label-free |
| Objective | tái tạo pixel hoặc toạ độ (MAE, MAMP, SignBERT+) / hồi quy prior (SignRep) / latent EMA (JEPA: S-JEPA, Skeleton2vec, Path-JEPA) / prototype hoặc cluster (DINO/iBOT, SLiM, HuBERT/SHuBERT) / contrastive (AimCLR, ActCLR, STC, SSL-SLR) / decorrelation (VICReg, USDRL) / diffusion (MacDiff) | latent EMA + centered-CE (REF) |
| Chống collapse | EMA + stop-grad + predictor / centering, sharpening / SIGReg, VICReg / frozen teacher (SALT) | EMA + centering |
| Masking | random / tube / theo bộ phận (SLiM STM, SkeletonMAE) / motion-aware (MAMP) / thích nghi (AdaMAE, ASGM) | theo kênh âm vị học (PL) |
| Token | per-joint × đoạn thời gian (S-JEPA: 750 token) / compact (SLiM: 8 frame) / theo bộ phận / theo kênh | kênh âm vị học × 32 bin |
| Đánh giá frozen | dictionary retrieval (SignRep T4, SignSeek) / linear probe (STC) / kNN / phonology probe / minimal pair (Yin 2026) | VAL-DICT + M1–M9 |

### 3.2 Sign-language SSL và pose ISLR

**Bảng tham chiếu frozen dictionary retrieval trên ASL Citizen** (gallery = train, 2.731 gloss, signer-disjoint):

| Nhóm | Phương pháp | R@1 | Ghi chú |
|---|---|---:|---|
| Không train | góc khớp tay, lấy trung bình | 8.81 | SignRep T4 [V] |
| Label-free, RGB | HieraMAE-YTSL (pixel MAE) | 0.39 | [V] |
| **Label-free, pose** | **MASA** (146K clip ISLR) | **27.96** | SignSeek T2 [V\*] |
| Label-free, RGB + teacher pose 3D | **SignRep** avg / weighted | 37.47 / **49.95** | [V] |
| Label-free, RGB + readout nội bộ | SignRep + bins2 / chamfer / H008 | 54.93 / 58.42 / 57.28 | [I] |
| Text | SignCLIP | 4.00 | [V] |
| Gloss (các tập khác) | **SignSeek** (pose) | 57.66 | [V] |
| Dictionary BSL/ASL + pseudo-label | SignMatch pose / RGB (init từ SignRep) | 57.32 / 67.12 | [V] |
| Gloss ASL Citizen | I3D | 58.03 | [V\*] |

Các mốc khác:

- **Linear probe pose SSL** trên WLASL2000: SignBERT 18.24, BEST 21.19, SignBERT+ 24.50, STC (contrastive) 32.35 [V].
- **Fine-tune hoặc có giám sát** trên ASL Citizen: SignRep FT 81.37, PoseFormer 75.1, SignSeek FT 74.7, OneSign 65.4–68.4, Uni-Sign (tái chạy) 60.92 [V].

**Bài học chính:**

1. **Chưa có sign JEPA nào** tính đến 2026-10-14. Đã tìm trên arXiv API, Semantic Scholar (danh sách trích dẫn của SignRep, SHuBERT, Uni-Sign, MASA, SignCLIP), sign-lang@LREC 2026 và web. Đây là claim có phạm vi tìm kiếm, không phải chứng minh không tồn tại [A].
2. **Có ba khoảng cách cần lấp, đọc từ bảng trên:**
   - label-free skeleton (27.96) → label-free RGB (49.95): **khoảng cách về tín hiệu và dữ liệu**;
   - → supervised pose (57.7): **trần mà input skeleton cho phép**;
   - → SignMatch RGB (67.12): **trần khi có nhãn và scale**.
3. **Kezar et al. 2025 (VQ-ASL):** tách bộ phận cơ thể không dùng nhãn làm tái tạo tốt hơn (MSE −21%) nhưng **không** làm nhận dạng OOV tốt hơn (MRR .381 → .372); chỉ supervision âm vị học mới giúp [V]. Đây là cảnh báo cho mọi thiết kế "token theo bộ phận": cấu trúc thôi chưa đủ, phải có tín hiệu buộc tích hợp thông tin.
4. **Yin et al. 2026, minimal pair:**
   - pose có giám sát (ST-GCN) thắng về handshape (86%) và movement (81%); RGB (I3D) thắng về location (81% so với 68%);
   - pretraining Kinetics không cho độ nhạy âm vị học nào;
   - độ nhạy giảm khoảng một nửa khi ra ngoài domain [V].
   - Chưa model SSL nào được đo theo protocol này: **cơ hội**.
5. **Canonical hoá tay trội** cho khoảng +5 R@1 trong model pose có giám sát [V, cảnh báo về split]. Handedness suy được không cần nhãn từ độ cao của tay [V abstract].
6. **Scale và supervision ngôn ngữ** tạo bước nhảy lớn nhất cho pose ISLR: đường cong của Sigma 23.0 → 64.5 theo % dữ liệu pretrain; Uni-Sign; LA-Sign [V]. Pose SSL label-free chưa ai train ở quy mô YT-SL-25. **SignVerse-2M** (DWPose cho YT-SL-25) giờ cho phép làm điều đó [V].
7. **Bảy yếu tố quyết định pose ISLR có giám sát** [V-L4]:
   - tay là bắt buộc: chỉ body 21.4 so với body + tay 48.5 trên WLASL;
   - giảm graph xuống khoảng 27 node;
   - multi-stream joint/bone/motion khoảng +6;
   - tách nhánh tay và thân khoảng +5 (STC);
   - mặt và miệng có ích (miệng là vùng mặt quan trọng nhất) [V abstract 2507.20884];
   - extractor tạo khác biệt lớn (Sem-Lex: 57.1 → 72.2 khi trích lại pose);
   - hand AP của COCO-WholeBody thấp (58–66).
8. **Signer leakage có hệ thống:**
   - pose nhận diện signer 80% (may rủi 3%) [V-L4];
   - feature SignRep có neighbour cùng signer nhiều hơn cùng gloss [I];
   - chưa có skeleton SSL label-free nào nhắm thẳng vào signer invariance.

### 3.3 Skeleton SSL (action recognition, NTU/PKU) và bài học có thể chuyển giao

Gần như toàn bộ bằng chứng ở mục này đến từ dữ liệu NTU 3-D, 25 khớp, nên luôn có rủi ro khi chuyển sang ký hiệu 2-D (xem bảng rủi ro ở B §5).

| Câu hỏi | Bằng chứng [V] | Kết luận |
|---|---|---|
| (a) Target: toạ độ, motion hay latent? | MAMP: motion +10.1 so với joint thô, nhưng AMR với joint đã chuẩn hoá: joint ≥ motion. S-JEPA: target ở output teacher hơn input +7.2. SLiM (đối chứng khớp): prototype-CE 87.66 > feature MSE 84.53 (collapse, rank 1.12) ≈ toạ độ 84.30. | Latent chỉ thắng khi **có dạng đúng** (ngữ cảnh, phân phối, có centering). Toạ độ thô là tệ nhất, nhưng chuẩn hoá cứu được. |
| (b) Mask có cấu trúc? | Skeleton2vec: tube khoảng 15 frame +3.6 so với random; tube cả clip 78.2 < random 79.4. SLiM STM +3.5 (khoảng +2.0 từ vai trò augmentation). MAMP motion-aware +1.2. Bỏ ngẫu nhiên: 84.9 → 66.7. MacDiff (target toàn cục): random 86.4 > tube 83.1. | Cấu trúc thời gian **khoảng 0.5 giây** là yếu tố lớn nhất. Hiệu ứng **phụ thuộc target**. Phải giữ ngẫu nhiên. |
| (c) Per-joint hay theo bộ phận? | Chưa có ablation trực tiếp. SLiM: token compact chỉ ổn khi target là latent (MAMP mất 6.1 điểm). Path-JEPA: token quan hệ +1.8 [S]. | Còn là khoảng trống. Với 133 keypoint, token theo bộ phận/kênh hoặc compact là bắt buộc trên 1 GPU. |
| (d) EMA, frozen teacher hay regularizer? | S-JEPA không EMA → 1.6% (collapse); momentum 0.99 → 0.9999 dao động 80.0–85.3. USDRL decorrelation hơn InfoNCE: +9.4 linear, +9.9 retrieval. Chưa paper skeleton nào thử SIGReg hay frozen teacher. | EMA bắt buộc và nhạy cảm; frozen teacher là khoảng trống. |
| (e) Contrastive hay masked cho retrieval? | MAMP linear 84.9 nhưng kNN khoảng 62; STARS (MAE + 20 epoch NNCLR) kNN 79.9 (+16.8). SLiM: CL 73.6 < MFP 85.3 < MFP + GLCL 87.7. Benchmark IJCV: MAMP đứng đầu linear nhưng cuối few-shot. | **Retrieval cần thành phần global/instance.** Kết hợp mới là tốt nhất. |
| (f) Epoch, scale, độ ổn định thứ hạng | GFP: thứ hạng giữa các loại target cố định từ checkpoint đầu. EMA đảo thứ hạng theo độ dài lịch. PCM3++: giảm một nửa dữ liệu mất 7 điểm linear, 10 điểm retrieval. | Screening ở budget vừa phải là được cho hiệu ứng lớn; hyperparameter phải tinh chỉnh lại ở full budget. |
| (g) Khớp bị thiếu hoặc nhiễu | SLiM: khớp thiếu đi qua mask token 87.2, parent-fill 66.5, zero-fill 49.6. MAMP chịu nhiễu kém (2.4% ở IGM). Low-pass augmentation +2.4 (BYOL-skeleton). | Khớp thiếu **phải** đi qua mask token. Lọc thông thấp có ích cho tay 2-D bị rung. |
| Augmentation | BYOL-skeleton: mạnh 79.6 so với nhẹ 51.1. HiCLR: mask ngẫu nhiên ngây thơ làm contrastive −16. ActCLR: augment vùng tĩnh +4.8 kNN. | Augmentation quyết định chất lượng invariance; augmentation mạnh chỉ áp cho student, hoặc theo thứ bậc từ yếu đến mạnh. |

**Các paper chủ chốt:**

| Paper | Nội dung | Linear (NTU) |
|---|---|---|
| **S-JEPA** (ECCV 2024) | JEPA skeleton với centered-CE, EMA, view chỉ áp cho student. Không có "mẹo 2-D → 3-D". Không có code. | 85.3 (báo cáo ở 800 epoch) |
| **Path-JEPA** (ECCV 2026) | Target log-signature, token quan hệ (joint/edge/chain), mask lan truyền chống leak. Không có code, không có arXiv. | 87.6 |
| **SLiM** (NeurIPS 2026, arXiv 2603.10648) | Token compact, STM, MFP + GLCL, có code | 87.9 |
| **Skeleton2vec** (arXiv 2401.00921) | Latent kiểu data2vec, tube motion-aware | 85.7 |
| MAMP (ICCV'23) / MacDiff (ECCV'24) / GFP (ICCV'25) / AMR (CVPR'26) | | 84.9 / 86.4 / 85.9 / 87.4 |
| BYOL-skeleton / SCD-Net (contrastive) | thường bị bỏ khỏi bảng so sánh | 86.8 / 86.6 |

### 3.4 JEPA và cơ chế chống collapse

- **I-JEPA / V-JEPA [V-L1]:**
  - hình học mask là hyperparameter nhạy nhất: multi-block 54.2 so với random 17.6;
  - target lấy ở output teacher tốt hơn ở input;
  - target latent tốt hơn pixel;
  - predictor hẹp có ích;
  - attentive probe tăng mạnh so với linear (V-JEPA +16 đến +24).
- **V-JEPA 2 / 2.1 [V-L1]:**
  - EMA cố định, lịch warmup–constant–cooldown;
  - context loss trên token còn thấy cải thiện feature dày đặc nhưng có hại cho tác vụ toàn cục, trừ khi có deep supervision đa tầng.
- **Target đa tầng** (Bootleg, 2603.15553):
  - nối {1, 4, 8, 12} đạt 67.8; trung bình các tầng kiểu data2vec chỉ 44.1;
  - seed std 2.9 → 0.6;
  - gần như không nhạy với EMA τ [V].
- **JEPA nhiễu theo seed:** I-JEPA 51.5 ± 2.9 so với MAE 43.2 ± 0.2 [V]. → So sánh 1 seed không phân giải được hiệu ứng 2–3 điểm.
- **Joint-embedding hay reconstruction** (Van Assel et al. 2025): JE tốt hơn khi nhiễu không liên quan có biên độ lớn; reconstruction tốt hơn khi nhiễu nhỏ [V]. Với skeleton, nhiễu lớn là hình thể signer và rung tay. Dự đoán kiểm chứng được: khoảng cách JEPA−MAE *tăng* khi bỏ chuẩn hoá hoặc thêm nhiễu keypoint.
- **Lý thuyết về predictor** (Koopman-JEPA 2511.09783; Sobal): predictor gần-đồng-nhất đẩy encoder về feature bất biến theo thời gian → **rủi ro signer** [V].
- **Chống collapse:**
  - BYOL τ: vùng tối ưu 0.99–0.999; target random cố định chỉ 18.8 [V];
  - EMA scaling rule ρ̂ = ρ^κ khi đổi batch [V];
  - DINO centering + sharpening: log teacher entropy [V];
  - LeJEPA/SIGReg [V-L1], nhưng nội bộ L1+SIGReg kém hơn centered-CE 13.8 điểm [I];
  - SALT (frozen teacher): loss dự báo được chất lượng, R² = 0.951 [V-L1].

### 3.5 Masking: lý thuyết và cạm bẫy

- **Lý thuyết shared-latent** (Kong et al. [V-L2]): mask quyết định biến ẩn chung giữa phần bị che và phần còn thấy. Mask quá nhỏ chỉ dạy động học cục bộ; mask quá lớn chỉ dạy yếu tố cấp thấp (dáng người).
- **AvgDist** (SimMIM): hiệu năng có dạng sống núi theo khoảng cách từ token bị che đến token còn thấy → đo "hop distance" và R² của P-M [V-L2].
- **Leak:**
  - token quan hệ hoặc vị trí làm lộ khớp bị che (Point-MAE, Path-JEPA);
  - khớp đối xứng hai tay (mirror shortcut);
  - nội suy từ frame lân cận;
  - mask token trông giống lỗi của detector [V-L2].
- **Mask theo cả bộ phận** dùng một mình thường hại (SemMAE 52.9 so với 66.8; BEST body-only thấp hơn không pretrain). Có lợi khi trộn hoặc có curriculum từ dễ đến khó [V-L2]. Nội bộ: curriculum G2 lại âm [I]. → Tuỳ domain.

### 3.6 Đánh giá và chẩn đoán representation

| Công cụ | Dùng làm gì | Bằng chứng [V] |
|---|---|---|
| RankMe (trên σ) | **Cổng** phát hiện collapse | Xếp hạng JEPA kém: τ 0.58 / 0.29 trên I-JEPA, 0.22 trên data2vec |
| LiDAR | Xếp hạng trong cùng họ (mask, τ, λ) | τ 0.82 / 0.79 trên I-JEPA |
| TwoNN / IdEst intrinsic dimension | Xếp hạng giữa các model, chỉ sau cổng rank | ρ −0.74; trên 260 model ρ ∈ [−0.95, −0.26] |
| kNN / MAP@R | Proxy rẻ có dùng nhãn | Gần bằng linear (DINO 74.5 so với 77.0) |
| Linear và attentive probe | Trọng tài cuối, báo cáo cả hai | Tương tác method × probe có thể đảo thứ hạng |
| ABX / minimal pair | Nội dung âm vị học, bất biến signer | Speech ABX; sign: Yin 2026 |
| Signer probe linear **và MLP**, post-hoc | Rò rỉ nuisance | Speech: linear 14.8% nhưng MLP 89.0% sau chuẩn hoá |
| Control task / MDL | Probe đọc representation hay tự ghi nhớ | Selectivity (Hewitt & Liang) |
| CKA / Procrustes | Hai run có học cùng thứ không (không phải metric chất lượng) | CKA thiếu nhạy |

**Độ chặt chẽ thống kê:**

- **Proxy ở scale nhỏ:** khoảng 80% quyết định cặp còn đúng khi lên scale lớn (DataDecide); metric có SNR cao cho quyết định chuyển giao tốt hơn (R = 0.79).
- **So sánh công bằng thu hẹp các lợi ích được claim** (metric learning reality check; "GANs are created equal"). Nguồn biến thiên lớn nhất là cách lấy mẫu dữ liệu.
- **Factorial thay cho thay đổi một nhân tố mỗi lần;** ghép nối leave-one-out với add-one; Shapley.
- **TOST** cho kết luận "không có hiệu ứng"; probability of improvement ≥ 0.75 [V, C §4–5].

### 3.7 Đặc thù ngôn ngữ ký hiệu ảnh hưởng đến thiết kế

1. **Tham số âm vị học:** handshape, location, movement, orientation, non-manual. ASL-LEX 2.0 có 2.723 mục và 58 handshape [V\*]; mã ASL-LEX có sẵn trong split ASL Citizen [I].
2. **Thời lượng:** sign trung vị 2.5 giây [I], nhưng các sự kiện âm vị học (đổi handshape) chỉ khoảng 100–300 ms [K]. → Tube mask khoảng 0.3–0.6 giây, bin 2 frame.
3. **Hai tay bất đối xứng:** tay trội/không trội, các loại sign (một tay, đối xứng, bất đối xứng). → Canonical hoá theo tay trội, và cẩn thận với mirror shortcut.
4. **Nhiễu keypoint tập trung ở tay** và ở các frame tay chạm tay hoặc tay chạm mặt [V-L4]. → Mask token cho khớp thiếu; lọc thông thấp; target bền với nhiễu.
5. **Signer:** danh tính đan xen với động học (tỉ lệ xương, nhịp, phong cách). Val ASL Citizen chỉ có 5 cụm signer hiệu dụng [I].
6. **Từ vựng trùng giữa train và test** trong ASL Citizen: retrieval không phải open-vocabulary. Phonology probe nên có split giữ lại theo từ vựng [V-A].

### 3.8 Mười hai nguyên tắc thiết kế rút ra từ literature

| # | Nguyên tắc | Nguồn chính |
|---|---|---|
| P1 | *Cái được dự đoán* quan trọng hơn *chỗ bị che*; target phải mang thông tin mới | SignRep T4/T5, MAMP, SLiM T10, nội bộ |
| P2 | Target latent cần dạng đúng: lấy ở output teacher, phân phối/CE có centering, đa tầng nối lại | S-JEPA, SLiM, Bootleg |
| P3 | Retrieval cần thành phần global/instance bên cạnh dự đoán cục bộ | STARS, SLiM, IJCV benchmark |
| P4 | Mask: bền khoảng 0.5 giây, có ngẫu nhiên, tỉ lệ ~U(0.5, 0.9), không che cả clip | Skeleton2vec, HA-CM, SLiM |
| P5 | Hiệu ứng cấu trúc mask phụ thuộc target, nên phải kiểm định chứ không mặc định | MacDiff, L2 |
| P6 | EMA bắt buộc nhưng nhạy; tinh chỉnh theo batch và độ dài lịch | S-JEPA, Skeleton2vec, EMA scaling |
| P7 | Khớp thiếu đi qua mask token; augmentation mạnh chỉ cho student | SLiM, BYOL-skeleton, S-JEPA |
| P8 | Canonical hoá tay trội; chuẩn hoá danh tính (độ dài xương) | Bootstrapping, SignRep A1 |
| P9 | Báo cáo readout tách riêng; activity-weighted và có thứ tự tạo khác biệt lớn | SignRep T4, nội bộ C012 |
| P10 | Đo cơ chế bằng minimal pair âm vị học và signer leakage (MLP, post-hoc) | Yin 2026, speech ABX |
| P11 | Chỉ số không nhãn chỉ dùng làm cổng (RankMe) hoặc xếp hạng trong cùng họ (LiDAR, ID) | LiDAR, IdEst |
| P12 | Factorial, nhiều seed cho JEPA, TOST cho "không có hiệu ứng", proxy chỉ được tin sau khi đã kiểm định | Bootleg T17, DataDecide, Bouthillier |

---

## 4. Hiện trạng nội bộ: điều đã biết, điều cần kiểm tra lại, điều chưa từng thử

Mục này **không** dùng để giới hạn hướng đi. Nó có ba mục đích:

- (i) giữ lại những gì đã có bằng chứng tốt, dùng làm anchor;
- (ii) chỉ ra những kết luận cũ cần kiểm tra lại, vì chúng được đo bằng một protocol có vấn đề (§4.3);
- (iii) vẽ ra phần lớn không gian **chưa được khám phá** (§4.4). Đây mới là trọng tâm của Phase C.

### 4.1 Kết quả đã có

Cùng evaluator `oj/evaluate.py`, pretrain trên 3.000 video (ASLC002), 1 seed. Các cột:

- A = ASLC300 R@1, white512 / raw;
- L = LOSO-val R@1, white512 / raw;
- LP = ASLC300 linear-probe macro;
- hs, mv = probe R² cho handshape / movement.

| Model | A white | A raw | L white | L raw | LP | hs | mv |
|---|---:|---:|---:|---:|---:|---:|---:|
| PLctx 4k | 46.3 | 30.5 | 22.6 | 10.3 | 49.4 | .28 | −.04 |
| PLctx 16k | **48.9** | **44.2** | 25.4 | 17.0 | 60.8 | .28 | −.03 |
| PLctx full-data 160 epoch | 47.7 | 40.7 | **29.5** | **19.7** | **68.9** | .32 | −.03 |
| PLint 4k / PLboth 4k | 41.3 / 41.9 | 26.5 / 30.8 | 22.1 / 22.5 | 9.3 / 10.4 | 48.9 / 47.1 | | |
| R0 (random mask) ctx / int / both | 39.1 / 33.6 / 41.0 | | 20.3 / 18.5 / 20.7 | | | | |
| R4 (articulator span) ctx / int / both | 31.3 / 34.7 / 39.8 | | 15.9 / 18.5 / 19.4 | | | | |
| B-MAE / B-MAMP (motion target) | 34.6 / 30.4 | 19.6 / 12.4 | 17.9 / 15.2 | | 46.1 / 34.5 | | |
| Global-only (Bglob) | 26.5 | 12.7 | 12.5 | 3.7 | 33.5 | | |
| Random-init encoder (sàn) | 23.9 | 9.1 | 10.8 | 3.2 | 32.8 | .22 | −.07 |
| Descriptor không cần train (B0) | 17.3 | 22.1 | 4.2 | 4.6 | 24.2 | .15 | −.07 |
| S1 (torso + explicit relation targets) / S2 (motion + log-signature targets) | 23.3 / 15.7 | | 10.5 / 7.2 | | 44.0 / 36.1 | | |
| Graph-JEPA per-joint, 67 landmark (B10–B13) | 17–21 | | 5–8 | | 15–25 | | |
| Pose-JEPA (proposal14), ASLC3000 | — | 20.8 | — | 5.1 | | | |
| RGB: JH3D (Hiera-MAE init, JEPA + priors) | 56.8 | 43.1 | 39.2 | 12.5 | 61.7 | .21 | .28 |
| RGB: SR3D (tái tạo SignRep, cùng budget) | 10.1 | 4.9 | 5.1 | 1.0 | 17.6 | .03 | .21 |

**Test (frozen dictionary, raw, readout avg).**

| Model | ASL Citizen R@1 | WLASL2000 R@1 |
|---|---:|---:|
| PLctx full160 | 16.68 | 9.62 |
| SignRep public, activity-weighted | 49.02 | 29.42 |
| SignRep public, avg | — | 26.50 |

**Tái tạo SignRep trên full ASL Citizen** (`jepa_baseline`, audit 2026-10-14): representation **sụp đổ**.

| Bước | Effective rank | Proxy top-1 |
|---|---:|---:|
| 10k | 63 | 1.87% |
| 30k | 6 | 0.07% |

Gradient của variance/covariance regularizer lớn gấp khoảng 3.700–4.800 lần gradient của pose reconstruction, và style discriminator đã ngừng hoạt động. **Recipe của SignRep rất nhạy với điều kiện dữ liệu và scale.** Ta không nên coi nó là "module đã chứng minh" khi chuyển sang ASL Citizen.

### 4.2 Phân loại module hiện có

Mọi phân loại dưới đây đều dựa trên **ASLC300-white, 1 seed**. Chúng là *giả thuyết làm việc* và phải được kiểm tra lại trên metric mới ở Phase A, bước A3.

| Nhóm | Module | Bằng chứng (Δ R@1 ASLC300 so với tham chiếu) |
|---|---|---|
| **Kế thừa (anchor)** | Latent contextual target: EMA teacher, centered-CE kiểu S-JEPA | ctx > int, khoảng +5 khi đi với PL. L1+SIGReg kém hơn 13.8; prototype loss kém hơn 4.2. |
| | Phonological-channel masking (PL: near time-block + far single-hand-channel span) | Main effect của mask: PL 43.2 > R0 37.9 > R4 35.3. |
| | Readout theo nhóm token (group concat) | grp 48.9 so với CLS 17.4. |
| | Train lâu hơn | 4k→16k: raw +13.7, white +2.6. Full data: LOSO +4.2, LP +8.1. |
| **Bỏ (bằng chứng âm)** | Global CLS cross-view alignment (G1) | −5.7 |
| | Intrinsic/explicit descriptor target | shufint 42.1 ≈ PLboth 41.9, tức target không mang thêm thông tin. S1 và S2 thấp hơn nhiều. |
| | Finger-group masking + curriculum (G2) | −2.9 |
| | Var/cov + adversarial regularizer kiểu SignRep | sụp đổ trong lần tái tạo |
| | Per-joint graph-JEPA (B-series) | khoảng 20, nhưng **bị nhiễu bởi khác codebase** → xem I1 ở §7.4 |
| **Chưa kết luận** | Bỏ kênh mặt (hands-only) | +1.6 ASLC300, LOSO ≈ 0 |
| | Handedness ngẫu nhiên | +1.6 |
| | Morphology augmentation | +0.5 |
| | Full data so với pilot | ASLC300 −1.3 [CI chứa 0], LOSO +4.2 |

### 4.3 Bốn khoảng trống đo được trong chính dữ liệu của ta

1. **Pilot đánh giá quá cao.** PLctx full160 đạt 47.7 trên ASLC300 nhưng chỉ 16.7 trên full test. Nguyên nhân:
   - ASLC300 chỉ có 300 gloss, có whitening, gallery là 3k clip;
   - full test có 2.731 gloss, raw, gallery 40k clip;
   - corpus pilot 3k video chỉ phủ 300 gloss.

   → Cần một dev metric **mô phỏng đúng protocol cuối** (VAL-DICT, §6.3) và một **corpus screening phủ đủ 2.731 gloss** (§9.1).
2. **Dị hướng (anisotropy).** Khoảng cách white−raw rất lớn: LOSO 25.4 so với 17.0 ở 16k. Protocol cuối lại là raw.
3. **Movement và non-manual hầu như không được mã hoá.** Probe R² < 0 ở **mọi** model skeleton. Các model RGB đạt 0.2–0.28. Có thể một phần do readout trung bình theo thời gian (§7.5).
4. **Signer leakage cao và không phân biệt được giữa các model** (signer probe 84–94%). Nghiên cứu trước trên feature SignRep (H010) cho thấy neighbour cùng signer chiếm 48.7%, cao hơn neighbour cùng gloss (43.8%).

Kendall τ với ASLC300 R@1, tính trên 50 model skeleton:

| Chỉ số | τ |
|---|---:|
| raw R@1 | .86 |
| probe handshape | .64 |
| probe location | .52 |
| effective rank | .45 |
| mean cosine | −.24 |
| signer probe | ≈ 0 |

Trên 69 model của mọi họ, effective rank chỉ còn τ = .14. → **Chỉ số hình học chỉ dùng làm cổng phát hiện collapse, không dùng để xếp hạng** (§6.4).

### 4.4 Không gian chưa từng thử

Toàn bộ chương trình nội bộ đến nay xoay quanh **một paradigm**: masked latent prediction với EMA teacher, trên **một kiểu token** (kênh âm vị học × 32 bin), **một corpus** (3k video) và **một kiểu readout** (mean-pool theo nhóm). Các họ tín hiệu dưới đây **chưa bao giờ được thử trong cùng codebase và cùng protocol** (chi tiết ở mục 7):

- **Paradigm/target:**
  - masked cluster prediction kiểu HuBERT/SHuBERT (đơn vị rời rạc học từ dữ liệu);
  - contrastive cấp clip với negative (alignment + uniformity);
  - DINO/iBOT với multi-crop thời gian (recipe SLiM đầy đủ);
  - teacher RGB frozen (cross-modal distillation);
  - frozen-teacher hai giai đoạn (kiểu SALT);
  - dự báo latent nhân quả theo thời gian (predictive coding);
  - target đa tầng nối lại (thay cho trung bình 4 tầng cuối mà REF đang dùng);
  - tinh chỉnh contrastive ngắn giai đoạn 2 (kiểu STARS/NNCLR);
  - random-teacher (control).
- **Masking:** motion-aware có ngẫu nhiên (MAMP), semantic tube theo thể tích không đổi (SLiM) với độ dài khoảng 0.5 giây, masking thích nghi theo loss (HPM/AdaMAE/ASGM).
- **View/invariance:** retarget skeleton về khung xương chuẩn, positive cross-video/cross-signer bằng nearest neighbour, signer-adversarial, view mạnh và corruption chỉ áp cho student (low-pass, mask token cho khớp thiếu).
- **Input/kiến trúc:** token per-joint compact (so sánh sạch trong cùng codebase), input đa luồng joint/bone/motion, chia sẻ encoder tay trội/không trội, token quan hệ tay–mặt/thân, model lớn hơn.
- **Readout:** ordered bins, activity-weighted, whitening fit trên train (không dùng nhãn), DTW/chamfer giữa các token cửa sổ.
- **Scale:** corpus 40k video ngay từ screening, đường cong dữ liệu 10/30/100%, lịch WSD để có nhiều mốc budget trong một run.

---

## 5. Hệ thống giả thuyết: điều ta tin về domain và cách kiểm chứng

Một method chỉ "được kiểm chứng đàng hoàng" khi mỗi module của nó trả lời được câu hỏi: **module này đúng vì giả thuyết nào về domain, và thí nghiệm nào đã có thể bác bỏ giả thuyết đó nhưng không bác bỏ được?**

Mục này đăng ký trước các giả thuyết, trước khi chạy bất cứ thứ gì. Các phase ở mục 9 tồn tại để kiểm chứng chúng. Kết quả âm của một giả thuyết cũng là tri thức về domain, và cũng công bố được.

### 5.1 Quy ước

Có ba tầng:

| Tầng | Ví dụ | Vai trò |
|---|---|---|
| **H0, câu hỏi gốc** | "Khoảng cách 16.7 → 49.95 đến từ đâu?" | Định hướng toàn chương trình |
| **H1–H12, giả thuyết domain** | "Target chỉ giúp khi mang thông tin mà input cục bộ không có" | Mỗi giả thuyết sinh ra một hoặc nhiều arm |
| **Dự đoán của arm (chữ ký cơ chế)** | "O7: σ_seed giảm ≥ 50%" | Điền trong run card (§6.4, Phụ lục B) |

Mỗi giả thuyết phải có đủ 6 trường:

1. Phát biểu.
2. Cơ sở: literature [V] và nội bộ [I].
3. Dự đoán định lượng.
4. **Tiêu chí bác bỏ.**
5. **Giải thích thay thế và control để loại trừ chúng.**
6. Hệ quả cho thiết kế method.

**Luật:**

- Không có arm nào không gắn với ít nhất một giả thuyết.
- Trạng thái của giả thuyết (OPEN / SUPPORTED / REFUTED / INCONCLUSIVE) chỉ được cập nhật theo luật quyết định §6.6, không theo cảm nhận.
- Một giả thuyết chỉ được coi là SUPPORTED khi cả *hiệu năng* (Δ VAL-DICT) **và** *cơ chế* (metric chữ ký) cùng đúng như dự đoán, **và** control đã loại được giải thích thay thế.

### 5.2 H0: nguồn gốc khoảng cách hiệu năng

- **Phát biểu.** Khoảng cách giữa skeleton SSL của ta (16.7 R@1 trên test) và SignRep (49.95) chủ yếu đến từ **(a) tín hiệu tự giám sát** và **(b) readout/cách đo**. Thông tin có trong input skeleton không phải nguyên nhân chính.
- **Cơ sở:**
  - pose có giám sát đạt 57.3–57.7 trên cùng protocol [V];
  - readout chiếm khoảng 12.5 R@1 trong con số của SignRep [V];
  - pilot ASLC300 so với full test: 48.9 so với 16.7 [I].
- **Dự đoán:**
  - cận trên có giám sát (A6a), cùng encoder và input, đạt **≥ 45** trên VAL-DICT;
  - readout tốt nhất (B1) thêm ≥ 3 R@1 so với mean.
- **Bác bỏ khi:** cận trên có giám sát < 35. Khi đó input (keypoint 2-D COCO-WholeBody, tập khớp) hoặc kiến trúc mới là nút thắt.
- **Hệ quả:**
  - nếu đúng: dồn lực vào họ O/K/V (Phase C);
  - nếu sai: Phase B3 trước (input, extractor), hạ ưu tiên mọi tinh chỉnh objective.

### 5.3 Giả thuyết về tín hiệu học (target)

**H1. Lượng thông tin mới của target.**

- **Phát biểu.** Một target tự giám sát chỉ cải thiện representation khi nó mang thông tin về gloss mà các token input cục bộ không trực tiếp chứa. Thông tin đó có thể đến từ ba nguồn: tích hợp ngữ cảnh (contextual latent), phân bố của toàn bộ dữ liệu (cluster/prototype), hoặc modality khác (RGB). Một target là hàm tất định cục bộ của input thì không giúp.
- **Cơ sở:**
  - SignRep: pixel → prior pose +37 R@1 [V], là một dạng cross-modal;
  - S-JEPA: target lấy ở output teacher hơn target lấy ở input +7.2 [V];
  - nội bộ: intrinsic, S1 và S2 đều âm; shufint ≈ PLboth [I].
- **Dự đoán:**
  - trên ≥ 6 loại target (REF-ctx, int, O1, O4, O7, O-ctrl), điểm P-T của target (VAL-DICT của chính target, đã trừ phần signer) có **Kendall τ ≥ 0.6** với VAL-DICT của student;
  - O-ctrl (random teacher) ≤ REF − δ_keep.
- **Bác bỏ khi:** τ < 0.3, **hoặc** random teacher ≈ REF. Trường hợp sau nghĩa là prior kiến trúc, chứ không phải ngữ nghĩa target, tạo ra kết quả.
- **Giải thích thay thế:** khác biệt về độ khó tối ưu hoặc thang loss. Control: chuẩn hoá thang loss theo từng target và ghi tỉ lệ gradient (M7).
- **Hệ quả:** nếu đúng, **P-T trở thành công cụ chọn target trước khi train**. Đây là một nguyên tắc thiết kế có thể tổng quát hoá, và tự nó là một đóng góp.

**H2. Dạng phân phối / rời rạc của target.**

- **Phát biểu.** Trên token skeleton, target dạng phân phối (CE trên prototype/cluster, có centering) cho embedding gộp phân biệt tốt hơn target hồi quy liên tục.
- **Cơ sở:**
  - SLiM: CE trên prototype 87.66 so với feature MSE 84.53 (collapse) [V];
  - S-JEPA: CE so với MSE +1.6 [V];
  - nội bộ: centered-CE hơn L1+SIGReg 13.8 điểm [I].
  - **Phản chứng nội bộ:** prototype loss −4.2 [I].
- **Dự đoán:** O1 (cluster CE) ≥ REF, và cluster vòng 1 tốt hơn vòng 0 (cluster tinh khiết hơn → R@1 cao hơn).
- **Bác bỏ khi:** O1 < REF − δ dù purity theo ASL-LEX của cluster cao. Điều kiện purity loại trừ khả năng "cluster kém".
- **Giải thích thay thế:** số cluster K không phù hợp. Control: thử hai giá trị K ở C2.

**H3. Target đa tầng.**

- **Phát biểu.** Nối target từ nhiều độ sâu của teacher vừa tăng chất lượng vừa giảm nhiễu theo seed, so với trung bình các tầng cuối.
- **Cơ sở:** Bootleg: concat 67.8 so với trung bình 44.1; seed std 2.9 → 0.6 [V].
- **Dự đoán:** O7 − REF ≥ +1 R@1, và σ_seed(O7) ≤ 0.5 σ_seed(REF). Đo bằng 3 seed ở C2.
- **Bác bỏ khi:** Δ ≤ 0 **và** σ không giảm.
- **Hệ quả:** nếu đúng, O7 thành mặc định của mọi arm ở Phase D. Nó vừa là module, vừa là công cụ giảm nhiễu giúp mọi so sánh sau rẻ hơn.

### 5.4 Giả thuyết về hình học embedding (local và global)

**H4. Dự đoán cục bộ không tạo ra không gian metric toàn cục.**

- **Phát biểu.** Masked prediction ở mức token cho token giàu thông tin nhưng embedding gộp lại dị hướng và kém tính metric. Một thành phần toàn cục *có negative* (instance discrimination) sửa được điều này cho retrieval.
- **Cơ sở:**
  - MAMP: linear 84.9 nhưng kNN khoảng 62; STARS +16.8 kNN [V];
  - SLiM: MFP + global 87.7 > MFP 85.3 > CL 73.6 [V];
  - nội bộ: anisotropy gap lớn; G1 (alignment *không* có negative) −5.7 [I].
- **Dự đoán:**
  - O8 hoặc O2h tăng VAL-DICT raw ≥ +3 và giảm anisotropy gap ≥ 50%;
  - O2 đứng một mình kém O2h (vẫn cần phần cục bộ).
- **Bác bỏ khi:** O8 ≤ REF + 1 **và** anisotropy gap không đổi.
- **Giải thích thay thế:** O8 thêm khoảng 10% số bước train. **Control: REF train thêm đúng số bước đó.** Một giải thích khác là whitening làm thay; control là so với REF + white-train readout. Nếu whitening cho cùng mức tăng, thành phần global chỉ là "whitening học được".

### 5.5 Giả thuyết về cấu trúc âm vị học của masking

**H5. Mask theo kênh âm vị học buộc model suy luận chéo giữa các kênh.**

- **Phát biểu.** Mask bám theo kênh âm vị học (cấu hình tay / vị trí-chuyển động / mặt) buộc model suy luận chéo kênh, tức là mã hoá danh tính từ vựng. Lợi ích phải xuất hiện ở MPA của tham số tương ứng.
- **Cơ sở:**
  - nội bộ: PL > R0 > R4, chênh 5–8 [I, pilot];
  - SLiM STM +3.5 [V];
  - lý thuyết shared-latent của Kong et al. [V-L2].
  - **Cảnh báo:** Kezar 2025 cho thấy tách bộ phận mà không có nhãn không cải thiện nhận dạng [V].
- **Dự đoán:** trên corpus 40k, REF − R0ctx ≥ +2 R@1 và MPA_handshape tăng ≥ 3.
- **Bác bỏ khi:** REF ≈ R0ctx trong khoảng ±1. Khi đó lợi thế PL cũ là do corpus pilot.
- **Giải thích thay thế:** khác biệt về tỉ lệ token bị che. Control: R0 và PL có cùng ngân sách che (đã khớp trong code PhonoJEPA, deviation #8).

**H6. Độ bền thời gian của mask.**

- **Phát biểu.** Có một độ dài tube tối ưu khoảng 0.3–0.6 giây. Ngắn hơn thì phần bị che bị nội suy ra (shortcut); che cả clip thì quá khó.
- **Cơ sở:** Skeleton2vec: hình chữ U ngược với đỉnh khoảng 15 frame; cả clip 78.2 < random 79.4 [V].
- **Dự đoán:** quét {4, 8, 16, 32} frame ở 25 fps (D2) cho đỉnh ở 8–16 frame. R² của P-M giảm đơn điệu theo độ dài.
- **Bác bỏ khi:** đường cong phẳng (dao động < 1 R@1). Khi đó độ dài tube không phải đòn bẩy, bỏ trục này.

### 5.6 Giả thuyết về movement và thời gian

**H7. Thiếu hụt movement do objective và readout, không do input.**

- **Phát biểu.** Thông tin movement có trong input (vận tốc cổ tay theo bin, quỹ đạo LOC) nhưng bị mất vì (a) readout trung bình theo thời gian và (b) objective không có áp lực dự đoán theo thời gian.
- **Cơ sở:**
  - mọi model skeleton có probe movement R² < 0, trong khi RGB đạt 0.2–0.28 [I];
  - reversal cosine 0.959 trên SignRep [I];
  - Yin et al.: pose có giám sát *thắng* về movement (80.8% wins) [V].

  Vậy skeleton có thể mã hoá movement nếu có tín hiệu đúng.
- **Dự đoán:**
  - (a) readout ordered-bins (B1) tăng MPA_movement ≥ +3 và velocity probe R² > 0;
  - (b) O6 hoặc I2 tăng MPA_movement ≥ +3 ở cùng readout R\*.
- **Bác bỏ khi:** cả readout lẫn objective đều không làm MPA_movement dịch chuyển. Kiểm tra thêm cận trên có giám sát (A6a): nếu nó cũng có MPA_movement thấp, giới hạn nằm ở input (nhiễu keypoint tay).
- **Hệ quả:** nếu (a) đúng mà (b) sai, movement là chuyện của readout và method không cần module thời gian riêng. Nếu (b) đúng, đóng góp "skeleton SSL mã hoá movement" có bằng chứng cơ chế.

### 5.7 Giả thuyết về nhiễu signer và handedness

**H8. Thông tin signer đang làm hại retrieval khác signer.**

- **Phát biểu.** Thông tin signer trong embedding *gây hại* cho retrieval giữa các signer. Loại nó đi (kể cả bằng nhãn) sẽ cải thiện VAL-DICT.
- **Cơ sở:**
  - neighbour cùng signer 48.7% > cùng gloss 43.8% trên SignRep [I];
  - pose nhận diện được signer ở 80% [V];
  - lý thuyết: JEPA với predictor gần-đồng-nhất ưu tiên feature bất biến theo thời gian, và danh tính signer là thứ bất biến nhất trong một clip [V 2511.09783].
- **Dự đoán:**
  - V3 (GRL oracle) tăng VAL-DICT ≥ δ_keep, với SNR giảm theo probe MLP post-hoc;
  - V1 (retarget) đạt ≥ 50% mức tăng của V3;
  - readout hiệu thời gian (B1) giảm SNR và tăng R@1. Trên feature SignRep, mức tăng này là +3.1 đến +3.8 [I].
- **Bác bỏ khi:** V3 giảm SNR rõ (probe post-hoc) mà VAL-DICT không đổi. Khi đó signer leakage không phải nút thắt → bỏ luận điểm T-D.
- **Giải thích thay thế:** GRL làm thay đổi tối ưu hoá theo cách khác. Control: GRL với signer ID xáo trộn.

**H8b. Canonical hoá tay trội/không trội loại bỏ một nuisance có thật.**

- **Cơ sở:** model có giám sát +5 R@1 nhờ canonical hoá tay [V, có cảnh báo về split]; handedness ngẫu nhiên nội bộ +1.6 [I].
- **Dự đoán:** I3 ≥ REF + 1; mức tăng không chỉ tập trung ở signer thuận tay trái.

**H8c. Nhiễu keypoint là một nuisance cần học bất biến.**

- **Phát biểu.** Rung tay và mất tay (do detector) là nuisance. View mạnh chỉ áp cho student, cùng với mask token cho khớp thiếu, cải thiện cả retrieval sạch lẫn độ bền.
- **Cơ sở:** BYOL-skeleton: augmentation mạnh 79.6 so với nhẹ 51.1, low-pass +2.4; SLiM: mask token 87.2 so với zero-fill 49.6 [V]. Hand AP của COCO-WholeBody chỉ 58–66 [V-L4]. JE thắng reconstruction khi nhiễu có biên độ lớn [V 2505.12477].
- **Dự đoán:** V4 ≥ REF (không giảm R@1 sạch); độ dốc suy giảm theo σ nhiễu trong M9 giảm ≥ 30%.
- **Bác bỏ khi:** V4 làm giảm R@1 sạch quá δ_eq mà robustness không cải thiện.

### 5.8 Giả thuyết về cross-modal, dữ liệu và độ tin cậy của pilot

**H9. RGB và skeleton bổ sung cho nhau, và có thể chưng cất.**

- **Phát biểu.** Feature RGB SSL chứa thông tin mà skeleton không có (vị trí tay so với mặt/thân, chi tiết ngón khi bị che), và thông tin đó chưng cất được vào student skeleton.
- **Cơ sở:** Yin et al.: RGB thắng về location (80.6 so với 67.7), pose thắng về handshape và movement [V]; prior của SignRep là trường hợp ngược lại [V].
- **Dự đoán:** O4 − REF ≥ +5 R@1, với mức tăng tập trung ở MPA_location; control teacher xáo trộn (feature của video khác) không tạo ra mức tăng.
- **Bác bỏ khi:** O4 ≤ REF + δ, **hoặc** teacher xáo trộn cho mức tăng tương đương. Trường hợp sau nghĩa là regularization chứ không phải thông tin.

**H10. Hiệu năng skeleton SSL đang bị giới hạn bởi dữ liệu.**

- **Phát biểu.** Ở quy mô ASL Citizen, hiệu năng bị giới hạn bởi dữ liệu, và objective tốt hơn có độ dốc theo dữ liệu lớn hơn.
- **Cơ sở:**
  - PCM3++: giảm một nửa dữ liệu mất 10.3 điểm retrieval [V];
  - đường cong scale của Sigma [V];
  - nội bộ: full data cho LOSO +4.2 [I];
  - MASA dùng 146K clip, ta chỉ có 40K.
- **Dự đoán:** E1 có độ dốc ≥ +3 R@1 mỗi lần gấp đôi dữ liệu, và FINAL dốc hơn REF.
- **Bác bỏ khi:** độ dốc < 1 → không đầu tư dữ liệu ngoài (SignVerse-2M).

**H11. Pilot có giá trị dự báo** (giả thuyết phương pháp luận).

- **Phát biểu.** Thứ hạng ở B_C1 trên corpus 40k dự báo được thứ hạng ở B_C2 (τ ≥ 0.6). Một phần kết luận cũ trên ASLC300 sẽ *không* giữ được trên VAL-DICT.
- **Kiểm chứng:** A3 (chấm lại), A5 (độ ổn định theo budget), C2 (τ giữa C1 và C2).
- **Hệ quả:** quyết định budget screening, và mức độ tin vào bảng §4.2.

**H12. Phần lớn lợi ích của các module cộng dồn được** (giả thuyết về tổ hợp).

- **Phát biểu.** Các module KEEP thuộc các họ khác nhau (O, K, V, I) có lợi ích phần lớn cộng dồn; module cùng họ thì thay thế cho nhau.
- **Cơ sở:** tương tác mask × target nội bộ rất mạnh: ctx tốt nhất khi đi với PL nhưng tệ nhất khi đi với R4 [I].
- **Dự đoán:** trong factorial D1, tương tác giữa các họ < ½ main effect; tương tác trong cùng họ âm.
- **Bác bỏ khi:** tương tác giữa các họ lớn. Khi đó method cuối phải được chọn như một tổ hợp, không ghép từng module.

### 5.9 Ma trận giả thuyết × thực nghiệm

| Giả thuyết | Arm / thực nghiệm | Metric chính | Metric cơ chế | Control loại trừ giải thích thay thế | Phase |
|---|---|---|---|---|---|
| H0 | A6a cận trên có giám sát; B1 readout | VAL-DICT | — | random-init, B0 | A, B |
| H1 | REF, int (cũ), O1, O4, O7, O-ctrl | τ(P-T, VAL-DICT) | P-T | random teacher; chuẩn hoá thang loss | C0–C2 |
| H2 | O1 (K = 2 mức), (O3) | VAL-DICT | cluster purity; M2 lift | — | C1–C2 |
| H3 | O7 × 3 seed | VAL-DICT, σ_seed | MPA_handshape | REF × 3 seed | C2 |
| H4 | O8 (a, b), O2, O2h | VAL-DICT raw | anisotropy gap | REF + thêm bước; REF + white-train | C1–C2 |
| H5 | REF so với R0ctx; K2 | VAL-DICT | MPA_handshape, group-drop | ngân sách che bằng nhau | A5, C1 |
| H6 | quét độ dài tube | VAL-DICT | R² của P-M | — | D2 |
| H7 | B1 ordered-bins; O6; I2; K1 | MPA_movement | velocity probe, ρ_rev | cận trên có giám sát (MPA_mov) | B, C |
| H8 | V3, V1, B1 temporal-diff | VAL-DICT | SNR@10, probe MLP post-hoc | GRL với nhãn signer xáo trộn | B, C |
| H8b | I3 | VAL-DICT | MPA trên sign hai tay | phân tầng theo signer thuận tay trái | C |
| H8c | V4 | VAL-DICT sạch + độ dốc robustness | M9 (dùng một bản lite ở C) | REF với cùng augmentation hiện có | C, F |
| H9 | O4; I4 (đối chứng cạnh tranh) | VAL-DICT | MPA_location | teacher xáo trộn; I4 lấy lại location mà không cần RGB? | C0–C2 |
| H10 | E1 (10/30/100%) | độ dốc VAL-DICT | — | cùng số epoch | E |
| H11 | A3, A5, τ(C1, C2) | τ | — | — | A, C |
| H12 | D1 factorial | main effect và tương tác | — | — | D |

### 5.10 Từ giả thuyết được ủng hộ đến nguyên tắc thiết kế method

Method cuối = **hội của các module mà mỗi module đứng sau một giả thuyết đã được ủng hộ**:

| Giả thuyết được ủng hộ | Nguyên tắc thiết kế rút ra |
|---|---|
| H1 | chọn target bằng P-T trước khi train |
| H2 / H3 | dạng target: phân phối, đa tầng |
| H4 | luôn kèm thành phần global có negative |
| H5 / H6 | mask theo kênh âm vị học, tube khoảng 0.5 giây |
| H7 | readout có thứ tự + áp lực dự đoán theo thời gian |
| H8 / H8b / H8c | module nuisance (retarget, handedness, view mạnh + mask token) |
| H9 | teacher cross-modal |
| H10 | đầu tư dữ liệu không nhãn |

Một claim về method trong paper sẽ có đủ bốn lớp bằng chứng:

1. Δ hiệu năng với CI, cả ở dev lẫn test.
2. Metric cơ chế dịch chuyển **đúng như dự đoán đã đăng ký trước**.
3. Control đã loại các giải thích thay thế.
4. Hiệu ứng giữ được khi scale (Phase E) và khi kết hợp với các module khác (Phase D/F).

Giả thuyết bị bác bỏ được ghi vào mục "điều không đúng trong domain này". Ví dụ: "target là hàm tất định của skeleton không giúp", hay "loại signer không cải thiện retrieval". Đây là tri thức giúp các đề xuất sau không lặp lại.

---

## 6. Khung đo lường: làm sao biết module nào mang tín hiệu

### 6.1 Năm nguyên tắc

1. **Đo đúng thứ sẽ báo cáo.** Metric chính khi phát triển phải giống protocol test cuối: full vocabulary, raw, cùng readout. Chỉ khác ở chỗ query lấy từ val.
2. **Tách representation khỏi readout.** Mọi so sánh giữa các module dùng *một* readout chuẩn R* được chốt ở Phase B. Cải thiện do readout được báo cáo riêng.
3. **Mỗi module phải khai báo trước "chữ ký cơ chế"** (mechanism signature): metric cơ chế nào phải dịch chuyển, theo hướng nào, nếu module hoạt động đúng như giả thuyết. Một module tăng R@1 mà chữ ký đi ngược hướng thì bị đánh dấu "nghi shortcut".
4. **Mọi Δ phải đặt cạnh mức nhiễu.** Nhiễu gồm seed σ (đo một lần ở Phase A) và bootstrap theo gloss.
5. **Một module chỉ bị bỏ khi có bằng chứng.** Bằng chứng là kiểm định tương đương (TOST) hoặc có hại. Việc "không thấy tăng" trong một run 1 seed **không** đủ để bỏ.

### 6.2 Bốn tầng đánh giá

| Tầng | Khi nào | Nội dung | Chi phí ước tính |
|---|---|---|---|
| **T0** | Trong lúc train | Từng thành phần loss; tỉ lệ norm gradient giữa các loss; teacher entropy và thống kê centering; RankMe và mean cosine của student/teacher trên một batch cố định 2.048 clip; collapse alarm | ≈ 0 |
| **T1** | Mỗi checkpoint (25/50/100% lịch) | **VAL-DICT-lite**: 2.000 query val cố định, gallery là toàn bộ train; raw và white-train; minimal-pair accuracy; signer-neighbour rate | ~3–5 phút |
| **T2** | Cuối run | **VAL-DICT đầy đủ** (§6.3) + bộ metric cơ chế M1–M8 + LOSO và ASLC300 (để giữ liên tục với các kết quả cũ) | ~15–20 phút |
| **T3** | Một lần, Phase F | Test ASL Citizen + WLASL2000 (cross-dataset). MS-ASL và robustness sweep là tuỳ chọn. | ~1 giờ |

### 6.3 Metric chính: VAL-DICT

- **Gallery:** toàn bộ 40.154 clip train, gồm 2.731 gloss và 35 signer, **giống hệt gallery của test**.
- **Query:** toàn bộ 10.304 clip val, gồm 6 signer, không trùng signer với train.
- **Điểm số:** embedding clip được L2-normalize rồi tính cosine. Điểm của một gloss là max cosine trên các clip gallery thuộc gloss đó, theo đúng protocol SignRep Table 4.
- **Metric:** R@1 (chính), R@5, MRR, DCG. Báo cáo cả micro và macro theo gloss.
- **Biến thể:**
  - **raw** (chính);
  - **white-train**: PCA-whitening chỉ fit trên feature gallery train, không dùng nhãn và không nhìn query, nên không transductive;
  - LOSO-white cũ, chỉ để tham chiếu.
- Gallery là các video đã thấy khi pretrain (không nhãn). Điều này giống hệt protocol test, nên không phải là leak.

### 6.4 Bộ metric cơ chế

Bộ metric này là công cụ để biết **tín hiệu nào đến từ đâu**.

| ID | Metric | Định nghĩa | Trả lời câu hỏi |
|---|---|---|---|
| **M1** | **Minimal-pair accuracy** theo tham số âm vị học $p$ ∈ {handshape, location, movement, sign-type} | Với các cặp gloss $(g,g')$ mà mã ASL-LEX chỉ khác nhau ở nhóm $p$, và mỗi query val $q$ thuộc $g$: tính $\mathrm{MPA}_p=\Pr[s(q,g)>s(q,g')]$, với $s$ là điểm gloss của VAL-DICT. Mức may rủi là 50%. Báo cáo Δ so với sàn random-init. Code có sẵn: `method/p4_signssl/p4/phonology.py`. | Module có làm representation phân biệt được tham số mà nó nhắm tới không? |
| **M2** | **Error attribute-sharing lift** | Trong các lỗi rank-1: $\text{lift}_p=\Pr[\text{top-1 sai chung } p]/\Pr[\text{gloss ngẫu nhiên chung } p]$ | Model đang "nhầm theo" tham số nào? |
| **M3** | **Group-only / group-drop readout** | R@1 khi chỉ dùng nhóm token G, và ΔR@1 khi bỏ G. Các nhóm: CONF và LOC của tay trội, tay không trội, UP, MO, HD. Không cần train lại. | Bộ phận nào mang thông tin? Module có chuyển thông tin sang bộ phận dự kiến không? |
| **M4** | **Độ nhạy với thứ tự thời gian** | (a) $\rho_{rev}$ = cos giữa embedding của clip và của clip đảo ngược; (b) R@1 với query đảo ngược; (c) mức tăng của readout ordered-bins so với mean | Có mã hoá động học/movement không? |
| **M5** | **Signer leakage** | (a) **SNR@10**: trong pool val, tỉ lệ 10 neighbour gần nhất (trừ chính video đó) cùng signer nhưng khác gloss, chuẩn hoá theo tỉ lệ may rủi; (b) khoảng cách R@1 giữa gallery cùng signer và khác signer; (c) ABX across-signer trừ ABX within-signer; (d) signer probe *mới, post-hoc*, cả linear lẫn MLP 1 tầng, train trên các gloss bị giữ lại. Lý do cần (d): trong speech, chuẩn hoá theo utterance làm linear speaker probe giảm từ 93.4% xuống 14.8%, nhưng MLP vẫn đạt 89.0% [V 2108.00917]; adversary lúc train về mức may rủi nhưng attacker post-hoc vẫn khôi phục được [V 1808.06640]. | Module có *thật sự* giảm nhiễu signer, hay chỉ làm nó khó thấy bằng probe tuyến tính? |
| **M6** | **Probe ΔR² so với random-init** | Probe hình học sẵn có (handshape, location/relations, body, movement, non-manual), trừ đi R² của random-init (ví dụ handshape: 0.22). Thêm **probe tuyến tính cho vận tốc/gia tốc** của cổ tay và ngón: latent JEPA từng được chỉ ra là "mù vận tốc" [S 2610.04585]. Probe phonology chạy kèm **control task** (gán nhãn ngẫu nhiên theo từng gloss) và báo cáo selectivity [V 1909.03368]. | Thông tin đến từ học hay đã có sẵn ở input? Probe đọc representation hay đang tự ghi nhớ? |
| **M7** | **Health và proxy không nhãn** | *Cổng:* RankMe (tính trên singular value σ, **không** phải trên eigenvalue σ²); TE32 = 1 − Σ_{i≤32}λ_i/Σλ_i; std theo từng chiều của target teacher; tỉ lệ gradient giữa các loss. *Xếp hạng trong cùng một họ, chỉ sau khi qua cổng:* **LiDAR** (dùng một họ view chung cho mọi arm) và **TwoNN intrinsic dimension** (thấp hơn là tốt hơn). *Mô tả:* mean cosine; **anisotropy gap** = R@1(white-train) − R@1(raw). | Collapse? Một loss lấn át loss khác (bài học từ SignRep repro)? Có proxy không nhãn nào xếp hạng đúng không? |
| **M8** | **Chi phí** | GPU-h, số tham số, clip/s khi train, độ trễ inference | Δ trên mỗi GPU-h |
| M9 | Robustness (đầy đủ ở Phase F; bản lite với 2 mức nhiễu cho V4 ở Phase C) | Quét nhiễu Gaussian theo kích thước tay (tương đương 5–20 px); bỏ một tay trong 20% số frame, đi qua mask token | Bền với nhiễu keypoint? |

**Bảng chữ ký cơ chế.** Mỗi họ module ở mục 7 phải điền dự đoán trước khi chạy. Ví dụ:

| Module | Dự đoán nếu đúng cơ chế | Dấu hiệu shortcut |
|---|---|---|
| Masking motion-aware | MPA_movement ↑, ρ_rev ↓, M6 movement ↑ | R@1 ↑ nhưng MPA_movement đứng yên và SNR ↑ |
| Retarget skeleton chuẩn | SNR ↓, cross-signer gap ↓, MPA không đổi | R@1 ↓ mạnh: mất thông tin handshape |
| Teacher RGB | MPA_handshape ↑, R@1 ↑ | SNR ↑: teacher mang ngoại hình signer |
| Contrastive cấp clip | anisotropy gap ↓, R@1 raw ↑ | SNR ↑: phân biệt instance theo signer |
| Cluster prediction | M2 lift_handshape ↑; cluster purity theo ASL-LEX cao | cluster purity theo signer cao |

### 6.5 Pre-screen không cần train: tín hiệu trước khi bỏ GPU

- **P-T: chất lượng target** (cho mọi đề xuất target: descriptor, cluster ID, feature RGB, random network, EMA teacher). Dùng chính target $y(x)$, pool theo thời gian, làm representation và đo:
  - VAL-DICT R@1 của $y$;
  - SNR@10 của $y$;
  - MPA của $y$.

  **Giả thuyết:** target tốt phải tự nó mang thông tin gloss và ít mang thông tin signer. Dữ liệu nội bộ đã ủng hộ điều này: intrinsic target có chất lượng xấp xỉ descriptor B0 (17.3) và không giúp gì (shufint ≈ PLboth). EMA-contextual target thì mang thông tin và có giúp.

  **Luật:** target có VAL-DICT ≤ B0 *và* SNR cao hơn B0 sẽ bị hạ ưu tiên, trừ khi có lý do cơ chế khác.
- **P-M: độ khó của mask.** Với mỗi họ mask, đo R² khi dựng lại feature của token bị che từ token còn thấy, bằng hai cách: (a) nội suy tuyến tính theo thời gian; (b) ridge regression từ các token cùng thời điểm.

  | R² | Đánh giá |
  |---|---|
  | > 0.9 | quá dễ, có shortcut |
  | ≈ 0 | quá khó |
  | ở giữa | họ mask nên nằm ở vùng này |

  Cơ sở lý thuyết: Kong et al., và quan sát của SimMIM về AvgDist (§3.5).
- **P-V: view có giữ nội dung và xoá nuisance không?** Với một augmentation hoặc retarget $a$: so sánh khoảng cách descriptor B0 giữa $x$ và $a(x)$ với khoảng cách giữa hai signer khác nhau ký cùng một gloss. Một view tốt làm thay đổi phần signer nhiều hơn phần gloss.

### 6.6 Thống kê và luật quyết định

Ký hiệu: $\Delta$ = metric(arm) − metric(ref), tính trên từng query (paired).

- **CI 95%:** bootstrap theo cụm gloss (resample gloss cùng toàn bộ query của gloss đó), B = 2.000.
- **Tính nhất quán theo signer:** số signer val có Δ > 0, trên 5 signer hiệu dụng (bỏ P13).
- **Seed noise:** $\sigma_s$ đo bằng 3 seed của cấu hình tham chiếu ở budget screening (Phase A, A4). Với hai arm mỗi arm $n$ seed, $\sigma_\Delta=\sigma_s\sqrt{2/n}$.
- **Hiệu ứng nhỏ nhất phát hiện được** (power 80%, α = 0.05): $\mathrm{MDE}\approx 2.8\,\sigma_\Delta$. Nếu MDE của một so sánh lớn hơn hiệu ứng kỳ vọng, phải tăng seed hoặc dùng thiết kế factorial (xem dưới).

**Luật bốn nhánh** cho mỗi module, so với tham chiếu cùng budget:

| Kết luận | Điều kiện |
|---|---|
| **KEEP** | $\Delta \ge \delta_{keep}=\max(1.0,\,2\sigma_\Delta)$ R@1 **và** CI_low > 0 **và** ≥ 4/5 signer **và** chữ ký cơ chế không ngược hướng **và** M7 không xấu đi |
| **DROP (vô hiệu)** | CI 90% nằm trọn trong $(-\delta_{eq}, +\delta_{eq})$ với $\delta_{eq}$ = 1.0 R@1 (TOST tương đương) **và** module làm tăng chi phí hoặc độ phức tạp |
| **DROP (có hại)** | CI_high < 0 |
| **INCONCLUSIVE** | Các trường hợp còn lại. Chỉ chạy thêm seed nếu module rẻ *hoặc* chữ ký cơ chế dương rõ. |
| **KEEP-MECH** | R@1 không đổi nhưng metric chữ ký tăng ≥ 3 lần SE (ví dụ MPA_movement +5). Giữ làm ứng viên cho Phase D/E, vì hiệu ứng có thể chỉ hiện ra ở scale lớn hoặc ở task khác. |

**Bổ sung khi có ≥ 3 seed mỗi arm:**

- **Probability of improvement:** $P(A>B)=\frac{1}{k^2}\sum_{i,j}\mathbb 1[a_i>b_j]$. KEEP thêm điều kiện P(A>B) ≥ 0.75, theo ngưỡng của Bouthillier et al. [V 2103.03098].
- **Permutation đảo dấu theo cụm signer:** $T=\sum_c\bar\Delta_c$, đảo dấu độc lập từng cụm. Với 5 cụm hiệu dụng, p nhỏ nhất đạt được (hai phía) là 2/32 = 0,0625. **Vì vậy kiểm định theo signer trên val chỉ dùng làm kiểm tra tính nhất quán, còn CI chính lấy từ bootstrap theo gloss.**

**Tổng nhiễu của một so sánh 1 run:** $\sigma^2_\Delta\approx2\sigma^2_{seed}+2\sigma^2_{ckpt}+\sigma^2_{test,paired}$.

- $\sigma_{ckpt}$ là độ lệch chuẩn tương đối của 3–5 checkpoint cuối. Có miễn phí nhờ lịch WSD [V 2508.13144].
- **Kỳ vọng:** literature cho thấy JEPA nhiễu theo seed hơn MAE nhiều. Ví dụ I-JEPA ViT-S đạt 51.5 ± 2.9 so với MAE 43.2 ± 0.2 [V 2603.15553]. Vì vậy σ_s phải được **đo riêng cho từng họ objective**, không lấy một giá trị dùng chung.

**SNR của metric:** SNR = (độ phân tán giữa các arm) / (độ lệch chuẩn giữa các checkpoint cuối) [V 2508.13144, R = 0.79 với độ chính xác quyết định]. Metric có SNR < 2 sau C1 bị loại khỏi vai trò cổng, chỉ giữ để mô tả.

**Sửa lỗi so sánh bội:**

- Screening (Phase C1) là exploratory: không claim p-value, chỉ xếp hạng và chọn nhánh vào vòng sau.
- Confirmatory (C2, D, F) dùng Holm trong từng phase.

**Factorial (Phase D):** ước lượng main effect và tương tác bằng hồi quy logistic trên tính đúng của từng query (predictor là các factor được mã hoá ±1), CI bằng gloss bootstrap. Một tương tác được đánh dấu "không cộng dồn" nếu |tương tác| > ½ |main effect|.

**Lợi thế ẩn của factorial:** với $2^{k-p}$ run, mỗi main effect được ước lượng từ một nửa số run so với nửa còn lại. Vì vậy 1 seed mỗi ô vẫn cho $\sigma_{effect}=2\sigma_s/\sqrt{N}$. Với 16 run, $\sigma_{effect}=\sigma_s/2$, tốt hơn so sánh 1 nhân tố mỗi lần (OFAT) 2 seed với cùng chi phí, và còn đo được tương tác.

### 6.7 Proxy validity và độ ổn định thứ hạng

- **Giữa các mức budget:** Kendall τ giữa thứ hạng ở C1 (budget ngắn) và C2 (budget dài), tính trên các arm đi qua cả hai. Nếu τ < 0.6, tăng budget của C1 cho các vòng sau và ghi chú rằng kết quả C1 không đáng tin.
  - Literature cho kỳ vọng: xếp hạng ở scale nhỏ đúng khoảng 80% số quyết định cặp khi lên scale lớn, và không phương pháp ngoại suy scaling-law nào thắng việc xếp hạng ở một scale nhỏ [V 2504.11393]. Tương quan thứ hạng giữa các scale của DataComp là 0.71–0.90 [V 2304.14108].
  - Nhưng với JEPA, rank của representation có thể tăng sớm rồi giảm trong khi accuracy vẫn tăng, nên RankMe sớm có thể chỉ sai hướng [V 2312.04000].
  - **Hệ quả:** hiệu ứng lớn thì tin được ở C1; hiệu ứng nhỏ hoặc module mà lợi ích là "scale tốt hơn" thì đánh dấu **DEFERRED** (để Phase E), không DROP.
- **Proxy không nhãn:** trong 10–20 run đầu (A3 và C1), tính τ giữa từng proxy (RankMe, LiDAR, ID, VAL-DICT-lite ở 25/50/100%) và VAL-DICT đầy đủ. Một proxy chỉ được dùng làm cổng xếp hạng nếu cận dưới CI của τ > 0.4 và **top-1 regret** (điểm của run tốt nhất thật trừ điểm của run mà proxy chọn) < MDE. Mức tham chiếu từ literature: LiDAR τ 0.55–0.82, RankMe τ 0.22–0.58 trên masked SSL [V 2312.04000].
- **Giữa các corpus:** Phase A kiểm tra xem kết luận cũ (pilot 3k video) có giữ được trên corpus 40k không (A3, A5).
- **Giữa dev và test:** chỉ biết ở Phase F. Do VAL-DICT được dựng *giống hệt* protocol test, đây là proxy tốt nhất có thể có mà không chạm vào test.

---

## 7. Không gian thiết kế: các họ tín hiệu và giả thuyết

**Cách đọc mục này.** Mỗi họ gồm: cái gì → vì sao (literature + khoảng trống nội bộ) → các arm đại diện → chữ ký cơ chế → chi phí → ưu tiên.

Mỗi họ chỉ chọn **1–3 arm đại diện, khác nhau về bản chất**. Không quét hyperparameter ở giai đoạn này: quét chỉ làm ở Phase D cho các họ đã sống sót. Ký hiệu chi phí implement: S < 1 ngày, M 1–3 ngày, L > 3 ngày.

**Backbone chung để so sánh công bằng** (trừ họ I):

- encoder Transformer 256-d, 6 block;
- token theo kênh âm vị học × bin thời gian như PhonoJEPA;
- cửa sổ 64 frame ở ≤ 25 fps;
- cùng optimizer, cùng số bước, cùng corpus.

Mỗi arm thay **đúng một** thành phần so với `REF` = PLctx (PL mask + contextual EMA target + centered-CE). Riêng họ O thay toàn bộ objective/target.

### 7.1 Họ O: paradigm objective / target

Literature cho thấy *cái được dự đoán* quan trọng hơn *chỗ bị che*. Ví dụ: MAMP đổi target cho +10.1, đổi mask chỉ +1.2; SLiM cho thấy CE trên phân phối prototype tốt hơn MSE feature khoảng 3 điểm (§3.3, §3.5). Thế nhưng nội bộ mới chỉ thử một dạng target latent (EMA-ctx) và các descriptor tường minh. Đây là họ có kỳ vọng lớn nhất.

| ID | Arm | Cơ sở | Chữ ký cơ chế | Chi phí | Ưu tiên |
|---|---|---|---|---|---|
| **O1** | **Masked cluster prediction** kiểu HuBERT/SHuBERT. Vòng 0: k-means (K ≈ 256–1.024 cho mỗi loại kênh) trên feature kênh hand-crafted. Vòng 1: k-means trên feature của REF. Dự đoán ID cluster của token bị che bằng CE. | HuBERT (speech); SHuBERT (sign, trên crop RGB) cho thấy target đơn vị rời rạc lặp lại hiệu quả cho ký hiệu. Với skeleton, cluster của kênh CONF có thể tự khám phá "handshape units". | Cluster purity theo ASL-LEX handshape > theo signer (P-T); M2 lift_handshape ↑; MPA_handshape ↑ | M (k-means offline + head CE) | **Cao** |
| **O2** | **Contrastive cấp clip** (InfoNCE/MoCo-v3, queue negative, augmentation mạnh trên skeleton: crop/tốc độ thời gian, affine, mirror, joint dropout) | Skeleton contrastive (AimCLR, ActCLR, CMD…) mạnh về linear eval. Trên sign, contrastive STC là pose SSL frozen tốt nhất trên WLASL (32.35 so với 18–25 của các phương pháp masked). Ta có anisotropy lớn; contrastive tối ưu trực tiếp uniformity. | anisotropy gap ↓; R@1 raw ↑. Nguy cơ: SNR ↑ do phân biệt instance theo signer. | M | **Cao** |
| O2h | O2 + REF, tức ctx + InfoNCE (hybrid) | Khác G1 (G1 chỉ alignment, không negative, nên bị −5.7): có negative thì có lực đẩy uniformity. | như O2, giữ MPA của REF | S (sau khi có O2) | Trung bình |
| **O3** | **DINO/iBOT + multi-crop thời gian**: recipe SLiM đầy đủ (MFP trên token + GLCL trên CLS, local view 8–32 frame lấy mẫu lại từ cùng đoạn, Sinkhorn/KoLeo) | SLiM: GLCL+MFP, prototype-CE tốt hơn MSE khoảng 3 điểm. Nội bộ G1 và proto đã âm, nhưng **chưa từng thử recipe hoàn chỉnh**; các mảnh lẻ có thể thất bại vì thiếu thành phần đi kèm. | RankMe ổn định; R@1 ↑; local view → M4? | M–L | Trung bình (có prior âm) |
| **O4** | **Teacher RGB frozen → student skeleton** (cross-modal): student dự đoán feature cửa sổ của SignRep public (Hiera, 16 frame) từ token skeleton. Dùng làm aux loss cạnh REF và làm arm độc lập. | Distillation từ frozen teacher (SALT, V-JEPA 2.1 distillation). SignRep mạnh ở handshape nhờ pixel. Teacher label-free, nhưng pretrain trên YT-SL-25 và **chọn checkpoint bằng WLASL val** → phải khai báo. | MPA_handshape ↑; R@1 ↑. Nguy cơ: SNR ↑ (teacher mang ngoại hình). | M (feature SignRep cho 40k clip **phải trích lại**, ~6–10 GPU-h) | **Cao**, nhưng chỉ chạy sau khi P-T dương |
| **O5** | **Frozen-teacher hai giai đoạn** (SALT-style): đóng băng teacher REF@B_C2, train student mới với mask mạnh hơn; tuỳ chọn lặp 2 vòng | EMA vừa tốn kém vừa bất ổn; frozen teacher tách hai vai trò và cho phép dùng mask khó hơn. Tương tự vòng lặp của HuBERT. | Đường học dốc hơn; RankMe ổn định | S–M | Trung bình |
| **O6** | **Dự báo latent nhân quả theo thời gian** (predictive coding/DPC): predictor chỉ thấy các bin ≤ t, dự đoán latent teacher ở các bin t+1…t+k | OpenHands DPC trên pose tốt hơn reconstruction. Movement đang là lỗ hổng lớn nhất (probe R² < 0). V-JEPA causal kém multi-block với video, nhưng với movement của ký hiệu thì chưa ai đo. | MPA_movement ↑; ρ_rev ↓ (nhạy thứ tự); M6 movement ↑ | S | **Cao** |
| **O7** | **Target đa tầng nối lại** (multi-level, kiểu "Bootleg"): nối output đã z-score của các block teacher {1, 3, 6} (encoder 6 block), thay cho *trung bình* LayerNorm 4 block cuối mà REF đang dùng | Trên IN-1k, ViT-S [V 2603.15553]: concat {1,4,8,12} đạt 67.8; trung bình kiểu data2vec 44.1; chỉ block cuối 60.2. Seed std giảm từ 2.9 (I-JEPA) xuống 0.6; gần như không nhạy với EMA τ (dao động < 1.3 so với 20 điểm). **REF đang dùng đúng kiểu trung bình bị chỉ ra là kém.** | R@1 ↑; **σ_seed ↓** (đo ở C2); MPA_handshape ↑ (tầng thấp giữ hình học) | S | **Cao** |
| **O8** | **Tinh chỉnh contrastive ngắn giai đoạn 2** (kiểu STARS): lấy checkpoint REF, tinh chỉnh nửa sau của encoder bằng NNCLR trong khoảng 10% budget (layer-decay). Có hai biến thể: (a) positive là neighbour gần nhất *khác video*; (b) như (a) nhưng loại neighbour có pseudo-signer giống (cụm descriptor tư thế nghỉ, không dùng nhãn). Biến thể (b) chính là ý tưởng "positive cross-signer" (trước đây là V2). | Masked prediction thuần cho feature kNN kém: MAMP linear 84.9 nhưng kNN chỉ khoảng 62–63. STARS tinh chỉnh NNCLR 20 epoch → **kNN 79.9 (+16.8)**, linear chỉ +2.2 [V 2407.10935]. Metric chính của ta là retrieval. | R@1 raw ↑; anisotropy gap ↓; (b): SNR ↓ | S | **Cao** |
| O-ctrl | **Random teacher**: target là feature của một mạng random-init đóng băng | Control quan trọng, cho biết "ngữ nghĩa của teacher" đóng góp bao nhiêu so với prior kiến trúc. BYOL: target random cố định chỉ đạt 18.8 so với 72.5 [V 2006.07733]. | — | S | Bắt buộc (control) |

**Không đưa vào:**

| Ứng viên | Lý do loại |
|---|---|
| VICReg/Barlow/SIGReg làm objective chính | L1+SIGReg đã −13.8; lần tái tạo SignRep sụp đổ vì var/cov |
| Diffusion (MacDiff) | Chi phí L, chưa có lợi thế frozen rõ ràng so với O1–O3 trên skeleton |
| Tokenizer VQ-VAE | Trùng ý tưởng với O1 nhưng tốn hơn; chỉ dùng nếu O1 dương mà k-means là nút thắt |
| Path-signature target | S2 đã âm rõ |

### 7.2 Họ K: chiến lược masking (giữ objective của REF)

Literature (§3.3, §3.5) cho thấy: mask có cấu trúc giúp khi nó (a) chặn shortcut nội suy, (b) giữ context giàu thông tin, (c) đặt loss trên token bị che. Trên skeleton, hiệu ứng là +1.2 đến +3.6 và **phụ thuộc target**. Cảnh báo cũ của L2 rằng "target latent làm giảm hiệu ứng mask" không đúng với skeleton (báo cáo B §6). Hiệu ứng kỳ vọng nhỏ hơn họ O, nên họ K có ưu tiên **thấp hơn**, nhưng rẻ.

| ID | Arm | Cơ sở | Chữ ký | Chi phí |
|---|---|---|---|---|
| **K1** | **Motion-aware có ngẫu nhiên**: điểm chọn block = U(0,1) + β·(năng lượng chuyển động *tương đối với cổ tay* của ngón, cộng chuyển động cổ tay), β ≈ 0.1. **Không bao giờ chọn tất định.** | Mức tăng nhỏ và dễ vỡ: MAMP +1.2; Skeleton2vec +0.5 bên trong tube. β = 0.3 mất 4 điểm; bỏ tính ngẫu nhiên (deterministic top-k) làm MAMP rơi từ 84.9 xuống 66.7 [V 2409.17951]. Rủi ro riêng của ký hiệu: chuyển động bị chi phối bởi tịnh tiến của cánh tay. | MPA_movement ↑; ρ_rev ↓ | S |
| **K2** | **Semantic tube thể tích không đổi** (SLiM STM): bộ phận nhỏ (ngón, miệng) bị che lâu hơn, bộ phận lớn bị che ngắn hơn; tỉ lệ ~U(0.5, 0.9); **tube dài khoảng 0.3–0.6 giây, không che cả clip** | SLiM +3.5 so với joint masking, nhưng khoảng +2.0 trong đó đến từ vai trò augmentation ở local view. Skeleton2vec: tube khoảng 15 frame +3.6 so với random, nhưng **tube cả clip (78.2) còn tệ hơn random (79.4)** [V 2401.00921]. Cấu trúc mask có lợi khi loss đặt trên token bị che, và có thể có hại khi target là toàn chuỗi được điều kiện hoá toàn cục (MacDiff: random 86.4 > tube 83.1). | MPA_handshape ↑ | S |
| **K3** | **Thích nghi theo loss**, từ dễ đến khó: chọn kênh/thời đoạn để che theo running loss (HPM / AdaMAE / ASGM-lite), có ngân sách cố định | HPM, AdaMAE, SignSeek ASGM +3.1 so với luôn che tay trội | phân phối mask dịch dần về token khó; R@1 ↑ | M |

**Không đưa vào:**

- R1 finger-group, R4 articulator-span, curriculum G2: đã thử và âm.
- Tỉ lệ mask: là hyperparameter, quét ở Phase D cho cấu hình thắng cuộc.

### 7.3 Họ V: view, bất biến và nhiễu signer

Khoảng trống 4 ở §4.3: signer leakage cao và chưa có phương pháp skeleton SSL label-free nào nhắm thẳng vào nó.

| ID | Arm | Cơ sở | Chữ ký | Chi phí |
|---|---|---|---|---|
| **V1** | **Retarget về khung xương chuẩn**: thay độ dài từng xương bằng độ dài template, giữ hướng xương. Tất định, áp ở input cho cả train và test. | SignRep chuẩn hoá bone-length cho *target*. Identity rò qua tỉ lệ xương. G3morph (augmentation ngẫu nhiên) ≈ 0, nhưng chuẩn hoá tất định là cơ chế khác. | SNR ↓; cross-signer gap ↓; MPA giữ nguyên | S |
| (V2) | *Gộp vào O8(b)*: positive cross-video/cross-signer bằng nearest neighbour | Positive cùng video luôn cùng signer, nên invariance học được bị lệch về signer. p4 H012b cho thấy precision khi mining không truyền sang student trong setting cũ. | — | — |
| V3 | **Signer-adversarial (GRL) với participant ID** | Khai báo là weak supervision; dùng làm **cận trên chẩn đoán**: loại signer có giúp retrieval bao nhiêu? Đánh giá bằng probe post-hoc *mới* (M5d), vì adversary về mức may rủi không có nghĩa là thông tin đã bị xoá. | SNR ↓↓ | S |
| **V4** | **View mạnh và corruption chỉ cho student:** low-pass Savitzky–Golay; scaling giữ độ dài xương; dropout cả tay qua **mask token** (không zero-fill); nhiễu tương đối theo kích thước tay; tuỳ chọn mix vùng tĩnh (thân, tay không trội lúc nghỉ) kiểu ActCLR | BYOL-skeleton: augmentation mạnh 79.6 so với nhẹ 51.1; low-pass +2.4 [V 2202.02232]. SLiM: scaling giữ xương +1.3; khớp thiếu đi qua mask token 87.2, so với parent-fill 66.5 và zero-fill 49.6 [V]. ActCLR: augment vùng tĩnh +4.8 kNN [V]. S-JEPA: view chỉ cho student +0.7. | M9 robustness ↑; R@1 ↑ hoặc không giảm | S |

### 7.4 Họ I: input, token hoá, kiến trúc

| ID | Arm | Cơ sở | Chữ ký | Chi phí |
|---|---|---|---|---|
| **I1** | **Token per-joint compact** (SLiM: 8 frame mỗi token, khoảng 60–100 joint × 8) so với token theo kênh. **Cùng codebase, cùng objective.** | B-series (per-joint) yếu nhưng khác codebase, nên chưa kết luận được. SLiM cho thấy token compact giữ hiệu năng với chi phí giảm 7.9×. | M3 phân bố thông tin; R@1 | M |
| **I2** | **Input đa luồng** joint + bone + motion (vận tốc) | Có giám sát: SL-GCN 4 luồng hơn 1 luồng khoảng 6 điểm; MAMP lấy motion làm target. Motion **làm input** chưa từng thử. | MPA_movement ↑; M6 movement ↑ | S |
| I3 | **Chia sẻ encoder tay trội/không trội** + mirror | Các model ISLR tốt nhất đều tách và canonical hoá tay (STC khoảng +5). Handedness ngẫu nhiên nội bộ +1.6 (chưa kết luận). | MPA_handshape ↑ trên các sign hai tay | S |
| **I4** | **Token quan hệ tay–mặt/thân**: thêm một kênh mã hoá vector dịch chuyển cổ tay và đầu ngón tới các mốc mặt và thân (tương tự prior P^{b,d} của SignRep, nhưng làm *input*). Phải lan truyền mask: khi kênh tay bị che thì kênh quan hệ tương ứng cũng bị che. | Yin et al.: skeleton thua RGB về location (67.7 so với 80.6% wins), vì chuẩn hoá vị trí làm mất vị trí tay so với cơ thể [V]. Path-JEPA: token quan hệ (edge, chain) +1.8 [S]. Đóng vai **control cạnh tranh cho O4** ở H9: nếu I4 lấy lại được MPA_location thì không cần RGB. | MPA_location ↑ | S |
| (I5) | Model M (khoảng 20M) | → Phase E (scale), không screening | | |

**Không đưa vào:** lifting 2D→3D (MotionBERT không phủ bàn tay; HaMeR quá đắt cho 83k video) và trích lại keypoint bằng extractor khác (83k video). Hai hướng này chỉ mở lại nếu Phase B chứng minh input là nút thắt.

### 7.5 Họ R: readout (chỉ đánh giá, không train)

Dữ liệu nội bộ: readout grp hơn CLS 31 điểm; trên SignRep, ordered bins +5.9 so với avg (C012); readout chamfer theo hiệu thời gian T4 đạt 0.726, là readout không cần train tốt nhất trên feature SignRep. Probe movement âm có thể một phần vì readout lấy trung bình theo thời gian.

| ID | Readout | Ghi chú |
|---|---|---|
| R1 | mean (toàn bộ token) | giống protocol test hiện tại |
| R2 | concat trung bình theo nhóm (dom, ndom, face) | primary hiện tại |
| R3 | ordered bins K = 2 / 4, concat theo thứ tự | giữ thứ tự thời gian |
| R4 | activity-weighted (trọng số = năng lượng chuyển động tay) | tương tự weighted của SignRep |
| R5 | R* + white-train | không dùng nhãn, không transductive |
| R6 | chamfer/DTW giữa chuỗi token | đắt hơn khi query; báo cáo riêng |

**Luật:** chốt **R\*** là readout tốt nhất trong nhóm R1–R5 trên VAL-DICT, với điều kiện không dùng nhãn. R6 chỉ để tham khảo, vì là matching chứ không phải vector embedding. **Mọi so sánh từ Phase C trở đi dùng R\*.**

### 7.6 Họ S: scale, dữ liệu, lịch học

| ID | Arm | Mục đích |
|---|---|---|
| S1 | Đường cong dữ liệu 10% / 30% / 100% của 40k video ở cùng số epoch | Ước lượng độ dốc: có nên đầu tư dữ liệu không nhãn bên ngoài không? |
| S2 | Lịch **WSD** (warmup–stable–decay) với cooldown ngắn tại 16k / 64k / 200k bước | Lấy nhiều mốc budget từ một run; kiểm tra thứ hạng có đảo khi train dài không (V-JEPA 2 dùng warmup-constant-cooldown) |
| S3 | Model S (5M) và M (khoảng 20M) ở full data | Capacity có phải nút thắt không? |
| S4 | Dữ liệu không nhãn ngoài (pose YouTube-ASL / How2Sign / train của WLASL hoặc MS-ASL) | **Có điều kiện:** chỉ khi S1 cho độ dốc dương rõ. Phải loại video trùng với các tập test. |

### 7.7 Bảng ưu tiên

Mỗi tiêu chí chấm 1–3:

- **E**: bằng chứng literature hoặc nội bộ;
- **G**: lấp được khoảng trống đo được ở §4.3;
- **C**: rẻ (3 = rẻ nhất);
- **N**: tiềm năng đóng góp hoặc độ mới.

Điểm = E + G + C + N; arm được vào Phase C1 nếu ≥ 8.

| Arm | E | G | C | N | Tổng | Vào C1? |
|---|---|---|---|---|---|---|
| O1 cluster prediction | 3 | 2 | 2 | 3 | 10 | ✔ |
| O2 contrastive | 3 | 3 | 2 | 1 | 9 | ✔ |
| O2h hybrid | 2 | 3 | 3 | 1 | 9 | ✔ (sau O2) |
| O3 SLiM recipe | 2 | 1 | 1 | 2 | 6 | ✘ → chỉ vào nếu O1 hoặc O2 dương, như "họ prototype" |
| O4 teacher RGB | 3 | 3 | 1 | 2 | 9 | ✔ có điều kiện P-T |
| O5 frozen teacher | 2 | 1 | 2 | 2 | 7 | ✘ → Phase E, nếu đường học REF phẳng |
| O6 causal forecasting | 2 | 3 | 3 | 2 | 10 | ✔ |
| O7 multi-level concat target | 3 | 2 | 3 | 2 | 10 | ✔ |
| O8 STARS-style NNCLR tuning | 3 | 3 | 3 | 1 | 10 | ✔ |
| O-ctrl random teacher | — | — | 3 | — | control | ✔ |
| K1 motion-aware | 3 | 3 | 3 | 1 | 10 | ✔ |
| K2 semantic tube | 3 | 1 | 3 | 1 | 8 | ✔ |
| K3 adaptive | 2 | 1 | 2 | 2 | 7 | ✘ → Phase D nếu K1 hoặc K2 dương |
| V1 retarget | 2 | 3 | 3 | 2 | 10 | ✔ |
| V3 GRL signer | 2 | 3 | 3 | 1 | 9 | ✔ (chẩn đoán) |
| V4 strong student views + mask token | 3 | 2 | 3 | 1 | 9 | ✔ |
| I1 per-joint compact | 2 | 1 | 2 | 2 | 7 | ✘ → chỉ nếu Phase B cho thấy channel token mất thông tin |
| I2 multi-stream | 3 | 3 | 3 | 1 | 10 | ✔ |
| I3 shared hand encoder | 2 | 2 | 3 | 1 | 8 | ✔ |
| I4 relation tokens | 2 | 3 | 3 | 1 | 9 | ✔ |

→ **Phase C1 gồm 15 arm ứng viên + 3 control** (REF, R0ctx làm control generic mask, O-ctrl):

- O1, O2, O2h, O4, O6, O7, O8, K1, K2, V1, V3, V4, I2, I3, I4;
- O4 chỉ chạy khi P-T đạt; O2h chạy sau O2;
- O8 là tinh chỉnh từ checkpoint REF, đi kèm control "REF + cùng số bước thêm".

Mỗi arm gắn với giả thuyết ở ma trận §5.9.

---

## 8. Các hướng method tổng thể: những luận điểm sẽ được phễu chọn ra

Phễu ở mục 9 **không định sẵn method cuối**. Nó chọn ra một trong các luận điểm dưới đây, hoặc tổ hợp của chúng, dựa trên bằng chứng. Mỗi luận điểm có "điều kiện cần" đo được: nếu điều kiện không thoả, bỏ luận điểm đó.

### T-A. PhonoJEPA++: dự đoán latent theo kênh âm vị học, có cấu trúc thời gian

- **Giả thuyết nền:** H5, H6, H7 (và H3 nếu O7 được KEEP).
- **Nội dung:** REF + các module K1/I2/O6/O7 sống sót + readout R\* có thứ tự.
- **Đóng góp:** JEPA skeleton cho ký hiệu. Theo tìm kiếm đến 2026-10-14, chưa có sign JEPA nào được công bố (claim có phạm vi, §3.2). Token và mask bám theo tham số âm vị học, kèm phân tích cơ chế theo từng tham số (MPA, group-drop).
- **Điều kiện cần:**
  - VAL-DICT của tổ hợp hơn REF ≥ δ_keep;
  - MPA_movement tăng có ý nghĩa, tức khoảng trống 3 được lấp.
- **Rủi ro:** độ mới ở mức gia tăng (S-JEPA, Path-JEPA, SignSeek đều là họ hàng gần).

### T-B. Đơn vị âm vị học rời rạc tự khám phá (Cluster/HuBERT-skeleton)

- **Giả thuyết nền:** H1, H2.
- **Nội dung:** O1 trên kênh tay và mặt, lặp 2 vòng.
- **Đóng góp:** câu chuyện ngôn ngữ học mạnh:
  - model *tự khám phá* các đơn vị con của ký hiệu từ skeleton không nhãn;
  - các đơn vị này được đánh giá bằng độ tinh khiết theo ASL-LEX (handshape, location) và theo signer;
  - chúng cải thiện retrieval.
- **Điều kiện cần:**
  - P-T: purity theo ASL-LEX của cluster vòng 0 > mức may rủi rõ ràng, và purity theo signer thấp;
  - C1: O1 ≥ REF − 1, tức không thua;
  - C2: O1 hoặc O1+ctx > REF.
- **Rủi ro:** k-means trên keypoint nhiễu có thể ra cluster theo người ký. P-T sẽ báo sớm.

### T-C. Skeleton học từ RGB SSL (cross-modal distillation)

- **Giả thuyết nền:** H9 (và H1: target cross-modal mang thông tin mới).
- **Nội dung:** O4 là thành phần chính, kết hợp REF. I4 (token quan hệ) là đối chứng cạnh tranh.
- **Đóng góp:** model skeleton khoảng 5M tham số đạt phần lớn chất lượng retrieval của SignRep, với chi phí inference nhỏ hơn nhiều; kèm phân tích tham số âm vị học nào truyền được qua cầu RGB→skeleton.
- **Điều kiện cần:**
  - P-T: feature SignRep có VAL-DICT cao và SNR không tệ hơn skeleton;
  - C1: O4 ≫ REF;
  - **control shuffled-teacher** (feature của video khác) cho thấy lợi ích đến từ thông tin, không phải regularization.
- **Rủi ro:** claim phụ thuộc SignRep, nên không thể nói "vượt SignRep". Teacher dùng dữ liệu ngoài và được chọn bằng nhãn WLASL val → phải khai báo.

### T-D. Skeleton SSL bất biến signer, có đo lường

- **Giả thuyết nền:** H8, H8b, H8c.
- **Nội dung:** V1, O8(b), I3, V4 (V3 làm cận trên) + REF.
- **Đóng góp:** phương pháp skeleton SSL label-free đầu tiên nhắm thẳng vào nhiễu signer, kèm bộ metric SNR và cross-signer gap.
- **Điều kiện cần:**
  - V3 (cận trên dùng signer ID) cải thiện VAL-DICT ≥ δ_keep. Nếu loại hẳn signer mà vẫn không giúp retrieval, hướng này không đáng theo;
  - V1 hoặc O8(b) đạt ≥ 50% mức cải thiện của V3 mà không dùng nhãn.
- **Rủi ro:** hiệu ứng nhỏ trên val chỉ có 5 cụm signer hiệu dụng.

### T-E. Objective cấp clip có uniformity (contrastive/hybrid)

- **Giả thuyết nền:** H4.
- **Nội dung:** O8(a), O2, O2h.
- **Đóng góp:** chứng minh rằng với retrieval raw (protocol test), uniformity là thứ JEPA token-level đang thiếu. Kèm phân tích anisotropy gap.
- **Điều kiện cần:**
  - anisotropy gap giảm ≥ 50%;
  - R@1 raw > REF, **và** > control "REF + cùng số bước thêm" **và** > "REF + white-train readout" (nếu whitening cho cùng mức tăng, thành phần global chỉ là whitening học được);
  - SNR không tăng.

> Các luận điểm **có thể cộng dồn**: ví dụ T-A + T-E + T-D. Phase D (factorial) kiểm tra đúng việc cộng dồn đó. Method cuối là tổ hợp nhỏ nhất đạt hiệu năng tốt nhất, tức nguyên tắc parsimony: module nào không qua cổng leave-one-out ở Phase F thì bị bỏ.

---

## 9. Chương trình thực nghiệm chi tiết: phễu 6 phase có cổng

```
Phase A  Đo lường & neo ─► G-A ─► Phase B  Readout & cận trên input ─► G-B
   │                                                              │
   ▼                                                              ▼
Phase C  Khám phá tín hiệu:  C0 pre-screen (0 train) ─► C1 screening (1 seed, budget ngắn)
                             ─► C2 xác nhận (3 seed, budget dài) ─► G-C
   ▼
Phase D  Tổ hợp (fractional factorial) + tinh chỉnh ─► G-D
   ▼
Phase E  Scale dữ liệu / bước / capacity ─► G-E
   ▼
Phase F  Leave-one-out, 3 seed full budget, test một lần ─► báo cáo
```

### 9.1 Quy ước chung cho mọi run

- **Corpus:** từ Phase A trở đi, screening dùng **toàn bộ 40.154 clip train**, phủ 2.731 gloss và 35 signer, thay cho corpus pilot 3k video chỉ có 300 gloss.
  - Lý do: metric chính là full vocabulary.
  - Chi phí mỗi bước giống pilot: khoảng 13,8 bước/giây; 16k bước mất khoảng 20–30 phút; 64k bước khoảng 80–90 phút.
- **Budget:**
  - B_C1 = 16k bước × batch 32, khoảng 12,8 epoch;
  - B_C2 = 64k bước, khoảng 51 epoch;
  - full = 200k bước, khoảng 160 epoch.

  B_C1 có thể điều chỉnh sau A4 và A5.

  Lý do không chọn budget ngắn hơn: literature skeleton thường screening ở khoảng 100 epoch và cho thấy thứ hạng giữa các loại target ổn định từ checkpoint đầu (GFP). Ngược lại, hyperparameter như EMA lại đảo thứ hạng theo độ dài lịch (Skeleton2vec) [V]. Nội bộ: 4k → 16k bước làm R@1 raw tăng 13.7 điểm, nên budget quá ngắn đo sai độ dị hướng.
- **Lịch học:** WSD với cooldown 10%, để các mốc 25/50/100% so sánh được và có thể rẽ nhánh cooldown.
- **Lưu trữ:** checkpoint ở 25/50/100% và **không ghi đè**. Run full160 trước đây mất mốc trung gian vì bị ghi đè.
- **Bằng chứng mỗi run:** manifest (hash code, config, seed), log T0, kết quả T1/T2 dạng JSON, và một "run card" (Phụ lục B).
- **Lập lịch:** một job GPU tại một thời điểm, có runner chờ VRAM trống, không quá 2 tiến trình. Dữ liệu cache sẵn.

### 9.2 Phase A: đo lường và neo (không thử method mới)

| ID | Việc | Đầu ra | Chi phí |
|---|---|---|---|
| A1 | Viết evaluator v2: VAL-DICT (raw, white-train), readout R1–R6, M1–M8. Kiểm tra bằng cách tính lại đúng các số ASLC300/LOSO đã lưu cho 3 checkpoint; unit test trên dữ liệu tổng hợp. **Lưu ý repo:** hàm `effective_rank` trong `signrel_jepa` (`embedding_health`) và trong `pose_jepa` (`representation_diagnostics`) tính entropy trên eigenvalue σ², tức không phải RankMe (RankMe dùng σ). Phải thêm RankMe dựa trên σ, hoặc đổi tên field. Code minimal-pair có sẵn để tái dùng: `signrel_jepa/.../diagnostics.py::minimal_pair_retrieval` và `p4_signssl/p4/phonology.py`. | `evaluate_v2` đã kiểm chứng | CPU + ít GPU |
| A2 | Tải ASL-LEX 2.0 `signdata.csv`, join qua cột `ASL-LEX Code`, dựng danh sách minimal pair cho từng tham số; báo cáo số cặp và số query val phủ được | danh sách cặp + thống kê | CPU |
| A3 | **Chấm lại** mọi checkpoint đang có (25 run PJ, S1, S2, B10–B13) bằng VAL-DICT và M1–M7. Phân loại lại bảng §4.2. | Bảng mới; τ giữa ASLC300 và VAL-DICT; danh sách kết luận cũ bị đảo | ~2–3 GPU-h (chỉ trích feature) |
| A4 | REF (PLctx) trên corpus 40k: 3 seed × B_C1, cộng 1 seed × B_C2 | **σ_s**, MDE, mức tăng từ B_C1 lên B_C2 | ~3 GPU-h |
| A5 | **Độ ổn định thứ hạng theo budget và corpus (H5, H11):** REF, R0ctx và B-MAE trên corpus 40k, mỗi cấu hình một run WSD 64k bước, rẽ nhánh cooldown ở 16k; so với kết quả cũ trên corpus 3k | Thứ hạng ở 16k có giữ ở 64k không? Lợi thế PL so với R0 có còn trên corpus 40k không? | ~5 GPU-h |
| A6 | **Neo:** (a) **cận trên có giám sát**: cùng encoder và input, train bằng CE theo gloss (nhãn *chỉ* dùng cho neo này), B_C2; (b) random-init; (c) descriptor B0; (d) SignRep public trên VAL-DICT. (d) chỉ chạy nếu đã trích feature RGB cho O4. | Vị trí của REF giữa sàn và trần | ~1,5 GPU-h (+ trích feature RGB nếu làm) |

**Cổng G-A:**

- A1 tái tạo đúng các số cũ (sai khác ≤ 0.1 điểm).
- Biết σ_s; chọn B_C1 sao cho MDE của so sánh 1 seed ≤ 2 R@1. Nếu không đạt, C1 dùng 2 seed.
- **Cận trên có giám sát (A6a), quyết định theo H0:**
  - **< 35** R@1 trên VAL-DICT: H0 bị bác bỏ, input hoặc kiến trúc đang giới hạn → chạy B3 trước Phase C, nâng ưu tiên I1/I2/I4;
  - **35–45**: chưa kết luận → chạy B3 song song với C1;
  - **≥ 45**: H0 được ủng hộ → vào thẳng Phase C.

### 9.3 Phase B: readout và cận trên của input

| ID | Việc | Quyết định |
|---|---|---|
| B1 | Ma trận readout R1–R6 × {REF@B_C2, PLctx full160, SignRep public nếu có} | Chốt **R\***, không dùng nhãn. Nếu ordered-bins hơn mean ≥ MDE: thông tin movement có trong token nhưng bị mất khi pool → R\* có thứ tự, và hạ ưu tiên những module chỉ nhắm vào readout. |
| B2 | Group-drop (M3) trên REF | Tay trội, tay không trội và mặt đóng góp bao nhiêu. Nếu bỏ mặt không làm giảm R@1 (khớp với hands-only +1.6), cân nhắc bỏ kênh mặt để đơn giản hoá. |
| B3 | *(chỉ khi G-A báo input giới hạn)* Cận trên có giám sát với: tập khớp (103 so với hands+arms so với thêm torso); xử lý khớp độ tin cậy thấp (mask-token so với zero-fill); cửa sổ (64 frame so với 32 frame ở 25 fps) | Chọn input theo cận trên, đổi chỉ khi tăng ≥ 3 R@1 |

**Cổng G-B:** chốt R\* và pipeline input. Mọi run từ đây dùng chúng.

### 9.4 Phase C: khám phá tín hiệu (trọng tâm)

**C0: pre-screen không cần train** (≤ 2 GPU-h, cộng thêm 6–10 GPU-h trích feature RGB nếu xét O4)

| Arm | Kiểm tra | Đạt khi |
|---|---|---|
| O1 | k-means vòng 0 trên CONF/LOC/face: purity theo handshape/location của ASL-LEX và theo signer; VAL-DICT của histogram cluster | purity ASL-LEX > 2× may rủi và purity signer ≤ của B0 |
| O4 | VAL-DICT, SNR, MPA của feature SignRep pool theo clip | VAL-DICT ≥ 2× REF@B_C1 và SNR ≤ SNR của REF |
| O-ctrl | VAL-DICT của feature từ mạng random | (chỉ để tham chiếu) |
| K1, K2 | P-M: R² dựng lại phần bị che (nội suy, ridge) so với PL và R0 | R² không vượt R² của R0 (không dễ hơn mask ngẫu nhiên) |
| V1, I3 | P-V: tỉ số thay đổi phần signer / phần gloss của descriptor | tỉ số > 1 |

Arm không đạt C0 vẫn có thể vào C1 nếu có lý do cơ chế độc lập, nhưng phải ghi lý do.

**C1: screening** (1 seed, B_C1, T1 ở 50%/100% và T2 ở cuối)

- **Gồm:** 15 arm ứng viên đạt C0 (O1, O2, O2h, O4, O6, O7, O8, K1, K2, V1, V3, V4, I2, I3, I4), 3 control (REF, R0ctx, O-ctrl), và control "REF + bước thêm" cho O8.
- **Chi phí:** khoảng 19 run × khoảng 40 phút ≈ 13 GPU-h, cộng k-means cho O1.
- **Luật thăng hạng lên C2** (tối đa 6 arm):
  1. mọi arm KEEP hoặc KEEP-MECH theo §6.6, với $\sigma_\Delta=\sqrt2\sigma_s$;
  2. **bảo toàn đa dạng:** arm tốt nhất của mỗi họ (O, K, V, I) được thăng hạng nếu Δ ≥ −1 σ_Δ, kể cả khi chưa đạt KEEP. Mục đích là không loại cả một họ tín hiệu chỉ vì nhiễu của 1 seed ở budget ngắn. Đây là successive halving có nhận biết nhiễu: mọi arm nằm trong khoảng MDE của arm dẫn đầu đều được giữ;
  3. arm làm M7 (health) xấu đi hoặc kích hoạt collapse alarm thì bị loại, trừ khi lỗi đã rõ là do implement.
- **Kết quả phụ:**
  - so sánh O-ctrl với REF cho biết ngữ nghĩa teacher đóng góp bao nhiêu;
  - so sánh R0ctx với REF xác nhận lại hiệu ứng của mask PL trên corpus mới.

**C2: xác nhận** (3 seed, B_C2)

- **Gồm:** các arm được thăng hạng và REF, 3 seed (có thể giảm còn 2 seed nếu thiếu GPU; khi đó MDE tăng khoảng 22%).
- **Chi phí:** khoảng 7 arm × 3 seed × khoảng 1,6 giờ (64k bước + T2) ≈ 30–35 GPU-h.
- **Phân loại cuối mỗi module:** KEEP / KEEP-MECH / DROP theo §6.6, Holm trong phase.
- **Proxy validity:** τ(C1, C2). Nếu < 0.6, coi screening là kém tin cậy và tăng B_C1 cho các vòng sau.

**Cổng G-C:** có ít nhất một module KEEP mới ngoài REF. Nếu không có:

1. ghi lại kết quả âm vào ledger. Đây vẫn là tri thức: biết những họ tín hiệu nào không giúp trong domain này;
2. chạy vòng thứ hai của C1 với các arm còn lại (O3, O5, K3, I1) **hoặc** chuyển sang Phase E để kiểm tra xem scale có phải nút thắt.

### 9.5 Phase D: tổ hợp và tinh chỉnh

- **D1: fractional factorial** trên k ≤ 5 module KEEP/KEEP-MECH từ C2:

  | k | Thiết kế | Số run |
  |---|---|---|
  | 4 | $2^4$ full | 16 |
  | 5 | $2^{5-1}$, Resolution V | 16 |
  | 6 | $2^{6-2}$, Resolution IV | 16 |

  1 seed mỗi ô, B_C2, khoảng 26 GPU-h. Kết quả gồm main effect, tương tác 2 chiều, và tổ hợp tốt nhất. Module có main effect ≤ 0 trong bối cảnh tổ hợp thì bị loại, kể cả khi nó KEEP lúc đứng một mình.
- **D2: tinh chỉnh** chỉ cho tổ hợp tốt nhất: tỉ lệ mask (2–3 mức, có thể đặt riêng theo bộ phận: mặt dư thừa nên che được nhiều, ngón tay thì không), **độ dài tube** {4, 8, 16, 32} frame (kiểm chứng H6), momentum EMA (2 mức), độ sâu predictor (2 mức), trọng số các loss nếu có nhiều loss. Thiết kế Latin hypercube khoảng 8–10 run, khoảng 12 GPU-h.
  - **EMA scaling rule:** khi batch đổi từ B sang κB thì $\hat\rho=\rho^\kappa$ [V 2307.13813]. Recipe gốc (S-JEPA batch 256; V-JEPA lớn hơn) chạy ở batch 32 cần momentum gần 1 hơn. Ví dụ 0.996 ở B = 256 tương đương khoảng 0.9995 ở B = 32.
  - BYOL cho thấy vùng tối ưu rộng: 0.99–0.999 [V 2006.07733].
  - Nếu O7 (target đa tầng) được KEEP, độ nhạy EMA được kỳ vọng giảm mạnh [V 2603.15553]. Khi đó bỏ trục EMA khỏi D2.

**Cổng G-D:** chốt **FINAL-S**, và biết rõ các module có cộng dồn hay không.

### 9.6 Phase E: scale

| ID | Việc | Câu hỏi |
|---|---|---|
| E1 | FINAL-S và REF ở 10/30/100% dữ liệu, cùng số epoch | Độ dốc theo dữ liệu. FINAL-S có dốc hơn REF không? Thứ hạng giữa chúng có đảo không? |
| E2 | Một run WSD của FINAL-S đến 200k bước, rẽ nhánh cooldown ở 16k/64k/200k; so với REF full160 | Đường học; lợi thế có giữ được khi train dài không? |
| E3 | FINAL ở model M (khoảng 20M) với full data | Capacity có giới hạn không? |
| E4 | *(có điều kiện: E1 dốc dương rõ, H10)* thêm dữ liệu pose không nhãn từ ngoài. Ứng viên hàng đầu: **SignVerse-2M** (DWPose cho khoảng 2M clip của YT-SL-25, 55+ ngôn ngữ ký hiệu [V 2605.01720]), cho phép so sánh **khớp dữ liệu** với SignRep. Phải loại ID video của WLASL và MS-ASL, và kiểm tra giấy phép. Khác biệt extractor (DWPose so với keypoint của ASL Citizen) phải được kiểm soát bằng một bước chuẩn hoá chung. | Dữ liệu ngoài đóng góp bao nhiêu? Skeleton ở scale SignRep đứng ở đâu? |
| (E5) | *(có điều kiện: đường học của E2 phẳng)* O5 frozen teacher hai giai đoạn | Có phá được trần của EMA không? |

Chi phí khoảng 25–35 GPU-h. **Cổng G-E:** nếu thứ hạng FINAL so với REF đảo ở full scale, quay lại D với budget lớn hơn. Đây là bài học từ việc pilot không dự báo được test.

### 9.7 Phase F: xác nhận và test

1. **F1. Leave-one-out** từ FINAL ở full data, 64k bước, 2 seed. Module nào bỏ đi mà kết quả *tương đương* theo TOST thì bị loại (parsimony).
   - Báo cáo song song **add-one-to-REF** (đã có từ C2) và **LOO-from-FINAL**:
     - LOO đo *tính cần thiết khi đã có các module khác*, nên đánh giá thấp những module dư thừa lẫn nhau;
     - add-one đo *tính đủ khi đứng một mình*, nên đánh giá cao những module trùng vai trò.
   - Khi hai cách này mâu thuẫn, nghĩa là có tương tác. Khi đó dùng attribution kiểu Shapley tính từ bảng factorial của D1.
2. **F2.** FINAL × 3 seed ở full budget, báo cáo VAL-DICT dạng mean ± sd.
3. **F3. Test một lần duy nhất:**
   - ASL Citizen test và WLASL2000 test (cross-dataset, không train trên WLASL);
   - báo cáo cả readout avg và R\*;
   - so sánh với SignRep public (avg và weighted), số MASA đã báo cáo, và REF full160;
   - báo cáo đầy đủ M1–M9;
   - **minimal pair do người Điếc tuyển chọn** của Yin et al. 2026: 259 cặp trên ASL Citizen test, 178 cặp Sem-Lex (ngoài domain), code và dữ liệu công khai [V 2606.28667]. Các cặp này nằm trong test, nên **chỉ dùng ở F3**;
   - tuỳ chọn: zero-shot BSL SignBank, nơi MASA (pose) 6.80 > SignRep 4.30 [V\*]. Đây là phép thử luận điểm "skeleton bổ sung cho RGB khi chuyển domain";
   - công khai lịch sử truy cập test trước đó.
4. **F4** (tuỳ chọn): linear probe và fine-tune ISLR (WLASL2000, MS-ASL1000), phonology probe theo split lexical.

### 9.8 Ngân sách và tiến độ

| Phase | GPU-h, bản đầy đủ | GPU-h, bản tiết kiệm | Ghi chú |
|---|---:|---:|---|
| A | 10–12 | 8 | trích feature + A4/A5 |
| B | 2–5 | 2 | B3 có điều kiện |
| C0 | 2 (+6–10 nếu O4) | 2 | trích feature RGB chỉ khi làm O4 |
| C1 | 13 | 13 | |
| C2 | 30–35 | 22 (2 seed) | |
| D | 35–40 | 18 (8 run, Res IV + 4 run tinh chỉnh) | |
| E | 25–35 | 12 (chỉ E1, E2) | |
| F | 25 | 15 | |
| **Tổng** | **≈ 150–170** | **≈ 90** | GPU dùng chung → khoảng 4–6 tuần lịch |

Có thể chạy song song (trong giới hạn 2 tiến trình): trong khi GPU chạy C1, phần CPU của C0 (k-means, P-M, P-V) và danh sách ASL-LEX (A2) làm cùng lúc.

---

## 10. Ra quyết định, rút kinh nghiệm và vận hành

### 10.1 Cây quyết định sau mỗi phase

```
G-A (H0): cận trên có giám sát trên VAL-DICT?
 ├─ < 35   → B3 (input) trước; nâng ưu tiên I1/I2/I4; xem xét extractor khác (chi phí L)
 ├─ 35–45  → B3 chạy song song với C1
 └─ ≥ 45   → B → C
G-C: có module KEEP mới?
 ├─ có (họ O)  → luận điểm T-B/T-C/T-E dẫn đầu; D tổ hợp với K/V/I đã KEEP
 ├─ có (chỉ K/V/I) → luận điểm T-A/T-D; D tổ hợp
 ├─ chỉ KEEP-MECH → D vẫn chạy (cơ chế có thể cộng dồn); E kiểm tra xem hiệu ứng có hiện ra ở scale
 └─ không có gì → C1 vòng 2 (O3, O5, K3, I1) hoặc E (scale có phải nút thắt?)
G-E: FINAL > REF ở full scale?
 ├─ có  → F
 └─ không (đảo thứ hạng) → D lại ở budget lớn hơn; ghi bài học "tín hiệu chỉ có ở budget ngắn"
```

Sau mỗi cổng, cập nhật **trạng thái của từng giả thuyết** H0–H12 (OPEN / SUPPORTED / REFUTED / INCONCLUSIVE), kèm bằng chứng. Các nhánh trong cây trên tương ứng với trạng thái giả thuyết:

- G-A ↔ H0;
- G-C ↔ H1–H9;
- G-E ↔ H10.

### 10.2 Rút kinh nghiệm sau mỗi đợt: viết gì vào ledger

Cuối mỗi phase, viết một mục `LESSONS` gồm năm phần:

1. **Bảng quyết định:** module → KEEP / DROP / INCONCLUSIVE / KEEP-MECH, kèm Δ, CI, k/5 signer và chữ ký cơ chế (dự đoán so với thực tế).
2. **Bản đồ tín hiệu theo trục:** module nào làm dịch chuyển MPA_handshape, MPA_location, MPA_movement, SNR, anisotropy. Đây là kiến thức tái sử dụng được cho method sau, kể cả khi module không tăng R@1.
3. **Bất ngờ:** những dự đoán chữ ký bị sai. Đây là nơi sinh ra giả thuyết mới.
4. **Proxy validity:** τ giữa các mức budget và giữa các corpus.
5. **Chi phí:** GPU-h thực tế so với kế hoạch.

### 10.3 Quy tắc vận hành

- **Đăng ký trước** (một trang mỗi phase, viết trước khi chạy): arm, control, budget, metric chính, luật quyết định, chữ ký cơ chế dự đoán. Không đổi sau khi đã thấy kết quả. Nếu buộc phải đổi, ghi lại thành *deviation*.
- **Một thay đổi mỗi arm** so với tham chiếu (trừ Phase D). Compute khớp theo số bước × batch và ghi FLOPs mỗi bước, vì các arm hai view (near + far) tốn hơn.
- **Không chọn checkpoint, hyperparameter hay readout bằng test.** Val chỉ dùng cho dev; test chỉ dùng ở F3.
- **Luôn có control cùng budget:** REF, generic mask, random teacher, random-init, B0.
- **Collapse alarm:** dừng sớm khi RankMe giảm > 50% trong 10% số bước, hoặc khi mean cos > 0.95. Log tỉ lệ gradient giữa các loss; cảnh báo nếu một loss chiếm > 100× loss khác. Đây là bài học từ lần tái tạo SignRep.

### 10.4 Anti-pattern cần tránh

1. Kết luận từ ASLC300-white trên corpus 300 gloss rồi tin rằng nó chuyển sang full test (cùng model PLctx full160: 47.7 → 16.7).
2. So sánh hai module với readout khác nhau, hoặc gán công của readout cho module.
3. Diễn giải loss SSL giảm thành representation tốt hơn. Loss JEPA không so sánh được giữa các objective.
4. Bỏ cả một họ tín hiệu chỉ vì một run 1 seed ở budget ngắn (đã có luật bảo toàn đa dạng ở C1).
5. Tối ưu từng module rời rạc rồi ghép lại. Tương tác đã từng rất mạnh: ctx tốt nhất khi đi với PL nhưng tệ nhất khi đi với R4.
6. Dùng effective rank hoặc signer probe để xếp hạng model (τ ≈ 0.14 và τ ≈ 0).
7. Thêm regularizer phân phối (var/cov, SIGReg) mà không theo dõi tỉ lệ gradient.
8. Dùng một số test đã bị truy cập nhiều lần như bằng chứng độc lập.

---

## 11. Tài liệu tham khảo chính

Danh sách rút gọn. Thẻ đầy đủ, số liệu đã xác minh và ghi chú độ tin cậy nằm trong báo cáo L1–L4 (`docs/proposal12/literature/`) và A/B/C (`docs/research/literature/`). Venue có ghi [S] nghĩa là chỉ lấy từ metadata hoặc trang dự án.

**Sign-language representation**

- SignRep: Wong, Camgoz, Bowden. ICCV 2025. arXiv 2503.08529.
- SHuBERT: Gueuwou et al. ACL 2025. arXiv 2411.16765.
- SignBERT: ICCV 2021, arXiv 2110.05382. SignBERT+: TPAMI 2023, arXiv 2305.04868.
- BEST: AAAI 2023, arXiv 2302.05075.
- MASA: TCSVT 2024, arXiv 2405.20666.
- STC: arXiv 2406.10501 (TIP [S]).
- OpenHands: ACL 2022, arXiv 2110.05877.
- SSL-SLR: arXiv 2509.05188 (TMLR [S]).
- SignMAE: ICPR 2026, arXiv 2605.02094.
- SignSeek: arXiv 2609.03695.
- SignDino: arXiv 2609.06296.
- SignMatch: arXiv 2609.01886.
- Uni-Sign: ICLR 2025, arXiv 2501.15187.
- Sigma: arXiv 2509.21223.
- SignCLIP: EMNLP 2024, arXiv 2407.01264.
- LA-Sign: arXiv 2603.29057.
- OneSign: arXiv 2609.33090.
- Geo-Sign: arXiv 2506.00129 (NeurIPS 2025 [S]).
- VQ-ASL (Kezar et al.): arXiv 2509.04745.
- Kezar et al., phonology-aux: EACL 2023, arXiv 2302.05759.
- Phonological Perception of Sign Language Models (Yin, Carter, Lu, Kocab): arXiv 2606.28667. Code/dữ liệu: github.com/kayoyin/sign-phonology.
- Bootstrapping Sign Language Annotations: arXiv 2604.07606 (CVPR 2026 Findings [S]).
- Vandendriessche, De Coster, Dambre: sign-lang@LREC 2026.
- SignVerse-2M: arXiv 2605.01720.
- PoseFormer one-shot ISLR: arXiv 2502.20171.
- NLA-SLR: CVPR 2023, arXiv 2303.12080.
- SL-GCN / SAM-SLR: arXiv 2103.08833 và 2110.06161.
- DSTA-SLR: LREC-COLING 2024, arXiv 2403.12519.
- Pose identifies signers: arXiv 2410.13675.
- Importance of facial features: arXiv 2507.20884.
- SignReGraph: Multimedia Systems 2026, DOI 10.1007/s00530-026-02686-z. Chỉ có metadata.

**Dataset và tài nguyên**

- ASL Citizen (Desai et al., NeurIPS 2023 D&B [K]).
- ASL-LEX 2.0 [K].
- Sem-Lex (ASSETS 2023).
- WLASL (WACV 2020).
- YouTube-SL-25.
- YouTube-ASL.
- COCO-WholeBody / RTMW / DWPose [V-L4].

**Skeleton SSL**

- S-JEPA: Abdelfattah, Alahi. ECCV 2024. DOI 10.1007/978-3-031-73411-3_21.
- Path-JEPA: Singh, Singh. ECCV 2026. DOI 10.1007/978-3-032-37258-1_7.
- SLiM: Do et al. arXiv 2603.10648 (NeurIPS 2026 [S]).
- Skeleton2vec: arXiv 2401.00921.
- MAMP: ICCV 2023.
- MacDiff: ECCV 2024, arXiv 2409.10473.
- IGM: ECCV 2024, arXiv 2410.20349.
- GFP: ICCV 2025.
- AMR: CVPR 2026.
- STARS: arXiv 2407.10935.
- PCM3 / benchmark IJCV 2025: arXiv 2406.02978.
- USDRL: TPAMI, arXiv 2508.12586.
- BYOL-skeleton: CVPR-W 2022, arXiv 2202.02232.
- HA-CM: arXiv 2409.17951.
- PSTL: AAAI 2023, arXiv 2302.09018.
- CrosSCLR: CVPR 2021. AimCLR: AAAI 2022. ActCLR: CVPR 2023. CMD: ECCV 2022. HiCLR: AAAI 2023. UmURL: ACM MM 2023. HYSP: ICLR 2023, arXiv 2303.06242.
- SkeletonMAE: ICCV 2023.
- MotionBERT: ICCV 2023.
- LA2DS: ECCV 2024, arXiv 2305.19480.

**JEPA, SSL nói chung, chống collapse**

- I-JEPA: CVPR 2023, arXiv 2301.08243.
- V-JEPA: TMLR 2024, arXiv 2404.08471.
- V-JEPA 2: arXiv 2506.09985.
- V-JEPA 2.1: arXiv 2603.14482.
- LeJEPA / SIGReg [V-L1].
- SALT [V-L1].
- Bootleg (multi-level targets): arXiv 2603.15553.
- data2vec / data2vec 2.0 [K].
- BYOL: arXiv 2006.07733.
- DINO: arXiv 2104.14294.
- iBOT / DINOv2 [K].
- HuBERT [K].
- EMA scaling rule: arXiv 2307.13813.
- IsoLoss: arXiv 2212.04858.
- Stepwise SSL: arXiv 2303.15438.
- JE vs reconstruction (Van Assel et al.): arXiv 2505.12477.
- Koopman-JEPA: arXiv 2511.09783.
- Dimensional collapse: arXiv 2110.09348.
- Cookbook of SSL: arXiv 2304.12210.

**Masking**

Tổng hợp trong L2:

- MAE, SimMIM, BEiT, VideoMAE, ST-MAE, MGMAE, AdaMAE, SemMAE, EPM, AttMask, ADIOS, HPM, MaskAlign, Point-MAE, Point-JEPA.
- Lý thuyết shared-latent của Kong et al.

**Đánh giá, chẩn đoán, thống kê**

- RankMe: arXiv 2210.02885.
- LiDAR: arXiv 2312.04000.
- α-ReQ: arXiv 2202.05808.
- IdEst: arXiv 2606.03338.
- Label-free metric study: arXiv 2608.23182.
- CLID: arXiv 2206.01251.
- Alignment/uniformity: arXiv 2005.10242.
- Control tasks: arXiv 1909.03368.
- MDL probing: arXiv 2003.12298.
- CKA: arXiv 1905.00414. Similarity testing: arXiv 2108.01661.
- Risk decomposition: arXiv 2302.03068.
- Benchmarking SSL protocols: arXiv 2407.12210.
- Guillotine regularisation: arXiv 2206.13378.
- Layer by Layer: arXiv 2502.02013.
- SUPERB: arXiv 2105.01051.
- Speaker info và ABX: arXiv 2108.00917.
- Adversarial removal: arXiv 1808.06640.
- LEACE: arXiv 2306.03819.
- DataDecide: arXiv 2504.11393.
- Signal and Noise: arXiv 2508.13144.
- DataComp: arXiv 2304.14108.
- Variance in ML benchmarks: arXiv 2103.03098.
- rliable: arXiv 2108.13264.
- Metric learning reality check: arXiv 2003.08505.
- Efficiency misnomer: arXiv 2110.12894.
- JEPA spectral audit: arXiv 2609.09630.
- Velocity blind spot (JEPA world models): arXiv 2610.04585 [S].

**Bằng chứng nội bộ** [I]:

- `docs/proposal12/comparison_plctx_20261013/REPORT_VI.md` và `all_methods_white512.csv`;
- `method/phonojepa/runtime/eval/` (P1–P3, các file models/*.json);
- `method/jepa_baseline/experiments/FULLASLC_AUDIT_20261014/REPORT_VI.md`;
- `docs/proposal4/sign-ssl-research-agent/research/` (STATE, CLAIM_LEDGER, gap_map);
- `method/pose_jepa/docs/COMPLETION_REPORT_VI.md`;
- `method/phonojepa_s1`, `method/phonojepa_s2` (README).

---

## Phụ lục A: công thức

- **RankMe:** $\exp\!\big(-\sum_k p_k\log p_k\big)$ với $p_k=\sigma_k/\sum_j\sigma_j$, trong đó $\sigma_k$ là singular value của ma trận embedding đã center (N ≥ 2.048).
- **Anisotropy gap:** $A=\mathrm{R@1}_{\text{white-train}}-\mathrm{R@1}_{\text{raw}}$.
- **SNR@10:** $\frac{1}{|Q|}\sum_q \frac{|\{n\in \mathrm{NN}_{10}(q): s(n)=s(q),\, g(n)\ne g(q)\}|}{10\cdot \pi_{s(q)}}$, trong đó $\pi_{s}$ là tỉ lệ clip của signer $s$ trong pool.
- **MPA:** $\mathrm{MPA}_p=\frac{1}{|\mathcal{P}_p|}\sum_{(q,g,g')\in\mathcal{P}_p}\mathbb{1}[s(q,g)>s(q,g')]$.
- **TOST:** kết luận tương đương nếu CI 90% của Δ nằm trong $(-\delta_{eq},\delta_{eq})$.
- **Main effect trong factorial ±1:** $\hat\beta_j=\bar y_{x_j=+1}-\bar y_{x_j=-1}$, với $\mathrm{SE}=2\sigma/\sqrt N$.

## Phụ lục B: mẫu run card

```
run_id: C1_O6_s0            phase: C1         date:
hypotheses tested: H7 (primary), H1 (target type)
change_vs_ref: causal latent forecasting (k=4 bins) replaces multi-block ctx target
alternative explanations & controls: extra compute? (FLOPs matched) | readout effect? (R* fixed)
hypothesis / predicted signature: MPA_movement +3; rho_rev down; R@1 >= REF-1
budget: 16k steps (B_C1) x 32, WSD, corpus ASLC40k     seed: 0     FLOPs/step:
T0: loss terms ok? grad ratio max: ; RankMe min: ; collapse alarm: no
T2 (R*): VAL-DICT R@1 raw / white-train:  | Δ vs REF [CI] | k/5 signers
mechanism: MPA hs/loc/mov | SNR@10 | ρ_rev | anisotropy gap | ΔR² mov
decision (§6.6): KEEP / KEEP-MECH / INCONCLUSIVE / DROP   reason:
surprises / new hypotheses:
```

## Phụ lục C: ánh xạ arm cũ sang mới

| Cũ (proposal12/PhonoJEPA) | Vai trò mới |
|---|---|
| PJ_PLctx | REF |
| PJ_PLctx_FULL_160ep | mốc L0 trên test; neo full-scale cho E2 |
| PJ_R0ctx | control generic mask |
| PJ_B1rand, PJ_B0 | sàn |
| PJ_PLboth_shufint | bằng chứng cho luật P-T (target không thông tin thì không giúp) |
| PJ_G1glob, PJ_PLboth_proto, PJ_PLboth_l1sig | bằng chứng âm cho các objective toàn cục kiểu alignment-only, prototype rời rạc và SIGReg-thay-CE; O2/O3 phải khác về cơ chế (negative, recipe đầy đủ) |
| PJ_PLboth_hands, PJ_A8g_randhand, PJ_G3morph | chưa kết luận → B2 và I3 kiểm tra lại |
| S1, S2, BMAMP | bằng chứng âm cho target tường minh (hình học, motion, log-signature) |
