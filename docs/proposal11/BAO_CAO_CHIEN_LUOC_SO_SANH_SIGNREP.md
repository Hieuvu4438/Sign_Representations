# Báo cáo: chiến lược train và cách xác định method của ta có tốt hơn SignRep hay không

**Ngày:** 10/10/2026  
**Phạm vi:** các phương án A, B, C của proposal11 (code ở `method/obs_jepa/`) so với baseline SignRep tái dựng (code ở `method/jepa_baseline/`), train trên dữ liệu đang có trên server.  
**Trạng thái:** code A/B/C đã cài đặt, đã qua 7 bài test và smoke test. **Chưa train nhánh nào của proposal11**, nên chưa có kết luận nào về hiệu quả. Các số liệu trong báo cáo này lấy từ EX16 (SignRep tái dựng so với JEPA, cùng prior), chỉ trên tập validation, 1 seed.

---

## 1. Câu hỏi cần trả lời và vì sao không so với SignRep public

Câu hỏi: **với cùng dữ liệu và cùng ngân sách train, method của ta có học được biểu diễn ký hiệu tốt hơn SignRep không?**

Không so với checkpoint SignRep do tác giả công bố, vì checkpoint đó:
- pretrain trên YouTube-SL-25, một tập lớn hơn dữ liệu của ta hàng trăm lần;
- chọn checkpoint dựa trên val WLASL (paper, Phụ lục A3);
- có danh sách video pretrain không công bố đầy đủ, nên không kiểm tra được việc trùng lặp với dữ liệu đánh giá.

Vì vậy baseline phải là **SignRep tái dựng, train từ đầu** (từ Hiera-B MAE-Kinetics, đúng như paper) trên cùng dữ liệu với method của ta. Bản tái dựng ở `method/jepa_baseline` đã được đối chiếu với paper (mục 3–5, 7.2, A1–A4) và khớp:
- 9 nhóm prior 3D từ teacher pose thật;
- sign decoder lấy đúng code của tác giả;
- trọng số loss theo Tab. 9;
- regularization phương sai/hiệp phương sai, discriminator chống phụ thuộc người ký;
- optimizer, mask 80%.

Checkpoint public chỉ được dùng làm **mốc tham chiếu bên ngoài**, không dùng để kết luận thắng/thua.

---

## 2. Protocol cố định cho mọi method (kể cả baseline)

| Hạng mục | Giá trị | Ghi chú |
| --- | --- | --- |
| Dữ liệu pretrain | 3.000 video train ASL Citizen, không nhãn (corpus ASLC002) | Chính là tập SignRep tái dựng đã dùng; có sẵn pose 3D của teacher SignRep |
| Ngân sách | 4.000 bước × 32 view, tức 128 nghìn view | Bằng SR3D trong EX16 |
| Nhãn gloss | Không dùng trong train | Chỉ dùng khi đánh giá |
| Đánh giá | Chỉ tập **validation** | Không đụng test trong suốt quá trình phát triển |
| Code đánh giá | `method/obs_jepa/oj/evaluate.py`, giống hệt cho mọi model | Mỗi model dùng đúng tiền xử lý lúc train của nó |
| Seed | 1 seed để sàng lọc; ứng viên cuối × 3 seed | |

Lý do chọn tập 3.000 video: pose 3D (target prior của SignRep) cần khoảng 7,6 giây/video với 4 worker. Toàn bộ 40.154 video sẽ mất khoảng 85–170 giờ CPU, hiện chưa khả thi.

---

## 3. Thước đo: chỉ số chính chốt trước khi chạy

### 3.1. Chỉ số chính

1. **Linear probe macro Top-1 trên ASLC300** (chỉ số chính theo proposal11 §12.2):
   - Gallery là 3.000 video train (300 gloss, mỗi gloss 10 video); query là 1.500 video val của 5 người ký không có trong train.
   - Encoder đóng băng → lấy trung bình feature các window (pooling cố định) → logistic regression.
   - Hệ số L2 chọn trên một phần gallery tách theo người ký, không dùng query.
   - Kết quả là độ chính xác trung bình theo lớp.
2. **R@1 LOSO-val** (đồng chính, giữ liên tục với EX15/EX16):
   - Toàn bộ 10.304 clip val. Lần lượt mỗi người ký làm query, gallery là clip của những người ký còn lại.
   - Embedding = trung bình window sau whitening PCA, không dùng nhãn.

### 3.2. Chỉ số phụ, để diễn giải chứ không dùng để chọn

- Retrieval R@1 trên ASLC300.
- **Probe theo bộ phận cơ thể** (R²): hình dạng tay, thân, vị trí và quan hệ tay–mặt/thân, non-manual, chuyển động. Cho biết model nắm được thông tin gì.
- **Probe người ký**: mức lộ danh tính người ký.
- **Kiểm tra sụp biểu diễn**: cosine trung bình giữa các clip, effective rank.
- Đề xuất thêm: val WLASL để kiểm tra khả năng tổng quát sang dataset khác.

### 3.3. Thống kê

- **Bootstrap ghép cặp theo cụm người ký** (10.000 lần lặp) cho mức chênh giữa hai model trên cùng query.
- Val chỉ có 5–6 người ký, nên luôn báo kết quả **theo từng người ký**, không chỉ con số trung bình.
- Độ dao động giữa các seed được báo riêng với độ bất định của mẫu đánh giá.

---

## 4. Thang so sánh: làm sao biết method của ta hơn SignRep

Đây là phần quan trọng nhất. Một phép so sánh đơn lẻ không đủ, vì method của ta và SignRep khác nhau **nhiều điểm cùng lúc**: cách train (objective), backbone (ViT so với Hiera) và khởi tạo (V-JEPA 2.1 so với MAE-Kinetics).

### 4.1. Bài học từ EX16 (đã chạy, val, 1 seed)

| Model | Khởi tạo | Cách train | R@1 LOSO-val (whitening) |
| --- | --- | --- | --- |
| SR3D: SignRep tái dựng | Hiera-B MAE-Kinetics | Prior + phương sai/hiệp phương sai + đối kháng | **5,1%** |
| ASLC002: bản tái dựng của Codex | Hiera-B MAE-Kinetics | như trên (batch 4) | 5,0% |
| JV3D | **V-JEPA 2.1 ViT-B** | JEPA + cùng prior SignRep | **28,0%** |
| SignRep public (tham chiếu ngoài) | — | pretrain YouTube-SL-25 | 37,5% |

JV3D hơn SR3D **+22,9 R@1, CI [18,4; 25,1]**, thắng ở 5/6 người ký. Nhưng JV3D khác SR3D cả về khởi tạo lẫn cách train. Khởi tạo V-JEPA 2.1 đã được Meta pretrain trên lượng video rất lớn (chưng cất từ ViT-G). Vì vậy **không thể quy mức chênh này cho JEPA**. Phần lớn có thể đến từ khởi tạo.

Hệ quả cho proposal11: A và C cũng khởi tạo từ V-JEPA 2.1, nên **gần như chắc chắn sẽ thắng SR3D**. Thắng SR3D vì vậy chưa nói lên điều gì về method.

### 4.2. Các bậc so sánh cần có

| Bậc | So sánh | Câu hỏi được trả lời | Kết luận được phép nếu method thắng |
| --- | --- | --- | --- |
| 1 | Method vs **SR3D** | Cùng dữ liệu và ngân sách, pipeline nào tốt hơn? | "Pipeline của ta tốt hơn pipeline SignRep ở điều kiện này", **kể cả lợi thế khởi tạo** |
| 2 | Method vs **SR-VJ** (công thức SignRep đặt lên V-JEPA 2.1; *cần thêm*) | Cùng backbone, cùng khởi tạo, cùng dữ liệu: objective nào tốt hơn? | "**Objective/cách train của ta tốt hơn SignRep**" (không còn do khởi tạo) |
| 3 | **JH3D vs SR3D** (JEPA trên chính Hiera-B của SignRep; đang chạy) | Riêng paradigm JEPA có hơn công thức SignRep khi giữ nguyên backbone không? | Đóng góp riêng của paradigm JEPA |
| 4 | Ablation trong method | Thành phần nào tạo ra cải thiện? | Đóng góp cụ thể của từng thành phần |
| 5 | Kiểm tra tổng quát: val WLASL, tăng dữ liệu hoặc ngân sách | Ưu thế có giữ khi đổi dataset hoặc quy mô? | Kết luận không chỉ đúng ở một cấu hình nhỏ |

Bậc 4 cụ thể:
- **A:** A10 → A13 và đối chứng độ phủ A10c (xem mục 5).
- **B:** B12 latent so với B13 thêm anchor.
- **C:** C so với A13 dùng cùng mask C-reserved.

**SR-VJ là gì:** giữ nguyên 100% công thức SignRep (sign decoder trên trung bình token, 9 prior 3D, regularization phương sai/hiệp phương sai, discriminator chống phụ thuộc người ký, cùng optimizer và mask), chỉ thay backbone Hiera-B bằng V-JEPA 2.1 ViT-B. Đây là đối chứng **"SignRep với cùng lợi thế khởi tạo"**, và là phép so sánh công bằng nhất cho A và C. Chi phí khoảng 1,5 giờ GPU. Code tái sử dụng được từ `jepa_targets/jt/ex16.py` và `jepa_baseline`.

### 4.3. Luật kết luận (chốt trước khi chạy)

Chỉ tuyên bố **"method của ta tốt hơn SignRep"** khi **đồng thời** thỏa:

1. Hơn **SR3D và SR-VJ** ít nhất **+1,0 điểm** ở **cả hai** chỉ số chính.
2. CI 95% của mức chênh (bootstrap ghép cặp theo người ký) **> 0** ở cả hai chỉ số.
3. Cùng chiều ở **cả 3 seed**.
4. **Không** thua trên val WLASL (chỉ số phụ), để tránh kết luận chỉ đúng trên ASL Citizen.
5. Ablation bậc 4 chỉ ra được thành phần tạo ra cải thiện. Nếu không, chỉ ghi "pipeline tốt hơn", không ghi "cơ chế X tốt hơn".

Nếu chỉ thỏa một phần thì kết luận bị thu hẹp tương ứng:

| Tình huống | Kết luận đúng |
| --- | --- |
| Thắng SR3D nhưng **hòa hoặc thua SR-VJ** | Khởi tạo V-JEPA là yếu tố chính; method chưa hơn objective của SignRep |
| Thắng SR-VJ ở 1/2 chỉ số chính | Có tín hiệu nhưng chưa đủ; cần thêm seed hoặc dữ liệu |
| Thắng nhưng CI chứa 0 hoặc chỉ do 1–2 người ký | Chưa kết luận; mức chênh có thể là nhiễu |
| Thắng ở 1 seed, không lặp lại ở 2 seed còn lại | Không kết luận |
| Thắng trên ASL Citizen nhưng thua trên WLASL | Ưu thế có thể do khớp đặc điểm của ASL Citizen (cùng nguồn webcam), chưa tổng quát |
| JH3D ≈ SR3D nhưng JV3D ≫ SR3D | Paradigm JEPA không phải nguồn cải thiện; khởi tạo mới là nguồn |
| Probe người ký tăng mạnh còn retrieval khác người ký giảm | Model đang học đặc trưng người ký, không phải ký hiệu |

### 4.4. Riêng phương án B

B dùng **landmark 2D** làm input và train từ đầu, nên không so trực tiếp với SignRep (input RGB) như cùng loại. B được so với:
- các baseline cùng loại input: B10 (tái tạo tọa độ raw), B11 (S-JEPA), feature hình học dùng trực tiếp (B00, sẽ thêm);
- và đóng vai trò **teacher cho C**. Nếu teacher B không giữ thông tin tốt hơn feature hình học trực tiếp, C không có cơ sở để chạy (proposal11 §9.1).

---

## 5. Kế hoạch chạy theo giai đoạn

| Giai đoạn | Nội dung | Câu hỏi | Điều kiện để đi tiếp |
| --- | --- | --- | --- |
| **P0** | Đánh giá lại baseline trong protocol mới: SR3D, ASLC002, JV3D, V-JEPA gốc (A00); **train thêm SR-VJ**; JH3D (đang chạy) | Mốc so sánh trong cùng protocol | Pipeline đánh giá tái lập được |
| **P1a** | A-R generic so với A-D generic | Recipe nền cho A | Chọn theo chỉ số chính trên val, rồi khóa |
| **P1b** | Factorial A (bảng dưới) | Mask theo vùng, cân bằng loss, hay chỉ do để lại nhiều tay hơn? | Xem cách đọc ở bảng dưới |
| **P2** | B12 latent, B13 latent + anchor, B11 centered-CE, B10 raw | Học hình học tự giám sát có hơn tái tạo tọa độ raw không? | Teacher B13 phải tốt hơn B12 và feature trực tiếp ở probe hình dạng/vị trí/chuyển động |
| **P3** | C1 = A13 + chưng cất từ B (mask C-reserved); đối chứng A13 với cùng mask C-reserved | Chưng cất hình học có giúp RGB ngoài tác dụng của mask không? | Chỉ chạy nếu qua cổng P2 |
| **P4** | Ứng viên tốt nhất × 3 seed; A ở 384 px; đối chứng A14 (che trọn bộ phận) và A15 (vùng dịch ngẫu nhiên); val WLASL | Ổn định, độ phân giải, tác dụng có thật sự đến từ thông tin vùng không | Luật kết luận ở mục 4.3 |

**Factorial A (P1b):**

| Mã | Chính sách mask | Trọng số loss | Mục đích |
| --- | --- | --- | --- |
| A10 | Generic | Đồng đều | Domain adaptation đơn thuần |
| A11 | Generic | Cân bằng theo bộ phận | Chỉ tác dụng của cân bằng loss |
| A12 | Partial-region | Đồng đều | Chỉ tác dụng của mask theo vùng |
| A13 | Partial-region | Cân bằng | Giả thuyết đầy đủ |
| A10c | Generic, giữ ≥ 25% tay như A12/A13 | Đồng đều | Tách tác dụng "để lại nhiều tay hơn" |

Cách đọc:
- Nếu A12/A13 thắng A10 nhưng **không** thắng A10c, cải thiện đến từ việc để lại nhiều tay nhìn thấy, không phải cách sắp xếp mask.
- Nếu A10 ≈ A13 và cả hai > A00, chỉ domain adaptation là yếu tố chính; claim về mask bị bác.

---

## 6. Chi phí ước tính

Đo trên GPU dùng chung, khi có job khác chạy song song:

| Hạng mục | Ước tính |
| --- | --- |
| Một lượt train A (224 px, 4.000 bước, batch 32) | khoảng 55 phút |
| Một lượt train B (4.000 bước) | khoảng 75 phút |
| SR-VJ | khoảng 1,5 giờ |
| A ở 384 px | khoảng 3–4 lần A ở 224 px |
| Trích feature, mỗi model | khoảng 45 phút (val) + 20 phút (ASLC300); có thể nhanh hơn 3–4 lần sau khi đọc video song song |
| P0–P4 | khoảng 25–30 giờ GPU, chạy tuần tự, một job GPU mỗi lúc |

Tất cả chạy qua runner tự chờ GPU trống, tự thử lại khi OOM, và resume được từ checkpoint nếu server sập.

---

## 7. Rủi ro và giới hạn đã biết

1. **Ở 224 px tay rất nhỏ:**
   - Mask partial chỉ tạo khoảng 19 target có cấu trúc mỗi window và 13% số mẫu phải fallback; ở 384 px là khoảng 69 target và 3%.
   - Tác dụng của mask theo vùng có thể chỉ thấy rõ ở 384 px (P4), nên ở P1 một kết quả âm tính chưa đủ để bác bỏ hoàn toàn.
2. **3.000 video là ít cho ViT-B 86 triệu tham số.** Model có thể overfit nhanh; cần theo dõi loss trên holdout và chỉ số sụp biểu diễn.
3. **Gallery ASLC300 chính là video pretrain.** Mọi method đều được hưởng như nhau, nhưng phải ghi rõ là đánh giá transductive.
4. **Landmark COCO-WholeBody có score chưa hiệu chỉnh.** Prior 3D của SignRep cũng thiếu ở một phần frame (tay chỉ hợp lệ khoảng 35–45%); activity prior gần như không có nhãn.
5. **Val chỉ có 5–6 người ký**, nên độ bất định thống kê lớn; mọi kết quả phải kèm số liệu theo từng người ký.
6. **Thắng ở đây chỉ là thắng SignRep tái dựng ở quy mô nhỏ**, chưa phải thắng SignRep trong paper (pretrain trên YouTube-SL-25). Muốn kết luận ở quy mô paper cần:
   - mở rộng dữ liệu pretrain (ví dụ How2Sign, rồi đa ngôn ngữ);
   - tính pose 3D cho tập lớn hơn;
   - cuối cùng mới chạy một lần trên test, với protocol khóa trước.

---

## 8. Việc cần bạn quyết định

1. Có thêm **SR-VJ** (SignRep đặt trên V-JEPA 2.1) vào P0 không? Mình khuyên **có**, vì thiếu nó thì không thể kết luận method tốt hơn SignRep.
2. Có thêm **val WLASL** làm chỉ số phụ không? Mình khuyên **có**.
3. Có dùng luật kết luận ở mục 4.3 như trên không, hay bạn muốn ngưỡng khác (ví dụ +2 điểm)? Ngưỡng phải chốt **trước** khi chạy P1.

Sau khi bạn đồng ý, thứ tự chạy sẽ là: hoàn tất JH3D (đang chạy) → P0 (gồm SR-VJ) → P1 → P2 → P3 → P4.

---

## 9. Lịch chạy đã cài đặt (1 seed, theo yêu cầu ngày 10/10/2026)

- **Đã thêm SR-VJ.** Nhánh này giữ nguyên công thức SignRep (dùng `Pretrainer` của bản tái dựng), backbone V-JEPA 2.1 ViT-B, và giữ đúng hình học mask của SignRep: 80% của 392 mask unit, mỗi unit 32×32 pixel × 2 frame.
- **Bộ lập lịch:** `method/obs_jepa/oj/schedule.py`. Log ở `method/obs_jepa/runtime/logs/schedule.log`, trạng thái ở `runtime/schedule_state.json`.
  - Mỗi lúc chỉ chạy một job GPU.
  - Trước mỗi job, chờ cho đến khi job trước đó (kể cả job còn sót lại của runner cũ) kết thúc và GPU còn đủ bộ nhớ trống. Ngưỡng: train 224 px 20 GB, train 384 px 32 GB, B 12 GB, trích feature 8 GB.
  - Khi OOM hoặc lỗi: chờ 5 phút rồi thử lại, tối đa 6 lần. Job train thì resume từ checkpoint.
  - Nếu server sập, chạy lại scheduler: nó bỏ qua các job đã xong.
- **Thứ tự:**
  1. Hoàn tất EX16 (feature và đánh giá JH3D).
  2. P0: train SR-VJ; trích feature và đánh giá SR3D, SR-VJ, ASLC002, JV3D, JH3D, A00.
  3. P1a: A10_AR, A10_AD. **Tự chọn recipe:** chọn bên có linear probe macro cao hơn; nếu chênh không quá 0,5 điểm thì chọn bên có LOSO R@1 cao hơn.
  4. P1b: A11, A12, A13, A10c.
  5. P2: B12, B13, B11, B10.
  6. **Cổng B:** chỉ chạy P3 khi B13 có R² probe hình dạng tay, vị trí/quan hệ và chuyển động đều ≥ B12.
  7. P3: A13r (mask C-reserved, không teacher) và C1.
  8. P4: cấu hình A tốt nhất (tự chọn theo linear probe macro) ở 384 px, cùng A14 (che trọn bộ phận) và A15 (vùng dịch ngẫu nhiên).
  9. FINAL: đánh giá tất cả, so ghép cặp với SR3D và SR-VJ.
  10. Sinh `runtime/REPORT_1seed.md`.
- **1 seed nên chỉ là sàng lọc.** Model "qua sàng lọc" (≥ +1 điểm ở cả hai chỉ số, CI > 0, so với cả SR3D và SR-VJ) mới đủ điều kiện chạy thêm seed và kiểm tra trên WLASL.
