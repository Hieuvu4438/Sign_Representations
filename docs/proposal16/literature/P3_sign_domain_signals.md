# P3 — Tín hiệu đặc thù ngôn ngữ ký hiệu: từ âm vị học đến skeleton và path signature (cho proposal16, Path-JEPA trên skeleton)

Ngày: **2026-10-16**. Chỉ đọc tài liệu và chạy phân tích CPU, không dùng GPU.

**Phạm vi (delta).** Phần sign SSL, SignRep, SignSeek, MASA, SHuBERT, Yin et al. (minimal pair), Kezar VQ-ASL, các bảng frozen retrieval và các con số "hands vs body" của NLA-SLR, Uni-Sign, SL-GCN **đã có** trong:
- `docs/research/literature/A_signrep_and_sign_ssl.md` (gọi tắt **A**)
- `docs/proposal12/literature/L4_sign_ssl.md` (gọi tắt **L4**)

Số liệu local (fps, độ phân giải, số clip, định dạng keypoint) có trong `docs/proposal16/literature/D1_server_dataset_audit.md` (gọi tắt **D1**). Ghi chú đọc kỹ Path-JEPA nằm ở `P0_path_jepa_close_reading.md` (**P0**).

File này chỉ bổ sung bốn phần:
1. Ánh xạ âm vị học → tín hiệu skeleton → mức signature, kèm phân bố thật của ASL-LEX 2.0, đếm trong phiên này.
2. Bằng chứng mới về kênh tay, mặt và miệng, so sánh ISLR với CSLR.
3. Thẻ dataset ở mức paper.
4. Xếp hạng tín hiệu nên ép mô hình học và tín hiệu nhiễu nên bất biến.

**Nhãn.**
- **[V]**: grep trong phiên này từ HTML hoặc abs của arXiv, hoặc từ Crossref/OSF API. Text thô nằm ở `/tmp/claude-1007/-home-haipd-Sign-Representations/32c5829a-820b-4c3b-9d20-e66b08713d70/scratchpad/raw/` (thư mục tạm của phiên).
- **[V-A] / [V-L4]**: đã xác minh trong A hoặc L4, chỉ chép lại.
- **[I]**: đo trên máy này trong phiên (ASL-LEX CSV, split ASL Citizen, ffprobe).
- **[K]**: kiến thức nền, chưa xác minh trong phiên.
- **[chưa xác minh]**: con số cụ thể chưa kiểm được.

---

## 1. Âm vị học ASL → tín hiệu động học

### 1.1 Các khung lý thuyết (tóm tắt có chọn lọc)

| Khung | Ý chính dùng được cho thiết kế | Nguồn |
|---|---|---|
| **Stokoe (1960)** | Ký hiệu được phân tích thành các tham số đồng thời: *tab* (location), *dez* (handshape), *sig* (movement). Orientation và non-manual được bổ sung về sau (Battison 1978 và các tác giả sau). | *Sign Language Structure*, Studies in Linguistics Occ. Papers 8 [K] |
| **Battison (1978): Symmetry Condition** | Nếu cả hai tay cùng chuyển động thì chúng có cùng handshape, location đối xứng, và chuyển động **đồng thời (in-phase)** hoặc **luân phiên (alternating)**. | *Lingering Loan Signs*, Linstok [K] |
| **Battison (1978): Dominance Condition** | Nếu hai tay khác handshape thì tay không trội **đứng yên** và lấy handshape từ một tập nhỏ không đánh dấu (B, A, S, C, O, 1, 5). | [K]. Được ASL-LEX xác nhận ở §1.2: 96,6 % [I] |
| **Liddell & Johnson (1989): Movement–Hold** | Ký hiệu là **chuỗi** segment Hold (H) và Movement (M), ví dụ H-M-H. Mỗi segment mang một bó đặc trưng: handshape, location, orientation, non-manual. Đây là mô hình tuần tự, đối lập với Stokoe. | *Sign Language Studies* 64:195–277, DOI 10.1353/sls.1989.0027 [V Crossref] |
| **Sandler: Hand-Tier** | Khuôn L–M–L (Location–Movement–Location). Handshape là một tier tự trị trải trên cả âm tiết. | Sandler & Lillo-Martin 2006, CUP, DOI 10.1017/CBO9781139163910 [V Crossref; nội dung K] |
| **Brentari (1998): Prosodic Model** | Đặc trưng chia làm hai nhóm. **Inherent features (IF)**: articulator/handshape và place of articulation, **tĩnh** trong suốt từ. **Prosodic features (PF)**: mọi thứ **động**. Movement là **nhân âm tiết** (syllable nucleus). Có thang "sonority thị giác" theo khớp: path (vai/khuỷu) > orientation change (cổ tay) > aperture/handshape change (đốt ngón) > wiggle (ngón, thứ cấp). Hầu hết ký hiệu đơn âm tiết. | *A Prosodic Model of Sign Language Phonology*, MIT Press, DOI 10.7551/mitpress/5644.001.0001 [V Crossref; nội dung K] |

Phân loại movement theo Brentari [K]:
- **Path movement**: tay di chuyển trong không gian. Hình dạng có thể là thẳng (straight), cung (arc), tròn (circle), zigzag/Z, hoặc tới-lui (back-and-forth). Có hướng (lên/xuống, xa/gần thân, ipsi/contra). Có thể có setting change, tức location đầu khác location cuối.
- **Local/internal movement**, không di chuyển tay trong không gian:
  - đổi handshape (mở/đóng aperture);
  - đổi orientation (xoay cẳng tay hay *wrist twist*, gập cổ tay);
  - **wiggle** ngón (dao động lệch pha giữa các ngón).
- **Repetition**: đơn và lặp có thể là đối lập hình thái. Ví dụ các cặp danh–động từ kiểu CHAIR/SIT (Supalla & Newport 1978) [K]. Riêng **số lần lặp vượt quá 2** thường là biến thể tự do [K].

**Non-manual trong ký hiệu rời.** Có mouth gesture/mouth morpheme từ vựng hoá (ví dụ NOT-YET có lưỡi) và mouthing tiếng Anh, vốn rất dao động theo người ký. Lông mày và đầu chủ yếu mang ngữ pháp, tức cấp câu. Với ký hiệu rời, vai trò từ vựng của NMM **hạn chế hơn nhiều** so với câu [K]. ASL-LEX 2.0 **không mã hoá NMM**, cũng **không mã hoá orientation** (§1.2).

### 1.2 ASL-LEX 2.0: tệp, cột, phân bố, nối với ASL Citizen

**Nguồn [V].**
- Sehyr, Caselli, Cohen-Goldberg, Emmorey. *The ASL-LEX 2.0 Project: A Database of Lexical and Phonological Properties for 2,723 Signs in American Sign Language*. **JDSDE 26(2):263–277, 2021**, DOI **10.1093/deafed/enaa038** (Crossref).
- OSF project **osf.io/zpha4**. Thư mục `Data Files/`:
  - `signdata.csv`: 2.010.308 B, tải tại `https://osf.io/download/9nygd/`.
  - `signdataKEY.csv`: 45.508 B, `https://osf.io/download/ygq4v/`. Đây là từ điển biến.
  - `Phonology/NeigborPairs.csv`: **2,29 GB, không tải**.
  - `Documentation/Supplementary_Materials_Phonological_Coding.docx`: hướng dẫn mã hoá.

**Đã tải [I]** về `/tmp/claude-1007/-home-haipd-Sign-Representations/32c5829a-820b-4c3b-9d20-e66b08713d70/scratchpad/asllex/`:
- `signdata.csv`: sha256 `080ecc3d…40fbd3`, 2.723 dòng, mã hoá latin-1.
- `signdataKEY.csv`: sha256 `b7ff6dc0…b83a6`.

Đây là thư mục tạm của phiên. Nếu cần giữ lâu dài thì copy vào `data/` hoặc tải lại từ URL trên. Script thống kê nằm ở `…/scratchpad/scripts/asllex_stats.py` và `asllex_mp.py`.

**Cột âm vị học (tên chính xác).** Mọi cột mô tả **morpheme đầu tiên**. Các bộ `…M2.2.0` đến `…M6.2.0` lặp lại cho morpheme 2–6 (259 mục có M2).

| Cột CSV | Tên trên web | Định nghĩa (signdataKEY) |
|---|---|---|
| `Handshape.2.0` | Handshape | Tổ hợp selected fingers, flexion, spread, thumb position, thumb contact. **58 giá trị.** |
| `SelectedFingers.2.0` | Selected Fingers | Ưu tiên ngón chuyển động, rồi ngón không gập hẳn hay duỗi hẳn, rồi ngón duỗi. |
| `Flexion.2.0`, `FlexionChange.2.0` | Flexion, Flexion Change | Độ mở (aperture) của ngón được chọn lúc bắt đầu, và có đổi hay không. |
| `Spread.2.0`, `SpreadChange.2.0` | Spread, Spread Change | |
| `ThumbPosition.2.0`, `ThumbContact.2.0` | | |
| `SignType.2.0` | Sign Type | "Symmetry of the hands according to **Battison's sign types**" |
| `Movement.2.0` | **Path Movement** | Path movement của morpheme đầu. |
| `RepeatedMovement.2.0` | Repeated Movement | Movement có lặp không (path, wrist twist hoặc handshape change). |
| `MajorLocation.2.0`, `MinorLocation.2.0`, `SecondMinorLocation.2.0` | | Location của **tay trội lúc bắt đầu**. Second minor location chỉ có khi có setting change. |
| `Contact.2.0` | Contact | Tay trội có chạm major location hay không (với ký hiệu đối xứng thì là chạm tay kia). |
| `NonDominantHandshape.2.0` | Nondominant Handshape | |
| `UlnarRotation.2.0` | **Wrist Twist** | |

Các cột khác có ích:
- `Code`: khoá nối với ASL Citizen, ví dụ `A_03_054`.
- `SignDuration(ms)`.
- `Neighborhood Density 2.0`, `Parameter.Neighborhood.Density.2.0`.
- `Phonological Complexity`, theo Morgan et al. 2019.
- `Compound.2.0`, `Initialized.2.0`, `FingerspelledLoanSign.2.0`, `LexicalClass`, `Iconicity(M)`, `SignFrequency(M)`.
- 15 cột `*.2.0Frequency`: tần suất dưới-từ-vựng.

**Phân bố trên 2.723 mục [I]** (đếm trực tiếp từ `signdata.csv`):

| Đặc trưng | Phân bố |
|---|---|
| **Sign type** | OneHanded **39,2 %**. SymmetricalOrAlternating **34,5 %**. AsymmetricalDifferentHandshape 17,5 %. AsymmetricalSameHandshape 5,9 %. DominanceViolation 2,3 %. SymmetryViolation 0,6 %. → **60,8 % hai tay.** Điều kiện Battison bị vi phạm ở 2,9 %. |
| **Path movement** (`Movement.2.0`) | Straight 45,7 %. Curved (arc) 23,2 %. **None 15,3 %**. Circular 7,6 %. Z-shaped 3,4 %. BackAndForth 2,6 %. Other 1,7 %. X-shaped 0,4 %. → **84,7 % có path movement.** |
| **Path và local** | Chỉ path: 58,2 %. Path + local: 26,5 %. Chỉ local: 9,6 %. Không cả hai: 5,7 % (155 mục). Local = FlexionChange, SpreadChange hoặc wrist twist. |
| **Repeated movement** | **44,6 %**. Thời lượng trung vị 901 ms nếu lặp, 801 ms nếu không lặp. |
| **Local movement** | FlexionChange 19,5 %. SpreadChange 7,1 %. **Wrist twist 16,5 %**. |
| **Major location** | Neutral 36,3 %. **Head 26,7 %**. **Hand 25,0 %** (tay kia làm nền). Body 9,4 %. Arm 2,6 %. |
| **Minor location** | 37 giá trị. Trong Head: HeadAway 179, Mouth 125, Forehead 108, Chin 101, CheekNose 96, Eye 43. Trong Hand: Palm 209, HandAway 96, FingerRadial 94, PalmBack 66, FingerTip 64. |
| **Second minor location** (setting change) | NA 29,5 %. Neutral 30,0 %. HeadAway 12,6 %. HandAway 10,9 %. |
| **Contact** | **62,2 %**. Theo vùng: Head 512/726 có chạm. Hand 611/682. Body 209/255. Neutral 304/988. |
| **Selected fingers** | imrp 44,8 %. i 26,1 %. im 14,0 %. t 5,7 %. m 4,5 %. p 3,3 %. |
| **Flexion** | FullyOpen 38,6 %. Curved 21,3 %. Flat 15,2 %. Bent 9,3 %. FullyClosed 6,4 %. Stacked 2,4 %. Crossed 1,2 %. |
| **Handshape** (58 giá trị) | 1: 10,1 %. open_b: 8,6 %. s: 6,0 %. a: 6,0 %. curved_5: 5,3 %. 5: 5,1 %. baby_o: 4,2 %. Phân bố đuôi dài. |
| **Handshape tay không trội** (với 476 mục asym-different) | B 57 %. 1 15,8 %. C 11,8 %. S 5,5 %. 5 5,3 %. O 1,1 %. A 0,4 %. → **96,6 % thuộc tập không đánh dấu của Battison.** |
| **Morpheme** | 1 morpheme: 90,4 %. Compound: 8,3 %. Initialized: 13,3 %. Fingerspelled loan: 1,4 %. |
| **SignDuration** (trên video kích thích của ASL-LEX) | Trung vị **834 ms**, p10 467 ms, p90 1201 ms. Ở 30 fps là khoảng **25 frame**, p90 khoảng 36 frame. |

**Nối với ASL Citizen [I].**
- Split `/home/dongvk/datasets/ASL_Citizen/splits/{train,val,test}.csv` có các cột `Participant ID, Video file, Gloss, ASL-LEX Code`.
- 2.731 gloss tương ứng 2.723 chuỗi mã. 2.722 mã khớp `Code` trong `signdata.csv`. Chuỗi `NA` gán cho 6 gloss: ABOVE, INSTINCT, LETTER2, MINIMUM1, SENTENCE1, WELCOME1.
- Số dòng khớp: train 40.066/40.154, val 10.278/10.304, test 32.872/32.941, tức **99,8 %**.
- Mỗi gloss chỉ có 1 mã. Có 2 mã dùng chung cho nhiều gloss (biến thể): `C_02_054` cho WHATFOR1/2/3, `B_03_084` cho RESEARCH1/2.
- Như vậy **mọi probe âm vị học trên ASL Citizen đều có nhãn sẵn**, không cần tải thêm gì khác.

**Minimal pair suy ra từ mã ASL-LEX [I].**

Cách đếm: so 15 đặc trưng của morpheme đầu. NA được coi là một giá trị. Các nhóm:
- handshape = SelectedFingers, Flexion, Spread, ThumbPosition, ThumbContact;
- location = Major, Minor;
- movement = Movement, Repeated, FlexionChange, SpreadChange, UlnarRotation;
- sign type = SignType, NonDominantHandshape;
- contact.

Đếm các cặp khác nhau ở **đúng một nhóm**:

| Nhóm khác biệt duy nhất | Số cặp | Số ký hiệu tham gia |
|---|---|---|
| movement | **7.510** | 1.880 |
| handshape | 3.048 | 982 |
| location | 1.886 | 1.043 |
| sign type / tay không trội | 468 | 479 |
| contact | 462 | 353 |
| **giống hệt trên cả 15 đặc trưng** | **1.393** | **955** |

Hệ quả:
1. Movement là trục đối lập nhiều nhất trong từ vựng. Trong khi đó dự án đã đo được movement là tham số **yếu nhất** của SignRep (lift 1,6× so với handshape 8,8× [V-A, GAP2]).
2. **955 ký hiệu (35 %) có "bản sao âm vị học" theo mã ASL-LEX.** Phần phân biệt chúng nằm ở thứ ASL-LEX **không mã hoá**: orientation, NMM, morpheme thứ hai, hướng movement chi tiết, chi tiết iconic. Đây là trần thông tin của mọi probe chỉ dựa trên ASL-LEX. Nó cũng là lý do mô hình phải giữ orientation và hướng.
3. Bộ 259 cặp Deaf-curated của Yin et al. [V-A] là tập con chất lượng cao. Danh sách tự động ở đây rộng hơn nhiều nhưng nhiễu hơn: chỉ morpheme đầu, nhóm do ta tự định nghĩa.

**Những gì ASL-LEX không có:**
- orientation (palm/finger orientation);
- NMM;
- phân biệt **symmetric in-phase** với **alternating**: chúng gộp chung một nhãn `SymmetricalOrAlternating`;
- hướng path (lên/xuống, xa/gần);
- chiều quay (cw/ccw);
- số lần lặp.

Vì vậy probe cho những trục này phải tự gán nhãn, hoặc dùng nguồn khác (Sem-Lex, Yin et al.).

### 1.3 Bảng ánh xạ: tham số → tín hiệu skeleton → mức signature → bất biến/nhiễu

Quy ước: D = tay trội, N = tay không trội (sau khi canonical hoá handedness). Keypoint local là COCO-WholeBody 133 2D, score SimCC thô [D1]. Gọi $S^{(1)}_{s,t}=X_t-X_s$ là level-1. Gọi $A_{ij}=\tfrac12\int(x_i\,dx_j-x_j\,dx_i)$ là Lévy area (phần phản đối xứng của level-2).

| Tham số (cột ASL-LEX) | Kênh skeleton mang thông tin | Tĩnh / động; hệ quy chiếu | Mức signature mã hoá được | Bất biến / nhiễu cần xử lý |
|---|---|---|---|---|
| **Handshape**: SelectedFingers, Flexion, Spread, Thumb\* | 21 điểm tay D. Góc gập từng đốt. Khoảng cách đầu ngón–cổ tay và đầu ngón–ngón cái. | **Tĩnh** (IF) trong frame cục bộ của bàn tay: gốc ở cổ tay, trục cổ tay→MCP giữa, chuẩn hoá theo độ dài lòng bàn tay. | **Signature không thấy tư thế tĩnh** (đứng yên thì signature = 1). Cần kênh tĩnh song song: tư thế ở frame hold, hoặc thêm basepoint (xem P0 §7). | Kích thước tay, tỉ lệ ngón (danh tính). Hand AP chỉ 58–66, mất tay ở 20–24 % frame [V-L4]. Che khuất khi chạm. |
| **Handshape change** (FlexionChange 19,5 %, SpreadChange 7,1 %) | Aperture = ‖đầu ngón được chọn − đầu ngón cái‖. Spread = góc giữa các ngón. | **Động, local** trong frame bàn tay. | Level-1 của kênh aperture: dấu cho biết mở hay đóng. Level-2 giữa aperture và path: aperture thay đổi *trong khi* tay di chuyển, so với trước/sau. | Biên độ dao động theo người ký. Jitter ngón tay rất lớn (J_acc tay 14,5 so với mặt 0,27 [V-L4]). |
| **Wrist twist** (UlnarRotation 16,5 %); orientation change | Vector ngang lòng bàn tay (MCP trỏ → MCP út) và pháp tuyến lòng bàn tay (tích có hướng, nếu có 3D). | **Động, local** quanh trục cẳng tay. | **Lévy area của đầu mút vector orientation** (vết quay trong mặt phẳng) xấp xỉ góc quay có dấu. Dấu = chiều xoay (sup/pronation). | Ở 2D, xoay quanh trục sâu bị chiếu thành co ngắn vector, nên mơ hồ. Dấu của area **đổi khi lật gương**. |
| **Orientation tĩnh** (không có trong ASL-LEX) | Như trên, nhưng lấy giá trị tại hold. | Tĩnh; tay so với thân. | Không phải signature, cần kênh tĩnh. | 2D gây mơ hồ trước/sau lòng bàn tay. |
| **Path movement shape** (Straight 45,7 / Curved 23,2 / Circular 7,6 / Z 3,4 / BackAndForth 2,6 %) | Quỹ đạo cổ tay D (hoặc tâm bàn tay), **tương đối với một mốc thân**: giữa vai, chuẩn hoá theo độ rộng vai. | **Động** (PF, nhân âm tiết). | Level-1: hướng và độ lớn độ dời. **Lévy area $A_{xy}$**: thẳng ≈ 0, cung thì area vừa phải và một dấu, tròn thì area lớn ($\pi r^2$ mỗi vòng), dấu cho cw/ccw. Z và zigzag: area đổi dấu giữa các cửa sổ con, nên cần signature đa cửa sổ hoặc level ≥ 3. **BackAndForth là path tree-like**: mọi mức signature của vị trí đều tầm thường (xem §1.4). | Signature bất biến reparametrization nên **tự động bất biến tempo**. Area tỉ lệ $s^2$ theo scale, nên phải chuẩn hoá. |
| **Hướng path** | Như trên. | Động; trong frame thân (lên/xuống, xa/gần, ipsi/contra). | Level-1 theo trục thân. Trục ipsi/contra phải lấy theo **tay trội**, không theo trái/phải ảnh. | Camera nghiêng (yaw) làm trục trộn lẫn. Xa/gần thân (trục sâu) gần như mất ở 2D. |
| **Repeated movement** (44,6 %) | Quỹ đạo cổ tay, aperture, orientation. | Động, tuần hoàn. | Vòng kín lặp k lần: area nhân k. **Đi-về thẳng lặp lại là tree-like**: signature chỉ còn độ dời ròng. Cần **lead-lag** (level-2 = quadratic variation, tỉ lệ năng lượng dao động) hoặc kênh thời gian/cửa sổ con. | **Số lần lặp > 2 là nhiễu.** Đơn hay lặp là tín hiệu. Thêm kênh thời gian thì mất bất biến tempo. |
| **Major / Minor location** (Head 26,7 %, Hand 25 %, 37 minor) | Vị trí tay D **tương đối** với mốc mặt (mũi, cằm, miệng, trán, mắt), mốc thân (ngực, vai) và tay N (lòng, mu, đầu ngón). | **Tĩnh** (IF), tại onset/hold. | **Signature bất biến tịnh tiến nên không thấy location tuyệt đối.** Có hai cách: (i) đưa **path tương đối** $D-\text{mốc}$ kèm basepoint, hoặc (ii) đưa kênh khoảng cách tĩnh. Đây là chỗ skeleton đang thua RGB (location 67,7 so với 80,7 % wins [V-A, Yin et al.]). | Tỉ lệ cơ thể và chiều cao người ký: chuẩn hoá theo độ rộng vai. Khung camera: cắt đầu, lệch tâm. |
| **Setting change** (SecondMinorLocation ≠ NA ở 70,5 %, trong đó 30 % là `Neutral`; tỉ lệ setting change thật [chưa xác minh]) | Location đầu → location cuối. | Động, thô. | **Level-1 của path tương đối mốc** là đúng đối tượng: $\Delta(D-\text{mốc})$. | Như location. |
| **Contact** (62,2 %) | min‖D − mốc‖ theo t. Tốc độ giảm về 0 tại lúc chạm (một hold). | Sự kiện. | Level-1 của kênh khoảng cách: tiến lại rồi lùi ra. Hold không đóng góp gì vào signature (bất biến reparam), nên không tự thấy lúc dừng. | Ở 2D, che khuất và độ sâu làm "chạm" và "gần" lẫn nhau. Keypoint hỏng nhiều nhất khi tay chạm mặt hoặc tay chạm tay [V-L4]. |
| **Sign type / Battison** (60,8 % hai tay) | Năng lượng chuyển động của N, tương quan D–N, handshape N. | Động. | Năng lượng N (level-1 tổng biến thiên, hoặc lead-lag) phân biệt một tay, asym (N đứng yên) và sym. **In-phase vs alternating**: xem §1.4, Lévy area chéo trên cả chu kỳ **không** phân biệt được. Cần lead-lag cross-covariation hoặc kênh tổng/hiệu sau khi gương hoá N. | Tay N "nghỉ" có thể ở trong khung hoặc ngoài khung. Hoạt động tay N hay bị nhầm thành nhiễu. |
| **Tay không trội** (handshape N; 96,6 % thuộc tập Battison) | 21 điểm tay N, frame cục bộ. | Tĩnh. | Kênh tĩnh, như handshape D. | Ít thông tin hơn tay D. Có thể dùng encoder tay chung với gương hoá (SignSeek [V-L4]). |
| **Finger wiggle** (không có cột riêng) | Góc MCP các ngón. | Dao động nhanh, lệch pha giữa các ngón. | **Lévy area chéo giữa hai ngón kề nhau** tỉ lệ $\sin\varphi$, khác 0 khi lệch pha. Gập đồng pha thì tree-like. | Cần fps ≥ 25 (ASL Citizen 30 fps [I]). Jitter ngón dễ giả wiggle. |
| **NMM: mouthing / mouth gesture** | 20 điểm miệng (COCO-WB 71–90). Độ mở miệng, độ rộng môi. | Động. | Level-1 và level-2 trên (độ mở, độ rộng): ví dụ "o"→"m". | Mouthing tiếng Anh rất dao động theo người ký. Giả thuyết chưa đo: miệng là kênh dễ rò rỉ danh tính. |
| **NMM: lông mày, đầu** | Khoảng cách lông mày–mắt; góc mũi–cổ (gật, lắc). | Động. | Level-1. | Chủ yếu mang ngữ pháp câu. Với ISLR thì gần như là nhiễu (§2). |

### 1.4 Năm hệ quả toán học của signature cần đưa vào thiết kế (delta so với P0)

1. **Bất biến tịnh tiến, nên mất location và handshape tĩnh.** P0 §7 đã nêu. Bổ sung từ dữ liệu: location (IF) và handshape (IF) là hai tham số tĩnh. Chúng chiếm phần lớn đối lập trong từ vựng ngoài movement (3.048 + 1.886 cặp, §1.2). → Bắt buộc có **kênh tĩnh hoặc basepoint**, và path phải được biểu diễn **tương đối với mốc** (mặt, thân, tay N), không theo toạ độ ảnh.
2. **Tree-like equivalence.** Path đi rồi quay lại đúng đường cũ có signature tầm thường. Trong ASL-LEX, BackAndForth chiếm 2,6 %. Nhưng **nhiều ký hiệu Straight kèm Repeated** (một phần trong 44,6 %) cũng là đi-về trong không gian vị trí. → Phải dùng **lead-lag transform** (level-2 của lead-lag cho quadratic variation, phản ánh năng lượng và số lần dao động), hoặc **signature theo cửa sổ con** (Path-JEPA đã có token local, P0), hoặc thêm kênh thời gian. Riêng thêm kênh thời gian sẽ phá bất biến tempo.
3. **Cross-hand: in-phase vs alternating.** Đặt $y_D=\sin\omega t$, $y_N=\sin(\omega t+\varphi)$. Trên một chu kỳ, $A(y_D,y_N)=-\pi\sin\varphi$, **bằng 0 cả khi φ = 0 (in-phase) lẫn φ = π (alternating)**. Thực tế path $(y_D,y_N)$ khi đó là đoạn thẳng đi-về trên đường chéo hoặc phản chéo, tức tree-like. Lévy area chéo trên cả chu kỳ chỉ phân biệt được lệch pha 1/4 chu kỳ.
   → Có ba cách sửa:
   - (a) tính cross quadratic covariation $\sum\Delta y_D\Delta y_N$ (dương khi in-phase, âm khi alternating), lấy từ level-2 của lead-lag;
   - (b) dùng **kênh tổng và hiệu** $D\pm\mathcal M(N)$ sau khi gương hoá N qua mặt phẳng dọc giữa thân: in-phase thì hiệu ≈ hằng, alternating thì tổng ≈ hằng;
   - (c) dùng level-1 trên cửa sổ con nửa chu kỳ.

   Đây là **chỉnh sửa cho P0 dòng "Lévy area chéo… mã hoá đối xứng cùng pha so với luân phiên"**: nhận định đó chỉ đúng khi dùng cửa sổ con hoặc lead-lag.
4. **Gương hoá đổi dấu Lévy area.** Phép phản xạ có det = −1, nên cw và ccw đổi chỗ. Người ký thuận tay trái thực hiện ký hiệu đối xứng gương, nên dấu area trong toạ độ ảnh **phụ thuộc handedness**. Hai lựa chọn:
   - Lấy |area| để bất biến: **sai**, vì mất chiều quay, vốn có thể là đối lập.
   - Canonical hoá: gương hoá toàn bộ clip để tay trội luôn ở cùng một phía, rồi giữ dấu. Đây là cách đúng.

   ASL Citizen dùng random horizontal flip "to simulate left and right handed signers" [V, §4]. Với signature, flip phải đi kèm hoán đổi nhãn D/N.
5. **Đảo thời gian thì đổi dấu level-1 và Lévy area.** Signature nhạy với thứ tự, khác với SignRep: embedding của clip đảo ngược có cosine 0,959 với clip gốc [V-A, I]. Đây là lợi thế có thể đo được (§4). Ngược lại, signature **bất biến reparametrization**: holds và tốc độ biến mất. Điều này tốt cho tempo, nhưng mất thông tin hold-duration. Theo Liddell & Johnson, hold-duration chủ yếu là prosody, không phải từ vựng [K].

Biến đổi affine của camera: scale nhân level-k với $s^k$; xoay trong mặt phẳng giữ area và xoay level-1. → Chuẩn hoá scale theo độ rộng vai **trước** khi tính signature. Nếu muốn bất biến hướng, chỉ xoay theo trục vai.

---

## 2. Bằng chứng thực nghiệm: kênh nào quan trọng (delta so với A/L4)

### 2.1 Tay, thân, miệng, mặt: ISLR so với CSLR, cùng một họ mô hình (HRNet COCO-WholeBody heatmap + S3D)

| Paper | Task / dataset | Upper body | + Tay | + Miệng | + Mặt còn lại | Nguồn |
|---|---|---|---|---|---|---|
| NLA-SLR (CVPR 2023, 2303.12080) Table 11 | **ISLR** WLASL2000, top-1 P-I | 21,37 (11 kp) | **48,54** (53 kp) | 49,10 (63 kp). Tay + miệng không thân: 48,64 | — | [V] |
| TwoStream-SLR (NeurIPS 2022, 2211.01367) Table 5a | **CSLR** Phoenix-2014T, WER test ↓ | 48,46 | **36,88** | **28,20** (giảm 8,7 WER) | 27,19 | [V] |

Như vậy **miệng gần như không đóng góp cho ISLR tiếng ASL** (+0,56 top-1), nhưng **cực kỳ quan trọng cho CSLR tiếng DGS** (−8,7 WER, gần bằng tác dụng của hai tay).

Các bằng chứng hỗ trợ:
- **Uni-Sign**: thêm face sub-pose được +1,89 trên MSASL1000 khi huấn luyện từ đầu [V-L4].
- **Pham et al. (2507.20884, SLTAT @ IVA'25, DOI 10.1145/3742886.3756718; DGS Corpus, ký hiệu rời)** [V]:
  - Body ROI + mouth tăng F1 từ 0,816 lên 0,889 với CSN, và từ 0,850 lên 0,874 với MViT.
  - Mouth đứng riêng được 0,486. Eyes đứng riêng được 0,185 F1.
  - Với MViT, phần tăng top-1 không có ý nghĩa thống kê.
  - (Các ô accuracy bị mất khi render HTML, chỉ cột F1 đọc được.)
  - DGS là ngôn ngữ có mouthing dày đặc [K].
- **NMFs-CSL** (TOMM, 2008.10428) [V]: 1.067 từ CSL, trong đó **610 từ "NMF-aware confusing"**. 32.010 mẫu, 10 người ký × 3 lần. Đây là dataset *được thiết kế* để NMM quyết định. Không đại diện cho ASL ISLR.
- **OpenASL** (EMNLP 2022, 2205.12870) [V]: dùng đặc trưng mouthing + handshape cho SLT. Nêu ví dụ SENATE/COMMITTEE chỉ khác handshape.

→ **Kết luận cho ASL Citizen / WLASL / MS-ASL (ISLR): ưu tiên tay > quan hệ tay–mặt/thân > thân > miệng ≫ lông mày/mắt.** Kênh miệng chỉ nên là kênh phụ có thể mask. Nó trở nên quan trọng nếu sau này mở rộng sang How2Sign/CSLR.

### 2.2 Movement, vận tốc, signature
- **Multi-stream** joint/bone/motion (SL-GCN, +6 điểm) và motion residual (MASA, mask 90 %) đã có trong L4.
- **Dữ liệu dự án** [V-A, I]:
  - Clip chỉ gồm frame tĩnh vẫn giữ **83 % R@1** của SignRep.
  - Lift nhầm lẫn: movement 1,6×, location 2,9×, handshape 8,8×.
  - Embedding gần như mù thứ tự (cosine 0,959 với clip đảo ngược).
  - Các con số này cộng với §1.2 (movement là trục minimal pair lớn nhất) là **lý do chính để dùng signature**.
- **Path signature trong nhận dạng cử chỉ/hành động**: Li et al. 2018/2019 (arXiv 1811.07081) dùng PS features + temporal transformer cho gesture dựa trên skeleton [V tiêu đề qua arXiv API]. Truy vấn arXiv `abs:"path signature" AND abs:"sign language"` **trả về 0 kết quả** [V]. Signature cho ISLR là khoảng trống, trong phạm vi tìm kiếm này.

### 2.3 Minimal pair và trần thông tin
- Yin et al. 2026 (A §3.1): pose thắng RGB về handshape và movement, thua về location. Kết quả này khớp với hệ quả 1 ở §1.4: skeleton được chuẩn hoá vị trí sẽ mất location tuyệt đối.
- §1.2: 35 % ký hiệu không phân biệt được bằng 15 đặc trưng ASL-LEX. Phần còn lại nằm ở orientation, NMM và chi tiết hướng.
- Biến thể (WHATFOR1/2/3, RESEARCH1/2) dùng chung mã ASL-LEX nhưng là các gloss khác nhau trong ASL Citizen. Đây là nguồn lỗi "không thể tránh" khi retrieval theo gloss.

---

## 3. Thẻ dataset (mức paper; số local xem D1)

| Dataset | Paper | Kích thước | Người ký | fps / clip | Pose có sẵn | Split / metric chuẩn |
|---|---|---|---|---|---|---|
| **ASL Citizen** | Desai et al., arXiv **2304.05934**. NeurIPS 2023 D&B [K]. | **83.399 video, 2.731 ký hiệu** (từ vựng ASL-LEX), 30,5 video/ký hiệu [V] | **52** Deaf/HH, gồm seed signer. Split 35/6/11, **rời nhau theo người**. Video: 40.154 / 10.304 / 32.941 [V Table 4] | 30 fps, 640×480 chủ yếu [I]. Clip trung vị 2,54 s (p10 1,75, p90 4,39) [I, mẫu 208]. Sign chỉ chiếm khoảng 1/3 clip (ASL-LEX 834 ms). | Paper chỉ dùng MediaPipe Holistic 27 kp cho ST-GCN [V]. Bản local là COCO-WB 133 [D1]. | Dictionary retrieval: **R@1/5/10, DCG, MRR** trên test [V] |
| **WLASL** | Li et al., WACV 2020, arXiv **1910.11006** [V] | 21.083 video, 2.000 gloss, khoảng 14 h. Subset 100/300/1000 [V] | **119**. Chia 4:1:1 **theo mẫu** trong từng gloss, nên **không rời người** [V]. ASL Citizen gọi đây là "random split" [V]. | Nguồn YouTube/web. Bản local đã crop 256×256, 25 fps, trung vị 2,2 s [I, D1]. | Paper: OpenPose 55 kp (13 thân + 2×21 tay) [V] | top-1/5/10 P-I và P-C. Pose-TGCN 23,65; I3D 32,48 [V] |
| **MS-ASL** | Joze & Koller, BMVC 2019 [K], arXiv **1812.01053** | 25.513 mẫu, 1.000 lớp, 24 h 39 phút. Subset 100/200/500 [V] | **222** (gom cụm qua face embedding). Train/val/test 165/37/20, **rời người** [V] | Hỗn hợp 15–60 fps, nhiều độ phân giải [I]. Số frame phân bố kiểu Poisson, trung bình khoảng 60 [V] | Không phát hành | **Per-class** top-1/5. I3D ASL1000: 57,69 [V] |
| **How2Sign** | Duarte et al., CVPR 2021, arXiv **2008.08143** [V] | >80 h, >35k câu, từ vựng 16k (khoảng 20 % fingerspelling) [V] | **11** (5 hearing, 4 Deaf, …). Test có 26 video trùng của một người ký không có trong train [V] | Green screen 1280×720, **30 fps** theo paper [V]. Bản local chủ yếu 24 fps [D1]. | OpenPose 2D body, tay, mặt. **3 h Panoptic 3D** [V] | SLT: BLEU, ROUGE, BLEURT |
| **YouTube-ASL** | Uthus et al., arXiv **2306.15162**. NeurIPS 2023 D&B [K]. | **984 h**, 11.093 video, khoảng 60K caption [V] | **>2.519** (số kênh, là cận dưới) [V] | Lọc 15–60 fps, độ dài 10 s–5 h [V] | Không phát hành pose. Baseline dùng MediaPipe [V-L4] | SLT. **Chứa 38,2 % test MS-ASL** [V-L4, SHuBERT] |
| **OpenASL** | Shi et al., EMNLP 2022, arXiv **2205.12870** [V] | 288 h, 98.417 cặp, 33.549 từ [V] | >200 (bảng ghi 220) [V] | web | — | SLT. 72,4 % test nằm trong YouTube-ASL [V-L4] |
| **YouTube-SL-25** | arXiv **2407.11144** [V] | **3.207 h, 39.197 video**, 25 ngôn ngữ ký hiệu, khoảng 3.000 người ký [V] | | | Bản DWPose: SignVerse-2M [V-A] | Corpus pretrain của SignRep (đã loại ID WLASL) [V-A] |

**Vấn đề đã biết.**
- **WLASL:**
  - không rời người;
  - link YouTube chết dần. Bản local có đủ 21.095 mục theo `nslt_2000` [D1], nhưng tỉ lệ mất ở bản tải công khai [chưa xác minh];
  - nhãn biến thể/phương ngữ chỉ được giữ khi có ≥5 mẫu [V];
  - SignRep chọn checkpoint trên WLASL val [V-A].
- **MS-ASL:**
  - link YouTube chết dần. Bản local còn **18.855/25.513 = 73,9 %** [D1, I];
  - trùng với YouTube-ASL (38,2 % test);
  - có thể trùng video với WLASL vì cùng nguồn YouTube [chưa xác minh].
- **ASL Citizen:**
  - có 6 gloss không có mã ASL-LEX, và biến thể dùng chung mã (§1.2);
  - clip tự quay qua webcam. Thời điểm bắt đầu/kết thúc dao động, đầu và cuối clip là tư thế nghỉ [V §3];
  - nền và ánh sáng tự do;
  - làm sạch: lọc video <150 KB, làm mờ 268 + 293 + 32 video, bỏ 513 video [V];
  - **từ vựng trùng giữa train và test**, nên retrieval không phải open-vocabulary [V-A].
- **How2Sign:** chỉ 11 người ký, nên nghiêng về danh tính. Thư mục test_label local bị rỗng [D1].

**Protocol frozen retrieval bắt nguồn từ chính ASL Citizen [V §5.3].**
- Lấy "globally pooled feature … nearest-neighbor classifier using the cosine distance from ASL Citizen training set representation".
- Kết quả: I3D-ASLC features **58,03** R@1 (DCG 0,7652); I3D-WLASL features 9,85; I3D-Kinetics 0,41 (DCG 0,1234).
- SignRep (Table 4) và SignSeek dùng cùng quy tắc này [V-A].

**Baseline supervised [V Table 5]:**

| Model | DCG | MRR | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| I3D | 0,7913 | 0,7332 | 0,6310 | 0,8609 | 0,9086 |
| ST-GCN (MediaPipe 27 kp) | 0,7637 | 0,6997 | 0,5952 | 0,8268 | 0,8813 |

**Đường cong theo lượng dữ liệu** (I3D, R@1) [V Table 9]: 25 % → 0,162; 50 % → 0,481; 75 % → 0,592; 100 % → 0,631. Chưa bão hoà, dù mỗi ký hiệu đã có khoảng 30 video.

**SignRep trên ASL Citizen** (lấy từ A): frozen retrieval R@1 37,47 (avg) / **49,95** (weighted), DCG 61,40 / 71,21. Fine-tuned R@1 81,37. Bản tái lập của dự án: 49,02 [I].

---

## 4. Tổng hợp cho thiết kế

### 4.1 "Tín hiệu nên ép mô hình học" (xếp hạng)

Tiêu chí xếp hạng gồm ba ý: (a) độ phủ trong từ vựng (ASL-LEX); (b) bằng chứng thực nghiệm rằng tín hiệu phân biệt được; (c) mức SignRep hoặc skeleton hiện **đang yếu** ở đó (cơ hội vượt).

1. **Handshape của tay trội, trong frame cục bộ bàn tay** (tĩnh và thay đổi aperture).
   - Phân bố đuôi dài trên 58 giá trị; 3.048 cặp minimal pair.
   - Loại bỏ tay làm NLA-SLR mất 27 điểm. Pose đã thắng RGB ở đây (Yin et al.).
   - Hình thức: kênh tĩnh (không signature) + signature của aperture. Encoder tay dùng chung cho D và N đã gương hoá.
2. **Path movement của tay trội tương đối mốc thân: hình dạng, hướng, chiều quay.**
   - 84,7 % ký hiệu có path; movement là trục minimal pair lớn nhất (7.510 cặp).
   - SignRep yếu nhất ở đây (lift 1,6×, mù thứ tự).
   - Hình thức: level-1 (hướng) + Lévy area (cung/tròn, cw/ccw) trên cửa sổ con; sau khi gương hoá canonical.
3. **Quan hệ tay–mặt, tay–thân, tay–tay (location + contact + setting change).**
   - Head 26,7 %, Hand 25 %; contact 62,2 %; SecondMinorLocation khác NA ở 70,5 % (cận trên của setting change).
   - Skeleton thua RGB về location (67,7 so với 80,7). SignRep có prior "hand-interaction distance" 12×22×3 và prior này có lợi (dist 47,2 ≫ kpt 32,9 [V-A]).
   - Hình thức: path $D-\text{mốc}$ kèm basepoint, cộng khoảng cách tĩnh tại onset. Không dùng signature thuần, vì nó bất biến tịnh tiến.
4. **Cấu trúc hai tay theo Battison** (một tay / asym / sym in-phase / alternating).
   - 60,8 % hai tay. Gần như tất định nhờ điều kiện Battison: N đứng yên khi khác handshape, tới 96,6 %.
   - Ít bị khai thác: SignRep chỉ có prior activity.
   - Hình thức: năng lượng N + **cross quadratic covariation lead-lag hoặc kênh tổng/hiệu** (§1.4-3), **không** dùng Lévy area chéo trên toàn cửa sổ.
5. **Local movement: wrist twist (16,5 %), đổi aperture (19,5 %), wiggle.**
   - Theo Brentari, đây là các "nhân âm tiết" có sonority thấp. Chúng là thứ duy nhất phân biệt 9,6 % ký hiệu chỉ có local movement.
   - Hình thức: Lévy area của vector orientation lòng bàn tay; Lévy area chéo giữa các ngón cho wiggle.
6. **Đơn hay lặp** (44,6 % lặp).
   - Hình thức: lead-lag hoặc cửa sổ con. Phải **bất biến với số lần lặp > 2**.
7. **Thứ tự thời gian trong ký hiệu** (onset → offset, setting change).
   - Signature nhạy thứ tự sẵn. Đây là lợi thế đo được trực tiếp so với SignRep.
8. **Miệng: kênh phụ, có thể mask.**
   - ISLR ASL chỉ +0,56 (NLA-SLR). Quan trọng cho CSLR (−8,7 WER) và NMFs-CSL.
   - Giữ kênh để dùng cho How2Sign sau này, nhưng không để nó chi phối loss.
9. *(Không ưu tiên)* Lông mày, mắt, gật/lắc đầu: chủ yếu mang ngữ pháp câu.

**Probe kiểm chứng tương ứng (frozen, tất cả CPU sau khi đã trích feature):**
- 16 head tuyến tính ASL-LEX trên **split rời gloss**.
- Minimal pair §1.2 theo từng nhóm, và 259 cặp của Yin et al.
- Probe tự gán nhãn cho cw/ccw và in-phase/alternating, vì ASL-LEX không có hai nhãn này.
- Δ cosine khi đảo thời gian.
- Kiểm tra lỗi tập trung ở 955 ký hiệu "giống hệt ASL-LEX": khoảng trần.

### 4.2 "Tín hiệu nhiễu nên bất biến"

| Nhiễu | Bằng chứng gây hại | Cách bất biến phù hợp với signature |
|---|---|---|
| **Danh tính người ký** (tỉ lệ xương, phong cách chuyển động, độ lớn biên độ) | Pose nhận diện người ký 80,18 % so với 3,23 % may rủi [V-L4]. Láng giềng gần nhất của SignRep frozen: cùng người ký 48,7 % so với cùng gloss 43,8 % [I] | Retarget độ dài xương (median skeleton, như SignRep A1). Chuẩn hoá biên độ theo người ký. Báo cáo signer-ID probe như **metric âm**. |
| **Tempo / tốc độ, hold-duration** | Thời lượng ASL-LEX p10–p90 là 467–1201 ms, chênh 2,6× | **Signature tự bất biến reparametrization.** Không thêm kênh thời gian thô. Nếu cần lead-lag thì chuẩn hoá theo số mẫu. |
| **Rest / đoạn đầu cuối clip** | Sign chiếm khoảng 1/3 clip ASL Citizen. Activity weighting của SignRep được +12,5 R@1 [V-A] | Path đứng yên đóng góp ≈ 0 vào signature, đây là lợi thế. Nhưng đoạn tay hạ/nâng đi vào và ra khỏi nghỉ thì lại là path lớn. Cần cắt theo năng lượng chuyển động hoặc trọng số activity. |
| **Kích thước cơ thể, khoảng cách tới camera** | Area tỉ lệ $s^2$, level-k tỉ lệ $s^k$ | Chuẩn hoá theo độ rộng vai (ST-GCN của ASL Citizen làm như vậy [V]) trước khi tính signature. |
| **Khung hình camera** (lệch tâm, nghiêng, cắt) | Webcam tự quay (ASL Citizen), YouTube (WLASL, MS-ASL) | Toạ độ tương đối mốc thân. Chỉ xoay theo trục vai, vì xoay giữ area. Mask khi điểm nằm ngoài khung (toạ độ <0 hoặc >1 [D1]). |
| **Handedness** (người thuận tay trái) | +5 R@1 khi canonical hoá dominant/non-dominant trong ISR supervised [V-A, split chưa chắc chuẩn]. ASL Citizen dùng flip để mô phỏng [V] | **Gương hoá canonical** rồi giữ dấu Lévy area. **Không** dùng |area|. Xác định tay trội theo năng lượng hoặc độ cao tay (sign-lang 2026 [V-A]). |
| **Số lần lặp > 2** | Biến thể tự do [K] | Lead-lag chuẩn hoá theo năng lượng, hoặc pooling max trên cửa sổ con. |
| **Jitter và mất keypoint tay** | Hand AP 58–66. MediaPipe mất tay ở 20–24 % frame. J_acc tay gấp khoảng 50 lần mặt [V-L4] | Signature cộng dồn nhiễu ở level-2 (Lévy area của nhiễu trắng ≠ 0 trên cửa sổ dài) → làm trơn trước, dùng mask theo confidence, và coi frame mất tay là mask tự nhiên. |
| **Mouthing tiếng Anh tuỳ người** | Rất dao động theo người ký [K]. Đóng góp ISLR nhỏ (§2.1) | Kênh miệng tách riêng, có thể drop/mask ngẫu nhiên. Không để nó làm target chính. |
| **Nguồn/extractor** (MediaPipe ↔ RTMW) | Đổi extractor làm số liệu Sem-Lex lệch tới 15 điểm [V-L4] | Mọi dataset local dùng chung COCO-WB 133 RTMW [D1]. Kiểm tra chéo khi đổi extractor. |

---

## 5. Tóm tắt các điểm mới (so với A/L4/P0)

1. **ASL-LEX 2.0 đã có trên máy (tạm thời)** cùng đủ tên cột. 99,8 % dòng ASL Citizen nối được qua `ASL-LEX Code`, nên có thể làm mọi probe âm vị học ngay.
2. **Phân bố:**
   - 60,8 % hai tay; 84,7 % có path; 44,6 % lặp; 62,2 % chạm;
   - Head 26,7 % và Hand 25 % (location quan hệ chiếm đa số);
   - wrist twist 16,5 %; đổi aperture 19,5 %.
3. **Movement là trục minimal pair lớn nhất** (7.510 cặp). 35 % ký hiệu không phân biệt được bằng mã ASL-LEX, nên orientation, NMM và hướng là các tín hiệu "ẩn" bắt buộc.
4. **Ba bẫy toán học của signature:**
   - bất biến tịnh tiến, nên mất location và handshape;
   - tree-like, nên mất back-and-forth và lặp đi-về;
   - Lévy area chéo trên toàn chu kỳ **không** phân biệt được in-phase với alternating.

   Kèm theo đó: gương hoá đổi dấu area. Cách sửa: kênh tĩnh/basepoint, lead-lag, cửa sổ con, kênh tổng/hiệu sau gương hoá.
5. **Miệng: ISLR ASL +0,56 so với CSLR DGS −8,7 WER**, cùng họ mô hình. Kênh miệng là phụ cho ASL Citizen/WLASL/MS-ASL.
6. **Bảng dataset đã xác minh lại:**
   - ASL Citizen 83.399 / 2.731 / 52, split 35/6/11, 30 fps, clip khoảng 2,5 s;
   - WLASL 21.083 / 2.000 / 119, chia không rời người;
   - MS-ASL 25.513 / 1.000 / 222 (165/37/20), bản local còn 73,9 %;
   - How2Sign 80 h / 11 người ký / 30 fps;
   - YouTube-ASL 984 h / ≥2.519;
   - YT-SL-25 3.207 h.
   - Protocol frozen retrieval có gốc ở ASL Citizen §5.3 (I3D features 58,03).
