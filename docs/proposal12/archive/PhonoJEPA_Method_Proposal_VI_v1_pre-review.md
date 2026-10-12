# Proposal 12 — **PhonoJEPA**: tự giám sát skeleton cho ngôn ngữ ký hiệu bằng cách che *kênh âm vị học* và dự đoán biểu diễn của kênh bị che

> **Trạng thái:** đề xuất phương pháp để review, **chưa implement, chưa train**. Mọi con số "kết quả" trong tài liệu đều thuộc một trong ba loại:
> 1. số liệu đã công bố, kèm nguồn;
> 2. số liệu đo được trong repo này;
> 3. chẩn đoán *training-free* tôi chạy cho proposal (§5, `pilot_diagnostics/`).
>
> **Ngày:** 2026-10-11. **Phạm vi:** skeleton-only SSL (2-D COCO-WholeBody keypoints), có nhánh mở rộng nối sang SignRep (RGB) ở §15.
> **Tài liệu kèm theo:**
> - `pilot_diagnostics/`: script và kết quả D1–D5.
> - `literature/`: 4 báo cáo literature review chi tiết bằng tiếng Anh, khoảng 25k từ, số liệu gắn tag đã/không kiểm tra raw text.

---

## 0. Tóm tắt (TL;DR)

**Ý tưởng gốc của anh:** che một số joint hoặc bộ phận cơ thể, rồi dùng phần còn lại để dự đoán *representation* của phần bị che (skeleton SSL, structured anatomical masking, JEPA).

**Phản biện ngắn:** ở dạng thuần, ý tưởng này **đã tồn tại**.
- **S-JEPA** (ECCV'24): JEPA trên joint, mask theo motion.
- **SLiM** (arXiv 2603.10648, "Semantic Tube Masking" — anatomical tube; anh gọi là SLim NeurIPS'26).
- **Path-JEPA** (ECCV'26): mask theo cây giải phẫu, lan truyền phụ thuộc.
- **SignSeek** (arXiv 2609.03695, 9/2026): *masked articulator prediction* trong sign language, nhưng có dùng nhãn gloss.
- Trong chính repo này: **obs_jepa arm B** (chưa train).

Đồng thời có **4 lỗ hổng** khi áp dụng nguyên bản cho ngôn ngữ ký hiệu. Cả 4 đều có bằng chứng, trong literature hoặc trong chẩn đoán tôi chạy trên ASL Citizen:
1. **Shortcut nội suy và động học.**
   - Handshape tại frame *t* dự đoán được từ *t±1* với R² = 0.69.
   - Toạ độ tay dự đoán được với R² = 0.87.
   - Vị trí cổ tay lộ qua khuỷu tay.
2. **Che toàn bộ một bộ phận là bài toán gần như bất định.**
   - Ngay cả "oracle từ vựng" (biết gloss) chỉ giải thích R² = 0.25 phương sai handshape.
   - Repo đã thấy cross-articulator *forecasting* thất bại: CIP 0/36.
3. **Danh tính signer là "slow feature" mà JEPA ưu tiên học.**
   - Probe signer trên mọi model trong repo: 0.92–0.99.
   - Đặc trưng miệng (chưa chuẩn hoá) nhận diện signer với balanced-acc 0.83 nhưng gần như không mang thông tin gloss (0.24%).
4. **Hồi quy latent thuần (MSE/L1) trên skeleton dễ collapse.**
   - SLiM: feature-MSE xuống effective rank 1.12.
   - S-JEPA: bỏ EMA thì đạt 1.6%.
   - Repo: `signrel_jepa` có cos 0.998.

**Đề xuất: PhonoJEPA.** Chuyển đơn vị che từ *bộ phận giải phẫu* sang **kênh âm vị học** (phonological channels). Mỗi tay được phân rã thành:
- **SHAPE**: cấu hình ngón trong hệ toạ độ lòng bàn tay, đã chuẩn hoá độ dài xương;
- **ORI**: hướng và co rút phối cảnh của lòng bàn tay;
- **LOC**: vị trí cổ tay và khuỷu tay so với thân, kèm vận tốc. Cánh tay được gộp vào LOC để đóng rò rỉ.

Mặt được phân rã thành UPPER (lông mày, mí mắt), MOUTH và HEAD. Tất cả được chuẩn hoá *leak-free* để khử hình thái cơ thể signer.

**Năm thành phần của PhonoJEPA:**
1. **Phonological channel tokens.** 9 kênh × 32 time-bin = 288 token, so với 2,144 token của obs_jepa B.
2. **Predictability-Ladder Channel Masking (PLCM).**
   - Hỗn hợp các họ mask có độ khó được *đo* trước (§5), từ nội-tay → khối thời gian → kênh âm vị → toàn articulator, có curriculum dễ→khó.
   - Luật **đóng phụ thuộc**: cánh tay nằm chung token với LOC; ngón bị che không gồm MCP; mọi chuẩn hoá chỉ dựa vào anchor.
   - Span tối thiểu ≥ 4 bin để chặn nội suy.
3. **Hai loại target cho "representation của phần bị che":**
   - **contextual**: teacher EMA nhìn toàn cục, như I-JEPA;
   - **intrinsic**: teacher chỉ nhìn chính kênh đó trong cửa sổ thời gian cục bộ. Đây đúng nghĩa đen là "biểu diễn của phần bị che", ít nhiễm danh tính signer.

   Target intrinsic dùng **prototype riêng cho từng nhóm kênh**, tức một bộ "đơn vị âm vị học" học online. Cách này kiểm chứng được bằng NMI với nhãn ASL-LEX.
4. **Chống collapse theo bằng chứng skeleton:** EMA + softmax-CE có centering/Sinkhorn (S-JEPA, SLiM), có nhánh thay thế L1 + SIGReg (LeJEPA/PointTube) để pilot quyết định.
5. **Khử signer không cần nhãn:**
   - augment hình thái *bất đối xứng* giữa student và teacher (độ dài tay/cẳng tay, vai, mặt) và tốc độ ký;
   - chuẩn hoá biểu cảm theo "neutral face" của clip;
   - đo bằng signer-probe như một **negative metric**.

Kèm theo một **global term** (DINO-CLS hoặc LeJEPA) cho clip embedding đẳng hướng, phục vụ retrieval. Lý do: repo cho thấy whitening thay đổi R@1 rất mạnh.

**Mục tiêu định lượng** (frozen, label-free) trên ASL Citizen dictionary retrieval:
- Vượt rõ MASA, skeleton SSL label-free tốt nhất đã công bố: **27.96 R@1**.
- Thu hẹp khoảng cách tới SignRep (RGB): **49.95**.
- Cùng lúc giảm signer-probe và tăng phonology-probe (ASL-LEX, *lexically disjoint*).

**Khả năng thất bại được nêu thẳng:**
- Literature cho thấy JEPA trên skeleton chỉ hơn MAMP (motion-MAE) **+0.4** điểm (S-JEPA 85.3 vs MAMP 84.9).
- Lợi ích của masking có cấu trúc thường chỉ **+1–3** điểm.

Vì vậy kế hoạch thực nghiệm được thiết kế để kết quả *null cũng có giá trị*: baseline matched, pre-registered decision rules, kill criteria.

---

## 1. Mục tiêu, phạm vi và câu hỏi nghiên cứu

**Mục tiêu:** học encoder *f* trên chuỗi keypoint 2-D (không nhãn) sao cho biểu diễn:
- (a) phân biệt được từ vựng ký hiệu khi *frozen* (retrieval, linear/attentive probe);
- (b) mã hoá được các tham số âm vị học (handshape, location, movement, orientation, non-manual) theo cách *tổng quát hoá sang ký hiệu chưa thấy*;
- (c) ít mang danh tính signer;
- (d) fine-tune tốt cho ISLR, và tuỳ chọn cho CSLR/SLT.

**Phạm vi:**
- Skeleton-only, 2-D, COCO-WholeBody 133 điểm (định dạng đồng nhất trên 6 dataset local).
- Không dùng nhãn trong pretraining.
- RGB chỉ xuất hiện ở phần mở rộng (§15).

**Câu hỏi nghiên cứu:**
- **RQ1:** Che theo *kênh âm vị học* có tốt hơn che theo *bộ phận giải phẫu* (SLiM/S-JEPA-style) và che ngẫu nhiên không, khi cùng token, cùng target và cùng budget?
- **RQ2:** "Representation của phần bị che" nên là gì: *contextual* (teacher toàn cục), *intrinsic* (teacher cục bộ theo kênh) hay cả hai? Lựa chọn này ảnh hưởng thế nào tới thông tin âm vị học và danh tính signer?
- **RQ3:** Thang độ khó của mask (đo bằng R² dự đoán được) và curriculum dễ→khó có cần thiết không, hay chỉ cần một họ mask?
- **RQ4:** Có khử được danh tính signer mà không cần nhãn signer không, bằng chuẩn hoá leak-free và augment hình thái bất đối xứng?
- **RQ5:** Latent target có thắng target tường minh (toạ độ, chuyển động kiểu MAMP) trên skeleton sign không? Thắng ở protocol nào (frozen hay fine-tune)?

---

## 2. Kiểm kê codebase: đã làm gì, có bằng chứng gì, và bài học ràng buộc thiết kế

Tôi rà toàn bộ `docs/proposal1–11`, `docs/storage`, `docs/proposal4/.../research` và 12 thư mục `method/*`. Thứ tự ưu tiên bằng chứng: eval JSON/log trước, ledger sau, spec cuối cùng. Lưu ý: `STATE.md` (v9, 2026-10-07) đã cũ hơn EX16/P0.

### 2.1 Bảng tổng hợp hướng nghiên cứu

| Hướng (proposal) | Code | Trạng thái | Số đo tốt nhất (nguồn) | Kết luận |
|---|---|---|---|---|
| Readout bins2/chamfer trên SignRep frozen (p4) | `p4_signssl` | test 1 lần, locked | ASLC test R0 49.02 → R1 54.93 → R2 58.42 (`runtime/eval/EX05_result.json`) | **Supported**. Readout thay đổi kết quả nhiều hơn mọi tinh chỉnh encoder. |
| H008 set→vector distillation | `p4_signssl` | test, 3 seeds | 57.28–57.31 ASLC test | Supported, novelty thấp |
| CIP-Sign cross-articulator innovation (p5) | `p4_signssl/cip` | premise test ×3 | **0/36 gates** (`runtime/cip/gates.json`) | **Negative**. Forecasting chéo articulator không thêm thông tin so với lịch sử của chính nó. |
| Spectral aux, SignRep continuation (p3) | `signrep_spectral` | val | B1 8.4 / M1 7.6 vs B0 37.5 | Negative. Continuation phá representation, eff-rank 24. |
| ASC-JEPA (p8) | `asc_jepa` | val | 48.99 vs 49.50 gốc | Negative |
| SignRel-JEPA (p6) | `signrel_jepa` | 95 clip | collapse (cos 0.998) | Negative |
| SIGNREL temporal, pose + DINO (p7) | `signrel_temporal` | locked test | A4 ≤ 11.25 vs **raw input 21.25** | Negative. Train còn làm tệ hơn đặc trưng thô. |
| JEPA vs MAE + 2-D priors, V-JEPA init (p10/EX15) | `jepa_targets` | val, 1 seed | JMOT 20.73, JP 19.93, J 7.38, M 4.11 (LOSO white512) | Prior chuyển động tốt nhất; non-manual có hại (JNM 1.96) |
| SignRep repro vs JEPA + priors, cùng Hiera init (EX16/P0) | `jepa_targets`, `jepa_baseline` | val, 1 seed | **JH3D 39.15 LOSO / 61.7 LP** vs SR3D 5.10 / 17.6 | Latent + prior ≫ repro SignRep ở quy mô 3k video (repro có thể under-fit) |
| obs_jepa A/B/C (p11) | `obs_jepa` | implemented, **chưa train** | — | Untested. **B là graph-JEPA 67 landmark, gần nhất với ý tưởng của anh.** |
| SignCoord (p9) | `signcoord_ssl` | smoke only | — | Untested |

### 2.2 Mười bài học ràng buộc thiết kế proposal này

1. **Readout và protocol quyết định kết luận.**
   - Cùng weight SignRep: plain mean 30.39, white512 37.55, R0 49.50, R1 53.88 (LOSO-val).
   - ⇒ Protocol đánh giá phải **khóa trước** và báo cáo nhiều readout. Literature cũng nói vậy: linear vs attentive probe đổi xếp hạng MAE/JEPA tới 8–17 điểm.
2. **Danh tính signer chi phối mọi biểu diễn.**
   - Signer probe 0.92–0.999 với mọi model đã train.
   - 82% láng giềng tự tin nhất là cùng signer.
   - ⇒ Phải có cơ chế khử signer và đo signer-probe như negative metric.
3. **Collapse xảy ra thường xuyên:** spectral (rank 24), signrel_jepa (cos 0.999), ASLC002 (cos 0.945, rank 25). ⇒ Phải chọn cơ chế chống collapse đã được chứng minh trên skeleton và theo dõi rank/cos ở mọi run.
4. **Bug từng đảo ngược kết luận:**
   - XWP student copy teacher;
   - aggregator input-scale;
   - JHOG/JPRM OOM nhưng log vẫn ghi "DONE".
   - ⇒ Cần unit test *hidden-value perturbation* (obs_jepa đã có mẫu tốt) và kiểm tra run thật sự đã train.
5. **Cross-articulator forecasting không có tín hiệu (CIP 0/36), nhưng đó là *forecasting* có lịch sử của chính kênh.**
   - Bài toán *masked* (che cả quá khứ lẫn hiện tại của kênh) là khác.
   - Tôi đo lại trực tiếp ở §5 (D2): có tín hiệu, nhưng khiêm tốn.
6. **Prior chuyển động là target phụ tốt nhất (JMOT); non-manual làm hại retrieval (JNM).** D4 (§5) giải thích nguyên nhân: kênh mặt chưa chuẩn hoá là "vân tay signer".
7. **Latent + prior thắng prior thuần rất xa (JH3D vs SR3D).** Đây là bằng chứng nội bộ cho thiết kế "latent target + target tường minh/cục bộ".
8. **Đặc trưng thô của pose đã mạnh hơn một số model SSL đã train (SIGNREL A0).** ⇒ Bắt buộc có **baseline training-free** (B0) và baseline random-init.
9. **Không có baseline pose-SSL nào trong repo** (không ST-GCN, MAMP, S-JEPA, SignBERT, MASA). ⇒ Proposal phải tự dựng các baseline matched.
10. **Test split đã bị dùng.**
    - ASL Citizen test và WLASL test mỗi cái dùng một lần (EX05), MS-ASL test một lần (EX07); ASL Citizen test còn bị các dự án legacy dùng thêm.
    - ⇒ Mọi phát triển chỉ trên val. Final test dùng *một lần* theo protocol khóa. Thêm trục đánh giá mới là phonology probe (§8).

### 2.3 Tài nguyên dữ liệu thực tế (đã kiểm tra)

Tất cả dataset dùng **cùng một định dạng**: `keypoints` (T×1×133×2, toạ độ chuẩn hoá x/W, y/H) cùng `scores` (T×1×133, không hiệu chỉnh, khoảng 0.5–10).

| Dataset | Số clip keypoint | Ghi chú |
|---|---|---|
| ASL Citizen | 83,399 (train 40,154 / val 10,304 / test 32,941) | signer-disjoint, 2,731 gloss, có mã **ASL-LEX** |
| WLASL2000 | 21,095 | trùng nguồn YouTube với MS-ASL, cần dedup |
| MS-ASL | 18,855 | |
| How2Sign | train 31,047 câu | ASL liên tục |
| CSL-Daily | 20,654 | CSL |
| Phoenix14T | ~8,257 | DGS |

Thêm vào đó, ASL-LEX 2.0 (`/home/haipd/Handshape-GNN/data/external/asllex_official_metadata_v1/sign_props.json`, 2,723 mục) có Handshape, SelectedFingers, Flexion, Spread, ThumbPosition, MajorLocation, MinorLocation, Movement, SignType, Contact, UlnarRotation… Nhờ đó có thể làm **phonology probe** ngay trên ASL Citizen.

Ràng buộc compute:
- 1 GPU RTX 5880 (49 GB) dùng chung; tối đa 2 tiến trình, mỗi lúc một job GPU.
- Disk còn khoảng 78 GB.
- Skeleton rẻ hơn RGB vài bậc. Toàn bộ khoảng 180k clip cache float16 tốn khoảng 6–9 GB.

---

## 3. Literature review có phản biện

Bốn báo cáo đầy đủ nằm trong `literature/`:
- L1: JEPA và các paradigm;
- L2: masking;
- L3: skeleton SSL;
- L4: sign SSL.

Số liệu dưới đây là các số đã được grep từ raw text bài báo (tag [V] trong báo cáo). Số chỉ có từ tóm tắt được đánh dấu *(tóm tắt)*.

### 3.1 JEPA và các paradigm hiện nay: điều thực sự quan trọng

| Phát hiện | Bằng chứng |
|---|---|
| **Hình học mask là siêu tham số chính của JEPA** | I-JEPA (1% IN, linear): multi-block 54.2 vs random 17.6 vs single block 20.2. Chỉ 1 target block: 9.0. Context scale 0.40 → 31.2. V-JEPA: multi-block 72.9 vs random-tube 51.5 (K400 attentive). |
| **Target lấy ở *output* teacher (contextual)** | I-JEPA: 67.3 vs 56.1 khi mask ở input. S-JEPA: mask input teacher → 78.1 vs 85.3. |
| **Latent target thắng pixel target lớn trên đầu vào nhiễu, chiều cao** | I-JEPA 66.9 vs 40.7. V-JEPA +5.1 (K400 frozen) nhưng ≈ 0 khi fine-tune. MaskFeat: HOG (82.2) ≈ DINO feature (82.5). **Một descriptor thủ công tốt lấy được phần lớn lợi ích.** |
| **Thiên kiến ngầm của JEPA** (Littwin et al., NeurIPS'24) | Deep-linear JEPA ưu tiên đặc trưng có *hệ số hồi quy* ρ cao (dự đoán được). MAE ưu tiên đặc trưng có phương sai cao. Thiên kiến tăng theo độ sâu. ⇒ **Phân phối mask quyết định đặc trưng nào "dự đoán được", tức đặc trưng nào được học.** |
| **JEPA bám "slow features"** (Sobal et al. 2022) | Nhiễu cố định theo thời gian bị học thay cho tín hiệu. Trong sign language, danh tính signer chính là nhiễu cố định trong clip. |
| **Chống collapse là lựa chọn thiết kế có bằng chứng** | EMA + stop-grad (BYOL → I/V-JEPA). Teacher đóng băng (SALT). Regulariser phân phối: VICReg, **SIGReg** (LeJEPA: λ = 0.05, BN-MLP projector, 256–1024 slice, Epps–Pulley). Prototype/Sinkhorn (DINO, iBOT, SIGMA). |
| **SIGReg cho dự đoán token** mới có bằng chứng quy mô nhỏ | PointTube-JEPA (point cloud, 2608.24093): không EMA, SIGReg λ = 0.01 → 94.08. Dùng EMA thay: 88.50. Stop-grad target: 79.44. Laya (EEG): λ < 0.02 bị collapse. LeWM: "SIGReg yếu khi dữ liệu ít đa dạng, chiều thấp". |
| **Probe đổi kết luận** | V-JEPA: linear → attentive +17. Efficient probing (ICLR'26): MAE-B +7.9 nhưng DINO +0.5. ⇒ Báo cáo cả linear lẫn attentive. |
| **Feature dense suy biến âm thầm** | V-JEPA 2.1 (context loss + deep supervision), DINOv3 (Gram anchoring), Human-JEPA ("silent collapse"; block mask hoạt động như "copy task" và tốn 4.8 điểm trên NTU-120). |
| **data2vec** | Target là trung bình top-K layer đã chuẩn hoá. Instance-norm cho tín hiệu có target kề nhau tương quan mạnh: **skeleton đúng là loại này**. Multi-mask M = 4–8 trên mỗi teacher pass (data2vec 2.0). |

### 3.2 Masking: ngẫu nhiên hay có cấu trúc

| Phát hiện | Bằng chứng |
|---|---|
| **Che trọn bộ phận từ đầu thường hại; đi qua curriculum hoặc trộn thì có lợi** | SemMAE: whole-part 52.9 vs random 66.8 vs curriculum nội-part → whole-part 68.7. EPM: part 75.71 vs grid 78.63 vs schedule 79.89. HPM: dễ→khó +0.46, khó→dễ −0.78. |
| **Trong sign: che *body-only* còn tệ hơn không pretrain** | BEST (AAAI'23), MSASL100: không pretrain 73.44; hand-only 76.75; **body-only 68.82**; trộn cả ba 80.98. |
| **Target quan trọng hơn mask trên skeleton** | MAMP: motion-aware masking +1.2, nhưng đổi target từ joint sang motion **+10.1**. |
| **Latent target làm giảm lợi ích của mask "thông minh", nhưng JEPA vẫn cần mask *liền khối*** | BEiT +4.3 (pixel) vs +1.7 (token). MaskAlign 85.3 vs 85.1. Nhưng I-JEPA/V-JEPA (ở trên) bắt buộc multi-block. |
| **Rò rỉ là failure mode số một** | Point-MAE/Point2Vec: positional embedding của patch bị che lộ hình dạng (che đi được +0.7). Point-M2AE: mask không nhất quán giữa các scale tốn −4.5. VideoMAE: frame masking 61.5 vs tube 69.6. |
| **Lý thuyết** (Kong et al., CVPR'23) | Mask khiến model học latent *tối thiểu được chia sẻ* giữa phần che và phần thấy. Mask quá nhỏ hoặc quá lớn đều chỉ cho đặc trưng mức thấp. Với skeleton, latent chia sẻ dễ bị **coupling động học** chiếm (cổ tay ∈ cả cánh tay lẫn bàn tay; tay không thuận gương tay thuận trong ký hiệu đối xứng). |
| **Frozen nhạy với mask hơn fine-tune** | MAE: LP dao động khoảng 20 điểm theo mask ratio, FT phẳng. AutoMAE: LP +3.0, FT ±0.1. ⇒ Claim nên đặt ở frozen/retrieval. |

**Kỳ vọng thực tế** (L2): mask có cấu trúc mang lại **+1–3 điểm** frozen so với random-block JEPA. Thiết kế target có thể còn quan trọng hơn.

### 3.3 Skeleton SSL: trạng thái nghệ thuật (NTU, linear, joint-only)

| Method | Venue | Mask | Target | Chống collapse | NTU60 xsub / xview | NTU120 xsub / xset |
|---|---|---|---|---|---|---|
| MAMP | ICCV'23 | motion-aware 90% | Δx (motion) | — | 84.9 / 89.1 | 78.6 / 79.1 |
| **S-JEPA** | ECCV'24 | motion-aware 90% (MAMP) | output teacher EMA, **softmax-CE + centering** | EMA (0.9999 → 1) | 85.3 / 89.8 | 79.6 / 79.9 |
| GFP | ICCV'25 | motion-aware | feature học phân cấp | VICReg | 85.9 / 92.0 | 79.1 / 80.3 |
| AMR | CVPR'26 | motion-aware | **toạ độ chuẩn hoá** (focal) | — | 87.4 / 92.3 | 81.1 / 81.9 |
| **Path-JEPA** | ECCV'26 | seed theo motion → **cây con giải phẫu**, lan truyền phụ thuộc | latent EMA của token *path-signature* | EMA + centering | 87.6 / 92.9 *(trang tác giả)* | 81.5 / 81.8 |
| **SLiM** | arXiv 2603.10648 (trang dự án ghi NeurIPS'26) | **Semantic Tube Masking**, r ~ U(0.5, 0.9) | prototype iBOT + DINO-CLS global-local | EMA + Sinkhorn + KoLeo | **87.9 / 93.2** | 81.2 / 83.6 |

Các điểm cốt lõi rút ra:

- **S-JEPA (internals):**
  - Không có code; repo GitHub rỗng.
  - 750 token (25 joint × 30 đoạn 4 frame). Encoder 8 × 256, predictor 5 × 256. Teacher EMA nhìn chuỗi 3-D sạch.
  - Bỏ EMA → **collapse 1.6%**. MSE thay CE → 83.7 / 76.2. Bỏ centering → 84.5.
  - 1200 epoch, khoảng 4.5× thời gian của MAMP.
  - Không có "2-D → 3-D depth trick" trong bản camera-ready; đó là MotionBERT.
- **SLiM:**
  - STM che một vùng giải phẫu liên thông qua các frame liên tiếp, với "constant volume": vùng nhỏ thì che lâu hơn.
  - Thắng joint masking **+3.5** (87.7 vs 84.2).
  - Ratio U(0.5–0.9) thắng 90% cố định (+0.9) và thắng 0–50% (+3.2).
  - **Phân tích target có đối chứng (bảng 10):**
    - raw coordinates: 86.96 (avg);
    - feature-MSE không predictor: 86.92, *loss → 3.3e-5, effective rank → 1.12*;
    - feature-MSE có predictor: 87.59;
    - **prototype CE: 90.17**.
  - "Bone-aware scaling" (scale độ dài xương U(0.85, 1.15), giữ hướng) có ích: 87.7 vs 86.4 với scaling thông thường.
  - Joint bị thiếu cần route qua mask token. Zero-fill làm sụp 49.6%.
- **Path-JEPA:**
  - Token là log-signature của đường đi (joint, cạnh, chuỗi động học), mang thông tin displacement và signed area, bền với thay đổi frame rate.
  - Mask: chọn seed theo motion, mở rộng ra cả cây con, rồi **lan truyền tới mọi token phụ thuộc** joint bị che. Đây đúng là nguyên tắc chống rò rỉ cần có.
- **Kết luận phản biện:**
  - Trên skeleton, JEPA ≈ MAE được thiết kế tốt (S-JEPA 85.3 vs MAMP 84.9; AMR chỉ dùng toạ độ chuẩn hoá đã đạt 87.4).
  - Latent target chỉ thắng khi hội đủ ba điều kiện: (i) lấy từ output teacher, (ii) có chống collapse mạnh (EMA + CE/prototype), (iii) mask khó và có cấu trúc.

### 3.4 Sign language SSL và ISLR trên pose

| Nhóm | Phương pháp (venue) | Ý chính | Số liệu then chốt |
|---|---|---|---|
| Pose masked modelling | SignBERT (ICCV'21), SignBERT+ (TPAMI'23) | token tay, mask joint/frame/clip, decoder MANO | WLASL2000 FT 48.85. **Linear chỉ 24.50** (STC bảng VII) |
| | BEST (AAAI'23) | token bộ ba (LH, RH, body) bằng d-VAE | FT 46.25; linear 21.19 |
| | MASA (TCSVT'24) | GCN tay/thân riêng, motion residual, mask 90% | FT 49.06. **Frozen retrieval ASL Citizen 27.96** (trích từ bảng SignSeek) |
| | SignMAE (ICPR'26) | heatmap keypoint, chỉ tái tạo vùng tay | random 24.36 → tube 26.48 → ST-mask 40.31 → **chỉ tái tạo tay 47.08** |
| Contrastive | STC (TIP) | nhánh tay và thân riêng | linear WLASL 32.35; tách tay/thân +4.9 |
| Predictive coding | OpenHands (ACL'22) | DPC trên pose | **DPC 94.7 > masked 91.3 > contrastive 90.8** (INCLUDE). Bằng chứng sớm nhất cho mục tiêu *dự đoán latent* trên sign pose. |
| Multi-stream masked cluster | **SHuBERT** (ACL'25) | 4 stream: mặt và 2 tay là *crop ảnh* qua DINOv2; body 14-d. Che span 3 frame **từng stream riêng**; target là k-means 256 cụm/stream, offline | ASL Citizen LoRA R@1 0.65. Phonology probe 85.02 nhưng **fine-tune, không lexically disjoint** |
| Temporal DINO trên crop | SignDino (9/2026, chưa review) | DINOv3 frozen + DINO/iBOT theo thời gian | Tự báo 0.63 frozen-linear ASL Citizen. Có cờ độ tin cậy (gán sai output cho SHuBERT) |
| **Gần nhất về ý tưởng** | **SignSeek** (2609.03695, 9/2026) | 4 articulator (body, face, LH, RH); **MAP = masked articulator prediction** dạng latent, stop-grad, cosine; chọn articulator theo saliency | **Frozen ASL Citizen 57.66**. Nhưng **dùng nhãn gloss** (contrastive giám sát) trên 266k mẫu có nhãn, và Sem-Lex trùng từ vựng ASL Citizen. Saliency mask 34.75 > luôn dominant hand 31.65 > uniform 29.66 (WLASL) |
| RGB + prior pose | **SignRep** (ICCV'25) | Hiera-B, mask ngẫu nhiên 80% theo khối 32 px (không giải phẫu). **Một vector toàn cục** được unroll ra 16 frame để regress prior pose 3-D. Bone length chuẩn hoá theo median dataset để tránh lộ danh tính. Adversarial style loss | **Frozen ASL Citizen 49.95** (activity-weighted) / 37.47 (avg). Raw hand-angle baseline 8.81 |
| Pose giám sát / ngôn ngữ | DSTA-SLR, Uni-Sign, LA-Sign | | WLASL2000 pose: 53.68 (giám sát thuần), 63.13 / 64.73 (pretrain có văn bản) |

Bốn kết luận quan trọng cho proposal:
1. **Chưa có JEPA kiểu latent-distillation (EMA/prototype) label-free nào được công bố cho sign skeleton.** Đã tìm theo tiêu đề và abstract arXiv tới 2026-10-11. Gần nhất có hai công trình:
   - OpenHands DPC (2021): *predictive coding* nhưng dùng loss contrastive.
   - SignSeek MAP: dự đoán latent articulator nhưng huấn luyện **cùng nhãn gloss**.
2. **Khoảng trống định lượng rõ:** frozen ASL Citizen của skeleton SSL label-free ≤ 27.96 (MASA), RGB SignRep 49.95, pose *có nhãn* SignSeek 57.66.
3. **Tay quyết định** (body only 21.37 vs body + hands 48.54, NLA-SLR). Tách nhánh tay, chuẩn hoá theo hệ toạ độ tay, và chia sẻ encoder L/R sau khi mirror đều thắng.
4. **Keypoint tay là mắt xích yếu nhất:**
   - Hand AP của RTMW khoảng 58–66, so với body 76 và face 88.
   - MediaPipe mất trọn bàn tay ở 20–24% frame trên Phoenix.
   - Đổi bộ trích xuất pose làm số liệu benchmark dịch chuyển 6–15 điểm.

### 3.5 Định vị: khoảng trống mà proposal này nhắm vào

| Tiêu chí | S-JEPA | SLiM | Path-JEPA | SignBERT+/BEST/MASA | SHuBERT | SignSeek | obs_jepa B (repo) | **PhonoJEPA** |
|---|---|---|---|---|---|---|---|---|
| Miền | action | action | action | sign | sign | sign | sign | **sign** |
| Label-free | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (gloss) | ✓ | **✓** |
| Đơn vị che | joint (motion) | vùng giải phẫu × thời gian | cây con giải phẫu | joint / frame / clip | stream × 3 frame | cả articulator | event phần / chuỗi ngón | **kênh âm vị học × span (thang độ khó)** |
| Đóng phụ thuộc / chống rò rỉ | — | — | ✓ (lan truyền) | — | — | — | ✓ (dependency-safe) | **✓ (luật đóng kênh + chuẩn hoá chỉ từ anchor)** |
| Target của phần bị che | contextual | contextual (prototype) | contextual | toạ độ / token | k-means offline | contextual (cosine) | contextual + anchor | **contextual + intrinsic (prototype theo nhóm kênh)** |
| Khử signer label-free | — | (bone scaling để tăng đa dạng view) | — | — | — | (contrastive có nhãn) | — | **✓ (augment bất đối xứng, neutral-face, đo signer-probe)** |
| Đánh giá âm vị học lexically disjoint | — | — | — | — | ✗ (FT, lexical overlap) | — | dự kiến | **✓ (frozen, lexically disjoint, + NMI prototype)** |

---

## 4. Phản biện ý tưởng gốc: "che joint/bộ phận → dự đoán representation phần bị che"

### 4.1 Ý tưởng gốc đã có ở đâu

Có bốn hiện thân của ý tưởng này:

| Hiện thân | Mô tả |
|---|---|
| **S-JEPA** | Che joint theo motion, dự đoán latent EMA (action). |
| **SLiM** | Che vùng giải phẫu liên thông theo thời gian. Đây chính là "structured anatomical masking" (action). |
| **SignSeek** | Che *một articulator nổi bật* và dự đoán latent. Đây là sign language, nhưng có giám sát. |
| **obs_jepa B** (repo, chưa train) | Graph-JEPA 67 landmark, mask event 50/30/20 theo chuỗi ngón, gap cùng bộ phận và block chung. |

Vì vậy, đóng góp *không thể* là "lần đầu che bộ phận cơ thể và dự đoán latent". Đóng góp phải đến từ việc giải quyết các vấn đề *đặc thù sign language* mà các hiện thân trên chưa giải quyết. Các vấn đề đó được liệt kê dưới đây.

### 4.2 Bảy failure mode, có bằng chứng

| # | Failure mode | Cơ chế | Bằng chứng |
|---|---|---|---|
| F1 | **Shortcut nội suy thời gian** | Che một bộ phận ở frame *t* nhưng để lộ *t±1*. | D2: handshape tại *t* từ trung bình (*t−1*, *t+1*) có R² = **0.69**; toạ độ tay thô **0.87**. VideoMAE: frame masking 61.5 vs tube 69.6. |
| F2 | **Rò rỉ động học / phụ thuộc** | Cổ tay của thân (COCO 9/10) trùng gốc bàn tay. Khuỷu tay cùng hướng cẳng tay ⇒ suy ra vị trí cổ tay. Hướng lòng bàn tay (từ MCP) ⇒ một phần hình dạng. | D3: ORI → SHAPE cùng frame, R² 0.11 (ridge) / 0.21 (kNN). Point-MAE / Point-M2AE: rò rỉ positional (+0.7 / −4.5). Path-JEPA phải lan truyền mask sang token phụ thuộc. |
| F3 | **Bài toán gần bất định khi che trọn articulator** | Biết vị trí và chuyển động chưa chắc biết handshape: cặp tối thiểu (minimal pairs). | D2: oracle từ vựng (trung bình handshape cùng gloss, khác signer) chỉ đạt R² **0.25**. CIP 0/36. SemMAE whole-part 52.9 vs random 66.8. BEST body-only < không pretrain. |
| F4 | **Thiên kiến JEPA nén mất chi tiết khó đoán** | JEPA học đặc trưng có ρ cao trước (Littwin). Nếu chỉ có mask khó, chi tiết handshape tinh (ρ thấp từ ngữ cảnh) bị teacher bỏ qua. | Lý thuyết Littwin. EX15: probe handshape R² thấp ở mọi arm (0.03–0.18). |
| F5 | **Danh tính signer là đặc trưng dễ đoán nhất** | Target contextual "phát tán" thông tin toàn cục của clip (hình thái, phong cách) vào *mọi* token. Từ ngữ cảnh nào cũng đoán được nó ⇒ JEPA học nó đầu tiên (slow feature). | Signer-probe 0.92–0.999 (repo). D1: chỉ hình thái tĩnh đã nhận ra signer với balanced acc 0.82. D4: kênh miệng thô 0.83 signer / 0.24% gloss. Pose nhận diện người ký 80.18% trên AUTSL (literature). |
| F6 | **Collapse khi hồi quy latent trên đầu vào chiều thấp** | Token skeleton ít chiều, tương quan mạnh, target kề nhau gần giống nhau. | SLiM: feature-MSE có rank 1.12. S-JEPA: bỏ EMA → 1.6%. T-JEPA cần token [REG]. Repo: signrel_jepa cos 0.998. |
| F7 | **Khuyết thiếu keypoint bị nhầm với mask** | Detector đánh rơi tay ở 20–24% frame. Nếu "thiếu" và "bị che" dùng chung token hay zero-fill, model học cách điền nhiễu. | SLiM: zero-fill sụp còn 49.6% so với 87.7. L2: tách [MASK] khỏi [ABSENT]. |

### 4.3 Từ phản biện sang nguyên tắc thiết kế

| Nguyên tắc | Xử lý failure mode | Hiện thực trong PhonoJEPA |
|---|---|---|
| **P1** Che theo *tube* thời gian. Span ≥ 4 bin (8 frame), không có mask 1-bin. | F1 | §6.6 (mọi họ) |
| **P2** Đơn vị che = *kênh âm vị học*, với **luật đóng phụ thuộc**: khuỷu tay gộp vào LOC. Ngón bị che chỉ gồm PIP/DIP/TIP (MCP thuộc lòng bàn tay). LOC không chứa quan hệ với mặt. Predictor query chỉ mang (loại kênh, thời gian), không mang toạ độ. | F2 | §6.4, §6.6 |
| **P3** **Thang độ khó** đo bằng R² (§5) và curriculum dễ→khó. Không dùng riêng mask "toàn articulator". | F3, F4 | §6.6 |
| **P4** Có **target intrinsic** (biểu diễn cục bộ của chính kênh) bên cạnh target contextual. | F4, F5 | §6.7 |
| **P5** Chuẩn hoá **chỉ từ anchor không bao giờ bị che** (vai) và theo từng frame. Mặt dùng neutral-face. **Augment hình thái và tốc độ bất đối xứng** giữa student và teacher. | F5 | §6.3, §6.10 |
| **P6** Loss dạng **softmax-CE có centering/Sinkhorn** (đã chứng minh trên skeleton), cộng EMA. Pilot so với L1 + SIGReg. Theo dõi rank và cos ở mọi run. | F6 | §6.8 |
| **P7** Tách hai token **[MASK] ≠ [ABSENT]**. Target có trọng số theo độ tin cậy phía teacher. | F7 | §6.5, §6.8 |

---

## 5. Bằng chứng pilot D1–D5 (training-free, CPU)

Tất cả nằm trong `pilot_diagnostics/` (script, JSON, README). Điều kiện chung:
- Dữ liệu: ASL Citizen val, 10,304 clip, 5 signer có trên 50 clip.
- Gallery: 14k clip train (35 signer khác), trừ D5 dùng toàn bộ train.
- Mô tả: resample 16 bước trên các frame có tay thuận hợp lệ, sau khi chuẩn hoá theo vai và mirror tay thuận.
- **Không** có model SSL nào được train ở đây.

### D1 — Rò rỉ signer và thông tin gloss của các cách biểu diễn skeleton đơn giản

| Biểu diễn | Signer probe (balanced acc, 5 lớp, chance ≈ 0.20) ↓ | Gloss 1-NN cross-signer ↑ |
|---|---|---|
| Hình thái tĩnh (median độ dài xương) | 0.819 | 1.70% |
| Toạ độ chuẩn hoá vai (tay thuận + tay không thuận + cánh tay) | 0.796 | 7.33% |
| **Phân rã kênh** (LOC, ORI, SHAPE hệ lòng bàn tay) × 2 tay | 0.641 | 9.68% |
| **Phân rã kênh + chuẩn hoá độ dài từng xương** (chia cho p90 của chính xương đó trong clip) | **0.611** | **12.34%** |

**Diễn giải:** chỉ đổi *cách biểu diễn* (chưa học gì) đã tăng thông tin gloss **+68% tương đối** (7.33 → 12.34) và giảm rò rỉ signer **−18.5 điểm** (0.796 → 0.611). Đây là căn cứ cho token kênh âm vị học có chuẩn hoá xương (§6.4).

### D2 — Thang độ dự đoán được của handshape tay thuận

Mục tiêu dự đoán: 20 joint trong hệ lòng bàn tay. R² được tính sau winsorize ±5σ.

| Bậc | Ngữ cảnh → mục tiêu | R² |
|---|---|---|
| Rất dễ (shortcut) | toạ độ tay thô tại *t* ← trung bình *t ± g* với g = 1 / 2 / 4 / 8 frame | 0.87 / 0.82 / 0.71 / 0.55 |
| Dễ | handshape (hệ lòng bàn tay) tại *t* ← *t ± g* với g = 1 / 2 / 4 / 8 | 0.69 / 0.60 / 0.47 / 0.30 |
| Trung bình | 3 ngón (giữa, áp út, út) ← ngón cái + trỏ + lòng bàn tay, cùng frame (kNN / ridge) | 0.56 / 0.40 |
| Khó, chéo kênh | handshape cấp clip ← quỹ đạo LOC | 0.05 |
| Khó, chéo kênh | ← LOC + ORI | 0.31 |
| Khó, chéo kênh | ← LOC + ORI + tay không thuận | 0.26 (kNN bị nhiều chiều làm loãng) |
| Trần từ vựng | ← trung bình handshape cùng gloss của *signer khác* (oracle) | 0.25 |
| Không có tín hiệu | ← trung bình handshape các clip khác của *cùng signer* | 0.005 |

**Diễn giải:**
1. Mask ngắn (≤ 2 frame) gần như là bài toán nội suy. ⇒ span tối thiểu 4 bin (P1).
2. Có một thang độ khó tự nhiên. Mask nội-tay (0.4–0.56) và chéo kênh (khoảng 0.3) nằm trong vùng "học được nhưng không tầm thường".
3. Oracle từ vựng chỉ đạt 0.25. Mô tả handshape 2-D chứa nhiều biến thiên phát âm và nhiễu, nên một *target latent* có thể bỏ nhiễu sẽ có lợi. Đồng thời che trọn articulator là bài toán có trần thấp (F3). ⇒ Không dùng riêng họ mask này (P3).
4. Handshape gần như **không** mang danh tính signer (0.005), trái hẳn với kênh mặt (D4). ⇒ Target intrinsic của kênh tay ít bị nhiễm signer.

> Lưu ý: lần chạy đầu (`D1_D2_results_v1_nonrobust.json`) **không hợp lệ**. Frame có lòng bàn tay suy biến (tay nhìn từ cạnh) làm toạ độ nổ, nên R² âm kể cả với oracle. Bản hiện tại loại frame có độ dài lòng bàn tay dưới 0.5× median của clip. Đây chính là lý do §6.3 có bước kiểm tra độ hợp lệ của lòng bàn tay.

### D3 — Rò rỉ ORI → SHAPE: hai cách chuẩn hoá bàn tay

| Chuẩn hoá | R² SHAPE ← ORI (ridge / kNN) ↓ | Gloss 1-NN chỉ dùng tay ↑ |
|---|---|---|
| Similarity (xoay theo trục cổ tay → MCP giữa, scale theo chiều dài lòng bàn tay) | 0.111 / 0.205 | **3.58%** |
| Affine (khớp lòng bàn tay vào template, khử cả co rút phối cảnh) | 0.066 / 0.154 | 1.16% |

**Diễn giải:** affine giảm rò rỉ một chút nhưng **mất 2/3 thông tin**. ⇒ Chọn similarity và chấp nhận rò rỉ vừa phải (R² ≤ 0.2, vẫn thấp hơn bậc "trung bình"). Không ép phân rã "sạch tuyệt đối" vì ở 2-D, hướng và hình dạng vốn đan xen qua phép chiếu.

### D4 — Thông tin gloss và signer theo từng kênh

Tất cả là mô tả training-free, resample 16 bước.

| Kênh | Signer ↓ | Gloss 1-NN ↑ |
|---|---|---|
| D-SHAPE | 0.447 | **4.17%** |
| N-SHAPE | 0.407 | 1.95% |
| D-LOC | 0.404 | 1.20% |
| D-ORI | 0.381 | 1.24% |
| D-LOC vận tốc | 0.297 | 0.88% |
| N-LOC / N-ORI | 0.31 / 0.32 | 0.79% / 0.78% |
| **MOUTH** (toạ độ thô so với mũi) | **0.826** | **0.24%** |

**Diễn giải:**
1. Từng kênh riêng lẻ yếu, nhưng ghép lại đạt 9.7–12.3%: các kênh **bổ sung** cho nhau. Đây là điều kiện cần để dự đoán chéo kênh có ý nghĩa.
2. Kênh miệng chưa chuẩn hoá là "vân tay signer". Điều này giải thích vì sao target non-manual làm hại retrieval trong EX15 (JNM 1.96). ⇒ Mặt **phải** dùng neutral-face, scale theo khoảng cách hai mắt, có augment hình dạng mặt, và có trọng số loss thấp hơn (§6.4, §6.8).

### D5 — Baseline training-free theo giao thức dictionary retrieval chính thức

Giao thức: gallery là toàn bộ 40,154 clip train ASL Citizen, query là val, điểm = class-max cosine.

| Mô tả training-free (vector 16 bước, z-score theo train, cosine) | R@1 | R@5 |
|---|---|---|
| Toạ độ chuẩn hoá vai | 11.49 | 23.95 |
| Phân rã kênh | 14.34 | 31.55 |
| **Phân rã kênh + chuẩn hoá xương** (= B0 trong §9) | **16.84** | **36.52** |
| *Tham chiếu, chỉ để định hướng, **khác split**:* raw hand-angle của SignRep (test) | 8.81 | — |
| *Tham chiếu:* MASA (test) / SignRep avg / SignRep weighted (test) | 27.96 / 37.47 / 49.95 | — |
| *Tham chiếu trong repo:* SignRep đã phát hành, cùng protocol, **val**, readout R0 | 59.16 | — |

**Diễn giải:**
1. Phân rã kênh và chuẩn hoá xương, chưa học gì, đã đạt **16.84 R@1 trên val**. Đây là **mốc sàn bắt buộc** (B0). Một model SSL không vượt rõ mốc này thì không học được gì có ích. Bài học từ SIGNREL A0: model train còn thua đặc trưng thô.
2. Val dễ hơn test khá nhiều: SignRep 59.16 trên val so với 49.02 trên test. Vì vậy so sánh với số trên test của literature chỉ để định hướng. Mọi so sánh chính thức đều làm trên cùng split và cùng readout (§8).

---

## 6. Phương pháp đề xuất: PhonoJEPA

### 6.1 Luận điểm trung tâm

Trong ngôn ngữ ký hiệu, các tham số âm vị học (handshape, orientation, location/movement, non-manual) được **từ vựng ràng buộc lẫn nhau**. Một ký hiệu là một tổ hợp kênh có cấu trúc chứ không phải chuyển động tự do. Một biểu diễn tốt phải "biết" các ràng buộc này: *cho trước một số kênh, những kênh còn lại có thể là gì*.

PhonoJEPA buộc encoder học đúng tri thức đó bằng ba bước:
1. Biểu diễn skeleton thành **các kênh âm vị học** đã khử hình thái signer.
2. **Che kênh** ở nhiều mức khó đã được đo trước, từ nội suy cục bộ tới suy luận chéo kênh dài hạn.
3. Dự đoán **biểu diễn của kênh bị che** ở hai cấp:
   - *intrinsic*: chính kênh đó, cục bộ trong thời gian, dưới dạng phân phối trên các "đơn vị âm vị" học online;
   - *contextual*: biểu diễn toàn cục của teacher tại vị trí đó.

Phần còn lại của thiết kế chỉ có hai nhiệm vụ: (i) chặn các lối tắt (nội suy, rò rỉ phụ thuộc, danh tính signer) và (ii) chống collapse.

### 6.2 Sơ đồ tổng thể

```
 keypoints 2-D (T×133×2) + scores
          │  §6.3 tiền xử lý: pixel-aspect, 25 fps, mirror tay thuận, anchor = vai + 4 khoé mắt
          ▼
 9 kênh × 32 bin (bin = 2 frame; cửa sổ 64 frame = 2.56 s)
   D-SHAPE  D-ORI  D-LOC   N-SHAPE  N-ORI  N-LOC   F-UPPER  F-MOUTH  F-HEAD
          │
   ┌──────┴─────────────────────── augment bất đối xứng (§6.10) ───────────────────────┐
   │ Teacher T1, T2 (EMA, nhìn đủ)                    Student: near, far (bị che, §6.6) │
   │   tokenizer → E_c (cục bộ kênh) → u^T  ── intrinsic target                         │
   │                       → E_f (toàn cục) → h^T  ── contextual target                 │
   │                                                  tokenizer → E_c → E_f (token thấy) │
   │                                                      → Predictor (query = loại kênh │
   │                                                        + thời gian, KHÔNG toạ độ)   │
   │                                                      → p^int, p^ctx tại vị trí che  │
   │   CLS teacher(T1, T2) ◄── DINO-CLS ── CLS student(near, far, G1, G2)               │
   └───────────────────────────────────────────────────────────────────────────────────┘
 Loss = L_tok(ctx + int, prototype-CE, Sinkhorn) + λ_glob · L_glob + λ_KoLeo · KoLeo
```

### 6.3 Tiền xử lý leak-free

| Bước | Mô tả | Lý do |
|---|---|---|
| (a) Pixel aspect | x ← x·W, y ← y·H, với W, H từ metadata video (cache sẵn) | Toạ độ đang chuẩn hoá riêng theo W và H nên góc bị méo. obs_jepa cũng làm bước này. |
| (b) FPS và cửa sổ | Resample về 25 fps theo timestamp, nearest, không nội suy. Cửa sổ T = 64 frame, chia 32 bin × 2 frame. ISLR: đặt cửa sổ quanh vùng hoạt động (tay trên đường hông); clip ngắn được pad bằng [ABSENT]. Câu dài (How2Sign, CSL, Phoenix): lấy cửa sổ ngẫu nhiên. | Đồng nhất thời gian giữa các dataset |
| (c) Tập joint | body 9 {mũi, vai, khuỷu, cổ tay, hông × L/R}; tay 2×21; mặt 52 iBUG {lông mày 17–26, mũi 27–35, mắt 36–47, miệng 48–67, cằm 8} | Bỏ chân (BEST: che vùng ít thông tin có hại). Bỏ đường viền hàm (chủ yếu là danh tính). |
| (d) Độ tin cậy | c = clip((s − q10_g)/(q90_g − q10_g), 0, 1), với q10/q90 lấy theo nhóm (body, face, hand) trên corpus train. Quan sát được: o = s > τ_g. Tay hợp lệ ở frame t nếu mean score tay > τ_hand (khởi điểm 2.0, xem D-diagnostics) **và** chiều dài lòng bàn tay ‖h₉ − h₀‖ > κ·σ_t, với κ hiệu chỉnh từ dữ liệu, khởi điểm 0.10. | Score không hiệu chỉnh. Điều kiện lòng bàn tay chặn toạ độ nổ (bài học D2-v1). Ngưỡng theo σ_t (anchor) chứ không theo median clip ⇒ không rò rỉ. |
| (e) Tay thuận | side* = argmax_s Σ_t o_t ‖w^s_{t+1} − w^s_t‖, dùng cổ tay *thân*. Nếu là tay trái: lật x quanh đường giữa hai vai, hoán vị L/R cho body, tay và mặt (cặp đối xứng iBUG). Lưu cờ, không đưa vào model. | ASL-LEX mã hoá theo tay thuận/không thuận. Cho phép chia sẻ tokenizer giữa hai tay. |
| (f) Anchor | **Vai** (khung thân: c_t = trung điểm hai vai, σ_t = ‖vai_L − vai_R‖, median-filter ±2 frame) và **4 khoé mắt + sống mũi 27** (khung mặt: gốc tại trung điểm hai mắt, trục x theo đường mắt, scale = IOD). Anchor **không bao giờ bị che và không bao giờ là target**. | Mọi phép chuẩn hoá chỉ dùng anchor hoặc frame hiện tại ⇒ **không rò rỉ** thông tin của token bị che (P5). |
| (g) Chuẩn hoá xương (SHAPE) | Mỗi xương b của bàn tay: ℓ_b^ref = p90 chiều dài 2-D **trên các frame thấy được của view đó**. Student chỉ dùng frame ngữ cảnh của kênh; teacher dùng toàn bộ. | D1: thêm +2.6 điểm gloss và −3 điểm signer. Tính trên frame thấy được nên dependency-safe. |
| (h) Neutral face | f̄ = median hình dạng mặt (trong khung mặt) trên các frame thấy được của kênh mặt trong view đó. Đặc trưng mặt = f − f̄. | D4: miệng thô là vân tay signer. "Biểu cảm = mặt − mặt trung tính". |

> **Kiểm thử bắt buộc:** *hidden-value perturbation*. Ghi đè ngẫu nhiên toạ độ của mọi joint thuộc token bị che, cho mọi họ mask, thì đầu ra student (token, CLS) phải **bit-identical**. Mẫu kiểm thử này lấy từ `method/obs_jepa/tests/test_pose_dependency.py`.

### 6.4 Token kênh âm vị học (9 loại × 32 bin = 288 token)

Ký hiệu: h₀..h₂₀ là joint tay (sau mirror). Cổ tay w và khuỷu e lấy từ body. R_t là phép xoay đưa trục lòng bàn tay u_t = (h₉ − h₀)/‖·‖ về (0, 1), và ℓ_t = ‖h₉ − h₀‖. Mỗi token ghép 2 frame của một bin.

| Kênh (tay H ∈ {D, N}) | Đặc trưng mỗi frame | Chiều/bin | Phụ thuộc joint |
|---|---|---|---|
| **H-SHAPE** (cấu hình ngón) | 20 vector xương trong hệ lòng bàn tay, chuẩn hoá theo ℓ_b^ref: R_t(h_child − h_parent)/ℓ_b^ref. Thêm conf của 21 joint và cờ *joint-hidden* 21 (dùng cho họ R1). | 2×(40 + 21 + 21) = 164 | 21 joint tay |
| **H-ORI** (hướng lòng bàn tay) | u_t (2); dấu của (h₅ − h₀) × (h₁₇ − h₀), tức lòng hay mu bàn tay hướng camera (1); ‖h₅ − h₁₇‖/ℓ_t, tức tỷ lệ bề ngang và co rút phối cảnh (1); log(ℓ_t/σ_t), tức kích thước lòng bàn tay so với thân, proxy độ sâu (1); conf (4) | 2×9 = 18 | h₀, h₅, h₉, h₁₇ (lòng bàn tay) và anchor vai |
| **H-LOC** (vị trí và chuyển động, gồm cánh tay) | (w − c_t)/σ_t (2); (e − c_t)/σ_t (2); Δw trong bin (2); conf (2) | 2×8 = 16 | cổ tay, khuỷu thân và anchor |
| **F-UPPER** (lông mày và mí mắt) | độ lệch so với neutral của 10 điểm lông mày và 8 điểm mí mắt (bỏ 4 khoé mắt, là anchor); conf | 2×(36 + 18) = 108 | 18 điểm và anchor mặt |
| **F-MOUTH** | độ lệch so với neutral của 20 điểm miệng; conf | 2×(40 + 20) = 120 | 20 điểm miệng |
| **F-HEAD** (tư thế đầu) | vị trí gốc khung mặt so với thân (2), IOD/σ_t (1), roll (cos, sin) (2), yaw proxy (mũi 30 − gốc).x/IOD (1), pitch proxy (mũi 30 − gốc).y/IOD (1), cằm 8 so với gốc (2) | 2×9 = 18 | mũi 30, cằm 8 và anchor |

Thiết kế có chủ đích cho **đóng phụ thuộc** (P2):
- Khuỷu tay được gộp vào H-LOC, nên không thể suy ra cổ tay từ cánh tay khi LOC bị che (rò rỉ F2 bị đóng).
- MCP thuộc lòng bàn tay (ORI). Họ R1 chỉ che PIP/DIP/TIP. Che SHAPE thì ORI vẫn thấy, chấp nhận rò rỉ vừa phải (D3: R² ≤ 0.2).
- H-LOC **không** chứa quan hệ với mặt. Quan hệ tay–mặt (*location* ở mặt, cằm, má) phải được học qua attention giữa LOC và F-HEAD; nếu không, che mặt sẽ bị lộ qua LOC.
- 4 khoé mắt là anchor, nên chuẩn hoá mặt không lộ F-UPPER.

**Tokenizer:**
- Mỗi loại kênh κ có MLP_κ: in_κ → 256 → d (GELU, LayerNorm).
- **D và N dùng chung tokenizer**, phân biệt bằng side embedding (giống SignSeek: chia sẻ encoder tay sau mirror).
- Token = MLP_κ(x) + e_type[κ] + e_side.
- Token **[ABSENT]** (detector mất dấu, hoặc pad) là một embedding học riêng, khác **[MASK]** (P7).
- Token bị che bị **loại khỏi encoder student**. [MASK] chỉ xuất hiện trong predictor (I-JEPA/MAE).

### 6.5 Kiến trúc

| Mô-đun | Mô tả | S (screening) | M (full) |
|---|---|---|---|
| Tokenizer | 6 MLP (SHAPE, ORI, LOC dùng chung D/N; UPPER; MOUTH; HEAD) | in → 256 → d | in → 256 → d |
| **E_c** (intrinsic encoder) | Transformer; attention **chỉ trong cùng kênh** và cửa sổ thời gian ⎮b − b′⎮ ≤ 2 (5 bin = 0.4 s); RoPE thời gian; trọng số chung cho mọi kênh | 2 block, d = 256 | 2 block, d = 384 |
| **E_f** (contextual encoder) | Full attention trên mọi token thấy được + 1 [CLS]; RoPE thời gian; cộng lại e_type ở đầu vào | 6 block, d = 256, 4 head | 10 block, d = 384, 6 head |
| **Predictor P** | Narrow transformer. Đầu vào là Linear(d → d_p) của token ngữ cảnh, cộng [MASK] + e_type + e_side + RoPE(b) cho mỗi vị trí target. **Không có toạ độ** | 4 block, d_p = 128 | 6 block, d_p = 192 |
| Head contextual g_ctx | MLP 3 lớp (→ 2048 → 2048 → bottleneck 256, L2-norm) → K_ctx = 8192 prototype (weight-norm), kiểu iBOT | | |
| Head intrinsic g_int^grp | Một head mỗi nhóm kênh: **HS** (SHAPE, K = 256), **HO** (ORI, K = 64), **HL** (LOC, K = 128), **FA** (UPPER + MOUTH + HEAD, K = 128). D và N dùng chung head. | | |
| Head global g_cls | Head DINO cho [CLS], K = 8192 | | |
| Tổng tham số (encoder) | | ≈ 6 M | ≈ 22 M |

**Vì sao chọn kích thước K như vậy:** ASL-LEX 2.0 có 49–58 handshape và vài chục location. K được đặt dư khoảng 4× để prototype có thể tách các biến thể phát âm. **Teacher = EMA** của (tokenizer, E_c, E_f, các head). Đánh giá dùng **encoder teacher**: S-JEPA cho thấy teacher tốt hơn view-encoder, 85.3 vs 84.8.

### 6.6 Predictability-Ladder Channel Masking (PLCM)

**Đơn vị:** token (kênh κ, bin b). **Vùng hoạt động A:** các bin có cổ tay thuận trên đường hông, hoặc có tốc độ cổ tay > v₀. Điểm bắt đầu span được lấy 70% trong A và 30% đều.

| Họ | Bậc (R² đo ở §5) | Cái gì bị che | Tham số mặc định |
|---|---|---|---|
| **R1** Hoàn thiện nhóm ngón (nội-kênh, input một phần) | trung bình (0.4–0.56) | PIP/DIP/TIP của nhóm ngón G trong H-SHAPE, qua một span. Token vẫn vào encoder nhưng bị khuyết joint, và **vẫn là target**. | H ∈ {D: 0.75, N: 0.25}. G ~ đều trên {cái, trỏ, giữa, áp út, út, trỏ+giữa, giữa+áp út+út, cái+trỏ}. Span ~ U{4..8} bin. |
| **R2** Khối thời gian | dễ–trung bình (0.3–0.5 ở tâm khối) | Cả 9 kênh trong một khối liên tiếp | 1–2 khối, mỗi khối ~ U{4..8} bin (8–16 frame) |
| **R3** Span kênh âm vị (*phonological factor masking*) | khó (khoảng 0.05–0.3) | Một kênh κ qua một span dài; các kênh khác của *cùng tay* vẫn thấy | π_κ: D-SH .30, D-OR .15, D-LO .20, N-SH .10, N-OR .05, N-LO .05, F-MO .07, F-UP .05, F-HD .03. Span ~ U(0.3, 1.0)·⎮A⎮, tối thiểu 4 bin. |
| **R4** Span articulator (*anatomical masking* = ý tưởng gốc) | khó nhất (≤ trần từ vựng 0.25) | Mọi kênh của một articulator | {tay thuận .45, tay không thuận .15, **cả hai tay** .15, mặt .25}. Span ~ U(0.4, 1.0)·⎮A⎮. |
| R0 Ống token ngẫu nhiên | — | Ống (kênh, 2 bin) ngẫu nhiên | **Chỉ dùng làm đối chứng** (baseline B5) |

> **Token khuyết một phần (R1)** được xử lý như sau:
> - Token đi vào encoder student với các joint ẩn được đặt về 0 và gắn cờ *joint-hidden*.
> - Độ dài tham chiếu ℓ_b^ref của các xương thuộc ngón ẩn được tính **không dùng** các frame trong span ẩn.
> - Predictor nhận thêm một query [MASK] tại đúng vị trí (κ, b) đó, nên token vừa là ngữ cảnh (phần thấy được) vừa là target (toàn phần, phía teacher).
> - Test *hidden-value perturbation* phải bao gồm cả các joint ẩn này.

**Hai view mỗi clip** (theo V-JEPA: short-range + long-range; data2vec 2.0: nhiều mask trên một teacher pass):
- **near view** = R1 ∪ R2. Thêm proposal cho tới khi đạt tỷ lệ r_near ~ U(0.4, 0.7) của số token không-[ABSENT].
- **far view** = R4 (xác suất 0.4) hoặc R3, cộng thêm sự kiện R3 tới khi đạt r_far ~ U(0.3, 0.6).
- Thuật toán đạt budget theo kiểu SLiM: nhận proposal nếu nó thêm ít nhất một ô mới mà không vượt budget; dừng sau 10 proposal liên tiếp không thêm ô nào.
- Ở cả hai view, **context dropout** loại thêm 10% token còn thấy khỏi input student. Các token này không phải target; mục đích là mô phỏng detector mất dấu và tăng độ bền.

**Ràng buộc:**
1. Mỗi view giữ ít nhất 25% token không-[ABSENT] làm ngữ cảnh.
2. Chỉ token *quan sát được* ở phía teacher mới là target.
3. Không bao giờ che cả 9 kênh trên toàn clip.
4. Che "cả hai tay" (R4) được cố ý đưa vào để phá shortcut đối xứng: trong ký hiệu đối xứng, tay không thuận là ảnh gương của tay thuận (L2).

**Curriculum dễ→khó** (SemMAE, EPM, HPM đều cho thấy chiều ngược lại có hại):
- Trọng số loss far view: w_far(e) = 0.3 + 0.7·(e/E)².
- Cận trên độ dài span của R3/R4 tăng tuyến tính từ 0.5⎮A⎮ lên ⎮A⎮ trong 30% đầu quá trình train.

**Tỷ lệ che.** Tỷ lệ của PhonoJEPA (30–70%) thấp hơn MAMP/S-JEPA (90%) vì mỗi token kênh đã gộp nhiều joint và mask có cấu trúc. Literature: mask có cấu trúc có ratio tối ưu thấp hơn; SLiM U(0.5–0.9) thắng 0–50%. **Ratio là tham số quét ở P2** (§12).

```python
def sample_views(meta, epoch, E):            # meta: observed[k,b], active_bins A, ...
    near = Budget(rate=U(0.4, 0.7), cells=observed_cells(meta))
    while not near.saturated():
        near.propose(R2_block(A) if rand() < 0.5 else R1_fingers(A))      # R1: partial SHAPE tokens
    far = Budget(rate=U(0.3, 0.6), cells=observed_cells(meta))
    max_span = lerp(0.5, 1.0, min(1, epoch / (0.3 * E))) * len(A)
    far.propose(R4_articulator(A, max_span) if rand() < 0.4 else R3_channel(A, max_span))
    while not far.saturated():
        far.propose(R3_channel(A, max_span))
    for v in (near, far):
        v.context_dropout(p=0.10)            # removed from student input, never targets
        assert v.visible_fraction() >= 0.25
    return near, far
```

### 6.7 Target: "representation của phần bị che" là gì?

Với mỗi vị trí bị che (κ, b), teacher (EMA, nhìn view đầy đủ cùng trục thời gian) cung cấp hai target:

1. **Contextual (I-JEPA/data2vec/S-JEPA):**
   - Lấy trung bình 4 block cuối của E_f, mỗi block LayerNorm trước: h̄ = LN(¼ Σ_{l = L_f−3..L_f} LN(h^{(l)})).
   - Phân phối target: P^ctx = SK-softmax(g_ctx^T(h̄)/τ_t).
   - Ý nghĩa: "token này đóng vai trò gì trong *cả câu ký*".
2. **Intrinsic (mới; đúng nghĩa đen của ý tưởng gốc):**
   - Đầu ra E_c^T, chỉ nhìn *chính kênh đó* trong ±2 bin: u^T_{κ,b}.
   - Phân phối target: P^int = SK-softmax(g_int^{grp(κ)}(u^T)/τ_t).
   - Ý nghĩa: "kênh này *đang làm gì*", biểu diễn dưới dạng phân phối trên một **bộ đơn vị âm vị học học online**: prototype handshape, orientation, location, mặt.

Student: đầu ra predictor tại (κ, b) đi qua hai linear head H_ctx, H_int rồi qua g^S tương ứng, cho p^ctx và p^int (softmax với τ_s = 0.1).

**Vì sao cần target intrinsic** (luận cứ, sẽ kiểm chứng qua RQ2 và H2):
- (i) Đây là định nghĩa trực tiếp của "biểu diễn phần bị che".
- (ii) Target chỉ nhìn kênh cục bộ thì **không chứa thông tin toàn cục của clip** ngoài những gì có trong chính kênh. Kênh tay đã chuẩn hoá xương gần như không mang danh tính signer (D2: R² signer-prior 0.005). Do đó thành phần "slow feature" signer trong target co lại, và thiên kiến JEPA nghiêng về signer bị yếu đi (F5).
- (iii) Target nông giữ được chi tiết tinh như ngón và độ cong (chống F4). Target contextual đẩy sự tích hợp ngữ nghĩa. Bằng chứng từ repo: "latent + prior cục bộ" JH3D ≫ từng loại riêng.
- (iv) Prototype theo nhóm kênh **diễn giải và đo được**: NMI giữa phép gán prototype HS và nhãn Handshape của ASL-LEX, giữa HL và MajorLocation (§8). SHuBERT dùng k-means offline trên đặc trưng đầu vào; ở đây prototype được học online trên latent và phân rã theo tham số âm vị.

*(Tuỳ chọn, ablation)* **Target tường minh kiểu MAMP:** chuyển động của đặc trưng kênh x_{κ,b} − x_{κ,b−1}, chuẩn hoá theo kênh, head H_desc với smooth-L1, β_desc ∈ {0, 0.5}. Literature cho thấy motion target rất mạnh trên skeleton (MAMP +10.1), và hybrid an toàn hơn latent thuần (L1 R3).

### 6.8 Loss và chống collapse

**Loss token** cho view v (near hoặc far), với tập target M_v:

$$\mathcal{L}^{v}_{tok}=\frac{\sum_{(\kappa,b)\in M_v}\omega_{\kappa,b}\Big[\mathrm{CE}\big(P^{ctx}_{\kappa,b},p^{ctx}_{\kappa,b}\big)+\alpha_{int}\,\mathrm{CE}\big(P^{int}_{\kappa,b},p^{int}_{\kappa,b}\big)\Big]}{\sum_{(\kappa,b)\in M_v}\omega_{\kappa,b}},\qquad \mathcal{L}_{tok}=\mathcal{L}^{near}_{tok}+w_{far}(e)\,\mathcal{L}^{far}_{tok}$$

- ω_{κ,b} = r^T_{κ,b} · γ_{grp(κ)}.
  - r^T là độ tin cậy phía teacher: conf trung bình của các joint trong token; bằng 0 nếu [ABSENT].
  - γ: tay = 1.0, mặt = 0.5, theo D4 (mặt chủ yếu mang signer). Sau đó chuẩn hoá để mỗi nhóm đóng góp tỷ lệ γ, nhằm tránh kênh mặt nhiều token chiếm loss.
- α_int = 1.0 mặc định. Ablation: {0, 0.5, 1, ∞}, trong đó ∞ = chỉ dùng intrinsic.

**Loss global** (giống SLiM GLCL, DINO-CLS; SLiM ghi nhận MFP + GLCL tăng 85.3 → 87.7):
- Teacher xử lý T1, T2. Student xử lý near, far, G1, G2. G1 và G2 là hai view *không che* với tốc độ, hình thái và nhiễu độc lập.
- $$\mathcal{L}_{glob}=\frac{1}{8}\sum_{s\in\{near,far,G_1,G_2\}}\sum_{t\in\{T_1,T_2\}}\mathrm{CE}\big(\mathrm{softmax}((g^T_{cls}(\mathrm{cls}^T_t)-c)/\tau_t),\ \mathrm{softmax}(g^S_{cls}(\mathrm{cls}^S_s)/\tau_s)\big)$$
- Cộng thêm **KoLeo** trên CLS student (DINOv2: tăng retrieval Oxford-M từ 55.6 lên 63.9) với λ_KoLeo = 0.1, nhắm vào vấn đề bất đẳng hướng mà repo đã thấy (whitening thay đổi R@1 rất mạnh).

**Tổng:** 𝓛 = 𝓛_tok + λ_glob·𝓛_glob + λ_KoLeo·𝓛_KoLeo, với λ_glob = 1.0 (như SLiM).

**Chống collapse (mặc định, có bằng chứng trên skeleton):**
- EMA m: 0.996 → 1.0 (cosine). S-JEPA dùng 0.9999 → 1 cho 1200 epoch; run ngắn hơn cần m khởi điểm thấp hơn.
- Prototype target dùng **Sinkhorn-Knopp** (3 vòng) cho token, centering EMA cho CLS.
- τ_t warmup 0.04 → 0.07 trong 10% đầu; τ_s = 0.1.
- Bằng chứng: S-JEPA (CE + centering > MSE: 85.3 vs 83.7) và SLiM (prototype CE 90.17 vs feature-MSE 86.92, collapse rank 1.12).

**Nhánh thay thế (pilot P1 quyết định):**
- Hồi quy liên tục: L1 trên target đã LayerNorm, cộng **SIGReg** (LeJEPA; λ = 0.02, 1024 slice, Epps–Pulley 17 điểm trên [−5, 5]) đặt trên BN-MLP projector của token và CLS student.
- Chạy có và không có EMA. PointTube-JEPA thấy EMA *hại* (88.50 vs 94.08); Laya thấy stop-grad giúp run dài. Literature mâu thuẫn ⇒ phải đo.

**Chẩn đoán ghi lại ở mọi lần eval:**
- loss theo họ mask và theo kênh;
- entropy target teacher;
- perplexity sử dụng prototype theo nhóm;
- RankMe của token teacher và CLS;
- mean-cos của CLS;
- LiDAR trên clip held-out;
- signer probe.

**Báo động collapse** khi xảy ra một trong hai: mean-cos CLS > 0.9, hoặc perplexity prototype < 10% K.

### 6.9 Vì sao các thành phần này hợp lý về lý thuyết

1. **Latent chia sẻ (Kong et al.):** mask khiến model học latent tối thiểu chung giữa phần che và phần thấy.
   - Che *kênh* (R3) làm latent chung là **ràng buộc chéo kênh**: từ vựng, âm vị học, ngữ điệu.
   - Che *khối thời gian* (R2) làm latent chung là **động học hay chuyển động**.
   - Che *nhóm ngón* (R1) làm latent chung là **hình thái học bàn tay**.
   - Luật đóng phụ thuộc loại bỏ thành phần latent chung tầm thường (coupling động học), vốn là thứ Kong cảnh báo sẽ chiếm chỗ.
2. **Thiên kiến ρ của JEPA (Littwin et al.):** một đặc trưng được học khi nó *dự đoán được* dưới phân phối mask. PLCM là hỗn hợp có trọng số các họ mask, nên ρ_eff(feature) ≈ Σ_m π_m ρ_m(feature).
   - Một họ dễ duy nhất: mọi đặc trưng trơn đều có ρ cao, nên model học nội suy.
   - Một họ khó duy nhất: chi tiết tinh có ρ ≈ 0, nên bị bỏ (F4).
   - **Thang** bảo đảm mỗi loại đặc trưng mong muốn (chi tiết ngón, chuyển động, ràng buộc từ vựng) có ρ cao ở *ít nhất một* họ.
3. **Slow feature và augment bất đối xứng (Sobal; động lực BYOL/DirectPred):**
   - Một đặc trưng phụ thuộc hình thái (biên độ tuyệt đối của không gian ký, kích thước lòng bàn tay) mà student và teacher thấy với hình thái *độc lập* thì không dự đoán được (ρ ≈ 0), nên bị nén.
   - Đặc trưng bất biến hình thái (vị trí tương đối, cấu hình ngón) vẫn dự đoán được, nên được giữ.
   - Kết hợp với target intrinsic (ít mang thông tin toàn clip), cơ chế này khử signer mà không cần nhãn signer.
4. **Đẳng hướng và retrieval:** prototype CE + KoLeo trên CLS (hoặc SIGReg ở nhánh thay thế) đẩy embedding clip về phân phối đều, đẳng hướng.
   - Repo cho thấy whitening thay đổi R@1 rất mạnh (JH3D mean 12.5 vs white512 39.2).
   - Kỳ vọng: khoảng cách giữa mean-pool và whitened co lại.

### 6.10 Augmentation (bất đối xứng student / teacher)

| | Teacher T1, T2 (nhẹ) | Student near, far (căn cùng trục thời gian với T1, T2) | Student G1, G2 (chỉ cho loss global) |
|---|---|---|---|
| Hình thái, scale xương kiểu SLiM, mở rộng sang tay và mặt | cánh tay, cẳng tay, vai U(0.9, 1.1); mặt U(0.95, 1.05) | **độc lập với teacher**: cánh tay, cẳng tay U(0.85, 1.15); vai U(0.9, 1.1); từng đốt ngón U(0.85, 1.15); mặt dị hướng U(0.9, 1.1) | như student |
| Camera | xoay ±5°, scale ±5% | xoay ±10°, scale ±10%, shear ±0.1 | như student |
| Nhiễu keypoint | không | N(0, (σ₀(1 − c))²), σ₀ = 0.02·σ_t với tay | như student |
| Mất dấu | không | 5% frame tay chuyển thành [ABSENT] | như student |
| Tốc độ ký | s₁, s₂ ~ U(0.8, 1.25), *dùng chung* với near/far tương ứng | dùng chung với teacher, vì loss token cần thẳng hàng thời gian | **độc lập** U(0.8, 1.25) |

Scale xương thực hiện bằng forward kinematics 2-D trong không gian vector xương: tâm vai → vai → khuỷu → cổ tay (≡ gốc bàn tay) → các ngón; mặt scale quanh gốc khung mặt.

Lưu ý: scale ngón bị chuẩn hoá xương (§6.3g) triệt tiêu một phần trong SHAPE. Nó vẫn có tác dụng lên ORI (tỷ lệ bề ngang lòng bàn tay) và lên tương quan SHAPE–ORI. **Tác dụng chính nằm ở LOC:** độ dài tay quyết định toạ độ không gian ký so với vai, nên augment bất đối xứng buộc LOC mang *vị trí tương đối* thay vì *biên độ tuyệt đối* của signer.

### 6.11 Công thức train

| Siêu tham số | S (screening) | M (full) |
|---|---|---|
| Dữ liệu | Tier-S: ASL Citizen train (40,154 clip) | Tier-A: ASL Citizen + WLASL + MS-ASL train + How2Sign train (≈ 100k mẫu/epoch) |
| Batch / epoch / step | 256 / 40 / ≈ 6.3k | 512 / 100 / ≈ 20k |
| Optimizer | AdamW β = (0.9, 0.95); lr đỉnh 1e-3 ở batch 256 (scale tuyến tính, trần 2e-3); warmup 5 epoch; cosine về 1e-6; wd 0.04 → 0.2 | như S |
| EMA | 0.996 → 1.0 cosine | 0.996 → 1.0 |
| Precision | bf16, grad-clip 1.0 | bf16 |
| Mỗi bước, mỗi clip | 2 teacher forward + 4 student forward/backward. Token bị che không vào encoder student | như S |
| Ước tính thời gian (GPU dùng chung) | ≈ 0.12–0.2 s/step ⇒ **≈ 15–25 phút/run** | ≈ 1 s/step ⇒ **≈ 5–7 giờ/run** |
| Nạp dữ liệu | Cache float16 (toạ độ pixel và conf của 103 joint ở 25 fps) dạng memmap. **Toàn bộ đặc trưng kênh tính trên GPU**, theo lô, sau khi đã biết mask. CPU chỉ cắt mảng. | như S |

Về nạp dữ liệu: obs_jepa B bị nghẽn CPU ở khoảng 2 s/step do tính đặc trưng trên CPU. Thiết kế ở đây tránh lỗi đó và tuân thủ luật "tối đa 2 tiến trình".

### 6.12 Pseudo-code một bước train

```python
for x in loader:                                         # x: raw cached keypoints + conf (B, 64, 103, 3), W/H, fps
    sT1, sT2 = U(0.8, 1.25), U(0.8, 1.25)                 # shared speeds for (T1, near) and (T2, far)
    T1, T2 = aug_teacher(x, speed=sT1), aug_teacher(x, speed=sT2)
    near_x, far_x = aug_student(x, speed=sT1), aug_student(x, speed=sT2)   # independent morphology/noise
    G1, G2 = aug_student(x, speed=U()), aug_student(x, speed=U())
    m_near, m_far = sample_views(meta(T1), epoch, E), sample_views(meta(T2), epoch, E)  # PLCM

    with no_grad():                                      # teacher sees full views
        for T in (T1, T2):
            tok = channel_features(T, visible=ALL)       # GPU, anchors-only normalisation
            u   = Ec_T(tokenizer_T(tok))                 # intrinsic states
            h   = Ef_T(u, return_layers=last4)           # contextual states + cls
            P_int[T] = sinkhorn(g_int_T(u) / tau_t)      # per channel group
            P_ctx[T] = sinkhorn(g_ctx_T(LN(mean(LN(h_l) for h_l in h.layers))) / tau_t)
            P_cls[T] = softmax((g_cls_T(h.cls) - center) / tau_t)

    loss = 0
    for xs, m, T, w in ((near_x, m_near, T1, 1.0), (far_x, m_far, T2, w_far(epoch))):
        tok = channel_features(xs, visible=m.visible, joint_hidden=m.R1_joints)  # stats from visible only
        z   = Ef_S(Ec_S(tokenizer_S(tok)[m.visible]))    # masked tokens are dropped
        q   = predictor(z, queries=type_side_time(m.targets))                 # no coordinates
        loss += w * weighted_ce(P_ctx[T][m.targets], g_ctx_S(H_ctx(q)), P_int[T][m.targets], g_int_S(H_int(q)),
                                weights=reliability_T[m.targets] * group_gamma, alpha_int=1.0)
        cls_S.append(z.cls)
    cls_S += [Ef_S(Ec_S(tokenizer_S(channel_features(G)))).cls for G in (G1, G2)]
    loss += lam_glob * dino_cls(P_cls, [g_cls_S(c) for c in cls_S]) + lam_koleo * koleo(cls_S)
    loss.backward(); opt.step(); ema_update(teacher, student, m=ema_sched(step)); update_center()
```

### 6.13 Suy luận và readout (cố định trước khi chạy)

- **Encoder:** teacher (EMA).
- **Clip dài hơn 64 frame:** cửa sổ trượt bước 32 frame, ghép token. RoPE cho phép đưa trực tiếp clip tới 128 frame.
- **Các readout được báo cáo đồng thời:**
  - (i) [CLS];
  - (ii) mean-pool token;
  - (iii) mean-pool theo nhóm kênh (HS, HO, HL, FA) rồi nối lại, dùng cho phonology probe;
  - (iv) "bins2" (trung bình hai nửa có thứ tự; repo C012 +5.9);
  - (v) **attentive probe** (1 block cross-attention, 4 query, cộng linear) trên toàn bộ token.

---

## 7. Dữ liệu

### 7.1 Các tier pretraining

Chỉ dùng split train. Trainer không bao giờ đọc nhãn.

| Tier | Thành phần | Mục đích |
|---|---|---|
| **S** | ASL Citizen train (40,154 clip, 35 signer) | Screening nhanh (P1–P3) |
| **A** | S + WLASL2000 train + MS-ASL train + How2Sign train (31,047 câu, lấy cửa sổ ngẫu nhiên) | Kết quả chính (P4) |
| **M** | A + CSL-Daily train + Phoenix14T train | Kiểm tra đa ngôn ngữ (ablation A9) |
| **noASLC** | A − ASL Citizen | Đối chứng *in-domain*: SignRep và MASA không pretrain trên ASL Citizen |

### 7.2 Vệ sinh dữ liệu

1. WLASL và MS-ASL cùng nguồn YouTube. Loại khỏi pretraining mọi clip có *video id / URL* xuất hiện trong val/test của bất kỳ dataset đánh giá nào.
2. Kiểm tra trùng gần đúng bằng vân tay keypoint: 8 frame × (cổ tay + đầu ngón), cosine > 0.98 thì coi là trùng.
3. ASL Citizen: chỉ dùng train. Val và test có signer khác hẳn.
4. Mọi số liệu trên MS-ASL được gắn cờ trùng với YouTube-ASL nếu sau này mở rộng sang YouTube-ASL (SHuBERT: 38.2% test MS-ASL nằm trong YouTube-ASL).

### 7.3 Cache

Mỗi clip lưu:
- (T, 103, 3) float16, gồm x_px, y_px, score thô;
- W, H, fps, dataset, split, id.

Nhãn lưu riêng. Dung lượng ước tính khoảng 10 GB cho mọi tier cộng các tập đánh giá. Disk còn khoảng 78 GB.

### 7.4 Kiểm toán dữ liệu (P0)

Theo từng dataset, đo:
- tỷ lệ frame tay hợp lệ và tỷ lệ lòng bàn tay suy biến;
- phân bố fps;
- tỷ lệ tay trái là tay thuận;
- mức đồng thuận cổ tay thân với gốc bàn tay (obs_jepa đo được 90.8% trên ASL Citizen);
- phân bố độ dài clip.

---

## 8. Giao thức đánh giá (khoá trước khi chạy run đầu tiên)

**Dev = ASL Citizen val.** Mọi quyết định đều dựa trên tập này. Không chọn checkpoint trên test; SignRep từng chọn checkpoint trên WLASL-val.

| Mã | Đánh giá | Chi tiết |
|---|---|---|
| **E1a** | Dictionary retrieval chính thức (OFF-val) | Gallery = toàn bộ train, query = val, class-max cosine. Readout mean-pool, CLS và bins2, mỗi readout có thêm biến thể whitened (PCA-whitening fit trên gallery, không nhãn). Dùng code `method/p4_signssl/scripts/eval_retrieval.py` để giữ cùng protocol với các số SignRep trong repo. |
| **E1b** | **Attentive probe** top-1/5 (2,731 lớp) | Train trên đặc trưng train; chọn lr trong {1e-3, 3e-4} bằng 10% signer train để riêng; 30 epoch. |
| E1c | Linear probe trên mean-pool | logistic, quét C |
| E1d | LOSO-val whitened R@1 | Liên tục với các số EX15/EX16/P0 của repo |
| **E2a** | **Phonology probe, lexically disjoint (P-lex)** | Khoảng 16 đặc trưng ASL-LEX 2.0, cùng bộ với SHuBERT/SignDino: handshape (top-30 + other), selected fingers, flexion, spread, thumb position, thumb contact, major/minor location, contact, path movement, repeated movement, sign type, non-dominant handshape, wrist twist… Chia **5 fold theo gloss**: probe train trên clip train của gloss ở 4 fold, test trên clip val của gloss ở fold còn lại, nên vừa disjoint về từ vựng vừa disjoint về signer. Metric: macro-F1 mỗi đặc trưng và trung bình. Có thêm protocol "standard split" để đặt cạnh SHuBERT (85.02, fine-tune) và SignDino (91.85, tự báo). |
| E2b | NMI prototype | Phép gán prototype HS (argmax, đa số trên các bin tay thuận hoạt động) so với ASL-LEX Handshape; HL so với MajorLocation |
| E2c | Phân tầng theo SignType | R@1 tách theo một tay / đối xứng / bất đối xứng, để phát hiện shortcut đối xứng |
| **E3** | **Rò rỉ signer** (negative metric) | Signer probe trên *train* (35 signer, clip held-out của mỗi signer, logistic trên mean-pool) và trên val (5 signer). Tỷ lệ láng giềng gần nhất cùng signer so với cùng gloss (chỉ số H010). |
| E4 | Few-shot | K ∈ {1, 3, 5} clip/gloss làm gallery (3 lần rút), R@1 |
| E5 | Độ bền | Bỏ toàn bộ tay không thuận; bỏ tay thuận ở 30% frame; nhiễu σ ∈ {0.02, 0.05}·σ_t; đổi fps 15/30 |
| **E6** | Transfer và fine-tune (**final, một lần**) | WLASL2000 và MSASL1000: (i) linear P-I/P-C theo protocol STC (SignBERT+ 24.50, STC 32.35 trên WLASL); (ii) frozen retrieval (SignRep 29.92 WLASL); (iii) fine-tune toàn bộ (MASA 49.06, STC 51.98, DSTA 53.68). ASL Citizen test: E1a và E1b một lần. |
| E7 | *(tuỳ chọn)* CSLR | Token frozen → BiLSTM-CTC 2 lớp trên Phoenix14T và CSL-Daily → WER, để đánh giá chất lượng mức frame |

**Thống kê:**
- 3 seed cho các ứng viên cuối.
- 95% CI bằng **bootstrap theo cụm signer**.
- Báo cáo kết quả theo từng signer.
- So sánh có cặp.
- Mọi bảng báo cáo đủ các readout (bài học §2.2-1).

---

## 9. Baselines matched

Matched nghĩa là cùng pipeline dữ liệu, cùng encoder (nếu áp dụng), cùng tier, cùng epoch và cùng bộ đánh giá. Các baseline là **công tắc cấu hình** của cùng codebase, nhờ đó khác biệt chỉ đến từ thành phần đang xét.

| ID | Baseline | Cấu hình |
|---|---|---|
| **B0** | Descriptor training-free (D5) | Phân rã kênh + chuẩn hoá xương, 16 bước. **Val R@1 = 16.84.** |
| B1 | Encoder khởi tạo ngẫu nhiên, frozen | Mốc sàn |
| B2 | Có giám sát (nhãn ASL Citizen train), cùng encoder | Tham chiếu "trần có nhãn" cho frozen probe; không phải đối thủ |
| B3 | **MAE toạ độ** (họ SignBERT) | Token kênh, mask R0 60%, tái tạo đặc trưng kênh |
| B4 | **MAMP-style** | Token kênh, mask ngẫu nhiên theo motion 90%, target = Δ đặc trưng kênh |
| B5 | **JEPA-random** | PhonoJEPA với mask R0, chỉ target contextual (kiểu I-JEPA/V-JEPA) |
| **B6** | **JEPA-STM** (SLiM-like) = *ý tưởng gốc ở dạng mạnh nhất đã công bố* | Chỉ mask articulator-tube (R4, constant volume, ratio U(0.5, 0.9)); prototype contextual + DINO-CLS |
| B7 | **SHuBERT-style rời rạc** | k-means offline (K = 256) trên đặc trưng kênh; masked CE; span 3 bin theo từng kênh |
| B8 | **S-JEPA-style** | Token nhóm-joint (không phân rã kênh), motion-aware 90%, CE có centering, EMA. Dùng để kiểm định giá trị của token hoá. |
| Ext. | Không matched, chỉ tham chiếu | SignRep đã phát hành (RGB, cùng protocol repo); JH3D (RGB, repo); số công bố của MASA, SignBERT+, STC, SignSeek |

---

## 10. Giả thuyết, luật quyết định và tiêu chí dừng (pre-registered)

**Metric chính** (khai báo trước):
- **E1b** attentive top-1 và **E1a** OFF-val R@1 (mean-pool);
- với RQ2/RQ4: thêm **E2a** P-lex macro-F1 và **E3** signer probe.

**"Khác biệt có ý nghĩa"** khi đồng thời thoả:
- ≥ +1.5 điểm trên metric chính;
- cùng chiều ở cả 3 seed;
- CI bootstrap theo signer có cận dưới > 0.

| Giả thuyết | Phát biểu kiểm định được |
|---|---|
| **H1** (RQ1) | PhonoJEPA > B6 (JEPA-STM) và > B5 (JEPA-random) trên E1a/E1b. |
| **H2** (RQ2) | ctx + int > chỉ ctx: ≥ +2 macro-F1 P-lex; signer probe giảm ≥ 5 điểm; metric chính giảm không quá 0.5. |
| **H3** (RQ3) | Thang đầy đủ > chỉ far (R3 ∪ R4) và > chỉ near (R1 ∪ R2); curriculum > không curriculum (≥ +1). |
| **H4** (RQ4) | Augment bất đối xứng: signer probe (35 signer) giảm ≥ 5 điểm, metric chính giảm không quá 0.5, few-shot cross-signer ≥ +1. |
| **H5** (RQ5) | PhonoJEPA > B4 (MAMP) và > B3 (MAE) trên frozen. Khi fine-tune WLASL, chênh lệch ≤ 1 điểm (dự đoán từ literature). |
| **H6** | Loss global + KoLeo tăng R@1 mean-pool ≥ +3 và thu hẹp khoảng cách whitened − mean ≥ 50%. |
| **H7** | Tier A > Tier S (≥ +2). noASLC nằm trong khoảng 3 điểm so với Tier A trên ASL Citizen val. |

**Mục tiêu tuyệt đối** (định hướng, không phải luật quyết định):
- OFF-val R@1 ≥ 2 × B0, tức khoảng 34.
- OFF-test R@1 vượt rõ MASA (27.96).
- Thu hẹp một nửa khoảng cách tới SignRep (49.95).

**Tiêu chí dừng / chuyển hướng:**
- **K1:** sau P2, nếu PhonoJEPA-S không vượt B0 ít nhất 5 điểm OFF-val R@1, thì SSL không thêm gì đáng kể so với descriptor ⇒ dừng và phân tích.
- **K2:** nếu B4 (MAMP) ≥ PhonoJEPA trên *mọi* metric frozen (3 seed), thì claim "latent target" thất bại ⇒ chuyển phương pháp thành "token kênh + PLCM + target motion" và báo cáo trung thực.
- **K3:** nếu B6 ≥ PhonoJEPA, thì claim về masking thất bại ⇒ chỉ báo cáo các phát hiện về target, token và khử signer (RQ2, RQ4).
- **K4:** nếu mọi nhánh của P1 đều collapse ⇒ chuyển sang teacher đóng băng kiểu SALT: train một teacher MAMP ngắn, đóng băng, rồi distill.

---

## 11. Kế hoạch ablation (theo thứ tự ưu tiên)

| # | Trục | Biến thể |
|---|---|---|
| A1 | Họ mask | đầy đủ / −R1 / −R2 / −R3 / −R4 / chỉ R4 / chỉ R3 / chỉ R0; curriculum bật/tắt; quét r_near, r_far |
| A2 | Target | ctx / int / ctx + int / + desc (motion); α_int ∈ {0.5, 1, 2}; cửa sổ E_c ∈ {±1, ±2, ±4} |
| A3 | Token hoá | 9 kênh âm vị / 5 token bộ phận không phân rã (tay×2, mặt, thân, đầu) / nhóm-joint (S-JEPA-like) |
| A4 | Chuẩn hoá | chuẩn hoá xương bật/tắt; neutral face bật/tắt; mirror tay thuận bật/tắt |
| A5 | Augment | hình thái bất đối xứng / đối xứng / không; tốc độ bật/tắt |
| A6 | Chống collapse | prototype-CE + SK / L1 + SIGReg (có hoặc không EMA) / L1 + EMA (dự kiến collapse; đối chứng âm) |
| A7 | Global | không / DINO-CLS / LeJEPA; KoLeo bật/tắt |
| A8 | **Đối chứng rò rỉ** | LOC không chứa khuỷu (rò rỉ) / R1 che cả MCP / LOC có quan hệ tới mặt ⇒ định lượng hiệu ứng shortcut |
| A9 | Dữ liệu | Tier S / A / M / noASLC |
| A10 | Quy mô | S vs M; cửa sổ 64 vs 128 frame |

---

## 12. Các phase thực nghiệm và ngân sách compute

Tuân thủ luật server: mỗi lúc một job GPU, tối đa 2 tiến trình. Runner tuần tự có chờ bộ nhớ GPU và retry khi OOM (tái dùng mẫu `obs_jepa/oj/schedule.py`).

| Phase | Nội dung | GPU-h | Gate để đi tiếp |
|---|---|---|---|
| **P0** | Không GPU. Cache, splits, dedup, join ASL-LEX, harness E1–E5, B0, toàn bộ unit test. **Leak audit:** mở rộng D3 thành R² giữa mọi cặp kênh cùng frame, kết quả là bảng quyết định các luật đóng phụ thuộc cần có. | 0 | Test pass; B0 trong pipeline mới (chuẩn hoá chỉ theo anchor) xấp xỉ D5 (16.84 ± 1.5). Nếu lệch nhiều hơn, phải giải thích trước khi đi tiếp. |
| **P1** | Pilot chống collapse: 4 nhánh A6 × model S × Tier S × 40 epoch, cộng B1 | ≈ 2 | Ít nhất một nhánh không collapse và vượt B0 trên E1b ⇒ khóa họ loss |
| **P2** | Screening lõi, 1 seed: B3, B4, B5, B6, B7, B8, Ours, Ours-ctx, Ours-int, Ours-no-curriculum | ≈ 5–6 | Xét K1–K3 |
| **P3** | Ablation A2–A8 chọn lọc (khoảng 12 run) | ≈ 6 | Khóa cấu hình M |
| **P4** | Tier A, model M: Ours, baseline JEPA tốt nhất (B5/B6), B4 × 1 seed; sau đó top-2 × thêm 2 seed; Ours trên Tier noASLC | ≈ 35–45 | Xét H1–H7 |
| **P5** | Final, một lần: ASL Citizen test, WLASL/MSASL linear + retrieval + fine-tune | ≈ 6–8 | — |
| | **Tổng** | **≈ 55–65** (+ khoảng 10 giờ extract/eval) | |

---

## 13. Kế hoạch implement

Chỉ là kế hoạch. **Chưa code gì** cho tới khi anh duyệt.

```
method/phonojepa/
  pj/data/        cache_build.py  splits.py (tiers, dedup)  audit.py  asllex.py
  pj/features/    channels.py   # GPU: anchors, mirror, bone/neutral norm, 9 channel features, dependency map
  pj/masks/       plcm.py       # R0–R4, budget, closure rules, curriculum, context dropout
  pj/models/      tokenizer.py  encoder.py (E_c windowed attn, E_f)  predictor.py  heads.py (prototypes, Sinkhorn)
  pj/             train.py  losses.py (weighted CE, DINO-CLS, KoLeo, SIGReg-alt)  ema.py  diagnostics.py
  pj/eval/        extract.py  retrieval.py  probes.py  phonology.py  signer.py  robustness.py  finetune.py
  configs/        S.yaml  M.yaml  baselines/{B3..B8}.yaml  ablations/*.yaml
  tests/          test_dependency_perturbation.py  test_mask_budget_closure.py  test_mirror_roundtrip.py
                  test_anchor_only_normalisation.py  test_absent_vs_mask.py  test_teacher_no_grad_ema.py
                  test_sinkhorn_centering.py  test_baselines_share_pipeline.py
  runtime/        logs, small checkpoints, eval JSON, run manifests
```

Các nguyên tắc kỹ thuật rút ra từ những lỗi trước đây trong repo:
- **Run manifest:** mỗi run ghi hash cấu hình, hash dữ liệu, *số step thực sự đã chạy* và loss cuối. Bộ đánh giá từ chối checkpoint có số step < 95% kế hoạch. Mục đích là tránh lặp lại lỗi JHOG/JPRM (OOM mà log vẫn ghi "DONE").
- **Kiểm tra rò rỉ tự động:** *hidden-value perturbation* cho mọi họ mask và mọi baseline.
- **Tái dùng code:** protocol OFF của `p4_signssl`; LOSO của `jepa_targets/jt/evaluate.py`; runner của `obs_jepa`.
- Theo dõi collapse và tự dừng run khi có báo động (§6.8).

---

## 14. Rủi ro và phương án dự phòng

| Rủi ro | Khả năng | Ảnh hưởng | Giảm thiểu |
|---|---|---|---|
| Trần chất lượng pose 2-D: hand AP khoảng 58–66; 2-D làm hướng và hình dạng đan xen | cao | trần tuyệt đối thấp hơn RGB | Đặt claim ở mức *tương đối* với baseline matched. Mở rộng X2 (3-D hands). Báo cáo E5. |
| JEPA ≈ MAMP trên skeleton (literature: +0.4) | trung bình–cao | claim RQ5 yếu | K2. Thiết kế để kết quả null vẫn có giá trị: RQ1, RQ2, RQ4 độc lập với RQ5. |
| Collapse (SLiM E.4, repo) | trung bình | run vô dụng | P1 trước tiên; SK, centering, EMA; báo động; K4 |
| Tín hiệu chéo kênh khiêm tốn (D2 khoảng 0.3; CIP 0/36) | trung bình | far view chủ yếu là nhiễu | Thang + curriculum; A1 đo đóng góp của R3/R4; target intrinsic giúp teacher bỏ nhiễu |
| Danh tính signer vẫn rò qua *phong cách* (tốc độ, biên độ) | cao | H4 thất bại một phần | Augment tốc độ cho view global; E3 dùng 35 signer; báo cáo trung thực |
| Quá nhiều núm vặn | trung bình | khó quy kết nguyên nhân | Phase có gate; mặc định cố định; chỉ A1–A8 được quét |
| Rò rỉ dữ liệu: WLASL/MS-ASL trùng; ASL Citizen nằm trong pretraining | trung bình | số bị thổi phồng | §7.2; Tier noASLC; gắn cờ khi báo cáo |
| Test đã từng bị dùng | chắc chắn | thiên lệch thích nghi (nhẹ, vì là phương pháp mới) | Chỉ dùng val để quyết định; test một lần; thêm trục mới P-lex |
| Nhầm tay thuận | thấp–trung bình | nhiễu kênh D/N | Kiểm toán ở P0; ablation A4; tay thuận được quyết định theo clip |

---

## 15. Mở rộng tuỳ chọn (sau khi lõi đạt gate)

| # | Mở rộng | Mô tả |
|---|---|---|
| **X1** | **Cầu nối sang SignRep** | (a) Dùng latent intrinsic/contextual của PhonoJEPA frozen làm **prior target học được** thay cho prior 3-D thủ công trong công thức JH3D (RGB JEPA + priors), mạnh nhất repo. Câu hỏi kiểm định: *prior âm vị học học từ skeleton có tốt hơn prior hình học thủ công của SignRep không?* Đây đúng là lựa chọn thiết kế cốt lõi của SignRep. (b) Late fusion embedding SignRep (RGB) với PhonoJEPA cho retrieval; rẻ và kiểm tra tính bổ sung. |
| X2 | Tay 3-D | MediaPipe 3-D hoặc HaMeR, để tách ORI và SHAPE triệt để hơn (D3 cho thấy 2-D buộc phải đan xen) |
| X3 | LOC bằng path-signature | Lấy cảm hứng từ Path-JEPA. Signed area mã hoá chiều quay của chuyển động, một tham số movement của ASL. Bền với thay đổi fps. |
| X4 | Predictor đa giả thuyết | K giả thuyết, relaxed winner-takes-all cho R3/R4, để biểu diễn tính đa trị (cặp tối thiểu) thay vì hồi quy về trung bình |
| X5 | Ký liên tục | CSLR/SLT với token mức frame. Thêm *context loss* kiểu V-JEPA 2.1 nếu feature dense suy biến. |
| X6 | Mở rộng quy mô | Trích keypoint RTMW cho YouTube-ASL / YouTube-SL-25 (cần tải video, hiện chưa có local) |

---

## 16. Đóng góp kỳ vọng và đánh giá novelty trung thực

**Mới, nếu thực nghiệm xác nhận:**
1. **Masking theo kênh âm vị học có đóng phụ thuộc.** Đơn vị che là tham số âm vị (SHAPE, ORI, LOC, mặt) thay vì bộ phận giải phẫu (SLiM, SignSeek) hay joint (S-JEPA). Các luật đóng phụ thuộc được *suy ra từ đo rò rỉ* (D3, leak audit ở P0).
2. **Phân đôi target contextual / intrinsic** cho câu hỏi "representation của phần bị che là gì", kèm **prototype theo nhóm kênh**. Chúng tạo thành một bộ đơn vị âm vị học học online, đo được bằng NMI với ASL-LEX. Luận cứ cơ chế: target intrinsic giảm thiên kiến slow-feature về danh tính signer.
3. **Thang độ khó đã được hiệu chỉnh bằng đo đạc** (D2) cùng curriculum, thay cho tỷ lệ heuristic.
4. **Khử signer không cần nhãn** trong sign SSL bằng augment hình thái và tốc độ *bất đối xứng*, chuẩn hoá chỉ dựa trên anchor, và neutral face. Signer probe được báo cáo như một negative metric.
5. **Giao thức đánh giá:** phonology probe *frozen, lexically disjoint* cộng phân tầng SignType. Các công trình trước không làm: SHuBERT fine-tune và có trùng từ vựng; SignDino tự báo.

**Không mới (kế thừa, có trích dẫn):**
- JEPA/EMA và target ở output teacher (I-JEPA, V-JEPA, S-JEPA);
- che bộ phận giải phẫu theo tube (SLiM) và lan truyền phụ thuộc (Path-JEPA);
- prototype CE với Sinkhorn và KoLeo (DINO, iBOT, DINOv2, SLiM);
- scale xương (SLiM);
- SIGReg (LeJEPA);
- tách nhánh tay và mirror tay (STC, SignSeek).

**Đánh giá thẳng thắn:**
- Một bài báo mạnh cần **H1 và H2 thành công**.
- Nếu chỉ H2 và H4 thành công (target intrinsic và khử signer), đóng góp vẫn có giá trị nhưng là dạng phân tích.
- Nếu K2 xảy ra (MAMP ≥ JEPA), đóng góp chuyển thành "token kênh âm vị + PLCM cho sign skeleton SSL". Phần này vẫn mới so với MASA, SignBERT+ và BEST, nhưng kém hấp dẫn hơn.

---

## 17. Các câu hỏi cần anh review hoặc quyết định trước khi implement

1. **Phạm vi:** đồng ý skeleton-only cho lõi, còn cầu nối SignRep (X1) chỉ làm sau gate P4?
2. **Dữ liệu:** có đưa CSL-Daily và Phoenix14T (Tier M, đa ngôn ngữ) vào kết quả chính, hay chỉ làm ablation?
3. **Test:** chấp nhận dùng ASL Citizen, WLASL và MS-ASL test *một lần nữa* ở P5 cho phương pháp mới, có ghi rõ lịch sử sử dụng? Hay anh muốn dựng holdout mới (ví dụ tách một phần signer của ASL Citizen test làm "fresh")?
4. **Họ loss mặc định:** chấp nhận để P1 quyết định giữa prototype-CE và L1 + SIGReg, hay anh muốn khóa ngay prototype-CE theo bằng chứng S-JEPA/SLiM?
5. **Ngân sách:** khoảng 55–65 GPU-h (cộng khoảng 10 giờ eval) trên GPU dùng chung có chấp nhận được không?
6. **Quan hệ với obs_jepa B** (đang chờ trong hàng đợi P2 của proposal11): tạm dừng B để tránh trùng lặp, hay chạy B như một baseline thêm? Tôi đề xuất **dùng B làm baseline bổ sung** vì nó đã implement xong.
7. **Đặt tên:** "PhonoJEPA" có ổn không?

---

## Phụ lục A — Bảng siêu tham số mặc định (model M)

| Nhóm | Giá trị |
|---|---|
| Dữ liệu | 25 fps; T = 64; bin = 2 frame; 9 kênh × 32 bin = 288 token; 103 joint |
| Ngưỡng | τ_hand: mean score > 2.0 (P0 hiệu chỉnh); lòng bàn tay > 0.10·σ_t; conf theo q10/q90 của từng nhóm |
| Mô hình | d = 384, 6 head; E_c 2 block (cửa sổ ±2 bin, trong kênh); E_f 10 block + CLS; predictor 6 × 192; RoPE thời gian |
| Head | K_ctx = 8192, K_cls = 8192; K_HS = 256, K_HO = 64, K_HL = 128, K_FA = 128; bottleneck 256 |
| Mask | near: R1 ∪ R2, r ~ U(0.4, 0.7); far: R4 (p = 0.4) hoặc R3, r ~ U(0.3, 0.6); span ≥ 4 bin; context dropout 0.1; ngữ cảnh tối thiểu 25% |
| Curriculum | w_far = 0.3 + 0.7(e/E)²; span tối đa 0.5 → 1.0⎮A⎮ trong 30% đầu |
| Loss | α_int = 1; γ_tay = 1, γ_mặt = 0.5; λ_glob = 1; λ_KoLeo = 0.1 |
| Teacher | EMA 0.996 → 1; τ_t 0.04 → 0.07 (warmup 10%); τ_s = 0.1; Sinkhorn 3 vòng; center CLS momentum 0.9 |
| Optimizer | AdamW (0.9, 0.95); lr 1e-3 @ batch 256 (batch 512 → 2e-3, trần 2e-3); warmup 5 epoch; cosine; wd 0.04 → 0.2; bf16; clip 1.0 |
| Augment | teacher: hình thái ±10%, xoay ±5°; student: hình thái ±15% độc lập, xoay ±10°, shear 0.1, nhiễu σ₀ = 0.02σ_t·(1 − c), drop tay 5%; tốc độ U(0.8, 1.25) |
| Nhánh thay thế A6 | L1 trên target LN + SIGReg λ = 0.02 (1024 slice, 17 điểm trên [−5, 5], BN-MLP projector); có hoặc không EMA |

## Phụ lục B — Nguồn chính

ArXiv ID hoặc venue. Đầy đủ hơn, kèm tag kiểm chứng, nằm trong `literature/`.

**JEPA và paradigm:**
- I-JEPA 2301.08243 (CVPR'23); V-JEPA 2404.08471; V-JEPA 2 2506.09985; V-JEPA 2.1 2603.14482.
- LeJEPA 2511.08544; SALT 2509.24317; C-JEPA 2410.19560.
- Littwin et al. 2407.03475 (NeurIPS'24); Sobal et al. 2211.10831; DirectPred 2102.06810.
- data2vec 2202.03555; data2vec 2.0 2212.07525; iBOT 2111.07832; DINOv2 2304.07193; DINOv3 2508.10104.
- MaskFeat 2112.09133; RankMe 2210.02885; LiDAR 2312.04000; efficient probing 2506.10178.
- PointTube-JEPA 2608.24093; Laya 2603.16281; LeWM 2603.19312; Human-JEPA 2608.21160; T-JEPA 2410.05016.
- Point-JEPA 2404.16432; Graph-JEPA 2309.16014.

**Masking:**
- Kong et al. (CVPR'23); SemMAE (NeurIPS'22); EPM (CVPR'23); AttMask (ECCV'22); ADIOS (ICML'22); HPM (CVPR'23).
- MAE 2111.06377; VideoMAE 2203.12602; MGMAE; AdaMAE; MaskAlign.
- Point-MAE; Point-M2AE; CroCo 2210.10716; SiamMAE 2305.14344.

**Skeleton:**
- S-JEPA (ECCV'24, DOI 10.1007/978-3-031-73411-3_21); MAMP 2308.07092 (ICCV'23).
- SLiM 2603.10648; Path-JEPA (ECCV'26, DOI 10.1007/978-3-032-37258-1_7); GFP 2509.03609 (ICCV'25); AMR (CVPR'26).
- MacDiff 2409.10473; IGM 2410.20349; SkeletonMAE 2307.08476 / 2209.02399; MaskSem 2508.12948; MotionBERT 2210.06551.

**Sign language:**
- SignRep 2503.08529 (ICCV'25); SHuBERT 2411.16765 (ACL'25); SignSeek 2609.03695; SignDino 2609.06296.
- SignBERT 2110.05382; SignBERT+ 2305.04868; BEST 2302.05075; MASA 2405.20666; STC 2406.10501; SignMAE 2605.02094.
- OpenHands 2110.05877; Uni-Sign 2501.15187; DSTA-SLR 2403.12519; NLA-SLR 2303.12080; SL-GCN 2103.08833.
- Kezar et al. 2302.05759; Sem-Lex; WLASL-LEX; 2609.18772 (handshape zero-shot); 2410.13675 (danh tính qua pose).
