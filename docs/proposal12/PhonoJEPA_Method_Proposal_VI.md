# Proposal 12 (v2) — **PhonoJEPA**: tự giám sát skeleton cho ngôn ngữ ký hiệu bằng cách che *kênh âm vị học* và dự đoán biểu diễn của kênh bị che

> **Trạng thái:** **anh đã review và duyệt hướng (2026-10-11)**. Quyết định ở §17. **Chưa implement, chưa train.** Bước tiếp theo là P0, chỉ bắt đầu khi anh ra lệnh.
>
> **Phiên bản:** v2.1 (2026-10-11). v2 được viết lại sau vòng phản biện thứ nhất; v2.1 vá theo vòng phản biện thứ hai. Nhật ký phản biện ở §18. Bản v1 lưu ở `archive/`.
>
> Mọi con số trong tài liệu thuộc một trong ba loại:
> 1. đã công bố, có nguồn;
> 2. đo trong repo;
> 3. chẩn đoán *training-free* tôi chạy cho proposal này (§5, `pilot_diagnostics/`).
>
> **Tài liệu kèm theo:**
> - `literature/`: 4 báo cáo literature, khoảng 25k từ, số liệu gắn tag kiểm chứng raw text.
> - `pilot_diagnostics/`: script và JSON của D1–D7.
> - `archive/`: bản v1 trước phản biện.

---

## 0. Tóm tắt

### 0.1 Ý tưởng gốc của anh và phản biện

**Ý tưởng gốc:** che một số joint hoặc bộ phận cơ thể, rồi dùng phần còn lại để dự đoán *representation* của phần bị che. Đây là skeleton SSL kết hợp structured anatomical masking và JEPA.

**Ý tưởng này ở dạng thuần đã có người làm:**
- S-JEPA (ECCV'24): JEPA trên joint.
- **SLiM** (arXiv 2603.10648, chính là "SLim NeurIPS'26" anh nhắc): *Semantic Tube Masking*, tức che vùng giải phẫu liên thông theo thời gian.
- Path-JEPA (ECCV'26): mask theo cây giải phẫu, lan truyền tới các token phụ thuộc.
- SignSeek (2609.03695): *masked articulator prediction* cho sign language, nhưng dùng nhãn gloss.
- obs_jepa arm B trong chính repo này (chưa train).

**Năm vấn đề đặc thù sign language mà dạng thuần chưa giải.** Mỗi vấn đề có bằng chứng từ literature hoặc từ chẩn đoán trên ASL Citizen (§5):

| # | Vấn đề | Bằng chứng |
|---|---|---|
| 1 | **Rò rỉ phụ thuộc và nội suy** | Hình bàn tay thô (toạ độ tương đối cổ tay, chưa chuẩn hoá) tại *t* đoán được từ *t±1* với R² 0.87. Cổ tay thân trùng gốc bàn tay. Hướng lòng bàn tay và cấu hình ngón đan xen trong 2-D (R² 0.38–0.50). |
| 2 | **Che trọn một articulator gần như là bài toán bất định** | Oracle từ vựng (biết gloss) chỉ giải thích R² 0.27 của cấu hình bàn tay. Dự đoán từ các kênh khác chỉ đạt 0.15. |
| 3 | **Danh tính signer là "slow feature"** | Probe signer 0.92–0.999 với mọi model trong repo. Kênh tư thế đầu nhận ra signer ở mức 0.89 nhưng gần như không mang thông tin gloss. |
| 4 | **Hồi quy latent thuần trên skeleton dễ collapse** | SLiM: feature-MSE còn effective rank 1.12. S-JEPA: bỏ EMA còn 1.6%. |
| 5 | **Lợi ích kỳ vọng nhỏ** | S-JEPA chỉ hơn MAMP (motion-MAE) +0.4. Mask có cấu trúc thường chỉ +1–3 điểm. |

### 0.2 Đề xuất v2

PhonoJEPA đổi **đơn vị che** từ bộ phận giải phẫu sang **kênh âm vị học**:
- Mỗi bàn tay có hai kênh:
  - **CONF**: cấu hình, gồm handshape cộng orientation.
  - **LOC**: vị trí và chuyển động, gồm cổ tay và khuỷu tay trong khung thân.
- Mặt có ba kênh: **UPPER** (lông mày, mí mắt), **MOUTH**, **HEAD** (tư thế đầu).

Tổng cộng 7 kênh × 32 bin = **224 token**. Mỗi joint thuộc đúng một kênh. Chuẩn hoá chỉ dùng anchor (vai) hoặc thống kê trên các frame *thấy được*.

Bốn thành phần lõi:
1. **Che kênh theo "thang độ khó"** (PLCM), từ dễ đến khó: nhóm ngón trong CONF → khối thời gian → span của một kênh → cả articulator.
   - Lấy mẫu mask **không phụ thuộc nội dung**.
   - Lan truyền mask qua phép resample thời gian.
   - Predictor được query ở mọi ô bị che.
2. **Hai loại target cho phần bị che:**
   - *contextual*: latent của teacher EMA, loss CE có centering kiểu S-JEPA, đã được chứng minh trên skeleton;
   - *intrinsic tường minh*: mô tả đã chuẩn hoá của chính kênh bị che cộng chuyển động của nó, kiểu MAMP/MaskFeat.

   Câu hỏi khoa học trung tâm là kênh bị che nên được biểu diễn bằng *ngữ cảnh*, bằng *chính nó*, hay cả hai.
3. **Khử signer không nhãn** bằng ba cơ chế:
   - neutral-face, đã đo: probe signer của kênh mặt giảm từ 0.98 xuống 0.67;
   - đặc trưng tỷ lệ tính tương đối theo median của clip;
   - augment hình thái bất đối xứng nhưng *giữ nguyên vị trí cổ tay*, để không phá location âm vị.
4. **Kế hoạch tối thiểu trước, mở rộng sau** (MVM):
   - Bắt đầu bằng một ma trận nhân tố có đối chứng: mask {ngẫu nhiên, ống giải phẫu kiểu SLiM, PLCM-lite} × target {contextual, intrinsic, cả hai}, cộng các baseline global-only, chỉ-tay, MAE và MAMP. Kế hoạch đã chốt theo §17: P0–P3, model S, **1 seed**, khoảng 11–30 GPU-giờ.
   - Chỉ khi qua gate mới thêm curriculum, global term và augment bất đối xứng.

### 0.3 Mục tiêu và rủi ro

**Mục tiêu định lượng** (frozen, label-free, cùng protocol của repo):
- Vượt rõ mốc sàn training-free B0 = 13.46 R@1 OFF-val (§5, D8).
- Vượt các baseline matched.
- Giảm signer-probe và tăng phonology-probe *lexically disjoint*.

Số công bố như MASA 27.96 hay SignRep 49.95 dùng readout và protocol khác (§2.2), nên chỉ để định hướng.

**Rủi ro được nêu thẳng:** literature cho thấy lợi ích của JEPA trên skeleton nhỏ. Thiết kế thực nghiệm vì vậy được chuẩn bị để **kết quả null vẫn có giá trị**.

---

## 1. Mục tiêu, phạm vi và câu hỏi nghiên cứu

**Mục tiêu:** học encoder *f* trên chuỗi keypoint 2-D không nhãn sao cho biểu diễn:
- (a) phân biệt được từ vựng khi *frozen*;
- (b) mã hoá được tham số âm vị học, tổng quát hoá sang ký hiệu chưa thấy;
- (c) ít mang danh tính signer;
- (d) fine-tune tốt cho ISLR; tuỳ chọn cho CSLR và SLT.

**Phạm vi:**
- Skeleton 2-D COCO-WholeBody 133 điểm. Đây là định dạng chung của 6 dataset local.
- Không dùng nhãn trong pretraining.
- RGB chỉ xuất hiện ở phần mở rộng X1 (§15).

**Câu hỏi nghiên cứu:**
- **RQ1:** Che theo *kênh âm vị học* có tốt hơn che theo *bộ phận giải phẫu* (kiểu SLiM/SignSeek) và che ngẫu nhiên không, khi cùng token, cùng target, cùng view và cùng compute?
- **RQ2:** "Representation của phần bị che" nên là *contextual* (latent teacher), *intrinsic tường minh* (mô tả của chính kênh) hay cả hai? Mỗi lựa chọn ảnh hưởng tới thông tin âm vị học và danh tính signer *trong target* lẫn *trong biểu diễn học được* như thế nào?
- **RQ3:** Thang độ khó cộng curriculum có cần không, so với một họ mask duy nhất?
- **RQ4:** Có khử được danh tính signer mà không cần nhãn signer không?
- **RQ5:** Latent target có thắng target tường minh (toạ độ, chuyển động) trên sign skeleton không? Nếu có thì ở protocol nào?

---

## 2. Kiểm kê codebase và các bài học ràng buộc thiết kế

Tôi đã rà `docs/proposal1–11`, `docs/storage`, `docs/proposal4/.../research` và 12 thư mục `method/*`. Thứ tự ưu tiên bằng chứng: eval JSON/log, rồi ledger, rồi spec. Lưu ý: `STATE.md` v9 cũ hơn EX16/P0.

### 2.1 Bảng tổng hợp

| Hướng (proposal) | Code | Trạng thái | Số đo tốt nhất (nguồn) | Kết luận |
|---|---|---|---|---|
| Readout bins2/chamfer trên SignRep frozen (p4) | `p4_signssl` | test 1 lần, locked | ASLC test R0 49.02 → R1 54.93 → R2 58.42 (`EX05_result.json`) | **Supported.** Readout tạo khác biệt lớn hơn mọi tinh chỉnh encoder. |
| H008 set→vector distillation | `p4_signssl` | test, 3 seed | 57.28–57.31 | Supported; novelty thấp |
| CIP-Sign cross-articulator innovation (p5) | `p4_signssl/cip` | premise ×3 | **0/36 gate** | **Negative.** Forecast chéo articulator không thêm gì so với lịch sử của chính nó. |
| Spectral aux, SignRep continuation (p3) | `signrep_spectral` | val | B1 8.4 / M1 7.6 vs B0 37.5 | Negative. Continuation phá biểu diễn (rank 24). |
| ASC-JEPA (p8) | `asc_jepa` | val | 48.99 vs 49.50 | Negative |
| SignRel-JEPA (p6) | `signrel_jepa` | 95 clip | collapse (cos 0.998) | Negative |
| SIGNREL temporal (p7) | `signrel_temporal` | locked test | A4 ≤ 11.25 vs **raw 21.25** | Negative. Train làm tệ hơn đặc trưng thô. |
| JEPA vs MAE + 2-D priors (p10/EX15) | `jepa_targets` | val, 1 seed | JMOT 20.73, JP 19.93, J 7.38, M 4.11 (LOSO white512) | Prior chuyển động tốt nhất; prior non-manual có hại (1.96) |
| SignRep repro vs JEPA + priors, cùng init Hiera (EX16/P0) | `jepa_targets`, `jepa_baseline` | val, 1 seed | **JH3D 39.15 LOSO / 61.7 LP** vs SR3D 5.10 / 17.6 | Latent + prior cục bộ ≫ repro SignRep ở quy mô 3k video. Repro có thể under-fit. |
| obs_jepa A/B/C (p11) | `obs_jepa` | implemented, **chưa train** | — | B là graph-JEPA 67 landmark, gần ý tưởng của anh nhất |
| SignCoord (p9) | `signcoord_ssl` | smoke | — | Untested |

### 2.2 Mười bài học

1. **Readout quyết định kết luận.** Cùng weight SignRep cho các số rất khác nhau theo readout:

   | Readout | R@1 (LOSO-val) |
   |---|---|
   | plain mean | 30.39 |
   | white512 | 37.55 |
   | R0 | 49.50 |
   | R1 | 53.88 |

   *Lưu ý:* LOSO-val (truy hồi trong val) khác OFF-val (gallery = train). SignRep đạt OFF-val R0 59.16 và OFF-test R0 49.02. ⇒ Khoá protocol trước và báo cáo nhiều readout.
2. **Signer chi phối:** probe 0.92–0.999. 82% láng giềng tự tin nhất là cùng signer.
3. **Collapse thường gặp:** rank 24, cos 0.999, cos 0.945.
4. **Bug từng đảo kết luận:** student copy teacher; lỗi input-scale; OOM nhưng log vẫn "DONE". ⇒ Cần unit test chống rò rỉ và run manifest.
5. **CIP 0/36 là kết quả về *forecasting* khi đã có lịch sử của chính kênh.** Bài toán *masked* khác ở chỗ che cả quá khứ lẫn hiện tại của kênh. D2 (§5) đo trực tiếp: có tín hiệu nhưng khiêm tốn (0.15).
6. **Prior chuyển động tốt nhất; prior non-manual có hại.** D4 (§5) giải thích: kênh mặt chưa chuẩn hoá là vân tay signer.
7. **Latent + prior cục bộ ≫ prior thuần (JH3D).** Đây là bằng chứng nội bộ cho thiết kế target contextual + intrinsic tường minh.
8. **Đặc trưng thô có thể thắng model đã train (SIGNREL A0).** ⇒ Bắt buộc có baseline training-free B0.
9. **Repo chưa có baseline pose-SSL nào** (MAMP, S-JEPA, SignBERT, MASA). ⇒ Phải tự dựng baseline matched.
10. **Mọi test split ASL đã được dùng ít nhất một lần.** ⇒ Mọi quyết định dựa trên dev. Test dùng một lần. Bổ sung dev-2 gồm các signer tách ra từ train (§7).

### 2.3 Dữ liệu thực tế

Định dạng chung: `keypoints` (T×1×133×2, chuẩn hoá x/W, y/H) và `scores` (không hiệu chỉnh, khoảng 0.5–10).

| Dataset | Số clip | Ghi chú |
|---|---|---|
| ASL Citizen | 83,399: train 40,154 / val 10,304 / test 32,941 | Signer-disjoint (35/6/11 signer); 2,731 gloss; có mã **ASL-LEX** |
| WLASL2000 | 21,095 | Trùng nguồn YouTube với MS-ASL |
| MS-ASL | 18,855 | |
| How2Sign | train 31,047 câu | ASL liên tục |
| CSL-Daily | 20,654 | |
| Phoenix14T | khoảng 8,257 | |

ASL-LEX 2.0 nằm ở `/home/haipd/Handshape-GNN/data/external/asllex_official_metadata_v1/sign_props.json` (2,723 mục). Nó có Handshape, SelectedFingers, Flexion, MajorLocation, Movement, SignType, Contact…, đủ cho phonology probe.

Compute: 1 RTX 5880 dùng chung; tối đa 2 tiến trình, mỗi lúc một job GPU; disk còn khoảng 78 GB.

---

## 3. Literature review có phản biện

Tóm tắt từ `literature/` (L1–L4). Các con số đã được kiểm tra bằng raw text của bài báo. Những số chỉ đến từ trang tác giả hoặc bản tóm tắt được ghi rõ.

### 3.1 JEPA và paradigm

- **Hình học mask là siêu tham số chính.**
  - I-JEPA (1% IN): multi-block 54.2, random 17.6, một block lớn 20.2, một target 9.0.
  - V-JEPA: multi-block 72.9 vs random-tube 51.5.
- **Target lấy ở *output* teacher:** I-JEPA 67.3 vs 56.1; S-JEPA 85.3 vs 78.1.
- **Latent target thắng lớn trên đầu vào nhiễu, chiều cao, nhưng không phải luôn luôn.**
  - V-JEPA: +5.1 khi frozen, ≈ 0 khi fine-tune.
  - MaskFeat: target HOG 82.2 ≈ target DINO 82.5.
  - Skeleton2vec (2401.00921) cho rằng target contextual thắng joint và motion. AMR (CVPR'26) phản bác: toạ độ chuẩn hoá đạt 87.4.
- **Thiên kiến ngầm** (Littwin et al., NeurIPS'24): JEPA ưu tiên đặc trưng có hệ số hồi quy cao, tức đặc trưng *dự đoán được*. Phân phối mask quyết định đặc trưng nào được học.
- **Slow features** (Sobal et al.): nhiễu cố định trong clip bị học. Danh tính signer chính là loại nhiễu đó.
- **Chống collapse:**
  - EMA + stop-grad;
  - teacher đóng băng (SALT);
  - SIGReg (LeJEPA, λ = 0.05);
  - prototype/centering (DINO, iBOT).
  - SIGReg cho dự đoán token mới chỉ có bằng chứng quy mô nhỏ: PointTube-JEPA, không EMA, λ = 0.01. EMA *hại* ở đó (88.50 vs 94.08). Laya: λ < 0.02 thì collapse.
- **Probe đổi kết luận:** đổi linear sang attentive làm V-JEPA tăng +17, MAE-B +7.9, nhưng DINO chỉ +0.5.

### 3.2 Masking

- **Che trọn bộ phận từ đầu thì hại; curriculum hoặc trộn họ mask thì có lợi.**
  - SemMAE: whole-part 52.9 vs random 66.8 vs curriculum 68.7.
  - EPM: 75.71 vs 78.63 vs 79.89.
  - HPM: dễ→khó +0.46, khó→dễ −0.78.
- **Sign:** BEST (AAAI'23), MSASL100. Che *body-only* còn tệ hơn không pretrain; trộn các họ mask là tốt nhất.

  | Cấu hình | MSASL100 |
  |---|---|
  | không pretrain | 73.44 |
  | hand-only | 76.75 |
  | body-only | 68.82 |
  | trộn | 80.98 |

- **Target quan trọng hơn mask trên skeleton:** MAMP motion-aware masking +1.2, nhưng đổi target sang motion **+10.1**.
- **Rò rỉ là failure mode số một:**
  - Point-MAE: positional embedding làm lộ hình dạng (+0.7).
  - Point-M2AE: mask không nhất quán (−4.5).
  - VideoMAE: frame masking 61.5 vs tube 69.6.
  - Path-JEPA giải bằng cách lan truyền mask tới mọi token phụ thuộc.
- **Lý thuyết** (Kong et al., CVPR'23): mask buộc model học latent tối thiểu *được chia sẻ* giữa phần che và phần thấy. Coupling động học có thể chiếm chỗ latent đó.
- **Frozen nhạy với mask hơn fine-tune:** MAE LP dao động khoảng 20 điểm, FT phẳng.
- **Kỳ vọng thực tế:** +1–3 điểm frozen.

### 3.3 Skeleton SSL (NTU, linear, joint-only)

| Method | Mask | Target | Chống collapse | NTU60 xs / xv | NTU120 xs / xset |
|---|---|---|---|---|---|
| MAMP (ICCV'23) | motion-aware 90% | Δx | — | 84.9 / 89.1 | 78.6 / 79.1 |
| **S-JEPA** (ECCV'24) | motion-aware 90% | output EMA; **softmax trên chiều feature, CE + centering** (τ_t 0.06, τ_s 0.1, β 0.9) | EMA | 85.3 / 89.8 | 79.6 / 79.9 |
| GFP (ICCV'25) | motion-aware | feature phân cấp | VICReg | 85.9 / 92.0 | 79.1 / 80.3 |
| AMR (CVPR'26) | motion-aware | toạ độ chuẩn hoá | — | 87.4 / 92.3 | 81.1 / 81.9 |
| **Path-JEPA** (ECCV'26) | seed theo motion → cây con → lan truyền | latent EMA của token path-signature | EMA + centering | 87.6 / 92.9 (trang tác giả) | 81.5 / 81.8 |
| **SLiM** (2603.10648; trang dự án ghi NeurIPS'26) | **Semantic Tube Masking**, r ~ U(0.5, 0.9) | prototype iBOT + DINO-CLS | EMA + centering | 87.9 / 93.2 | 81.2 / 83.6 |

Chi tiết then chốt:
- **S-JEPA:**
  - Không có code; repo rỗng.
  - Bỏ EMA → 1.6%. MSE thay CE → 83.7. Bỏ centering → 84.5.
  - 1200 epoch, khoảng 4.5× MAMP.
- **SLiM:**
  - STM thắng joint masking +3.5.
  - Phân tích target có đối chứng: raw coordinates 86.96; feature-MSE 86.92 (rank 1.12); thêm predictor 87.59; prototype-CE 90.17.
  - Cũng ở SLiM, *nhánh token collapse vẫn cho 84.5* vì term global DINO gánh biểu diễn. ⇒ **Bắt buộc có baseline global-only.**
  - Bone-aware scaling có ích.
  - Zero-fill joint bị thiếu làm sụp còn 49.6%.
- **Kết luận phản biện:** trên skeleton, latent target chỉ thắng khi có đủ (i) target từ output teacher, (ii) chống collapse mạnh, (iii) mask khó và có cấu trúc. Ngoài các điều kiện đó, một target tường minh được chuẩn hoá tốt cũng ngang hoặc hơn.

### 3.4 Sign language SSL và pose ISLR

| Nhóm | Phương pháp | Ý chính | Số liệu then chốt |
|---|---|---|---|
| Masked pose | SignBERT+ (TPAMI'23) | token tay; mask joint/frame/clip | WLASL2000 FT 48.85; **linear 24.50** |
| | BEST (AAAI'23) | token bộ ba bằng d-VAE | FT 46.25; linear 21.19 |
| | MASA (TCSVT'24) | GCN tay/thân, motion residual, mask 90% | FT 49.06; frozen ASLC 27.96 (theo bảng của SignSeek; *không có weight công khai*) |
| | SignMAE (ICPR'26) | heatmap; chỉ tái tạo vùng tay | random 24.36 → tube 26.48 → ST-mask 40.31 → **chỉ tay 47.08** |
| Contrastive | STC (TIP) | nhánh tay và thân riêng | linear WLASL 32.35; tách nhánh +4.9 |
| Predictive coding | OpenHands DPC (2021) | dự đoán latent (loss contrastive) | DPC 94.7 > masked 91.3 (INCLUDE) |
| Multi-stream cluster | SHuBERT (ACL'25) | mặt và tay là *crop ảnh* qua DINOv2; mask span 3 frame theo từng stream; target k-means **offline** theo stream | ASLC LoRA 0.65; phonology 85.02, nhưng là **fine-tune và có trùng từ vựng** |
| Gần nhất | **SignSeek** (2609.03695) | 4 articulator; latent masked-articulator prediction | Frozen ASLC 57.66, nhưng **dùng nhãn gloss** |
| RGB + prior | **SignRep** (ICCV'25) | Hiera; mask ngẫu nhiên 80%; một vector toàn cục regress prior 3-D; **chuẩn hoá bone length theo median để tránh lộ danh tính**, kèm adversarial style loss | Frozen ASLC 49.95 (weighted) / 37.47 (avg) |
| Phonology probe | WLASL-LEX (ACL'22); SVT-ISLR; 2606.28667 | lexical holdout trên skeleton; probe frozen từng lớp; nhận thức âm vị của model SLR | — |

Bốn kết luận cho proposal:
1. **Chưa có JEPA latent-distillation label-free nào công bố cho sign skeleton.** Gần nhất: OpenHands DPC (contrastive) và SignSeek MAP (có nhãn).
2. **Khoảng trống:** skeleton SSL label-free frozen ≤ 27.96, so với RGB 49.95 và pose có nhãn 57.66. Các số này khác readout và protocol.
3. **Tay quyết định:** body-only 21.37 vs body + hands 48.54. Tách nhánh tay, chia sẻ encoder L/R và mirror đều thắng.
4. **Keypoint tay là mắt xích yếu nhất:**
   - hand AP 58–66, so với body 76;
   - MediaPipe mất tay ở 20–24% frame;
   - đổi bộ trích xuất làm số liệu lệch 6–15 điểm.

### 3.5 Định vị

| Tiêu chí | S-JEPA | SLiM | Path-JEPA | SignBERT+/BEST/MASA | SHuBERT | SignSeek | SignRep | obs_jepa B | **PhonoJEPA** |
|---|---|---|---|---|---|---|---|---|---|
| Miền | action | action | action | sign | sign | sign | sign (RGB) | sign | **sign** |
| Label-free | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | **✓** |
| Đơn vị che | joint | vùng giải phẫu × thời gian | cây con | joint / frame / clip | stream × 3 frame | articulator | khối pixel | event phần | **kênh âm vị × span, thang độ khó** |
| Chống rò rỉ phụ thuộc | — | — | ✓ lan truyền | — | — | — | — | ✓ | **✓ sở hữu joint + lan truyền + test** |
| Target phần bị che | contextual | contextual (prototype) | contextual | toạ độ / token | k-means offline theo stream | contextual | prior toàn cục | contextual + anchor | **contextual vs intrinsic tường minh theo kênh** |
| Khử danh tính | — | (bone scaling để tăng đa dạng) | — | — | — | contrastive có nhãn | bone-norm median + adversarial | — | **neutral-face, tỷ lệ tương đối, augment bất đối xứng, đo trên cả target lẫn biểu diễn** |

---

## 4. Phản biện ý tưởng gốc

### 4.1 Đóng góp phải nằm ở đâu

"Che bộ phận rồi dự đoán latent" đã có ở S-JEPA, SLiM, SignSeek và obs_jepa B. Vì vậy đóng góp phải đến từ ba chỗ:
1. *Đơn vị che* đúng với cấu trúc ngôn ngữ ký hiệu.
2. *Target* đúng cho "phần bị che".
3. *Chặn được các lối tắt* đặc thù của sign language.

### 4.2 Bảy failure mode

Bằng chứng đã cập nhật theo chẩn đoán v2 (§5).

| # | Failure mode | Bằng chứng |
|---|---|---|
| F1 | **Nội suy thời gian** | Hình bàn tay thô (toạ độ tương đối cổ tay, v1): R² 0.87 (gap 1) → 0.55 (gap 8 frame). Vector xương ngón đã chuẩn hoá thì nhiễu hơn: 0.25 → < 0. ⇒ Nội suy là shortcut ở cấp *vị trí*. Ở cấp *ngón*, target đơn-frame bị nhiễu chi phối. VideoMAE frame vs tube: 61.5 vs 69.6. |
| F2 | **Rò rỉ phụ thuộc** | Cổ tay thân trùng gốc bàn tay. Trước khi gộp kênh, SHAPE → ORI cùng frame đạt R² 0.38–0.50 (D6′, đã bỏ artifact). Khung mặt dùng chung giữa các kênh mặt (D6: UPPER → MOUTH 0.41–0.48). Path-JEPA và Point-MAE cho thấy rò rỉ phải được chặn có hệ thống. |
| F3 | **Bất định khi che trọn articulator** | Oracle từ vựng 0.27; từ các kênh khác 0.15. SemMAE whole-part 52.9 vs 66.8. BEST body-only < không pretrain. |
| F4 | **JEPA bỏ qua chi tiết khó đoán** | Lý thuyết Littwin. EX15: probe handshape R² chỉ 0.03–0.18. |
| F5 | **Signer là đặc trưng dễ đoán nhất** | Repo 0.92–0.999. D4: HEAD 0.89 signer / 0.29% gloss; mặt chưa chuẩn hoá 0.92–0.98. |
| F6 | **Collapse khi hồi quy latent** | SLiM rank 1.12. S-JEPA bỏ EMA → 1.6%. T-JEPA cần [REG]. Repo cos 0.998. |
| F7 | **Khuyết thiếu bị nhầm với bị che, và rò rỉ trước khi che** | Zero-fill → 49.6% (SLiM). Vị trí mask, quyết định mirror, tập query và phép resample đều có thể mang thông tin của phần bị che (phản biện §18, C2). |

### 4.3 Nguyên tắc thiết kế

| Nguyên tắc | Xử lý | Ở đâu |
|---|---|---|
| **P1** Mask theo ống thời gian ≥ 4 bin; không có mask 1-bin. Target ngón được làm trơn theo bin. | F1 | §6.6, §6.7 |
| **P2** **Sở hữu joint:** mỗi joint thuộc đúng một kênh. Kênh phụ thuộc khung của kênh khác thì phải bị che cùng (*closure*). Rò rỉ được đo (D6) và có ngưỡng. Query của predictor chỉ mang (loại kênh, thời gian). | F2 | §6.4 |
| **P3** Thang độ khó + curriculum dễ→khó. Không dùng riêng họ mask che trọn articulator. | F3, F4 | §6.6 |
| **P4** So sánh **target contextual vs target intrinsic tường minh** của kênh bị che. Đo danh tính signer *trong chính các target*. | F4, F5 | §6.7, §8 |
| **P5** Neutral-face; đặc trưng tỷ lệ tương đối theo median của clip (trên frame thấy được); augment hình thái bất đối xứng giữ nguyên cổ tay. | F5 | §6.3, §6.10 |
| **P6** CE có centering kiểu S-JEPA + EMA làm mặc định. Pilot so với prototype và L1 + SIGReg. Theo dõi rank. | F6 | §6.8 |
| **P7** Tách [MASK] khỏi [ABSENT]. Mask không phụ thuộc nội dung. Lan truyền mask qua resample. Query mọi ô bị che. Kiểm thử perturbation end-to-end. | F7 | §6.3, §6.6 |

---

## 5. Bằng chứng pilot (training-free, CPU) — bản v2 đã sửa

Thiết lập:
- Dữ liệu: ASL Citizen **val** (10,299 clip hợp lệ), gồm 5 signer có trên 50 clip.
- Gallery OFF: toàn bộ 40,154 clip train (35 signer khác).
- Mô tả: resample 16 bước trên các frame có tay thuận hợp lệ.
- Script: `pilot_diagnostics/diag_v2.py`, kết quả trong `diag_v2_results.json`.

> **Minh bạch.** Bản v1 của chẩn đoán có hai lỗi:
> - **Artifact:** cột MCP-giữa gần như hằng số, sau khi z-score trở thành proxy của 1/chiều dài lòng bàn tay, tức kích thước tay. Reviewer phát hiện lỗi này.
> - **So sánh lẫn biến:** biến thể "chuẩn hoá xương" của v1 đổi ba thứ cùng lúc.
>
> Kết luận của v1 rằng chuẩn hoá xương giúp +2.6 gloss và −3 signer **không còn đứng vững** (xem D1). Kết quả v1 vẫn được giữ trong `pilot_diagnostics/D*_results.json` nhưng *không* được dùng làm căn cứ.

### D1 / D5′ — Từng thay đổi một trên tokenization v2

Thiết lập: tay = CONF + LOC × 2 tay; OFF-val R@1 dùng protocol chính thức.

| Biến thể | Signer, 5 lớp ↓ | Signer, 3 lớp cân bằng gloss ↓ | OFF-val R@1 ↑ |
|---|---|---|---|
| CONF hệ lòng bàn tay, chia theo chiều dài lòng bàn tay | 0.800 | 0.902 | **12.13** |
| + chuẩn hoá từng xương (p90) | 0.803 | 0.907 | 12.06 |
| + tỷ lệ tương đối median-clip | 0.803 | 0.902 | 11.85 |
| … + mặt **thô** (UPPER, MOUTH, HEAD) | 0.996 | 0.997 | 6.24 |
| … + mặt **neutral**, mặc định v2 | 0.974 | 0.984 | 10.16 |

**Diễn giải:**
1. Chuẩn hoá xương và tỷ lệ tương đối **không** cải thiện mô tả training-free. Chúng không làm hại, nên được giữ vì lý do khử hình thái, nhưng phải kiểm định lại bằng ablation khi đã học.
2. Thêm mặt vào một mô tả cosine **làm hại** retrieval, và mặt thô đẩy signer lên 0.996. Neutral-face giảm thiệt hại (6.24 → 10.16).
3. ⇒ Trong model học, mặt cần trọng số thấp. Kênh HEAD chỉ dùng phần *động* (§6.4).

### D4 — Thông tin theo từng kênh

Probe signer: 5 lớp / 3 lớp cân bằng gloss. Retrieval: OFF-val R@1.

| Kênh | Signer 5 ↓ | Signer 3 cân bằng ↓ | R@1 ↑ |
|---|---|---|---|
| D-CONF | 0.611 | 0.743 | **4.33** |
| N-CONF | 0.582 | 0.736 | 3.77 |
| D-LOC | 0.599 | 0.817 | 0.99 |
| N-LOC | 0.561 | 0.786 | 0.99 |
| F-UPPER neutral (thô) | 0.669 (0.980) | 0.853 (0.987) | 0.12 (0.09) |
| F-MOUTH neutral (thô) | 0.645 (0.923) | 0.847 (0.959) | 0.33 (0.24) |
| F-HEAD | **0.894** | **0.940** | 0.29 |

**Diễn giải:**
1. **Neutral-face giảm mạnh rò rỉ signer** của kênh mặt (0.98 → 0.67). Đây là bằng chứng cho P5.
2. **HEAD là vân tay signer** với gần như không có thông tin gloss. ⇒ Đầu vào HEAD chỉ giữ thành phần tương đối/động, và target của HEAD chỉ là Δ.
3. LOC mang nhiều signer (0.82 khi cân bằng) hơn CONF (0.74). Lý do là tư thế và độ dài tay. ⇒ Augment hình thái nhắm vào LOC (§6.10).

### D2′ — Thang dự đoán được của cấu hình tay thuận

| Bậc | Ngữ cảnh → mục tiêu | R² |
|---|---|---|
| Hình bàn tay thô (v1; toạ độ tương đối cổ tay, chưa chuẩn hoá, tức *cấu hình + hướng*, không phải vị trí) | ← *t ± g* (g = 1/2/4/8 frame) | 0.87 / 0.82 / 0.71 / 0.55 |
| Ngón, đơn-frame | 15 xương ngón tại *t* ← *t ± g* | 0.25 / 0.13 / 0.01 / −0.11 |
| Nội-tay | xương ngón giữa, áp út, út ← xương cái + trỏ + đa giác lòng bàn tay (ridge / kNN) | 0.27 / 0.34 |
| Chéo kênh, cấp clip | CONF ← D-LOC | 0.04 |
| | ← D-LOC + tay không thuận | **0.15** |
| | ← + mặt | 0.13 |
| | ← chỉ N-CONF | 0.09 |
| Trần từ vựng | ← trung bình CONF cùng gloss của signer khác (oracle) | **0.27** |
| Offset signer | ← trung bình CONF các clip *khác gloss* của cùng signer | 0.007 |

**Diễn giải** (đã sửa theo phản biện):
1. Xương ngón đơn-frame **rất nhiễu**. ⇒ Target intrinsic phải được làm trơn theo bin. Ở đây target latent của teacher có lợi thế tự nhiên là lọc nhiễu, và đây là một điểm phải kiểm định (RQ5).
2. Khoảng một nửa phần cấu hình tay *do từ vựng quyết định* (0.15 trên 0.27) dự đoán được từ các kênh khác bằng kNN thô. Tín hiệu chéo kênh **có nhưng khiêm tốn**, nhất quán với CIP 0/36 cho forecasting.
3. Hàng "offset signer 0.007" chỉ cho biết **không có một handshape trung bình đặc trưng cho từng signer**. Nó **không** chứng minh CONF không mang danh tính: D4 cho thấy CONF vẫn nhận ra signer ở 0.61–0.74. v1 đã đọc sai chỉ số này.

### D6 — Kiểm toán rò rỉ tức thời giữa các kênh v2

R² ridge / kNN cùng frame, cross-signer, đã winsorize và lọc frame lỗi.

| nguồn ↓ / đích → | D-CONF | D-LOC | N-CONF | N-LOC | F-UP | F-MO | F-HD |
|---|---|---|---|---|---|---|---|
| D-CONF | — | 0.04 / 0.06 | 0.06 / 0.09 | ≤ 0 | ≤ 0 | ≤ 0 | < 0 |
| D-LOC | ≤ 0 | — | ≤ 0 | 0.13 / 0.16 | ≤ 0 | ≤ 0 | < 0 |
| N-CONF | 0.06 / 0.05 | ≤ 0 | — | 0.05 / 0.08 | ≤ 0 | ≤ 0 | < 0 |
| N-LOC | ≤ 0 | 0.13 / 0.16 | ≤ 0 | — | ≤ 0 | ≤ 0 | < 0 |
| F-UP | < 0 | < 0 | < 0 | < 0 | — | **0.48 / 0.41** | ≤ 0.08 |
| F-MO | < 0 | < 0 | < 0 | < 0 | 0.23 / 0.16 | — | 0.07 / 0.12 |
| F-HD | < 0 | < 0 | < 0 | < 0 | 0.10 / 0.06 | **0.32 / 0.29** | — |

**D6′ (biến thể tách SHAPE/ORI, `D6_split_test_results.json`):**
- SHAPE → ORI: 0.38 / 0.50.
- ORI → SHAPE: 0.21 / 0.29.
- Trước khi bỏ cột artifact: 0.47 / 0.69.

**Quyết định thiết kế suy ra từ D6:**
1. **Gộp handshape và orientation thành một kênh CONF.** Trong 2-D, hai thành phần này đan xen qua phép chiếu, và tách ra thì vi phạm sở hữu joint (khung lòng bàn tay). Tách SHAPE/ORI chỉ còn là ablation.
2. Giữa các kênh tay, rò rỉ tức thời ≤ 0.16, nên **không cần closure**.
3. Các kênh mặt ghép cặp (trung bình 0.3–0.48; theo cột tới 0.9, xem D8) vì tư thế đầu chiếu lên mọi điểm mặt. Thêm vào đó, UPPER và MOUTH được tính trong khung mắt, mà khung mắt thuộc HEAD. ⇒ **Closure: che HEAD thì che cả mặt.** v2.1 bỏ luôn các mask chỉ che UPPER hoặc MOUTH (D8).

### D7 — Khung toạ độ cho CONF: hệ lòng bàn tay hay hệ thân

Thiết lập: CONF chỉ khác nhau ở khung toạ độ. Mô tả cho cả hai tay là CONF + LOC (đã tính tỷ lệ tương đối). Script: `diag_v2b.py`.

| Khung CONF | Chuẩn hoá xương | OFF-val R@1, hai tay | R@1 chỉ D-CONF | Signer 5 (hai tay) | R² nội suy xương ngón, gap 1 / 4 |
|---|---|---|---|---|---|
| hệ lòng bàn tay (xoay theo trục cổ tay → MCP giữa) | không | 11.98 | 4.59 | 0.799 | 0.19 / −0.05 |
| hệ lòng bàn tay | p90 | 11.85 | 4.33 | 0.803 | 0.25 / 0.01 |
| hệ thân/ảnh (không xoay) | không | 11.96 | **4.94** | 0.817 | 0.37 / 0.13 |
| **hệ thân/ảnh** | **p90** | **12.65** | 4.49 | 0.816 | **0.41 / 0.17** |

**D7b — Anchor vai theo từng frame hay theo median của clip** (`diag_v2c.py`). Cả hai cách đều leak-free, vì anchor không bao giờ bị che.

| Anchor | OFF-val R@1, hai tay | R@1 chỉ LOC | Signer 5 / 3 cân bằng (hai tay) |
|---|---|---|---|
| theo frame (c_t, σ_t) | 12.65 | 1.62 | 0.816 / 0.905 |
| **median clip** | **12.83** | 1.64 | 0.811 / 0.892 |

**Diễn giải và quyết định:**
1. Khác biệt về retrieval nhỏ (≤ 0.8 điểm). Phép xoay theo trục lòng bàn tay **cộng thêm nhiễu**: R² nội suy giảm từ 0.41 xuống 0.25, vì trục lòng bàn tay 2-D rung theo keypoint. Phép xoay này cũng *loại bỏ* hướng tuyệt đối của ngón, vốn là thông tin orientation có nghĩa âm vị.
2. ⇒ Mặc định **CONF trong hệ thân** (căn theo đường vai, tức anchor, nên không rò rỉ), có chuẩn hoá xương p90, và **anchor là median của clip** (ít nhiễu hơn, chênh lệch nhỏ). Hệ lòng bàn tay và anchor theo frame chỉ còn là ablation A4.
3. B0 = 12.83 R@1 OFF-val trước khi hiệu chỉnh tỷ lệ khung hình. **D8 thay bằng 13.46**. Con số 16.84 của v1 *không tái lập được* khi bỏ artifact và các đặc trưng kích thước tay, nên không dùng nữa.

### D8 (v2.1) — Chạy lại sau phản biện vòng 2

Thay đổi so với D6/D7:
- **Hiệu chỉnh tỷ lệ khung hình:** x ← x·W/H theo từng video. ASL Citizen có 50,243 video 640×480 và 215 video 960×540.
- Khung thân là phép xoay theo góc median của đường vai.
- Cằm được chuyển sang F-MOUTH. F-HEAD chỉ giữ thành phần tương đối.
- Anchor lấy median theo clip.
- Script `diag_v21.py`, kết quả `diag_v21_results.json`.

**D7′ — Khung CONF sau khi hiệu chỉnh tỷ lệ khung hình:**

| Khung | OFF-val R@1, hai tay | R@1 chỉ D-CONF | Signer 5 / 3 cân bằng | R² nội suy xương ngón, gap 1/2/4 |
|---|---|---|---|---|
| lòng bàn tay | 12.69 | 4.36 | 0.789 / 0.885 | 0.25 / 0.14 / 0.02 |
| **thân** | **13.46** | **4.80** | 0.809 / 0.892 | 0.44 / 0.33 / 0.21 |
| thân + mặt (UPPER, MOUTH có cằm, HEAD tương đối) | 13.08 | — | 0.928 / 0.971 | — |
| thân + UPPER + MOUTH | 12.92 | — | 0.910 / 0.962 | — |

**D6″ — Kiểm toán rò rỉ theo từng cột** (khung thân, hiệu chỉnh tỷ lệ). Mỗi ô ghi R² ridge *trung bình / lớn nhất* trên các cột của kênh đích. Kết quả gộp toàn bộ (pooled) và kết quả trong từng clip (đã trừ trung bình theo clip) gần như trùng nhau; chỉ ghi bản pooled, trừ khi khác biệt đáng kể.

| nguồn ↓ / đích → | D-CONF | D-LOC | N-CONF | N-LOC | F-UP | F-MO | F-HD |
|---|---|---|---|---|---|---|---|
| D-CONF | — | 0.01 / 0.21 | 0.12 / 0.31 | −0.04 / 0.16 | ≈0 | ≈0 / 0.05 | ≈0 |
| D-LOC | 0.03 / **0.15** | — | 0.01 / 0.10 | 0.12 / **0.47** | ≈0 | 0.03 / 0.10 | 0.03 / 0.10 |
| N-CONF | 0.11 / 0.32 | ≈0 / 0.18 | — | 0.04 / 0.27 | ≈0 | ≈0 | ≈0 |
| N-LOC | ≈0 / 0.12 | 0.10 / 0.45 | 0.07 / 0.23 | — | ≈0 | 0.03 / 0.10 | 0.03 / 0.10 |
| F-UP | ≈0 | < 0 | ≈0 | < 0 | — | 0.35 / 0.67 | 0.24 / **0.77** |
| F-MO | ≈0 | < 0 | ≈0 | < 0 | 0.18 / 0.54 | — | 0.31 / **0.91** |
| F-HD | ≈0 | < 0 | ≈0 | < 0 | 0.13 / 0.36 | 0.47 / **0.84** | — |

**Diễn giải và quyết định (thay cho phần tương ứng của D6/D7):**
1. **Khung thân thắng khung lòng bàn tay** (13.46 vs 12.69) sau khi hiệu chỉnh tỷ lệ. Mặc định giữ khung thân.
   - Phản biện vòng 2 lưu ý R² nội suy cao hơn trong khung thân phản ánh chủ yếu *chuyển động xoay chậm của cả bàn tay* được giữ lại, không phải "ít nhiễu hơn". Nói cách khác, đó là thành phần dễ nội suy, tức đúng loại shortcut F1.
   - ⇒ Giữ P1 (span ≥ 4 bin). R² nội suy ở gap 4 vẫn khoảng 0.2, đủ thấp.
2. **B0 chính thức = 13.46 R@1 OFF-val.** Con số này thay cho 12.83 trước khi hiệu chỉnh tỷ lệ.
3. **Kênh tay:**
   - LOC → CONF tối đa 0.15 theo cột (v2 ghi "≤ 0.06" là trung bình cột trong khung lòng bàn tay; nay sửa).
   - Ghép cặp giữa hai tay: LOC ↔ LOC tối đa 0.47, CONF ↔ CONF tối đa 0.32. Đây là cấu trúc âm vị thật (ký hiệu đối xứng/song song), không phải rò rỉ phụ thuộc. Nó được xử lý bằng mask "hai tay" ở R4 và phân tầng theo SignType.
4. **Kênh mặt ghép cặp rất mạnh theo từng cột** (tối đa 0.67–0.91) qua hình chiếu tư thế đầu.
   - ⇒ **Bỏ F-UP và F-MO khỏi R3.** Mặt chỉ bị che *nguyên khối* (R4-mặt) hoặc trong khối thời gian (R2). Khi đó R3 chỉ còn các kênh tay (§6.6).
   - Thêm mặt vào mô tả hai tay giờ chỉ làm giảm 0.4 điểm (trước khi sửa là 1.7), nhưng làm signer probe tăng lên 0.93. ⇒ Cánh **chỉ-tay** C-PL-both-hands ở P2 là bắt buộc.

---

## 6. Phương pháp đề xuất: PhonoJEPA (v2)

### 6.1 Luận điểm trung tâm

Trong ngôn ngữ ký hiệu, cấu hình tay, vị trí/chuyển động và non-manual bị **từ vựng ràng buộc lẫn nhau**. Một biểu diễn tốt phải biết *cho trước một số kênh thì các kênh còn lại có thể là gì*, đồng thời vẫn giữ được *chi tiết của chính từng kênh*.

PhonoJEPA kiểm định điều này một cách có kiểm soát:
1. Biểu diễn skeleton thành **kênh âm vị học có sở hữu joint rõ ràng**.
2. **Che kênh** theo thang độ khó, với lấy mẫu không phụ thuộc nội dung.
3. So sánh hai cách biểu diễn "phần bị che":
   - **contextual**: latent của teacher;
   - **intrinsic tường minh**: mô tả của chính kênh.
4. Đo cả hiệu năng lẫn **danh tính signer trong target và trong biểu diễn**.

### 6.2 Sơ đồ (v2)

```
keypoints 2-D (T×133×2) + scores
  │ §6.3 thứ tự leak-free: crop theo timestamp → quyết định tay thuận + mirror → resample tốc độ (ghi support)
  │      → anchor vai → lấy mẫu mask (chỉ phụ thuộc độ dài clip) → lan truyền mask qua support
  │      → tính đặc trưng trên GPU, thống kê chỉ từ frame thấy được
  ▼
7 kênh × 32 bin (bin = 2 frame ở 25 fps; cửa sổ 64 frame = 2.56 s)
  D-CONF  D-LOC  N-CONF  N-LOC  F-UPPER  F-MOUTH  F-HEAD
  │
  ├─ Teacher (EMA) trên T1, T2 (view đủ, augment nhẹ)
  │    → token output (trung bình LN của 4 block cuối) ⇒ target contextual P^ctx (softmax trên chiều, centering)
  │    → mô tả kênh tường minh x̄, Δx̄ (z-score)    ⇒ target intrinsic y^int
  └─ Student trên near (căn thời gian với T1) và far (căn với T2), bị che theo PLCM, augment hình thái độc lập
       tokenizer → encoder E (chỉ token thấy) → predictor (query = loại kênh + bên + thời gian, mọi ô bị che)
       → head ctx: CE(P^ctx, p^ctx);  head int: smooth-L1(y^int, ŷ^int)
  [Gate G1] CLS student(near, far) ⇄ CLS teacher(T1, T2): DINO-CLS + KoLeo theo từng view
```

### 6.3 Pipeline leak-free và sở hữu joint (P2, P7)

**Thứ tự xử lý.** Thứ tự này là một phần của đặc tả; test end-to-end kiểm tra đúng thứ tự này.

1. **Pixel aspect:** x ← x·W, y ← y·H (W, H lấy từ metadata video, cache sẵn).
2. **Crop thời gian không phụ thuộc nội dung:**
   - ISLR: cửa sổ 64 frame ở 25 fps, đặt theo *timestamp* quanh tâm clip, jitter ±8 frame khi train. Clip ngắn hơn được pad bằng [ABSENT]; vị trí pad chỉ phụ thuộc độ dài clip.
   - Câu dài: cửa sổ ngẫu nhiên.
   - *Bỏ "vùng hoạt động A" của v1*, vì nó lộ thông tin cổ tay của phần bị che.
3. **Tay thuận:** quyết định trên cả cửa sổ từ chuyển động cổ tay thân, rồi mirror (lật x quanh trục vai, hoán vị L/R cho thân, tay và mặt theo cặp đối xứng iBUG).
   - Đây là **rò rỉ 1 bit được khai báo**.
   - Test: thay nội dung bị che bằng nội dung *thật* của một clip khác (swap-in), báo cáo tỷ lệ lật theo từng họ mask. Ngưỡng ngẫu nhiên tuyệt đối không dùng được vì với R4 hai tay thì quyết định lật là hợp lý.
   - A8 có thêm cánh **lật L/R ngẫu nhiên, dùng chung cho teacher và student** (không dựa vào nội dung) để đo giá trị của canonical hoá tay thuận.
   - Với dữ liệu liên tục, quyết định theo *video*, không theo cửa sổ.
4. **Tốc độ:** resample bằng **nội suy tuyến tính**, s ~ U(0.8, 1.25), dùng chung cho cặp teacher/student. Ghi lại *support* (các frame nguồn) của từng frame đích.
5. **Anchor:** chỉ vai (c, σ, và góc đường vai cho khung thân), lấy **median theo clip** (D7b); không bao giờ bị che, không bao giờ là target. Anchor *theo frame* là **rò rỉ có chủ đích** (vai 2-D dịch theo khi giơ tay, nên mang thông tin LOC) và chỉ xuất hiện trong A8.
6. **Lấy mẫu mask** trên lưới 7 × 32. Chỉ phụ thuộc số bin hợp lệ (độ dài clip), **không** phụ thuộc vị trí tay hay việc detector có thấy hay không.
7. **Lan truyền mask:**
   - Với mỗi kênh, tập frame nguồn bị ẩn = hợp các support của các bin bị che.
   - Bin *thấy được* của cùng kênh mà support giao với tập đó thì chuyển thành *context-drop*: không vào student, không làm target.
   - Lan truyền theo từng kênh, vì sở hữu joint tách bạch giữa các kênh.
   - **Support = support của resample ∪ stencil thời gian của đặc trưng.** Đặc trưng có stencil (ví dụ Δw của LOC dùng frame t−1) chỉ được tính khi *mọi* frame trong stencil thấy được; nếu không thì đặt 0 và bật cờ. Nhờ vậy frame thấy được đầu tiên sau span không lộ vị trí cuối của span.
   - **R1 (nhóm ngón):** lan truyền theo *nhóm joint*, không theo kênh; conf của joint ẩn đặt về 0 và cờ joint-hidden bật.
   - **Closure áp dụng cho mọi họ mask, kể cả R0:** ô F-HEAD bị che ⇒ ô F-UPPER và F-MOUTH cùng bin bị che.
8. **Đặc trưng tính trên GPU theo view.** Mọi thống kê theo clip (p90 xương, neutral face, median log-scale) chỉ dùng **frame thấy được của kênh đó** trong view đó. Teacher dùng toàn bộ frame.
9. **Token hoá:** token thấy được vào encoder; token bị che bị loại; token detector mất dấu nhận [ABSENT].
10. **Query predictor:** **mọi ô bị che**. Ô có [ABSENT] ở phía teacher vẫn được query, nhưng trọng số target bằng 0.

**Sở hữu joint:**

| Kênh | Joint sở hữu | Khung/chuẩn hoá phụ thuộc | Closure |
|---|---|---|---|
| D-CONF / N-CONF | 21 joint bàn tay (theo hand root) | anchor vai; thống kê của chính kênh | — |
| D-LOC / N-LOC | cổ tay thân và khuỷu tay thân của bên đó | anchor vai | — (D8: LOC → CONF ≤ 0.15 theo cột) |
| F-UPPER | 10 điểm lông mày, 8 điểm mí mắt | **khung mắt** (thuộc F-HEAD) | được che riêng; F-HEAD bị che thì che cả mặt |
| F-MOUTH | 20 điểm miệng + **cằm 8** (độ mở hàm thuộc miệng; vòng 2) | khung mắt | như trên |
| F-HEAD | 4 khoé mắt, mũi 27–35 | anchor vai | **che F-HEAD ⇒ che cả mặt** |
| Anchor | 2 vai | — | không bao giờ che |

> **Test bắt buộc (P0)**, chạy cho mọi họ mask, mọi baseline, và có cả speed augmentation:
> 1. *End-to-end hidden-source perturbation*: ghi đè ngẫu nhiên các frame nguồn và joint thuộc ô bị che *trước* crop, resample và lấy mẫu mask (cùng seed). Đầu ra student phải **bit-identical**, trừ trường hợp quyết định tay thuận bị lật; trường hợp này được đếm riêng.
> 2. *Content-independence của sampler*: với cùng seed và cùng độ dài clip, mask phải giống hệt nhau khi nội dung thay đổi.

### 6.4 Đặc trưng kênh (7 loại × 32 bin = 224 token)

**Ký hiệu:**
- Khung thân: xoay theo đường vai, gốc c, đơn vị σ.
- L = ‖h₉ − h₀‖; L_eff = max(L, 0.1σ). Dùng sàn số học thay vì loại bỏ frame, vì lòng bàn tay chĩa về camera là orientation có thật (YOU, HE/SHE…).

| Kênh | Đặc trưng mỗi frame (khung thân) | Chiều/bin |
|---|---|---|
| **H-CONF** (handshape + orientation) | đa giác lòng bàn tay P₁, P₅, P₁₃, P₁₇ = (h_k − h₀)/L_eff (8); 15 xương ngón chia cho p90 (frame thấy được) của từng xương (30); trục lòng bàn tay u (2); dấu lòng/mu bàn tay (1); log(L_eff/σ) − median (1); cờ co rút 1 − L/L_eff (1); conf 21; cờ joint-hidden 21 (cho R1) | 2 × 85 = 170 |
| **H-LOC** | (w − c)/σ (2); (e − c)/σ (2); Δw (2); conf (2) | 2 × 8 = 16 |
| **F-UPPER** | 18 điểm trong khung mắt trừ neutral (36); conf (18) | 2 × 54 = 108 |
| **F-MOUTH** | 20 điểm miệng + cằm trong khung mắt trừ neutral (42); conf (21) | 2 × 63 = 126 |
| **F-HEAD** (chỉ thành phần tương đối/động, theo D4) | (gốc khung mắt − c)/σ trừ median (2); log(IOD/σ) − median (1); roll cos/sin (2); mũi 30 trong khung mắt trừ median (2); conf (3) | 2 × 10 = 20 |

**Tokenizer:**
- CONF dùng chung cho D/N (side embedding); LOC dùng chung; ba tokenizer mặt riêng.
- MLP_κ: in → 256 → d, GELU, LN.
- Token = MLP(x) + e_type + e_side.
- [ABSENT] ≠ [MASK].

### 6.5 Kiến trúc (v2: một encoder)

Bỏ hai tầng E_c/E_f của v1 (phản biện C4/M7).

| Mô-đun | S (screening) | M (full) |
|---|---|---|
| Encoder E: Transformer, full attention trên token thấy được + [CLS], RoPE thời gian, e_type/e_side | 6 block, d = 256, 4 head | 10 block, d = 384, 6 head |
| Predictor: narrow; đầu vào = Linear(d → d_p)(token thấy) cộng query [MASK] + e_type + e_side + RoPE(b); **không có toạ độ** | 4 block, d_p = 128 | 6 block, d_p = 192 |
| Head contextual | Linear(d_p → d) rồi softmax trên d chiều (kiểu S-JEPA) | |
| Head intrinsic | 5 linear head theo loại kênh, d_p → chiều target (CONF 2 × 43, LOC 2 × 6, UP 2 × 36, MO 2 × 42, HD 7) | |
| Tham số encoder | ≈ 5 M | ≈ 18 M |

**Teacher** = EMA của (tokenizer, E). **Đánh giá** dùng encoder teacher.

### 6.6 PLCM v2 — Predictability-Ladder Channel Masking

| Họ | Bậc (§5) | Che gì | Tham số |
|---|---|---|---|
| **R1** Nhóm ngón (nội-CONF, token khuyết một phần) | trung bình: R² 0.27–0.34 | PIP/DIP/TIP của nhóm ngón G (MCP và lòng bàn tay vẫn thấy) qua một span. Token CONF vẫn vào encoder với cờ joint-hidden, *và* được query làm target. | H ∈ {D .75, N .25}. G ∈ {cái, trỏ, giữa, áp út, út, trỏ+giữa, giữa+áp út+út, cái+trỏ}. Span U{4..8} bin. |
| **R2** Khối thời gian | dễ–trung bình | cả 7 kênh trong một khối | 1–2 khối × U{4..8} bin |
| **R3** Span một kênh tay (*phonological channel masking*) | khó: CONF ← các kênh khác 0.04–0.15 (cấp clip); rò rỉ theo cột ≤ 0.15 (D8) | một kênh tay qua một span dài; các kênh khác, kể cả kênh còn lại của cùng tay, vẫn thấy | κ: D-CONF .35, D-LOC .25, N-CONF .25, N-LOC .15. Span U(0.3, 1.0)·T_hợp lệ, tối thiểu 4 bin. **Không có kênh mặt** (D8: các kênh mặt ghép cặp tới 0.9 theo cột). |
| **R4** Span articulator (*anatomical*, ý tưởng gốc) | khó nhất (≤ trần từ vựng 0.27) | mọi kênh của một articulator | {tay thuận .40, tay không thuận .15, **hai tay** .15, mặt (gồm HEAD) .30}. Span U(0.4, 1.0)·T. |
| R0 Ống ngẫu nhiên | — | ống (kênh, 2 bin) ngẫu nhiên | chỉ dùng làm đối chứng |

**Hai view mỗi clip:**
- **near** = R2, cộng thêm R1 khi gate G2 được bật, tới khi đạt r ~ U(0.4, 0.7) số ô hợp lệ.
- **far** = R3 (PLCM-lite, dùng trong MVM). Khi bật gate G2 (PLCM đầy đủ): R4 với xác suất 0.4, còn lại R3, rồi thêm R3 tới khi đạt r ~ U(0.3, 0.6).
- **Hàng R0 và hàng R4 của ma trận nhân tố** dùng *cùng một họ* cho cả hai view, với cùng phân phối tỷ lệ (near U(0.4, 0.7), far U(0.3, 0.6)), cùng context dropout và cùng số pass. Nhờ vậy ba hàng chỉ khác nhau ở họ mask.
- Thuật toán budget: nhận proposal nếu thêm ít nhất một ô mới mà không vượt budget; dừng sau 10 proposal liên tiếp không thêm ô nào.
- Context dropout 10%.
- Mỗi view giữ ít nhất 25% ô làm ngữ cảnh.
- Mask "hai tay" (R4) phá shortcut "tay không thuận là ảnh gương". Tuy nhiên tính đối xứng (điều kiện Battison) cũng là âm vị học thật, nên đánh giá **phân tầng theo SignType** thay vì coi mọi đối xứng là shortcut.

**Curriculum** (chỉ bật khi qua gate G2):
- w_far(e) = 0.3 + 0.7(e/E)².
- Cận trên span của R3/R4 tăng từ 0.5T lên T trong 30% đầu.

```python
def sample_views(n_valid_bins, rng, epoch=None, E=None, curriculum=False, use_R1=False, use_R4=False):  # content-independent
    cells = grid(7, n_valid_bins)
    near = Budget(cells, rate=rng.uniform(0.4, 0.7))
    while near.open():                                   # MVM: R2 only; gate G2 enables R1 (+ curriculum)
        near.propose(R1_fingers(rng) if (use_R1 and rng.random() < 0.5) else R2_block(rng))
    max_span = (0.5 + 0.5 * min(1, epoch / (0.3 * E))) if curriculum else 1.0
    far = Budget(cells, rate=rng.uniform(0.3, 0.6))
    far.propose(R4_articulator(rng, max_span) if (use_R4 and rng.random() < 0.4) else R3_channel(rng, max_span))  # MVM: R3 only
    while far.open():
        far.propose(R3_channel(rng, max_span))
    for v in (near, far):
        v.context_dropout(rng, p=0.10); v.require_visible(0.25)
    return near, far          # later: propagate(v, resample_support) -> extra context-drops per channel
```

### 6.7 Target cho phần bị che

Câu hỏi khoa học RQ2 được trả lời qua ma trận nhân tố §9.

1. **Contextual** (theo S-JEPA, được chứng minh trên skeleton):
   - Lấy h̄ = LN(¼ Σ 4 block cuối, mỗi block đã LN) của teacher, tại vị trí bị che.
   - Target: P^ctx = softmax((h̄ − center)/τ_t) trên d chiều. Center là EMA theo batch với β = 0.9; τ_t = 0.06.
   - Student: p^ctx = softmax(H_ctx(q)/τ_s), τ_s = 0.1. Loss CE.
   - Các phương án khác do pilot P1 quyết định:
     - (i) prototype head kiểu iBOT/SLiM, K = 4096, centering;
     - (ii) L1 trên target LN + SIGReg (λ = 0.02, 1024 slice, BN-MLP projector), có và không có EMA.
2. **Intrinsic tường minh** (thay cho intrinsic-latent của v1):
   - Với mỗi loại kênh: y^int_{κ,b} = [x̄_{κ,b}; x̄_{κ,b} − x̄_{κ,b−1}].
     - x̄ là trung bình trên **3 bin liền kề (b−1, b, b+1), tức 6 frame**, lấy từ đặc trưng *view teacher*, đã bỏ conf và cờ. Lý do: xương ngón đơn-frame có khoảng 60% phương sai là rung (R² nội suy gap-1 chỉ khoảng 0.4). Làm trơn ở phía teacher không rò rỉ.
     - Phần Δ được đặt trọng số 0 khi bin b−1 là [ABSENT] hoặc pad. Loss lấy trung bình theo chiều trong mỗi loại kênh, rồi trung bình theo token.
     - z-score theo thống kê của corpus train.
     - F-HEAD chỉ dùng phần Δ.
   - Loss: smooth-L1 (β = 0.5).
   - Token R1: target là CONF đầy đủ.
   - Lý do chọn bản tường minh (phản biện C4): bản latent của v1 có receptive field chồng lên ngữ cảnh, head không khớp loại đầu vào, và Sinkhorn áp prior đều. Bản tường minh **không collapse**, diễn giải được, và đo được lượng danh tính signer mang theo (D4). Đây cũng là dạng "target cục bộ" đã có bằng chứng: MAMP (+10.1), MaskFeat-HOG, và JH3D trong repo.
3. **(Ablation) intrinsic rời rạc:** k-means theo loại kênh (K ∈ {64, 128}) trên x̄ của corpus train; target là mã cụm; loss CE. Đây là phiên bản "SHuBERT theo kênh âm vị trên skeleton". Đánh giá bằng **AMI** với ASL-LEX, so với k-means trên mô tả thô và với khởi tạo ngẫu nhiên có cùng K.

**Loss token:**

$$\mathcal{L}_{tok}=\frac{\sum_{v}\sum_{(\kappa,b)\in M_v}\omega_{\kappa,b}\big[\mathrm{CE}(P^{ctx},p^{ctx})/\bar\ell_{ctx}+\alpha_{int}\,\mathrm{SL1}(y^{int},\hat y^{int})/\bar\ell_{int}\big]}{\sum\omega}$$

- Mỗi thành phần được chia cho giá trị trung bình chạy của chính nó (ℓ̄), để hai loại target có tỷ trọng gradient tương đương. **ℓ̄ được đóng băng sau warmup**, vì ℓ̄_ctx chứa entropy của teacher: nếu để chạy, trọng số ctx sẽ tăng khi teacher sắc dần, tạo vòng phản hồi về phía collapse. Theo dõi phần KL (CE − H(P^ctx)) để chẩn đoán. α_int = 1.
- **Đối chứng cho H2** (thay cánh "×2" của v2, vốn chỉ là phép đổi thang dưới AdamW):
  - quét α_int ∈ {0.5, 1, 2};
  - cánh **ctx + shuffled-int**: y^int bị hoán vị giữa các clip có cùng loại kênh. Cánh này giữ nguyên lượng gradient và tính chính quy hoá, nhưng phá nội dung. Nếu ctx + int > ctx + shuffled-int thì lợi ích đến từ *nội dung* của kênh bị che.
- ω = r^T · γ. **Chuẩn hoá Σω theo batch**, không theo mẫu, để clip mất nhiều tay không tạo loss phương sai cao:
  - r^T = độ tin cậy phía teacher, bằng 0 nếu [ABSENT];
  - γ_tay = 1, γ_mặt = 0.5 (D1: mặt làm nhiễu retrieval).
  - Chuẩn hoá theo nhóm.

### 6.8 Global term (gate G1), chống collapse và chẩn đoán

**Global (chỉ bật khi qua G1):**
- CLS student (near, far) khớp với CLS teacher (T1, T2) bằng CE softmax trên chiều, có centering.
- **KoLeo theo từng view** (λ = 0.05). DINOv2 cho thấy KoLeo cải thiện retrieval; áp riêng từng view để không đẩy các view của cùng clip ra xa nhau (phản biện m2).
- Bỏ các view G1/G2 không che của v1 để tiết kiệm compute.

**Chống collapse mặc định:**
- EMA m: 0.996 → 1 (cosine).
- Centering cho target CE; τ_t warmup 0.04 → 0.06.
- Không dùng Sinkhorn trong mặc định. Nếu dùng prototype, Sinkhorn phải loại [ABSENT]/pad.

**Chẩn đoán ghi ở mọi lần eval:**
- loss theo họ mask và theo kênh;
- entropy target; độ lệch chuẩn target theo chiều;
- RankMe và LiDAR (held-out);
- mean-cos CLS;
- signer probe;
- **probe signer trên chính không gian target** (P^ctx vs y^int).

**Báo động collapse:** mean-cos > 0.9, hoặc RankMe < 10% của d.

### 6.9 Cơ sở lý thuyết (là heuristic, không phải định lý)

1. **Latent chia sẻ** (Kong et al.):
   - R3 buộc học ràng buộc chéo kênh (từ vựng, âm vị).
   - R2 buộc học động học.
   - R1 buộc học hình thái bàn tay.
   - Sở hữu joint loại bỏ coupling động học tầm thường khỏi latent chia sẻ.
2. **Thiên kiến "dự đoán được"** (Littwin): đặc trưng nào có hệ số hồi quy cao dưới phân phối mask thì được học trước. Một *hỗn hợp* họ mask cho mỗi loại đặc trưng mong muốn ít nhất một họ mà nó dự đoán được.
   - Phản biện m6: hệ số hiệu dụng của hỗn hợp là tỷ số của các hỗn hợp, không phải hỗn hợp của các tỷ số. Vì vậy đây là trực giác thiết kế, và chỉ được xác nhận qua ablation A1.
3. **Slow feature** (Sobal):
   - Thành phần phụ thuộc hình thái mà student và teacher thấy với *hình thái độc lập* thì khó dự đoán hơn, nên bị nén.
   - Target intrinsic tường minh đã chuẩn hoá chứa ít danh tính hơn target contextual hay không là một **giả thuyết đo được**: probe signer trên không gian target, §8 E3b.
4. **Đẳng hướng cho retrieval:** centering cộng KoLeo (hoặc SIGReg) đẩy embedding về phân phối đều. Kỳ vọng thu hẹp khoảng cách mean-pool vs whitened; trong repo khoảng cách này rất lớn (JH3D 12.5 vs 39.2 LOSO).

### 6.10 Augmentation v2 (giữ cổ tay, phản biện M4)

| | Teacher T1, T2 | Student near, far (cùng trục thời gian với T1, T2) |
|---|---|---|
| Camera | xoay ±5°, scale ±5% | xoay ±10°, scale ±10%, shear ±0.1 |
| Hình thái (độc lập giữa student và teacher) | không | (i) **giải lại khuỷu tay bằng IK 2 đoạn**, scale cánh tay và cẳng tay U(0.9, 1.1), **giữ nguyên cổ tay** so với thân và mặt; (ii) độ rộng vai U(0.95, 1.05) quanh c, giữ cổ tay; (iii) đốt ngón U(0.9, 1.1) bằng FK từ MCP; (iv) mặt scale dị hướng U(0.95, 1.05) quanh gốc khung mắt |
| Nhiễu và mất dấu | không | N(0, (σ₀(1 − c))²), σ₀ = 0.02σ cho tay; 5% frame tay → [ABSENT] |
| Tốc độ | s dùng chung với student tương ứng | như teacher; ±20% cho dữ liệu liên tục, vì tốc độ có thể mang nghĩa thể (aspect) |

Augment hình thái chỉ bật khi qua gate G3. Khi đó student và teacher có hình thái độc lập. Giới hạn trung thực: khác biệt *tầm với* trong không gian trung tính giữa các signer (phong cách) không bị augment này khử.

### 6.11 Phiên bản tối thiểu (MVM) và các gate mở rộng

**MVM**, dùng ở P2:
- token 7 kênh; encoder S;
- mask ∈ {R0, R4 (ống kiểu SLiM), PLCM-lite = near R2 / far R3};
- target ∈ {ctx, int, ctx + int};
- EMA + CE kiểu S-JEPA;
- **không** global term, **không** R1, **không** curriculum, augment nhẹ đối xứng;
- 3 seed.

**Các gate,** mỗi gate được bật thì giữ lại:

| Gate | Thêm gì | Điều kiện chấp nhận |
|---|---|---|
| G1 | Global DINO-CLS + KoLeo | Metric chính tăng ≥ 1, **hoặc** khoảng cách whitened − mean giảm ≥ 50% mà metric chính không giảm |
| G2 | R1 + R4 trong far view + curriculum (PLCM đầy đủ) | Metric chính tăng ≥ 1, hoặc P-lex macro-F1 tăng ≥ 2 |
| G3 | Augment hình thái bất đối xứng | Signer probe (35 signer: 30 signer pretrain + 5 signer dev-2) giảm ≥ 5, metric chính giảm không quá 0.5 |
| G4 | (ablation) tách CONF thành SHAPE/ORI, khai báo phụ thuộc khung lòng bàn tay | Chỉ để báo cáo |

Mọi ngưỡng "≥ 1" đều cần thêm điều kiện > 2 × độ lệch chuẩn giữa các seed (đo ở P1).

### 6.12 Công thức train và compute

Các số compute dưới đây **sẽ được đo ở P0**.

| | S | M |
|---|---|---|
| Dữ liệu | Tier S: ASL Citizen train, trừ 5 signer dev-2 | Tier A |
| Batch / epoch | 256 / 40 | 512 / 100 |
| Optimizer | AdamW β = (0.9, 0.95); lr 1e-3 tại batch 256 (scale tuyến tính, trần 2e-3); warmup 5 epoch; cosine; wd 0.04 → 0.2; bf16; clip 1.0 | như S |
| Số pass mỗi bước | 2 teacher forward + 2 student forward/backward (khoảng 50% token) | như S |
| FLOP ước tính | khoảng 15 GFLOP/clip ⇒ khoảng 4 TFLOP/step | khoảng 50 GFLOP/clip ⇒ khoảng 26 TFLOP/step |
| Thời gian (GPU dùng chung, ước tính) | 0.1–0.3 s/step ⇒ **15–45 phút/run** | 0.6–1.5 s/step ⇒ **4–9 giờ/run** |
| Nạp dữ liệu | cache float16 memmap; **toàn bộ đặc trưng tính trên GPU** | như S |

So với ước tính 45–80 phút của reviewer: số đó dành cho kiến trúc v1 (6 pass và head 8192 prototype). Kiến trúc v2 bỏ cả hai.

### 6.13 Pseudo-code một bước (v2)

```python
for batch in loader:                                   # cached raw keypoints + conf + W/H + fps + length
    crop = timestamp_crop(batch, T=64, jitter=8)       # content-independent
    crop, flip = canonicalise_handedness(crop)         # declared 1-bit leak (flip-rate test)
    s1, s2 = U(0.8, 1.25), U(0.8, 1.25)
    T1, sup1 = resample(aug_teacher(crop), s1); T2, sup2 = resample(aug_teacher(crop), s2)
    S1, _    = resample(aug_student(crop), s1); S2, _    = resample(aug_student(crop), s2)  # independent morph/noise
    m_near, _ = sample_views(n_valid(T1), rng);  _, m_far = sample_views(n_valid(T2), rng)
    m_near = propagate(m_near, sup1); m_far = propagate(m_far, sup2)                         # per-channel context-drop

    with no_grad():
        for T in (T1, T2):
            f = channel_features(T, visible=ALL)               # GPU; anchors + own-channel stats
            h = E_teacher(tokenizer_teacher(f), return_last4=True)
            P_ctx[T] = softmax((LN(mean(LN(l) for l in h.last4)) - center) / tau_t)
            Y_int[T] = zscore(descriptor_targets(f))           # [x_bar, delta x_bar], HEAD: delta only
            cls_T[T] = h.cls

    loss = 0
    for S, m, T, w in ((S1, m_near, T1, 1.0), (S2, m_far, T2, w_far(epoch))):
        f = channel_features(S, visible=m.visible, joint_hidden=m.R1)      # stats from visible frames only
        z = E_student(tokenizer_student(f)[m.visible])
        q = predictor(z, queries=type_side_time(m.masked_cells))           # ALL masked cells, no coordinates
        wgt = reliability_teacher(T)[m.masked_cells] * gamma_group          # 0 for ABSENT
        loss += w * token_loss(P_ctx[T][m.masked_cells], H_ctx(q), Y_int[T][m.masked_cells], H_int(q), wgt)
        if G1: loss += lam_glob * cls_ce(cls_T, z.cls, center_cls) + lam_koleo * koleo(z.cls)   # per view
    loss.backward(); opt.step(); ema_update(); update_centers()
```

### 6.14 Readout (khoá trước)

- **Encoder:** teacher.
- **Clip dài hơn 64 frame:** cửa sổ trượt bước 32 frame, ghép token lại.
- **Báo cáo đồng thời:**
  - (i) CLS;
  - (ii) mean-pool;
  - (iii) mean-pool theo nhóm kênh rồi nối lại (cho phonology);
  - (iv) bins2 (hai nửa có thứ tự);
  - (v) **attentive probe** (1 block cross-attention, 4 query, linear);
  - (vi) mỗi readout có thêm một biến thể whitened.

---

## 7. Dữ liệu

| Tier | Thành phần (chỉ dùng split train; trainer không đọc nhãn) | Mục đích |
|---|---|---|
| **S** | ASL Citizen train, **trừ 5 signer dev-2** | screening (P1–P3) |
| **A** | S + WLASL2000 train + MS-ASL train + How2Sign train (cửa sổ ngẫu nhiên) | kết quả chính |
| **noASLC** | A − ASL Citizen | **dùng cho các claim so với bên ngoài**: SignRep và MASA không pretrain trên ASL Citizen, mà gallery retrieval lại là chính ASL Citizen train (phản biện M3) |
| **M** | A + CSL-Daily + Phoenix14T | ablation đa ngôn ngữ (A9) |

**Các tập dev:**
- **Dev-1** = ASL Citizen val (5–6 signer).
- **Dev-2** = 5 signer tách khỏi ASL Citizen train, chọn ngẫu nhiên có seed và ghi vào protocol. Các signer này không dùng cho pretrain và không dùng cho train probe.
- Tổng cộng 10 signer dev, đủ cho kiểm định nhất quán theo signer (§8).

**Vệ sinh dữ liệu:**
- Loại khỏi pretraining các clip trùng *video id/URL* với val/test của bất kỳ dataset đánh giá nào. Bổ sung kiểm tra trùng gần đúng bằng vân tay keypoint (cosine > 0.98).
- Kiểm tra *cùng bộ trích keypoint* giữa các dataset (P0), vì đổi extractor làm số liệu lệch 6–15 điểm.

**Cache:** (T, 103, 3) float16 cùng W, H, fps, id; khoảng 10 GB.

**Kiểm toán ở P0:**
- tỷ lệ tay hợp lệ;
- tỷ lệ lòng bàn tay co rút;
- fps;
- tỷ lệ người thuận tay trái;
- độ lệch giữa cổ tay thân và hand root (khoảng 9% frame lệch trên ASL Citizen);
- phân bố độ dài clip.

---

## 8. Giao thức đánh giá (khoá trước run đầu tiên)

| Mã | Đánh giá | Chi tiết |
|---|---|---|
| **E1a** | OFF dictionary retrieval | Gallery = ASL Citizen train (bỏ dev-2), query = dev-1 và dev-2, class-max cosine. Mọi readout ở §6.14. Code của `p4_signssl`. |
| **E1b** | **Attentive probe** top-1/5 (2,731 lớp) | Train trên đặc trưng train; chọn lr bằng 10% signer train để riêng |
| E1c | Linear trên mean-pool | |
| E1d | LOSO-val whitened | Liên tục với EX15/EX16 |
| **E2a** | **Phonology P-lex** (frozen, lexically disjoint, kiểu WLASL-LEX) | 16 đặc trưng ASL-LEX 2.0; 5 fold theo gloss; probe train trên clip train của gloss trong fold-train, test trên clip dev của gloss trong fold-test. Báo cáo macro-F1. Kèm protocol "standard split" để đặt cạnh SHuBERT (fine-tune) và SignDino (tự báo). |
| E2b | AMI mã kênh (khi bật ablation intrinsic rời rạc) | So với k-means trên mô tả thô và với khởi tạo ngẫu nhiên, cùng K |
| E2c | Phân tầng SignType | một tay / đối xứng / bất đối xứng |
| **E3a** | **Rò rỉ signer trong biểu diễn** (negative metric) | Probe 35 signer train (clip held-out), probe 5 signer dev-1, bản cân bằng gloss. Tỷ lệ láng giềng gần nhất cùng signer so với cùng gloss. |
| **E3b** | **Rò rỉ signer trong target** (kiểm định cơ chế, phản biện M6) | Probe signer trên P^ctx và trên y^int (pool theo clip). Trước khi probe, **chiếu cả hai về cùng số chiều** (PCA 64). Báo cáo kèm probe gloss trong cùng không gian, để so sánh tỷ số signer/gloss. |
| E4 | Few-shot | K ∈ {1, 3, 5} |
| E5 | Độ bền | bỏ tay không thuận; bỏ 30% frame tay thuận; nhiễu σ ∈ {0.02, 0.05}σ; fps 15/30 |
| **E6** | Final (dùng một lần) | ASL Citizen test (E1a, E1b); WLASL2000 / MSASL1000: linear (protocol STC), frozen retrieval, fine-tune P-I/P-C |
| E7 | (tuỳ chọn) CSLR | BiLSTM-CTC trên token frozen; Phoenix14T, CSL-Daily |

**Thống kê** (phản biện M2, điều chỉnh theo quyết định chạy **1 seed**, §17):
- Một khác biệt "có ý nghĩa" cần đủ ba điều kiện:
  - (i) **Δ ≥ 1.5 điểm** trên metric chính. Ngưỡng này cố định và bảo thủ, thay cho max(1.0, 2·SD_seed) vì không có nhiều seed để ước lượng SD_seed.
  - (ii) **permutation test theo query có cặp**, p < 0.01;
  - (iii) cùng chiều ở **≥ 8/10 signer dev**.
- Khác biệt nhỏ hơn 1.5 điểm được báo cáo là **"không kết luận được"** (biến thiên seed với 1 seed chưa biết), không phải "bằng nhau".
- Không dùng CI bootstrap trên 5 cụm.

---

## 9. Baselines: ma trận nhân tố matched

Tất cả dùng **cùng** token, encoder, predictor, cặp view near/far, số pass và epoch. Khác biệt chỉ ở yếu tố đang xét. Báo cáo thêm theo GPU-giờ.

| | target **ctx** | target **int** (tường minh) | **ctx + int** |
|---|---|---|---|
| mask **R0** (ngẫu nhiên) | C-R0-ctx (≈ JEPA-random) | C-R0-int (≈ MAE/MAMP theo kênh) | C-R0-both |
| mask **R4** (ống articulator, kiểu SLiM) | C-R4-ctx (≈ **ý tưởng gốc**, kiểu SLiM/SignSeek) | C-R4-int | C-R4-both |
| mask **PLCM-lite** (near R2 / far R3) | C-PL-ctx | C-PL-int | **C-PL-both (PhonoJEPA-MVM)** |

Thêm các baseline sau:

| ID | Baseline | Vai trò |
|---|---|---|
| B0 | Mô tả training-free | 13.46 OFF-val (D8; khung thân, hiệu chỉnh tỷ lệ) |
| B1 | Random-init, frozen | mốc sàn |
| B2 | Có giám sát (nhãn ASL Citizen) | trần tham chiếu, không phải đối thủ |
| **B-glob** | **Chỉ global term (DINO-CLS + KoLeo), không có loss token** | SLiM cho thấy global term có thể gánh cả biểu diễn |
| B-MAE | Tái tạo đặc trưng kênh thô, R0 60%, cùng encoder | |
| **C-PL-both-hands** | MVM nhưng **chỉ có token tay** (không có kênh mặt ở input) | Mặt làm hại ở mọi dạng đo được (D1); phải kiểm tra mặt có đáng giữ không |
| **C-PL-ctx + shuf-int** | Như C-PL-both nhưng y^int bị hoán vị giữa các clip cùng loại kênh | Đối chứng cơ chế cho H2 |
| B-MAMP | Target motion của kênh, mask R0 90% | |
| Ext. | SignRep đã phát hành (RGB); số công bố của MASA, SignBERT+, STC, SignSeek | **chỉ tham chiếu, không matched.** MASA không có weight công khai nên không chạy lại được |

---

## 10. Giả thuyết, luật quyết định và tiêu chí dừng

**Metric chính:** E1b (attentive top-1) và E1a (R@1 mean-pool) trên dev-1 + dev-2. Với RQ2/RQ4, thêm E2a và E3.

| | Giả thuyết |
|---|---|
| **H1** (RQ1) | Trong cột ctx + int, PLCM-lite > R4 và > R0. Kiểm định thêm bằng tương tác: lợi ích của PLCM so với R4 phải có ở cả cột ctx. |
| **H2** (RQ2) | ctx + int > ctx-only **và** > ctx + shuffled-int: ≥ +2 P-lex, hoặc ≥ +1 metric chính. **Cơ chế:** E3b cho thấy target intrinsic mang ít danh tính signer hơn target contextual, *và* E3a của biểu diễn giảm theo. |
| **H3** (RQ3) | G2 (R1 + R4 trong far view + curriculum, tức PLCM đầy đủ) cải thiện so với PLCM-lite. |
| **H4** (RQ4) | G3 giảm signer (35) ≥ 5 điểm, metric chính giảm không quá 0.5. |
| **H5** (RQ5) | Có target ctx (ctx hoặc ctx + int) thắng int-only (≈ MAMP/MAE theo kênh) trên frozen. Khi fine-tune, khoảng cách ≤ 1. |
| **H6** | G1 vượt B-glob và không thua MVM. Khoảng cách whitened − mean giảm ≥ 50%. |
| **H7** | Tier A > Tier S. noASLC kém Tier A không quá 3 điểm. Kiểm định bằng 3 seed với khoảng tương đương (equivalence margin) được khai báo trước. |

**Mục tiêu định hướng:**
- E1a ≥ 2 × B0, tức khoảng 27 trên dev.
- Trên noASLC, OFF-test vượt MASA 27.96 (khác readout, ghi rõ).

**Tiêu chí dừng / chuyển hướng:**
- **K1:** sau P2, nếu cấu hình tốt nhất không vượt B0 ít nhất 5 điểm (E1a), thì dừng và phân tích.
- **K2:** nếu C-R0-int (≈ MAMP) ≥ mọi cấu hình có ctx trên **metric chính**, *và* xếp hạng này giữ nguyên ở cả 40 và 160 epoch (P1b), thì claim latent thất bại ⇒ phương pháp trở thành "token kênh + PLCM + target tường minh".
- **K3:** nếu R4 ≥ PLCM ở mọi cột, thì claim về masking thất bại ⇒ chỉ báo cáo RQ2/RQ4.
- **K4:** nếu P1 collapse ở mọi nhánh ⇒ chuyển sang teacher đóng băng kiểu SALT.
- **K5:** nếu B-glob ≥ mọi cấu hình có loss token, thì loss token không đóng góp ⇒ báo cáo và dừng.

---

## 11. Ablation (sau MVM, theo thứ tự ưu tiên)

| # | Trục | Biến thể |
|---|---|---|
| A1 | Họ mask | −R2 / −R3 / thêm R1 / chỉ R3 / tỷ lệ near/far / curriculum bật-tắt |
| A2 | Target | α_int ∈ {0.5, 1, 2}; intrinsic rời rạc (k-means K = 64/128); bỏ phần Δ; bỏ làm trơn theo bin |
| A3 | Token hoá | 7 kênh / 5 token bộ phận (không phân rã) / nhóm-joint kiểu S-JEPA / tách SHAPE-ORI (G4) |
| A4 | Chuẩn hoá | khung thân vs khung lòng bàn tay; chuẩn hoá xương bật/tắt; neutral face bật/tắt; tỷ lệ tương đối bật/tắt; anchor theo frame vs median clip |
| A5 | Augment | hình thái bất đối xứng / đối xứng / không; IK giữ cổ tay vs FK kéo cổ tay |
| A6 | Chống collapse | CE-centering / prototype / L1 + SIGReg (có hoặc không EMA) |
| A7 | Global | không / DINO-CLS / LeJEPA; KoLeo bật/tắt |
| A8 | **Đối chứng rò rỉ có chủ đích** | (a) bỏ lan truyền mask qua resample; (b) mask phụ thuộc hoạt động (vùng A của v1); (c) chỉ query ô quan sát được; (d) LOC chứa quan hệ tới mặt; (e) anchor vai theo frame (vai 2-D dịch khi giơ tay); (f) bỏ stencil, tức Δw tính qua mép span; (g) lật L/R ngẫu nhiên dùng chung cho teacher và student, thay cho canonical hoá tay thuận theo nội dung. Mục tiêu: định lượng mỗi rò rỉ thổi phồng loss/metric bao nhiêu. |
| A9 | Dữ liệu | Tier S / A / M / noASLC |
| A10 | Quy mô | S vs M; cửa sổ 64 vs 128 frame |

---

## 12. Phase thực nghiệm và ngân sách (đã hiệu chỉnh)

Luật server: mỗi lúc một job GPU, tối đa 2 tiến trình, runner tuần tự có chờ bộ nhớ và retry khi OOM.

| Phase | Nội dung | GPU-giờ | Gate |
|---|---|---|---|
| **P0** | Không GPU, trừ bước đo throughput. Cache; tách dev-2; dedup; kiểm tra extractor; join ASL-LEX; harness E1–E5; B0; **toàn bộ test** (§6.3, §13); leak audit D6 trên pipeline thật (gồm ±k frame); đo throughput S/M | ≈ 1 | Test pass. B0 được tính lại *trong pipeline* (cửa sổ 64 frame, gallery Tier S) và dùng làm mốc sàn chính thức; chênh lệch so với 13.46 (D8, toàn clip, gallery đủ 35 signer) phải được giải thích. |
| **P1** | (a) Chống collapse: 4 nhánh A6 × MVM (C-PL-both), S, Tier S; cộng **3 seed** của nhánh mặc định để có SD_seed. (b) **Kiểm tra lịch train:** C-PL-ctx và C-PL-int ở 40 vs 160 epoch, vì target EMA/ctx có thể cần train lâu hơn (S-JEPA dùng 1200 epoch). | ≈ 5–8 | Có nhánh không collapse và > B0 ⇒ khoá họ loss. Chọn số epoch cho P2 sao cho xếp hạng ctx/int không còn đổi. |
| **P2** | **Ma trận nhân tố** 3 × 3 + B-glob + B-MAE + B-MAMP + **C-PL-both-hands** (chỉ token tay) + **C-PL-ctx+shuf-int** (§6.7) + B1, 1 seed. **3 seed** cho C-PL-both, C-PL-ctx, C-PL-int, C-R4-both, C-R0-both, B-glob, tức các ô mà H1, H2, H5 và K2 cần. | ≈ 8–20 | K1–K3, K5; H1, H2, H5 |
| **P3** | Gate G1–G3 và ablation A1–A8 chọn lọc | ≈ 8 | khoá cấu hình M |
| **P4** | Tier A và noASLC, model M: PhonoJEPA, ô tốt nhất không-PLCM, C-R0-int, B-glob (8 run × 4–9 giờ); sau đó top-2 × thêm 2 seed | ≈ 50–110 | H1–H7 |
| **P5** | Final một lần: ASL Citizen test, WLASL/MS-ASL (linear, retrieval, fine-tune) | ≈ 8 | — |
| | **Tổng** | **≈ 80–155** (cộng khoảng 10 giờ extract/eval) | |

**Phương án tiết kiệm.** Nếu ngân sách bị giới hạn: dừng ở P0–P3 (khoảng 22–37 GPU-giờ). Như vậy đã trả lời được RQ1, RQ2 và RQ5 ở quy mô S, với 3 seed cho các ô then chốt.

**Kế hoạch đã chốt (§17): P0–P3, model S, Tier S, 1 seed.**

| Phase | Run (1 seed) | GPU-giờ ước tính |
|---|---|---|
| P0 | đo throughput; mọi thứ khác chạy trên CPU | ≈ 1 |
| P1 | 4 nhánh chống collapse (A6) ở 40 epoch; C-PL-ctx và C-PL-int ở 160 epoch (kiểm tra lịch train) | ≈ 3–9 |
| P2 | ma trận 3 × 3, B-glob, B-MAE, B-MAMP, C-PL-both-hands, C-PL-ctx + shuf-int, B1 (khoảng 15 run) | ≈ 4–11 |
| P3 | gate G1–G3 và ablation chọn lọc (khoảng 10–12 run) | ≈ 3–9 |
| | **Tổng** | **≈ 11–30**, nằm trong ngân sách 22–37 |

Phần dư (nếu còn) để chạy lại các run lỗi hoặc thêm một ablation. P4 (model M) và P5 (test cuối) cần anh duyệt thêm.

---

## 13. Kế hoạch implement (chưa code cho tới khi anh duyệt)

```
method/phonojepa/
  pj/data/      cache_build.py  splits.py (tiers, dev-2, dedup)  audit.py  asllex.py
  pj/pipeline/  crop.py  handedness.py  resample.py (support tracking)  anchors.py
  pj/features/  channels.py  (GPU; ownership map; visible-only stats; descriptor targets)
  pj/masks/     plcm.py  (R0–R4, budget, content-independent sampler, closure, propagation, curriculum)
  pj/models/    tokenizer.py  encoder.py  predictor.py  heads.py
  pj/           train.py  losses.py (CE-centering, smooth-L1, loss balancing, DINO-CLS, KoLeo, SIGReg/prototype alts)
                ema.py  diagnostics.py (RankMe, LiDAR, target-space signer probe, collapse alarms)  manifest.py
  pj/eval/      extract.py  retrieval.py  probes.py  phonology.py  signer.py  robustness.py  finetune.py  stats.py
  configs/      S.yaml  M.yaml  factorial/*.yaml  baselines/*.yaml  gates/*.yaml
  tests/        test_e2e_hidden_source_perturbation.py   # every family x {speed aug on/off} x baselines
                test_sampler_content_independence.py  test_handedness_flip_rate.py
                test_resample_propagation.py  test_ownership_closure.py  test_absent_vs_mask.py
                test_teacher_no_grad_ema.py  test_query_all_masked_cells.py  test_baselines_share_pipeline.py
```

Các nguyên tắc rút từ lỗi cũ trong repo:
- **Run manifest:** ghi hash cấu hình, hash dữ liệu và *số step thực sự đã chạy*. Bộ eval từ chối checkpoint đạt dưới 95% số step kế hoạch.
- **Tái dùng code:** protocol OFF (`p4_signssl`); LOSO (`jepa_targets`); runner (`obs_jepa`).

---

## 14. Rủi ro và phương án dự phòng

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Trần chất lượng pose 2-D (hand AP 58–66; CONF đan xen trong 2-D) | cao | Claim ở mức *tương đối* so với baseline matched; X2 (tay 3-D); E5 |
| JEPA ≈ MAMP trên skeleton | cao | K2. RQ1, RQ2, RQ4 vẫn có giá trị dù RQ5 âm tính |
| Global term gánh hết biểu diễn (SLiM) | trung bình | B-glob, K5, G1 có gate |
| Tín hiệu chéo kênh khiêm tốn (0.15) | trung bình | Thang độ khó; A1 đo đóng góp của R3/R4; target intrinsic giữ thông tin cục bộ |
| Danh tính rò qua phong cách | cao | E3a/E3b; G3; báo cáo trung thực |
| Rò rỉ chưa phát hiện | trung bình | Test end-to-end; A8 đo hiệu ứng của rò rỉ có chủ đích |
| Thống kê yếu (ít signer) | trung bình | dev-2; 3 seed; permutation test cộng nhất quán theo signer |
| Compute vượt ước tính | trung bình | Đo ở P0; phương án tiết kiệm khoảng 25 GPU-giờ |
| Dữ liệu trùng lặp; ASL Citizen nằm trong pretraining | trung bình | Dedup; claim bên ngoài dùng noASLC |
| Test từng bị dùng | chắc chắn | dev-1 + dev-2 cho mọi quyết định; test một lần |
| Đặc thù sign: NMM kéo dài trong ký liên tục bị neutral-face theo cửa sổ xoá mất; fingerspelling và các handshape bị che ngón cái (A/S/T/N/M) khó phân biệt ở 2-D; cổ tay thân lệch hand root khoảng 9% frame | trung bình | Ký liên tục: neutral face theo *video*. Phân tầng lỗi theo handshape. LOC dùng cổ tay thân, CONF dùng hand root. Không thêm cờ lệch, vì cờ này phụ thuộc hai kênh và vi phạm sở hữu joint; chỉ báo cáo tỷ lệ lệch ở P0. |

---

## 15. Mở rộng (sau khi lõi qua gate)

| # | Mở rộng | Nội dung |
|---|---|---|
| **X1** | **Cầu nối SignRep** | (a) Dùng biểu diễn token PhonoJEPA frozen làm *prior học được* thay cho prior 3-D thủ công trong công thức JH3D. Câu hỏi: prior học từ skeleton có tốt hơn prior hình học của SignRep không? (b) Late fusion embedding SignRep + PhonoJEPA cho retrieval. |
| X2 | Tay 3-D | MediaPipe 3-D hoặc HaMeR, để tách SHAPE/ORI đúng nghĩa (G4) |
| X3 | LOC bằng path-signature | Lấy cảm hứng từ Path-JEPA. Signed area mã hoá chiều quay của chuyển động. |
| X4 | Predictor đa giả thuyết | Cho R3/R4, xử lý các cặp tối thiểu |
| X5 | Intrinsic học được kiểu iBOT | Head dùng chung cho token thấy được của student, E_c theo từng token, prior lũy thừa thay cho Sinkhorn (phản biện C4) |
| X6 | Ký liên tục và quy mô lớn | CSLR/SLT; trích RTMW cho YouTube-ASL (cần tải video) |

---

## 16. Đóng góp kỳ vọng và novelty (đã hiệu chỉnh theo phản biện)

**Có thể bảo vệ là mới** (nếu thực nghiệm xác nhận):
1. **Masking theo kênh âm vị học trên sign skeleton với sở hữu joint, đóng phụ thuộc và lan truyền mask được kiểm thử end-to-end.**
   - Đơn vị che là (cấu hình, vị trí/chuyển động, các kênh non-manual), không phải bộ phận giải phẫu (SLiM, SignSeek) hay joint (S-JEPA).
   - Tìm kiếm của reviewer không thấy tiền lệ cho masked prediction phân rã theo tham số âm vị.
2. **So sánh có kiểm soát "phần bị che nên được biểu diễn bằng gì"**: contextual latent vs mô tả tường minh của chính kênh vs cả hai.
   - Đo danh tính signer *trong target* và *trong biểu diễn*, tức một kiểm định cơ chế chứ không chỉ một con số accuracy.
   - Tiền lệ gần: SHuBERT (k-means offline theo stream, crop ảnh), Skeleton2vec (contextual), MAMP (motion). Nhưng chưa có công trình nào so sánh *theo kênh âm vị* trên sign skeleton.

**Không mới, kế thừa và trích dẫn:**
- JEPA/EMA và CE-centering (I/V-JEPA, S-JEPA);
- ống giải phẫu (SLiM) và lan truyền phụ thuộc (Path-JEPA);
- DINO-CLS và KoLeo (DINO, DINOv2, SLiM);
- scale xương (SLiM);
- **khử danh tính bằng chuẩn hoá xương và loss đối kháng** (SignRep đã làm);
- **phonology probe lexically held-out** (WLASL-LEX; probe frozen theo lớp của SVT-ISLR; 2606.28667);
- thang độ khó đo trên mô tả thô (mới chỉ là căn cứ heuristic, chưa đo trên latent).

**Đánh giá thẳng thắn:**
- Một bài báo mạnh cần **H1 hoặc H2 thành công rõ** ở 10 signer dev. Giai đoạn đã chốt chỉ chạy 1 seed, nên đây là *screening*; trước khi viết claim chính thức cần thêm seed cho các ô thắng.
- Nếu chỉ có RQ2 + RQ4, đóng góp ở dạng phân tích.
- Nếu K2 xảy ra, phương pháp trở thành "token kênh + PLCM + target tường minh". Kết quả đó vẫn mới so với MASA, BEST và SignBERT+ nhưng kém hấp dẫn hơn.

---

## 17. Quyết định của anh (review ngày 2026-10-11)

| # | Câu hỏi | Quyết định |
|---|---|---|
| 1 | Hướng phương pháp | **Duyệt, giữ nguyên v2.1.** Các mặc định đi kèm vì vậy cũng được chấp nhận: lõi chỉ skeleton (cầu nối SignRep X1 để sau); tách dev-2 gồm 5 signer từ ASL Citizen train; CONF dùng khung thân; tên "PhonoJEPA". |
| 2 | Ngân sách | **Phương án tiết kiệm (P0–P3, ≤ 22–37 GPU-giờ), chỉ chạy 1 seed.** Kế hoạch đã điều chỉnh ở §12; thống kê ở §8. |
| 3 | Test split | **Dùng lại test đúng 1 lần** ở bước cuối, ghi rõ lịch sử đã dùng (EX05/EX07 và các dự án legacy). |
| 4 | obs_jepa B | **Dừng hẳn.** Không dùng làm baseline; đã bỏ khỏi §9. |

**Hệ quả của việc chạy 1 seed** (cần ghi nhớ khi đọc kết quả):
- Không ước lượng được SD_seed. Luật quyết định ở §8 dùng ngưỡng cố định bảo thủ hơn.
- P5 (test cuối) và P4 (model M) nằm *ngoài* ngân sách hiện tại. Nếu muốn có số test, cần thêm một lần chạy cấu hình tốt nhất ở model S, khoảng 1–3 GPU-giờ cho final eval. Việc này sẽ được hỏi lại khi tới bước đó.

## 18. Nhật ký phản biện (v1 → v2)

Một reviewer độc lập đọc toàn bộ proposal, literature và script chẩn đoán. Reviewer chạy lại một kiểm tra số học và tìm kiếm novelty.

| Mã | Vấn đề reviewer nêu | Đánh giá của tôi | Xử lý trong v2 |
|---|---|---|---|
| C1 | SHAPE chứa xương cổ tay → MCP nên ORI suy ra được từ SHAPE. Khuỷu tay có thể lộ trục lòng bàn tay. F-HEAD dựng từ anchor "không bao giờ che". TL;DR và §6 mâu thuẫn (11 vs 9 kênh). | **Đúng.** D6′ đo được SHAPE → ORI 0.38–0.50 (0.69 là số còn dính artifact). | Gộp thành CONF (§5 D6, §6.4). LOC → CONF ≤ 0.15 theo cột (D8, khung thân). F-HEAD sở hữu khoé mắt và có closure cả mặt. Đồng bộ toàn văn bản. |
| C2 | Rò rỉ trước bước mask: vùng A, quyết định mirror, tập query chỉ gồm ô quan sát được, resample nearest nhân bản frame bị che. Test perturbation sẽ không qua. | **Đúng.** | Pipeline có thứ tự khoá (§6.3); sampler không phụ thuộc nội dung; query mọi ô bị che; nội suy tuyến tính + lan truyền theo support; mirror là rò rỉ 1 bit được khai báo, có test tỷ lệ lật; A8 đo hiệu ứng rò rỉ. |
| C3 | Chẩn đoán v1 bị lẫn biến, đọc sai chỉ số (signer-offset), và có **artifact** (cột MCP-giữa thành proxy 1/L). | **Đúng; lỗi do tôi.** | Viết lại `diag_v2.py`, thay một yếu tố mỗi lần. Kết luận "chuẩn hoá xương giúp" bị **rút lại**. Chỉ số signer-offset được đọc lại đúng nghĩa. B0 sửa thành 12.83, rồi 13.46 sau khi hiệu chỉnh tỷ lệ khung hình (D8). |
| C4 | Target intrinsic-latent của v1 có receptive field ±4 bin chồng ngữ cảnh, head không khớp loại đầu vào, Sinkhorn áp prior đều trong khi handshape phân bố Zipf. | **Đúng.** | Thay bằng intrinsic **tường minh** (§6.7). Bản học được chuyển sang X5. Bỏ hai tầng encoder. |
| M1 | Baseline không matched; thiếu baseline global-only. | Đúng | Ma trận nhân tố 3 × 3 + B-glob + K5 (§9, §10) |
| M2 | Thống kê trên 5 signer quá yếu. | Đúng | dev-2 (10 signer); SD_seed; permutation test + nhất quán ≥ 8/10 signer |
| M3 | Mục tiêu so với bên ngoài không cùng protocol; vấn đề in-domain. | Đúng | Claim bên ngoài dùng noASLC; ghi rõ khác readout; MASA không có weight công khai |
| M4 | Scale hình thái bằng FK kéo lệch cổ tay khỏi điểm tiếp xúc; còn đặc trưng tỷ lệ tuyệt đối. | Đúng | IK giữ cổ tay; mọi tỷ lệ tính tương đối theo median (§6.4, §6.10) |
| M5 | Loại frame lòng bàn tay ngắn xoá mất các orientation có thật. | Đúng | Sàn số học + cờ co rút |
| M6 | H2 không kiểm định đúng cơ chế. | Đúng | E3b (probe signer trên target, cùng số chiều); cánh ctx + shuffled-int; quét α_int |
| M7 | Quá phức tạp; ước tính compute quá lạc quan. | Đúng một phần | MVM + gate; một encoder; bỏ head 8192 và view G; ước tính lại, đo ở P0; phương án tiết kiệm |
| M8 | Novelty claim quá mức (SignRep, WLASL-LEX, SHuBERT, Skeleton2vec). | Đúng | §16 viết lại; §3.5 thêm SignRep |
| m1–m8 | pseudo-code; KoLeo theo view; bằng chứng cho Sinkhorn; R² nội-tay; mirror mặt trong diag; công thức ρ; các khoảng trống đặc thù sign; D5 đúng | Đúng | Đã sửa (§6.6–§6.13, §14; `diag_v2.py` mirror iBUG đúng) |

**Vòng phản biện 2 (v2 → v2.1).** Reviewer thứ hai kiểm tra lại C1–C4 và tìm vấn đề mới. Kết luận: *"v2 đã đủ để trao cho anh review; không còn rò rỉ nghiêm trọng; trước khi implement cần vá 1–5"*. Toàn bộ đã được vá trong v2.1:

| Mã | Vấn đề | Xử lý trong v2.1 |
|---|---|---|
| R2-1 | PLCM-lite có hai định nghĩa (R2 + R3 hay có cả R4); hàng R0/R4 của ma trận chưa matched | PLCM-lite = near R2 / far R3; thêm cờ `use_R4` (chỉ bật ở G2). Hàng R0/R4 dùng cùng họ cho cả hai view, cùng phân phối tỷ lệ (§6.6) |
| R2-2 | Các ô có 3 seed không đủ cho H1/H5/K2; ước tính chi phí mâu thuẫn; thiếu cánh chỉ-tay | 3 seed cho 6 ô then chốt cộng B-glob; thêm C-PL-both-hands; sửa chi phí P2 thành 8–20 giờ, tổng 80–155 giờ (§12) |
| R2-3 | Screening 40 epoch thiên vị *chống lại* target EMA/ctx | P1b so 40 vs 160 epoch; K2 chỉ kích hoạt nếu xếp hạng ổn định |
| R2-4 | Rò rỉ qua stencil thời gian (Δw ở mép span); R1 cần lan truyền theo nhóm joint; test tỷ lệ lật tay thuận không thể đạt; anchor theo frame là rò rỉ; Σω theo mẫu gây phương sai cao; closure R0 với HEAD chưa định nghĩa | Support = resample ∪ stencil; R1 lan truyền theo nhóm joint và zero conf; test swap-in theo từng họ + cánh lật ngẫu nhiên trong A8; anchor theo frame chuyển sang A8; Σω theo batch; closure cho mọi họ (§6.3) |
| R2-5 | Thiếu hiệu chỉnh tỷ lệ khung hình làm lẫn D7; D6 chỉ trung bình cột, sai khung; cằm làm lộ miệng; nhãn hàng 0.87 sai | Chạy lại D7′/D6″ có hiệu chỉnh tỷ lệ, theo cột, cả trong clip (D8); cằm chuyển sang MOUTH; bỏ F-UP/F-MO khỏi R3; B0 = 13.46; sửa nhãn |
| R2-6 | Target intrinsic làm trơn quá yếu; Δ không xác định ở mép; ℓ̄_ctx chứa entropy teacher (phản hồi dương); cánh ×2 vô nghĩa dưới AdamW | Làm trơn 3 bin phía teacher; Δ có trọng số 0 khi thiếu bin trước; đóng băng ℓ̄ sau warmup; thay bằng quét α_int + cánh ctx + shuffled-int (§6.7) |
| R2-m | Lệch số signer (35 vs 30); E3b khác số chiều; P0 gate cho B0 không xác định rõ; cờ lệch cổ tay vi phạm sở hữu joint | Ghi rõ 35 = 30 + 5; PCA-64 cho E3b; B0 tính lại trong pipeline; bỏ cờ lệch |

**Những gì phản biện xác nhận là vững:**
- Ý tưởng lõi (che *kênh âm vị học* thay vì bộ phận giải phẫu) hợp lý và chưa thấy công bố.
- Code D5 (class-max retrieval, train/val disjoint, z-score theo train) đúng.

---

## Phụ lục A — Siêu tham số mặc định (MVM, model S)

| Nhóm | Giá trị |
|---|---|
| Dữ liệu | 25 fps; T = 64; bin = 2 frame; 7 kênh × 32 = 224 token; 103 joint |
| Ngưỡng | tay hợp lệ: mean score > 2.0 (P0 hiệu chỉnh); L_eff = max(L, 0.1σ); conf theo q10/q90 của từng nhóm |
| Mô hình | d = 256, 6 block, 4 head; predictor 4 × 128; RoPE thời gian; CLS |
| Mask | near: R2 (+ R1 sau G2), r ~ U(0.4, 0.7); far: R3 (MVM, PLCM-lite), hoặc R4 (p = 0.4) / R3 sau G2, r ~ U(0.3, 0.6); span ≥ 4 bin; context dropout 0.1; ngữ cảnh tối thiểu 25% |
| Target | ctx: softmax trên chiều, centering β = 0.9, τ_t 0.04 → 0.06, τ_s = 0.1; int: [x̄ (trung bình 3 bin), Δx̄] z-score, smooth-L1 β = 0.5; cân bằng loss bằng trung bình chạy, đóng băng sau warmup; Σω chuẩn hoá theo batch; α_int = 1; γ_tay = 1, γ_mặt = 0.5 |
| Teacher | EMA 0.996 → 1 (cosine) |
| Optimizer | AdamW (0.9, 0.95); lr 1e-3 @ batch 256; warmup 5 epoch; cosine; wd 0.04 → 0.2; bf16; clip 1.0 |
| Augment | teacher: xoay ±5°, scale ±5%; student: xoay ±10°, scale ±10%, shear 0.1, nhiễu 0.02σ(1 − c), 5% tay → [ABSENT]; tốc độ U(0.8, 1.25) dùng chung theo cặp; (G3) hình thái IK giữ cổ tay ±10% |
| Global (G1) | DINO-CLS (softmax trên chiều, centering), KoLeo theo view λ = 0.05 |
| Phương án thay thế (A6) | prototype K = 4096 + centering; L1 + SIGReg λ = 0.02 (1024 slice, 17 điểm trên [−5, 5], BN-MLP) có hoặc không EMA |

## Phụ lục B — Nguồn chính

Đầy đủ, kèm tag kiểm chứng: `literature/`.

**JEPA và paradigm:**
- I-JEPA 2301.08243; V-JEPA 2404.08471; V-JEPA 2 2506.09985; V-JEPA 2.1 2603.14482.
- LeJEPA 2511.08544; SALT 2509.24317; C-JEPA 2410.19560.
- Littwin 2407.03475; Sobal 2211.10831; DirectPred 2102.06810.
- data2vec 2202.03555 / 2212.07525; iBOT 2111.07832; DINOv2 2304.07193; DINOv3 2508.10104; MaskFeat 2112.09133.
- RankMe 2210.02885; LiDAR 2312.04000; efficient probing 2506.10178.
- PointTube-JEPA 2608.24093; Laya 2603.16281; LeWM 2603.19312; Human-JEPA 2608.21160; T-JEPA 2410.05016.
- Point-JEPA 2404.16432; Graph-JEPA 2309.16014.

**Masking:**
- Kong et al. (CVPR'23); SemMAE; EPM; AttMask; ADIOS; HPM.
- MAE 2111.06377; VideoMAE 2203.12602; Point-MAE; Point-M2AE; CroCo 2210.10716; SiamMAE 2305.14344.

**Skeleton:**
- S-JEPA (ECCV'24, DOI 10.1007/978-3-031-73411-3_21); MAMP 2308.07092.
- SLiM 2603.10648; Path-JEPA (ECCV'26, DOI 10.1007/978-3-032-37258-1_7).
- GFP 2509.03609; AMR (CVPR'26); Skeleton2vec 2401.00921; MacDiff 2409.10473; IGM 2410.20349; MotionBERT 2210.06551.

**Sign language:**
- SignRep 2503.08529; SHuBERT 2411.16765; SignSeek 2609.03695; SignDino 2609.06296.
- SignBERT 2110.05382; SignBERT+ 2305.04868; BEST 2302.05075; MASA 2405.20666; STC 2406.10501; SignMAE 2605.02094.
- OpenHands 2110.05877; Uni-Sign 2501.15187; DSTA-SLR 2403.12519; NLA-SLR 2303.12080; SL-GCN 2103.08833.
- Kezar 2302.05759; WLASL-LEX 2205.12072; SVT-ISLR 2309.02450; 2606.28667 (phonological perception); 2609.18772 (handshape zero-shot); 2410.13675 (danh tính qua pose).
