# P2 — Cập nhật (delta) về JEPA và skeleton SSL cho một Path-JEPA trên skeleton ký hiệu

Biên soạn 2026-10-16. Tài liệu này **chỉ bổ sung** cho năm báo cáo nội bộ đã có, không viết lại chúng:
- `docs/proposal12/literature/L1_jepa_paradigms.md` (I-JEPA, V-JEPA 1/2/2.1, SALT, C-JEPA, LeJEPA/SIGReg, data2vec, DINOv2/v3, Littwin, domain JEPA),
- `L2_masking_structured.md` (masking có cấu trúc, BEST/SignBERT+),
- `L3_skeleton_ssl.md` (S-JEPA, SLiM, MAMP, GFP, AMR, MacDiff, IGM, ASMa, STARS, MotionBERT),
- `docs/research/literature/B_skeleton_ssl_masking.md` (Skeleton2vec, HA-CM, BYOL-skeleton, USDRL, quy tắc R1–R12),
- `docs/research/literature/C_jepa_diagnostics_rigour.md` (Bootleg multi-level targets, RankMe/LiDAR/α-ReQ/ID, EMA scaling rule, probe confound).

Khi một mục đã được các file trên phủ, ở đây chỉ có một dòng trỏ tới đó. Nội dung mới gồm: (i) **toàn văn Path-JEPA**, đọc lần đầu; nó sửa và bổ sung các mục [from-summary] trong L3/B; (ii) dòng SIGReg dùng cho *masked token prediction* (2026), lấp lỗ hổng "Not found" của L1; (iii) skeleton SSL mới (SOfA, HSP, A4Mer, MaskCLR, SignSeek, DiMoP, MASQ); (iv) tokenizer dựa trên path signature và các tokenizer có cấu trúc; (v) tổng hợp 12 nguyên tắc thiết kế.

**Nhãn bằng chứng.**
- **[V]**: tôi (hoặc agent phụ) đã grep con số trong text thô của PDF arXiv/CVF; kèm vị trí, ví dụ `[V: 2609.07078 Tab.5]`.
- **[V\*]**: con số lấy từ bảng so sánh của một bài khác.
- **[S]**: chỉ có từ tóm tắt.
- **[chưa xác minh]**: chưa grep được.
- **[suy luận]**: suy luận của tôi, không phải phát biểu của bài báo.

Text thô nằm ở `scratchpad/p2raw/txt/<id>.txt`; toàn văn Path-JEPA ở `scratchpad/pathjepa/5207.txt`. Hệ thống không dùng GPU.

---

## 0. Mười điểm mới quan trọng nhất

1. **Path-JEPA che rất ít, và chỉ che theo không gian.** Tập mask là một tập *khớp* M; mọi token (mọi window, mọi scale) có path chạm vào M đều bị che. Tỉ lệ tối ưu là **20–25% số khớp**: 10% cho 85.1, 40% cho 84.9, còn mặc định cho 87.6 [V: 5207 Tab.5, §4.5]. Con số này trái ngược với quy ước 90% của MAMP/S-JEPA. Path-JEPA **không có mask thời gian nào**.
2. **Signature bỏ mất vị trí tuyệt đối và cấu hình tĩnh.** Signature của một path hằng là (1, 0, 0, …). Một bàn tay giữ nguyên handshape, hoặc vị trí tay so với mặt (location), là *vô hình* với log-signature của quỹ đạo, nếu không nối thêm điểm đầu hoặc vị trí tương đối [V: 2206.14674; LogSig-RNN 2110.13008 và GCN-DevLSTM 2403.15212 đều nối start point]. Với ký hiệu, handshape và location là hai tham số âm vị học, nên đây là rủi ro số 1 khi bê nguyên Path-JEPA sang.
3. **Token [global ‖ local] làm rò rỉ nếu thêm mask thời gian.** Theo đẳng thức Chen S₀,T = S₀,t₁ ⊗ … ⊗ S_{tK−1},T, signature level ≤2 của một window bị che có thể tính **chính xác** từ phần global cộng các window còn thấy. Agent phụ đã kiểm bằng numpy với d=4 và 4 window [V: 2206.14674 Eq.2.53 + kiểm số]. Path-JEPA tránh được lỗi này chỉ vì nó che cả path qua mọi window.
4. **SIGReg cho masked latent prediction đã có công thức chạy được (2026), nhưng chưa ở quy mô ViT video.**
   - GLaS-JEPA (speech): shared encoder, stop-grad target qua projector tuyến tính 128-d, SIGReg *marginal trên mọi token B×T*, λ=0.01, mask 50% theo span. Bỏ SIGReg thì rank về 1 [V: 2609.37798 §3, §5.2].
   - LeVJEPA **không** làm masked prediction: nó là invariance [CLS] + SIGReg [V: 2608.27395 §3]. Brief giả định ngược lại.
5. **Đặt SIGReg sai chỗ sẽ triệt tiêu tín hiệu động.**
   - TC-SIGReg trừ trung bình cửa sổ cục bộ (W=4) trước khi chuẩn hoá: LIBERO 63.6→83.8 [V: 2607.26924 Tab.2].
   - LeNEPA: SIGReg *theo thời gian trong từng mẫu* là cách đặt duy nhất cho lợi ích bền vững [V: 2607.00958 Tab.5].
   - Clip ký hiệu có thành phần tĩnh lớn (người ký, khung hình), nên đây là ablation rẻ và nên làm.
6. **Đừng che toàn bộ khớp tại một bước thời gian.** EEG-JEPA gặp "bias-inflation collapse": predictor học bảng tra theo vị trí kênh, mean mỗi kênh phồng 2–3×, và VICReg/C-JEPA không phát hiện được [V: 2609.33487 §5]. JEPA nhạy với hình học mask gấp 2.4× MAE (0.77 so với 0.32) [V: §4].
7. **Có hai bằng chứng mới cho tokenizer bất biến topology.**
   - SOfA: joint-name embedding (T5) + 5 canonical slot. Bỏ slot làm NTU-60 rơi 85.7→76.4; bỏ cả hai còn 75.2 [V: 2609.07078 Tab.5].
   - HSP: luồng 2-D 17 khớp (HRNet) *thắng* luồng 3-D Kinect, 84.4 so với 80.2 [V: 2506.03481 Tab.1]. 2-D không phải là bất lợi.
8. **Token theo phân đoạn chuyển động, độ dài biến thiên, thắng token cố định.** A4Mer: token per-frame cho lỗi dự đoán 309 mm, token theo segment 150 mm. Không tách global/local thì sụp: recog 38.1→15.1 [V: 2604.28173 Tab.3].
9. **Mô tả đa thang tay thiết kế thắng patch thô trong JEPA.** W-JEPA (wavelet) đạt 80.6 so với 71.9/73.1 macro-F1 [V: 2610.19411 §4.2]. LogSig-SSM: log-sig depth 2 đạt 71.7, depth 1 đạt 69.5, wavelet scattering 67.6, conv 62.9 [V: 2610.05051 Tab.4]. Cái lợi nằm ở level 2 (Lévy area giữa các kênh).
10. **Một số trick DINO/iBOT nhỏ ở quy mô nhỏ.**
    - Sinkhorn-Knopp so với softmax-centering: không đổi [V: 2304.07193 Tab.1].
    - KoLeo: +2.3 kNN và retrieval Oxford 55.6→63.9 [V: Tab.1, Tab.3a]. Hữu ích cho dictionary retrieval.
    - MIM không có [CLS] loss thì sụp ngữ nghĩa (kNN 9.5) [V: 2111.07832 Tab.9].
    - Chưa tìm thấy follow-up 2025–26 nào cho Littwin et al.

---

## 1. Path-JEPA (ECCV 2026): sửa và bổ sung từ toàn văn

Singh & Singh, ECCV 2026, LNCS 17054 pp.110–128 (L3 đã xác minh qua Crossref). Code vẫn "coming soon" (B). Toàn văn đã đọc; phụ lục (chi phí tính toán, độ nhạy β, frame drop) **không có** trong bản text này [V: 5207 chỉ dài 18 trang].

| Thành phần | Công thức Path-JEPA | Nguồn |
|---|---|---|
| Path | Nội suy tuyến tính từng khớp, sau đó lead-lag [X(t), X(t−δ)] ∈ R⁶ (3-D) | [V: Eq.1] |
| Đặc trưng | Log-signature level n=2, D_log=22, D_sig=43 | [V: §3.1, §4] |
| Họ path | J (từng khớp), P (cạnh xương, Y=X_v−X_u nên bất biến tịnh tiến, \|P\|=J−1), C (chuỗi chi, "nối các quỹ đạo displacement cạnh theo thứ tự giải phẫu") | [V: §3.2] |
| Token | m_{p,ℓ,k} = [logS(0,T) ‖ logS(window ℓ,k)]. Window W_ℓ=2^ℓ·W₀; ℓ=0 dài 25, stride 12; ℓ=1 dài 50; tổng 4 window, **N=152 token/chuỗi** | [V: Eq.3, §4] |
| Embedding | Projection riêng theo loại + type emb + e_struct(ℓ, độ sâu động học d_G). Attention bias tuỳ chọn theo \|Δt\|, scale, IoU window, khoảng cách đồ thị | [V: Eq.5–6] |
| Mask | Seed khớp lấy mẫu theo motion, mở rộng ra cả **subtree động học**, rồi lan sang mọi token có path chạm khớp bị che (cả global lẫn local). Mask áp *sau* khi tính signature | [V: §3.4, Eq.7] |
| Kiến trúc | Student 6 lớp, d=512, 8 head; predictor 4 lớp với query học được (type, path ID, thời điểm, scale, depth) cross-attend vào context | [V: §3.5, §4] |
| Target / loss | Teacher EMA (0.996→1, cosine) nhìn toàn bộ token; centering (β=0.9); **loss cosine**; đánh giá bằng teacher | [V: Eq.9–10, §4] |
| Huấn luyện | 300 epoch, AdamW 3e-4, wd 0.05, batch 256 | [V: §4] |

**Ablation (NTU-60 xsub, linear) [V: Tab.5, Tab.6].**

| Masking | Acc | Tokenisation | Acc |
|---|---|---|---|
| Random 20% khớp | 84.1 | J, chỉ global | 84.4 |
| Motion-aware (kiểu S-JEPA) | 86.8 | J, chỉ local | 83.3 |
| Signature-augmented, seed ngẫu nhiên | 86.3 | J, global + local | 85.8 |
| + mở rộng subtree (mặc định) | **87.6** | + cạnh (P) | 86.4 |
| Kinematic, 10% | 85.1 | + chuỗi (C), đầy đủ | **87.6** |
| Kinematic, 40% | 84.9 | | |

**Sửa và ghi chú so với L3/B.**
- Các mục kiến trúc trong L3 trước đây gắn [from-summary] (6/512, predictor 4 lớp, EMA 0.996→1, centering 0.9, cosine, 300 epoch) nay **đã xác minh [V]**. Tokenisation ablation (B: [S]) nay cũng **[V]**.
- *Mới:* tỉ lệ mask tối ưu là 20–25% khớp, không phải 90%. Hàng "seed ngẫu nhiên + propagate" (86.3) **thấp hơn** motion-aware không propagate (86.8). Bảng không cô lập riêng lợi ích của việc chống rò rỉ, và cũng không có cấu hình motion-seed + propagate mà không mở rộng subtree [suy luận].
- *Không nhất quán:* với 25 khớp NTU, \|P\|=24 và \|C\|≥1, tức ≥50 path/window. 152/4 = 38 token/window không khớp với số đó. Theo công thức Witt, log-sig level 2 với d=6 có **21** chiều, không phải 22. Con số 22 có lẽ cộng thêm hằng số hoặc một kênh [suy luận]. Cần hỏi tác giả hoặc chờ code.
- *Giao thức fps:* clip được resample rồi **nội suy tuyến tính trở lại độ dài gốc** [V: §4.5]. Đây là nhiễu loạn nhẹ. Kết hợp với ghi chú của B (giá trị 30 fps của baseline trùng đúng headline), nên coi tuyên bố bền vững 15–60 fps là bằng chứng vừa phải.
- *Rủi ro khi chuyển sang ký hiệu [suy luận]:*
  - Window 25/50 frame (≈0.8/1.7 s ở 30 fps) gần bằng độ dài cả một ký hiệu đơn, nên quá thô cho fingerspelling và chuyển handshape.
  - Không có mask thời gian, nên mô hình không bị ép dự đoán *khi nào* có chuyển động.
  - Signature của chuỗi ngón tay (wrist→MCP→PIP→DIP→tip) có kích thước bùng nổ nếu ghép theo kênh (§4.1).

---

## 2. JEPA family: con trỏ ngắn và phần delta

### 2.1 Bảng con trỏ (đã có trong L1/C, không lặp lại chi tiết)

| Phương pháp | Target | Predictor | Masking | Chống sụp | Bằng chứng then chốt | Ở đâu |
|---|---|---|---|---|---|---|
| I-JEPA 2301.08243 (CVPR'23) | EMA output, che ở output | Hẹp (384), 6–12 lớp | 4 block 0.15–0.2, context 0.85 | EMA | multi-block 54.2 so với random 17.6 | L1 §1 |
| V-JEPA 2404.08471 (TMLR'24) | EMA, L1 | 12×384 | short+long block, ~90% | EMA | random-tube 51.5 so với 72.9 | L1 §1 |
| V-JEPA 2 / 2.1 (2506.09985 / 2603.14482) | LayerNorm target; 2.1: deep supervision 4 mức + context loss | 12→24 lớp | như V-JEPA | EMA cố định; distil dùng frozen teacher | ADE 22.2→38.6 | L1 §1 |
| SALT 2509.24317 | Teacher pixel-MAE đóng băng | — | multi-block | frozen teacher | loss↔acc R²=0.951 | L1 §1 |
| C-JEPA 2410.19560 (NeurIPS'24) | I-JEPA + VICReg | — | — | EMA + VC | +0.8 linear | L1 §1 |
| LeJEPA 2511.08544 | multi-crop invariance (không có masked prediction) | không | — | SIGReg, λ=0.05 | ViT-L 77.1 | L1 §2 |
| data2vec 2.0 2212.07525 (ICML'23) | trung bình top-K lớp | conv decoder | inverse block, 8 mask/mẫu | EMA | 2–16× nhanh hơn | L1 §3 |
| Bootleg 2603.15553 | **nối** z-score các block {1,4,8,12} | — | I-JEPA | EMA, ít nhạy τ | +10 so với I-JEPA; seed std 2.9→0.6 | C §1.2 |
| DINOv2/iBOT/DINOv3 | prototype CE, MIM patch | — | block | centering, KoLeo, Gram anchoring | xem §2.3 | L1 §3 |
| Littwin 2407.03475 (NeurIPS'24) | lý thuyết: JEPA thiên về đặc trưng có hệ số hồi quy cao, MAE thiên về đặc trưng có phương sai cao | — | — | — | thiên lệch tăng theo độ sâu | L1 §2 |
| Van Assel 2505.12477 | lý thuyết: JE thắng khi nhiễu không liên quan có biên độ lớn | — | — | — | MAE rơi 25.1% so với DINO 10.5% trên IN-C | C §1.1 |
| RankMe / LiDAR / α-ReQ / ID | chẩn đoán | — | — | — | RankMe kém trên I-JEPA (τ 0.58), LiDAR 0.82 | C §2 |

### 2.2 Delta: SIGReg và các regulariser cho *masked/token-level* latent prediction

- **GLaS-JEPA** (2609.37798, preprint, speech, 57M Conformer, LibriSpeech) [V].
  - Không có EMA và không có predictor: encoder tự suy ra vị trí mask từ mask token.
  - Target = projector tuyến tính theo token (128-d) của *cùng encoder* chạy trên view đầy đủ, có stop-grad trong số hạng MSE.
  - Mask 50%, span 10 latent (200 ms).
  - SIGReg trên các token của view đầy đủ, **không pool**, λ=0.01.
  - Với target liền kề tương quan cao: dùng phân phối marginal p(z) (trộn mọi B×T token) tốt hơn conditional theo t: WER 13.99→12.24, DER 7.65→6.65 [V: Fig.3].
  - Bỏ SIGReg thì effective rank về 1 sau vài nghìn bước. Bản ~95M chưa ổn định [V: §5.2].
  - **Bài học:** đây là khuôn gần nhất cho một sign-JEPA không EMA. Hai điểm then chốt là regulariser marginal trên token và projector nhỏ.
- **Laya** (2603.16281, EEG; L1 đã có công thức). Delta mới: bỏ stop-grad làm acc 0.603→0.559 ở 10k bước, và 0.597→0.466 ở 20k bước. Latent target 0.603 so với MAE raw 0.529 [V: Tab.3, Tab.5]. Stop-grad càng quan trọng khi huấn luyện càng dài.
- **LeNEPA** (2607.00958, MiLeTS'26, time series) [V: Tab.5].
  - Next-latent prediction nhân quả, không stop-grad.
  - **Temporal SIGReg** (regularise tập token *trong từng mẫu* theo thời gian) là cách đặt duy nhất cho lợi ích bền vững; batch-wise, pooled và difference đều kém hơn.
  - Projector giúp trong 22/24 so sánh probe.
  - Lớp tốt nhất nằm ở giữa (L4–5), không phải lớp cuối.
- **TC-SIGReg** (2607.26924, preprint) [V: Eq.2, Tab.2, Tab.6].
  - r_t = z_t − mean(z trong cửa sổ W_t); SIGReg chỉ áp lên r_t.
  - Lý do: SIGReg thường chia chung "ngân sách phương sai" giữa phần bền vững và phần dư, còn prediction loss ưu tiên phần bền vững, nên dynamics bị nén.
  - LIBERO trung bình 63.6→83.8. W = 2/4/8/16 cho 62.0/63.6/65.8/52.1.
  - **Liên hệ ký hiệu [suy luận]:** hình thể và tư thế người ký là phần "bền vững"; handshape và chuyển động là phần dư.
- **LeVJEPA** (2608.27395, preprint) [V: §3–5]. **Không phải masked prediction.** Nó là invariance [CLS] giữa view toàn cục và cục bộ, không stop-grad, không predictor, SIGReg λ=0.02 trên [CLS] sau projector BN.
  - "Masking" là token drop 95%, đóng vai augmentation.
  - Drop dạng tube kém hơn uniform: IN1K 39.6 so với 50.7.
  - Thêm local view: V 4→10 cho 47.6→50.2.
  - Ở cùng FLOP với V-JEPA 2: IN1K 61.0 so với 51.6, nhưng SSv2 40.4 so với 42.5. Nhánh invariance thua ở bài toán chuyển động.
- **λ-JEPA / SACReg** (2609.35288, preprint) [V: Tab.1a, Tab.2, Tab.5].
  - Regulariser log-det covariance trên biểu diễn *đã trung bình qua view*, áp cả ở backbone lẫn projector, vì regulariser chỉ ở sau projector để lại backbone low-rank.
  - SSv2 video ViT-B 43.9 so với LeVJEPA 30.4 (240 epoch).
  - Thêm SIGReg trực tiếp vào backbone thì RankMe chỉ nhích 0.07→0.09 và linear *giảm* 60.4→58.9.
  - **Bài học:** đo rank ở backbone, không chỉ ở projector.
- **OOB dataset-level regularisation** (2610.20158, preprint) [V: Tab.2, App.A.3].
  - Cache phân vị Gaussian cho toàn bộ dataset, cập nhật mỗi 0.5 epoch.
  - Với batch nhỏ (1–128): 5-NN 34.6→48.8 trên ảnh tự nhiên. Cache cho N=13k chỉ 12.7 MiB.
  - Với khoảng 40k clip, chi phí không đáng kể.
- **Beyond Gaussian Worlds** (2609.21656, preprint, toy) [V: Tab.1].
  - Latent trên đa tạp (vòng tròn, torus) bị Gaussian target bóp méo: R²(Y→Z)=0.493 so với 0.996 khi target khớp hình học.
  - **Liên hệ [suy luận]:** góc khớp và hướng xương sống trên S¹/S², nên cần ablation target cầu. Chưa có bằng chứng nào trên skeleton.
- **Hình học mask đa kênh** (2609.33487, preprint, EEG, EMA-JEPA) [V: §4–5].
  - Che *mọi kênh* tại một bước thời gian gây "bias-inflation collapse" mà VICReg/C-JEPA không thấy.
  - Tối ưu là bán kính không gian vừa phải với block thời gian ngắn.
  - Khớp skeleton tương đương với kênh EEG.

**Kết luận 2.2.** Với masked token prediction trên chuỗi tín hiệu thấp chiều, đã có ba khuôn **không EMA** chạy được: GLaS, Laya, LeNEPA. Điểm chung: (a) stop-grad trên target của cùng encoder qua projector nhỏ; (b) SIGReg *trên token*, marginal hoặc theo thời gian, chứ không chỉ trên [CLS] pool; (c) λ 0.01–0.05. Vẫn **chưa có** skeleton SSL nào dùng SIGReg (B §d vẫn đúng). EMA + centering của Path-JEPA và S-JEPA vẫn là baseline an toàn.

### 2.3 Delta: MC-JEPA, DINOv2, iBOT (số liệu ablation)

- **MC-JEPA** (2307.12698, preprint) [V: Tab.1, Tab.5, Tab.11].
  - Encoder chung cho flow tự giám sát (PWC, coarse-to-fine) và VICReg content.
  - VC regularisation đặt **ở mọi tầng pyramid**: VOC 47.3 (không VC) → 65.6 (chỉ lớp cuối) → 66.2 (mọi tầng) → 67.1 (+1 epoch warm-up).
  - Motion và content giúp lẫn nhau: chỉ flow cho VOC 9.4, MC 67.1; DAVIS 70.5 so với VICReg 58.1.
  - Hệ số đa nhiệm tối ưu 0.1.
  - **Liên hệ:** một mục tiêu "motion" tường minh (increment hoặc signature) cùng một mục tiêu "content" trên chung encoder là hợp lý, nhưng phải cân bằng (~0.1) và regularise nhiều tầng.
- **DINOv2** (2304.07193, TMLR'24) [V: Tab.1, Tab.3].
  - SK centering: không đổi (kNN 81.7 / lin 84.7).
  - KoLeo (trọng số 0.1, trên CLS crop global thứ nhất): kNN 76.6→78.9, Oxford-M 55.6→63.9, ADE gần như không đổi.
  - Untied head: +0.3 kNN, −0.2 linear.
  - MIM term: ADE 44.2→47.1.
- **iBOT** (2111.07832, ICLR'22) [V: Tab.9, Tab.18, App.E].
  - Tỉ lệ dự đoán r=0 với xác suất 0.5, ngược lại lấy từ U[0.1, 0.5], mặc định 0.3±0.2, không nhạy trong khoảng 0.05–0.4. Biến thiên ± giúp ích (cùng hướng với SLiM U(0.5, 0.9)).
  - Head chung so với riêng: 69.1/74.2 so với 68.9/73.9.
  - Patch centering có tác dụng nhỏ hơn so với [CLS].
  - **MIM không có [CLS] loss: kNN 9.5 / lin 29.8.**
- **Littwin follow-up:** không tìm thấy bài 2025–26 nào mở rộng sang phi tuyến; nhiều bài chỉ trích dẫn làm động lực (5 truy vấn arXiv) [S].

---

## 3. Skeleton SSL: phần delta (2025 – 10/2026)

### 3.1 Sửa brief
- **S-JEPA (ECCV'24) không có pretext lifting 2D→3D.** Cả hai nhánh đều nhận 3-D; lifting là pretext của MotionBERT. Đã xác minh từ bản camera-ready (L3 §1, B §2.2). Centering và CE là đúng: CE so với MSE là 85.3 so với 83.7; centering chỉ thêm +0.2–0.8.
- **STARS** 2407.10935 là **WACV 2026**, pp.2858–2868 [V: Path-JEPA ref 38].
- **HeterSkel 2506.03481** tự gọi là **HSP**. Hàng "HSP" trong bảng của SOfA/L3 (80.7) là hàng J+S, không phải hàng tốt nhất (87.8) [V: 2506.03481 Tab.1 so với 2609.07078 Tab.2].
- **MaskCLR (CVPR'24) là phương pháp có giám sát** (CE + AGPM + contrastive), không phải SSL. Bài không có trên arXiv; đã đọc PDF của CVF.
- Các phương pháp MAMP, MacDiff, IGM, ASMa, Skeleton2vec, GFP, AMR, SLiM, STARS: xem L3 §4, B §2. Không có thay đổi.

### 3.2 Thẻ mới

**SOfA, "One for All"** (2609.07078, KAIST, cùng nhóm với SLiM/SkateFormer; preprint) [V].
- *Tokenisation:* zero-pad về 25 khớp; patch (8 frame × 1 khớp); T=64.
  - Thay positional embedding tuyệt đối bằng **Semantic Joint Embedding**: mã T5 của "A human skeleton joint of the {name}".
  - 5 **Canonical Joint Slots** học được, lặp theo T/8, được điền bằng masked self-attention; chỉ giữ [CLS] và slot.
- *Mục tiêu:* EMA teacher (0.994→1). L_Cano là CE prototype trên slot, mask 90% luôn bật. L_DINO trên [CLS]. K=65,536. KoLeo 0.1 + Sinkhorn.
- *Kết quả:*
  - NTU-60 xsub linear 85.7 (SOfA-L 87.0).
  - Cảm biến chưa thấy (Florence, SBU) với encoder đóng băng: 91.7/97.0 so với MAMP 73.7/80.2.
  - 4.30 so với 28.32 GFLOPs.
- *Ablation:*
  - Không slot, không SJE: 75.2; chỉ slot: 84.9; chỉ SJE: 76.4; cả hai: 85.7.
  - Số slot 0/5/10/15 cho 76.4/85.7/85.5/85.3.
  - Topology tổng hợp 30 khớp: 85.5.
- *Giới hạn:* chỉ 3-D, 15–25 khớp, không có tay/mặt.
- *Liên hệ:* đây là cơ chế đã kiểm chứng để có token không phụ thuộc bố cục keypoint, và là một **nút thắt slot nhỏ**: vài slot cho mỗi bộ phận (tay trái, tay phải, mặt, thân) đặt trên token signature theo path.

**HSP, Heterogeneous Skeleton** (2506.03481, CVPR'25) [V].
- Hợp nhất skeleton Kinect 3-D 25 khớp (J) và HRNet 2-D 17 khớp (C) thành một "prompted unified skeleton" 30 khớp: prompt học được lấp các khớp thiếu, MLP nâng 2-D lên 3-D.
- Thêm luồng S: hướng chuyển động theo trục, ánh xạ sang CLIP-text.
- Mục tiêu: nhất quán MSE + VICReg; không có mask.
- NTU-60 xsub: J 80.2, **C 84.4**, J+C 86.1, J+C+S 87.8.
- Bỏ prompt (zero-pad): 58.2→57.2 trên PKU-II.

**A4Mer / Action Motifs** (2604.28173 v3, Kyoto; "CVPR 2026 Highlight" theo comment arXiv) [V].
- *Token biến độ dài:* ranh giới "Action Atom" khi sai số ngoại suy tuyến tính (z-norm theo khớp) nhảy vọt. Motif được tạo bằng k-means 512 cụm cộng khai phá chuỗi con.
- Mỗi segment được thu về **1 token** bằng cross-attention; self-attention bị giới hạn trong segment.
- *JEPA:* smooth-L1 với EMA target (0.996), che 50% token, **mask phân cấp** (che motif thì che luôn các atom của nó).
- *Chống sụp:* tách token thành phần trung bình chuỗi (global) và độ lệch (local), L = L_local + 0.05·λ_k·L_global.
- *Ablation (Tab.3):*

  | Biến thể | Recog | Pred (mm) |
  |---|---|---|
  | Đầy đủ | 38.1 | 150 |
  | Token per-frame | 31.5 | 309 |
  | Token per-clip | 26.3 | 208 |
  | Target BERT | 34.8 | 169 |
  | Không tách global/local | 15.1 | 222 |
  | Full attention | 20.6 | 212 |

- *Liên hệ:* một segment cắt tại ranh giới động học sinh ra một token, và log-signature của segment là bản tóm tắt cỡ cố định tự nhiên cho token đó [suy luận].

**SignSeek** (2609.03695, Surrey; venue [chưa xác minh]) [V].
- Keypoint ký hiệu 2-D kèm confidence, tách theo articulator (thân, mặt, tay trái, tay phải); encoder tay dùng chung, tay phải được lật gương.
- Lõi là contrastive *có nhãn gloss* (266K mẫu), cộng MAP: masked latent prediction với stop-grad, cosine.
- Che articulator nổi bật nhất: R@1 ASL Citizen 48.40 (chỉ GAC) → 55.93 (+MAP) → 57.66 (đầy đủ).
- Chọn articulator ngẫu nhiên: 52.22.
- Lợi ích của MAP được đo *trên nền có giám sát*.

**SignDino** (2609.06296, NYUAD; dự án đã ghi nhận ở P102) [V: §4.6, Tab.4].
- Dùng crop RGB của từng articulator (DINOv3 đóng băng) chứ không dùng skeleton, nhưng là bằng chứng ở miền ký hiệu cho **view toàn cục và cục bộ theo thời gian**.
  - Teacher nhận crop toàn cục T_g ∈ U{64..96}; student nhận thêm crop cục bộ T_l ∈ U{10..32}.
  - Ép teacher chỉ dùng cửa sổ ngắn là ablation tệ nhất: −2.1 BLEU, R@1 0.704→0.65.
  - Bỏ Gram anchoring: −1.4 BLEU, R@1 0.67.
  - Mask chỉ theo frame là tốt nhất trong 5 chiến lược.
- Các số này đo ở chế độ fine-tune.

**Khác (đóng góp thấp, chỉ ghi lại).**
- *MaskCLR* (CVPR'24, có giám sát): mask theo attention (top δ=0.7) thắng random 91.7 so với 89.7. Kiểm tra dịch chuyển bộ ước lượng tư thế (HRNet→ViTPose): 78.3 so với MotionBERT 51.7 [V: Tab.2, Tab.4]. Đây là **giao thức đáng chép** cho keypoint 2-D.
- *MASQ* (2608.29891): che *toàn bộ quỹ đạo* một khớp thắng che frame, MoF 52.3 so với 45.5 [V: Tab.5].
- *DiMoP* (2609.35444, T-BIOM): diffusion trên khớp bị che; 90% là tốt nhất; random thắng motion-aware; NTU-60 86.8 [V: §IV]. Cùng hướng với MacDiff: target toàn chuỗi thì không cần mask có cấu trúc.
- *QCLR* (ACCV'26), M3GCLR, VCL, DoGCLR, TranCLR: contrastive, không có latent prediction; bỏ qua.

### 3.3 Sự kiện về tỉ lệ mask (đơn vị token khác nhau nên không so trực tiếp)

| Phương pháp | Đơn vị token | Tỉ lệ tốt nhất | Nguồn |
|---|---|---|---|
| MAMP / S-JEPA / Skeleton2vec | khớp × 3–4 frame | 90%; 95% thì rơi mạnh | L3, B R5 |
| SLiM | khớp × 8 frame | U(0.5, 0.9) | L3 |
| SOfA | khớp × 8 frame (L_Cano) | 90% | [V: 2609.07078 §E] |
| **Path-JEPA** | path × window signature, che theo khớp qua mọi window | **20–25% khớp** (10%: 85.1; 40%: 84.9) | [V: 5207 Tab.5] |
| A4Mer | segment biến độ dài | 50% | [V: 2604.28173] |
| GLaS-JEPA / Laya / W-JEPA | span 200 ms / block 5–10 patch / cặp bin | 50% / 60% / 65% | [V] |
| iBOT | patch (tỉ lệ dự đoán) | 0.3±0.2 | [V] |

**Đọc bảng [suy luận].** Token càng "tóm tắt" (signature cửa sổ dài, segment) thì tỉ lệ tối ưu càng thấp. Token càng thô và dư thừa (khớp × vài frame) thì càng cần 90% để chặn đường nội suy. Tỉ lệ cho signature token *phải được quét lại*; không được kế thừa 90%.

---

## 4. Tokenizer có cấu trúc: signature, part, motion, đa thang, global/local

### 4.1 Path signature như tokenizer

| Bài | Thiết kế token | Bằng chứng | Nguồn |
|---|---|---|---|
| **Rough Transformer** (2405.20799, NeurIPS'24) | "multi-view": [S(0,t_k) ‖ S(t_{k−1},t_k)]; global là **prefix nhân quả**, khác nhau theo từng token (Path-JEPA dùng [0,T] chung) | Global/local được chọn *theo từng dataset*: G ở 4/6, L ở 1/6, G+L chỉ ở 1/6 (Tab.6). Level 1 tương đương downsample: EW 64.1 so với 90.2 ở level 2. Drop frame 50%: 87.7 so với Neural-RDE 60.7. Nhanh hơn 1.4–26× | [V: Eq.8, Tab.5, 6, 14, 15] |
| **LogSig-SSM** (2610.05051, NeurIPS'26) | log-sig đa thang trên các cửa sổ trailing, K thang, nhánh global tuỳ chọn | Log-sig d2 71.7 > d1 69.5 > scattering 67.6 > conv 62.9 > Mamba thô 58.6. Ép d1 tốn −7.2 trên dataset ít kênh. Signature (không log) −3.1. Nhánh global chỉ được chọn ở 2/6 | [V: Tab.4, Tab.11, §5.5] |
| **LogSig-RNN** (2110.13008, BMVC'21) | embedding → cumsum → +time → log-sig M=2 trên N segment, **+ điểm đầu segment** | N=4 tốt nhất (92.94); N=64 còn 91.50. Bền khi drop hoặc lặp frame ≥50% | [V: Tab.1c, §5.3] |
| **GCN-DevLSTM** (2403.15212, TMLR'26) | *path development* vào nhóm Lie, kích thước m² không phụ thuộc d | NTU-60 joint 90.8/95.7. Drop 47%: −6.07 so với CTR-GCN −18.18 | [V: Tab.2, §5.4] |
| Li et al. (1811.07081, AAAI'19) | cặp khớp (không gian), khớp+time (thời gian), dyadic L_D=3 | level 1→4: 85.5→89.2, lợi ích giảm dần | [V: Tab.2] |
| Yang, Lyons et al. (1707.03993) | mọi cặp (level 2) và bộ ba (level 3), multi-delay lead-lag, dyadic h=3 | Tư thế 2-D AlphaPose: 68.2; **+confidence làm kênh: 75.7**; GT: 84.2. h>3 bị nhiễu cục bộ lấn át | [V: §6.5, §6.7] |
| DyG-WM (2610.19364) | log-sig level 2 làm **ngữ cảnh** cho predictor JEPA (không làm target) | bỏ signature: AUC 74.99→59.09 | [V: Tab.4] |
| Cheng et al. TMM (learnable paths) | — | không có trên arXiv; 94.18 Chalearn chỉ có dạng [V\*] | [chưa xác minh] |

**Lý thuyết dùng cho thiết kế** [V: 2206.14674; 2110.13008 Thm B.2; kiểm số CPU]:
- *Số chiều log-sig (Witt).* d=4 (2-D lead-lag): level 2 → 10, level 3 → 30. d=6: 21/91. Chuỗi 3 khớp 2-D ghép kênh có lead-lag (d=12): 78/650. Hai bàn tay ghép phẳng (d=42): 903/25,585. Không bao giờ ghép phẳng 21 khớp tay.
- *Bất biến reparameterisation, nhưng mất tốc độ.* Thêm kênh thời gian khôi phục tốc độ nhưng làm giảm độ bền với fps; ở level cao thì giảm nhanh hơn [V: 1707.03993 Fig.7].
- *Bỏ mất vị trí tuyệt đối và cấu hình tĩnh.* Phải nối điểm đầu, vị trí trung bình hoặc vị trí tương đối với thân.
- *Lead-lag* bắt quadratic (cross-)variation [V: 1707.03993 §4.2.3]. Theo khảo sát HAR 2610.06553, lead-lag cạnh tranh được nhưng không bao giờ tốt nhất, và số chiều bùng nổ [V].
- *Đẳng thức Chen:* xem §0 điểm 3. Với global dạng prefix, local bị che = prefix_{k−1}⁻¹ ⊗ prefix_k, tức là rò rỉ *chỉ từ hai token lân cận*.
- *Tìm kiếm:* không thấy bài nào dùng signature làm *target* trong SSL masked/contrastive ngoài Path-JEPA [S: 5 truy vấn].

### 4.2 Token theo bộ phận (part tokens)
- **SOKE** (2411.17799) [V: Tab.4]. Tokenizer VQ tách thân / tay trái / tay phải thắng VQ toàn thân trên mọi chỉ số DTW, ví dụ How2Sign body/hand 9.49/3.56 → 6.82/2.35. Tách L/R tốt hơn gộp hai tay. Không có codebook riêng cho mặt, và đây là bằng chứng về *sinh*, không phải biểu diễn.
- **SkateFormer** (2403.09508, có giám sát) [V: Tab.4]. Attention theo phân hoạch (bộ phận gần/xa × thời gian cục bộ/strided) đạt 92.6, so với full attention 90.6 và baseline không attention 90.7. Chỉ phân hoạch khớp: 91.8; chỉ phân hoạch thời gian: 91.9.
- **SOfA** slot (§3.2) và **SignSeek** articulator (§3.2). Sự thiếu vắng ablation per-joint vs part *trong SSL* (B §c) **vẫn chưa được lấp**. SOfA chỉ so có slot với không slot.

### 4.3 Motion tokens, đa thang và global/local
- *Motion token biến độ dài:* A4Mer (§3.2). *Lượng tử hoá theo frame:* MASQ.
- *Đa thang trong JEPA:*
  - W-JEPA (2610.19411) [V]: token (dải wavelet × bin) gắn embedding kênh/dải/bin; mask *theo cặp*, ρ=0.65; chỉ dùng hinge phương sai. Wavelet 80.6 so với patch 73.1. Swap regulariser: SIGReg 89.74 < VICReg 90.66 < hinge 92.91 (UCI-HAR, fine-tune).
  - GFP: target 5/10/30 frame + global (L3). AMR: cửa sổ 4/8/12 (L3).
  - Path-JEPA: 2 thang (25, 50).
- *Global + local:*
  - Path-JEPA: G 84.4, L 83.3, G+L 85.8.
  - SignDino: cửa sổ teacher ngắn −2.1 BLEU.
  - SLiM GLCL: +2.4 trên MFP (B R8).
  - LeVJEPA: thêm local view tăng 47.6→50.2.
  - A4Mer: tách global/local là bắt buộc (15.1 so với 38.1).
  - Ngược lại, Rough Transformer và LogSig-SSM thường **không** chọn nhánh global.
  - **Kết luận:** ngữ cảnh toàn cục là cần, nhưng *dạng* của nó (nối vào từng token, token riêng, hay view teacher) phải được ablate.

---

## 5. Tổng hợp: 12 nguyên tắc thiết kế cho sign-skeleton JEPA với token path-signature

Mức bằng chứng: **M** = mạnh (≥2 nguồn độc lập, hoặc ablation khớp lớn hơn 3 điểm); **V** = vừa; **Y** = yếu hoặc suy luận.

**N1. Signature phải đi kèm trạng thái: nối thêm vị trí và cấu hình. (M lý thuyết, V thực nghiệm)**
Signature bỏ mất vị trí tuyệt đối và triệt tiêu đoạn tĩnh (2206.14674). LogSig-RNN và GCN-DevLSTM đều nối start point. Yang/Lyons cho thấy thêm confidence làm kênh tăng 68.2→75.7. Mỗi token nên gồm [log-sig(window) ‖ vị trí đầu hoặc trung bình của window, tương đối với vai/cằm ‖ cấu hình ngón chuẩn hoá theo cổ tay ‖ confidence trung bình]. Đây là nguyên tắc **khác biệt nhất** so với Path-JEPA gốc. NTU gần như không có "hold", còn ký hiệu thì có.

**N2. Dùng level ≥ 2, log-signature, trên path ít chiều; không ghép phẳng bàn tay. (M)**
- Level 1 tương đương downsample (RFormer 64.1 so với 90.2).
- Depth 2 tốt hơn depth 1 2.3 điểm (LogSig-SSM); log-signature tốt hơn signature 3.1 điểm.
- Với 2-D lead-lag, path đơn khớp hoặc cạnh có 10 chiều. Chuỗi ngón nên là chuỗi các cạnh *tương đối* (wrist→MCP→…, mỗi cạnh 2-D), hoặc path development (GCN-DevLSTM) nếu số chiều vượt khoảng 100.
- Không ghép 21 khớp tay thành một path (903 chiều trở lên).

**N3. Chỉ dùng kênh thời gian khi cần, và ablate nó. (V)**
Bất biến reparameterisation là lý do Path-JEPA bền với fps. Ký hiệu cần một phần thông tin tốc độ (hold so với movement, lặp lại), nhưng không cần tốc độ riêng của từng người ký. Có hai lựa chọn: (a) không có kênh t, và đưa tốc độ vào qua độ dài window hoặc embedding thời gian; (b) thêm t với trọng số nhỏ. Thử cả hai, đo bằng probe tốc độ người ký như một chỉ số tiêu cực (B "Frozen in a Frame" cũng gợi ý probe vận tốc).

**N4. Thu nhỏ window; đa thang dyadic, nhưng không quá ngắn. (V)**
- Window 25/50 frame của Path-JEPA xấp xỉ độ dài cả ký hiệu.
- Bằng chứng về giới hạn dưới: LogSig-RNN N=4 tốt nhất và N=64 kém hơn; Yang h=3 tốt nhất, sâu hơn thì bị nhiễu; RFormer cần window lớn hơn 2 điểm.
- Đề xuất cho ASL Citizen ở 30 fps [suy luận]: các thang khoảng 8/16/32 frame, stride một nửa, cộng một view toàn clip.
- Cân nhắc segment biến độ dài tại ranh giới động học (A4Mer: per-frame 309 mm so với segment 150 mm).

**N5. Chống rò rỉ là ràng buộc cứng; global view phải tương thích với mask thời gian. (M lý thuyết, V thực nghiệm)**
- Lan mask theo phụ thuộc: khớp → cạnh → chuỗi (Path-JEPA 87.6; L2 Point-M2AE −4.5 khi lộ).
- *Mới:* nếu thêm mask thời gian (che window k của path p), thì phần [0,T] global của mọi token cùng path, và phần prefix của token lân cận, sẽ cho phép tính chính xác window bị che (Chen).
- Ba lựa chọn hợp lệ: (a) bỏ global khỏi token context và để encoder tự tổng hợp; (b) tính "global" chỉ từ các window còn thấy; (c) coi global là một token riêng, bị che cùng mọi window của path.

**N6. Mask hỗn hợp: không gian (subtree / articulator) + thời gian (span ngắn); quét tỉ lệ từ thấp. (V)**
- Path-JEPA: subtree +1.3; tối ưu 20–25% khớp.
- Mask trải qua mọi window tương đương "whole-trajectory" của MASQ (+6.8).
- Mask articulator theo saliency của SignSeek: +5.4 R@1 trên ASL Citizen (57.66 so với 52.22) so với chọn ngẫu nhiên, trong bối cảnh có giám sát.
- BEST và SignBERT+: hỗn hợp tốt hơn từng họ đơn (L2).
- EEG 2609.33487: bán kính không gian vừa phải với block thời gian ngắn.
- **Cấm:** che mọi path tại cùng một window (bias-inflation), và che chỉ thân hoặc chân (BEST body-only thấp hơn cả không pretrain).
- Quét tỉ lệ token theo lưới {15, 25, 40, 60}%, có lấy mẫu ngẫu nhiên tỉ lệ (SLiM, iBOT).

**N7. Token theo bộ phận, với tay trái và tay phải tách riêng; định danh khớp bằng ngữ nghĩa và cấu trúc. (V)**
- SOKE: tách L/R thắng gộp hai tay.
- SkateFormer: attention theo phân hoạch thắng full attention (+2.0).
- SOfA: joint-name embedding + slot thắng tuyệt đối 75.2→85.7; 5 slot tốt hơn 15.
- Đề xuất: positional = (loại path, articulator, độ sâu động học d_G, thang ℓ, thời gian dạng RoPE). Có thể thêm 2–4 slot cho mỗi articulator làm nút thắt hoặc token đọc ra.

**N8. Chống sụp: EMA + centering làm mặc định; prototype-CE và SIGReg-trên-token là hai nhánh ablation. (M cho EMA, V cho các nhánh còn lại)**
- EMA là bắt buộc trong S-JEPA (không có EMA: 1.6%). Path-JEPA dùng centering + cosine.
- Prototype-CE: SLiM 87.66 so với feature-MSE 85.07 (B R1); SOfA cũng dùng.
- Nhánh không EMA theo GLaS/Laya: stop-grad target qua projector khoảng 128-d, SIGReg marginal trên token, λ 0.01–0.05; λ<0.02 thì Laya sụp.
- Điều chỉnh τ của EMA theo batch (C §1.5).

**N9. Đặt regulariser theo hướng giữ lại dynamics. (V)**
- Trừ trung bình cục bộ trước SIGReg (TC-SIGReg, W=4–8: 63.6→83.8).
- Hoặc dùng SIGReg theo thời gian trong từng mẫu (LeNEPA).
- Lý do: hình thể và tư thế người ký là "slow feature" (Sobal, L1 R8).
- Regularise và đo rank cả ở backbone (λ-JEPA). Với batch nhỏ, cân nhắc OOB cache.

**N10. Target đa mức và loss ở cả token context nếu cần đặc trưng theo frame. (V)**
- Nối z-score nhiều block thay vì trung bình: Bootleg +10, seed std 2.9→0.6, ít nhạy τ (C §1.2).
- V-JEPA 2.1 context loss có trọng số theo khoảng cách cho đặc trưng dày (L1).
- A4Mer: tách loss global/local là bắt buộc.

**N11. Luôn có một mục tiêu cấp clip (global view hoặc [CLS]) bên cạnh masked prediction. (M)**
- iBOT không có [CLS] loss: kNN 9.5.
- SLiM MFP+GLCL 87.7 so với MFP 85.3.
- STARS: kNN +16.8 (B R7).
- SignDino: view teacher dài là quan trọng nhất.
- Thêm KoLeo (0.1) cho retrieval (DINOv2 Oxford +8.3).
- Mục tiêu motion và content trên chung encoder: hệ số khoảng 0.1, VC ở nhiều tầng (MC-JEPA).

**N12. Giao thức đánh giá chống tự lừa. (M)**
- Missing keypoint đi qua mask token, không bao giờ zero-fill (SLiM: 87.2 so với 49.6).
- Kiểm tra dịch chuyển bộ ước lượng tư thế (MaskCLR: HRNet→ViTPose).
- Kiểm tra fps đúng nghĩa: thay đổi số frame, không nội suy trở lại.
- Probe handshape và location, với signer-ID làm chỉ số tiêu cực.
- ≥3 seed vì JEPA nhiễu theo seed (C: I-JEPA ±2.9).
- Đo rank ở backbone và ở token (EEG: VICReg không thấy bias-inflation).

---

## 6. Khoảng trống và rủi ro

1. **Không có bằng chứng** nào về signature-JEPA trên dữ liệu 2-D, tay, mặt hay ký hiệu. Toàn bộ bằng chứng Path-JEPA là NTU 3-D 25 khớp, một lần chạy, chưa có code.
2. Chưa có ablation per-joint vs part trong skeleton SSL (B §c vẫn đúng). Chưa có SIGReg trên skeleton. Chưa có frozen teacher (SALT) trên skeleton.
3. Lợi ích của Path-JEPA (+2.3 so với S-JEPA) chủ yếu đến từ token *quan hệ* (+1.8 từ cạnh và chuỗi) và global/local (+1.4), không phải từ "signature" thuần: global-joint-only 84.4 < S-JEPA 85.3. Cần một baseline token quan hệ *không* signature (ví dụ increment hoặc vận tốc theo window) để tách hai hiệu ứng.
4. Gaussian target có thể bóp méo latent dạng góc (2609.21656, chỉ toy). Ablation target cầu có chi phí thấp nhưng chưa có tiền lệ.
5. Phần lớn công trình SIGReg 2026 là preprint đơn lẻ, quy mô nhỏ (speech, EEG, robot). Chỉ coi chúng là giả thuyết.

**Ablation tối thiểu gợi ý, theo thứ tự:**
- (a) N1: có và không có state channel.
- (b) N5/N6: chỉ mask không gian kiểu Path-JEPA, so với hỗn hợp không gian + thời gian (với global đã sửa rò rỉ); tỉ lệ ∈ {15, 25, 40}%.
- (c) N4: thang window.
- (d) N8: EMA-cosine, so với prototype-CE, so với SIGReg-token (±TC).
- (e) Baseline token quan hệ không signature (rủi ro 3).

---

## 7. Chỉ mục nguồn (id → venue như bài tự ghi; [S] nếu chỉ từ comment)

- **Path-JEPA:** ECCV'26 (toàn văn `pathjepa/5207.txt`).
- **JEPA và regulariser (mới):**
  - GLaS-JEPA 2609.37798 (preprint)
  - LeVJEPA 2608.27395 (preprint)
  - λ-JEPA 2609.35288 (preprint)
  - TC-SIGReg 2607.26924 (preprint)
  - OOB 2610.20158 (preprint)
  - Beyond Gaussian 2609.21656 (preprint)
  - LeNEPA 2607.00958 (MiLeTS'26 [S])
  - EEG masking geometry 2609.33487 (preprint)
  - Laya 2603.16281
- **JEPA (đọc lại để lấy số liệu):** MC-JEPA 2307.12698 · DINOv2 2304.07193 (TMLR'24) · iBOT 2111.07832 (ICLR'22).
- **Skeleton:**
  - SOfA 2609.07078 (preprint)
  - HSP 2506.03481 (CVPR'25)
  - A4Mer 2604.28173 (CVPR'26 Highlight [S])
  - SkateFormer 2403.09508 (ECCV'24, theo trích dẫn Path-JEPA)
  - MaskCLR (CVPR'24, PDF CVF)
  - MASQ 2608.29891
  - DiMoP 2609.35444 (T-BIOM)
  - STARS 2407.10935 (WACV'26)
- **Ký hiệu:** SignSeek 2609.03695 · SignDino 2609.06296 · SOKE 2411.17799.
- **Signature:** Rough Transformer 2405.20799 (NeurIPS'24) · LogSig-SSM 2610.05051 (NeurIPS'26 [S]) · LogSig-RNN 2110.13008 (BMVC'21) · GCN-DevLSTM 2403.15212 (TMLR'26) · Li 1811.07081 (AAAI'19) · Yang/Lyons 1707.03993 · Lyons & McLeod 2206.14674 · HAR study 2610.06553 · DyG-WM 2610.19364.
- **Wearable:** W-JEPA 2610.19411.
