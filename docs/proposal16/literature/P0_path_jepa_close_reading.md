# P0. Đọc kỹ Path-JEPA (ECCV 2026)

*Singh, A. & Singh, A. "Path-JEPA: Path Signature Based Predictive Learning for Skeleton Action Recognition". ECCV 2026, LNCS 17054, tr. 110–128, DOI `10.1007/978-3-032-37258-1_7`. PDF đọc toàn văn: `https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf`, 18 trang, tải ngày 2026-10-16.*

Quy ước nhãn: **[P]** là nội dung paper nói rõ (kèm số mục, bảng hoặc phương trình). **[S]** là suy luận của ta khi đọc. Code chính thức **chưa phát hành**: repository mới chỉ có trang dự án (kiểm tra 2026-10-11, ghi ở `docs/proposal12/research_update_20261011/code/pathjepa/README.md`). Vì vậy mọi chi tiết không có trong PDF đều là [S].

---

## 1. Method: paper thực sự làm gì

### 1.1 Pipeline

| Bước | Nội dung [P] |
|---|---|
| Input | $X\in\mathbb R^{T\times J\times 3}$: NTU 25 khớp 3D, PKU-MMD |
| Liên tục hoá | Nội suy tuyến tính từng khớp. **Lead–lag theo độ trễ** $\tilde X^{LL}_j(t)=[\tilde X_j(t),\tilde X_j(t-\delta)]$, $d'=6$ (Eq. 1). Giá trị $\delta$ không được nêu |
| Signature | Log-signature bậc $n=2$, $D_{log}=22$ hệ số mỗi path (§3.1, §4) |
| Họ path | Joint ($J$ path), joint-pair theo cạnh động học ($J-1$ path, $Y_{u,v}=X_v-X_u$), chain (shoulder→elbow→wrist, hip→knee→ankle…; ghép các displacement cạnh theo thứ tự giải phẫu) (§3.2) |
| Thang thời gian | Hai thang: $\ell=0$ cửa sổ 25 frame stride 12, $\ell=1$ cửa sổ 50 frame. Tổng cộng $L_{total}=4$ cửa sổ và $N_{total}=152$ token mỗi chuỗi (§4) |
| Token multi-view | $m_{p,\ell,k}=[\log S(\tilde X_p)_{0,T},\ \log S(\tilde X_p)_{t_{k-1},t_k}]$, $D_{mv}=44$. Tức **mỗi token mang cả signature toàn cục của path đó** (Eq. 3) |
| Embedding | Projection tuyến tính riêng cho từng loại token (joint/pair/chain), type embedding, structural embedding $(\ell, d_G(p))$ với $d_G$ là độ sâu động học (Eq. 5) |
| Attention | Logit attention có thể cộng bias học được theo khoảng cách thời gian, thang, IoU cửa sổ và khoảng cách đồ thị động học (Eq. 6, "optionally") |
| Masking | Tính token trên chuỗi đầy đủ rồi mới che (không bao giờ zero toạ độ trước khi tính signature). Chọn tập khớp $\mathcal M$ và che **mọi token mà path của nó chạm vào khớp bị che, ở mọi cửa sổ** (Eq. 7). Seed được lấy theo xác suất tỉ lệ với chuyển động, rồi mở rộng ra cả subtree động học (§3.4) |
| JEPA | Student chỉ thấy token context. Predictor 4 layer cross-attention từ query metadata (type, path ID, tâm thời gian, thang, độ sâu) vào context. Teacher EMA thấy toàn bộ token. Target là feature teacher tại vị trí bị che, trừ đi running-mean center. **Loss cosine** (Eq. 8–10) |
| Kiến trúc | Encoder 6 layer, predictor 4 layer, d = 512, 8 head, FFN 2048. 300 epoch, AdamW lr 3e-4, wd 0.05, batch 256, EMA 0.996→1.0, centering β = 0.9 (§4) |
| Feature cho downstream | Teacher encoder (§3.5) |

### 1.2 "Target là path signature": diễn giải đúng

Abstract viết rằng method "adopts path signatures as the prediction target". Nhưng §3.5, Eq. 9–10 cho thấy **target là embedding của teacher EMA** tính trên các token signature. Paper **không** regress hệ số signature [P].

→ Path-JEPA = **(i) tokenizer signature** + **(ii) target latent kiểu S-JEPA/I-JEPA** + **(iii) masking đóng theo phụ thuộc** [S].

Phân biệt này quyết định cách ta đọc kết quả nội bộ: các thử nghiệm trước của ta dùng log-signature làm **target tường minh**, và đều không giúp:
- S2: λ_signature, `docs/proposal12`;
- SignPathLab E03/E04 (proposal13).

Những thử nghiệm đó **không phải** là thử Path-JEPA. Riêng dạng "signature làm input token + target latent" chưa từng được chạy: arm `signature_input_4k` của S2 có cài nhưng không chạy.

### 1.3 Kết quả chính [P]

| Benchmark | Path-JEPA | Tốt nhất trước đó | Δ |
|---|---:|---:|---:|
| Linear, NTU60 X-Sub | 87.6 | 85.9 (GFP) | +1.7 |
| Linear, NTU60 X-View | 92.9 | 92.0 (GFP) | +0.9 |
| Linear, NTU120 X-Sub / X-Set | 81.5 / 81.8 | 79.6 / 80.3 | +1.9 / +1.5 |
| Linear, PKU-I / PKU-II | 93.4 / 56.7 | 92.2 / 56.2 | +1.2 / +0.5 |
| Fine-tune, NTU60 X-Sub | 94.4 | 93.9 (GFP) | +0.5 |
| Semi 1%, NTU60 X-Sub | 72.2 | 71.8 (GFP) | +0.4 |
| Transfer NTU60→PKU-II | 72.6 | 71.4 (S-JEPA) | +1.2 |

**Ablation** (Table 5–6, linear NTU60 X-Sub):

| Thay đổi | Acc. | Ý nghĩa cho ta |
|---|---:|---|
| Joint + global only | 84.4 | |
| + local | 85.8 (+1.4) | đa thang có ích |
| Local only, không global | 83.3 | global view quan trọng (−2.5) |
| + edge | 86.4 (+0.6) | |
| + chain | 87.6 (+1.2) | path nhiều khớp có ích |
| Random 20% joint | 84.1 | |
| Motion-aware (S-JEPA) | 86.8 | |
| Signature-augmented closure, seed ngẫu nhiên | 86.3 | |
| + kinematic subtree | 87.6 | che theo khối giải phẫu |
| Kinematic 10% / 40% | 85.1 / 84.9 | tỉ lệ che tối ưu 20–25% khớp (tính trên khớp, không phải token) |

**Tần số khung hình** (Table 4, test resample): Path-JEPA dao động 87.1–88.1 trên dải 15–60 fps. S-JEPA mất 2.9 ở 15 fps.

Một điểm [S] quan trọng: giao thức này nội suy chuỗi về lại độ dài gốc, nên đây là kiểm tra **sampling jitter / mất chi tiết**, không phải kiểm tra tốc độ ký thật sự.

---

## 2. Điểm mơ hồ và điểm yếu cần lưu ý [S]

1. **Mọi thí nghiệm chỉ 1 run, không có seed hay CI.** Phần lớn Δ nằm trong khoảng 0.5–2 điểm.
2. **Không có control "cùng tokenizer, khác target".** Không biết lợi ích đến từ tokenizer signature hay từ cách che mới. Bảng 6 thay đổi tokenizer nhưng giữ target latent; không có arm "token toạ độ + cùng closure masking + cùng kiến trúc".
3. **$D_{log}=22$ với $d'=6, n=2$:** công thức Witt cho 6 + 15 = 21 hệ số Lie. Con số 22 có thể là có giữ hệ số 0 hoặc một kênh thời gian. Không xác định được khi chưa có code (đã ghi ở `docs/proposal12/research_update_20261011/RESEARCH_LOG.md`).
4. **Số token 152 không khớp rõ ràng** với "25 + 24 + chain" × 4 cửa sổ. Ví dụ (25 joint + 24 edge + ~?) × 4 > 152 với NTU 2 người. Có thể chain ít, hoặc token chỉ tính cho một số cửa sổ. Không tái tạo đúng được.
5. **Global view được copy vào mọi token của cùng path.** Vì masking chỉ theo không gian (che path ở mọi cửa sổ), điều này không rò rỉ trong setting của paper. Nhưng **nếu ta thêm masking theo thời gian** (PLctx dùng near time-block 50%), global view của chính path đó ở cửa sổ khác sẽ chứa thông tin của cửa sổ bị che → rò rỉ. **Phải sửa khi chuyển sang sign** (xem §7 của proposal: global view chỉ tính trên các cửa sổ nhìn thấy, ghép bằng định lý Chen).
6. **Lead–lag theo độ trễ δ phá một phần tính bất biến reparameterization:** δ tính bằng frame (hoặc giây) là một thang thời gian cố định. Định lý $S(x\circ\phi)=S(x)$ chỉ đúng tuyệt đối cho path gốc, không cho path lead–lag có độ trễ cố định.
7. **Signature bất biến tịnh tiến:** level-1 chỉ là độ dời. Path đứng yên → signature = 0. Với NTU (3D, nhiều chuyển động toàn thân) điều này ít hại. Với sign, **handshape và location chủ yếu là tĩnh** → cần kênh tĩnh song song (basepoint / anchor).
8. **Motion-aware seed phụ thuộc nội dung:** vị trí mask tự nó mang thông tin về nơi có chuyển động. Đây là rò rỉ nhẹ mà PLctx cố ý tránh (mask độc lập nội dung).

---

## 3. Chuyển giao sang sign language: cái gì mang theo được [S]

| Thành phần Path-JEPA | Mang theo? | Lý do / điều chỉnh |
|---|---|---|
| Target latent EMA, centering | **Có** | Trùng với bằng chứng nội bộ: target latent ctx thắng target tường minh (PLctx > PLint; S1, S2 âm) |
| Loss cosine | Thử. Mặc định giữ centered-CE của PLctx | PLctx đã ổn định với centered-CE. Đổi loss là một factor riêng |
| Token signature đa thang (global + local) | **Có**, nhưng **lai** với token tĩnh | Signature mất static; sign có handshape và location tĩnh |
| Path joint/edge/chain | **Có**, đổi nghĩa | Chain = ngón tay (5 × 2 tay) và cánh tay. Edge = xương bàn tay |
| Path quan hệ khác nhánh động học | **Thêm mới** | Path tay–mặt, tay–ngực, tay–tay. Trong sign, location và contact là quan hệ giữa các khớp **không kề nhau** trên cây động học |
| Path ghép hai tay 4D | **Thêm mới** | Phân biệt chuyển động hai tay cùng pha với luân phiên (Battison). **Đã sửa sau P3 §1.4:** Lévy area chéo $(y_D,y_N)$ trên cả chu kỳ tỉ lệ với $\sin\varphi$, nên bằng 0 ở *cả* cùng pha lẫn luân phiên. Phải dùng kênh tổng/hiệu $D\pm\mathcal M(N)$ sau khi gương hoá, hoặc signature theo cửa sổ con, hoặc cross-QV của lead–lag |
| Closure theo phụ thuộc (Eq. 7) | **Có, tổng quát hoá** | Support = (khớp × khoảng thời gian), không chỉ khớp. Mask đóng theo cả không gian lẫn thời gian |
| Kinematic subtree | **Có**, dạng "articulator / kênh âm vị học" | Subtree ngón tay, subtree tay = kênh CONF; arm = LOC. Trùng với R3 của PLctx |
| Motion-aware seed | **Arm thử nghiệm**, không mặc định | Rò rỉ nhẹ (§2.8); PLctx cố ý content-independent |
| Lead–lag | **Arm** | Lead–lag kiểu trễ δ của Path-JEPA không bằng QV/2 và phụ thuộc lưới lấy mẫu (P1 §1.5). Trên tay 2D nó chủ yếu đo jitter (P1 §1.8). Chẩn đoán D-SIG: +0.6 R@1 với số chiều ×3.2, thấp hơn mức tăng của smoothing (+0.7) — xem proposal §5.3 |
| Attention bias cấu trúc (Eq. 6) | Có, rẻ | Bias theo khoảng cách thời gian và khoảng cách giải phẫu |
| Bất biến tốc độ | **Đặc biệt có giá trị cho sign** | Chẩn đoán D-TEMPO: signer giải thích phần lớn biến thiên thời lượng. Tempo là nhiễu theo signer |
