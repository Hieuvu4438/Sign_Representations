# P1. Path signature cho skeleton, gesture và sign language: tổng quan có kiểm chứng

*Viết ngày 2026-10-16 cho proposal16, bản chuyển thể Path-JEPA (ECCV 2026, Singh & Singh, LNCS 17054, tr. 110–128, DOI `10.1007/978-3-032-37258-1_7`) sang skeleton SSL cho ngôn ngữ ký hiệu.*
*Đọc cùng `P0_path_jepa_close_reading.md` (đọc kỹ Path-JEPA), `D1_server_dataset_audit.md` (số liệu pose trên server) và `P3_sign_domain_signals.md`.*

**Quy ước nhãn**
- **[P]**: paper nói rõ. Số liệu đã được lấy từ text PDF/HTML gốc (arXiv, CVF, UCL Discovery) bằng `grep`, không lấy từ bản tóm tắt của WebFetch.
- **[V]**: ta tự kiểm bằng mô phỏng **CPU** (numpy và pySigLib 4.0.0, cài tạm vào scratchpad, không dùng GPU). Script nằm trong scratchpad của phiên: `sigcheck*.py`.
- **[S]**: suy luận của ta.
- **[chưa xác minh]**: chưa đối chiếu được với nguồn gốc.

---

## 0. Tóm tắt cho người thiết kế

1. Log-signature bậc 2 của một path tuyến tính từng đoạn có **công thức đóng**: gồm increment và diện tích Lévy, tính bằng hai `einsum`. Không cần thư viện. Ta đã đối chiếu công thức này với pySigLib: sai số khoảng 1e-14 [V].
2. Với $d'=6$, $n=2$, số chiều Lie (Witt) là **21**, không phải 22 như Path-JEPA ghi. Con số 43 trong paper là $1+6+36$, tức có tính cả số hạng hằng 1 [V].
3. Signature **mù với thông tin tĩnh**: một path đứng yên có signature bằng 0, và signature bất biến khi tịnh tiến. Nó cũng mù với chuyển động "đi rồi về" (tree-like, ví dụ động tác gõ lặp lại): log-signature bằng **0 chính xác** [V]. Muốn thấy số lần lặp phải có lead-lag hoặc kênh thời gian.
4. Nhiễu jitter phá phần bậc 2.
   - Diện tích Lévy của một path chỉ có nhiễu có độ lệch chuẩn $\approx 0.70\,\sigma^2\sqrt N$ (N là số frame).
   - Biến phân bậc hai (QV) do nhiễu tăng tuyến tính, $\approx 4(N-1)\sigma^2$ trong 2D [V].
   - Như vậy **tăng fps làm feature bậc 2 nhiễu hơn**. Tay trong pose 2D trên ASL Citizen có score thấp ở khoảng 42–48 % số frame (D1). Đây là rủi ro số 1 khi đưa sang sign language.
5. Chỉ tìm thấy **hai** công trình dùng signature trên dữ liệu sign language:
   - Cheng et al., TMM 2024, trên AUTSL: tăng **+0.25 đến +0.68** top-1 khi cắm vào GCN.
   - Shi et al., ICPR 2024, trên WLASL-300: **60.85 %** với 1.25M tham số.
   - **Không** có công trình nào dùng signature cho SSL trên sign language, ngoài Path-JEPA trên action.
6. Những lựa chọn có bằng chứng giúp ích:
   - path cặp khớp theo không gian, đây là mức tăng lớn nhất trong ablation của Li et al.;
   - bậc 2–3 là đủ;
   - cửa sổ phân cấp hoặc dyadic tốt hơn cửa sổ trượt (Morrill et al.);
   - kết hợp global và local (Path-JEPA);
   - một phép chiếu tuyến tính học được trước khi lấy signature (LogSig-RNN, Cheng et al.).
   **Chưa paper nào kiểm tra độ bền với nhiễu toạ độ.** Các bài chỉ kiểm tra việc bỏ hoặc chèn frame.
7. Global view $[0,T]$ trong token của Path-JEPA sẽ **rò rỉ chính xác** cửa sổ bị che nếu thêm temporal masking. Theo Chen identity, chỉ cần global và các cửa sổ còn lại là khôi phục đúng log-signature của cửa sổ bị che (sai số 1e-15) [V].
8. Path-JEPA **vẫn chưa có code**. Repo `singh17ashu/Path-JEPA` mới có README "Code coming soon", `index.html` và LICENSE; push cuối ngày 2026-10-05 (kiểm tra GitHub API ngày 2026-10-16).

---

## 1. Toán cần cho thiết kế

### 1.1 Định nghĩa

Cho path $x:[s,t]\to\mathbb R^d$ có biến phân bị chặn. Signature bậc $n$ là
$$S(x)_{s,t}^{\le n}=\Big(1,\ S^{(1)},\dots,S^{(n)}\Big),\qquad S^{(k)}_{i_1\dots i_k}=\int_{s<u_1<\dots<u_k<t}dx^{i_1}_{u_1}\cdots dx^{i_k}_{u_k}.$$

- Mức $k$ có $d^k$ hệ số. Tổng số hệ số là $\sum_{k=0}^n d^k$ (có số hạng hằng) [P, Path-JEPA §3.1; Lyons & McLeod arXiv:2206.14674].
- **Log-signature** là $\log S$ trong đại số tensor. Nó nằm trong đại số Lie tự do, nên bỏ được các quan hệ dư thừa kiểu shuffle, ví dụ $S^{(1,2)}+S^{(2,1)}=S^{(1)}S^{(2)}$ [P, Morrill et al. arXiv:2006.00873].
- Cái giá: log-signature **mất tính phi tuyến phổ quát**, tức không còn xấp xỉ mọi hàm liên tục bằng một hàm tuyến tính [P, Morrill et al.].
- Trên 26 bộ dữ liệu, với độ sâu tốt nhất chọn riêng cho mỗi bên, Morrill et al. thấy signature **tốt hơn** log-signature một cách có ý nghĩa thống kê: Wilcoxon p = 0.01; rank trung bình 1.25 so với 1.75 [P].
  Path-JEPA chọn log-signature mà không có ablation so với signature [P, đọc toàn văn].

### 1.2 Ý nghĩa bậc 1 và bậc 2

- **Bậc 1** là increment $S^{(1)}_i=x_i(t)-x_i(s)$. Nó chỉ phụ thuộc hai đầu mút, không phụ thuộc đường đi.
- **Bậc 2, phần đối xứng**: $\tfrac12(S^{(2)}_{ij}+S^{(2)}_{ji})=\tfrac12 S^{(1)}_iS^{(1)}_j$. Phần này **hoàn toàn suy ra được từ bậc 1** [V, sai số 1.4e-14]. Đó chính là phần dư thừa mà log-signature bỏ đi.
- **Bậc 2, phần phản đối xứng** là **diện tích Lévy**
  $$A_{ij}=\tfrac12\big(S^{(2)}_{ij}-S^{(2)}_{ji}\big)=\tfrac12\int_s^t\big[(x_i-x_i(s))\,dx_j-(x_j-x_j(s))\,dx_i\big].$$
  Về hình học, đây là diện tích có dấu giữa hình chiếu của path lên mặt phẳng $(i,j)$ và dây cung nối hai đầu.
  - **Dấu** của nó cho biết **chiều quay**: dương là ngược chiều kim đồng hồ trong hệ $(i,j)$.
  - Khi $i,j$ là hai kênh khác nhau, ví dụ toạ độ x của tay phải và x của tay trái, $A_{ij}>0$ nghĩa là kênh $i$ có xu hướng **đi trước** kênh $j$. Đây là thước đo lead/lag giữa hai kênh [Chevyrev & Kormilitzin arXiv:1603.03788; Lyons & McLeod].
  - **Với sign language** [S]: diện tích $(x,y)$ của cổ tay mã hoá chuyển động vòng tròn và chiều vòng. Diện tích giữa kênh của hai tay mã hoá việc tay nào dẫn trước, đúng loại thông tin phân biệt các sign hai tay đối xứng và bất đối xứng.
- Log-signature bậc 2 gồm $d$ increment cộng $\binom d2$ diện tích Lévy.

### 1.3 Công thức đóng cho path tuyến tính từng đoạn

Đặt $y_k = x_k - x_0$ và $\Delta_k = x_{k+1}-x_k$ với $k=0..T-2$. Khi đó
$$\log S^{\le2}=\Big(y_{T-1},\ \ A_{ij}=\tfrac12\sum_{k}\big(y_{k,i}\Delta_{k,j}-y_{k,j}\Delta_{k,i}\big)\Big)_{i<j}.$$

Đây chính là toạ độ Lyndon mà pySigLib trả về, với `method=1` lẫn `method=2`: 10 hệ số đầu trùng tới 4 chữ số thập phân với $d=6$, $T=40$ [V]. Cài đặt torch:

```python
def logsig2(X):                      # X: [B, T, d], tuyến tính từng đoạn
    Y = X - X[:, :1]; D = X[:, 1:] - X[:, :-1]
    M = 0.5 * (torch.einsum('bti,btj->bij', Y[:, :-1], D) - torch.einsum('bti,btj->bij', D, Y[:, :-1]))
    iu = torch.triu_indices(X.shape[-1], X.shape[-1], 1)
    return torch.cat([Y[:, -1], M[:, iu[0], iu[1]]], -1)   # [B, d + d(d-1)/2]
```

- Tốc độ: 9 600 path ($T=64$, $d=6$) mất 0.137 s trên CPU [V].
- Hàm này khả vi và chạy được trên GPU.
- Với cửa sổ chồng nhau thì dùng prefix-sum. Chen identity (mục 1.4) cho phép ghép các đoạn.

### 1.4 Chen identity: ghép path, path có khoảng trống, chỉ đoạn nhìn thấy

$$S(x*y)=S(x)\otimes S(y).$$
Cắt ngắn ở bậc 2: $(1,a_1,a_2)\otimes(1,b_1,b_2)=(1,\ a_1+b_1,\ a_2+b_2+a_1\otimes b_1)$.
Ở dạng log, theo BCH bậc 2: increment cộng lại, và diện tích là $A=A_x+A_y+\tfrac12(a_1\wedge b_1)$.

Ba hệ quả cho thiết kế:
- **Path có khoảng trống**, ví dụ tay mất detection ở giữa clip.
  - Nội suy tuyến tính qua khoảng trống tương đương chèn một **cầu thẳng**: $S=S_1\otimes S(\text{cầu})\otimes S_2$ [V, sai số 9e-15]. Cầu này là chuyển động bịa ra.
  - Phương án khác là **bỏ khoảng trống**, tức tịnh tiến đoạn 2 cho khớp vào điểm cuối đoạn 1: $S=S_1\otimes S_2$ [V]. Cách này giữ đúng hình học của các đoạn nhìn thấy, nhưng mất bước nhảy vị trí.
  - Trên ví dụ thử, hai cách cho kết quả lệch nhau tới 52 (trên thang toạ độ của ví dụ) [V]. **Lựa chọn này không vô hại.**
  - Cũng có thể thêm kênh hiển thị, theo phép biến đổi visibility của Wu, Ni, Lyons, Hudson (arXiv:2004.04006).
- **Streaming và cửa sổ**: signature của mọi cửa sổ $[t_a,t_b]$ tính được từ prefix, bằng $S_{0,a}^{-1}\otimes S_{0,b}$. pySigLib có sẵn `SigStream` / `LogSigWindowStream` cho truy vấn khoảng thời gian trong thời gian hằng [P, README pySigLib].
- **Rò rỉ khi che theo thời gian**: xem mục 5.7.

### 1.5 Các tính chất bất biến và các phép augmentation phá chúng

- **Bất biến với reparameterization**: $S(x\circ\varphi)=S(x)$ với mọi $\varphi$ đơn điệu.
  - Ta kiểm tra: upsample tuyến tính 4× cho log-signature y hệt, sai số 7e-15 [V].
  - Đây là cơ sở cho khẳng định "bền với frame-rate" của Path-JEPA [P, §4.5].
- **Thêm kênh thời gian** $(t,x_t)$ phá bất biến này. Ta lấy đúng polyline cũ, chèn thêm điểm vào nửa đầu để đổi tốc độ.
  - Không có kênh thời gian: log-signature lệch 7e-15, tức trùng khớp.
  - Có kênh thời gian: lệch 0.60, trên thang hệ số khoảng 11 [V].
  Đổi lại:
  - kênh thời gian mã hoá tốc độ;
  - nó loại bỏ hiện tượng tree-like, nên signature xác định path duy nhất (Hambly–Lyons).
  Morrill et al. thấy **time + basepoint** cho rank trung bình tốt nhất, 2.5 trên thang 1–6. Rank khi không augmentation là 4.69 [P, Table 7].
  **Path-JEPA không dùng cả hai** [P].
- **Bất biến với tịnh tiến**: signature chỉ phụ thuộc các increment. Có ba cách đưa lại vị trí tuyệt đối:
  - **basepoint**: chèn điểm 0 vào đầu path;
  - **invisibility-reset**: thêm một kênh bằng 1 rồi rơi về 0 ở cuối. Signature của path gốc vẫn nằm trong signature này, nhưng tốn thêm một kênh [P, Morrill et al. App. A];
  - **nối điểm bắt đầu** vào cạnh log-signature, như LogSig-RNN ("Add Starting Points") và GCN-DevLSTM ("append the starting point") [P].
  Ta kiểm tra: path đứng yên tại $(3,-2)$ có log-signature $(0,0,0)$; sau khi thêm basepoint thì thành $(3,-2,0)$ [V].
- **Lead-lag**: có hai định nghĩa khác nhau, và chúng **không** tương đương.
  - **Lead-lag rời rạc chuẩn** (Flint, Hambly, Lyons, SPA 2016, arXiv:1310.4054; Chevyrev & Kormilitzin).
    - Các điểm là $(x_1,x_1),(x_2,x_1),(x_2,x_2),\dots$, tạo thành đường bậc thang có $2T-1$ điểm, trong $\mathbb R^{2d}$.
    - Diện tích Lévy giữa kênh lead và lag đúng bằng **QV/2** $=\tfrac12\sum\Delta x^2$ [V: 21.9266 so với 21.9266].
  - **Path-JEPA** (Eq. 1) dùng phép nhúng trễ $[\tilde X(t),\tilde X(t-\delta)]$ trên nội suy liên tục, và **không nêu giá trị $\delta$** [P].
    - Với $\delta$ bằng 1 frame, diện tích thu được **không bằng** QV/2. Ví dụ: 2.57 so với 5.86 khi $N=20$; 68.5 so với 84.5 khi $N=200$, trên random walk.
    - Với tín hiệu trơn, nó xấp xỉ $\tfrac12\sum\Delta_k^2+\tfrac12\sum\Delta_k\Delta_{k+1}$, tức gần **QV** chứ không phải QV/2 [V].
  - **Hệ quả** [S]:
    - QV của một path trơn khi lấy mẫu dày thì tiến về 0, cỡ $O(1/N)$. Như vậy feature lead-lag **phụ thuộc vào lưới lấy mẫu**.
    - Phép trễ $\delta$ cố định theo thời gian thực cũng không giao hoán với reparameterization: $X(\varphi(t))-\delta$ khác $\varphi(t-\delta)$.
    - Vậy lý do "bất biến reparam" mà Path-JEPA nêu **không áp dụng nguyên vẹn** cho token lead-lag của chính nó.
  - Morrill et al. thấy lead-lag "usually represents a good choice". Nó tốt nhất cho HAR, rank 1.78, và MOTION, rank 1.75 [P, Table 1], nhưng không tốt hơn "không augmentation" một cách có ý nghĩa thống kê [P].

### 1.6 Số chiều log-signature (công thức Witt)

Mức $k$ có $\frac1k\sum_{m|k}\mu(m)d^{k/m}$ chiều [P, docstring pySigLib `log_sig_length`; Reutenauer]. Kết quả từ `pysiglib.log_sig_length` và `sig_length` [V]:

| $d$ | $n$ | signature, không hằng | signature, có hằng | log-signature |
|---|---|---|---|---|
| 2 (khớp 2D) | 2 | 6 | 7 | **3** |
| 3 (khớp 3D) | 2 | 12 | 13 | 6 |
| 4 (2D + lead-lag) | 2 | 20 | 21 | **10** |
| 6 (3D + lead-lag) | 2 | 42 | **43** | **21** |
| 7 (3D + lead-lag + time) | 2 | 56 | 57 | 28 |
| 6 | 3 | 258 | 259 | 91 |

- **Lệch số với Path-JEPA** [S]: paper ghi $D_{sig}=43$, tức có hằng, và $D_{log}=22$. Đúng ra là 21 Lie coordinates. Rất có thể 22 = 21 + số hạng hằng, nhưng log-signature không có số hạng hằng. Không có code để đối chiếu.
- Với pose **2D** (ASL Citizen), lead-lag và $n=2$ cho **10** hệ số mỗi path, và chỉ có **một** diện tích không gian thật sự, là diện tích $(x,y)$.

### 1.7 Tính duy nhất và tương đương tree-like

- Signature xác định path **đến mức tương đương tree-like**. Hambly & Lyons, *Annals of Mathematics* 171 (2010) 109–167, DOI `10.4007/annals.2010.171.109`, chứng minh cho path biến phân bị chặn. Boedihardjo, Geng, Lyons, Yang (arXiv:1406.7871) mở rộng cho rough path.
- Một đoạn "đi rồi quay lại đúng đường cũ" thì bị triệt tiêu.
- Ta kiểm tra trên động tác **gõ lặp lại** dọc một đường thẳng, 1, 2 hoặc 3 lần: log-signature là **0 chính xác**. Diện tích lead-lag thì tăng tuyến tính theo số lần lặp: 0.085, 0.167, 0.249 [V].
- **Với sign language** [S]: sự lặp lại có tính âm vị học (ví dụ các sign có chuyển động lặp lại kiểu gõ hoặc chạm), nhưng signature không có lead-lag hoặc kênh thời gian sẽ **không thấy** nó.

### 1.8 Độ nhạy với nhiễu và jitter

**Tài liệu**: chưa tìm thấy paper skeleton hoặc gesture nào đo độ bền của signature với nhiễu toạ độ cộng thêm. Các paper chỉ đo bỏ hoặc chèn frame.
- Phần lý thuyết gần nhất là bài về độ tập trung của signature bị cắt cho rough path Gaussian: mức $k$ có đuôi $\exp(-ct^{2/k})$, tức bậc càng cao thì đuôi càng nặng (Lechiheb, arXiv:2512.09588; chỉ đọc abstract).
- Chevyrev & Oberhauser, JMLR 23(176), 2022, arXiv:1810.10971, chỉ ra moment signature **không robust** với outlier nếu không chuẩn hoá. Họ đề xuất tensor normalization (mục 1.9).
- Yang et al. ghi rằng dyadic sâu hơn $h_{TJ}=3$ thì "harmful" vì nhiễu cục bộ [P].
- Li et al. ghi rằng trên ChaLearn 2016, keypoint tay của OpenPose kém chính xác nên hạn chế việc phân biệt các lớp tĩnh về hình dạng tay [P].

**Mô phỏng của ta** [V]. Nhiễu i.i.d. $\sigma$ trên mỗi toạ độ, 2D, 2 000 lần lặp:

| N frame | std(diện tích), chỉ có nhiễu | std/√N | QV trung bình (≈ 4(N−1)σ²) |
|---|---|---|---|
| 16 | 2.865 | 0.716 | 60.7 (60) |
| 64 | 5.647 | 0.706 | 251.9 (252) |
| 256 | 11.198 | 0.700 | 1019.8 (1020) |

Với một vòng tròn bán kính 1 cộng nhiễu $\sigma=0.05$, sai số diện tích là 0.058, 0.043, 0.034, 0.034 khi $N$ = 30, 60, 120, 240 [V]. Có hai chế độ:
- số hạng chéo giữa tín hiệu và nhiễu, cỡ $\sigma L/\sqrt N$, giảm khi $N$ tăng;
- số hạng nhiễu thuần, cỡ $\sigma^2\sqrt N$, tăng khi $N$ tăng.

→ Có một $N$ tối ưu, cỡ $L/\sigma$ [S]. Upsample, hoặc giữ nguyên 30–60 fps khi keypoint tay rung, sẽ làm diện tích và QV bị nhiễu chiếm.

Thêm một số liệu [V]: tín hiệu sin trơn, 1 chu kỳ, $N=120$, có QV/2 = 0.083. Jitter $\sigma=0.02$ (2 % biên độ) nâng con số này lên 0.127 (+54 %). Nghĩa là **feature lead-lag của tay chủ yếu đo jitter**.

### 1.9 Chuẩn hoá

- **Factorial**: hệ số mức $k$ có độ lớn $O(1/k!)$.
  - Post-scaling nghĩa là nhân với $k!$.
  - Pre-scaling nghĩa là nhân path với $\alpha=(N!)^{1/N}$.
  - Kết quả của Morrill et al.: pre-scaling **tệ hơn có ý nghĩa thống kê**; post-scaling và không scale không khác nhau có ý nghĩa. Rank: None 1.73, Post 1.92, Pre 2.35 [P, Table 11].
- **Tensor normalization** (Chevyrev & Oberhauser, Def. 12): $\Lambda(t)=\delta_{\lambda(t)}t$, với phép giãn $\delta_c(t)=(t_0,ct_1,c^2t_2,\dots)$. Nó chặn chuẩn bởi $R$ mà vẫn giữ tính đại số và khả năng tách điểm, từ đó cho ra thống kê có influence function hữu hạn [P].
- **Thang hình học** [S]: mức $k$ tỉ lệ với $s^k$ khi phóng toạ độ lên $s$ lần. Vì vậy phải chuẩn hoá theo **độ rộng vai** hoặc kích thước signer **trước** khi lấy signature, nếu không diện tích sẽ mã hoá khoảng cách tới camera. Sau đó nên chuẩn hoá z-score riêng cho từng loại token và từng mức. Path-JEPA không nói nó chuẩn hoá hệ số như thế nào [P].

---

## 2. Signature trong nhận dạng skeleton, gesture và sign

### 2.1 Bảng tổng hợp

| Công trình | Venue / ID | Path | Bậc, cửa sổ, augmentation | Dữ liệu, kết quả chính [P] |
|---|---|---|---|---|
| Graham, *Sparse arrays of signatures for online character recognition* | arXiv:1308.0371 | Nét bút | Bậc 1–3 làm input cho CNN | CASIA-OLHWDB1.1: lỗi 3.58 % so với 5.61 % của CNN thường |
| Li, Zhang, Jin, **LPSNet** | ICCVW 2017, DOI 10.1109/ICCVW.2017.80 | Quỹ đạo tay | Log path signature | SKIG: 96.7 % (RGB), 98.7 % (RGB+D) |
| Yang, Lyons, Ni, Schmid, Jin | arXiv:1707.03993; chương sách Springer 2022 | Spatial: **mọi** cặp và bộ ba khớp trong một frame (S-P/S-T-PSF). Temporal: quỹ đạo từng khớp có thêm thời gian (T-J-PSF). Lead-lag đa trễ trên từng chiều của feature không gian (T-S-PSF) | $n_{SP}=2$, $n_{ST}=3$, $n_{TJ}=5$, $n_{TS}=2$. Dyadic chồng nhau $h=3$. Invisibility-reset. Augmentation: lật, nhiễu Gaussian | JHMDB GT pose 84.23 %; sub-JHMDB AlphaPose 68.2 %, lên 75.7 % khi thêm confidence làm một chiều |
| Li, Zhang, Liao, Jin, Yang | AAAI 2019, arXiv:1811.07081 | **Chỉ phần tay**: khuỷu, cổ tay, bàn tay × 2 tay = 6 khớp. Cặp khớp trong cùng tay, giữa hai tay, và tay–thân. Quỹ đạo khớp + thời gian. Lead-lag 2 chiều trên các chiều của S_PS | $m_T=4$, $m_S=2$, $m_{TS}=3$. Dyadic $L_D$ = 3 và 2. Resample 39 frame | ChaLearn 2013 (20 cử chỉ Ý): 92.08 %. ChaLearn 2016 IsoGD (OpenPose): 39.95 % |
| Liao, Zhang, Li, Multi-path CNN | VCIP 2019, DOI 10.1109/VCIP47243.2019.8965816 | [chưa đọc toàn văn] | | ChaLearn 2013: 93.13 % (theo Table 3 của Shi et al. 2024) |
| Liao, Lyons, Yang, Schlegel, Ni, **LogSig-RNN** | BMVC 2021, arXiv:2110.13008 | Toàn bộ skeleton ($d=75$) đi qua Embedding Layer, giảm còn 30 chiều. Bản GCN thì tính per-joint trên feature GCN | Log-signature $M=2$, $N=4$ đoạn rời nhau. Thêm kênh thời gian. Lớp cộng dồn. Nối điểm bắt đầu | Chalearn 2013: 92.94 ± 0.21 (aug). NTU120 X-Sub/X-Set: 75.8 / 78.0 (bản 2 stream) |
| *Short-term path signature for skeleton-based action recognition* | SIViP 2022, DOI 10.1007/s11760-022-02404-y | [chưa đọc; chỉ có metadata OpenAlex] | | |
| Cheng, Shi, Li, Li, Ni, Jin, Zhang | **IEEE TMM 26 (2024) 3951–3961**, DOI 10.1109/TMM.2023.3318242 | Spatial: các khớp lân cận trong $k$-hop. Temporal: cửa sổ trượt trên một khớp. **Learnable path** (L-PSM): attention Gaussian-embedding chọn top-$(L-1)$ khớp cho mỗi khớp gốc, mỗi frame. Có map tuyến tính $\omega_1$ trước signature | Signature (Signatory), $k=2$. Không lead-lag, không kênh thời gian | ChaLearn 2013 94.18; IsoGD 51.60. **AUTSL** (xem 2.2) |
| Shi, Zhang, Cheng, Xiong, Ni, **AGGP** | ICPR 2024, LNCS, DOI 10.1007/978-3-031-78354-8_18; PDF green OA tại UCL Discovery | **Một** path toàn cục thích nghi: các token khớp × thời gian, chọn bằng Gumbel-Softmax từ hai nhánh (global self-attention và cluster attention theo vận tốc), sắp theo thứ tự thời gian | Signature (Signatory). Bậc không nêu trong text [P] | ChaLearn 2013 95.18 (1.1M tham số). **WLASL-300** (xem 2.2) |
| Jiang, Yang, Zhang, Ni, **GCN-DevLSTM** | arXiv:2403.15212, TMLR 2026 | Path development (nhóm Lie SE, ma trận $m=10$) trên feature GCN có thêm thời gian, per-node; cửa sổ trượt | Output có $m^2$ chiều, **không phụ thuộc $d$** | Chalearn 2013 95.61. NTU60 X-sub joint-only 90.8; 5 stream 92.9 |
| Lou, Li, Ni, Path Development Network | arXiv:2204.00740 (journal-ref: TMLR 08/2024) | Development khả vi trên nhóm Lie hữu hạn chiều | | Cơ sở lý thuyết của GCN-DevLSTM |
| Moreno-Pino et al., **Rough Transformer** | NeurIPS 2024, arXiv:2405.20799 | Time series tổng quát | **Multi-view**: $M(X)_k=(S_{0,t_k},\,S_{t_{k-1},t_k})$, tức prefix cộng cửa sổ. Mặc định là signature. Kênh thời gian là tuỳ chọn. Signature tính offline, không backprop qua nó | UEA trung bình 64.5; HR RMSE 2.66 |
| Walker et al., **Log-NCDE** | ICML 2024, arXiv:2402.18512 | Path có kênh thời gian | Log-signature bậc 1–2 trên các đoạn rời nhau | UEA trung bình 64.3 |
| Morrill et al., NRDE | ICML 2021, arXiv:2009.08295 | | Log-signature theo đoạn | Chuỗi dài |
| Oury et al., **LogSig-SSM** | NeurIPS 2026, arXiv:2610.05051 (chỉ đọc abstract) | | Log-signature cửa sổ đa thang làm token cho SSM | |
| Malzard et al., **SHARDeg** | arXiv:2505.18048 | Benchmark suy giảm dữ liệu trên NTU120 | | LogSigRNN thắng DeGCN ở 5/6 trường hợp frame-rate thấp, trung bình 6 %, dù thua 11–12 % ở 30 FPS (abstract, curl+grep) |
| Morrill, Fermanian, Kidger, Lyons, *Generalised signature method* | arXiv:2006.00873 | Phân loại augmentation, cửa sổ, transform, rescaling | Pipeline chuẩn đề xuất: $S^N\circ W^j\circ\varphi_b\circ\varphi_t$ với cửa sổ dyadic | Rank cửa sổ: Dyadic 1.73 < Expanding 2.17 < Global 2.83 < Sliding 3.04 |
| Signature kernel: Király & Oberhauser; Salvi et al. (Goursat PDE) | arXiv:1601.08169; arXiv:2006.14794 | | Kernel thay cho feature cắt bậc | Venue: JMLR 2019 và SIAM J. Math. Data Sci. 2021 [chưa xác minh] |

### 2.2 Các công trình dùng dữ liệu sign language (chỉ tìm thấy hai)

Phạm vi tìm kiếm:
- arXiv API, các truy vấn `abs:"signature" AND abs:"sign language"`, `abs:"log-signature"`, `abs:"path signature" AND abs:gesture|skeleton|self-supervised`;
- OpenAlex, 10 truy vấn;
- web search, cả chế độ extended, với các từ khoá WLASL, AUTSL, LSA64, IsoGD, CSLR, fingerspelling, Chinese sign language;
- GitHub search.

**Không** tìm thấy signature nào được áp dụng cho ASL Citizen, WLASL-2000, MS-ASL, LSA64, CSL, continuous SLR hay fingerspelling. Kết quả "Arabic SL … Radial Signature" (2013) là một descriptor hình dạng, không phải path signature.

1. **Cheng et al., TMM 2024, trên AUTSL** (226 sign, 43 signer; pose HRNet/MMPose 133 điểm theo SL-GCN). Chỉ stream joint. Top-1, Table VI [P]:

   | Backbone | Js | Js + module PS | 4 stream |
   |---|---|---|---|
   | ST-GCN | 94.25 | 94.93 (+0.68) | 95.06 |
   | AS-GCN | 94.84 | 95.04 | 95.29 |
   | Shift-GCN | 94.86 | 95.16 | 95.88 |
   | SL-GCN | 94.84 | 95.09 (+0.25) | 95.56 |

   → Gain nhỏ, và AUTSL đã gần bão hoà. Module PS ngang mức ensemble 2 stream nhưng thấp hơn 4 stream. Paper không nói dùng tập con nào trong 133 điểm.
   Thêm chi phí: ST-GCN từ 3.59 ms lên 26.02 ms khi gắn cả hai module [P].

2. **Shi et al., ICPR 2024 (AGGP), trên WLASL-300**. Skeleton 55 khớp từ [WLASL], trong đó 42 là khớp tay [P]. Pose có lẽ là 2D, suy từ câu "When utilizing the 2D skeleton as input" [S].
   - Đạt **60.85 %** với 1.25M tham số. So sánh: P3D(2D) 52.17, SPOTER 43.78, Pose-TGCN 38.32, I3D RGB 56.14 [P, Table 4].
   - **Mâu thuẫn nội bộ**: ablation Table 2 có một cấu hình đạt 61.93 trên WLASL-300, cao hơn con số 60.85 báo cáo ở Table 4 [P].
   - Nhận xét của tác giả: WLASL "more sensitive to dynamic information". Họ cũng cho rằng hand joint ít trong ChaLearn2013 có thể là lý do gain ở đó nhỏ hơn [P].
   - Ablation trên ChaLearn 2013: thay PS bằng FC hoặc bỏ PS thì còn 92.23–93.27, so với 95.18 [P, Table 1].

3. Li et al., AAAI 2019, chạy trên ChaLearn 2013, gồm 20 cử chỉ văn hoá Ý (gần với emblem gesture, **không** phải ngôn ngữ ký hiệu), và IsoGD.
   Kết quả IsoGD bằng skeleton OpenPose chỉ 39.95 % [P], cho thấy giới hạn của keypoint tay cũ.

### 2.3 Signature trong SSL, contrastive hoặc masked modeling

- **Path-JEPA** là công trình duy nhất tìm thấy dùng signature làm token hoặc target trong khung masked SSL.
  Lưu ý (theo P0): target của nó là **embedding của teacher EMA trên token signature**, không phải regress hệ số signature [P].
- Không tìm thấy contrastive hay MAE nào trên time series dùng signature (truy vấn arXiv `abs:"path signature" AND abs:"self-supervised"` chỉ trả về một bài imitation learning, arXiv:2407.04856).
- Liên quan xa:
  - "Early autism diagnosis based on path signature and Siamese unsupervised feature compressor" (Cerebral Cortex 2024) [chưa đọc];
  - SigFormer (deep hedging, arXiv:2310.13369);
  - SigDiffusions (arXiv:2406.10354, log-signature làm embedding cho diffusion).
- **Bằng chứng nội bộ** (theo P0): khi dùng log-signature làm **target tường minh**, các thử nghiệm trước đều không giúp, gồm proposal12 S2 λ_signature và SignPathLab E03/E04. Biến thể "signature làm input token + target latent" **chưa được chạy**.

### 2.4 Rút ra từ ablation: cái gì thực sự giúp

| Yếu tố | Bằng chứng [P] | Ý nghĩa cho proposal16 [S] |
|---|---|---|
| Path cặp khớp theo không gian | Li et al. Table 2, bậc 2: RC 77.84 → +S_PS 87.62, so với +T_PS (quỹ đạo khớp) 81.28. Yang: spatial PSF tăng khoảng 20 % so với baseline | Cặp tay–tay, tay–mặt và ngón–cổ tay là ứng viên tự nhiên cho location và contact |
| Bậc | Li: S_PS bão hoà ở bậc 2–3 (88.12 ở bậc 3, 88.06 ở bậc 4). Yang: $n_{TS}$ > 2 "negligible". Cheng chọn $k=2$ | Giữ $n=2$; chỉ thử $n=3$ cho path ngắn chiều |
| Cửa sổ | LogSig-RNN: $N=4$ đoạn là tối ưu (92.94), $N=64$ còn 91.50. Morrill: dyadic tốt nhất. Yang: dyadic sâu hơn 3 thì hại | Clip sign ngắn nên dùng 2–3 thang dyadic là đủ |
| Global + local | Path-JEPA: chỉ local 83.3, chỉ global 84.4, cả hai 85.8 (joint-only). RFormer: global hay local là tuỳ dataset; trong 6 bộ UEA chỉ HB dùng cả hai | Giữ global, nhưng xem mục 5.7 về rò rỉ |
| Joint, edge, chain | Path-JEPA: 85.8 → 86.4 (+edge) → 87.6 (+chain) | Cho tay: thay chain chi bằng chain ngón tay |
| Chiếu tuyến tính học được trước signature | LogSig-RNN: EL +1.87 pp và giảm hơn 60 % trọng số. Cheng: $\omega_1,\omega_2$ giảm từ 6.00M xuống 1.27M tham số (bỏ chúng thì 94.09 so với 94.18) | Nhưng signature học được thì không còn là target cố định cho JEPA |
| Time + basepoint | Morrill: rank 2.5, tốt nhất | Path-JEPA bỏ cả hai. Cần ablation riêng cho sign, vì location là thông tin tĩnh |
| Bền với bỏ hoặc chèn frame | GCN-DevLSTM: bỏ 30/64 frame thì −6.07, so với −18.18 của CTR-GCN. SHARDeg: LogSigRNN thắng ở frame-rate thấp | Có ích cho ASL Citizen vốn trộn nhiều fps (11–120) |
| Bền với nhiễu toạ độ | **Không paper nào đo** | Phải tự đo trước khi tin vào token bậc 2 của tay |

---

## 3. Phần mềm (kiểm tra PyPI/GitHub ngày 2026-10-16)

Môi trường của ta: `.venv-signrep` có Python 3.13.5 và torch 2.11.0+cu128.

| Thư viện | Bản mới nhất / ngày | GPU | Autograd | Tương thích Py3.13 + torch 2.11 | Ghi chú |
|---|---|---|---|---|---|
| **signatory** (Kidger & Lyons, ICLR 2021, arXiv:2001.00706) | 1.2.6.1.9.0 (2021-07-19); README nói 1.2.7 | Có | Có | **Không**. README: Python 3.7–3.9, torch 1.8–1.11. Repo push cuối 2024-01 | Nhiều paper dùng (Cheng, AGGP, LogSig-RNN bản PyTorch). Không dùng được ở ta |
| **iisignature** (Reizenstein & Graham, arXiv:1802.08252; ACM TOMS 46(1) 2020) | 0.24 (2019-12-01); repo push 2026-06 | Không | Có backprop thủ công (`sigbackprop`, `logsigbackprop`) | Chỉ có wheel Windows ≤ cp39 và sdist. Build trên Linux/3.13 [chưa xác minh] | Yang, Li và LogSig-RNN bản TF dùng thư viện này |
| **pySigLib** (Shmelev & Salvi, arXiv:2509.10613) | **4.0.0 (2026-09-27)**, manylinux wheel, Python ≥ 3.9; repo push 2026-10-15 | CUDA, qua `pysiglib[cuda]` | PyTorch và JAX | **Có**. Đã import và chạy trên CPU ở ta (cài `--target` scratchpad, phụ thuộc `kauri`) | Có signature, log-signature (Lyndon), signature kernel, stream/window query, `time_aug`, `lead_lag`. `log_sig` với method 1/2 cần gọi `prepare_log_sig` trước |
| **RoughPy** (DataSig) | 0.3.0 (2026-03-23), Python ≥ 3.11 | [chưa xác minh] | [chưa xác minh] | Python thì được | Hướng tới stream algebra, không phải lớp torch |
| **esig** | 1.0.0 (2024-12-17), py3-none-any | — | — | Phụ thuộc `roughpy` và `pyrecombine` | Thư viện kế thừa (Yang et al. dùng bản cũ) |
| **keras_sig** | 1.0.2 (2025-01-02) | Qua backend Keras 3 | Có | **Không**: yêu cầu Python < 3.13 | |
| **signax** (JAX) | 0.2.1 (2023-09-05); repo push 2026-05 | Có (JAX) | Có | Không phải torch | |
| **sigkernel** (Salvi) | Chỉ có trên GitHub, push 2025-11 | Có | Có | [chưa thử] | Chỉ cho signature kernel |
| `torchsig` trên PyPI | 2.2.0 | — | — | — | **Không liên quan.** Đây là toolkit xử lý tín hiệu RF; đừng nhầm |

**Khuyến nghị** [S]:
- Với $n=2$, dùng **công thức đóng** ở mục 1.3, thuần torch. Nó khả vi, chạy được trên GPU và không phụ thuộc thư viện ngoài.
- Dùng pySigLib (CPU) làm oracle cho unit test, và nếu cần $n\ge3$ hoặc kernel.
- Tránh signatory và keras_sig vì không tương thích môi trường.

---

## 4. Rủi ro khi áp dụng cho sign language

### 4.1 Mất thông tin tĩnh

Âm vị học của ngôn ngữ ký hiệu có **handshape**, **location**, **orientation** và **movement**. Ba thành phần đầu phần lớn là tĩnh trong một sign.
- Một path thời gian đứng yên có signature bằng 0 [V].
- Signature bất biến khi tịnh tiến, nên mất location.
- Như vậy token signature **theo thời gian** chủ yếu chỉ mang movement [S].

Cách giảm thiểu [S]:
- **(a) Path không gian trong từng frame**, kiểu S-P/S-T-PSF của Yang et al. Ví dụ một path đi qua các khớp của một ngón: cổ tay → MCP → PIP → DIP → TIP.
  - Increment là vector từ cổ tay tới đầu ngón.
  - Diện tích $(x,y)$ là diện tích có dấu của đa giác ngón, tức đo **độ gập ngón** và chiều gập.
  - Đây là signature của **hình dạng**, không phải của chuyển động. Yang et al. và Li et al. đều thấy path không gian cho gain lớn nhất.
- **(b) Path cặp khớp tương đối với mặt hoặc thân**, ví dụ cổ tay → mũi hoặc cằm. Cộng thêm basepoint hoặc nối điểm bắt đầu để giữ location.
- **(c) Giữ một luồng token toạ độ thô** song song. Lưu ý Path-JEPA bỏ hẳn toạ độ thô.

### 4.2 Chuyển động lặp lại và tree-like

Động tác gõ hoặc đi-về dọc cùng một đường cho log-signature bằng 0 [V].
- Cần lead-lag, có QV tăng theo số lần lặp, hoặc kênh thời gian.
- Lead-lag lại nhạy với jitter (mục 1.8), và lead-lag kiểu delay của Path-JEPA phụ thuộc $\delta$ cũng như lưới lấy mẫu (mục 1.5).
- **Đề xuất ablation**: so sánh {không aug, lead-lag chuẩn, delay-$\delta$, time-aug} trên tập sign có nhãn "lặp lại" của ASL-LEX.

### 4.3 Nhiễu pose 2D của tay

Số liệu ASL Citizen theo D1, pose RTMW COCO-WholeBody 133 điểm 2D:
- score tay p50 là 3.23 trên thang SimCC khoảng 0–10;
- tỉ lệ frame có mean score < 3 là 48.2 % với tay trái và 42.3 % với tay phải;
- **alias trái/phải** ở 17.3 % số frame và 78.5 % số clip;
- tay ra ngoài khung ở 12.5 % (trái) và 8.6 % (phải) số frame.

Hệ quả [S, dựa trên V]:
- Diện tích và QV của tay sẽ chủ yếu là nhiễu nếu không làm mượt. Nhiễu tăng theo $\sqrt N$ (diện tích) và theo $N$ (QV).
- Alias, tức hai tay đổi nhãn, tạo bước nhảy lớn. Bước nhảy này đi thẳng vào increment và diện tích, và nhờ Chen identity nó **lan sang mọi cửa sổ chứa nó, kể cả global**.

Giảm thiểu:
- sửa alias **trước** khi lấy signature;
- làm mượt có kiểm soát (Savitzky–Golay hoặc One-Euro) hoặc hạ xuống khoảng 15 fps cho path tay;
- bỏ hoặc giảm trọng số token có coverage confidence thấp;
- tính và báo cáo tỉ số SNR của diện tích (tín hiệu so với nhiễu) trên tập val **trước** khi pretrain. Có thể dùng `docs/proposal16/diagnostics/sig_diagnostics.py` nếu nó đo đại lượng này.

### 4.4 Tay bị mất hoặc bị alias, và nội suy

Có ba cách xử lý khoảng trống, mỗi cách cho signature khác nhau (mục 1.4):
- cầu tuyến tính: bịa ra chuyển động;
- bỏ khoảng trống: mất bước nhảy;
- kênh visibility: thêm chiều.

Ngoài ra, Path-JEPA tính token trên chuỗi đầy đủ rồi mới che [P]. Với sign, "đầy đủ" vốn đã gồm frame nội suy, nên **target có thể chứa chuyển động bịa ra** [S].
Đề xuất: mask token khi coverage của cửa sổ dưới ngưỡng, và không dùng token đó làm target.

### 4.5 Ảnh hưởng của 2D so với 3D

- Mỗi khớp 2D chỉ có **một** diện tích không gian. Hai chiều thêm từ lead-lag là $(x,x_{lag})$ và $(y,y_{lag})$, cộng 4 diện tích chéo.
- Chuyển động theo trục sâu (về phía hoặc ra xa camera) gần như biến mất.
- Chiều quay đo trong mặt phẳng ảnh, nên phụ thuộc góc camera.
- Hệ số mức $k$ tỉ lệ $s^k$, nên **bắt buộc** chuẩn hoá theo độ rộng vai.
- Với ASL Citizen, webcam chủ yếu chính diện (96.1 % clip ở 640×480 theo D1), nên ảnh hưởng của góc nhìn có lẽ nhỏ hơn NTU, nhưng không có kiểm chứng [S].

### 4.6 Clip ngắn và fps hỗn hợp

ASL Citizen theo D1:
- độ dài p5/p50/p95 là 1.47 / 2.57 / 5.27 s;
- fps 30 ở 84.8 % clip, nhưng có clip từ 11.4 đến 120 fps.

Hệ quả [S]:
- Ở 30 fps, $T_{p50}\approx77$ frame. Cửa sổ 25 frame stride 12 của Path-JEPA cho khoảng 5 cửa sổ, và cửa sổ 50 frame cho khoảng 1–2.
- Clip bắt đầu và kết thúc ở tư thế nghỉ, nên increment global của cổ tay xấp xỉ 0. Thông tin dồn vào diện tích, đúng phần nhạy nhiễu nhất. Chuyển động nâng và hạ tay chiếm một phần lớn của global view.
- Cửa sổ phải được định nghĩa theo **giây** chứ không theo số frame, vì fps thay đổi. Bất biến reparam chỉ giúp khi **không** có kênh thời gian **và** không có lead-lag trễ theo frame.

### 4.7 Rò rỉ khi kết hợp global view với temporal masking

Mỗi token của Path-JEPA chứa $\log S_{0,T}$ của path, **giống hệt nhau ở mọi cửa sổ** (Eq. 3) [P].
- Path-JEPA chỉ che theo **khớp**, ở mọi cửa sổ, nên không rò theo thời gian [P].
- Nếu ta thêm **temporal masking** (che cửa sổ $[a,b]$ của một path đang hiển thị ở cửa sổ khác) thì có thể khôi phục chính xác log-signature của cửa sổ bị che từ global và các local còn hiển thị:
  - $\Delta_{ab}=\Delta_{0T}-\Delta_{0a}-\Delta_{bT}$;
  - $A_{ab}=A_{0T}-A_{0a}-A_{bT}-\tfrac12(\Delta_{0a}\wedge\Delta_{ab}+\Delta_{0a}\wedge\Delta_{bT}+\Delta_{ab}\wedge\Delta_{bT})$.
  - Ta kiểm tra: sai số 9e-16 và 6e-15 [V].
- Cửa sổ chồng nhau (stride 12 < size 25) và thang $\ell=1$ chứa các cửa sổ $\ell=0$ làm rò thêm.
- Path cặp khớp giữa hai tay hoặc tay–mặt cũng mang thông tin của khớp bị che nếu không áp quy tắc Eq. 7 cho mọi path chạm vào khớp đó.

Quy tắc an toàn [S]: khi che theo thời gian thì phải bỏ global view khỏi token context, hoặc tính global chỉ trên phần hiển thị (Chen trên các đoạn còn thấy). Che mọi token có khoảng thời gian giao với vùng bị che, ở mọi thang.

### 4.8 Những điểm Path-JEPA còn mơ hồ (không có code để đối chiếu)

1. **Code**: repo `github.com/singh17ashu/Path-JEPA` (MIT, tạo 2026-07-01, push cuối 2026-10-05) chỉ có README "Code coming soon", `index.html` và LICENSE.
2. **$\delta$ của lead-lag** không được nêu. Cũng không rõ phần $t<\delta$ được pad thế nào.
3. **$D_{log}=22$ so với 21** (mục 1.6).
4. **Chain path**: "concatenating the constituent joint-pair displacement trajectories". Nếu là ghép kênh thì $d'$ tăng theo số cạnh và $D_{log}$ không thể cố định bằng 22. Vậy hoặc là ghép theo thời gian (Chen), hoặc là tổng [S].
5. **Số token**: $N_{total}=152$ với $L_{total}=4$ cửa sổ, tức 38 token mỗi cửa sổ. Con số này không khớp với $J+|E|+|C|$ của NTU, vì $25+24+\text{chain}>38$ [S]. Bốn cửa sổ (25/12 cộng một cửa sổ 50) khớp với $T\approx50$, nhưng paper không nêu $T$ [S].
6. **Giao thức frame-rate** (Table 4): resample rồi "linear interpolation back to the original sequence length".
   - Ta mô phỏng: 60 fps theo đúng mô tả này cho lại **đúng chuỗi gốc** (sai số 0.0000) [V]. Vậy việc baseline tụt ở 60 fps (MAMP 84.9 → 81.2) khó giải thích nếu giao thức đúng như mô tả.
   - Bảng 4 ghi 87.7 ở 30 fps, còn Bảng 1 ghi 87.6 [P].
7. **Không nêu**: chuẩn hoá hệ số signature, thư viện tính signature, cách xử lý hai người trong NTU, và chuẩn hoá toạ độ.
8. **Lead-lag phá bất biến reparam** mà paper dùng để giải thích độ bền frame-rate (mục 1.5) [S].

---

## 5. Hàm ý thiết kế cho proposal16 (đề xuất, chưa chạy)

1. **Tokenizer**: log-signature bậc 2 tính bằng công thức đóng, sau khi đã chuẩn hoá theo độ rộng vai và sửa alias tay. Viết unit test so với pySigLib.
2. **Họ path cho sign**:
   - quỹ đạo cổ tay, temporal, có lead-lag;
   - cặp tay–tay và tay–mặt, cho location và contact;
   - **path không gian trong frame qua từng ngón**, cho handshape;
   - chain cánh tay;
   - luồng toạ độ thô hoặc basepoint để giữ thông tin tĩnh.
3. **Thời gian**: cửa sổ theo giây, 2–3 thang dyadic. Hạ khoảng 15 fps cho path tay, hoặc làm mượt. Đo SNR diện tích trước khi chọn.
4. **Masking**: dependency masking của Eq. 7 mở rộng thêm cho các cặp tay–mặt. Nếu thêm temporal hoặc tube masking thì bỏ global $[0,T]$ khỏi context, hoặc tính lại global chỉ trên phần hiển thị.
5. **Ablation bắt buộc**:
   - signature so với log-signature;
   - {không aug, lead-lag chuẩn, delay-δ, time, basepoint};
   - có và không có path không gian theo ngón;
   - độ bền khi thêm jitter tổng hợp lên tay ở các mức σ khác nhau. Đây là khoảng trống mà chưa paper nào lấp.

---

## 6. Tài liệu tham khảo (ID đã kiểm tra qua arXiv API, Crossref hoặc OpenAlex)

- Singh & Singh, Path-JEPA, ECCV 2026, LNCS 17054, tr. 110–128, DOI 10.1007/978-3-032-37258-1_7 (Crossref); repo `singh17ashu/Path-JEPA`.
- Chen, K.-T., Trans. AMS 89(2) (1958) 395–407 (theo trích dẫn trong Shi et al. 2024).
- Hambly & Lyons, Annals of Math. 171 (2010) 109–167, DOI 10.4007/annals.2010.171.109.
- Boedihardjo, Geng, Lyons, Yang, arXiv:1406.7871.
- Flint, Hambly, Lyons, SPA 126(9) (2016) 2593–2614, arXiv:1310.4054.
- Chevyrev & Kormilitzin, arXiv:1603.03788.
- Lyons & McLeod, arXiv:2206.14674 (EMS Surveys).
- Cass & Salvi, arXiv:2404.06583.
- Lyons, ICM 2014, arXiv:1405.4537.
- Chevyrev & Oberhauser, JMLR 23(176) 2022, arXiv:1810.10971.
- Morrill, Fermanian, Kidger, Lyons, arXiv:2006.00873.
- Fermanian, arXiv:1911.13211 (CSDA 2021).
- Wu, Ni, Lyons, Hudson, arXiv:2004.04006.
- Graham, arXiv:1308.0371.
- Li, Zhang, Jin, LPSNet, ICCVW 2017, DOI 10.1109/ICCVW.2017.80.
- Yang, Lyons, Ni, Schmid, Jin, arXiv:1707.03993.
- Li, Zhang, Liao, Jin, Yang, AAAI 2019, arXiv:1811.07081, DOI 10.1609/aaai.v33i01.33018585.
- Liao, Zhang, Li, VCIP 2019, DOI 10.1109/VCIP47243.2019.8965816.
- Liao, Lyons, Yang, Schlegel, Ni, BMVC 2021, arXiv:2110.13008; code `steveliao93/GCN_LogsigRNN`.
- Cheng et al., IEEE TMM 26 (2024) 3951–3961, DOI 10.1109/TMM.2023.3318242.
- Shi, Zhang, Cheng, Xiong, Ni, ICPR 2024, DOI 10.1007/978-3-031-78354-8_18.
- Jiang, Yang, Zhang, Ni, arXiv:2403.15212 (TMLR 2026); code `DeepIntoStreams/GCN-DevLSTM`.
- Lou, Li, Ni, arXiv:2204.00740.
- Moreno-Pino et al., NeurIPS 2024, arXiv:2405.20799; code `AlvaroArroyo/RFormer`.
- Walker et al., ICML 2024, arXiv:2402.18512; code `Benjamin-Walker/Log-Neural-CDEs`.
- Morrill et al., NRDE, ICML 2021, arXiv:2009.08295.
- Oury et al., LogSig-SSM, arXiv:2610.05051.
- Malzard et al., SHARDeg, arXiv:2505.18048.
- Lechiheb, arXiv:2512.09588.
- Kidger et al., Deep Signature Transforms, NeurIPS 2019, arXiv:1905.08494.
- Kidger & Lyons, Signatory, ICLR 2021, arXiv:2001.00706.
- Reizenstein & Graham, iisignature, arXiv:1802.08252.
- Shmelev & Salvi, pySigLib, arXiv:2509.10613.
- Király & Oberhauser, arXiv:1601.08169.
- Salvi et al., arXiv:2006.14794.
