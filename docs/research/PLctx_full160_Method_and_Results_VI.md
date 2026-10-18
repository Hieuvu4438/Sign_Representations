# PLctx full160: method chi tiết và kết quả cuối

Ngày tổng hợp: 2026-10-15. File này chỉ tổng hợp lại các kết quả đã có. Không chạy training hay evaluation mới.

Nguồn chính (đường dẫn tính từ `method/phonojepa/`):

| Nội dung | File |
|---|---|
| Manifest và log train | `runtime/runs/PJ_PLctx_FULL_160ep/{manifest.json,train_log.jsonl}` |
| Đánh giá pilot-compatible (LOSO, ASLC300) | `runtime/eval/PJ_PLctx_FULL_160ep.json` |
| Đánh giá frozen dictionary (test chính thức) | `runtime/eval/PJ_PLctx_FULL_160ep_dictionary/{aslc,wlasl}/results.json`, `comparison.csv` |
| Audit toàn bộ run | `runtime/eval/PJ_PLctx_FULL_160ep_audit/{REPORT.md,audit.json}` |
| Mã nguồn | `pj/{masks,features,model,losses,train,extract}.py`, `IMPLEMENTATION.md` |
| Đặc tả method | `docs/proposal12/PhonoJEPA_Method_Proposal_VI.md` |

Quy ước: Rec = Recall@K tính bằng %. DCG ghi theo thang 0–100. "white512" là PCA-whitening 512 chiều, không dùng nhãn. "raw" là không whitening.

---

## 1. Tóm tắt

| Mục | Giá trị |
|---|---|
| Tên | PLctx, bản full-data 160 epoch (`PJ_PLctx_FULL_160ep`) |
| Họ method | PhonoJEPA: skeleton SSL, che kênh âm vị học, dự đoán latent theo ngữ cảnh |
| Dữ liệu train | Toàn bộ train ASL Citizen: 40.154 video, 2.731 gloss, 35 signer. Không dùng nhãn |
| Ngân sách | 200.640 bước, batch 32, 160 epoch, 1 seed, train từ đầu |
| Tham số | Encoder 5,19M; predictor 0,83M; heads 0,07M |
| Test ASL Citizen (raw, dictionary) | **Rec@1 16,68; Rec@5 36,90; DCG 39,62** (32.941 query) |
| Test WLASL2000 | Rec@1 9,62; Rec@5 24,42; DCG 30,55 |
| LOSO-val white512 | Rec@1 29,54 |
| ASLC300 white512 | Rec@1 47,67; linear probe macro 68,87 |
| Khoảng cách tới SignRep public (ASLC test) | 16,68 so với 49,02 (weighted) |

---

## 2. Method

### 2.1 Ý tưởng

PLctx là biến thể đơn giản nhất của PhonoJEPA. Nó che các **kênh âm vị học** của skeleton (hình dạng tay, vị trí/chuyển động tay, các vùng mặt) và dự đoán **biểu diễn latent theo ngữ cảnh** của phần bị che. Target do một teacher EMA tạo ra. Không có target tường minh nào khác.

Hai đặc điểm phân biệt PLctx với các biến thể khác:
- **PL**: kiểu mask là PLCM-lite (view gần che khối thời gian, view xa che một kênh tay đơn lẻ).
- **ctx**: chỉ dùng target latent theo ngữ cảnh. Không dùng intrinsic descriptor (khác PLint và PLboth).

### 2.2 Dữ liệu và tiền xử lý

| Thành phần | Cấu hình |
|---|---|
| Corpus | 40.154 clip train chuẩn của ASL Citizen; 3.462.848 frame (khoảng 32,17 giờ); độ dài trung vị 2,53 s (25–75%: 2,00–3,24 s) |
| Chia tập | Train/val/test không trùng video ID hoặc participant ID |
| Skeleton | COCO-WholeBody, chọn 103 khớp |
| Window | 64 mẫu tại min(fps, 25) |
| Kênh | 7 kênh × 32 bin thời gian = 224 token |
| Kênh tay (mỗi tay) | D/N-CONF: hình dạng + hướng tay trong khung thân. D/N-LOC: cổ tay + khuỷu |
| Kênh mặt | UPPER, MOUTH, HEAD |
| Khung toạ độ | Khung thân, hiệu chỉnh tỉ lệ pixel (ASL Citizen chủ yếu 640×480) |
| Handedness | Canonical: tay trội quyết định theo nội dung toàn clip (khai báo là rò rỉ 1 bit) |
| Augmentation | Đổi tốc độ, camera affine; hai view near/far lấy window độc lập |

### 2.3 Masking (PLCM-lite)

Mỗi clip, mỗi bước có hai view bị che, so với hai view teacher.

| View | Họ mask | Cách che | Tỉ lệ che (trên lưới 7×32) |
|---|---|---|---|
| Near | R2 | Khối thời gian 4–8 bin, tất cả kênh | khoảng 50% |
| Far | R3 | Một kênh **tay** trong một đoạn dài (≥ 4 bin); không bao giờ che cả hai kênh của cùng một tay trong một bin | khoảng 11% |

Chi tiết:
- Kênh tay được chọn với xác suất D-CONF 0,35, D-LOC 0,25, N-CONF 0,25, N-LOC 0,15.
- Kênh mặt không bị che riêng lẻ, vì chúng liên kết tới 0,9 theo cột (chẩn đoán D8). Closure F_HD ⇒ F_UP, F_MO luôn được áp dụng.
- Việc chọn mask độc lập với nội dung: sampler chỉ biết số bin hợp lệ và RNG.
- Ngân sách far là U(0,3; 0,6) × 2 × số bin hợp lệ "hand slot", giống nhau cho mọi họ mask (PL, R4, R0).
- Phần bị che bị xoá khỏi student; nguồn bị che được ghi đè và propagate qua đổi tốc độ. Có test rò rỉ: ghi đè mọi giá trị bị che bằng 1e6 hoặc nhiễu thì đầu vào student vẫn giống bit-for-bit.

### 2.4 Kiến trúc

| Thành phần | Cấu hình |
|---|---|
| Tokenizer | Theo kênh, 7 × 32 token, có embedding kênh và vị trí |
| Student encoder | d = 256, 6 block Transformer, 4 head, RoPE, CLS token. Chạy cả 224 slot dưới key mask (không gather token nhìn thấy) |
| Teacher | EMA của student, momentum 0,996 → 0,9999 (cosine) |
| Predictor | d = 128, 4 block, mỗi target cell có một query |
| Heads | Context head (33.024 tham số) và các head phụ, tổng 66.693 |
| Số tham số | Student 5.186.816; teacher 5.186.816; predictor 827.264 |

### 2.5 Target và loss

| Mục | Cấu hình |
|---|---|
| Target | Token ngữ cảnh của teacher: trung bình LayerNorm của 4 layer cuối |
| Loss chính | Centered cross-entropy trên feature dimension (kiểu S-JEPA), tổng hai view near + far |
| Centering | Chỉ cập nhật sau một bước optimizer hữu hạn |
| Không dùng | Intrinsic descriptor, CLS cross-view alignment (`--glob`) và KoLeo đi kèm nó, SIGReg, prototype loss, curriculum (các cờ này đều tắt trong manifest; các hệ số `lam_*` trong manifest chỉ là giá trị mặc định, không có hiệu lực) |
| Ghi chú | CLS chỉ nhận gradient gián tiếp, không có loss ép CLS của hai view khớp nhau |

### 2.6 Optimizer và lịch học

| Mục | Giá trị |
|---|---|
| Optimizer | AdamW, lr đỉnh 3e-4, weight decay 0,05 (không áp dụng cho bias/embedding) |
| Warmup | 20.064 bước = 16 epoch |
| Lịch | Cosine tới 3e-6 |
| Clip gradient | Norm 1,0 |
| Precision | bf16 |
| Batch / bước | 32 clip |
| Tổng bước | 200.640 (6.420.480 lượt lấy clip; trung bình 159,90 lượt/clip do drop-last) |
| Khởi tạo | Scratch, seed 0, không warm-start từ pilot 16k |
| Thời gian | 4 giờ 3 phút (14.580 giây), 13,76 bước/s |

### 2.7 Readout dùng khi đánh giá

| Readout | Mô tả | Vai trò |
|---|---|---|
| `grp` | Mean-pool theo nhóm token (tay trội, tay còn lại, mặt), nối lại: 768 chiều. Lấy từ teacher encoder | **Chính**, cố định trước khi chấm test |
| `mean` | Mean-pool toàn bộ token | Phụ |
| `cls` | CLS token | Phụ |

Embedding clip = trung bình các window (64 frame, stride 2 ở bước test) rồi chuẩn hoá L2.

---

## 3. Kiểm tra tính hợp lệ của lần train

| Kiểm tra | Kết quả |
|---|---|
| Hoàn tất đủ 200.640 bước | Có (`ckpt.pt` và `final.pt` cùng bước 200.640, tensor giống hệt) |
| Số dòng log | 4.013 dòng (bước 1 và mỗi 50 bước); 803 mốc diagnostics (mỗi 250 bước) |
| NaN/Inf, bước bị bỏ | 0 |
| Traceback / collapse alarm | 0 / 0 |
| Tensor model hữu hạn | Student, teacher, predictor, heads đều hữu hạn |
| Hash file nguồn | 14/14 khớp với launcher |
| Rò rỉ train/val/test | Không trùng video ID hay participant ID giữa các tập |
| Hash checkpoint | `8f8a10687f2df869fd0d7f4703edfeb467c59977d7baaacb2d567c15789e41f9`, khớp protocol của đánh giá dictionary |

Hạn chế đã biết:
- Không có đánh giá validation trong lúc train và checkpoint trung gian bị ghi đè. Chưa biết epoch nào tốt nhất.
- Canonical handedness là rò rỉ 1 bit về nội dung (khai báo trong `IMPLEMENTATION.md`, mục 3). Chưa có ablation random-handedness ở full-data.
- 100% mốc log có gradient norm > 1 trước clipping (trung vị 9,58), nên clipping luôn hoạt động.

---

## 4. Diễn biến train

Trung vị các batch được log trong từng giai đoạn. `loss` là tổng near + far. Cột CTX là loss của view far (code ghi đè giá trị này).

| Epoch | Loss tổng | CTX far | Grad norm trước clip | Token effective rank (batch cố định) |
|---|---:|---:|---:|---:|
| 1–16 (warmup) | 4,8411 | 2,6029 | 13,17 | 132,86 |
| 17–40 | 4,2359 | 2,1557 | 7,11 | 164,28 |
| 41–80 | 3,6487 | 1,8017 | 7,67 | 178,40 |
| 81–120 | 3,6388 | 1,7970 | 9,78 | 189,70 |
| 121–144 | 3,6330 | 1,7634 | 11,07 | 195,46 |
| 145–160 | 3,6508 | 1,7676 | 11,32 | 196,52 |

| Chỉ số | Đầu → cuối |
|---|---|
| Token effective rank | 94,58 → 196,64 |
| Token std | 0,804 → 1,004 |
| CLS mean cosine (batch cố định) | 0,810 → 0,289 |
| Teacher entropy far (10% cuối) | 0,1785 nat (pilot 16k: 0,2621) |
| Loss trung bình epoch 41–80 so với 145–160 | 3,6561 so với 3,6379 (chênh khoảng 0,5%) |

Nhận xét: loss gần phẳng từ khoảng epoch 60–80. Loss không dự đoán được retrieval, nên không kết luận được nửa sau của quá trình train có ích hay không.

---

## 5. Kết quả cuối

### 5.1 Test chính thức (frozen dictionary, raw, không whitening)

Protocol: raw window mean → L2 → cosine → điểm lớp là max cosine trong gallery. Rank đếm theo lớp, ties tính bi quan. Không classifier, không fine-tune. Skeleton 64 frame, stride 2, không có activity head.

| Dataset | Gallery (train) | Query (test) | Số lớp | DCG | Rec@1 | Rec@5 | Rank trung vị |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ASL Citizen** | 40.154 | 32.941 | 2.731 | 39,62 | **16,68** | 36,90 | 13 |
| WLASL2000 | 14.296 | 2.879 | 2.000 | 30,55 | 9,62 | 24,42 | 39 |

Số đếm: ASL Citizen đúng 5.494 query ở top-1 và 12.154 ở top-5. WLASL đúng 277 và 703. Class-macro Rec@1 là 16,79 (ASLC) và 9,22 (WLASL).

Chi phí trích xuất: ASLC train 333.667 window (479 s), test 199.852 window (320 s).

### 5.2 Ba readout trên test

| Readout | ASLC Rec@1 | ASLC Rec@5 | ASLC DCG | WLASL Rec@1 | WLASL Rec@5 | WLASL DCG |
|---|---:|---:|---:|---:|---:|---:|
| `grp` (chính) | **16,68** | **36,90** | **39,62** | **9,62** | **24,42** | **30,55** |
| `mean` | 14,23 | 32,94 | 36,89 | 9,62 | 23,86 | 30,48 |
| `cls` | 0,26 | 0,83 | 11,39 | 2,19 | 4,52 | 14,62 |

Readout chính được cố định trước khi chấm test. Không dùng test để đổi readout.

### 5.3 Test ASL Citizen theo signer (readout `grp`)

| Signer | Số query | DCG | Rec@1 | Rec@5 |
|---|---:|---:|---:|---:|
| P47 | 2.966 | 52,26 | **27,34** | 55,36 |
| P18 | 2.998 | 46,38 | 22,05 | 47,06 |
| P35 | 3.004 | 46,42 | 21,24 | 47,27 |
| P15 | 3.000 | 42,40 | 19,43 | 40,63 |
| P9 | 2.988 | 39,67 | 16,63 | 37,22 |
| P17 | 3.000 | 36,34 | 13,83 | 32,37 |
| P22 | 2.997 | 36,43 | 13,58 | 32,33 |
| P6 | 2.993 | 35,58 | 13,26 | 30,91 |
| P42 | 2.996 | 33,77 | 12,98 | 27,77 |
| P49 | 3.000 | 33,94 | 11,80 | 28,10 |
| P48 | 2.999 | 32,73 | **11,40** | 27,01 |

Rec@1 dao động 2,4 lần giữa signer (11,40–27,34). Đây là mô tả, chưa chứng minh nhân tố gây ra.

### 5.4 Đánh giá pilot-compatible (validation, không phải test)

| Protocol | Cấu hình | DCG | MRR | Rec@1 | Rec@5 | Rec@10 |
|---|---|---:|---:|---:|---:|---:|
| LOSO-val (10.304 clip, 6 signer) | white512 | 51,32 | 40,61 | **29,54** | 53,73 | 62,23 |
| LOSO-val | raw (`mean`) | 42,59 | 30,15 | 19,68 | 41,50 | 51,63 |
| ASLC300 (gallery 3.000, query 1.500, 300 gloss) | white512 | 67,12 | 58,59 | **47,67** | 72,00 | 80,33 |
| ASLC300 | raw (`mean`) | 62,72 | 53,15 | 40,67 | 67,47 | 76,33 |
| ASLC300 linear probe | C = 1,0 | | | macro Top-1 **68,87** | | |

Lưu ý: LOSO white512 fit PCA trên window của cả validation (không nhãn), nên là thiết lập transductive. ASLC300 fit whitening trên gallery train. Frozen dictionary không whitening. Ba thiết lập không dùng thay thế cho nhau.

### 5.5 Probe đặc trưng (validation, tách theo signer)

Ridge R² trên các prior hình học, 2 outer fold tách signer, alpha chọn qua inner signer CV.

| Prior | Full160 | Pilot 16k |
|---|---:|---:|
| Handshape | 0,3246 | 0,2766 |
| Location relations | 0,1711 | 0,1242 |
| Body | 0,0320 | 0,0467 |
| Movement | −0,0295 | −0,0349 |
| Non-manual | −0,2599 | −0,2755 |

| Hình học và signer | Full160 | Pilot 16k |
|---|---:|---:|
| Signer probe accuracy | 93,73% | 90,27% |
| Mean cosine (không whitening) | 0,4397 | 0,3565 |
| Effective rank (embedding) | 481,79 | 423,21 |

Movement và non-manual có R² âm: probe tuyến tính không giải mã tốt hơn baseline trung bình. Không suy ra model hoàn toàn không chứa thông tin đó.

---

## 6. So sánh

### 6.1 Full160 so với pilot 16k (cùng method, khác corpus và ngân sách)

| Đánh giá | 16k (3k video) | Full160 (40k video) | Chênh | CI95 (bootstrap theo signer) | Số signer tăng |
|---|---:|---:|---:|---|---|
| LOSO white512 Rec@1 | 25,38 | 29,54 | **+4,16** | [+1,91; +5,12] | 4/6 |
| LOSO raw Rec@1 | 16,96 | 19,68 | +2,72 | | |
| LOSO white512 Rec@5 | 49,12 | 53,73 | +4,61 | | |
| ASLC300 white512 Rec@1 | 48,93 | 47,67 | −1,27 | [−3,01; +2,25] | 1/5 |
| ASLC300 raw Rec@1 | 44,20 | 40,67 | −3,53 | | |
| ASLC300 linear probe macro | 60,80 | 68,87 | **+8,07** | [+4,06; +12,12] | 5/5 |

Hai ghi chú:
- Trên LOSO, full160 đúng thêm 1.322 query và sai thêm 893, lợi ròng 429/10.304.
- Pilot 16k chỉ train trên đúng 3.000 clip thuộc 300 gloss của ASLC300, nên có lợi thế chuyên biệt trên tập này. Đây là giả thuyết giải thích việc retrieval ASLC300 không tăng. Chưa tách được tác động riêng của lượng dữ liệu, vì lịch lr, lịch EMA và số bước cũng thay đổi theo.

Theo signer (Rec@1, full160 − pilot 16k):

| Protocol | Signer | n | 16k | Full160 | Chênh (điểm %) | Query sửa được | Query mất |
|---|---|---:|---:|---:|---:|---:|---:|
| LOSO | P12 | 697 | 28,84 | 26,83 | −2,01 | 79 | 93 |
| LOSO | P13 | 2 | 0,00 | 0,00 | 0,00 | 0 | 0 |
| LOSO | P21 | 2.997 | 24,02 | 29,26 | +5,24 | 412 | 255 |
| LOSO | P26 | 2.999 | 24,67 | 29,91 | +5,24 | 397 | 240 |
| LOSO | P39 | 3.000 | 28,10 | 31,83 | +3,73 | 369 | 257 |
| LOSO | P5 | 609 | 18,23 | 21,02 | +2,79 | 65 | 48 |
| ASLC300 | 1ea34f8f… | 182 | 41,76 | 40,66 | −1,10 | 18 | 20 |
| ASLC300 | 41150924… | 475 | 54,11 | 50,11 | −4,00 | 58 | 77 |
| ASLC300 | 57742f9f… | 385 | 47,01 | 46,23 | −0,78 | 43 | 46 |
| ASLC300 | 9567c431… | 54 | 31,48 | 48,15 | +16,67 | 14 | 5 |
| ASLC300 | b706cc4b… | 404 | 50,25 | 49,26 | −0,99 | 41 | 45 |

### 6.2 Các PLctx và biến thể khác trên cùng evaluator pilot (validation)

| Method | LOSO white Rec@1 | ASLC300 white Rec@1 | ASLC300 LP macro |
|---|---:|---:|---:|
| PLctx 4k (3k video) | 22,64 | 46,27 | 49,40 |
| PLctx 16k (3k video) | 25,38 | **48,93** | 60,80 |
| **PLctx full160 (40k video)** | **29,54** | 47,67 | **68,87** |
| PLint 16k | 24,67 | 43,27 | 55,93 |
| JH3D (RGB, Hiera-MAE init) | 39,19 | 56,80 | 61,73 |
| JV3D (RGB, V-JEPA init) | 27,98 | 22,60 | 56,40 |
| B13 (skeleton graph-JEPA) | 7,93 | 21,27 | 25,13 |
| SR3D (tái tạo SignRep) | 5,14 | 10,07 | 17,60 |

Chênh có CI (bootstrap theo signer) của full160 so với một số model khác:

| Cặp | LOSO Rec@1 chênh | CI95 | ASLC300 Rec@1 chênh | CI95 |
|---|---:|---|---:|---|
| Full160 − PLctx 4k | +6,90 | [+5,68; +7,62] | +1,40 | [−1,59; +3,74] |
| Full160 − PLint 16k | +4,87 | [+4,38; +5,18] | +4,40 | [+2,11; +8,45] |
| Full160 − JH3D | −9,65 | [−11,52; −3,90] | −9,13 | [−22,49; +11,11] |
| Full160 − JV3D | +1,56 | [−0,28; +4,74] | +25,07 | [+15,00; +37,89] |
| Full160 − SR3D | +24,40 | [+21,94; +26,20] | +37,60 | [+35,49; +41,01] |
| Full160 − B13 | +21,61 | [+20,18; +22,97] | +26,40 | [+17,58; +32,43] |

Các model RGB và SR3D/JH3D/JV3D khác điều kiện (input, init, pretrain) với PLctx. Chỉ so sánh để tham chiếu.

### 6.3 Full160 và SignRep trên test chính thức

| Dataset | Method | Pooling | Pretrain | DCG | Rec@1 | Rec@5 |
|---|---|---|---|---:|---:|---:|
| ASL Citizen | **PLctx full160** | avg (grp) | ASL Citizen train, 40k | 39,62 | 16,68 | 36,90 |
| ASL Citizen | SignRep public (đo local) | weighted | YouTube-SL-25 | 70,42 | 49,02 | 79,29 |
| ASL Citizen | SignRep paper, Table 4 | avg | YouTube-SL-25 | 61,40 | 37,47 | 68,77 |
| ASL Citizen | SignRep paper, Table 4 | weighted | YouTube-SL-25 | 71,21 | 49,95 | 80,09 |
| WLASL2000 | **PLctx full160** | avg (grp) | ASL Citizen train, 40k | 30,55 | 9,62 | 24,42 |
| WLASL2000 | SignRep public (đo local) | avg | YouTube-SL-25 | 53,89 | 26,50 | 60,75 |
| WLASL2000 | SignRep public (đo local) | weighted | YouTube-SL-25 | 57,13 | 29,42 | 65,47 |
| WLASL2000 | JH3D | avg | ASL Citizen 3k | 32,04 | 8,37 | 25,49 |
| WLASL2000 | JV3D | avg | ASL Citizen 3k | 25,31 | 4,38 | 14,38 |
| WLASL2000 | SR3D | avg | ASL Citizen 3k | 15,89 | 2,33 | 4,48 |

- Full160 thấp hơn SignRep public trên cả hai tập. Điều kiện khác nhau: SignRep dùng RGB 16 frame, pretrain YouTube-SL-25 và có thể dùng activity head.
- Trên WLASL, full160 cao hơn JH3D về Rec@1 (9,62 so với 8,37) nhưng thấp hơn về DCG và Rec@5, nên không vượt toàn diện.

---

## 7. Điều rút ra và giới hạn

1. Full160 là một run hoàn tất, ổn định số học, không collapse. Nó cải thiện LOSO (+4,16) và linear probe (+8,07) so với pilot 16k.
2. Retrieval trên ASLC300 không tăng so với pilot 16k (−1,27, CI chứa 0). Proxy này chỉ có 300 gloss nên phóng đại điểm: 47,67 trên ASLC300 so với 16,68 trên test đầy đủ.
3. Trên test chính thức, readout `grp` quan trọng: `mean` thấp hơn 2,45 điểm, `cls` gần bằng 0. CLS không có loss trực tiếp nên không hữu ích cho retrieval.
4. Probe cho thấy hình dạng tay và quan hệ vị trí được mã hoá tốt hơn pilot. Movement và non-manual vẫn không được giải mã tuyến tính. Signer probe tăng lên 93,7%, tức danh tính người ký vẫn dễ đọc từ đặc trưng.
5. Giới hạn:
   - Chỉ 1 seed. CI chỉ đo biến thiên theo signer, không đo biến thiên giữa các seed.
   - LOSO có 6 signer nhưng P13 chỉ có 2 clip; ASLC300 có 5 signer. CI mang tính mô tả và không điều chỉnh nhiều phép so sánh.
   - Không có đường cong theo epoch, nên chưa chọn được checkpoint tốt nhất hay phát hiện overfit.
   - 8 query test (0,024%) và 39 gallery clip có embedding `grp` bằng 0 vẫn được giữ trong đánh giá. Con số này quá nhỏ để giải thích mức điểm thấp.
   - Test đầy đủ ASL Citizen và WLASL đã được dùng cho run này một lần. Các project khác trong repo cũng đã chấm trên các test split này.
6. Đề xuất bước tiếp (chưa thực hiện): train lại với checkpoint ở epoch 20/40/80/120/160, chấm VAL-DICT từng mốc, thêm nhiều seed và ablation random-handedness. Xem `Skeleton_SSL_Sign_Research_Design_VI.md`, Phase A.
