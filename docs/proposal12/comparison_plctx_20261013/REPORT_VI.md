# PLctx16k / full160epoch và các method đã chạy

Ngày tổng hợp:2026-10-13. Chỉ đọc và tổng hợp các kết quả đã có; không chạy training/evaluation mới.
DCG trong mọi bảng dùng **DCG×100**; Rec@K và LP dùng %. Không gộp protocol có/không whitening hoặc validation/test.

## Ngân sách training

| Run | Video train | Steps | Batch | Lượt clip trung bình/video |
|---|---:|---:|---:|---:|
| PLctx16k | 3.000 | 16.000 |32|170,67|
| PLctxfull160epoch |40.154|200.640|32|159,90|

Full160 train scratch,seed0; không warm-start pilot. Video train tăng13,38×, số optimizer update tăng12,54×. Không thể quy khác biệt riêng cho số epoch; lượt tiếp xúc trung bình mỗi clip tương đương, và corpus mở rộng từ300 lên2731 gloss.

## ASLC300: cùng evaluator white512

Gallery3.000 clip train/query1.500 clip validation,300 gloss. Encoder frozen; whitening fit gallery. Đây là protocol đã dùng trong các câu trả lời trước.

| Method | DCG×100 | Rec@1 | Rec@5 | LP macro |
|---|---:|---:|---:|---:|
| JH3D | 73.57 | 56.80 | 77.93 | 61.73 |
| PLctx 16k | 68.62 | 48.93 | 74.07 | 60.80 |
| PLctx full160epoch | 67.12 | 47.67 | 72.00 | 68.87 |
| PLctx 4k | 66.51 | 46.27 | 72.53 | 49.40 |
| PLint 16k | 64.26 | 43.27 | 69.60 | 55.93 |
| PLboth 4k | 63.62 | 41.87 | 69.53 | 47.13 |
| B-MAE 4k | 56.84 | 34.60 | 58.93 | 46.13 |
| B-MAMP 4k | 53.03 | 30.40 | 54.40 | 34.47 |
| S1 4k | 47.81 | 23.33 | 47.60 | 44.00 |
| JV3D | 46.74 | 22.60 | 44.73 | 56.40 |
| B13 | 44.25 | 21.27 | 40.67 | 25.13 |
| S2 4k | 39.93 | 15.73 | 35.60 | 36.13 |
| SR3D | 32.20 | 10.07 | 23.07 | 17.60 |

Bảng đầy đủ mọi primary/group readout đã có cùng protocol: [all_methods_white512.csv](all_methods_white512.csv). Không chọn giữa grp/CLS theo test; các biến thể CLS không đưa vào bảng này.

## LOSO validation: cùng white512

10.304 clip validation,6 signer; gallery nằm ở các signer khác trong validation. PCA whitening không nhãn fit trên validation nên thiết lập này có tính transductive.

| Method | DCG×100 | Rec@1 | Rec@5 |
|---|---:|---:|---:|
| PLctx 4k | 44.86 | 22.64 | 44.90 |
| PLctx 16k | 47.89 | 25.38 | 49.12 |
| PLctx full160epoch | 51.32 | 29.54 | 53.73 |
| JH3D | 60.34 | 39.19 | 65.19 |
| JV3D | 51.29 | 27.98 | 53.29 |
| S1 4k | 30.63 | 10.49 | 24.46 |
| S2 4k | 25.93 | 7.20 | 17.94 |

## Full160 −16k: paired signer-cluster bootstrap

| Metric | Chênh lệch (điểm%) | CI95 |
|---|---:|---|
| loso_R@1_white512 | +4.16 | [+1.91, +5.12] |
| aslc300_linear_probe_macro | +8.07 | [+4.06, +12.12] |
| aslc300_retrieval_R@1 | -1.27 | [-3.01, +2.25] |

Full160 tăng LOSO Rec@1 và LP; ASLC300 retrieval giảm nhẹ với CI95 chứa0. CI đo biến thiên theo signer của một cặp model, không đo biến thiên qua seed. Chỉ một seed và nhiều phép so sánh chưa điều chỉnh.

## Frozen dictionary: official test, không whitening

Raw window mean → video L2 → cosine → max gallery score theo gloss. PLctx dùng skeleton64frame,stride2,avg; RGB references dùng16frame,stride2; weighted dùng activity head. Input, pretrained weights và data budget khác nhau.

| Dataset | Method/pooling | DCG×100 | Rec@1 | Rec@5 |
|---|---|---:|---:|---:|
| ASL Citizen | PLctx_FULL_160ep_grp / avg | 39.62 | 16.68 | 36.90 |
| ASL Citizen | SignRep_public / weighted | 70.42 | 49.02 | 79.29 |
| WLASL2000 | PLctx_FULL_160ep_grp / avg | 30.55 | 9.62 | 24.42 |
| WLASL2000 | SignRep_public / avg | 53.89 | 26.50 | 60.75 |
| WLASL2000 | SignRep_public / weighted | 57.13 | 29.42 | 65.47 |
| WLASL2000 | JH3D / avg | 32.04 | 8.37 | 25.49 |
| WLASL2000 | JH3D / weighted | 31.95 | 8.34 | 25.32 |
| WLASL2000 | JV3D / avg | 25.31 | 4.38 | 14.38 |
| WLASL2000 | SR3D / avg | 15.89 | 2.33 | 4.48 |
| WLASL2000 | SR3D / weighted | 16.34 | 2.43 | 4.97 |

ASL Citizen test:40.154 gallery/32.941 query,2731 gloss. WLASL2000:14.296 gallery/2.879 query,2000 gloss. Chưa có PLctx16k trên hai full-test protocol trong các artifact đã kiểm tra; không dùng điểm pilot validation thay thế.

Full160 thấp hơn SignRep public trong cả hai tập. Trên WLASL, full160 hơn JH3D về Rec@1 nhưng thấp hơn DCG/Rec@5, nên không vượt toàn diện.

## Kết luận từ kết quả đã đo

- Trong các skeleton primary/group run đã thống kê, PLctx16k có ASLC300 Rec@1 cao nhất; full160 không cải thiện metric này.
- Full160 có representation tốt hơn trên LOSO và linear probe; LP68,87% cao hơn JH3D61,73% trong bảng cùng evaluator, nhưng retrieval thấp hơn JH3D.
- Raw dictionary full-test còn yếu. Các điểm SignRep public dùng pretrainRGB bên ngoài và pooling khác phải được ghi như reference khác điều kiện.
- Geometric probes full160:handshapeR²=.3246,relations=.1711,body=.0320,movement=−.0295,nonmanual=−.2599. Signer accuracy93,73%; chuyển động/nonmanual chưa được probe tuyến tính giải mã tốt.

## Nguồn và kiểm tra

- Các cache JSON trùng tên trong nhiều runtime được đối chiếu nội dung, không cộng thành nhiều run.
- `sources.json` ghim SHA256 của các kết quả/manifest đã đọc; script `summarize.py` tạo lại bảng.
- Kết quả full dictionary đối chiếu results/protocol/verification với CSV; không đưa paperTable4 chưa tái chạy vào bảng measured.
- ASLC300 raw avg trong bảng proposal11 dùng protocol/window pooling khác white512; SignRep_public84,56/72,20/90,80 ở bảng đó không được chèn vào bảng white512.
- Không chạy model mới và không chọn hyperparameter/checkpoint theo test.
