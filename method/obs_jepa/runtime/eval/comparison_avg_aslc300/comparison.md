# So sánh dictionary retrieval ASLC300: SignRep(avg)

Thời điểm: 2026-10-11T16:09:12+07:00

Gallery: 3.000 video official train, 300 gloss × 10. Query: 1.500 video official validation, 300 gloss × 5, 5 người ký không có trong gallery. Không dùng test.

Pooling: trung bình feature window gốc → L2 ở cấp clip → cosine; không PCA whitening, không activity weighting. Class score là cosine cao nhất trong các video gallery của lớp. DCG = mean(1/log2(1+r)); Rec@K = mean(r≤K). Mọi cột dưới đây nhân 100.

RGB: window 16 frame, stride 8; các nhánh A/A00 dùng letterbox và effective fps ≤25 theo preprocessing của chúng, Hiera/JV3D/SR-VJ/public dùng native-frame squash 224. B: landmark 2D, window 64 frame, stride 32, effective fps ≤25. Cache public stride 2 được chọn lại tại start % 8 = 0.

Lưu ý: retrieval_mean cũ chuẩn hóa L2 từng window trước khi pooling. Bảng này tính lại từ feature gốc, chỉ L2 sau pooling; vì vậy số liệu có thể khác các báo cáo trước.

| Method | Input | DCG | Rec@1 | Rec@5 | Dữ liệu train ký hiệu | Khởi tạo |
| --- | --- | ---: | ---: | ---: | --- | --- |
| SignRep_public | RGB | 84.56 | 72.20 | 90.80 | YouTube-SL-25 (tác giả); không train thêm trên ASLC002 | Hiera-B MAE-Kinetics |
| JH3D | RGB | 68.93 | 45.93 | 78.60 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | Hiera-B MAE-Kinetics |
| JV3D | RGB | 56.65 | 29.80 | 62.80 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | V-JEPA 2.1 ViT-B distilled từ ViT-G |
| B11 | Landmark 2D | 32.50 | 10.07 | 24.07 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | Ngẫu nhiên |
| B13 | Landmark 2D | 31.10 | 9.47 | 22.67 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | Ngẫu nhiên |
| B12 | Landmark 2D | 31.53 | 8.93 | 24.00 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | Ngẫu nhiên |
| B10 | Landmark 2D | 26.14 | 5.53 | 15.73 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | Ngẫu nhiên |
| SR3D | RGB | 25.17 | 5.00 | 14.13 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | Hiera-B MAE-Kinetics |
| SRVJ | RGB | 21.83 | 2.67 | 8.93 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | V-JEPA 2.1 ViT-B distilled từ ViT-G |
| A11_AD | RGB | 21.84 | 2.53 | 8.93 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | V-JEPA 2.1 ViT-B distilled từ ViT-G |
| A10_AD | RGB | 20.00 | 1.80 | 6.33 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | V-JEPA 2.1 ViT-B distilled từ ViT-G |
| ASLC002 | RGB | 18.99 | 1.73 | 4.87 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | Hiera-B MAE-Kinetics |
| A00 | RGB | 19.48 | 1.73 | 5.87 | Không train thêm trên ASLC002; dùng V-JEPA 2.1 public | V-JEPA 2.1 ViT-B distilled từ ViT-G |
| A10_AR | RGB | 19.62 | 1.47 | 5.33 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | V-JEPA 2.1 ViT-B distilled từ ViT-G |
| A12_AD | RGB | 17.79 | 0.87 | 3.13 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | V-JEPA 2.1 ViT-B distilled từ ViT-G |
| A13_AD | RGB | 18.24 | 0.80 | 3.87 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | V-JEPA 2.1 ViT-B distilled từ ViT-G |
| A10c_AD | RGB | 17.72 | 0.73 | 3.47 | ASLC002: 3.000 video ASL Citizen train, không nhãn gloss | V-JEPA 2.1 ViT-B distilled từ ViT-G |

Tất cả nhánh tự train dùng checkpoint cuối, một seed. ASLC002 là lượt 5.000 bước, batch 4 view; EX16/SR-VJ là 4.000 bước, 32 view; A/B là 4.000 bước, batch 32. A00 không train thêm. Public SignRep là checkpoint tác giả, SHA256 f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b.

Kiểm tra: toàn bộ 16 manifest tự train/foundation trùng thứ tự; public phủ đủ sample ID, gloss, path và split; gallery/query không trùng video hoặc người ký; xếp hạng đối chiếu bằng sort độc lập và nearest-video Top-1. Tie true-class được xử lý bất lợi cho query và ghi số lượng trong JSON.

P3 bị bỏ qua do B gate không đạt; P4 chưa có kết quả ở thời điểm tạo bảng. Không bổ sung các thí nghiệm cũ ngoài EX16/proposal11 vào bảng này.

Nguồn phương pháp: SignRep §6, §7.1–7.4, https://arxiv.org/html/2503.08529v1 ; public checkpoint https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1 .
