# JEPA cho biểu diễn ngôn ngữ ký hiệu: literature review phản biện và ba thiết kế end-to-end

**Ngày chốt tra cứu:** 09/10/2026.  
**Tài liệu được phản biện:** `JEPA_Sign_Language_Deep_Research_End_to_End(1).md` do người dùng cung cấp.  
**Trạng thái:** nghiên cứu literature và thiết kế phương pháp; chưa huấn luyện, chưa có checkpoint mới, chưa có kết quả thực nghiệm của các phương án đề xuất.

Tài liệu này thay thế cách chọn kiến trúc bằng một danh sách thành phần bằng cách chốt ba pipeline có thể triển khai và bác bỏ được. Mỗi pipeline quy định tín hiệu đầu vào, hệ tọa độ, encoder, mask, teacher, loss, lịch training, inference và đối chứng. Các giá trị cấu hình là điểm khởi đầu để pilot, không phải hyperparameter đã chứng minh tối ưu.

Quy ước bằng chứng:

- **[LIT-M]**: đã kiểm tra phần phương pháp, kết quả hoặc phụ lục liên quan trong nguồn gốc. Không đồng nghĩa đã kiểm tra mã nguồn hay tái lập kết quả.
- **[LIT-A]**: chỉ xác minh abstract, metadata hoặc mô tả chính thức; không dùng để suy luận chi tiết chưa đọc được.
- **[LIT-D]**: đã kiểm tra specification hoặc dataset description chính thức.
- **[INFER]**: phân tích suy ra từ phương pháp và điều kiện bài toán, chưa phải kết quả thực nghiệm về sign language.
- **[DESIGN]**: lựa chọn mới trong tài liệu này, cần thí nghiệm.
- **[VERIFY]**: điều kiện chưa được xác minh; không dùng làm tiền đề cho claim.

## Mục lục

1. [Kết luận và quyết định nghiên cứu](#1-kết-luận-và-quyết-định-nghiên-cứu)
2. [Cách nghiên cứu và các vòng phản biện](#2-cách-nghiên-cứu-và-các-vòng-phản-biện)
3. [Audit bản gốc: giữ, sửa, bỏ hoặc hoãn](#3-audit-bản-gốc-giữ-sửa-bỏ-hoặc-hoãn)
4. [Literature theo cơ chế và mức độ cạnh tranh](#4-literature-theo-cơ-chế-và-mức-độ-cạnh-tranh)
5. [Bài toán representation thực sự cần giải](#5-bài-toán-representation-thực-sự-cần-giải)
6. [Hợp đồng dữ liệu, preprocessing và chống leakage](#6-hợp-đồng-dữ-liệu-preprocessing-và-chống-leakage)
7. [Phương án A: RGB-JEPA với mục tiêu quan sát và dự đoán tách biệt](#7-phương-án-a-rgb-jepa-với-mục-tiêu-quan-sát-và-dự-đoán-tách-biệt)
8. [Phương án B: graph-JEPA trong các không gian geometry tương thích](#8-phương-án-b-graph-jepa-trong-các-không-gian-geometry-tương-thích)
9. [Phương án C: RGB student học từ geometry teacher đã kiểm chứng](#9-phương-án-c-rgb-student-học-từ-geometry-teacher-đã-kiểm-chứng)
10. [Masking, augmentation và curriculum chốt cho từng phương án](#10-masking-augmentation-và-curriculum-chốt-cho-từng-phương-án)
11. [Training recipes, pseudocode và kiểm tra implementation](#11-training-recipes-pseudocode-và-kiểm-tra-implementation)
12. [Đánh giá representation và ngôn ngữ](#12-đánh-giá-representation-và-ngôn-ngữ)
13. [Đối chứng, ablation và diễn giải nhân quả](#13-đối-chứng-ablation-và-diễn-giải-nhân-quả)
14. [Compute, thứ tự triển khai và điều kiện quyết định](#14-compute-thứ-tự-triển-khai-và-điều-kiện-quyết-định)
15. [Novelty, phản biện reviewer và giới hạn](#15-novelty-phản-biện-reviewer-và-giới-hạn)
16. [Tài liệu gốc và độ sâu kiểm tra](#16-tài-liệu-gốc-và-độ-sâu-kiểm-tra)

## 1. Kết luận và quyết định nghiên cứu

**Khuyến nghị:** triển khai A để có một baseline RGB đáng tin và một thí nghiệm nhân quả nhỏ; triển khai B như một hướng geometry độc lập và nguồn teacher có thể kiểm chứng; chỉ triển khai C khi B hoặc một teacher khác vượt qua kiểm tra chất lượng. Không bắt đầu bằng một tổng loss gồm video, handshape, motion, relations, face và text cùng lúc.

Giả thuyết nên nghiên cứu là:

> Với cùng dữ liệu, backbone và ngân sách, representation của ký hiệu có thể tốt hơn nếu giữ thông tin chi tiết của articulator đang quan sát được, đồng thời chỉ dùng masked prediction ở độ hạt và mức che mà context thực sự hỗ trợ. Lợi ích phải còn tồn tại khi kiểm soát teacher, lượng thông tin nhìn thấy và metadata của mask.

Đây là **[DESIGN]**, chưa phải kết luận thực nghiệm. Cụm “context thực sự hỗ trợ” phải được vận hành bằng kiểm tra predictability trên validation và các đối chứng; không thể thay bằng một confidence score tùy ý.

Những sửa đổi có ảnh hưởng lớn nhất:

1. **Novelty đã bị thu hẹp bởi literature 2026.** SignMAE đã dùng masking liên quan tay, cánh tay và chuyển động; SignSeek đã kết hợp graph theo articulator, saliency và dự đoán latent phần bị che. Vì vậy “hand-aware mask + graph + JEPA” không đủ là đóng góp mới. SignSeek dùng gloss supervision trong pretraining, cần phân biệt với thiết lập không nhãn của đề xuất này. [S09, S10]
2. **EMA không buộc model giữ handshape hoặc non-manuals.** Nếu một đặc trưng riêng của tay/mặt không suy ra được từ context, loss dự đoán có thể ưu tiên phần thông tin dễ đoán hơn. Cần thêm mục tiêu ở vùng thực sự quan sát được hoặc teacher cố định đủ tốt; thêm một head có tên “handshape” chưa giải quyết được vấn đề.
3. **Không được coi mọi hệ tọa độ là cùng không gian.** Hand-world và pose-world của MediaPipe có gốc khác nhau. Quan hệ tay–tay/tay–mặt phải tính từ tọa độ ảnh chung hoặc từ 3D đã đăng ký về cùng hệ. Khi chỉ có 2D, không được gọi các vector suy ra là palm normal 3D. [S35, S36]
4. **Bắt buộc che trước các phép trộn và kiểm tra dependency của feature.** Đạo hàm thời gian, bone vectors, canonicalization, smoothing, graph và crop queries đều có thể mang thông tin target vào context. Mask RGB và pose “cùng vùng” chưa đủ ngăn các đường rò này.
5. **V-JEPA 2.1 ViT-B là checkpoint distilled.** Khởi tạo từ checkpoint này rồi tạo EMA teacher mới là một recipe adaptation mới; không phải tái lập nguyên xi pretraining của V-JEPA 2.1. [S04, S05]
6. **Phải tách true linear probe khỏi attentive probe.** Frozen encoder với một attention pooling được học là một phép đánh giá khác. Protocol cần kiểm tra đồng thời phân biệt ký hiệu, thuộc tính ngôn ngữ và khả năng tổng quát qua signer/lexical identity.

### 1.1. Ba hướng, giá trị dự kiến và điều kiện tiếp tục

| Phương án | Đầu vào lúc inference | Cơ chế chính | Câu hỏi nghiên cứu | Rủi ro quyết định |
|---|---|---|---|---|
| **A — RGB-JEPA** | RGB của toàn vùng ký hiệu | EMA latent prediction; target weighting và partial masking theo vùng; giữ mục tiêu quan sát tách khỏi suy luận | Structured masking có giúp hơn generic domain adaptation khi lượng thông tin và target được kiểm soát? | Teacher generic có thể chưa giữ chi tiết ngón tay; RGB nhỏ không chứa đủ chi tiết |
| **B — geometry graph-JEPA** | Landmarks, trạng thái quan sát, confidence nếu có; RGB không bắt buộc | Spatial graph + temporal attention; geometry trong các không gian hợp lệ; masked latent và observable geometry anchors | Geometry có giữ shape, orientation, trajectory và relations tốt hơn raw-coordinate SSL? | Sai detector, thiếu 3D và non-manuals; normalization xóa đặc trưng cần phân biệt |
| **C — RGB + frozen geometry teacher** | RGB; teacher chỉ dùng khi train | Visible-region geometry distillation và masked prediction cho phần structure được kiểm chứng | Geometry giúp RGB nhờ target tốt hơn hay chỉ nhờ extra supervision? | Lợi ích có thể bằng simple distillation; teacher sai hoặc target không quan sát được trong RGB |

Không có phương án nào hiện có cơ sở để hứa tăng accuracy. A có rủi ro triển khai thấp nhất vì tận dụng encoder có sẵn. B cho kiểm soát tín hiệu và cost tốt hơn nhưng biểu diễn bị giới hạn bởi pose. C là hướng nhiều điều kiện nhất, chỉ đáng chạy sau khi biết teacher giữ được thông tin gì.

## 2. Cách nghiên cứu và các vòng phản biện

Tra cứu bao phủ năm nhóm: predictive SSL/JEPA; video masked learning và self-distillation; sign-specific representation; skeleton/graph/motion; pose quality, dữ liệu và linguistic evaluation. Ưu tiên paper, proceedings, repository và tài liệu API chính thức. Nguồn thứ cấp không được dùng để xác nhận nội dung kỹ thuật của một paper chưa lấy được bản gốc.

Các chuỗi tìm kiếm gồm tên phương pháp trong file gốc, “sign language self-supervised masked representation”, “sign language JEPA latent prediction”, “motion-aware masking skeleton”, “sign language pose estimation evaluation”, “phonological evaluation signer independent”, cùng các công trình 2025–2026. Danh mục cuối tài liệu chỉ rõ nguồn được đọc ở mức nào. Đây là critical literature review theo vấn đề, **không phải systematic review đăng ký trước hoặc tuyên bố đã bao quát mọi công trình**.

### 2.1. Những thay đổi sau từng vòng

| Vòng | Giả định ban đầu được kiểm tra | Bằng chứng/phản ví dụ | Thiết kế thay đổi |
|---|---|---|---|
| 1 — novelty | Hand-aware masking và relation targets là khoảng trống đủ lớn | SignRep đã tái tạo sign priors và quan hệ; SignMAE đã mask vùng/chuyển động; SignSeek đã graph + masked latent | Chuyển trọng tâm sang observability, tính đúng của feature dependencies và protocol nhân quả |
| 2 — target | Latent teacher tránh nuisance nên sẽ giữ linguistic semantics | Teacher có thể bỏ chi tiết khó đoán; SHuBERT cho thấy discrete targets là đối thủ thực tế; S-JEPA dùng distributional CE | Đòi hỏi teacher audit; so latent, cluster và raw priors với controls phù hợp |
| 3 — input | Local 3D hand + body 3D có thể tạo quan hệ trực tiếp | Gốc world-hand/world-body khác nhau; monocular 2D không xác định palm normal | Chốt image coordinates chung; tách registered global 3D khỏi local-hand 3D |
| 4 — masking | Mask cùng RGB/pose là đủ | Velocity, normalization, GCN, query vị trí có thể lộ target | Dependency closure và metadata-only controls trở thành yêu cầu implementation |
| 5 — evaluation | Frozen classification + minimal pairs đủ xác nhận representation | SHuBERT đã có phonological evaluation; known-word labels có thể cho phép ghi nhớ ánh xạ | Thêm lexical-disjoint attribute probes, signer/source controls và teacher-matched distillation |

### 2.2. Những gì chưa xác minh được

- Không tái lập training, không chạy benchmark và không kiểm tra toàn bộ source code của các đối thủ.
- AHA-JEPA trong bản gốc: chưa truy xuất được nguồn gốc đủ để xác nhận các chi tiết “frozen Sign-JEPA carrier”. Giữ ở mục cần xác minh, không dùng nó làm nền kỹ thuật.
- MS-MAE: xác minh proceedings và abstract chính thức; chưa đọc được đầy đủ PDF phương pháp trong lượt này. Không tự bổ sung chi tiết mask/loss.
- Path-JEPA: xác minh bản ghi ECCV 2026 và nội dung được index, nhưng bản PDF đầy đủ không lấy được. Không dùng làm bằng chứng hiệu quả của một target path-signature trên sign language.
- Mã và weights được công bố không tự động chứng minh giấy phép phù hợp, không overlap evaluation hoặc phù hợp camera/ngôn ngữ của người dùng. Các điều kiện này cần audit ở lúc chọn dữ liệu và checkpoint.

## 3. Audit bản gốc: giữ, sửa, bỏ hoặc hoãn

Các mục dưới đây tham chiếu số section của file gốc để có thể truy ngược quyết định.

| Thành phần trong bản gốc | Kết luận phản biện | Thay đổi cụ thể |
|---|---|---|
| §1, §9: RGB-centric là hướng ưu tiên | **Giữ có điều kiện** | Chạy frozen transfer trước; chỉ tiếp tục masked adaptation khi input có đủ hand resolution và teacher probe không suy giảm |
| §6: novelty quanh hand/relation-aware JEPA | **Sửa đáng kể** | Bổ sung SignMAE/SignSeek; không claim first-use, first relation target hoặc first linguistic probe |
| §6.2: SignRep gần MAE với sign priors | **Mô tả chính xác hơn** | SignRep thay pixel reconstruction bằng sign-prior reconstruction; vì thế là đối chứng target quan trọng, không chỉ đối chứng RGB reconstruction |
| §6.4: skeleton JEPA đều latent regression | **Sửa** | S-JEPA dùng centered/sharpened distributions và cross-entropy; phải có baseline tương ứng, không gắn nhãn reproduction cho L1 variant |
| §10, §18: ViT-B/16 2.1 + EMA + deep supervision | **Sửa provenance** | ViT-B/L 2.1 là distilled; đặt tên adaptation riêng, ghi hash checkpoint và predictor mới. Deep supervision là biến thí nghiệm |
| §10.3: factorization handshape/location/orientation/motion | **Giữ mục tiêu, bỏ bảo đảm** | Chỉ gọi là factor-targeted heads; không tuyên bố disentanglement vì chưa có identification assumptions/independence evidence |
| §11: palm/body canonicalization | **Sửa định nghĩa đầu vào** | 2D và 3D xử lý riêng; giữ phép biến đổi/handedness ngoài shape branch; không dùng hidden joints tạo basis cho student |
| §11.5: relations từ landmarks | **Giữ sau registration** | Dùng common image coordinates trước; nếu dùng 3D, bắt buộc nêu cùng camera/world frame và calibration |
| §13.5: motion-guided masking | **Giữ như ablation** | Motion cao có thể do jitter; motion thấp vẫn quan trọng. Thêm uniform floor, natural-occlusion strata và mask-metadata controls |
| §13.7: cross-modal mask alignment | **Mở rộng** | Kiểm tra duplicate wrists, derivative support, temporal smoothing, full-input graph và ROI query positions |
| §14: frozen geometry teachers chưa chỉ rõ cách tạo | **Thay thế** | Phương án B là recipe xây teacher hoàn chỉnh; phải so direct descriptor, frozen teacher và random teacher controls |
| §15: nhiều loss optional | **Thu gọn** | A có latent loss và biến context; B thêm một nhóm observable anchors; C thêm visible geometry trước masked geometry |
| §19: FPS 8–12, 16/32 frames | **Không lấy làm mặc định mọi nhiệm vụ** | Pilot giữ source timing, ưu tiên 25 fps nếu dữ liệu hỗ trợ; so cùng physical duration; đánh giá ngón tay/fingerspelling riêng |
| §20: mean/attentive pool đều “linear” | **Sửa** | Mean fixed pooling + linear là linear probe; trainable pooling là attentive probe, báo cáo riêng |
| §20: signer independent và linguistic contrast | **Giữ, cụ thể hóa** | Xác minh split từng corpus; bổ sung lexical-disjoint phonology, expert validation và paired cluster bootstrap |
| §23 YAML: target_fraction = null | **Chốt có thể thực thi** | Định nghĩa visible/context, target và guard; budget hidden 80% cho RGB pilot, 50% cho pose pilot; không để mask count tùy sampler |
| §27: claim relational contribution | **Hoãn tới khi có evidence** | Nếu simple distillation cùng teacher đạt tương đương, bỏ claim JEPA-specific; nếu mask metadata giải thích gain, thiết kế lại |

Các tỷ lệ 80%/50% trên là **[DESIGN]** cho pilot, không phải giá trị tối ưu đã biết. Bản gốc có nhiều cảnh báo đúng; phần thiếu chủ yếu là hợp đồng đầu vào, cơ chế làm target có ý nghĩa và phép thử có thể bác bỏ lựa chọn đó.

## 4. Literature theo cơ chế và mức độ cạnh tranh

### 4.1. JEPA và self-distillation: phải phân biệt target và recipe

| Công trình | Cơ chế đã kiểm tra | Hàm ý cho đề xuất |
|---|---|---|
| I-JEPA [S01] | Predictive learning từ context sang latent của target blocks; EMA target encoder | Hình dạng target và informative context là một phần phương pháp, không chỉ mask ratio |
| V-JEPA [S02] | Video masked latent prediction; full clean teacher, L1; short/long multi-block masking | Baseline generic cần giữ đúng teacher, predictor và temporal support khi so mask |
| V-JEPA 2 [S03] | Scaling video SSL và bước action-conditioned riêng cho world modeling | Không suy diễn rằng pretrained video features tự là linguistic world model |
| V-JEPA 2.1 [S04] | Dense context supervision, intermediate layers, image/video tokenizers; small models distilled | Dense/local và global không luôn cùng tăng; recipe large teacher khác adaptation ViT-B |
| S-JEPA [S06] | Skeleton EMA teacher; centered/sharpened distribution prediction với CE | Skeleton JEPA không đồng nghĩa raw L1; dữ liệu action 3D không đủ chứng minh fine-hand transfer |
| LeJEPA [S07] | Predictive objective cùng SIGReg chống collapse bằng regularization tường minh | Một lựa chọn anti-collapse khác để so, không tự động áp lý thuyết cho EMA + sign priors |
| DSeqJEPA [S08] | Saliency từ attention và ordered latent prediction, chủ yếu image setting; bản mới ghi accepted ECCV 2026 | Attention saliency chưa phải linguistic importance; không dùng kết quả image như bằng chứng sign |

**[LIT-M]** V-JEPA 2.1 công bố ViT-B/L là distilled từ model lớn; phụ lục distillation dùng frozen teacher, không dùng EMA teacher và deep supervision như recipe large-model. Table 1 cho thấy thêm context loss riêng làm SSv2 giảm 72.8 → 62.5 trong khi ADE20K tăng 22.2 → 33.8; multi-level prediction khôi phục phần lớn SSv2. Đây là trade-off trên benchmark đó, không phải số liệu sign language. [S04]

**[INFER]** Không thể lấy “dense features tốt” làm suy luận đủ rằng model phân biệt ngón tay tốt. Một vùng tay vài pixels có thể không chứa thông tin để xác định handshape; supervision chi tiết hơn không tạo ra thông tin bị mất trong đầu vào. Phải kiểm tra input resolution và target quality riêng.

Các cầu nối SSL hữu ích nhưng không nên nhập hết vào core:

- **data2vec/data2vec 2.0:** contextual teacher targets và tái sử dụng teacher cho nhiều corrupted views; gợi ý giảm cost, không tự cho biết target tốt cho ký hiệu. [S17, S18]
- **BYOL/SimSiam:** predictor, asymmetry và stop-gradient; cho thấy không cần negatives trong mọi thiết lập. Stop-gradient không phải một bảo đảm độc lập về semantics. [S19, S20]
- **iBOT/DINOv2:** local image token/self-distillation hữu ích cho hand/face teacher; chưa chứng minh giữ hand orientation hoặc mouth grammar. [S21, S22]
- **VICReg:** kiểm soát variance/covariance; high variance có thể vẫn do signer, camera hoặc position metadata. [S23]

### 4.2. Các đối thủ sign-specific trực tiếp

| Công trình | Supervision khi pretrain | Đầu vào và target | Điểm phải đối chứng |
|---|---|---|---|
| SignBERT/SignBERT+ [S11, S12] | Masked gesture SSL; sử dụng pose/model priors | Hand gestures; multi-granularity corruption, model-aware reconstruction | Hand-local tokenization, confidence filtering, handshape prior đã tồn tại |
| MASA [S13] | SSL trên skeleton | Motion-aware reconstruction và momentum semantic alignment | Motion targets và motion-biased masks không mới; so reconstruction/latent tách biệt |
| SignRep [S14] | Sign-prior supervision từ pose; không cần gloss cho bước representation | RGB masked encoder → hand/body geometry, angles, relations; thêm regularization/style handling | So cùng prior; cải thiện của C có thể chỉ do teacher/prior |
| SHuBERT [S15] | Masked pseudo-labels từ clustering, không gloss ở SSL | Hand/face visual features và body landmarks; cluster targets theo stream | Baseline discrete cực kỳ liên quan; crop quality và feature encoder góp phần đáng kể |
| MS-MAE [S16] | Skeleton SSL theo abstract chính thức | Multistream masked autoencoder, ASL pretraining và cross-language transfer | Cần lấy full methods/code trước khi chọn implementation tái lập |
| SignMAE [S09] | Masked pretraining rồi downstream supervision | Mask theo moving hands/arms; RGB và pose heatmap streams | Không claim first hand/motion mask; giữ partial active-hand context là lựa chọn đã có |
| SignSeek [S10] | Pretraining có gloss labels | Per-articulator adaptive graphs, Conformer, contrastive và masked alignment/prediction | Rất gần graph + articulator mask + latent; so ở regime supervision đúng |
| SSL-SLR [S29] | Negative-free SSL | Multi-view learning, boundary-aware augmentation | Không giới hạn literature ở JEPA; phải kiểm tra augmentation làm mất sign boundaries |

**[LIT-M] SignRep.** Paper tái tạo các sign priors thay vì pixels, gồm geometry tay/cơ thể, góc khớp và quan hệ giữa các phần; student là RGB. Vì vậy đề xuất mới “RGB dự đoán geometry/relations” cần chứng minh thêm được điều gì so với target-prior reconstruction. Việc thay target decoder bằng latent head chưa đủ chứng minh lợi ích. [S14]

**[LIT-M] SHuBERT.** Paper dùng stream riêng cho hands/face và body; các crop visual encoders được thích nghi trước khi clustering. Appendix A cho kết quả masking phụ thuộc metric: channel/time/random lần lượt có BLEU 2.6/2.3/2.2 và BLEURT 29.9/31.2/31.4 ở thí nghiệm được báo cáo. Không được chuyển thành claim “cross-channel luôn tốt” hoặc “random luôn tốt”. Paper cũng đã đánh giá phonological attributes, và báo overlap một số benchmark với dữ liệu YouTube. [S15]

**[LIT-M] SignSeek.** Bản arXiv tháng 09/2026 dùng pose của body, face và hai tay; graph riêng từng articulator, fusion bằng Conformer, chọn articulator theo saliency, đồng thời có contrastive objectives và masked latent prediction. Clean target được stop-gradient; không nên tự gọi là một EMA teacher độc lập. Pretraining có cùng-gloss supervision qua signer, nên hiệu quả không xác lập được cho regime thuần không nhãn. [S10]

**[LIT-M] SignMAE.** Bản 2026 đã có mask dựa trên hands/arms/motion và các stream video/pose. Đây là bằng chứng trực tiếp chống novelty chung của “mask đúng vùng cơ thể”. Phải so partial masks và temporal support thay vì chỉ đặt một tên mới cho hand-aware masking. [S09]

### 4.3. Reconstruction, clusters và latent: không có thứ hạng phổ quát

| Target | Ưu điểm có cơ sở | Điều dễ thất bại | Phép kiểm tra cần có |
|---|---|---|---|
| Pixels/HOG | Có anchor quan sát cụ thể; không phụ thuộc semantic teacher | Nuisance/background chiếm loss; pixel error không phản ánh linguistic error | Same encoder/data; foreground-balanced và raw video MAE controls |
| Coordinates/angles | Có ý nghĩa hình học, dễ kiểm tra | Detector errors, ambiguities, khác hệ tọa độ | Stratify theo quality; direct-prior reconstruction baseline |
| Motion/residuals | Tập trung biến đổi, ít background | Amplify jitter; không giữ shape đứng yên | Temporal support/fps matched; static handshape probes |
| Cluster pseudo-labels | Giảm một phần noise, objective hữu hạn | Quantization có thể nhập các contrast gần; clusters có thể mã hóa style | Same upstream features, cluster purity/attribute probes |
| EMA latents | Contextual target; linh hoạt bỏ nuisance | Teacher/student cùng học bỏ thông tin khó dự đoán | Teacher probes, target rank/content controls, raw/cluster matched targets |
| Frozen teacher latents | Target cố định, có thể audit | Chất lượng và provenance teacher là trần; thêm supervision/cost | Teacher-alone và simple distillation controls |

VideoMAE/MaskFeat cho cơ sở của masked observation targets; MAMP và AMR cho thấy motion/reconstruction trên skeleton vẫn là đối thủ mạnh. AMR 2026 còn báo lợi ích nhỏ từ all-patch reconstruction trong setting large patches của paper. Điều đó không chứng minh raw target luôn tốt, nhưng bác bỏ cách loại raw-coordinate losses chỉ vì “JEPA semantic hơn”. [S24–S27]

### 4.4. Graph, temporal encoders và linguistic priors

ST-GCN/CTR-GCN là các cơ sở để dùng topology và learned adjacency; MotionBERT cho một encoder motion 2D/3D mạnh nhưng nhiệm vụ gốc không chuyên về fingers/non-manuals. EGNN là lựa chọn equivariant có cơ sở toán học khi tọa độ thực sự cùng không gian; không phải lý do mặc định loại bỏ orientation hoặc phản chiếu trong ký hiệu. [S28, S30–S32]

Phải phân biệt:

| Loại graph | Node/edge đại diện | Vai trò | Điều không được suy diễn |
|---|---|---|---|
| Skeleton graph | Khớp, bones và temporal links | Local geometry/motion bias | Không tự thành linguistic grammar |
| Interaction graph | Tay, mặt, thân và relations quan sát được | Relative location, co-motion, contact candidates | Distance nhỏ trong 2D không chứng minh contact 3D |
| Linguistic knowledge graph | Lexical/phonological/semantic entities | External structured supervision | Không còn là thuần SSL nếu dùng lexical labels/annotations |

ASLKG đã sử dụng linguistic knowledge graphs; Sem-Lex và ASL-LEX hỗ trợ phonological evaluation. Đây là nguồn cho target/evaluation có nhãn trong regime riêng, không nên trộn vào pretraining không nhãn mà vẫn claim hoàn toàn self-supervised. [S39–S41]

### 4.5. Những phương pháp khác regime

SSVP-SLT có SSL trên video rồi paired translation; SignCLIP dùng video/pose–text pairs; UniSign và Two-Stream phục vụ sign understanding với supervised/paired objectives. VL-JEPA có text-space targets. Chúng là đối chứng downstream hoặc một regime có ngôn ngữ supervision, không phải bằng chứng trực tiếp cho unlabeled sign-video JEPA. [S42–S46]

Nếu dùng captions, glosses, dictionary descriptions hoặc pretrained text teacher trong bất cứ giai đoạn nào, phải ghi rõ. “Không sử dụng gloss loss” chưa đồng nghĩa “không sử dụng linguistic supervision”. Pose detectors và pretrained vision teachers cũng có external supervision/provenance riêng, dù không phải gloss labels.

## 5. Bài toán representation thực sự cần giải

### 5.1. Định nghĩa đích học, tránh ép model học tên module

Một representation hữu ích phải cho phép truy xuất các contrast cần cho nhiệm vụ, đồng thời bền vững vừa đủ với signer/camera. Không yêu cầu mọi video token độc lập mã hóa mọi yếu tố.

| Yếu tố cần giữ | Tín hiệu có thể quan sát | Không gian phù hợp | Điều phải giữ ngoài canonicalization |
|---|---|---|---|
| Handshape | Finger configuration, texture/occlusion, local pose nếu đáng tin | Hand-local geometry và visual tokens | Hand side, orientation, quality/missingness |
| Orientation | Directed bones, palm appearance; palm normal nếu valid 3D | Image/body-relative direction hoặc registered 3D | Rotation transform; không chỉ distances/angles invariant |
| Location | Wrist/hand so với mặt/thân, signing space | Common image/body frame | Absolute relative height/side; không wrist-center toàn bộ tín hiệu |
| Movement | Path, direction, speed, repetition, holds | Time-indexed common coordinates và local articulation changes | Real timestamps, duration, phase; không tùy tiện reverse/speed-invariant |
| Two-hand relations | Relative position/orientation, synchrony, independent roles | Common coordinates + pair features | Anatomical left/right và active/non-active roles |
| Non-manuals | Mouth, brows, gaze/head/torso | Face visual + head/body geometry | Detail riêng của mặt; không xem face chỉ là signer nuisance |
| Long context | Transitions, coarticulation, spatial reference, boundary | Dense temporal features, downstream context model | Không cắt window rồi claim hiểu cả utterance |

Đây là các yêu cầu operational cho thiết kế. Không có cam kết rằng một skeleton 2D biểu diễn đủ mọi cấu trúc ngôn ngữ, hay các yếu tố trên độc lập. “Handshape head” chỉ là head dự đoán một target; phải có probe để biết target/head giữ đúng contrast.

### 5.2. Vấn đề bất khả đoán khi che articulator

Gọi (C) là context student, (Z_M) là target latent của articulator bị che. Với squared loss và một teacher cố định:

$$
\arg\min_h\;\mathbb E[\|h(C)-Z_M\|_2^2]
=\mathbb E[Z_M\mid C].
$$

Sai số tối thiểu chứa conditional variance (\mathbb E[\operatorname{Var}(Z_M\mid C)]). Với L1, nghiệm tối ưu theo từng coordinate là conditional median, không phải luôn conditional mean.

**[INFER]** Nếu cùng một context còn cho phép nhiều handshapes/non-manuals hợp lệ, point prediction không xác định được chi tiết riêng của target. Teacher EMA có thể cùng student đổi representation để giảm những thành phần khó đoán. Vì vậy một mask “rất khó” có thể làm loss lớn, làm predictor ghi nhớ priors hoặc làm encoder bỏ đặc trưng; chưa chắc làm representation mạnh.

Điều này dẫn đến hai nhóm objective:

1. **Mục tiêu giữ chi tiết quan sát:** student nhìn thấy chính vùng chứa thông tin; học local representation, directed geometry hoặc teacher embedding của vùng đó.
2. **Mục tiêu suy luận từ context:** student che một phần; chỉ dự đoán target ở độ hạt phù hợp. Partial finger masks/short gaps khác toàn bộ face hoặc dominant hand cho cả clip.

Không cần tuyên bố cross-articulator prediction vô ích. Nó có thể học synchrony/coordination hoặc robustness. Nhưng whole-hand masks phải là ablation có mục đích, không phải mặc định để ép học intrinsic handshape.

### 5.3. Ba đại lượng không được gộp thành một confidence

| Đại lượng | Ý nghĩa | Cách ước lượng khả thi | Không thay thế được |
|---|---|---|---|
| Observation validity (a) | Detector có thực sự trả observation hay là missing? | API outputs, tracking continuity, missing flags | Không đo correctness |
| Measurement quality (q) | Observation có đáng tin không? | Manual sample audit, calibrated quality model hoặc available visibility scores | Không đo context predictability |
| Conditional ambiguity (u(C)) | Context có đủ để đoán target không? | Frozen-teacher prediction pilot, residual stratification, controlled masks | Không biết từ detector confidence riêng |

Confidence của detector không thể chứng minh handshape che hoàn toàn là suy ra được. Ngược lại, target khó đoán nhưng measurement tốt vẫn đáng giữ ở một objective quan sát. Cần log ba thứ riêng, không thêm một learned uncertainty head được phép hạ tất cả weights về zero.

### 5.4. Các giả thuyết đã chốt

| ID | Giả thuyết **[DESIGN]** | Kết quả sẽ bác bỏ hoặc thu hẹp |
|---|---|---|
| H1 | Regional target balancing tốt hơn generic adaptation ở fine-grained transfer | Gain mất khi cùng mask/data/teacher; chỉ signer probe tăng |
| H2 | Partial local/short-gap prediction giữ private details tốt hơn whole-articulator prediction | Whole-part control tốt ngang hoặc hơn trên lexical-disjoint phonology và retrieval |
| H3 | Geometry ở nhiều không gian hợp lệ tốt hơn một canonical hand stream | Giữ orientation/location branch không cải thiện contrasts tương ứng |
| H4 | C cải thiện RGB từ teacher B đủ tốt | Simple visible distillation cùng teacher đạt tương đương; masked geometry không thêm lợi ích |
| H5 | Dependency-safe masking giảm shortcut và tăng transfer | Unsafe path có loss thấp hơn nhưng transfer ngang; hoặc safe version không cải thiện sau equal compute |

H5 có thể cho ra một đóng góp đánh giá/implementation hữu ích ngay cả khi không tăng SOTA; không đổi thành claim accuracy breakthrough nếu chỉ có bằng chứng leakage.

## 6. Hợp đồng dữ liệu, preprocessing và chống leakage

### 6.1. Những giả định triển khai cần ghi ngay từ đầu

**[DESIGN]** Pilot mặc định dùng ASL vì có dữ liệu và nguồn phonological evaluation phù hợp. Kết quả không tự chuyển sang ngôn ngữ ký hiệu Việt Nam, BSL, DGS hoặc CSL. Nếu mục tiêu thực tế là một ngôn ngữ khác, giữ kiến trúc nhưng thay inventory, dữ liệu và người đánh giá ngôn ngữ; không ánh xạ lexical identities chỉ bằng từ tiếng Anh.

Pretraining thuần không nhãn được định nghĩa ở đây là không dùng gloss, transcript, phonological labels hoặc class labels trong objective/sampling. Detector và foundation checkpoints là external pretrained components, phải báo cáo riêng. Sampling “cùng gloss khác signer” thuộc regime có nhãn.

### 6.2. Data manifest tối thiểu

Mỗi sample cần có:

```yaml
sample_id: stable_unique_id
source_video_id: original_recording_id
source_collection: corpus_version
language: verified_or_unknown
signer_id: verified_or_unknown
session_id: verified_or_unknown
start_sec: 12.4
end_sec: 13.68
frame_timestamps_sec: [12.40, 12.44, 12.48]  # ví dụ rút gọn
source_width: 1920
source_height: 1080
source_fps: 25.0
decode_transform_id: versioned_transform
split: pretrain_train
content_hash: hash_of_available_bytes
duplicate_group_id: source_or_near_duplicate_group
burned_caption_status: checked_or_unknown
detector_version: optional_version_and_checkpoint_hash
pose_coordinate_system: image_xy_or_registered_3d
quality_audit_stratum: clean_occluded_fast_small_hands_or_unknown
```

Timestamp arrays được lưu đầy đủ trong dữ liệu thật; sample YAML chỉ minh họa schema. Unknown là một trạng thái, không âm thầm thay bằng signer khác hoặc confidence 1.0 có nghĩa chắc chắn.

Các clip cùng recording/session, overlapping intervals và near-duplicates không được tách tùy tiện qua train/eval. Dùng original video IDs, time ranges và perceptual deduplication; text title hoặc checksum byte riêng lẻ không bắt được re-encode. Foundation pretraining có thể không công bố đủ IDs: ghi **unresolved contamination risk**, không tuyên bố “clean tuyệt đối” chỉ vì pretrain corpus mới đã được lọc.

### 6.3. RGB preprocessing

1. Decode theo timestamp, giữ source aspect ratio và thực hiện sampling theo giây. Pilot đặt effective fps = min(source fps,25), không upsample nguồn chậm bằng lặp/interpolation frames. T cố định do đó có physical duration khác trên nguồn chậm; mọi methods trong causal comparison nhận cùng timestamps và support. Lưu effective fps, frame-valid flags và khoảng thời gian thật.
2. Xác định vùng ký hiệu đủ chứa đầu, hai tay và thân trên. Pilot dùng crop cố định theo source/clip hoặc toàn frame phù hợp. Nếu dùng detector tạo crop ở inference, phải tính detector vào pipeline và latency.
3. Resize giữ tỷ lệ rồi pad/letterbox đến 384×384; loại padding khỏi loss. Crop/resize phải khớp timestamp/landmarks bằng transform đã lưu. Nếu checkpoint chuẩn dùng một preprocessing khác, baseline chính dùng chuẩn đó; letterbox là một biến riêng, không thay im lặng.
4. Chuẩn hóa RGB bằng đúng thông số checkpoint. Không đoán mean/std hoặc positional interpolation. Chụp một batch và kiểm tra shape/range/decoded order trước train.
5. RGB-only student không nhận tọa độ, confidence hoặc bounding box của target như một feature. Pose chỉ được dùng cho sampling/loss ở các biến có công bố.
6. Cố định teacher/student spatial transform trong core JEPA view để target token thẳng hàng. Color jitter nhẹ chỉ ở student là ablation. Nếu spatial crops khác nhau, bắt buộc có map support chính xác; không so cùng token index của hai crop khác nhau.

Burned subtitles, dictionary labels và presenter layout có thể thành shortcuts. Kiểm tra OCR/caption regions; blur hoặc loại clip nếu chữ chồng lên tay/mặt. Không blur một vùng rộng làm mất non-manuals rồi gọi đó là de-identification trung tính. SSVP-SLT đã chỉ ra nhu cầu xử lý caption shortcuts trong video sign pretraining. [S42]

### 6.4. Từ landmarks normalized sang geometry đúng

Với detector trả (x,y) normalized riêng theo width/height, trước tiên đổi về pixels:

$$
p_{tj}^{img}=(W_t x_{tj},\;H_t y_{tj}).
$$

Sau đó mới tính Euclidean directions/distances hoặc chuẩn hóa body scale. Trừ normalized x/y rồi coi hai trục cùng đơn vị sẽ sai khi (W\ne H).

**[LIT-D]** MediaPipe hand image-z có quy ước khác global camera depth; hand-world origin nằm tại hand geometric center, pose-world origin tại mid-hips. Handedness score và tracking threshold không phải per-joint calibrated confidence. Do đó output interface phải phản ánh đúng API thực tế. [S35, S36]

Hợp đồng coordinate:

| Loại tọa độ | Được dùng cho | Không được dùng trực tiếp cho |
|---|---|---|
| Common image xy đã undo crop | Global wrist/body/face relations, trajectories 2D | True 3D contact/depth/palm normal |
| Hand-local world xyz | Local handshape/orientation theo hệ riêng, với uncertainty | Hand–body global displacement nếu chưa registration |
| Body-world xyz | Body pose trong origin của body detector | Ghép với independent hand-world như cùng camera frame |
| Registered global 3D | Global relations và 3D motion | Bất kỳ claim nào nếu registration/calibration chưa đo lỗi |

How2Sign có subset 3D chi tiết, không phải mọi video trong toàn corpus đều có ground-truth 3D. Có thể dùng subset này kiểm tra detector/registration, nhưng không xem 3D pseudo-label là ground truth trên phần còn lại. [S50]

### 6.5. Validity, quality và missingness

Interface cho mỗi joint là (p_{tj},a_{tj},q_{tj},r_{tj}): coordinate, observed-valid flag, available quality score, và reason code. Reason code phân biệt natural missing, synthetic mask, padding, rejected measurement và interpolated observation.

- Nếu SDK không có joint confidence, để `quality_known=false`; loss weight mặc định cho valid observation là một trọng số tính toán, không phải xác suất correctness.
- Interpolation cần giữ provenance và không tính như observation độc lập. Không dùng interpolation đi qua synthetic target gaps cho student.
- Gaps dài/occlusion có thể chính là lúc hands interact. Loại toàn bộ low-quality cases sẽ đổi distribution và có thể loại bỏ các ký hiệu quan trọng. Báo coverage theo signer, tốc độ, kích thước tay, occlusion và sign class khi có nhãn đánh giá.
- Detector comparison nên bắt đầu bằng hai ứng viên trên một tập kiểm tra stratified nhỏ, có annotations/inspection phù hợp. Không mặc định estimator mới hoặc 3D estimator thắng mọi dataset.

Một nghiên cứu pose-to-SLT 2026 đánh giá nhiều estimators trong cùng downstream pipeline cho thấy smoothness không bảo đảm chất lượng translation. Nghiên cứu eyebrow tracking cũng nêu khó khăn dưới head movement. Đây là lý do audit measurement thay vì dùng confidence/smoothing như chứng chỉ linguistic fidelity. [S37, S38]

### 6.6. Dependency-safe masking

Gọi (V\) là tập raw observations còn nhìn thấy và (D(u)\) là tập raw observations dùng để tính feature (u\). Student chỉ được nhận:

$$
u\text{ hợp lệ cho context}\quad\Longleftrightarrow\quad D(u)\subseteq V.
$$

Nếu điều kiện không thỏa, bỏ feature hoặc thay bằng missing token kèm trạng thái. Không điền bằng feature clean teacher. Đây là quy tắc cho student; full teacher được phép thấy full input trong offline masked-learning task.

| Feature/phép toán | Dependency dễ bị quên | Cách làm hợp lệ |
|---|---|---|
| Bone (p_j-p_k) | Cả hai endpoints | Nếu một endpoint hidden, bone cũng hidden |
| Velocity (p_t-p_{t-1}) | Hai timestamps và transform/scale tương ứng | Mask cả derivative nếu một endpoint hidden; dùng Δt thật |
| Acceleration | Ít nhất ba observations | Không đưa qua target gap; chỉ optional sau kiểm tra detector jitter |
| Palm basis từ wrist/index-MCP/pinky-MCP | Các joints dựng basis | Không tạo basis cho student từ hidden MCPs; dùng safe frame hoặc mark unavailable |
| Clip mean/scale | Toàn tập dùng để estimate | Estimate từ visible body anchors; teacher dùng cùng anchor parameters cho global targets |
| Smoothing/tracker | Neighbor frames hoặc tương lai | Student chạy trên visible observations; causal task dùng past-only version |
| GCN/temporal conv | Neighborhood/receptive field | Mask raw inputs trước; thông tin target không được trộn vào rồi mới drop embedding |
| Adaptive adjacency | Inputs dùng để tạo edge weights | Xây từ context-only; không full-pose graph trước masking |
| ROI pooling/query | Center/extent do detector cung cấp | Không cấp hidden target location cho một head đang được đánh giá là dự đoán location |
| Duplicated wrist | Body wrist và hand-root cùng vị trí | Tie masks hoặc nêu rõ wrist location được condition như context |
| Image patch stem | Pixels trong receptive field | Disjoint tubelet stem có thể patchify rồi drop; overlapping stem cần account support/guard |

Ví dụ: che wrist tại (t\) nhưng giữ clean velocity (p_t-p_{t-1}\) và (p_{t-1}\) là một đường khôi phục wrist gần như trực tiếp. Che các joints ngón tay nhưng dùng palm canonicalization tính từ chính những joints bị che là một leakage khác. Lỗi này không biến mất khi thêm `stop_gradient`.

### 6.7. Mask metadata cũng chứa thông tin

Mask được tạo từ vị trí tay hoặc motion của full input có thể tiết lộ **vị trí, extent hoặc thời điểm có động tác**. Predictor cần biết target index, nhưng không được xem index như thông tin trung tính trong mọi bài toán.

Core A được phép dùng pose để ưu tiên vùng loss/masking và phải công bố điều kiện này. C không được cấp wrist center clean để rồi tuyên bố đã học dự đoán wrist location từ context. Các controls bắt buộc:

1. Model chỉ nhận positional/mask metadata, không nhận RGB/pose content.
2. Giữ mask của clip A nhưng thay context bằng clip B có source/length tương tự.
3. Pose-guided masks được jitter hoặc thay bằng masks có distribution tương đương nhưng không dùng pose của chính clip.
4. So region weighting trên cùng masks với thay masks trên cùng loss weights.

Nếu metadata-only giải thích phần lớn gain, claim phải chuyển sang “external localization prior” hoặc thiết kế lại sampling; không gọi đó là learned sign semantics.

### 6.8. Offline learning, anticipation và online inference là ba hợp đồng khác

Full bidirectional teacher hợp lệ cho offline SSL trên một clip. Nó không chứng minh causal future prediction. Với anticipation:

- Context RGB, pose, trackers, normalization và mask metadata đều chỉ dùng lịch sử.
- Teacher có thể tạo future targets, nhưng future position/boxes không được cấp cho student queries.
- Train/eval timestamps và prediction horizon được ghi theo giây; không dùng centered smoothing đi qua thời điểm cắt.

Các proposal chính trong tài liệu là **offline representation learning**. Online/anticipation là một nghiên cứu mở rộng với pipeline causal riêng.

## 7. Phương án A: RGB-JEPA với mục tiêu quan sát và dự đoán tách biệt

### 7.1. Mục đích và phạm vi

**[DESIGN]** Student dùng RGB, teacher video là EMA của encoder, pose chỉ hỗ trợ region sampling/loss ở training. Mục tiêu là cải thiện dense/local representation mà không buộc model suy ra toàn bộ chi tiết của một articulator bị mất. Không dùng gloss/text targets.

Đây là hướng để kiểm tra H1/H2 với ít confounders nhất. Nó chưa có target độc lập bảo đảm handshape; vì vậy teacher audit là điều kiện bắt buộc. Nếu foundation/EMA teacher không giữ finger contrasts, không giả định masking sẽ tự sửa: chuyển sang visible local supervision của C hoặc giữ nguyên baseline.

### 7.2. Tensor contract và encoder

| Module | Đầu vào | Đầu ra | Quy tắc |
|---|---|---|---|
| Decoder/preprocessor | Source frames + timestamps | (X\in\mathbb R^{B\times3\times T\times384\times384}\), valid map | Whole signing area; transform đồng bộ |
| Local tubelet embedding | X, disjoint kernel/stride (2,16,16) | (E\in\mathbb R^{B\times N\times768}\) với checkpoint B phù hợp | Không trộn qua tubelet trước drop; xác nhận dim từ model thực |
| Mask sampler | Grid, valid map, optional train-only regions | C, M, G, region memberships | C=context, M=targets, G=guard; disjoint |
| Student encoder | E[C], positions | (H_C^{(\ell)}\in\mathbb R^{B\times\lvert C\rvert\times768}\) | V-JEPA 2.1 B initialization, đúng positional implementation |
| Predictor | Projected context + mask queries + positions | $\hat Z_M^{(\ell)}$, optional $\hat Z_C^{(\ell)}$ | New predictor; không load một predictor lớn không khớp |
| EMA teacher | Full valid E | (Z_N^{(\ell)}\), stop-gradient | Teacher tokenizer/backbone cùng architecture; update sau optimizer |
| Region loss | Predictions, teacher targets, quality/regions | Scalar losses | Không truyền region coordinates vào encoder |

Với T=16, (N=8\times24\times24=4608\). Với T=32, N=9216. Teacher vẫn encode full N; không được dự toán cost chỉ từ số context tokens của student.

**Cấu hình pilot chốt:** T=16, 25 fps nếu source hỗ trợ, 384 px, ViT-B dimension kiểm tra từ checkpoint; predictor 4 Transformer blocks, width 384, 6 heads, output projection về teacher dimension. Đây là predictor **mới [DESIGN]**, không phải official distilled predictor. Dùng đúng positional mechanism của adapter; không mặc định absolute embeddings nếu checkpoint dùng RoPE.

T=32 là cấu hình xác nhận tiếp theo sau memory profiling. T=16@25 chỉ quan sát khoảng 0.64 giây; nó không bao trọn mọi isolated sign. Downstream dùng fixed multi-window aggregation; pretraining lấy nhiều start times. Không gọi window ngắn là whole-sign understanding.

### 7.3. Hai background recipes, tách khỏi novelty masking

| Recipe | Target layers | Loss | Vai trò |
|---|---|---|---|
| **A-R** | Final encoder layer | Masked latent L1 | Generic adaptation control tối giản |
| **A-D** | Blocks 3, 6, 9, 12 của B 12-block, nếu implementation khớp | Masked L1 + visible-context L1, equal layer averaging | Dense adaptation control, lấy cảm hứng từ 2.1 nhưng recipe riêng |

A-D concatenate/project các context features nhiều layers về predictor width; cùng một predictor trunk có bốn output heads. Teacher trả bốn normalized targets. Không chỉ thêm bốn losses trên cùng final-layer feature rồi gọi deep supervision.

Pilot so A-R và A-D bằng **generic mask giống nhau**, kiểm tra global và local probes. Chọn background recipe bằng validation đã định trước và khóa nó cho thí nghiệm mask chính. Nếu compute giới hạn, dùng A-R cho nhóm causal mask ablations và báo A-D như mở rộng, không ghép kết quả giữa hai recipe.

### 7.4. Mask và region target balancing

Định nghĩa (N_v\) là valid token count; hidden budget:

$$
\rho_h=\frac{|M|+|G|}{N_v}=0.80,\qquad
\lvert C\rvert=N_v-\lvert M\rvert-\lvert G\rvert.
$$

G là hidden guard không tính target loss. `target_fraction` vì vậy không đồng nghĩa hidden fraction. Nếu geometry guards làm |G| tăng, sampler phải giảm |M| hoặc điều chỉnh chọn regions để vẫn đúng budget; không âm thầm giảm context.

Đề xuất **partial-region sampler**:

**Event policy pilot chốt:** chọn đều một reliable region trong LH/RH/face/body đang hiện diện; 50% event là partial spatial block phủ 25–50% region patches qua 1–3 bins, 50% là whole-region temporal gap 1–2 bins với same-region anchors quanh gap. Core không bias bằng motion/dominant-hand score. Region quá nhỏ để chia block, ví dụ dưới 4 patches, chuyển sang short-gap hoặc generic fallback; báo fallback rate. Đây là distributions **[DESIGN]**, phải khớp trong đối chứng mask/loss.

1. Tạo regions LH, RH, face, upper-body và other trên token grid từ train-only landmarks; vùng không đáng tin chuyển về generic policy.
2. Chọn partial hand tube hoặc short temporal gap; không mặc định che tay trọn clip. Giữ ít nhất 25% support của mỗi reliable hand region trên toàn window, và temporal anchors hai phía của local gap nếu có.
3. Hand gap core dài 1–3 tubelet bins, tương đương 0.08–0.24 giây ở 25 fps/tubelet 2. Face core dùng partial short gaps; whole-face masks là ablation.
4. Guard khởi đầu: một spatial patch quanh vùng target và một temporal bin ở boundary khi đủ budget. Đây là conservative guard để thử shortcut từ adjacency, không là bảo đảm independence; báo actual guard size và visible support.
5. Điền các hidden slots còn lại bằng generic block masks để đủ 80%. Nếu constraints không khả thi, giảm kích thước vùng target hoặc fallback generic; không vi phạm constraints mà vẫn ghi sampler thành công.

Global generic masked target count và structured target count phải ghi riêng. Masked loss của A-D/generic luôn có counterpart cùng predictor/layers/context weights.

Với target weights, dùng exclusive ownership cho một token theo ưu tiên hands → face → body → other để không đếm một wrist nhiều lần. Tính mean loss trong từng nhóm rồi average các nhóm tồn tại; không để background area quyết định toàn gradient. Pilot khởi đầu equal group weights; so với uniform-per-token trên **cùng mask**. Group vắng không lấy zero làm loss tốt; ghi coverage.

### 7.5. Objective chốt

Gọi $z_i^{(\ell)}=\operatorname{LN}_{no\text{-}affine}(f_{\bar\theta}^{(\ell)}(X)_i)$. Teacher target được detach; predictor output không buộc normalize cùng cách trừ khi đó là ablation có công bố.

$$
e_i^{(\ell)}=\frac{1}{d_\ell}\|\hat z_i^{(\ell)}-\operatorname{sg}(z_i^{(\ell)})\|_1.
$$

Với R là các target groups hiện diện:

$$
L_M=\frac{1}{|\mathcal L|}\sum_{\ell\in\mathcal L}
\sum_{r\in R}\alpha_r
\frac{\sum_{i\in M_r}w_i e_i^{(\ell)}}{\sum_{i\in M_r}w_i+\epsilon},
\qquad\sum_r\alpha_r=1.
$$

Trong A, w chỉ loại invalid/padding hoặc dùng quality đã công bố; không học w để giảm objective. Low detector quality không làm mất generic RGB loss ở vùng đó, chỉ làm mất region prior không đáng tin.

$$
L_C=\frac{1}{|\mathcal L|}\sum_\ell
\frac{\sum_{i\in C}v_i e_i^{(\ell)}}{\sum_{i\in C}v_i+\epsilon},
\qquad L_A=L_M+\lambda_C L_C.
$$

Pilot A-R: $\lambda_C=0$. Pilot A-D: $\lambda_C=0.05$, v=1; thử 0.1 và distance-weighted v như ablations nếu cần. Layer weights equal. Các trị số này là **[DESIGN]**, không sao chép như official 2.1 recipe. Không thêm geometry, contrastive, signer adversary hoặc SIGReg vào core A.

L_C là mục tiêu trên vùng student quan sát, nhưng teacher feature vẫn có contextual receptive field. Nó không phải ground-truth handshape supervision. Phải đo liệu local token thực sự tăng khả năng truy xuất contrasts hay chỉ học layout.

### 7.6. Training stages và inference

| Stage | Làm gì | Điều kiện ra khỏi stage |
|---|---|---|
| A0 | Load exact foundation checkpoint; extract fixed clip/per-frame embeddings; frozen evaluation | Loader/preprocessing đúng, có kết quả baseline và quality strata |
| A1 | Generic sign-domain adaptation A-R; A-D nếu đủ budget | Không collapse, validation transfer ổn; biết context/global trade-off |
| A2 | Region weighting only, same generic masks | Gain không chỉ từ background downweight hoặc thêm prior metadata |
| A3 | Partial-region mask only, same uniform loss | Visible-information budget matched; metadata-only không giải thích gain |
| A4 | Weighting + mask; whole-part mask negative/control experiment | H1/H2 được kiểm tra với predeclared metrics, không chỉ training loss |
| A5 | T=32 hoặc input resolution variant | Lợi ích còn sau exposure/compute accounting; không đổi data và teacher đồng thời |

Inference bỏ predictor, EMA teacher, mask sampler và pose extractor dùng cho training-only guidance. Encoder nhận full valid RGB grid. Mean spatial pooling tạo feature mỗi tubelet; fixed temporal mean hoặc fixed multi-window mean tạo clip embedding. Có thể dùng downstream temporal head riêng cho CSLR/SLT. Nếu chọn ROI crop dựa detector tại inference, đó là biến khác và phải công bố.

### 7.7. Khi nào A không đủ?

- Hand tokens quá nhỏ: thử hợp lý crop toàn vùng ký hiệu/tăng resolution, nhưng kiểm tra không cắt active hand. Nếu vẫn không có đủ pixels, không diễn giải latent prediction như phục hồi thông tin chắc chắn.
- Teacher có clip classification tốt nhưng handshape probe yếu: A-R/A-D chưa có anchor bảo đảm chi tiết. Học visible geometry/visual targets ở C là lựa chọn có lý hơn whole-hand imputation mạnh hơn.
- Domain adaptation làm probe giảm so foundation: kiểm tra LR, dense loss và forgetting; early-stop bằng validation, không cứ scale thêm training.
- Gains chỉ trên same-signer split: representation có thể đang khai thác style/layout; chưa đạt mục tiêu cross-signer.

## 8. Phương án B: graph-JEPA trong các không gian geometry tương thích

### 8.1. Mục đích và topology cụ thể

**[DESIGN]** B là một pose-based model độc lập, không chỉ một “geometry teacher tùy chọn”. Nó cần có giá trị inference/representation riêng trước khi được xuất làm teacher cho C.

Pilot 2D dùng J=67 nodes:

| Nhóm | Số nodes | Schema khởi đầu |
|---|---:|---|
| Upper body | 7 | Nose, L/R shoulder, L/R elbow, L/R wrist; nếu MediaPipe pose: indices 0,11,12,13,14,15,16 |
| Left hand | 21 | Wrist + 4 joints mỗi finger theo hand detector schema |
| Right hand | 21 | Như left hand; anatomical side được giữ |
| Face subset | 18 | Bảng IDs dưới đây; không đủ cho mọi mouth/gaze grammar |

Face subset khởi đầu theo MediaPipe mesh schema tương thích: brow candidates `[70,63,105,66,107,336,296,334,293,300]`; mouth `[61,291,13,14]`; eye corners `[33,133,362,263]`. IDs là lựa chọn **[DESIGN]**, cần kiểm tra mapping bằng overlay ở SDK version thực, đặc biệt mirror/left-right. Không gọi 18 points này là đầy đủ non-manual representation; không có iris nên không claim đo gaze.

Edges cố định gồm anatomical hand bones, shoulder–elbow–wrist chains, facial local groups và temporal same-joint links. Interaction edges chỉ nối part-level representations hoặc safe observed anchors. Body wrist và hand root có duplicate geometry: khi thử whole-hand/location masks phải mask đồng bộ hoặc khai báo rõ body wrist còn cung cấp location.

### 8.2. Coordinate branches và features

Không nối tất cả coordinates thành một vector rồi để model “tự hiểu”. Dùng ba nhóm có semantics và validity riêng:

**(i) Body-relative common-image branch.** Với shoulder midpoint c và shoulder distance s ước lượng từ **visible valid body anchors**:

$$
\tilde p^{body}_{tj}=\frac{p^{img}_{tj}-c}{\max(s,s_{min})}.
$$

c/s khởi đầu là robust clip estimate để tránh motion do changing normalization; nếu shoulders thiếu, dùng fixed image-scale fallback, lưu flag và không claim body-relative. Teacher dùng cùng c/s từ context cho global targets. Không dùng target hand extent để estimate s. Retain body tilt/head displacement; không rotation-normalize toàn bộ video mặc định.

**(ii) Hand-local shape branch.** Root-relative hand vectors $p_j-p_{wrist}$, chuẩn hóa bằng safe scale. Pilot dùng body shoulder scale hoặc visible hand bone scale có dependency rõ. Giữ directed vectors, projected angles và per-feature validity. Một target local-hand descriptor của full teacher có thể dùng full hand; student context descriptor không được dùng hidden root/bones để chuẩn hóa.

Nếu dùng palm-aligned 3D shape trong bản mở rộng: dựng basis từ ba non-collinear reliable joints trong local 3D, kiểm tra degeneracy/chirality; lưu basis/quaternion vào orientation branch. Không loại rotation rồi mất orientation. Không tính basis 3D từ 2D chỉ bằng thêm z=0.

**(iii) Location, motion và relations branch.** Wrist-to-body/face vectors và inter-wrist vectors trong cùng image frame; velocity $(p_t-p_{t-1})/\Delta t$; optional local articulation differences. Inter-hand relations khởi đầu là directed displacement, distance và temporal synchrony; projected distance gần không được gọi là physical contact. Acceleration không nằm trong core vì nhạy detector jitter.

Tín hiệu node token khởi đầu: normalized xy, root-relative xy khi valid, bone-directed xy khi valid, velocity xy khi valid, observation flags và available q. Mỗi nhóm feature được normalize bằng statistics của training set, không evaluation set. Learned node/part/hand-side embeddings bổ sung identity của joint; real-time encoding nhận timestamp/Δt. Numeric zeros phải có flags để phân biệt valid root zero và missing observation.

**Feature schema pilot cụ thể:** main node vector có 19 slots mỗi frame: common/body xy (2), hand-root-relative xy (2), parent-bone xy (2), velocity xy (2), validity flags cho bốn nhóm (4), observed flag (1), available q (1), quality-known flag (1), reason-code one-hot bốn trạng thái (4). Core reason states: measured, natural-missing/rejected, synthetic-mask, padding; core không interpolate landmarks. Unknown q được encode zero + quality-known=false, tách khỏi default loss weight của valid targets. Nhóm không áp dụng có value zero và validity false; nếu có nhiều reason cùng lúc, dùng priority convention đã version. Hai frames concatenate thành 38 slots rồi project lên 256. Face/body không giả tạo hand-root features. Parent-bone schema theo anatomical chains, không dùng adjacent joint tùy ý.

Hand chains: `[0,1,2,3,4]`, `[0,5,6,7,8]`, `[0,9,10,11,12]`, `[0,13,14,15,16]`, `[0,17,18,19,20]`. Body chains: nose–shoulder–elbow–wrist ở mỗi bên. Face local edges: hai brow chains, mỗi cặp eye corners, và vòng mouth corners–upper/lower lip; cross-part context đi qua fusion. Mọi edge/parent dùng IDs từ schema đã kiểm tra bằng overlay.

Để phục vụ **observed targets của C**, B có một local articulation branch riêng: hand/face root-relative xy và directed parent-bone xy, các validity/quality fields, chỉ trong một 2-frame bin. Hand scale lấy từ reliable local palm bones của bin; face center/scale từ eye anchors của bin. Không dùng velocity, clip-wide statistics hoặc cross-part context trong branch này. Khi dependencies thiếu, feature/target tương ứng unavailable. Branch giữ image-axis directions, không rotation-normalize tùy tiện.

### 8.3. Encoder và tensor flow

Pilot T=64 ở 25 fps: khoảng 2.56 giây. Temporal grouping 2 frames không overlap; T'=32. Token count $N_p=T'J=2144$; width d=256, 6 blocks, 8 heads. Với group có một frame synthetic-hidden, bỏ feature hoặc đánh dấu partial support; không clean-average target vào visible group.

Local articulation branch dùng 2 spatial graph blocks width256; core hands share weights cùng anatomical-side embedding, không mirror input; face có weights riêng. Nó trả node/part features trước temporal/cross-part fusion. Các local features được project rồi cộng vào main node embeddings; vì vậy loss chính và observable anchors đều train được branch này. Các controls B phải giữ branch/capacity này như nhau hoặc báo thành một architectural factor riêng.

Sau local branch, mỗi main block:

1. Spatial graph message passing trong từng time bin, với anatomical fixed edges và optional learned residual adjacency từ context.
2. Temporal attention per joint/part qua valid time bins, có real-time positional information.
3. Cross-part attention giữa part summary tokens; part summaries chỉ pool visible nodes trong student.
4. Residual MLP và normalization.

Core spatial layer dùng normalized anatomical adjacency, self projection và neighbor projection; learned residual adjacency khởi đầu tắt. Temporal self-attention per joint dùng 8 heads; cross-part self-attention trên bốn part summaries rồi broadcast update về observed nodes. Local branch không có các bước temporal/cross-part này. Pooling local teacher là mean observed nodes với fixed topology/side embeddings; đó là articulation latent có thể probe, không một disentanglement guarantee.

Như vậy full attention trên T'J tokens không phải mặc định. Masked nodes được exclude khỏi context aggregation hoặc represented bằng learned missing token không chứa coordinates. Receptive field chỉ được tiếp cận observations còn nhìn thấy. Predictor 4 blocks width 256 trả latents cho masked queries; baseline centered-CE theo objective của S-JEPA được xây riêng. Adapt input/backbone sang signing phải ghi là objective-matched adaptation, không là exact reproduction toàn paper.

| Dòng tensor | Shape | Teacher/student |
|---|---|---|
| Raw common coordinates | B×T×67×2 | Student che raw trước derived features; teacher full |
| Observation/quality flags | B×T×67×k | Không cấp “confidence của hidden coordinate” nếu nó tạo oracle quality shortcut |
| Feature tokenizer | B×32×67×256 | Context-only derivation cho student |
| Dense joint features | B×32×67×256 | Teacher full; student available nodes |
| Part features | B×32×4×256 | Body, face, LH, RH; pool theo validity |
| Local observed hand/face features | B×32×3×256 | Trước global temporal/cross-part mixing; support chỉ 2 frames của part |
| Predictor outputs | B×\lvert M\rvert×256 | Query là time/joint IDs, không target xy |
| Observable descriptor heads | B×selected-visible-support×descriptor_dim | Chỉ predict từ support đủ quan sát |

Transformer width/block numbers là **[DESIGN]**, phải profile và so với encoder controls cùng capacity. Không cho rằng “graph nhẹ hơn” khi adaptive full adjacency hoặc cross-part attention được triển khai không sparse.

### 8.4. Teacher không được tạo tùy ý

Teacher B là EMA copy của backbone, tokenizer và mọi projection tạo latent target. Nếu target projector learnable, projector cũng thuộc EMA; không để một online projector tự thu target về constant trong khi backbone teacher frozen.

Teacher latents được probe ở checkpoints 0, pilot và final. Bổ sung **observable geometry anchors** để có một áp lực giữ thông tin cụ thể. Các anchors không dùng gloss:

| Anchor | Target | Context support bắt buộc |
|---|---|---|
| Shape | Root-relative finger vectors và projected flexion descriptor | Corresponding observed joints/root/scale |
| Orientation | Directed MCP/bone vectors so với image/body axes | Endpoints và axes quan sát được |
| Location | Wrist-to-shoulder/face anchor vectors | Wrist và body/face anchor observed |
| Motion | Time-normalized displacement trong một short span | Hai endpoints observed, cùng coordinate transform |
| Relation | Inter-wrist vector hoặc safe hand-to-body vector | Hai phần/anchors observed |

Shape/orientation anchors được predict từ local articulation branch để nó giữ details độc lập với toàn clip. Location/motion/relation anchors lấy từ main context features với support hợp lệ. Optional face anchor gồm projected mouth width/opening và brow-to-eye offsets trong current bin; chỉ là measurement proxies sau quality audit, không grammar labels. Core teacher export không gắn tên “disentangled shape/orientation” cho một part embedding chưa được probe.

Đây là hybrid objective, không được mô tả là pure latent-only JEPA. Predicting descriptors từ visible support có nguy cơ dễ; mục tiêu của nó là giữ thông tin và ổn định teacher, không giả vờ là một hard predictive task. Kiểm tra nó có làm encoder chỉ sao chép coordinates bằng downstream transfer và direct-coordinate baseline.

### 8.5. Loss, masking và curriculum

$$
L_B=L_{joint\text{-}latent}(M)
+\lambda_{obs}L_{observable\text{-}geometry}(C)
+\lambda_{part}L_{part\text{-}latent}(M_{part}).
$$

Core B: $\lambda_{part}=0$, $\lambda_{obs}=0.1$ sau khi mean-normalize từng descriptor group. Main latent loss là target-LN L1; observable descriptors dùng Smooth L1 trên normalized values, directed-angle target dùng sin/cos hoặc cosine có sign convention rõ. Weight từng nhóm không theo số joints để hand descriptors không lấn át location/motion.

Pilot raw hidden ratio 50%, phân biệt raw hidden observations và token hidden count sau grouping. Sampler thực hiện một trong các event:

- 50% event: partial finger-chain/joint blocks, che 10–40% joints của một tay qua 1–3 bins; giữ root/MCP anchors nếu objective cần biết location/orientation.
- 30% event: short temporal gaps trên một part, có same-part context quanh gap.
- 20% event: generic joint-time blocks.

Sau event, fill random safe masks để đạt budget 50%; derived-feature invalidation có thể tăng hidden feature count, phải log riêng. Không dùng target-quality threshold để chỉ giữ clean easy fingers rồi bỏ natural occlusions khỏi downstream.

Whole-hand/body/face masking là một **separate ablation**, không chiếm core mixture. Nếu cần part latent, dùng teacher part features đã có probe tốt và chỉ che spans đã kiểm tra predictability; khởi đầu $\lambda_{part}=0.05$. Không thêm full-resolution raw handshape target khi entire hand bị ẩn dài.

### 8.6. Training B và cách xuất teacher cho C

1. **B0 — detector/schema audit:** overlay joints, hand identities, aspect ratio; 2D frame registration và quality coverage. Nếu detector chuyển hand side thường xuyên, sửa tracking trước train.
2. **B1 — direct controls:** frozen coordinate descriptors + temporal head; raw masked reconstruction; centered-CE S-JEPA-objective adaptation với cùng input; generic L1 EMA B không anchors. Không lấy benchmark NTU của paper gốc làm score sign baseline.
3. **B2 — core hybrid:** train joint-latent + observable anchors; checkpoints 2k/10k/50k updates theo ngân sách ở §11. Không đóng băng một teacher ngẫu nhiên rồi gọi output semantic target.
4. **B3 — factor probes:** shape/orientation/location/motion/relations và signer. Freeze encoder; test mỗi target family có thêm thông tin gì ngoài raw coordinates.
5. **B4 — export:** freeze backbone, norm statistics và chosen part projections; ghi extraction transforms/hash. Xuất cả local pre-fusion articulation features và contextual part latents 256-D theo bin; ghi raw dependency support cho từng family, chỉ xuất targets vượt audit.

Nếu geometry teacher không tốt hơn direct descriptors hoặc không giữ contrast đang cần, C dùng direct-prior reconstruction control trước, không giả định teacher đã “semantic hóa” coordinates. Khi fine-tune B cho downstream có nhãn, checkpoint đó thuộc regime teacher supervised và phải tách khỏi B SSL-only.

Inference B gồm detector → coordinate preprocessing → graph encoder → downstream head. Detector cost và failures là một phần kết quả. Face-subset model chỉ có non-manual proxies; thêm face RGB stream là một phương án khác có compute/input khác, không công bố B hiện tại đủ cho SLT.

### 8.7. Các mở rộng nên hoãn

- **Registered 3D/EGNN:** chỉ sau khi đo lỗi registration và giữ orientation/chirality targets. Rotation/reflection invariance vô điều kiện có thể xóa contrast.
- **Path signatures:** cần giữ time channel nếu duration/speed là thông tin; không xem reparameterization invariance luôn có lợi. Literature skeleton đã có Path-JEPA, và bằng chứng đọc trong lượt này chưa đủ để chọn recipe. [S34]
- **Acceleration/flow:** chỉ sau quality audit cho nhanh/chậm và occlusion; motion magnitude cao có thể là noise.
- **Discrete linguistic graph:** dùng như supervised auxiliary hoặc evaluation priors riêng; không trộn vào unlabeled baseline.

## 9. Phương án C: RGB student học từ geometry teacher đã kiểm chứng

### 9.1. Điều kiện cần và mô hình causal của lợi ích

**[DESIGN]** C giữ RGB encoder/predictor/EMA của A, thêm một frozen geometry teacher từ B. Teacher không nhìn gloss/text và không được fine-tune bằng evaluation labels. Mục tiêu là transfer một số structure đã kiểm chứng sang RGB encoder, không gọi mọi coordinate embedding là semantic target.

Trước khi chạy C phải trả lời:

1. B teacher có giữ shape/orientation/location/relations tương ứng không, trên signer và lexical identities held out?
2. Teacher có thêm thông tin hữu ích so với direct normalized descriptors không?
3. RGB student có nhìn thấy đủ support/resolution để học target đó không?
4. Teacher có được tạo từ cùng pretraining split, không benchmark-test videos/annotations?
5. Latent target tốt hơn direct-prior reconstruction hay chỉ thêm một learned encoding?

Nếu chưa trả lời được, C chỉ là “distillation từ một teacher chưa được đánh giá”, không phải một phương pháp có cơ sở ưu tiên hơn A/B.

### 9.2. Luồng dữ liệu và alignment

| Nhánh | Đầu vào train | Đầu ra | Vai trò inference |
|---|---|---|---|
| RGB student | Masked X, positions | Dense visible token features | Giữ encoder; downstream head dùng full RGB |
| RGB EMA teacher | Full same X | Video latent targets | Bỏ |
| Frozen geometry teacher | Common-frame P, validity; cùng physical clip | Part/joint latents 256-D, optional descriptor predictions | Bỏ trong RGB-only deployment |
| Visible geometry heads | Student visible features, part IDs | Geometry latents/descriptors tại visible support | Bỏ nếu chỉ cần general representation |
| Masked geometry predictor | Context features + time/part query IDs | Coarse geometry latent của short gaps | Bỏ |

RGB tubelet và pose temporal group đều 2 source frames ở pilot. Nếu FPS khác, alignment dựa timestamps và support intervals, không dựa frame index sau resampling. Geometry teacher phải re-encode **cùng physical clip**; không lấy latent contextual từ một long clip đã cache rồi cắt ra vì latent đó có thể mang ngoài-window information. Variable-length inference của B phải được probe ở T=16/32 trước C; nếu không tốt, đổi target sang local descriptor hoặc chọn matched window, không âm thầm dùng teacher context dài hơn.

Không cần geometry teacher là frozen pretrained “có sẵn”. B0–B4 quy định cách train/audit/export. Frozen targets có thể cache offline sau khi chốt toàn bộ extraction/preprocessing, nhưng hash phải bao gồm clip support, normalization, topology và teacher checkpoint.

### 9.3. Tách visible distillation khỏi masked inference

**C-observed — core được ưu tiên.** Dùng local pre-fusion articulation features từ B, có dependency chỉ trong current 2-frame part bin. Không dùng một contextual part latent có receptive field toàn clip rồi gọi nó là observed local target. Với part r và time bin t có đủ visible RGB support, dùng student features tương ứng để dự đoán local articulation embedding:

$$
L_{G,obs}=\sum_{r\in\mathcal R}\beta_r
\frac{\sum_t b_{tr}q_{tr}\,
\ell(D_r(h^{visible}_{tr}),\operatorname{sg}(g_r(P)_{tr}))}
{\sum_t b_{tr}q_{tr}+\epsilon}.
$$

Trong đó b=1 chỉ khi **toàn bộ raw support của target family**, gồm scale/anchor dependencies, nằm trong RGB region giữ lại của bin, target measurement valid và family đã được audit. Core C sampler reserve một số complete visible hand bins rồi fill masks ở vùng khác để giữ total hidden80%; partial hand bins không nhận observed whole-hand loss. `quality_known=false` có policy rõ như §6; q không được học để làm objective biến mất. Normalize target theo training statistics cố định hoặc target LN, thống nhất giữa train/probe.

Feature extraction cho từng nhóm:

- Shape/orientation: ROI-weighted pool các **visible** hand patches; pose coordinates là training-only alignment supervision, không cấp cho encoder. Không pool masked token predictions rồi gọi là observed support.
- Face visual/non-manual proxy: chỉ dùng visible face patches có đủ resolution; teacher B không cung cấp ground-truth mouth/gaze grammar.
- Location/relations: dùng global visible feature pool hoặc fixed spatial-grid queries; target core là direct descriptors có short-bin support rõ, hoặc local teacher head đã audit. Không dùng full-context B latent như observed target; không dùng clean wrist-coordinate queries cho mục tiêu wrist location.
- Motion: visible span có đủ timestamps và endpoints; teacher/student cùng support. Motion của teacher nằm ngoài visible support không thuộc observed loss.

ROI selection tự mang location prior. Vì vậy shape/orientation distillation không được dùng để claim student đã tự localization. Local ROI probes là diagnostic có privileged alignment; primary RGB-only evaluation không nhận ROI metadata.

**C-masked — mở rộng sau audit.** Dùng contextual part latent của frozen B cho coarse prediction trên short gaps; nó thuộc predictive objective, khác local observed target. Queries chỉ có part/time IDs và fixed image grid nếu cần. Target-shape/private detail không được buộc chính xác khi entire part bị ẩn dài. Chỉ thêm nhóm target có predictable residuals tốt hơn simple conditional baseline trên validation. So no-mask, partial-mask và whole-part-mask versions.

$$
L_C^{total}=L_A+\lambda_{G,obs}L_{G,obs}
+\lambda_{G,mask}L_{G,mask}.
$$

Core: $\lambda_{G,obs}=0.1$, $\lambda_{G,mask}=0$. Mở rộng: $\lambda_{G,mask}=0.05$ cho nhóm đã qua kiểm tra. Mỗi family được mean-normalize trước cộng. Các weights là **[DESIGN]** và được kiểm tra bằng ratio của gradient norms, không so trực tiếp hai loss scalars có scale khác.

### 9.4. Predictability audit có thể thực hiện thế nào?

Không đo conditional uncertainty của tất cả dữ liệu bằng một con số tuyệt đối. Dùng một pilot hữu hạn:

1. Freeze B teacher; tạo một tập validation train-domain tách recording/signer nếu metadata cho phép, không dùng downstream test.
2. Chọn target family và mask modes; giữ target teacher/features như nhau.
3. Fit các conditional baselines nhỏ từ allowed context: temporal interpolation, constant/mean target, context-only MLP hoặc small predictor. Interpolation chỉ là baseline từ visible endpoints, không đưa vào training context như clean feature.
4. So teacher-space error theo gap length, target family, observed quality, same-hand anchors, motion và source. Thêm linear attribute probes khi có label validation được phép.
5. Chỉ chọn masked objectives có error giảm khi thêm meaningful context và không gần bằng metadata-only. Ngưỡng quyết định được khóa từ pilot, không chọn theo downstream test gain.

Residual thấp không tự chứng minh semantics: target có thể quá constant. Phải kiểm tra content variance/rank và attribute retention đồng thời. Residual cao không khiến target vô dụng: chuyển target về observed loss thay vì loại mọi khó khăn khỏi representation.

### 9.5. Các controls bắt buộc cho C

| Control | Giữ giống C | Khác gì | Trả lời câu hỏi |
|---|---|---|---|
| Teacher-alone | Same B checkpoint/extraction | Không RGB student | Teacher có mang thông tin cần học? |
| Direct-prior decoder | RGB/data/priors | Raw normalized geometry thay latent | Latent encoding có thêm giá trị? |
| Simple visible distillation | Same teacher và alignment | Không masked geometry objective | Gain có thật sự cần masked cross-modal prediction? |
| A + visible distillation | Same video JEPA và B | $\lambda_{G,mask}=0$ | C-masked có thêm gì? |
| Shuffled teacher targets | Teacher distribution/compute | Sai video nhưng matched source/length | Student học meaningful correspondence hay prior/style? |
| Random frozen encoder teacher | Target dims và head capacity | Teacher chưa được train | Lợi ích có phụ thuộc teacher quality? |
| Same-feature clusters | Same upstream B/features | K-means targets thay continuous latents | Continuous target tốt hơn discretization ở đâu? |

Shuffled/random teachers là diagnostic, không phải baseline có triển vọng. Cũng không cần chạy tất cả trên full scale: pilot đủ để loại cơ chế sai. Teacher-matched controls là điều kiện quan trọng nhất để không gọi extra supervision là JEPA-specific gain.

### 9.6. Training stages, inference và lựa chọn visual teacher

- C0: khóa checkpoint A background và frozen B; audit short-window teacher targets.
- C1: thêm visible geometry heads với warmup 10% pilot updates; video JEPA không đổi.
- C2: chọn một target family mạnh nhất theo audit, không thêm shape/motion/relations cùng lượt; so teacher-matched controls.
- C3: thêm short-gap masked geometry nếu C1/C2 có lợi và predictability đạt yêu cầu.
- C4: scale window/data hoặc backbone sau khi kết quả còn trên frozen/lexical-disjoint/signer-independent evaluation.

Inference RGB-only giống A. Training-only pose, ROI selection và teacher branches bị bỏ. Nếu downstream cần part queries có trainable attention, ghi rõ là downstream head khác mean linear probe.

Optional visual teacher cho hands/face có thể dùng DINO-style crop encoder đã được audit/adapt trên training-only crops, như kinh nghiệm SHuBERT. Đây là **một factor riêng** với extra data/crop cost; không mặc định thêm vào C. High-resolution teacher không thể đảm bảo student low-resolution phân biệt được cùng contrast. Muốn dùng, kiểm tra teacher-at-student-resolution và teacher-at-high-resolution, báo external signal rõ ràng.

## 10. Masking, augmentation và curriculum chốt cho từng phương án

### 10.1. Chọn mask theo câu hỏi, không theo một tỷ lệ trộn tùy ý

| Mask | Mục đích hợp lệ | Core hay ablation? | Rủi ro và constraint |
|---|---|---|---|
| Partial finger/hand region | Local structure completion | Core A/B sau generic control | Giữ same-part anchors; patch resolution đủ |
| Short same-part temporal gap | Continuity, local dynamics | Core A/B | Real Δt; không cho derivative/smoothing đi qua hidden observations |
| Long temporal span | Long dynamics/phase | Ablation | Window có thể không chứa enough context; teacher semantic audit |
| Whole hand | Cross-hand coordination/robustness | Ablation | Không coi other hand xác định private handshape |
| Whole face | Cross-articulator shared structure | Ablation | Mouth/brows có independent information; observed face loss vẫn cần |
| Hand-to-body relation | Location/interaction structure | Target family ở B/C | Same coordinate frame; không query clean target coordinates |
| Motion-guided | Allocate capacity cho dynamic segments | Ablation | Motion can be noise; quiet holds vẫn có lexical contrasts |
| Generic/random | Control, diversity, fallback | Bắt buộc | Không chọn làm universal weak baseline; tune với equal budget |

Không triển khai ngay mixture sáu mask tasks như bản gốc. Những mode đó khác mục tiêu và độ khó, nên phải biết một mode đóng góp gì trước khi trộn. Sau ablation, mixture cuối phải ghi event probability, hidden/target count, span distributions, guard count và visible articulator coverage.

### 10.2. Motion saliency có uniform floor

Nếu thử motion-guided mask, estimate từ reliable common-frame trajectories và real Δt, loại camera/source motion khi có thể. Clip/robust-normalize speed để outliers không độc chiếm sample probability:

$$
\pi_i=\eta\,\frac{1}{N_v}+(1-\eta)\,
\frac{s_i}{\sum_j s_j+\epsilon},\qquad\eta=0.5\;\text{ở pilot}.
$$

Đây là **[DESIGN]**. Saliency high không có nghĩa phonologically important. So velocity-based, detector-confidence-aware và uniform sampling trên cùng budget. Attention norm/saliency từ clean teacher cũng là target-dependent metadata; cần các controls §6.7.

### 10.3. Augmentation có semantic contract

| Transform | Policy core | Vì sao/điều kiện |
|---|---|---|
| Modest color/brightness | Nhẹ; ablation student-only | Tránh thay texture/mouth visibility quá mạnh |
| Resize/translation | Đồng bộ geometry và teacher/student | Không cắt tay/face, giữ location relations trong crop |
| Aggressive random crop | Không core | Có thể biến sign thành partial/unobserved example |
| Temporal jitter | Nhỏ trong physical window | Không cắt onset/offset tùy tiện |
| Speed change | Chỉ evaluation/ablation có giới hạn | Duration/repetition có thể mang thông tin |
| Frame reversal | Diagnostic, không positive augmentation core | Motion direction/order có thể phân biệt sign |
| Horizontal flip | Không mặc định positive pair | Phải swap anatomy/handedness, kiểm tra ngôn ngữ và spatial reference |
| 3D rotation | Không vô điều kiện | Camera equivariance khác linguistic orientation invariance |
| Hand/face blur | Robustness diagnostic | Có thể xóa đúng target cần giữ |
| Pose noise/dropout | B ablation theo error model | Noise phù hợp detector thực, không arbitrary geometric distortions |

Teacher/student content phải cùng sign support. Hai random temporal crops của một utterance dài có thể chứa hai ký hiệu khác nhau; không tự coi là positive pair tương đương. Không dùng caption equality hoặc English gloss equality để tạo positives across signed languages mà không kiểm tra lexical equivalence.

### 10.4. Curriculum dựa validation, không dựa “càng khó càng tốt”

1. **Ổn định:** generic masks, đúng preprocessing, teacher/student asymmetry hoạt động.
2. **Giữ chi tiết:** partial masks và observable objectives; kiểm tra local probes.
3. **Tăng thời gian bị thiếu:** thêm short gaps; theo dõi residual và representation transfer.
4. **Cross-articulator:** chỉ thử whole-part mode sau; nếu private-detail probes giảm, giữ mode này cho robustness branch hoặc loại bỏ.

Không cần tăng mask ratio theo thời gian nếu ratio hiện tại đã thiếu informative context. Một curriculum tốt phải được so với fixed-policy control cùng updates/exposure; không chọn curriculum sau khi thấy test score.

## 11. Training recipes, pseudocode và kiểm tra implementation

### 11.1. Hyperparameters khởi đầu có thể thực hiện

Toàn bộ bảng sau là **[DESIGN]**. Đây là một recipe pilot được chốt để tránh lựa chọn tùy tiện, không phải benchmark đã tối ưu và không phải config tương thích trực tiếp với mọi official repo.

| Hạng mục | A | B | C |
|---|---|---|---|
| Backbone | Official distilled V-JEPA 2.1 B init | d256: 2 local spatial +6 main blocks | Same A checkpoint/background recipe |
| Input core | 16 frames, 384 px, native timing/target 25 fps | 64 frames, J67, image xy | Same RGB as A; matched support geometry |
| Temporal token group | 2 frames | 2 frames | Đồng bộ hai nhánh |
| Optimizer | AdamW | AdamW | AdamW |
| Encoder LR peak | 1e-4 | 3e-4 từ scratch | 1e-4 hoặc giảm theo adaptation pilot |
| Predictor/new heads LR | 3e-4 | 3e-4 | 3e-4 |
| Betas / weight decay | (0.9,0.95) / 0.05 | (0.9,0.95) / 0.05 | Như A |
| No decay | Norms/biases theo implementation | Norms/biases | Như A |
| LR schedule | 10% warmup, cosine decay về 1% peak | Như A | Geometry loss warmup đồng thời |
| Gradient clipping | Global norm 1.0, sau unscale | Như A | Như A |
| Precision | BF16 nếu hỗ trợ; FP32 loss/norm diagnostics | Như A | Như A |
| Global batch pilot | 64 clips/update nếu khả thi | 64 clips/update | Như matched A control |
| Microbatch | Profile thực, accumulation để đạt global batch | Profile thực | Có teacher cache nếu đúng support |
| EMA m | 0.996 → 0.9999 theo cosine updates | Như A | RGB teacher như A; geometry frozen |
| Hidden ratio | 0.80 valid RGB tokens | 0.50 raw joint-time support | Same RGB ratio; aux support constraints |
| Loss core | Mask L1; A-D context 0.05 | Joint L1 + observed anchors 0.1 | A + observed geometry 0.1 |
| Geometry masked loss | Không | Optional part latent 0.05 | 0 core; 0.05 after audit |
| Pilot length | 2k stability, 10k comparison; 50k confirm nếu có tín hiệu | Như A | Stage-wise, cùng exposure controls |

Không scale LR tuyến tính theo batch một cách máy móc; tìm hẹp quanh 0.5×/1×/2× peak trên train-domain validation, cùng search budget cho các variants. Nếu global batch 64 không khả thi, chốt batch nhỏ hơn và so tất cả methods ở setting đó; accumulation không thay thế được batch-dependent regularization hoàn toàn.

10k updates ×64 là 640k **clip views**, không phải 640k unique clips hoặc một số giờ signing độc lập. Repeated/overlapping samples phải được tính. Khi dùng hai masks cùng video, báo video exposures và student views riêng.

### 11.2. EMA, normalization và anti-collapse

$$
\bar\theta\leftarrow m_s\bar\theta+(1-m_s)\theta,
\quad
m_s=m_{end}-(m_{end}-m_0)\frac{1+\cos(\pi s/S)}2.
$$

EMA update đúng một lần sau **successful optimizer step**, không sau từng microbatch. Bỏ update nếu overflow/non-finite làm optimizer step bị skip. Teacher eval mode, không nhận gradient; rules cho buffers cần rõ. Tránh BatchNorm trong new heads ở pilot batch nhỏ; nếu có, define synchronization/teacher buffer handling. Initialize EMA từ student đã load checkpoint, không từ random pre-load weights.

Không mặc định thêm VICReg/SIGReg vào mọi recipe. Trước hết kiểm tra:

- Teacher và student content features có nonzero across-video variance, không chỉ variation theo token positions.
- Effective rank trên clip features và per-part features; covariance rank tối đa là min(B−1,d) với centered B samples, nên B<d không có full-rank estimate.
- Nearest neighbors theo sign vs signer/source; part features có đáp ứng khi content thay đổi không.
- Metadata-only backbone/probe có đạt gần representation thật không.
- Predictor loss giảm có đi kèm frozen transfer tăng không.

Nếu cần regularization, so một centered-CE target variant, một VICReg-style auxiliary hoặc một LeJEPA-style objective trong nhóm riêng. Không nói “EMA + SIGReg bảo đảm giữ ngôn ngữ”. Theorem/anti-collapse regularizer không chứng minh teacher giữ handshape hay mọi nuisance đã được loại.

### 11.3. YAML chốt cho một experiment A

Config minh họa **interface của implementation mới**, không phải file nạp trực tiếp vào official repository. `checkpoint_sha256` phải được điền bằng artifact thực trước run; đây là provenance field, không là hyperparameter còn bỏ ngỏ.

```yaml
experiment: A_partial_region_dense_pilot
status: proposed_not_run
seed: 17
supervision:
  gloss: false
  text: false
  phonology: false
  train_only_pose_guidance: true
input:
  frames: 16
  target_fps: 25
  upsample: false
  preserve_timestamps: true
  resolution: 384
  preprocessing: exact_checkpoint_baseline_then_controlled_variant
  tubelet: 2
  patch: 16
encoder:
  initialization: official_vjepa_2_1_distilled_vit_b
  checkpoint_sha256: REQUIRED_AT_RUN
  feature_dim: 768              # assert against loaded model
  target_layers: [3, 6, 9, 12]
  adaptation_recipe: new_ema_dense_not_official_reproduction
predictor:
  depth: 4
  width: 384
  heads: 6
  output_heads: 4
mask:
  hidden_fraction: 0.80
  budget_unit: valid_tubelet_tokens
  target_policy: partial_region_plus_generic_fill
  target_guard_disjoint: true
  spatial_guard_patches: 1
  temporal_guard_bins: 1
  min_hand_visible_support: 0.25
  hand_gap_bins: [1, 3]
  fallback: generic_if_constraints_infeasible
loss:
  masked_latent: 1.0
  visible_context: 0.05
  geometry: 0.0
  contrastive: 0.0
  target_normalization: layer_norm_without_affine
  target_group_weights: equal_over_present_groups
optimizer:
  name: adamw
  encoder_lr: 0.0001
  new_modules_lr: 0.0003
  weight_decay: 0.05
  betas: [0.9, 0.95]
  global_batch: 64
  updates: 10000
  warmup_fraction: 0.10
  final_lr_fraction: 0.01
  grad_clip_norm: 1.0
teacher:
  type: ema
  start_momentum: 0.996
  end_momentum: 0.9999
  update_after_successful_optimizer_step: true
eval:
  primary: frozen_fixed_mean_linear_macro_class_accuracy
  secondary: retrieval_and_lexical_disjoint_phonology
  test_used_for_selection: false
```

Đối chứng của experiment này là **same A-D config**, thay `target_policy` bằng generic và `target_group_weights` bằng uniform. Sau đó thay từng biến riêng để không gộp hai nguyên nhân.

### 11.4. Training loop cho A/C

```python
# Pseudocode kiến trúc; chưa phải code đã chạy.
for optimizer_step in range(total_updates):
    optimizer.zero_grad(set_to_none=True)
    any_valid = False

    for batch in accumulation_microbatches():
        X, time, valid, train_regions, P = prepare_same_support(batch)
        C, M, G, weights = sample_and_validate_masks(valid, train_regions)
        assert_disjoint(C, M, G)

        # Local, non-overlapping tubelet stem; no global mixing here.
        tokens = student.local_tokenize(X)
        context = student.encode(select(tokens, C), positions=C)

        with no_grad():
            targets = video_teacher.full_encode(X)
            targets = normalize_target_layers(targets)
            # C only; frozen B, exact same time window/transforms.
            geometry_targets = frozen_geometry(P, time) if use_C else None

        pred = predictor(context, query_positions=M, visible_positions=C)
        loss = weighted_mask_L1(pred, targets, M, weights)
        if use_dense_context:
            loss += lambda_context * context_target_L1(pred, targets, C)

        if use_C:
            observed_support = intersect_visible_and_reliable_support(C, P)
            loss += lambda_geo_obs * visible_geometry_distill(
                context, geometry_targets, observed_support
            )
            if audited_masked_geometry_enabled:
                loss += lambda_geo_mask * masked_geometry_predict(
                    context, geometry_targets, safe_part_time_queries(M)
                )

        # Valid-count weighted reduction across workers; not means of means
        # when coverage differs. Accumulation weighting is defined globally.
        loss = globally_normalized_loss(loss)
        backward_with_precision_policy(loss)
        any_valid |= has_valid_targets(M)

    if any_valid and gradients_finite():
        unscale_if_needed()
        clip_gradients()
        optimizer.step()
        scheduler.step()
        update_ema_after_step(video_teacher, student)
    else:
        record_skipped_step_without_ema_or_schedule_advance()
```

Predictor phải trả cả C outputs nếu context loss được bật; dòng `context_target_L1` không thể dùng masked-only outputs. `safe_part_time_queries` không mang clean target xy. `globally_normalized_loss` là yêu cầu distributed reduction: gradients cần khớp global valid counts, không chỉ average loss scalar sau khi backward. Cách chia accumulation loss phải nhất quán với batch size/coverage.

Teacher forward có thể reuse cho nhiều masks **trong cùng optimizer step với cùng weights, X và target preprocessing/anchors**. EMA features không được cache lâu qua nhiều updates như frozen geometry features. Với B, context-estimated normalization có thể đổi theo mask: chỉ reuse khi anchor parameters giống nhau, hoặc recompute targets. Nếu input augmentation thay đổi, target reuse cũng không còn cùng target.

### 11.5. Student feature pipeline của B

```python
P_full, observed, quality, timestamps = load_measurements(batch)
raw_mask = sample_joint_time_mask(observed)
raw_mask = expand_to_nonoverlapping_temporal_groups(raw_mask, group_size=2)
raw_mask = enforce_budget_and_tied_duplicate_nodes(raw_mask)

P_context = hide_raw_observations(P_full, raw_mask)
visible = observed & ~raw_mask
anchor_params = fit_normalization_from_visible_body(P_context, visible)

F_context, F_valid = derive_features_with_dependencies(
    P_context, visible, timestamps, anchor_params
)
# Missing derivative/bone/basis != clean value. No interpolation over masks.
context_latents = graph_encoder(F_context, F_valid, context_only_adjacency=True)

with no_grad():
    F_teacher = derive_full_teacher_features(
        P_full, observed, timestamps, shared_global_anchors=anchor_params
    )
    target_latents = ema_graph_encoder(F_teacher)

pred = graph_predictor(context_latents, masked_joint_time_ids(raw_mask))
loss = masked_latent_loss(pred, target_latents)
loss += 0.1 * observable_descriptor_loss(
    context_latents, descriptor_targets(P_full, anchor_params), support=F_valid
)
```

Temporal-group expansion diễn ra trước chốt budget, nên không có hidden frame bị clean-average vào một token. (D(u)\subseteq V\) được kiểm tra từng descriptor, không chỉ cả node. Teacher global anchors dùng parameters từ context; teacher local shape targets có own full support và phải gắn target-family contract riêng.

### 11.6. Những kiểm tra implementation thực sự cần thiết

Đây là kiểm tra cho research implementation tương lai, **chưa được chạy** trong lượt nghiên cứu này.

| Kiểm tra | Cách làm | Failure bị phát hiện |
|---|---|---|
| Hidden-value perturbation | Giữ mask/metadata cố định, thay raw target values; student context features không đổi | Leakage qua normalization/derivative/graph |
| Teacher/student gradients | Inspect graph/parameters; target branch không có grad | Teacher optimization ngoài EMA/frozen policy |
| Token support | Overlay raw/tubelet/pose groups; verify disjoint supports | Patch/time off-by-one, partial clean averaging |
| Coordinate registration | Plot known image points và relation vectors trước/sau crop | Aspect ratio/gốc tọa độ sai |
| Missingness | Compare natural missing, synthetic mask, valid zero | Model dùng zero như coordinate có thật |
| Left/right continuity | Check crossings, mirror, duplicate wrists | Hand swapping, chirality mất |
| Mask constraints | Log count/coverage từng sample, kể cả fallback | Hidden budget đúng trung bình nhưng từng part bị mất sạch |
| Finite losses/valid denominators | Test all-invalid geometry samples | NaN hoặc loss “tốt” do không có targets |
| Resume determinism | Restore RNG, sampler, optimizer, EMA và schedule | Runs không thể so sánh/reproduce |
| Context swap/metadata-only | Fixed masks, source-matched content swap | Mask oracle/layout shortcuts |

“Loss giảm trong một batch” chỉ chứng minh một phần wiring hoạt động, không chứng minh learned representation hữu ích.

## 12. Đánh giá representation và ngôn ngữ

### 12.1. Dữ liệu, phiên bản và vai trò

| Nguồn | Vai trò đề xuất | Cần kiểm tra |
|---|---|---|
| ASL Citizen [S48] | Primary isolated-sign frozen/low-shot/retrieval; official signer split | Version, lexical variants, joins với phonological metadata |
| WLASL [S49] | Secondary ISLR/retrieval và source diversity | Available video version; official split không được tự xem là strict signer-disjoint |
| Sem-Lex + ASL-LEX [S40, S41] | Lexical/phonological attribute probes | Exact lexical IDs và variant mapping; unknown/absent label handling |
| How2Sign [S50] | Continuous signing transfer, SLT riêng và quality audit subset | Camera/view split, alignment, 3D subset coverage |
| YouTube-ASL / YT-SL-25 [S51, S52] | Unlabeled sign-domain pretraining sau overlap audit | Caption leakage, removed IDs, accessibility, language metadata |
| BOBSL [S53] | BSL cross-domain/cross-language transfer | Interpreter/TV domain, source episode overlap |
| PHOENIX-2014T [S54] | DGS CSLR/SLT downstream | Weather-domain specificity, official protocol |
| CSL-Daily [S55] | CSL CSLR/SLT downstream | Language-specific labels, signer/protocol verification |

**[LIT-D]** ASL Citizen công bố participant được gán vào một split; version 1.0 có 83,399 videos, 2,731 signs và 52 signers. Counts có thể khác giữa releases, nên manifest phải ghi version. Đây là lý do phù hợp chọn làm primary signer-independent benchmark, không phải bảo đảm không overlap với mọi external foundation data. [S48]

Ưu tiên pretrain tập vừa đủ sạch cho matched comparisons hơn đưa ngay toàn bộ nhiều ngôn ngữ vào model. Nếu train multilingual, report per-language exposure; lợi ích có thể đến từ data scale, lexical overlap hoặc camera diversity. Domain/language mixing là factor riêng.

### 12.2. Ba chế độ đánh giá không được nhập chung

| Chế độ | Encoder | Aggregation/head | Kết luận cho phép |
|---|---|---|---|
| **True linear probe** | Frozen | Fixed mean pooling/fixed concatenation + linear classifier | Information linearly accessible dưới feature definition đó |
| **Frozen attentive probe** | Frozen | Learned temporal/spatial attention + classifier | Information accessible với trainable aggregation, không phải linear separability |
| **Fine-tuning** | Trainable một phần/toàn bộ | Same downstream architecture/budget | Initialization/adaptation utility, không đủ chứng minh frozen representation |

Primary A/C dùng fixed spatial mean per tubelet rồi fixed temporal/multi-window mean. B dùng fixed node/part/time mean hoặc fixed concatenation các part means; chốt feature definition cho mọi B variants. Không fine-tune normalization stats trên test.

Linear classifier fit cross-entropy/logistic regression, validation chọn LR/L2 cùng grid và cùng budget. Report per-instance Top-1/Top-5 và mean per-class accuracy; primary metric là per-class Top-1. Early stopping/probe selection chỉ dùng validation. Nếu chọn intermediate layer hoặc concat layers, ghi rõ và áp cùng lựa chọn/search budget cho controls.

### 12.3. Dictionary retrieval và held-out lexicon

Chốt gallery/query split; query signer không trùng gallery signer khi metadata cho phép. Gallery có reference clip(s) của sign cần tìm, nên đó là retrieval từ reference dictionary, không là “zero-shot không cần reference”. Report R@1/R@5, mAP, per-class và per-signer aggregation.

Thực hiện hai setting:

1. **Known-lexicon retrieval:** các lexical classes có trong representation/probe development, nhưng query videos/signers held out.
2. **Held-out-lexicon retrieval:** lexical IDs chưa xuất hiện trong supervised probe training; gallery chứa reference clips của các lexical IDs đó. Để claim lexically unseen pretraining, còn cần xác minh SSL corpus không chứa chúng—điều thường không thể bảo đảm trên unlabeled web video.

Giữ fixed cosine/L2-normalized embedding baseline. Nếu train metric projection bằng gloss labels, đó là supervised retrieval head, không còn frozen zero-training retrieval. Multiple gallery instances và max/mean aggregation phải chốt vì chúng ảnh hưởng đáng kể rank.

### 12.4. Phonological probes với lexical-disjoint split

Một classifier sign identity có thể học bảng ánh xạ `word → handshape`, nên attribute accuracy trên lexical classes đã thấy không đủ chứng minh sublexical generalization. Thiết kế chính:

1. Join bằng curated lexical IDs và variant mapping, không dùng English gloss string đơn giản. Một lexical entry có thể có nhiều realizations; sample không phù hợp label phải được loại/annotate rõ.
2. Split train/val/test theo lexical ID; phối hợp với signer disjoint nếu đủ samples. Báo ma trận signer×lexical overlap thay vì gắn một tên chung.
3. Fit frozen linear heads cho attributes có metadata đáng tin. Khởi đầu handshape, selected fingers/flexion, major location, path/repetition, nondominant handshape theo inventory thật. Orientation/non-manual grammar cần annotations phù hợp riêng; không tạo labels từ detector rồi xem là linguistic ground truth.
4. Stratify để attribute classes có đủ support; report seen/unseen label categories. Một lexical-heldout class với label category chưa có training không thể xử lý như standard closed-set accuracy.
5. Report macro-F1, balanced accuracy, per-attribute coverage và confusion. Đặt family metric trước; không chọn attribute tăng nhiều nhất sau khi thấy results.

ASL-LEX cung cấp lexical/phonological coding; sử dụng như nguồn **evaluation labels**, không objective pretraining trong pure SSL regime. SHuBERT đã có phonological evaluation bằng fine-tuning; đóng góp cần là protocol và causal evidence tốt hơn, không claim lần đầu đo phonology. [S15, S41]

### 12.5. Linguistic contrast sets và non-manuals

**[DESIGN]** Tạo một benchmark nhỏ, được người dùng ngôn ngữ ký hiệu/chuyên gia xác nhận, cho các contrast shape, orientation, location, motion và two-hand relation. Không đưa tên các cặp cụ thể nếu chưa xác minh thuộc ngôn ngữ/variant đang dùng.

Với mỗi query a, chọn positive p cùng lexical/linguistic target nhưng khác signer, và hard negative n có contrast được annotation. Đo:

$$
\Pr\big[\operatorname{sim}(a,p)>\operatorname{sim}(a,n)\big],
$$

cùng margin distribution và per-family accuracy. Chuyên gia ghi yếu tố khác nhau, variant, độ rõ video và mức agreement. “Khác đúng một yếu tố” là điều cần xác minh trên realizations, không mặc định từ dictionary label.

Non-manual grammar thường phụ thuộc utterance/context; isolated-sign datasets và face landmarks không đủ. Muốn claim giữ mouth/brow grammatical contrasts, cần continuous examples có annotation tương ứng, contextual window và consent phù hợp. Nếu chưa có, báo đây là **coverage gap**, không dùng một face-motion regression score thay bằng grammar accuracy.

Synthetic reverse/flip/blur là diagnostics về sensitivity/robustness, không tự là minimal pairs ngôn ngữ. Sensitivity mạnh với một transform cũng có thể do low-level artifacts, nên phải đối chiếu linguistic annotations.

### 12.6. Few-shot, signer/source leakage và robustness

- Few-shot 1/5/10 labeled examples mỗi class nếu dữ liệu cho phép, cùng sampled examples/seeds giữa methods; report frozen-head và fine-tune riêng. Tránh dùng nhiều clips cùng signer cho cả support/query.
- Signer-ID/source-ID probes trên frozen features với overlap protocol phù hợp. Low signer accuracy không tự tốt; sign utility và rare/dialectal variants vẫn cần giữ. Pose features có thể mang identity, không tự anonymized. [S47]
- Quality strata: small hands, occluded/crossing hands, fast finger movement, face tilt, low illumination, missing joints. Report target-valid coverage cạnh accuracy.
- Counterfactual body-normalization/orientation tests: normalization variant có xóa đúng contrast cần giữ không; compare with no-rotation/no-mirroring controls.
- Temporal diagnostics: time shuffle/reverse và clip-duration variation; xác định model dùng order/motion hay chỉ appearance. Không train invariance với các diagnostics này mặc định.

### 12.7. CSLR/SLT chỉ sau core representation evidence

CSLR: xuất dense time features, dùng cùng temporal head/CTC architecture và labels cho mọi encoder. Report WER, deletion/insertion/substitution và input feature rate. CTC cần đủ steps cho target gloss sequence, kể cả repeated-label blank separation; downsample quá nhiều có thể làm alignment bất khả thi, không chỉ “giảm accuracy”.

SLT: fixed pretrained language decoder, cùng paired training data và tuning budget; report SacreBLEU signature, chrF và semantic/human evaluation phù hợp. BLEURT hoặc LM fluency không thay đánh giá faithfulness/omissions. Encoder ablation/frozen decoder controls giúp tránh kết luận encoder tốt hơn chỉ do language model mạnh hơn.

Đừng pretrain bằng benchmark test videos rồi gọi translation label-free transfer clean chỉ vì không dùng test transcripts. Pure SSL cũng là nhìn thấy input distribution; phải công bố transductive nếu dùng.

### 12.8. Thống kê và model selection

Primary run comparison dùng cùng seeds và dữ liệu. Tối thiểu ba training seeds cho các finalists nếu budget cho phép; mọi pilot một seed được ghi exploratory. Paired bootstrap cluster theo signer/source/session, không xem từng frame/video cùng source là independent. Với ít signers, report per-signer results và uncertainty rộng; không tạo precision giả bằng millions frames.

Report effect sizes, paired confidence intervals và run variation. Nhiều mask/loss variants cần tách exploratory và confirmatory experiments, giữ final test chưa dùng chọn policy. Chốt mức cải thiện có ý nghĩa thực tiễn dựa pilot variance/cost trước confirmation; không đặt một ngưỡng arbitrary rồi hứa paper đạt A*.

## 13. Đối chứng, ablation và diễn giải nhân quả

### 13.1. Matching phải bao gồm thông tin nhìn thấy

| Yếu tố | Cần khớp ở causal ablation | Nếu không khớp, kết luận bị giới hạn |
|---|---|---|
| Data | Same video IDs, time supports, language/source exposure | Không tách method gain khỏi data gain |
| Initialization | Same exact checkpoint/hash | Không tách objective khỏi foundation quality |
| Teachers | Same teacher/extraction khi so targets/loss | Không tách latent mechanism khỏi extra prior |
| Encoder/predictor | Same blocks/dims/query budget nếu không phải factor nghiên cứu | Không tách capacity/decoder changes |
| Context | Same hidden count + articulate coverage/span distributions được báo | Same ratio vẫn có thể giữ nhiều relevant information hơn |
| Target count | Same evaluated queries/validity policy | Nhiều informative labels/weights hơn có thể là nguồn gain |
| Time | Same duration, timestamps và frame rate ở mask/loss ablation | Tăng T có thể chỉ tăng physical context |
| Resolution/crops | Same information content | Hand crop lớn hơn không phải masking effect |
| Compute | Report GPU-hours, measured throughput, teacher/detector preparation | Same updates không phải same cost |
| Model selection | Same validation/search budget | Nhiều tuning hơn có thể thắng |

Không thể luôn khớp đồng thời tất cả giữa RGB và pose architectures. Vì vậy cần **hai loại bảng**: leaderboard transfer theo published checkpoints với provenance khác nhau, và controlled experiments để suy luận nguyên nhân. Không trình bày hai loại như cùng mức bằng chứng.

### 13.2. Ma trận A gọn nhưng đủ nhân quả

Chốt một background recipe A-R hoặc A-D, same random seeds/data sequence. Core 2×2 factorial:

| ID | Mask policy | Loss region weighting | Vai trò |
|---|---|---|---|
| A00 | Không adaptation | Không | Foundation frozen baseline |
| A10 | Generic | Uniform | Domain adaptation baseline |
| A11 | Generic | Part-balanced | Weighting-only effect |
| A12 | Partial-region | Uniform | Masking-only effect |
| A13 | Partial-region | Part-balanced | Combined hypothesis |
| A14 | Whole-articulator | Part-balanced | Private-detail/predictability control |
| A15 | Jittered/source-matched regions | Part-balanced | Pose/mask metadata specificity |

A10–A13 là vòng chính. A14/A15 chỉ confirmation nếu pilot có tín hiệu. Add A-R vs A-D riêng trước/ngoài factorial; không đổi deep supervision cùng lúc với mask rồi gắn gain cho mask.

Với coverage control mạnh hơn: chọn generic masks theo constraint giữ cùng amount hand/face support như structured masks. Nếu A12 vẫn tốt, temporal/spatial arrangement có đóng góp ngoài “để lại nhiều tay hơn”. Nếu gain biến mất, báo đó là allocation-of-visible-information effect, không tự gọi semantic prediction tốt hơn.

### 13.3. Ma trận B

| ID | Context-safe features | Objective | Câu hỏi |
|---|---|---|---|
| B00 | Direct coordinates/descriptors | Frozen features + same downstream head | Có cần learned representation? |
| B10 | Có | Raw masked geometry/motion reconstruction | Observation-target control |
| B11 | Có | Centered distribution CE kiểu S-JEPA | Skeleton predictive baseline đúng family |
| B12 | Có | EMA latent L1 | Continuous latent control |
| B13 | Có | B12 + observable anchors | Anchors giữ thêm contrast nào? |
| B14 | Có | B13, only hand-local branch | Global location/orientation có cần? |
| B15 | Có | B13, remove motion or relation family lần lượt | Tín hiệu nào thực sự góp ích? |
| B16 | Cố tình unsafe chỉ diagnostic | Same B13 | Loss/transfer bị shortcut làm sai ra sao? |

B16 không dùng làm final model hoặc benchmark ngang hàng. Nó là phép thử để phát hiện hướng rò; không cần train full scale nếu hidden-value perturbation đã chứng minh lỗi. Frame-wise GCN với whole-frame masking có support khác joint-wise masking; không kết luận mọi paper dùng GCN-before-mask đều sai nếu mask unit của họ là whole frame.

Graph inductive bias được kiểm tra bằng thay spatial graph với per-joint MLP + same temporal blocks, khớp parameters/cost gần nhất có thể. Learned adjacency chỉ thêm sau fixed anatomical graph baseline; kiểm tra overfitting source/signer.

### 13.4. Ma trận C và các đối thủ external

Core C controls: A13; A13+direct priors; A13+visible frozen-B distillation; A13+visible+masked-B; simple same-B distillation; same-feature clusters. **Mọi control, gồm A13 không geometry loss, dùng cùng C-reserved visible-support masks** để không gán ích lợi từ nhìn thấy nhiều tay hơn cho distillation. Giữ geometry target quality/coverage như nhau. Nếu khác sample exclusions, báo riêng.

Đối thủ external ưu tiên:

- RGB: generic V-JEPA/VideoMAE adaptation; SignRep và SignMAE nếu checkpoint/implementation/pretraining provenance đủ.
- Pose: SignBERT+, MASA, S-JEPA variant; MS-MAE sau full-method access.
- Multistream features: SHuBERT, kiểm soát crop encoder/pretraining khi suy luận target-type contribution.
- Gloss-supervised transfer ceiling/competitor: SignSeek, dùng nhãn đúng regime và không gọi baseline SSL thuần.
- Paired text/SLT: SignCLIP/UniSign/Two-Stream/VL-JEPA ở bảng riêng nếu nhiệm vụ cần.

Không sao chép scores từ papers khác rồi khẳng định superiority ở một split/data version khác. Published numbers chỉ là bối cảnh; controlled comparison cần chạy đúng protocol hoặc ghi rõ không tái lập được.

### 13.5. Cách diễn giải kết quả sau thí nghiệm

| Nếu quan sát được | Diễn giải phù hợp | Claim cần bỏ |
|---|---|---|
| A10≈A13 nhưng cả hai >A00 | Domain adaptation là yếu tố chính | Novel mask superiority |
| A11>A10; A13≈A11 | Balanced supervision/allocation là yếu tố chính | Mask strategy là nguyên nhân |
| A12 gain mất ở coverage-matched generic | Giữ nhiều relevant pixels/context giải thích gain | Latent inference mechanism mạnh hơn |
| C-visible≈C-masked | Visible teacher transfer đủ | Masked cross-modal objective cần thiết |
| Raw priors≈B/C latents | Geometry signal hữu ích nhưng learned latent chưa thêm giá trị | Latent semantic target superiority |
| Cluster target thắng ở rare handshapes | Quantization/target stability có ích ở setting này | Continuous latent universal advantage |
| Fine-tune tốt, frozen/phonology không tốt | Initialization hữu ích cho supervised adaptation | Richer frozen linguistic representation |
| Signer/source probe tăng và cross-signer giảm | Shortcut/style có thể chiếm representation | Signer invariance |
| Whole-part mask cải thiện robustness nhưng giảm handshape | Có trade-off; tách objective/branch hoặc chọn theo use case | Một mask tốt cho mọi linguistic factors |

Không cần một kết quả thắng tất cả để có nghiên cứu tốt. Một cơ chế được kiểm soát, giới hạn rõ và có thí nghiệm bác bỏ đáng tin có giá trị hơn một tổng loss không biết thành phần nào hiệu quả.

## 14. Compute, thứ tự triển khai và điều kiện quyết định

### 14.1. Token/memory accounting

Với tubelet 2 và patch16:

$$
N_v=(T/2)(H/16)(W/16).
$$

| Input | Full teacher tokens | Context tokens xấp xỉ ở 80% hidden | Lưu ý |
|---|---:|---:|---|
| 16×384×384 | 4,608 | 922 | Pilot chốt A |
| 32×384×384 | 9,216 | 1,843 | Teacher/predictor có thể thành bottleneck |
| 64×384×384 | 18,432 | 3,686 | Không mặc định phù hợp GPU hạn chế |
| 32×256×256 | 4,096 | 819 | Resolution khác checkpoint; controlled variant |
| 32×224×224 | 3,136 | 627 | Hand information có thể giảm đáng kể |

Counts context là số xấp xỉ do rounding/valid masks. Full attention có thành phần O(N²d); tăng T gấp đôi làm attention term gấp bốn, không phải toàn training FLOPs luôn gấp bốn. FlashAttention giảm materialized attention memory, không loại quadratic attention arithmetic.

Student 20% context làm riêng student attention term còn khoảng 4% full-token version. Nhưng full teacher và predictor vẫn cần chạy; do đó không thể quảng cáo end-to-end 25× cheaper. EMA teacher không cần lưu backward activations nhưng vẫn cần forward compute/model weights. A-D multi-layer targets/output heads cũng tăng memory.

B factorized spatial/temporal attention có leading interactions gần $T'J^2+JT'^2$, thay vì $(T'J)^2$, cộng graph/cross-part/MLP costs. Đây là complexity estimate, không throughput đo thực. Pose extraction có thể tốn hơn B encoder và phải được tính.

### 14.2. Profiling trước khi hứa ngân sách

Chạy 100–500 updates gồm decode, detector/cache reading, full teacher, student, predictor, backward và optimizer. Log peak memory, sec/update, samples/sec, I/O stalls và valid-target coverage. Từ đó:

$$
\text{GPU-hours}\approx
\frac{\text{sec/update}\times\text{updates}\times\text{GPU count}}{3600}
+\text{teacher/detector preparation cost}.
$$

Không đưa một con số GPU-hours cố định khi chưa có GPU/model implementation thật. Thứ tự giảm cost:

1. Profile T16@384 trước T32; giữ baseline input information nhất quán.
2. BF16, activation checkpointing và optimized attention nếu implementation hỗ trợ.
3. Reuse clean teacher trong cùng update cho nhiều masks, khi target preprocessing giống nhau.
4. Cache frozen B targets đúng support, không cache stale EMA targets.
5. Sample một budget target queries cố định hoặc cross-attention predictor như biến hiệu năng riêng; dùng cùng query budget/capacity cho controls.
6. Chỉ giảm resolution khi đã biết ảnh hưởng hand resolution; report matched physical support khi giảm fps/T.

Teacher high-resolution/crop adaptation/detector preparation được tính cả trong cost của C; inference cost riêng vì các branches bị bỏ. Không xem precomputed priors là “miễn phí”.

### 14.3. Một lộ trình theo gates, không hứa lịch 10–12 tuần vô căn cứ

| Gate | Work package | Evidence cần có để tiếp tục |
|---|---|---|
| G0 — dữ liệu | Versioned manifests, dedup, captions, detector audit | Split/coordinates đúng; biết unresolved contamination và coverage |
| G1 — baseline | A00, direct B00, teacher/feature probes | Pipeline inference/eval reproducible; hình học và hand resolution hợp lý |
| G2 — pilot | 2k stability →10k A10–A13; B10–B13 nếu geometry là mục tiêu | Không collapse/shortcut; signal trên validation, không chỉ pretext loss |
| G3 — confirmation | Finalists 50k hoặc matched exposure, nhiều seeds | Frozen/class retrieval + attribute effects; compute accounting |
| G4 — teacher transfer | C-visible và teacher-matched controls | Teacher quality đủ và extra loss có giá trị ngoài simple distillation |
| G5 — mở rộng | Longer clips/data, cross-language, CSLR/SLT | Core claim vẫn tồn tại; không đổi mọi yếu tố cùng lúc |

Nếu ngân sách chỉ đủ một hướng: A00/A10–A13 và evaluation tốt. Nếu nguồn vào chỉ là pose: chọn B, không cố tái tạo RGB priors không có. Nếu cần inference RGB nhưng có pose chất lượng khi train: C sau teacher audit. Không có GPU/dataset thực tế nên thứ tự này là decision framework, không thời gian hoàn thành đã cam kết.

### 14.4. Deliverables của một implementation thật

- `data_manifest` và split hashes; detector/extraction contracts.
- Experiment configs khóa trước run, model/teacher hashes, masks statistics.
- Checkpoints có optimizer, EMA, RNG/sampler state và normalization statistics.
- Frozen/attentive/fine-tune evaluation scripts với fixed feature definitions.
- Tables báo mean/uncertainty, per-signer/per-attribute coverage, resource usage.
- Failure examples và hidden-value/context-swap diagnostics.

Tài liệu hiện tại cung cấp blueprint cho các deliverables đó; chưa tạo repository/train scripts chạy được và chưa có những artifacts thực nghiệm này.

## 15. Novelty, phản biện reviewer và giới hạn

### 15.1. Contribution có thể bảo vệ nếu có evidence

Một framing hợp lý là **observability-aware predictive representation learning for signing** với ba thành phần được chứng minh riêng:

1. Tách observed articulation preservation khỏi missing-part prediction theo target family/support.
2. Dependency-safe geometry/multimodal masking, kèm evidence về shortcut và representation transfer.
3. Evaluation phân biệt class memorization, sublexical transfer, signer/source và teacher contribution.

**[VERIFY]** Chưa xác lập rằng framing này hoàn toàn mới trên toàn literature. Trước submission cần thêm backward/forward citation search và full-method access cho các nguồn còn thiếu. Protocol engineering hoặc đổi tên loss không tự đạt novelty A*; phải có một cơ chế/insight được thí nghiệm hỗ trợ và đối chứng mạnh.

Không claim:

- Lần đầu JEPA/skeleton/graph cho ký hiệu.
- Lần đầu hand-aware/motion-aware/cross-articulator masks.
- Lần đầu hand/body relations hoặc phonological evaluation.
- EMA hoặc latent loss tự loại signer/style và giữ semantics.
- Pure SSL khi dùng gloss/text/lexical annotations trong objective/sampling.
- 3D linguistic geometry khi thực chất dùng independent local-world estimates hoặc 2D projections.

### 15.2. Những phản biện mạnh nhất và phép trả lời bằng evidence

| Reviewer objection | Trả lời bằng thiết kế/thí nghiệm |
|---|---|
| “Đây chỉ là SignRep với learned target decoder.” | Raw-prior, same-B visible distillation và masked-B comparisons; nếu tương đương thì thu hẹp claim |
| “SignSeek đã mask articulator và predict latent.” | Acknowledge overlap; tách gloss-supervised regime; so support/observability mechanism, không dựa tên JEPA |
| “Structured mask chỉ giữ nhiều tay hơn.” | Coverage-matched generic masks + weighting-only factorial |
| “Teacher đã cung cấp tất cả ích lợi.” | Teacher-alone, same-feature cluster, simple distillation và equal provenance |
| “Geometry canonicalization làm mất orientation/location.” | Separate branches; removal ablations và attribute contrasts |
| “Mask hoặc ROI tiết lộ đáp án.” | Metadata-only, context swap, coordinate/query restrictions |
| “Pose quality quyết định mọi thứ.” | Estimator/quality strata audit; coverage và common-frame validation |
| “Frozen probes chỉ học word-to-phonology mapping.” | Lexical-disjoint attributes, expert contrast sets; labels không tham gia SSL |
| “Đã nhìn test data trong pretraining.” | Source/time/signer overlap manifests; unresolved foundation exposure ghi rõ |
| “Gains nhỏ và tuning nhiều hơn.” | Equal search budget, paired uncertainty, final held-out test và cost curves |

### 15.3. Những giới hạn không thể giải bằng thêm loss

- Occlusion và low-resolution RGB có thể làm nhiều configurations không phân biệt được; latent target không tạo thông tin chắc chắn từ nguồn không đủ.
- Monocular pose không xác định true 3D orientation/contact mọi lúc. Confidence cao không loại systematic bias.
- Body/face movement có thể mang cả identity và thông tin ngôn ngữ. Adversarial removal toàn bộ signer signal có thể xóa dialect/grammar; không thêm signer adversary vào core chưa kiểm chứng.
- Isolated-sign benchmark không đủ cho spatial reference, coarticulation và non-manual grammar trong utterance.
- Cross-language transfer không có một nhãn phonology universal duy nhất; inventory và lexical equivalence cần được xác minh.
- Dữ liệu nhiều nhưng captions/layout/source structure có thể làm representation học shortcut. Không chỉ kiểm tra pretext loss.

### 15.4. Community và data considerations gắn với chất lượng nghiên cứu

Linguistic contrast annotations cần người sử dụng ngôn ngữ ký hiệu/chuyên gia phù hợp và trả công. Ghi variant/dialect và uncertainty thay vì áp một form chuẩn duy nhất. Consent/usage conditions và dataset version phải được tôn trọng; pose không tự anonymized. Không đề xuất xóa mặt như mặc định để chống identity khi face là một kênh thông tin ngôn ngữ.

### 15.5. Lý thuyết và các hướng xa hơn

**[LIT-A]** seq-JEPA nghiên cứu việc tách invariant/equivariant streams; auxiliary-task analysis cho JEPA đặt các điều kiện lý thuyết cụ thể trên môi trường/representation. Các kết quả đó không bảo đảm factorization hoặc semantic sufficiency trên signing noisy, occluded và partially observed. Chúng hỗ trợ đặt giả thuyết có điều kiện, không thay validation. [S56, S57]

MC-JEPA gợi ý kết hợp motion/content; HuBERT là tiền lệ cho masked discrete units. Flow targets hoặc cluster IDs không tự là linguistic primitives. Những hướng đáng thử sau core là uncertainty/multiple-hypothesis prediction cho genuinely ambiguous targets, causal context models và high-resolution local encoders; chúng cần protocol và controls riêng. [S33, S58]

### 15.6. Quyết định cuối cùng cho bản đề xuất hiện tại

Thay “thêm relational latent objective là core novelty” bằng **một nghiên cứu có điều kiện về target observability, support-safe masking và sự giữ lại contrast**. Ưu tiên A factorial; B là hướng geometry có thể train/audit độc lập; C chỉ đi tiếp khi teacher quality và teacher-matched experiments chứng minh giá trị.

Một kết quả âm tính đáng tin—ví dụ whole-articulator prediction làm mất handshape, hoặc simple visible distillation ngang masked geometry—phải được giữ trong kết luận. Không cứu giả thuyết bằng cách thêm nhiều heads/losses rồi chọn score tốt nhất. Đây là tiêu chuẩn để biến file proposal thành một chương trình nghiên cứu có thể kiểm chứng, thay vì một kiến trúc nhiều tên module.

## 16. Tài liệu gốc và độ sâu kiểm tra

Danh mục dùng nguồn gốc, không dùng blog/review thứ cấp làm căn cứ cho claims kỹ thuật. **M** = đã kiểm tra phần methods/results/appendix liên quan; **A** = abstract/metadata chính thức; **D** = official documentation/dataset description; **U** = chưa đủ source access. “M” không có nghĩa đọc từng reference của paper hay tái lập code. Links về phiên bản cụ thể được ưu tiên khi phân tích methods; venue ghi theo nguồn xác minh được tại ngày chốt tra cứu.

### 16.1. JEPA và predictive representation

**[S01]** Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA, 2023). [Paper](https://arxiv.org/html/2301.08243v3). **M:** context/target masks, target encoder.

**[S02]** Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA, 2024). [Paper](https://arxiv.org/html/2404.08471v1). **M:** objective, masking, transfer.

**[S03]** Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* (2025). [Paper](https://arxiv.org/html/2506.09985v1). **M:** video pretraining và action-conditioned stage.

**[S04]** Mur-Labadia et al. *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning* (2026). [Paper v2](https://arxiv.org/html/2603.14482v2). **M:** §2, Table 1, §3.10, Appendix B.

**[S05]** Meta FAIR. *Official V-JEPA 2 / 2.1 repository and checkpoints.* [Repository](https://github.com/facebookresearch/vjepa2). **D:** checkpoint family/specifications; chưa audit toàn bộ code.

**[S06]** Abdelfattah & Alahi. *S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition* (ECCV 2024). [Proceedings PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf), [project](https://sjepa.github.io/). **M:** masking, centered-CE targets, EMA và evaluation.

**[S07]** Balestriero & LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* (2025). [Metadata](https://arxiv.org/abs/2511.08544), [methods bản v1](https://arxiv.org/html/2511.08544v1). **M:** objective/SIGReg; metadata title cập nhật.

**[S08]** He et al. *DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture* (2025; v4 ghi accepted ECCV 2026). [Metadata](https://arxiv.org/abs/2511.17354), [paper v4](https://arxiv.org/html/2511.17354v4). **M:** saliency/order; xác minh status theo arXiv comments.

### 16.2. Sign-specific SSL và representation

**[S09]** Xie, Cai & Stefanov. *SignMAE: Segmentation-Driven Self-Supervised Learning for Sign Language Recognition* (2026). [Paper v1](https://arxiv.org/html/2605.02094v1). **M:** moving hand/arm masks, video/pose streams. Dùng bản arXiv; không gán venue chưa xác minh.

**[S10]** Asasi, Sincan & Bowden. *SignSeek: Learning Transferable Representations for Sign Dictionary Retrieval* (2026). [Paper v1](https://arxiv.org/html/2609.03695v1). **M:** §3, retrieval/evaluation; pretraining có gloss supervision.

**[S11]** Hu et al. *SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition* (ICCV 2021). [Paper record](https://arxiv.org/abs/2110.05382). **A:** original framework; chi tiết trong review dựa thêm SignBERT+.

**[S12]** Hu et al. *SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding* (2023). [Paper v1](https://arxiv.org/html/2305.04868v1). **M:** gesture inputs, multilevel masking, model-aware decoder.

**[S13]** Zhao et al. *MASA: Motion-aware Masked Autoencoder with Semantic Alignment for Sign Language Recognition* (2024). [Paper v1](https://arxiv.org/html/2405.20666v1). **M:** motion targets, masking và momentum alignment.

**[S14]** Wong, Camgoz & Bowden. *SignRep: Enhancing Self-Supervised Sign Representations* (ICCV 2025). [Paper v1](https://arxiv.org/html/2503.08529v1), [official proceedings](https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.pdf). **M:** sign priors, preprocessing, loss.

**[S15]** *SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction* (ACL 2025; preprint 2024). [Proceedings](https://aclanthology.org/2025.acl-long.1397/), [paper v3](https://arxiv.org/html/2411.16765v3). **M:** stream targets, Appendix A/B, data overlap.

**[S16]** *Improving Sign Language Understanding with a Multi-Stream Masked Autoencoder Trained on ASL Videos* (MS-MAE, IJCNLP–AACL 2025). [Proceedings](https://aclanthology.org/2025.ijcnlp-long.66/). **A:** official abstract/status; full methods cần đọc tiếp.

### 16.3. SSL objectives, image/video targets và temporal learning

**[S17]** Baevski et al. *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language* (ICML 2022). [Proceedings](https://proceedings.mlr.press/v162/baevski22a.html). **A:** contextual teacher-target framework.

**[S18]** Baevski et al. *Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language* (data2vec 2.0, ICML 2023). [Proceedings](https://proceedings.mlr.press/v202/baevski23a.html). **A:** teacher-target amortization/efficiency.

**[S19]** Grill et al. *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning* (BYOL, 2020). [Paper record](https://arxiv.org/abs/2006.07733). **A:** asymmetric online/target framework.

**[S20]** Chen & He. *Exploring Simple Siamese Representation Learning* (SimSiam, 2021). [Paper record](https://arxiv.org/abs/2011.10566). **A:** predictor/stop-gradient mechanism.

**[S21]** Zhou et al. *iBOT: Image BERT Pre-Training with Online Tokenizer* (2022). [Paper v3](https://arxiv.org/html/2111.07832v3). **M:** local/global token self-distillation.

**[S22]** Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision* (2023). [Paper record](https://arxiv.org/abs/2304.07193). **A:** visual teacher foundation; không xác lập linguistic crop fidelity.

**[S23]** Bardes, Ponce & LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning* (2022). [Paper v3](https://arxiv.org/html/2105.04906v3). **M:** variance/covariance terms.

**[S24]** Tong et al. *VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training* (2022). [Paper v3](https://arxiv.org/html/2203.12602v3). **M:** masked reconstruction/high-ratio setting.

**[S25]** Wei et al. *Masked Feature Prediction for Self-Supervised Visual Pre-Training* (MaskFeat, 2022). [Paper v2](https://arxiv.org/html/2112.09133v2). **M:** target features/HOG và video pretraining.

**[S26]** Mao et al. *Masked Motion Predictors are Strong 3D Action Representation Learners* (MAMP, ICCV 2023). [Official paper page](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html). **A:** motion-prediction baseline; chưa tái lập configs.

**[S27]** Sun et al. *Exploring Adaptive Masked Reconstruction for Self-Supervised Skeleton-Based Action Recognition* (AMR, CVPR 2026). [Paper v1](https://arxiv.org/html/2606.11450v1), [official proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Exploring_Adaptive_Masked_Reconstruction_for_Self-Supervised_Skeleton-Based_Action_Recognition_CVPR_2026_paper.html). **M:** decoder decoupling, focal reconstruction, Appendix C.2.

**[S28]** Zhu et al. *MotionBERT: A Unified Perspective on Learning Human Motion Representations* (ICCV 2023). [Official paper page](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html). **A:** unified motion encoder/lifting framework.

**[S29]** Madjoukeng et al. *SSL-SLR: Self-Supervised Representation Learning for Sign Language Recognition* (preprint 2025; TMLR 2026 record). [Paper v2](https://arxiv.org/html/2509.05188v2), [OpenReview](https://openreview.net/forum?id=buTZkTXijy). **M:** negative-free objective và augmentation.

### 16.4. Graph, motion và coordinate specifications

**[S30]** Yan, Xiong & Lin. *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* (ST-GCN, AAAI 2018). [Proceedings](https://ojs.aaai.org/index.php/aaai/article/view/12328), [paper record](https://arxiv.org/abs/1801.07455). **A:** spatial-temporal graph framework.

**[S31]** Chen et al. *Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition* (CTR-GCN, ICCV 2021). [Paper v2](https://arxiv.org/html/2107.12213v2). **M:** learned adjacency mechanism.

**[S32]** Satorras, Hoogeboom & Welling. *E(n) Equivariant Graph Neural Networks* (ICML 2021). [Paper record](https://arxiv.org/abs/2102.09844). **A:** equivariant graph framework; không sign-specific evidence.

**[S33]** Bardes, Ponce & LeCun. *MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features* (2023). [Paper record](https://arxiv.org/abs/2307.12698). **A:** joint motion/content objectives.

**[S34]** *Path-JEPA: Path Signature Based Predictive Learning for Skeleton Action Recognition* (ECCV 2026 record). [Conference-hosted PDF](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf). **A/U:** indexed primary record/excerpts; chưa lấy được full PDF. Không dùng results như evidence cho sign.

**[S35]** Google AI Edge. *Hand landmarks detection guide for Python.* [Official specification](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python). **D:** normalized coordinates, world origin, output fields.

**[S36]** Google AI Edge. *Pose landmark detection guide for Python.* [Official specification](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python). **D:** body coordinate origin, visibility/confidence interface.

**[S37]** O’Brien, Sant, Müller & Ebling. *Evaluation of Pose Estimation Systems for Sign Language Translation* (LREC SignLang Workshop 2026). [Proceedings](https://aclanthology.org/2026.signlang-1.39/), [PDF](https://aclanthology.org/2026.signlang-1.39.pdf). **M:** controlled estimator comparisons, stability/missing hands/occlusion.

**[S38]** Kuznetsova & Kimmelman. *Testing MediaPipe Holistic for Linguistic Analysis of Nonmanual Markers in Sign Languages* (2024). [Paper v1](https://arxiv.org/html/2403.10367v1), [updated metadata](https://arxiv.org/abs/2403.10367). **M:** eyebrow/head-motion measurement limitations.

### 16.5. Linguistic resources, other supervision regimes và identity

**[S39]** *The American Sign Language Knowledge Graph: Infusing ASL Models with Linguistic Knowledge* (ASLKG, Findings NAACL 2025). [Proceedings](https://aclanthology.org/2025.findings-naacl.389/). **A:** linguistic KG và supervised knowledge use.

**[S40]** Kezar et al. *The Sem-Lex Benchmark: Modeling ASL Signs and Their Phonemes* (2023). [Paper v1](https://arxiv.org/html/2310.00196v1). **M:** dataset alignment và phonological experiments.

**[S41]** Sevcikova Sehyr et al. *The ASL-LEX 2.0 Project: A Database of Lexical and Phonological Properties for 2,723 Signs in American Sign Language* (2021). [Official contents](https://asl-lex.org/about.html), [data](https://asl-lex.org/download.html), [article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685/). **D/A:** verified inventory/metadata; evaluation mapping vẫn cần audit.

**[S42]** *Towards Privacy-Aware Sign Language Translation at Scale* (SSVP-SLT, ACL 2024). [Paper v2](https://arxiv.org/html/2402.09611v2), [proceedings](https://aclanthology.org/2024.acl-long.467/). **M:** SSL/paired stages và caption shortcut handling.

**[S43]** *SignCLIP: Connecting Text and Sign Language by Contrastive Learning* (EMNLP 2024). [Proceedings](https://aclanthology.org/2024.emnlp-main.518/). **A:** paired text/sign regime.

**[S44]** *Uni-Sign: Toward Unified Sign Language Understanding at Scale* (ICLR 2025). [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html), [official code](https://github.com/ZechengLi19/Uni-Sign). **A:** multimodal sign understanding, khác unlabeled-only SSL.

**[S45]** *Two-Stream Network for Sign Language Recognition and Translation* (NeurIPS 2022). [Official proceedings](https://papers.neurips.cc/paper_files/paper/2022/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html). **A:** RGB/pose supervised fusion.

**[S46]** *VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language* (2025). [Paper v1](https://arxiv.org/html/2512.10942v1), [metadata](https://arxiv.org/abs/2512.10942). **M:** text-embedding targets; khác sign-video-only regime.

**[S47]** Battisti et al. *Person Identification from Pose Estimates in Sign Language* (LREC SignLang Workshop 2024). [Proceedings](https://aclanthology.org/2024.signlang-1.2/), [PDF](https://aclanthology.org/2024.signlang-1.2.pdf). **M:** identity retained in pose estimates.

### 16.6. Datasets và external provenance

**[S48]** Microsoft Research. *ASL Citizen: Dataset Description.* [Official description](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/). **D:** versions, signer-independent split.

**[S49]** *WLASL.* [Official dataset page](https://dxli94.github.io/WLASL/). **D:** corpus/version/access conditions; strict signer split phải kiểm tra thêm.

**[S50]** *How2Sign.* [Official dataset page](https://how2sign.github.io/). **D:** multimodal corpus và detailed 3D subset.

**[S51]** *YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus* (2023). [Paper record](https://arxiv.org/abs/2306.15162). **A:** dataset scope; actual available IDs cần audit.

**[S52]** *YouTube-SL-25: A Large-Scale, Open-Domain Multilingual Sign Language Parallel Corpus* (2024). [Paper record](https://arxiv.org/abs/2407.11144). **A:** multilingual corpus; not all language pairs interchangeable.

**[S53]** *BOBSL.* [Official dataset page](https://www.robots.ox.ac.uk/~vgg/data/bobsl/). **D:** BSL interpreted-TV dataset và release information.

**[S54]** *RWTH-PHOENIX-Weather 2014 T: Parallel Corpus of Sign Language Video, Gloss and Translation.* [Official dataset page](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/). **D:** downstream corpus scope.

**[S55]** *CSL-Daily.* [Official dataset page](https://ustc-slr.github.io/datasets/2021_csl_daily/). **D:** downstream corpus scope.

### 16.7. Theory extensions và nguồn cần xác minh

**[S56]** *seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models* (2025). [Paper record](https://arxiv.org/abs/2505.03176). **A:** invariant/equivariant streams.

**[S57]** *Why and How Auxiliary Tasks Improve JEPA Representations* (2025). [Paper record](https://arxiv.org/abs/2509.12249). **A:** conditional theoretical analysis; không chuyển thành guarantee cho signing.

**[S58]** Hsu et al. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units* (2021). [Paper record](https://arxiv.org/abs/2106.07447). **A:** masked discrete-target precedent.

**[U01]** AHA-JEPA trong file gốc, DOI được cung cấp: [10.1109/ACCESS.2026.3716809](https://doi.org/10.1109/ACCESS.2026.3716809). **U:** chưa truy xuất được primary full text/metadata đủ để xác nhận mô tả carrier. Không kết luận nguồn này giả; cũng không dùng như verified evidence. Cần bản tác giả hoặc trang publisher truy xuất được trước khi trích dẫn kỹ thuật.

Các nguồn U và A chỉ có thể hỗ trợ claims đúng mức đã xác minh. Bất cứ implementation nào dựa vào chi tiết của MS-MAE, Path-JEPA hoặc U01 đều cần đọc bản đầy đủ trước khi chốt reproduction.
