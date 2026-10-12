# PhonoJEPA: truy vết tín hiệu và chiến lược skeleton JEPA cho sign SSL

**Ngày:** 11/10/2026. **Phạm vi:** proposal12, implementation PhonoJEPA, kết quả frozen evaluation hiện có và literature primary 2021–2026. **Trạng thái:** báo cáo nghiên cứu và thiết kế thí nghiệm; các chiến lược mới bên dưới chưa được train. Không thay đổi hay khởi chạy job training trong lượt nghiên cứu này.

## 1. Kết luận có thể hành động

**Nên giữ cấu trúc tay hiện tại, bổ sung thân trên và quan hệ có kiểm soát, đồng thời kiểm tra lại cách biểu diễn chuyển động. Chưa có bằng chứng đủ để bỏ face hoặc motion.** Handshape đang được giải mã tuyến tính khá hơn nhiều reference RGB, nhưng chưa phải tín hiệu duy nhất có ích, cũng chưa chứng minh nó là nguyên nhân chính của retrieval tốt.

Ba phân biệt quyết định thiết kế:

1. **Thông tin có trong raw keypoints, thông tin có trong tokens và thông tin đọc được từ embedding là ba thứ khác nhau.** Có thể mất tín hiệu ngay ở chuẩn hóa, trước khi JEPA được học; cũng có thể encoder giữ nhưng pooling làm yếu nó.
2. **Upper-body chưa được bao phủ đầy đủ, nhưng không hoàn toàn vắng mặt.** LOC có wrist/elbow; shoulders làm anchor. Thiếu stream riêng cho torso, shoulder/hip dynamics và explicit hand–body–head relations.
3. **Probe không phải ablation causal.** R² thấp không cho phép xóa một bộ phận; R² cao không bảo đảm retrieval tốt. Cần giữ cùng data, steps, capacity và readout khi thử thêm/bỏ/thay tín hiệu.

Ưu tiên đề xuất: **S0 audit/readout → S1 body/relations → S2 motion và signature bổ sung → S3 graph tokenizer → S4/S5 teacher pretrained**. S2 bắt đầu bằng motion targets đơn giản trước khi thêm toàn bộ log-signature hierarchy. Không có paper nào trong review chứng minh sẵn chiến lược này sẽ thắng trên dữ liệu hiện tại.

Các link nguồn được đặt cạnh nhận định; danh mục, phiên bản, mức bằng chứng và code commit đầy đủ nằm trong [LITERATURE_AND_CODE.md](research_update_20261011/LITERATURE_AND_CODE.md). Nhật ký ba vòng phản biện: [RESEARCH_LOG.md](research_update_20261011/RESEARCH_LOG.md).

## 2. Method hiện tại thực sự nhận và học gì?

### 2.1 Input và encoder

`PLctx 16k` là tên một cấu hình PhonoJEPA: **PL** = Predictability-Ladder Channel Masking, **ctx** = contextual latent target, **16k** = 16.000 optimizer steps. Đây không phải context length 16.000 tokens hay tên backbone pretrained.

Raw COCO-WholeBody 133 keypoints được chọn còn 103 joints. Cache dùng tọa độ pixel, có missingness và confidence đã scale nhưng chưa calibrated. Một window gồm 64 samples ở `min(fps,25)`, ghép mỗi hai samples thành một bin. Bảy channels × 32 bins = 224 tokens:

| Channel | Nội dung vào tokenizer | Thông tin chính / giới hạn |
|---|---|---|
| D_CONF, N_CONF | Hình học mỗi bàn tay: palm axis, polygon, finger bones chuẩn hóa, foreshortening/scale residual; score và missingness | Cấu hình ngón **và một phần orientation**, không phải handshape thuần tách khỏi góc nhìn |
| D_LOC, N_LOC | Wrist, elbow trong body frame; wrist displacement trong cặp hai samples | Vị trí tay/cánh tay và motion rất ngắn; không có explicit trajectory dài |
| F_UP | Brows/eyelids trong eye frame, trừ median theo window | Biến thiên quanh baseline; có thể mất biểu cảm được giữ lâu |
| F_MO | Mouth/chin trong eye frame, trừ median | Động học miệng; sustained mouth shape có thể bị loại |
| F_HD | Eye center/log scale/nose residual quanh median, cùng eye-axis direction | Chuyển động đầu tương đối; thiếu một số tham chiếu head-to-body tuyệt đối theo window |

Dominant hand được canonicalize sang bên phải bằng mirror + đổi indices; dominant suy từ wrist motion cả clip. Đây là quyết định có nhìn content, đã được implementation ghi là một bit side-information. Không gọi toàn bộ pipeline content-independent chỉ vì mask sampler không đọc keypoints.

Encoder là **Transformer tự xây, train từ đầu: 6 blocks, d256, 4 heads, FFN1024, temporal RoPE**, có channel-specific MLPs và CLS. Predictor 4 blocks d128. Teacher EMA có cùng encoder. Contextual targets lấy trung bình các tầng cuối sau normalization; centered CE khớp phân bố theo chiều feature. Predictor query chỉ có channel/time metadata, không chứa tọa độ target. Đây là JEPA theo cơ chế masked context → teacher representation, gần S-JEPA nhưng không phải bản tái lập nguyên recipe của S-JEPA. [Code local model](../../method/phonojepa/pj/model.py), [loss](../../method/phonojepa/pj/losses.py), [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf).

Transformer cho phép kết hợp nhiều thời điểm và bộ phận: hình dạng tay tại một thời điểm có thể được suy từ các thời điểm khác, tay còn lại, vị trí tay và mặt. Tuy nhiên, **không có bảo đảm attention tự học đúng cấu trúc giải phẫu**. Graph tokenizer là một inductive bias đáng kiểm tra, không phải điều kiện bắt buộc để gọi JEPA.

### 2.2 Mask và readout

PL-lite hiện có near view che all-channel time spans 4–8 bins; far view che một channel tay trên đoạn dài, giữ partner channel của tay đó. Face có dependency `F_HD hidden ⇒ F_UP/F_MO hidden`. Raw source masking được truyền qua support của interpolation. Student xử lý đủ slots nhưng hidden slots không làm attention keys; **visible-only attention không đồng nghĩa đã gather để giảm compute**. [Masks](../../method/phonojepa/pj/masks.py), [window](../../method/phonojepa/pj/data/window.py), [dataset](../../method/phonojepa/pj/data/dataset.py).

Readout `_grp` nối ba vectors d256: dominant hand pair, nondominant hand pair và face; mỗi vector mean các present tokens theo thời gian. Tổng 768 dimensions. Các tokens đã tương tác trong encoder nên “vector tay” có thể chứa thông tin từ mặt/cánh tay. Không thể diễn giải ba vectors như các nguồn độc lập. Window embeddings lại được pool thành clip embedding. [Extract](../../method/phonojepa/pj/extract.py).

## 3. Handshape, vị trí/quan hệ và các probe cho biết gì?

Các probe dùng geometry tính từ **keypoints 2D gốc**, lấy trung bình theo source span của window, rồi ridge regression từ frozen window embedding. Chúng không phải nhãn ngôn ngữ học do chuyên gia gán.

| Probe | Target thực tế | Ví dụ phân biệt được / điều chưa đo được |
|---|---|---|
| Handshape, 164 dims | Mỗi tay: 42 wrist-relative coordinates, 30 sin/cos bend angles, 10 fingertip pair distances | Tay nắm/mở, ngón co/duỗi và layout chiếu 2D; không phải phân loại phonological handshape chuẩn, cũng không tách hoàn toàn orientation |
| Location/relations, 17 dims | Mỗi tay: wrist/index tip so với nose, wrist so với shoulder center, min hand–face distance; thêm wrist-to-wrist vector và min hand–hand distance; scale theo shoulder width | Cùng dạng tay nhưng đặt gần trán/ngực; tay gần/chạm nhau trên ảnh. **Khoảng cách 2D gần không chứng minh tiếp xúc 3D** |
| Body, 30 dims | COCO joints 0–12 so với shoulders (26 dims) + hai elbow angles (4 dims) | Có cả nose/eyes/ears/hips, không chỉ torso; target rộng hơn body signals đang đưa vào encoder |
| Movement, 14 dims | Mỗi tay: wrist velocity/acceleration, index-tip velocity và handshape-change speed | Một số biến thiên ngắn; mean signed derivatives dễ triệt tiêu, không đo đầy đủ trajectory/repetition/rhythm |
| Non-manual, 65 dims | Mouth/brow geometry, eye/mouth opening, head-roll/nose proxies | Geometric expression proxies; không đánh giá đầy đủ mouthing/grammar/affect |

Nguồn định nghĩa: [jt/priors.py](../../method/jepa_targets/jt/priors.py), [probe slices và ridge](../../method/jepa_targets/jt/evaluate.py), [window targets](../../method/obs_jepa/oj/evaluate.py).

**R²≈0,277 nghĩa là linear probe đạt mức giải thích phương sai target đó trên protocol hiện tại; không có nghĩa embedding chứa “27,7% toàn bộ thông tin handshape”.** R² âm nghĩa là lỗi bình phương lớn hơn baseline mean dùng trong phép tính, không đồng nghĩa không còn thông tin. Probe nonlinear có thể đọc được thêm, nhưng không được thay classifier tùy model để làm đẹp kết quả.

### 3.1 Có thể truy vết để chọn input không?

Có. Cần theo chuỗi sau, rồi chỉ ra bước bắt đầu làm mất thông tin:

```mermaid
flowchart LR
  A[Raw joints + scores + timestamps] --> B[Observation gate và canonicalization]
  B --> C[Resampling và hệ tọa độ]
  C --> D[CONF / LOC / face / body features]
  D --> E[Tokenization]
  E --> F[Encoder theo thời gian]
  F --> G[Pooling window và clip]
  G --> H[Frozen retrieval / linear probe]
```

Đề xuất đo target ở bốn điểm: **raw descriptor → normalized features → token features → final readout**. Nếu raw/normalized descriptor dự đoán được motion nhưng readout không được, thay pooling/objective trước khi đổi pose estimator. Nếu tín hiệu đã biến mất ở normalization, thêm Transformer sâu hơn không khôi phục được dữ liệu đã bị loại. Nếu raw pose không có evidence do occlusion, cân nhắc modality/teacher khác.

Ví dụ cụ thể do phân tích của báo cáo:

- Hai signs có handshape tương tự nhưng tay ở đầu và ngực: giữ body-referenced location/head reference; wrist-relative hand shape riêng không đủ.
- Hai chuyển động đi thẳng và đi vòng có cùng điểm đầu/cuối: mean velocity có thể giống nhau. Cần ordered trajectories, signed area, temporal bins hoặc multi-lag changes.
- Với `v_t = x_t − x_(t−1)`, tổng velocity bằng `x_last − x_first`. Đi ra rồi quay lại có thể có mean velocity gần 0 dù chuyển động lớn; probe movement hiện tại có rủi ro này.
- Nếu mouth shape gần như không đổi cả window, `mouth_t − median(mouth)` gần 0 cho nhiều biểu cảm khác nhau. Nên thử **level branch + residual branch**, không mặc định bỏ mặt.
- Per-frame torso alignment có thể xóa projected shoulder roll/lean. Body frame cố định theo window/clip giữ được biến thiên đó; camera motion vẫn là yếu tố cần đối chứng.

## 4. Eval hiện có: ngoài handshape còn tín hiệu tốt nào?

Snapshot P1 lấy lúc **21:21:32 +07, 11/10/2026**. [JSON đóng băng](research_update_20261011/local_snapshot/P1.json), [CSV tất cả models/readouts](research_update_20261011/local_snapshot/P1_summary.csv), [manifest/hash](research_update_20261011/local_snapshot/provenance.json). P1 complete, P2/P3 chưa có JSON tại snapshot; không dùng train loss của các arm đang chạy thay downstream eval.

### 4.1 Kết quả frozen, đơn vị %

| Model | Steps local | LOSO R@1 white512 | ASLC300 retrieval R@1 white512 | ASLC300 linear probe macro |
|---|---:|---:|---:|---:|
| B0 handcrafted descriptor | 0 | 4,15 | 17,27 | 24,20 |
| PLboth `_grp` | 4k | 22,53 | 41,87 | 47,13 |
| PLboth prototype `_grp` | 4k | 20,25 | 37,73 | 45,47 |
| PLboth L1+SIGReg `_grp` | 4k | 14,21 | 28,07 | 43,40 |
| PLboth L1+SIGReg no-EMA `_grp` | 4k | 16,30 | 33,07 | 47,13 |
| **PLctx 16k `_grp`** | 16k | **25,38** | **48,93** | **60,80** |
| PLint 16k `_grp` | 16k | 24,67 | 43,27 | 55,93 |
| B12 reference graph | 4k recipe khác | 7,54 | 20,67 | 22,33 |
| B13 reference graph | 4k recipe khác | 7,93 | 21,27 | 25,13 |
| SR3D reference RGB | recipe khác | 5,14 | 10,07 | 17,60 |
| JH3D reference RGB | recipe khác | 39,19 | 56,80 | 61,73 |
| JV3D reference RGB | recipe/pretrained khác | 27,98 | 22,60 | 56,40 |

SR3D/JH3D/JV3D là tên reference local, **không phải toàn bộ kết quả paper SignRep/V-JEPA**. B0 trong bảng là descriptor implementation của P1; không trộn với con số B0=13,46 của pilot D8, vì descriptor/protocol khác nhau.

### 4.2 Geometric probes, R²

| Model `_grp` nếu có | Handshape | Body | Location/relations | Movement | Non-manual | Signer acc. % |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0,150 | −0,071 | −0,144 | −0,074 | −0,331 | 89,44 |
| PLboth 4k | 0,274 | 0,055 | 0,118 | −0,051 | −0,304 | 87,36 |
| **PLctx 16k** | **0,277** | **0,047** | **0,124** | **−0,035** | **−0,276** | **90,27** |
| PLint 16k | 0,300 | 0,061 | 0,163 | −0,022 | −0,271 | 92,61 |
| B12 | 0,290 | 0,281 | 0,749 | −0,006 | −0,309 | 88,94 |
| B13 | 0,321 | 0,327 | 0,712 | 0,006 | 0,158 | 91,60 |
| SR3D | 0,031 | −0,049 | 0,204 | 0,209 | −0,225 | 98,42 |
| JH3D | 0,213 | 0,163 | 0,584 | 0,284 | −0,227 | 92,37 |
| JV3D | 0,181 | 0,178 | 0,490 | 0,249 | −0,227 | 98,09 |

Đọc bảng đúng mức:

- **Location/relations và body có tín hiệu dương trong PhonoJEPA**, mạnh hơn B0, nhưng relations vẫn thấp hơn nhiều references. Đây là vùng đáng điều tra/bổ sung.
- PLint giải mã handshape/relations/body nhỉnh hơn PLctx nhưng retrieval và LP thấp hơn. Không chọn objective chỉ bằng geometric probes.
- B13 có relations/body cao mà lexical retrieval thấp. Ngược lại, JH3D handshape thấp hơn PLctx nhưng retrieval cao hơn. Cần complementary cues và representation thích hợp cho nhiệm vụ.
- Motion là khoảng trống đáng kiểm tra vì JH3D/JV3D đạt R² dương rõ hơn; **chưa chứng minh chỉ thêm motion sẽ thu hẹp khoảng cách**, vì modality/pretraining/recipe đều khác.
- Non-manual âm ở phần lớn models, cả references RGB. Đây có thể là giới hạn probe/geometry/representation; chưa phải bằng chứng non-manual vô dụng. B13 dương cho thấy phép đo này không hoàn toàn bất khả thi.
- Signer accuracy PLctx còn 90,27%; hand normalization chưa tạo signer-invariant embedding. Giảm signer accuracy đơn độc cũng chưa đủ: bỏ mọi signal có thể giảm cả signer lẫn sign information.

### 4.3 Readout và độ chắc chắn

PLctx16k cùng checkpoint: `_grp` LOSO 25,38% so với `_cls` 6,04%; LP 60,80% so với 28,87%; ASLC300 retrieval 48,93% so với 17,40%. Riêng `_grp`, whitening đổi LOSO từ 16,96% lên 25,38%. **Pooling và feature geometry là biến thực nghiệm lớn**, không nên vội quy thành công cho masking.

Paired bootstrap đã có trong P1, chênh lệch tính theo percentage points:

| PLctx16k so với | LOSO Δ [CI95] | LP Δ [CI95] | ASLC300 retrieval Δ [CI95] |
|---|---|---|---|
| B13 | +17,45 [16,14;19,58] | +35,67 [23,63;46,91] | +27,67 [16,85;34,69] |
| SR3D | +20,23 [17,84;23,15] | +43,20 [37,26;47,12] | +38,87 [34,42;41,93] |
| JH3D | −13,81 [−16,41;−6,42] | −0,93 [−9,49;7,05] | −7,87 [−20,52;10,44] |

Các CI này resample ít signer clusters (LOSO 6; ASLC300 5), không đo seed variance. CI chứa 0 không chứng minh equivalence. P1 chỉ seed0, nhiều comparisons mang tính exploratory. 4k→16k không phải đối chứng riêng contextual vs intrinsic hoặc riêng mask. Phải đợi so sánh đồng budget trong P2, rồi kiểm tra seed mới.

Protocol còn có hai điểm cần công bố:

1. LOSO `white512` fit PCA trên evaluated windows, mang tính **transductive label-free**. ASLC300 whitening fit gallery-only; LP fit gallery và chọn C qua signer holdout. Không gộp hai protocol thành tuyên bố hoàn toàn inductive.
2. Geometric probe chia signer-disjoint outer2 folds, chọn ridge alpha inner LOSO; feature standardization `X` hiện dùng toàn bộ window set trước outer split. Target còn khác hệ tọa độ/handedness với encoder. Cần thêm bản **fit-train-only và coordinate-aligned**, giữ bản cũ để đối chiếu lịch sử.

## 5. Giữ, bổ sung hay thay thế tín hiệu nào?

| Tín hiệu | Quyết định hiện tại | Thay đổi có thể thử | Bằng chứng cần có trước khi loại |
|---|---|---|---|
| Finger/palm CONF | **Giữ** | Thêm graph-local encoding hoặc latent head bảo toàn tay; không ép palm-frame invariance vô điều kiện | Train không CONF hoặc giản lược CONF, cùng budget; đánh giá handshape tối thiểu và retrieval qua signer |
| Palm orientation | **Giữ cùng shape** | Tách diagnostic head để biết orientation và configuration đóng góp khác nhau; không tách thành mask dễ suy rò | Counterfactual orientation, matched sign examples; kiểm tra mất lexical distinctions |
| LOC wrist/elbow | **Giữ, mở rộng** | Explicit hand–head/torso/hand relations, trajectories nhiều lag | Ablation bỏ tọa độ nhưng giữ quan hệ và ngược lại; target/readout cùng hệ tọa độ |
| UBody torso/shoulders/hips | **Bổ sung có đối chứng** | Stable body reference, torso dynamics, arm angles; không chỉ thêm tọa độ absolute | +body thật so với +capacity control và body-shuffled control; signer-held-out |
| Motion | **Thay cách biểu diễn/đo trước khi bỏ** | Signed/magnitude changes nhiều lag, temporal bins, signature hỗ trợ | Trajectory/order/repetition probes, corruption theo thời gian, matched reconstruction baseline |
| Face/head level | **Chưa bỏ** | Thử level + neutral residual; explicit head-to-body reference | Hands-only train từ đầu và face replacement controls; subset gần mặt/khác mouth/head cues |
| Camera translation, global scale, morphology | Giảm nuisance có chọn lọc | Center/scale và augmentation train-only, đo cả signer và lexical information | Invariance cần giữ downstream performance; 2D camera/body motion dễ lẫn |
| Score/missingness | **Giữ quality information** | Boolean observed flag + confidence soft weight có floor; đánh giá theo occlusion | Không coi normalized score là xác suất hay label semantic; kiểm tra mask-location shortcut |

Pilot v2.1 hỗ trợ giữ body-frame CONF + LOC và cho thấy face có thể tăng signer signal; đó là handcrafted training-free diagnostic, chưa quyết định lợi ích face sau SSL. Claim cũ “per-bone normalization tự nó giúp” đã bị rút ở v1. Chỉ dùng bản v2/v2.1 được [README pilot](pilot_diagnostics/README.md) xác định authoritative.

**Phép thử thay tín hiệu cần phân biệt inference ablation với retraining.** Che face của model đã học đo độ phụ thuộc/OOD robustness; train model không face từ đầu đo khả năng bỏ face trong thiết kế. Zero, shuffle, drop và replacement đều có thể đổi phân bố; nên dùng nhiều control và ghi rõ câu hỏi tương ứng. Không dùng permutation importance như causal importance tuyệt đối.

## 6. Literature dẫn đến lựa chọn nào?

Review được nhóm thành các trục thực nghiệm thay vì xếp hạng accuracy giữa paper:

| Trục | Nguồn gốc | Điều có thể chuyển sang ta | Điều cần giữ lại làm phản chứng |
|---|---|---|---|
| Learned contextual targets | S-JEPA, Skeleton2vec, I-JEPA [R01,R02,R19] | EMA clean teacher, missing-token prediction, target nhiều tầng | Contextual target vẫn có thể học nuisance; không đồng nghĩa học phonology |
| Explicit motion target | MAMP, MASA [R03,R14] | Residuals/kinematics nhiều scale, quality weighting | Static holds có nghĩa; pose jitter có thể giả high-motion |
| Target/decoder efficiency | GFP, AMR [R04,R05] | Target granularity khác input, cross-attention predictor, local/global supervision | Reconstruction vẫn là baseline hợp lệ; EMA lớn không phải cách duy nhất |
| Anatomical masks | SLiM, BEST [R06,R13] | Tubes theo bộ phận/chuỗi, mask ratio sweep | Anatomy group của Kinect không phải fine-hand graph; random spans có thể tốt hơn trong sign |
| Geometry trajectories | Path-JEPA [R07] | Joint/pair/chain hỗ trợ temporal relations; dependency-aware masking | Không thay được static shape/location; global view và temporal masks cần thiết kế lại |
| Complementary dynamics | ASMa, STC [R08,R15] | Không chỉ tập trung high-motion; tách rồi fuse cues bổ sung | Learned downstream attention không phải true linear probe |
| Sign articulators | SignRep, SignSeek, SHuBERT [R09,R10,R17] | Bao phủ hands/body/head/face, phối hợp local/global features | RGB vs skeleton, labels và data scale khác nhau; không coi body đã được chứng minh riêng |
| Other teachers/objectives | MotionBERT, SignDino, LeJEPA, V-JEPA2.1 [R16,R18,R20,R21] | Geometric/RGB teacher, anti-collapse và dense intermediate supervision | External supervision/pretraining và objective khác phải ghi rõ |

Nguồn của từng hàng và phiên bản đã đọc: [bảng literature](research_update_20261011/LITERATURE_AND_CODE.md). Các kết luận tổng hợp này là suy luận thiết kế của báo cáo, không phải kết quả benchmark mới.

### 6.1 Path-JEPA người dùng bổ sung: phần nên học và phần cần kiểm chứng

Hai URL của người dùng là cùng Path-JEPA, venue xác nhận qua ECCV và publisher. Paper dùng global/local trajectory descriptors, EMA/cosine latent prediction và mask closure theo joint–edge–chain. Table 6 báo thêm local rồi relational levels tăng nội bộ; kết quả NTU không xác nhận trực tiếp sign. [PDF §§3–4.5](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf).

**Những nhận xét kỹ thuật dưới đây là phân tích của báo cáo để chuyển domain:**

1. Signature dựa trên increments bất biến với tịnh tiến path. Một đường hằng ở vị trí A và đường hằng ở B đều có increments bằng 0. Nếu các ngón không chuyển động, chỉ lấy signature quỹ đạo sẽ không phân biệt nhiều handshapes tĩnh. Pair displacement path đứng yên cũng không khôi phục vị trí tương đối tĩnh. Cần giữ **CONF + level relation/location**, có thể thêm start/end level bên cạnh signature.
2. Global signature chứa toàn bộ thời gian. Nếu lặp global component trong các local tokens rồi chỉ che một đoạn thời gian kiểu PL, phần visible vẫn có thể chứa target interval. Dependency closure phải xét **joint × raw-frame support**, không chỉ tên body part. Với temporal mask, có thể bỏ global component ở student hoặc chỉ dùng global summary ở teacher/auxiliary head.
3. Không zero các tọa độ bị che trước khi tính signature: zero tạo bước nhảy giả. Với clean teacher, tính descriptor đầy đủ; với student, bỏ token nếu support chạm hidden source, hoặc chỉ biểu diễn contiguous observed segments có gap flag. Không nối tắt hai phía occlusion như một đường được quan sát thật.
4. Định lý reparameterization nói về cùng continuous path. Subsampling mất đường cong/chi tiết nhanh tạo path khác. Fixed-lag lift cũng có thể phụ thuộc lịch sampling. Nên đo robustness thực nghiệm theo fps, drop frames và jitter, không hứa bất biến mọi tốc độ hay giữ rhythm tự động.
5. Các chi tiết tái lập cần xác minh: paper nêu 22 log-signature coefficients cho lift 6D/degree2, trong khi free-Lie dimensions chuẩn là `6 + (6²−6)/2 = 21` nếu bỏ scalar convention, theo [Witt formula trong tài liệu iisignature của tác giả thư viện](https://github.com/bottler/iisignature/blob/main/doc.md#logsiglengthd-m). Tương tự, 152 tokens/4 windows và danh sách paths chưa đủ để dựng topology chính xác từ full NTU joints. Đây là **điểm cần tác giả/code làm rõ**, không đủ kết luận kết quả sai. Ta nên tính shape từ thư viện/định nghĩa đã chọn, không hard-code số paper.

Do repo hiện chỉ có trang dự án, chiến lược S2 bên dưới là **adaptation có định nghĩa riêng**, không nhận là official reproduction Path-JEPA.

## 7. Các chiến lược đề xuất

Mọi hyperparameter mới dưới đây là **điểm bắt đầu đề xuất**, chưa phải lựa chọn đã tối ưu. Chỉ so hiệu quả sau đối chứng đồng ngân sách.

### S0 — Sửa phép đo và đọc representation trước khi đổi model

**Input:** giữ toàn bộ pipeline hiện tại. **Encoder/pretrained:** frozen teacher của PLctx16k/PLint16k, từ scratch local; không dùng checkpoint bên ngoài. **Masking:** không đổi training mask; dùng corruption ở eval chỉ để chẩn đoán. **Ép học:** không có training mới ở bước đầu này.

**Việc làm:**

- Geometric probe phiên bản aligned: cùng pixel aspect, dominant-side convention và body coordinate với encoder; fit normalization trong outer-training signers. Giữ bản raw-coordinate cũ song song để biết invariance đã loại gì.
- Tách static/dynamic và handedness: shape angles/distances, palm orientation, torso residual, head reference; motion probes có speed/path-length/signed area, lag displacement, order/repetition và temporal bins thay mean signed velocity đơn độc.
- Cùng checkpoint so `_grp`, mean, CLS; thử temporal pyramid 1/2/4 bins với projection/PCA fit gallery/training only. Nếu cần token-level readout, phải re-extract; không thể dựng lại tokens đã mất chỉ từ `_grp` cache.
- Kiểm tra per-channel observability, error theo occlusion/fps/signer; không gộp absence và training mask.

**Inference/readout:** giữ `_grp` làm anchor, đề xuất temporal pyramid có số chiều sau projection bằng baseline. **Chi phí:** thấp hơn retraining; token extraction cần forward riêng. **Bác bỏ:** nếu corrected probes vẫn thấp và token-level/temporal readout không cứu motion/body, chuyển trọng tâm sang input/objective. Nếu chỉ readout đã cải thiện, chưa cần tăng backbone.

### S1 — Giữ CONF, thêm torso và explicit relations với thay đổi nhỏ

**Input:** giữ 7 channels; thêm một `TORSO` token mỗi bin từ nose, shoulders, hips trong body frame ổn định và các residual theo thời gian. Nose/hips đã được cache chọn vào 103 joints nhưng hiện không dùng làm nội dung. Không cần giả định raw pose mới. Level quan hệ tay–torso/tay–head/tay–tay ban đầu dùng **auxiliary targets**; sau đó mới thử `REL` input token như một factor riêng. Elbow/wrist đã có trong LOC; arm angles là derived geometry, không tính là joint mới.

**Encoder:** cùng Transformer 6×d256, heads4; MLP torso→256; 8×32=256 tokens + CLS, predictor d128/4 blocks. Tăng attention slots từ 224→256 có chi phí lý thuyết khoảng `(256/224)²≈1,31` ở phần attention, không phải 1,31× toàn bộ runtime. So thêm capacity control có cùng slots/parameters nhưng không thêm body evidence.

**Pretrained:** primary experiment scratch, không có external weights. Warm-start PLctx16k là nhánh triển khai riêng; phải ghi tổng budget đã tiêu thụ, không so “4k warm-start” với “4k scratch” như cùng compute.

**Masking:** giữ PL làm anchor; thêm torso tubes theo thời gian. Nếu che raw nose/hips/elbow/wrist, mọi derived relation và duplicate joint representation liên quan phải bị che. Median anchors dùng chung cho student/teacher phải được tính từ allowed context, hoặc được khai báo là luôn-visible side-information và không tính là target đã hidden. Khi raw shoulders thành target, không dùng statistics chứa hidden shoulders để dựng student frame. Tránh dùng face nose/body nose như hai đường copy cùng điểm.

**Target/loss/ép học:** contextual EMA teacher như baseline; thêm head dự đoán masked torso dynamics và static hand–body relations, mỗi nhóm có normalization/weight riêng. Mô hình phải suy vị trí/tư thế thân từ context và phối hợp hand placement, thay vì chỉ học thêm signer morphology. `λ_body=0` và `λ_body>0` tách effect input khỏi effect supervision.

**Inference/readout:** concat pooled D/N/face/torso, rồi fixed train/gallery-only projection về dimension thống nhất; giữ temporal bins ở nhánh motion. **Rủi ro:** hips out-of-frame, hình thể/signer shortcut, tăng dimension, mask closure khiến task quá khó. **Bác bỏ:** body thật không hơn capacity control/body-shuffled, hoặc chỉ tăng signer accuracy và geometric R² mà không tăng retrieval/LP qua signer. Khi đó xét representation/quality trước khi tăng body channels.

### S2 — Contextual JEPA + chuyển động nhiều thang; signature là nhánh bổ sung

**Input:** primary arm giữ S1/hoặc baseline input nguyên vẹn; chưa thêm signature vào student để cô lập **target effect**. Chỉ trong arm sau mới bổ sung motion descriptors. Luôn giữ hand CONF và static LOC/relations.

**Encoder:** Transformer 6×256 hiện tại; predictor masked-context d128; motion head MLP hoặc cross-attention queries cho từng interval. Nếu thêm descriptor input, dùng type-specific projection→256 và interval metadata, không đổi toàn backbone.

**Pretrained:** scratch; teacher EMA cùng model. **Target giai đoạn đầu:** wrist/index/arm-relative changes ở lag 1/4/8 samples, path length, speed magnitude và shape-change; dùng timestamps/Δt đúng với speed-resampling, valid-support mask và robust normalization fit train. Giữ signed direction cùng magnitude; không smooth mạnh làm mất chuyển động ngón nhanh. So target frame difference nguyên bản để tách lợi ích Δt normalization khỏi lợi ích multiple lags.

**Target giai đoạn signature:** selected 2D trajectories (wrist, selected tips, shoulder/elbow chains, hand-to-head relative paths), 3 interval scales đề xuất 8/16/32 samples. Degree2 trước, không all finger-pair combinatorics. Với path 2D không lift có 2 increment components + 1 area component; nếu dùng lead–lag 4D, số log components chuẩn degree2 là 10 trước convention scalar. Giữ static level/start-end features và separate timing/speed branch để không xóa holds/rhythm.

**Masking:** interval/anatomical masks với raw dependency closure qua interpolation, derivatives, overlapping scales và chain endpoints. Student token chỉ visible nếu mọi source entries cần đọc đều allowed. Global signature **không lặp vào visible local input khi đang che thời gian**. Teacher có thể nhìn full clean path; query chỉ chứa path/time/scale metadata. Target validity yêu cầu đủ observed contiguous samples; không tự nội suy băng qua gap dài.

**Loss/ép học:** `L_ctx + λ_m L_motion + λ_s L_signature`, với `λ_s=0` ở ablation đầu; motion/signature là masked interval supervision, không dựng lại mọi raw coordinate. Per-group balancing không để torso high motion lấn handshape. So contextual latent target, intrinsic deterministic target và motion-MAE ở cùng input/mask/budget. Anti-collapse của contextual branch giữ nguyên trước khi thử loss mới.

**Inference/readout:** pose-only; concat static and temporal summaries, equal-dimension projection; bỏ auxiliary heads. **Rủi ro:** estimator jitter, combinatorial tokens, mất static shape, hidden-region leakage, excessive speed invariance. **Bác bỏ:** tăng motion probe nhưng lexical metrics không tăng; gain biến mất khi dùng equal-capacity motion-MLP thay signatures; hoặc chỉ khỏe dưới synthetic fps mà hại fine handshape/minimal pairs. Khi đó chọn motion representation đơn giản hơn.

Nguồn thiết kế: [MAMP](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html), [MASA](https://arxiv.org/abs/2405.20666v1), [Path-JEPA](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf). Các chi tiết adaptation/lag/scales/loss mixing ở trên do báo cáo đề xuất.

### S3 — Graph tokenizer cục bộ + Transformer hợp nhất

**Input:** hand joints/confidence/missingness và body geometry của S1; giữ current CONF/LOC/face features như residual pathway. Không thay mọi input bằng chỉ coarse upper-body nodes.

**Encoder:** hand graph tokenizer dùng chung D/N, hai spatial message-passing blocks d128 trên 21-joint hand topology; body graph riêng cho shoulder–elbow–wrist/hip links; pooled features project lên d256 và cộng/gate với MLP tokens. Giữ 32 bins, 8 token types như S1 và Transformer 6 blocks d256/heads4. Body graph outputs và LOC có shared raw support, được khai báo dependency. Temporal Conformer là ablation sau, không đổi đồng thời tokenizer và temporal backbone ở primary arm.

**Pretrained:** scratch; graph weights không lấy từ SignSeek chỉ vì cùng topology. **Masking:** raw masking/scrubbing trước graph encoding; message passing không được truyền hidden nodes qua visible neighbors. Một graph-pooled token chỉ visible nếu không chứa forbidden sources; nếu dùng partial graph, message masks và learned absence phải rõ. So PL, anatomical tubes và uniform tubes với realized budget tương đương.

**Target/loss/ép học:** EMA contextual targets + auxiliary geometry của S1; hand-local blocks học cấu trúc liên kết ngón, global Transformer học hand–body/temporal coordination. Không ép hand/body embeddings giống hoàn toàn vì chúng chứa thông tin bổ sung. **Inference:** graph+Transformer pose-only, pooling/projection giống S1.

**Rủi ro:** graph smoothing làm mất khác biệt ngón, topology sai do pose noise, tăng parameter count. **Bác bỏ:** không hơn MLP/flat Transformer có cùng parameter/FLOPs/token count, hoặc chỉ tăng pose reconstruction. Thử graph residual off và token-preserving hand readout để xác định lỗi ở graph hay pooling.

Nguồn kiến trúc: [STC official code](https://github.com/sakura2233565548/Self-Supervised-Representation-Learning-with-Spatial-Temporal-Consistency-for-SLR), [SignSeek](https://arxiv.org/abs/2609.03695v1). Chỉ chuyển thiết kế encoder; same-gloss objectives của SignSeek không thuộc arm SSL này.

### S4 — Frozen MotionBERT làm body teacher, giữ hand encoder riêng

**Input student:** S1 hoặc S3 skeleton 2D. **Input teacher:** mapped body skeleton đủ cho H36M17, 2D+confidence theo preprocessing của checkpoint. Cache 103 joints hiện tại không có knees/ankles; muốn dùng teacher này cần quay lại raw COCO133 và audit coverage. Không tự điền legs=0 rồi xem teacher output như ground truth.

**Encoder:** student d256 PhonoJEPA/graph; frozen teacher DSTformer MotionBERT, config full d512/depth5/heads8, original maxlen243. Adapter/temporal sampling phải theo official inference; masked/distilled windows đồng bộ timestamps. Chỉ chọn valid upper-body outputs, projection head từ student→teacher feature space.

**Pretrained:** official MotionBERT pretrained model từ [repo/weights](https://github.com/Walter0807/MotionBERT); có AMASS/H36M/2D external training và 3D supervision. Đây là **external geometric teacher**, không thuộc strict same-corpus-from-scratch SSL. Chưa tải hay kiểm checksum weights trong lượt này; khi triển khai phải ghim file/hash/config.

**Masking:** student thiếu body/arm intervals; teacher full clean, không backprop. Missing input joints của teacher phải qua coverage gate/robustness audit; checkpoint không mặc định hỗ trợ mọi kiểu partial body. Không cho teacher outputs vào student input dưới dạng cache chứa target vùng bị che.

**Loss/ép học:** contextual JEPA giữ nhánh hand; auxiliary cosine/Huber match body features/relative 3D geometry ở valid outputs. Mục tiêu là geometric depth/coordination priors, không giả định pseudo-3D luôn đúng. **Inference:** chỉ student pose encoder; teacher/aux heads bỏ.

**Rủi ro/bác bỏ:** camera/domain gap, hallucinated depth, upper-body-only coverage thấp; so cùng student với 2D geometry head và random/frozen untrained teacher control. Nếu teacher quality thấp hoặc gains chỉ từ external supervision/capacity, không gọi là cải tiến JEPA thuần. Đây là nhánh sau S1/S2, không phải thay thế CONF ngay.

### S5 — RGB teacher hỗ trợ những vùng pose không quan sát tốt

**Input student:** skeleton S1/S3; **input teacher:** RGB đồng bộ, ưu tiên anatomical hand/face crops và frame reference của body. Thử một teacher ở mỗi arm.

**Encoder/pretrained:** student từ scratch d256 như baseline. Teacher lựa chọn A: frozen SignRep Hiera-B released sign representation; lựa chọn B: frozen DINOv3 ViT-B/16 cho crops, temporal teacher/projector phù hợp. SignRep không phải skeleton checkpoint; không load Hiera weights trực tiếp vào student. Không dùng current test labels hoặc same-gloss retrieval neighbors để làm SSL targets.

**Masking:** student che articulator/interval; teacher thấy clean aligned RGB. Crop vị trí được tính ngoài student nhưng không đưa crop coordinates/target feature vào visible input. Teacher target validity kiểm occlusion và detection; kết hợp body context để teacher không chỉ học appearance.

**Loss/ép học:** masked cross-modal latent distillation, giữ contextual/handshape objective và quality gates. So uniform weighting với weight theo pose reliability; normalized score không được coi là probability. Ép student khai thác context để phục hồi yếu tố mà pose extractor thiếu, nhưng **student không thể suy chắc thông tin thị giác không có trong context**, nên tránh đặt loss quá nặng lên targets bất khả suy.

**Inference/readout:** skeleton-only để giữ deployment cost; RGB teacher chỉ train. **Rủi ro:** signer/background/style transfer, temporal alignment và external data advantage. **Bác bỏ:** chỉ cải thiện cùng signer hoặc chỉ khi RGB cùng background, không hơn same-budget pose teacher, hoặc làm mất handshape robustness. Report riêng external-pretrained setting; không so như cùng dữ liệu với strict pose SSL.

Nguồn: [SignRep](https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html), [SHuBERT](https://aclanthology.org/2025.acl-long.1397/), [SignDino preprint](https://arxiv.org/abs/2609.06296v1). Các nguồn gợi ý modality/stream design, chưa thử cross-modal strategy này trên PhonoJEPA.

## 8. Masking nên thiết kế và so sánh thế nào?

Đề xuất ba family chính khi input đã cố định: **M0 PL anchor; M1 anatomical interval tubes; M2 motion/hold-balanced sampling**. M1 dùng fingers/palm, forearm/arm, torso, mouth/upper-face theo semantic ownership. M2 lấy mixture uniform/high-motion/low-motion với confidence gate; proportions là hyperparameters trên validation, không cố định bằng kết quả NTU.

Ngân sách phải báo tối thiểu:

- Fraction raw joint×frame bị che, fraction token bị loại sau closure, target count và tỷ lệ mỗi channel có target.
- Visible context còn lại sau propagation, kể cả short clips; matched budget không chỉ matched sampler input rate.
- Các masks động đọc score/motion có thể tiết lộ vị trí biến thiên qua **mask pattern**; so content-independent matched control và metadata-only predictor.
- Target observability/missingness; không đánh đồng “không được quan sát” với “bị che để học”. Confidence không quyết định mask của nhiều channels cùng lúc theo cách rò score hidden.

Đề xuất sweep nhỏ realized hidden-token rates 30/50/70% cho sign channel model trước; mask ratio theo raw joints của signature model phải báo riêng. Không gọi 20% Path-JEPA, 90% MAMP và 50–90% SLiM là cùng một đại lượng. Chọn range bằng task feasibility và minimal-pair preservation, không chọn loss thấp nhất.

Dependency rule chung cho features mới:

`visible(token) ⇔ raw_support(token) ∩ hidden_raw_entries = ∅`, trừ side-information được khai báo rõ và giữ giống giữa arms.

Support bao gồm raw joints, frames đọc bởi interpolation, derivative stencil, normalization/anchor statistics, graph receptive field và signature intervals. Existing closure theo channel đủ cho layout hiện tại theo các tests đã ghi; thêm body/relations/signature phải mở rộng dependency audit. Một perturbation test có ý nghĩa phải sửa hidden raw coordinates/scores, xác nhận student input/prediction không đổi, đồng thời sửa visible evidence phải làm output đổi để tránh test vacuous.

Anti-collapse là factor riêng: giữ centered CE ở primary comparison; sau khi có gain ổn định mới so cosine+variance/covariance, prototype targets hoặc SIGReg. Theo dõi per-stream variance/rank/entropy cùng downstream; loss giảm hoặc zero collapse alarms không đủ xác nhận representation tốt. `l1sig_noema` hiện tại chưa chứng minh EMA nên bỏ.

## 9. Ma trận thí nghiệm tối thiểu và quy tắc quyết định

Không train tất cả combination ngay. Chạy theo gates, tránh đổi input+encoder+mask+loss một lúc.

| Gate | Arms tối thiểu | Giữ cố định | Câu hỏi và quyết định |
|---|---|---|---|
| G0 measurement | Current probes vs aligned/train-only; grp vs temporal readout | Checkpoints hiện có | Tín hiệu mất ở preprocessing hay representation/readout? |
| G1 body | Baseline; +TORSO; +same-capacity control; +shuffled torso | Scratch, same steps/data/backbone/loss | UBody có bổ sung lexical evidence ngoài capacity/signer shortcut không? |
| G2 relations | G1 winner ± relation target; rồi ±REL input | Same body input/mask | Explicit relations có hơn chỉ để attention tự học không? |
| G3 motion | ctx-only; +multi-lag target; +signature target; matched motion reconstruction | Same input/encoder/budget | Target geometry cụ thể có lợi hơn latent-only hoặc reconstruction đơn giản không? |
| G4 graph | MLP tokenizer vs residual graph tokenizer | Same slots, close params/FLOPs, same recipe | Spatial inductive bias có lợi không? |
| G5 mask | PL, anatomical, mixed motion/holds, matched-uniform | Best fixed input/encoder/target | Mask nào tạo task học được, bảo toàn tín hiệu tĩnh và động? |
| G6 pretrained | Scratch; MotionBERT teacher; RGB teacher | Same student architecture | External teacher có cải thiện transferable pose representation tương xứng chi phí không? |

Baseline anchor đề xuất là PLctx và PLint **cùng 16k, seed0/1/2**, sau khi dùng P2 4k để screening; giữ 3.000 clips, batch32 và data sampling thống nhất. Các arms 16k đã có chỉ là seed0. Với warm start, báo tổng steps/examples/FLOPs từ checkpoint trước. Với graph/extra tokens, báo parameters, train wall time và inference throughput; equal steps không luôn equal compute.

Primary metrics: ASLC300 true LP macro và frozen retrieval với gallery-only transform, LOSO historical để continuity, thêm strict inductive variant. Secondary: coordinate-aligned probes, per-signer/per-gloss errors, missingness robustness, signer probe, temporal minimal pairs nếu có labels hợp lệ. Giữ split cố định; whitening/projection fit training/gallery; không chọn checkpoint/hyperparameters bằng test.

Screening seed0 chỉ chọn shortlist. Confirmation ≥3 seeds và paired signer-cluster CI, báo riêng seed variation; ít signer clusters nên không dựa p-value duy nhất. Trước train xác định practical margin; ví dụ +2 percentage points LP/retrieval là **mốc đề xuất cần thống nhất**, không phải significance threshold suy từ P1. Giữ handshape/minimal-pair quality như guardrail, không ràng buộc raw R² phải bất biến nếu thay intentional invariance.

Để nói “bỏ face/part X”, yêu cầu: retrain không X cùng budget trên nhiều seeds; main metrics không suy giảm vượt margin đã đặt; kiểm tra subsets có liên quan X và external domain. Inference masking/permute chỉ hỗ trợ chẩn đoán. Nếu chi phí là mục tiêu, báo trade-off cost–performance thay vì tuyên bố X không có thông tin.

Nếu có ASL-LEX, thêm frozen linguistic probes với lexical-family/gloss-disjoint split và signer separation; manual phonology labels chỉ dùng eval, không lẫn vào SSL training. Hiện evaluation này chưa được implementation hoàn tất, nên báo cáo không dùng geometric R² thay thế nó.

## 10. Giới hạn và kết quả của các vòng phản biện

**Checkpoint 1 — sửa tiền đề:** UBody chưa có stream riêng nhưng LOC đã có arms; probes không causal; SignSeek không label-free; tài liệu “chưa train” đã stale so với P1.

**Checkpoint 2 — tìm phản chứng:** reconstruction AMR và target-MLP GFP làm suy yếu khẳng định EMA JEPA luôn tốt nhất; ASMa/SHuBERT làm suy yếu giả định high-motion/whole-articulator masking luôn đúng. Path-JEPA full text được bổ sung từ người dùng, chuyển từ citation chưa đủ sang phân tích method/ablation; vẫn thiếu code để xác minh một số conventions.

**Checkpoint 3 — sau viết:** ưu tiên ít yếu tố thay đổi, tách target/input/pretrained effects, không đề xuất xóa face/motion từ probe âm, ghi rõ transductive protocol và chưa có evidence cho strategies mới. Audit cuối và trạng thái P2/P3 ở [nhật ký](research_update_20261011/RESEARCH_LOG.md).

Review này đọc 21 primary PDFs và 8 official repository snapshots; mức đọc khác nhau và đã ghi trong danh mục. Không chạy thêm training, không tái lập accuracy paper, không thực hiện meta-analysis giữa NTU và ASL Citizen. Dataset nhỏ, một seed local, pose 2D, thiếu frozen linguistic evaluation và code placeholder là các giới hạn thực chất. Các đề xuất S1–S5 là giả thuyết có cách bác bỏ, chưa phải method đã được xác nhận.

Tổng hợp và ba lượt phản biện do cùng một AI thực hiện tuần tự bằng skill academic-research-suite; không có reviewer độc lập. Nguồn và dữ liệu gốc được dẫn rõ; không upload human videos hay gửi thông điệp tới tác giả. Tài liệu cũ được giữ nguyên để bảo toàn lịch sử; dùng báo cáo này cùng JSON/manifest khi đọc trạng thái ngày 11/10/2026.
