# Tín hiệu skeleton nào hữu ích cho self-supervised Sign Video Retrieval, và đo như thế nào?

**Phạm vi:** literature review và thiết kế thực nghiệm, cập nhật ngày **19/10/2026**. **Tác vụ:** video-to-video sign dictionary retrieval theo SignRep; encoder frozen. Các khuyến nghị bên dưới là đề xuất nghiên cứu, chưa phải kết quả thực nghiệm mới.

## 1. Câu trả lời nghiên cứu

**Chưa có căn cứ để xếp một thứ tự phổ quát như hand > motion > face cho mọi sign.** Nên ưu tiên kiểm tra **hình dạng hai bàn tay + vị trí tương đối với cơ thể**, tiếp theo là **chuyển động và phối hợp hai tay**, rồi **thứ tự/tiến trình động tác**. Face/non-manual và context cần được đánh giá theo nhóm sign và tác vụ; điểm trung bình trên isolated dictionary retrieval có thể che mất lợi ích của chúng. Đây là thứ tự đầu tư thực nghiệm đề xuất, không phải ranking được literature chứng minh.

Để trả lời “tín hiệu tốt”, cần phân biệt bốn tầng:

1. **Có trong observation:** landmark hiện có có đo đủ tín hiệu không?
2. **Chứa thông tin phân biệt sign:** tín hiệu tự nó có giúp retrieval khác signer không?
3. **Được encoder học:** sau SSL, embedding có giữ thông tin đó không?
4. **Có ích khi kết hợp:** thêm tín hiệu có cải thiện retrieval ngoài những gì baseline đã có, với chi phí chấp nhận được không?

Một mô hình có thể đạt tầng 3 mà không đạt tầng 4. Ví dụ, linear probe đọc được velocity từ embedding không có nghĩa cosine nearest neighbour sẽ dùng velocity để xếp hạng đúng. Ngược lại, thêm velocity không tăng điểm có thể vì encoder đã suy ra nó từ tọa độ, vì derivative bị nhiễu, hoặc vì objective/readout không giữ nó.

**Khuyến nghị chính:** xây một **bản đồ giá trị của tín hiệu** trước khi ghép nhiều module. Với mỗi tín hiệu, đo đồng thời **only / add / remove / perturb / matched-control**, rồi xác nhận bằng retrieval và diagnostic độc lập.

## 2. Bám đúng bài toán của SignRep

SignRep dùng RGB lúc inference; skeleton cung cấp prior lúc pretraining. Table 4 là frozen retrieval: ASL Citizen R@1 **37,47% mean / 49,95% activity-weighted**. Table 3 có **81,37%**, nhưng là recognition sau downstream training. Table 5, WLASL stride 8: DCG của angle/kpt/dist lần lượt **45,1 / 32,9 / 47,2**, kết hợp **48,5**. Vì vậy các kết quả này hỗ trợ việc thử geometry và readout, chưa chứng minh thứ hạng giữa các skeleton input stream. [Wong et al., ICCV 2025, §3–7, Tables 3–5](https://arxiv.org/html/2503.08529v1).

### 2.1 Protocol cần khóa

Gọi embedding video là \(z(v)=\mathrm{L2}(\mathrm{pool}(f(v)))\). Với gallery clip \(G_c\) của gloss \(c\), sử dụng scorer gloss-max trong pipeline repo:

\[
s(q,c)=\max_{g\in G_c}\langle z(q),z(g)\rangle.
\]

Xếp hạng **các gloss**, không để nhiều video cùng gloss chiếm nhiều vị trí top-K. Cần đối chiếu implementation upstream khi tái tạo bảng paper; không mặc định mọi repository dùng cùng cách xử lý tie, frame lỗi và pooling. Xem [protocol local đã ghi](../../proposal11/WLASL2000_FROZEN_RETRIEVAL.md).

| Thành phần | Thiết lập chính đề xuất |
|---|---|
| Gallery | Official train, toàn bộ video khả dụng; giữ manifest cố định |
| Query khi phát triển | Official val; không dùng test để chọn module |
| Relevance | Cùng gloss/lexical entry của benchmark |
| Primary metric | Micro R@1 theo gloss ranking |
| Secondary | R@5/10, MRR, DCG, macro-gloss R@1, per-signer R@1 |
| Normalization | L2; center/PCA/whitening là arm riêng, fit trên train |
| Readout chính | Một vector, cùng chiều, cùng coverage thời gian giữa arm |
| Frame/pose lỗi | Không loại query khó khỏi denominator; báo failure riêng |
| Ties | Quy tắc xác định trước, ví dụ pessimistic rank; báo rõ |

Nếu chỉ có một gloss relevant và rank là \(r\), DCG dùng gain nhị phân có thể tính \(1/\log_2(r+1)\). Phải xác minh với scorer thực tế trước khi so paper; không đổi tên thành nDCG hoặc mAP. Clip-level retrieval với nhiều positive là benchmark phụ riêng.

Đã kiểm tra code local: [p4 scorer](../../../method/p4_signssl/p4/retrieval.py) dùng class-max và pessimistic ties; [signrep_baseline scorer](../../../method/jepa_baseline/src/signrep_baseline/retrieval.py) xử lý ties theo gallery order. Activity clamping/zero-weight fallback cũng khác. Cùng tên “SignRep protocol” chưa bảo đảm byte-for-byte parity. Đợt ablation mới phải chọn **một** contract và dùng cho mọi arm, đồng thời ghi sai khác khi so số cũ. Vì class-max có thể ưu ái class có nhiều gallery clip, báo thêm sensitivity trên gallery cân bằng số clip/gloss; giữ official gallery làm kết quả chính.

**Không đồng nhất các khái niệm:** same-gloss retrieval kiểm tra lexical identity; semantic similarity như “hai sign khác nhau nhưng gần nghĩa” cần annotation relevance khác. ASL Citizen train/test chia signer nhưng chia sẻ vocabulary; điều này **không tự tạo đánh giá unseen-gloss/open-vocabulary**.

### 2.2 Tách encoder khỏi readout

Chạy một ma trận nhỏ **checkpoint cố định × readout** trước khi đánh giá module:

- R0: temporal mean + cosine.
- R1: activity-weighted mean; activity chỉ từ pose observation, không từ nhãn.
- R2: hai temporal bins có thứ tự; giữ tổng chiều bằng R0 qua projection thống nhất.
- R3: temporal token matching, ví dụ Chamfer hoặc DTW, báo riêng số token, latency và storage.

Sau khi chọn trên val, khóa readout cho screening encoder. Chamfer là set matching, bản thân nó không giữ thứ tự token; DTW có ràng buộc thứ tự. Mean của **contextual token** vẫn có thể chứa thứ tự mà encoder đã mã hóa. Không suy ra model không hiểu thời gian chỉ vì cuối cùng có pooling.

## 3. Literature nói gì, và chưa nói gì?

Ma trận nguồn đầy đủ ở [LITERATURE_MATRIX_VI.md](LITERATURE_MATRIX_VI.md). Dưới đây là những kết nối quan trọng cho thiết kế.

### 3.1 Hand configuration và geometry: căn cứ tốt để bắt đầu

SignBERT xem hand pose là token và dùng masking/reconstruction với hand prior; SignBERT+ mở rộng hướng này. BEST tổ chức hai tay và body thành triplet rồi học discrete pseudo-target. Chúng cho cơ sở thử local hand geometry và cấu trúc đa bộ phận, nhưng trọng tâm đánh giá là recognition, không phải ablation frozen dictionary retrieval của từng tín hiệu. [SignBERT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html), [SignBERT+, TPAMI 2023](https://arxiv.org/abs/2305.04868), [BEST, AAAI 2023](https://arxiv.org/abs/2302.05075).

**Suy luận cho thí nghiệm:** nên tách handshape trong wrist/palm frame khỏi hand location trong torso frame. Nếu chỉ lấy tay làm gốc, ta dễ bỏ vị trí thực hiện sign; nếu chỉ dùng body coordinates, hình dạng ngón có thể bị lấn bởi dịch chuyển toàn tay. Hai loại tọa độ cần được kiểm tra riêng và kết hợp.

### 3.2 Motion có căn cứ trực tiếp trong sign SSL

STC học joint/motion branches và hand/trunk consistency; Table V báo WLASL2000 linear evaluation **32,35%**, so với các baseline trong bảng **18,24 / 21,19 / 24,50%**. Đây là kết quả của cả recipe. MASA kết hợp motion-aware masking, motion residual reconstruction và momentum semantic alignment. Hai hướng này hỗ trợ việc kiểm tra motion có đối chứng; chúng không xác lập velocity là stream tốt nhất cho retrieval. [STC, §III–IV, Table V](https://arxiv.org/html/2406.10501v1), [MASA, TCSVT 2024, §III–IV](https://arxiv.org/html/2405.20666v1).

**Suy luận cho thí nghiệm:** thử velocity làm input và làm target là hai câu hỏi khác nhau. Đừng gộp chúng trong một arm; kiểm tra derivative đã hiệu chỉnh timestamp, smoothing và missingness trước.

### 3.3 Multi-stream/context tốt cho transfer, nhưng không phải mọi stream đều là skeleton

SHuBERT dự đoán cluster cho bốn stream: hai hand crop, face crop qua DINOv2, và body pose. Nó cho cơ sở học contextual target theo stream; không phải bằng chứng cho một encoder chỉ dùng landmark mặt/tay. Frozen feature + head được train cũng khác training-free nearest-neighbour retrieval. [SHuBERT, ACL 2025, §2–4](https://arxiv.org/html/2411.16765v3).

**Suy luận cho thí nghiệm:** context cần được kiểm tra bằng target dài hạn, cắt context và lexical holdout, không chỉ bằng tăng temporal window. Pseudo-cluster là latent unit, chưa có nghĩa là phoneme/morpheme có khả năng tổ hợp.

### 3.4 Retrieval trực tiếp: SignSeek hữu ích nhưng có gloss supervision

SignSeek dùng graph theo articulator, temporal fusion, gloss contrastive alignment và saliency-guided masking. ASL Citizen R@1 **57,66%**; random articulator selection **52,22%**. GAC-only **48,40%**. Các ablation này trực tiếp liên quan retrieval, nhưng model pretrain trên **266.915 clip có gloss label**; không dùng nó làm baseline SSL thuần ngang supervision. MASA **27,96%** trong Table 2 là số được nhóm SignSeek báo lại/đánh giá, không phải một số frozen retrieval được xác minh từ paper MASA gốc. [SignSeek v1, §§3–4, Tables 1–2, 7–9](https://arxiv.org/html/2609.03695v1).

**Suy luận cho thí nghiệm:** thử random articulator masking trước. Chỉ thêm saliency/gating khi đã chứng minh các stream có giá trị khác nhau theo sign; chọn stream bằng attention hoặc gradient không đủ để gọi là “tín hiệu ngôn ngữ quan trọng”.

### 3.5 Face: cần đánh giá có điều kiện

CoSign dùng group-specific GCN và static/dynamic streams trong supervised continuous recognition. Một nghiên cứu facial regions cho thấy mouth hữu ích trong setting RGB isolated recognition của họ; không chuyển trực tiếp kết luận sang mouth landmarks. [CoSign, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.html), [Pham & Avramidis, IVA Adjunct 2025](https://arxiv.org/abs/2507.20884).

**Suy luận cho thí nghiệm:** tách mouth, brows/eyes và head pose; đo riêng nhóm sign có non-manual annotation. Mặt cũng làm anchor để định vị tay, nên phải tách **face expression** khỏi **face reference geometry**.

### 3.6 Composition: reconstruction tốt hơn chưa đủ

VQ-ASL báo OOV ISR-probe MRR **.381 baseline / .372 parameter disentanglement / .452 full model**; full model thêm phonological supervision. Tách stream không tự bảo đảm generalization tốt hơn. [Kezar, Sehyr & Thomason, 2025, Table 3](https://arxiv.org/html/2509.04745v1).

Phonological Perception đánh giá minimal pairs; tập được tuyển chọn gồm **259 cặp ASL Citizen**, **178 cặp Sem-Lex**. Nghiên cứu ghi nhận trade-off theo kiến trúc giữa các tham số âm vị học. Đây là nguồn phù hợp cho diagnostic hơn một bảng accuracy tổng. [Yin et al., CogSci 2026, §2–3](https://arxiv.org/html/2606.28667v1).

**Suy luận cho thí nghiệm:** gọi một latent “semantic” hoặc “compositional” phải đi kèm kiểm tra tổ hợp chưa thấy/lexeme chưa thấy. Group tokenization hay codebook đa tầng tự chúng chưa chứng minh claim đó.

### 3.7 Cập nhật 2026 cần phân biệt tên và phiên bản

- **SignDINO, Gan et al., CVPR 2026:** teacher global frame, student local views giữ hands/face; downstream chính là gloss-free translation. [Nguồn CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html).
- **SignDino v1 → SignDEG v2, Hu et al., arXiv 2609.06296:** v2 ngày 15/10/2026 thêm geometric latent prediction và geometric relation transfer từ hand reconstruction teacher. Là RGB/crop input, geometry target; các recognition result có downstream training không được gán thành SignRep-style frozen retrieval. [SignDEG v2](https://arxiv.org/html/2609.06296v2).
- **SSL-SLR, TMLR 02/2026:** negative-free objective và augmentation tập trung temporal regions. Không mặc định mọi augmentation trong paper đều giữ lexical identity trên mọi sign. [OpenReview chính thức](https://openreview.net/forum?id=buTZkTXijy).

### 3.8 Skeleton action SSL: nguồn ý tưởng, cần kiểm tra lại trong sign

MAMP thử motion target và motion-guided masking; CrosSCLR khai thác nhiều skeleton views; Skeleton2vec và S-JEPA học contextual latent target. Đây là đối chứng objective phù hợp. Kết quả trên NTU/PKU với body skeleton không chứng minh hiệu quả trên fine-grained fingers, face hoặc sign retrieval. [MAMP, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html), [CrosSCLR, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Li_3D_Human_Action_Representation_Learning_via_Cross-View_Consistency_Pursuit_CVPR_2021_paper.html), [Skeleton2vec](https://arxiv.org/abs/2401.00921), [S-JEPA, ECCV 2024](https://sjepa.github.io/).

## 4. Taxonomy tín hiệu: định nghĩa để có thể ablate

**Quy tắc:** bộ phận cơ thể, loại feature, độ dài context và SSL objective là các trục khác nhau. Ví dụ “right hand” không cùng loại với “velocity”; phải ghi “velocity của wrist phải trong torso frame”. Bảng dưới là thiết kế đo đề xuất.

| ID | Tín hiệu | Cách biểu diễn ban đầu | Diagnostic cần đi cùng | Ưu tiên |
|---|---|---|---|---|
| H | Hand configuration | 21 joint/tay, wrist-relative; bone unit vectors hoặc góc ngón | Handshape/flexion/spread minimal pairs | P0 |
| L | Location | Wrist/tip tương đối shoulders, chest, nose/head | Cặp gần handshape, khác location | P0 |
| O | Orientation | Palm basis, directed bone vectors; 3D nếu có estimator đáng tin | Orientation pairs; lỗi theo view | P1 |
| M | Dynamics | Velocity đa khoảng thời gian, direction + speed tách riêng | Movement pairs; fps/speed perturbation | P0–P1 |
| B | Bimanual relation | Vector LH↔RH, relative velocity, synchrony/lag | One/two-handed, symmetry, alternation | P1 |
| F | Non-manual | Mouth shape, brows/eyes local-frame, head motion | Nhóm non-manual có nhãn độc lập | P1 có điều kiện |
| T | Temporal structure | Thứ tự, repetition, hold/move transitions, duration | Order/repetition-specific pairs | P1 |
| C | Hierarchy/composition | Joint→part→segment tokens; local→global objective | Held-out phonological combinations | P2 |
| K | Contextual latent | Masked contextual targets, multi-stream units | Cắt context; lexical holdout; transfer | P1–P2 |
| Q | Observation quality | Missing mask, detector score, track continuity | Score-only baseline, quality-stratified gain | Control bắt buộc |

### 4.1 Những quyết định preprocessing có thể làm mất sign content

- **Torso frame:** dùng shoulder midpoint và một scale ổn định theo clip. Chuẩn hóa dịch chuyển camera/scale; giữ lại vị trí tay với mặt/thân.
- **Hand local frame:** wrist-relative là bước đầu. Palm-frame alignment chỉ ở arm riêng vì phép xoay chuẩn hóa có thể xóa orientation có nghĩa. Giữ transform local→body ở stream riêng nếu cần cả hai.
- **Left/right:** physical side và dominant/non-dominant là hai biểu diễn. So sánh chúng; không giả định tay chuyển động nhiều hơn luôn là dominant. Báo lỗi heuristic. Không dùng frame tương lai ngoài clip khi mục tiêu là online inference.
- **2D/3D:** 2D không quan sát đầy đủ palm depth/orientation. 3D monocular là estimated target có sai số, không phải ground truth; so cùng estimator coverage và missingness.
- **Mirror:** hoán vị joints, side và vectors đồng bộ; không dùng mirror của riêng một bộ phận làm positive mặc định. Cần kiểm tra lexical validity.
- **Motion:** tính sau xử lý camera/scale nhưng kiểm soát anchor jitter; không cho biến thiên normalization theo frame tạo chuyển động giả.
- **Face:** head-relative expression và body-relative head location là hai feature riêng. Bỏ expression không được vô tình bỏ head anchor.

Với timestamp thật \(t_i\), dùng:

\[
v_i^{(\tau)}=\frac{x(t_i+\tau)-x(t_i-\tau)}{2\tau},
\quad d_i=\frac{v_i}{\|v_i\|+\epsilon},\quad u_i=\|v_i\|.
\]

Kiểm tra \(\tau\) theo **giây**, chẳng hạn 0,08/0,16/0,32s nếu fps hỗ trợ. Mask derivative khi endpoint thiếu hoặc gap quá dài; không đặt velocity tại gap bằng zero như một hold thật. Acceleration là arm muộn hơn vì khuếch đại nhiễu. Recompute derivative sau temporal warp với timestamp mới, hoặc nói rõ target là phase-normalized displacement, không gọi nhầm velocity vật lý.

### 4.2 Feature tất định vẫn có giá trị

Bone, angle, velocity và pairwise relation là hàm của tọa độ quan sát. Chúng không tự tăng lượng thông tin quan sát theo nghĩa information-theoretic, nhưng **có thể đổi inductive bias, sample efficiency, optimization và robustness**. Vì vậy không loại explicit geometry chỉ vì “nó đã nằm trong input”. Ngược lại, target từ teacher RGB/3D chứa prior bên ngoài; phải ghi nguồn pretrained supervision và không gọi là pose-only pretraining.

## 5. Dataset nào trả lời câu hỏi nào?

### 5.1 Bộ dữ liệu đề xuất

| Dataset/tài nguyên | Vai trò | Tín hiệu kiểm tra tốt | Giới hạn và nguồn |
|---|---|---|---|
| ASL Citizen | Train SSL trên train; val chọn arm; test xác nhận | Lexical retrieval khác signer; link ASL-LEX | 83.399 video, 2.731 signs, 52 signers; vocab chung giữa split. [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f29cf8f8b4996a4a453ef366cf496354-Abstract-Datasets_and_Benchmarks.html) |
| ASL-LEX 2.0 | Annotation cho probe/slice, không đưa vào SSL loss | Handshape, location, movement và thuộc tính từ vựng | Join theo ID/variant, không chỉ English string. [Trang gốc](https://asl-lex.org/) |
| Sem-Lex | Cross-corpus diagnostic và lexical transfer | Phonology, khác cách elicitation | Có alignment với tài nguyên ASL; kiểm tra overlap lexeme/source trước claim. [Paper](https://arxiv.org/abs/2310.00196) |
| WLASL2000 | Cross-dataset retrieval theo thiết lập SignRep | Thay đổi view/source, vocabulary | Công bố không bảo đảm mọi URL còn tải được; thống kê bản local. [WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.html) |
| MS-ASL | Replication cross-corpus bổ sung | Signer/view variability | Audit video ID/timestamp overlap với WLASL/YouTube pretrain. [Paper gốc](https://arxiv.org/abs/1812.01053) |
| How2Sign | Pretrain continuous context; một phần kiểm tra 3D/multiview | Transition/coarticulation/context | >80 giờ, có multiview/depth; sentence split không tự tạo dictionary relevance. [CVPR 2021 / project](https://how2sign.github.io/) |
| YouTube-ASL | Scale SSL cùng ngôn ngữ ASL | Nhiều signer, continuous dynamics | 984 giờ; bản local ghi trong repo trước đây chỉ có video IDs. [Nguồn gốc](https://arxiv.org/abs/2306.15162) |
| YouTube-SL-25 | Scale đa ngôn ngữ, liên hệ SignRep | Domain/language transfer | >3.000 giờ; không gộp gloss khác ngôn ngữ thành cùng class theo English word. [ICLR 2025](https://arxiv.org/abs/2407.11144) |
| SignVerse-2M | Ứng viên pose-native scale sau pilot | So dữ liệu lớn cùng preprocessing | v3: khoảng 2M clips, 55+ languages, DWPose; phải xác nhận release/coverage/overlap thực. [v3](https://arxiv.org/html/2605.01720v3) |

PHOENIX/CSL-Daily hữu ích cho continuous transfer và non-manual/context nhưng không phải ưu tiên đầu khi câu hỏi chính là isolated ASL retrieval. Không gán relevance cho sentence pair bằng gloss equality đơn giản.

### 5.2 Audit local thực hiện trong lượt này

Đã đọc **CSV metadata**, chưa đọc lại video hay đo độ chính xác detector. Chi tiết và hashes ở [local_dataset_audit.json](local_dataset_audit.json).

| ASL Citizen local | Video rows | Signers | Glosses | ASL-LEX codes |
|---|---:|---:|---:|---:|
| Train | 40.154 | 35 | 2.731 | 2.723 |
| Val | 10.304 | 6 | 2.731 | 2.723 |
| Test | 32.941 | 11 | 2.731 | 2.723 |

Signer sets rời nhau; glossary trùng hoàn toàn. Val signer **P13 chỉ có 2 video**; P5/P12 có **609/697**, P39/P21/P26 có **3.000/2.997/2.999**. Vì thế 10k query không tương đương 10k mẫu độc lập cho suy luận về signer. Một ASL-LEX code có thể tương ứng nhiều gloss local: probe split phải group theo lexeme/code và xử lý variant, tránh lexical leakage qua mã dùng chung.

Code hiện tại đọc COCO-WholeBody **133 joints**, rồi chọn **103 joints = body9 + face52 + LH21 + RH21**. Có pipeline cũ dùng confidence gate và pipeline mới giữ điểm hữu hạn với observation filtering khác. So arm dùng các cache đó mà không khớp preprocessing sẽ không cô lập tín hiệu. Nguồn: [joint map](../../../method/phonopath_jepa/pp/base/joints.py), [data adapter](../../../method/phonopath_jepa/pp/data.py).

**Audit cần bổ sung trước train:** khoảng missing của từng part; track switch; wrist disagreement body↔hand; frame timestamps; nguồn confidence và scale; chất lượng theo contact/occlusion; tỷ lệ lexical classes mất pose. Detector score chỉ là score, chưa phải xác suất đúng. Sample video kiểm tra thủ công chỉ lấy train/dev, không chọn hard subset bằng test.

## 6. Bằng chứng nội bộ và các kết luận cần tránh

Đọc lại [tổng hợp official test 19/10/2026](../../experiments/method_comparison_20261019/REPORT_VI.md), không rerun model:

| Method/readout | ASL Citizen test R@1 | Diễn giải được |
|---|---:|---|
| PLctx full160, raw grp | 16,678% | Anchor skeleton hiện có |
| PhonoPath A1 full160, raw grp | 11,511% | Thấp hơn anchor trong cấu hình này |
| PhonoPath A1 full160, grp+P | 6,928% | Readout chính của run, có PCA; khác raw |
| Pose-JEPA Hybrid raw / white512 | 5,744 / 9,317% | Postprocessing ảnh hưởng đáng kể |
| SignRep public weighted, reproduced | 49,020% | RGB + external pretraining, setting khác |
| SignRep + Chamfer20 | 58,420% | Khác scorer/storage với một vector |

Các số là **internal recorded results**, không phải số mới được xác nhận bằng checkpoint trong lượt này. Một số checkpoint/cache đã được archive hoặc xóa; không giả định có thể trích lại stream từ chúng.

Ba kết luận hợp lệ:

1. SSL loss giảm chưa đủ để chọn checkpoint cho retrieval.
2. Cần tách hiệu ứng readout/preprocessing khỏi hiệu ứng encoder/objective.
3. Cần A0/A1 đối chứng cùng code trước khi quy kết cho log-signature hay bất kỳ loại motion nào.

Ba kết luận **chưa hợp lệ**:

- “Motion/relations không có ích” chỉ vì một family run thất bại.
- “Skeleton không phải bottleneck” chỉ vì có supervised pose model mạnh ở paper khác. Phải thử reference có nhãn với **cùng input và scorer**. Đây là empirical reference, không phải upper bound toán học.
- “Probe R² âm chứng minh tín hiệu không tồn tại”: linear probe, estimator target và lexical split đều có thể giới hạn phép đo.

## 7. Chương trình thực nghiệm để xác định tín hiệu

### 7.1 Vòng A — sửa phép đo và kiểm tra observation, chưa tối ưu method

**A0. Khóa data/evaluator.** Fingerprint manifest, splits, feature schema, normalization, timestamps, crop coverage, dimension, gallery và failure policy. Nhãn gloss không được sampler SSL dùng để tạo positive/negative theo class. Nhãn chỉ xuất hiện trong downstream evaluation và supervised reference riêng.

**A1. Baseline không SSL.** Raw normalized hand/body descriptors, mean + variance và sequence/DTW baseline; random-init encoder. Descriptor có dimensionality lớn cần bản projection/PCA cùng chiều fit train. Mean descriptor không đại diện cho toàn bộ khả năng của tín hiệu động.

**A2. Baseline readout.** Ma trận R0–R3 ở §2.2 trên checkpoint còn tồn tại; nếu phải train lại, ghi rõ là run mới. Cùng model, gallery và temporal coverage.

**A3. Quality-only và nuisance-only control.** Chỉ score/missingness, chỉ duration, chỉ bone-length/static proportion. Nếu các control này retrieval bất thường, kiểm tra collection/pose shortcuts. Signer probe trên tập phụ có signer lặp lại giữa probe-train/probe-val; main retrieval signer-disjoint không dùng để đo same-signer neighbour vì gallery không có signer của query.

**A4. Supervised reference có giới hạn.** Cùng observation/architecture train recognition hoặc metric head bằng gloss train, chọn val, rồi frozen retrieval. Đây là cách kiểm tra pipeline có dùng được tín hiệu hay không; không nhập kết quả vào bảng SSL. Nếu nó yếu, kiểm tra input/evaluator/capacity; chưa thể kết luận information ceiling.

### 7.2 Vòng B — bản đồ giá trị từng tín hiệu

Với signal set \(S\), gọi \(R(S)\) là retrieval metric của model **được train với S** dưới protocol cố định. Tách ba estimand:

\[
\text{Sufficiency}_i=R(\{i\}),\quad
\text{Increment}_i=R(B\cup\{i\})-R(B),\quad
\text{Removal}_i=R(A)-R(A\setminus\{i\}).
\]

Đây là model comparisons trong điều kiện thí nghiệm, không phải causal effect của bộ phận cơ thể trong ngôn ngữ tự nhiên. Với frozen ablation, dùng notation khác \(R_f(A)-R_f(\mathrm{corrupt}_i(A))\); không đồng nhất nó với retraining removal.

**Bộ stream ban đầu:** physical LH, RH, upper body/arm location, face. Với bốn group chỉ có 16 subset. Có thể đo đủ 16 ở budget screening nếu đủ compute; bản tiết kiệm dùng full, 4 only, 4 leave-one-out, hands-only, hands+body — bỏ arm trùng nhau. Giữ reference anchors rõ trong mọi arm, vì “hand-only” dùng body scale thực ra là hand + static anchor.

**Sau đó tách feature với baseline hands+location:**

| So sánh | Câu hỏi | Matched control |
|---|---|---|
| +bone/angle | Explicit local geometry có giúp học nhanh hơn? | Cùng số token/params; duplicate-joint projection |
| +velocity | Explicit motion có giá trị thêm? | Nhánh cùng capacity nhận position thay motion |
| +direction vs +speed | Đường chuyển động hay tốc độ đang hữu ích? | Cùng normalization và missing mask |
| +LH↔RH relation | Cần quan hệ hai tay hay chỉ thêm capacity? | Non-interacting matched two-stream model |
| +hand↔head/body relation | Location rõ ràng hơn có giúp? | Giữ head anchors; không xóa local handshape |
| +mouth / +brows / +head dynamics | Nhóm non-manual nào hữu ích? | Capacity/token-count matched; reference geometry giữ nguyên |

Tín hiệu nên được kiểm tra bằng **cùng một objective đơn giản** trước. Chỉ khi đã chọn được một vài feature triển vọng mới đổi objective; không chạy tích Descartes tất cả signals × objectives × masks ngay từ đầu.

### 7.3 Vòng C — phá có chủ đích để biết model dùng gì

Đánh giá frozen **cả query-only corruption và query+gallery corruption**; cái thứ nhất đo robustness với mismatch, cái thứ hai kiểm tra signal trong cùng domain biến đổi. Giữ gallery sạch là primary; cả hai báo riêng.

| Intervention | Cái bị phá / cái cố giữ | Cách tránh kết luận sai |
|---|---|---|
| Part dropout | Bỏ một stream | Dùng missing token đã thấy khi train; retrain-minus để xác nhận |
| Temporal mean pose lặp | Phá dynamics, giữ mean pose | Có distribution shift; ghép với static-trained control |
| Shuffle frame/block | Phá order/continuity ở nhiều scale | Recompute velocity; không giữ derivative của chuỗi gốc |
| Reverse | Đảo hướng/thứ tự | Không yêu cầu giảm trên mọi sign; focus direction-sensitive pairs |
| Mild monotonic time warp | Thay speed, cố giữ lexical sequence | Cần kiểm tra semantic validity và đủ temporal coverage |
| LH/RH temporal desync | Phá phối hợp, giữ marginal trajectories | Đo theo lag giây; bỏ boundary/padding artifact |
| Hand location translation | Phá body-relative location, giữ local shape | Vị trí mới có thể không hợp lệ; chỉ claim model sensitivity |
| Local-frame shape perturbation | Phá handshape, cố giữ path | Dùng perturbation nhỏ có kiểm soát, không coi pose giả là sign thật |
| Face expression neutralization | Phá expression, giữ head geometry | Trung hòa từ train stats; kiểm tra landmark artifacts |
| FPS/jitter/occlusion | Phá observation quality | Cường độ theo body/hand scale, gap theo giây |

**Không xếp tín hiệu theo mức tụt do zero-out đơn thuần.** Tụt nhiều có thể vì input OOD; tụt ít có thể vì tín hiệu dư thừa. Attribution, attention, gradients và gates chỉ giúp sinh giả thuyết. Phép xác nhận là matched retraining và diagnostic độc lập.

Với derived feature, phải áp closure: mask joint nguồn thì mask angle/bone/velocity/relation liên quan; smoothing/interpolation không được dùng hidden endpoint để điền lại visible input. Nếu không, masked modeling có thể nhìn thấy target qua feature dẫn xuất.

### 7.4 Vòng D — chọn SSL objective, mask và target cho tín hiệu đã có ích

Chỉ so trên 1–2 input recipes đã chốt, cùng encoder/readout:

| Arm | Cơ chế | Control cần có |
|---|---|---|
| O0 masked coordinate reconstruction | Baseline generative | Random mask; loss normalized theo group |
| O1 masked motion prediction | Ép học dynamic target | Cùng mask với O0 trước; sau đó mới đổi masking |
| O2 masked EMA latent prediction | Target contextual, giảm ép tái tạo tọa độ nhiễu | Cùng target layers/temperature schedule; monitor collapse |
| O3 clustered-unit prediction | Multi-stream pseudo-target | K-means chỉ fit train; codebook usage/stability, same capacity |
| O4 clip-level negative-free/contrastive | Metric geometry và invariance | Positive augmentation có giữ sign; false negatives phân tích hậu kiểm bằng nhãn |

Không so raw SSL loss giữa các objective. Predictor/projection head có thể làm metric tốt ở một layer và xấu ở layer khác: chọn representation layer trên val với budget tuning ngang nhau, rồi khóa.

Sau đó mới so **random mask vs part mask vs temporal tube vs motion-aware mask** với cùng tỷ lệ và span coverage. Motion-aware phải đối chứng sampling theo confidence, vì vùng motion lớn cũng có thể là detector jitter. Teacher/student dùng input/target visibility contract được ghi rõ.

**Context test:** cùng sequence length, so local attention với global attention; cùng compute gần tương đương. Tăng window phải có control tăng số frame nhưng không tăng receptive field. Long-context chỉ có ích cho retrieval nếu readout giữ sign-relevant state, không chỉ thêm rest/context.

### 7.5 Vòng E — tương tác, composition và scale

- Chọn tối đa **3 module**, chạy **2³ = 8 cấu hình**, đo main effects và pairwise interactions. Nếu hai module dư thừa, không yêu cầu cả hai cùng giữ.
- Từ tổ hợp tốt nhất chạy retrained leave-one-out, 3 seed; dùng cùng budget với tổ hợp đầy đủ.
- Chạy data scaling và update scaling tách nhau: thay số clip với số update cố định kiểm tra data diversity; thay update với data cố định kiểm tra optimization. Báo thêm compute/epochs vì không thể đồng thời giữ mọi biến.
- **Lexical holdout:** chia theo ASL-LEX code/variant family trước SSL; chỉ train trên train lexical subset. Unseen-gloss gallery được đưa vào lúc evaluation, không vào pretraining/tuning. Đây mới là test unseen lexeme rõ ràng.
- **Composition:** chọn các feature combination có từng thành phần đã xuất hiện trong train nhưng tổ hợp chưa có; group theo gloss và signer. Annotation thiếu/không độc lập phải được báo; không gọi split tổ hợp nếu thực chất chỉ là random video split.
- **Context semantics:** continuous dataset là track riêng; đánh giá window/context shuffle và downstream task tương ứng. Không hứa isolated retrieval đo được discourse semantics.

## 8. Diagnostic và thống kê

### 8.1 Ba nhóm diagnostic

**Linguistic:** minimal-pair triplet accuracy \(\mathbb{1}[s(q,g^+)>s(q,g^-)]\) với positives cùng lexical entry và hard negative khác một thuộc tính đã được kiểm chứng; lấy khác signer, match chất lượng và số gallery. Report từng handshape/location/movement/orientation/non-manual slice, đủ sample counts và CI. Một cặp chỉ khác trong database chưa chắc đúng minimal pair trong mọi video variant; cần người có chuyên môn ASL kiểm tra các trường hợp mơ hồ.

**Representation:** linear/ridge probe và một nonlinear probe nhỏ để phân biệt linear accessibility với absence; train probe theo lexical/signer grouping, fit normalization trên probe-train. Nhãn phonology dùng để chọn module là **supervised model selection**, dù loss pretrain không dùng nhãn. Phải khai báo điều đó. Probe dự đoán derivative từ cùng pose chỉ đo fidelity với measurement, không phải bằng chứng độc lập về nghĩa.

**Geometry/robustness:** effective rank, cosine distribution, duplicate/zero vectors, alignment, calibration score, missingness sensitivity, signer prediction. Dùng phát hiện collapse và shortcut; không lấy chúng thay R@1. Same-signer nuisance probe cần pool chẩn đoán có cùng signer hai phía và cân bằng gloss.

### 8.2 Thiết kế thống kê khả thi trên ASL Citizen

- Primary comparison dùng **cùng query**, lưu rank từng query và seed. Báo delta theo **điểm phần trăm**, không nhầm relative percent.
- 1 seed chỉ screening. Confirmation tối thiểu **3 paired seeds**; cùng split, sample order/augmentation seeds khi có thể, báo từng seed và mean±SD. Ba seed vẫn cho ước lượng training variance yếu.
- CI conditional-on-checkpoint: paired resampling theo signer và gloss; có thể dùng crossed cluster bootstrap với trọng số signer×gloss và renormalize metric. Báo thêm cluster theo từng trục và leave-one-signer-out stability. Không giả định query i.i.d.
- Val chỉ có 6 signer, một signer 2 clip: CI theo signer dễ bất ổn. Báo macro-signer gồm tất cả, rồi sensitivity analysis trên nhóm có đủ query, không âm thầm bỏ P13. Dùng signer-holdout nội bộ từ train để kiểm tra stability; không gộp nó thành official val score.
- CI query-only hẹp không chứng minh generalization sang signer population. Nên tách uncertainty do seed và do evaluation clusters; không cộng SD/CI tùy tiện.
- Holm correction cho các primary contrasts đăng ký trong một family. Screening là exploratory; confirmation dùng cấu hình đã khóa và báo toàn bộ arm, không chỉ winner.

**Luật chọn module đề xuất:** đặt practical margin \(\delta\) trước khi xem kết quả, ví dụ **+1,0 điểm R@1** nếu chi phí tương đương. Đây là ngưỡng vận hành ban đầu, cần hiệu chỉnh bằng noise pilot, không phải chuẩn literature.

| Kết quả | Quyết định |
|---|---|
| Gain ổn định qua seed, CI hỗ trợ dương, effect đủ lớn, cost hợp lý | Giữ, xác nhận full budget |
| Diagnostic đúng giả thuyết nhưng primary gain chưa rõ | Giữ làm nghiên cứu cơ chế; chưa ghép vào final method |
| CI rộng hoặc đảo giữa signer/seed | Chưa kết luận; tăng replication hoặc đơn giản hóa |
| CI nằm trong equivalence margin, hoặc có hại ổn định | Bỏ phiên bản module này |

Không gọi “không significant” là “không có tín hiệu”. Muốn kết luận gain vượt \(\delta\), lower confidence bound phải vượt \(\delta\); gain point estimate >\(\delta\) với CI chỉ >0 là claim yếu hơn.

## 9. Quy tắc từ bằng chứng → module

| Kết quả quan sát | Module nên thử | Điều cần loại trừ trước |
|---|---|---|
| Local handshape tốt, location pairs yếu | Dual local/body frames; hand↔body relation tokens | Wrist normalization đã xóa location? |
| Location tốt, ngón yếu | Hand-specific spatial encoder, shared LH/RH weights với side transform | Hand detector có đủ độ phân giải/coverage? |
| Motion-only hữu ích, joint model movement yếu | Velocity branch hoặc motion target | FPS, jitter, derivative gap và capacity |
| Bimanual desync gây lỗi có chọn lọc | Cross-hand attention/relative motion, lag-aware features | Artificial boundary artifact |
| Order pairs yếu dù motion probe tốt | Temporal positional/relative bias; multi-scale ordered readout | Readout set-based đang bỏ thứ tự? |
| Face chỉ giúp non-manual subset | Separate face-expression branch, calibrated late fusion | Expression gain hay head-location gain? |
| Nonlinear probe tốt, cosine retrieval yếu | Pooling/projector/metric objective | Dimension/whitening/layer choice |
| Supervised matched reference tốt, SSL yếu | Đổi objective/positive views/target | Budget/tuning mismatch |
| Mọi matched reference đều yếu ở occlusion | Quality-aware masking, extractor/3D audit | Đừng thêm temporal module để chữa missing data |
| Thêm stream làm giảm average nhưng giúp minority | Late fusion/conditional computation được kiểm chứng | Metadata shortcut, worst-group harm |
| Học được attributes nhưng OOV combination kém | Compositional objectives/codebooks có control | Lexical split/variant leakage và label supervision |

**Kiến trúc khởi đầu đề xuất, chưa phải winner:** encoder tay dùng chung cho local geometry, body-relative location stream, temporal encoder đơn giản, một vector retrieval. Motion là một biến thể được so riêng. Face-expression, cross-hand relation, hierarchy và adaptive gating chỉ thêm theo kết quả. Giữ baseline nguyên trạng để đo add-one và retrain-minus.

## 10. Budget và thứ tự chạy thực tế

Không dùng GPU-hour cũ trong proposal như ước lượng hiện tại. Trước tiên profile **200–500 optimizer updates sau warmup**, extraction 1k clip và scoring toàn val; đo peak VRAM, clips/s, sec/update, storage. Công thức sơ bộ:

\[
\mathrm{GPUh}_{train}=\sum_a\frac{n_{seed,a}\,U_a\,t_{update,a}}{3600};
\quad \mathrm{total}=\mathrm{train}+\mathrm{extract}+\mathrm{score}+\mathrm{diagnostics}.
\]

| Vòng | Số cấu hình/train đề xuất | Budget/seed đề xuất | Điều kiện đi tiếp |
|---|---:|---:|---|
| A: metadata, descriptors, readouts | 0 SSL run mới; 1–2 supervised references nếu cần | Theo profiler | Evaluator và coverage hợp lệ |
| B: stream/input screening | Tối đa 12 cấu hình, 1 seed | 10k updates | Giữ nhiều family; báo uncertainty |
| D: objective screening | 5 cấu hình trên input đã chọn, 1 seed | 10k updates | Có gain trên full-vocab val |
| Confirmation | 4 cấu hình gồm baseline, 3 seed | 40k updates | Gain/diagnostic/cost ổn định |
| Interaction | 8 cấu hình, 1 seed | 40k updates | Tương tác đủ rõ để chốt candidate |
| Final + retrain-minus | Full candidate và tối đa 3 leave-one-out, 3 seed | Budget khóa theo learning curve | Val ổn định; test batch cuối |

Các mức 10k/40k là **đề xuất**, không phải hyperparameter đã chứng minh. Với batch32 và 40.154 clip, 10k updates tương đương khoảng 8 lần qua dữ liệu nếu sampler đơn giản. Khớp số update/batch nhưng báo thêm FLOPs và wall time vì multi-stream/multi-view đắt khác nhau.

**Bản tiết kiệm:** A → 6 input arms → 3 objective arms → baseline và 2 candidate ×3 seed → final/LOO. Không train tất cả 16 subsets nếu profiler vượt budget. Pilot vocabulary nhỏ chỉ để debug; không dùng nó làm tiêu chuẩn duy nhất chọn winner cho full 2.731 gloss.

### 10.1 Đợt đầu cụ thể nếu bắt đầu triển khai

Sau vòng A, đề xuất **8 cấu hình, mỗi cấu hình 1 seed ×10k updates**, cùng một objective, backbone, readout, dimension và toàn bộ vocab. Đây là một initial batch ưu tiên trong không gian arm của CSV, không cộng thêm vào mọi nhánh trong kế hoạch đầy đủ.

| Cấu hình | Input | Câu hỏi trực tiếp |
|---|---|---|
| S0 | H: hai tay wrist-relative, static scale anchors khai báo | Local hand configuration đủ đến đâu? |
| S1 | L: wrist/arm trajectories trong torso frame | Path/location không có chi tiết ngón đủ đến đâu? |
| S2 | H + L | Baseline kết hợp shape và location |
| S3 | S2 + M | Velocity đa khoảng thời gian có incremental gain? |
| S4 | S2 + B | Relative bimanual features có incremental gain? |
| S5 | S2 + mouth-expression | Mouth landmark dynamics giúp toàn bộ hay một subset? |
| S6 | S2 + brows/eyes/head dynamics | Non-manual ngoài mouth có ích ở đâu? |
| S7 | S2 + explicit bone/angle | Geometry feature engineering giúp ngoài raw H+L? |

H và L đều là **chuỗi theo thời gian**, nên S0/S1 không phải static-only baseline. S3 cần capacity-matched position-branch control trước khi claim gain riêng từ velocity. S5/S6 giữ head/body reference geometry thống nhất; không để lợi ích định vị tay bị gán nhầm cho expression. Dùng batch này để chọn **câu hỏi tiếp theo**, không tuyên bố winner chỉ từ một seed. Sau đó xác nhận S2 và tối đa hai candidate bằng 3 seed, thêm retrained removal và matched controls tương ứng; chỉ khi đó mới đầu tư objective/hierarchy/gating.

## 11. Tính toàn vẹn của evaluation và claim

- Official tests trong repo đã được truy cập ở các nghiên cứu trước. Vì vậy lần test sắp tới chỉ là **khóa quyết định trong chương trình mới**, không thể mô tả là một dataset chưa từng được nhìn. Nếu cần independent confirmation mạnh hơn, dùng corpus/signer cohort mới hoặc split thực sự chưa được sử dụng.
- Không dùng các minimal pairs thuộc official test để chọn module; dành chúng cho đánh giá cuối. Dev diagnostics cần tạo từ train/val hoặc tài nguyên độc lập phù hợp.
- Không trộn label-free pretrain với gloss-positive, phonology-supervised loss, text pairing hay signer-adversarial loss có ID mà không khai báo. Skeleton extractor và RGB/3D teacher có thể đã dùng external supervision.
- Cross-corpus phải audit source video ID + temporal segment, không chỉ filename hoặc gloss name. Cross-language cần language tag và relevance riêng.
- Chỉ hứa learned “phonology/semantics” đến mức diagnostic hỗ trợ. Reconstruction MSE tốt, attention heatmap đẹp hoặc cluster tách biệt chưa đủ.

## 12. Deliverable khoa học nên đạt

Kết quả cần là bảng **signal × sufficiency × incremental gain × retrain-removal × perturbation × linguistic slice × cost**, có confidence và supervision scope. Một module được chọn khi nó cải thiện retrieval trong đối chứng phù hợp, hoặc mang cơ chế có giá trị được chứng minh riêng; không chọn vì tên module giống cấu trúc ngôn ngữ.

Từ đó mới có thể trả lời cụ thể: tín hiệu nào đã có sẵn, tín hiệu nào SSL chưa giữ được, tín hiệu nào dư thừa, và tín hiệu nào chỉ có giá trị cho một nhóm sign. Đó là cơ sở để chọn module thay vì tiếp tục ghép nhiều prior rồi chỉ đọc một số R@1 cuối.
