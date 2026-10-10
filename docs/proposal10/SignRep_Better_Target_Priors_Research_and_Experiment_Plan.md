# SignRep Beyond Sign Priors
## Deep Research Memo: Better Target Priors, Candidate Objectives, and Controlled Experimental Plan

**Mục tiêu tài liệu:** xây dựng một kế hoạch nghiên cứu có thể triển khai thực nghiệm để kiểm tra liệu các target priors mới có thể **thay thế, bổ sung, hoặc tái cấu trúc** bộ Sign Priors của SignRep và tạo representation tốt hơn trong điều kiện so sánh công bằng.

**Baseline trung tâm:**  
Ryan Wong, Necati Cihan Camgoz, Richard Bowden. **SignRep: Enhancing Self-Supervised Sign Representations.** ICCV 2025.  
Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html  
arXiv: https://arxiv.org/abs/2503.08529  
Code: https://github.com/ryanwongsa/SignRep  
Supplement: https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wong_SignRep_Enhancing_Self-Supervised_ICCV_2025_supplemental.pdf

**Phạm vi literature review:** cập nhật đến **07/10/2026**.  
Các paper 2026 chưa peer-reviewed được ghi rõ là **preprint** và không được dùng như bằng chứng tương đương với paper venue chính thức.

---

# 0. Executive Summary

Câu hỏi nghiên cứu trung tâm:

> **What should a masked sign representation model predict?**  
> Liệu các geometric Sign Priors hiện tại của SignRep có thực sự là target space tối ưu, hay có thể cải thiện representation bằng các target giàu motion, semantic, non-manual hoặc linguistic structure hơn?

SignRep đã đưa một bước tiến quan trọng: thay vì reconstruction thuần pixel, model học các **pose-derived sign priors** gồm:

- hand/body keypoints;
- joint angles;
- pairwise geometric distances;
- hand activity;
- representation regularization;
- signer adversarial regularization.

Tuy nhiên, chính ablation của SignRep cho thấy **raw keypoints là prior đơn lẻ yếu đáng kể hơn angle/distance**. Điều này gợi ý rằng target có tính **invariant, relational và semantic** có thể đáng học hơn target tọa độ thô.

Từ literature review, các hướng có evidence mạnh nhất để thử nghiệm là:

1. **Multi-stream discrete semantic units** cho left hand / right hand / face / body, theo tinh thần SHuBERT.
2. **Masked teacher-latent prediction** thay vì coordinate reconstruction, theo MVD/data2vec và gần đây là SignSeek-style masked articulator prediction.
3. **Explicit motion residuals, velocity, acceleration, bone motion, angular motion**, được hỗ trợ mạnh bởi MASA và spatial-temporal consistency SLR.
4. **Non-manual targets**: face, mouth, eyebrow/head orientation, vì SignRep tự thừa nhận chưa mô hình hóa đầy đủ non-manuals và literature cho thấy chúng mang discriminative linguistic information.
5. **Cross-modal RGB–pose latent consistency** thay exact pose coordinate regression, theo CCL-SLR và related cross-modal work.
6. **Phonology-factorized targets** theo Handshape / Orientation / Location / Movement / Non-manual, là hướng high-risk/high-reward để chuyển target space từ “pose geometry” sang “linguistically meaningful state”.
7. **HOG/local shape target cho hand regions**, một low-cost baseline đáng thử theo MaskFeat.
8. **Learned interaction latent** thay hoặc bổ sung pairwise Euclidean distance matrix.
9. **Articulator-aware masking**, không phải target nhưng có thể cộng hưởng với các target mới; cần test ở phase riêng để tránh confounding.

Kết luận thực nghiệm quan trọng nhất của memo này:

> **Không nên thay toàn bộ SignRep priors ngay từ đầu.**  
> Nên giữ angle/distance như geometric inductive bias, thay raw XYZ trước, rồi kiểm tra semantic + motion + non-manual targets theo controlled experiments.

Một cấu hình được ưu tiên cao:

\[
\mathcal{L}_{\text{Hybrid}}
=
\lambda_g \mathcal{L}_{\text{geometry}}
+
\lambda_m \mathcal{L}_{\text{motion}}
+
\lambda_s \mathcal{L}_{\text{semantic}}
+
\lambda_n \mathcal{L}_{\text{nonmanual}}
+
\mathcal{L}_{\text{reg}}
\]

với:

- geometry = angle + selected relative distances;
- motion = velocity + bone motion + angular velocity;
- semantic = discrete hand/face/body clusters hoặc teacher latent;
- non-manual = face/mouth/head targets;
- regularization = giữ SignRep variance/covariance + signer adversarial khi cần.

---

# 1. Research Question và Evidence Standard

## 1.1 Câu hỏi chính

Ta cần phân biệt ba câu hỏi khác nhau:

### Q1 — Target quality
Nếu giữ nguyên:

- Hiera backbone;
- pretraining data;
- number of steps;
- masking;
- augmentations;
- optimizer;
- regularizers;
- decoder capacity gần tương đương;

thì target nào tạo frozen representation tốt nhất?

Đây là experiment quan trọng nhất.

### Q2 — Complementarity
Một target mới có bổ sung thông tin mà SignRep priors đang thiếu không?

Ví dụ:

\[
L_{\text{SignRep}} + L_{\text{motion}}
\]

có tốt hơn cả hai độc lập không?

### Q3 — Full-system optimization
Sau khi xác định target tốt, liệu:

- articulator-aware masking;
- adaptive loss weighting;
- saliency masking;
- target-specific decoders;

có tiếp tục giúp không?

Không được trộn Q1 và Q3 ngay từ đầu, vì nếu đổi cả target lẫn masking/architecture thì không thể biết improvement đến từ đâu.

---

## 1.2 Các mức độ bằng chứng

Trong tài liệu này, evidence được chia thành:

### Level A — Direct sign-language evidence
Paper trực tiếp làm sign language representation / recognition với objective tương tự.

Ví dụ:

- SignRep;
- SHuBERT;
- BEST;
- MASA;
- CCL-SLR;
- SignBERT;
- non-manual SLR studies.

### Level B — Cross-domain but methodologically direct
Paper general video/skeleton SSL chứng minh loại target đó tốt hơn low-level reconstruction.

Ví dụ:

- Masked Video Distillation;
- MaskFeat;
- data2vec;
- skeleton SSL.

### Level C — Linguistic / theoretical evidence
Các công trình ngôn ngữ học hoặc phonological modeling cho thấy loại tín hiệu đó thực sự mang contrastive meaning.

Ví dụ:

- handshape;
- orientation;
- location;
- movement;
- non-manual markers.

### Level D — Research hypothesis
Hợp lý về mặt theory/engineering nhưng chưa có direct controlled evidence trên SignRep.

---

# 2. SignRep Baseline: Phân rã chính xác target space

## 2.1 Backbone

SignRep sử dụng Hiera-B encoder, pretrained từ Kinetics MAE initialization và tiếp tục self-supervised pretraining trên dữ liệu sign video.

Trong quá trình pretraining:

\[
V \rightarrow E_{\theta}(V_{\text{visible}})
\rightarrow D_{\phi}
\rightarrow \hat T
\]

với \(T\) là Sign Priors.

Decoder chỉ phục vụ pretraining; downstream dùng encoder.

---

## 2.2 Keypoint priors

### Hand keypoints

Mỗi tay:

\[
P_{h,k} \in \mathbb{R}^{21 \times 3}
\]

với coordinate normalization quanh wrist.

Ưu điểm:

- trực tiếp;
- dễ generate;
- tương ứng articulation.

Nhược điểm:

- phụ thuộc pose estimator;
- chịu viewpoint/perspective noise;
- occlusion hai tay;
- motion blur;
- hand-face contact;
- signer morphology;
- metric distance không đồng nghĩa linguistic distance.

### Full body

\[
P_{b,k} \in \mathbb{R}^{61 \times 3}
\]

gồm hand + upper-body points.

---

## 2.3 Angle priors

### Hand angles

Khoảng 41 joint-angle components, biểu diễn bằng:

\[
[\sin \theta, \cos \theta]
\]

để tránh discontinuity tại \(2\pi\).

### Body angles

Khoảng 22 angle components.

Angle có tính relative/invariant hơn raw coordinate.

---

## 2.4 Distance priors

SignRep sử dụng pairwise geometric relations:

- fingertip–knuckle;
- fingertip–wrist;
- fingertip–fingertip;
- hand–body;
- hand–face;
- inter-hand relations.

Tổng quát:

\[
d_{ij} = \|p_i - p_j\|_2
\]

Distance prior có lợi vì encode **interaction geometry**, không chỉ isolated joint state.

---

## 2.5 Hand activity prior

Binary / heuristic target:

\[
a_t \in \{0,1\}
\]

cho mỗi hand, biểu diễn active/inactive signing state.

---

## 2.6 Loss

Các continuous priors dùng Smooth-L1:

\[
\mathcal{L}_{prior}
=
\operatorname{SmoothL1}(\hat T,T)
\]

và confidence thấp từ pose estimator được loại/mask.

Tổng loss về cơ bản:

\[
\mathcal{L}
=
\mathcal{L}_{recon}
+
w_{var}\mathcal{L}_{var}
+
w_{cov}\mathcal{L}_{cov}
+
w_{adv}\mathcal{L}_{adv}
\]

với variance/covariance regularization và signer-style adversarial objective.

---

## 2.7 Loss weights từ SignRep supplemental

Các trọng số đáng giữ nguyên khi reproduce baseline:

| Target | Approx. loss weight | Scale |
|---|---:|---:|
| Body angles | 10 | 1 |
| Left hand angles | 10 | 1 |
| Right hand angles | 10 | 1 |
| Body keypoints | 10 | 1 |
| Left hand keypoints | 10 | 2 |
| Right hand keypoints | 10 | 2 |
| Body distance | 20 | 1 |
| Left hand distance | 20 | 4 |
| Right hand distance | 20 | 4 |
| Hand activity | 0.2 | — |
| Variance regularizer | 1 | — |
| Covariance regularizer | 0.004 | — |
| Signer adversarial | 2 | — |

**Lưu ý:** khi thêm target mới, không nên tùy tiện so trực tiếp cùng coefficient nếu dimensionality/loss scale khác. Xem phần loss normalization.

---

# 3. SignRep Ablation cho thấy vấn đề gì?

Một kết quả quan trọng trong ablation retrieval WLASL là:

| Target configuration | DCG, xấp xỉ theo paper |
|---|---:|
| Angle only + masking | 45.1 |
| Keypoint only + masking | 32.9 |
| Distance only + masking | 47.2 |
| Angle + keypoint + distance + masking | 48.5 |
| Angle + keypoint + distance, no masking | 46.3 |
| Angle + keypoint + masking + adversarial | 49.9 |
| Full configuration | 50.7 |

Điểm quan trọng không phải chỉ là absolute number mà là ordering:

\[
\text{raw keypoint}
<
\text{angle / relational distance}
\]

Điều này hỗ trợ hypothesis:

> **Target càng invariant / relational thì có khả năng cung cấp supervision tốt hơn raw XYZ.**

Do vậy raw XYZ là candidate đầu tiên cần thử **replace/downweight**, còn angle/distance chưa nên bỏ vội.

---

# 4. Literature Map

---

## 4.1 SHuBERT — Multi-Stream Cluster Prediction

**Paper:** Shester Gueuwou et al.  
**Title:** *SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction*  
**Venue:** ACL 2025, Oral  
**DOI:** 10.18653/v1/2025.acl-long.1397  
**Paper:** https://aclanthology.org/2025.acl-long.1397/  
**Project:** https://shubert.pals.ttic.edu/  
**Public code:** https://github.com/ShesterG/SHuBERT

### Ý tưởng chính

Tách signing frame thành:

- left hand;
- right hand;
- face;
- body.

Hand và face dùng pretrained visual features, body dùng pose-based representation.

Mỗi stream được cluster độc lập:

\[
z^s_t \xrightarrow{k\text{-means}} c^s_t
\]

với \(s\in\{LH,RH,Face,Body\}\).

Masked encoder dự đoán:

\[
\hat c^s_t
\]

bằng cross entropy.

### Tại sao rất quan trọng với SignRep?

SignRep:

\[
RGB \rightarrow XYZ,\theta,d
\]

SHuBERT:

\[
RGB/context \rightarrow \text{semantic-like visual units}
\]

Đây là direct sign-language evidence rằng target có thể được nâng từ continuous geometry lên **discrete hidden units**.

### Kết luận cho project

**Candidate Priority: S**

Nên thử cả:

- hard k-means target;
- soft cluster distribution;
- raw teacher latent;
- multi-stream vs hand-only;
- cluster + geometry hybrid.

---

## 4.2 BEST — Discrete Pose Units

**Paper:** Weichao Zhao et al.  
**Title:** *BEST: BERT Pre-training for Sign Language Recognition with Coupling Tokenization*  
**Venue:** AAAI 2023  
**DOI:** 10.1609/aaai.v37i3.25470  
**Paper:** https://ojs.aaai.org/index.php/AAAI/article/view/25470  
**Official code:** không xác nhận được public official implementation tại thời điểm review.

### Motivation

Continuous pose units là low-level và khó phù hợp với BERT-like categorical masked prediction.

BEST học tokenizer để biến pose components thành:

\[
P_t \rightarrow q_t
\]

với \(q_t\) là discrete pseudo-label.

### Các biến thể trong paper

- k-means;
- separate vector quantization;
- coupling tokenization / coupled VQ.

### Relevance

BEST là evidence trực tiếp cho hypothesis:

\[
\text{continuous pose regression}
\rightarrow
\text{discrete pose-state classification}
\]

có thể tạo representation giàu semantic hơn.

### Candidate Priority

**S / A+**

Đặc biệt phù hợp làm experiment rẻ nhất vì vẫn dùng chính pose source của SignRep.

---

## 4.3 MASA — Motion-aware reconstruction

**Paper:** Weichao Zhao et al.  
**Title:** *MASA: Motion-Aware Masked Autoencoder With Semantic Alignment for Sign Language Recognition*  
**Venue:** IEEE Transactions on Circuits and Systems for Video Technology, 2024  
**DOI:** 10.1109/TCSVT.2024.3409728  
**arXiv:** https://arxiv.org/abs/2405.20666  
**Public code:** https://github.com/sakura2233565548/MASA

### Ý tưởng

Không chỉ reconstruct appearance/frame state. MASA xây motion-aware masked autoencoding và reconstruct **motion residuals** của masked frames.

Khái niệm cốt lõi:

\[
R_t = X_t - X_{t-1}
\]

hoặc motion information tương đương, cùng semantic alignment.

### Relevance

Sign language là dynamic language:

\[
\text{sign identity}
\neq
f(\text{static pose only})
\]

mà phụ thuộc:

\[
f(\text{handshape}, \text{location}, \text{orientation}, \text{movement})
\]

SignRep có geometric state nhưng chưa có explicit derivative targets.

### Candidate Priority

**S**

---

## 4.4 Spatial-Temporal Consistency SSL for SLR

**Paper:** Weichao Zhao et al.  
**Title:** *Self-Supervised Representation Learning With Spatial-Temporal Consistency for Sign Language Recognition*  
**Venue:** IEEE Transactions on Image Processing, 2024  
**DOI:** 10.1109/TIP.2024.3416881  
**arXiv:** https://arxiv.org/abs/2406.10501  
**Code:** https://github.com/sakura2233565548/Self-Supervised-Representation-Learning-with-Spatial-Temporal-Consistency-for-SLR

### Ý tưởng liên quan

Paper sử dụng consistency giữa:

- hand / trunk spatial components;
- joints / first-order motion temporal components;
- bidirectional information transfer.

### Relevance

Củng cố việc không nên supervise chỉ pose state:

\[
P_t
\]

mà nên thêm:

\[
\Delta P_t = P_t-P_{t-1}.
\]

---

## 4.5 CCL-SLR — RGB/Pose Cross-modal Consistency

**Paper:** Kepeng Wu et al.  
**Title:** *Cross-Modal Consistency Learning for Sign Language Recognition*  
**Venue:** CVPR Workshops 2025  
**DOI:** 10.1109/CVPRW67362.2025.00392  
**Paper:** https://openaccess.thecvf.com/content/CVPR2025W/SLRTP/html/Wu_Cross-Modal_Consistency_Learning_for_Sign_Language_Recognition_CVPRW_2025_paper.html  
**arXiv:** https://arxiv.org/abs/2503.12485  
**Code:** https://github.com/dueToLife/CCL-SLR

### Ý tưởng chính

- RGB stream;
- pose stream;
- within-modal consistency;
- cross-modal consistency;
- motion-preserving masking;
- semantic positive mining.

### Insight quan trọng

Pose rất compact nhưng:

- có estimation noise;
- có blur/occlusion problems;
- không chứa toàn bộ semantic appearance.

Do đó có thể tốt hơn nếu dùng pose như **latent teacher / cross-modal anchor** thay vì ép RGB encoder reconstruct từng coordinate.

### Candidate Priority

**A+**

---

## 4.6 SignSeek — Articulator-aware masked latent prediction

**Paper:** Sobhan Asasi, Ozge Mercanoglu Sincan, Richard Bowden  
**Title:** *SignSeek: Learning Transferable Representations for Sign Dictionary Retrieval*  
**Status:** arXiv preprint, September 2026 — **chưa nên coi như peer-reviewed evidence**  
**arXiv:** https://arxiv.org/abs/2609.03695  
**Institutional page:** https://openresearch.surrey.ac.uk/esploro/outputs/preprint/SignSeek-Learning-Transferable-Representations-for-Sign/991160273502346  
**Code:** chưa xác nhận public code release tại thời điểm 07/10/2026.

### Các objective

- GAC: Global Articulator-aware Contrastive;
- MAC: Masked/Modality Articulator Contrastive;
- MAP: Masked Articulator Prediction.

MAP đặc biệt liên quan:

1. xác định articulator quan trọng;
2. mask articulator đó;
3. dùng context còn lại;
4. predict clean latent target của articulator bị mask.

Khái niệm:

\[
z_a = T(x_a)
\]

\[
\hat z_a =
D(E(x_{\setminus a}))
\]

\[
L_{MAP} = d(\hat z_a, \operatorname{sg}(z_a))
\]

### Kết quả đáng chú ý

Paper báo frozen cross-corpus retrieval cao hơn SignRep trên nhiều benchmark, nhưng:

**không được diễn giải là MAP target đã chứng minh tốt hơn SignRep priors**, vì:

- training data khác;
- supervision khác;
- GAC/MAC dùng gloss;
- architecture/protocol khác.

### Experiment có thể lấy từ SignSeek

Ta có thể xây phiên bản **self-supervised MAP-only**:

- không dùng gloss;
- articulator selection từ motion/saliency;
- predict frozen teacher latent;
- giữ nguyên Hiera/YouTube-SL/compute của SignRep.

Đây là controlled test rất giá trị.

### Candidate Priority

**S high-potential, nhưng evidence hiện tại là preprint + confounded comparison.**

---

## 4.7 SignMAE — Sign-aware masking

**Paper:** Kunyuan Xie et al.  
**Title:** *SignMAE: Segmentation-Driven Self-Supervised Learning for Sign Language Recognition*  
**Status:** arXiv preprint, May 2026  
**arXiv:** https://arxiv.org/abs/2605.02094

### Relevance

SignMAE không phải target replacement trực tiếp, nhưng nhấn mạnh rằng masking nên tập trung vào các body regions có ý nghĩa đối với signing.

Kết luận cho project:

- target study trước;
- masking study sau;
- không đồng thời đổi cả hai ở first-stage ablation.

---

## 4.8 Masked Video Distillation — Teacher feature targets

**Paper:** Rui Wang et al.  
**Title:** *Masked Video Distillation: Rethinking Masked Feature Modeling for Self-Supervised Video Representation Learning*  
**Venue:** CVPR 2023  
**Paper:** https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Masked_Video_Distillation_Rethinking_Masked_Feature_Modeling_for_Self-Supervised_Video_CVPR_2023_paper.html  
**Code:** https://github.com/ruiwang2021/mvd

### Core idea

Không reconstruct pixels. Student predict latent features từ pretrained teacher.

Spatial teacher và temporal teacher cung cấp complementary signals.

### Mapping sang SignRep

Có thể dùng:

- image teacher cho handshape/face;
- video teacher cho movement;
- pose teacher cho geometry;
- co-teaching.

Ví dụ:

\[
T_{spatial,t}=DINOv2(Crop_{hand,t})
\]

\[
T_{temporal}=VideoTeacher(V_{t:t+k})
\]

\[
L =
\lambda_s d(\hat T_s,T_s)
+
\lambda_t d(\hat T_t,T_t)
\]

### Candidate Priority

**S**

---

## 4.9 data2vec — Contextual teacher targets

**Paper:** Alexei Baevski et al.  
**Title:** *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language*  
**arXiv:** https://arxiv.org/abs/2202.03555

### Relevance

Thay modality-specific handcrafted target bằng contextual teacher representation.

Điều này gợi ý một SignRep variant:

\[
T_t =
EMAEncoder(V)_t
\]

và student predict masked contextual embedding.

Có thể kết hợp sign priors như regularizer thay vì target chính.

---

## 4.10 MaskFeat — HOG target

**Paper:** Chen Wei et al.  
**Title:** *Masked Feature Prediction for Self-Supervised Visual Pre-Training*  
**Venue:** CVPR 2022  
**Paper:** https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html  
**Code base:** https://github.com/facebookresearch/SlowFast

### Relevance

HOG là target đơn giản nhưng encode:

- local edge;
- contour;
- orientation;
- shape.

Đặc biệt có thể hữu ích cho handshape.

Candidate:

\[
H_t^{LH}=HOG(crop^{LH}_t)
\]

\[
H_t^{RH}=HOG(crop^{RH}_t)
\]

và predict masked hand HOG features.

### Candidate Priority

**B+/A-**

Không nên kỳ vọng đây là final solution, nhưng là strong low-cost baseline.

---

## 4.11 Non-manual information — LREC 2020

**Paper:** Medet Mukushev et al.  
**Title:** *Evaluation of Manual and Non-manual Components for Sign Language Recognition*  
**Venue:** LREC 2020  
**Paper:** https://aclanthology.org/2020.lrec-1.745/  
**Project/data:** https://krslproject.github.io/krsl20/

### Experimental evidence

Dataset được thiết kế để chứa signs có manual components gần nhau nhưng khác non-manual cues.

Các cues gồm:

- facial expression;
- eyebrows;
- mouth;
- head orientation.

Reported performance cho thấy adding non-manual components có thể tăng recognition so với manual-only.

Một số configurations được paper báo:

| Input | Accuracy |
|---|---:|
| Manual only | ~73.4 |
| Manual + all NMM | ~77.0 |
| Manual + face + eyebrow + mouth | ~78.2 |
| Manual + eyebrow + mouth | ~77.2 |
| Manual + mouth | ~77.5 |

### Relevance

SignRep supplemental thừa nhận limitation về non-manual facial expressions/mouthing.

Do đó đây là research gap trực tiếp.

### Candidate Priority

**A+/S-**

---

## 4.12 SignCLIP — semantic visual-language representation

**Paper:** Zifan Jiang et al.  
**Title:** *SignCLIP: Connecting Text and Sign Language by Contrastive Learning*  
**Venue:** EMNLP 2024  
**DOI:** 10.18653/v1/2024.emnlp-main.518  
**Paper:** https://aclanthology.org/2024.emnlp-main.518/

### Relevance

SignCLIP cho thấy sign video có thể align vào semantic language space.

Có hai cách dùng:

1. **External teacher track:** predict frozen SignCLIP video latent.
2. **Weakly supervised track:** video-text contrastive alignment.

Track 2 không còn apples-to-apples với pure SignRep self-supervision.

Do đó phải báo riêng.

---

## 4.13 SAM-SLR — multimodal complementarity

**Paper:** Songyao Jiang et al.  
**Title:** *Skeleton Aware Multi-modal Sign Language Recognition*  
**Venue:** CVPR Workshops 2021  
**Paper:** https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html  
**Code:** https://github.com/jackyjsy/CVPR21Chal-SLR

### Relevance

Skeleton dynamics, RGB và flow có complementary information.

Không phải target study, nhưng hỗ trợ việc:

- motion stream;
- flow;
- skeleton/RGB complementarity

là tín hiệu đáng khai thác.

---

## 4.14 SignBERT

**Paper:** Hezhen Hu et al.  
**Title:** *SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition*  
**Venue:** ICCV 2021  
**Paper:** https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html

Official public code không xác nhận được trong review này. Có community/unofficial implementations nhưng không nên coi là reference implementation.

### Relevance

Hand reconstruction được regularize bằng structural/hand-model prior thay vì xem raw pose estimator output là perfect target.

---

# 5. Candidate C1 — Motion Residual Priors

## 5.1 Hypothesis

SignRep mô hình hóa mạnh **state**, nhưng movement là phonologically relevant.

Hai sign có thể có:

- handshape tương tự;
- starting location tương tự;

nhưng khác:

- motion direction;
- speed;
- cyclicity;
- trajectory.

Do đó:

\[
P_t
\]

chưa đủ.

---

## 5.2 Target definitions

Pose trước tiên được normalize:

\[
\tilde p_{j,t}
=
\frac{p_{j,t}-p_{\text{root},t}}{s_t+\epsilon}
\]

với \(s_t\) có thể là shoulder width hoặc hand scale.

### Velocity

\[
v_{j,t}
=
\tilde p_{j,t}
-
\tilde p_{j,t-1}
\]

### Acceleration

\[
a_{j,t}
=
\tilde p_{j,t+1}
-
2\tilde p_{j,t}
+
\tilde p_{j,t-1}
\]

### Angular velocity

Vì angle dùng sin/cos:

\[
u_t=[\sin\theta_t,\cos\theta_t]
\]

predict:

\[
\Delta u_t=u_t-u_{t-1}
\]

tránh discontinuity.

### Relative-hand movement

\[
r_t = p_{LH,t}-p_{RH,t}
\]

\[
\Delta r_t=r_t-r_{t-1}
\]

### Fingertip trajectory

\[
\tau^k =
\left\{
p_{tip_k,t-L},...,p_{tip_k,t}
\right\}
\]

có thể predict derivative thay entire trajectory.

---

## 5.3 Confidence gating

Không tính motion target nếu một trong hai frame không đáng tin:

\[
m_{j,t}
=
\mathbb{1}[c_{j,t}>\tau_c]
\mathbb{1}[c_{j,t-1}>\tau_c]
\]

Acceleration yêu cầu 3-frame validity.

---

## 5.4 Loss

\[
L_v =
\frac{\sum m_{j,t}\operatorname{Huber}(\hat v_{j,t},v_{j,t})}
{\sum m_{j,t}+\epsilon}
\]

Tương tự cho \(a_t\).

---

## 5.5 Experiments

- C1.1 velocity only;
- C1.2 acceleration only;
- C1.3 velocity + acceleration;
- C1.4 angular velocity;
- C1.5 relative-hand motion;
- C1.6 all motion;
- C1.7 SignRep + motion;
- C1.8 SignRep minus raw XYZ + motion.

### Critical test

Nếu:

\[
\text{SignRep-noXYZ + motion}
>
\text{SignRep}
\]

thì có evidence mạnh rằng motion derivatives là target tốt hơn raw coordinates.

---

## 5.6 Expected failure modes

- pose jitter làm derivative amplify noise;
- low FPS clips;
- variable frame rate;
- inaccurate depth component;
- acceleration rất noisy.

### Countermeasures

- confidence mask;
- optional low-pass/Savitzky–Golay preprocessing;
- compare smoothing vs no smoothing;
- normalize by \(\Delta t\) nếu timestamp available;
- report target noise diagnostics.

---

# 6. Candidate C2 — Bone / Bone-Motion Priors

## 6.1 Target

Với edge skeleton \(i\rightarrow j\):

\[
b_{ij,t}=p_{j,t}-p_{i,t}
\]

Unit bone vector:

\[
\hat b_{ij,t}
=
\frac{b_{ij,t}}{\|b_{ij,t}\|+\epsilon}
\]

Separate length:

\[
l_{ij,t}=\|b_{ij,t}\|
\]

Bone motion:

\[
\Delta b_{ij,t}=b_{ij,t}-b_{ij,t-1}
\]

---

## 6.2 Why

So với coordinate:

- translation-invariant;
- structurally local;
- closer to articulation;
- lower nuisance from body position.

So với angle:

- vẫn giữ directional information;
- có thể recover shape tốt hơn.

---

## 6.3 Suggested loss

\[
L_b =
\lambda_{dir}(1-\cos(\hat b^{pred},\hat b^{gt}))
+
\lambda_l Huber(\hat l,l)
\]

hoặc Huber vector trực tiếp.

---

## 6.4 Experiments

- bone only;
- bone + bone motion;
- bone + angle;
- bone replacing keypoint;
- bone + semantic cluster.

**Priority: A**

---

# 7. Candidate C3 — Discrete Pose Tokens

## 7.1 Main idea

Thay:

\[
\hat P_t \approx P_t
\]

bằng:

\[
\hat q_t \approx q(P_t)
\]

với \(q\) là pose-state tokenizer.

---

## 7.2 Option A — k-means

1. collect normalized pose features;
2. PCA optional;
3. k-means;
4. save cluster id per frame/part;
5. train CE.

\[
q_t=\arg\min_k \|z_t-\mu_k\|^2
\]

---

## 7.3 Option B — soft clusters

Hard boundary có thể gây noise.

Define:

\[
p(k|z)=
\frac{
\exp(-\|z-\mu_k\|^2/\tau)
}{
\sum_j \exp(-\|z-\mu_j\|^2/\tau)
}
\]

và:

\[
L_{soft}=
KL(p_{target} \Vert p_{pred})
\]

---

## 7.4 Option C — learnable VQ

Train codebook:

\[
z\rightarrow e_k
\]

với EMA VQ hoặc VQ-VAE style.

Cần tránh joint end-to-end collapse; tốt hơn pretrain tokenizer rồi freeze trong experiment đầu.

---

## 7.5 Feature choices trước tokenization

Test riêng:

1. hand keypoints;
2. angle;
3. bone vectors;
4. keypoint + angle;
5. keypoint + angle + motion;
6. part-wise tokenization.

---

## 7.6 Codebook sizes

\[
K\in\{64,128,256,512,1024\}
\]

Không tune trên test.

Primary sweep trên WLASL validation frozen retrieval.

---

## 7.7 Key diagnostics

- cluster entropy;
- effective perplexity;
- cluster occupancy;
- mutual information với sign labels chỉ dùng **sau** khi cluster fit;
- signer leakage;
- visualization nearest samples.

---

## 7.8 Priority

**S / A+**

Đây là experiment đặc biệt quan trọng vì cùng pose source với SignRep, nên attribution sạch hơn SHuBERT/DINO external teacher.

---

# 8. Candidate C4 — Multi-Stream Semantic Hidden Units

## 8.1 Streams

\[
S=\{LH,RH,Face,Body\}
\]

Mỗi frame:

- crop left hand;
- crop right hand;
- crop face;
- body pose vector.

---

## 8.2 Teacher features

Suggested baseline:

- DINOv2 for hand/face image crops;
- normalized body pose for body.

Alternative:

- SigLIP / CLIP / SignCLIP for visual crops;
- hand-specialized teacher if available;
- ViT image teacher.

---

## 8.3 Cluster extraction

\[
z_t^s=f_s(x_t^s)
\]

\[
c_t^s=kmeans_s(z_t^s)
\]

Separate codebooks tránh dominant-scale issues.

Recommended first setting:

\[
K=256
\]

cho mỗi stream để match SHuBERT-style starting point.

---

## 8.4 Prediction heads

Shared hidden token \(h_t\), four classification heads:

\[
\hat y^s_t=W_s h_t
\]

\[
L_{cluster}
=
\sum_s \lambda_s CE(\hat y^s_t,c^s_t)
\]

---

## 8.5 Variants

### C4-A
Hands only.

### C4-B
Hands + body.

### C4-C
Hands + face.

### C4-D
Hands + body + face.

### C4-E
Hard labels.

### C4-F
Soft assignment.

### C4-G
Cluster + geometry.

---

## 8.6 What should be replaced?

Recommended first test:

Remove:

- raw hand keypoint regression;
- raw body coordinate regression.

Keep:

- angle;
- distance;
- activity.

Add:

- four stream cluster CE.

Reason:

raw coordinates là candidate nuisance-heavy nhất; angle/distance vẫn có explicit relation priors.

---

## 8.7 Expected upside

- robust to small coordinate noise;
- hand crop captures texture/silhouette;
- face stream captures non-manuals;
- discrete state classification có thể align better with categorical phonological state.

---

## 8.8 Risks

- generic image teacher may cluster skin/background rather than sign state;
- face identity leakage;
- k-means codebooks may be signer-dependent;
- crop error;
- storage overhead.

### Mitigation

- tight crop;
- photometric augmentation;
- identity-adversarial regularization;
- whitening/PCA;
- cluster quality audit by signer;
- freeze teacher.

---

# 9. Candidate C5 — Teacher Latent Feature Prediction

## 9.1 Motivation

Không cần quantize.

Predict:

\[
z_T
\]

directly.

---

## 9.2 Spatial teacher

For hands/face:

\[
z^{LH}_t=f_{img}(crop^{LH}_t)
\]

\[
z^{Face}_t=f_{img}(crop^{Face}_t)
\]

Projected target:

\[
\bar z = LN(Pz)
\]

Student:

\[
\hat z = g(h_t)
\]

Loss:

\[
L_{latent}=1-\cos(\hat z,\operatorname{sg}(\bar z))
\]

---

## 9.3 Temporal teacher

A video teacher:

\[
z^{motion}_{t:t+w}=f_{video}(V_{t:t+w})
\]

provides temporal target that static DINO features cannot.

---

## 9.4 Co-teaching

\[
L=
\lambda_{img}L_{img}
+
\lambda_{video}L_{video}
\]

Có thể dùng separate decoders để tránh conflict.

---

## 9.5 EMA teacher variant

Không cần external pretraining:

\[
\theta_T \leftarrow
\tau \theta_T + (1-\tau)\theta_S
\]

và predict contextual hidden state.

Đây là data2vec-like setup và có ưu điểm:

- self-contained;
- không thêm external semantic supervision.

---

## 9.6 Target normalization

Rất quan trọng:

- LayerNorm;
- feature standardization;
- optional PCA;
- L2-normalization.

Nếu không, scale của teacher representation làm loss weight khó so.

---

## 9.7 Experiments

- hand image teacher only;
- hand + face;
- image teacher only;
- video teacher only;
- image + video;
- EMA teacher;
- latent + geometry;
- latent replacing XYZ.

**Priority: S**

---

# 10. Candidate C6 — Non-Manual Priors

## 10.1 Why

SignRep tập trung mạnh manual features.

Nhưng sign languages dùng:

- mouthings;
- mouth gestures;
- eyebrows;
- head tilt;
- head shake;
- gaze;
- facial expression.

---

## 10.2 Recommended target hierarchy

Không nên bắt đầu bằng raw 468-face-landmark reconstruction vì target rất noisy.

Ưu tiên:

### Level 1 — semantic face latent

\[
z^{face}_t = f_{teacher}(crop_{face,t})
\]

### Level 2 — mouth latent

\[
z^{mouth}_t=f_{teacher}(crop_{mouth,t})
\]

### Level 3 — head pose

\[
r_t=[yaw,pitch,roll]
\]

### Level 4 — normalized geometric non-manuals

- mouth openness;
- eyebrow distance;
- head motion;
- selected facial ratios.

---

## 10.3 Loss

\[
L_{NMM}=
\lambda_f L_{face}
+
\lambda_m L_{mouth}
+
\lambda_h L_{head}
\]

Semantic face/mouth targets dùng cosine hoặc cluster CE.

Head pose dùng Huber/sin-cos.

---

## 10.4 Key ablations

- face only;
- mouth only;
- head pose only;
- face + mouth;
- all NMM;
- NMM + existing geometry;
- NMM + semantic hand targets.

---

## 10.5 Critical benchmark

NMFs-CSL nên là một benchmark quan trọng vì non-manuals có relevance cao.

Nhưng phải báo cả WLASL/ASL-Citizen để biết gain có generalize không.

---

# 11. Candidate C7 — Hand-Focused HOG / Local Shape Target

## 11.1 Definition

For cropped hand:

\[
H_t=HOG(crop_{hand,t})
\]

Predict HOG of masked hand-associated patches.

---

## 11.2 Advantages

- cheap;
- offline extraction;
- encodes contour/orientation;
- little dependency on pose z-depth;
- strong low-level shape supervision.

---

## 11.3 Variants

- LH/RH separately;
- HOG + angle;
- HOG + cluster;
- HOG full-frame vs hand-region.

Expected:

hand-only HOG should be more targeted than full-frame HOG.

**Priority: B+/A-**

---

# 12. Candidate C8 — Cross-Modal Latent Consistency

## 12.1 Instead of coordinate regression

Pose encoder:

\[
z^P_t=f_P(P_t)
\]

RGB encoder:

\[
z^R_t=f_R(V_t)
\]

Align:

\[
L_{align}
=
1-\cos(z^R_t,\operatorname{sg}(z^P_t))
\]

or contrastive InfoNCE.

---

## 12.2 Why it may outperform coordinate regression

Exact coordinate regression penalizes every local estimation error.

Latent consistency allows pose encoder to compress away noise.

Pose becomes:

> a semantic structural teacher,

not an exact regression target.

---

## 12.3 Pose encoder choices

- frozen ST-GCN;
- Transformer;
- MLP over normalized pose;
- pretrained pose SSL encoder;
- BEST-like tokenizer encoder.

---

## 12.4 Experiments

- direct coordinate SignRep;
- latent pose alignment;
- latent + angle/distance;
- bidirectional consistency;
- stop-grad pose teacher;
- EMA pose teacher.

**Priority: A+**

---

# 13. Candidate C9 — Articulator-Guided Masked Latent Prediction

## 13.1 Inspiration

SignSeek-style masked articulator prediction, nhưng build một version **không gloss-supervised**.

---

## 13.2 Articulators

\[
A=\{LH,RH,Face,Body\}
\]

For each stream:

\[
z_a=f_a(x_a)
\]

---

## 13.3 Saliency score without gloss

Không được dùng label nếu muốn fair self-supervised track.

Các lựa chọn:

### Motion saliency

\[
s_a=\frac{1}{T}\sum_t \|\Delta p_{a,t}\|
\]

### Teacher change magnitude

\[
s_a=
\frac{1}{T-1}
\sum_t
\|z_{a,t}-z_{a,t-1}\|
\]

### Attention saliency

Từ frozen teacher attention, nhưng phải kiểm tra stability.

---

## 13.4 Mask + predict

Chọn articulator \(a^*\):

\[
a^*=\arg\max_a s_a
\]

Mask tokens/crop region tương ứng.

Predict:

\[
\hat z_{a^*}
=
g(E(V_{\setminus a^*}))
\]

Loss:

\[
L_{MAP}
=
1-\cos(
\hat z_{a^*},
\operatorname{sg}(z_{a^*})
)
\]

---

## 13.5 Essential controls

- random articulator mask;
- motion-saliency mask;
- teacher-change mask;
- uniform region mask;
- SignRep random patch mask.

Nếu saliency > random ở cùng target, ta chứng minh masking policy có independent value.

---

## 13.6 Risks

Model luôn chọn hands.

Mitigation:

- temperature sampling:

\[
P(a)=softmax(s_a/\tau)
\]

- entropy regularization;
- balanced selection quota;
- stochastic top-2.

**Priority: S high-potential**

---

# 14. Candidate C10 — Phonology-Factorized Targets

## 14.1 Motivation

Thay vì dự đoán generic geometry, dự đoán các phonological dimensions gần sign linguistics hơn:

\[
T=
[H,O,L,M,N]
\]

- \(H\): Handshape
- \(O\): Orientation
- \(L\): Location
- \(M\): Movement
- \(N\): Non-manual

---

## 14.2 Pseudo-label construction

### H — Handshape

Option 1:

\[
H=q(z^{hand}_{DINO})
\]

Option 2:

cluster normalized finger bone/angle geometry.

### O — Orientation

Palm plane normal:

\[
n_t=
\frac{
(p_{index}-p_{wrist})
\times
(p_{pinky}-p_{wrist})
}{
\|
\cdot
\|
}
\]

Quantize into orientation bins or continuous unit vector.

### L — Location

Hand position relative to body landmarks:

\[
r_{hand,body}=p_{hand}-p_{anchor}
\]

Assign region:

- face/head;
- torso;
- neutral space;
- left/right;
- upper/lower.

### M — Movement

Cluster trajectory primitives using:

\[
[\Delta P,\Delta^2P,\Delta\theta]
\]

### N — Non-manual

Cluster face/mouth/head latent.

---

## 14.3 Loss

\[
L_{phon}
=
\lambda_H CE(\hat H,H)
+\lambda_O L_O
+\lambda_L CE(\hat L,L)
+\lambda_M CE(\hat M,M)
+\lambda_N CE(\hat N,N)
\]

---

## 14.4 Strong scientific value

Nếu model này thắng geometric SignRep, claim có thể là:

> **A linguistically factorized target space is more transferable than direct pose reconstruction.**

Đây có thể là contribution mạnh hơn chỉ “thêm velocity”.

---

## 14.5 Risks

- pseudo-label definition arbitrary;
- phonology differs across sign languages;
- cluster ≠ true linguistic category;
- target parser/annotation may introduce supervision leakage.

Do đó cần hai tracks:

### Pure pseudo target
Không dùng manual linguistic annotations.

### Annotated/weak-supervision target
Chỉ dùng nếu dataset cho phép và báo riêng.

**Priority: S high-risk / high-reward**

---

# 15. Candidate C11 — Learned Interaction Latent

## 15.1 Problem with distance matrix

SignRep uses:

\[
D_{ij}=\|p_i-p_j\|
\]

nhưng interaction có thể phụ thuộc:

- orientation;
- contact type;
- temporal approach/separation;
- relative motion;
- hand topology.

---

## 15.2 Graph target

Create interaction graph:

\[
G_t=(V_t,E_t)
\]

nodes:

- fingertips;
- wrists;
- face anchors;
- body anchors.

Features:

- position;
- velocity;
- bone direction;
- confidence.

Teacher graph encoder:

\[
z^{inter}_t=GNN(G_t)
\]

RGB student predicts:

\[
\hat z^{inter}_t
\]

---

## 15.3 Baseline comparison

- pairwise distance only;
- graph latent only;
- distance + graph latent.

**Priority: B+/A-**  
Novel nhưng evidence trực tiếp ít hơn C1–C9.

---

# 16. Candidate C12 — Optical Flow / Dense Motion Target

## 16.1 Motivation

Pose misses:

- local finger movement;
- cloth/body detail;
- motion when pose tracking fails.

Dense flow:

\[
F_t=Flow(V_t,V_{t+1})
\]

---

## 16.2 Why lower priority

- expensive preprocessing;
- flow errors under blur;
- background/camera motion;
- high-dimensional;
- may encode nuisance.

Suggested only after cheaper motion targets.

---

## 16.3 Better version

Hand-region flow:

\[
F^{hand}_t
=
Crop(F_t, bbox_{hand})
\]

or teacher-compressed flow embedding.

**Priority: B**

---

# 17. Candidate C13 — External Semantic Teacher / SignCLIP

## 17.1 Target

Frozen SignCLIP visual encoder:

\[
z_{SC}=SignCLIP_{video}(V)
\]

Student SignRep encoder predicts:

\[
\hat z_{SC}
\]

---

## 17.2 Supervision warning

SignCLIP itself was trained with video-text pairs.

Vì vậy:

- không được gộp vào pure self-supervised table;
- report là **external semantic teacher track**.

---

## 17.3 Scientific question

Có semantic knowledge từ language-supervised teacher giúp sign representation transfer tốt hơn geometry-only teacher không?

**Priority: A, nhưng separate supervision class.**

---

# 18. Candidate Priority Matrix

| ID | Candidate | Evidence strength | Novelty | Cost | Recommended priority |
|---|---|---:|---:|---:|---|
| C1 | Motion residual / velocity | Very strong sign-specific | Medium | Low | **S** |
| C2 | Bone + bone motion | Strong indirect/sign skeleton | Medium | Low | **A** |
| C3 | Discrete pose tokens | Strong sign-specific | High | Medium | **S** |
| C4 | Multi-stream semantic clusters | Very strong sign-specific | High | Medium | **S** |
| C5 | Teacher latent prediction | Very strong video SSL | High | Medium | **S** |
| C6 | Non-manual priors | Strong linguistic/sign evidence | High | Medium | **S-** |
| C7 | Hand HOG | Strong generic SSL | Medium | Low | **B+/A-** |
| C8 | RGB-pose latent consistency | Strong sign-specific | High | Medium | **A+** |
| C9 | Articulator masked latent prediction | Strong recent hypothesis | Very high | Medium | **S*** |
| C10 | H/O/L/M/N phonology | Strong linguistic basis | Very high | High | **S high-risk** |
| C11 | Learned interaction latent | Moderate | Very high | Medium | **A-/B+** |
| C12 | Optical flow | Moderate | Low | High | **B** |
| C13 | SignCLIP teacher | Strong semantic | High | Medium | **A external-supervision** |

`S*`: SignSeek evidence hiện là 2026 preprint và comparison không controlled.

---

# 19. Recommended Target Replacement Map

| Original SignRep component | Recommendation |
|---|---|
| Raw left/right hand XYZ | **First target to replace/downweight** |
| Raw body XYZ | **Replace/downweight** |
| Hand angles | **Keep initially** |
| Body angles | Keep, later ablate |
| Fingertip distances | **Keep initially** |
| Hand-body interaction distances | Keep; compare learned interaction |
| Activity | Keep as cheap auxiliary |
| — | Add velocity |
| — | Add bone motion |
| — | Add semantic hand clusters |
| — | Add face/mouth targets |
| — | Add teacher latent |
| — | Optionally add phonological factor targets |

Most promising first hybrid:

\[
T=
\{
Angle,
Distance,
Activity,
Motion,
SemanticClusters,
NonManual
\}
\]

without raw XYZ.

---

# 20. Experiment Governance: Nguyên tắc để comparison thực sự khoa học

## 20.1 Không thay nhiều variable cùng lúc

Phase 1:

- same backbone;
- same data;
- same mask ratio;
- same random masking;
- same optimizer;
- same steps;
- only target changes.

Phase 2:

- target combinations.

Phase 3:

- masking strategy.

Phase 4:

- architecture/weight optimization.

---

## 20.2 Same data

Giữ đúng:

- YouTube-SL-25 split;
- same de-duplication;
- same preprocessing;
- same exclusion against downstream test sets.

Nếu không reproduce được exact dataset do link availability, phải:

1. freeze một new canonical dataset snapshot;
2. rerun SignRep baseline trên snapshot đó;
3. compare tất cả candidate trên cùng snapshot.

Không được dùng published SignRep number làm baseline nếu training corpus khác đáng kể.

---

## 20.3 Same compute

Main controlled experiments:

- Hiera-B;
- 16 frames;
- 224 resolution;
- 80% mask;
- 500k steps;
- same batch/effective batch;
- same LR schedule;
- same checkpoint selection rule.

Teacher target preprocessing có thể offline.

Report riêng:

- target extraction GPU-hours;
- training GPU-hours;
- peak memory;
- number of decoder/head params;
- encoder params at inference.

---

# 21. Baseline Reproduction Plan

## B00 — Initialization only

Hiera-B Kinetics MAE initialization, không YouTube-SL SignRep pretraining.

Mục đích:

\[
\Delta_{\text{pretrain}}
\]

---

## B01 — Generic MAE on same sign corpus

Nếu compute cho phép:

Hiera-B MAE / pixel target trên cùng sign corpus.

Mục đích phân biệt:

\[
\text{domain adaptation}
\]

vs

\[
\text{sign-prior target advantage}.
\]

---

## B02 — SignRep exact

Reproduce:

- exact original priors;
- same loss;
- same confidence mask;
- same variance/covariance;
- signer adversarial;
- same augmentations;
- same schedule.

B02 là baseline chính.

---

# 22. SignRep Reproduction Checklist

Pretraining config từ paper/supplement nên cố gắng giữ:

- backbone: Hiera-B;
- initialization: Kinetics MAE;
- clip: 16 consecutive frames;
- crop: 224 from 256;
- mask ratio: 80%;
- batch: effective equivalent của paper;
- optimizer: AdamW;
- peak LR: \(1\times10^{-4}\);
- warmup: khoảng 50k iterations;
- cosine decay;
- layer-wise LR decay: ~0.85;
- betas: ~0.9/0.95;
- weight decay: ~0.5;
- grad clipping: ~1.0;
- training length: 500k iterations;
- augmentation:
  - random resized crop;
  - Planckian jitter;
  - Gaussian blur;
  - grayscale;
- validation interval cố định.

Nếu hardware khác:

- giữ effective batch;
- giữ total optimizer updates;
- log throughput;
- không thay LR arbitrarily.

---

# 23. Phase 1 — Single-Target Screening

Mục đích:

> target nào tự thân tạo representation tốt?

Recommended experiment IDs:

| ID | Target | Original priors? |
|---|---|---|
| E01 | Velocity | No |
| E02 | Velocity + acceleration | No |
| E03 | Bone + bone motion | No |
| E04 | Discrete pose tokens K=256 | No |
| E05 | Soft pose clusters K=256 | No |
| E06 | Semantic LH/RH clusters | No |
| E07 | Semantic LH/RH/Face/Body clusters | No |
| E08 | Hand DINO latent | No |
| E09 | Hand + face latent | No |
| E10 | Temporal video-teacher latent | No |
| E11 | Spatial + temporal co-teacher | No |
| E12 | Hand HOG | No |
| E13 | Non-manual face/mouth/head | No |
| E14 | RGB↔pose latent alignment | No |
| E15 | MAP latent prediction + random articulator mask | No |
| E16 | MAP latent prediction + saliency mask | No |
| E17 | H/O/L/M/N pseudo-target | No |
| E18 | Hand optical-flow target | No |

### Compute-saving shortlist

Nếu compute hạn chế, ưu tiên:

1. E01;
2. E04;
3. E07;
4. E08 hoặc E11;
5. E13;
6. E16.

---

# 24. Phase 2 — Replacement Experiments

Mục đích:

> target nào có thể replace phần yếu của SignRep?

Define:

\[
G^\star =
\{angles,distances,activity\}
\]

tức giữ geometry nhưng bỏ raw coordinates.

---

## R01 — SignRep minus XYZ

\[
L=L_{angle}+L_{distance}+L_{activity}+L_{reg}
\]

Nếu R01 gần B02, raw coordinate contribution rất nhỏ.

---

## R02 — Replace XYZ with motion

\[
G^\star + Motion
\]

---

## R03 — Replace XYZ with semantic clusters

\[
G^\star + SemanticCluster
\]

---

## R04 — Replace XYZ with teacher latent

\[
G^\star + TeacherLatent
\]

---

## R05 — Semantic + motion

\[
G^\star
+
L_{semantic}
+
L_{motion}
\]

---

## R06 — Semantic + motion + non-manual

\[
G^\star
+
L_{semantic}
+
L_{motion}
+
L_{NMM}
\]

**Đây là configuration mình dự đoán có xác suất cao nhất vượt original SignRep.**

---

## R07 — Additive full SignRep + best candidate

Giữ tất cả old priors:

\[
L_{SignRep}+L_{candidate}
\]

Mục đích phân biệt:

- replacement value;
- complementary value.

---

# 25. Phase 3 — Masking Strategy Study

Chỉ chạy sau khi khóa target tốt nhất.

| ID | Mask strategy |
|---|---|
| M0 | Original random masking |
| M1 | Motion-preserving masking |
| M2 | Body-part/segmentation guided |
| M3 | Random articulator masking |
| M4 | Motion-saliency articulator masking |
| M5 | Teacher-saliency articulator masking |

---

## 25.1 Fair masking comparison

Mọi run giữ:

- target;
- backbone;
- number masked tokens;
- total compute;

chỉ thay location distribution của mask.

---

# 26. Phase 4 — Loss Weighting

Multi-target learning có thể thất bại chỉ vì scale mismatch.

## 26.1 Baseline weighting

Normalize mỗi loss theo:

\[
\tilde L_i
=
\frac{L_i}{EMA(L_i)+\epsilon}
\]

sau đó coefficient ban đầu:

\[
\lambda_i=1.
\]

---

## 26.2 Manual sweep

Cho target mới:

\[
\lambda\in
\{0.25,0.5,1,2,4\}
\]

tune trên **validation retrieval**, không test.

---

## 26.3 Gradient diagnostics

Log:

\[
g_i =
\|\nabla_{\theta_E} L_i\|
\]

Nếu một loss có gradient norm lớn gấp hàng chục lần loss khác thì raw coefficient comparison vô nghĩa.

---

## 26.4 Adaptive weighting

Optional follow-up:

- GradNorm;
- uncertainty weighting;
- dynamic weight averaging.

Nhưng adaptive weighting không nên dùng trong initial target-quality comparison vì thêm confound.

---

# 27. Phase 5 — Discrete Target Ablations

For C3/C4.

## 27.1 Codebook size

\[
K\in\{64,128,256,512,1024\}
\]

---

## 27.2 Hard vs soft

Hard:

\[
CE(\hat p,c)
\]

Soft:

\[
KL(p_{target}\|\hat p)
\]

---

## 27.3 Stream ablation

- LH;
- RH;
- both hands;
- face;
- body;
- hands + face;
- hands + body;
- all.

---

## 27.4 Cluster quality

Report:

- occupancy histogram;
- entropy;
- effective perplexity;
- average intra-cluster distance;
- inter-cluster distance;
- signer prediction from cluster IDs;
- gloss purity only as diagnostic, never used for clustering.

---

# 28. Phase 6 — Motion Target Ablations

| ID | Motion target |
|---|---|
| MV1 | wrist velocity |
| MV2 | all joint velocity |
| MV3 | fingertip velocity |
| MV4 | acceleration |
| MV5 | angular velocity |
| MV6 | bone motion |
| MV7 | LH-RH relative motion |
| MV8 | combined |

---

## 28.1 Noise ablation

Compare:

- no smoothing;
- temporal Gaussian smoothing;
- Savitzky-Golay.

Do not assume smoothing helps.

---

## 28.2 Frame-rate sensitivity

Downsample target input:

- 1x;
- 1/2 fps;
- 1/4 fps.

Measure robustness.

---

# 29. Phase 7 — Non-Manual Ablations

| ID | Target |
|---|---|
| NM1 | Face latent |
| NM2 | Mouth latent |
| NM3 | Head pose |
| NM4 | Brow/mouth geometry |
| NM5 | Face + mouth |
| NM6 | Face + mouth + head |
| NM7 | NM6 + hand semantic target |

Primary hypothesis:

\[
NM6
\]

should particularly help categories where manual forms are ambiguous.

---

# 30. Phase 8 — Cross-Modal Consistency Ablations

## CM1
Pose latent → RGB student, stop-gradient teacher.

## CM2
Symmetric RGB↔pose.

## CM3
Contrastive cross-modal.

## CM4
Latent alignment + original angles/distances.

## CM5
Latent alignment + motion.

Important:

same pose detector as SignRep if possible.

---

# 31. Evaluation Protocol

## 31.1 Primary: Frozen cross-corpus retrieval

Đây phải là main metric vì ít downstream adaptation nhất.

Benchmarks:

- WLASL2000;
- ASL Citizen;
- NMFs-CSL.

Metrics:

- DCG;
- Recall@1;
- Recall@5;
- MRR nếu available.

Use:

- official SignRep-style temporal sampling;
- stride 2 cho main;
- stride 8 reproduction nếu muốn match Table 5 ablation.

---

## 31.2 Secondary: Isolated Sign Recognition

### WLASL2000

Report:

- Instance Top-1;
- Instance Top-5;
- Class Top-1;
- Class Top-5.

SignRep reference:

- instance Top-1 ~61.05;
- instance Top-5 ~90.27;
- class Top-1 ~58.89;
- class Top-5 ~89.44.

### NMFs-CSL

Report:

- Top-1;
- Top-5.

SignRep reference approximately:

- Top-1 ~84.1;
- Top-5 ~98.8.

---

## 31.3 ASL Citizen downstream

Report:

- DCG;
- MRR;
- R@1;
- R@5.

SignRep reported downstream representation performance approximately:

- DCG 90.84;
- MRR 88.05;
- R@1 81.37;
- R@5 96.11.

---

## 31.4 Translation transfer

Nếu compute cho phép:

- PHOENIX14T;
- CSL-Daily;
- How2Sign.

Goal:

> better target should not only improve isolated recognition but transferable sequence-level linguistic information.

Freeze encoder first, then use same decoder protocol across all pretrained models.

---

# 32. Frozen Retrieval Reference Numbers from SignRep

Approximate paper values:

| Dataset | DCG | R@1 | R@5 |
|---|---:|---:|---:|
| ASL-Citizen | 71.21 | 49.95 | 80.09 |
| WLASL2000 | 57.93 | 29.92 | 67.41 |
| NMFs-CSL | 83.05 | 63.04 | 95.63 |

**Rule:** published numbers chỉ là sanity reference.  
Final comparison phải dùng **our reproduced SignRep checkpoint** dưới cùng code/data/hardware protocol.

---

# 33. Representation Diagnostics

Final Top-1 alone không đủ để hiểu target.

## 33.1 Linear probes

Nếu có labels:

- handshape;
- location;
- movement;
- orientation;
- non-manual;
- signer identity.

---

## 33.2 Signer leakage probe

Train linear classifier:

\[
z\rightarrow signerID
\]

Desirable:

- sign accuracy up;
- signer predictability down hoặc không tăng.

---

## 33.3 Nearest-neighbor inspection

For each query:

- inspect same-gloss neighbors;
- signer mismatch;
- similar handshape but wrong movement;
- same motion but wrong handshape.

Categorize errors.

---

## 33.4 Target predictability

Measure:

\[
R^2
\]

hoặc CE accuracy của each target head.

Cảnh báo:

> target dễ predict không đồng nghĩa representation tốt.

Need correlate target quality with downstream metrics.

---

# 34. Robustness Tests

## 34.1 Pose noise

Inject:

\[
p' = p+\epsilon,\quad
\epsilon\sim\mathcal N(0,\sigma^2)
\]

at target generation for controlled sensitivity.

---

## 34.2 Keypoint dropout

Randomly drop:

- 10%;
- 20%;
- 40%.

Measure how candidate target degrades.

---

## 34.3 Hand occlusion

Mask rectangular/semantic hand regions during input.

Latent/cluster methods should ideally degrade less than raw coordinate models.

---

## 34.4 Temporal jitter

- random frame drop;
- duplicate frame;
- irregular stride.

Motion target robustness must be measured explicitly.

---

## 34.5 Crop noise

Perturb hand bbox:

- translation;
- scale;
- jitter.

Important for semantic crop teachers.

---

# 35. Statistical Testing

## 35.1 Seeds

Screening:

- 1 seed initially only để loại rõ losers.

Serious candidate:

- minimum 3 seeds.

Final paper-quality:

- ideally 5 seeds cho top configurations.

---

## 35.2 Report

\[
mean \pm std
\]

Không chỉ best checkpoint/best seed.

---

## 35.3 Retrieval confidence interval

Dùng paired bootstrap trên queries.

For each resample:

1. sample queries with replacement;
2. compute metric difference:
   \[
   \Delta = M_{candidate}-M_{SignRep}
   \]
3. obtain 95% CI.

---

## 35.4 Nonparametric paired test

Có thể dùng:

- Wilcoxon signed-rank;
- permutation test

trên per-query reciprocal rank/DCG.

Không nên chỉ báo p-value.

Report:

- effect size;
- CI;
- absolute delta;
- relative delta.

---

# 36. Success Criteria

Một candidate được coi là **convincingly better than SignRep** nếu:

### Minimum criterion

- frozen retrieval cải thiện trên ít nhất 2/3 datasets;
- không giảm nghiêm trọng trên dataset còn lại;
- gain không chỉ ở one seed.

### Strong criterion

95% bootstrap CI của:

\[
\Delta DCG
\]

và/hoặc:

\[
\Delta R@1
\]

không chứa 0 trên ít nhất 2 datasets.

### Strongest criterion

- better frozen retrieval;
- better/neutral recognition;
- better cross-dataset transfer;
- equal encoder inference cost;
- equal or modest pretraining overhead.

---

# 37. Compute-normalized Comparison

Một target có thể thắng nhờ teacher cực lớn.

Do đó report hai bảng:

## 37.1 Fixed student training compute

Encoder updates identical.

Teacher targets precomputed offline.

## 37.2 Total system cost

Include:

- teacher extraction;
- k-means;
- target storage;
- pretraining;
- GPU-hours.

Metric optional:

\[
\frac{\Delta DCG}{GPU\text{-hour}}
\]

để đánh giá practical value.

---

# 38. Target Cache Design

Recommended directory:

```text
targets/
  metadata.json
  train/
    <video_id>.npz
  val/
    <video_id>.npz
```

Per file:

```text
hand_kpt              [T, 2, 21, 3]
hand_conf             [T, 2, 21]
body_kpt              [T, J, 3]
body_conf             [T, J]

hand_angle            [T, 2, A, 2]
body_angle            [T, B, 2]

hand_distance         [...]
body_distance         [...]

velocity              [T, J, 3]
acceleration          [T, J, 3]
bone                  [T, E, 3]
bone_motion           [T, E, 3]

dino_lh               [T, D]
dino_rh               [T, D]
dino_face             [T, D]

cluster_lh            [T]
cluster_rh            [T]
cluster_face          [T]
cluster_body          [T]

cluster_prob_lh       [T, K]      # optional
cluster_prob_rh       [T, K]
cluster_prob_face     [T, K]
cluster_prob_body     [T, K]

head_pose             [T, 3]
mouth_latent          [T, Dm]

validity_mask         [...]
```

---

## 38.1 metadata.json

Store:

```json
{
  "pose_model": "...",
  "pose_checkpoint_sha256": "...",
  "teacher_model": "...",
  "teacher_checkpoint_sha256": "...",
  "crop_version": "...",
  "cluster_k": 256,
  "cluster_seed": 0,
  "pca_dim": null,
  "normalization": "...",
  "code_commit": "..."
}
```

This is essential for reproducibility.

---

# 39. Architecture Integration

Keep same encoder.

```text
Video
  |
  v
Hiera Encoder
  |
  v
Shared masked decoder
  |
  +--> continuous head --> geometry/motion
  |
  +--> cluster head ----> discrete semantic target
  |
  +--> latent MLP ------> teacher feature
  |
  +--> NMM head --------> face/mouth/head
```

---

## 39.1 Continuous head

\[
h\rightarrow Linear(d,D_c)
\]

Loss:

Huber / Smooth-L1.

---

## 39.2 Cluster head

\[
h\rightarrow Linear(d,K)
\]

Loss:

CE / KL.

---

## 39.3 Latent head

Recommended:

\[
h
\rightarrow
Linear
\rightarrow GELU
\rightarrow Linear
\rightarrow LayerNorm
\]

Loss:

cosine.

---

## 39.4 Inference

All pretraining heads are dropped.

Therefore downstream encoder inference should remain equivalent to SignRep if encoder architecture unchanged.

This is an important practical advantage.

---

# 40. Suggested Pseudocode

```python
video = batch["video"]
targets = target_cache[batch["video_id"]]

visible, mask = mask_video(video, ratio=0.80)

z = encoder(visible)
h = decoder(z, mask)

losses = {}

if cfg.use_angles:
    pred = heads.angle(h)
    losses["angle"] = masked_huber(
        pred, targets["angle"], targets["angle_valid"]
    )

if cfg.use_motion:
    pred = heads.motion(h)
    losses["motion"] = masked_huber(
        pred, targets["velocity"], targets["motion_valid"]
    )

if cfg.use_clusters:
    for stream in ["lh", "rh", "face", "body"]:
        logits = heads.cluster[stream](h)
        y = targets[f"cluster_{stream}"]
        losses[f"cluster_{stream}"] = masked_ce(
            logits, y, targets[f"{stream}_valid"]
        )

if cfg.use_latent:
    pred = normalize(heads.latent(h))
    target = normalize(targets["teacher_latent"])
    losses["latent"] = 1.0 - cosine_similarity(
        pred, target.detach()
    ).mean()

if cfg.use_nonmanual:
    ...

loss = weighted_sum(losses)
loss += cfg.var_w * variance_loss(z)
loss += cfg.cov_w * covariance_loss(z)
loss += cfg.adv_w * signer_adversarial_loss(z)

optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

---

# 41. Recommended Configuration Objects

```yaml
target:
  geometry:
    keypoints: false
    angles: true
    distances: true
    activity: true

  motion:
    velocity: true
    acceleration: false
    bone_motion: true
    angular_velocity: true

  semantic:
    type: cluster
    streams: [left_hand, right_hand, face, body]
    k: 256
    soft_target: false

  nonmanual:
    face_latent: true
    mouth_latent: true
    head_pose: true

mask:
  type: random
  ratio: 0.80

regularization:
  variance: true
  covariance: true
  signer_adversarial: true
```

---

# 42. Prioritized Execution Roadmap

## Stage 0 — Reproduction

**Must complete before claims.**

- reproduce B02 SignRep;
- frozen WLASL retrieval;
- at least one recognition benchmark.

Go/no-go:

If reproduced baseline deviates > ~1–2 points substantially, debug before new priors.

---

## Stage 1 — Cheap candidate screen

Run:

- E01 velocity;
- E03 bone motion;
- E04 discrete pose;
- E12 HOG.

Reason:

cheap target generation.

Goal:

determine whether non-XYZ target families show positive signal.

---

## Stage 2 — Semantic teacher screen

Run:

- E06 hands clusters;
- E07 all streams;
- E08 hand latent;
- E09 hand+face latent.

Goal:

hard cluster vs continuous latent.

---

## Stage 3 — Replacement

Run:

- R01 no XYZ;
- R02 motion replacement;
- R03 clusters replacement;
- R04 latent replacement;
- R05 semantic+motion;
- R06 semantic+motion+NMM.

This is the core scientific table.

---

## Stage 4 — Articulator masking

Only top target configuration.

Compare M0–M5.

---

## Stage 5 — Phonological target

Build H/O/L/M/N target only after simpler semantic/motion evidence is established.

High engineering cost, high publication potential.

---

# 43. Early-Stop Screening Strategy

500k × dozens of runs rất đắt.

Có thể dùng a two-stage schedule:

### Screening

- 50k;
- 100k;
- 200k checkpoints.

Evaluate frozen WLASL validation retrieval.

Kill if candidate is consistently far below reproduced SignRep learning curve.

### Full run

Only top ~30–40% candidates → 500k.

Important:

Use **same early-stop rule** for all candidates.

---

# 44. Learning-Curve Comparison

Do not compare only final checkpoint.

Plot:

\[
Metric(step)
\]

at:

- 25k;
- 50k;
- 100k;
- 200k;
- 300k;
- 400k;
- 500k.

Questions:

1. target learns faster?
2. final ceiling higher?
3. overfits?
4. unstable?

A target with same final score but 2x faster convergence may still be valuable.

---

# 45. Ablation: Target Noise vs Representation Quality

For each target, estimate reliability:

### Pose

mean confidence.

### Cluster

distance to assigned centroid:

\[
u_t =
\|z_t-\mu_{q_t}\|
\]

### Latent teacher

augmentation consistency:

\[
c_t=
\cos(
f(x_t^{aug1}),
f(x_t^{aug2})
)
\]

Analyze downstream retrieval conditioned on target confidence.

Hypothesis:

semantic targets may provide better robustness in low-pose-confidence frames.

---

# 46. Ablation: What Information Does Each Target Encode?

Use controlled probes.

Given frozen encoder:

### Probe H
Handshape.

### Probe O
Orientation.

### Probe L
Location.

### Probe M
Movement.

### Probe N
Non-manual.

Then build radar/table.

Expected:

- geometry → H/O/L;
- motion → M;
- face target → N;
- semantic latent → mixed.

This makes paper explanation much stronger than only Top-1.

---

# 47. Ablation: Which Frames Benefit?

Partition samples by:

- high vs low pose confidence;
- high vs low hand motion;
- one-handed vs two-handed;
- face-visible vs face-occluded;
- static vs dynamic signs.

Then compute:

\[
\Delta metric
\]

per bucket.

This directly tests candidate hypotheses.

---

# 48. Failure Mode Analysis

## 48.1 Semantic cluster learns identity

Check:

- signer classifier;
- nearest neighbors grouped by signer;
- cluster occupancy per signer.

Mitigate:

- signer adversarial;
- stronger appearance augmentation;
- crop normalization.

---

## 48.2 Motion target learns camera motion

Mitigate:

- body-relative coordinates;
- stabilize with torso;
- hand-relative motion;
- subtract shoulder/torso flow.

---

## 48.3 Face target learns identity instead of NMM

Mitigate:

- mouth/head subregions;
- stronger photometric augmentation;
- identity adversarial;
- geometric facial target;
- temporal difference target.

---

## 48.4 Cluster target collapses

Monitor:

\[
H(C)
\]

and perplexity.

Re-fit with:

- balanced k-means;
- more data;
- whitening;
- smaller K.

---

## 48.5 Multi-task gradients conflict

Measure cosine between gradients:

\[
cos(g_i,g_j)
=
\frac{g_i^\top g_j}
{\|g_i\|\|g_j\|}
\]

If strongly negative:

- separate decoders;
- PCGrad optional;
- staged training;
- different loss weights.

---

# 49. Essential Control Experiments

## Control 1 — Random labels

Replace cluster IDs with shuffled labels.

Should not improve.

---

## Control 2 — Same-dimensional random projection

Teacher latent vs random frozen projection of pixels/pose.

Tests whether gains simply come from output dimensionality.

---

## Control 3 — Same extra head without target information

Ensures gain not from extra parameters.

---

## Control 4 — Confidence-matched subsets

Compare on clips where pose is high confidence.

If semantic target only wins low-confidence clips, interpretation is robustness—not necessarily richer semantics.

---

## Control 5 — External teacher size

If possible compare:

- small;
- base;
- large teacher.

Show whether gains depend on huge external foundation model.

---

# 50. Proposed Main Paper Table

```markdown
| Method | XYZ | Ang/Dist | Motion | Semantic | NMM | External Teacher | WLASL DCG | WLASL R@1 | ASL-C DCG | ASL-C R@1 | NMFs DCG | NMFs R@1 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| SignRep reproduced | ✓ | ✓ | — | — | — | No | | | | | | |
| No XYZ | — | ✓ | — | — | — | No | | | | | | |
| + Motion | — | ✓ | ✓ | — | — | No | | | | | | |
| + Discrete pose | — | ✓ | — | ✓ | — | No | | | | | | |
| + Semantic units | — | ✓ | — | ✓ | ✓ | Yes | | | | | | |
| + Latent teacher | — | ✓ | — | ✓ | ✓ | Yes | | | | | | |
| Hybrid | — | ✓ | ✓ | ✓ | ✓ | Yes | | | | | | |
```

---

# 51. Baseline Reproduction Table Template

| Run | Dataset snapshot | Seed | Steps | DCG | R@1 | R@5 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| SignRep published | original | — | 500k | | | | paper |
| B02 reproduce #1 | ours | 0 | 500k | | | | |
| B02 reproduce #2 | ours | 1 | 500k | | | | |
| B02 reproduce #3 | ours | 2 | 500k | | | | |

---

# 52. Single-Target Screening Table

| Run | Target | Loss | Teacher | Steps | WLASL Val DCG | R@1 | GPU-h | Keep? |
|---|---|---|---|---:|---:|---:|---:|---|
| E01 | velocity | Huber | pose | | | | | |
| E03 | bone motion | Huber | pose | | | | | |
| E04 | pose token | CE | pose+kmeans | | | | | |
| E07 | multi-stream cluster | CE | DINO+pose | | | | | |
| E08 | hand latent | cosine | DINO | | | | | |
| E13 | NMM latent | cosine | DINO/face | | | | | |
| E16 | articulator MAP | cosine | teacher | | | | | |

---

# 53. Replacement Table

| Run | Base geometry | Removed | Added | WLASL | ASL-C | NMFs | Conclusion |
|---|---|---|---|---:|---:|---:|---|
| B02 | all SignRep | — | — | | | | |
| R01 | angle/dist/activity | XYZ | — | | | | |
| R02 | angle/dist/activity | XYZ | motion | | | | |
| R03 | angle/dist/activity | XYZ | clusters | | | | |
| R04 | angle/dist/activity | XYZ | latent | | | | |
| R05 | angle/dist/activity | XYZ | clusters+motion | | | | |
| R06 | angle/dist/activity | XYZ | clusters+motion+NMM | | | | |

---

# 54. Masking Table

| Mask | Same target? | Ratio | WLASL DCG | ASL-C DCG | NMFs DCG |
|---|---|---:|---:|---:|---:|
| Random | ✓ | 80% | | | |
| Motion-preserving | ✓ | 80% | | | |
| Random articulator | ✓ | 80% equivalent | | | |
| Saliency articulator | ✓ | 80% equivalent | | | |

---

# 55. Robustness Table

| Method | Clean | Pose noise | 20% kp drop | Hand occlusion | Temporal jitter |
|---|---:|---:|---:|---:|---:|
| SignRep | | | | | |
| Motion | | | | | |
| Cluster | | | | | |
| Latent | | | | | |
| Hybrid | | | | | |

---

# 56. Compute Table

| Method | Target extraction GPU-h | Train GPU-h | Peak VRAM | Pretrain-only params | Inference encoder params |
|---|---:|---:|---:|---:|---:|
| SignRep | | | | | |
| Cluster | | | | | |
| Latent | | | | | |
| Hybrid | | | | | |

---

# 57. Decision Tree

## Step 1
Does R01 (remove XYZ) maintain performance?

### Yes
Raw keypoint regression is largely redundant.

Proceed with semantic/motion replacement.

### No
Keep a lower-weight coordinate loss.

---

## Step 2
Does motion-only beat XYZ-only?

### Yes
Explicit dynamics are a better target family.

### No
Check derivative noise and frame-rate normalization.

---

## Step 3
Do clusters beat continuous latent?

### Yes
Discretization likely suppresses nuisance.

### No
Hard quantization may discard useful detail; use soft cluster or latent.

---

## Step 4
Does face/mouth improve NMFs but not WLASL?

Then NMM is task-dependent, not universally better.

Keep as conditional branch.

---

## Step 5
Does hybrid beat individual targets?

If yes:

> geometric invariance + semantic state + motion are complementary.

This is likely the strongest scientific story.

---

# 58. Potential Paper Hypotheses

## H1
**Raw coordinate reconstruction is not the optimal sign prior.**

Evidence expected:

\[
R03/R04 > B02
\]

or:

\[
R01 \approx B02
\]

plus semantic replacement wins.

---

## H2
**Temporal derivatives provide supervision orthogonal to static geometry.**

Expected:

\[
R05 > R03
\]

---

## H3
**Non-manual targets improve linguistically ambiguous signs.**

Expected stronger gain on:

- NMF-heavy subset;
- samples with similar manual features.

---

## H4
**Discrete semantic units are more invariant than continuous geometric targets.**

Expected:

- lower signer leakage;
- better cross-corpus retrieval;
- more robustness to pose noise.

---

## H5
**Sign-aware masking matters only after target quality is solved.**

Expected:

\[
M4>M0
\]

with same target.

---

# 59. Claims We CAN Make If Experiments Succeed

Examples:

> Under identical backbone, pretraining corpus, optimization budget and downstream protocol, semantic multi-stream targets outperform original coordinate-based SignRep priors on frozen cross-corpus retrieval.

> Replacing raw XYZ reconstruction with motion and semantic targets improves transfer while retaining angle and relational distance constraints.

> Non-manual feature prediction provides complementary gains beyond manual geometric supervision.

> Articulator-aware latent prediction improves masked sign representation learning independent of gloss labels.

---

# 60. Claims We MUST NOT Make Without Extra Evidence

Do not claim:

> SHuBERT proves SignRep priors are worse.

Different training and architecture.

Do not claim:

> SignSeek proves masked latent prediction is superior to SignRep.

SignSeek is a 2026 preprint and includes different supervision/data/objectives.

Do not claim:

> SignRep keypoints are useless.

Ablation shows weaker alone, not zero marginal contribution.

Do not claim:

> face target universally helps all sign languages.

Non-manual relevance differs by dataset/language/sign category.

Do not claim:

> cluster IDs are linguistic phonemes.

They are pseudo-units unless validated linguistically.

---

# 61. Recommended First Six Experiments

Nếu team chỉ có ngân sách cho 6 candidate runs sau reproduction:

## 1. R01 — SignRep without raw XYZ
**Question:** XYZ có thực sự cần không?

## 2. R02 — Geometry + velocity/bone motion
**Question:** explicit dynamics có thay XYZ được không?

## 3. R03 — Geometry + SHuBERT-style multi-stream clusters
**Question:** semantic discrete units có thay XYZ được không?

## 4. R04 — Geometry + continuous teacher latent
**Question:** cluster hay continuous latent tốt hơn?

## 5. R06-lite — Geometry + semantic + motion + face
**Question:** hybrid information có complementary không?

## 6. M4 — best target + saliency articulator mask
**Question:** structured masking có thêm gain không?

Nếu chỉ một experiment có thể chạy trước:

> **R03: remove raw XYZ, giữ angle/distance/activity, thêm LH/RH/Face/Body semantic cluster targets.**

Đây là test cân bằng tốt nhất giữa:

- novelty;
- literature support;
- feasibility;
- interpretability;
- khả năng tạo measurable improvement.

---

# 62. Recommended “Paper-Grade” Final Model

Working name:

## SignRep-SemanticMotion

\[
\boxed{
L =
L_{angle}
+
L_{distance}
+
L_{activity}
+
\lambda_mL_{motion}
+
\lambda_sL_{semantic}
+
\lambda_nL_{nonmanual}
+
L_{var}
+
L_{cov}
+
L_{adv}
}
\]

Recommended target composition:

### Manual geometry
- hand angle;
- selected hand/body relational distances.

### Motion
- normalized fingertip/wrist velocity;
- bone motion;
- angular velocity.

### Semantic
- LH/RH hand latent or cluster;
- body stream cluster.

### Non-manual
- face latent;
- mouth latent;
- head pose optional.

### Removed/downweighted
- raw full hand XYZ;
- raw full body XYZ.

---

# 63. Alternative High-Novelty Model

Working name:

## PhonoSignRep

Predict:

\[
\{
Handshape,
Orientation,
Location,
Movement,
NonManual
\}
\]

from masked video.

All labels are pseudo-derived from:

- pose geometry;
- trajectory;
- visual teacher clusters.

No gloss required.

Scientific framing:

> move from reconstructing sensor/estimator outputs to reconstructing linguistically structured latent states.

Potential paper title:

**From Pose to Phonology: Rethinking Pretraining Targets for Sign Language Representation Learning**

---

# 64. Public Code / Resource Index

## SignRep
Paper:  
https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html

arXiv:  
https://arxiv.org/abs/2503.08529

Code:  
https://github.com/ryanwongsa/SignRep

---

## SHuBERT
Paper:  
https://aclanthology.org/2025.acl-long.1397/

Project:  
https://shubert.pals.ttic.edu/

Code:  
https://github.com/ShesterG/SHuBERT

---

## BEST
Paper:  
https://ojs.aaai.org/index.php/AAAI/article/view/25470

DOI:  
https://doi.org/10.1609/aaai.v37i3.25470

---

## MASA
arXiv:  
https://arxiv.org/abs/2405.20666

Code:  
https://github.com/sakura2233565548/MASA

DOI:  
https://doi.org/10.1109/TCSVT.2024.3409728

---

## Spatial-Temporal Consistency SSL
arXiv:  
https://arxiv.org/abs/2406.10501

Code:  
https://github.com/sakura2233565548/Self-Supervised-Representation-Learning-with-Spatial-Temporal-Consistency-for-SLR

DOI:  
https://doi.org/10.1109/TIP.2024.3416881

---

## CCL-SLR
Paper:  
https://openaccess.thecvf.com/content/CVPR2025W/SLRTP/html/Wu_Cross-Modal_Consistency_Learning_for_Sign_Language_Recognition_CVPRW_2025_paper.html

arXiv:  
https://arxiv.org/abs/2503.12485

Code:  
https://github.com/dueToLife/CCL-SLR

---

## SignSeek
arXiv preprint:  
https://arxiv.org/abs/2609.03695

Institutional record:  
https://openresearch.surrey.ac.uk/esploro/outputs/preprint/SignSeek-Learning-Transferable-Representations-for-Sign/991160273502346

---

## SignMAE
arXiv preprint:  
https://arxiv.org/abs/2605.02094

---

## Masked Video Distillation
Paper:  
https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Masked_Video_Distillation_Rethinking_Masked_Feature_Modeling_for_Self-Supervised_Video_CVPR_2023_paper.html

Code:  
https://github.com/ruiwang2021/mvd

---

## MaskFeat
Paper:  
https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html

Code base:  
https://github.com/facebookresearch/SlowFast

---

## data2vec
Paper:  
https://arxiv.org/abs/2202.03555

---

## Non-manual evaluation
Paper:  
https://aclanthology.org/2020.lrec-1.745/

Project:  
https://krslproject.github.io/krsl20/

---

## SignCLIP
Paper:  
https://aclanthology.org/2024.emnlp-main.518/

DOI:  
https://doi.org/10.18653/v1/2024.emnlp-main.518

---

## SAM-SLR
Paper:  
https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html

Code:  
https://github.com/jackyjsy/CVPR21Chal-SLR

---

## SignBERT
Paper:  
https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html

---

# 65. Linguistic / Phonological References

## Deep-learning SLP review
DOI:  
https://doi.org/10.1145/3477498

Use for high-level decomposition:

- handshape;
- orientation;
- location;
- movement;
- facial/non-manual information.

---

## Classification of phonological parameters
arXiv:  
https://arxiv.org/abs/2205.12072

Useful for conceptual H/O/L/M parameterization.

---

## SL CatForm coding schema, 2026
Record:  
https://www.fdr.uni-hamburg.de/record/18270

DOI:  
https://doi.org/10.25592/uhhfdm.18270

Useful as inspiration for structured categorical target design.

---

## HamNoSys parser
Code:  
https://github.com/hearai/parse-hamnosys

Useful only if experiment later includes explicit phonological notation.

---

# 66. Reproducibility Checklist

Before any run:

- [ ] fixed git commit;
- [ ] dataset manifest hash;
- [ ] target cache version;
- [ ] teacher checkpoint hash;
- [ ] pose estimator version;
- [ ] random seed;
- [ ] exact training config;
- [ ] exact downstream config;
- [ ] no test-based hyperparameter tuning.

During run:

- [ ] train losses per head;
- [ ] gradient norms per head;
- [ ] validation frozen retrieval;
- [ ] throughput;
- [ ] GPU memory;
- [ ] target-valid rate;
- [ ] cluster occupancy.

After run:

- [ ] best validation checkpoint;
- [ ] frozen retrieval;
- [ ] recognition;
- [ ] seed aggregation;
- [ ] bootstrap CI;
- [ ] robustness;
- [ ] signer leakage;
- [ ] compute report.

---

# 67. Research Risks

## Risk A — “Better target” actually means “stronger external teacher”
Solution:

Maintain two tracks.

### Pure track
Only pose/video itself, no externally supervised semantic teacher.

Examples:

- velocity;
- bones;
- pose tokens;
- EMA teacher.

### External-teacher track
DINOv2/SignCLIP/video foundation model.

Never merge their claims.

---

## Risk B — Large target dimensionality artificially increases supervision
Control:

- project all latent targets to matched dimension;
- equal decoder capacity where possible;
- report head params.

---

## Risk C — Different masking makes comparison unfair
Use original random 80% masking for Phase 1/2.

---

## Risk D — Downstream fine-tuning hides representation differences
Primary endpoint = frozen retrieval.

---

## Risk E — Dataset leakage
Deduplicate pretraining corpus against:

- WLASL;
- ASL Citizen;
- NMFs-CSL;
- any translation test.

---

# 68. Strongest Expected Scientific Story

Nếu expected experiments thành công, câu chuyện hợp lý nhất sẽ là:

1. SignRep chứng minh sign-specific pretraining targets tốt hơn generic reconstruction.
2. Nhưng current geometric priors still approximate signing as pose-estimator state.
3. Controlled ablations show raw coordinates are not the most transferable target.
4. Motion derivatives encode temporal phonological structure.
5. Semantic units encode appearance and articulatory state robust hơn coordinate regression.
6. Non-manual units recover information missing from manual pose.
7. Geometry vẫn hữu ích như inductive constraint.
8. Best model therefore combines:
   - invariant geometry;
   - dynamics;
   - semantic articulator state;
   - non-manual information.

Conceptually:

\[
\boxed{
\text{Good sign representation}
=
\text{Geometry}
+
\text{Dynamics}
+
\text{Articulator semantics}
+
\text{Non-manual language}
}
\]

chứ không đơn thuần:

\[
\text{Good sign representation}
=
\text{Reconstruct pose coordinates}.
\]

---

# 69. Recommended Immediate Engineering Tasks

## Task 1
Fork/clone SignRep and obtain exact B02 reproduction.

## Task 2
Add modular `TargetProvider` abstraction.

```python
class TargetProvider:
    def __call__(self, sample):
        return {
            "target_name": tensor,
            "valid_mask": tensor,
        }
```

## Task 3
Build target caching pipeline.

## Task 4
Implement motion targets first.

## Task 5
Implement pose k-means tokenizer.

## Task 6
Implement DINO-based crop feature cache.

## Task 7
Add cluster/latent heads.

## Task 8
Automate frozen retrieval evaluation every N checkpoints.

## Task 9
Build experiment registry with run IDs Bxx/Exx/Rxx/Mxx.

## Task 10
Only after R-table is complete, implement articulator-aware masking.

---

# 70. Suggested Repository Layout

```text
signrep-targets/
├── configs/
│   ├── baseline/
│   ├── motion/
│   ├── cluster/
│   ├── latent/
│   ├── nonmanual/
│   └── hybrid/
├── datasets/
│   ├── manifests/
│   └── target_cache/
├── target_providers/
│   ├── signrep_geometry.py
│   ├── motion.py
│   ├── bone.py
│   ├── pose_tokenizer.py
│   ├── semantic_cluster.py
│   ├── teacher_latent.py
│   ├── nonmanual.py
│   └── phonology.py
├── models/
│   ├── heads/
│   └── decoders/
├── eval/
│   ├── retrieval.py
│   ├── recognition.py
│   ├── bootstrap.py
│   ├── robustness.py
│   └── probes.py
├── scripts/
│   ├── extract_pose.py
│   ├── extract_dino.py
│   ├── fit_kmeans.py
│   ├── build_motion.py
│   └── run_ablation.py
└── results/
    ├── tables/
    ├── curves/
    └── statistical_tests/
```

---

# 71. Minimal Core Experiment Budget

Nếu training full 500k rất đắt, minimum publishable logic:

### Required

- B02 × 3 seeds;
- R01 × 3;
- R02 × 3;
- R03 × 3;
- R04 × 3;
- R05/R06 × 3.

Tổng:

18 runs.

Nếu screening trước:

- 100k all candidates;
- 500k top candidates.

---

# 72. Extended Experiment Budget

For strong top-tier submission:

- 3–5 seeds;
- 3 frozen retrieval datasets;
- 2 recognition datasets;
- 1 translation dataset;
- robustness;
- target diagnostic;
- masking ablation;
- loss ablation;
- compute report.

---

# 73. Final Research Recommendation

## Highest confidence

### 1. Replace raw XYZ with semantic clusters
Best balance of evidence and novelty.

### 2. Add explicit motion derivatives
Low engineering risk, strong domain motivation.

### 3. Add non-manual semantic target
Addresses explicit SignRep limitation.

### 4. Keep angle + distance
They are strong relation/invariant priors.

### 5. Test latent prediction against cluster prediction
This is a key scientific question.

### 6. Only later optimize masking
Saliency/articulator masking should not contaminate the initial target comparison.

---

# 74. Concrete Proposed Final Ablation Sequence

Run in exactly this order:

```text
B02  SignRep exact
R01  - raw XYZ
R02  R01 + velocity/bone motion
R03  R01 + hand/body/face semantic clusters
R04  R01 + teacher latent
R05  R01 + clusters + motion
R06  R01 + clusters + motion + NMM
R07  B02 + best new candidate
M01  R06 + motion-preserving mask
M02  R06 + articulator saliency mask
P01  H/O/L/M/N pseudo-target
```

Interpretation matrix:

```text
R01 ~= B02
=> raw XYZ redundant.

R02 > R01
=> dynamics provide missing information.

R03 > R01
=> semantic units are better than raw coordinate target.

R04 > R03
=> continuous latent retains useful detail lost by quantization.

R03 > R04
=> discretization suppresses nuisance and is beneficial.

R05 > max(R02,R03)
=> geometry/motion/semantic complementarity.

R06 > R05
=> non-manual information adds independent signal.

M02 > R06
=> sign-aware masking adds benefit after target quality is controlled.

P01 > R06
=> explicit phonological factorization may be a stronger next-generation target space.
```

---

# 75. Bottom Line

### What is most likely to beat original SignRep priors?

Not a single magical replacement, but a target stack where each component captures a different linguistic axis:

\[
\boxed{
Angles/Relations
+
Motion
+
Semantic Hand Units
+
Face/Mouth Units
}
\]

The weakest-looking component to remove first is **raw XYZ regression**.

The strongest replacement candidates are:

\[
\boxed{
\text{semantic clusters}
\quad\text{or}\quad
\text{teacher latent targets}
}
\]

The strongest complementary target is:

\[
\boxed{
\text{explicit motion residual / velocity / bone motion}
}
\]

The most important missing linguistic target is:

\[
\boxed{
\text{non-manual face/mouth/head information}
}
\]

The highest-novelty long-term direction is:

\[
\boxed{
H/O/L/M/N\ \text{phonology-factorized masked prediction}
}
\]

A successful controlled result here would move the research question from:

> “How can we reconstruct better pose priors?”

to:

> **“What target space best represents the linguistic information carried by sign video?”**

That is the research direction with the clearest potential to extend SignRep into a stronger and more general sign representation framework.

---

# 76. Bibliography / Source Checklist

1. Wong, R.; Camgoz, N. C.; Bowden, R. **SignRep: Enhancing Self-Supervised Sign Representations.** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html

2. Gueuwou, S. et al. **SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction.** ACL 2025.  
   https://aclanthology.org/2025.acl-long.1397/

3. Zhao, W. et al. **BEST: BERT Pre-training for Sign Language Recognition with Coupling Tokenization.** AAAI 2023.  
   https://ojs.aaai.org/index.php/AAAI/article/view/25470

4. Zhao, W. et al. **MASA: Motion-Aware Masked Autoencoder With Semantic Alignment for Sign Language Recognition.** IEEE TCSVT 2024.  
   https://arxiv.org/abs/2405.20666

5. Zhao, W. et al. **Self-Supervised Representation Learning With Spatial-Temporal Consistency for Sign Language Recognition.** IEEE TIP 2024.  
   https://arxiv.org/abs/2406.10501

6. Wu, K. et al. **Cross-Modal Consistency Learning for Sign Language Recognition.** CVPR Workshops 2025.  
   https://openaccess.thecvf.com/content/CVPR2025W/SLRTP/html/Wu_Cross-Modal_Consistency_Learning_for_Sign_Language_Recognition_CVPRW_2025_paper.html

7. Asasi, S.; Mercanoglu Sincan, O.; Bowden, R. **SignSeek: Learning Transferable Representations for Sign Dictionary Retrieval.** arXiv preprint, 2026.  
   https://arxiv.org/abs/2609.03695

8. Xie, K. et al. **SignMAE: Segmentation-Driven Self-Supervised Learning for Sign Language Recognition.** arXiv preprint, 2026.  
   https://arxiv.org/abs/2605.02094

9. Wang, R. et al. **Masked Video Distillation: Rethinking Masked Feature Modeling for Self-Supervised Video Representation Learning.** CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Masked_Video_Distillation_Rethinking_Masked_Feature_Modeling_for_Self-Supervised_Video_CVPR_2023_paper.html

10. Wei, C. et al. **Masked Feature Prediction for Self-Supervised Visual Pre-Training.** CVPR 2022.  
    https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html

11. Baevski, A. et al. **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language.** 2022.  
    https://arxiv.org/abs/2202.03555

12. Mukushev, M. et al. **Evaluation of Manual and Non-manual Components for Sign Language Recognition.** LREC 2020.  
    https://aclanthology.org/2020.lrec-1.745/

13. Jiang, Z. et al. **SignCLIP: Connecting Text and Sign Language by Contrastive Learning.** EMNLP 2024.  
    https://aclanthology.org/2024.emnlp-main.518/

14. Jiang, S. et al. **Skeleton Aware Multi-modal Sign Language Recognition.** CVPR Workshops 2021.  
    https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html

15. Hu, H. et al. **SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition.** ICCV 2021.  
    https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html

16. Review / linguistic decomposition reference. **Deep Learning Methods for Sign Language Translation.** ACM Transactions on Accessible Computing.  
    https://doi.org/10.1145/3477498

17. **Classification of Phonological Parameters in Sign Languages.**  
    https://arxiv.org/abs/2205.12072

18. **SL CatForm coding schema.** 2026.  
    https://www.fdr.uni-hamburg.de/record/18270  
    https://doi.org/10.25592/uhhfdm.18270

---

## End of memo
