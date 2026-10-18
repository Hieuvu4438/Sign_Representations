## Experiment 1 - Pose-Jepa Hybrid

### 1. Tổng quan và mục tiêu

Pose-JEPA Hybrid là phương pháp học representation cho ngôn ngữ ký hiệu từ chuỗi pose 2D. Model học bằng cách che một phần thông tin của clip rồi dự đoán representation, hình học và chuyển động tại các vị trí bị che. Mục tiêu là tạo feature giữ được cấu hình bàn tay, vị trí tương đối giữa các bộ phận và diễn biến theo thời gian, để phục vụ dictionary retrieval và các tác vụ downstream.

JEPA viết tắt của Joint Embedding Predictive Architecture. Trong thí nghiệm này, tên implementation là `pose_jepa_hybrid_v1`; full run là `pose_jepa_full_aslc_hybrid_100epochs`. Đây là adaptation riêng theo proposal14, không phải kết quả reproduction nguyên bản của một paper đã công bố.

Đầu ra gồm sequence features theo thời gian và một clip embedding 1.024 chiều. Giá trị ngữ nghĩa của embedding được đánh giá qua retrieval và frozen linear probe; việc tối ưu SSL loss không tự chứng minh model nhận diện gloss tốt.

### 2. Dữ liệu đầu vào và các bộ phận cơ thể

Encoder nhận pose cache COCO-WholeBody gồm tọa độ 2D, confidence, validity, source frame indices, timestamps và track continuity.

$$
X\in\mathbb{R}^{T\times133\times2},\qquad Q\in[0,1]^{T\times133}
$$

Ở đây, T là số frames của clip; X chứa tọa độ x,y; Q là độ tin cậy. Các feature đầu vào của mỗi joint là x,y,q. Confidence là reliability, không phải tọa độ chiều sâu z.

Model chọn 69 keypoints từ 133 điểm:

| Part | Số joints | Thông tin cần giữ |
|---|---:|---|
| Body | 9 | Tư thế thân trên, vai, khuỷu và cánh tay |
| Left hand | 21 | Cấu hình bàn tay và ngón tay trái |
| Right hand | 21 | Cấu hình bàn tay và ngón tay phải |
| Face | 18 | Một phần tín hiệu mặt và đầu |
| Tổng | 69 | Cấu trúc cơ thể phục vụ biểu diễn ký hiệu |

Pose extractor nằm ngoài encoder. Full run sử dụng pose đã được trích sẵn; detector không được huấn luyện cùng Pose-JEPA. Cache nguồn giữ all-person raw scores và áp conversion q=s/(1+s) cho finite nonnegative SimCC scores. Mapping này là uncalibrated reliability; weights detector nguồn chưa được xác minh.

### 3. Chuẩn hóa không gian, validity và quality

Tọa độ đầu vào được đưa về viewport-normalized coordinates. Trên các frames có hai vai hợp lệ, lấy trung vị tâm hai vai làm origin và trung vị khoảng cách vai làm scale:

$$
o=\operatorname{median}_{t\in\mathcal{V}_{sh}}\frac{x_{t,Lsh}+x_{t,Rsh}}{2},\qquad
s=\operatorname{median}_{t\in\mathcal{V}_{sh}}\left\|x_{t,Lsh}-x_{t,Rsh}\right\|_2
$$

$$
g_{t,j}=\frac{x_{t,j}-o}{s}
$$

Reference được tính một lần trên toàn clip trước khi subsampling, windowing hoặc artificial masking. Khi không đủ shoulder support hoặc scale không hợp lệ, clip được đánh dấu normalization_failed.

Body sử dụng tọa độ trong hệ reference cơ thể. Tay và mặt dùng local coordinates quanh root tương ứng, đồng thời giữ global anchor để không mất vị trí của part so với cơ thể:

$$
\ell_{t,j}^{(p)}=g_{t,j}^{(p)}-g_{t,r_p}^{(p)},\qquad
a_t^{(p)}=\left[g_{t,r_p}^{(p)};q_{t,r_p}^{(p)}\right]
$$

Ví dụ, cấu hình ngón tay được mô tả bởi local coordinates; global anchor giúp phân biệt tay ở gần mặt hay gần ngực. Ngưỡng confidence là 0,3. Một part-frame cần ít nhất 50% joints hợp lệ; local part còn yêu cầu root hợp lệ. Token quality được tính từ confidence và support quan sát thực tế.

Đây là chuẩn hóa offline sử dụng toàn clip, không phải causal streaming. Masking diễn ra ở cấp part-segment tokens sau chuẩn hóa; không được mô tả là che tuyệt đối mọi thông tin tọa độ raw trước preprocessing.

### 4. Spatial frontend và part-segment tokenization

Mỗi part-frame đi qua Spatial GCN, học quan hệ giữa các joints theo graph của bộ phận. GCN chỉ xử lý trong một frame và một part; quan hệ xuyên thời gian được học ở Transformer.

Spatial channels là 64→64→128→256, sử dụng channel LayerNorm, GELU và residual connections. Joint features được flatten theo thứ tự joints rồi project, thay vì chỉ lấy joint mean. Hai tay chia sẻ frontend weights. Global anchor của tay/mặt được encode qua MLP và fuse với spatial feature.

Hai frames liên tiếp tạo thành một segment. Với từng part, hai frame features và hai validity bits được concatenate rồi project thành token 256 chiều:

$$
z_{k,p}=\operatorname{LN}\left(W_{seg}\left[h_{2k,p};h_{2k+1,p};v_{2k,p};v_{2k+1,p}\right]+b_{seg}\right)
$$

Mỗi segment có bốn tokens theo thứ tự body, left hand, right hand, face. Token có positional encoding thời gian và learned part embedding. Một token hợp lệ khi ít nhất một trong hai part-frames hợp lệ; padding không tạo observation mới.

$$
K=\left\lceil\frac{T}{2}\right\rceil,\qquad N=4K,\qquad z_{k,p}\in\mathbb{R}^{256}
$$

Với 128 frames, có 64 segments và tối đa 256 part tokens. Khi train isolated signs, clips dài hơn 128 frames được subsample đều trên toàn clip; timestamps và source indices được giữ để xác định thời gian thực giữa các quan sát.

### 5. Student, predictor và EMA teacher

Kiến trúc training có ba thành phần:

| Thành phần | Cấu hình | Vai trò |
|---|---|---|
| Student encoder | Transformer 8 layers, dim 256, 8 heads, FFN 1.024 | Encode các visible tokens; học quan hệ part/time |
| Predictor | Transformer 4 layers, dim 128, 4 heads, FFN 512 | Dự đoán latent và tạo hidden features cho physical heads |
| Teacher encoder | Cùng kiến trúc encoder, cập nhật EMA | Đọc toàn bộ valid tokens và tạo latent targets |
| Geometry head | MLP cho từng part; hai tay chia sẻ weights | Dự đoán tọa độ local/body của hai frames trong segment |
| Motion head | Bộ physical heads riêng | Dự đoán displacement có hiệu chỉnh thời gian |

Student Transformer chỉ nhận các visible tokens được pack. Predictor project feature từ 256 xuống 128 chiều, đưa context về vị trí ban đầu, chèn learned mask embeddings theo part vào target positions, rồi xử lý chuỗi valid positions. Latent head project predictor output lên 256 chiều.

$$
H_V=f_{\theta}(Z_V),\qquad
\widehat{Z}_M=p_{\phi}\left(H_V,\operatorname{mask\_tokens}_M,\operatorname{positions}\right),\qquad
Z^{teacher}=f_{\bar\theta}(Z_{valid})
$$

Teacher targets được stop-gradient. Student encoder, predictor và physical heads được cập nhật bằng optimizer; teacher không được cập nhật bằng backpropagation.

$$
\bar\theta\leftarrow m\bar\theta+(1-m)\theta
$$

Teacher bắt đầu từ bản sao student. Momentum tăng theo cosine từ 0,999 về 1,0 trên budget successful optimizer updates:

$$
m(u)=1-\frac{1-m_0}{2}\left[1+\cos\left(\pi\frac{u}{U}\right)\right],\qquad m_0=0.999
$$

U là tổng budget updates; u là chỉ số update trong schedule. EMA tạo mục tiêu thay đổi chậm. Cơ chế này hỗ trợ ổn định training, nhưng không thay thế kiểm tra representation collapse hoặc downstream quality.

### 6. Part-temporal masking

Mask generator chọn khoảng 60% valid tokens làm targets. Một nửa ngân sách mask được tạo bằng contiguous spans của một part; phần còn lại được chọn từ các vị trí hợp lệ theo part weights.

| Thuộc tính | Giá trị |
|---|---|
| Mask ratio mục tiêu | 60% valid tokens |
| Part weights body/LH/RH/face | 0,15 / 0,30 / 0,30 / 0,25 |
| Span budget | 50% tổng số target tokens |
| Độ dài span | 2–8 segments |
| Visible tối thiểu | Ít nhất 20% và ít nhất 4 tokens |
| Ràng buộc mỗi segment | Giữ ít nhất một part visible nếu segment có valid support |
| Ràng buộc tay | Giữ tối thiểu hai valid segments mỗi tay nếu có đủ support |

$$
\mathcal{M}\subseteq\mathcal{V}_{natural},\qquad
\mathcal{V}_{visible}=\mathcal{V}_{natural}\setminus\mathcal{M}
$$

Natural validity mask mô tả dữ liệu có thật; artificial mask tạo nhiệm vụ dự đoán. Targets không được tạo trên invalid tokens. Actual mask ratio có thể dưới 60% do các constraints. Run này chưa bật motion-aware masks, learned saliency hoặc hand canonical flip.

### 7. Hybrid objective

Objective kết hợp latent prediction với các physical targets:

$$
\mathcal{L}=\mathcal{L}_{latent}+0.25\mathcal{L}_{geometry}+0.10\mathcal{L}_{motion}
$$

Ba terms được normalize riêng theo tổng valid quality support của effective optimizer batch. Các hệ số 0,25 và 0,10 là cấu hình recipe, không phải tỷ lệ đóng góp thực nghiệm cố định.

#### 7.1. Latent cross-entropy

Implementation dùng centered/sharpened cross-entropy giữa distributions trên các chiều latent, không phải gloss classification loss. Teacher feature sau trừ center được softmax với temperature 0,06; predictor dùng temperature 0,10:

$$
P^{t}_{i}=\operatorname{softmax}\left(\frac{z^{t}_{i}-c}{\tau_t}\right),\qquad
P^{s}_{i}=\operatorname{softmax}\left(\frac{\widehat z_i}{\tau_s}\right),\qquad
\tau_t=0.06,\quad\tau_s=0.10
$$

$$
\mathcal{L}_{latent}=\frac{\sum_{i\in\mathcal{M}}w_i\left[-\sum_{d=1}^{256}P^t_{i,d}\log P^s_{i,d}\right]}{\sum_{i\in\mathcal{M}}w_i}
$$

w là token quality từ confidence và validity. Center được cập nhật bằng weighted teacher target mean trên masked support, momentum 0,9:

$$
c\leftarrow0.9c+0.1\frac{\sum_{i\in\mathcal{M}}w_i z^t_i}{\sum_{i\in\mathcal{M}}w_i}
$$

Các distributions có 256 chiều latent, không đại diện cho 2.731 gloss labels. Zero-support terms không tạo optimizer contribution; implementation bảo vệ phép chia cho zero support.

#### 7.2. Geometry reconstruction

Geometry head dự đoán tọa độ pose chuẩn hóa tại masked parts. Geometry targets dùng local coordinates cho tay/mặt và body-reference coordinates cho body. Local roots của tay/mặt được loại vì tọa độ bằng 0 theo định nghĩa chuẩn hóa.

$$
\mathcal{L}_{geometry}=\frac{\sum_{(t,j,p)\in\mathcal{S}_g}a_pq_{t,j,p}\,\rho_{0.05}(\widehat\ell_{t,j,p}-\ell_{t,j,p})}{\sum_{(t,j,p)\in\mathcal{S}_g}a_pq_{t,j,p}}
$$

Trong công thức, local/body target đều được ký hiệu gọn là ell; Sg chứa masked, observed, real-frame và root-supported targets. a là part weight: body 0,25, mỗi tay và face 1,0. rho là mean Smooth L1 trên hai tọa độ x,y, với beta 0,05.

#### 7.3. Timestamp-supported motion prediction

Motion target được tính trên global body-normalized trajectories, trước local-root subtraction. Delta dùng hai quan sát liên tiếp trong chuỗi được đưa vào training và timestamps gốc:

$$
d_{t,j,p}=\left(g_{t+1,j,p}-g_{t,j,p}\right)\frac{1/25}{\Delta t_t},\qquad
\Delta t_t=\operatorname{timestamp}_{t+1}-\operatorname{timestamp}_{t}
$$

Đây là displacement quy về khoảng thời gian tham chiếu 1/25 giây, không phải FPS cố định giả lập. Target chỉ hợp lệ khi timing known, timestamps hữu hạn, track liên tục, hai joints observed và khoảng thời gian trong (0;0,2] giây.

$$
\mathcal{L}_{motion}=\frac{\sum_{(t,j,p)\in\mathcal{S}_m}a_pq_{t,j,p}q_{t+1,j,p}\,\rho_{0.02}(\widehat d_{t,j,p}-d_{t,j,p})}{\sum_{(t,j,p)\in\mathcal{S}_m}a_pq_{t,j,p}q_{t+1,j,p}}
$$

Motion support yêu cầu target part tại thời điểm bắt đầu thuộc artificial mask. Loss dùng Smooth L1 beta 0,02. Tích confidence của hai quan sát giảm ảnh hưởng của các trajectories thiếu tin cậy.

### 8. Một training update diễn ra như thế nào?

1. Lấy batch từ official train; load pose, canonicalize toàn clip và subsample nếu dài hơn 128 frames.
2. Tạo part-segment validity/quality và artificial masks; xác định geometry/motion support và denominators.
3. Teacher encode toàn bộ valid tokens, không gradient, lấy latent targets tại masked positions.
4. Student encode visible tokens; predictor và physical heads dự đoán latent/geometry/motion targets.
5. Tính quality-weighted losses, backward, kiểm tra finite gradients và gradient clipping, rồi optimizer update.
6. Sau successful update, cập nhật EMA teacher và latent center; tăng step và ghi metrics/checkpoint theo lịch.

Stage SSL không dùng gloss/text trong encoder inputs hoặc loss. Gloss labels vẫn được dùng để định nghĩa protocol, tính validation retrieval và chọn checkpoint. Cách mô tả supervision là pose-only self-supervised pretraining với target-validation selection.

### 9. Inference và readout

Inference dùng EMA encoder từ validation-selected checkpoint. Predictor và physical heads không nằm trong inference encoder. Encoder có 10.255.872 parameters; raw output không có learned metric projector trong thí nghiệm này.

Chuỗi dài được chia thành windows 128 frames, stride 64, aligned theo two-frame segments. Features tại cùng segment trong nhiều windows được average trước khi clip pooling. Validity được xét riêng từng part.

$$
\bar h_p=\frac{\sum_k v_{k,p}h_{k,p}}{\max(1,\sum_kv_{k,p})},\qquad
e_{clip}=\left[\bar h_{body};\bar h_{LH};\bar h_{RH};\bar h_{face}\right]\in\mathbb{R}^{1024}
$$

Sequence output có khoảng ceil(T/2) bước, mỗi bước 1.024 chiều do concatenate bốn part features. Clip output là một vector 1.024 chiều. Part không có support tạo zero contribution; clip không đủ observation nhận failure status rõ ràng.

PCA/whitening của evaluation là adapter sau feature extraction. Chúng không phải thành phần đã học bằng Hybrid SSL objective. Frozen linear probe là bước supervised riêng trên encoder cố định; chưa thực hiện fine-tuning encoder hoặc Stage 2 metric adaptation trong full run này.

### 10. Cấu hình Experiment 1 trên full ASL-Citizen

| Thuộc tính | Thiết lập full run |
|---|---|
| Dataset | ASL-Citizen, 2.731 glosses |
| Train | 40.154 clips, 35 signers |
| Validation | 10.304 clips, 6 signers |
| Test | 32.941 clips, 11 signers |
| Initialization | Scratch; chỉ reuse pose preprocessing |
| Seed | 42 |
| Batch | 32, accumulation 1 |
| Precision | BF16 nếu GPU hỗ trợ |
| Optimizer | AdamW, betas (0,9;0,95), weight decay 0,05 |
| LR schedule | Peak 3e-4, warmup 5.000 updates, cosine xuống 1e-5 |
| Budget | Tối đa 125.482 successful updates, khoảng 100 source epochs |
| Gradient clipping | Global norm 1,0 |
| Validation interval | Mỗi 5.000 updates và final update |
| Selection metric | Raw dictionary validation Rec@1 |
| Early stopping | 5 lượt validation liên tiếp không cải thiện Rec@1 |
| Checkpoint | Mỗi 100 updates, tại validation và graceful stop |
| Export encoder | EMA từ best validation checkpoint |

Validation chững lại không chứng minh dataset đã được học hết. Model được chọn bằng metric đã chốt; thay đổi recipe hoặc tăng data cần là một thí nghiệm được ghi riêng. Các budget, modality và external pretraining của baseline phải được công bố khi so sánh.

### 11. Protocol validation, test và metrics

Gallery dùng toàn bộ official train. Validation dùng toàn bộ official val; official test được encode/evaluate sau khi hoàn tất SSL checkpoint selection. Official split IDs và signer membership không trùng nhau. Full run dùng official Participant IDs và Gloss từ CSV; hashed signer IDs/aliases của pose cache subset chỉ được map để reuse preprocessing.

Với clip embedding được L2-normalize, cosine là inner product. Điểm một gloss bằng max video similarity trong gallery:

$$
\widetilde e=\frac{e}{\|e\|_2},\qquad
S(q,c)=\max_{g\in\mathcal{G}_c}\widetilde e_q^{\top}\widetilde e_g
$$

Rank dùng pessimistic ties, đếm các gloss khác có điểm lớn hơn hoặc bằng gloss đúng:

$$
r_q=1+\sum_{c\ne y_q}\mathbf{1}\left[S(q,c)\ge S(q,y_q)\right]
$$

$$
\operatorname{Rec@K}=\frac{1}{N_q}\sum_q\mathbf{1}[r_q\le K],\qquad
\operatorname{MRR}=\frac{1}{N_q}\sum_q\frac{1}{r_q},\qquad
\operatorname{DCG}=\frac{1}{N_q}\sum_q\frac{1}{\log_2(1+r_q)}
$$

| Metric | Vai trò |
|---|---|
| Rec@1 | Tỷ lệ gloss đúng đứng đầu; dùng chọn SSL checkpoint |
| Rec@5 / Rec@10 | Tỷ lệ gloss đúng trong top 5 / top 10 |
| MRR | Nhấn mạnh việc đưa gloss đúng lên các vị trí đầu |
| DCG | Giảm đóng góp theo logarithm thứ hạng; single-relevant-gloss formulation |
| Macro Rec@1 | Trung bình Rec@1 từng gloss, giảm ảnh hưởng mất cân bằng số clips |
| Coverage | Số valid/failed gallery và query, hỗ trợ kiểm tra mẫu số |
| Frozen linear-probe Top-1/Top-5 | Khả năng phân tách gloss bằng classifier tuyến tính trên feature cố định |

Invalid gallery không tạo candidate; invalid/missing-positive query nhận rank infinity và vẫn ở mẫu số, nên đóng góp 0 cho Recall/MRR/DCG. Raw và White512 reports được trình bày riêng. Whitening chỉ fit valid train/gallery features, không fit query/test.

Frozen linear probe dùng raw1.024 features, train-only standardization và linear softmax head; AdamW lr1e-3, weight decay1e-3, batch512, tối đa30epochs, validation patience5. Head chọn bằng validation, sau đó đánh giá instance/macroTop-1/Top-5 trên test. Recipe này khác historical ASLC300 sklearn C-search, cần khai báo khi so sánh.

Cùng validation query IDs chưa đủ để gọi cùng protocol: LOSO bên trong val có gallery khác train→val dictionary retrieval. Không so trực tiếp các bảng này. Test được giữ ngoài SSL checkpoint selection; nếu dùng test để sửa recipe/chọn lại model thì không còn là đánh giá cuối độc lập theo thiết kế này.

### 12. Giả thuyết, giới hạn và phạm vi kết luận

Giả thuyết nghiên cứu là latent prediction giúp học ngữ cảnh, còn geometry/motion giữ tín hiệu vật lý của ký hiệu. Đây là động lực thiết kế; muốn xác định contribution phải chạy ablation JEPA-only, geometry-only, motion-only và Hybrid với cùng data, selection và budget.

Các giới hạn chính gồm chất lượng keypoint extraction, thiếu chiều sâu/orientation trong pose2D, chỉ 18 face points và khả năng embedding còn giữ đặc điểm signer. Local normalization không bảo đảm signer invariance. Physical targets là pose-derived proxies, không phải nhãn ngôn ngữ học gán thủ công. Low SSL loss không đồng nghĩa high retrieval accuracy.

Full job hiện mô tả một seed42. Chưa có bằng chứng multi-seed robustness hoặc kiểm định vượt baseline. Body-part/signer probes không nằm trong final evaluation schedule của full job này; không coi metrics của subset3k là kết quả full run. Continuous recognition và translation cần downstream training với transcripts/translations riêng.

### 13. Nguồn gốc thiết kế và implementation evidence

- [S-JEPA](https://sjepa.github.io/): tham khảo EMA teacher, predictor và centered/sharpened latent CE. Run này chưa bật motion-aware masks của nguồn tham khảo.
- [Uni-Sign](https://arxiv.org/abs/2501.15187): tham khảo pose2D COCO-WholeBody và phân chia body/hands/face. Experiment này không chạy generative sign-text pretraining hoặc RGB fusion của Uni-Sign.
- [SignRep](https://arxiv.org/abs/2503.08529): động lực cho physical priors và frozen representation evaluation. Encoder pose2D hiện tại khác RGB backbone và priors3D của nguồn tham khảo.

Evidence trong repository: `method/pose_jepa/runtime/full_aslc_job/train.yaml`; source snapshot `runtime/full_aslc_job/code/signrepr/`; `docs/method_provenance.md`; `docs/FULL_ASLC_BACKGROUND_JOB_VI.md`. Frozen training source SHA256: `90bc239a06688ba3b73af17a471aca85ebad61bbf47fc7e2d3a1d2b2c11d75a9`.

Tài liệu này mô tả method và protocol đã triển khai. Các kết quả theo checkpoint và test cuối được báo cáo riêng cùng scope; không suy ra chất lượng full benchmark từ các correctness/smoke tests.
