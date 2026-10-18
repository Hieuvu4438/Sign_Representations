# Audit chất lượng SignRep trên full ASL-Citizen

Ngày 2026-10-14. Material Passport: academic-research-suite / experiment-agent;
mode validate; Verification Status: ANALYZED. Đọc paper ICCV cuối cùng, supplementary,
frozen runtime code, log và features của run hiện tại. Không chạy lại full training;
không sửa cấu hình, không dừng service, không đánh giá test.

## Kết luận

Có suy giảm representation sau bước 10.000, đi cùng tín hiệu regularization áp đảo
pose reconstruction và style discriminator mất tác dụng. Đây là vấn đề optimization
có bằng chứng, chưa xác định duy nhất chi tiết triển khai gây ra bằng ablation.
Không có cơ sở nói chỉ cần chạy lâu hơn sẽ khắc phục.

## Phạm vi reproduce

Paper dùng continuous YouTube-SL-25, 500.000 updates, warmup 50.000, khởi tạo
Hiera-B MAE Kinetics, chọn checkpoint bằng WLASL validation retrieval. Run này dùng
40.153 video ASL-Citizen isolated signs, scheduler 100.000, warmup 10.000, review ở
50.000 và proxy ASL-Citizen với 2 train gallery + 1 validation video mỗi lớp.
Cùng architecture và initialization không làm hai protocols tương đương.

Table 3 của paper: 81,37% là downstream recognition sau fine-tuning. Table 4:
frozen retrieval ASL-Citizen Top1 = 37,47% với mean, 49,95% với activity weighting.
Proxy của ta khác dictionary/split, nên không so trực tiếp các số đó. Tuy nhiên,
proxy cố định qua checkpoints: sự sụt giảm 10k→20k→30k là vấn đề nội tại của run.

Code tác giả local cung cấp architecture/heads/inference, không cung cấp toàn bộ
trainer, target preprocessing và historical config. Tests/checkpoint parity chứng minh
implementation đáp ứng các quy ước đã chọn, không chứng minh reproduce scores.
Các bất định U01–U10 trong REPRODUCTION_SPEC vẫn còn, nhất là reduction/beta,
Gram normalization, missingness, detach/update ordering và preprocessing.

## Các phép đo mới trên run này

| Bước | Proxy Top1 | Proxy Top5 | MRR | Effective rank gallery | 5 PCs đầu giải thích variance |
|---:|---:|---:|---:|---:|---:|
| 10.000 | 1,8674% | 4,7235% | 0,037758 | 63,35 | 33,71% |
| 20.000 | 0,1098% | 0,4394% | 0,004973 | 6,48 | 88,09% |
| 30.000 | 0,0732% | 0,2197% | 0,003892 | 5,72 | 90,25% |

Effective rank = exp(-Σp log p), từ covariance đã center của toàn bộ 5.462 gallery
vectors L2-normalized. Đây là phân bố biến thiên inference; không phải minibatch
rank, cũng không chứng minh mọi vectors đều bằng nhau. Số hướng biến thiên hữu hiệu
sụt mạnh và thông tin phân biệt sign suy yếu.

Đã đối chiếu video identity của 8.193 pooled caches ở mỗi checkpoint, tính lại
ranking/metrics và khớp số đã lưu. Chưa tìm thấy bằng chứng nhầm labels/checkpoint.
Điều này không chứng minh teacher chính xác tuyệt đối hoặc mọi decoder path đúng.

### 1. Gradient regularization lấn át pose

Trung bình updates 29.001–30.000: reconstruction 1,09368; variance 49,9001;
covariance sau nhân 0,004 = 121,6688; adversarial = 0. Reconstruction khoảng 0,63%
scalar loss. Var/cov dạng sum và weights khớp các phương trình paper: tỷ lệ scalar
không tự chứng minh implementation sai hoặc gradient lấn át.

Đo tiếp trên snapshot bước 37.876, hai batches khác nhau, 20 views từ 10 videos,
training augmentation/drop-path/mask 80%, CPU float32, không cập nhật optimizer:

| Seed | Recon loss | Norm gradient reg / recon tại features | Cosine gradient recon với total | Δrecon khi tráo video embeddings |
|---:|---:|---:|---:|---:|
| 20261014 | 1,11346 | 4.826,5× | 0,01227 | −0,1083% |
| 20261015 | 1,20024 | 3.685,6× | −0,00494 | −0,0372% |

Reg ở đây là variance + 0,004 covariance. Gradient tổng gần vuông góc với gradient
pose. Tráo features giữa các videos khác nhau không làm reconstruction tăng.
Decoder phụ thuộc yếu vào đúng video trong hai probes; loss thấp chưa chứng minh
embedding chứa sign information tốt. Đây không phải replay GPU BF16 hoặc gradient
toàn backbone; muốn chứng minh cải thiện cần controlled ablation.

Điểm cần kiểm tra là cân bằng reduction/scale: reconstruction dùng valid-element
mean với SmoothL1 beta=1, reg dùng sum trên D=768. Không tự chia reg cho 768 rồi
khẳng định đó là recipe tác giả. Với N=20, covariance batch có rank tối đa 19;
diagonal bằng 1/off-diagonal bằng 0 không thể đồng thời đạt trên 768 chiều.
Hạn chế toán học này không riêng run ta, cũng không đủ để quy lỗi cho batch20.

### 2. Style discriminator đã mất khả năng phân biệt

Snapshot 37.876: EMA matched = EMA unmatched = 0,499396861. Trong hai probes dùng
dữ liệu thực, gradient output theo style khoảng 1e-14; matched/unmatched probabilities
giống nhau trong float32. Synthetic probe cho gradient theo content 3,63e-3 nhưng
theo style chỉ 2,31e-13. Norm hai weight matrices style projection khoảng 0,00243 và
0,00268, không phải subnormal như diagnostics ASLC002.

Updates 19.001–20.000 và 29.001–30.000 có adv=0 trên toàn 1.000 updates,
BCE≈0,693147 và EMA gap=0. Gate tắt vì discriminator không phân biệt styles;
không được diễn giải thành encoder đã loại bỏ style thành công. Trước đó, window
4.001–5.000 có BCE≈0,611 và EMA gap≈0,156: nhánh từng hoạt động rồi suy yếu.

Nghi ngờ cần ablate: Gram đang chia K×D rồi mean theo D. Nếu chỉ chia K, scale lớn
hơn 768 lần; normalization này là lựa chọn U04 chưa được xác nhận. Scale tương tác
với Adam coupled weight decay 1e-3 và training ordering. Chưa có ablation xác nhận
đây là nguyên nhân duy nhất hoặc normalization khác là đúng defaults tác giả.

Trong gallery sample 1.024 videos, nearest other video cùng signer chiếm 78,03%,
59,08%, 40,23% tại 10k/20k/30k; tỷ lệ uniform-other theo signer frequencies chỉ 5,92%.
Signer bias còn mạnh. Nhưng bias giảm đồng thời retrieval giảm nên không thể quy
mọi suy giảm cho signer bias.

### 3. Supervision thưa và crop mismatch

Actual training log tới bước 38.153: valid target elements RH≈46,06%, LH≈35,19%,
interaction≈28,32%, activity≈0,265%. Đây là coverage của sampled training windows,
khác phép audit chỉ chọn first valid clip của mỗi video.

Activity hiện đòi đủ 21 hand points và 4 torso references qua toàn bộ 16 frames;
rất ít examples chịu loss. A1 nói visible-hand motion và low position, không công bố
exact missingness aggregation. Policy hiện tại là engineering choice. Chưa xác minh
positive/negative balance toàn corpus; không suy kết luận all-active của pilot sang
full run. Thiếu activity không giải thích hết suy giảm mean pooling.

Training crop signer rồi random crop, inference full frame. Sample 128 train videos
(seed1014) có median bbox/frame area≈77,49%. Đây là distribution mismatch; chưa có
crop-consistent counterfactual retrieval để định lượng tác động.

## Vì sao không thể chỉ gọi là thiếu steps

Paper dùng corpus đa dạng hơn và budget lớn hơn; những khác biệt này ảnh hưởng score.
Nhưng chúng không đủ giải thích metric của cùng proxy giảm khoảng 25 lần từ 10k tới
30k, trong khi gradient/style/geometry đều cho dấu hiệu suy giảm. BF16, một clip bị
loại và những giả thuyết khác chưa được chứng minh là nguyên nhân chính.

Các cảnh báo tương tự đã có ở ASLC002. Tăng batch4→20 và full train chưa giải quyết;
không nên dùng test pass thay thế quality gates hoặc mặc định tăng tới100k sẽ tốt hơn.

## Hướng xử lý đề xuất

1. Giữ run hiện tại và checkpoint10k làm evidence; không tự extend100k bằng recipe này.
   Audit này không dừng job hoặc sửa live source/config.
2. Đo raw Kinetics latent và wrapper step0 trên cùng proxy để tách ảnh hưởng random fc.
   Chưa có phép đo này trong report.
3. Chạy các short ablations cùng seed/membership/views/budget, từng yếu tố riêng:
   cân bằng gradient pose/reg; Gram scale/decay/update ordering. Chấp nhận khi pose
   phụ thuộc đúng RGB, style sensitivity còn hoạt động và validation cải thiện.
   Đổi normalization/loss scale phải được báo là ablation, không exact reproduction.
4. Audit teacher quality/visibility và activity labels trên subset; không gán missing
   thành inactive. Thử eval crop consistent và raw latent/post-fc để tách failure modes.
5. Chọn bằng fixed validation, rồi full validation; test không dùng để cứu recipe.

## Artifacts và nguồn

- training_and_crop_summary.json: train loss, coverage và crop statistics.
- training_gradient_probes.json: seeds/indices và gradient probes.
- discriminator.json: checkpoint identity, norms và style sensitivity.
- pooled_embedding_geometry.json: gallery spectra và sampled query cosines.
- retrieval_identity_and_signer_audit.json: identity checks, recomputed metrics, signer bias.
- latest_training_snapshot.pt: hardlink checkpoint atomic 37.876, không sửa production.
- best_step10000.model.pt: bản bảo tồn checkpoint tốt nhất tại lúc audit.

Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.pdf
Supplement: https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wong_SignRep_Enhancing_Self-Supervised_ICCV_2025_supplemental.pdf

Final PDFs đọc local từ artifacts/legacy/evidence/papers/P01.pdf và P02.pdf.
Primary metadata/search được đối chiếu online; web fetch PDFs trả403.
Không dùng PDF page anchors chưa có preflight.
