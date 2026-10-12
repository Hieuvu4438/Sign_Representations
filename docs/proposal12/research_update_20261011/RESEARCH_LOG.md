# Nhật ký deep research: skeleton JEPA → sign SSL

Ngày bắt đầu: 2026-10-11, Asia/Ho_Chi_Minh. Người thực hiện: Codex, một mô hình; các vòng phản biện được thực hiện tuần tự, không phải đánh giá độc lập bởi nhiều reviewer.

## Phạm vi và tiêu chuẩn hoàn tất

1. Đọc proposal12 và kiểm tra code/eval đang có; phân biệt dữ liệu thực nghiệm với đề xuất, tài liệu cũ và kết quả chưa hoàn tất.
2. Đọc nguồn gốc của S-JEPA, các skeleton predictive/MAE phương pháp liên quan, và sign representation learning (đặc biệt SignRep, SignSeek, body–hand integration).
3. Xác minh paper/version/venue; đọc source code chính thức khi có, ghi commit và đường dẫn. Không lấy bản tái triển khai không chính thức làm bằng chứng code của tác giả.
4. Phân tích tín hiệu handshape, upper-body, location/relations, motion, non-manual; chỉ đề nghị loại tín hiệu khi có đối chứng, không suy từ R² thấp.
5. Đề xuất nhiều phương án với input, encoder, initialization/pretrained, masking, target/loss, inference/readout và kiểm định bác bỏ cụ thể.
6. Thực hiện ít nhất ba checkpoint phản biện: giả định ban đầu → tổng hợp literature → bản đề xuất cuối; lưu các thay đổi sau phản biện.
7. Viết báo cáo tiếng Việt và bảng nguồn có link gốc, giới hạn, kế hoạch đối chứng và provenance. Không thay code training, không khởi chạy/dừng job GPU.

## Câu hỏi nghiên cứu

- RQ-A: Với cùng handshape stream, thêm hình học/chuyển động upper-body có tăng chất lượng sign embedding frozen qua người ký mới không?
- RQ-B: Sai số probe hiện tại đến từ pose, chuẩn hóa/tokenization, mục tiêu học, pooling hay mismatch hệ tọa độ?
- RQ-C: Masking/latent target nào từ skeleton JEPA có bằng chứng đủ mạnh để chuyển sang 2D sign, và điều gì không thể chuyển trực tiếp?
- RQ-D: Khi nào nên bổ sung một tín hiệu, thay cách biểu diễn, hoặc dùng teacher khác modality thay vì loại bỏ bộ phận?

## Tìm kiếm và tiêu chí

Nguồn ưu tiên: ECVA/CVF/ACL/proceedings chính thức, PDF/HTML arXiv của tác giả, trang dự án và repository tác giả. Bao phủ nền tảng 2021–2026, cập nhật tới ngày tìm kiếm; các kết quả NTU không được so trực tiếp với ASL Citizen. Bài supervised/text-supervised được giữ làm bằng chứng thiết kế, tách khỏi label-free SSL. Không tuyên bố systematic review/PRISMA hay coverage tuyệt đối.

Nhóm truy vấn: skeleton JEPA/S-JEPA/Skeleton2vec/SLiM/Path-JEPA; motion-aware masked skeleton/MAMP/GFP/AMR; SignRep/SignSeek/SignBERT/BEST/MASA/STC; whole-body/hands/face sign SSL/SHuBERT; corruption/2D–3D MotionBERT; JEPA dense targets/collapse.

## Checkpoint 1: phản biện phạm vi

Verdict: REVISE → sửa khung đánh giá trước khi tổng hợp.

- Major: “body chưa có” không chính xác tuyệt đối: LOC đã có elbow/wrist, shoulders làm anchor; thiếu body stream độc lập và torso dynamics. Đánh giá thêm UBody phải tách tăng thông tin khỏi tăng token/capacity.
- Major: Handshape R²≈0.277 chưa chứng minh các chiều handshape gây ra retrieval tốt; cần bỏ/đổi tín hiệu có đối chứng.
- Major: R² thấp của motion/face có thể do window averaging, neutralization và coordinate/handedness mismatch; không dùng làm lý do xóa đầu vào.
- Major: SignSeek dùng same-gloss supervision; không sao chép positive sampling đó vào label-free SSL rồi vẫn gọi phương pháp không nhãn.
- Minor: Proposal/README còn ghi “chưa train” nhưng log hiện có P1; dùng JSON và manifest làm nguồn trạng thái.

Phản đề mạnh nhất: mở rộng body chỉ thêm morphology/signer nuisance; lợi ích hiện tại có thể đến từ lịch train hoặc readout thay vì mask/JEPA. Báo cáo phải nêu phép thử có thể bác bỏ từng đề xuất.

## Tiến độ nguồn và eval

- 2026-10-11 ~21:09: P1 đã hoàn tất; P2 đang chạy `PJ_R4both`; chưa có P2.json/P3.json. Runner PID 108344 và child train được xác nhận live bằng ps. Không suy kết quả từ checkpoint chưa eval.
- Đã phát hiện literature cũ có các mệnh đề chưa được chứng minh bằng full text (Path-JEPA), khác nhau về nhận diện SLiM, và đường dẫn raw cache không còn tồn tại. Sẽ xác minh lại thay vì kế thừa nhãn “verified”.

## Nhật ký tiếp theo

### Nguồn và code đã thu thập

- `source_downloads.json` + `source_supplement.json`: 21 PDF khác nhau tải được, gồm S-JEPA, Skeleton2vec, MAMP, GFP, AMR, SLiM, Path-JEPA, ASMa, SignRep, SignSeek, BEST, MASA, STC, SignBERT/+, MotionBERT, SHuBERT, SignDino, I-JEPA, LeJEPA, V-JEPA 2.1. Có SHA256, thời điểm tải, bản text và kết quả PDF preflight. SignBERT preflight UNAVAILABLE nên không dùng page anchor của bản này; các PDF còn lại PASS. Không xem UNAVAILABLE như chứng nhận đọc toàn bộ PDF an toàn.
- BEST: tải endpoint AAAI thất bại; chuyển sang PDF arXiv v1, xác minh venue/DOI qua landing page AAAI. ASMa: endpoint PDF OpenReview thất bại; đọc PDF/HTML arXiv v1, bản PDF ghi TMLR 01/2026. Các thất bại giữ nguyên trong manifest.
- SLiM: đọc v3 (03/08/2026), tên khác v1; danh mục NeurIPS 2026 đã có bài dưới tên “Feature Learning”, khác “Feature Prediction” trên v3. Không gán v3 thành proceedings final.
- 8 repository snapshot chính thức, commit đầy đủ trong `code_downloads.json`. Đọc chọn lọc SLiM masking/backbone/config/loss wiring, MAMP motion sampling/target, STC hand-body graph/temporal fusion, MotionBERT config/mapping/pretrain, SignRep inference/model. Skeleton2vec, Path-JEPA, AMR hiện chỉ có README/trang dự án; không có implementation để xác nhận chi tiết paper. Repo tác giả S-JEPA tìm thấy nhưng rỗng lúc kiểm tra; không dùng bản làm lại của bên thứ ba làm code chính thức.
- Không chạy code tải từ tác giả, không tải trọng số, không dùng GPU. Các clone chỉ là tài liệu đọc.

### Bổ sung từ người dùng

Người dùng gửi `https://eccv.ecva.net/virtual/2026/poster/4270` và `https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf` trong khi đang nghiên cứu. Đã xác nhận trang poster liên kết đúng PDF Path-JEPA, tải 18 trang PASS và đọc §§3–4.5/Tables 1,4–6. Nguồn publisher xác nhận ECCV 2026, first online 02/10/2026, DOI `10.1007/978-3-032-37258-1_7`. Kết luận “không có full text Path-JEPA” ở lượt trước đã được thay bằng phân tích full text. Repository vẫn chỉ có trang dự án, chưa có code.

### Snapshot local

`local_snapshot/provenance.json`, P1 JSON, CSV tóm tắt, hai manifest 16k và schedule_state đã lưu lúc 21:21:32 +07. P1 complete; P2/P3 JSON chưa xuất hiện. PLctx/PLint đều seed 0, batch 32, 16.000 optimizer steps, 3.000 video. P1 arm 4k không phải đối chứng đồng ngân sách cho 16k. JH3D/JV3D/SR3D là encoder RGB/tham chiếu khác recipe; probes của chúng không phải ablation upper-body trong PhonoJEPA.

## Checkpoint 2: phản biện tổng hợp literature

Verdict: REVISE → thực hiện lượt tra cứu nhắm vào phản chứng, rồi sửa chiến lược.

1. **Không có bằng chứng “JEPA luôn hơn reconstruction”.** AMR (CVPR 2026) vẫn dùng reconstruction có cải tiến và có kết quả mạnh; GFP dùng target MLP cùng học + variance/covariance, không cần EMA teacher lớn. Giữ đối chứng MAMP/MAE và đánh giá chi phí.
2. **Không có mask ratio chuẩn chung.** S-JEPA/MAMP 90% token, SLiM ngân sách 50–90%, Path-JEPA 20% joint trước dependency closure: khác mẫu số và lượng dữ liệu bị che. Không so trực tiếp các tỷ lệ.
3. **Không mặc định ưu tiên high-motion.** Đọc thêm ASMa: hai hướng masking bổ sung stable/dynamic, nhưng fusion được học có nhãn ở downstream nên “linear evaluation” của tổ hợp không phải linear head đơn thuần. SHuBERT có phản chứng: random spans được chọn sau ablation, không phải whole-channel luôn tốt hơn.
4. **Path signature có thể làm mất thứ sign cần.** Static path có signature tăng lượng bằng 0; vị trí và tư thế tĩnh phải giữ qua nhánh level/CONF. Lead–lag fixed delay và resampling gây mất chi tiết không được bảo đảm bất biến tuyệt đối chỉ bằng định lý reparameterization.
5. **Global+local token làm thay đổi masking.** Path-JEPA che joint ở mọi window cùng các edge/chain phụ thuộc. Nếu đem global signature lặp lại vào PLCM che thời gian cục bộ, global component có thể chứa vùng đích. Phải truy support theo raw joint×frame; không chỉ thêm channel ID vào mask sampler.
6. **Dữ liệu chứ không chỉ encoder.** JH3D có movement/relations tốt và handshape thấp hơn PLctx nhưng retrieval tốt hơn; B13 relations cao hơn nhiều song retrieval thấp. Không suy quan hệ nhân quả từ bảng probe.
7. **Protocol có tính transductive.** LOSO whitening fit trên evaluated windows, chuẩn hóa X trong geometric probes cũng dùng toàn bộ windows. Ghi rõ và thêm kiểm định inductive fit-train-only trong kế hoạch, giữ protocol cũ để continuity.

Sau phản biện, thứ tự đề xuất đổi thành: sửa trace/probe/readout → body/relations tối thiểu → target motion/signature bổ sung → graph architecture → teacher ngoài domain. Không mở đầu bằng thay toàn bộ encoder hay xóa face.

## Checkpoint 3

Verdict: PASS sau chỉnh sửa, với giới hạn bằng chứng được ghi rõ; đây là tự phản biện tuần tự, không phải independent review.

- Major được sửa trong lượt cuối: SignBERT+ xuất 133 keypoints từ detector không đồng nghĩa đưa body vào pretraining encoder. Đọc lại method/implementation details và xóa nhận định unsupported về body stream khỏi source card. Không kế thừa claim của related work khác.
- Major: không đánh đồng probe cao với lexical utility. Bảng kết quả ghi cả phản ví dụ B13 và JH3D; strategies yêu cầu ablation đồng budget và coverage/minimal-pair evaluation trước khi bỏ tín hiệu.
- Major: Path-JEPA chuyển từ thiếu full text thành verified ECCV primary PDF; phân tích riêng static information loss và global/local support leakage khi adaptation sang PL temporal masks. Kiểm tra thêm tài liệu thư viện iisignature về Witt dimensions; discrepancy 22/21 được gọi là convention chưa rõ, không kết luận paper sai.
- Major: reference modality/external pretraining khác được ghi rõ; S4/S5 tách khỏi strict same-corpus SSL. Missing legs của MotionBERT không bị giả định là xử lý được bằng zero-fill.
- Minor: sửa khả năng hiểu nhầm “16k context”, “visible-only là tiết kiệm compute”, “B0 D8 là P1 B0”, “ASMa fusion là linear head”, “SignSeek MAP dùng EMA”.

## Audit hoàn tất

| Yêu cầu | Bằng chứng bàn giao |
|---|---|
| Đọc proposal12 và re-audit code/eval | Báo cáo §§2–5, P1 snapshot/hash/CSV và manifests 16k |
| Skeleton JEPA / masking trên phạm vi rộng | 21 primary PDFs; R01–R08/R19–R21, có reconstruction và masking phản chứng |
| SignRep/SignSeek/UBody | R09–R18; báo cáo input-ownership và supervised-vs-SSL distinction |
| Hai URL Path-JEPA người dùng bổ sung | Poster xác nhận PDF; nguồn R07 và phân tích §6.1/S2 |
| Source code chính thức | 8 commit snapshots, 5 repos có model/training code đọc chọn lọc; 3 placeholders được chỉ rõ |
| Input/encoder/pretrained/mask/ép học | S0–S5 đều có cấu hình và kiểm định bác bỏ; G0–G6 cô lập factors |
| Giữ/bỏ/thay signal theo bằng chứng | §5, không loại face/motion từ probe âm; retraining controls và guardrails |
| Vòng lặp phản biện → nghiên cứu → sửa | CP1–CP3, các nguồn/claims bổ sung và sửa ghi ở trên |
| Markdown tiếng Việt và nguồn đầy đủ | Báo cáo chính + LITERATURE_AND_CODE + README bundle |

Kiểm tra tự động đã thực hiện: mọi local Markdown link tồn tại; 21 PDF hashes khớp manifests (20 PASS/1 UNAVAILABLE); 8 repo HEAD/file lists khớp manifest; metric và paired CI của bảng khớp P1 JSON; số source cards và strategies đủ. Không chạy tests của training vì không sửa training code. Các hash source local tại snapshot được đối chiếu lại trong audit cuối.

Không còn việc cần làm để bàn giao **nghiên cứu/chiến lược đã yêu cầu**. P2/P3 downstream kết quả vẫn pending tại lần kiểm tra cuối; đây là công việc scheduler độc lập, không phải kết quả đã có hay thí nghiệm mới của báo cáo. Xem `local_snapshot/final_status.json` cho thời điểm live-check. Việc triển khai/chạy S0–S5 và giải đáp conventions từ code Path-JEPA tương lai là các bước tiếp theo, không được giả là đã thực hiện.

Live-check cuối **21:39:16 +07**: scheduler PID 108344 còn chạy; `PJ_R4both` đã train xong và đang extract val. Vẫn chỉ có P1.json, chưa có P2/P3.json. Audit 20 source/eval/manifest hashes không đổi, mọi local links PASS; bản ghi máy đọc được ở `completion_audit.json`.
