# Nhật ký tìm kiếm, nguồn và giới hạn

## Phạm vi và cách tìm

- Ngày: **19/10/2026**, giới hạn nguồn công bố tới thời điểm này.
- Research question: tín hiệu skeleton nào có ích cho frozen sign dictionary retrieval như SignRep, và cần đối chứng nào để chọn module?
- Cách làm: targeted literature review + backward snowballing từ SignRep, SHuBERT, MASA và SignSeek; mở rộng sang phonology, face, dataset và skeleton action SSL.
- Nguồn ưu tiên: CVF official proceedings, ACL Anthology, NeurIPS/ICLR proceedings, OpenReview TMLR, arXiv tác giả và project/resource chính thức.
- Nguồn thứ cấp xuất hiện trong search results chỉ dùng phát hiện ứng viên; không dùng blog, ResearchGate hay AI paper reviews làm căn cứ scientific claim.
- Đọc tài liệu nội bộ để định vị: `docs/research/Skeleton_SSL_Sign_Research_Design_VI.md`, literature A, `docs/experiments/method_comparison_20261019`, dataset README và data/joint adapters.

Các họ query đã dùng, giữ tên paper để có thể lặp lại qua công cụ tìm kiếm:

| Họ | Query tiêu biểu | Mục đích |
|---|---|---|
| Sign SSL chung | `self supervised skeleton sign language representation learning retrieval SignBERT SignCL SignRep` | Breadth và nhận diện literature chính |
| Cập nhật gần đây | `self supervised sign language representation skeleton retrieval SPOT SignDINO 2025 2026`; `site:arxiv.org sign language skeleton self supervised representation 2026 retrieval` | Recent methods và retrieval |
| Masked hand models | `SignBERT hand model aware Hu site:openaccess.thecvf.com/content/ICCV2021`; `BEST coupling tokenization site:ojs.aaai.org`; `SignBERT+ 2023 TPAMI` | Primary source và venue |
| Retrieval | `SignSeek 2609.03695` | Trực tiếp đọc method/protocol/ablation |
| Temporal/motion | Exact titles của MASA, STC, SSL-SLR; arXiv IDs mở trực tiếp | Input/target/objective separation |
| Current self-distillation | `Learning Effective Sign Features without Text Gan CVPR 2026 SignDINO`; metadata `2609.06296` | Phân biệt SignDINO và SignDEG, kiểm tra revision |
| Phonology | `Phonological Perception of Sign Language Models 2606.28667`; `Phonological Representation Learning for Isolated Signs conference` | Diagnostic/OOV/composition |
| Non-manual/nuisance | Exact titles của facial-features và pose appearance transfer; `CoSign Exploring Co-occurrence ICCV 2023` | Face/identity/interaction controls |
| Action SSL | `Masked Motion Predictors skeleton ICCV 2023`; `Skeleton2vec 2024`; `Self-Supervised Skeleton Joint Embedding ECCV 2024`; `3D Human Action Representation Learning via Cross-View Consistency` | Objective baselines, domain boundary |
| Dataset | `ASL Citizen dataset NeurIPS 2023 83399 2731 signer independent`; `ASL LEX 2.0 database phonological 2723 signs`; exact WLASL/MS-ASL/How2Sign/YouTube-ASL titles | Data applicability và scale |
| Pose scale | `SignVerse 2M 2605.01720` và versioned project/paper | Pose-native corpus, latest version |

Inclusion: đóng góp trực tiếp cho sign SSL/retrieval, tín hiệu/cơ chế quan trọng, linguistic diagnostic, hoặc dataset cần thiết cho protocol. Giữ action SSL dưới nhãn ngoại suy. Loại các bài chỉ trùng từ khóa, review thứ cấp thay nguồn gốc, hoặc duplicate version. Peer-reviewed sources ưu tiên; preprint quan trọng vẫn giữ và ghi rõ trạng thái.

Không có tổng số hit/screened/excluded đáng tin từ giao diện tìm kiếm; **không dựng PRISMA flow**. Số 34 là selected evidence matrix, không là recall của toàn lĩnh vực. Metadata/tên tác giả được ghi gọn phục vụ nghiên cứu, chưa xuất bibliography APA hoàn chỉnh cho manuscript.

## Việc kiểm chứng thực hiện

1. Truy xuất **18 văn bản HTML phiên bản**, tương ứng **16 arXiv paper IDs**; gồm hai bản lịch sử để phát hiện thay đổi phiên bản. Đọc method/results và tables liên quan; không nhận là đọc hết mọi phần của cả 18 bản.
2. Các metadata/abstract/project còn lại có mức đọc riêng trong matrix. Không nâng abstract-only thành full-text evidence.
3. Numeric claims của SignRep: Tables 3–5; SignSeek: Tables 1–2, 7–9; STC: Table V; VQ-ASL: Table 3. Dùng table/section anchors, không dùng PDF page anchors.
4. Phonology pairs: phần datasets của Yin et al.; các pair này thuộc evaluation data, không chuyển sang dev mặc định.
5. SignDEG: latest abs metadata và HTML v2. SignVerse: latest abs metadata và HTML v3. Không lấy tên cũ làm current title.
6. ASL Citizen local: đếm CSV, signer overlap, gloss overlap và ASL-LEX mapping; SHA256 trong JSON. Không score model, không dùng test labels để lựa chọn arm. Chạy lại audit bằng `python docs/research/skeleton_signal_retrieval_20261019/audit_metadata.py` từ repository root.
7. Số internal results là đọc artifact đã lưu, **không tái lập bằng weights**. Working tree đã có thay đổi từ công việc khác; nghiên cứu mới chỉ thêm thư mục này.
8. Đọc `method/p4_signssl/p4/retrieval.py`, `method/jepa_baseline/src/signrep_baseline/retrieval.py` và upstream example usage: xác nhận local code có khác biệt ties/activity handling, nên report yêu cầu khóa contract thay vì gán parity mặc định.

`source_retrieval.json` lưu URL, số ký tự, số table parser thấy, SHA256 của text extraction. Full text tạm ở `/tmp/sign_signal_review_20261019`, không đưa paper full text vào Git. Hash bảo đảm trace một extraction cụ thể, không chứng minh parser đúng hết công thức/bảng.

## Evidence / inference / recommendation

- **Evidence:** tác giả báo gì, trên task/split/supervision nào; local audit đo gì.
- **Inference:** chuyển từ các evidence sang lý do cần kiểm tra handshape/location/motion/context; không gọi là kết quả đã biết.
- **Recommendation:** arm, budget, practical margin, perturbation và module-selection rules do nghiên cứu này đề xuất, chưa chạy.
- **Secondary evidence:** MASA frozen ASLC 27,96% là do SignSeek Table 2 báo, chưa kiểm tra bằng code/checkpoint MASA gốc.

## Hạn chế và phản biện tự kiểm tra

| Nguy cơ | Cách báo cáo/xử lý |
|---|---|
| Thiên lệch về ASL và isolated signs | Chủ đích theo task người dùng; continuous/cross-language là replication, không claim universality |
| Chỉ đọc sâu các paper thuận với prior đang quan tâm | Giữ cả negative lesson VQ-ASL và thất bại internal; không chọn evidence chỉ có gain |
| Gộp SSL với supervised metric learning | SignSeek/gloss, SignCLIP/text, VQ-ASL/Phono và downstream training ghi riêng |
| Tên paper/recipe đổi theo revision | Pin versions, ghi khác biệt SignDino→SignDEG và SignVerse 25+→55+ |
| Confounding trong so sánh method | Không quy numeric gap cho một stream; yêu cầu same-code matched controls |
| Frozen perturbation bị OOD | Ghép retrain-minus và train-time missingness; chỉ gọi sensitivity khi chưa có counterfactual hợp lệ |
| Metadata phonology thiếu hoặc variant không khớp | Split/join theo code/lexeme, báo coverage và nhờ expert kiểm tra cặp mơ hồ |
| Test đã được dùng nhiều lần trong repo | Công khai lịch sử; khóa thiết kế mới trên val, cần corpus mới cho confirmation độc lập mạnh |
| Không có training mới | Mọi arm là planned; không ghi số dự kiến thành measured |

Đợt này không phải đánh giá độc lập đa reviewer hay cross-model review. Ba bước tự phản biện được thực hiện inline: (i) RQ/protocol có trả lời đúng task không; (ii) evidence có vượt phạm vi supervision/modality không; (iii) final claims có phân biệt thiếu bằng chứng với bằng chứng không có hiệu ứng không.

Không có claim “đầu tiên”, “duy nhất”, “chưa ai làm” hoặc SOTA tuyệt đối dựa trên lượt tìm kiếm này. Văn bản được AI hỗ trợ tổng hợp từ nguồn đã truy xuất; người dùng cần xác minh lại bibliography và các claim trọng yếu trước khi đưa vào manuscript.
