# Đặc tả thực thi end-to-end cho AI Agent: nghiên cứu Sign Representations

**Phiên bản:** 1.0 — 30/09/2026  
**Ngôn ngữ:** Tiếng Việt; tên file, cấu hình và CLI dùng tiếng Anh.  
**Máy đích theo người dùng:** ptitiec; các đường dẫn ở dưới phải được kiểm tra trên máy đích.  
**Đầu ra yêu cầu:** repository thí nghiệm, manifest, features, kết quả, phân tích phản biện và kết luận có bằng chứng.  
**Phạm vi:** tái sử dụng representation của SignRep/SHuBERT để nghiên cứu ngữ pháp phi thủ công; chuyển sang temporal grounding nếu dữ liệu phù hợp hơn. Không mặc định phải vượt điểm recognition/translation của paper gốc.

> Tài liệu này là chỉ dẫn để agent thực thi, không phải báo cáo về những thí nghiệm đã chạy. Các đường dẫn dataset do người dùng cung cấp; tác giả tài liệu chưa truy cập server để xác nhận file bên trong. Các script thuộc project ở phần sau là **script agent phải tạo**, trừ những script được đánh dấu **UPSTREAM VERIFIED**. Không được coi một tên script trong kế hoạch là code đã tồn tại.

## Mục lục

1. [Hợp đồng nhiệm vụ](#1-hợp-đồng-nhiệm-vụ)
2. [Dataset đã có và chính sách không tải lại](#2-dataset-đã-có-và-chính-sách-không-tải-lại)
3. [Câu hỏi nghiên cứu và nhánh quyết định](#3-câu-hỏi-nghiên-cứu-và-nhánh-quyết-định)
4. [Nguồn paper bắt buộc](#4-nguồn-paper-bắt-buộc)
5. [Source code và revision](#5-source-code-và-revision)
6. [Checkpoint và nguồn tải](#6-checkpoint-và-nguồn-tải)
7. [Nguồn dữ liệu bổ sung](#7-nguồn-dữ-liệu-bổ-sung)
8. [Bootstrap trên máy đích](#8-bootstrap-trên-máy-đích)
9. [Schema và adapters dữ liệu](#9-schema-và-adapters-dữ-liệu)
10. [Khóa protocol và kiểm soát leakage](#10-khóa-protocol-và-kiểm-soát-leakage)
11. [Môi trường và tích hợp backbone](#11-môi-trường-và-tích-hợp-backbone)
12. [Feature cache và smoke tests](#12-feature-cache-và-smoke-tests)
13. [Baseline chung luôn phải thực hiện](#13-baseline-chung-luôn-phải-thực-hiện)
14. [Track A: grammar và semantic verification](#14-track-a-grammar-và-semantic-verification)
15. [Track B: query-by-example temporal grounding](#15-track-b-query-by-example-temporal-grounding)
16. [Thích nghi representation có điều kiện](#16-thích-nghi-representation-có-điều-kiện)
17. [Đánh giá thống kê và kiểm tra cơ chế](#17-đánh-giá-thống-kê-và-kiểm-tra-cơ-chế)
18. [Task graph và hợp đồng CLI](#18-task-graph-và-hợp-đồng-cli)
19. [Tài nguyên, scheduling và recovery](#19-tài-nguyên-scheduling-và-recovery)
20. [Quy tắc đưa ra kết luận](#20-quy-tắc-đưa-ra-kết-luận)
21. [Deliverables và definition of done](#21-deliverables-và-definition-of-done)
22. [Prompt bàn giao cho agent](#22-prompt-bàn-giao-cho-agent)
23. [Checklist cuối](#23-checklist-cuối)

## 1. Hợp đồng nhiệm vụ

### 1.1. Agent phải hoàn thành gì?

Agent thực hiện liên tục các bước độc lập còn khả thi, không dừng ở việc đề xuất kế hoạch:

1. Kiểm kê dữ liệu có sẵn, annotation, môi trường, GPU và dung lượng.
2. Đọc paper/README/code liên quan; ghi evidence và pin revision.
3. Chuẩn hóa dataset thành manifest mà không sửa dữ liệu nguồn.
4. Thiết lập code và checkpoint, kiểm tra tải đúng trọng số.
5. Chạy feature extraction trên subset trước, rồi baseline chung.
6. Xác định Track A hay B có đủ dữ liệu để kiểm chứng.
7. Chạy diagnostic trước khi thiết kế method; chỉ mở rộng khi gate đạt.
8. Thực hiện đối chứng, ablation, transfer và thống kê phù hợp.
9. Audit lỗi, đối chiếu prior art, viết kết luận giới hạn đúng phạm vi.
10. Xuất toàn bộ cấu hình, provenance, log, prediction và báo cáo tái lập.

Không được tối ưu cho việc tìm một kết quả dương. Kết quả âm hoặc thiếu dữ liệu là kết luận hợp lệ nếu được chứng minh và ghi đúng.

### 1.2. Ràng buộc cứng

- **Không tải lại bất kỳ dataset nào trong mục 2.**
- Không sửa, rename, move, xóa hoặc ghi cache vào hai thư mục dữ liệu nguồn.
- Không tự đồng nhất CSL-Daily với một dataset CSL khác; không tự đồng nhất ASL, CSL, DGS và BSL.
- Không gọi nhãn suy từ bản dịch/gloss/punctuation là gold non-manual grammar.
- Không gọi CTC alignment, subtitle alignment hoặc boundary dự đoán là ground truth temporal.
- Không dùng ASL-MTP để train/tune rồi báo cáo nó như external test.
- Không dùng kết quả NLL/BLEURT công bố của tác giả làm input/label cho model mới.
- Không invent số liệu, dataset size thực tế, GPU runtime, hash hoặc link checkpoint.
- Không gọi dictionary retrieval, QA, phonology-VQ, headshake detection hoặc online processing là task lần đầu xuất hiện.
- Không kết luận encoder mất hoàn toàn thông tin chỉ vì một probe thất bại.
- Không gọi adaptation dùng semantic/span labels là pure SSL.
- Không thay đổi test protocol sau khi xem kết quả để tạo improvement.
- Không gửi video/dữ liệu nguồn lên dịch vụ demo bên ngoài. Code demo công khai có thể được tải và chạy local.
- Không gửi email, mở issue hay liên hệ tác giả tự động. Nếu thiếu quyền truy cập hoặc annotation, ghi yêu cầu cụ thể vào báo cáo.

### 1.3. Cách ghi trạng thái

Mỗi task có một trong các trạng thái:

| Trạng thái | Ý nghĩa |
|---|---|
| PENDING | Chưa thực hiện |
| RUNNING | Đang chạy; có command, PID/job ID và log |
| PASS | Có artifact và kiểm tra đạt |
| FAIL | Đã chạy nhưng kiểm tra/thí nghiệm không đạt |
| BLOCKED_DATA | Thiếu dữ liệu/annotation có tính quyết định |
| BLOCKED_ACCESS | Không đọc được đường dẫn, checkpoint hoặc nguồn được yêu cầu |
| BLOCKED_COMPUTE | Tài nguyên thực tế không đủ trong budget |
| SKIPPED_NOT_APPLICABLE | Không phục vụ track đã chọn; phải ghi lý do |

BLOCKED không được đổi thành PASS. Task downstream phụ thuộc task blocked phải giữ blocked hoặc chuyển nhánh có cơ sở. Agent vẫn hoàn thành các task độc lập và báo cáo cuối.

### 1.4. Tách bằng chứng

Trong evidence ledger, phân biệt:

- PAPER_FACT: nguồn công bố; ghi URL, version, section/table.
- CODE_OBSERVATION: quan sát ở commit cụ thể.
- LOCAL_MEASUREMENT: số đo từ server này; ghi manifest/run ID.
- PROPOSED: thiết kế hoặc kỳ vọng cần kiểm chứng.
- UNKNOWN: thông tin chưa xác minh.

Một paper claim không phải LOCAL_MEASUREMENT. Thiếu thí nghiệm trong paper không tự động là empirical failure.

## 2. Dataset đã có và chính sách không tải lại

### 2.1. Roots do người dùng cung cấp

| ID root | Đường dẫn | Chính sách |
|---|---|---|
| ROOT_A | /home/shared_data/sign_language | READ_ONLY; ưu tiên bản shared khi xác nhận đầy đủ |
| ROOT_B | /home/dongvk/datasets | READ_ONLY; dùng các dataset đã được người dùng liệt kê |

Agent phải kiểm tra existence, permission, symlink target và nội dung. Không mặc định việc một folder tồn tại đồng nghĩa đủ video, annotation và split.

### 2.2. Danh sách từng dataset/folder

| Tên chuẩn trong project | Đường dẫn ứng viên đã được người dùng nêu | Vai trò dự kiến | download_allowed |
|---|---|---|---|
| csl_daily | /home/shared_data/sign_language/CSLDaily | Transfer/auxiliary CSL, gloss/sentence tasks | false |
| how2sign | /home/shared_data/sign_language/How2Sign | Continuous ASL; paired text; ứng viên grammar/spotting nếu có annotation phù hợp | false |
| how2sign_alt | /home/dongvk/datasets/How2Sign | Bản thứ hai; kiểm tra trùng và khác version | false |
| asl_citizen | /home/dongvk/datasets/ASL_Citizen | Isolated ASL; lexical retention; query exemplars | false |
| ms_asl | /home/dongvk/datasets/MS-ASL | Isolated ASL; external lexical control | false |
| wlasl2000 | /home/dongvk/datasets/WLASL2000 | Isolated ASL; external lexical control | false |
| youtube_asl | /home/dongvk/datasets/youtubeASL | Unlabeled/paired ASL nếu caption và IDs tồn tại | false |
| phoenix14t | /home/dongvk/datasets/phoenix14T | Continuous DGS; transfer/QA auxiliary | false |
| csl_daily_sentence_crop | /home/dongvk/datasets/CSL_Daily_Sentence_Crop | Derived CSL clips; phải ánh xạ tới corpus gốc | false |
| places365 | /home/dongvk/datasets/places365 | Không dùng trong pipeline chính | false |
| local_exploration_scripts | /home/dongvk/datasets/script_data_exploring | Đọc để hiểu format; không chạy mù | false |
| local_download_scripts | /home/dongvk/datasets/src_for_download | Tham khảo provenance; không dùng để tải lại dataset | false |

Tên CSL-Daily/CSL_Daily/CSLDaily trong mô tả không được tạo thêm ba bản dữ liệu. Chỉ chuẩn hóa alias sau khi kiểm tra metadata.

### 2.3. Các quyết định bắt buộc

1. Nếu cả hai bản How2Sign tồn tại, lập comparison theo video IDs, annotation versions, splits và sampled file hashes; chọn một canonical root.
2. Giữ official splits khi baseline phục vụ so sánh công bố. Tạo protocol mới ở file riêng.
3. Đối chiếu CSL_Daily_Sentence_Crop với CSLDaily: ghi parent video, interval và các thay đổi crop/resample.
4. phoenix14T có thể là image sequences thay vì mp4; adapter phải hỗ trợ frames directory.
5. youtubeASL không tự động là YouTube-SL-25. Ghi đúng corpus/version từ metadata; không giả định toàn bộ nội dung bằng tên folder.
6. script_data_exploring và src_for_download là folders code, không tính vào số dataset khoa học.
7. Nếu dataset có trong danh sách nhưng không đọc được: báo MISSING_EXISTING_DATA hoặc BLOCKED_ACCESS; không tự tải một bản thay thế.
8. Đối với file lỗi/mất: ghi exclusion manifest. Không phục hồi bằng download tự động.

### 2.4. Dữ liệu đang có chưa bảo đảm điều gì?

CSL-Daily/PHOENIX/How2Sign sentence hoặc gloss annotations không mặc nhiên cung cấp:

- Temporal boundaries của mọi sign.
- Scope của non-manual grammatical markers.
- Negative clips đã xác nhận sign vắng mặt.
- Minimal pairs tự nhiên, cân bằng theo hai chiều.
- Signer-disjoint hoặc lexical-disjoint splits phù hợp câu hỏi mới.

Agent phải audit các năng lực này trước khi chạy track tương ứng.

## 3. Câu hỏi nghiên cứu và nhánh quyết định

### 3.1. Track A — mặc định ưu tiên

**Mục tiêu:** đo và học representation giữ được chức năng ngữ pháp phi thủ công, phân biệt thông tin trong encoder với thông tin decoder sử dụng.

Tasks:

- A1: frozen information-access probing trên labels ngữ pháp có nguồn đáng tin.
- A2: semantic verification bằng video và hai candidates gần nhau.
- A3: marker/function recognition và scope localization nếu có annotation theo thời gian.
- A4: frozen transfer sau adaptation; kiểm tra lexical retention.

**Điều kiện mở Track A:** có input giữ cue, nguồn annotation hợp lệ và train/dev độc lập với external test. Nếu thiếu scope labels, A3 bị blocked; không đổi sentence labels thành scope labels.

### 3.2. Track B — nhánh dự phòng

**Mục tiêu:** query-by-example localization trong continuous signing, kiểm tra transfer dưới coarticulation.

Input: query video q và continuous video V.  
Output: intervals của sign tương ứng hoặc absent.

**Điều kiện mở Track B:** query-to-lexical mapping cùng ngôn ngữ, occurrence labels đáng tin, test có absent cases và split thích hợp.

Không đủ occurrence labels thì chỉ được làm weakly supervised exploratory alignment, không temporal-mAP claim dựa trên gold.

### 3.3. Baseline chung — luôn thực hiện nếu feature extractor chạy được

Với ASL Citizen/WLASL/MS-ASL đã có, chạy frozen lexical probes và dictionary retrieval có kiểm soát. Mục đích:

- Xác nhận adapter/feature pipeline hoạt động.
- Đo lexical information trước/sau adaptation.
- Tạo artifact khoa học hữu ích khi A/B blocked.

Baseline chung không được tự đổi thành claim novelty về dictionary retrieval; SignSeek là prior gần.

### 3.4. Decision gate G0

| Dữ liệu sau audit | Hành động |
|---|---|
| Grammar train/dev hợp lệ và test độc lập tồn tại | Chạy A; B chỉ làm transfer nếu phù hợp |
| Grammar thiếu nhưng spotting gold hợp lệ tồn tại | Chạy B; ghi lý do chuyển track |
| Chỉ có lexical/sentence/gloss labels | Hoàn thành baseline chung, exploratory diagnostics; xuất annotation request; chưa train method dựa trên gold giả |
| Model hoặc data access bị chặn | Hoàn thành inventory/evidence/available baselines; kết luận giới hạn và blocker |

Không chạy cả ba hướng đầy đủ chỉ để tăng số benchmark. Chọn một scientific claim chính.

## 4. Nguồn paper bắt buộc

### 4.1. Core papers: phải đọc phần liên quan trước implementation

| Paper ID | Công trình | Nguồn chính | Agent phải lấy gì |
|---|---|---|---|
| P01 | SignRep, ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.pdf | Input, objective, evaluation settings, feature pooling |
| P02 | SignRep supplement | https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Wong_SignRep_Enhancing_Self-Supervised_ICCV_2025_supplemental.pdf | Training, model selection, limits, implementation |
| P03 | SHuBERT, ACL 2025 | https://arxiv.org/pdf/2411.16765v3 ; https://aclanthology.org/2025.acl-long.1397/ | Four streams, layers, preprocessing, task-specific adaptation; Appendix B |
| P04 | ASL-MTP linguistic analysis | https://arxiv.org/html/2604.27232v3 ; https://arxiv.org/pdf/2604.27232v3 | Evaluation by minimal translation pairs; decoder confounds; limits |
| P05 | Lost in Expression, LREC workshop 2026 | https://www.sign-lang.uni-hamburg.de/lrec/pub/26056.html | Non-manual generalization; which model actually tested |
| P06 | SignSeek, arXiv 2026 | https://arxiv.org/abs/2609.03695 | Frozen retrieval, use of gloss supervision, subtitle alignment |
| P07 | VQ-ASL, arXiv 2025 | https://arxiv.org/abs/2509.04745 | Phonology/VQ/OOV prior art |
| P08 | Grounding in Phonology, LREC workshop 2026 | https://www.sign-lang.uni-hamburg.de/lrec/pub/26012.pdf | Linguistic subspaces; difference from grammar/scope |
| P09 | Phonological Perception, arXiv 2026 | https://arxiv.org/abs/2606.28667 | Minimal-pair representation protocol, data availability |
| P10 | SignQA, arXiv 2026 | https://arxiv.org/abs/2607.27826 | Existing QA, annotation generation, non-manual questions |
| P11 | Scaling up sign spotting, IJCV 2022 | https://arxiv.org/abs/2205.04152 ; https://www.robots.ox.ac.uk/~vgg/research/bsldict/ | Isolated-to-continuous spotting prior and baselines |
| P12 | Sign Spotting Disambiguation, 2025 | https://arxiv.org/abs/2507.03703 | Dictionary/DTW/LLM prior; input assumptions |
| P13 | Online CSLR/SLT, EMNLP 2024 | https://aclanthology.org/2024.emnlp-main.619/ | Avoid stale claims about online processing/distillation |
| P14 | Drop-DTW, NeurIPS 2021 | https://proceedings.neurips.cc/paper_files/paper/2021/hash/729c68884bd359ade15d5f163166738a-Abstract.html | Alignment baseline; do not claim invented algorithm |
| P15 | Calibrate Before Use, ICML 2021 | https://proceedings.mlr.press/v139/zhao21c.html | Prior calibration inspiration; not evidence it works for signs |
| P16 | Headshake annotation, LREC workshop 2026 | https://lrec.elra.info/lrec2026-ws-signlang-21 | Existing grammatical headshake detection |
| P17 | Pose-Based Sign Spotting, WSLP 2025 | https://aclanthology.org/2025.wslp-main.10/ | Existing query-video presence prediction |

### 4.2. Expansion papers: đọc để positioning, không bắt buộc triển khai

| ID | Paper / nguồn | Vì sao đọc |
|---|---|---|
| P18 | SignCLIP, EMNLP 2024 — https://aclanthology.org/2024.emnlp-main.518/ | Sign-text alignment và multilingual pretraining đã có |
| P19 | Privacy-aware SSVP-SLT, ACL 2024 — https://aclanthology.org/2024.acl-long.467/ | RGB SSL/privacy context |
| P20 | Uni-Sign, ICLR 2025 — https://openreview.net/forum?id=0Xt7uT04cQ | Unified tasks không tự là novelty |
| P21 | SLU-2K — https://arxiv.org/abs/2606.03788 | Semantic evaluation of translations |
| P22 | CNSL-bench — https://aclanthology.org/2026.acl-long.1896/ | Existing sign-understanding benchmark |
| P23 | SignBind-LLM — https://arxiv.org/abs/2509.00030 | Expert fusion; không chỉ thêm face/lipreading stream |
| P24 | UPRet — https://arxiv.org/abs/2405.19689 | Uncertainty-aware retrieval prior |
| P25 | SOKE, ICCV 2025 — https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html | Generation/tokenization prior; không thuộc track chính |
| P26 | M3T — https://arxiv.org/abs/2603.23617 | Motion-token prior |
| P27 | SAGE, ICCV workshop 2025 — https://openaccess.thecvf.com/content/ICCV2025W/MSLR/html/Low_SAGE_Segment-Aware_Gloss-Free_Encoding_for_Token-Efficient_Sign_Language_Translation_ICCVW_2025_paper.html | Segment/token efficiency prior |
| P28 | SignMAE — https://arxiv.org/abs/2605.02094 | Segmentation-driven masking prior |
| P29 | PHONSSM — https://arxiv.org/abs/2604.08761 | Compositional phonology proposal; verify workshop status |
| P30 | ASL Citizen — https://arxiv.org/abs/2304.05934 | Dataset, split và retrieval assumptions |
| P31 | How2Sign, CVPR 2021 — https://how2sign.github.io/ | Modalities, sentence alignment và access terms |
| P32 | YouTube-ASL, NeurIPS 2023 — https://arxiv.org/abs/2306.15162 | Corpus identity và pretraining overlap |
| P33 | YouTube-SL-25 — https://arxiv.org/abs/2407.11144 | Không nhầm với local youtubeASL |
| P34 | Sem-Lex — https://arxiv.org/abs/2310.00196 | Expert mappings, phonology resources |

Agent tải paper PDF nếu máy chưa có, lưu URL/version/hash và extract text. Nếu link lỗi, tìm bản tương ứng ở arXiv/ACL/CVF/trang tác giả; không thay bằng bài khác cùng tên gần giống. Không coi venue workshop/preprint là main-conference A*.

### 4.3. Literature deliverable

Tạo evidence/literature_matrix.csv với columns:

~~~text
paper_id,title,version,venue_status,url,read_level,relevant_sections,
claim,explicit_limit,nearest_to_track,what_is_already_done,remaining_question
~~~

read_level: FULL_RELEVANT / ABSTRACT_ONLY / METADATA_ONLY. Không đánh FULL nếu chỉ đọc abstract.

## 5. Source code và revision

### 5.1. Repositories chính đã kiểm tra

Các commit dưới đây là snapshot xác minh ngày 30/09/2026. Agent clone và checkout chính xác; nếu cần cập nhật, ghi lý do, diff và pin mới.

| Code ID | Public source | Revision đã xác minh | Dùng cho |
|---|---|---|---|
| C01 | https://github.com/ryanwongsa/SignRep | 06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2 | SignRep feature extraction |
| C02 | https://github.com/ShesterG/SHuBERT | cc1929326075bfbad7ad73159b2acf84356059bb | Original SHuBERT preprocessing/inference |
| C03 | https://github.com/serpilkarabuklu/SL-Models-Analysis | 55054020dfe61263360cab703d59282406031817 | ASL-MTP metadata and published analysis |
| C04 | https://huggingface.co/spaces/ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText | 69d3d77aa4a4fec89048f5417d44fa717e36f6f8 | Author demo code; local translation adapter |
| C05 | https://github.com/gulvarol/bsldict | 8629693ae210062a7cdb41373123ba2ca4e5d856 | Spotting baseline/reference |
| C06 | https://github.com/kayoyin/sign-phonology | 25c563189cb2321fc2134289f888eff85e3fabd4 | Representation diagnostics reference |
| C07 | https://github.com/SamsungLabs/Drop-DTW | Resolve and lock at execution | Alignment implementation |
| C08 | https://github.com/facebookresearch/dinov2 | Resolve and lock at execution | DINO architecture used by SHuBERT |
| C09 | https://github.com/microsoft/ASL-citizen-code | Resolve and lock if used | ASL Citizen metadata/baseline reference |
| C10 | https://github.com/dxli94/WLASL | Resolve and lock if used | WLASL annotation format |
| C11 | https://github.com/google-research/google-research/tree/master/youtube_asl | Pin parent commit if used | YouTube-ASL metadata documentation |
| C12 | https://github.com/leekezar/SemLex | Resolve and verify if needed | Optional lexical/phonology mappings |

Không có checkpoint/code SignSeek hoặc VQ-ASL đã được xác minh trong tài liệu này. Khi chúng cần làm baseline, agent kiểm tra paper/project/tác giả, ghi VERIFIED hoặc UNAVAILABLE; không tự invent repository.

### 5.2. Những điểm code phải lưu ý

- C01: entry point thật là example_usage.py; nó yêu cầu sửa checkpoint/video paths. Không có CLI chính thức đầy đủ như scripts/extract_signrep.py.
- C02: downstream fine-tuning trong README còn TODO. Agent phải tự tạo heads, training loop và evaluation trong project này.
- C02: shell wrappers có Slurm loops và placeholder paths; không chạy thẳng để submit 256 jobs.
- C03: model/placeholder.md không phải implementation của model dịch.
- C04: app.py là app UI có initialization/download/cache side effects. Không import để thực thi batch nghiên cứu.
- C04: code app có kiểm tra HF_TOKEN kiểu legacy; metadata model hub đã thấy public, non-gated ở ngày xác minh. Chạy direct local adapter; recheck access thực tế, không dùng token không có quyền.
- C06: curated minimal-pair CSV còn TODO ở revision đã kiểm tra. Không lấy tên file trong README làm bằng chứng đã có stimuli.
- C07: chỉ tái sử dụng alignment algorithm; không tải COIN/HowTo100M để chạy task sign.

### 5.3. Clone pin, không sửa upstream trực tiếp

Ví dụ cho C01, sau khi bootstrap đã tạo project:

~~~bash
git clone https://github.com/ryanwongsa/SignRep.git third_party/SignRep
git -C third_party/SignRep checkout --detach 06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2
git -C third_party/SignRep rev-parse HEAD
~~~

Tương tự C02/C03/C05/C06. Nếu folder đã tồn tại, xác minh remote và commit trước; không clone đè, không git reset --hard lên thay đổi của người dùng. Lưu patches trong patches/ và diffs trong evidence/code_changes/.

C04 có thể lấy bằng huggingface_hub snapshot_download với repo_type="space", revision đã pin và allow_patterns chỉ source files cần thiết. Không tải example videos mặc định.

## 6. Checkpoint và nguồn tải

### 6.1. SignRep — W01

- Release: https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1
- Direct asset: https://github.com/ryanwongsa/SignRep/releases/download/v0.0.1/ckpt.pt
- Asset filename: ckpt.pt
- Size từ release API: 222676746 bytes.
- SHA-256 từ release metadata:

~~~text
f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b
~~~

- API để kiểm tra lại: https://api.github.com/repos/ryanwongsa/SignRep/releases/tags/v0.0.1
- Local target: checkpoints/signrep/ckpt.pt
- C01 example đọc dictionary key model.
- Khi dùng checkpoint hoàn chỉnh này với constructor init_weights=False và không load thêm init checkpoint, không cần tải lại Kinetics MAE chỉ để feature extraction.
- Release nêu CC BY-NC-SA 4.0 và hạn chế biometric identification, surveillance, synthetic media generation, commercial exploitation. Track trong tài liệu là nghiên cứu hiểu/định vị; không mở nhánh generation/identity từ checkpoint này.

### 6.2. SHuBERT official Google Drive — nguồn đối chiếu

- Folder tác giả được README C02 liên kết:
  https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV?usp=sharing
- Tên assets thấy trong folder listing:
  checkpoint_836_400000.pt, face_dinov2_checkpoint.pt, hands_dinov2_checkpoint.pt.
- Dùng như nguồn fallback/đối chiếu; không invent file IDs.
- Không mặc định Drive và HF là byte-identical. Nếu lấy hai bản, so SHA-256 và state keys.

### 6.3. SHuBERT author model hub — W02 đến W09

**Nguồn public metadata đã xác minh:**  
https://huggingface.co/ShesterG/SHuBERT  
https://huggingface.co/api/models/ShesterG/SHuBERT

**Revision:** 578a0233e770c8ce4dc75d859b91fdea7c34f5aa

Metadata tại ngày xác minh: private=false, gated=false. Điều này xác nhận public listing, chưa thay thế smoke test khi agent tải và chạy.

Prefix tải cụ thể:

~~~text
https://huggingface.co/ShesterG/SHuBERT/resolve/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/
~~~

Ghép prefix với relative path ở bảng:

| ID | Relative path chính xác | Bytes | SHA-256 từ LFS metadata |
|---|---|---:|---|
| W02 | models/checkpoint_836_400000.pt | 1057496908 | 9087f47602401c07c44a9c70164ca1387373938ac4642149126a43e61c20cd7d |
| W03 | models/dinov2face.pth | 112850895 | ebf07e5720a2ccbee559a89f2a8afd7b85ba14bf9ff156842cefb3de5be59bcb |
| W04 | models/dinov2hand.pth | 112850895 | c6ffaf9fbf57b906a5f288b8dbbbeb8651a5cf7a37623fa284cb8e56bf640d85 |
| W05 | models/face_landmarker_v2_with_blendshapes.task | 11577701 | 8e0044e474913045fff3ace464ab14db1646f5f49b42f7c38cacc5f85a5bb8e9 |
| W06 | models/hand_landmarker.task | 7819105 | fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1 |
| W07 | models/yolov8n.pt | 6549796 | f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 |
| W08 | models/checkpoint-11625/pytorch_model.bin | 2683976334 | cb605058abf7fd50b80f055fe8a6e271ed64f04ef7b1b450c7a364f88495526a |
| W09 | models/byt5_base/pytorch_model.bin | 2326696954 | 4651bb6e44016ece08e5c614478da3c6b919191019764d7a43ff78d586e254a6 |

Download rules:

- Feature-only pipeline: W02-W06; W07 nếu cần signer detector.
- Native translation diagnostic: thêm W08 và config/tokenizer files ở dưới.
- Không tải W09 mặc định: trained translation weights W08 đã chứa model parameters; config/tokenizer ở models/byt5_base vẫn cần.
- Không tải optimizer.pt, rng_state_*.pth, scheduler.pt, training_args.bin hoặc example_video mặc định.
- Không gọi W08 là đúng model dùng trong ASL-MTP nếu không xác minh provenance/training conditions.
- Report name: SHuBERT-author-demo-11625. Chỉ đổi thành paper-reproduction khi đối chiếu đủ bằng chứng.

Small files cần cho translation:

~~~text
models/checkpoint-11625/config.json
models/checkpoint-11625/generation_config.json
models/byt5_base/config.json
models/byt5_base/generation_config.json
models/byt5_base/special_tokens_map.json
models/byt5_base/tokenizer_config.json
~~~

Có thể lấy trainer_state.json để đọc provenance; không dùng log tác giả làm kết quả local.

### 6.4. Download snippet có revision và allowlist

Snippet sau chạy trong môi trường có huggingface_hub. Mặc định chỉ lấy feature checkpoints; agent thêm translation files khi Track A diagnostic cần W08.

~~~python
from pathlib import Path
from huggingface_hub import snapshot_download
import os

root = Path(os.environ["SIGN_RESEARCH_ROOT"])
patterns = [
    "models/checkpoint_836_400000.pt",
    "models/dinov2face.pth",
    "models/dinov2hand.pth",
    "models/face_landmarker_v2_with_blendshapes.task",
    "models/hand_landmarker.task",
    "models/yolov8n.pt",
]
snapshot_download(
    repo_id="ShesterG/SHuBERT",
    revision="578a0233e770c8ce4dc75d859b91fdea7c34f5aa",
    allow_patterns=patterns,
    local_dir=root / "checkpoints" / "shubert_author_hf",
)
~~~

Không chạy snippet nếu files hợp lệ đã ở local/cache. Viết checkpoint registry trước khi tải. Sau tải, hash bytes local và so bảng; mismatch thì quarantine file trong project, không load và không tự cập nhật expected hash để bỏ lỗi.

### 6.5. Pretrained bổ sung — chỉ dùng khi cần

| Nguồn | URL | Điều kiện |
|---|---|---|
| Generic ByT5 Base | https://huggingface.co/google/byt5-base | Text-only baseline hoặc train decoder riêng; không phải sign-trained checkpoint |
| DINOv2 | https://github.com/facebookresearch/dinov2 | Architecture; generic visual baseline riêng; không thay DINO sign-tuned một cách im lặng |
| YOLOv8n official | https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8n.pt | Nếu bundled W07 không dùng; phải ghi đổi version/hash |
| MediaPipe face | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker | Chỉ fallback; latest có thể khác W05 |
| MediaPipe hand | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker | Chỉ fallback; latest có thể khác W06 |

DINOv2 architecture ở C02 dùng dinov2_vits14_reg. Loader có xử lý teacher keys và positional embedding; không thay bằng generic DINO weights rồi gọi feature đó là native SHuBERT.

### 6.6. BSL spotting checkpoints — tùy chọn, không cần cho ASL track

Public source C05:

- https://www.robots.ox.ac.uk/~vgg/research/bsldict/data/pretrained_models/i3d.pth.tar
- https://www.robots.ox.ac.uk/~vgg/research/bsldict/data/pretrained_models/mlp.pth.tar
- https://www.robots.ox.ac.uk/~vgg/research/bsldict/data/pretrained_models/i3d_mlp.pth.tar
- Loader/reference: C05/models/download_models.sh và demo.py.

Chỉ tải khi đã chọn BSL setting cùng corpus phù hợp. BSL model không là cùng-language baseline mặc định cho ASL Citizen/How2Sign.

## 7. Nguồn dữ liệu bổ sung

### 7.1. Metadata/annotation có thể cần tải

| Data ID | Nguồn chính xác | Chính sách |
|---|---|---|
| D01 ASL-MTP metadata | C03/data/asl-mtp.csv tại pinned commit | Có thể lấy metadata; giữ external evaluation |
| D02 ASLLRP corpus/annotations | https://dai.cs.rutgers.edu/dai/s/dai ; https://dai.cs.rutgers.edu/dai/s/utterancesearch ; https://www.bu.edu/asllrp/indexright.html | Tải targeted subset/annotations khi cần; theo access terms |
| D03 SignQA annotation-only | https://huggingface.co/datasets/hulala/SignQA-2026 | Optional auxiliary QA; nối với local CSL-Daily/PHOENIX; không tải lại videos |
| D04 BSLDict metadata | https://www.robots.ox.ac.uk/~vgg/research/bsldict/data/bsldict_v1.pkl | Optional BSL track; không có nghĩa dense continuous labels |
| D05 BOBSL | https://www.robots.ox.ac.uk/~vgg/data/bobsl/ | Default disabled; corpus lớn, cần access/annotation audit |
| D06 Sem-Lex | https://github.com/leekezar/SemLex | Optional phonology/lexical transfer; default disabled |
| D07 Phonological Perception stimuli | C06 README | Curated CSV availability chưa đủ ở snapshot; không giả định download được |

Ưu tiên ASL Citizen + How2Sign đang có. Không tải BOBSL/BSLDict videos chỉ để thêm benchmark khi chưa chứng minh cần BSL.

### 7.2. Nguồn tài liệu của dataset đã có — chỉ để đọc format

- CSL-Daily: https://ustc-slr.github.io/datasets/2021_csl_daily/
- How2Sign: https://how2sign.github.io/
- ASL Citizen: https://www.microsoft.com/en-us/research/project/asl-citizen/ và C09.
- MS-ASL: https://microsoft.github.io/data-for-society/dataset?d=MS-ASL-American-Sign-Language-Dataset
- WLASL: https://github.com/dxli94/WLASL
- YouTube-ASL: C11.
- PHOENIX14T: xác nhận local README và version; paper/project SLT: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

Đọc URL không đồng nghĩa được phép redownload. PHOENIX-2014 và PHOENIX-2014T khác versions; không dùng trang 2014 để tự thay local 2014T.

### 7.3. ASL-MTP: schema thật và các bẫy đã quan sát

Pinned file D01 có 1275 rows. Header đọc bằng encoding utf-8-sig:

~~~text
file,video_id,phenomenon,matched translation,mismatched translation
~~~

Giá trị phenomenon thấy ở snapshot:

~~~text
classifiers
conditional
fingerspelling
negation
negation-reversed
numbers
wh-question
yes/no
yes/no-reversed
~~~

Điểm phải xử lý:

1. CSV có BOM; utf-8 thông thường có thể tạo column tên \ufefffile.
2. Column file có URLs tới DAI search/result pages; không mặc định là direct mp4.
3. Một link có thể chứa nhiều videos. C03/data/asl-mtp-readme.md hướng dẫn kết hợp video_id và matched translation để chọn.
4. Ở snapshot kiểm tra, video_id có 48 distinct raw values và file có 938 distinct URLs. **Không suy ra chỉ có 48 utterances.** Đây là lý do phải resolve canonical utterance/video/interval IDs trước splitting và inference.
5. Nhiều rows/phenomena có thể liên quan một utterance/source. Không random split rows độc lập.
6. asllrp-collective.csv và asllrp-collective-nmm-annotated.csv chứa published results: good_nll, bad_nll, nll_diff, bleurt, training_condition, inference_condition. **Chúng không phải gold scope annotations.**
7. Không dùng published hypothesis, NLL, BLEURT hoặc condition-specific results để train/select model mới.
8. Không lấy matched translation của test để chọn layer, calibration coefficient hoặc sampling policy.

### 7.4. Resolve DAI videos

Agent phải tạo resolver có các bước:

- Dùng metadata/pinned README để lấy page đúng; lưu URL và content hash.
- Parse video candidates cùng utterance IDs/labels từ public page hoặc download metadata được corpus cung cấp.
- Chọn frontal view tương ứng sau đối chiếu IDs/translation; ghi resolution evidence.
- Nếu có ambiguity, giữ status UNRESOLVED, không chọn video đầu tiên.
- Download targeted verified clip; kiểm tra content type, ffprobe, duration và frames.
- HTML page lưu đuôi mp4 không phải video hợp lệ.
- Giữ original view và interval. Nhiều camera/views phải cùng split.
- Audit thủ công một subset theo nguồn/phenomenon; nếu không có người kiểm tra, ghi chưa xác nhận linguistic/video alignment.

Không crawl toàn bộ ASLLRP/SignBank vô hạn. Nếu access cần điều kiện chưa có, ghi blocker, tiếp tục task không phụ thuộc.

## 8. Bootstrap trên máy đích

### 8.1. Khởi tạo project riêng

Chạy từ directory writable của người thực thi; không chạy bên trong dataset roots.

~~~bash
export SIGN_RESEARCH_ROOT="$PWD/sign_representation_research"
export SIGN_DATA_ROOT_A="/home/shared_data/sign_language"
export SIGN_DATA_ROOT_B="/home/dongvk/datasets"
mkdir -p "$SIGN_RESEARCH_ROOT"
cd "$SIGN_RESEARCH_ROOT"
~~~

Không gán lại HOME hoặc thay môi trường conda base. Các commands phải chạy dưới account có quyền trên máy đích; tài liệu không tự tạo SSH access.

### 8.2. Bootstrap Python độc lập, có thể chạy ngay

Agent lưu code dưới đây thành bootstrap.py trong project rồi chạy python3 bootstrap.py. Script không tải mạng, không ghi vào datasets và chỉ scan có giới hạn.

~~~python
import json
import os
import platform
import shutil
import subprocess
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["SIGN_RESEARCH_ROOT"]).resolve()
data_roots = [
    Path(os.environ.get("SIGN_DATA_ROOT_A", "/home/shared_data/sign_language")),
    Path(os.environ.get("SIGN_DATA_ROOT_B", "/home/dongvk/datasets")),
]
for source in data_roots:
    if root == source.resolve() or source.resolve() in root.parents:
        raise RuntimeError("Project must be outside dataset roots")

folders = [
    "configs", "scripts", "src", "tests", "third_party", "patches",
    "evidence", "evidence/code_changes", "data/manifests", "data/splits",
    "data/derived", "data/external", "features", "checkpoints", "runs",
    "results/predictions", "results/tables", "reports", "state", "logs",
]
for name in folders:
    (root / name).mkdir(parents=True, exist_ok=True)

def bounded_scan(base, max_entries=5000, max_depth=3):
    queue = deque([(base, 0)])
    count = 0
    suffixes = Counter()
    metadata_examples = []
    errors = []
    while queue and count < max_entries:
        current, depth = queue.popleft()
        try:
            with os.scandir(current) as iterator:
                for entry in iterator:
                    if count >= max_entries:
                        break
                    count += 1
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth < max_depth:
                            queue.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False):
                        suffix = Path(entry.name).suffix.lower()
                        suffixes[suffix] += 1
                        if suffix in {".csv", ".tsv", ".json", ".jsonl", ".pkl",
                                      ".eaf", ".xml", ".txt"}:
                            if len(metadata_examples) < 40:
                                metadata_examples.append(entry.path)
        except OSError as exc:
            errors.append({"path": str(current), "error": str(exc)})
    return {
        "entries_examined": count,
        "bounded_scan": True,
        "depth_limit": max_depth,
        "entry_limit": max_entries,
        "extension_counts_are_partial": dict(suffixes),
        "metadata_examples": metadata_examples,
        "errors": errors,
    }

inventory = []
for base in data_roots:
    item = {"root": str(base), "exists": base.exists()}
    if base.exists():
        try:
            item["children"] = sorted(p.name for p in base.iterdir())
            item["scan"] = bounded_scan(base)
        except OSError as exc:
            item["error"] = str(exc)
    inventory.append(item)

def command_output(args):
    if shutil.which(args[0]) is None:
        return {"available": False}
    completed = subprocess.run(args, capture_output=True, text=True,
                               timeout=20, check=False)
    return {"available": True, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}

disk = shutil.disk_usage(root)
report = {
    "utc_time": datetime.now(timezone.utc).isoformat(),
    "hostname": platform.node(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "disk_free_bytes": disk.free,
    "dataset_roots": inventory,
    "gpu": command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,"
                           "memory.free,utilization.gpu", "--format=csv"]),
    "ffprobe": command_output(["ffprobe", "-version"]),
    "git": command_output(["git", "--version"]),
}
(root / "evidence/bootstrap_inventory.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote evidence/bootstrap_inventory.json")
print("This is a bounded inventory, not complete dataset counts.")
~~~

### 8.3. Cấu hình master agent phải tạo

File configs/project.yaml:

~~~yaml
project_name: sign_representation_research
research_track: auto_gate_A_then_B
source_snapshot_date: "2026-09-30"
data:
  roots:
    shared: /home/shared_data/sign_language
    lab: /home/dongvk/datasets
  source_mode: read_only
  redownload_existing_datasets: false
  canonical_roots: {}
  allow_new_metadata: true
  allowed_new_data:
    - asl_mtp_targeted_videos
    - asllrp_targeted_annotations
  bobsl_download_enabled: false
  semlex_download_enabled: false
compute:
  max_concurrent_gpu_jobs: 1
  pilot_gpu_hours_cap: 8
  adaptation_gpu_hours_cap: 24
  new_download_cap_gib: 10
  feature_cache_cap_gib: 100
  disk_free_min_gib: 20
  seeds: [0, 1, 2]
protocol:
  test_tuning_allowed: false
  aslm_tp_role: external_test
  primary_metric: unset_until_track_selected
  freeze_before_test: true
  full_finetuning_default: false
  linguistic_gold_from_text_allowed: false
~~~

Các budget là operational defaults của kế hoạch, **không phải runtime estimates đã đo**. Agent ghi vào resource_plan.md, đo throughput rồi chọn subset phù hợp. Không vượt cap tự động; hoàn thành kết luận trong phạm vi có thể đo.

### 8.4. State và resumability

Tạo state/tasks.json và state/decisions.jsonl. Mỗi task lưu:

~~~json
{
  "task_id": "T01",
  "status": "PENDING",
  "depends_on": ["T00"],
  "inputs": [],
  "input_hashes": {},
  "command": null,
  "run_id": null,
  "outputs": [],
  "checks": [],
  "reason": null
}
~~~

Sau interruption, kiểm tra artifacts/hash/checks và resume, không chạy lại toàn bộ chỉ vì mất context. Mọi quyết định scientific phải có evidence references.

## 9. Schema và adapters dữ liệu

### 9.1. File manifest chung

Một row JSONL cho mỗi sample/utterance/citation clip, không cho mỗi decoded frame:

~~~json
{
  "sample_id": "EXAMPLE_ONLY",
  "dataset_id": "how2sign",
  "sign_language": "ase",
  "source_root": "/home/shared_data/sign_language/How2Sign",
  "relative_path": "EXAMPLE_ONLY.mp4",
  "input_kind": "video",
  "parent_source_id": null,
  "utterance_id": null,
  "signer_id": null,
  "view_id": null,
  "official_split": "train",
  "research_split": null,
  "start_sec": null,
  "end_sec": null,
  "fps_observed": null,
  "duration_sec": null,
  "num_frames_observed": null,
  "width": null,
  "height": null,
  "lexical_id": null,
  "gloss_raw": null,
  "translation_raw": null,
  "metadata_source": null,
  "annotation_quality": "UNKNOWN",
  "decode_status": "NOT_CHECKED",
  "content_hash": null,
  "duplicate_group": null
}
~~~

Ví dụ là schema, không phải một sample thật. Null phải giữ null; không suy signer_id từ tên file nếu quy tắc chưa được xác nhận.

### 9.2. Dataset adapter tasks

| Dataset | Việc phải kiểm tra và thực hiện |
|---|---|
| ASL Citizen | Tìm CSV/official split; map video path, lexical label, signer; giữ original label IDs; audit missing files |
| WLASL2000 | Tìm WLASL metadata; map video_id/split/gloss/instance fields; kiểm tra cropped/full videos và clip interval |
| MS-ASL | Đọc local train/val/test metadata; xử lý start/end và missing videos; không reuse WLASL parser |
| How2Sign | Xác định frontal/side, full video/sentence clip, original/manual realignment, transcript TSV và signer metadata |
| youtubeASL | Xác định source video IDs/captions/clips/crop provenance; don't assume original YouTube-ASL format |
| CSLDaily | Tìm official annotation/pickle/JSON; phân biệt gloss tokens và Chinese translation; map split và video |
| CSL_Daily_Sentence_Crop | Recover parent IDs/interval/crop metadata; cùng parent phải cùng research split |
| phoenix14T | Hỗ trợ ordered image sequence; lấy FPS từ README/source metadata, không từ frame count suy bừa |

Pickle annotation từ local dataset có thể cần dependency/version tương ứng; đọc trong environment riêng, ghi schema observed. Không mass-convert rồi overwrite nguồn.

### 9.3. Gold grammar/contrast schema

data/manifests/grammar_pairs.jsonl cần:

~~~text
pair_id, sample_id, canonical_utterance_id, phenomenon,
candidate_1, candidate_2, correct_candidate_index,
contrast_type, articulators_required, marker_function,
marker_intervals, grammatical_scope_intervals,
annotation_source, annotation_status, annotator_review_status,
signer_id, lexical_template_id, duplicate_group, research_split
~~~

- annotation_status: GOLD_EXISTING / HUMAN_VERIFIED / WEAK_TEXT_DERIVED / UNKNOWN.
- Chỉ GOLD_EXISTING hoặc HUMAN_VERIFIED làm confirmatory grammar test.
- marker_intervals và grammatical_scope_intervals là hai trường khác nhau.
- Nếu chưa có scope labels: null, không [] có nghĩa không tồn tại.
- Correct index phải được randomize có seed khi tạo candidate order; không đưa tag matched/mismatched vào model input.
- Phenomenon, annotator metadata và correct index không được đưa vào input model.

### 9.4. Spotting schema

Hai files:

~~~text
queries.jsonl:
query_id, sample_id, sign_language, lexical_id,
query_signer_id, task_split, mapping_source, mapping_status

occurrences.jsonl:
target_sample_id, lexical_id, intervals_sec,
occurrence_annotation_source, annotation_status,
coverage_status, absent_verification_status, signer_id, task_split
~~~

intervals_sec=[] chỉ hợp lệ làm negative nếu absence được xác nhận trong phạm vi video. Thiếu annotation không phải absent.

### 9.5. Quality checks bắt buộc

- sample_id unique; relative paths không thoát source root.
- start/end hữu hạn; start < end; interval trong duration.
- Frame sequences được sort đúng số thứ tự, không lexicographic sai 1,10,2.
- CSV delimiter/encoding được phát hiện và ghi.
- IDs từ same video/utterance/views/crops có group chung.
- Decode sampling có seed; báo tỷ lệ lỗi và coverage.
- Không dùng path/filename, dataset ID, signer ID làm model features.
- Official split và research split không bị ghi đè lên nhau.
- Label mapping version/hash cố định; không refit label vocabulary bằng test labels cho train head.
- Normalize language enum bằng metadata được xác nhận: ASL=ase, DGS=gsg, BSL=bfi; CSL identity cần ghi corpus-specific nếu chưa xác minh biến thể.

## 10. Khóa protocol và kiểm soát leakage

Tạo protocol.yaml trước khi chạy test. File phải chứa task, population, exclusions, split hashes, primary metric, seeds, model/checkpoint hash, nguồn supervision, ngân sách tuning và điều kiện dừng. Sau khi khóa, mọi thay đổi tạo protocol version mới; không sửa kết quả cũ.

### 10.1. Quy tắc chia dữ liệu

1. Duy trì official train/val/test để tái lập benchmark. Research split bổ sung phải có tên riêng.
2. Gom mọi crop, camera view, phiên bản encode, query và câu từ cùng source recording vào recording_group. Ưu tiên signer-disjoint khi signer metadata đáng tin cậy; không suy đoán signer bằng khuôn mặt.
3. Query exemplar của Track B chỉ lấy từ train, hoặc từ designated query pool tách biệt. Không lấy exemplar từ chính target recording.
4. Với ASL-MTP, gom các matched/mismatched translations và hiện tượng của cùng utterance. Không shuffle từng dòng độc lập. Dataset này mặc định là external test, không phải train/validation.
5. Kiểm tra overlap theo canonical source ID, video URL, exact hash và sampled-frame perceptual similarity. Similarity chỉ tạo candidate để audit; không tự khẳng định duplicate dựa vào một ngưỡng tùy ý.
6. Ghi pretraining overlap: known, absent-by-documented-list, hoặc unknown. Unknown không được biến thành “unseen data”. Một checkpoint có thể từng nhìn thấy video test dù downstream split đúng.
7. Chỉ fit normalizer, PCA, codebook, label vocabulary, threshold và calibration trên train/validation. Test không tham gia chọn layer, crop, checkpoint hoặc hyperparameter.

Xuất split_audit.json: counts theo recording/signer/language, overlap candidates, exclusions, unresolved IDs và từng lý do. Nếu canonical IDs không giải được, không kết luận external transfer.

### 10.2. Khóa metric và tuning

Mỗi track có đúng một primary endpoint được chọn trước. Secondary metrics vẫn báo đầy đủ nhưng không thay thế primary sau khi thấy kết quả. Chọn tối đa 3 learning rates cho probe và 2 capacities; cùng grid và số trial cho các representation. Native benchmark reproduction có thể dùng upstream protocol riêng, phải được đánh dấu riêng với controlled comparison.

Không dùng các loại dữ liệu mang nghĩa vụ khác nhau như cùng một benchmark: isolated sign classification, continuous gloss recognition, sentence translation và nonmanual grammar đều có nhãn/đơn vị đánh giá khác nhau.

## 11. Môi trường và tích hợp backbone

### 11.1. Ba môi trường độc lập

| Environment | Nguồn | Việc agent phải làm | Điều kiện PASS |
|---|---|---|---|
| env_signrep | C01 requirements và example_usage.py | Cài trong env mới, chọn torch/CUDA tương thích máy, khóa dependency sau smoke test | Import model, strict checkpoint load, forward video thật |
| env_shubert_native | C02 ba environment YAML và vendored fairseq | Tách extraction/DINO/model nếu cần; bỏ prefix/build hardware-specific trong bản sao, lưu diff | Features đủ bốn streams; tải model với mismatch audit |
| env_shubert_translation | C04 requirements và custom inference classes | Dùng code author demo được pin; không import app.py/UI, không gửi video lên Space | Local teacher-forced loss/generation trên một mẫu hợp lệ |

Không cài mọi package vào conda base. Không nâng cấp fairseq/transformers để sửa lỗi mà không lưu diff và test lại. C02 native dùng fairseq vendored; C04 là tích hợp khác với requirements riêng. Ghi pip freeze, conda list, CUDA driver, torch version, GPU model, commit và tất cả compatibility patches. Không khẳng định bitwise reproduction trên hardware khác.

### 11.2. Adapter SignRep

Đọc C01 example_usage.py và model source trước khi triển khai. Dùng model class FINAL_hiera_latent_model_head_v25_active.Model với cấu hình đúng checkpoint. Ví dụ upstream dùng RGB, resize/normalize theo Transformation, tensor [B,3,16,224,224], segment length 16 và stride 2 frame. Output features là representation cho window, không được gọi là feature từng frame độc lập. Kiểm tra thực tế dimension; published wrapper hiện dùng features 768 chiều và latent riêng.

Implement SignRepAdapter.extract(video, sampling_policy): trả embeddings [T,D], window_start_sec, window_end_sec, center_sec, valid_mask và metadata. T là số windows. Clip ngắn có padding phải ghi padding; không bỏ mẫu âm thầm. Giữ frame indices theo FPS thực tế. Khởi tạo đúng head và load checkpoint strict; nếu upstream dynamic import bắt exception, adapter phải báo lỗi head rõ ràng. Không dùng random head rồi mô tả là pretrained feature.

Đánh giá default features và latent chỉ là ablation đã đăng ký; không chọn bằng test. Không thay sampling 16-frame thành 16 giây. Khi tốc độ video khác, temporal span khác: native policy và matched-context policy phải được báo riêng.

### 11.3. Adapter SHuBERT native

Pipeline C02: landmarks/crops → DINO face, left hand, right hand → body features → SHuBERT. Body script chọn upper-body joints và sinh 14 chiều; ba DINO streams hiện dùng 384 chiều mỗi stream. Kiểm tra shape/frame alignment thực tế trước forward. Chỉ chạy những .sh sau khi đã đọc và thay private paths/SLURM partition trong bản sao; không launch 256 jobs mặc định.

**UPSTREAM VERIFIED:** features/shubert_inference.py nhận --index, --csv_path, --checkpoint_path, --output_dir, --batch_size. CSV upstream dùng một field gồm bốn paths ghép bằng tab, thứ tự face, left hand, right hand, body; không phải CSV bốn cột thông thường. --batch_size chia danh sách việc, không phải batch GPU. Viết adapter mới để tránh output basename collision. Code upstream dùng strict=False: agent phải ghi missing/unexpected keys, từ chối thiếu trọng số encoder trọng yếu.

Output native có layer_results; audit axes bằng một mẫu rồi lưu [L,T,D]. Không giả định layout hoặc số layers chỉ từ tên model. Extract với eval và mask=False. Khóa layer trên validation; báo last-layer và layer-average như baselines cố định nếu chưa có validation hợp lệ.

Các preprocessing CLIs phải đọc trực tiếp source pinned revision. kpe_mediapipe.py cần face_model_path và hand_model_path; dinov2_features.py cần dino_path. DINO loader native mong teacher checkpoint và key layout cụ thể, không thể tráo bằng bất kỳ DINOv2 HF weight nào chỉ vì dimension giống nhau. Tải đúng W03/W04/W05 từ registry hoặc BLOCKED_ACCESS; không tự tạo weight substitute.

### 11.4. SHuBERT + ByT5 public translator

C04 custom SignLanguageByT5ForConditionalGeneration và config không phải AutoModel chuẩn. Dùng local inference.py/shubert.py đúng revision, tải đúng subdirectory checkpoint trong registry. Full translator có thể chứa encoder đã fine-tune; không coi encoder đó là W02 frozen pretrained encoder. Đặt tên riêng shubert_native và shubert_demo_finetuned_encoder.

Demo feature extractor có implementation FPS/stride cần audit; comment “15 fps” không đủ chứng minh code chạy ở 15 fps. Wrapper phải lưu actual sampled indices. Native demo layer weights không tự softmax; giữ nguyên nếu tái lập native output.

Teacher-forced scoring dùng tokenizer/config đi kèm và labels với padding=-100. Average negative log likelihood trên số token hợp lệ theo cùng BOS/EOS policy; ByT5 token length không phải word count. Không cần beam generation để xếp hạng cặp. Giữ cả sum NLL, token count, mean NLL để audit length confound. Checkpoint public demo chưa được xác nhận là đúng checkpoint tác giả chạy ASL-MTP: kết quả phải ghi “public-demo checkpoint evaluation”, không “exact ASL-MTP reproduction”.

### 11.5. Fair input comparison

Hai bảng độc lập: native pipeline và controlled input/context. Native SHuBERT có streams/crops và context khác SignRep. Controlled comparison dùng cùng recording IDs, cùng thời đoạn, resolution policy được mô tả, cùng train labels, probe capacity và trial budget. Khi không thể match input information, không quy mọi chênh lệch cho encoder architecture. Ghi branch missing rate, face/hand detection confidence, số valid frames và context span.

## 12. Feature cache và smoke tests

Cache key là hash của video/source ID, dataset revision, crop/sampling config, model commit, checkpoint SHA256, output layer và preprocessing version. Không chỉ dùng filename. Mỗi shard chứa embeddings, timestamps, valid mask; metadata JSON chứa dimension, dtype, cache key và provenance. Atomic write temporary file → validate → rename. Không lưu object pickle nếu không cần; dùng .npz hoặc safetensors với metadata JSON.

Smoke subset: tối đa 20 mẫu train, có short clip, missing landmarks, variable FPS, hai video cùng basename và một mẫu valid dài. Không lấy test để debug preprocessing.

| Kiểm tra | Tiêu chí |
|---|---|
| RGB/order | Đọc cùng một frame bằng hai cách, xác nhận conversion theo upstream |
| Shape và thời gian | D đúng, finite values, timestamps monotonic và nằm trong duration |
| Mask | Không đưa padding/missing frames vào pooling/loss |
| Checkpoint | Hash trùng registry; mismatch được giải thích trước extraction |
| Determinism | Cùng input/config eval cho output tương đương trong tolerance ghi rõ |
| Collision | Hai basename giống nhau có cache/output IDs khác nhau |
| Resume | Interrupted shard không được coi là complete |
| Coverage | Mọi manifest row có SUCCESS hoặc error code, không mất mẫu |

Pilot extraction tối đa 100 clips hoặc ngân sách pilot, lấy điều kiện đến trước. Đo samples/hour, peak GPU memory, CPU time, detection failure và disk bytes/sample. Ước lượng full-cache cost bằng phân bố duration, không lấy một clip ngắn nhân tổng số. Nếu vượt ngân sách, giảm cohort có stratification và báo population mới; không cắt test sau khi thấy metric.

## 13. Baseline chung luôn phải thực hiện

Đây là kiểm tra representation có sử dụng được trên dữ liệu hiện có, không phải novelty claim. B00 label-prior/majority; B01 normalized frozen embedding + nearest centroid; B02 frozen linear probe; B03 equal-capacity shallow temporal head. Chạy trên một corpus isolated ASL có annotation/splits đầy đủ, ưu tiên ASL_Citizen hoặc WLASL2000 sau inventory. MS-ASL là transfer tùy coverage; không tải lại các videos thiếu.

Với classification: báo top-1, macro recall, coverage và class counts; chọn primary theo official protocol được đọc từ paper/dataset. Với retrieval: gallery/query tách recording và định nghĩa relevance bằng nhãn gold; báo Recall@K và mAP nếu multiple positives. Nếu vocabulary giữa datasets chưa có verified mapping, train head riêng; không gọi đó là zero-shot cross-dataset recognition.

Pool mean theo valid windows, L2 normalization và linear head là baseline đủ rõ. Cùng train IDs, augmentations, probe parameters, grid và seeds cho các backbone. B03 dùng cùng hidden width/capacity hoặc báo số parameters và matching rule. Trình bày upstream reported result và local measured result ở hai cột; không copy số paper thành số tái lập.

## 14. Track A: grammar và semantic verification

### 14.1. Gate dữ liệu

A-G0 PASS khi có video–utterance mapping đáng tin, matched/mismatched semantic pair gold, phenomenon label và recording groups. ASL-MTP có thể hỗ trợ evaluation này sau resolving đúng video. A-G1 PASS khi có train/val grammar labels độc lập với external test và đủ signer/recording groups để ước lượng. A-G2 PASS khi có temporal nonmanual scope gold từ annotation source hoặc human annotation độc lập. Không có A-G2 vẫn làm sentence-level A-G0/A-G1, nhưng không claim scope localization.

Lập annotation_request.csv cho phần thiếu: canonical ID, duration, target phenomenon, requested label/scope, source license, ambiguity và reviewer requirement. LLM/caption không được xác nhận grammar gold. Scope cần người có năng lực ASL, guideline và adjudication; dual annotation subset báo agreement phù hợp với interval/label. Không liên lạc hay thuê người tự động.

### 14.2. Experiment matrix

| ID | Thí nghiệm | Mục đích | Gate |
|---|---|---|---|
| A00 | Text-only candidate prior | Đo khả năng đoán từ câu/length/lexical cues | A-G0 |
| A01 | Public translator matched vs mismatched mean NLL | Chẩn đoán decoder-based semantic compatibility | A-G0 + translator |
| A02 | Frozen encoder sentence-level grammar probe | Representation có thông tin linearly/shallow decodable không? | A-G1 |
| A03 | Face/body/hand controlled ablation với cùng head | Thông tin nằm ở branch nào? | A-G1 + valid aligned streams |
| A04 | Temporal scope head | Có xác định được interval grammar không? | A-G2 |
| A05 | Validation-selected calibrated score | Kiểm tra calibration, không method novelty riêng | Independent validation |
| A06 | Conditional adaptation ở mục 16 | Có cải thiện trên external held-out groups không? | Diagnostic + đủ gold |

A00 không phải dùng matched-candidate position. Randomize candidate order deterministically, lưu permutation. Text prior có thể là frozen language-model likelihood hoặc train-only textual classifier; ghi pretraining/label sources và input. Nếu dùng LM khác, registry thêm paper/code/checkpoint chính xác trước tải. Không bắt buộc bổ sung một LM nặng: candidate length, n-gram/template và label prior là controls tối thiểu.

A01 dùng cùng video với hai candidates, prediction là mean NLL nhỏ hơn; tie policy khóa trước. Báo balanced accuracy theo phenomenon và aggregate theo utterance, không chỉ pooled rows. Chạy video-shuffle giữa recordings, blank/padded visual input nếu model hỗ trợ hợp lệ, và wrong-video control. Các controls là distribution-shift diagnostics; loss thay đổi không tự chứng minh causal reliance. So sánh video-conditioned ranking với text-only ranking trên cùng pairs.

A02 chỉ train trên gold train. Frozen linear pooling head trước; temporal shallow head sau. Nếu probe tốt nhưng decoder ranking kém, evidence phù hợp với readout/decoder limitation trong protocol này; không chứng minh encoder hoàn hảo. Nếu probe kém, chưa đủ kết luận encoder mất thông tin: input preprocessing, labels, sample size và head capacity cần audit.

A03 dùng missing-modality masks và ghi detection failures. Zeroing có thể out-of-distribution; nếu train head trên branch subsets, giữ budget/capacity ngang nhau. Không làm “manual signs giống nhau” từ gloss giống nhau; manual-semantic matching cần annotation thật hoặc audit chuyên gia. Không gọi face classifier là grammar representation nếu nó dự đoán signer/background.

A04 output frame/window probabilities, map về timestamp gốc. Primary metric chọn trước: interval IoU/F1 tại thresholds đăng ký, hoặc onset/offset error; báo tolerance theo annotation resolution. Class “không có phenomenon” khác scope chưa annotated. Không dùng sentence duration làm scope gold. Candidate scope phải giữ giới hạn confidence/uncertainty.

### 14.3. Luận điểm tiềm năng, chưa được chứng minh

Hướng paper có cơ sở là benchmark/protocol và representation-readout analysis cho nonmanual semantic distinctions, sau đó một adaptation được kiểm chứng. Điều mới cần chứng minh so với nonmanual/phonology/QA prior art: dataset/protocol nào mới, granularity nào mới, transfer nào chưa có, và vì sao model/module giải đúng failure đã quan sát. “Thêm face branch/attention” đơn lẻ không đủ novelty.

## 15. Track B: query-by-example temporal grounding

### 15.1. Gate và task definition

Input là isolated exemplar q và continuous signing video v; output là các intervals cùng lexical sign theo gold, với confidence. Đây khác sentence translation và dictionary retrieval. B-G0 cần gloss identity mapping được kiểm chứng giữa query corpus và target corpus; B-G1 cần occurrence start/end gold và negatives; B-G2 cần recording/signer-disjoint query và target. CSL-Daily và PHOENIX sentence gloss sequences không tự cho temporal occurrence gold. CTC alignment/pseudo-label chỉ là weak supervision, không evaluation gold.

Nếu ASL dataset continuous không có gold occurrence: tìm đúng phần annotation công khai đã liệt kê ở mục 7 hoặc tạo annotation_request; không mua/download corpus lớn tự động. Không ghép ASL exemplar với DGS/CSL targets vì gloss English/Chinese trùng string. BSLDict/BSL spotting code là prior art và baseline implementation; dữ liệu BSL có điều kiện truy cập phải kiểm tra trước.

### 15.2. Baselines và representation diagnosis

| ID | Baseline | Supervision |
|---|---|---|
| Q00 | Random/length prior với seed khóa | Không train labels |
| Q01 | Cosine sliding-window matching | Frozen features, val threshold |
| Q02 | Subsequence DTW | Frozen features, val distance/length policy |
| Q03 | Drop-DTW theo paper/code đã pin | Ghi exact implementation và cost policy |
| Q04 | Equal-capacity query-conditioned temporal head | Gold train occurrences |
| Q05 | Adaptation ở mục 16 | Cùng gold labels như Q04 |

Không gọi Drop-DTW module mới. DTW không giải mọi variation/coarticulation: phân tích failures theo speed, duration, signer và context chỉ khi những attributes có annotation hoặc tiêu chí objective. Query-to-target cosine phải dùng compatible dimensions; PCA/projection fit trên train. Normalization không fit bằng toàn bộ target test.

### 15.3. Metrics và controls

Primary: interval AP tại một IoU threshold được khóa theo resolution/task; báo mAP tại tập threshold đã đăng ký như secondary. Matching one-to-one giữa prediction và gold; overlapping duplicate detections bị tính false positive. Negatives có no occurrence phải tham gia đánh giá; báo false positives/minute, recall và boundary error. Không chỉ tính accuracy trên đoạn đã crop quanh gold.

Threshold/NMS/length prior chọn trên val. Báo per-query macro và pooled metric; grouped bootstrap theo recording, không bootstrap mỗi overlapping window. Controls: same-sign isolated-to-isolated retrieval để phân biệt lexical feature yếu với context failure; query swap khác sign; context-matched negatives; short/long windows. Nếu gold chỉ partial annotation, đánh giá trong vùng exhaustively annotated và ghi rõ; không coi unannotated events là negatives.

Hướng paper khả thi là characterizing và cải thiện isolated-to-continuous transfer qua temporal context, với occurrence gold đủ tốt. Không tuyên bố coarticulation-specific gain nếu không có phân nhóm hay annotation phân biệt coarticulation với speed/domain shift.

## 16. Thích nghi representation có điều kiện

Chỉ mở task này sau diagnostics. Không bắt đầu bằng pretraining quy mô lớn. Chọn đúng một mechanism có evidence và đăng ký trước trên validation.

### 16.1. Track A candidate

Nếu face/body information có ích nhưng global pooling làm yếu temporal scope, thử frozen multi-stream encoder + temporal adapter nhỏ: branch projections, missing-stream mask, temporal convolution hoặc 2-layer transformer, sentence/scope heads. Không có scope gold thì bỏ scope loss. Loss là classification/BCE với gold labels; optional scope loss chỉ trên annotated frames. Sampling phải cân bằng recording groups, không nhân nhiều window thành nhiều independent labels.

Đối chứng bắt buộc: same-capacity global head; temporal head không branch conditioning; added parameters matched; hand-only/face-only khi streams hợp lệ. Adapter không được dùng candidate text làm shortcut cho scope gold. Chỉ thêm contrastive pair loss nếu pairs gold và semantics rõ, không tạo negatives từ arbitrary translations.

### 16.2. Track B candidate

Nếu Q01/Q02 có lexical retrieval tốt nhưng continuous grounding kém, thử query-conditioned local temporal adapter với background/drop handling. Train từ gold positives và exhaustively annotated negatives; contrastive/ranking loss cộng interval loss nếu có boundaries. Đối chứng Q04 cùng parameter count, DTW/Drop-DTW và adapter không context. Nếu chỉ pseudo-boundaries, train weakly supervised được nhưng evaluation vẫn cần gold; ghi rõ noise source.

### 16.3. Budget công bằng

Frozen-backbone adapter là default. Full fine-tune SKIPPED nếu vượt caps; không mượn external test làm validation. Với mọi runs ghi trainable/total params, labeled utterances, labeled intervals, GPU-hours, peak memory, trials. Same decoder/text supervision nếu so representation; thêm text supervision là yếu tố khác phải ablate. Không kết luận adaptation hiệu quả nếu chỉ hơn head yếu hơn hoặc dùng nhiều labels hơn.

## 17. Đánh giá thống kê và kiểm tra cơ chế

1. Báo point estimates và 95% confidence interval. Dùng paired grouped bootstrap tối thiểu 2.000 resamples cho delta giữa models trên cùng recording groups; seed riêng cố định. Nếu groups quá ít, interval không cứu được inference: báo counts và giới hạn.
2. Ba training seeds [0,1,2] cho learned heads/adapters khi ngân sách cho phép. Deterministic frozen scoring không cần train 3 lần; bootstrap không được gọi là 3 seeds.
3. Chọn primary comparison trước test. Secondary multiple comparisons phải được ghi exploratory hoặc correction policy. Không chọn best seed làm headline.
4. Calibration chỉ fit validation: reliability/ECE cần bins locked và sample counts; semantic ranking accuracy không tự là calibrated uncertainty. Báo Brier/log loss nếu probability definition hợp lệ.
5. Audit ít nhất 30 lỗi theo fixed sampling, hoặc toàn bộ nếu ít hơn: video mapping, decode, landmarks, signer/domain, label ambiguity, manual lexical mismatch, scope missing, text prior, timing. Nếu review cần ASL expertise thì đánh dấu unresolved, không tự đoán từ khuôn mặt.
6. Hiệu ứng subgroup chỉ báo khi có group metadata hợp lệ và đủ counts. Không infer tuổi/giới tính/dân tộc từ video.

Xuất predictions.parquet/csv với canonical IDs, gold status, scores, thresholds, prediction và split; metrics.json chứa metric implementation revision và denominator. Lưu failed samples riêng và coverage denominator. Đưa cả completed và failed runs vào experiment_registry.csv, không chỉ runs đẹp.

## 18. Task graph và hợp đồng CLI

Các scripts dưới đây là **agent phải triển khai**, không phải file đã tồn tại trong upstream. Bootstrap mục 8 tạo scaffold; agent triển khai tối thiểu từng contract, test rồi chạy. Dùng python -m project_module hoặc scripts tương đương, giữ interface/output mapping.

| Task | Dependency | Inputs | Outputs và điều kiện PASS |
|---|---|---|---|
| T00 inventory | Không | roots/config | inventory + hardware + source existence, không sửa roots |
| T01 evidence | T00 | P/C/W registries | literature_matrix + provenance, unverified facts marked |
| T02 manifests | T00 | local labels/videos | manifests + adapters + coverage + stable IDs |
| T03 leakage/gates | T01,T02 | IDs/gold | split_audit + gate_decisions + locked protocol |
| T04 environments | T00,T01 | pinned code/weights | env locks + hashes + load audit |
| T05 smoke/cache | T02,T04 | train pilot | feature shards + smoke reports + cost forecast |
| T06 common probes | T03,T05 | isolated gold | B00–B03 results + predictions |
| T07A grammar diagnostics | T03,T05 | grammar gold; translator optional | A00–A05 hoặc BLOCKED_DATA chi tiết |
| T07B grounding diagnostics | T03,T05 | queries/occurrence gold | Q00–Q04 hoặc BLOCKED_DATA chi tiết |
| T08 adaptation | T07A hoặc T07B PASS | evidence + protocol | matched-capacity ablations, tối đa một candidate/track |
| T09 external/statistics | Relevant tasks | frozen config/predictions | external result, grouped CIs, leakage caveats |
| T10 synthesis | T01–T09 states | all logs/evidence | final_report.md + claim_ledger + reproduction commands |

### 18.1. CLI contract

~~~bash
# CONTRACT: agent triển khai các file này trước khi chạy.
python scripts/validate_manifest.py --manifest data/manifests/isolated.jsonl --report reports/manifest_validation.json
python scripts/audit_splits.py --manifest data/manifests/all.jsonl --config configs/protocol.yaml --output reports/split_audit.json
python scripts/extract_features.py --manifest data/manifests/pilot.jsonl --backbone signrep --config configs/signrep.yaml --output features/signrep --resume
python scripts/extract_features.py --manifest data/manifests/pilot.jsonl --backbone shubert_native --config configs/shubert.yaml --output features/shubert_native --resume
python scripts/run_probe.py --config configs/experiments/B02.yaml --seed 0
python scripts/run_semantic_pairs.py --config configs/experiments/A01.yaml
python scripts/run_grounding.py --config configs/experiments/Q02.yaml
python scripts/bootstrap_metrics.py --predictions runs/EXPERIMENT/predictions.csv --group-key recording_group --resamples 2000 --seed 42
python scripts/build_report.py --registry runs/experiment_registry.csv --gates reports/gate_decisions.json --output reports/final_report.md
~~~

EXPERIMENT là run ID agent chọn từ registry, không chạy literal path nếu chưa có file. Config phải có manifest path/hash, feature key, split IDs, model revision/checkpoint hash, metric, tuning grid, seed và budget. validate_manifest kiểm tra timestamp, duplicate IDs, source existence, annotation status và split consistency; không coi parse được JSON là đủ.

Mỗi script: --help, nonzero exit khi schema/load invalid, JSON log, atomic outputs, deterministic ID, resume chỉ khi config hash trùng. --dry-run hiển thị estimated work/bytes trước download/extraction. Paths được pathlib xử lý; không eval shell strings từ annotations. Unit tests tập trung vào split grouping, metric one-to-one matching, masks, NLL normalization và cache invalidation; không viết test chỉ để khẳng định hằng số implementation.

### 18.2. Run registry schema

~~~yaml
run_id: A02_shubert_seed0
status: PENDING
protocol_version: v1
config_sha256: null
code_commit: null
checkpoint_sha256: null
dataset_manifest_sha256: null
train_recording_count: null
validation_recording_count: null
test_recording_count: null
trainable_parameters: null
gpu_hours: null
coverage: null
metrics_path: null
failure_reason: null
~~~

null có nghĩa chưa đo, không phải zero. Không điền con số suy đoán để report generator chạy được.

## 19. Tài nguyên, scheduling và recovery

Caps mục 8 là mặc định bảo thủ, không phải dự báo runtime. Đọc GPU availability và tài nguyên được cấp trên server; không chiếm toàn bộ GPU hay kill process khác. Một worker extraction trên GPU được cấp trước, CPU decode workers nhỏ rồi đo. Nếu GPU không được cấp, inventory/manifests/review vẫn chạy; model tasks BLOCKED_COMPUTE.

Lịch thực hiện theo gate, không hứa hoàn thành trong số ngày cố định: inventory/evidence → data mapping → load/smoke → cost forecast → common probes → track diagnostics → adaptation có điều kiện → statistics/report. Khi thiếu gold, tiếp tục common probes và viết kết luận data-readiness; không loop tìm một dataset mới vô hạn.

Recovery: OOM giảm inference batch/chunk và ghi thay đổi; không giảm context test riêng để làm metric đẹp. Network failure retry tối đa 3 với backoff; checksum mismatch xóa riêng artifact lỗi, không xóa dataset. Disk low dừng shard mới, giữ logs và báo byte requirement. Decode error lưu ID/error, kiểm tra sampling coverage bias. Checkpoint inaccessible không thay bằng random weights. Metric NaN là FAIL với denominator audit, không thay bằng 0.

Downloads chỉ từ registry, public authorized URLs, sau dry-run size check. Không bypass access controls hay dùng credential tìm thấy trong repo. Owned datasets download_allowed=false luôn được ưu tiên hơn bất kỳ hướng dẫn upstream redownload nào.

## 20. Quy tắc đưa ra kết luận

### 20.1. Claim ladder

| Evidence | Kết luận được phép | Kết luận không được phép |
|---|---|---|
| Paper/README mô tả hạn chế | Câu hỏi cần kiểm chứng, provenance | Hạn chế chắc chắn trên local data |
| Frozen public-demo decoder lỗi | Pipeline này thất bại trong protocol này | Mọi SHuBERT representations mất grammar |
| Probe tốt, decoder yếu với controls | Có evidence cho readout limitation | Causal proof encoder hoàn hảo |
| Adapter gain + CIs + fair ablations | Gain đo được ở population/split này | Universal improvement/cross-language nếu chưa test |
| Occurrence gold thiếu | BLOCKED_DATA, annotation requirement | Temporal grounding đã được kiểm chứng |
| Coverage thấp/overlap unknown | Exploratory result với caveats | Clean external generalization |
| Prior art trùng task/method | Reframe hoặc dừng novelty claim | Đổi tên module rồi tuyên bố mới |

Chỉ đề xuất “paper candidate” nếu đủ: verified novelty matrix, gold/protocol đáng tin, ít nhất một nontrivial diagnostic hoặc method result, controls phù hợp, reproducibility và giới hạn. Venue quality không được đảm bảo từ một tên task. Nếu cải thiện chỉ nhỏ/unstable, ưu tiên báo diagnosis/benchmark rõ hơn thay vì thêm module tùy tiện.

### 20.2. final_report.md template

1. Executive decision: GO_TRACK_A / GO_TRACK_B / DIAGNOSTIC_ONLY / NO_GO_DATA / NO_GO_METHOD; lý do gắn evidence IDs.
2. Exact task/population và difference so với SignRep, SHuBERT, ASL-MTP, spotting/nonmanual prior art.
3. Dataset inventory: available, missing, gold, licensing/access, duplicate roots, coverage và split audits.
4. Models: native/demo distinction, source commit, checksum, preprocessing/context, mismatch audit.
5. Results: all planned experiments, point estimates/CIs, seeds, denominators, budgets; unavailable cells ghi NA + reason.
6. Mechanism evidence: what supports, what contradicts, alternatives chưa loại trừ.
7. Research-gap verdict: verified untested capability trong papers khác với measured local weakness; ghi hai loại riêng.
8. Candidate contribution và novelty đối chiếu từng paper; không claim first nếu search chưa exhaustive.
9. Limitations: annotation, pretraining overlap, test reuse, pipeline differences, sample size, access.
10. Reproduction commands và exact required files; next executable action nếu còn blocker.

claim_ledger.csv có claim_id, claim_text, evidence_type, evidence_path, supporting_results, contradicting_results, confidence_level, permissible_scope. Confidence level là định tính với rationale, không gán phần trăm tùy ý.

## 21. Deliverables và definition of done

Agent giao một project folder riêng gồm:

- configs/: locked protocols, experiments, dataset roots và resource registry.
- data/manifests/ và data/splits/: normalized samples, queries, occurrences, grammar pairs, splits; metadata không copy video trừ khi cần workspace cache hợp lệ.
- provenance/: source commits, download URLs, hashes, dependency locks, license/access notes.
- scripts/ và tests/: adapters, extraction, probes, scoring, statistics, report builder.
- reports/: inventory, literature_matrix, novelty_matrix, split_audit, gate_decisions, smoke/cost reports, annotation_request, final_report, claim_ledger.
- runs/: registry, configs, logs, predictions và metrics cho mọi run.
- features/: cache index và valid shards trong budget; large cache không bắt buộc giao qua chat.
- REPRODUCE.md: environment setup, minimum data prerequisites, commands theo task, expected output files, known blockers.

Done không có nghĩa mọi experiment phải PASS. Done là mọi task có terminal state và evidence; các task BLOCKED có điều kiện unblock cụ thể; các task độc lập đã thực hiện; final report không bịa kết quả. Nếu compute/data đủ, agent phải thực sự chạy chứ không chỉ viết scaffold. Nếu thiếu human gold, không thể tự động hóa hợp lệ phần tạo kết luận semantic gold; phải nói rõ phần này và hoàn tất các việc còn lại.

## 22. Prompt bàn giao cho agent

Copy đoạn dưới vào agent có quyền làm việc trên server chứa datasets, kèm toàn bộ file này:

~~~text
Bạn là research engineer và AI researcher. Thực thi tài liệu này end-to-end trên máy đích, không chỉ đề xuất. Đọc toàn bộ trước khi chạy. Dữ liệu có sẵn tại /home/shared_data/sign_language và /home/dongvk/datasets là read-only, tuyệt đối không redownload hay ghi đè. Tạo workspace riêng ngoài hai roots.

Bắt đầu T00 inventory và bootstrap; xác minh tài nguyên/annotations, pin sources và kiểm tra registry. Triển khai scripts được đánh dấu CONTRACT; không giả định chúng đã có. Ghi task state, provenance và logs sau mỗi milestone. Ưu tiên frozen baselines và diagnostics; không pretrain/fine-tune lớn trước gates.

Track A là ưu tiên khi semantic gold đủ; Track B chỉ mở khi query mapping và occurrence boundaries gold đủ. Common lexical probes vẫn chạy khi khả thi. Caption, gloss sequence, CTC alignment và author NLL tables không thay thế nonmanual/occurrence gold. ASL-MTP là external evaluation mặc định; không tune trên test.

Mọi kết luận phải gắn PAPER_FACT, CODE_OBSERVATION hoặc LOCAL_MEASUREMENT; PROPOSED và UNKNOWN không được viết thành kết quả. Không bịa metrics, checkpoints, paths, novelty hoặc hashes. Nếu decoder public demo khác paper checkpoint, ghi đúng. Nếu task blocked, ghi lý do và file/input cần bổ sung, tiếp tục task độc lập. Không tự gửi email, upload video lên dịch vụ, bypass quyền truy cập hoặc chiếm GPU chưa được cấp.

Khóa protocol trước test. Chạy fair controls, grouped statistics, error audit, lưu cả failures. Cuối cùng xuất final_report.md, claim_ledger.csv, REPRODUCE.md và toàn bộ configs/predictions. Cho kết luận GO/NO_GO/DIAGNOSTIC_ONLY theo evidence; kết quả âm là hợp lệ. Không tuyên bố có paper A* chỉ từ ý tưởng hoặc cải thiện chưa kiểm chứng.
~~~

## 23. Checklist cuối

- [ ] Hai dataset roots được inventory; không yêu cầu tải lại corpus đã có.
- [ ] Không giả định quyền truy cập server/GPU/annotations từ tên directory.
- [ ] Papers, code, weights đều có URL và version/hash khi đã lấy được.
- [ ] Không dùng placeholder model của SL-Models-Analysis như checkpoint thật.
- [ ] Public-demo translator và native SHuBERT encoder được phân biệt.
- [ ] Video mapping và recording groups được audit trước split.
- [ ] ASL-MTP results CSV không bị nhầm với temporal annotation gold.
- [ ] Missing scope khác absent phenomenon; missing annotation khác negative.
- [ ] Controlled input/context và native benchmark được báo riêng.
- [ ] Primary metric, tuning budget, seeds và test lock có trước test.
- [ ] Downstream scripts thực sự implemented/tested trên máy đích trước claim đã chạy.
- [ ] Baselines/capacity/label budgets công bằng; không copy paper scores thành local results.
- [ ] All task states terminal; blockers có file/input unblock cụ thể.
- [ ] Final claims nằm trong phạm vi gold, coverage, overlap và protocol.

**Phạm vi xác minh của tài liệu này:** đây là đặc tả nghiên cứu/thực thi, không phải log thí nghiệm trên server của bạn. Nguồn công khai và cấu trúc code đã được kiểm tra để lập kế hoạch; việc mount datasets, tải weight trên máy đích, GPU forward và các metrics vẫn là nhiệm vụ agent. Không có kết quả thực nghiệm hay cải thiện được giả lập trong file này.
