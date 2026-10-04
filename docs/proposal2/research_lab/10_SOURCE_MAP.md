# 10 — Source map và trạng thái đầu vào

Ngày kiểm tra: **2026-10-03**. Đây là seed map cho chương trình nghiên cứu, không phải systematic literature review hoàn tất. Mỗi agent tương lai phải cập nhật sources cho hypothesis của mình. Những rule/process trong bộ tài liệu là đề xuất vận hành của lab, không phải tuyên bố được các paper bên dưới yêu cầu.

## Paper và official code

| ID | Nguồn | Đã kiểm tra | Hệ quả cho rules |
| --- | --- | --- | --- |
| S01 | [SHuBERT paper](https://arxiv.org/pdf/2411.16765), bản PDF v3, 2025-07-02 | Nội dung methods/evaluation và Appendix A–C | Biểu diễn contextual multi-stream; đã có SLT, ISLR, fingerspelling detection và phonological feature recognition. Không gọi phonological prediction là task chưa từng có. Layer mixing/fine-tuning là đối chứng cần xem xét. |
| S02 | [SHuBERT official repo](https://github.com/ShesterG/SHuBERT), head quan sát `cc1929326075bfbad7ad73159b2acf84356059bb` | Cây thư mục và `features/shubert_inference.py` | Reference interface, không chứng minh downstream reproduction. Script đọc bốn streams và all-layer output; kiểm tra load coverage và layout khi wrap. |
| S03 | [SignRep paper](https://arxiv.org/html/2503.08529v1), [abstract/version](https://arxiv.org/abs/2503.08529) | HTML v1, methods, evaluation và limitations; PDF URL trực tiếp lỗi fetch | RGB inference, sign priors lúc pretraining; đã recognition, dictionary retrieval và feature extraction cho translation. Short-window vs contextual representation là câu hỏi cần kiểm chứng, không tự suy thành advantage. |
| S04 | [SignRep official repo](https://github.com/ryanwongsa/SignRep), head quan sát `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2` | Cây thư mục và `example_usage.py` | Reference loader/inference; phải xác minh checkpoint/config/preprocessing trên máy thực. |
| S05 | [ASL Citizen official code](https://github.com/microsoft/ASL-citizen-code), revision quan sát `17f0148b00ef87a5957ec103da8d10bd8eb1fa23` | Cây thư mục, `I3D/aslcitizen_testing.py` và hàm `eval_metrics` | Có DCG/MRR và top-k trên class ranking; binary first-relevant rank dùng discount logarithm. Đây chưa phải full retrieval gallery specification. Chưa chạy parity. |
| S06 | [WLASL official code](https://github.com/dxli94/WLASL/blob/master/code/I3D/test_i3d.py), file blob SHA `ccdd653bd38795d8c8672fb1655dab013ac8747a` | File evaluation, aggregation per-instance/per-class | Phải kiểm tra class support/missing samples và inference aggregation; ghim repository commit thực trước sử dụng, không dùng blob SHA làm commit. Chưa chạy parity. |
| S07 | [SacreBLEU](https://github.com/mjpost/sacrebleu) | Tài liệu official về signature, tokenization, corpus scoring | Ghi version/signature; không mặc định equivalent historical scorer. Chưa chọn package version cho experiment. |
| S08 | [How2Sign project](https://how2sign.github.io/) | Official project page | Điểm vào xác minh release/annotations và modality; chưa xác minh temporal sign labels của bản local. |

Không dùng observed head ở bảng trên để ép checkpoint local thuộc cùng version. Agent phải ghi revision thật của asset mình sử dụng.

## Work gần đây cần novelty audit

| ID | Nguồn | Trạng thái đọc | Cảnh báo |
| --- | --- | --- | --- |
| L01 | [SignSeek — University of Surrey record](https://openresearch.surrey.ac.uk/esploro/outputs/preprint/SignSeek-Learning-Transferable-Representations-for-Sign/991160273502346), [arXiv ID](https://arxiv.org/abs/2609.03695), record ghi 2026-09-03 | Đã đọc abstract/metadata từ institutional source; arXiv full text chưa fetch được | Có công trình về transferable dictionary retrieval/cross-corpus. Phải đọc full protocol/method/code trước khi chốt novelty hướng F; chưa kết luận reproducibility hoặc conference acceptance. |
| L02 | [SMART](https://arxiv.org/abs/2608.25493) | Đã mở abstract page; chưa review full methods/evaluator | Có lead về recognition/spotting temporal alignment; cần kiểm tra gần hướng B thế nào. Không mặc định chỉ cần làm spotting là mới. |

Các lead không phải “related work đã review đầy đủ”. Search results/abstract chỉ giúp ưu tiên đọc; chưa đủ để sao chép method hoặc đưa baseline numbers vào main table.

## Repo mục tiêu và phạm vi đã đọc

Repo: [Hieuvu4438/Sign_Representations](https://github.com/Hieuvu4438/Sign_Representations). Base commit khi soạn rules: `6da1e45894dbbc46faa73f8f1be64d841ed5dcd5`. Đã kiểm tra cây file và [docs/datasets/README.md](https://github.com/Hieuvu4438/Sign_Representations/blob/6da1e45894dbbc46faa73f8f1be64d841ed5dcd5/docs/datasets/README.md), file blob SHA `b96265a0bc27f0810ff569c805b14b988df30dc0`.

`method/` tại base chỉ có `.gitkeep`; chưa có root/nested `AGENTS.md` trong cây file quan sát. `third_party/` có SignRep và SHuBERT cùng các thư viện khác. Các thư mục legacy/code cũ và các Markdown cũ ngoài dataset không được đọc để định hướng nghiên cứu.

## Đường dẫn local làm seed inventory

Đây là thông tin từ dataset document của repo, **không phải kiểm tra filesystem máy nghiên cứu trong phiên này**:

| Dataset | Root được tài liệu liệt kê | Cần xác minh cho hypothesis |
| --- | --- | --- |
| ASL Citizen | `/home/dongvk/datasets/ASL_Citizen/` | Split CSV, signer IDs, ontology và metadata phụ |
| WLASL2000 | `/home/dongvk/datasets/WLASL2000/` | Release annotation, missing videos, class support |
| MS-ASL | `/home/dongvk/datasets/MS-ASL/` | Original/cut video, subsets, synonym mapping |
| How2Sign | `/home/shared_data/sign_language/How2Sign/` | train/eval/test, raw-vs-clipped videos, annotation version |
| How2Sign labels bổ sung | `/home/dongvk/datasets/How2Sign/` | Provenance và khác biệt với shared_data labels |
| CSLDaily | `/home/shared_data/sign_language/CSLDaily/` | Split/labels và overlap với bản crop |
| CSL Daily Sentence Crop | `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/` | Preprocessing/timestamp, duplicate corpus |
| Phoenix14T | `/home/dongvk/datasets/phoenix14T/` | PHOENIX-2014-T release và annotations chính thức |
| youtubeASL | `/home/dongvk/datasets/youtubeASL/` | Tài liệu ghi chỉ có video ID list tại root này; người dùng nói đã tải dataset, nên xác minh current path trước kết luận thiếu |

Trạng thái checkpoints đã nạp/inference chạy là do người dùng cung cấp. Chưa có checkpoint hash/path, GPU allocation hoặc feature cache manifest được kiểm tra ở đây. Agent trên máy thực cần bổ sung, không giả tạo giá trị.
