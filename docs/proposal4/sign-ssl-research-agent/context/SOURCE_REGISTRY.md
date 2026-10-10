# Nguồn khởi đầu — không phải completed literature review

Kiểm tra URL/metadata ban đầu ngày 2026-10-05. Agent phải recheck phiên bản khi chạy. Source row không tương đương FULL_READ. Chỉ anchor và các file code ở audit đã được đọc method/protocol/code; các nguồn literature còn lại mới là điểm bắt đầu.

## Sources

| ID | Nguồn chính thức | Mức đọc của bộ này | Mục đích đọc tiếp |
| --- | --- | --- | --- |
| S01 | [SignRep: Enhancing Self-Supervised Sign Representations — v1](https://arxiv.org/html/2503.08529v1) | Method, experiments, appendix đã đọc; agent phải đọc lại để viết note | Anchor và protocol audit |
| S02 | [SignRep official repository](https://github.com/ryanwongsa/SignRep) | CODE_AUDITED một số file; không runtime | Reproduction feasibility |
| S03 | [SignRep — ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.pdf) | DISCOVERED; chưa so bản final | Version differences |
| S04 | [SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](https://arxiv.org/abs/2411.16765) | ABSTRACT_READ | Cluster prediction, target streams, data/protocol |
| S05 | [SSL-SLR: Self-Supervised Representation Learning for Sign Language Recognition](https://arxiv.org/abs/2509.05188) | ABSTRACT_READ | Negative-free objective, augmentation, transfer |
| S06 | [Self-Supervised Representation Learning with Spatial-Temporal Consistency for Sign Language Recognition](https://arxiv.org/abs/2406.10501) | ABSTRACT_READ | Hand/trunk, joint/motion consistency |
| S07 | [SignCLIP: Connecting Text and Sign Language by Contrastive Learning](https://aclanthology.org/2024.emnlp-main.518/) | ABSTRACT_READ / proceedings metadata | Text-paired comparison; thêm supervision |
| S08 | [SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.pdf) | DISCOVERED / abstract-conclusion excerpt | Masked hand priors |
| S09 | [BEST: BERT Pre-training for Sign Language Recognition with Coupling Tokenization](https://ojs.aaai.org/index.php/AAAI/article/view/25470) | ABSTRACT_READ / proceedings metadata | Triplet/tokenization prior art |
| S10 | [SignBERT+ — tác giả/lab publication page](https://ustc-slr.github.io/publication/) | Metadata xác minh, chưa full paper | Resolve đúng paper, venue và method |
| S11 | [ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition](https://arxiv.org/abs/2304.05934) | ABSTRACT_READ | Official split/retrieval/data usage protocol |
| S12 | [Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding](https://arxiv.org/abs/2509.21223) | ABSTRACT_READ | Semantic pretraining; phân loại supervision |
| S13 | [Claude Code memory / project instructions](https://code.claude.com/docs/en/memory) | Phần CLAUDE.md/rules đã đọc | Cách nạp hướng dẫn, không phải research evidence |

S05 từng xuất hiện dưới tên SL-SLR ở preprint cũ; URL hiện hiển thị SSL-SLR. Agent ghi version/title mapping, không đếm hai tên như hai methods.

## Cần bổ sung bằng tìm kiếm

- Video MAE / Hiera, VICReg, predictive latent video learning/JEPA và các alternatives mới.
- Sign language phonology, non-manual cues và semantic augmentation validity.
- Domain/signer robustness, pose uncertainty/distillation và data contamination.
- Official WLASL2000, NMFs-CSL, YouTube-SL-25 cùng splits/terms và latest releases.
- Các nghiên cứu sau những seed trên đến **ngày agent chạy**.

Không tự điền title/DOI/year/citation từ trí nhớ. Cập nhật source IDs, read levels và locators trong paper notes/search log. Bảng này không chứng minh novelty của bất kỳ hypothesis nào.

