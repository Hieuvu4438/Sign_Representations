# SignRep — audit khởi đầu

Ngày kiểm tra: **2026-10-05**. Đây là bằng chứng đọc nguồn, không phải reproduction report. Chưa tải checkpoint, chưa chạy model, chưa kiểm tra dataset, chưa huấn luyện.

## Paper anchor

Nguồn người dùng: [arXiv v1](https://arxiv.org/html/2503.08529v1). Nguồn final để đối chiếu: [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.pdf). Bản final chưa được đối chiếu đầy đủ với v1 trong bộ này.

Các fact dưới ghi ngắn bằng tiếng Anh để giữ thuật ngữ chính xác; agent phải đọc các section tương ứng khi dùng.

| Paper fact / reported result | Locator in v1 |
| --- | --- |
| Hiera-B video encoder; pose-derived priors, feature regularization, adversarial style loss; RGB downstream | §3–5 |
| 16-frame pretraining; 64-frame recognition; retrieval stride 2 | §7.2 |
| Confidence filtering already exists | §5.1 |
| WLASL validation retrieval used for checkpoint selection | Appendix A3 |
| WLASL2000: fine-tuned instance Top-1 61.05%; frozen weighted retrieval Rec@1 29.92% | Tables 1, 4 |

**REPORTED_RESULT**, chưa phải số project đo. Agent không dùng hai con số khác setting này thay thế nhau.

## Repo snapshot đã đọc

- Repo: [ryanwongsa/SignRep](https://github.com/ryanwongsa/SignRep).
- Branch: `main`.
- Commit: `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2`, commit date 2025-11-19.
- [Cây file cố định](https://github.com/ryanwongsa/SignRep/tree/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2).
- [Release v0.0.1](https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1): asset `ckpt.pt`, **222,676,746 bytes**; chưa download hoặc verify content/checksum.

## Code map và observation

| Thành phần | File pin ở snapshot | Điều đã kiểm tra |
| --- | --- | --- |
| Model wrapper | [FINAL_hiera_latent_model_head_v25_active.py](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/models/final_models/FINAL_hiera_latent_model_head_v25_active.py) | backbone, fc, projection head, activity head, outputs |
| Decoder/prior projection | [dict_head_v6_FINAL_thoughtful.py](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/models/head_models/dict_head_v6_FINAL_thoughtful.py) | Conv1d/GELU/ConvTranspose1d và per-prior MLP |
| Video preprocessing | [base_video_aug.py](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/augmentation/video/base_video_aug.py) | Resize 224, Normalize, tensor conversion; example validation transform |
| Example inference | [example_usage.py](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/example_usage.py) | local path placeholders, CUDA, 16 frames, stride 2, features/latent |
| Dependencies | [requirements.txt](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/requirements.txt) | six unpinned entries; chưa verify imports/runtime |
| Usage description | [README.md](https://github.com/ryanwongsa/SignRep/blob/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2/README.md) | inference/example và release checkpoints |

Code-level findings, cần runtime verify:

1. Example có `<ckpt_dir>` và `<video_dir>`, không phải đường dẫn thực hoặc dataset runner.
2. Example gọi CUDA trực tiếp; chưa có capability check hoặc CPU path trong example.
3. Example tạo segments đủ dài, bỏ cửa sổ cuối thiếu frames và chỉ tích lũy feature arrays. Không phải một dictionary retrieval evaluation hoàn chỉnh.
4. Model forward trả `features`, `latent`, `cls_head`, `cls_active` và intermediate outputs. Phải chọn feature/head/pooling rõ khi tái lập.
5. Forward hiện vẫn gọi projection/activity head; không giả định benchmark wrapper này đã loại decoder khỏi cost. Cần đo hoặc xây encoder-only inference đúng scope.
6. Constructor có broad exception khi tạo head. Import/config errors có thể bị che bởi thông báo fallback; phải kiểm tra modules/checkpoint compatibility thay vì tin thông báo load.
7. Requirements chưa pin. Lock môi trường sau khi import/runtime được xác thực, không đoán phiên bản.

## Phần chưa thấy trong recursive tree đã kiểm tra

Chưa thấy dedicated pretraining entrypoint, dataset loaders/manifests, sign-prior extraction pipeline, complete losses/discriminator training, full downstream recognition/translation/retrieval runner hoặc full training configs.

Đây là observation **của snapshot này**, không phải kết luận về tất cả tài nguyên tác giả có thể cung cấp. Task T02 phải kiểm tra releases, branches, linked resources và cập nhật nếu tìm được.

## Hệ quả cho thiết kế nghiên cứu

- Không viết lệnh upstream train/eval tưởng tượng. Agent xây runner riêng và ghi adaptation nếu cần.
- Ghi cấp độ: AUDIT_ONLY → INFERENCE_VERIFIED → EVAL_REPRODUCED → PRETRAIN_REPRODUCED. Chưa đạt mức runtime nào ở thời điểm tạo bộ.
- Checkpoint evaluation không đủ để đo causal benefit của objective mới; cần matched continued-training control hoặc train-from-init comparison phù hợp.
- Trace validation history trước khi claim unseen/zero-shot; downstream không train encoder chưa đủ nói toàn pipeline không dùng thông tin benchmark.
- Với hypothesis về reliability, chứng minh delta vượt baseline filtering; không định vị filtering đơn giản là contribution mới.
- Tái lập dictionary protocol trước khi dùng retrieval numbers làm mục tiêu; giữ supervised fine-tuning và frozen retrieval riêng.

## UNKNOWN quan trọng

Teacher availability/training provenance; exact pretraining data/version/dedup manifest; full checkpoint contents; original environment; code–paper final-version differences; dataset accessibility hiện tại; actual runtime/cost; reproduce scores. Task T00/T02 giải quyết từng phần bằng evidence.

