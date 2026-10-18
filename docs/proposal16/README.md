# proposal16: PhonoPath-JEPA

Mục tiêu: đem ý tưởng **Path-JEPA** (ECCV 2026: token path-signature + JEPA target latent + masking đóng theo phụ thuộc) vào học biểu diễn ngôn ngữ ký hiệu tự giám sát trên **skeleton**. Downstream là sign video retrieval và isolated sign recognition. Proposal xây trên anchor PLctx (proposal12) và dùng các bài học từ mọi method đã thử trong repo.

Trạng thái 2026-10-16: **chỉ đề xuất, v1.1** (đã qua một vòng tự rà soát và một vòng review độc lập; xem §12b của proposal). Chưa triển khai và chưa train. Chỉ có chẩn đoán CPU không cần train.

| File | Đọc khi |
|---|---|
| **`PhonoPath_JEPA_Proposal_VI.md`** | Bắt đầu ở đây: tóm tắt, method, giả thuyết, arm, phase, ngân sách, rủi ro |
| `literature/P0_path_jepa_close_reading.md` | Muốn biết Path-JEPA thật sự làm gì, ablation nói gì, và chỗ nào phải sửa cho sign |
| `literature/P1_path_signatures.md` | Toán signature (Chen, lead–lag, nhiễu, chuẩn hoá), signature trong gesture/sign, phần mềm |
| `literature/P2_jepa_skeleton_update.md` | JEPA và skeleton SSL mới (2024–10/2026), 12 nguyên tắc thiết kế N1–N12 |
| `literature/P3_sign_domain_signals.md` | Âm vị học ASL → tín hiệu skeleton → mức signature; ASL-LEX 2.0; dataset; tín hiệu nên học và nhiễu nên bất biến |
| `literature/D1_server_dataset_audit.md` | Dataset nào có trên server, định dạng keypoint, signer, nhiễm chéo, dùng được cho việc gì |
| `diagnostics/README.md` | Chẩn đoán D-TEMPO, D-SIG, D-LEX: script, cách tái lập, giới hạn |
| `data/` | ASL-LEX 2.0 `signdata.csv` và `signdataKEY.csv` (OSF `zpha4`, CC BY-NC). Không commit |

Literature trước đó, không lặp lại ở đây: `docs/proposal12/literature/L1–L4`, `docs/research/literature/A–C`.
