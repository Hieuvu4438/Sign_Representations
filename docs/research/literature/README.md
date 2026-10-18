# Báo cáo literature hỗ trợ (2026-10-14)

Ba báo cáo này do các research sub-agent biên soạn bằng tiếng Anh, và session chính đã đối chiếu lại. Chúng bổ sung và sửa các báo cáo L1–L4 trong `docs/proposal12/literature/`.

| File | Phạm vi |
|---|---|
| `A_signrep_and_sign_ssl.md` | Đọc kỹ toàn bộ SignRep (ICCV 2025, arXiv 2503.08529): kiến trúc, loss, mọi bảng ablation, giới hạn. Sign SSL cập nhật đến 2026-10 (Yin 2026, SignMatch, VQ-ASL, Sigma, SignVerse-2M…). Bảng benchmark frozen đã sửa. 11 khoảng trống có test bác bỏ. |
| `B_skeleton_ssl_masking.md` | Skeleton SSL (S-JEPA, Path-JEPA, SLiM, Skeleton2vec, MAMP, MacDiff, STARS, USDRL, BYOL-skeleton…), chiến lược masking và augmentation; bảng 32 lựa chọn thiết kế kèm rủi ro khi chuyển sang ký hiệu; 15 điểm sửa cho L2/L3. |
| `C_jepa_diagnostics_rigour.md` | Phần cập nhật của họ JEPA; danh mục khoảng 20 chỉ số chẩn đoán (RankMe, LiDAR, ID, ABX…); độ tin cậy của proxy; measurement stack cho 1 GPU; công thức thống kê; mẫu thiết kế ablation. |

Quy ước tag:

- `[V]`: grep trực tiếp từ văn bản gốc;
- `[V*]`: lấy từ bảng của paper khác;
- `[S]`: chỉ có từ tóm tắt hoặc tìm kiếm;
- `[K]`: kiến thức nền;
- `[I]`: đo nội bộ.

Văn bản gốc (raw HTML/PDF→txt) nằm trong scratchpad của session và **không được chép vào repo**. Muốn kiểm tra lại một con số, tải lại paper rồi grep theo caption bảng.

Tài liệu thiết kế chính: `../Skeleton_SSL_Sign_Research_Design_VI.md`.
