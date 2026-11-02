# Skeleton signals cho self-supervised Sign Video Retrieval

Ngày khảo sát: **19/10/2026**. Tác vụ theo xác nhận của người dùng: **Sign Video Retrieval như SignRep**, ưu tiên encoder frozen và sign dictionary retrieval.

- [Báo cáo và thiết kế thực nghiệm](REPORT_VI.md): kết luận, tín hiệu, dataset, protocol, ablation, thống kê và chọn module.
- [Ma trận tài liệu](LITERATURE_MATRIX_VI.md): nguồn chính, phiên bản, supervision, phạm vi đã đọc và giới hạn bằng chứng.
- [Danh sách arm](experiment_matrix.csv): kế hoạch có đối chứng; **chưa phải kết quả chạy**.
- [Audit metadata local](local_dataset_audit.json): đếm split, signer, gloss, liên kết ASL-LEX và fingerprint CSV.
- [Nhật ký nguồn và giới hạn](SEARCH_AND_EVIDENCE_VI.md).
- [Metadata truy xuất HTML](source_retrieval.json): URL phiên bản và SHA256 của văn bản đã phân tích; không chứa toàn văn paper.

Đây là literature review có tìm kiếm và đối chiếu nguồn, kèm thiết kế thực nghiệm; không phải systematic review PRISMA hay một phương pháp mới đã được chứng minh. Không chạy train hoặc score lại official test trong đợt nghiên cứu này. Chỉ đọc metadata split và các kết quả đã lưu trong repo.
