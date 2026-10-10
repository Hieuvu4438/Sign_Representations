# Data audit — chưa có dataset được kiểm tra

Không có local data paths, manifests hoặc counts được xác thực trong bộ. Dataset đề xuất ở config chưa phải quyền tải/sử dụng.

| Dataset/version | Access/terms | Language/task | Train/val/test manifest+hash | Counts/classes/signers | Duplication/overlap | Teacher/init contamination | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Checklist cho từng dataset thực

- [ ] Official source/version và điều kiện sử dụng.
- [ ] Dữ liệu còn truy cập được; actual available clips khác published counts nếu có.
- [ ] Official splits/class map và valid counts.
- [ ] Signer/source/video/session/channel overlap.
- [ ] Exact/near duplicates; clips từ cùng recording.
- [ ] Pretraining/eval overlap, bao gồm unlabeled data.
- [ ] Teacher/init provenance biết gì, chưa biết gì.
- [ ] Missing videos, pose confidence/masks, sample exclusions.
- [ ] Normalization/statistics fit đúng split.
- [ ] FPS/timestamp, temporal crop, decode failures.
- [ ] Gallery/query policy, labels access, self-match.
- [ ] Semantic augmentation audit.
- [ ] Data retention/sharing phù hợp terms.

Không dùng face recognition để tự suy ra signer metadata. Nếu identity/source overlap không kiểm tra được, giữ UNKNOWN và hạn chế claim.

