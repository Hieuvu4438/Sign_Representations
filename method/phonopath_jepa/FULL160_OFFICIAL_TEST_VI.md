# A1 full160 — official ASL Citizen test

> Đã archive19/10/2026: runtime/checkpoint/cache và service của run này đã được xóa theo yêu cầu người dùng. Xem [ARCHIVED.md](ARCHIVED.md) để truy cập kết quả đã lưu. Các path/lệnh dưới đây mô tả setup lịch sử.


Job: `runtime/retrieval_e160_test/`. Service systemd user: `phonopath-a1-e160-test-retrieval.service`, đã enable và dùng linger. Lần chạy này theo yêu cầu người dùng sau khi train đủ160epoch, không phụ thuộc chat/SSH.

## Vật liệu cố định

- Checkpoint: `runtime/full160/train/final.pt`, EMA teacher,200.640updates/160epoch. SHA256 `c9474e0fd9bd77deb440694da29c4bb9795b0c99fcfdedb852131805db913723`.
- Copy nguyên `code/pp` từ production snapshot; không sửa model/preprocessing và không dùng cache epoch40 (khác weights).
- Gallery: official ASL Citizen train40.154clip; query: official test32.941clip,2.731gloss; không trùng ID, metadata/pose tồn tại và sourcehash khớp checkpoint.
- Native inference:64mẫu/25fps,window stride2,batch cửa sổ32; raw mean-window rồi cosine/class-max/pessimistic ties, FP16 cache.
- Readout chính vẫn `grp+P` (PCA768); `grp` raw768, `mw`, `ord2` được báo phụ. PCA chỉ fit gallery train; không chọn checkpoint hoặc readout theo test.
- Không tự thêm WLASL, fine-tune hoặc chọn lại checkpoint. Test epoch40 trước đó đã ghi ledger; đây là final160 cố định theo lịch train đã chốt.

## Chạy và phục hồi

Tự chạy: extract train → extract test → PCA/scorer → ghi metrics/ranks và báo cáo. Shard128clip có fsync/atomiccommit, resume theo ID/checkpoint/sourcehash. Chờ disk nếu trống dưới8GiB; chờ GPU nếu không đủ2GiB hoặc có hơn1 GPU process khác của cùng user. CUDA allocator cap4,5% VRAM để cùng tồn tại với job tài khoản khác đang dùng phần lớn GPU; không đổi batch/precision khoa học để vượt lỗi. Finite checkpoints, empty/mixed cache và duplicate test jobs bị chặn.

Service restart khi crash; watchdog extraction20phút không tiến triển;8lỗi không tăng output sẽ chuyển `needs_attention`. Không dừng/chỉnh job người khác. Kết quả từng readout được commit riêng để scoring resume được. Ledger dùng checkpoint SHA+testmanifest SHA+readout; giữ lịch sử epoch40 và đánh dấu scheduled/extracting/scoring/complete cho final160.

```bash
cd /home/haipd/Sign_Representations/method/phonopath_jepa
scripts/a1_e160_official_test.sh status
scripts/a1_e160_official_test.sh logs
scripts/a1_e160_official_test.sh pause
scripts/a1_e160_official_test.sh resume
```

## Artifact

- `status.json`, `train_progress.json`, `test_progress.json`, `score_progress.json`: stage, PID, số clip đã xử lý/commit, speed/ETA.
- `lock.json`: config/source/code/checkpoint hash và readout cố định.
- `shards/train/`, `shards/test/`: feature có thể resume; `train.npz`, `test.npz`: cache đầy đủ.
- `results/{grp,grpP,mw,ord2}.json`: R@1/5/10,MRR,DCG,macroR@1; `*_query_ranks.npz`: từng query.
- `results/*_pca.npz`: components/mean chỉ fit trên gallery.
- `results.json`, `REPORT_VI.md`: bảng kết quả cuối, so sánh từng readout với epoch40. Chỉ `status=complete` mới là hoàn tất cả4readout.
- `runtime/test_ledger.json` (ở method root): lịch sử lần dùng test.

Thời gian cần lấy từ progress thực tế, không dùng lại ETA epoch40 do GPU load khác. File kết quả chưa tồn tại trong giai đoạn extraction là bình thường.
