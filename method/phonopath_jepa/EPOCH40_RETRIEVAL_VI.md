# Retrieval checkpoint A1 epoch40

Job: `runtime/retrieval_e040/`; systemd user service `phonopath-a1-e040-retrieval.service`, enable và linger. Training full160 tiếp tục trong service riêng, không đổi config/optimizer của training.

Checkpoint cố định: `runtime/full160/train/epoch_040.pt`, EMA teacher, epoch40/bước50160. Job giữ hard link và SHA256 của checkpoint, copy nguyên `code/pp` từ snapshot production; không dùng model đang thay đổi ở `latest.pt`.

## Protocol

- Gallery: official ASL Citizen **train 40.154 clip**.
- Query: official **val 10.304 clip**, cùng 2.731 gloss, ID rời gallery.
- COCO133 → native preprocessing proposal16; 64 mẫu, đúng25fps, window stride2.
- Raw window-embedding mean, cache FP16, L2 tại cosine scorer.
- Class score = max cosine trên gallery clip của từng gloss; ties bi quan.
- `grp`: 768 chiều, không PCA, so sánh trực tiếp cùng chiều với A0.
- `grp+P`: readout chính, PCA768 fit **gallery train**.
- `mw`, `ord2`: báo cáo phụ, cũng PCA768 fit gallery train. Không chọn readout theo test.
- R@1/5/10, MRR, DCG, macro R@1; lưu ranks từng query.
- **Không đọc hoặc chấm official test.**

## Độ bền job

Extraction lưu shard128 clip bằng fsync + atomic rename; SIGTERM dừng sau clip đang làm và commit buffer. Khi restart, kiểm tra ID/order/SHA của shard rồi tiếp tục từ prefix đã commit. Không bỏ qua clip lỗi để giữ đủ protocol. Tổng số clip hoàn tất phải đúng50.458 mới score full VAL-DICT.

Sau khi có đủ train/val cache, tự chạy scorer và PCA cho các readout. Metrics từng readout được lưu riêng để score có thể tiếp tục nếu bị ngắt. Có lock supervisor, restart khi lỗi, watchdog extraction20phút không tiến triển; dừng/chờ khi disk dưới8GiB. Không tác động process user khác. Admission: tối đa1 processGPU khác của cùng user trước khi mở worker, free VRAM≥4GiB.

## Theo dõi

```bash
cd /home/haipd/Sign_Representations/method/phonopath_jepa
scripts/a1_e040_retrieval.sh status
scripts/a1_e040_retrieval.sh logs
scripts/a1_e040_retrieval.sh pause
scripts/a1_e040_retrieval.sh resume
```

| File trong job | Ý nghĩa |
|---|---|
| `status.json` | Stage và worker PID |
| `train_progress.json`, `val_progress.json` | Clip hoàn tất/đã commit, clips/s, ETA từng split |
| `shards/train`, `shards/val` | Cache có thể resume |
| `train.npz`, `val.npz` | Feature cache đầy đủ, tương thích native evaluator |
| `results/grp.json`, `grpP.json`, `mw.json`, `ord2.json` | Metrics từng readout |
| `results/*_query_ranks.npz` | ID/gloss/signer và rank từng query |
| `results/*_pca.npz` | Center/components fit trên gallery |
| `results.json` | Tổng hợp; chỉ `status=complete` khi cả4 readout xong |
| `REPORT_VI.md` | Bảng kết quả cuối, tự tạo khi hoàn tất |
| `resume_verification.json` | Kiểm tra restart service không mất prefix |

Tốc độ/ETA phải lấy từ progress hiện tại; startup ít clip và tải GPU chung có thể làm ước lượng dao động. Loss train không được dùng thay metric retrieval. Không so trực tiếp R@1 val epoch40 với R@1 test full160 của baseline.

Người dùng đã yêu cầu thêm official ASL Citizen test ở cùng checkpoint40. Job VAL-DICT này giữ nguyên; service test riêng đợi gallery/PCA và kết quả val hoàn tất rồi chạy tự động. Xem [EPOCH40_TEST_RETRIEVAL_VI.md](EPOCH40_TEST_RETRIEVAL_VI.md).
