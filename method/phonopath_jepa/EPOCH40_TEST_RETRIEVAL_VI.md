# ASL Citizen test — checkpoint A1 epoch40

Người dùng yêu cầu bổ sung test ở epoch40: “đánh giá trên cả test luôn”. Đây là lần sử dụng test sớm theo yêu cầu trực tiếp, đã đăng ký vào `runtime/test_ledger.json`; không coi test này là tập chưa từng dùng khi báo cáo full160 sau này. Không chọn checkpoint/readout hoặc thay đổi training theo kết quả test.

- Checkpoint EMA cố định epoch40, step50.160, SHA256 `f33e1b88a8124a02d11c14a2a9d8ff700fc50bc8ab509c69cb2e8040e9364014`.
- Gallery official ASL Citizen train: **40.154 clip**; query official test: **32.941 clip**, **2.731 gloss**. Đã kiểm tra không trùng ID với train/val và đủ file pose.
- Cùng native snapshot/extractor như VAL-DICT: cửa sổ64,25fps,stride2, FP16 mean-window cache.
- Readout cố định `grp` và `grp+P` (chính), cùng `mw`, `ord2` phụ. Cosine class-max, ties bi quan; R@1/5/10, MRR, DCG và macroR@1.
- Dùng lại gallery đã extract và PCA đã fit trên **gallery train** từ VAL-DICT. Không fit PCA trên test, không fit lại theo query. Projection có kiểm thử bằng kết quả sklearn PCA.transform.

## Job và thứ tự

Service `phonopath-a1-e040-test-retrieval.service`, job `runtime/retrieval_e040_test/`.

Test đợi job `runtime/retrieval_e040/results.json` có `status=complete`, sau đó tự extract toàn bộ test rồi score. Nhờ chạy tuần tự, chỉ có train full160 và một worker evaluation dùng GPU cùng lúc trong tài khoản haipd; không đổi hoặc dừng training. Trong thời gian chờ, test supervisor sống nhưng không chiếm GPU.

Job có source/code/checkpoint hash, cache shard128clip có thể resume, fsync/atomic commit, systemd restart và watchdog, chờ khi disk dưới8GiB. Ledger phân biệt scheduled/extracting/scoring/complete theo từng readout và checkpoint, cho phép resume cùng job nhưng không tạo lần đánh giá trùng lặp ở job khác.

```bash
cd /home/haipd/Sign_Representations/method/phonopath_jepa
scripts/a1_e040_test_retrieval.sh status
scripts/a1_e040_test_retrieval.sh pause
scripts/a1_e040_test_retrieval.sh resume
```

Khi hoàn tất: `runtime/retrieval_e040_test/results.json`, `REPORT_VI.md`, ranks từng query trong `results/`. Test này chỉ là ASL Citizen; không tự thêm WLASL hoặc dataset khác.
