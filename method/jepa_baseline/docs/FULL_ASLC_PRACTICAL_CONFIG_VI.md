# Full ASL-Citizen: huấn luyện SignRep từ encoder Kinetics

**Cập nhật thực thi:** job nền đã được setup/khởi chạy theo
`FULL_ASLC_BACKGROUND_JOB_VI.md`. Custom job runner đã triển khai 4 data workers,
ordered prefetch/resume, inference BF16/batch4, preprocess mỗi frame một lần và
pooled caches. CLI generic vẫn chưa có DataLoader. Phần dưới giữ nội dung review
cấu hình ban đầu; trạng thái vận hành và config thực nằm trong
`runtime/full_aslc_kinetics/{config.json,options.json,control.json}`.
Không chạy CLI pretrain riêng vào cùng output khi service đang chạy.

Cập nhật 2026-10-13 theo yêu cầu người dùng: khởi tạo encoder bằng Hiera-B MAE
Kinetics, không dùng checkpoint SignRep đã pretrain. Đề xuất transfer ngày
2026-10-12 đã được thay thế. Đây là review/config proposal; chưa chạy full training
và chưa chứng minh budget rút gọn đạt score ngang recipe 500.000 bước.
Workflow: academic-research-suite / experiment-agent, kiểm tra inline.

## Dataset và initialization

Dataset có sẵn tại `/home/dongvk/datasets/ASL_Citizen/`; không copy video.

| Split | Videos | Lớp | Người ký | Min/median/max videos mỗi lớp |
|---|---:|---:|---:|---|
| train | 40.154 | 2.731 | 35 | 9 / 15 / 24 |
| val | 10.304 | 2.731 | 6 | 2 / 4 / 7 |
| test | 32.941 | 2.731 | 11 | 7 / 12 / 20 |

Đã đối chiếu CSV, đủ file video và source/signer disjoint giữa splits trong session.
Giữ toàn bộ train, sampler uniform-video, mỗi video hai clips liên tiếp 16 frames.
Không đưa val/test vào SSL loss hoặc bone statistics. Audit valid pose windows
trước training; không âm thầm đổi membership bằng model score.

Checkpoint: `runtime/kinetics/mae_hiera_base_16x224.pth`, 952.882.133 bytes.
`initialize_backbone()` chỉ nạp encoder; fc, decoder, prior/activity heads giữ
khởi tạo mới. Discriminator được khởi tạo riêng trong Pretrainer. Không nạp các
MAE decoder weights từ Kinetics vào SignRep sign decoder.
Đây là cùng loại initialization với SignRep nhưng corpus/budget khác paper YT-SL.
Khi so với phương pháp mới, giữ cùng initialization, membership, budget và protocol.

## Cấu hình đề xuất

File: `configs/pretrain_full_aslc_kinetics.yaml`.
Full-corpus cache và proxy manifests trong YAML chưa được tạo. YAML không tự bật
DataLoader workers, early stopping, inference BF16 hoặc inference batch 16.

| Mục | Giá trị |
|---|---|
| Train set | Toàn bộ 40.154 official train videos |
| Batch | 20 views / 10 videos |
| Input / masking | 16 frames, 224×224 / 0,8 |
| First checkpoint review | 50.000 updates, khoảng 12,45 lượt qua train |
| Maximum budget | 100.000 updates, khoảng 24,90 lượt qua train |
| Peak LR | 1e-4 cho encoder và discriminator |
| Warmup / schedule | 10.000 updates / cosine tới 100.000 |
| Encoder optimizer | AdamW, betas (0,9; 0,95) |
| Weight decay / layer decay / clip | 0,5 / 0,85 / 1,0 |
| Discriminator | Adam, betas (0,5; 0,9), weight decay 1e-3 |
| AMP | BF16; batch 20 cần benchmark trên GPU thực |
| Loss weights / targets / augmentation | Giữ implementation hiện tại |
| Proxy validation | 2 gallery train + 1 val/class: 8.193 videos, đủ 2.731 lớp |
| Proxy / save interval | 10.000 / 2.000 updates |
| Primary pooling | Mean; activity-weighted báo riêng |

Giữ batch 20 thay vì batch 4 để regularization/style có nhiều video khác nhau hơn.
50k/100k là budget thực dụng chưa được xác minh tối ưu. Không áp dụng giả định
20k đủ tốt của phương án pretrained SignRep cũ vào initialization Kinetics mới.

Sau khi chuẩn bị/audit corpus, stage đầu chạy:

```bash
PYTHONPATH=method/jepa_baseline/src .venv-signrep/bin/python -m signrep_baseline pretrain \
  --config method/jepa_baseline/configs/pretrain_full_aslc_kinetics.yaml --stop-after 50000
```

Mốc 50k là review ở giữa scheduler 100k, không phải một run anneal xong ở 50k.
Nếu validation còn cải thiện và nhánh học hoạt động, resume `train/last.pt` tới
100k với config không đổi. CLI không tự early-stop. Không sửa total_steps từ 50k
sang 100k sau training: resume kiểm tra run contract và scheduler.
Nếu cần vượt 100k, lập protocol/schedule extension riêng, không sửa YAML để bypass guard.

## Các phần tăng tốc cần triển khai và benchmark

### Data loading

CLI hiện đọc từng item, augmentation/collate rồi mới GPU update. Log ASLC002:
mean update 0,2803 s nhưng wall/update 0,9086 s. Khoảng 69% thời gian nằm ngoài
update, gồm data, logging và checkpoint. CPU probe 12 video theo seed 1012:
hai views/video mất mean 0,3550 s, median 0,3092 s. Với 10 videos/batch,
CPU tuần tự là bottleneck rõ.

Đề xuất: 4 data workers, 2 CPU threads/worker, persistent workers, prefetch_factor 2,
pinned memory, RAM queue bounded. Phải giữ pairing, batch unique videos và sampler
order; kiểm tra resume/RNG. Thêm num_workers vào YAML hiện chưa có tác dụng vì CLI
chưa có DataLoader. Những thay đổi này chưa được triển khai trong review.

### Evaluation

Giữ stride 2, đầy đủ video và preprocessing tác giả. Preprocess mỗi frame một lần
rồi dùng lại cho overlapping windows. Probe 40 real frames / 13 windows:
0,05964 s → 0,02718 s, nhanh 2,19× riêng preprocess, tensors bitwise equal.
Không chia whole-evaluation ETA cho 2,19 khi chưa đo inference.
Inference batch 16 / BF16 là candidate cần đo so với ASLC002 batch 4 / BF16;
kiểm cosine/ranking/metrics do numerical batch effects.
CLI evaluate_live hiện batch 2 / FP32: YAML AMP chỉ áp dụng training.
Hiera đã dùng SDPA khi có; không cộng FlashAttention speedup giả định.

Proxy cố định bằng seed trước run chỉ dùng theo dõi/chọn candidates. Sau đó đánh giá
best candidate và mốc 50k nếu cần trên full official validation với full train gallery,
rồi chốt checkpoint cho full test. Proxy có thể chọn checkpoint khác full validation.
Giữ last/best và caches cần thiết, không lưu features mọi checkpoint.

### Pose

Dùng parallel pose stage, 4 workers đã đo. Thử 6/8 workers trước khi chọn vì máy có
công việc khác; không giả định speedup tuyến tính. prepare-corpus extraction tuần tự
nếu thiếu raw caches, nên cần parallel extraction stage trước để đạt ETA 4 workers.
Reuse 3.000 raw ASLC002 caches chỉ khi video hash, teacher weights/root mode khớp;
recompute median bones trên full train và dựng normalized caches mới.
Không dùng bone table 3.000 videos làm full-corpus statistics.
Keypoints có sẵn được kiểm tra một file: 86 frames, 133 điểm 2D/frame và scores;
không có body22/hand41 Euler/FK offsets, nên chưa thay được teacher cache SignRep.

## Điều kiện để có kết quả tốt

ASLC002 từng có activity supervision 0,19% và discriminator mất phụ thuộc vào style;
adversarial loss bằng 0 ở 1.000 updates cuối. Tăng steps không tự chứng minh khắc phục.
Trước long run, kiểm full-corpus coverage/activity labels, pose/angle gradients,
regularization balance, matched/unmatched/style gradients và crop protocol.
Không coi EMA gate tắt/adv0 là style-agnostic learning thành công.
Config giữ convention U01–U10 hiện tại; không tự sửa Gram normalization, confidence
masks hoặc loss weights khi chưa có ablation.

Lưu step0 fresh SignRep với encoder Kinetics làm reference; wrapper có fc mới nên
không mô tả step0 là raw Kinetics backbone-only performance. Theo dõi Top1, MRR và
macro metrics trên cùng protocol. Nếu so ASLC002, dùng dictionary 300 lớp như cũ;
không so trực tiếp score 300 lớp với 2.731 lớp. Không dùng test để chỉnh config/steps.

Full 500k ASL-Citizen baseline chưa chạy: không có evidence đảm bảo 50k/100k ngang
score đó. Nếu cần tương đương recipe, giữ 500k và chỉ tối ưu execution pipeline.

## ETA và storage

Số đo ASLC002: pose 3.000 videos / 4 workers mất 6,3288 h; full train ngoại suy
84,7 h trước reuse/normalization. Train 5.000 updates / batch 4 views mất 4.543 s
walltime. Ngoại suy tuyến tính batch 20: 50k khoảng 63,1 h, 100k khoảng 126,2 h;
chưa benchmark batch 20. Proxy 8.193 videos mỗi lần khoảng 4–5 h theo video-count
extrapolation của ASLC002 BF16/batch 4; CLI FP32/batch 2 cần đo riêng.

| Budget | Training-only ngoại suy legacy | Toàn pipeline tới full test |
|---|---|---|
| Mốc 50k | khoảng 2,6 ngày | dự trù 10–14 ngày |
| Tới 100k | khoảng 5,3 ngày | dự trù 15–20 ngày |

Range tính pose, proxy eval, full validation chọn model và full test; phụ thuộc
số candidates full validation, clip length và GPU sharing. Không có benchmark GPU
mới ở review này. Các tối ưu có thể giảm ETA nhưng chưa được đo end-to-end.
50k là mốc review giữa scheduler, không bảo đảm convergence hoặc tương đương 500k.
Máy kiểm tra 2026-10-13 còn khoảng 58 GiB disk trống; GPU 48 GiB có công việc khác.

Disk phát sinh legacy khoảng 23–35 GB; nên chuẩn bị 40 GB trống. Không cộng video đã
có, không copy RGB frames toàn corpus. Pooled-only caches có thể giảm storage cho
retrieval nhưng cần exporter support, không tự bật qua YAML. Translation cần
unpooled features và không nằm trong phạm vi ETA này.

## Nguồn

- Paper/supplement local: artifacts/legacy/evidence/papers/P01.pdf và P02.pdf.
- Code: src/signrep_baseline/{cli,model,data,optim,training,retrieval}.py.
- Logs: experiments/ASLC002/train.log, runtime/ASLC002/train/metrics.jsonl,
  runtime/ASLC002/pose_report.json.
- CPU probes từ session 2026-10-12, chưa có GPU timing mới.
- Official data-loading guidance:
  https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html
