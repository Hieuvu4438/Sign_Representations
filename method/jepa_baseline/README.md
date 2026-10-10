# SignRep baseline reconstruction

Thư mục có tên `jepa_baseline`, nhưng phương pháp được dựng ở đây là **SignRep** của
ICCV2025. Code công khai của tác giả được giữ nguyên và đối chiếu với implementation
bổ sung. Đây là bản tái dựng các module; chưa chạy pretraining500000 bước trên YT-SL-25
và chưa xác nhận lại các điểm số trong paper.

| Phần | Implementation | Bằng chứng đã kiểm tra |
|---|---|---|
| Hiera, decoder, heads | `model.py`, `upstream_signrep/` | 327 tensor checkpoint khớp; forward khớp bitwise code tác giả |
| Pose teacher, chuẩn hoá | `scripts/pose_worker.py`, `bones.py`, `cache.py`, `priors.py` | Teacher thật22/41 Euler; RGB thật; giữ frame thất bại; FK và chiều dài xương |
| Objectives/style | `losses.py`, `discriminator.py` | Numerical oracle; hệ số; mask; hướng gradient; EMA |
| Pretraining | `data.py`, `augmentation.py`, `optim.py`, `training.py`, `preparation.py` | Batch hai view/video; AdamW/Adam; resume bitwise; loại WLASL |
| Retrieval/recognition | `retrieval.py`, `recognition.py`, `recognition_training.py` | Stride2; nearest training video; inflation64 frame; KL; backward thật |
| Translation | `translation*.py`, `integrations/` | Sign2GPT và Tarrés/Fairseq gốc: loader, forward, backward, generation |

Chi tiết paper, các thông tin chưa công bố và quy ước được chọn:
[docs/REPRODUCTION_SPEC.md](docs/REPRODUCTION_SPEC.md). Review trước/sau từng giai đoạn
nằm trong [docs/reviews/](docs/reviews/); kết quả kiểm chứng nằm trong `evidence/`.
Audit giai đoạn tái dựng ban đầu: [evidence/FINAL_AUDIT.json](evidence/FINAL_AUDIT.json), 28 tests pass và
1595 file upstream khớp hash. Có thể chạy lại
`.venv-signrep/bin/python method/jepa_baseline/scripts/final_audit.py` sau các smoke.
Những lựa chọn chưa được xác nhận gồm khoảng/probability augmentation, signed/absolute
xyz distance, Gram normalization, confidence cho góc/activity, temporal sampling,
class aggregation/temperature grid và danh sách/statistics corpus nguyên bản. Đối chiếu
các head trong checkpoint hỗ trợ distance `source − destination` và sin/cos interleaved;
đây là mặc định hiện tại, kèm diagnostic trong `evidence/`.

## Chạy trong workspace này

Từ `/home/haipd/Sign_Representations`:

```bash
export PYTHONPATH="$PWD/method/jepa_baseline/src"
.venv-signrep/bin/python -m signrep_baseline --help
.venv-signrep/bin/python -m signrep_baseline verify-sources
.venv-signrep/bin/python -m pytest method/jepa_baseline/tests -q
```

Cài editable ở một môi trường riêng nếu cần executable `signrep-baseline`:
`python -m pip install -e 'method/jepa_baseline[test,assets]'`.
Code RGB đã kiểm tra với Python3.13/Torch2.11. Pose teacher chạy riêng ở Python3.10,
Torch2.1.1/MediaPipe0.10.14; xem `evidence/*environment.json`. Không cài MediaPipe vào
Python3.13. Các môi trường hiện có chỉ được dùng để kiểm tra, không bị thay đổi.
Translation có môi trường riêng dưới `runtime/translation_env`.

## Weights và pose

Weights/caches/checkpoints mới nằm trong `runtime/` và được gitignore. Source kèm
URL/SHA256 nhưng không đóng gói các file weights lớn. Có thể tải lại:

```bash
.venv-signrep/bin/python -m signrep_baseline download-assets --kind all
.venv-signrep/bin/python -m signrep_baseline pose inspect \
  --python .venv-shubert-native/bin/python \
  --weights method/jepa_baseline/runtime/teacher/models
.venv-signrep/bin/python -m signrep_baseline pose extract \
  --python .venv-shubert-native/bin/python \
  --weights method/jepa_baseline/runtime/teacher/models \
  --video /path/video.mp4 --output /path/raw.npz
```

Teacher giữ **mọi frame đã decode**, confidence/missingness, góc Euler gốc, expanded
FK offsets và SHA video. Confidence thân là MediaPipe visibility; bàn tay dùng cờ
quan sát vì MediaPipe không cung cấp confidence từng joint. Không coi điểm handedness
là confidence joint. Các bước fallback/filter/clamp của demo teacher vẫn có trong
vendor nhưng không được áp vào target thiếu quan sát.
Body root dùng góc native/canonical như demo teacher gốc; hand root giữ cam_rot.
`pose extract ... --body-root-mode camera` bật lựa chọn body root khác, được ghi trong cache;
normalization từ chối khi mode không khớp. Xem các review amendment S2.

Tạo input JSONL cho corpus, mỗi dòng:

```json
{"video_id":"unique_source_id","youtube_id":"actual_youtube_id","split":"pretrain","video":"/path/video.mp4","signer_bbox_xyxy":[0,0,512,512]}
```

Bỏ bbox để dùng quy ước crop tự động đã ghi rõ trong metadata. Không drop/reindex
frame khi crop. Chuẩn hoá xương phải dùng **toàn bộ corpus đã loại WLASL**, không dùng
median riêng từng video. Pipeline chuẩn bị corpus:

```bash
.venv-signrep/bin/python -m signrep_baseline prepare-corpus \
  --manifest /path/yt_sl_sources.jsonl --wlasl /path/WLASL_v0.3.json \
  --output method/jepa_baseline/runtime/corpus \
  --pose-python .venv-shubert-native/bin/python \
  --weights method/jepa_baseline/runtime/teacher/models
```

Lệnh này loại ID WLASL từ tất cả split, trích pose, tính median corpus, dựng lại pose
bằng FK, ghi caches chuẩn hoá và audit/hash. Cache raw không được dùng để supervision.
Default mask góc dùng cờ teacher dự đoán được từng phần; tùy chọn
`angle_confidence: all_observed_points` bật chính sách conservative đã kiểm tra.
Activity được tính lại theo từng clip16 frame, không gán một nhãn cho cả video dài.

## Pretrain, retrieval, recognition

Sửa đường dẫn trong `configs/pretrain.yaml`, giữ `scope: paper_recipe` cho recipe
500000 bước/20 views/80% mask. Config dùng đường dẫn tương đối với file YAML. Manifest
WLASL gallery có `split: train`, queries có `split: val`, `dataset: wlasl`, `label`
nguyên từ0 và `video`. Pretraining CLI bắt buộc audit/statistics corpus, khởi tạo
Kinetics MAE và chọn best bằng retrieval WLASL validation mỗi25000 bước.

```bash
.venv-signrep/bin/python -m signrep_baseline pretrain \
  --config method/jepa_baseline/configs/pretrain.yaml
# --stop-after 2: kiểm tra kỹ thuật, không hoàn thành recipe.
# --resume /path/last.pt: giữ optimizer/scheduler/EMA/scaler/RNG/sampler.
```

`last.pt`/`best.pt` giữ đầy đủ trạng thái train. `last.model.pt`/`best.model.pt` là
bản model riêng để downstream load với `weights_only=True`. Không dùng file training
đầy đủ làm released checkpoint.

```bash
.venv-signrep/bin/python -m signrep_baseline features \
  --manifest /path/train.jsonl --checkpoint /path/best.model.pt \
  --output /path/train_features --device cuda
.venv-signrep/bin/python -m signrep_baseline features \
  --manifest /path/test.jsonl --checkpoint /path/best.model.pt \
  --output /path/test_features --device cuda
.venv-signrep/bin/python -m signrep_baseline retrieval \
  --gallery /path/train_features/features.jsonl \
  --queries /path/test_features/features.jsonl --output /path/retrieval.json
.venv-signrep/bin/python -m signrep_baseline class-distribution \
  --train-manifest /path/train_features/features.jsonl \
  --classes 2000 --output /path/class_distribution.npz
.venv-signrep/bin/python -m signrep_baseline recognition \
  --config method/jepa_baseline/configs/recognition.yaml
.venv-signrep/bin/python -m signrep_baseline recognition-eval \
  --checkpoint /path/recognition/best.pt --manifest /path/test.jsonl \
  --output /path/recognition_test.json --device cuda
```

Retrieval default activity-weighted, dùng raw active head và L2 sau pooling; không
sigmoid/clamp ngầm. `--unweighted` bật mean pooling. Top1 là nearest training video;
Top5/DCG/MRR xếp các nhãn riêng biệt bằng video tốt nhất của mỗi nhãn, đồng thời báo
Top5 theo video hits để không lẫn hai quy ước. DCG/MRR đối chiếu hàm Microsoft ASL-Citizen.
Video dưới16 frame mặc định bị từ chối; `--short-policy repeat_last` là lựa chọn tường
minh. `--stride 8` dành cho ablation; benchmark/default translation dùng2.

Recognition dùng train manifest đã có features để SHA của class table và training
membership nhất quán. Temporal kernel được inflation đúng A6; decoder/activity bị
loại. CE smoothing.1 + .2*phi[c,c]*KL đúng Eq13. Nếu không có nhiệt độ nào trong
[.001,.1] thoả max probability<.5, class-distribution CLI báo lỗi; không tự nới bound.

## Translation và kiểm chứng

Hướng dẫn chạy trainer gốc, schemas, recipes và giới hạn environment:
[docs/TRANSLATION.md](docs/TRANSLATION.md).

Các smoke script thực thi phép kiểm chứng sâu, dùng dữ liệu nhỏ có scope kỹ thuật:

```bash
.venv-shubert-native/bin/python method/jepa_baseline/scripts/verify_pose_pipeline.py
.venv-signrep/bin/python method/jepa_baseline/scripts/smoke_pretrain.py
.venv-signrep/bin/python method/jepa_baseline/scripts/smoke_downstream.py
.venv-signrep/bin/python method/jepa_baseline/scripts/smoke_cli.py
method/jepa_baseline/runtime/translation_env/bin/python method/jepa_baseline/scripts/smoke_sign2gpt.py
method/jepa_baseline/runtime/translation_env/bin/python method/jepa_baseline/scripts/smoke_how2sign.py
```

Smoke fixtures trong workspace sử dụng đoạn ASL-Citizen, bone table nhỏ và có nhãn
kỹ thuật/synthetic; không phải YT-SL/WLASL experiment. Chạy trên dataset chính thức,
bone table corpus đầy đủ, pretrained language weights và đủ recipe là bước riêng để
xác nhận benchmark scores. Các repo public được pin commit và hash; không có
manifest/bone table/config lịch sử của tác giả nên không khẳng định tái lập số học100%.

Source giữ licenses tại từng `vendor/` và namespace copy. SignRep/pose teacher
dùng CC-BY-NC-SA4.0; Sign2GPT dùng Apache2.0; Planckian dùng MIT;
Tarrés/Fairseq giữ license gốc.

## ASL-Citizen pilot

Thí nghiệm thay corpus có protocol riêng tại
[experiments/ASLC001/protocol.md](experiments/ASLC001/protocol.md).
Audit corpus local dùng `prepare-corpus --held-out-manifest ...
--training-reference-manifest ...` để đối chiếu official train và loại val/test;
không tạo YouTube ID giả. Corpora local phải dùng scope khác `paper_recipe`.
Checkpoint AMP ghi rõ `amp_dtype`; BF16 dành cho lần chạy trên GPU Ada này.
Tests và review của adaptation nằm trong `experiments/ASLC001/`, riêng với audit
ban đầu. Outputs/model/feature caches của pilot nằm tại `runtime/ASLC001/`.

ASLC001 đã chạy đủ 1000 bước và đánh giá validation trên 64 lớp,
128 video gallery/64 query. Xem [báo cáo trước/sau train](experiments/ASLC001/REPORT.md),
[kiểm chứng số học](experiments/ASLC001/VERIFICATION.json) và
[audit dữ liệu/code/checkpoint](experiments/ASLC001/audit.json).
Mean pooling: Top5 15.625% → 25%; Top1 giữ nguyên 1.5625%.
Đây là pilot một seed; không phải benchmark đầy đủ của ASL-Citizen.
