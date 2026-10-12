# Pose extraction và Uni-Sign inference trên ptitiec

Kiểm tra và setup ngày 2026-10-12. Repo fork `Hieuvu4438/Uni-Sign` đã pull từ
`2732773` lên `2879410`. HEAD của repo tác giả `ZechengLi19/Uni-Sign` là
`eed438bcb49e30405cd6ccdfcccca330c134e830`, đã nằm trong lịch sử fork; fork có
thêm 25 commit tại thời điểm kiểm tra.

## Trạng thái trước và sau setup

Trước setup, Python mặc định 3.13 thiếu RTMLib; ONNX Runtime 1.24.4 chỉ có CPU
provider. Không có cache RTMLib ONNX hoặc pose `.pkl` trong checkout. Có trọng số
mT5-base và `wlasl_pose_only_islr.pth`. Log CoSign trong Git ghi GPU L40S, không
chứng minh đã train hay extract trên server RTX 5880 này. Metadata CoSign có
3.862 hàng nhưng chỉ 79 đường dẫn video trong metadata tồn tại và 0 đường dẫn
pose tồn tại; checkout không chứa đầy đủ dataset CoSign.

Sau setup, CLI chạy được trên NVIDIA RTX 5880 Ada 48 GB:

- Python: `/home/haipd/Sign_Representations/.venv-unisign/bin/python`, 3.10.19.
- PyTorch 2.1.1+cu121, torchvision 0.16.1+cu121, CUDA 12.1/cuDNN 8.
- ONNX Runtime GPU 1.18.0, bản CUDA 12 từ feed chính thức Microsoft.
- RTMLib 0.0.13 từ bản vendored trong repo; transformers 4.40.0,
  tokenizers 0.19.1, sentencepiece 0.1.97, numpy 1.26.3.
- Cache ONNX: `/home/haipd/.cache/rtmlib/hub/checkpoints/`.
- Checkpoint encoder: `pretrained_weight/unisign/wlasl_pose_only_islr.pth`.
- mT5: `pretrained_weight/mt5-base/`.

Venv dùng `--system-site-packages` từ conda `h4wpp` để dùng lại PyTorch/CUDA đã
có; các package bổ sung được cài vào venv, không sửa package của môi trường
gốc. Venv phụ thuộc vào việc giữ môi trường `h4wpp`. Wrapper tự thiết lập đường
dẫn CUDA/cuDNN và chọn đúng Python, không cần activate conda.

## Chạy trên một video

Từ thư mục Uni-Sign:

```bash
bash script/run_pose_inference.sh \
  --video /duong/dan/video.mp4 \
  --output-dir out/my_clip
```

Đầu ra:

- `pose.pkl`: keypoints chuẩn hoá theo width/height `(1, 133, 2)` và scores
  `(1, 133)` cho mỗi frame được tracker giữ lại.
- `features.npz`: `pose_embeddings` có shape `(min(T, 64), 768)`,
  `pooled_embedding` `(768,)`, `sampled_pose_indices`.
- `result.json`: nguồn video, số frame, pose mode, ONNX providers, checkpoint,
  kích thước features và thời gian chạy.

Features là output pose encoder sau `pose_proj`, trước mT5 encoder, đã bỏ
text prompt. Chúng dùng checkpoint WLASL được huấn luyện có giám sát. Không
phải pose thô hoặc representation SSL không nhãn.

Tracker chọn một người, bỏ frame không đủ keypoint confidence. Vì vậy
`sampled_pose_indices` đánh chỉ số trong `pose.pkl`, không phải timestamp/frame
index gốc của video. Raw RTMLib scores có thể lớn hơn 1, không phải xác suất
đã được hiệu chuẩn.

Chỉ trích xuất pose, không nạp Uni-Sign/mT5:

```bash
bash script/run_pose_inference.sh \
  --video /duong/dan/video.mp4 \
  --output-dir out/my_pose --pose-only
```

Thêm sinh nhãn ASL từ checkpoint WLASL:

```bash
bash script/run_pose_inference.sh \
  --video /duong/dan/asl_video.mp4 \
  --output-dir out/my_prediction --predict
```

Nhãn được sinh theo open vocabulary, không có calibrated confidence/top-k.
Checkpoint WLASL không nhận diện được 30 nhãn tiếng Việt CoSign một cách có
cơ sở; cần checkpoint CoSign đã fine-tune để làm việc đó. File
`out/cosign_pose_islr_seed42/best_checkpoint.pth` hiện còn thiếu.

Encode lại pose pickle tin cậy, không chạy detector:

```bash
bash script/run_pose_inference.sh \
  --pose out/my_clip/pose.pkl \
  --output-dir out/my_features
```

Mặc định `--mode performance`, `--device cuda`, `--max-length 64`. Có thể chọn
`lightweight` hoặc `balanced`; ONNX của cả ba mode được cache trong setup.
CLI báo lỗi nếu CUDA session fallback sang CPU. `--device cpu` phải được chọn
tường minh nếu muốn chạy CPU. Dùng thư mục đầu ra mới hoặc thêm `--overwrite`.

## Batch pose extraction bằng script có sẵn

CoSign lưu video trong thư mục nhãn nên cần `--recursive`. Với server này:

```bash
source /home/haipd/Sign_Representations/.venv-unisign/bin/activate
export LD_LIBRARY_PATH="/home/haipd/miniconda3/envs/h4wpp/lib/python3.10/site-packages/torch/lib:/home/haipd/miniconda3/envs/h4wpp/lib:/home/haipd/miniconda3/envs/h4wpp/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python demo/pose_extraction.py \
  --src_dir /duong/dan/videos --tgt_dir /duong/dan/poses \
  --recursive --device cuda --backend onnxruntime \
  --mode performance --max_workers 1
```

Script batch gốc giữ mọi người phát hiện trong frame; CLI mới dùng tracker
của serving để giữ một người tương thích preprocessing Uni-Sign. Với dữ liệu
nhiều người, cần kiểm tra schema trước khi dùng output batch cho encoder.

## Tái tạo môi trường

Danh sách package chính: `requirements-inference-cu121.txt` (Python 3.10,
Linux x86_64). Trên server hiện tại:

```bash
/home/haipd/miniconda3/envs/h4wpp/bin/python -m venv \
  --system-site-packages /home/haipd/Sign_Representations/.venv-unisign
/home/haipd/Sign_Representations/.venv-unisign/bin/python -m pip install \
  -r requirements-inference-cu121.txt
```

Trên máy khác, dùng Python 3.10 và venv riêng; file requirements có thể tải
PyTorch/CUDA nếu chưa có. Các trọng số Uni-Sign/mT5 không nằm trong Git;
`script/download_checkpoint.py --mode pose-only` tải chúng. RTMLib tự tải ONNX
ở lần chạy đầu của từng mode nếu cache chưa có. Wrapper đặt chế độ offline
cho Hugging Face để dùng các trọng số local.

Nguồn lựa chọn ORT: [CUDA/cuDNN compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
và [CUDA 12 wheel downloads](https://onnxruntime.ai/docs/install/#cuda-12x).
Không thay ORT 1.18.0 CUDA 12 bằng wheel cùng số phiên bản từ PyPI (CUDA 11).

## Kiểm chứng thực tế

Checkpoint đã nạp với `strict=True`; cả detector YOLOX và estimator RTMW dùng
`CUDAExecutionProvider`. Các file pose/features đều được kiểm tra shape và
finite values.

| Video | Frame | Features | Kết quả | Tổng thời gian |
| --- | ---: | --- | --- | ---: |
| CoSign `BAN_NGÀY_PHẠM_ĐÌNH_HẢI_(4).mp4` | 164 | `64 × 768` | Pipeline pass; nhãn WLASL sinh ra không dùng để đánh giá tiếng Việt | 14,296 s |
| WLASL `66039.mp4` | 65 | `64 × 768` | Sinh `a`, khớp nhãn mẫu `a` | 10,951 s |

Thời gian gồm khởi tạo session và nạp checkpoint, không phải benchmark
steady-state. PyTorch peak memory khoảng 2387,5 MiB, không tính bộ nhớ ONNX.
Đây là smoke test, không phải đánh giá accuracy toàn tập.

Artifact nằm trong `out/pose_inference_smoke_cosign/` và
`out/pose_inference_smoke_wlasl/`. File `runtime.json` trong thư mục smoke CoSign
ghi phiên bản package và SHA-256 checkpoint/ONNX performance.

Đã kiểm tra thêm đường `--pose`: features từ pose cache khớp features của
đường video với `rtol=1e-5, atol=1e-6`. Script batch gốc
`demo/pose_extraction.py` cũng chạy thành công trên 65 frame WLASL với mode
`lightweight`. Cả ba mode đã khởi tạo được CUDA session. `pip check` pass;
CLI từ chối input không tồn tại và output đã có để tránh ghi đè ngoài ý muốn.

18 test core/API pass trước setup. Các test này dùng model giả và không thay
thế các lần chạy GPU thật ở trên. CLI mới không import DeepSpeed/decord hay
các package training. Docker daemon không truy cập được với user `haipd`,
cổng service `56568` chưa chạy; setup này cung cấp inference bằng CLI, chưa
triển khai API daemon. Filesystem còn khoảng 53 GiB; cần chọn nơi lưu phù hợp
trước khi trích xuất toàn bộ dataset.
