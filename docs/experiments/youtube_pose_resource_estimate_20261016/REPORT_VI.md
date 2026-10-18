# RTMW133 — VRAM, concurrency và storage

Đo16/10/2026 trên RTX5880Ada đang chia sẻ với workload khác. YOLOX-m640 + RTMW-x384x288, ONNX CUDA, batch1;72predecodedframes từ3ASL-Citizen clips,15giây mỗi profile. Mẫu có1người/frame.

| Profile | Workers | Aggregatefps | Sum sampledpeakGiB |
|---|---:|---:|---:|
| default | 1 | 9.79 | 10.73 |
| default | 2 | 15.22 | 16.89 |
| limited_workspace | 1 | 9.57 | 1.35 |
| limited_workspace | 2 | 15.70 | 2.70 |
| limited_workspace | 4 | 41.30 | 5.40 |

Limited options: cudnn_conv_algo_search=HEURISTIC; cudnn_conv_use_max_workspace=0; arena_extend_strategy=kSameAsRequested; gpu_mem_limit=4GiB/session. Detector và pose có session riêng; arena cap không phải toàn process cap.
VRAM:nvidia-smi perPID sampled~0,2s, gồm warmup; sumprocesspeaks không nhất thiết là simultaneousGPUpeak. FPS chưa gồm decode/tracking/cachewrite. SharedGPUload có thể giải thích speedup4workers>4x; không chứng minh superlinear scaling. Chưa benchmark YouTube nhiều người/highresolution/long-running hoặc kiểm tra quality parity giữa runtime options.

## Storage chưa nén,1tracked signer/frame

| Format |3000giờ25fps,GiB|3207giờ25fps,GiB|
|---|---:|---:|
|PLctx103:xyFP32,confFP16|259,0|276,9|
|COCO133:xyFP32,scoreFP32|401,3|429,0|
|COCO133:xy/scoreFP16|200,7|214,5|

Chưa gồm RGB, timing/validity/receipts, rawallpersondetections hoặc duplicate arrays. FP16coords đổi precision và cần qualitycheck; không giả định compression ratio trước khi đo data thực.

Free disk khi kết thúc benchmark:73.57GiB.
Measurements ở benchmark.json,limited_workspace/benchmark.json,workerresult/logs,storage_estimate.json. Không sửa extractor/model/caches của các job khác.
