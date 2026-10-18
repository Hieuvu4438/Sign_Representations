# E04 — ASLC3000, cấu hình để review trước khi chạy

**Cập nhật:** người dùng đã cho phép launch. Job hiện được quản lý bằng
[systemd service](ASLC3000_SERVICE_VI.md), không dùng launcher nohup phía dưới.
Các câu “chưa khởi động” bên dưới là hồ sơ preview trước lần cho phép này.

Job full-ASLC PID1411678 và toàn bộ4worker/process phụ đã dừng theo yêu cầu người dùng.
Trạng thái gốc lưu `artifacts/full_aslc/job_seed17/status.json` = `stopped_by_user`.
Job đó mới ở cache_train, chưa train SSL main model. Cache/intermediates đã tạo được giữ,
không xóa dataset hoặc artifacts của method khác.

**Training mới chưa được khởi động.** Đây là bản setup để người dùng xem trước.

## Vì sao job trước có nhiều process, và cấu hình mới

Job trước có1Python coordinator +4CPUworkerpose +1multiprocessing resource tracker.
Tôi chọn4worker để overlap IMAGE detectors với GPU DINO khi xử lý83.399videos; đây là
quyết định runtime, không phải yêu cầu của E04. Không có4models hoặc4GPUtraining runs.

Giới hạn mới của người dùng: tối đa2process. Lượt mới dùng **1Pythonprocess** chạy cả
pipeline; pose dùng một worker **thread** trong cùng process, DINO và trainer ở main
thread. Không tạo multiprocessing pool hoặc resource tracker; không torchrun/DDP.
OMP/MKL/OpenBLAS giới hạn1thread trong launcher. MediaPipe/PyTorch có thể có native
threads bên trong process; chúng không phải Python worker processes.

`build_full_cache(workers!=1)` và runner từ chối trực tiếp. Script launch mới truyền
`--workers1`; script launchfull/preflight cũng đã đổi sang1để không vô tình tạo lại pool4.
Một số thao tác asset probe gọi git subprocess ngắn; cùng coordinator vẫn nằm trong
giới hạn2, không sinh pool background.

## Tập dữ liệu khớp với method tham chiếu

Tham chiếu read-only:

- `../jepa_baseline/runtime/ASLC002/corpus/pretrain.jsonl` —3.000videos dùng pretrain.
- `../obs_jepa/runtime/feats/JH3D/aslc300/clips.jsonl` —canonical3.000gallery/1.500query.
- `../obs_jepa/oj/schedule.py` —proposal11budget4.000steps/effectivebatch32.
- `../obs_jepa/oj/compare_retrieval_avg.py` —DCG/Rec@1/Rec@5 gloss-ranking evaluator.

| Split/role | Videos | Glosses | Signers | Videos/gloss |
| --- | ---: | ---: | ---: | ---: |
| Train và retrieval gallery | 3.000 | 300 | 26 | 10 |
| Validation và retrieval query | 1.500 | 300 | 5 | 5 |

Membership train khớp **100%** với ASLC002pretrain manifest; source SHA-256 khớp
manifest tham chiếu và video hiện tại. Canonical query membership cũng giữ nguyên.
Sample/source/hash/signer audit pass, không thay official train/val split. Label IDs
giữ taxonomy của canonical300glosses để tiện so sánh ranks với các method trước.

Đây là pool **đã chọn theo lớp** từ experiment tham chiếu. SSL sampler/loss vẫn chỉ
nhận pretrain view không chứa label/gloss/signer; labels chỉ dùng dictionary evaluation.
Không dùng test và không train classifier. Không random lấy một bộ3.000khác rồi gọi
là same-data comparison.

Manifests mới: `data/aslc3000/manifests/{train,val}.jsonl`.
Evidence: [membership/audit](evidence/aslc3000_selection.json),
[300glosses và counts](evidence/aslc3000_classes.csv).

## Config đề xuất

File: `configs/aslc3000/E04_seed17.yaml`.

| Thành phần | Giá trị |
| --- | --- |
| Method | E04, semantic JEPA + logsig2/anchors/timing auxiliarytarget |
| Seed | 17 |
| Model | D384, encoder6layers/6heads, predictor2layers |
| Foundation | Frozen pinned DINOv2ViT-S/14; IMAGE pose/hand/face priors |
| Input | 4visual +3staticpose streams |
| Budget | 4.000successful updates |
| Batch | micro8 × accumulation4 =32; same effectivebatch as proposal11peers |
| Data sampling | Uniform shuffled source epochs rồi random48sample contiguous segment |
| Approximate source draws | 128.000 ≈42,67passes qua3.000sources nếu không skip |
| Loss | normalized semanticL1 +.1×geometrySmoothL1; q32 train-onlyprojection |
| Optimizer/precision | AdamWlr3e−4; BF16; successful-step cosineLR/EMA |
| Checkpoint/diagnostics | Mỗi1.000updates;3recentstep checkpoints +best/last |
| Evaluation | Fixed-budget final checkpoint tại4.000; frozen300gloss retrieval |
| Process policy | 1Pythonprocess;1CPUextractionthread;1GPUjob |

4.000updates thay vì10.000để match successful-update budget của proposal11. Điều này
không đồng nghĩa same compute: architecture, foundations, multimodal inputs và auxiliary
targets khác. Temporal48samples/25fps của E04 cũng khác native16frame/stride8Hiera peer.

DCG/Rec@1/Rec@5 dùng mean raw clip representation→L2once→cosine, scoregloss=maxgallery
exemplar, pessimistic ties. E04 giữ timestamp-aligned dense tiling48/overlap16 và valid
distinct-frame mean. Không PCA/activityweighting/classifier; paper reproduction không
được claim chỉ vì trùng3metrics. Generic video-ranking recall của pilot8không dùng ở đây.

## Dung lượng full: đo lại từ700cache thật

[Storage measurement](evidence/full_aslc_storage.json) đo completed train prefix700clips,
không dùng24clippreflight nhỏ để quyết định full storage. Đây vẫn là **ngoại suy**, không
phải full-corpus exact total; prefix có thể khác averageclip của dataset.

Mean mỗi video khoảng436.557bytes logical,443.047bytesallocated trên filesystem:

- `features.npz` ≈171,9KB.
- `raw_supports.npz` ≈163,6KB.
- `meta.json` ≈101,1KB.

| Phạm vi | Cache lossless logical (GiB) | Nên dành, gồmallocatedblocks +20–30%margin +2–3GiBartifacts |
| --- | ---: | ---: |
| 3.000train +1.500val | 1,83 | 4–6GiB |
| Full40.154train | 16,33 | 22–25GiB |
| Fulltrain +10.304val | 20,51 | 27–30GiB |
| Fulltrain +val +32.941test | 33,91 | 43–48GiB |

Các planning rows là lựa chọn phạm vi riêng, **không cộng chồng**. GiB=2³⁰bytes.
Checkpoint thực đo256.822.481bytes≈245MiB/file;3retainedsteps+best+last khoảng1,2GiB,
atomic-save peak thêm một checkpoint; budget2–3GiBbao gồm logs/targets/provenance và overhead.

Venv đã có≈7,34GiB; pretrainedweights/detectors đang reuse nên không tính thành cache
tạo thêm. Các số trên **không bao gồm original ASL Citizen videos**. Nếu cài từ máy trống,
cần cộng venv/weights và datasetvideo footprint. Đã stat đủ83.399sourcefiles: videogốc
train22,90GiB +val5,83GiB +test17,47GiB =**46,19GiB**. Tính cảvideo +venv +fullgenerated
cache/artifacts và margin thì khoảng**97–102GiB**, chưa cộng thêm các methodcachekhác.
Các video/venv này đã có trênmáy, không cần tạo bảncopy mới. Giữ raw/normalization intermediates
toàn corpus sẽ tốn thêm; pipeline hiện chỉ giữ boundedqueue và receipts sau featurevalidate.

Nếu giữ feature/support arrays **không nén**, extrapolated payload fulltrain/val/test
lên≈376,6GiB, chủ yếu do conservative primitive supports. Lossless NPZ là lý do full
pipeline có thể xuống≈34GiBcache; không giảm sourceframes, precision hoặc modalities.

Free disk hiện≈43,37GiB, shared với tác vụ khác. Fulltrain+val có thể nằm trong budget;
fullbao gồmcảtest thì headroom mỏng, nên có **ít nhất50–55GiBtrống** nếu giữ6GiBreserve.
ASLC3000planned4–6GiBphù hợp hơn nhiều. Job full đã dừng;700cache/receipts được giữ để
audit và tính lại dung lượng, không tiếp tục ghi dữ liệu.

## Kiểm tra và lệnh launch sau khi review

Config validation pass; **127tests pass** trong unit/integration chọn lọc chạy mộtprocess; không chạy lại
multi-processDDPtests khi userlimit2. Kiểm tra realextraction2videos cũng pass với
1Pythonprocess, no childprocesses sau extraction; không train model trong check này.
Evidence: `evidence/aslc3000_single_process_check.json`.

Lệnh đã chuẩn bị nhưng **chưa chạy**:

```bash
cd /home/haipd/Sign_Representations/method/signpathlab
bash scripts/launch_aslc3000.sh
```

Runner sẽ cache3.000train/1.500val→fittrain-onlytargets→verifygate→profile→train4.000
updates→frozenvalretrieval. `--evaluation-splitval` kết thúc sau ASLC300comparison,
không tạo fulltestcache hoặc evaluate32.941testvideos ngoài phạm vi này.

Sau khi được khởi động, xem một lần trạng thái:

```bash
.venv-locked/bin/python scripts/status_full_aslc.py --job-dir artifacts/aslc3000/job_seed17
tail -n 40 artifacts/aslc3000/job_seed17/output.log
```

Kết quả cuối: `artifacts/aslc3000/job_seed17/result.json`. Không automaticretry/bypass
gate khi near-duplicate candidates hoặc lỗisource/cache. Không dùng checkpoint3step
preflight để thay finalmodel. Chờ người dùng review setup này trước khi chạy lại training.
