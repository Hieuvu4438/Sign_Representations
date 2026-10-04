# Inventory — PREP001

Ngày 2026-10-03 UTC, host `ptitiec`, repo base
`6da1e45894dbbc46faa73f8f1be64d841ed5dcd5`.
Evidence trực tiếp: [inventory JSON](PREP001_inventory.json),
[status/exit 0](../experiments/PREP001_status.json). Script một CPU process,
1,393 s wall time, 0 GPU-hours. Không decode/hash video corpus.

## Máy và allocation

- GPU 0: RTX 5880 Ada, 49.140 MiB; snapshot dùng 46.898 MiB.
- Người dùng đã cấp GPU 0 khi trống qua câu trả lời trong phiên 2026-10-03.
  Snapshot `gpu_allocation_status=unknown` trong PREP001 là trạng thái trước
  câu trả lời; allocation hiện hành trong STATE có quyền ưu tiên.
- Không có scheduler/CUDA_VISIBLE_DEVICES trong environment quan sát.
- RAM quan sát: 62 GiB tổng, khoảng 43 GiB available; quota riêng unknown.
- Disk free 76.157.091.840 bytes / 3.934.171.283.456 bytes (~1,94%).
  Gate reserve 20% không đạt: hold cache/extraction lớn và tải corpus.
  Small records/source fixtures tiếp tục với footprint <10 MiB; đây không
  phải allocation override cho dữ liệu lớn. Chưa có storage quota riêng.

## Dữ liệu

Cả chín root trong `docs/datasets/README.md` tồn tại; entry names trong JSON.
Existence không chứng minh release, annotation correctness hoặc coverage.
Chỉ ASL Citizen đã audit coverage metadata toàn split:

| Split | CSV rows / file tồn tại | Glosses | Participant IDs | Min–max class support |
| --- | --- | --- | --- | --- |
| train | 40.154 / 40.154 | 2.731 | 35 | 9–24 |
| val | 10.304 / 10.304 | 2.731 | 6 | 2–7 |
| test | 32.941 / 32.941 | 2.731 | 11 | 7–20 |

Không có duplicate filename trong từng split; mọi cặp split có 0 filename
và 0 participant ID chung. Val/test không có gloss ngoài train; không có
participant/gloss rỗng. Tổng counts khớp Table 2 của
[paper release](https://papers.neurips.cc/paper_files/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf).
Chưa đối chiếu exact CSV release hoặc kiểm chứng danh tính participant.
Metadata test chỉ dùng cho exclusivity/coverage, không dùng chọn model/cohort.

MS-ASL train/val JSON đã đọc schema: 16.054/5.287 rows, có signer, URL,
start/end, FPS và box. Chưa audit availability hoặc dùng làm benchmark.
youtubeASL root chỉ thấy ID list; vị trí video hiện có vẫn unknown, không tải lại.
Nhãn phonology/non-manual và sign-level boundary chưa xác minh ở bất kỳ corpus.

## Checkpoint/interface

| Encoder | Asset | Bytes | SHA-256 |
| --- | --- | --- | --- |
| SignRep | `checkpoints/signrep/ckpt.pt` | 222.676.746 | `f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b` |
| SHuBERT | `checkpoints/shubert_author_hf/models/checkpoint_836_400000.pt` | 1.057.496.908 | `9087f47602401c07c44a9c70164ca1387373938ac4642149126a43e61c20cd7d` |

Read-only upstream revisions: SignRep `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2`;
SHuBERT `cc1929326075bfbad7ad73159b2acf84356059bb`.
Checkpoint origin/release mapping và pretrain sample exclusions: unknown.
Không lấy tên checkpoint làm chứng cứ mapping.

SignRep example: RGB, resize 224, ImageNet normalize, windows 16 frames,
stride 2, bỏ window cuối không đủ 16; có `features`, `latent`, activity head.
SHuBERT example: input face/left/right/body arrays, all-layer output; loader
dùng `strict=False` mà không audit coverage; skip cache chỉ theo filename.
Hai interface đã đọc, chưa chạy upstream inference hoặc feature parity.
Nguồn code là các file được ghim trên máy, hash bổ sung theo run.
Config ByT5 `checkpoint-11625` có architecture translation riêng; không
đánh đồng translation checkpoint với frozen pretrained encoder.

Unknown còn lại: decode quality, PTS/FPS, near duplicates, sessions, official
CSV identity/license, pretraining overlap, SHuBERT raw stream preprocessing
và cache provenance. Không khảo sát code/caches từ phương pháp cũ.

## H003 prerequisite assets and raw detector smoke — D016

[ASSET002](ASSET002_asset_equality.json) main/face/hand SHA/size equal published author LFS metadata; [DINOSRC001](DINOSRC001_source_manifest.json) pinned official architecture31files/156624B, historical author DINO commit unknown. Strict CPU model/teacher-backbone coverage and synthetic GPU interface completed; no inference-install/shared environment mutation.

[LANDMARK001](LANDMARK001_landmark_audit.json) all four fixed train/dev native clips396frames have pre-fill API counts/coordinates/handedness and aligned timestamps. Raw5,777,579B in /dev/shm/shubert_h003_landmarks_85a92d70adc2bcd37fcf; hashes/paths in report, ephemeral storage must remain available or be rebuilt with a fresh registered run/store. No test access, anatomical crop assignment or natural semantic failure ground truth. Corrected full-wall graphics resource cost is in GLAUDIT001/D016; raw GPU_hours0 does not mean no GPU resources.

## H003 native input and GPU compatibility — D017–D019

CROP00216predeclared native boundary fixtures/full396geometry rows identify
initialNone frame/body-shape problems, degenerate shoulder nonfinite values,
empty-hand and offscreen face exceptions and14same-detection left/right selections
on BORROW. These are API/geometry observations, not human true tracking failures.
All4realclips have valid finite native inputs. DecoderRGB/OpenCVBGR inversion
exact, mp4v compression lossy; the original author writer color conversion is
correct for the full path tested.

Own stateful adapter retains frame alignment/14Dbody/observed-before-fill masks,
rejects invalid observed inputs transactionally and preserves duplicate selection.
INPUTTEST0016fixtures pass; INPUTPARITY001 all396encoded pixels/body exact native.
[INPUTPARITY001_input_parity.json](INPUTPARITY001_input_parity.json) references
16,527,451B RAM input store. Adapted initial-None branches are absent from these
realclips, so exact compatibility remains limited to covered branches.

SOFTGL001 validates existing Mesa/llvmpipe per-child CPU graphics for128prefix
frames/12contexts with raw coordinate equality and no worker NVIDIA maps/fd.
No install/globalenvironment change/historical resource refund; conservatively
charged14.095760175GPUseconds.

Original STREAMFEAT001 real CPU/GPU feature gate fails seven cells atlayers10/11
in first T156context. NUMDIAG001 preserves arrays, NUMCPU001 precision promotion
is diagnostic only. Separately registered AUTHORGPU001 all4/all396/12layers are
bitwise equal to native author CUDA function after independent complete loads.
34,123,612B native/reference/own features are in bounded RAM with hashes; no
new task scores/natural-failure ground truth or paper reproduction claim.
Historical DINO version/pretraining overlap and true anatomy remain unknown.

## D020 — Auxiliary annotation source and exported confidence

[TRACKREL001](TRACKREL001_tracking_release_audit.json) records official manual DGS
XML provenance/schema; exact train/dev ID joins0/0.
[TRACKJOIN001](TRACKJOIN001_tracking_join_audit.json) also finds0/0 after literal
name normalization. Only524288B of the1,904,876,596B official image archive fetched
under enforced HTTP206/ETag/range conditions. Three complete PNG prefix members
have no candidate local path. No pixel comparison/frame identity or label join
accepted. RAM paths/full hashes in reports; no bulk download.

[CONFSEM001](CONFSEM001_confidence_semantics.json) reads the installed result
constructor/native exporter and all396 saved rows. Hand score is a handedness
classification quantity; localization confidence/pose visibility not exported.
This closes score semantics, not correctness or calibration. Ground truth remains
missing on the existing ASL cohort. The [intake specification](../protocols/H003_annotation_intake_v1.md)
defines acceptable source/frame/coordinate identity and independent review;
no manual labels exist as an outcome. GPU cost unchanged, all attempts terminal.
