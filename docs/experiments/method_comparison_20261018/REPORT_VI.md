# So sánh kết quả các method — 18/10/2026

Chỉ tổng hợp số đã đo trong repo, không chạy lại model. Metric chính: R@1 frozen dictionary retrieval trên official ASL Citizen test. Không trộn ASLC300/LOSO/val hoặc recognition accuracy vào bảng test.

**Skeleton tốt nhất đã đo: PLctx full160, R@1=16,6783%. Pipeline RGB/SignRep tốt nhất trong các kết quả đã đăng ký: chamfer20, R@1=58,42%. H008 single-vector khoảng57,30% trên3seed.**

Đây là xếp hạng theo task; input, pretrain, budget, feature extraction và readout khác nhau. Không suy ra nhân quả hoặc SOTA ngoài repo.

## Skeleton — raw readout, không PCA/whitening

| Method | Readout | Budget | R@1 (%) | R@5 (%) | DCG×100 |
|---|---|---|---:|---:|---:|
| PLctx full160 | grp | 160 epochs / 200640 updates | 16.678 | 36.896 | 39.618 |
| PLctx continuation | grp | selected global210640; branches searched separately | 16.633 | 36.784 | 39.578 |
| PhonoPath-JEPA A1 epoch40 | grp | 40 epochs / 50160 updates | 11.305 | 27.613 | 33.538 |
| Pose-JEPA Hybrid | raw_test | run125482 updates; selected115000 | 5.744 | 15.279 | 24.860 |
| ASM-JEPA | state_part_concat | val-selected checkpoint5000 | 1.069 | 3.324 | 14.640 |

## Skeleton — các readout có PCA/whitening

| Method | Readout | Budget | R@1 (%) | R@5 (%) | DCG×100 |
|---|---|---|---:|---:|---:|
| PhonoPath-JEPA A1 epoch40 | mw | 40 epochs / 50160 updates | 11.059 | 28.317 | 33.960 |
| PhonoPath-JEPA A1 epoch40 | ord2 | 40 epochs / 50160 updates | 9.365 | 23.803 | 31.008 |
| Pose-JEPA Hybrid | white512_test | run125482 updates; selected115000 | 9.317 | 24.046 | 31.030 |
| PhonoPath-JEPA A1 epoch40 | grp+P | 40 epochs / 50160 updates | 8.567 | 22.510 | 30.015 |

## RGB — các pipeline dựa trên SignRep

| Method | Readout | Budget | R@1 (%) | R@5 (%) | DCG×100 |
|---|---|---|---:|---:|---:|
| SignRep + chamfer20 | R2 | external RGB-pretrained author encoder; no extra encoder training | 58.420 | 84.210 | 76.240 |
| H008 set-to-vector distillation | single vector1024 | label-free head; frozen external SignRep teacher | 57.300 | — | — |
| SignRep + bins2 | R1 | external RGB-pretrained author encoder; no extra encoder training | 54.930 | 82.390 | 74.180 |
| SignRep public weighted | R0 | external RGB-pretrained author encoder; no extra encoder training | 49.020 | 79.290 | 70.420 |

## Diễn giải

- PLctx continuation16,6328% không cải thiện test so với baseline16,6783%; chênh lệch0,0455điểm, một seed và chưa có kiểm định để gọi là khác biệt có ý nghĩa.
- PhonoPath A1 chỉ có test epoch40. Readout chính grp+P8,5668%; grp11,3051% là readout phụ cố định đã báo cùng lần. Chưa có A0 ở40epoch nên không kết luận riêng tác dụng của P hoặc hiệu quả compute-matched.
- PhonoPath full160 đang train; không dùng checkpoint đang train như một kết quả đã đo.
- Pose-JEPA whitening5129,3167% là postprocessing khác raw5,7436%, không gộp thành kết quả raw. Archive giữ metric nhưng đã xóa weights/runtime.
- ASM đã loại22gallery không dùng được;6query lỗi vẫn nằm trong mẫu số. Pose có28gallery/4query không dùng được; mẫu số test vẫn giữ32941. PLctx giữ zero vectors theo policy gốc. Có khác biệt failure/preprocessing giữa method.
- SignRep+chamfer là set-to-set/window matching; H008 là learned single-vector head không dùng gloss. Đây là cải tiến pipeline/readout trên cùng encoderRGB ngoại dữ liệu, không phải skeleton encoder mới.
- P4RGB sử dụng bảng kết quả đã đăng ký/claim ledger lưu trong docs; không tuyên bố rerun hoặc kiểm tra lại binary đã xóa.
- PhonoJEPA PLint/PLboth, S1/S2, Graph-JEPA, SignRel-JEPA/SignRel-Temporal và SignCoord có các pilot/protocol riêng; không điền điểm pilot thành full-test. PLctx–Pose fusion đã dừng và xóa runtime17/10, chưa có accuracy fusion được xác minh để xếp hạng.
- SignRep tái tạo trên ASLC (40ksteps) chỉ có proxy5462gallery/2731val, bestproxyR@1=1,8674%; không phải official fulltest.

## Nguồn

- `method/phonojepa/runtime/eval/PJ_PLctx_FULL_160ep_dictionary/aslc/results.json` — SHA256 `f548b6797b77d343b3914c87b571e7897cbe4100899a8a696d3f4751db542161`
- `method/phonojepa/runtime/plctx160_continuation/final_test/metrics.json` — SHA256 `01c96effa7e1f2932763211ab695507b9e63d198f9f3e79d60ced3d399e612af`
- `method/phonopath_jepa/runtime/retrieval_e040_test/results.json` — SHA256 `b8fdc46ad09800ff3d7163bd53d840c5a66216bbbfa85dd462926f3dc3ba29d4`
- `docs/experiments/pose_jepa_results_20261015/official_test_report.json` — SHA256 `eb21c13d22f4b9550639c0aa548fe5c3e20d54ab7bbd5951a611ad2a36ef87ae`
- `method/asm_jepa/runs/final_s0/pretrain/reports/frozen_test.json` — SHA256 `a62a650deaf75cd6c2092c3a4f81a0a97281ff537fc98048368fd0b0cdfa4f31`
- `docs/proposal4/sign-ssl-research-agent/research/paper/tables.json` — SHA256 `037eebc50ffe2c2af45cb8aa9f70f046eb26e2ffc3dc095313b15e3927ed23bc`
- `docs/proposal4/sign-ssl-research-agent/research/CLAIM_LEDGER.md` — SHA256 `268bc8afbca0c352903b66604b90df1cf3870ec4f68aac763d144b639600b9bf`

## Pilot ASLC300 — không cùng full-test

Gallery3000/query1500,300gloss,white512. Bảng đầy đủ giữ ở `pilot_ASLC300_white512.csv`.

| Run | R@1 (%) | LP macro (%) |
|---|---:|---:|
| JH3D | 56.800 | 61.733 |
| PJ_PLctx_16k_grp | 48.933 | 60.800 |
| PJ_PLctx_FULL_160ep_grp | 47.667 | 68.867 |
| PJ_PLctx_grp | 46.267 | 49.400 |
| PJ_PLint_16k_grp | 43.267 | 55.933 |
| PJ_PLboth_grp | 41.867 | 47.133 |
| B13 | 21.267 | 25.133 |

JH3D đứng cao trong pilot này nhưng là RGB; PLctx16k cao hơn PLctx160 trên ASLC300. Không dùng các con số này thay cho official test2731gloss.
