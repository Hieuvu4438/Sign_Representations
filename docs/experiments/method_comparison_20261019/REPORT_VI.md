# Các method sau train và official test ASL Citizen — 19/10/2026

Tổng hợp số đo đã lưu, không chạy model hay chấm lại test. Galleryofficialtrain40.154 / testqueries32.941 /2.731gloss; R@K và DCG×100 là%. MRR giữ0–1. Không trộn pilot/val/recognition vào bảng retrieval.

**PLctx full160 giữ kết quả skeleton tốt nhất: R@1=16,678%. PhonoPath A1 full160 không vượt baseline: readout chính grp+P6,928%; rawgrp11,511%. Trong các pipelineRGB đã có kết quả đăng ký, SignRep+Chamfer58,42% cao nhất; H008singlevector khoảng57,30%.**

## Readout chính của từng run skeleton

| Method | Readout | Budget | Test R@1 (%) | R@5 (%) | DCG×100 |
|---|---|---|---:|---:|---:|
| PLctx full160 | grp | 160ep/200640updates | 16.678 | 36.896 | 39.618 |
| PLctx continuation | grp | selected210640;twoLRbranches searched | 16.633 | 36.784 | 39.578 |
| PhonoPath-JEPA A1 epoch40 | grp+P | 40ep/50160updates | 8.567 | 22.510 | 30.015 |
| PhonoPath-JEPA A1 epoch160 | grp+P | 160ep/200640updates | 6.928 | 18.305 | 26.706 |
| Pose-JEPA Hybrid | raw_test | run125482;valselected115000 | 5.744 | 15.279 | 24.860 |
| ASM-JEPA | state_part_concat | valselected5000 | 1.069 | 3.324 | 14.640 |

## Cùng raw readout, không PCA/whitening

| Method | Readout | Test R@1 (%) | R@5 (%) | R@10 (%) | DCG×100 |
|---|---|---:|---:|---:|---:|
| PLctx full160 | grp | 16.678 | 36.896 | — | 39.618 |
| PLctx continuation | grp | 16.633 | 36.784 | 46.844 | 39.578 |
| PhonoPath-JEPA A1 epoch160 | grp | 11.511 | 28.005 | 37.100 | 33.476 |
| PhonoPath-JEPA A1 epoch40 | grp | 11.305 | 27.613 | 37.303 | 33.538 |
| Pose-JEPA Hybrid | raw_test | 5.744 | 15.279 | 21.939 | 24.860 |
| ASM-JEPA | state_part_concat | 1.069 | 3.324 | — | 14.640 |

## Readout có PCA/whitening — báo riêng

| Method | Readout | Test R@1 (%) | R@5 (%) | R@10 (%) | DCG×100 |
|---|---|---:|---:|---:|---:|
| PhonoPath-JEPA A1 epoch40 | mw | 11.059 | 28.317 | 38.709 | 33.960 |
| PhonoPath-JEPA A1 epoch160 | mw | 9.872 | 25.078 | 34.298 | 31.463 |
| PhonoPath-JEPA A1 epoch40 | ord2 | 9.365 | 23.803 | 33.077 | 31.008 |
| Pose-JEPA Hybrid | white512_test | 9.317 | 24.046 | 33.360 | 31.030 |
| PhonoPath-JEPA A1 epoch40 | grp+P | 8.567 | 22.510 | 31.438 | 30.015 |
| PhonoPath-JEPA A1 epoch160 | ord2 | 7.340 | 19.259 | 26.769 | 27.286 |
| PhonoPath-JEPA A1 epoch160 | grp+P | 6.928 | 18.305 | 25.640 | 26.706 |

## RGB / SignRep — khác modality và external pretrain

| Method | Readout | Test R@1 (%) | R@5 (%) | R@10 (%) | DCG×100 |
|---|---|---:|---:|---:|---:|
| SignRep + chamfer20 | R2 | 58.420 | 84.210 | — | 76.240 |
| H008 set-to-vector | single_vector1024 | 57.300 | — | — | — |
| SignRep + bins2 | R1 | 54.930 | 82.390 | — | 74.180 |
| SignRep public weighted | R0 | 49.020 | 79.290 | — | 70.420 |

## PhonoPath: epoch40 →160

| Readout | Epoch40 R@1 (%) | Epoch160 R@1 (%) | Delta(points) |
|---|---:|---:|---:|
| grp | 11.305 | 11.511 | +0.206 |
| grp+P | 8.567 | 6.928 | -1.639 |
| mw | 11.059 | 9.872 | -1.187 |
| ord2 | 9.365 | 7.340 | -2.025 |

## Recognition official test — tác vụ khác retrieval

| Method | Regime | Top1 (%) | Top5 (%) |
|---|---|---:|---:|
| Pose-JEPA linear | frozen linear probe | 22.905 | 46.362 |
| ASM-JEPA linear | frozen linear probe | 5.689 | 14.936 |
| ASM-JEPA fine-tune | supervised fine-tune | 55.414 | 75.632 |
| ASM scratch | supervised scratch | 53.808 | 74.309 |

Không xếp chung fine-tune có nhãn với frozen linear probe. PLctx68,87% LP là ASLC300, không phải official2731gloss test; PhonoPath chưa có recognition official test nên không điền điểm smoke.

## Phạm vi và kết luận

- PLctx160 so với continuation16,633%: chênh0,046điểm R@1; một seed, chưa có kiểm định để gọi khác biệt có ý nghĩa.
- PhonoPath grp full160 thấp hơn PLctx grp khoảng5,167điểm R@1. Cùng tổngbudget160ep, nhưng preprocessing/masking/kiến trúc khác; không quy riêng cho signature nếu chưa có A0/A2 cùng code.
- PhonoPath grp+P giảm1,639điểm từ40→160; grp chỉ tăng0,206điểm. Modeltrainloss giảm nhưng retrieval không tăng tương ứng. Readout chính được giữ grp+P, không đổi winner theo test.
- Mọi điểm cùng officialsplit nhưng không phải mọi preprocessing/failurepolicy giống hệt. Pose có40126usablegallery/4failedquery; ASM40132usablegallery/6failedquery, querydenominator giữ32941. PLctx giữzero vectors. PCA/whitening được ghi riêng.
- SignRepRGB dùng pretrain bên ngoài; Chamfer là setmatching khác singlevector. P4/H008 là số đã đăng ký còn trong docs, không tuyên bố đã rerun weights/cache đã xóa.
- S1/S2, PLint/PLboth, Graph-JEPA, SignRel-JEPA, SignCoord và SignPathLab có pilot/protocol khác hoặc chưa có fullofficialtest được xác minh. Không gán số pilot thành test. PLctx–Pose fusion đã dừng, không có kết quả fusionfulltest để xếp hạng.
- SignRep tái tạo trên ASLC chỉ có proxy5462gallery/2731val, không phải officialfulltest.

## Lưu trữ

Kết quả/ranks/logs/config của PhonoPath đã được archive tại `artifacts/phonopath_jepa_results_20261019`. Runtime/checkpoint/cache nặng được dọn theo yêu cầu; code giữ lại. Bảng này có sourcebundles riêng và SHA256, nên vẫn tạo lại được sau cleanup.

