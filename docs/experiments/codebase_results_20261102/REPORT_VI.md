# Thống kê kết quả ASL-Citizen trong codebase — 02/11/2026

Rà soát các số đo đã lưu tại workspace. Không train, không chạy inference hoặc chấm lại test. Ngày rà soát không phải ngày thực nghiệm; phần lớn kết quả chính được lưu trong tháng 10/2026.

## Kết quả chính

- Skeleton: **PhonoJEPA PLctx full160**, official test R@1 **16,678%**, R@5 **36,896%**, DCG×100 **39,618**; train scratch trên 40.154 clip, 160 epochs / 200.640 updates, một seed.
- Pipeline RGB: **SignRep + Chamfer20**, R@1 **58,420%**. Đây là frozen encoder pretrain bên ngoài và set matching; bins2/Chamfer là readout, không phải các encoder được train độc lập trên ASLC.
- Head SSL **H008**: single vector 1024D trên frozen SignRep; ba seed đăng ký 57,28 / 57,31 / 57,31%, trung bình **57,300%**. Đây là học head không nhãn trên ASLC, khác train encoder từ đầu.
- ASLC300 white512: JH3D R@1 **56,800%**; PLctx16k **48,933%**; PLctx full160 **47,667%**. Linear probe full160 **68,867%** chỉ thuộc pilot 300 gloss.

## Official test: frozen dictionary retrieval

Gallery danh nghĩa: official train 40.154 clip; query: official test 32.941 clip; 2.731 gloss. Điểm gloss là max similarity trên các video gallery cùng gloss. R@K là tỷ lệ rank≤K; DCG = mean[1/log2(1+rank)]. Mọi DCG trong báo cáo được nhân 100. Các bảng giữ readout chính/phụ đã khai báo trước, không chọn lại theo test.

### Skeleton: raw readout

| Method | Readout | R@1 % | R@5 % | DCG×100 |
|---|---|---|---|---|
| PLctx full160 | grp | 16.678 | 36.896 | 39.618 |
| PLctx continuation | grp | 16.633 | 36.784 | 39.578 |
| PhonoPath-JEPA A1 epoch160 | grp | 11.511 | 28.005 | 33.476 |
| PhonoPath-JEPA A1 epoch40 | grp | 11.305 | 27.613 | 33.538 |
| Pose-JEPA Hybrid | raw_test | 5.744 | 15.279 | 24.860 |
| ASM-JEPA | state_part_concat | 1.069 | 3.324 | 14.640 |

### Skeleton: PCA/whitening

| Method | Readout | R@1 % | R@5 % | DCG×100 |
|---|---|---|---|---|
| PhonoPath-JEPA A1 epoch40 | mw | 11.059 | 28.317 | 33.960 |
| PhonoPath-JEPA A1 epoch160 | mw | 9.872 | 25.078 | 31.463 |
| PhonoPath-JEPA A1 epoch40 | ord2 | 9.365 | 23.803 | 31.008 |
| Pose-JEPA Hybrid | white512_test | 9.317 | 24.046 | 31.030 |
| PhonoPath-JEPA A1 epoch40 | grp+P | 8.567 | 22.510 | 30.015 |
| PhonoPath-JEPA A1 epoch160 | ord2 | 7.340 | 19.259 | 27.286 |
| PhonoPath-JEPA A1 epoch160 | grp+P | 6.928 | 18.305 | 26.706 |

### RGB: external pretraining / SignRep

| Method | Readout | R@1 % | R@5 % | DCG×100 |
|---|---|---|---|---|
| SignRep + chamfer20 | R2 | 58.420 | 84.210 | 76.240 |
| H008 set-to-vector | single_vector1024 | 57.300 | — | — |
| SignRep + bins2 | R1 | 54.930 | 82.390 | 74.180 |
| SignRep public weighted | R0 | 49.020 | 79.290 | 70.420 |

PhonoPath A1 có readout **chính grp+P**: epoch40 8,567%, epoch160 6,928%. grp là readout phụ, tương ứng 11,305% và 11,511%. Pose raw là chính; white512 là hậu xử lý báo riêng. PLctx continuation không cải thiện full160 (−0,0455 điểm). ASM chọn checkpoint 5.000 bằng validation dù train đủ 60.000 steps.

Pose chỉ có 40.126 gallery dùng được và 4 query lỗi; ASM 40.132 gallery dùng được và 6 query lỗi. Query denominator vẫn là 32.941; PLctx giữ zero vectors. Khác biệt preprocessing/failure policy và pretrain làm các hàng không trở thành đối chứng nhân quả.

Nguồn: [bảng và source bundle 19/10](../method_comparison_20261019/REPORT_VI.md). 11 source bundle và 11 file nguồn gốc khớp SHA256 đã đăng ký; xem `source_audit.csv`. P4/H008 là kết quả đăng ký còn trong tài liệu, chưa xác minh lại bằng weights/inference.

## Recognition official test

| Method | Cách train head | Top-1 % | Top-5 % |
|---|---|---|---|
| ASM-JEPA linear | frozen linear probe | 5.689 | 14.936 |
| ASM-JEPA fine-tune | supervised fine-tune | 55.414 | 75.632 |
| ASM scratch | supervised scratch | 53.808 | 74.309 |
| Pose-JEPA linear | frozen linear probe | 22.905 | 46.362 |

Linear probe dùng nhãn để train classifier trên frozen encoder. Fine-tune/scratch là supervised downstream. ASM fine-tune hơn scratch 1,606 điểm Top-1 ở một seed; chưa đủ để kết luận ổn định qua seed. Không gộp accuracy classifier với retrieval R@1.

## Pilot ASLC300: white512 và LOSO

44 cấu hình/readout được đọc lại từ JSON gốc, đối chiếu R@1 và LP với bảng lưu trước đó; không tính các bản JSON sao chép là seed độc lập. ASLC300: 3.000 gallery train, 1.500 query validation, 300 gloss. LOSO: 10.304 validation clip, gallery là các signer khác trong validation. LOSO whitening fit trên validation nên mang tính transductive. Hai cột là hai protocol khác nhau.

| Run | ASLC300 R@1 % | R@5 % | DCG×100 | LOSO R@1 % | LP macro % |
|---|---|---|---|---|---|
| JH3D | 56.800 | 77.933 | 73.571 | 39.189 | 61.733 |
| PJ_PLctx_16k_grp | 48.933 | 74.067 | 68.622 | 25.378 | 60.800 |
| PJ_PLctx_FULL_160ep_grp | 47.667 | 72.000 | 67.119 | 29.542 | 68.867 |
| PJ_PLctx_grp | 46.267 | 72.533 | 66.514 | 22.642 | 49.400 |
| PJ_A8g_randhand_grp | 43.533 | 69.600 | 64.515 | 23.525 | 48.200 |
| PJ_PLboth_hands_grp | 43.467 | 69.400 | 64.542 | 22.574 | 51.000 |
| PJ_PLint_16k_grp | 43.267 | 69.600 | 64.262 | 24.670 | 55.933 |
| PJ_G3morph_grp | 42.400 | 69.933 | 63.950 | 23.195 | 48.067 |
| PJ_PLboth_shufint_grp | 42.133 | 69.867 | 63.733 | 22.263 | 47.733 |
| PJ_PLboth_grp | 41.867 | 69.533 | 63.624 | 22.535 | 47.133 |
| PJ_PLint_grp | 41.333 | 67.933 | 62.739 | 22.069 | 48.867 |
| PJ_R0both_grp | 41.000 | 67.067 | 62.366 | 20.662 | 47.933 |
| PJ_R4both_grp | 39.800 | 68.933 | 62.393 | 19.420 | 47.867 |
| PJ_R0ctx_grp | 39.133 | 67.000 | 61.459 | 20.312 | 50.133 |
| PJ_G2full_grp | 39.000 | 66.133 | 61.447 | 21.099 | 47.400 |
| PJ_PLboth_proto_grp | 37.733 | 64.733 | 60.249 | 20.254 | 45.467 |
| PJ_G1glob_grp | 36.200 | 63.133 | 58.879 | 18.080 | 43.133 |
| PJ_R4int_grp | 34.733 | 60.800 | 57.553 | 18.507 | 46.733 |
| PJ_BMAE_grp | 34.600 | 58.933 | 56.841 | 17.896 | 46.133 |
| PJ_R0int_grp | 33.600 | 57.600 | 55.504 | 18.478 | 42.600 |
| PJ_PLboth_l1sig_noema_grp | 33.067 | 57.333 | 55.508 | 16.304 | 47.133 |
| PJ_R4ctx_grp | 31.333 | 59.267 | 55.461 | 15.936 | 42.800 |
| PJ_BMAMP_grp | 30.400 | 54.400 | 53.031 | 15.237 | 34.467 |
| PJ_PLboth_l1sig_grp | 28.067 | 54.467 | 52.493 | 14.208 | 43.400 |
| PJ_Bglob_grp | 26.533 | 52.467 | 50.764 | 12.539 | 33.467 |
| PJ_B1rand_grp | 23.867 | 45.800 | 46.913 | 10.773 | 32.800 |
| S1_FULL_4k_grp | 23.333 | 47.600 | 47.808 | 10.491 | 44.000 |
| JV3D | 22.600 | 44.733 | 46.743 | 27.979 | 56.400 |
| B13 | 21.267 | 40.667 | 44.249 | 7.929 | 25.133 |
| B12 | 20.667 | 41.333 | 44.177 | 7.541 | 22.333 |
| B11 | 19.133 | 40.600 | 43.049 | 6.066 | 24.200 |
| B10 | 17.333 | 35.333 | 40.026 | 5.192 | 14.467 |
| PJ_B0 | 17.267 | 35.800 | 39.808 | 4.154 | 24.200 |
| S2_FULL_4k_grp | 15.733 | 35.600 | 39.927 | 7.201 | 36.133 |
| SR3D | 10.067 | 23.067 | 32.195 | 5.144 | 17.600 |
| ASLC002 | 5.467 | 13.467 | 25.474 | 4.988 | 19.133 |
| SRVJ | 4.867 | 12.400 | 24.676 | 1.737 | 7.800 |
| A11_AD | 3.067 | 10.400 | 23.113 | 11.908 | 23.200 |
| A10_AR | 2.200 | 8.133 | 21.410 | 8.812 | 16.067 |
| A13_AD | 2.200 | 8.333 | 21.697 | 10.967 | 18.200 |
| A10_AD | 2.133 | 7.133 | 20.958 | 9.152 | 16.933 |
| A00 | 1.467 | 6.000 | 20.094 | 7.133 | 12.133 |
| A10c_AD | 1.067 | 5.533 | 19.578 | 8.249 | 12.533 |
| A12_AD | 0.667 | 5.733 | 19.399 | 7.861 | 12.933 |

## Pilot ASLC300: raw-average đã tính lại

Protocol: raw window mean → video L2; không PCA, không activity weighting. Bảng này khác `retrieval_mean` cũ vốn L2 từng window trước pooling. JH3D 45,93% ở đây và 56,80% white512 ở trên đều là số đúng theo từng protocol. SignRep public không cập nhật encoder trên ASLC.

| Run | Input | R@1 % | R@5 % | DCG×100 |
|---|---|---|---|---|
| SignRep_public | RGB | 72.200 | 90.800 | 84.555 |
| JH3D | RGB | 45.933 | 78.600 | 68.929 |
| JV3D | RGB | 29.800 | 62.800 | 56.648 |
| B11 | Landmark 2D | 10.067 | 24.067 | 32.498 |
| B13 | Landmark 2D | 9.467 | 22.667 | 31.102 |
| B12 | Landmark 2D | 8.933 | 24.000 | 31.533 |
| B10 | Landmark 2D | 5.533 | 15.733 | 26.139 |
| SR3D | RGB | 5.000 | 14.133 | 25.172 |
| SRVJ | RGB | 2.667 | 8.933 | 21.834 |
| A11_AD | RGB | 2.533 | 8.933 | 21.843 |
| A10_AD | RGB | 1.800 | 6.333 | 20.003 |
| ASLC002 | RGB | 1.733 | 4.867 | 18.987 |
| A00 | RGB | 1.733 | 5.867 | 19.483 |
| A10_AR | RGB | 1.467 | 5.333 | 19.625 |
| A12_AD | RGB | 0.867 | 3.133 | 17.791 |
| A13_AD | RGB | 0.800 | 3.867 | 18.239 |
| A10c_AD | RGB | 0.733 | 3.467 | 17.718 |

## JEPA-targets và ASC-JEPA: validation bổ sung

Các nhánh RGB J/JP/M/MP/S/JREL/JNM/JMOT có log/config và LOSO validation, 4.000 steps, batch32. CSV `jepa_targets_LOSO.csv` giữ mọi readout đã có; riêng eval_M.json chỉ có mean/bins2, không tự điền whitening. Không gán các điểm này thành official test.

| Run | Target | LOSO white512 R@1 % | R@5 % |
|---|---|---|---|
| J | latent=True; pixel=False; priors= | 7.376 | 17.120 |
| JP | latent=True; pixel=False; priors=POSE | 19.934 | 39.897 |
| MP | latent=False; pixel=True; priors=POSE | 18.546 | 36.976 |
| S | latent=False; pixel=False; priors=POSE | 10.142 | 22.671 |
| JREL | latent=True; pixel=False; priors=REL | 13.519 | 29.552 |
| JNM | latent=True; pixel=False; priors=NM | 1.960 | 5.765 |
| JMOT | latent=True; pixel=False; priors=MOT | 20.730 | 39.024 |

ASC-JEPA warm-start SignRep public, 3.000 steps (B4 config), đánh giá LOSO validation 10.304 clip. B0 là baseline, asc_B* là continuation. Cả B4 và B2 vẫn thấp hơn baseline ở weighted R@1; bins2 là readout khác.

| Run | Weighted R@1 % | Bins2 R@1 % |
|---|---|---|
| B0 | 49.495 | 53.882 |
| asc_B0 | 48.185 | 53.057 |
| asc_B1 | 47.943 | 52.572 |
| asc_B4 | 48.991 | 53.174 |
| asc_B2 | 48.874 | 53.280 |
| asc_C4a | 48.622 | 53.115 |

## Pilot nhỏ và trạng thái method khác

SignRel-JEPA: local pilot 40 gloss, ba training seed, không phải full benchmark. Mean ± SD theo seed:

| Run | R@1 mean % | SD điểm % | R@5 mean % | Seeds |
|---|---|---|---|---|
| A_generic | 2.083 | 3.608 | 12.083 | 3 |
| B_articulator | 2.917 | 1.909 | 14.583 | 3 |
| C_relation | 1.667 | 0.722 | 19.583 | 3 |
| D_combined | 2.083 | 1.909 | 17.917 | 3 |

Nguồn kết luận pilot: `relation_benefit_not_supported_by_this_pilot`. SignRel-Temporal là thí nghiệm riêng: SSL train 47 clip How2Sign + 3 dev, retrieval subset ASLC 40 gloss. A4−A1 macro R@1 trung bình +0,42 điểm qua ba seed, hướng không nhất quán; không phải method pretrain trên ASLC full. Xem [report gốc](../../../method/signrel_temporal/results/confirmatory_report.md).

| Thư mục / nhóm | Bằng chứng và phạm vi |
|---|---|
| `phonojepa` | PLctx/PLint/PLboth và ablation có pilot; PLctx full160 và continuation có official test. Checkpoint full160 còn local. |
| `phonojepa_s1`, `phonojepa_s2` | Scratch ASLC002 3.000 clip, 4.000 steps; có ASLC300 + LOSO, chưa có full official test được xác minh. |
| `jepa_targets` | JEPA + prior/ablation trên RGB, gồm JH3D/JV3D/SR3D/SRVJ; pilot và LOSO. |
| `obs_jepa` | A00 foundation, các nhánh A adaptation và B10–B13 graph landmark; 4.000 steps trên 3.000 clip. B gate không đạt, P3 chưa chạy. |
| `asc_jepa` | SignRep warm-start + state/change; có validation, chưa thấy full official test. |
| `jepa_baseline` | ASLC001/002 pilot; full-ASLC reproduction dừng ở 40.000 steps và archive. Best proxy R@1 1,867% tại 10k, 5.462 gallery/2.731 val; không có test. |
| `pose_jepa` | Official retrieval + linear probe đã archive; weights/cache/runtime đã xóa, không thể inference từ archive hiện tại. |
| `asm_jepa` | Full-ASLC SSL và downstream linear/fine-tune/scratch có official test; một seed. |
| `phonopath_jepa` | Epoch40 và epoch160 có official test; metric/ranks/config đã archive, weights không giữ. |
| `p4_signssl` | SignRep readout/continuation/SSL head H008; số official test dùng ledger và bảng đăng ký. |
| `signrel_jepa` | Controlled local pilot 40 gloss, ba seed; không có lợi ích retrieval rõ của relation. |
| `signrel_temporal` | SSL How2Sign; ASLC retrieval subset riêng, không xếp cùng full-ASLC train. |
| `signpathlab` | PILOT8: 32 train/16 val, ba seed ×100 updates. E03/E04 cùng PC Top1 27,08%, SD 3,61 điểm. Chỉ tìm thấy evidence khởi động ASLC3000; chưa thấy metric hoàn tất để thống kê. |
| `signcoord_ssl` | Smoke/control evidence trên dữ liệu thật, có run 4 updates; chưa xác minh score full ASLC. Không dùng smoke làm benchmark. |
| `plctx_pose_fusion` | Đã dừng/dọn runtime 17/10; chưa có full-test fusion để xếp hạng. |
| `signrep_spectral`, `frozen_probe`, `stream_reliability`, `research_lab` | Có records/probe/audit/continuation riêng; không xác định là skeleton SSL full-test encoder mới trong bảng chính. |

## Diễn giải thống kê

Đa số run chính chỉ một seed. Chênh lệch điểm không mặc nhiên có ý nghĩa thống kê; CI bootstrap theo signer chỉ phản ánh lấy mẫu signer, không thay thế độ dao động do train seed. Không tính trung bình giữa các method/protocol. H008 là ba seed; SignRel pilot ba seed nhưng phạm vi nhỏ.

Full160 so với PLctx16k: LOSO +4,16 điểm (CI95 [+1,91; +5,12]); LP ASLC300 +8,07 ([+4,06; +12,12]); retrieval ASLC300 −1,27 ([−3,01; +2,25]). Đây là kết quả bootstrap đã lưu trong [báo cáo 13/10](../../proposal12/comparison_plctx_20261013/REPORT_VI.md), một seed/cặp model; không tính lại CI mới.

## File xuất và tái lập

`official_test_retrieval.csv/json`, `recognition_test.csv/json`, `pilot_ASLC300_white512.csv/json`, `pilot_ASLC300_raw_avg.csv/json`, `jepa_targets_LOSO.csv/json`, `asc_validation.csv/json`, `signrel_40gloss_pilot.csv/json` và `source_audit.csv/json`. `sources.json` ghim file đọc bằng SHA256; `summarize.py` tạo lại báo cáo bằng Python stdlib. Các cột trống là chưa có giá trị, không phải 0. Script chỉ tổng hợp text/JSON, không mở checkpoint.
