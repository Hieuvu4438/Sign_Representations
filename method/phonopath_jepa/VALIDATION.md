# Verification

Kiểm tra trên server bằng Python 3.13 / torch 2.11, CPU. Official test ASL Citizen/WLASL không được score; chỉ đọc danh sách WLASL để tạo manifest.

| Kiểm tra | Bằng chứng |
|---|---|
| Core/support/evaluation | 42 test ban đầu; thêm 3 recovery test cho production; includes end-to-end hidden-source overwrite, observation gate dependency, interpolation endpoints, smoothing dilation, target-query preservation, A0 numerical parity, all registered arms forward/backward, train-only PCA, scorer ties |
| pySigLib 4.0.0 oracle | 6 test; d=2/3/4/8/9 và Chen gap-skipping; tolerance 1–2e-12 float64 |
| Tổng suite trong oracle venv | `evidence/tests.xml`: 48 passed; core env skips oracle module khi không cài pySigLib |
| Model đầy đủ 256/6 + predictor128/4 | `evidence/full_model_cpu/results.json`: train/save/reload, optimizer replay bit-for-bit CPU1; grp768, grp+P1536, mw1536, ord23072 |
| Public CLI với dữ liệu thật | `evidence/cli_smoke.json`, `evidence/cli_smoke.log`: 8 train/8 val, fit stats → train → resume → extraction → custom retrieval → G0/linear/fine-tune; 2–3 bước, không có giá trị khoa học |
| Classifier inference | `evidence/classifier_smoke.json`: supervised checkpoint → val inference |
| Raw WLASL pickle | `evidence/wlasl_loader_smoke.json`: 1 official train clip → teacher/readouts |

## Reproduce

```bash
cd /home/haipd/Sign_Representations/method/phonopath_jepa
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
$PY -m pytest tests -q
$PY scripts/smoke.py --full-model --out evidence/full_model_cpu
$PY scripts/smoke_cli.py --out runtime/another_cli_smoke
```

Oracle sử dụng `/tmp/phonopath-oracle-venv` riêng, module pySigLib/kauri nằm trong venv đó, đọc dependency numpy/torch từ môi trường đã có bằng `.pth`; không cài hoặc đổi dependency venv chính. Có thể tái tạo môi trường tạm khác rồi cài pySigLib với `--no-deps` để tránh tải lại torch/CUDA lớn:

```bash
/tmp/phonopath-oracle-venv/bin/python -m pytest tests -q --junitxml=evidence/tests.xml
```

## Giới hạn bằng chứng

- A1 full160 đã bắt đầu train bằng service production; chưa hoàn thành 160 epoch, G0 đầy đủ hoặc đo chất lượng cuối cùng. Smoke metrics trên 8 clip không phải VAL-DICT.
- Đã qua CUDA/BF16 batch32, finite gradient và reload checkpoint: `runtime/full160/gpu_preflight.json`. Tốc độ warm smoke khoảng 0,64 step/s và train thực ban đầu khoảng 0,5 step/s trên GPU dùng chung, chưa đạt mục tiêu ≥5 step/s. Job ghi speed/ETA thực liên tục.
- Chưa chạy probe ASL-LEX vì file `signdata.csv` không có trong `docs/proposal16/data` hiện tại. Code và protocol đã có; nhãn cơ chế không bị tạo từ input signature.
- Chưa có hai vòng review code độc lập theo gate trước Phase 2; tests và rà soát triển khai không được coi là thay thế reviewer.
- Statistics của smoke chỉ fit 4 clip để kiểm tra kết nối, không được dùng cho run khoa học. Config chính fit mặc định 640 clip train, sampling cố định seed123.

## Production deployment

Xem `FULL160_JOB_VI.md`. `evidence/job_recovery_cli.json` xác nhận SIGUSR1, SIGTERM, corrupt latest fallback và epoch/cursor. `evidence/full160_live_recovery.json` xác nhận systemd tự restart sau supervisor SIGKILL và trainer tiếp tục sau checkpoint125. Không dùng official test.
