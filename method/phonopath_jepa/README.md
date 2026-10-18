# PhonoPath-JEPA — proposal16 v1.1

Triển khai độc lập của [proposal16](../../docs/proposal16/PhonoPath_JEPA_Proposal_VI.md), package `pp`. Không sửa hoặc ghi vào runtime của các method cũ. Đã có code train SSL, resume, G0 có giám sát, extraction, retrieval, probe âm vị học, linear probe, fine-tune, few-shot và runner tuần tự. A1 full160 đã được khởi chạy bằng service systemd; xem `FULL160_JOB_VI.md`. Chưa chạy screening hoặc chấm official test.

Luồng mặc định A1:

- COCO-WB 133 → chọn 103 khớp PLctx, observation filter theo ảnh và khoảng cách cổ tay 0.30 độ rộng vai; confidence liên tục, không threshold score.
- Resample đúng 25 fps, 64 mẫu; near/far độc lập; teacher/student trong một view dùng chung warp PCHIP đơn điệu 4 nút.
- Giữ S = 7 × 32 token PLctx. P = 5 × (15 cửa sổ 8 mẫu + 7 cửa sổ 16 mẫu); dim LOC/CONF/REL = 19/195/37. G = 5 token. Cộng CLS: 340 token.
- Log-signature bậc 2 thuần torch, SG5 riêng cho P, nội suy tối đa 2 mẫu; median/IQR + asinh và ngưỡng still fit trên train, đóng băng trong checkpoint.
- Closure trên frame nguồn qua resample, nội suy và smoothing; G student dùng Chen trên đoạn visible, không nối bước nhảy qua gap. Teacher dùng cùng phân hoạch với đoạn cuối `[60,63]`.
- Transformer 256/6 block/4 head, predictor 128/4 block, bias cấu trúc khởi tạo 0, EMA 0.996→0.9999; target là latent LN của 4 layer cuối. Centered-CE riêng S/P, λP=1, không regress signature.
- AdamW 3e-4, wd .05, warmup 10%, cosine, clip 1, bf16 trên CUDA, batch 32. 40 epoch = 50,160 update; 160 epoch = 200,640 update trên 40,154 clip.

## Bắt đầu

```bash
cd /home/haipd/Sign_Representations/method/phonopath_jepa
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
$PY -m pytest tests -q
$PY scripts/smoke.py
$PY scripts/smoke.py --full-model --out evidence/full_model_cpu
```

Core dùng numpy, torch, scipy và scikit-learn trong `.venv-signrep`. Không cần signatory hoặc pySigLib khi train. Oracle pySigLib là dependency tùy chọn, chỉ dùng trong venv tạm; xem [VALIDATION.md](VALIDATION.md).

## Train và resume

Cấu hình mặc định đọc trực tiếp cache pose ASM có sẵn; không tạo bản sao 8 GB. `PoseSource` cũng đọc NPZ full-train PLctx, nhưng NPZ đó đã qua score gate cũ nên không phải observation recipe chính của proposal16.

```bash
$PY -m pp.fit_stats --config configs/A1_s0.json --out runtime/stats/A1.pt
$PY -m pp.train --config configs/A1_s0.json --out runtime/runs/A1_s0
$PY -m pp.train --config configs/A1_s0.json --out runtime/runs/A1_s0 --resume
```

A0 không cần fit P statistics. Các arm khác fit stats theo config tương ứng. `--max-steps 3` dành cho smoke, ghi rõ `smoke=true` trong manifest. Checkpoint chứa student, EMA teacher, predictor, optimizer, centers, frozen stats, RNG, epoch/cursor và fingerprint nguồn. Resume kiểm tra config, nguồn và lịch học; không đổi tổng budget khi resume. Runtime lưu `manifest.json`, `train_log.jsonl`, `latest.pt`, checkpoint EMA ở các epoch đăng ký và `final.pt`.

`configs/` có A0 (3 seed), A1, A2, A3, A4, A5, A6, A7, A8a, A8b, A9, A10, A1-pb, B1, B2, B3 và full160. Chi tiết các control và các điểm cần hiểu đúng nằm trong [IMPLEMENTATION.md](IMPLEMENTATION.md).

## G0 và downstream

G0 dùng encoder 256/4 block, A0/A1/A2; 40% nhãn trên toàn vocab, 5-shot trên 300 gloss cố định, và 100% nhãn tham khảo; mỗi chế độ 2 seed. Cùng số bước/augmentation. Checkpoint G0 được đánh dấu `supervised=true` và không dùng khởi tạo SSL.

```bash
$PY scripts/make_plan.py --stage g0 --out runtime/g0_plan.json
$PY scripts/schedule.py --plan runtime/g0_plan.json             # chỉ in lệnh
$PY scripts/schedule.py --plan runtime/g0_plan.json --run       # chạy tuần tự

TRAIN=/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_train.jsonl
VAL=/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_val.jsonl
$PY -m pp.downstream --ckpt runtime/runs/A1_s0/final.pt --train "$TRAIN" --val "$VAL" --out runtime/lp --linear
$PY -m pp.downstream --ckpt runtime/runs/A1_s0/final.pt --train "$TRAIN" --val "$VAL" --out runtime/ft --steps 30000
$PY -m pp.downstream --ckpt runtime/runs/A1_s0/final.pt --train "$TRAIN" --val "$VAL" --out runtime/fewshot5 --k 5
```

Dùng `--k 1` cho 1-shot; `--config` thay `--ckpt` để chạy scratch/G0. Chọn checkpoint bằng val; lưu danh sách clip/gloss được chọn và predictions val. Fine-tune dùng native encoder/readout PhonoPath; đây không phải tái tạo nguyên xi head/optimizer riêng của ASM-JEPA.

## Frozen retrieval

```bash
$PY -m pp.extract --ckpt runtime/runs/A1_s0/final.pt --source "$TRAIN" --split train --out runtime/A1_train.npz --device cuda
$PY -m pp.extract --ckpt runtime/runs/A1_s0/final.pt --source "$VAL" --split val --out runtime/A1_val.npz --device cuda
$PY -m pp.evaluate --gallery runtime/A1_train.npz --query runtime/A1_val.npz --readout grp --out runtime/A1_grp.json --device cuda
$PY -m pp.evaluate --gallery runtime/A1_train.npz --query runtime/A1_val.npz --readout grp+P --pca 768 --out runtime/A1_grpP.json --device cuda
```

Embedding cache mặc định FP16 để hạn chế disk; `--dtype float32` nếu cần audit độ chính xác. Extraction: cửa sổ 64 mẫu, stride 2; trung bình embedding thô rồi L2 trong scorer. `grp` = 768, `grp+P`/`mw` = 1536 trước PCA, `ord2` = 3072. PCA chỉ fit gallery train. So sánh chính A1-grp vs A0-grp không PCA; so sánh phụ qua cùng pipeline center/PCA768, kể cả A0. Scorer class-max cosine và ties bi quan; R@1/5/10, MRR, DCG, macro R@1, ranks từng query. Chặn các tập ASLC/WLASL thiếu số lượng chuẩn; subset phải ghi `--dataset custom`.

Official test chỉ được score với `--phase 4`; mỗi checkpoint/dataset/readout được ghi ledger và không score lại. Chọn readout/cấu hình trên val trước. Test command dùng `--split test` khi extract, và `--phase 4 --ledger runtime/test_ledger.json` khi evaluate; WLASL thêm `--dataset wlasl`.

## Probe, robustness và quyết định

```bash
$PY -m pp.probe --train runtime/A1_train.npz --val runtime/A1_val.npz \
  --asllex /path/to/signdata.csv --splits /home/dongvk/datasets/ASL_Citizen/splits --out runtime/A1_probe.json
```

Probe D-LEX v3: chia gloss 70/30 cố định, train/val rời signer; StandardScaler/PCA256 chỉ fit probe-train; balanced logistic probe, CI bootstrap theo gloss. Không dùng nhãn cw/ccw tự sinh từ signature. CSV ASL-LEX không được vendoring; cần cung cấp file dữ liệu như proposal mô tả.

Extraction robustness có `--fps 10/15/25`, `--drop .1/.3`, `--jitter .01/.02/.04`, `--reverse`; không nội suy sparse sampling trở lại fps 25. Score cùng gallery; dùng script compare cho paired ranks. `pp.probe.minimal_pairs` nhận nhãn âm vị học độc lập. `scripts/compare.py` áp bootstrap signer × gloss 2000 lần, Holm và τ=max(1.5,2.5 SD từ 3 seed A0). G0/G1 rules ở `scripts/gates.py`.

## WLASL / MS-ASL

```bash
$PY -m pp.prepare --dataset wlasl --root /home/dongvk/datasets/WLASL2000 --vocab 2000 --out runtime/wlasl
$PY -m pp.prepare --dataset msasl --root /home/dongvk/datasets/MS-ASL --out runtime/msasl
```

Adapter tạo metadata, đọc pickle COCO133 trực tiếp, không sao chép pose. WLASL dùng `--vocab 100/300/2000`; thêm `--signer-annotations /path/to/WLASL_v0.3.json` để có signer ID. Nếu thiếu file, ghi `unknown`, không giả định signer-disjoint. MS-ASL chỉ gồm clip local, báo rõ missing/fraction; dùng header thật lấy W/H/fps, số frame lấy từ pickle. Các manifest này dùng chung extraction/downstream CLI với ASLC.

## Runner và profile

```bash
$PY scripts/make_plan.py --stage screen --out runtime/screen_plan.json --asllex /path/to/signdata.csv
$PY scripts/schedule.py --plan runtime/screen_plan.json
$PY scripts/benchmark.py --config configs/A1_s0.json --out runtime/profile_A1.json
# Sau khi G0 và code review đạt yêu cầu:
setsid nohup "$PY" scripts/schedule.py --plan runtime/screen_plan.json --run > runtime/schedule.log 2>&1 &
```

Plan screening chạy A0×3 trước A1/A2, giữ checkpoint 10/20/40 epoch và chấm VAL-DICT/probe từng mốc. Plan full160 tương tự ở 40/80/120/160; không tự chấm test. Runner dùng process lock, mỗi lần một subprocess, chờ GPU còn tối đa 1 compute process khác; watchdog theo log tiến triển và restart có giới hạn. Hai vòng review độc lập của §12 vẫn là gate trước Phase 2, không phải một kết quả đã đạt bởi unit test.

Classifier checkpoint downstream có thể chấm bằng CLI riêng:

```bash
$PY -m pp.classify --ckpt runtime/ft/best.pt --source "$VAL" --split val --out runtime/ft_val.json
```

Đổi `--split test` và source tương ứng sau khi khóa model; classifier test có ledger riêng. `scripts/diagnose.py` đọc embeddings val và bản `--reverse`, báo signer/gloss NN, cosine đảo thời gian và minimal pairs từ annotations độc lập. NN được ghi nhãn rõ để phân biệt với linear probe.

Job A1 full160 chạy ngầm và phục hồi qua systemd: xem [FULL160_JOB_VI.md](FULL160_JOB_VI.md). Quản lý bằng `scripts/a1_full160.sh status|logs|checkpoint|pause|resume`.

VAL-DICT checkpoint epoch40 chạy bằng service riêng: xem [EPOCH40_RETRIEVAL_VI.md](EPOCH40_RETRIEVAL_VI.md), theo dõi `scripts/a1_e040_retrieval.sh status`. Kết quả hoàn tất ở `runtime/retrieval_e040/results.json` và `REPORT_VI.md`.

Official ASL Citizen test tại epoch40 được bổ sung theo yêu cầu người dùng, đợi VAL-DICT rồi chạy tự động: [EPOCH40_TEST_RETRIEVAL_VI.md](EPOCH40_TEST_RETRIEVAL_VI.md). Trạng thái: `scripts/a1_e040_test_retrieval.sh status`.
