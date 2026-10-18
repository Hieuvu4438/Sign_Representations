# asm_jepa — ASM-JEPA trên ASL-Citizen (proposal15)

Hiện thực PyTorch của **ASM-JEPA** (Anchored State–Motion JEPA) cho skeleton SSL, theo đặc tả
[docs/proposal15/ASM_JEPA_ASL_Citizen_AI_Agent_Implementation_Guide_VI.md](../../docs/proposal15/ASM_JEPA_ASL_Citizen_AI_Agent_Implementation_Guide_VI.md).
Trạng thái từng phần (done / tested / not run / blocked) nằm ở [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

Setting: **in-domain ASL-Citizen SSL with signer-independent evaluation**. SSL chỉ dùng official train
(40.154 video, không nhãn); đánh giá trên official val/test (signer khác train). Train gallery đã được SSL nhìn
thấy; đây không phải setting cross-corpus hay tái lập trực tiếp SignRep.

## Phương pháp tóm tắt

| Tầng | Nội dung | Config |
|---|---|---|
| Foundation | manifest chính thức, pose COCO-133 → 114 joint (body 4 / LH 21 / RH 21 / face 68), 7 kênh/joint, full-clip 64 frame, patch 4 frame | `configs/data`, `configs/pose` |
| JEPA base (R2) | EMA teacher đọc mọi token hợp lệ của shared clean view; online chỉ nhận token context-safe; predictor dự đoán latent của target; centred CE (S-JEPA) | `pretrain/base.yaml` |
| + state anchor (R3) | intact online pass dự đoán descriptor tọa độ (offset 0/2/3 của mỗi patch) | `pretrain/anchor_only.yaml` |
| **ASM-v1 (R4, pilot)** | + relation tokens (LH–RH, LH–mũi, RH–mũi) | `pretrain/state.yaml` |
| ASM-v2 (R5) | + motion tokens: log-signature bậc 2 (geometry 3-d, timing 7-d) của quỹ đạo cổ tay, 3 family × 4 tâm × 3 thang thời gian thực | `pretrain/v2_no_global.yaml` |
| ASM-v2 full (R6) | + global consistency (VICReg-style, teacher G là EMA) | `pretrain/full.yaml` |
| R1 control | coordinate MAE, cùng kiến trúc state+relations, không EMA target | `pretrain/mae_r1.yaml` |
| R0 baseline | supervised scratch, cùng backbone/head/augmentation | `eval/finetune.yaml --init scratch` |

Chống rò rỉ target: raw mask theo đơn vị (part, patch); mỗi token khai báo raw dependencies; **dependency
closure** loại mọi relation/motion token đọc giá trị bị che; online encoder chỉ nhận chuỗi token đã gather
(không phải chỉ mask attention). Hand scale tính theo từng frame; vai là anchor công khai. Test counterfactual
(thay tọa độ/score bị che bằng donor) kiểm chứng chuỗi đầu vào context không đổi.

Readout cố định: `state_part_concat1024_l2` = mean theo thời gian của state token mỗi part → L2 → concat
[body, LH, RH, face] → L2 (teacher encoder mặc định).

## Đánh giá Sign Video Retrieval

`cli/retrieval.py` hiện thực **dictionary retrieval gloss-level max-instance**: gallery = official train,
query = official val (chọn model) hoặc test (một lần, sau khi khóa config); điểm của gloss = cosine lớn nhất
giữa query và các video train của gloss đó; xếp hạng toàn bộ 2.731 gloss. Báo cáo:

- `metrics`: R@1/5/10, MRR, DCG (= mean 1/log2(rank+1)), dạng 0..1 và `*_pct`; tie phá theo class id tăng dần.
- `signrep_style`: DCG / Rec@1 / Rec@5 với tie *pessimistic* (rank = số class có điểm ≥ điểm đúng) — đúng
  quy ước SignRep §7.2 mà các method khác trong repo đang dùng
  (`method/phonojepa/runtime/eval/PJ_PLctx_FULL_160ep_dictionary`), nên số test so sánh trực tiếp được.
- coverage / failure_count (mẫu không có pose hợp lệ vẫn nằm trong mẫu số), per-signer, bootstrap CI theo query
  và theo signer.

Mốc tham chiếu đã có trong repo (ASL-Citizen test, gallery train 40.154, query 32.941; khác input/pretraining,
không phải mục tiêu nghiệm thu):

| Method | Pretraining | DCG | Rec@1 % | Rec@5 % |
|---|---|---:|---:|---:|
| SignRep paper Table 4 (avg) | YouTube-SL-25, RGB | 61.40 | 37.47 | 68.77 |
| SignRep public (đo local, weighted) | YouTube-SL-25, RGB | 70.42 | 49.02 | 79.29 |
| PhonoJEPA PLctx full160 (`_grp`) | ASL-Citizen train, skeleton | 39.62 | 16.68 | 36.90 |

Recognition (classifier ranking, không phải retrieval): ASL-Citizen paper I3D 63.10 % / ST-GCN 59.52 % Top-1;
MS-MAE 75.72 % (YT-ASL pretraining); SignRep 81.37 %.

## Môi trường

```bash
cd method/asm_jepa
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python      # training/eval (torch 2.11, cu128)
PYP=/home/haipd/Sign_Representations/.venv-unisign/bin/python     # chỉ cho extract_pose (rtmlib + onnxruntime-gpu)
$PY -m asm_jepa.cli.doctor --config configs/pretrain/state.yaml
```

Chạy từ `method/asm_jepa` (cwd nằm trên `sys.path`); không cần `pip install`. Mọi CLI có `--help`, trả mã khác 0
khi lỗi; đường dẫn tương đối tính từ `method/asm_jepa`. `data/raw/ASL_Citizen` là symlink local tới bản giải nén
có sẵn trên máy (`/home/dongvk/datasets/ASL_Citizen`); `data/` và `runs/` bị git-ignore (license dữ liệu cấm phân phối).

## Quy trình end-to-end

```bash
# A. manifest + audit split (idempotent)
$PY -m asm_jepa.cli.prepare_aslc --raw-root data/raw/ASL_Citizen --output-root data/processed/aslc --expected-version v1.0

# B. pose: nhập cache RTMW có sẵn (đã xác minh trùng RTMW performance đã pin) ...
$PY -m asm_jepa.cli.import_pose_cache --source-dir data/raw/ASL_Citizen/keypoints --split train --resume
$PY -m asm_jepa.cli.import_pose_cache --source-dir data/raw/ASL_Citizen/keypoints --split val --resume
#    ... hoặc trích lại từ RGB (môi trường pose riêng)
PYTHONPATH=. $PYP -m asm_jepa.cli.extract_pose --config configs/pose/rtmw133.yaml --split train --limit 100
$PY -m asm_jepa.cli.audit_pose --split train --sample-count 100 --render-overlays

# C. stats chỉ từ train + tests
$PY -m asm_jepa.cli.fit_stats --config configs/pretrain/full.yaml --split train
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q tests

# D/E. smoke, overfit, throughput
$PY -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --synthetic --run-dir runs/smoke_synthetic \
    --set train.max_steps=20 --set train.warmup_steps=2 --set train.batch_size_per_gpu=4 --set train.accum_steps=1 \
    --set data.num_workers=0 --set eval.frozen_every_steps=0 --set train.checkpoint_every_steps=10
$PY -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --run-dir runs/smoke_real --limit-train-samples 128 \
    --set train.max_steps=100 --set train.warmup_steps=10 --set eval.frozen_every_steps=0 --set train.checkpoint_every_steps=50
$PY -m asm_jepa.cli.train_classifier --config configs/eval/finetune.yaml --init scratch --overfit-samples 16 \
    --run-dir runs/sanity_overfit16 --max-steps 500
$PY -m asm_jepa.cli.pretrain --config configs/pretrain/full.yaml --run-dir runs/profile_full --profile-steps 1000 \
    --set eval.frozen_every_steps=0

# F. SSL pilot chính (ASM-v1), resume nếu bị dừng
setsid nohup $PY -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --run-dir runs/aslc_asm_state_s0 \
    > runs/aslc_asm_state_s0.log 2>&1 &
$PY -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --run-dir runs/aslc_asm_state_s0 \
    --resume runs/aslc_asm_state_s0/checkpoints/last.pt

# G. frozen retrieval (val để chọn; test chỉ ở bước J)
R=runs/aslc_asm_state_s0
$PY -m asm_jepa.cli.extract_embeddings --config configs/pretrain/state.yaml --checkpoint $R/checkpoints/best_val.pt --split train --output $R/embeddings/train.npz
$PY -m asm_jepa.cli.extract_embeddings --config configs/pretrain/state.yaml --checkpoint $R/checkpoints/best_val.pt --split val --output $R/embeddings/val.npz
$PY -m asm_jepa.cli.retrieval --gallery $R/embeddings/train.npz --queries $R/embeddings/val.npz --protocol gloss_max --output $R/reports/frozen_val.json

# H. linear probe, fine-tune, supervised scratch
$PY -m asm_jepa.cli.train_classifier --config configs/eval/linear.yaml --train-embeddings $R/embeddings/train.npz --val-embeddings $R/embeddings/val.npz --run-dir runs/aslc_linear_state_s0
$PY -m asm_jepa.cli.train_classifier --config configs/eval/finetune.yaml --init $R/checkpoints/best_val.pt --run-dir runs/aslc_ft_state_s0
$PY -m asm_jepa.cli.train_classifier --config configs/eval/finetune.yaml --init scratch --run-dir runs/aslc_scratch_s0

# J. test — chỉ sau khi khóa config
$PY -m asm_jepa.cli.import_pose_cache --source-dir data/raw/ASL_Citizen/keypoints --split test --resume
$PY -m asm_jepa.cli.extract_embeddings --config configs/pretrain/state.yaml --checkpoint $R/checkpoints/best_val.pt --split test --output $R/embeddings/test.npz
$PY -m asm_jepa.cli.retrieval --gallery $R/embeddings/train.npz --queries $R/embeddings/test.npz --protocol gloss_max --output $R/reports/frozen_test.json
$PY -m asm_jepa.cli.evaluate_classifier --checkpoint runs/aslc_ft_state_s0/checkpoints/best_val.pt --split test --output runs/aslc_ft_state_s0/reports/test.json
$PY -m asm_jepa.cli.report --runs $R runs/aslc_ft_state_s0 runs/aslc_scratch_s0 --output runs/final_report.md
```

Ablation khác: `configs/pretrain/{base,anchor_only,v2_no_global,full,mae_r1}.yaml`; fine-tune từ checkpoint v2
dùng `configs/eval/finetune_full.yaml` (kiến trúc phải khớp checkpoint, nếu không CLI dừng và liệt kê khác biệt).

## Chạy job final (method full, chạy ngầm)

Method final: `configs/pretrain/final.yaml` (ASM-v2 full + bộ lọc quan sát theo audit; xem IMPLEMENTATION_STATUS §6).

```bash
scripts/final_job.sh start              # chạy ngầm (setsid nohup); tham số runner truyền thêm được ghi nhớ
scripts/final_job.sh install-watchdog   # cron 10 phút: tự khởi động lại nếu runner chết / máy reboot
scripts/final_job.sh status             # stage hiện tại, step, samples/s, ETA, val gần nhất
scripts/final_job.sh log                # job.log + log của stage hiện tại
scripts/final_job.sh summary            # bảng kết quả (runs/final_s0/summary.md)
scripts/final_job.sh stop               # dừng (resume được bằng start); watchdog tạm dừng cho tới start
scripts/final_job.sh reset              # xóa bộ đếm lỗi sau khi sửa stage bị failed
```

Thư mục `runs/final_s0/`: `job_state.json`, `job.log`, `logs/<stage>.log`, `pilots/`, `pretrain/` (metrics.jsonl,
checkpoints best_val/last/milestone_*, embeddings, reports/frozen_{val,test}.json), `stage_eval/milestone_*/`,
`linear*/`, `finetune/`, `scratch/`, `final_report.md`, `results.csv`, `summary.md`, `summary.json`.
Test chỉ được đọc ở stage_eval/evaluate (sau khi train và chọn checkpoint bằng val); kết quả test của milestone
là đường cong chẩn đoán, con số chính thức là checkpoint `best_val`.

## Cấu trúc

```
asm_jepa/config.py               YAML inherits + --set + schema chặt (key lạ → lỗi)
asm_jepa/data/                   manifest, pose_store (npz contract), schema 133→114, sampling, normalization,
                                 augment, dataset (SampleBuilder: views, tokens, mask, closure), synthetic
asm_jepa/features/               state, relations, logsig2, paths, tokens (layout), dependencies (closure),
                                 masking (raw mask sampler), stats
asm_jepa/models/                 tokenizer, encoder (pre-LN, SDPA, DropPath), predictor, pooling, heads, asm_jepa
asm_jepa/training/               losses, ema, centers, optim, schedules, sampler, checkpoint, distributed,
                                 engine (SSL loop), classifier (linear probe / fine-tune)
asm_jepa/evaluation/             ranking, retrieval, classification, diagnostics, bootstrap, embeddings
asm_jepa/cli/                    doctor, prepare_aslc, extract_pose, import_pose_cache, audit_pose, fit_stats,
                                 pretrain, extract_embeddings, retrieval, train_classifier, evaluate_classifier, report
tests/                           split/vocab, schema/sampling/padding, normalisation leakage, closure,
                                 counterfactual, logsig2, EMA/centres, losses, ranking/retrieval, resume, probe
runs/<run_id>/                   config_resolved.yaml, provenance.json, metrics.jsonl, checkpoints/, embeddings/, reports/
```
