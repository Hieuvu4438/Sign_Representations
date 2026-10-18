# proposal16: chẩn đoán không cần train

Mọi chẩn đoán ở đây chỉ dùng CPU và không train model nào. Chúng đọc pose cache ASL Citizen của `method/asm_jepa/data/processed/aslc` ở chế độ chỉ đọc. Pose cache là output RTMW-x COCO-WholeBody 133, 2D, toạ độ pixel. Nhãn gloss, signer và ASL-LEX **chỉ dùng để chấm điểm**, không dùng để xây feature.

| Script | Kết quả | Nội dung |
|---|---|---|
| `tempo_anova.py` | in ra stdout (chép vào proposal §5.2) | Phân rã phương sai log thời lượng clip theo gloss và signer, trên full train 40.154 clip, chỉ dùng metadata |
| `prep_pose.py` | `<scratch>/val_pose.npz`, `<scratch>/train_pose_half.npz` | Tiền xử lý cho chẩn đoán: observation filter, khung thân, tay trội canonical, nội suy khoảng trống. Lưu FP16 |
| `active_tempo.py` | in ra stdout | Thời lượng hoạt động, tốc độ và độ dài đường đi cổ tay trội, phân rã theo gloss và signer |
| `sig_diagnostics.py` | `results/diag_results.json` | D-SIG: retrieval VAL-DICT / LOSO, signer-NN, độ bền tempo, ε² gloss/signer của 22 descriptor (tĩnh, bin, vận tốc, path signature theo họ path và augmentation) |
| `asllex_probe.py` | `results/probe_results.json` | D-LEX v1: probe tuyến tính độc lập signer (PCA 256), từ từng họ descriptor sang 9 tham số âm vị học ASL-LEX 2.0. Ghi cache feature `<out>.featK.npz` |
| `asllex_probe_v2.py` | `results/probe_v2_results.json` | D-LEX v2: PCA 1024, thêm các tổ hợp "static + X", dùng lại cache của v1. **Không rời gloss**, chỉ tham khảo |
| `asllex_probe_v3.py` | `results/probe_v3_results.json` | **D-LEX v3 (bảng chính, proposal §5.4)**: rời gloss (30 % gloss giữ lại làm test) và rời signer, PCA 256 cho mọi descriptor, static chỉ trung bình, CI 95 % bootstrap theo gloss. Viết sau review độc lập |
| `absent_rate.py` | `results/absent_rate.json` | Phase 0(b): tỉ lệ cửa sổ 8/16 mẫu có đủ support được quan sát, theo kênh, trên mọi cửa sổ và trên cửa sổ hoạt động |

## Tái lập

```bash
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
S=<scratch dir>
cd /home/haipd/Sign_Representations/docs/proposal16/diagnostics
$PY tempo_anova.py
$PY prep_pose.py --split val --out $S/val_pose.npz                    # ~1 phút
$PY prep_pose.py --split train --every 2 --out $S/train_pose_half.npz # ~2 phút
$PY active_tempo.py $S/train_pose_half.npz
OMP_NUM_THREADS=4 $PY sig_diagnostics.py --val $S/val_pose.npz --train $S/train_pose_half.npz --out results/diag_results.json   # ~1 giờ, 1 process
OMP_NUM_THREADS=4 $PY asllex_probe.py --val $S/val_pose.npz --train $S/train_pose_half.npz \
    --asllex ../data/signdata.csv --out results/probe_results.json                                                              # ~40 phút
OMP_NUM_THREADS=4 $PY asllex_probe_v2.py --val $S/val_pose.npz --train $S/train_pose_half.npz \
    --asllex ../data/signdata.csv --cache results/probe_results.json --out results/probe_v2_results.json                       # ~30 phút
OMP_NUM_THREADS=4 $PY asllex_probe_v3.py --val $S/val_pose.npz --train $S/train_pose_half.npz \
    --asllex ../data/signdata.csv --cache results/probe_results.json --out results/probe_v3_results.json                       # ~20 phút
$PY absent_rate.py $S/val_pose.npz > results/absent_rate.json
```

Cache feature `.featK.npz` (vài GB) nằm trong scratch, không lưu ở đây. Muốn chạy v2 thì phải chạy v1 trước, cùng giá trị `--out`/`--cache`.

ASL-LEX 2.0 (`../data/signdata.csv`, `signdataKEY.csv`) được tải từ OSF, dự án `zpha4`. Xem `../literature/P3_sign_domain_signals.md` §1.2 về URL và sha256. Giấy phép CC BY-NC, **không commit vào git**.

## Giới hạn

- Một lần chạy, không có seed (descriptor tất định). Không có CI. Chênh lệch dưới khoảng 0.3 R@1 nên coi là nhiễu.
- Gallery VAL-DICT chỉ là **nửa** train (mỗi clip thứ hai), không phải toàn bộ 40k như protocol chính thức. Số tuyệt đối vì vậy thấp hơn B0 = 13.46 của proposal12, vốn dùng descriptor CONF trong khung bàn tay. Chỉ nên so sánh tương đối giữa các descriptor ở đây.
- D-LEX v1/v2 dùng chung gloss giữa probe-train và val, nên probe có thể thắng nhờ nhận ra gloss. v3 sửa lỗi này. Mọi kết luận trong proposal dựa trên v3.
- z-score theo chiều fit trên gallery. Mỗi block (họ path) có trọng số bằng nhau. Cosine trên descriptor ghép lại là một cách đo **thô**: nó không nói được token signature có giúp một encoder học hay không, mà chỉ đo thông tin *tuyến tính, không cần học* trong từng họ.
- "Hoạt động" trong `active_tempo.py` là một heuristic: cổ tay trội cao hơn mức nghỉ, hoặc đang chuyển động nhanh.
