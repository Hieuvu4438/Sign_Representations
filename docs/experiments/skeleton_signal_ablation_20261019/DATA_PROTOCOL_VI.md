# Dataset, split và observation contract

## 1. Train trên bộ nào?

**Vòng tìm tín hiệu: chỉ ASL Citizen official train.** Lý do thiết kế: cùng ngôn ngữ, có hai tay/body/face, split khác signer và dictionary retrieval đúng tác vụ quan tâm. Dùng toàn bộ vocabulary ngay từ screening; subset nhỏ chỉ dùng smoke, không chọn winner trên ASLC300 rồi mặc định chuyển full vocabulary.

| Tập | Số video metadata | Signer | Gloss | Dùng để làm gì? |
|---|---:|---:|---:|---|
| ASLC train | 40.154 | 35 | 2.731 | SSL pretrain; gallery; fit normalization/PCA; supervised reference/probe train riêng |
| ASLC val | 10.304 | 6 | 2.731 | Chọn recipe/readout/module; probe validation; frozen retrieval development |
| ASLC test | 32.941 | 11 | 2.731 | Batch đánh giá cuối sau khi khóa quyết định |

Signer rời nhau; vocabulary giống nhau. Đây là **unseen-signer, seen-vocabulary retrieval**, không phải unseen-gloss learning. Số liệu đã đối chiếu metadata local; [snapshot](data_snapshot.json).

## 2. Nguồn local cần dùng

| Đầu vào | Đường dẫn |
|---|---|
| Official split CSV | `/home/dongvk/datasets/ASL_Citizen/splits/{train,val,test}.csv` |
| Pose manifest train | `/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_train.jsonl` |
| Pose manifest val | `/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_val.jsonl` |
| Pose manifest test | `/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_test.jsonl` |
| Pose từng video | `pose_relpath` trong JSONL, resolve tương đối với thư mục chứa manifest |
| Raw pose để đối chiếu provenance | `/home/dongvk/datasets/ASL_Citizen/keypoints/` |
| Video để kiểm tra detector khi cần | `/home/dongvk/datasets/ASL_Citizen/videos/` |
| ASL-LEX annotation | `docs/proposal16/data/signdata.csv` |
| Định nghĩa các cột ASL-LEX | `docs/proposal16/data/signdataKEY.csv` |

Đã xác nhận **mọi pose path trong ba manifest đều tồn tại**: train 40.154, val 10.304, test 32.941. Kiểm tra tồn tại file chưa xác nhận file đọc được, số frame đúng hoặc keypoint chính xác; preflight phải kiểm tra schema và missingness. Không ghi vào dataset/cache nguồn; không cần tạo thêm bản sao toàn bộ pose.

Không dùng lẫn NPZ cũ đã qua confidence gate của PLctx với pose cache ASM trong cùng ablation. Các cache này có observation policy khác nhau. Recipe mới phải đọc cùng nguồn qua một adapter duy nhất.

## 3. Nhãn nào được phép xuất hiện ở đâu?

| Thông tin | SSL pretrain | Evaluation/probe | Ghi chú |
|---|---|---|---|
| Skeleton, timestamp, observation validity | Có | Có | Input thật |
| Detector confidence | E1 không đưa vào embedding/loss weighting; chỉ detection-failure flag đã xác minh | Quality diagnostics | Score không phải calibrated correctness probability |
| Gloss/class ID | Không dùng loss, pair selection hoặc sampler | Retrieval scorer; supervised reference | Dataset label vẫn có trong file; training adapter phải không dùng |
| Signer ID | Không dùng representation loss hoặc positives | Per-signer metric/nuisance probe | Train bằng signer-adversarial loss là track supervision khác |
| ASL-LEX labels | Không | Probe/slices | Dùng chọn module là supervised model selection, phải khai báo |
| Test labels | Không | Batch cuối | Không tune bằng test |

Dùng video-uniform shuffled sampler, không balanced theo gloss; batch 32. Mỗi seed có sample order/window/mask stream xác định để so các arm. Resume lưu cursor đã tiêu thụ, optimizer, RNG, teacher và centers.

## 4. Observation và feature definition cho S0–S7

Raw layout là COCO-WholeBody 133; adapter hiện có chọn 103 joints: body9 + face52 + LH21 + RH21. Giữ map joint ID cố định, không suy ra side từ vị trí pixel. Screening dùng **physical LH/RH**, chưa canonical hóa tay trội và chưa random mirror.

| Ký hiệu | Định nghĩa dùng ở vòng đầu | Những gì không thêm sẵn |
|---|---|---|
| H | Mỗi hand: tọa độ 21 joints trừ wrist, chia scale shoulder theo clip; không xoay về palm frame | Không explicit angle/bone/velocity |
| L | Body-frame wrist và elbow của cả hai tay; shoulder center/scale/roll thống nhất | Không explicit wrist velocity |
| M | Wrist velocity 2D của LH/RH ở các half-span 0,08/0,16/0,32 giây khi đủ source support | Không acceleration, không pseudo motion labels |
| B | LH↔RH wrist displacement, distance, relative velocity | Không thêm handshape mới |
| Fmouth | Mouth/chin landmarks trong head-local frame, loại head rigid motion | Không đưa mouth RGB texture vào |
| Fother | Brows/eyelids local-frame + head translation/roll dynamics | Là bundle; phải tách tiếp nếu giúp |
| G | Directed hand bone vectors + finger-angle encoding từ H; angle dùng sin/cos nếu cần wrap | Không đổi spatial encoder/backbone riêng |

**S2 = H+L.** S0 dùng H và static shoulder scale anchors đã khai báo; không có dynamic L. S1 dùng L, không có chi tiết ngón. H/L đều là chuỗi, nên “không explicit motion” không nghĩa là không còn thông tin thời gian.

M/B/G là hàm của H/L: nếu có gain, kết luận là **explicit encoding/inductive bias giúp ở recipe này**, không phải thêm một observation mới. B dùng cùng derivative recipe với M; benefit của B có thể do explicit motion, nên confirmation phải có control M-only tương ứng.

Body normalization dùng shoulder statistics ổn định theo clip/view; không dùng normalization thay đổi tùy tiện theo frame gây fake velocity. Face normalization phải có source dependency với eye/head anchors. Ở masked student, feature/smoothing/anchor phải chỉ dùng source được phép nhìn; teacher không có stream bị bỏ khỏi arm.

**Không tái sử dụng nguyên feature adapter PLctx để gọi là “raw S2”:** [features.py](../../../method/phonojepa/pj/features.py) đã có finger bones và within-bin wrist velocity. Cần adapter mới tách các feature theo bảng trên. Nếu giữ nguyên adapter cũ, không thể dùng nó làm đối chứng cho “+velocity” hay “+bone”.

Missing point dùng observation mask, không coi tọa độ 0 là missing mặc định. Short gap interpolation tối đa 2 source samples; derivative chỉ hợp lệ nếu đầy đủ endpoints và không bắc qua gap dài. Dùng timestamp thật; tốc độ tính theo giây. Không cho score khác scale giữa dataset đi thẳng vào loss mà chưa kiểm tra provenance.

E1 dùng cùng observation policy cho mọi arm: finite coordinates, frame validity và image bounds; detection-failure flags chỉ dùng khi provenance đã xác minh. Chưa áp hard confidence threshold riêng theo part, chưa confidence-weight target. Valid target nhận trọng số1 trong group normalization. Confidence-aware input/weighting là ablation Q riêng sau E1, tránh thay đổi đồng thời feature content và reliability objective.

## 5. ASL-LEX: dữ liệu đã có và giới hạn

File `signdata.csv` hiện có **2.723 rows**, `Code` unique; đọc lossless bằng Latin-1 trong audit metadata. File key có 184 rows định nghĩa biến. Không mặc định UTF-8, không dùng `errors=ignore` làm mất ký tự.

Official ASLC CSV có **2.723 giá trị code thô nhưng gồm sentinel `NA`**. Sau loại sentinel: 2.722 code hợp lệ local, join được cho **2.725/2.731 gloss**. Sáu gloss không có code: ABOVE, INSTINCT, LETTER2, MINIMUM1, SENTENCE1, WELCOME1. Không đưa `NA` thành một lexeme/group thật. Có code dùng cho nhiều gloss variants; group code/variant khi chia probe/lexical holdout.

Những trường kiểm tra được trong header: `Handshape.2.0`, `Flexion.2.0`, `SelectedFingers.2.0`, `MajorLocation.2.0`, `MinorLocation.2.0`, `Movement.2.0`, `RepeatedMovement.2.0`, `SignType.2.0`, `NonDominantHandshape.2.0`, `UlnarRotation.2.0`, `Compound.2.0`. Ý nghĩa/class values phải đọc `signdataKEY.csv`, không diễn giải numeric code như số liên tục.

`UlnarRotation` chỉ là annotation hẹp, không đại diện toàn bộ palm orientation 3D. Header được kiểm tra **không có gold label mouth/non-manual đầy đủ**; không tự dựng non-manual ground truth từ feature đang train. Face diagnostic dùng perturbation/measurement proxy và subset expert-verified nếu tạo được; báo rõ trạng thái khi chưa có nhãn.

## 6. Khi nào dùng dataset khác?

| Track | Pretrain | Gallery/query | Mục tiêu và thời điểm |
|---|---|---|---|
| Main signal ablation | ASLC train | ASLC train / val | Tìm input/objective và chọn module |
| Final ASLC | ASLC train | ASLC train / test | Xác nhận cấu hình đã khóa |
| WLASL transfer | **Vẫn chỉ ASLC train** | WLASL train / test | Không fine-tune/tune WLASL; report availability, overlap, signer provenance |
| MS-ASL replication | Vẫn chỉ ASLC train | MS-ASL train / test khả dụng | Sau WLASL; dedup source IDs/timestamps với các corpus |
| Continuous context | How2Sign **train only** | ASLC train/val và continuous dev diagnostic riêng | Thêm dữ liệu continuous để kiểm tra long-context; không gán sentence cùng nghĩa thành same-gloss relevance |
| Scale | YouTube-ASL train hoặc SignVerse sau audit | Giữ target benchmark như trên | Chỉ mở sau khi baseline/candidate ổn định; baseline và candidate nhận cùng corpus/budget |
| Unseen lexical composition | ASLC train bỏ các held-out code/variant families | Gallery/query của held-out lexemes ở evaluation | Train mới, không dùng checkpoint đã pretrain thấy held-out lexemes |

Trước mọi transfer, kiểm tra source overlap và schema/extractor. Không coi tên dataset khác nhau là bảo đảm độc lập. Cross-corpus có thể vẫn cùng signer/lexeme; phải ghi phạm vi claim.
