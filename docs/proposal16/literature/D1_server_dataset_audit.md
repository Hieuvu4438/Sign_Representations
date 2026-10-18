# D1 — Audit các dataset sign language có trên server (cho proposal16, skeleton SSL)

Ngày audit: **2026-10-16**. Audit chỉ đọc (read-only), chạy một process, không dùng GPU.
Số liệu lấy từ việc đếm file và nhãn, cùng mẫu nhỏ: ffprobe 200 video/dataset, 100–600 file pose/dataset, seed 0.
Đường dẫn gốc được liệt kê trong `docs/datasets/README.md`.
Những số ghi "mẫu" là ước lượng; những số ghi "toàn bộ" là đếm đủ.

## 1. Tổng quan nhanh

| Dataset (bản local) | Loại | Clip local (train / val / test) | Lớp / vocab | Signer ID | Giờ (ước tính) | Keypoint có sẵn |
|---|---|---|---|---|---|---|
| ASL Citizen | ISLR, ASL | 40 154 / 10 304 / 32 941 (toàn bộ, = bản v1.0) | 2 731 gloss | Có (P1..P52): 35 / 6 / 11, **rời nhau** | 65,0 h (train 32,2) | COCO-WB 133, mọi clip |
| WLASL2000 | ISLR, ASL | 14 296 / 3 920 / 2 879 (21 095, đủ theo nslt_2000) | 2 000 | Chỉ có trong `WLASL_v0.3.json` (LA-Sign): 119 signer, **chồng giữa các split** | ≈ 14 h | COCO-WB 133, mọi clip |
| MS-ASL (local) | ISLR, ASL | 12 711 / 3 297 / 2 847 (18 855 / 25 513 = 73,9 %) | 1 000 (test local có 944) | Có: 126 / 26 / 14, rời nhau | ≈ 16 h | COCO-WB 133, mọi clip đã cắt |
| How2Sign (rgb_front) | CSLT câu, ASL | 31 047 / 1 739 / 2 343 video | văn bản tiếng Anh, không gloss | **Không có** trong nhãn local | ≈ 60 h | COCO-WB 133 |
| CSL-Daily (Sentence_Crop) | CSLR/CSLT, CSL | 18 401 / 1 077 / 1 176 | 2 000 gloss | Có (P0000..P0009): 10 signer, **có mặt ở cả 3 split** | 31,3 h | 133 + bản 77 điểm |
| Phoenix-2014T | CSLR/SLT, DGS | 7 096 / 519 / 642 | 1 115 gloss | Có (Signer01..09): 9 signer, có mặt ở cả 3 split | ≈ 10,5 h | COCO-WB 133 + bản 77 điểm |
| YouTube-ASL | SLT, ASL | **0 video**, chỉ có 11 095 ID | — | — | (paper YouTube-ASL báo ≈ 984 h; chưa tải, chưa kiểm) | Không |

Ghi chú:
- `/home/shared_data/sign_language/CSLDaily` là **bản trùng** với `CSL_Daily_Sentence_Crop`. Kích thước byte của 2 video mẫu giống nhau và `labels.train` trùng khóa. Vì vậy không được đếm CSL-Daily hai lần.
- How2Sign: nhãn train/val nằm trong shared_data. Thư mục `test/test_label/` **rỗng**; nhãn test chỉ có ở `/home/dongvk/datasets/How2Sign/from_uni_sign_source/labels.test` (2 349 mục).

## 2. Video: fps, độ phân giải, độ dài

| Dataset | Độ phân giải (mẫu 200) | fps | Độ dài clip (s) p5 / p25 / p50 / p75 / p95 |
|---|---|---|---|
| ASL Citizen (toàn bộ 83 399, từ metadata cache) | 640×480: 96,1 %; 960×540: 3,9 %; 480×640: 4 clip | 30: 84,8 %; 31: 7 %; 25: 3,8 %; 15: 2 % (min 11,4, max 120) | 1,47 / 1,98 / 2,57 / 3,23 / 5,27 (max 22,6) |
| WLASL2000 | **256×256** (đã crop/resize sẵn) | 25 | 1,16 / 1,72 / 2,24 / 2,84 / 4,24 |
| MS-ASL `cut_processed_video` | 1280×720 37 %, 1920×1080 20 %, 640×480 15 %, 854×480 9 %… | 29,97 phổ biến; 25; 30; 60; 15 | 1,20 / 2,10 / 2,86 / 3,68 / 5,52 |
| MS-ASL `video/` (4 269 video YouTube gốc) | hỗn hợp, tới 1920×1080 | hỗn hợp | p50 3,6, p95 105 (video dài) |
| How2Sign train / test | 1280×720 | 24 (89 %); 23,976; 30; 60 | train 0,79 / 2,79 / 4,73 / 8,68 / 15,2 |
| CSL-Daily | 1280×1280 (66 %), 1080×1080 (34 %) | 30 / 29,97 | 2,92 / 4,10 / 5,03 / 6,60 / 8,28 |
| Phoenix-2014T | **210×260** | 25 | 1,80 / 3,19 / 4,34 / 5,65 / 8,68 |

**Lưu ý MS-ASL:** với 82 % clip đã cắt, `nb_frames` trong header container sai (clip được cắt bằng stream-copy).
Số frame trong pickle khớp với số frame decode được (`-count_frames`), nên phải dùng T của pickle chứ không dùng header.
Với các dataset khác, T của pickle khớp với nb_frames ở 200/200 mẫu.

## 3. Định dạng keypoint (dùng chung một pipeline)

Mọi dataset có keypoint đều cùng một định dạng **pickle dict, mỗi file một clip**:
`{'keypoints': list[T] of ndarray (1,133,2) float64, 'scores': list[T] of ndarray (1,133) float32}`.
Dữ liệu là **COCO-WholeBody 133, 2D**, mỗi frame một người, toạ độ **chuẩn hoá x/W, y/H** theo khung video gốc. Giá trị có thể <0 hoặc >1 khi điểm nằm ngoài khung.
Score là **SimCC thô chưa hiệu chỉnh** (≈0–10), không phải xác suất.
Thang score ở mọi dataset giống nhau (p50 khoảng 6–8,6, max ≈9,6), phù hợp với output của rtmlib Wholebody (RTMW-x 384×288), cùng pipeline với Uni-Sign.
Đây là suy luận từ định dạng và thang score; đã xác minh trực tiếp cho ASL Citizen, chưa xác minh từng dataset.

Layout 133 (COCO-WholeBody): body 0–16 (5/6 vai, 7/8 khuỷu, 9/10 cổ tay), feet 17–22, face 23–90 (68 điểm thứ tự iBUG), tay trái 91–111, tay phải 112–132.
**Không có** dataset nào dùng OpenPose 137 hay MediaPipe 543.

| Dataset | Thư mục keypoint | Số file | Ghi chú |
|---|---|---|---|
| ASL Citizen | `/home/dongvk/datasets/ASL_Citizen/keypoints/<stem>.pkl` | 83 399 (= số video) | stem = tên video |
| WLASL2000 | `/home/dongvk/datasets/WLASL2000/keypoint/<video_id>.pkl` | 21 095 | toạ độ theo khung 256×256 |
| MS-ASL | `/home/dongvk/datasets/MS-ASL/keypoint/<NNNNNN>.pkl` | 18 855 | id = vị trí trong `msasl1k.json` cộng 1 (train+val+test nối nhau, đã kiểm 0 lệch) |
| How2Sign | `/home/shared_data/sign_language/How2Sign/{train,eval,test}/*_pose/*.pkl` | 31 047 / 1 739 / 2 343 | train thiếu 1 pose so với video |
| CSL-Daily | `.../CSL_Daily_Sentence_Crop/keypoint/*.pkl` | 20 654 (= nhãn; video có 20 668) | |
| CSL-Daily 77 | `.../keypoint_focus_hand_w_face/*.pkl` | 20 654 | `{'keypoint': (1,T,77,2), 'keypoint_score': (1,T,77), 'total_frames'}`; **score là hằng số 0,9** (đã mất confidence) |
| Phoenix-14T | `.../PHOENIX-2014-T/output/keypoint/*.pkl` (+ `keypoint_focus_hand_w_face`) | 8 257 | |

Bản 77 điểm (đã map ngược về 133, sai số 0) gồm body [5–10] (6 điểm), tay trái [91–111], tay phải [112–132], đường viền hàm [23–39] (17 điểm) và môi ngoài [71–82] (12 điểm).

Có ít frame nhiều người. Trong mẫu 200 clip, số clip có ≥1 frame mà mảng có `shape[0]>1`: WLASL 4, MS-ASL 19, CSL 1, các dataset khác 0. Loader phải lấy `[0]` hoặc chọn người ở giữa khung.

Snippet load (dùng chung cho mọi dataset 133 điểm):

```python
import pickle, numpy as np
def load_coco133_pkl(path, W, H):
    d = pickle.load(open(path, "rb"))
    xy = np.stack([k[0] if k.ndim == 3 else k for k in d["keypoints"]]).astype(np.float32)  # [T,133,2] chuẩn hoá
    sc = np.stack([s[0] if s.ndim == 2 else s for s in d["scores"]]).astype(np.float32)     # [T,133] SimCC thô
    xy_px = xy * np.array([W, H], np.float32)  # về pixel trước khi chuẩn hoá theo vai (tránh méo trục 16:9, 4:3)
    return xy_px, sc
# W,H: ASLC lấy từ metadata cache; WLASL 256x256; Phoenix 210x260; MS-ASL/How2Sign/CSL dùng ffprobe theo clip
```

**Dữ liệu lỗi cần xử lý:** các frame không phát hiện được người có `score` toàn 0. Ở ASL Citizen, những frame này chiếm 2,3 % (cache đánh dấu `frame_valid=False`).

## 4. ASL Citizen — cache pose đã xử lý (`method/asm_jepa/data/processed/aslc`)

- **8,1 GB.** Mỗi clip là một file `.npz` (`poses/<sample_id>.npz`) và đủ 83 399 clip. `unreadable_videos.jsonl` rỗng. Import summary báo `ok` cho toàn bộ. `dataset_version.json` khớp v1.0, có sha256 của CSV.
- Các key: `xy` float32 [T,133,2] **pixel**, gốc trên-trái; `score` float32 [T,133] SimCC thô; `observed` bool [T,133]; `timestamps_sec` (CFR = index/fps container); `frame_valid`; `image_wh` (640,480); `source_frame_index`; `meta_json` (provenance gồm sha256 pickle nguồn, schema `coco_wholebody_133_raw`, không nội suy, không làm mượt).
- Metadata: `metadata_{split}.jsonl` (sample_id, signer_id, gloss, class_id, duration_sec, fps, num_frames, width, height). Ngoài ra có `poses_index_*.jsonl`, `ssl_train.jsonl` (40 154 clip, chỉ train), `vocab.json` và `stats/train_v{1,2}.npz`.
- Loader `asm_jepa/asm_jepa/data/pose_store.py` có `read_clip(path)` trả về `PoseClip` và kiểm tra contract.
  `schema.py` (`coco133_to_aslc114_v1`) chọn 114 joint: vai và khuỷu [5–8], hai bàn tay 42 điểm, face 68.

```python
from asm_jepa.data.pose_store import read_clip
from asm_jepa.data.schema import raw_to_model
c = read_clip(".../processed/aslc/poses/a9f512bdb0be4a75983f64bc.npz")
xy, sc, ob = raw_to_model(c.xy, c.score, c.observed)   # [T,114,*]
```

**Phân bố T** (toàn bộ 83 399): p0 3, p1 32, p5 42, p25 58, **p50 75**, p75 96, p95 158, p99 211, max 680, mean 82,8.
Có 0,94 % clip có T<32 và 0,12 % clip có T>300. Ở mức median, T gần như bằng nhau giữa các split (75/74/74).

**Score tay** (mẫu 600 clip ≈ 49,6 k frame; số đo trên từng joint):

| Nhóm joint | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p99 |
|---|---|---|---|---|---|---|---|---|
| Tay (91–132) | 0 | 0,89 | 1,12 | 1,85 | **3,23** | 4,84 | 6,47 | 8,31 |
| Body 0–10 | 0 | 2,31 | 3,49 | 5,48 | 7,20 | 8,35 | 8,98 | 9,91 |
| Face 23–90 | 0 | 6,08 | 6,72 | 7,56 | 8,32 | 8,81 | 9,12 | 9,55 |

Tỷ lệ frame có score trung bình của một bàn tay dưới ngưỡng:

| Ngưỡng (mean score 21 điểm) | Tay trái thấp | Tay phải thấp | Cả hai thấp | Chỉ tính frame tay nằm trong khung: trái / phải |
|---|---|---|---|---|
| < 2 | 30,2 % | 25,6 % | 17,5 % | 22,4 % / 13,8 % |
| **< 3** (đề xuất) | 48,2 % | 42,3 % | 32,0 % | 43,9 % / 29,0 % |
| < 4 | 64,0 % | 58,4 % | 47,8 % | — |

- Tay chiếm >50 % điểm nằm ngoài khung (thường là tay buông xuống dưới mép ảnh 480 px): trái 12,5 %, phải 8,6 % số frame.
- **Lỗi aliasing tay:** frame có centroid tay trái và tay phải cách nhau < 0,15 × độ rộng vai chiếm 17,3 %, và 78,5 % clip có ít nhất một frame như vậy. Con số cao hơn mức ~14 % ghi trong memory vì tiêu chí khác. Đây là nhiễu lớn nhất của tay.
- **Cảnh báo:** cache dùng `confidence_threshold=0.2` để tạo `observed`. Với thang SimCC, ngưỡng này gần như luôn bị vượt: `observed` = `frame_valid` (97,7 %), nên `observed` **không** phản ánh tay mờ hoặc khuất.
  Proposal16 nên dùng một ngưỡng tương đối, ví dụ ≈3 (gần median score tay), hoặc dùng score như một feature liên tục, đồng thời thêm cờ alias.
- Aspect ratio: 96,1 % là 4:3 (640×480) và 3,9 % là 16:9 (960×540). Có 4 clip dọc 480×640, nên phải chuẩn hoá theo pixel và vai, không theo [0,1].

**Cột `ASL-LEX Code`** trong `splits/*.csv`: điền **100 %** hàng. Dạng `[A-K]_0[1-3]_NNN` (ví dụ `A_03_054`), có 2 723 mã khác nhau cho 2 731 gloss.
Mỗi gloss chỉ có một mã. 6 gloss (183 clip) mang mã `NA`: SENTENCE1, MINIMUM1, INSTINCT, ABOVE, LETTER2, WELCOME1.
Có 2 mã dùng chung nhiều gloss biến thể: `C_02_054` cho WHATFOR1/2/3 và `B_03_084` cho RESEARCH1/2.
Như vậy có thể join gần như toàn bộ vocab với ASL-LEX 2.0 để lấy nhãn phonology (handshape, location, movement…), một nguồn giám sát yếu hoặc probe rất có giá trị.

**Split và signer:** train 35, val 6 (P5, P12, P13, P21, P26, P39), test 11 signer, **giao nhau = 0**. Cả 3 split dùng cùng 2 731 gloss.
Mỗi signer train có từ 3 đến 3 000 clip (rất mất cân bằng).

## 5. Overlap và rủi ro nhiễm test (quan trọng cho SSL pretraining)

| Cặp | Kết quả | Hệ quả |
|---|---|---|
| WLASL2000 ↔ MS-ASL (YouTube ID) | 3 250 video YT trong WLASL2000 và 7 212 video YT trong MS-ASL có **1 579** video chung (686 video có trong bản MS-ASL local). **112 / 2 879 clip test WLASL** có video nguồn nằm trong MS-ASL local (104 thuộc MS-ASL train). Ngược lại, 0 clip test MS-ASL local có nguồn trong WLASL2000. | Khi pretrain trên MS-ASL rồi eval WLASL, phải loại các clip MS-ASL có YouTube ID trùng với WLASL val/test, hoặc báo cáo kèm số liệu đã lọc. |
| YouTube-ASL ↔ MS-ASL | 111 video YT chung. **1 599 / 2 847 clip test MS-ASL local (56 %)** đến từ 85 video có trong danh sách YouTube-ASL (482 clip train, 108 clip val). Ở MS-ASL100 test: 296/522; MS-ASL200: 541/942. | **Nhiễm nặng:** nếu sau này pretrain trên YouTube-ASL thì phải blacklist 85 video này (hoặc toàn bộ 111) trước khi eval MS-ASL. |
| YouTube-ASL ↔ WLASL2000 / How2Sign | 0 / 0 | An toàn |
| How2Sign ↔ MS-ASL | 0 video chung. Các YT ID của How2Sign train/val/test rời nhau. | An toàn |
| WLASL signer | 119 signer, có 104 signer vừa ở train vừa ở test | WLASL là benchmark **signer-dependent** |
| MS-ASL signer | rời nhau giữa train/val/test | Signer-independent |
| ASL Citizen signer | rời nhau (0/0/0) | Signer-independent. Nguồn webcam riêng, không trùng video YouTube. |
| CSL-Daily | 10 signer ở cả 3 split. Câu train rời với dev/test; dev và test chung 795 sentence ID. | Signer-dependent |
| Phoenix-14T | 9 signer ở cả 3 split | Signer-dependent |
| Vocab ASLC ↔ WLASL / MS-ASL | 2 301 gloss gốc ASLC (đã bỏ hậu tố số): 1 362 trùng WLASL, 778 trùng MS-ASL. WLASL2000 ∩ MS-ASL1000: 856 / 1 056 tên. | Có thể làm eval cross-dataset với lớp chung, nhưng phải đối chiếu biến thể và từ đồng nghĩa. |

Quy tắc đề xuất: SSL corpus **không** chứa val/test của bất kỳ dataset nào dùng để eval.
Với ASL Citizen, nếu muốn pretrain trên cả val/test (transductive), phải ghi rõ trong báo cáo; mặc định nên chỉ dùng `ssl_train.jsonl` (train).

## 6. Chất lượng tay giữa các dataset (mẫu 100 clip/dataset, cùng tiêu chí)

| Dataset | Score tay p10 / p50 / p90 | Frame có tay trái < 3 | Frame có tay phải < 3 | Alias L/R | Tay > 50 % ngoài khung | T median |
|---|---|---|---|---|---|---|
| ASL Citizen | 1,27 / 3,23 / 6,60 | 49,8 % | 40,5 % | 17,4 % | 6,8 % | 70 |
| WLASL2000 | 1,16 / 4,21 / 7,14 | 38,7 % | 21,7 % | 10,1 % | 4,2 % | 56 |
| MS-ASL | 2,00 / 5,24 / 8,05 | 23,2 % | 8,4 % | 15,3 % | 4,8 % | 81 |
| How2Sign | 4,35 / 7,04 / 8,80 | 0,5 % | 0,2 % | 3,6 % | 1,4 % | 115 |
| Phoenix-14T | 2,89 / 4,44 / 6,29 | 14,0 % | 5,0 % | 1,4 % | 0 % | 100 |
| CSL-Daily | 2,84 / 5,94 / 8,62 | 15,3 % | 5,6 % | 3,5 % | 22,1 % | 160 |

So với các dataset còn lại, ASL Citizen (webcam 640×480, người dùng tự quay) có tay kém nhất.
How2Sign (studio, 720p) có tay sạch nhất, nên có thể dùng How2Sign làm phần "sạch" của corpus SSL.
Phoenix có độ phân giải rất thấp (210×260), nên score tay thấp nhưng ổn định.

## 7. Dung lượng đĩa (du -sh)

| Đường dẫn | Kích thước |
|---|---|
| ASL_Citizen/videos · keypoints | 47 G · 19 G |
| asm_jepa processed/aslc (npz) | 8,1 G |
| WLASL2000 video · keypoint | 2,0 G · 3,4 G |
| MS-ASL video gốc · cut · keypoint · msasl.tar | 19 G · 13 G · 4,7 G · 6,6 G |
| How2Sign train (gồm zip) · eval · test | 78 G · 2,4 G · 5,5 G |
| CSL_Daily_Sentence_Crop videos · keypoint · keypoint77 | 28 G · 8,7 G · 5,9 G |
| shared CSLDaily (bản trùng) | 28 G |
| phoenix14T (toàn bộ, gồm frame PNG) | 48 G |
| youtubeASL | 136 K (chỉ có file ID) |

Đĩa `/` mới **dùng 98 %, chỉ còn trống 94 GB**. Mọi cache mới phải nhỏ, ví dụ FP16 hoặc subset joint.

## 8. YouTube-ASL

- `/home/dongvk/datasets/youtubeASL/youtube-asl_youtube_asl_video_ids.txt` có **11 095 ID** (chưa có video hay pose nào).
- Theo `docs/experiments/youtube_pose_resource_estimate_20261016/REPORT_VI.md`: RTMW-x với profile `limited_workspace` đạt 9,6 fps (1 worker) đến 41 fps (4 worker, cộng dồn) trên GPU dùng chung, chưa tính decode và tracking.
  Lưu trữ khi chưa nén cho 3 000 h ở 25 fps: COCO133 FP32 cần 401 GiB, FP16 cần 201 GiB, PLctx103 (xy FP32 + conf FP16) cần 259 GiB.
- Report dùng kịch bản 3 000–3 207 h (lớn hơn con số ≈ 984 h của paper YouTube-ASL; chưa đối chiếu nguồn số giờ). Hệ quả: với 94 GB trống, **không thể** đưa YouTube-ASL đầy đủ vào corpus hiện nay. Chỉ có thể dùng một subset (≈ 1 000 h ở FP16 và joint subset) sau khi giải phóng hoặc bổ sung đĩa.
  Nếu ước tính 3 000 h × 25 fps với 4 worker ở 41 fps, riêng phần pose mất ≈ 76 ngày GPU. Ngoài ra còn bài toán nhiều người và chọn signer, chưa được benchmark.
- Trước khi dùng phải blacklist 111 video trùng MS-ASL (xem §5).

## 9. Khuyến nghị

| Dataset | (a) Corpus SSL skeleton | (b) Eval ISLR | (c) Eval retrieval | Tiền xử lý cần làm |
|---|---|---|---|---|
| ASL Citizen | **Có, ngay** (train 40 k clip, 32 h; cache npz sẵn) | **Có, chính**: signer-independent, 2 731 lớp, top-1/top-5 trên test 11 signer | **Có**: query/gallery theo signer test; dùng mã ASL-LEX để chấm retrieval theo phonology | Thêm cờ alias L/R và ngưỡng score tương đối; chuẩn hoá pixel theo vai; xử lý 4 clip dọc |
| WLASL2000 | Có (train 14,3 k), sau khi lọc YT trùng | Có, nhưng signer-dependent: báo cáo kèm cảnh báo; có subset 100/300/1000 | Có (cross-dataset với ASLC qua 1 362 gloss chung) | Toạ độ theo khung 256×256; loại 112 clip test có nguồn trùng MS-ASL khi pretrain MS-ASL |
| MS-ASL (local 74 %) | Có (train 12,7 k), sau khi loại các video YT trùng WLASL val/test | Có (signer-independent), nhưng chỉ trên bản local 74 % nên **không so trực tiếp với số trong paper** | Tuỳ chọn | Dùng T của pickle, bỏ header; nhân W,H theo từng clip (độ phân giải hỗn hợp); chọn người khi frame có nhiều người |
| How2Sign | **Có, nên dùng** (≈ 55 h train, tay sạch nhất) | Không (câu liên tục, không có gloss) | Không (chỉ retrieval câu–văn bản, ngoài phạm vi) | Clip dài (p95 15 s): cần crop cửa sổ; không có signer ID nên không kiểm soát được signer |
| CSL-Daily | Tuỳ chọn (31 h, ngôn ngữ CSL, khác ASL) | Không cho ISLR ASL | Không | Dùng bản 133 điểm (bản 77 điểm mất score). Không dùng thư mục shared trùng. |
| Phoenix-14T | Tuỳ chọn (10,5 h DGS, 210×260) | Không | Không | Độ phân giải thấp, domain TV thời tiết |
| YouTube-ASL | **Chưa** (0 video, thiếu đĩa) | — | — | Tải, trích pose và blacklist 111 video trùng MS-ASL |

**Corpus SSL đề xuất có thể dùng ngay** (chỉ train split): ASLC train (32 h) + WLASL train (≈ 9,5 h) + MS-ASL train (≈ 10,8 h) + How2Sign train (≈ 53 h) ≈ **105 h ASL**.
Có thể thêm CSL-Daily và Phoenix train (≈ 37 h, thuộc ngôn ngữ ký hiệu khác) cho một ablation đa ngôn ngữ.
Mọi dataset dùng chung một extractor và layout 133, nên **không cần chuyển định dạng giữa các schema**. Chỉ cần thống nhất 4 bước:
1. Đưa toạ độ chuẩn hoá về pixel (nhân W,H của clip), rồi chuẩn hoá theo tâm và độ rộng vai. Làm vậy để tránh méo do aspect ratio khác nhau: 4:3, 16:9, 1:1, 210×260.
2. Dùng **subset joint chung**. Đề xuất mặc định gồm body 6 [5–10], hai bàn tay 42 [91–132] và face chọn lọc ≈ 30 điểm (lông mày 40–49, mũi 50–53 hoặc chỉ đỉnh mũi 53, môi 71–90), tổng ≈ 78–80 joint.
   Hai lựa chọn khác: subset 77 của Uni-Sign (có cổ tay body, viền hàm, môi ngoài) hoặc `aslc114` (đủ 68 điểm face) để so sánh với ASM-JEPA.
   Bỏ mũi, mắt, tai của body (0–4) vì trùng với face, và bỏ bàn chân.
3. Chuẩn hoá confidence: xem SimCC thô là feature liên tục (chia 10). Dùng mask "low-conf" với ngưỡng ≈ 3 cho tay, và cờ alias L/R (khoảng cách centroid < 0,15 × độ rộng vai).
   Frame có score toàn 0 thì đánh dấu là invalid.
4. Resample về fps chung (ví dụ 25 hoặc 30) dựa trên fps thật từng clip, vì ASLC có clip 11–120 fps. Đặt giới hạn T hoặc dùng cửa sổ cho How2Sign và CSL.
   Lưu cache FP16 với subset joint: ≈ 105 h × 25 fps × ~80 joint × 3 kênh × 2 B ≈ 4,5 GB, vừa với đĩa hiện có.
