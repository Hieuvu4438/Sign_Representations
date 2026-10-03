# Các dataset sign language trên máy local

Kiểm tra filesystem ngày **2026-10-03**. Tài liệu này ghi nhận dữ liệu hiện có
trong `/home/shared_data/sign_language/` và `/home/dongvk/datasets/`;
không khẳng định các bản local đầy đủ như bản phát hành gốc.
Các đường dẫn tuyệt đối bên dưới dùng trên máy hiện tại, không nằm trong repository.

## Tổng quan

| Dataset / bản local | Thư mục gốc | Nội dung quan sát được |
| --- | --- | --- |
| CSLDaily | `/home/shared_data/sign_language/CSLDaily/` | Video và nhãn train/dev/test |
| How2Sign | `/home/shared_data/sign_language/How2Sign/` | Video, pose và nhãn theo train/eval/test |
| CSL Daily Sentence Crop | `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/` | Video, keypoint, nhãn và metadata |
| ASL Citizen | `/home/dongvk/datasets/ASL_Citizen/` | Video, keypoint, split CSV và metadata |
| MS-ASL | `/home/dongvk/datasets/MS-ASL/` | Video tải về, video đã xử lý, keypoint và metadata |
| WLASL2000 | `/home/dongvk/datasets/WLASL2000/` | Video MP4, keypoint và metadata |
| Phoenix14T | `/home/dongvk/datasets/phoenix14T/` | Video theo split và thư mục bản release PHOENIX-2014-T |
| How2Sign — nhãn bổ sung | `/home/dongvk/datasets/How2Sign/` | Nhãn và script hỗ trợ; không thấy thư mục video tại đây |
| youtubeASL | `/home/dongvk/datasets/youtubeASL/` | Chỉ thấy danh sách video ID |

## CSLDaily

Bản trong shared_data chứa video và ba file nhãn:

- Video: `/home/shared_data/sign_language/CSLDaily/videos/videos/`
- Nhãn train: `/home/shared_data/sign_language/CSLDaily/labels.train`
- Nhãn dev: `/home/shared_data/sign_language/CSLDaily/labels.dev`
- Nhãn test: `/home/shared_data/sign_language/CSLDaily/labels.test`

Bản có tên `CSL_Daily_Sentence_Crop` nằm riêng trong thư mục của dongvk,
chứa video cùng hai biến thể keypoint và metadata số frame:

- Video: `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/videos/`
- Keypoint: `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/keypoint/`
- Keypoint tay và mặt: `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/keypoint_focus_hand_w_face/`
- Nhãn: `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/labels.train`, `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/labels.dev`, `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/labels.test`
- Metadata: `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/metadata_meaning_frames.json`
- CSV số frame: `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/train_data_with_num_frames.csv`, `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/dev_data_with_num_frames.csv`, `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/test_data_with_num_frames.csv`

Chưa đối chiếu nội dung video giữa hai bản CSLDaily.

## How2Sign

Dữ liệu video chính nằm trong shared_data. Split validation trên máy dùng tên `eval`.

| Split local | Video | Pose | Nhãn |
| --- | --- | --- | --- |
| train | `/home/shared_data/sign_language/How2Sign/train/raw_videos/` | `/home/shared_data/sign_language/How2Sign/train/train_pose/` | `/home/shared_data/sign_language/How2Sign/train/train_label/` |
| eval | `/home/shared_data/sign_language/How2Sign/eval/raw_videos/` | `/home/shared_data/sign_language/How2Sign/eval/eval_pose/` | `/home/shared_data/sign_language/How2Sign/eval/eval_label/` |
| test | `/home/shared_data/sign_language/How2Sign/test/raw_videos/` | `/home/shared_data/sign_language/How2Sign/test/test_pose/` | `/home/shared_data/sign_language/How2Sign/test/test_label/` |

- Subset local: `/home/shared_data/sign_language/How2Sign/train/subset_2000/`
- CSV validation realigned: `/home/shared_data/sign_language/How2Sign/eval/how2sign_realigned_val.csv`
- Nhãn bổ sung: `/home/dongvk/datasets/How2Sign/labels.train`
- Nhãn từ nguồn Uni-Sign: `/home/dongvk/datasets/How2Sign/from_uni_sign_source/labels.train`, `/home/dongvk/datasets/How2Sign/from_uni_sign_source/labels.test`
- Script hỗ trợ: `/home/dongvk/datasets/How2Sign/draft.py`

## ASL Citizen

Bản local có video, keypoint và split CSV riêng:

- Video: `/home/dongvk/datasets/ASL_Citizen/videos/`
- Keypoint: `/home/dongvk/datasets/ASL_Citizen/keypoints/`
- Split train: `/home/dongvk/datasets/ASL_Citizen/splits/train.csv`
- Split validation: `/home/dongvk/datasets/ASL_Citizen/splits/val.csv`
- Split test: `/home/dongvk/datasets/ASL_Citizen/splits/test.csv`
- Metadata tổng hợp: `/home/dongvk/datasets/ASL_Citizen/asl_citizen_all_in_one.json`
- Ánh xạ nhãn: `/home/dongvk/datasets/ASL_Citizen/label_maps.json`
- Ghi chú sử dụng local: `/home/dongvk/datasets/ASL_Citizen/use.txt`

## MS-ASL

Bản local có cả thư mục video tải về và thư mục video đã xử lý:

- Video: `/home/dongvk/datasets/MS-ASL/video/`
- Video xử lý/cắt: `/home/dongvk/datasets/MS-ASL/cut_processed_video/`
- Keypoint: `/home/dongvk/datasets/MS-ASL/keypoint/`
- Nhãn split: `/home/dongvk/datasets/MS-ASL/MSASL_train.json`, `/home/dongvk/datasets/MS-ASL/MSASL_val.json`, `/home/dongvk/datasets/MS-ASL/MSASL_test.json`
- Lớp và từ đồng nghĩa: `/home/dongvk/datasets/MS-ASL/MSASL_classes.json`, `/home/dongvk/datasets/MS-ASL/MSASL_synonym.json`
- Metadata local: `/home/dongvk/datasets/MS-ASL/metadata_msasl_like_wsasl.json`
- Biến thể metadata: `/home/dongvk/datasets/MS-ASL/metadata_msasl_like_wsasl_100.json`, `/home/dongvk/datasets/MS-ASL/metadata_msasl_like_wsasl_200.json`, `/home/dongvk/datasets/MS-ASL/msasl1k.json`
- Script tải: `/home/dongvk/datasets/MS-ASL/src_download/`

Tên file `wsasl` được giữ đúng theo filesystem.

## WLASL2000

Video MP4 nằm trong thư mục con trùng tên dataset:

- Video: `/home/dongvk/datasets/WLASL2000/WLASL2000/`
- Keypoint: `/home/dongvk/datasets/WLASL2000/keypoint/`
- Metadata: `/home/dongvk/datasets/WLASL2000/metadata_meaning_frames.json`
- Script preprocessing: `/home/dongvk/datasets/WLASL2000/preprocess/`

## Phoenix14T

Video local được chia theo train/dev/test:

- Train: `/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/train/`
- Dev: `/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/dev/`
- Test: `/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/test/`
- Annotation tổng hợp: `/home/dongvk/datasets/phoenix14T/phoenix14t.pami0.train.annotations_only.gzip`, `/home/dongvk/datasets/phoenix14T/phoenix14t.pami0.dev.annotations_only.gzip`, `/home/dongvk/datasets/phoenix14T/phoenix14t.pami0.test.annotations_only.gzip`

Thư mục release: `/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/`.
Các đường dẫn bên trong release:

- Annotation: `/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/`
- Frame/features: `/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/`
- Video output: `/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/output/videos/`
- Keypoint output: `/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/output/keypoint/`
- Keypoint tay và mặt: `/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/output/keypoint_focus_hand_w_face/`

## youtubeASL và thư mục hỗ trợ

`/home/dongvk/datasets/youtubeASL/` hiện chỉ có:

- `/home/dongvk/datasets/youtubeASL/youtube-asl_youtube_asl_video_ids.txt`

Chưa thấy video tải về trong thư mục này.
`/home/dongvk/datasets/script_data_exploring/` chứa các thư mục hỗ trợ
`CSL/`, `NMF_CSL/`, `How2Sign/`, `Phoenix14T/`; không tính là dataset video trong danh sách trên.
`places365`, `imagenet1k` và `outlier_dataset` không nằm trong danh sách sign language này.

## Quy tắc lưu vào Git

Chỉ lưu tài liệu, mã nguồn và metadata cần thiết. Video, checkpoint, feature cache,
archive tải về và manifest toàn bộ corpus được loại qua `.gitignore`.
Dữ liệu thực vẫn ở các đường dẫn ngoài repository nêu trên; tài liệu này không sao chép chúng vào Git.
