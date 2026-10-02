# Kiểm tra khả năng reproduce SignRep và SHuBERT — 2026-10-02

**Kết luận: đủ checkpoint đã đăng ký để chạy feature inference của cả hai backbone trên máy hiện tại; chưa đủ để reproduce toàn bộ training và benchmark của hai paper.** Đủ checkpoint inference không đồng nghĩa đủ code, dữ liệu, trạng thái training hay protocol đánh giá gốc.

## Kiểm tra thực hiện

- Commit nguồn local của SignRep, SHuBERT và DINOv2 khớp `provenance/source_commits.json`, không có thay đổi tracked trong các checkout này. GitHub API xác nhận HEAD hiện tại của hai repo chính vẫn trùng commit đã pin.
- W01–W08, sáu file config/tokenizer W08_CONFIG và các file source demo C04 đều khớp SHA256 đã đăng ký. Chi tiết: [reproduction_inventory_20261002.json](reproduction_inventory_20261002.json).
- Chạy lại strict load SignRep trên CPU: không thiếu/thừa key, projection head có mặt; [load audit](signrep_checkpoint_load_audit_20261002.json).
- Chạy lại encoder SHuBERT và hai DINO trên CPU: strict load không thiếu/thừa key; input synthetic 15/16 frame, deterministic difference bằng 0 và từ chối stream không đồng bộ. [Smoke audit](shubert_encoder_load_audit_20261002.json). Đây là kiểm tra interface, không phải benchmark video.
- Chạy lại `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv-signrep/bin/python -m unittest discover -s tests -v`: **65/65 PASS**. [Bằng chứng test](codebase_review_tests_20261002.json). Các test này không chứng minh reproduce paper hoặc độ chính xác model.
- Rà adapter, preprocessing, configs, cache/resume, probe và các entry point so với code gốc. Không chạy lại full training, full video extraction hoặc benchmark trong lần kiểm tra này.

## Checkpoint hiện có và phần còn thiếu

| Thành phần | Hiện có local | Đánh giá |
|---|---|---|
| SignRep final pretrained model | W01 `ckpt.pt`, 222,676,746 bytes | Đủ cho kiến trúc và example feature extraction được public. API release hiện chỉ liệt kê asset checkpoint này. |
| SHuBERT native pretrained encoder | W02 `checkpoint_836_400000.pt`, 1,057,496,908 bytes | Đủ strict-loaded encoder inference. Chưa kiểm tra khả năng resume optimizer/scheduler từ payload này. |
| DINO face/hand của tác giả | W03/W04, mỗi file 112,850,895 bytes | Đủ feature frontend; đã strict load với kiến trúc DINOv2 đã pin. |
| MediaPipe face/hand và YOLO | W05/W06/W07 | Đủ bộ model phụ trợ dùng trong raw-video pipeline đã đăng ký. |
| Translator demo SHuBERT | W08 `checkpoint-11625/pytorch_model.bin`, 2,683,976,334 bytes; configs và ByT5 tokenizer | Đủ đường inference đang dùng; numerical smoke lịch sử đã pass. Chưa xác nhận tương ứng checkpoint benchmark paper. |
| ByT5 ban đầu | Thiếu `models/byt5_base/pytorch_model.bin`, remote 2,326,696,954 bytes | Không cần tải riêng để strict-load W08 đầy đủ; cần nếu tái lập đường khởi tạo từ ByT5 ban đầu. |
| Trạng thái training translator | Thiếu `optimizer.pt` (5,361,549,242 bytes), `scheduler.pt`, `trainer_state.json`, `training_args.bin`, tám `rng_state_*.pth` | Có trên HF nhưng chưa tải/đăng ký. Cần kiểm tra đầy đủ nếu muốn resume training demo; chỉ tải các file này vẫn chưa cung cấp data/recipe gốc. |
| Checkpoint downstream paper | Chưa có mapping được xác minh cho translation/isolated sign/fingerspelling của SHuBERT và các evaluation task của SignRep | Không thể gọi W01/W02/W08 là bộ checkpoint cho tất cả task/ablation. |

Danh sách remote và trạng thái tồn tại của từng file được lưu trong JSON inventory; không tải thêm nhiều GB vì yêu cầu lần này là kiểm tra. Model card HF của SHuBERT trống, nên không cung cấp bằng chứng mapping demo checkpoint tới bảng kết quả paper.

## Các khoảng trống ngoài checkpoint

**SignRep:** repo public chứa model/head, augmentation và example feature extraction. Không có bộ runner pretraining, downstream fine-tuning/evaluation, split và recipe đầy đủ cho mọi benchmark paper. Adapter local dùng constructor đọc từ AST của example, strict load và kiểm tra transform equivalence; cấu hình 16 frame/stride 2 phù hợp example. Với clip dưới 16 frame, adapter local pad frame cuối còn example gốc bỏ clip; protocol probe local đã ghi rõ và loại trường hợp này. Feature local mặc định lấy `features`, trong khi example còn xuất `latent`. Các probe frozen local không thay thế các task/head/protocol gốc.

**SHuBERT:** `README.md` để downstream fine-tuning là `TODO`; `DATASETS.md` để downstream preparation là `TODO`. Pretraining có launcher/config nhưng còn placeholder cho training manifest và bốn file label k-means face/left hand/right hand/body. `raw_audio_dataset.py` đọc các file này như chuỗi integer label theo từng sample; chúng không phải chỉ là checkpoint hoặc centroid file có thể thay thế tùy ý. Cần exact dữ liệu/preprocessing, loại giao với OpenASL val/test, labels và recipe nếu muốn tái lập training gốc.

Adapter native local giữ output nhánh FFN trong `layer_results`, đúng đường feature extraction của tác giả, và strict load từ cấu hình checkpoint. Tuy nhiên FPS/crop/context và xử lý stream thiếu của protocol local không đủ bằng chứng cho exact downstream recipe paper. W08 chứa encoder embedded khác W02 về giá trị tensor theo audit lịch sử; không được thay thế W08 bằng W02 rồi coi là cùng translator.

**Kết quả hiện tại:** SignRep common cohort hoàn thành 4,726 clip nhưng chỉ là cohort 200 class đã chọn. SHuBERT common extractor vẫn có terminal `FAIL` do disk reserve; preprocessing recovery ghi `RUNNING`, chưa có terminal full PASS. Trạng thái timestamp cũ không chứng minh process hiện còn sống. NCSLGR native 222 clip có PASS nhưng là diagnostic nội bộ, không phải full benchmark gốc.

## Git và khả năng chạy từ clone mới

Đã push commit `637e5c1` chứa code/config/tài liệu/provenance và các textual result snapshot. `.gitignore` loại venv, third_party, pretrained/generated weights, features, video, PDF, archive, logs và full dataset manifests lớn. Các file này vẫn còn trên máy, không bị xóa. Kiểm tra trước commit không phát hiện credential theo các pattern token/private key phổ biến; file staged lớn nhất dưới 5 MB. Whitespace warnings ở dữ liệu CSV CRLF, HTML evidence, Markdown hard breaks và ba dòng trống EOF không ảnh hưởng test; không sửa evidence đã hash chỉ để làm sạch whitespace.

Clone Git **không kèm weights/source bên ngoài/media/cache**. Cần clone đúng commit ở `provenance/source_commits.json`, fetch asset qua `scripts/fetch_asset.py`, chuẩn bị environment, cấu hình lại đường dataset và tạo lại inventory lớn. Các manifest local chứa đường tuyệt đối trên máy nghiên cứu; freeze environment có dependency kế thừa và editable path. Chúng là bằng chứng môi trường đã chạy, không phải installer portable. Các lock/output snapshot đã commit có thể từ chối overwrite; cần namespace/workspace mới theo `REPRODUCE.md`.

## Việc cần có để đạt full reproduction

1. Xác định từng benchmark, bảng kết quả và checkpoint task tương ứng; xin/xác minh runner, config, split và recipe chưa public từ tác giả.
2. Với SHuBERT pretraining: phục hồi exact training corpus/filter/manifest, frontend features và bốn chuỗi k-means labels; kiểm tra training state cần thiết cho resume. Với translator: bổ sung initial ByT5 hoặc resume files tùy mục tiêu và recipe đã xác minh.
3. Thiết lập environment độc lập và dataset path portable trên clone mới; verify asset/source hashes trước chạy.
4. Chạy đầy đủ task gốc với metric, seeds và selection protocol gốc, rồi đối chiếu kết quả với paper. Hoàn thành local common extraction chỉ giải quyết local diagnostic, không đủ cho bước này.

## Nguồn đối chiếu

- [SignRep source](https://github.com/ryanwongsa/SignRep), [release v0.0.1](https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1), [release API](https://api.github.com/repos/ryanwongsa/SignRep/releases).
- [SHuBERT source](https://github.com/ShesterG/SHuBERT), [quickstart](https://github.com/ShesterG/SHuBERT/blob/main/QUICKSTART.md), [dataset preparation](https://github.com/ShesterG/SHuBERT/blob/main/DATASETS.md), [training config](https://github.com/ShesterG/SHuBERT/blob/main/fairseq/examples/shubert/config/base_random.yaml).
- [SHuBERT HF model](https://huggingface.co/ShesterG/SHuBERT), [model file listing API](https://huggingface.co/api/models/ShesterG/SHuBERT/tree/main/models?recursive=true&limit=1000), [author demo](https://huggingface.co/spaces/ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText).
