# Báo cáo implementation — vòng 1, 12/10/2026

**Bản lịch sử. Trạng thái mới nhất: [báo cáo vòng 2](IMPLEMENTATION_PHASE2_VI.md).**

Mục tiêu toàn bộ proposal **chưa hoàn thành**. Vòng này xây dựng và kiểm chứng nền tảng;
không thu nhỏ Definition of Done của proposal thành preprocessing hay synthetic tests.
Không có training score, downstream prediction hoặc claim reproduction.

## Kết quả có bằng chứng

| Nhóm | Code đã có | Evidence đã chạy | Phần chưa đủ để đóng milestone |
| --- | --- | --- | --- |
| M0 | Installable package; strict config inheritance; doctor; base/smoke/E01–E04 configs; source/checkpoint registries; asset downloader | Editable installation thành công; config tests; doctor; SHA-256 source snapshot | Clean-room environment và smoke CLI training; command-specific asset validation toàn pipeline |
| M1 | Actual ASL Citizen CSV adapter; local JSONL; availability/hashes; train-only vocabulary; label-free pretrain view; exact split audit | Adapter tests với đúng CSV columns; duplicate/source/hash/signer tests; đã đọc CSV local thật | Full corpus import/audit; near-duplicate fingerprints/candidate review; source sampler/collation |
| M2 | PTS decode/VFR nearest sampling; aliases; 49-joint mapping; raw candidate retention; wrist matching; body coords; ROI/affine; IMAGE-mode detector | Container VFR lossless thật; missing shoulders/aspect ratio; một full ASL Citizen clip; repeated-frame independence | Corpus ledger/shards/resume; CLI orchestration; corpus-scale handedness/quality review |
| M3 | DINOv2 pretrained strict load, per-frame 4 streams, FeatureBundle cache | Pinned source `7764ea0…`; thật weights SHA; cache roundtrip/tamper; full-clip DINO inference | Frozen visual probe và validation protocol |
| M4 | Default 11 paths; physical-time windows; NumPy/Torch degree2; anchors/timing; ordinary-motion; train-only q32 projectors | Analytical tests; float64/32/16 Torch parity/gradient; windows/missing/zero-dt; target fit/reload/held-out rejection | Target fitting CLI; train-pool sampling; integration vào SSL engine |
| M5 | Typed raw dependencies; tokenizer static pose; synchronized alias-aware sampler; closure/strict eligibility/counters | Shoulder/group closure; alias test; fixed-mask hidden-value counterfactual; missing-target policy | Retry/min-context logic trong trainer; anatomical/stream/modality strategies; actual pretrained counterfactual gate |
| M6–M10 | Chưa triển khai training/downstream/research runtime | Chưa chạy | Toàn bộ gates liên quan vẫn mở |

Validation hiện tại: **59 tests passed**, trên CPU, thời gian pytest khoảng 1.6 s.
Test video container cần PyAV; các test ROI cần OpenCV; môi trường hiện tại có cả hai,
không có skipped test trong lần ghi nhận. Geometry Torch backend được kiểm tra gradient,
nhưng chưa có SSL optimizer-step test, AMP skipped-step, DDP hoặc resume test.

## Review code từng giai đoạn

Review do cùng coding agent thực hiện, bằng cách đọc lại contracts/code và chạy tests;
**không phải independent reviewer**. Không spawn sub-agent ở vòng này.

1. **M0/config/assets:** Reject duplicate keys, unknown keys, cycles, wrong type/dim,
   hidden query values, held-out target stats và random fallback. Sửa validator để
   reject unsupported enum/fixed contracts thay vì chỉ check type. Downloader test
   reject HTML, checksum mismatch và file quá ngắn; không publish `.part` lỗi.
2. **M1/data:** Pretraining view không truyền metadata chứa nhãn. Audit verify source
   file content, không chỉ tin filename/hash khai báo. Empty manifest bị fail. Missing
   signer/hash làm gate fail khi protocol yêu cầu; near-duplicates ghi rõ chưa chạy.
3. **M2/pose/ROI:** Check upper7 MediaPipe indices thay vì seven-first landmarks;
   body and hand wrists riêng. Quality handedness không copy làm localization confidence.
   Invalid shoulders làm normalization invalid và model coordinates/quality zero.
   Real diagnostic tìm ra lỗi yêu cầu **cả hai** cổ tay phải reliable để map một tay;
   sửa thành check reliability của wrist được chọn và giữ ambiguity/distance gates.
   Regression test có crossed hands và unrelated missing wrist.
4. **M3/cache/backbone:** DINO dùng pinned local source, `pretrained=False` để construct,
   sau đó strict load **weights thật** trước inference. Reader yêu cầu expected identity,
   checksum mọi array, support tối thiểu RGB/normalization và aliases nhất quán. Sửa extractor
   hash để bao gồm code SHA của adapter/preprocessing; code đổi thì cache identity đổi.
5. **M4/geometry/projection:** Oracle Chen độc lập với Torch area accumulation. Kiểm tra
   L-path area +0.5, straight area0, reversal, affine determinant, scaling và constant
   static collision. Empty/invalid windows không pad bằng zero. Projection kiểm tra
   registered train sources, native/output dimensions, normalization và identity khi reload.
6. **M5/masking:** Closure dùng source primitive sets; geometry input-visible khác target-eligible.
   Phải fully-hidden cho strict temporal target. Sửa propagation từ extraction/grouped
   supports vào geometry; không chỉ khai báo joints mà bỏ preprocessing prerequisites.
   Counterfactual hiện kiểm tra feature/pose-derived geometry với fixed validity/schedule;
   raw-video→pretrained rerun counterfactual vẫn cần thực hiện trước research pilot.

## Clip thật và failure memo

Evidence chính: [real_clip_003.json](evidence/real_clip_003.json).

- Toàn bộ clip train đầu tiên của ASL Citizen, 71 raw frames → 59 selected samples.
- Valid visual streams `[59,0,15,59]`; clip này chủ yếu một tay, hand detector có 29/71
  candidate frames, và wrist quality gate loại thêm observations. Không biến missing
  left-hand stream thành background valid.
- Geometry complete/valid chỉ 10/495 scheduled tokens; đây là **target availability risk**,
  chưa đủ để suy geometry branch có ích hoặc train tốt trên corpus.
- Với mask seed17 của diagnostic này: 100 semantic targets nhưng **0 geometry targets
  fully-hidden**. Geometry empty phải có count0/zero-term trong trainer; cần profile
  availability qua nhiều masks/clips trước pilot, không claim geometry loss đã được train.
- Run001 là diagnostic với bug unrelated wrist; không dùng làm experiment.
  Run002 đã sửa matching nhưng chưa hash implementation; run003 thay thế nó cho evidence.
- Overlay của một frame (`artifacts/real_clip_002/overlay.png`) đã được quan sát: right
  hand gần body right wrist ở phía trái ảnh. Source mirror metadata vẫn unknown;
  cần nhiều clip, hai tay/crossing trước pilot. Không suy handedness từ x-position.
- Detector IMAGE repeated-frame test pass cho body/hand/face với black intervening frame.
  Đây là evidence temporal independence giới hạn, không chứng minh mọi detector failure mode.
- Khoảng 9.38 s cho run003 trên RTX 5880 Ada; số này gồm startup/one clip pipeline,
  không dùng extrapolate GPU-hours hoặc throughput corpus.

## Các yêu cầu còn mở của proposal

1. Package CLI preprocessing/cache/targets/verify/profile, stage-specific validation,
   labels-free source/segment sampling, batch padding và crop/full-clip inference contracts.
2. Visual frozen probe; cluster CE baseline/codebooks; online tokenizer/projectors,
   temporal encoder, predictor public metadata, exact-copy EMA teacher và E02–E04 objectives.
3. Global per-type/group-scale loss normalization; empty sets; AMP finite consensus,
   successful-step EMA/scheduler; accumulation, DDP, checkpoint/resume RNG/data order.
4. CPU synthetic training smoke và **real full MVP** tới optimizer/checkpoint/unmasked
   embeddings/downstream prediction; leakage/static-retention gates trong model thực.
5. ISLR linear/attentive/temporal FT, retrieval, phonology, functioning CTC/seq2seq heads,
   metrics/predictions/protocol hashes; toy continuous tests khi chưa chạy real continuous.
6. Fair ablation runner, successful-step/frame budgets, seeds, uncertainty, experiment
   summaries, deployment artifacts/export parity, best/last selection theo validation.
7. Full split/source/near-duplicate audit; target fit train pool; pilot theo budget,
   matched E03/E04 utility comparison, failure buckets và decision memo. Chưa có
   evidence để chọn hoặc loại geometry.
8. README/METHOD/data/model cards, failure memo, complete commands và reproduction handoff;
   optional backends phải explicit unavailable hoặc được implement/test trước claim.

Thứ tự tiếp tục: hoàn thiện orchestration/data batching → M3 probe → M6/M7 SSL runtime
và smoke/resume/DDP → M8 downstream → M9 pilot. Không cần chờ approval để tiếp tục
các thay đổi reversible đã được yêu cầu; báo cáo này để người dùng review code hiện có.
