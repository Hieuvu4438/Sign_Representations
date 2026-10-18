# Rà soát PLctx160epochs — iterations và dữ liệu train

Kết luận đã kiểm tra từ code, launcher, dữ liệu cache, manifest, logs và optimizer checkpoint:
**PLctx160 đã train200.640successful optimizer updates trên toàn bộ40.154official train clips của ASL-Citizen.**
Không dùng chung train+validation+test để train; validation10.304 và test32.941 độc lập.

| Nội dung | Kết quả |
|---|---|
| Iterations planned / completed |200.640 /200.640|
| Batch |32source clips/update|
| Views |Mỗi source clip sinh near/far views, losses gộp trong một optimizer update|
| Epoch budget |160|
| Steps/epoch |floor(40.154/32)=1.254|
| Công thức |160×1.254=200.640|
| Drop-last |26clips dư mỗi shuffled epoch|
| Tổng source-clip draws |6.420.480|
| Draws trung bình/source |159,8964|
| Cache train |40.154unique IDs, đúng thứ tự official train.csv|
| Coverage |2.731glosses,35train signers;3.462.848frames|
| Train overlaps val/test |0 /0|
| Initialization / seed |Scratch /0|
| Checkpoints |final.pt và ckpt.pt đều step200.640; mọi Adam step state hiện có=200.640|
| Manifest |steps_planned=steps_done=200.640|
| Log |DONE200640; periodic metrics cuối step200.600|

Một iteration là một optimizer update, không phải một clip, frame, view hay epoch.
`drop_last=True` loại batch dư26clips mỗi epoch; shuffle mỗi epoch nên đây không phải subset cố định26clips bị loại khỏi dataset. Cache chứa đầy đủ official train IDs, nhưng mỗi clip không được bảo đảm xuất hiện đúng160lần. Không có per-sample draw log để chứng minh exact exposure của từng clip.

## Code và provenance

- `method/phonojepa/scripts/launch_plctx_full160.py`: EPOCHS160, BATCH32, N_CLIPS40154; truyền explicit --cache aslc_train_full.npz và --steps200640 vào trainer.
- `method/phonojepa/scripts/build_full_train_cache.py`: đọc official train.csv, kiểm unique40154, disjoint với val/test; build mọi train cache entries.
- `method/phonojepa/pj/train.py`: load cache do CLI chỉ định; DataLoader shuffle/drop_last; tăng step sau optimizer.step; save final/manifest.
- `method/phonojepa/pj/data/cache.py`: dataset size từ cache offsets; không hardcode subset3000 khi dùng full cache.
- Docstring/default trainer vẫn nhắc pilot3000clips/4000steps; không mô tả run full160, vì launcher override cả cache và steps. Không được dùng default/help text thay evidence run.
- Hash toàn bộ training source được ghi lúc launch khớp code hiện tại; hash full train cache và official train CSV khớp manifest đã ghi.
- Không thay đổi code, config, cache hoặc weights của PLctx trong đợt rà soát này.

`audit.json` giữ hashes, checkpoint counters, source matches và các assertion cụ thể.
