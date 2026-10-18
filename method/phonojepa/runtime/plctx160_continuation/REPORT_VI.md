# PLctx160 continuation — báo cáo kết quả đầy đủ

Hoàn tất06:11ngày16/10/2026, giờViệtNam; khoảng8giờ8phút. Service kết thúc thành công.
**Kết quả đo: validation tăng nhẹ; officialtest của checkpoint chọn bằng validation không cải thiện so với PLctx160 gốc.**

## Hai nhánh continuation

| Nhánh | Updates thêm đã chạy | Globalstep khi dừng | Best globalstep | Best validationRec@1 (%) | Trạng thái |
|---|---:|---:|---:|---:|---|
| lr1e5 | 35000 | 235640 | 210640 | 22.9523 | early_stopped |
| lr3e5 | 30000 | 230640 | 205640 | 22.9523 | early_stopped |

Cả hai peak validationRec@1 bằng nhau. Tie policy đã chốt giữ ứng viên xuất hiện trước: chọnlr1e5/bestglobal210.640. Không chạy test của lr3e5 để chọn lại winner.

## Validation theo iterations

Cùng40.154traingallery và10.304valqueries,raw/_grp/window64/stride2; Recall hiển thị%,DCG/MRR giữ0–1.

| Nhánh | Updates thêm | Globalstep | DCG | MRR | Rec@1 (%) | Rec@5 (%) | Rec@10 (%) | MacroRec@1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 200640 | 0.4621 | 0.3399 | 22.7193 | 45.9433 | 56.7838 | 23.0239 |
| lr1e5 | 5000 | 205640 | 0.4626 | 0.3407 | 22.8552 | 45.9433 | 56.7061 | 23.1399 |
| lr1e5 | 10000 | 210640 | 0.4629 | 0.3411 | 22.9523 | 45.9918 | 56.7838 | 23.2577 |
| lr1e5 | 15000 | 215640 | 0.4630 | 0.3412 | 22.9523 | 45.9724 | 56.7352 | 23.2436 |
| lr1e5 | 20000 | 220640 | 0.4622 | 0.3402 | 22.8164 | 45.8172 | 56.7061 | 23.1326 |
| lr1e5 | 25000 | 225640 | 0.4615 | 0.3394 | 22.7387 | 45.8075 | 56.5897 | 23.0599 |
| lr1e5 | 30000 | 230640 | 0.4612 | 0.3391 | 22.6708 | 45.7395 | 56.4344 | 22.9568 |
| lr1e5 | 35000 | 235640 | 0.4613 | 0.3391 | 22.6320 | 45.8560 | 56.4732 | 22.9312 |
| lr3e5 | 5000 | 205640 | 0.4629 | 0.3412 | 22.9523 | 45.8463 | 56.7061 | 23.2491 |
| lr3e5 | 10000 | 210640 | 0.4627 | 0.3408 | 22.8746 | 45.8269 | 56.7255 | 23.1753 |
| lr3e5 | 15000 | 215640 | 0.4626 | 0.3406 | 22.8455 | 45.8560 | 56.7158 | 23.1441 |
| lr3e5 | 20000 | 220640 | 0.4617 | 0.3396 | 22.7193 | 45.6716 | 56.8226 | 23.0349 |
| lr3e5 | 25000 | 225640 | 0.4612 | 0.3390 | 22.5932 | 45.8948 | 56.6964 | 22.9183 |
| lr3e5 | 30000 | 230640 | 0.4610 | 0.3387 | 22.5641 | 45.8172 | 56.7158 | 22.8665 |

## Official test

Cùng40.154gallery và32.941testqueries,2.731glosses,rawfeatures/khôngwhitening. Native baseline zero-vector policy giữ nguyên;39galleryzero và8testzero; đủ query denominator.

| Metric | PLctx160 | Continuation đã chọn | Chênh lệch continuation−baseline |
|---|---:|---:|---:|
| DCG | 0.3962 | 0.3958 | -0.0004 |
| MRR | 0.2665 | 0.2660 | -0.0004 |
| Rec@1 (%, chênh lệch điểm%) | 16.6783 | 16.6328 | -0.0455 |
| Rec@5 (%, chênh lệch điểm%) | 36.8963 | 36.7839 | -0.1123 |
| Rec@10 (%, chênh lệch điểm%) | 46.9415 | 46.8444 | -0.0971 |
| macro_Rec@1 (%, chênh lệch điểm%) | 16.7922 | 16.7480 | -0.0442 |

Top1đúng:baseline5.494clips;continuation5.479clips,ít hơn15query. Chênh lệch nhỏ; chưa có CI/kiểm định hoặc nhiều seed để kết luận suy giảm có ý nghĩa thống kê.

## Diễn giải và giới hạn

- Continuation+LRrestart1e-5/3e-5 giữ nguyên PLctx recipe chỉ tăng validationRec@1 khoảng0,233điểm%. Validation sau peak giảm, cả hai nhánh dừng do5lượt không vượt best.
- Test của validationwinner giảm khoảng0,0455điểm%Rec@1. Không có bằng chứng cải thiện test trong experiment này; không suy ra mọi LR/objective/data change đều không có tác dụng.
- Một seed; testbaseline đã được xem trước khi đề xuất extension nên đây là exploratoryimprovement. LR/nhánh/checkpoint mới chỉ chọn trên val.
- Modelbaseline và caches gốc giữ nguyên; original checkpoint thiếu sampler/NumPy/CUDA state, nên firstfork dùng sampler mới đã khai báo; subsequentresume đầy đủ.

## Artifacts

- `report.json`: báo cáo gốc máy đọc, selection và history.
- `validation_iterations.csv`: mọi validationiterations.
- `selection.json`, `final_test/metrics.json`, `final_test/ranks.npy`: checkpoint đã chọn và test thật.
- `result_verification.json`: recompute từ ranks khớp report, SHAcheckpoint khớp selection.
- `arms/lr1e5/best.pt`: checkpointvalidationselected210.640; `arms/lr3e5/best.pt`: best205.640.
- `arms/*/last.pt`, `previous.pt`, `complete.json`: state cuối và earlystopping evidence.
