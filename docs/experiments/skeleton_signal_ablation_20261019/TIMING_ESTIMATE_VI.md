# Ước lượng thời gian nếu mỗi cấu hình chỉ chạy 1 seed

Ngày đối chiếu:19/10/2026. **Đây là ước lượng theo kịch bản, không phải benchmark code mới.** Không thay protocol3-seed đã viết; bảng này trả lời phương án chỉ dùng seed0. Không khởi chạy training hoặc evaluation mới.

## 1. Dữ kiện thời gian từ run cũ

| Run | Updates | Thời gian train được log | Trung bình giây/update | Nguồn |
|---|---:|---:|---:|---|
| PLctx full160 | 200.640 | 4,05 giờ | 0,0727 | [manifest](../../../method/phonojepa/runtime/runs/PJ_PLctx_FULL_160ep/manifest.json) |
| PLctx16k | 16.000 | 0,494 giờ | 0,1111 | [manifest](../../../method/phonojepa/runtime/runs/PJ_PLctx_16k/manifest.json) |
| PhonoPath full160 | 200.640 | 56,81 giờ | 1,0193 | [manifest đã archive](../../../artifacts/phonopath_jepa_results_20261019/metadata/manifest.json) |

Các run có input pipeline, augmentation, target computation và điều kiện GPU khác nhau. Không quy chênh lệch này riêng cho một module; không coi PhonoPath là tốc độ bắt buộc của skeleton model mới.

PLctx full-dictionary extraction đã log train40.154 clip478,58giây và test32.941 clip319,52giây: [kết quả có extraction timing](../../../method/phonojepa/runtime/eval/PJ_PLctx_FULL_160ep_dictionary/aslc/results.json). Đó là anchor full-corpus; không dùng timing ASLC300/pilot để dự đoán full-val pipeline.

Thiết bị quan sát ở thời điểm đọc: **RTX5880 Ada, khoảng48GB VRAM**; GPU đang bận khoảng88%, dùng khoảng45GB. Đây là snapshot, không bảo đảm GPU sẵn sàng khi bắt đầu chạy.

## 2. Giả định cho kịch bản nhanh có cache tối ưu

- Một GPU, các run chạy tuần tự, có lượt GPU đủ tài nguyên; không nhân đôi tốc độ bằng cách mặc định chạy song song.
- Cache/preprocessing tối ưu gần hướng PLctx, không tính lại feature đắt từ đầu mỗi sample. Code mới bỏ path signatures và các augmentation nặng của PhonoPath; hiệu quả thực chưa được đo.
- **Planning rate:0,10–0,20giây/update.** Đây là giả định có headroom so PLctx, không là số measured của S0–S7.
- **Một full validation pass:15–30phút**, gồm extraction gallery train+query val và retrieval scoring. Đây là allowance dựa trên anchor cũ, chưa đo cho adapter mới; chưa bao gồm mọi probe/perturbation sâu.
- Gallery embedding phải trích lại khi checkpoint đổi. Pose cache dùng chung được, nhưng embedding gallery của encoder khác không dùng lại được.
- Không bao gồm implement/debug runner mới, hàng đợi GPU, tải dataset, full pose re-extraction hay sửa dữ liệu.

## 3. Số update và tổng thời gian ước lượng

| Phạm vi1seed | Train updates | Số val passes giả định | Train | Extract/eval | Tổng làm tròn |
|---|---:|---:|---:|---:|---:|
| Chỉ S0–S7 | 80.000 | 16:5k và10k mỗi arm | 2,2–4,4h | 4–8h | **6–13h** |
| S0–S7 +4controls +2physical-hand arms | 140.000 | 28 | 3,9–7,8h | 7–14h | **11–22h; dành12–24h** |
| Thêm confirmation: baseline+2candidate, mỗi arm40k | 120.000 | 9:10k/20k/40k | 3,3–6,7h | 2,25–4,5h | **Thêm6–12h** |
| Confirmation có thêm1matched control40k | 160.000 | 12 | 4,4–8,9h | 3–6h | **Thêm8–16h** |
| Mọi arm hiện đăng ký, bỏ seed1/2, kể cả2full runs | 791.280 | Khoảng63 | 22–44h | 16–32h | **Khoảng38–76h**, chưa E4/diagnostics sâu |

Dòng cuối gồm28arm/seed0:14initial +1random-init +4confirmation +2supervised reference +5objective +2full-budget200.640. Một số arm là conditional; không bắt buộc chạy tất cả.63val passes là giả định scheduling để tính budget, không là measured count.

E4 tối đa8factorial40k +4full/LOO40k cho1seed thêm480.000updates: khoảng13–27htrain +3–6hendpoint evaluation. Thêm probes, robustness và final transfer, **toàn chương trình1seed nên dự trù khoảng3–6ngày thực thi** nếu các giả định nhanh giữ được. Vòng objective confirmation hoặc controls phát sinh cần cộng riêng theo số run; đây không là upper bound.

## 4. Nếu pipeline chậm hơn

14initial arms có140kupdates. Nếu đo ra0,5–1,0giây/update, riêng train là19,4–38,9h; cộng validation theo allowance trên thành khoảng26–53h, chưa tính contention có thể làm cả evaluation chậm. Khi đó không còn ngân sách12–24h.

Công thức cập nhật sau profiler:

```text
T_total_h = sum(updates_arm * measured_sec_per_update_arm)/3600
            + sum(measured_full_validation_seconds)/3600
            + probe/robustness/final_transfer_hours
wall_clock = T_total_h + GPU_queue_and_interruptions
```

Muốn estimate có giá trị vận hành, cần implement adapter/runner rồi profile200–500updates sau warmup của S2 và arm nặng nhất, kèm một full-val extraction. Chưa có các đo đó nên không cam kết ETA chính xác.

## 5. Cách dùng1seed hợp lý

Ưu tiên initial14arms trước: mục tiêu là khám phá tín hiệu trong khoảng1ngày theo kịch bản nhanh. Sau đó chỉ tăng budget40k cho baseline và2candidate.1seed không đo được training variance; CI qua query chỉ nói uncertainty của evaluation conditional-on-checkpoint. Không gọi một gain nhỏ từ1seed là bằng chứng ổn định qua initialization.
