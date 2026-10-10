# Đánh giá hướng nghiên cứu

Trạng thái: kết luận pilot đã có; kết luận matrix/test chờ các run kết thúc. Không dùng các hàng planned hoặc run chưa đủ update làm bằng chứng confirmatory.

## Câu hỏi thực sự đang được kiểm tra

Thêm supervision quan hệ cosine giữa đặc trưng LH(t) và RH(t+lag) có giúp biểu diễn frozen phân biệt gloss tốt hơn một encoder chỉ tái tạo state hay không? A4−A1 là đối chứng nhân quả chính trong phạm vi recipe, dữ liệu và ngân sách đã khóa. A0 là mốc tham chiếu thực dụng: đặc trưng teacher/body trực tiếp, không huấn luyện Transformer.

Metric chính là macro gloss R@1 trong dictionary retrieval: mỗi query được xếp hạng theo các gloss có exemplar trong gallery; đúng khi gloss đúng đứng đầu. Đây không phải accuracy dịch ngôn ngữ ký hiệu, không phải continuous recognition, và không đo trực tiếp phonology. Gallery tùy chỉnh40 gloss K=1; tập dev80 query cân bằng2/gloss nên macro và micro R@1 trùng nhau ở pilot.

## Bằng chứng pilot đã hoàn tất

Nguồn: `results/pilot_results.json`, checkpoint/source/cache/split hashes trong các run manifests. A1/A4 cùng seed17,1000 update, effective batch32, cùng khởi tạo và4000 microbatch window/mask plans.

| Variant | Macro R@1 | Macro R@5 | MRR |
|---|---:|---:|---:|
| A0 raw teacher/body | 13,75% | 35,00% | 0,2574 |
| A1 state-only | 10,00% | 23,75% | 0,1976 |
| A4 state + cross-hand relation | 8,75% | 26,25% | 0,1955 |

A4−A1 R@1 = −1,25 điểm phần trăm; paired gloss-bootstrap95% CI [−3,75;0,00],1000 replicate/40 gloss clusters. CI này mô tả sampling của evaluation gloss tại hai checkpoint cố định, chưa bao gồm biến thiên training seed. Không có bằng chứng cải thiện; CI chạm0 nên không kết luận chắc chắn A4 kém hơn.

Held-out SSL diagnostics (`artifacts/diagnostics_a1.json`, `diagnostics_a4.json`) cho thấy A4 giảm relation prediction error từ0,11357 xuống0,07892 so với A1. Tuy vậy, nearest-visible-frame predictor đạt0,03697, thấp hơn cả hai; hand reconstruction error của nearest-visible là0,32739, so với A1/A4 là0,41233/0,41800. Đây là các lỗi trên cùng fixed draw, không phải retrieval scores của predictor đó.

Diễn giải: relation objective được tối ưu tốt hơn nhưng chưa chuyển thành retrieval utility. Tính liên tục của đặc trưng có thể tạo shortcut; generic DINO/crop noise có thể hạn chế tín hiệu phối hợp. Đây là giả thuyết giải thích, chưa được chứng minh nguyên nhân. Target có biến thiên thật theo lag (std~0,135–0,150), nên không thể nói mọi relation target đều gần hằng số. Covariance diagnostics chỉ có3 SSL-dev clip, rank tối đa2; không dùng chúng để tuyên bố collapse.

## Quyết định hiện tại

Chưa nên mở rộng nguyên recipe để nhắm tới một paper tuyên bố method tốt hơn. Hoàn tất matrix đã khóa trong ngân sách hiện tại để đo độ lặp lại và các controls. Nếu A4 không có lợi ích ổn định so với A1, dừng mở rộng recipe này và giữ kết quả âm trong báo cáo.

Điều đó chưa bác bỏ toàn bộ hướng học phối hợp hai tay. Một nhánh tiếp theo phải có lý do cụ thể từ diagnostics: kiểm định chất lượng teacher/hand crops, hoặc kiểm tra masking dài hơn để giảm shortcut temporal interpolation. Mỗi thay đổi cần protocol mới, cache version phù hợp và chạy lại A1/A4 tương ứng; không thêm nhiều losses để cứu headline sau khi xem test.

## Điều kiện để đầu tư tiếp cho paper

- Primary retrieval gain phải lặp lại qua training seeds, có uncertainty theo gloss và effect size đủ hữu ích; một seed hoặc R@5 tăng trong khi primary R@1 giảm chưa đủ.
- Controls zero-lag, intra-hand và shuffled target phải hỗ trợ vai trò riêng của quan hệ cross-hand temporal, đồng thời công khai pair counts và compute khác nhau. Nếu controls thiếu, mechanism claim vẫn thiếu bằng chứng.
- So với A0, cần chứng minh giá trị thực dụng hoặc một lợi ích đã đăng ký khác. A4 hơn A1 nhưng vẫn thua raw features chưa đủ cho claim retrieval method hữu ích.
- Cần đánh giá rộng hơn47 SSL clip/40 gloss, trên test và điều kiện có kiểm soát source/signer. Không suy ra signer invariance từ query-gallery signer disjointness.
- Claim phonology cần natural minimal-pair annotations verified; hiện BLOCKED_EXTERNAL. Retrieval tốt hơn tự nó không chứng minh phonology tốt hơn.

Không có ngưỡng R@1 tuyệt đối hay một p-value duy nhất bảo đảm đủ viết paper. Gói implementation có thể hoàn tất với kết quả âm; paper cải tiến method cần bằng chứng utility, cơ chế, độ bền và novelty phù hợp. Một paper về failure mode cũng cần thí nghiệm rộng, giải thích có kiểm chứng và đóng góp rõ, không chỉ một pilot âm.
