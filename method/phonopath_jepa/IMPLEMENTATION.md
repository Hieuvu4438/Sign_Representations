# Đối chiếu proposal16 v1.1

| Yêu cầu | Code / bằng chứng |
|---|---|
| C1: wrist, arm, 5 finger chains, twist, aperture, tay–mặt, tổng/hiệu hai tay, tay–tay | `pp/paths.py`; kiểm tra dims 19/195/37 |
| C2: 224 S + 110 P + 5 G + CLS | `pp/features.py`, `pp/model.py`; unit test count và readout |
| C3: Chen-visible, đầu mút chung, không bridge gap | `pp/signature.py`, `pp/ptokens.py`; oracle và end-to-end leakage tests |
| C4: PLCM-lite, support frame nguồn, face closure, P target ≥50% thời gian, drop phần còn lại | `pp/data.py`, `pp/closure.py`, `pp/ptokens.py`; support tests |
| C5: in-image, wrist gate, SG5, gap≤2, train median/IQR/asinh, still | `pp/data.py`, `pp/ptokens.py`, `pp/fit_stats.py`; raw-observation leakage test |
| C6: grp, grp+P, mw, ord2; train-only PCA | `pp/extract.py`, `pp/evaluate.py`; real-cache extraction smoke |
| Shared encoder, continuous RoPE, zero-init bias, predictor metadata, EMA, LN-last4 target, centered CE | `pp/model.py`, `pp/train.py`; forward/backward all arms, A0 parity test |
| G0, few-shot, linear probe, fine-tune | `pp/downstream.py`; public CLI smoke trên cache thật |
| VAL-DICT, WLASL, pessimistic ties, metrics, ledger | `pp/evaluate.py`, `pp/prepare.py`; scorer tests |
| D-LEX v3, bootstrap/decision gates | `pp/probe.py`, `scripts/compare.py`, `scripts/gates.py` |
| Sequential runner/watchdog, manifests/logs/checkpoints/resume | `pp/train.py`, `scripts/schedule.py`; real resume replay |

## Chi tiết được chốt để biến proposal thành code

1. **Dependency mới phát hiện khi port observation filter:** ASM bỏ body wrist khỏi model, còn PLctx sở hữu body wrist trong LOC. Gate tay đọc body wrist; do đó khi LOC wrist bị che, hand CONF cùng frame cũng cần context-drop để không rò qua observation status. `source_hidden` bao gồm dependency này. Target S vẫn là mask PLCM gốc; P target theo support sau closure. Đây là closure bảo thủ cần thiết, có test sau observation filter. Nó có thể làm giảm P visibility hơn mô phỏng trong proposal; phải ghi run card trước Phase 2.
2. **Không có anchor vai:** body frame và mirror center dùng fallback hằng số, không lấy centroid của khớp có thể bị che. Không coi mũi/cằm là anchor công khai.
3. **Nội suy:** chỉ lấp natural gap ≤2 mẫu của mỗi khớp, giữ support cả hai đầu mút; SG5 dùng replicate ở biên thật của view và yêu cầu stencil quan sát đủ. Theo làm rõ ở §9.1(b), một local P token chỉ absent khi không còn path nào đủ quan sát; từng path thiếu có flag riêng và giá trị 0 sau chuẩn hóa. Không loại cả token vì một path thiếu, và không nội suy cầu dài để bịa chuyển động. Flag/still được giữ riêng theo path.
4. **Wrist twist 2D:** góc của vector h5−h17, unwrap riêng trong mỗi local window/segment; signed foreshortening dùng diện tích tam giác lòng bàn tay chia bình phương chiều dài h0→h9. Đây là proxy 2D, không phải ước lượng rotation 3D.
5. **A2:** level 2 là upper triangle của QV increment (gồm đường chéo), cắt về d(d−1)/2. Global QV cộng theo đoạn, không cộng wedge, để không đưa lại Lévy area vào control.
6. **G:** 15 đoạn `[4k,4k+4]` và đoạn cuối `[60,63]`; segment cuối repeat endpoint khi vector hóa. Global basepoint lấy từ đoạn visible đầu tiên của chính path. Statistics G fit riêng scale=0. Confidence/coverage G chỉ đọc các đoạn visible. Không có feature nhảy qua gap.
7. **A3:** giữ các họ path sign của §7.3, P-only; thêm mean local, motion-seeded whole-time hand+arm subtree khoảng 23/98 owned joints, global full và centered cosine latent. G chạm subtree bị che cũng bị ẩn; không để global của tay bị che lộ ra như A5. Paper gốc không có code/chi tiết đầy đủ; đây là port theo thiết kế A3 của proposal, không phải công bố tái tạo NTU Path-JEPA.
8. **A1-pb:** triển khai đúng experimental mask budget “chỉ các run ≤4 bin” cho P. Điều này cho P đọc được một phần S target, nên **không có cùng bảo đảm leak-free như A1**. Không gộp kết quả A1-pb với A1. A5, A10, A3 cũng có các leak/control được khai báo trong proposal (global full hoặc content-dependent mask).
9. **Plan B:** B1 nối thêm 8 feature quan hệ vào LOC token (không thay toạ độ gốc); B2 trộn subtree ở far; B3 teacher unwarped và student warped, nội suy latent teacher về source-time bin của student; B4 chạy readout trên checkpoint A0, không train.
10. **A10 giữ ngân sách:** motion seed đổi vị trí thời gian/kênh của mask PLCM bằng permutation/roll, giữ nguyên số cell bị che và hình dạng block; không giảm near budget thành một block duy nhất.
11. **Readout:** height y-up; rest = q10(height)+.25; mw không dùng D-LOC energy, chỉ CONF raw energy của path present/not-still. Rest/midpoint hoạt động lấy trên clip khi inference; pooling dùng các token present. PCA1536→768 và PCA3072→768 thực hiện sau mean window, fit gallery train. A3 `grp` là pooling nhóm P vì không có S.
12. **Checkpoint:** lưu model và optimizer đầy đủ, frozen stats cùng provenance, RNG CPU/CUDA/numpy/Python, epoch/cursor. Exact replay được kiểm bằng CPU single-thread; repeated-index gradients của bias có thể khác ~1e-10 trên CPU nhiều thread. CUDA bitwise determinism không được khẳng định khi chưa đo.
13. **Phạm vi triển khai:** đủ các arm chính và Plan B để chạy chương trình đề xuất. Các ý tưởng mở “chỉ khi còn ngân sách” (learnable projection trước signature, P-face stream, KoLeo, target nối nhiều block) không được bật/giả lập; `lead_lag` toán học đã có. Scale corpus 105h là giai đoạn tùy chọn, không tự tải/mix dữ liệu ngoài ASLC train.

## Các gate còn thuộc thực nghiệm

Chưa có kết quả hoàn tất screening/full160, chưa chứng minh lợi ích retrieval, chưa dùng official test. A1 full160 đã khởi chạy theo yêu cầu trực tiếp của người dùng. CUDA/BF16 batch32 đã qua smoke, tốc độ ban đầu khoảng 0,5 step/s trên GPU dùng chung, chưa đạt ≥5 step/s. `scripts/benchmark.py` ghi closure theo channel/scale và đo end-to-end throughput khi có slot. Đọc `VALIDATION.md` để phân biệt smoke kết nối pipeline với kết luận khoa học.
