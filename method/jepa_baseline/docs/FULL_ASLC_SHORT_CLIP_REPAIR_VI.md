# Xử lý một clip train ngắn và tiếp tục job SignRep

Ngày 2026-10-14, người dùng đồng ý loại đúng clip 11 frames theo recommendation.

- Video: `asl_citizen:011101656879706212-YAWN 1`, label2709 / glossYAWN1.
- Còn40.153 video SSL train và full gallery, đủ2.731 lớp; YAWN1 còn14 video.
- File video gốc, raw cache, `source.jsonl`, official CSV/manifests và
  `pose_report.json` được giữ nguyên, không sửa báo cáo thất bại thành PASS.
- Proxy gallery/val, official val/test giữ nguyên hash.
- Effective manifests: `runtime/full_aslc_kinetics/source_training.jsonl`,
  `manifests/gallery_train.jsonl`.
- Audit: `approved_exclusions.json`, `membership_exclusion_audit.json`,
  `short_clip_repair.json`, `effective_pose_check.json` trong job directory.

Lọc một dòng không được reindex raw caches theo dòng mới: mỗi dòng effective
manifest trỏ explicit raw cache/index/SHA gốc. Preparation kiểm tra hash và
video identity trước reuse. Đã preflight40.153 raw caches, video hashes,
teacher identity, frame alignment và signer crop trước khi start.

`corpus_policy.py` chỉ chấp nhận exception đã duyệt, đúng observed11-frame clip.
Nếu xuất hiện pose failure khác, class mất toàn bộ mẫu hoặc cache mapping sai,
pipeline vẫn báo lỗi; không silently skip dữ liệu. Training/gallery phải có cùng
membership và thứ tự. Prepared audit tiếp tục dùng nguyên officialtrain reference
và held-out exclusions, kèm policy audit ở job level.

Service vẫn Kinetics-only initialization,20views/BF16,scheduler100k và dừng50k.
Source snapshot/config/input pins được cập nhật có archive **trước khi training
có checkpoint đầu tiên**, không bypass resume guard của run đang train.

Bone medians tính lại trên40.153 video train; raw caches không trích lại.
Normalized caches ghi atomic và resume theo raw/bone SHA. Normalization worker
giữ immutable bone statistics trong RAM thay vì parse40.153 lần. Array và metadata
đối chiếu cache cũ khớp exact; khi restart chỉ xử lý phần chưa hoàn tất.
Progress normalization xuất trực tiếp ra progress.txt và console.log.

40tests pass, gồm exception scope, raw cache mapping sau lọc middle row,
unapproved-failure rejection, resume/RNG và baseline tests. Native normalization
probe trên4raw caches pass exact arrays/metadata và filehash reuse.

Theo dõi:

```bash
tail -f /home/haipd/Sign_Representations/method/jepa_baseline/runtime/full_aslc_kinetics/progress.txt
systemctl --user status signrep-full-aslc-kinetics.service
```

Khi audit xong, service tự chuyển sang training. Lỗi clip train đã được xử lý;
không thể bảo đảm không còn mọi lỗi phần cứng/điện/GPU. Checkpoint và controlled
stop/resume tiếp tục hoạt động như hướng dẫn FULL_ASLC_BACKGROUND_JOB_VI.md.

Header preflight cảval/test tìm thấy không có clip val ngắn, nhưng17 test clips
ngắn hơn16frame. Đây chưa phải finaltest stage; chưa đổi membership/testpolicy,
chưa evaluate model trên test. Trước khi chốt finaltest cần policy riêng được ghi
rõ để xử lý clips ngắn và giữ officialtest denominator.
