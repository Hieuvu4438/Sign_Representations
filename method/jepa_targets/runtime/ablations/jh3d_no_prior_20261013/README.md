## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-13
- Verification Status: UNVERIFIED (experiment running)
- Version Label: jh3d-no-prior-v1

## JH3D bỏ prior loss

Yêu cầu: kiểm tra bỏ prior loss có làm giảm chất lượng representation không.

Can thiệp duy nhất trong objective: `--prior-w 0`, bỏ toàn bộ chín prior hình học và activity loss; giữ JEPA latent loss. Train lại từ initial.model.pt của ASLC002, không fine-tune checkpoint JH3D đã học prior. Giữ 3.000 video train, seed 20261008, batch 16 video / 32 views, 4.000 bước, 6 workers, mask 80%, predictor và EMA 0.99925, optimizer/schedule gốc.

Bản gốc: runtime/JH3D/last.pt. Baseline val LOSO R@1 white512 + mean: 39,1498%; mean chưa whitening: 12,50%. Đánh giá cùng 10.304 video validation, cùng clip/window/pooling. Whitening fit trên validation như EX16 gốc; không đọc test. Đây là exploratory validation, một seed.

Khi bỏ prior, fc và prior heads không có gradient. Đánh giá hai cách: (1) embedding pipeline gốc fc(mean tokens); (2) mean tokens trước fc cho cả hai nhánh, để phân biệt thay đổi encoder với projection chưa train.

Smoke 6 bước đã pass: loss tổng bằng latent loss, reconstruction = 0; backbone thay đổi; fc và prior heads giữ nguyên trọng số khởi tạo. Không retry tự động; mỗi stage giới hạn 4 giờ, theo dõi process/exit code mỗi 30 giây. Không dừng hoặc sửa các GPU job khác.

Lệnh pipeline:

```bash
cd /home/haipd/Sign_Representations/method/jepa_targets
/home/haipd/Sign_Representations/.venv-signrep/bin/python jt/jh3d_no_prior.py
```

Theo dõi: status.json, runner.log, train.log và ../../JH3D_NOPRIOR_20261013/train_log.jsonl.

Khi hoàn tất: RESULTS.md, comparison.json (delta và signer-cluster bootstrap CI), evaluation.json (retrieval các kiểu pooling, probes, geometry). Nếu lỗi: status.json ghi failed và stage; không kết luận từ kết quả chưa đủ.
