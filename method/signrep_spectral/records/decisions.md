# Decisions and deviations from the spec (all made before quality measurement unless noted)

| # | Decision | Reason | Spec section |
|---|---|---|---|
| D1 | Targets are 2-D: heads `left_pose`/`right_pose` = 42 dims (21 joints x xy), `wrist_image_xy` = 4; spectral hand groups D=40 (joints 1..20) | Local pose has no z; inventing z would fake data | 6.3 |
| D2 | Joint observed = single-person frame & mean hand score >= 1.0 & joint score >= 1.0 & finite & coords in [-0.05, 1.05]; `confidence_mode = backend_score_unnormalized>=1.0` | Scores are unnormalized backend scores (semantics unverified); 0.5 threshold is meaningless on this scale; resting/out-of-frame hands score ~0.8-1.5, signing hands ~3-7 | 6.4 |
| D3 | Target policy r2 (`hand_image_canonical_v1_xy_r2`): `scale_floor = 0.02` width units, joints with abs(wrist-relative coord) > 4 palm lengths -> unobserved | r1 (floor 1e-4) let ~1% of clips with collapsed hand detections produce targets up to 226 palm lengths (scaler std up to 5.5). Changed after the r1 scaler/coverage audit, **before any training or quality measurement**. r1 artifacts kept: `scaler_001`, `targets_001`, `calibration_001` | 6.3 |
| D4 | Attention kernel `sdpa_4d_reshape` (scoped shim of the vendored module's `F`) for every arm incl. B0 | 2.7x throughput; parity within 1e-5 fp32 | 9.1 (deviation: scoped module attribute swap) |
| D5 | Extraction stride **16** (non-overlapping 16-frame windows) instead of 2; bf16 autocast | Budget: stride 2 = 2.85M clips/arm (~4.3 h); stride 16 = 0.39M (~35 min). Same for every arm | 12.2 |
| D6 | Only pooled features (`features` and `latent`) are stored, not per-window arrays | ~63 GB free disk | 12.2 |
| D7 | Aux strength: `lambda = 0.1 / median(||g_aux||/||g_base||)` on encoder params, 8 fixed train batches at the shared init (`calibration_002`): spectral 0.383, velocity 0.053, acceleration 0.028, window-time 0.929. Spectral warmup 100 updates for all aux arms | Matched gradient strength across auxiliaries (spec 12.4 option 1) | 8.2, 12.4 |
| D8 | Pilot training: bs 16, 3000 updates, AdamW (enc 1e-5, dec 1e-4, wd 0.05, betas 0.9/0.95), cosine LR with 200-update warmup, grad clip 1.0, mask ratio 0.6 (explicit masks keyed by batch index), bf16 autocast for the network, FP32 losses, seed 1701, decoder seed 1702 | Engineering defaults from the spec scaled to a ~15 min/arm pilot | 10.2 |
| D9 | Resume and DDP are rejected by config validation (not implemented) | MVP pilot does not need them; spec forbids silent fallback | 7.6, 10.4 |
| D10 | Gallery/probe-train = all train videos; queries = val for selection; test scored once after freezing | Protocol `asl_citizen_frozen_class_retrieval_v1` (adapted, not official) | 12.5 |
