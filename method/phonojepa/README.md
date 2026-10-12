# phonojepa — PhonoJEPA (proposal12 v2.1)

PhonoJEPA is skeleton SSL for sign language. It masks **phonological channels** (hand configuration, hand location/movement, face sub-channels) and predicts the representation of the masked channel. The method is specified in [docs/proposal12/PhonoJEPA_Method_Proposal_VI.md](../../docs/proposal12/PhonoJEPA_Method_Proposal_VI.md); see §6 and the user decisions in §17.

Implementation notes and deviations are in [IMPLEMENTATION.md](IMPLEMENTATION.md).

**Status:** implemented and tested (`pytest tests`, 40 tests), with a 60-step GPU smoke run. **No full training run yet.** Runs start only on the user's go.

## Comparison protocol

Every model is trained and evaluated under the same protocol as proposal11 (obs_jepa A/B) and EX16, so PhonoJEPA can be compared directly with SR3D (SignRep reproduction), SRVJ, ASLC002, JV3D, JH3D, A00 and the obs_jepa A/B arms:

| Item | Value |
| --- | --- |
| Pretraining data | 3,000 unlabelled ASL Citizen train videos (ASLC002 corpus, `method/jepa_baseline/runtime/ASLC002/corpus/pretrain.jsonl`) |
| Budget | 4,000 optimizer steps × 32 clips per step; window T = 64 samples at min(fps, 25); 1 seed |
| Optimizer | Same recipe as obs_jepa B: AdamW lr 3e-4, wd 0.05, 10% warm-up, cosine, EMA 0.996 → 0.9999, bf16 |
| Evaluation | Unchanged `method/obs_jepa/oj/evaluate.py`: LOSO-val whitened retrieval, ASLC300 retrieval + true linear probe, body-part probes, signer probe |

PhonoJEPA features use the same window extraction as obs_jepa B: 64 samples every 32. They are written in the same file format.

## Layout

```
pj/joints.py          103-joint layout, 7 channels, joint ownership / dependencies, mirroring, thresholds
pj/data/cache.py      COCO-WholeBody pickles -> pixel arrays (NaN = unobserved) + the ASLC002 training cache
pj/data/window.py     handedness, content-independent windows, speed resampling with source support, augmentation
pj/masks.py           PLCM sampler (R0-R4), budget, face closure, context dropout (content-independent)
pj/data/dataset.py    teacher/student pairs (near, far): mask -> hidden source frames -> scrub -> propagation -> views
pj/features.py        channel features (CONF/LOC/UP/MO/HD), tokens, intrinsic targets, reliability, target statistics
pj/model.py           tokenizers, RoPE transformer encoder (+CLS), predictor (query per target cell), heads, EMA
pj/losses.py          centred CE (S-JEPA), prototype CE, L1+SIGReg (LeJEPA), intrinsic smooth-L1, KoLeo, loss balancer
pj/train.py           training loop; all arms are flags (factorial, baselines, gates, controls)
pj/extract.py         window features in obs_jepa format (readouts _grp [primary], _mean, _cls; descriptor = B0)
pj/evaluate.py        runs obs_jepa's evaluate.py on runtime/ (+ paired stats vs SR3D/JH3D/JV3D/ASLC002/B12/B13)
scripts/schedule.py   sequential runner for the approved plan (P1 -> P2 -> P3, 1 seed); NOT started
tests/test_core.py    leakage, masks, features, model, EMA, arms
```

## Commands

```bash
cd /home/haipd/Sign_Representations/method/phonojepa
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
$PY -m pj.data.cache                       # (done) training cache runtime/cache/aslc002_train.npz, ~3 s
$PY -m pytest tests -q                     # 40 tests
$PY pj/train.py --out runtime/runs/PJ_PLboth --family PL --target both          # MVM cell, ~10-15 min on the shared GPU
$PY pj/extract.py --kind model --ckpt runtime/runs/PJ_PLboth/final.pt --name PJ_PLboth --set val      # and --set aslc300
$PY pj/evaluate.py --names PJ_PLboth_grp,PJ_PLboth_cls --out runtime/eval/one.json
$PY scripts/schedule.py --dry              # print the whole plan
setsid nohup $PY scripts/schedule.py > runtime/logs/schedule.log 2>&1 &         # full plan (only after approval)
```

## Arms (scripts/schedule.py)

| Phase | Runs |
| --- | --- |
| B0 | Training-free descriptor in the same protocol |
| P1 | MVM cell `PJ_PLboth` with each anti-collapse option: centred CE (default), prototypes, L1 + SIGReg with or without EMA. Schedule check: `PJ_PLctx_16k`, `PJ_PLint_16k` (4× steps) |
| P2 | 3 × 3 factorial: mask {PL, R4, R0} × target {ctx, int, both}. Baselines: B-glob, B-MAE, B-MAMP, hands-only, ctx + shuffled-int, random init |
| P3 | Gates G1 (global CLS + KoLeo), G2 (R1 + R4 + curriculum), G3 (morphology). A8(g) uses random handedness instead of content-based handedness |

Cost: the 60-step smoke ran at 0.15 s/step, so a 4k-step run takes ≈ 10–15 min. Extraction is mostly CPU and takes ≈ 10–20 min per model for val + aslc300. All 24 arms should take ≈ 10–14 h wall-clock and ≈ 6–9 GPU-h, within the approved 22–37 GPU-h.
