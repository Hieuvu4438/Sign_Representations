# Pilot diagnostics for proposal12

All diagnostics here are CPU-only and training-free. They run on the local ASL Citizen COCO-WholeBody 2-D keypoints. No SSL model was trained. The proposal (§5) cites **only the v2 results**.

## v2 (current; written after the independent review)

| Script | Result | Content |
| --- | --- | --- |
| `diag_v2.py` | `diag_v2_results.json` | D1/D5′, D2′, D4, D6 on the revised tokenisation: hand = CONF + LOC; face = UPPER, MOUTH, HEAD. Changes one factor at a time. Includes a gloss-balanced signer probe, OFF-val retrieval, the finger-bone predictability ladder, and the cross-channel leak audit. |
| `diag_v2b.py` | `diag_v2b_results.json` | D7: CONF in palm frame vs body/image frame, with and without per-bone p90 normalisation |
| `diag_v2c.py` | `diag_v2c_results.json` | D7b: per-frame vs clip-median shoulder anchors |
| `diag_v21.py` (+ `aspect_map.py`) | `diag_v21_results.json` | **D8 (v2.1, after the second review), authoritative for frame choice, B0 and the leak audit.** Adds pixel-aspect correction from per-video W/H, a body frame from the median shoulder angle, the chin moved to MOUTH, relative-only HEAD, and a per-column leak audit (pooled and within-clip). B0 = 13.46. |
| `diag6_split_test.py` | `D6_split_test_results.json` | D6′: leak audit with the hand split into SHAPE (finger bones) and ORI (palm polygon). Writes `diag6_results.json` (copied here under a new name). It motivated merging them into CONF. |

How v2 fixes the v1 problems:
- **No near-constant columns.** In v1 the middle-MCP column in the palm frame acted as a 1/palm-length proxy, i.e. a hand-size feature.
- **Floor instead of dropping frames.** The palm length gets a numeric floor of 0.1·σ instead of dropping foreshortened frames.
- **Correct face mirroring.** Mirroring now applies the iBUG left/right permutation.
- **Robust leak audit.** Frames are filtered for shoulder and face detection failures, and features are winsorised at the train 1st/99th percentiles.

## v1 (superseded; kept for transparency)

`diag.py`, `diag3.py`, `diag4.py`, `diag5.py` and `D1_D2_results*.json`, `D3_results.json`, `D4_results.json`, `D5_results.json` are the first-round diagnostics.

The reviewer showed that they contain a numerical artifact and confounded comparisons (see proposal §18, C3). **Do not cite them.** The v1 claim "bone normalisation helps" was withdrawn, and the B0 value 16.84 was not reproducible without the artifact and hand-size features.

## Reproduce

```bash
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
OUT=<scratch dir>
$PY -I load_kp.py val 0 $OUT/val.pkl
$PY -I load_kp.py train 0 $OUT/train_full.pkl
cp diag_v2.py diag_v2b.py diag_v2c.py $OUT/
cd $OUT
$PY -I diag_v2.py $OUT          # ~40 min on 8 threads
$PY    diag_v2b.py $OUT         # imports diag_v2.py from the same folder
$PY    diag_v2c.py $OUT
```

`diag_v2.py` also needs `$OUT/train14k.pkl`, created with `load_kp.py train 14000`, if you run the v1 scripts.

Caveats:
- One random seed.
- Probes are linear (logistic or ridge) or kNN on hand-crafted descriptors.
- The signer probe has 5 classes, or 3 classes in the gloss-balanced version.
- Pixel aspect was **not** corrected in v1/v2 (keypoints are x/W, y/H). It **is** corrected in `diag_v21.py` (D8) using `aspect.json` from `aspect_map.py`.
- `load_kp.py` now stores the video `stem`, which `diag_v21.py` needs.
