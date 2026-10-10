# obs_jepa — observability-aware JEPA for sign representation (proposal11)

Implementation of the three pipelines of `docs/proposal11/JEPA_Sign_Language_Critical_Review_End_to_End_v2.md`,
trained under the **same protocol as the SignRep reproduction** (no released SignRep weights are used for training):
same 3,000 unlabelled ASL Citizen train videos (ASLC002 corpus), same evaluation code for every model.

| Proposal | Code | Inference input |
| --- | --- | --- |
| **A** RGB-JEPA (A-R / A-D recipes; generic, coverage-matched generic, partial-region, whole-articulator, jittered masks; uniform or part-balanced loss) | `oj/train_rgb.py`, `oj/models/rgb.py`, `oj/masks/rgb_masks.py`, `oj/data/rgb_dataset.py` | RGB only |
| **B** graph-JEPA on 67 landmarks (B10 raw, B11 S-JEPA centred-CE, B12 latent L1, B13 latent + observable anchors; core / whole / generic masks) | `oj/train_graph.py`, `oj/models/graph.py`, `oj/features/geometry.py`, `oj/masks/pose_masks.py`, `oj/data/pose_dataset.py` | 2-D landmarks |
| **C** A + frozen-B visible geometry distillation (C-observed, C-reserved masks) | `oj/train_rgb.py --b-teacher ... --c-reserve` | RGB only |
| Shared evaluation (LOSO-val whitened retrieval; ASLC300 retrieval + true linear probe; body-part / signer probes) | `oj/extract.py`, `oj/evaluate.py` | — |

Details, proposal-section mapping, deviations and verification status: [IMPLEMENTATION.md](IMPLEMENTATION.md).
Training plan: [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md).

```bash
cd /home/haipd/Sign_Representations/method/obs_jepa
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
$PY -m pytest tests -q                                    # 7 implementation checks (§11.6)
$PY oj/train_rgb.py   --out runtime/A13 --recipe AR --policy partial --weighting balanced --steps 4000 --batch 32
$PY oj/train_graph.py --out runtime/B13 --objective anchors --steps 4000 --batch 64
$PY oj/train_rgb.py   --out runtime/C1 --recipe AR --policy partial --b-teacher runtime/B13/final.pt --c-reserve
$PY oj/extract.py --kind rgb --ckpt runtime/A13/final.pt --weights student --name A13 --set val      # and --set aslc300
$PY oj/evaluate.py --names A00,A13,SR3D --ref SR3D --out runtime/eval.json
```
Long runs: launch with `setsid nohup ... &`; every trainer saves a resumable `ckpt.pt` (`--resume`).
