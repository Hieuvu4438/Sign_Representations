# PLctx full160: frozen dictionary retrieval

## Material Passport

Experiment: PLctx-full160-frozen-SignRep-retrieval. Frozen final checkpoint, official train galleries
and test queries. No classifier, PCA, fine-tuning or WLASL pretraining. No external data upload.

Primary readout `_grp` was fixed before test scoring. Secondary `_cls`/`_mean` results are retained
in the CSV and JSON; they are not used to select a test-optimal configuration.

| Dataset | Gallery train | Query test | DCG ×100 | Rec@1 % | Rec@5 % |
|---|---:|---:|---:|---:|---:|
| ASL Citizen | 40154 | 32941 | 39.62 | 16.68 | 36.90 |
| WLASL2000 | 14296 | 2879 | 30.55 | 9.62 | 24.42 |

## Protocol and scope

[SignRep §6, §7.2 and Table 4](https://arxiv.org/html/2503.08529v1#S7.SS4): raw window mean -> video L2 -> cosine; class score is
maximum gallery cosine; rank counts distinct classes with pessimistic ties. DCG is 1/log2(1+rank).

PhonoJEPA uses its trained 64-frame skeleton encoder at min(fps,25), stride 2, native short-clip validity
masks and uniform pooling. SignRep uses 16 RGB frames and can additionally use its trained activity head.
These are declared model/input differences; this is the same frozen dictionary ranking protocol,
not a claim of identical feature extraction. WLASL videos are the local pretrimmed release.

PLctx pretraining uses all 40,154 ASL Citizen train clips, 160 epochs, one seed. This data budget differs
from both the earlier 3,000-clip pilot and the SignRep public checkpoint's YouTube-SL-25 pretraining.

Published Table 4 values and existing local SignRep/JH3D/JV3D/SR3D measurements are separate CSV rows.
ASL Citizen test has 32,941 queries; earlier LOSO validation and ASLC300 are different evaluations.

## Earlier pilot-compatible validation

LOSO whitened Rec@1: 29.54%; ASLC300 whitened Rec@1: 47.67%; linear probe macro Top-1: 68.87%.
