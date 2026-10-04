# PARTPROBE001 — H005 train-only participant geometry

Prelaunch registration retained; status **completed**, H005/H005_v0/D023/D024.
Primary question/metric/controls/threshold/data access and missing-ID rule fixed
in protocol/hypothesis cards. Fixed native full-activity768D vectors, no policy
trial/encoder adaptation/task-label quality. Train1313 IDs, seed1701,100 null
controls,180s CPU/0GPU, exclusive runtime under frozen_probe; <5MiB evidence.

PARTTEST001 four fixture groups passed, independent manual expected values and
installed public-reference parity; CLI --help smoke passed. Actual code/config
hash, run ID, argv, status/environment, prediction/control/report digests and
terminal state are recorded below. Immutable per-run snapshot is
the replay source; no completed directory reuse or background scheduler.

## Actual run

`PARTPROBE001_20261004_H005_v0_6143bb5b_s1701`, session49778 terminal exit0,
PID2835271/hostptitiec,2026-10-04T08:13:28.525263→08:13:30.753161UTC.
Instrumented CPU wall2.2279215729795396s/GPU0. No new encoder calls.
Config SHA6143bb5bfdb6645c816eb43bdef67719bc19c31dd7da5baf52c59060fdcfba2a.

Actual argv (thread environment OMP/OPENBLAS/MKL=1):

```
/home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/audit_participant_probe.py --config method/frozen_probe/configs/PARTPROBE001.json --seed 1701 --run-id PARTPROBE001_20261004_H005_v0_6143bb5b_s1701 --output-dir method/frozen_probe/runtime/PARTPROBE001_20261004_H005_v0_6143bb5b_s1701
```

Runtime root `method/frozen_probe/runtime/PARTPROBE001_20261004_H005_v0_6143bb5b_s1701`;
sibling `.log` preserves stdout/stderr. Status/config/environment/source_snapshot,
predictions/control scores/report stored. Per-run NumPy2.5.3/scikit-learn1.8.0.
Raw source bytes/native cache unchanged. Source47,594 cached train windows,
1313/1313 source predictions evaluated exactly once,31eval IDs per fold.

| Metric | Fold0 | Fold1 | Primary mean |
| --- | ---: | ---: | ---: |
| Balanced participant recall | .19276667494189306 | .19485966184899903 | .19381316839544604 |
| Micro participant accuracy | .31825037707390647 | .3292307692307692 | not primary |
| Candidate source coverage | .995475113122172 | .9984615384615385 | not primary |
| Train-majority balanced recall | .03225806451612903 | .03225806451612903 | not primary |

100 control means median.03037610982933869, range[.013342085702569149,.04985910012146026],
empirical2.5/97.5percentiles[.015793190090799276,.04840943168650947]. Primary effect
.16343705856610735 (16.343706pp), above registered15pp; candidate coverages pass95%.
No degenerate prototype. All100 controls retained. These percentiles are not CI
or a p-value. Each direction fits31people/evaluates31; the union includes32IDs,
and absent prototypes remain wrong in the denominator.

Report SHA3119b24c461ea3f5b946b930ec5472b48df2faffb2df953dfb50d1d20ac25056;
predictions8002c26ad5acb6a94af8310c4150506c4eef63991a815997f3040f4a843164f8;
control37c7a96efac0d5fb2d58a64f4ff69f849b592d26514851460366278e006d63f3.
`PARTPROBE001_verification.json` independently recounts all prediction IDs,
truth mapping/denominators and fold scores; all output/registered input digests
match. This is a count verifier, not an independent research review/rerun.

Initial registration helper overchecked all CACHE005 historic Python snapshots
including unrelated later-versioned run_baseline.py; failed before config/runtime
creation. Narrowed to seven immutable frozen-core bindings plus actual diagnostic
dependencies, all unchanged, then registered and launched once. Incident kept
in PARTPROBE001_registration.json; no failed scientific attempt or hidden retry.

Valid **exploratory participant-association diagnostic**, not harmful-nuisance
proof, retrieval quality, hubness, linguistic invariance or new method. Glosses
held out, people reused; camera/session/background or legitimate signing variation
can explain association. No val feature/test access, no H002 tuning. D024 continues
only to a preregistered task-harm test with known-method/matched controls.
