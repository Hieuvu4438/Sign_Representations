# COHORT002 — disjoint vocabulary sparse-gallery inventory

D011 proposed prospective1example/class replication. Train-only class selection seed1704 excluding original100classes; same fixedsupportseeds1701/2/3, allval metadata, no newtestaccess, no model/GPU/score. Abort missingclass/source/leakage/byte duplicates with no reselection. SourceSHA/scopedparticipant contracts and10GiBread/120s CPU bound. This inventory does not freeze/validate new mainmeasurement.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/prepare_replication_cohort.py --config method/frozen_probe/configs/COHORT002.json --seed 1704 --run-id COHORT002_20261004_P004v0_s1704 --output-dir method/frozen_probe/runtime/COHORT002_20261004_P004v0_s1704
```
