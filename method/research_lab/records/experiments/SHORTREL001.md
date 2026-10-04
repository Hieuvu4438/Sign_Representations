# SHORTREL001 — one short-source official archive equality

Bound original range-reader source/local byte SHA,32MiBtotal range cap/1MiB
member cap,300s CPU/network deadline. Verify exactly one contract-failing source
against official Microsoft archive with exact HTTP206ranges/ETag/ZIPCRC. No
source replacement/wholearchive fetch/restricted video export. Terminal4.446711s,
10759569B downloaded; official92741Bmember equals local SHA. Config digest
8238cee75836a5d60cbcd76bc4a3f71000bbcc719b3e7d11fc9270cdf097efde
stored; diagnostic run name omits hash prefix, authoritative argv/config digest
and immutable status retained rather than renaming historical evidence.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/verify_short_source_release.py --config method/frozen_probe/configs/SHORTREL001.json --seed 1701 --run-id SHORTREL001_20261004_H002_P004v1_s1701 --output-dir method/frozen_probe/runtime/SHORTREL001_20261004_H002_P004v1_s1701
```

Result is one-source equality only; [D014](../decisions/D014.md). Existing bounded
archive-reader helper snapshot/sourceSHA retained alongside ordinary own code
snapshot. No test task performance claim or new recipe.
