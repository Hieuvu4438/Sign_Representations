# ASSET001 — H003 prerequisite audit

Actual run `ASSET001_20261004_H003_v0_0395b8eb_s1701`; status **failed**, wall 0.318064810s, recorded GPU 0.000000000s.
Config SHA `0395b8eb2daf4269f59129ef3c5e381e4c6c5a85bf10ed297f958707eb31fca0`; config/environment/source snapshots and outputs in `method/stream_reliability/runtime/ASSET001_20261004_H003_v0_0395b8eb_s1701`.

Input/software/provenance diagnostic only; no scientific task metric.

```bash
/home/haipd/miniconda3/bin/python3 method/stream_reliability/scripts/audit_assets.py --config method/stream_reliability/configs/ASSET001.json --seed 1701 --run-id ASSET001_20261004_H003_v0_0395b8eb_s1701 --output-dir method/stream_reliability/runtime/ASSET001_20261004_H003_v0_0395b8eb_s1701
```

Retained HTTP404: models were requested under the demo-space endpoint. ASSET002 retry1 corrects to the separate author model repository identified by pinned author app; no local weight changed.

Decision/limitations: [D016](../decisions/D016.md). H003 scientific pilot remains hold; no test access or new search trial.
