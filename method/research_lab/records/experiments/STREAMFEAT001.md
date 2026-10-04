# STREAMFEAT001 — H003 registered software diagnostic

Run `STREAMFEAT001_20261004_H003_v1_c525a0ad_s1701`; status **failed**, wall7.040762268s / GPU6.602126127s. Configc525a0adc6a625d38a99da301f12142d38e4126f5d33df7e69b757a65bc7594b. Original config/source/environment/status/output/error retained at `method/stream_reliability/runtime/STREAMFEAT001_20261004_H003_v1_c525a0ad_s1701`. Diagnostic only; no task scores, HPO or test quality. See D018/D019 for gates and original CPU/GPU failure.

Actual invocation:

```bash
/home/haipd/Sign_Representations/.venv-shubert-native/bin/python method/stream_reliability/scripts/extract_stream_smoke.py --config method/stream_reliability/configs/STREAMFEAT001.json --seed 1701 --run-id STREAMFEAT001_20261004_H003_v1_c525a0ad_s1701 --output-dir method/stream_reliability/runtime/STREAMFEAT001_20261004_H003_v1_c525a0ad_s1701
```

Report: [STREAMFEAT001_feature_smoke.json](../experiments/STREAMFEAT001_feature_smoke.json). Large referenced RAM payloads are ephemeral; hashes and fresh-output rebuild path are retained. Completed NUMDIAG001/NUMCPU001 do not accept the failed STREAMFEAT001 gate. AUTHORGPU001 has only additional native-CUDA producer compatibility.
