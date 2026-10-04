# NUMDIAG001 — H003 registered software diagnostic

Run `NUMDIAG001_20261004_H003_v1_2986835f_s1701`; status **completed**, wall7.198581601s / GPU6.638495895s. Config2986835f63641188283ad09d66c0995bd3834d38a026478dccb70b1b3674424e. Original config/source/environment/status/output/error retained at `method/stream_reliability/runtime/NUMDIAG001_20261004_H003_v1_2986835f_s1701`. Diagnostic only; no task scores, HPO or test quality. See D018/D019 for gates and original CPU/GPU failure.

Actual invocation:

```bash
/home/haipd/Sign_Representations/.venv-shubert-native/bin/python method/stream_reliability/scripts/extract_stream_smoke.py --config method/stream_reliability/configs/NUMDIAG001.json --seed 1701 --run-id NUMDIAG001_20261004_H003_v1_2986835f_s1701 --output-dir method/stream_reliability/runtime/NUMDIAG001_20261004_H003_v1_2986835f_s1701
```

Report: [NUMDIAG001_numerical_diagnostic.json](../experiments/NUMDIAG001_numerical_diagnostic.json). Large referenced RAM payloads are ephemeral; hashes and fresh-output rebuild path are retained. Completed NUMDIAG001/NUMCPU001 do not accept the failed STREAMFEAT001 gate. AUTHORGPU001 has only additional native-CUDA producer compatibility.
