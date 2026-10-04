# NUMCPU001 — H003 registered software diagnostic

Run `NUMCPU001_20261004_H003_v1_0e5eeffb_s1701`; status **completed**, wall3.114718352s / GPU0.000000000s. Config0e5eeffb925a550a3403f68a797077c3e226369c19dfa059ddd5029c023f6230. Original config/source/environment/status/output/error retained at `method/stream_reliability/runtime/NUMCPU001_20261004_H003_v1_0e5eeffb_s1701`. Diagnostic only; no task scores, HPO or test quality. See D018/D019 for gates and original CPU/GPU failure.

Actual invocation:

```bash
/home/haipd/Sign_Representations/.venv-shubert-native/bin/python method/stream_reliability/scripts/audit_numerical_cpu.py --config method/stream_reliability/configs/NUMCPU001.json --seed 1701 --run-id NUMCPU001_20261004_H003_v1_0e5eeffb_s1701 --output-dir method/stream_reliability/runtime/NUMCPU001_20261004_H003_v1_0e5eeffb_s1701
```

Report: [NUMCPU001_cpu_precision_diagnostic.json](../experiments/NUMCPU001_cpu_precision_diagnostic.json). Large referenced RAM payloads are ephemeral; hashes and fresh-output rebuild path are retained. Completed NUMDIAG001/NUMCPU001 do not accept the failed STREAMFEAT001 gate. AUTHORGPU001 has only additional native-CUDA producer compatibility.
