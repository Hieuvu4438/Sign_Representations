# AUTHORGPU001 — H003 registered software diagnostic

Run `AUTHORGPU001_20261004_H003_v2_ac538b19_s1701`; status **completed**, wall4.501499658s / GPU3.883341263s. Configac538b19cfb1fa3d89b562d587b85be2034a5492f9486eaabd94444b8299e8a6. Original config/source/environment/status/output/error retained at `method/stream_reliability/runtime/AUTHORGPU001_20261004_H003_v2_ac538b19_s1701`. Diagnostic only; no task scores, HPO or test quality. See D018/D019 for gates and original CPU/GPU failure.

Actual invocation:

```bash
/home/haipd/Sign_Representations/.venv-shubert-native/bin/python method/stream_reliability/scripts/audit_author_gpu.py --config method/stream_reliability/configs/AUTHORGPU001.json --seed 1701 --run-id AUTHORGPU001_20261004_H003_v2_ac538b19_s1701 --output-dir method/stream_reliability/runtime/AUTHORGPU001_20261004_H003_v2_ac538b19_s1701
```

Report: [AUTHORGPU001_author_gpu_parity.json](../experiments/AUTHORGPU001_author_gpu_parity.json). Large referenced RAM payloads are ephemeral; hashes and fresh-output rebuild path are retained. Completed NUMDIAG001/NUMCPU001 do not accept the failed STREAMFEAT001 gate. AUTHORGPU001 has only additional native-CUDA producer compatibility.
