# INPUTPARITY001 — H003 registered software diagnostic

Run `INPUTPARITY001_20261004_H003_v1_3700038a_s1701`; status **completed**, wall10.332137229s / GPU0.000000000s. Config3700038a63762059667e6b8e2b83234e4b9176c8cb9712aa152a5f688fa56563. Original config/source/environment/status/output/error retained at `method/stream_reliability/runtime/INPUTPARITY001_20261004_H003_v1_3700038a_s1701`. Diagnostic only; no task scores, HPO or test quality. See D018/D019 for gates and original CPU/GPU failure.

Actual invocation:

```bash
/home/haipd/Sign_Representations/.venv-shubert-native/bin/python method/stream_reliability/scripts/audit_input_parity.py --config method/stream_reliability/configs/INPUTPARITY001.json --seed 1701 --run-id INPUTPARITY001_20261004_H003_v1_3700038a_s1701 --output-dir method/stream_reliability/runtime/INPUTPARITY001_20261004_H003_v1_3700038a_s1701
```

Report: [INPUTPARITY001_input_parity.json](../inventory/INPUTPARITY001_input_parity.json). Large referenced RAM payloads are ephemeral; hashes and fresh-output rebuild path are retained. Completed NUMDIAG001/NUMCPU001 do not accept the failed STREAMFEAT001 gate. AUTHORGPU001 has only additional native-CUDA producer compatibility.
