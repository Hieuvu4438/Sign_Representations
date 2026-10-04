# STREAMGPU001 — H003 prerequisite audit

Actual run `STREAMGPU001_20261004_H003_v0_aeb2c96d_s1701`; status **completed**, wall 3.325681433s, recorded GPU 2.900944370s.
Config SHA `aeb2c96db748e4a576eb1f1610d4455050a2e8c3340004d50ecf41a583d6591c`; config/environment/source snapshots and outputs in `method/stream_reliability/runtime/STREAMGPU001_20261004_H003_v0_aeb2c96d_s1701`.

Numerical software smoke only, authorized by D016; H003 method/pilot remains held

```bash
/home/haipd/Sign_Representations/.venv-shubert-native/bin/python method/stream_reliability/scripts/audit_gpu_parity.py --config method/stream_reliability/configs/STREAMGPU001.json --seed 1701 --run-id STREAMGPU001_20261004_H003_v0_aeb2c96d_s1701 --output-dir method/stream_reliability/runtime/STREAMGPU001_20261004_H003_v0_aeb2c96d_s1701
```

All four synthetic outputs pass registered np.allclose(atol=rtol=1e-4); largest absolute error0.010467529 on large-magnitude author layer-last output. This is combined absolute+relative tolerance, not an absolute1e-4 error bound. Peak tensor memory368437760B; allocated GPU2.900944370s. No crop/video/benchmark parity claimed.

Decision/limitations: [D016](../decisions/D016.md). H003 scientific pilot remains hold; no test access or new search trial.
