# Source reuse

| Source | Use | Changes |
|---|---|---|
| Vendored SignRep (`third_party/SignRep`, recorded upstream rev `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2`) | `Model`, `ProjectionModel`, `mae_hiera_base_16x224` imported read-only via `vendor.py`; constructor literal copied from `example_usage.py` | None to vendor files. Per-file SHA-256 recorded in every run's preflight/initial-state audit. |
| PyTorch `torch.fft.rfft`, `torch.hann_window`, `Tensor.unfold` (docs R3-R5, torch 2.11) | Spectral core | Native PyTorch; no external FFT/STFT code was copied. `torch.stft` not used. |
| Hiera `get_random_mask` semantics | Re-implemented as `make_explicit_mask` (same keep/remove polarity) with a CPU generator so every arm sees identical masks | Logic mirrors the vendored function (Meta license header in vendor). |

No code from other `method/*` directions or `artifacts/legacy` was read or imported.
Focal Frequency Loss / audio STFT losses were not reused (conventions differ from `[B,T,D]` pose signals).
