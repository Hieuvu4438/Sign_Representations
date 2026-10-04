# SignRep activity contract audit

Primary [paper](https://arxiv.org/html/2503.08529v1), accessed2026-10-04 local.
§3 activity targets are two hand indicators; §4 names output2; §5.1 explicitly
uses smooth-L1 for activity as well as priors. §6 Eq11/12 uses maximum hand
activity for normalized weighted aggregation. Appendix A2 excludes WLASL
YouTube IDs; A3 selects checkpoint via WLASL dev retrieval. These do not
establish ASL Citizen video/pretraining disjointness.

Local read-only upstream model has Linear768→128,GELU,Linear128→2 and a
commented-out sigmoid; forward returns raw `cls_active`. Source/hash captured
in FEAT001 report. Existing ten-window outputs span[-.141660,1.015702].
This diagnostic input-dependent range is not calibration/recognition evidence.

Inference/engineering decision for upcoming adapted pilot: treat outputs as
regression estimates, use max(clamp(raw,0,1)) per window; if sum≤1e-12, return
unweighted mean and record fallback. This bounded nonnegative convention
reflects target domain, but public native weighted-pooling implementation is
unavailable in the inspected repo: **not exact paper reproduction**.
No sigmoid/softmax conversion, threshold search or pose supervision added.
Raw outputs remain available for audit. Contract changes belong to P001_v3.
Feature key `features` matches post-LayerNorm/linear z_avg in §4; `latent` is
not interchangeable. Choose features before any performance feedback.
