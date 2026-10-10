# Search log G — novelty check for H012 (rest-pose pseudo-signer clusters, cross-cluster mining)

Date 2026-10-07. Engine: WebSearch (US). Snippet/abstract level (DISCOVERED / ABSTRACT_READ).

| # | Query | Kept | Note |
| --- | --- | --- | --- |
| G1 | unsupervised sign language representation pseudo signer clustering cross-signer positive mining | Cross-Modal Consistency Learning for SLR 2503.12485 (Semantic Positive Mining: top-k similar samples per modality as pseudo-positives, RGB+pose) ; SignDino 2609.06296 | positive mining exists in sign SSL, but not signer-aware; no pseudo-signer inference found |
| G2 | camera-aware proxies unsupervised person re-identification inter-camera contrastive | CAP 2012.10674 (AAAI 2021: split clusters into camera-aware proxies; inter-camera positives; **camera IDs given**); ICE 2103.16364; camera-aware style separation 2112.10089; inter-camera identity discrimination (TOMM 2024) | the mechanism "prefer cross-domain positives in unsupervised mining" is established in re-ID with known camera labels |

## Conclusion (INFERENCE, scope G1–G2, 2026-10-07)

Cross-domain positive mining is not new (re-ID with camera IDs). Not found in scope: inferring the nuisance domain (signer/session) label-free from the *rest segments of the sign video itself* and using it to mine cross-signer positives for sign representation learning. If H012 works, the defensible contribution is a sign-specific, metadata-free nuisance-domain estimate plus measured cross-signer precision gains, positioned against CAP-style camera-aware mining. Blind spots: speaker-aware unsupervised speech representation (speaker-invariant HuBERT variants, e.g. ContentVec) not yet searched — this is the closest analogue in speech and must be searched before any claim.

| G3 | ContentVec speaker disentanglement self-supervised speech representation speaker-invariant HuBERT | ContentVec 2204.09224 (ICML 2022: speaker disentanglement in teacher via voice conversion to a canonical speaker, in student via speaker-perturbation contrastive pairs, speaker conditioning with a pretrained speaker embedding); SPIN (Chang et al., Interspeech 2023, speaker-invariant clustering fine-tuning) | speech analogue of "teacher dominated by speaker identity" is established; ContentVec's "disentangle the teacher" is the speech counterpart of H010/H012's diagnosis. Sign has no voice-conversion equivalent; the rest-segment nuisance estimate is the sign-specific substitute |

Update to conclusion: the problem framing ("SSL teacher targets entangled with speaker/signer identity → disentangle the teacher") must cite ContentVec/SPIN as the speech precedent. The sign-specific contribution, if any, is how the nuisance is estimated (rest segments, no signer labels, no generative conversion) and measured (cross-signer same-gloss neighbour precision).
