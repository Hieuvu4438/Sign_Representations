# Search log F — novelty check for H010/H011 (temporal-difference teacher) and H008 (set→vector distillation)

Date 2026-10-07. Engine: WebSearch (US); abstracts verified from export.arxiv.org with curl (abstract level only, ABSTRACT_READ).

| # | Query | Kept | Note |
| --- | --- | --- | --- |
| F1 | temporal difference of clip embeddings signer-invariant sign language retrieval | CLIP2Video 2106.11097 (Temporal Difference Block over adjacent CLIP frame embeddings, *learned*, text-video retrieval); SignSeek 2609.03695 (already P112, supervised same-gloss contrastive, cross-signer dictionary retrieval); Kinematics-centric CSLRet 2610.05306 (SMPL-X motion space to reduce appearance reliance, gloss-span weak supervision, sign-text retrieval) | none of these is label-free on frozen RGB sign windows |
| F2 | sign language recognition feature difference between frames remove signer appearance self-supervised | SignDino 2609.06296 (temporal DINO self-distillation over frozen DINOv3 articulator crops; learns temporal evolution of articulator states); signer-removal CSLR (Zuo et al., TOMM 2024, disentanglement with signer labels) | SignDino is the closest SSL "learn only temporal dynamics over frozen features" design, but it trains a temporal encoder with DINO, not distillation from a training-free set-matching teacher |
| F3 | first-order temporal difference of frozen video features appearance-invariant retrieval chamfer late interaction video | **SemanticMoments 2602.09146** (training-free temporal higher-order moments over pretrained semantic features for motion-centric retrieval; argues pretrained features are appearance-biased); Video-ColBERT 2503.19009; ViSiL; TCA | SemanticMoments is the closest prior art to the H010 *readout* signal: "temporal statistics of frozen features reduce appearance bias" is published (generic video, not sign, not signer-level) |
| F4 | "set-to-vector" distillation chamfer teacher student video embedding | DnS (Distill-and-Select, fine-grained teacher → coarse student for video retrieval; already cited in log E); FGD 2212.10423 (multi-granular → single-vector distillation, documents); TeachCLIP 2308.01217; EmbedDistill 2301.12005 | multi-vector → single-vector distillation is established; H008's novelty is not the mechanism |

## Conclusion (INFERENCE, scope A–F, 2026-10-07)

- Temporal differences / temporal moments of pretrained features as an appearance-cancelling motion descriptor: **not new** (CLIP2Video TDB learned; SemanticMoments training-free).
- Fine-grained (set) teacher → single-vector student distillation: **not new** (DnS, FGD, TeachCLIP).
- Not found in scope: (i) a measurement that label-free neighbours of a sign encoder are signer-dominated and that appearance-cancelling similarities shift them toward cross-signer same-gloss pairs; (ii) label-free set→vector distillation for sign dictionary retrieval reported with signer-cluster statistics. These are measurement/setting contributions, not mechanism novelty.
- Required control added because of F3: a SemanticMoments-style training-free readout (temporal mean/std/3rd-moment of frozen SignRep windows) must be compared to T1/T4 on val before any claim that temporal differences are the right appearance-cancelling statistic (→ explore_moments, exploratory).
