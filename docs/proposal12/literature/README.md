# Literature review for proposal12

Four reports were compiled on 2026-10-11 by research sub-agents and checked by the main session. They are in English.

| File | Scope |
| --- | --- |
| `L1_jepa_paradigms.md` | JEPA family, anti-collapse (EMA, SIGReg, prototypes), theory (Littwin, Sobal), probing protocols, design rules for a skeleton JEPA |
| `L2_masking_structured.md` | Random vs structured masking: images, video, graphs, points, skeleton, sign. Theory (Kong et al.), leakage pitfalls, design rules for anatomical masking |
| `L3_skeleton_ssl.md` | Skeleton SSL state of the art (S-JEPA internals, MAMP, SLiM, Path-JEPA, AMR, GFP…), structured-vs-random and latent-vs-coordinate evidence |
| `L4_sign_ssl.md` | Sign SSL and pose ISLR (SignBERT+, BEST, MASA, STC, SHuBERT, SignSeek, SignDino, SignRep internals, Uni-Sign…), benchmark tables, phonology, signer invariance |

Tags:
- `[V]`/`[VR]`: grepped from the raw paper text.
- `[V*]`: from a citing paper's table.
- `[S]`/`[FS]`: from a search summary only.
- `[K]`/`[mem]`: background knowledge.

Raw texts were kept in the session scratchpad and are not copied here.

Known disagreements between reports, resolved in the proposal:
- **"SLim (NeurIPS 2026)".** L4 could not find it. L1 and L3 identified it as **SLiM** (Do et al., KAIST, arXiv 2603.10648, code github.com/KAIST-VICLab/SLiM). Its NeurIPS 2026 venue comes from the authors' project page; the arXiv text does not state a venue.
- **Path-JEPA.** It is ECCV 2026 (LNCS 17054, DOI 10.1007/978-3-032-37258-1_7). No arXiv version and no code are available yet. Its numbers come from the authors' project page.
