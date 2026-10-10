# SIGNREL temporal inference model card

Status: research pilot, not a deployed sign recognition or translation system. Consult results/research_report.md and docs/STATE.md for measured run status; no unsupported performance claims.

The encoder receives synchronized cached LH/RH DINOv2 original ViT-S/14 vectors (384D each) and seven normalized 2D upper-body landmarks with coordinate validity. Four pre-LN joint Transformer blocks (width256, 8heads) use anatomical stream embeddings and sinusoidal physical time. SSL reconstructs masked hand/body states; A4 adds sparse signed cross-hand cosine targets. State decoders are discarded from exported encoder.pt.

Input contract: an audited `.npz` recording and JSON metadata, preprocessing and original teacher fingerprints consistent with training. No gloss, text, phonological labels or signer ID enters the encoder. The cache is not raw RGB: MediaPipe and frozen DINO are required for a new raw video. No interpolation supplies hidden hand appearances.

Output: fixed 2304D normalized clip embedding; overlapping context windows merge by canonical frame and stream, then whole and physical half-time means are normalized separately and concatenated. Samples with no valid hand observation return insufficient_observation; missing streams have zero pooling blocks. This coverage convention is part of the evaluation protocol.

Intended use: compare state-only and relation learning under fixed small-data controls, explore frozen dictionary retrieval, inspect detector/target limitations. It does not output linguistic phonology or identify an unseen gloss without an exemplar dictionary. No claim of signer-invariant representation, SSL signer-disjoint training, broad generalization, accessibility readiness or real-time use.

Data: deterministic 47 How2Sign segmented clips train / 3 held-out SSL dev; no captions read. Existing segment boundaries contribute a segmentation prior. Custom 40-gloss ASL Citizen official train gallery K=1, 80 official validation queries and separately locked 80 test queries. Query-gallery signer disjointness is audited; continuous SSL signer identity and lexical overlap are unknown.

Limitations: tiny corpus and gallery; detector misses/landmark hallucination and crop errors remain; generic DINO similarity may not encode phonological coordination; face/mouth absent; low-rank diagnostics on three SSL-dev videos are sample limited; sparse pair controls have differing eligible-pair counts/compute. A positive pilot is insufficient for a paper-level efficacy claim. Natural minimal-pair annotation is unavailable.

Provenance/licensing: artifacts/sources.lock.json, third_party/NOTICE.md, per-run source archive and full hashes. Restricted local videos and caches are not redistributed. ASL Citizen local terms are read and hashed in artifacts/dataset_licenses.json; videos and derived caches remain local. Model distribution remains subject to training-data and upstream asset terms; this card makes no licensing grant.
