# S5 pre-implementation review

Authority: main §§6–8 equations11–13, supplement A3/A5/A6, referenced author repos.
Frozen retrieval uses 16-frame stride2 windows without padding an incomplete tail;
videos shorter than16 require an explicit policy, default reject (U06). Activity weight
is max(LH,RH) from the released raw head; denominator zero/nonfinite fails (U08).
Cosine normalizes **after** pooling; rank training videos, not class prototypes. Report
per-instance and macro per-class top1/top5; for ASL-Citizen also rank distinct labels
by their best dictionary-video score before DCG/MRR, with upstream metric cross-check.
Gallery/query IDs must be disjoint; every query class must exist in the gallery.

Class-distribution table is constructed from training-only normalized frozen features.
Use exact average of all cross-sample cosine similarities per class pair, include
self-similarity explicitly (U09); class mean products are algebraically equivalent.
Per-row grid .001:.001:.100; select feasible maximum probability nearest but <.5;
fail if no grid value meets bound. CE smoothing .1 plus .2*phi[c,c]*KL(phi[c]||softmax).
Do not renormalize away diagonal or select temperature with validation/test data.

Recognition reuses checked 64-frame inflation; optimize only encoder/projection and
new linear classifier, removing prior decoder/activity from the trainable path.
Temporal uniform sampling/padding is explicit U06 because author sampling is absent.
Validate numerical KL, train-only table provenance, real backward and checkpoint load.

Translation sources are pinned in evidence/translation_sources.json: Sign2GPT
55559355772fd074f73c39e32513af3468613b99; Tarrés Fairseq fork
3260d05f4804d3611a911adf3fab8cc50d8204cd. Frozen stride2 features require adapters
for original dataset/model interfaces, skip pseudo-gloss stage entirely, retain each
downstream recipe. No random LLM smoke or exported features prove BLEU reproduction.
Check source schemas and original components independently; record environment limits.
