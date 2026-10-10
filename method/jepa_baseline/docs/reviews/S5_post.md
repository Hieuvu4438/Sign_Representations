# S5 post-implementation review

Downstream mathematical tests PASS (`S5_downstream_pytest.txt`): weighted raw activity
pooling, zero denominator failure, nearest training-video label, DCG/MRR against the
exact pinned Microsoft function, all-sample inter-class cosine equivalence, temperature
feasibility and KL direction/diagonal weighting with independent scalar arithmetic.
Class distribution is training-only with manifest and encoder pins; unsupported
temperature bound is reported, never silently relaxed.

Actual released SignRep encoder produces11 stride2 windows for36 frames; inference
does not consume RNG. A real64-frame backward/AdamW update and recognition checkpoint
restore PASS (`S5_rgb_smoke.json`), with prior decoder/activity removed. This uses a
synthetic3-class probability table and short-video padding, not benchmark accuracy.

Actual upstream Sign2GPT MetaFormer and modified XGLM/adapters execute forward,
backward and generation with real frozen SignRep features (`S5_sign2gpt_smoke.json`).
The random tiny XGLM is a technical interface oracle, not a pretrained language model.
Review caught a dtype mismatch in the initial smoke: upstream casts frozen weights
to BF16, so the test was corrected to use the original trainer's CUDA autocast context.
No original module was changed. Original Trainer and text-only callback import PASS.

Actual upstream How2Sign SignFeatsDataset,6-encoder/3-decoder256-dim transformer,
Adam update and Fairseq beam generation PASS (`S5_how2sign_smoke.json`). Adapter changes
only input projection width1024→768. Public best YAML is retained verbatim; transport
tag i3d is documented as loader routing, not the feature modality.

Dependencies isolated in runtime/translation_env, historical version equivalence U10.
No translation datasets/tokenizers or pretrained1.7B LLM were trained here; no BLEU
claims. Production launch instructions use the original trainers and recipes.

Decision: downstream modules and original-component integration PASS;
benchmark/historical numerical equivalence UNVERIFIED.
