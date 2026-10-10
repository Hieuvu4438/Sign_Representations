# S4 post-implementation review

Four data/optimizer tests PASS (`evidence/S4_data_pytest.txt`). Official Planckian jitter
is copied byte-for-byte with license/source hash; video views use shared transform
parameters. Inference preprocessing calls the actual author implementation and matches
bitwise. WLASL dedup covers all splits and URL variants; sampler resumes permutations.
Layer decay covers each trainable parameter once; schedule endpoint checks PASS.

Two different real videos, four views: actual encoder/discriminator backward and Adam
updates PASS (`evidence/S4_training_smoke.json`). Interrupted-step replay reproduces the
next sampled frames, augmented tensors, every encoder/discriminator/EMA tensor, and all
metrics bitwise. Test-score selection is rejected; validation selection saves a best
checkpoint. Pinned official Kinetics MAE checkpoint was downloaded and strictly loads
all 298 encoder tensors, dropping only 110 original MAE decoder/head tensors
(`evidence/S1_kinetics_init.json`, `evidence/kinetics_weights.json`).

Technical smoke used the released SignRep checkpoint as initialization, batch four,
four-step schedule, full-frame crops and a clip-level bone table. It is explicitly
not the YT-SL/500000-step benchmark. U02/U06/U07 remain recorded. The production CLI
still needs manifest audit enforcement and scheduled retrieval integration in S6.

Decision: trainer core/replay PASS; production orchestration deferred to S6.
