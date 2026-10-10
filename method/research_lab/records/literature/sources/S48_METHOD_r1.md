# S48 method/code follow-up — supervised residual discrimination

Access2026-10-05. Original S48/S48_CODE_r1 preserved. Primary
[arXiv2603.04125v2](https://arxiv.org/html/2603.04125v2), selected §§3–5 and
§6.1 discussion; supplement/full remaining text not reviewed. Official
[repo](https://github.com/HSP-IIT/fsosar), HEAD rechecked unchanged
c8115c09c54d5480bf0adf972f66988bb2274c2b, API tree67entries/nontruncated.
Selected model/loss/training/loader/config symbols read; no full module execution.

Paper uses episodic source class labels, known/unknown queries and joint CE plus
weighted rejection BCE. Correctly recognized known queries are positives;
unknown queries negatives. It evaluates5-way1/5shot on class-disjoint splits.
SAFSAR combines VideoMAE and text-guided refinement. Thus a frozen SignRep
head-only study has different feature learning, supervision and task sizes.
Published gains do not supply our local baseline numbers or validity guarantees.

Code locators: utils.py::BinaryClassificationModelSAFSAR lines200–217 is
d→2d→d→64→1, hiddenReLU/finalSigmoid; d768 has2410881parameters.
utils.py::OpenSetLoss.discriminator_loss42–61 ignores wrong-known queries,
retains first at most Ncorrect unknowns and returnsNone when Ncorrect0; author
labels explicitlyFP32. models/safsar.py209–216 subtracts nearest prototype from
query before discriminator; train.py270–285/317–324 jointly backpropagates
known and weighted open-set losses. videoloader.py374–402 samples support/known
distinct clipindices and pseudo-unknowns outside episodeclasses, not our new
paired-query/signer-exclusion law. configs/SAFSAR.json has trainable backbone,
textual embedding and model-specific tiny learning rates/discriminator weights.

Exact five sourcefiles/fullBSDlicense captured61837B afterD086,128KiBcap:
[capturemanifest](../../inventory/FRDISC_MODEL_SOURCE_CAPTURE_r1.json).
Original utils/license read from historical capture, unchanged. No model/data
download/install/source module/training/published-prediction reproduction.
OwnHEADTEST001 ten FP64 sigmoid architecture parities exact; manual FP64 loss/
gradient plus nativeFP32 author BCE≤1e−6. Own stable BCEWithLogits/frozen unit
vectors/Adam/signer-matched paired galleries are explicit adaptations.

Nearest-residual MLP is established prior art. Paired sampling addresses a
metadata feasibility/control issue, not proven novelty. Full joint feature
learning and released benchmark checkpoints/evaluator result provenance remain
unreproduced; earlier pinned compute_oscr rejection does not invalidate paper
scores. Venue/peer-review/COI still unverified in this audit.
