# S47 code/parameter follow-up r1 — existing NAC frozen score control

2026-10-05. [arXiv2301.01922v1](https://arxiv.org/abs/2301.01922) primarymetadata
reports ICPR2022pp1026–1032/DOI10.1109/ICPR56361.2022.9956714; publisherpaper
not inspected, originalarXivdate2023-01-05 distinctfromconferenceyear2022.
SelectedoriginalPDF§III-D Eq4/§IV-D: k16 chosen globally followingpapergrid;
unscaledexpcosine, k=C equivalent cosine-logitsoftmax. This does not mean k16
is optimal for signs; ourcontrolledparameter fixed frompriorart, no targetsearch.

Author-linked [repo](https://github.com/1ho0jin1/OSFI-by-FineTuning) pinned
971f6767a2203ceed39877adbd5b70d65d01a2a7, gitlsremoteHEAD/complete24-entry
GitHubAPI tree/nontruncated. `utils.py`4582B/SHA
57ec4b2077012b548a18a6084430f6f867032fc42098d04799938372d40d288e,
selectedcosine lines8–13/NAC16–28 fullyinspected. Cosine normalizesfeatures,
matrixmultiply/clamp[−1,1]; NAC topkcosine, scaledsoftmax, returntop1conf/index,
defaults k16/s1. CompleteMITlicense1066B/SHA
97a39b95ccf0e0a78daf426c240233eff04ca3b01bc4a9f8fa438019c58a5d0c retained.
Publicsource+license5648B under frozen_probe/references/nac_971f6767/ afterD080.
Selectedmain.py matcherconfig only, no fulltraining/evaluator/facepipeline audit.

REJECTSCORETEST001 actuallyexecutes onlyauditedAST cosine/NAC onexistingCPU
Torch2.11.0+cu128, no upstreammodule import/training/environment install/GPU.
30fixedsynthetic no-tie cases, confidence maximumerror5.551115123125783e−17
versus NumPy, predictions exact. Manualscore/unitcosine/calibration/tie/error
oracles pass. K<16 boundary: upstreamdefaulttopk16 raises on10galleryclasses;
our min(16,N) and ownascendingglobalclassID tiepolicy explicitlyadapted.
Exactties confidenceparity only, not Torch tiedindex equivalence. Component
scoring only, not authorfine-tuned faceencoder/weightimprinting/BNreproduction.

H007: NAC16 is a mandatory existingcontrol, not inventedgallery-relative
rejection method. Scale1/defaultk16 cannotbe advertised as tuned/sign-optimal.
Rawcosine/margin/MSP/NAC share classificationranking here; differences reflect
rejectionscore/calibration, not newrepresentation. No publishednumbers imported.
Source-knownquantilecalibration empiricalonly, unknown/domainguarantee unsupported.
