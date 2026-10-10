# Search log

Ngày search: 2026-10-03 UTC. Provider: web search + direct primary URL browsing.
Không có year filter cho foundational sources; query năm 2025–2026 dùng để
refresh novelty. Language: English. Hits tổng của engine không được trả;
không fabricate PRISMA counts. Search kết quả không phải chứng cứ methods.

| Query/lookup nguyên văn | Include/decision | Exclusion / uncertainty |
| --- | --- | --- |
| `"sign language" frozen features low resource transfer signer independent` | S01/S03 baseline leads, SignCLIP S13 | Forum/aggregator chỉ leads, không evidence |
| `"sign language" missing modality robustness self supervised` | broaden next | Chưa tìm đủ direct robustness protocols |
| `"sign language" efficient representation temporal pooling distillation` | S12 TLP; S14 CTCA; S15 LiftSign | Các edge-device titles khác chưa screened full text |
| `"SignSeek" "SMART" sign language` | direct URLs L01/L02 | Search chung không đủ retrieval; mở arXiv theo seed |
| `"Temporal Lift Pooling" official paper` | S12 ECCV PDF accessible | ICCV2023 guessed URL sai: rejected locator, không rejected paper |
| `"Self-Supervised Video Transformers for Isolated Sign Language Recognition" paper` | S11 arXiv PDF, selected methods/eval | Các bản ResearchGate/PapersWithCode dedup, không sources riêng |
| `"ASL Citizen" NeurIPS 2023 paper` | S10 proceedings metadata/PDF | Hai proceedings host URLs dedup một paper |
| `"Distilling Cross-Temporal Contexts" arxiv` | S14 CVPR lead | CVF direct fetch internal error; methods chưa đọc |
| `"sign language" "few-shot" "representation" 2025 2026` with domains arxiv.org/openaccess.thecvf.com/aclanthology.org | L03 PhonSSM; S16 geometry-aware few-shot lead | Không claim novelty trước full protocol/code audit |
| Direct arXiv 2411.16765v3 / 2503.08529v1 | S01/S03 | Selected sections, chưa review mọi appendix |
| Direct arXiv 2609.03695v1 / 2608.25493v2 | L01/L02 HTML accessible | Abstract seed đã được mở rộng sang methods/eval |
| Direct official ASL Citizen pinned testing.py and GitHub tree | S05 scorer, MIT code license locator | Repo không chứa release split CSV; exact release equality pending |

13 unique paper candidates được retained/screened bằng title/abstract/section
trong matrix: S01,S03,L01,L02,S10,S11,S12,L03,S13,S14,S15,S16,S17.
Trong đó 8 paper có selected text/methods đã đọc; 5 chỉ lead/abstract.
S05 là evaluator source ngoài paper count. Chưa đạt tham khảo 20–30 screened
items của rules; không dùng quota như hard stop, nhưng search vẫn chưa đóng.
Không kiểm chứng citation existence cho rejected nonprimary search hits.

S17: SignBERT+ arXiv2305.04868, abstract lead; keep để tránh claim phonological
pretraining chưa có. Novelty next action: đọc S13/S17, mở rộng theo mechanism
H001/H003; nếu full-text closest không available thì giữ novelty unknown.

## Update1 — selected full-text review and release lookup

Direct ACL S13 PDF selected §5/6.1/Table5 and arXiv S17 PDF §3/4.1/Table1
now read; cards S13/S17 and NOVELTY_20261003_r1.md retain scope and inference.
Ten papers now have selected methods/evaluation text reviewed; three remain
leads. This does not mean ten complete papers/codebases were reviewed.

Queries: `site.microsoft.com ASL Citizen license non commercial dataset`;
`site.arxiv.org sign language frozen temporal pooling few shot low label representation`.
Official project/download pages found; broad mechanism query mainly returned
already-known TLP/SignCLIP and unrelated language-model prompt tuning (exclude).
Older zero-shot textual-description paper retained as a possible portfolio
lead, not included in screened matrix pending specific mechanism relevance.
Supporting-material tab currently returns no license text in browser parse.
Dataset local `use.txt` is the terms artifact; REL001 will verify official
metadata provenance with bounded range requests. No full archive download.

Source locator correction: S17 future-work discussion is §4.7, not §4.6;
S13 scope uses selected §5 paragraphs rather than an unverified §5.3 label.
No scientific result or source-derived mechanism conclusion changes.

## Update2 — mechanism search,2026-10-04 local

Queries: `"frozen" "sign language" "temporal" "pooling"`;
`"temporal pyramid" "sign language" recognition`; `"SignRep" "activity" "sigmoid"`.
S18 author-hosted ASSETS2009 PDF selected §4.3/5 reviewed; temporal bins
are established sign-language practice. Matrix now14 papers+scorer,11 selected
text reviews and3 leads; not a complete literature review. SpaMo NAACL2025
PDF returned as translation lead, not read/added or treated as a probe baseline.
Non-paper repository and aggregator-only matches excluded as method evidence.
SignRep §3/4/5.1/6,A2/A3 examined for regression activity and pretrain exclusions;
ACTIVITY_CONTRACT.md records source facts versus adapted-engineering decisions.
No literature-score comparison or method novelty certified.

Update2026-10-04, targeted primary-source follow-up (no new search items):
- arxiv2608.25493v2 §§3.3/4.1–4.3 Tables4–6; inclusion: temporal adapter supervision/baseline limits, unchanged H001 analysis scope.
- arxiv2604.08761v1 AppendixG/K; inclusion: mean-embedding phonological probe and group/pose/ontology limitations.
No full-domain review claim; no external reported scores inserted as local baselines.

## 2026-10-04 H002 targeted cost/stride follow-up

Queries: SignRep stride; sign language frozen encoder efficient inference
temporal sampling window stride retrieval. Opened primary SignRep v1 HTML
§6/§7.2/Table5; stride8 exists in ablations, method-novelty claim rejected.
Official proceedings PDF browser fetch failed; author copy exceeds browser
limit, no local PDF download. Broad search returns auxiliary leads, not full
read cards or a closed novelty review. No literature scores used as local data.


## 2026-10-04 — dossier primary-source spot verification while test inference runs

Opened pinned SignRepv1, SignSeekv1 and final SignCLIP/TemporalLiftPooling/ASLCitizen
primary PDFs. Rechecked SignRep§7.2/Table5 existingstrides, SignSeek§4.1per-glossMAX
cosine/normalizedtrain-gallery/testqueries, TLP§3prediction/update lossydownsampling
context and ASLCitizen§3.3rankedmetrics/signer-separatedusecase. SignCLIP ten-text
search returned no match; closest row uses only the previously read limited-gallery
KNN fact and does not strengthen its exact protocol/code parity. This is selected
source checking, not full-appendix/code/final SignRep proceedings or exhaustive
novelty review. No test scores read, method/config choices changed or paper numbers
used as controls. Original source URLs in the dossier/source cards.

## 2026-10-04 — terminal H002 source/direction refresh

Websearch exactqueries (no publication-year filter; judge sources through current2026-10-04):
`site.openaccess.thecvf.com SignRep 2025 ICCV representation`;
`"sign dictionary retrieval" "temporal" "sampling"`;
`"sign language" "retrieval" "efficiency" sampling window`;
`"sign dictionary retrieval" "sampling" -site:reddit.com -site:rt.http3.lol`;
`"sign language" "retrieval" "temporal sampling" -site:reddit.com`;
`"SignRep" "stride" cost latency -site:reddit.com`;
`"SignMatch: Matching Dictionary Signs to Continuous Sign Language Video" arxiv`;
`"Semantic Hardness Is Not Visual Hardness" arxiv`.

Primary CVF HTML/PDF browser403, direct official HEAD/GET200. HTMLlinkmatches
main/supp; LIT001cap failure retained, LIT002complete bounded16MiB/source31.787s.
Read S03Fselected main6/7.1–7.2/7.4–7.5/Table4–5/8,suppA1–A3/A7–A8; no fullversion
diff, numericpagepreflightUNAVAILABLE/no install. arXivabs showsv1only; attempted
v2HTML404, not evidence of a secondrevision. OfficialacceptedOA facts retain
stride8 prior and add WLASLvalcheckpointselection disclosure.

Four new distinct primary-paper items screened: L04SignMatch included nearest
task/backbone (selected3/4/5.1/5.4); L05SANincluded mechanismlimit (selected3/4.1/
Limitations); S19reducedfeatureASLdictionaryCHI2025 andS20YOLOActrecognition remain
primary-indexed leads/directfetchfailed. No methods/code/userstudy completeness
claims for leads. Secondarymirrors/aggregators/reddit excluded asmethod evidence;
SLVideo/Spotterpackage result notread/added and notbaseline. Matrix20rows =18distinct
papers+1scorer+1additionalSignRepversion;13distinct selectedtext reviews/5leads,
not18complete papers. No exhaustiveabsence or first-study claim. No reported
paper scores used as localcontrols. SignMatchofficialauthor project currently
code/model “soon”; venueBMVC2026 author-stated, proceedings unaudited.

## 2026-10-04 H003 primary asset/architecture source resolution

Decision targeted: existing local main/DINO weights and native strict-load compatibility. Read selected paper3.1/figure2/4.1 and official author project→GitHub/demo chain; author project states ACL2025Oral, proceedings not separately audited. Pinned demo app69d3d77aa4a4fec89048f5417d44fa717e36f6f8 references ShesterG/SHuBERT MODEL_REPO. Wrong demo-space models endpoint404 retained ASSET001; correct author model tree ASSET002 published LFS comparison succeeds. No method numeric literature baseline copied.

Official DINO hub/backbones source read, then bounded necessary architecture from commit7764ea0f912e53c92e82eb78a2a1631e92725fc8 under own ignored runtime:31files/156624B plus LICENSE, Gitblob/size/SHA checks. Selected builder/models/layer import contracts, not full repository review. Source version is a new pinned compatibility reference, not known historical author version. No install/weights download, no new paper matrix row, novelty counts unchanged. Source card S01 updated; strict CPU and synthetic GPU checks do not establish paper reproduction/exclusions or natural failure semantics.

## 2026-10-04 H003 technical primary-source audit (no new paper/novelty claim)

Target decisions: crop color/encoding compatibility and avoiding unnecessary
NVIDIA graphics contexts with CPU MediaPipe. Selected primary implementation/docs:
[decord FFmpeg threaded decoder](https://raw.githubusercontent.com/dmlc/decord/master/src/video/ffmpeg/threaded_decoder.cc)
RGB24 copy contract; [OpenCV VideoWriter](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html)
BGR convention; [MediaPipe Pose](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
33-pose landmark API. Installed-version behavior independently verified by known
color/lossless/lossy fixtures and all396realframes rather than inferred from
current docs. Author RGB→BGR before writer is correct; unencoded RGB differs
from native encoded model inputs because mp4v is lossy.

[libglvnd vendor loader](https://raw.githubusercontent.com/NVIDIA/libglvnd/master/src/EGL/libeglvendor.c),
[Mesa environment](https://docs.mesa3d.org/envvars.html),
[Mesa llvmpipe](https://docs.mesa3d.org/drivers/llvmpipe.html) selected contracts
for exclusive vendor JSON/per-child software renderer/cache disabling. Local
SOFTGL00112contexts confirms llvmpipe/raw equality/noNVIDIAworker handles. Guessed
egl-vendor.md page unavailable, not evidence. Current technical docs don't
establish historical SHuBERT DINO source or natural-failure annotation validity.
No additional paper matrix row or claim that closest-work search is complete.

## 2026-10-04 — H003 annotation and reliability decision refresh

Exact queries, no year filter:

- `"sign language" "tracking" "annotations" hand failure dataset`
- `"sign language" recognition missing modalities robustness hand occlusion`
- `"SHuBERT" "masking" hand face ablation`
- `"sign language" "hand pose" manually annotated benchmark`
- `"SignBERT+" confidence reconstruction hand pose sign language`
- `"sign language" hand tracking manually annotated "dataset" "ground truth"`
- `"How2Sign-Synth3D" "arxiv"`
- `"sign language" "confidence" "missing" "modality" reliability recognition`

Selected nine decision-relevant primary candidates screened: S21–S25, original
SignBERT predecessor, 3D-hands consistency benchmark, LSWH100 and CorSiL. Other
irrelevant index hits not comprehensively reviewed or counted. Original SignBERT
mechanisms covered by S17 follow-up; no separate matrix row/full-review claim.
Consistency-only benchmark, synthetic language-mismatched labels and email-only
access do not solve current manual natural-ASL GT. No external contacts.

New matrix items: S21 selected§3/4/5/Table5; S22 project-only lead (CVF fetch403,
no independently verified arXiv ID); S23 selected§3.1/3.3/Table10 narrative/§7 and
materials (image table values not read); S24 selected equations/state/metrics,
not full code or label-release audit; S25 indexed lead/fulltext429, no bypass.
Cards retain provenance, partial coverage and limitations. S01 AppendixA/Table4/8,
S17 III-B2/Eq8 and L01§3.1–3.4 targeted rereads refine relevant control semantics;
no claim of reading every appendix or producing numerical literature baselines.

Official Boston50/104 and RWTH database terms read as metadata follow-ups:
[Boston50](https://www-i6.informatik.rwth-aachen.de/web/Software/Databases/Signlanguage/details/rwth-boston-50/index.php),
[Boston104](https://www-i6.informatik.rwth-aachen.de/web/Software/Databases/Signlanguage/details/rwth-boston-104/index.php),
[database terms](https://www-i6.informatik.rwth-aachen.de/web/Software/Databases/Signlanguage/).
Dense manual GT provenance not established by these pages. No form submitted or
bulk archive fetched. Actual bounded PHOENIX XML/prefix audits separately preserve
response identity/hashes and find no accepted local join; C059/F055 is metadata,
not tracking quality. Installed score semantics are C060/F056, not inferred from
a current web API alone.

Matrix25 rows =23 distinct papers, one scorer and an extra SignRep version;
16 selected-text reviews/seven leads. This is bounded source review, not a
systematic search or exhaustive absence claim. [D020](../decisions/D020.md) and
[H003 review](NOVELTY_20261004_H003_r1.md) retain scientific hold while making the
annotation dependency and closest controls explicit. No new test outcomes/HPO.

## 2026-10-04 — H002 contribution and H004 acquisition mechanism

Exact queries, no publication-year restriction:

- `"sign dictionary retrieval" "efficient"`
- `"sign language" "temporal sampling" retrieval`
- `"SignRep" "stride" retrieval`
- `"SignMatch" sign retrieval code`
- `"ASL Dictionary Retrieval" "Reduced Feature Sets"`
- `"sign dictionary" "sampling" -site:pith.science -site:huggingface.co -site:reddit.com`
- `"adaptive" "frame selection" "sign language" recognition`
- `"sign language" "optimal transport" retrieval`
- `"video retrieval" "adaptive" "early" sampling confidence`
- `"video retrieval" "certified" sampling`
- `"adaptive sampling" "inner product" concentration retrieval`
- `"adaptive" "Monte Carlo" "ranking" retrieval`
- `"BanditMIPS" proceedings AAAI 2024`
- `"confidence sequences" "sampling without replacement" Waudby Smith Ramdas`

Nine selected decision-relevant primary candidates considered: S19, newS26–S29,
UATVR/UMIVR/HAFTR/ConvLSTM adaptive-frame works. The latter four remain metadata/
indexed leads only, not full-reviewed controls. Other irrelevant search hits
uncounted/unreviewed. Secondary AI reviews, aggregators and commercial signboard
hits excluded as method evidence. No exhaustive absence/priority claim.

S19 now selected author-PDF§3.1/3.2/4.1/5.3/5.4, failed-fetch history retained.
S26 selected§2/3/3.2/Algorithm1; S27 intro/definition/finite-population model only,
constructions/proofs unread; S28 motion/early-decision sections; S29 distribution/
OT§3.3–3.5. Cards retain partial scope/unknown code and comparability. Browser PDFs
only, no local numeric-page anchors. Guessed unrelated PMLR Waudby-Smith page
unavailable, not a citation. SignMatch author page still shows release placeholders;
no code/model release proved. Existing SignSeek text revisited, no new version audit.

Matrix29 rows/27 distinct papers/21 distinct selected-text reads/six leads, one
scorer and extra SignRep version. Selected does not mean full review. No paper
scores used as reproduced controls. D021/D022 execute two actual CPU feasibility
formulas on unchanged train/dev, no test access; both fail practical gate. See
H002_H004_r3 assessment for preserved prior-art/contribution limits.

## 2026-10-04 — H005 participant/gallery problem discovery

Exact web queries, no publication-year filter:

- `sign language frozen retrieval hubness signer invariance`
- `few shot dictionary retrieval support bias local scaling`
- `"sign language" "hubness"`
- `"sign language" "signer" "retrieval" "normalization"`
- `"sign language" "signer invariant" representation learning`
- `"hubness" "few shot" nearest neighbor classification`
- `"Cross-domain Similarity Local Scaling" Conneau 2018`
- `"sign language" "nuisance" disentanglement retrieval`
- `"Word Translation Without Parallel Data" CSLS arxiv`
- `"Nuisance Attribute Projection" speaker recognition paper`
- `"Learning Signer-Invariant Representations with Adversarial Training" Ferreira`
- `"Z-Score Normalization, Hubness, and Few-Shot Learning"`
- `"Signer-Independent Sign Language Recognition with Adversarial Neural Networks" site:ijmlc.org`
- `"Z-Score Normalization, Hubness, and Few-Shot Learning" arxiv`

Two closest new papers screened/included as **abstract-only leads**: S30
University of Porto indexed publication abstract (ICMV2019 event/2020publication),
S31 official ICCV2021 indexed abstract/intro. Their author/official PDFs returned
Timeout/403/InternalError; RUC author fallback timed out. Primary institutional
metadata and official proceedings support existence/topic, not formulas/code
or reproduction. IJMLC journal sequel guessed PDF URL inaccessible, identity
of that URL unverified; not an accepted source. ResearchGate/aggregators/AI
reviews/reddit excluded as method evidence, no secondary copy used to bypass
terms. These two cards do not count as full or selected-method reads.

Other returned primary/institution leads (dictionary joint-hand-position DTW,
ICCVW2025 signer conformer, Conneau bilingual retrieval, weighted/nonlinear NAP,
hubness graph/motion works) remain uncounted contextual leads here; no method
formula/code/venue priority inferred beyond the recorded read scope. Dictionary
ACL2020 PDF opened, selected methods not yet read and no source row added.
The decision is baseline association measurement, not a method/novelty claim.

Scorer search opened official scikit-learn balanced_accuracy_score documentation
(mean recall; adjusted=False); installed1.8.0 source audited and exact manual
fixture parity run, local source hash saved. Website1.9.1 is not claimed as the
installed version or pinned upstream commit. H005_v0 CPU baseline diagnostic
allowed while closest full-text novelty remains unresolved (03_LITERATURE).

Matrix31rows =29 distinct papers + scorer + extra SignRep version;21 selected
partial-text reads/eight abstract-only leads. No exhaustive/full-paper claim.
D023/D024, PARTPROBE001 and C063/F059 give actual train-only evidence. All1313
sources covered; no test/dev feature measurement, GPU, new encoder call, policy
search or causal nuisance/retrieval-quality conclusion.

## 2026-10-04 — H005 task-harm controls and primary projection precedents

Exact queries (no year filter):

- `"nuisance attribute projection" Solomonoff Campbell Quillen 2007 pdf`
- `"Signer-Independent Sign Language Recognition with Adversarial Neural Networks" pdf ijmlc`

Two new primary papers included with selected partial reads: S32 institutional
publisher PDF title/intro/3.1Eq4–6; S33 journal publisher PDF III architecture/
adversarial/transfer and initial IV dataset/baseline paragraphs. The legitimate
IJML publisher result resolves the earlier guessed inaccessible journal URL;
S30 conference remains abstract-only, not retroactive methods read. No local
PDF page anchors, code/reproduced scores or whole-text completeness claimed.

Original NAP CiteSeer PDF/institutional author list and Turkish2024 signer
disentanglement publisher page screened as contextual leads, not method-read
or matrix rows. Secondary ResearchGate/studylib mirrors excluded as method
evidence. Bibliographic existence/selected text only, no exhaustive absence.

Matrix33rows/31distinct paper publications +scorer+extraSignRep version,
23selected partial-text reads/eight leads. Known projection and signer-mean
invariance families constrain novelty. D025 preregisters one rank8 held-gloss
task diagnostic; D026 rejects participant-specific scaling when generic PCA
wins. No test/H002 policy tuning, GPU, new encoder or renamed-framework claim.

## H006/D027–D028 — Base-conditioned geometry and centering precedent

2026-10-04 UTC, exact queries (no year filter):
- `"few shot" "within-class covariance" whitening arxiv`
- `SimpleShot nearest neighbor centering normalization Wang arxiv 2019`

Primary arXiv1911.04623v2 selected sections2/2.1 and initial3.1 reviewed: S34,
base-trained image encoder, one-shot NN/multishot centroids, base mean centering
and L2. Author repository link uninspected; no local image benchmark reproduction.
Existing NAP-family residual covariance identity is algebra/oracle, not new method.
Unreviewed LFD-ProtoNet arXiv2006.08306, CVPR2020 Mahalanobis and multimodal
nearest-neighbor normalization arXiv2410.24114 hits remain next-read leads, not
matrix evidence or method facts. Secondary explanatory hits excluded as facts.
Current matrix34rows/32distinct paper publications +scorer+extraSignRep version,
24selected partial-text reads/eight leads. No exhaustive novelty-absence claim.
H006 passes exploratory dev controls; closest strong metric baseline remains open.

## H006_v1/D029–D030 — Strong standard geometry and mechanism collision

2026-10-04 UTC, web queries exactly (no year filter):
- `site.openaccess.thecvf.com CVPR 2020 few-shot Mahalanobis distance covariance`
- `site.arxiv.org 2006.08306 LFD ProtoNet`
- `few shot classification within class whitening frozen features Ledoit Wolf covariance`

Primary CVF metadata verifies S35 CVPR2020; PDF403, author arXiv1912.03432v3
selected§4Eq1–3 and initial6.1 read. LFD arXiv2006.08306v2 metadata/PDF selected
intro/code footnote,§3 and4.1Algorithm1: S36. Its author error warning excludes
reported result numbers from evidence. HTML unavailable, failed screenshot not
visual-page inspection. FROST arXiv2606.31136v1 HTML selected3.1–3.2/A.3: S37,
mechanism collision with within-class frozen-feature whitening. Preprint only,
no code/whole-paper/segmentation score reproduction. Other frozen augmentation/
Brownian covariance/newer secondary results not incorporated as method facts.
Secondary AlphaXiv/ResearchGate/blog/wiki/forum pages were leads only; primary
texts resolved selected decisions. Official sklearn API Parameters/Notes and
local installed1.8.0 _ledoit_wolf/empirical_covariance source: S38 software, not
additional paper; online docs1.9.1 distinct. Local files SHA-bound prelaunch.

38matrix rows/35distinct publications,27selected partial-paper reads/eight leads,
plus two software sources and extra SignRep version. Engine total hits unknown,
no PRISMA/exhaustive absence/full-code completeness claim. H006_v1 registered
standard-control falsifier fails when residual LW cosine beats hardrank8 PCA.
No new math/algorithm claim; next checkpoint-pretraining membership and analysis
contribution review before independent-protocol/scale claims. Resource estimate
failure preserved separately; no dataset/source changes or reported-paper baselines.

## LINEAGE/D031–D034 — Public source-pool audit

2026-10-04, primary direct opens/source-link following only, no new web search
query. SignRep arXiv2503.08529v1 selected7.2/A2/A3/A7; YouTube-SL25
arXiv2407.11144v1 introduction/2/2.1/initial2.2, metadata/author release S39.
No full-paper/model/code review. Public console login redirect unavailable; no
auth attempt. Registered bounded public GCS metadata/read URL works, generation
1718728410650400/MD5 and667188bytes verified, headerless source IDs recounted.
Exact local subset coverage is not SignRep training membership/disjointness.
39matrix rows/36paper publications/28selected partial-paper reads/eight leads
plus two software references/extra SignRep version; no exhaustive absence claim.

## H006_v2/D038–D040 — Bounded computational mechanism diagnostic

No new external query/publication this turn. Reviewed existing primaryselected
S35regularized covariance and S37withinclass frozenfeature whitening precedents,
with prior S32/S34/S38 family cautions. Factorial basis/spectrum equations/manual
oracles are own ordinary linear algebra; no claim this ablation is uniquely new
or absent from literature. Same-dev positive orientation result supports bounded
analysis, not new method or independent confirmation. Matrix39rows/36papers/
28partialselectedpaper reads/eightleads+2software+extraSignRep unchanged. Refresh
closest cross-corpus/representation-analysis review before promotion/scaling.

MS-ASL feasibility, 2026-10-04: exact query MS-ASL dataset official microsoft annotations download github, no date filter. Primary arXiv1812.01053v2 selected3.2–3.5/initial4, Microsoft project/download/catalog read. Guessed microsoft/MS-ASL GitHub URL404 retained as a failed lead, not global absence. Third-party downloader/aggregator/Reddit results not used as evidence or executed. S40 added; matrix40rows/37distinct paper publications/29selected partial-paper reads/eight leads plus two software references/extra SignRep version. No full-paper/code review or exhaustive novelty claim. Publisher archive planned under D041/MSMETA001; no video downloads.


## H006/D071 continuation — closest geometry/task refresh before MS-ASL scores

2026-10-04; twelve exact web queries, no date/domain filter except explicit site terms:
- `"sign language" "retrieval" "whitening"`
- `"few-shot" "within-class covariance" "cross-domain"`
- `"FROST" "Training-Free Few-Shot Segmentation"`
- `"Whitening and Fine-Tuning Are Substitutes"`
- `"Metric-Learning Retrieval for Cross-Signer Isolated Sign Language Recognition"`
- `"A Closer Look at Prototype Classifier for Few-shot Image Classification"`
- `"Whitening and Fine-Tuning" Poornima Mathias pdf`
- `"Whitening" "Poornima" sign`
- `"Metric-Learning Retrieval" "Asif" 2026`
- `"Fine-Tuning CNN Image Retrieval with No Human Annotation" arxiv`
- `"Whitening and Fine-Tuning Are Substitutes" site:arxiv.org`
- `"Whitening and Fine-Tuning Are Substitutes" site:aclanthology.org`

Four new source cards: S41/S42 organizer accepted-list titles only, S43 selected
NeurIPS2022 primary3.4/4.2/A.1/A.7, S44 selected primary3.4/p6/Eq7-8 +IEEE metadata.
FROST re-found, no new fullread. Aggregator/dental/otherdomain hits not evidence;
exact returned hit universe not complete and engine global hit count unknown.
WSLP page lists acceptance news2026-10-02; no manuscript links found. ACL search
URL unavailable and guessed arXiv1711.02512v3 HTML unavailable; actual arXivv2 PDF
read instead. No contacting authors, downloads to shared storage, code execution
or reported score importing. Matrix40->44rows: two new selected-text reads and
two new title-only leads; no full-paper/comprehensive absence claim. New audit r2
keeps empirical-transfer RQ and no novelalgorithm promotion.

## D073 — finite-gallery rejection intake, 2026-10-05

Bounded targeted search; no date/domain filters except literal query terms.
Includes the unpersisted four-query intake immediately preceding this continuation
and eight queries in this continuation:12submissions/10distinct strings.

- `"Whitening and Fine-Tuning Are Substitutes: One-Shot Indian Sign Language Recognition"` (twice)
- `"Metric-Learning Retrieval for Cross-Signer Isolated Sign Language Recognition"` (twice)
- `"sign language" "open-set" retrieval recognition`
- `"sign language" "conformal" prediction`
- `"Reliability-aware modality-adaptive" conformal risk`
- `few shot open set recognition conformal gallery rejection`
- `"Reliability-aware modality-adaptive mixture-of-experts" Nguyen`
- `"few-shot" "open-set" "sign language" rejection`
- `"conformal" "sign language" prediction uncertainty`
- `"Automatic hand sign recognition" "latent cognizance" Nakjai 2018`

Screened five newly included primary publications S45–S49; existing L01 refreshed
and S41/S42 leads rechecked. Not a complete hit census: engine totals/unique
returned records unknown. Older papers included deliberately for mechanisms.
Excluded wearable skin-conformal hardware, biomedical/hyperspectral irrelevant
task hits, aggregator/Reddit snippets as algorithm evidence. Other relevant
FSOS methods remain follow-up leads, not fully reviewed comparisons.

S45 publisher abstract only; direct publisher open failed but indexed primary
abstract accessible. Earlier SSRN abstract differs; final methods not accessible
in this cycle. arXiv2110.15548 intro/reference9 led to actual Thai fingerspelling
task paper2110.15542; do not conflate them. HTML2110.15542/2110.15548 unavailable,
read browser PDF selected text; no local PDF downloads/page preflight claim.
NAC guessed HTMLv2 unavailable; actual arXiv2301.01922v1 PDF read. FR-Disc first
v1 then currentv2 selected methods; guessed uppercase projectURL failed, followed
exact author link to lowercasefsosar and officialrepo. CRC currentv4 selected
conditions/proof only, extensions unread. Exacttitle searches yielded no public
S41/S42 manuscript in returned results; no global absence or contacting authors.

Matrix44→49rows; three new selected computational-paper reads, one selected
theory read, one abstract-only lead. No full-paper/code reproduction, author
credibility/COI/predatory-venue audit or comprehensive novelty claim. Citations
verified through primary arXiv/publisher/project metadata; programmatic S2/DOI
verification not invoked. All retrieved content treated as data. New D073/intake
changes next action to metadata-only calibration feasibility before H007 scores.


2026-10-05 D076–D079 / OSCR evaluator source follow-up:
- Official NeurIPS2018 abstract/metadata and PDF selected §4.3/Eq3/Fig3(c)/Table2 discussion; S50 added, matrix49→50rows. No local PDF/fullpaper-read claim.
- Exact public HSP-IIT/fsosar pinned c8115c09c54d5480bf0adf972f66988bb2274c2b utils.py and full root LICENSE fetched by HTTPS/SHA; browser raw/blob cachemiss retained. Captured35043B source+BSDlicense after D076, no model/data/shared-env download.
- Exact `git ls-remote https://github.com/Vastlab/Reducing-Network-Agnostophobia.git HEAD`; pinned581b3f06aeedb737e83a8610573f02299e2416b2; GitHub API recursive tree21entries/nontruncated; selected Tools/evaluation_tools.py4342B readfully inRAM/SHA recorded. No LICENSE filename in inspected tree, license unknown; no copying/execution/Python2 dependency install. Selected file plots TP/positives vs FP/unknowns, not scalarAUC. This bounded check does not establish no other usable scorer exists.
- Pinned FR-Disc compute_oscr synthetic audit executes only inspectedAST function, not module/training. Actual wrongknown/perfect/wrongties .25/1/.25 versus manual0/1/0 rejects this metric version. Original wrapper import-hook failure retained; no paper score/result-version reproduction or published-score invalidation. S48 original card unchanged; separate S48_CODE_r1 follow-up and matrix provenance updated.
- Own explicit inclusive-tie/grouped-trapezoid evaluator six softwarefixturegroups/30Python-Fraction cases pass; adapted specification, no exactofficialscalar parity or task-quality/newmethod/novelty claim.


2026-10-05 D080–D084 / exact NAC source/parameter follow-up:
Primary arXiv2301.01922v1 abstract/journalmetadata and PDF selected§III-D Eq4/§IV-D(defaultk16,scale1), officialauthorlink 1ho0jin1/OSFI-by-FineTuning. `git ls-remote https://github.com/1ho0jin1/OSFI-by-FineTuning.git HEAD` pinned971f6767a2203ceed39877adbd5b70d65d01a2a7; complete24-entry GitHubAPI tree/nontruncated. Raw utils.py4582B/MITLICENSE1066B SHA-read, selected main.py matcherconfig; fullNAC/cosine ASTfunctions inspected/captured afterD080 (5648B, no model/data/shared-env download). ExistingTorchCPU only30synthetic componentparities max5.551115123125783e-17/predictions exact, tiesconfidence only,10classkclamp adaptation explicit. Original S47card preserved, S47_CODE_r1/matrix follow-up update;50rows unchanged. No fullfacepipeline/training/scorer/releasedprediction/publishednumbers/publisherpaper reproduction or signoptimalparameter claim.


2026-10-05 D086–D090 / residual-head closest-method and supervision follow-up:
Actual browser queries: `"few shot open set" "feature residual" discrimination`;
`"open set" "one shot" gallery normalization conformal recognition`;
`"sign language" "open set" retrieval calibration`;
`"few-shot open-set" "relative displacement" recognition`;
`"few-shot open-set" "PEELER" recognition`;
`"sign language" "open-set" "unknown" recognition`;
`"Few-Shot Open-Set Recognition" "Relative" displacement`;
`"Feature-Residual Discriminator" "2022" one shot`;
`"sign language" "open-set recognition" video`.
Primary arXiv2603.04125v2 selected§§3–5/6.1, exact authorproject/repo. Authorrepo
HEAD rechecked unchangedc8115c09c54d5480bf0adf972f66988bb2274c2b; GitHubAPI
recursive67entry/nontruncatedtree. Selected utils BinaryClassificationModelSAFSAR
and OpenSetLoss.discriminator_loss, SAFSARmodel/forward, train loss/optimizer,
loader episode query roles and config read; not fullrepo/paper/supplement read.
After D086 capture5exact model/training/loader/config/fullBSDlicense files61837B
under128KiBcap, SHAinventory FRDISC_MODEL_SOURCE_CAPTURE_r1. No fullclone,
checkpoint/data/shared-env download or author-module training import. RAMonly
STRM30112B/config672B selected read, not captured/reproduced.
PEELER arXiv2005.13713/CVPR2020 abstract and SnaTCHer2103.01537/CVPR2021 abstract
screened; officialCVF PDFs returned403, no full-method claim. RFDNet authorGitHub
README selectedabstract only, task-adaptive displacement lead, fullmethods/code/
license unverified. No newS51/S52rows or comprehensive novelty conclusion.
S48_METHOD_r1 and NOVELTY_20261005_H007_r1 frozen before3head quality fits.
Component fixtures/local head-only adaptation executed under its own protocol;
jointauthorbackbone/CE+BCE/fullbenchmark notreproduced. All3seeds jointutility
screenfails FARgate; no novelty/newmethod or publishedpaper-score invalidation.
Matrix remains50rows, source-scopedselectedread. NEXT fullprimary PEELER/
SnaTCHer/RFDNet method distinctions before any new representation hypothesis;
no same-MS rescuedesign/selectedmetric/targetthreshold tuning.
