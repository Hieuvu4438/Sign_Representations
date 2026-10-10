# SignRel-JEPA — End-to-End Implementation Specification for an AI Coding Agent

> **Purpose:** This document is an executable research/engineering specification for implementing, validating, and evaluating **SignRel-JEPA: Relation-Aware Joint-Embedding Predictive Learning for Sign Language Representation**.
>
> **Audience:** autonomous/agentic coding systems and human researchers.
>
> **Status:** research proposal / implementation blueprint, **not** a claim that the proposed method has already been empirically validated.
>
> **Source verification date:** 2026-10-06.
>
> **Core constraint:** the main pretraining path is **self-supervised**. Do **not** use gloss labels, text translations, class IDs, signer IDs, or dataset labels as model inputs, positive-pair definitions, or losses during core pretraining.

---

## 0. Agent Operating Contract

The implementation agent MUST obey the following rules.

1. **Preserve the scientific hypothesis.** The central contribution is **inter-articulator relational latent prediction**. Do not silently replace it with pixel reconstruction, pose reconstruction, ordinary contrastive learning, or standard masked-autoencoding.
2. **No semantic labels in core pretraining.** Gloss/text/class labels may exist in manifests for evaluation, but the pretraining `Dataset` and training step MUST NOT expose them to the model or loss.
3. **Pose/landmarks are training-time structural metadata only by default.** They may define articulator regions/masks. They are **not** the primary target.
4. **RGB-only inference must remain possible.** The final encoder used for downstream feature extraction must not require a pose estimator.
5. **External code must be provenance-tracked.** Before copying or vendoring code, inspect the upstream license and record:
   - upstream repository + commit hash,
   - source file,
   - copied/adapted functions,
   - license,
   - local destination.
6. **Do not assume paper/checkpoint compatibility.** Validate tensor shapes, patch/tubelet configuration, image size, frame count, normalization, and checkpoint keys.
7. **No data leakage.** Pretraining video IDs overlapping downstream validation/test sets must be removed when identifiable.
8. **Every research feature requires an ablation.** Do not add a component to the flagship model unless it can be disabled by config.
9. **Every training run must be reproducible.** Save full resolved config, git commit, package lock, random seeds, dataset manifest hash, upstream checkpoint hash, and environment information.
10. **Do not optimize only training loss.** Frozen representation quality is the primary objective. A lower pretraining loss is not sufficient evidence.
11. **Fail loudly.** Assertions must catch:
    - labels accidentally entering pretraining,
    - zero-valid-articulator batches,
    - teacher receiving gradients,
    - incorrect EMA updates,
    - invalid token/region alignment,
    - resume-state mismatch,
    - overlap between pretraining and downstream held-out video IDs.
12. **Do not overclaim novelty.** `relation-aware latent prediction` is a research hypothesis. Before submission, rerun a dedicated prior-art search.

---

# 1. Research Objective

## 1.1 Core research question

> **Are inter-articulator relations a better self-supervised prediction target for sign-language representation learning than generic visual reconstruction or independent articulator prediction?**

A sign is modeled as a time-varying system of articulators:

\[
A = \{LH, RH, Face, Body\}.
\]

Existing approaches frequently model individual components, reconstruct sign-specific geometric priors, predict discrete hidden units, or mask one salient component.

SignRel-JEPA instead models a **dynamic relational graph**:

\[
G_t=(V_t,E_t),
\]

where nodes are articulator representations and edges encode learned relations such as:

\[
LH\leftrightarrow RH,\quad
LH\leftrightarrow Face,\quad
RH\leftrightarrow Face,\quad
LH\leftrightarrow Body,\quad
RH\leftrightarrow Body.
\]

The student sees a structurally masked sign video and predicts relation representations produced by an EMA teacher from the clean/full video.

---

## 1.2 Main hypotheses

### H1 — Relation-target hypothesis

\[
\text{JEPA + relation prediction}
>
\text{generic JEPA}
\]

on frozen sign retrieval and phonological probes.

### H2 — Structured-context hypothesis

Masking linguistically relevant regions/time intervals should produce stronger transferable features than purely random video masking.

### H3 — Multi-scale hypothesis

Predicting relations over short, medium, and long temporal scales should improve preservation of both fine articulatory cues and movement/trajectory cues.

### H4 — Invariance hypothesis

A relation-centric target should reduce sensitivity to signer appearance/background while preserving sign-relevant structure.

---

# 2. Scientific Positioning and What Is Actually Novel

## 2.1 Main contribution

**Relational self-supervised target**

Instead of predicting:

- RGB pixels,
- pose coordinates,
- handcrafted geometric priors,
- independent discrete clusters,
- only one missing/salient articulator,

the proposed model predicts:

\[
\boxed{
r^{a,b}_{t,s}
=
R_{\xi}
\left(
z^{a}_{t,s},
z^{b}_{t,s}
\right)
}
\]

where:

- \(a,b\) are articulators,
- \(s\) is temporal scale,
- \(z\) is a contextual visual latent from the teacher,
- \(R_\xi\) is the teacher relation projector,
- targets are stop-gradient.

The student predicts these targets from masked context.

## 2.2 Supporting contribution

**Multi-scale temporal relational prediction**

Relations are modeled at multiple temporal scales rather than only frame/token level.

## 2.3 Supporting mechanism

**Sign-aware structured masking**

Use hand/face/body localization and motion metadata to construct difficult masks that break articulatory relations.

**Important:** sign-aware masking by itself is **not** the primary novelty because recent work such as SignMAE and SignDINO already develops sign-aware masking/view construction.

## 2.4 Evaluation contribution

Build a representation diagnostic suite that measures both:

- preservation of phonologically relevant factors;
- removal/reduction of signer/style information.

---

# 3. What We Reuse, Adapt, and Do Not Reuse

Use the following provenance categories:

- **DIRECT-REUSE:** code/infrastructure can be directly adapted if license permits.
- **ADAPT:** conceptual or code-level adaptation; interface must be rewritten for SignRel-JEPA.
- **BASELINE:** use released model/checkpoint as external comparison.
- **REFERENCE-ONLY:** learn the idea; do not depend on code.
- **DO-NOT-REUSE-AS-CORE:** intentionally different from our core objective.

## 3.1 Source inheritance matrix

| Component | Upstream | Reuse level | What to use | What NOT to inherit |
|---|---|---:|---|---|
| EMA teacher/student paradigm | V-JEPA / V-JEPA 2 | DIRECT-REUSE / ADAPT | teacher copy, no-grad teacher, EMA momentum scheduler, checkpoint semantics | generic target definition is extended with relation targets |
| Video ViT backbone | V-JEPA 2 / V-JEPA 2.1 | DIRECT-REUSE / ADAPT | pretrained encoder initialization, tokenization, RoPE, tubelet patches | do not assume official giant-scale recipe is affordable |
| Mask collator infrastructure | V-JEPA 2 | ADAPT | multiblock/tube masking APIs and visible/target token bookkeeping | random mask alone is only baseline |
| Dense latent-prediction ideas | V-JEPA 2.1 | REFERENCE / OPTIONAL | dense prediction, intermediate supervision are later extensions | do not implement in MVP unless core relation method is stable |
| Sign-specific priors & regularization | SignRep | ADAPT / BASELINE | released checkpoint; Hiera baseline; optional variance/covariance regularizer; leakage prevention principle | do not make pose/keypoint reconstruction the proposed target |
| Multi-stream articulator preprocessing | SHuBERT | ADAPT / BASELINE | face/hand/body localization pipeline ideas; released weights as baseline | do not use clustered stream IDs as proposed target |
| Masked articulator prediction | SignSeek MAP | REFERENCE-ONLY / BASELINE REIMPLEMENTATION | stop-gradient latent prediction concept; retrieval protocol | do not use same-gloss SupCon/GAC/MAC in core SSL |
| Local sign-aware views | SignDINO | REFERENCE-ONLY | motivation that hands/face need sign-aware views | do not claim local hand/face masking itself is novel |
| Segmentation-driven masking | SignMAE | REFERENCE-ONLY | prior art for body-part/motion-aware masking | do not reduce proposed method to mask-and-reconstruct |
| Pixel masked autoencoding | VideoMAE / VideoMAEv2 | BASELINE | generic masked-video baseline | do not use as flagship target |
| Phonological metadata | ASL-LEX 2.0 | EVALUATION DATA | feature labels only | **never reuse/scrape ASL-LEX reference videos** |
| Phonological videos | Sem-Lex | EVALUATION DATA | videos + mapped phonological labels subject to ToU | respect access/license restrictions |

---

# 4. Authoritative External Sources

The coding agent should read these **before implementing inherited components**.

## 4.1 V-JEPA 2 / V-JEPA 2.1 — primary implementation substrate

- Official repo: https://github.com/facebookresearch/vjepa2
- README / model zoo: https://github.com/facebookresearch/vjepa2/blob/main/README.md
- Standard training loop: https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa/train.py
- V-JEPA 2.1 training loop: https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py
- Hub/backbone definitions: https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py
- Example V-JEPA 2.1 config:
  https://github.com/facebookresearch/vjepa2/blob/main/configs/train_2_1/vitl16/pretrain-256px-16f.yaml
- License: MIT, per upstream repository source headers/LICENSE.

Important implementation facts to verify from upstream:

- student encoder + predictor;
- EMA target encoder;
- target encoder `requires_grad=False`;
- multiblock/tube masking;
- mixed precision and distributed training;
- checkpoints carry encoder/predictor/target encoder/optimizer/scaler state;
- V-JEPA 2.1 pretrained model entry points include:
  - `vjepa2_1_vit_base_384`
  - `vjepa2_1_vit_large_384`
  - larger variants.

Example upstream loading pattern:

```python
backbone = torch.hub.load(
    "facebookresearch/vjepa2",
    "vjepa2_1_vit_base_384",
)
```

**Implementation policy:** use the official repo as architecture/checkpoint reference. Do not copy the full research repository into this project. Wrap the backbone behind a local adapter.

### Recommended backbone policy

For limited compute:

- first research-scale initialization: `V-JEPA 2.1 ViT-B/16 384`;
- maintain a lower-resolution debug mode;
- make backbone interchangeable via config.

384 px is attractive for sign because hand details occupy small image regions, but this is a **design hypothesis**, not a guaranteed optimum.

If the exact V-JEPA 2.1 training path for a chosen model is unstable/incompatible, fall back to a stable V-JEPA 2 encoder while keeping SignRel modules unchanged.

---

## 4.2 Original V-JEPA

- Official repo: https://github.com/facebookresearch/jepa
- Use as conceptual reference for the core JEPA principle:
  **predict representations in latent space rather than reconstructing pixels**.

---

## 4.3 SignRep — sign-specific SSL reference and baseline

Paper:

- https://arxiv.org/html/2503.08529v1

Official code:

- https://github.com/ryanwongsa/SignRep
- `models/`
- `augmentation/video/`
- `example_usage.py`
- README states checkpoints are published through repository releases.
- Model loading example in repo uses:
  `models.final_models.FINAL_hiera_latent_model_head_v25_active.Model`
- Video transform:
  `augmentation.video.base_video_aug.Transformation`
- License file currently declares **CC BY-NC-SA 4.0**. Treat code reuse as non-commercial/share-alike constrained unless legal review says otherwise.

Relevant ideas:

- replace generic pixel target with sign-specific structure;
- training-time pose priors, RGB-only downstream encoder;
- variance/covariance feature regularization;
- explicitly prevent WLASL ↔ YouTube pretraining overlap by video ID.

Paper implementation reference:

- Hiera-B initialized from Kinetics MAE;
- 16-frame pretraining clips;
- 80% masking;
- Planckian jitter / resized crop / blur / grayscale;
- long training schedule in the paper;
- only pose estimates above confidence threshold contribute to SignRep’s reconstruction target.

**Our usage:**

- use SignRep checkpoint as an **external frozen baseline**;
- optionally port the variance/covariance regularizer by reimplementing the equations;
- copy no CC BY-NC-SA code into a differently licensed project without explicit license compatibility review;
- do not reuse SignRep reconstruction targets as the flagship loss.

---

## 4.4 SHuBERT — contextual sign SSL and preprocessing reference

Paper / publication:

- ACL 2025: https://aclanthology.org/2025.acl-long.1397/
- Project/paper: https://arxiv.org/abs/2411.16765

Official repo:

- https://github.com/ShesterG/SHuBERT
- quick start:
  https://github.com/ShesterG/SHuBERT/blob/main/QUICKSTART.md
- datasets:
  https://github.com/ShesterG/SHuBERT/blob/main/DATASETS.md
- feature extraction:
  https://github.com/ShesterG/SHuBERT/blob/main/FEATURES.md
- pretraining entry:
  `train_shubert.sh`
- repo states project is primarily MIT licensed.
- README links pretrained SHuBERT plus DINO face/hand weights.

SHuBERT pipeline to understand:

1. signer crop;
2. MediaPipe landmarks;
3. face/hand crops;
4. DINO-derived face/hand features;
5. body pose features;
6. separate clustering per stream;
7. masked cluster prediction by transformer.

**Our usage:**

- borrow preprocessing ideas for articulator localization;
- use SHuBERT released weights as an external baseline when possible;
- do **not** make k-means cluster IDs the SignRel-JEPA target.

---

## 4.5 SignSeek — closest conceptual competitor

Paper:

- https://arxiv.org/html/2609.03695v1
- arXiv v1 date: 2026-09-03.

As of this specification’s verification date, treat this as a **paper-only reference unless an official code release is subsequently found**. Agent must re-check code availability before experiments.

Important components:

- pose streams: body, face, left hand, right hand;
- per-articulator encoders + temporal fusion;
- Articulator Saliency-Guided Masking;
- GAC: same-gloss supervised contrastive loss;
- MAC: dominant-articulator supervised contrastive path;
- MAP: mask dominant articulator, predict clean latent under stop-gradient.

SignSeek MAP is conceptually close to our auxiliary articulator-prediction loss. Therefore:

> **Do not claim “predict a masked articulator from context” as our novelty.**

Our differentiation must be:

1. relation target rather than only dominant articulator target;
2. multiple pairwise/graph relations;
3. multiple temporal scales;
4. no gloss-supervised objective in core pretraining;
5. RGB representation with pose/landmarks optional only as structural metadata.

For a future SignSeek-style baseline, reproduce only from paper equations and clearly mark it as an independent reimplementation.

---

## 4.6 SignDINO — sign-aware teacher/student local-view prior art

CVPR 2026:

- https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html

Key idea:

- teacher sees global sign frame;
- student sees masked/local hand and facial views;
- pretraining has no text supervision;
- global representation is used at inference.

**Our usage:** motivation/reference only. Do not claim that retaining/masking hands and face is novel by itself.

---

## 4.7 SignMAE — sign-aware masking prior art

Paper:

- https://arxiv.org/abs/2605.02094

Key idea:

- segmentation-driven masking adapts to presence/motion of key body parts;
- mask-and-reconstruct SSL.

**Our usage:** novelty guardrail. Structured body-part masking is a supporting mechanism, not the main contribution.

---

## 4.8 VideoMAE / VideoMAEv2 — generic masked-video baseline

- VideoMAE: https://github.com/MCG-NJU/VideoMAE
- VideoMAEv2: https://github.com/OpenGVLab/VideoMAEv2

Use only as controlled generic SSL baselines where compute allows.

---

# 5. Dataset Sources, Licensing, and Leakage Rules

## 5.1 YouTube-ASL — primary large-scale ASL pretraining candidate

Official dataset information:

- https://github.com/google-research/google-research/blob/master/youtube_asl/README.md
- paper: https://arxiv.org/abs/2306.15162

Reported upstream:

- 11,093 videos;
- 984 hours;
- video IDs are released rather than redistributing all source videos.

**Use:** ignore captions/text for core SSL.

**Agent rule:** store captions nowhere in the model batch. If dataset ingestion downloads caption metadata for segmentation, drop the text field before pretraining manifests.

---

## 5.2 YouTube-SL-25 — optional multilingual scale-up

Paper:

- https://arxiv.org/abs/2407.11144

Reported scale:

- >3,000 hours;
- >25 sign languages.

SignRep uses YouTube-SL-25 for pretraining.

**Use:** optional later-stage multilingual pretraining after MVP works.

**Agent rule:** first verify current official access mechanism, video-ID availability, and terms. Do not write an automated downloader that bypasses platform rules.

---

## 5.3 How2Sign — controlled continuous ASL source

Official site:

- https://how2sign.github.io/
- code/scripts: https://github.com/how2sign/how2sign-data

Reported:

- large continuous ASL dataset;
- dataset page states CC BY-NC 4.0.

**Use:** controlled pretraining/debug data while ignoring sentence translation.

**Split rule:** if How2Sign is later used for downstream evaluation, held-out evaluation videos must not have been included in SSL pretraining.

---

## 5.4 WLASL — downstream evaluation

Official repository:

- https://github.com/adiacla/WLASL

Metadata contains fields such as:

- `gloss`
- `bbox`
- `fps`
- `frame_start`
- `frame_end`
- `signer_id`
- `source`
- `split`
- `url`
- `variation_id`
- `video_id`

License is C-UDA; repo states academic/computational use and no commercial usage.

**Critical leakage rule:**

If pretraining from YouTube-derived corpora, remove any source `video_id` overlapping any WLASL evaluation source video. Implement an explicit overlap audit and persist a report.

---

## 5.5 ASL Citizen — signer-independent retrieval/recognition

Official Microsoft Research page:

- https://www.microsoft.com/en-us/research/project/asl-citizen/
- dataset description:
  https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/
- archived baseline code:
  https://github.com/microsoft/ASL-citizen-code

Reported:

- roughly 84k videos;
- roughly 2.7k signs;
- 52 Deaf/hard-of-hearing signers;
- signer-independent released splits.

Use as:

- frozen retrieval benchmark;
- frozen/linear recognition benchmark;
- signer-invariance probe where IDs are available.

---

## 5.6 Sem-Lex — phonological evaluation video source

Official repo:

- https://github.com/leekezar/SemLex

Repo reports:

- 91,148 isolated sign videos;
- vocabulary of 3,149 signs;
- 41 Deaf participants;
- alignment with ASL-LEX 2.0 and ASL SignBank.

Use subject to its Terms of Use.

---

## 5.7 ASL-LEX 2.0 — phonological labels only

Official:

- https://asl-lex.org/about.html
- https://asl-lex.org/download.html

ASL-LEX 2.0 reports:

- 2,723 signs;
- 60+ lexical/phonological properties;
- 23 phonological features.

License warning:

- database/visualization (excluding videos): CC BY-NC 4.0;
- **reference sign videos may not be saved, displayed, or reused without explicit permission.**

**Hard rule:** never scrape or use ASL-LEX reference videos in training/evaluation. Use its tabular metadata only and map permitted videos from Sem-Lex / ASL Citizen by lexical identifier where legally and technically valid.

---

## 5.8 NMFs-CSL — cross-language / non-manual evaluation

Official:

- https://ustc-slr.github.io/datasets/2020_nmfs_csl/

Reported:

- 1,067 Chinese sign words;
- designed to emphasize non-manual features;
- 30 FPS RGB;
- research-only access with signed release agreement.

Use only after access is properly granted.

---

# 6. Proposed Repository Layout

```text
signrel-jepa/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── train.txt
│   └── eval.txt
├── configs/
│   ├── pretrain/
│   │   ├── debug_tiny.yaml
│   │   ├── jepa_baseline.yaml
│   │   ├── articulator_mvp.yaml
│   │   ├── relation_mvp.yaml
│   │   └── relation_multiscale.yaml
│   ├── eval/
│   │   ├── retrieval_wlasl.yaml
│   │   ├── retrieval_asl_citizen.yaml
│   │   ├── linear_probe.yaml
│   │   ├── phonology_semlex.yaml
│   │   └── signer_probe.yaml
│   └── data/
├── signrel/
│   ├── data/
│   │   ├── manifest.py
│   │   ├── video_dataset.py
│   │   ├── sampler.py
│   │   ├── transforms.py
│   │   └── leakage.py
│   ├── preprocessing/
│   │   ├── signer_crop.py
│   │   ├── landmarks.py
│   │   ├── articulator_boxes.py
│   │   ├── motion.py
│   │   └── cache.py
│   ├── masks/
│   │   ├── base.py
│   │   ├── random_multiblock.py
│   │   ├── articulator.py
│   │   ├── relation_break.py
│   │   └── motion_salient.py
│   ├── models/
│   │   ├── backbone_adapter.py
│   │   ├── region_pooler.py
│   │   ├── relation_projector.py
│   │   ├── temporal_pyramid.py
│   │   ├── relation_predictor.py
│   │   ├── ema.py
│   │   └── signrel_jepa.py
│   ├── losses/
│   │   ├── latent.py
│   │   ├── relation.py
│   │   ├── regularization.py
│   │   └── total.py
│   ├── engine/
│   │   ├── train.py
│   │   ├── checkpoint.py
│   │   ├── distributed.py
│   │   └── logging.py
│   ├── eval/
│   │   ├── extract.py
│   │   ├── retrieval.py
│   │   ├── linear_probe.py
│   │   ├── phonology.py
│   │   ├── minimal_pairs.py
│   │   └── signer_probe.py
│   └── utils/
├── scripts/
│   ├── prepare_manifest.py
│   ├── extract_landmarks.py
│   ├── audit_leakage.py
│   ├── train.py
│   ├── extract_features.py
│   └── evaluate_all.py
├── tests/
│   ├── test_manifest_no_labels.py
│   ├── test_transform_alignment.py
│   ├── test_region_pool.py
│   ├── test_masks.py
│   ├── test_teacher_no_grad.py
│   ├── test_ema.py
│   ├── test_relation_target.py
│   ├── test_checkpoint_resume.py
│   └── test_leakage.py
└── third_party/
    └── PROVENANCE.md
```

---

# 7. Data Manifest Contract

Use JSONL or Parquet.

Minimal pretraining row:

```json
{
  "sample_id": "youtube_asl__abc123__001200_001360",
  "dataset": "youtube_asl",
  "language": "ase",
  "video_path": "/data/.../abc123.mp4",
  "source_video_id": "abc123",
  "start_sec": 40.0,
  "end_sec": 45.3,
  "fps": 30.0,
  "landmark_path": "/cache/landmarks/...npz",
  "split": "ssl_train"
}
```

Fields explicitly forbidden from the model-facing pretraining batch:

```text
gloss
class_id
translation
caption
text
signer_id
phonology_label
```

These may exist in a separate evaluation manifest.

Implement:

```python
FORBIDDEN_SSL_KEYS = {
    "gloss", "class_id", "translation", "caption",
    "text", "signer_id", "phonology_label"
}
```

At the final collate boundary:

```python
assert not (FORBIDDEN_SSL_KEYS & set(batch.keys()))
```

Do not merely trust that losses ignore them; do not provide them at all.

---

# 8. Preprocessing Pipeline

## 8.1 Stage P0 — decode and normalize video metadata

For each source video:

1. decode FPS, width, height, duration;
2. detect corrupted/zero-frame files;
3. normalize timestamps;
4. record SHA256 or stable source identifier where feasible;
5. never silently resample metadata without recording it.

## 8.2 Stage P1 — signer/person crop

Use one of:

- existing dataset bbox where reliable;
- lightweight person detector;
- full frame as fallback.

Store normalized crop box:

\[
b_t=[x_1,y_1,x_2,y_2]\in[0,1]^4.
\]

Do not permanently destructively crop raw data. Crop at load time or cache a derived representation.

## 8.3 Stage P2 — landmarks

MVP recommendation:

- use MediaPipe-derived landmarks, following SHuBERT’s practical preprocessing precedent;
- output:
  - left hand landmarks + confidence;
  - right hand landmarks + confidence;
  - face landmarks or selected facial anchors + confidence;
  - upper-body landmarks + confidence.

Offline cache format:

```python
{
    "xy": float32[T, J, 2],       # normalized 0..1
    "confidence": float32[T, J],
    "joint_type": int16[J],
    "valid_frame": bool[T],
    "source_size": int32[2],
    "version": "...",
}
```

No gradient through landmark detector.

## 8.4 Stage P3 — articulator boxes

Define:

```python
ARTICULATORS = ["left_hand", "right_hand", "face", "body"]
```

For each frame/articulator:

1. collect valid landmarks;
2. compute tight bbox;
3. enlarge by configurable margin;
4. clip to [0, 1];
5. assign confidence from valid landmark fraction / mean confidence;
6. temporally smooth box center/size;
7. interpolate short missing gaps only;
8. mark long missing gaps invalid.

Recommended initial margins:

```yaml
boxes:
  left_hand_margin: 0.35
  right_hand_margin: 0.35
  face_margin: 0.20
  body_margin: 0.10
  min_confidence: 0.5
  max_interp_gap_frames: 3
```

These values are proposed engineering defaults, not inherited paper constants.

## 8.5 Stage P4 — motion metadata

Compute an inexpensive motion score from landmarks:

\[
m_t^a =
\frac{1}{J_a}
\sum_j
\left\|
p_{t,j}^{a}-p_{t-1,j}^{a}
\right\|_2.
\]

Normalize by torso size or shoulder distance when available.

Store per-articulator and global motion:

```python
motion: float32[T, 4]
global_motion: float32[T]
```

No optical flow is required for MVP.

---

# 9. Spatial Augmentation Must Transform Landmarks Too

This is a critical engineering requirement.

If the video receives:

- resize,
- crop,
- horizontal flip,

the articulator coordinates must receive the exact same transform.

Implement a transform object that returns both video and transform parameters:

```python
video, transform_record = transform.sample_and_apply(video)
landmarks = transform_record.apply_points(landmarks)
boxes = transform_record.apply_boxes(boxes)
```

For horizontal flip:

- x coordinate becomes `1 - x`;
- **swap left/right semantic identities** after flipping.

Failure to swap LH/RH invalidates relation targets.

Color transforms do not affect coordinates.

Suggested augmentations:

- random resized crop;
- mild scale/aspect variation;
- grayscale with low probability;
- Gaussian blur;
- brightness/color perturbation;
- optional Planckian jitter inspired by SignRep.

Avoid overly destructive augmentations that remove finger information before mask construction.

---

# 10. Backbone Interface

Every backbone adapter must emit token geometry.

Required interface:

```python
class VideoBackboneAdapter(nn.Module):
    def forward_tokens(
        self,
        video: torch.Tensor,  # [B, C, T, H, W]
        visible_mask=None,
    ) -> dict:
        """
        Returns:
            tokens: [B, Tt, Ht, Wt, D] or losslessly reshapeable equivalent
            grid_shape: (Tt, Ht, Wt)
            global_feature: optional [B, D]
        """
```

The rest of SignRel-JEPA MUST NOT depend on a specific ViT implementation.

Default expectations for a tubelet ViT:

```text
input:     [B, 3, T, H, W]
tubelet:   2
patch:     16 x 16
token grid:
Tt = T / tubelet
Ht = H / patch
Wt = W / patch
```

Always query/derive actual grid size from the model; never hardcode it.

---

# 11. Convert Articulator Boxes to Token Membership

Given boxes in normalized image coordinates and token grid \(H_t\times W_t\), construct soft or binary membership:

\[
M^{a}_{b,t,h,w}\in[0,1].
\]

MVP: token-center membership.

Better later: patch-box intersection-over-area.

Tensor:

```text
articulator_membership:
[B, Tt, A=4, Ht, Wt]
```

Temporal reduction from raw frames to tubelets:

- aggregate raw-frame boxes belonging to a tubelet;
- default union/mean center box;
- confidence = max or mean valid confidence.

Invalid articulator/time positions receive zero weight and are excluded from relation loss.

---

# 12. Region Pooler

Teacher full tokens:

\[
Z^T\in\mathbb{R}^{B\times T'\times H'\times W'\times D}.
\]

For articulator \(a\):

\[
z^{a}_{b,t}
=
\frac{\sum_{h,w}
M^{a}_{b,t,h,w} Z^T_{b,t,h,w}}
{\sum_{h,w}M^{a}_{b,t,h,w}+\epsilon}.
\]

Output:

```text
node_latents: [B, Tt, 4, D]
node_valid:   [B, Tt, 4]
```

MVP pooler must be deterministic and parameter-free.

Optional later improvement:

- query-based attentive region pooling.

Do not start with learnable pooling; it complicates interpretation.

---

# 13. Relation Graph

Initial edge set:

```python
EDGES = [
    ("left_hand", "right_hand"),
    ("left_hand", "face"),
    ("right_hand", "face"),
    ("left_hand", "body"),
    ("right_hand", "body"),
]
```

Do not include `face-body` in MVP unless diagnostics show value.

Let \(E=5\).

## 13.1 Teacher relation projector

For an edge \((a,b)\), construct:

\[
u^{a,b}_t =
[
z^a_t,\,
z^b_t,\,
z^a_t-z^b_t,\,
z^a_t\odot z^b_t
].
\]

Then:

\[
r^{a,b}_t =
R_\xi(u^{a,b}_t)
\]

where `R_teacher` is EMA-copied from `R_student`.

Recommended MVP:

```text
input dim: 4D
LayerNorm
Linear(4D -> 2Dr)
GELU
Linear(2Dr -> Dr)
LayerNorm
Dr = 256 or 384
```

Avoid explicit raw landmark geometry in the default relation target. This makes the main result depend on **visual contextual latents**, not geometry supervision.

Add geometry as an ablation later:

```text
relative center dx, dy
relative scale ratio
box overlap/contact proxy
motion difference
```

## 13.2 Why EMA the relation projector?

If only the encoder is EMA but the target relation head updates directly by gradient, the target itself moves inconsistently.

Maintain:

```text
student encoder      f_theta
student relation     R_theta
teacher encoder      f_xi
teacher relation     R_xi
```

EMA both target components:

\[
\xi\leftarrow m\xi+(1-m)\theta.
\]

No gradients to teacher modules.

---

# 14. Student Relation Predictor

The student sees masked context.

A relation prediction query identifies:

- edge ID;
- temporal position/span;
- temporal scale.

Use embeddings:

```python
edge_embed[e]
scale_embed[s]
time_embed[t]
```

The relation predictor receives:

- visible/context student tokens;
- query token(s).

Two valid MVP implementations:

### Option A — cross-attention predictor — recommended

```text
query = edge_embedding + time_embedding + scale_embedding
K,V = projected visible student tokens
Transformer cross-attention
MLP -> Dr
```

### Option B — V-JEPA-style predictor adapter

Adapt upstream predictor machinery so target-mask locations carry relation query IDs rather than ordinary patch target IDs.

**Recommendation:** start with Option A because it separates research logic from V-JEPA patch predictor internals and is easier to test. Reuse V-JEPA’s training/EMA/masking infrastructure, not necessarily its exact predictor implementation.

Output:

```text
pred_relation: [B, Q, Dr]
target_relation: [B, Q, Dr]
query_valid: [B, Q]
```

---

# 15. Masking Strategies

Implement every mask strategy behind the same interface.

```python
class SignMaskGenerator:
    def sample(
        self,
        token_grid,
        articulator_membership,
        motion,
        rng,
    ) -> MaskSpec:
        ...
```

`MaskSpec`:

```python
@dataclass
class MaskSpec:
    visible_tokens: Tensor
    hidden_tokens: Tensor
    relation_queries: list
    metadata: dict
```

## 15.1 Random multiblock/tube mask — baseline

Adapt V-JEPA mask collator infrastructure.

Purpose:

- establish generic JEPA baseline;
- regularize model.

## 15.2 Articulator mask

Choose valid articulator \(a\) and temporal span.

Mask tokens overlapping its region.

Student predicts:

- optional node/articulator latent \(z^a\);
- relations containing \(a\).

## 15.3 Relation-breaking mask — flagship

Choose edge \((a,b)\).

Mask one endpoint over a temporal span while leaving useful surrounding context.

Example:

```text
edge: LH-RH
hide: RH tokens at t=5..10
visible: LH + face + body + RH outside span
target: teacher relation r^(LH,RH) at t=5..10
```

This creates the intended reasoning task:

> infer the missing relation from context, not simply copy both visible endpoints.

Safety rules:

- never hide all articulators for the entire span;
- enforce minimum visible-token ratio;
- skip query if either teacher endpoint is invalid;
- prevent masked target tokens from entering student context.

## 15.4 Motion-salient mask

Sample temporal spans with probability proportional to motion energy.

Within the selected span, prioritize hand regions.

This is a supporting strategy, not the novelty.

## 15.5 Initial mask mixture

Proposed MVP defaults:

```yaml
masking:
  mixture:
    random_multiblock: 0.30
    articulator: 0.30
    relation_break: 0.30
    motion_salient: 0.10
```

Full-study candidate:

```yaml
masking:
  mixture:
    random_multiblock: 0.15
    articulator: 0.25
    relation_break: 0.45
    motion_salient: 0.15
```

These are **our proposed defaults** and must be tuned/ablated.

---

# 16. Core Forward Pass

Pseudo-code:

```python
def forward_ssl(batch):
    x = batch["video"]                 # [B,C,T,H,W]
    regions = batch["articulator_meta"]

    mask_spec = mask_generator.sample(...)

    # ----- teacher: clean/full clip -----
    with torch.no_grad():
        zt_grid = teacher_encoder.forward_tokens(x)["tokens"]
        zt_nodes, node_valid = region_pooler(zt_grid, regions)

        zt_rel_short = teacher_relation_projector(zt_nodes, EDGES)

        targets = temporal_target_builder(
            zt_rel_short,
            node_valid=node_valid,
            scales=cfg.model.temporal_scales,
        )

    # ----- student: masked context -----
    zs = student_encoder.forward_tokens(
        x,
        visible_mask=mask_spec.visible_tokens
    )

    pred_rel = relation_predictor(
        context_tokens=zs["tokens"],
        queries=mask_spec.relation_queries,
    )

    target_rel = gather_teacher_targets(
        targets,
        mask_spec.relation_queries,
    )

    loss_rel = relation_loss(pred_rel, target_rel)

    # optional auxiliary node/articulator latent prediction
    loss_art = maybe_articulator_loss(...)

    loss_reg = maybe_feature_regularization(...)

    loss = (
        cfg.loss.rel * loss_rel
        + cfg.loss.art * loss_art
        + cfg.loss.reg * loss_reg
    )

    return loss, metrics
```

Teacher must run under `torch.no_grad()` and all teacher parameters must have `requires_grad=False`.

---

# 17. Relation Loss

Primary MVP loss should be aligned with JEPA-style latent regression.

Normalize target and prediction:

\[
\tilde r =
LN(r).
\]

Default:

\[
\mathcal L_{rel}
=
\frac{1}{|\mathcal Q|}
\sum_{q\in\mathcal Q}
\left\|
LN(\hat r_q)
-
sg(LN(r_q))
\right\|_1.
\]

Config:

```yaml
loss:
  relation:
    type: l1
    normalize: layernorm
```

Ablate:

- Smooth L1 / Huber;
- cosine:

\[
1-\cos(\hat r,r).
\]

SignSeek MAP uses a cosine-style stop-gradient latent prediction objective, while V-JEPA uses latent feature prediction. Do not present either loss function itself as novel.

---

# 18. Auxiliary Articulator Latent Loss

This is **not** the main contribution.

For masked articulator query:

\[
\mathcal L_{art}
=
\left\|
LN(\hat z^{a})
-
sg(LN(z^{a,T}))
\right\|_1.
\]

Config:

```yaml
loss:
  art: 0.25
  rel: 1.0
```

Essential ablation:

```text
generic JEPA
+ articulator latent
+ relational latent
```

If relation loss does not beat articulator-only under controlled compute, the central hypothesis is not supported.

---

# 19. Multi-Scale Temporal Relation Targets

Do not add this until single-scale relational MVP works.

Given short-scale teacher relations:

```text
R0: [B, Tt, E, Dr]
```

Construct:

```text
short: identity / kernel 1
medium: temporal average or learned pooling over ~4 token steps
long: temporal average or learned pooling over ~8 token steps
```

Example for 32 input frames, tubelet=2:

```text
Tt = 16
short  k=1
medium k=4
long   k=8
```

Target:

\[
r^{e,s}_{t}
=
Pool_s(
r^e_{t:t+k_s}
).
\]

MVP use deterministic average pooling to make results interpretable.

Only after validation consider:

- temporal convolution;
- attention pooling;
- multi-layer teacher targets.

Loss:

\[
\mathcal L_{multi}
=
\sum_s \alpha_s \mathcal L_{rel}^{(s)}.
\]

Initial:

```yaml
temporal_scales:
  - {name: short, kernel: 1, weight: 1.0}
  - {name: medium, kernel: 4, weight: 1.0}
  - {name: long, kernel: 8, weight: 1.0}
```

---

# 20. Anti-Collapse / Representation Regularization

Start without extra regularization if EMA latent prediction is stable.

If feature collapse/low variance occurs, add optional VICReg-like terms inspired by SignRep’s use of variance/covariance regularization.

For pooled student global features \(z\):

Variance:

\[
\mathcal L_{var}
=
\frac{1}{D}
\sum_j
\max(0,\gamma-\sigma_j).
\]

Covariance:

\[
\mathcal L_{cov}
=
\frac{1}{D}
\sum_{j\ne k}
C_{jk}^2.
\]

Config:

```yaml
regularization:
  enabled: false
  variance_weight: 1.0
  covariance_weight: 0.004
  gamma: 1.0
```

**Important:** the example weights above reflect SignRep-style scale references and are not guaranteed optimal for this model. Tune them rather than assuming transferability.

Do not add SignRep’s adversarial style discriminator in MVP. First measure signer leakage. Add explicit invariance only if necessary.

---

# 21. Total Objective by Research Stage

## Stage B — generic JEPA baseline

\[
L=L_{patch/latent}
\]

or a local baseline implementation of masked latent prediction.

## Stage C — articulator-only

\[
L=L_{art}.
\]

## Stage D — relation MVP

\[
\boxed{
L=L_{rel}+\lambda_{art}L_{art}
}
\]

Default:

```yaml
lambda_rel: 1.0
lambda_art: 0.25
```

## Stage E — multi-scale flagship

\[
L=
\lambda_{rel}
\sum_s \alpha_s L_{rel}^{s}
+
\lambda_{art}L_{art}
+
\lambda_{reg}L_{reg}.
\]

Every term must be disableable.

---

# 22. EMA Teacher Update

Use standard EMA:

\[
\xi\leftarrow m\xi+(1-m)\theta.
\]

EMA applies to:

- teacher backbone;
- teacher relation projector;
- any teacher-side projection head.

Never EMA optimizer state.

Recommended starting schedule:

```yaml
ema:
  start: 0.996
  end: 1.0
  schedule: cosine
```

This is a proposed starting point. Compare with current V-JEPA upstream configs for the chosen architecture.

Unit test:

```python
# after one update:
expected = m * old_teacher + (1-m) * new_student
assert_allclose(actual_teacher, expected)
```

---

# 23. Training Configuration

## 23.1 Debug config

Goal: validate code, not representation quality.

```yaml
data:
  num_frames: 16
  crop_size: 224
  batch_size_per_gpu: 2
  workers: 2

model:
  backbone: tiny_video_vit
  tubelet: 2
  patch: 16
  relation_dim: 128

trainer:
  max_steps: 500
  precision: bf16
  grad_clip: 1.0
```

## 23.2 MVP research config

```yaml
seed: 42

data:
  num_frames: 16
  sampling_stride: 2
  crop_size: 384
  batch_size_per_gpu: auto
  drop_last: true

model:
  backbone:
    family: vjepa2_1
    name: vjepa2_1_vit_base_384
    pretrained: true
  relation_dim: 384
  temporal_scales:
    - [short, 1]

masking:
  mixture:
    random_multiblock: 0.30
    articulator: 0.30
    relation_break: 0.30
    motion_salient: 0.10

loss:
  relation_weight: 1.0
  articulator_weight: 0.25
  regularization_weight: 0.0
  relation_type: l1

optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.04
  betas: [0.9, 0.95]

scheduler:
  warmup_fraction: 0.05
  type: cosine

ema:
  start: 0.996
  end: 1.0

trainer:
  precision: bf16
  gradient_checkpointing: true
  grad_clip: 1.0
  accumulate_steps: auto
```

These are **initial engineering defaults**, not paper-validated hyperparameters.

## 23.3 Full relation config

Only after MVP supports H1:

- increase frames to 32;
- add short/medium/long targets;
- increase data scale;
- test 64 frames if compute permits;
- test multilingual data.

---

# 24. Initialization Experiments

Keep pretraining initialization separate from method contribution.

Required comparisons:

1. **generic pretrained initialization**
   - V-JEPA 2/2.1;
2. optional **from-scratch or generic action/video pretrained**;
3. external frozen **SignRep** baseline;
4. external frozen **SHuBERT** baseline where input/protocol is compatible.

Do not initialize the proposed model from SignRep for the primary comparison against SignRep; that would confound attribution.

When using V-JEPA initialization, describe the experiment as:

> self-supervised domain adaptation / continued self-supervised pretraining on sign video.

Do not claim the entire representation was learned from scratch.

---

# 25. Optimizer and Scheduling

Use AdamW initially.

Do not inherit SignRep or SignSeek hyperparameters blindly because architectures/data differ.

Agent should implement automatic effective batch reporting:

```text
effective_batch =
batch_per_gpu * num_gpus * grad_accumulation
```

LR scaling experiment:

```text
lr_scaled = base_lr * effective_batch / reference_batch
```

but keep an option for fixed LR.

Log:

- LR;
- WD;
- EMA momentum;
- grad norm;
- AMP scaler;
- samples/sec;
- valid relation queries/sample.

---

# 26. Checkpoint Contract

Checkpoint must include:

```python
{
    "student_encoder": ...,
    "student_relation_projector": ...,
    "relation_predictor": ...,
    "teacher_encoder": ...,
    "teacher_relation_projector": ...,
    "optimizer": ...,
    "lr_scheduler": ...,
    "scaler": ...,
    "global_step": ...,
    "epoch": ...,
    "rng_state": ...,
    "config": ...,
    "dataset_manifest_hash": ...,
    "git_commit": ...,
    "upstream_init": {
        "name": ...,
        "url": ...,
        "sha256": ...
    }
}
```

Resume test:

1. train N steps;
2. save;
3. resume;
4. compare with uninterrupted N+M run under deterministic debug setup.

---

# 27. Feature Extraction API

Downstream must use RGB only.

```python
@torch.no_grad()
def encode_video(video) -> dict:
    """
    returns:
      token_features
      frame_or_tube_features
      global_embedding
    """
```

Default global embedding:

1. obtain backbone tokens;
2. mean pool space/time or use validated backbone pooling;
3. LayerNorm;
4. L2 normalize for retrieval.

Do **not** require articulator boxes at downstream inference for the flagship claim.

Optional diagnostic path may output region features if landmark metadata exists, but it is not the default inference API.

---

# 28. Evaluation Protocols

## 28.1 Frozen retrieval — primary representation test

Follow the spirit of SignRep/SignSeek frozen dictionary retrieval.

For each video:

\[
z=\frac{f(x)}{\|f(x)\|_2}.
\]

Cosine similarity:

\[
s(q,g)=q^\top g.
\]

For a gloss with multiple gallery instances, score can use max similarity as in SignSeek-style protocol:

\[
S(q,c)=
\max_{g:y_g=c}
q^\top g.
\]

Metrics:

- Recall@1;
- Recall@5;
- mAP;
- DCG if matching established protocol.

Datasets:

- WLASL;
- ASL Citizen;
- NMFs-CSL after licensed access.

Always report whether any evaluation corpus was seen during pretraining.

## 28.2 Frozen linear probe

Freeze encoder completely.

Train only:

```text
LayerNorm(optional)
Linear(D -> #classes)
```

Report:

- per-instance Top-1 / Top-5;
- per-class accuracy;
- mean/std over at least 3 seeds when feasible.

## 28.3 Fine-tuning

Secondary task only.

Use same recipe for all compared initializations.

## 28.4 Phonological probe — high priority

Use Sem-Lex videos mapped to ASL-LEX 2.0 metadata.

For each phonological property with sufficient coverage:

```text
handshape
major location
minor location
movement
selected fingers / finger configuration
contact
sign type
two-handedness / symmetry-related properties
...
```

Train one frozen-feature classifier per property.

Metrics:

- macro F1;
- balanced accuracy;
- sample count;
- majority baseline.

Avoid a random sample split that leaks identical glosses or signer identity in a misleading way. Report at least:

1. signer-independent split;
2. stricter gloss-aware/generalization split when feasible.

## 28.5 Minimal-pair diagnostic

Construct pairs from legal metadata/video sources where signs differ in one target phonological feature while matching as many other features as possible.

Measure:

```text
same-sign different-signer distance
minimal-pair distance
unrelated-sign distance
```

Desired tendency:

\[
d(same\ sign,different\ signer)
<
d(minimal\ pair)
<
d(unrelated).
\]

Also test retrieval confusion within minimal-pair sets.

## 28.6 Signer leakage probe

Freeze representation.

Train a lightweight classifier to predict signer ID.

Interpretation must be careful:

- lower signer prediction is desirable **only if linguistic performance is maintained**;
- do not optimize signer probe alone.

Report a Pareto view:

```text
x: phonological / retrieval performance
y: signer prediction accuracy
```

---

# 29. Required Baselines

Minimum:

1. random initialization / supervised task baseline;
2. generic pretrained V-JEPA frozen;
3. generic JEPA continued on same sign data with random masking;
4. articulator-latent-only variant;
5. SignRel relation variant;
6. SignRel + multi-scale;
7. SignRep released checkpoint;
8. SHuBERT released checkpoint if protocol/input can be made fair;
9. VideoMAE-family generic baseline if compute allows.

Paper-only methods without released code/checkpoint must not be presented as locally reproduced unless actually reimplemented and validated.

For SignSeek:

- cite reported paper values separately;
- if reimplemented, label `SignSeek-MAP reproduction`, not “official SignSeek”.

---

# 30. Ablation Matrix

The following ablations are mandatory for a serious paper.

| Ablation | Values |
|---|---|
| target | generic latent / articulator latent / relation latent |
| relation graph | LH-RH only / hand-face / full five-edge graph |
| masking | random / articulator / relation-break / mixture |
| temporal | short only / short+medium / short+medium+long |
| face | on / off |
| relation projector | concat / concat+diff+product |
| target loss | L1 / smooth-L1 / cosine |
| resolution | 224/256 vs 384 |
| frames | 16 / 32 / optionally 64 |
| pose dependency | landmark-guided mask / random or learned salience |
| initialization | generic V-JEPA / alternative |
| geometry in target | none / optional relative geometry |
| regularization | none / variance-covariance |

Keep backbone, data, effective batch, steps, and optimizer constant within each ablation family.

---

# 31. MVP Experiment Sequence

Do not build every feature at once.

## Milestone M0 — environment and source audit

Deliverables:

- `third_party/PROVENANCE.md`;
- dependency lock;
- upstream repos/commits documented;
- licenses checked;
- dataset terms recorded.

Acceptance:

```text
pytest passes
backbone checkpoint loads
one video decodes
one landmark file is generated
```

## M1 — data + geometry pipeline

Deliverables:

- manifests;
- person crop;
- landmark cache;
- articulator boxes;
- joint spatial augment transforms;
- token membership.

Visualization QA:

save debug frames showing:

- crop;
- LH/RH/face/body boxes;
- token-grid overlay;
- chosen masked patches.

Manually inspect at least 100 diverse clips before training.

## M2 — generic JEPA baseline

Use same backbone and sign pretraining corpus but generic mask.

Goal:

- establish stable teacher/student training;
- verify non-collapse;
- create fair baseline.

## M3 — articulator prediction

Add `L_art`.

This is a validation bridge to SignSeek-like masked latent reasoning.

## M4 — relation MVP — decisive experiment

Add:

- teacher relation projector;
- relation queries;
- relation-breaking mask;
- `L_rel`.

Run controlled comparison:

```text
A generic JEPA
B A + articulator target
C A + relation target
D A + articulator + relation
```

**Go/no-go criterion:** relation variants should improve at least two representation-focused metrics (e.g., frozen retrieval + phonological probe) consistently enough to justify further scaling.

## M5 — multi-scale

Only after M4.

Add medium/long relation targets.

## M6 — scale

Increase:

- clip length;
- pretraining hours;
- multilingual data;
- model size if justified.

## M7 — final diagnostics

Run:

- retrieval;
- linear probe;
- phonology;
- signer leakage;
- minimal pairs;
- robustness to occlusion/noisy localization.

---

# 32. Unit Tests

## 32.1 No-label test

```python
def test_ssl_batch_has_no_semantic_labels():
    batch = next(iter(ssl_loader))
    forbidden = {...}
    assert forbidden.isdisjoint(batch.keys())
```

## 32.2 Teacher gradient test

```python
loss.backward()
for p in teacher.parameters():
    assert p.grad is None
```

## 32.3 EMA test

Check exact update on a tiny parameter tensor.

## 32.4 Coordinate transform test

Create synthetic landmarks at known positions.

Apply crop/resize/flip.

Assert transformed boxes exactly match rendered point locations.

## 32.5 Left-right flip test

After horizontal flip:

```text
old LH spatial points -> new RH semantic slot
old RH spatial points -> new LH semantic slot
```

## 32.6 Region pooling test

Use synthetic token grid with unique cell IDs.

Verify pooler selects only intended patches.

## 32.7 Relation validity test

If either endpoint invalid:

```text
relation_query_valid == False
loss contribution == 0
```

## 32.8 Mask leakage test

For relation-breaking mask, verify target endpoint tokens are absent from student visible tokens during the target span.

## 32.9 Data leakage test

Given downstream held-out `source_video_id` set:

```python
assert pretrain_ids.isdisjoint(eval_source_ids)
```

## 32.10 Checkpoint resume test

Deterministic debug-mode resumed training should match uninterrupted training within numerical tolerance.

---

# 33. Training Health Metrics

Log every N steps:

```text
loss_total
loss_relation
loss_articulator
loss_reg
teacher_target_std
student_embedding_std
teacher_embedding_std
relation_target_std_per_edge
valid_relation_queries_per_sample
masked_token_ratio
mask_type_frequency
EMA_momentum
grad_norm
learning_rate
weight_decay
GPU memory
throughput
```

Edge diagnostics:

```text
LH-RH loss
LH-Face loss
RH-Face loss
LH-Body loss
RH-Body loss
```

This is important: a low total relation loss may hide one collapsed/easy edge.

---

# 34. Collapse Alarms

Stop or flag run if, for a sustained window:

```text
teacher target feature std ~ 0
student global feature std ~ 0
cosine similarity between unrelated samples -> ~1
one relation edge target is nearly constant
valid relation queries/sample too low
```

Do not automatically “fix” collapse by adding many losses. Diagnose:

1. mask leakage;
2. EMA momentum;
3. normalization;
4. target projector;
5. batch/data diversity;
6. then regularization.

---

# 35. Visualization / Debug Tools

Implement a research dashboard or saved artifacts:

1. original frames;
2. transformed frames;
3. landmark boxes;
4. token grid;
5. mask type;
6. relation edge being predicted;
7. temporal target span;
8. nearest neighbors in frozen embedding space;
9. UMAP/t-SNE only as qualitative supplement.

Do not treat t-SNE as quantitative evidence.

---

# 36. Data Leakage Audit

Create:

```bash
python scripts/audit_leakage.py \
  --pretrain manifests/pretrain.parquet \
  --eval manifests/wlasl_test.parquet \
  --key source_video_id \
  --out reports/leakage_wlasl.json
```

Audit levels:

1. exact source video ID;
2. exact file hash;
3. source URL;
4. optional perceptual duplicate detection.

If overlap exists:

- remove entire source video from pretraining, not only the matching segment;
- regenerate manifest hash;
- retrain affected final experiments.

---

# 37. Provenance File Format

`third_party/PROVENANCE.md` should contain entries like:

```markdown
## V-JEPA 2

- Repository: https://github.com/facebookresearch/vjepa2
- Commit: <HASH>
- License: MIT
- Accessed: 2026-10-06
- Files consulted:
  - app/vjepa/train.py
  - app/vjepa_2_1/train.py
  - src/hub/backbones.py
- Local adaptations:
  - EMA update pattern -> signrel/models/ema.py
  - mask collator interface -> signrel/masks/random_multiblock.py
- Copied verbatim code: NONE / list exact ranges if any
```

Do this for every external repository.

---

# 38. Reproducibility Metadata

Every run directory:

```text
runs/<run_id>/
├── config.resolved.yaml
├── command.txt
├── git_commit.txt
├── git_diff.patch
├── environment.txt
├── pip_freeze.txt
├── cuda_info.txt
├── dataset_manifest.sha256
├── upstream_checkpoints.json
├── metrics.jsonl
├── checkpoints/
└── eval/
```

Store only metadata/paths allowed by dataset terms; do not redistribute restricted videos.

---

# 39. Recommended CLI

```bash
# Preprocess
python scripts/prepare_manifest.py \
  --dataset youtube_asl \
  --out manifests/youtube_asl_ssl.parquet

python scripts/extract_landmarks.py \
  --manifest manifests/youtube_asl_ssl.parquet \
  --out-root cache/landmarks

python scripts/audit_leakage.py \
  --pretrain manifests/youtube_asl_ssl.parquet \
  --eval manifests/wlasl_all.parquet

# Debug train
torchrun --nproc_per_node=1 scripts/train.py \
  --config configs/pretrain/debug_tiny.yaml

# Relation MVP
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/pretrain/relation_mvp.yaml

# Extract frozen features
python scripts/extract_features.py \
  --checkpoint runs/.../best.pt \
  --dataset wlasl \
  --out features/wlasl_signrel.npz

# Evaluation
python scripts/evaluate_all.py \
  --features features/wlasl_signrel.npz \
  --config configs/eval/retrieval_wlasl.yaml
```

---

# 40. Minimal Model Skeleton

```python
class SignRelJEPA(nn.Module):
    def __init__(
        self,
        student_encoder,
        teacher_encoder,
        student_relation_projector,
        teacher_relation_projector,
        relation_predictor,
        region_pooler,
        mask_generator,
        temporal_target_builder,
    ):
        super().__init__()
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        self.student_relation_projector = student_relation_projector
        self.teacher_relation_projector = teacher_relation_projector
        self.relation_predictor = relation_predictor
        self.region_pooler = region_pooler
        self.mask_generator = mask_generator
        self.temporal_target_builder = temporal_target_builder

        freeze(self.teacher_encoder)
        freeze(self.teacher_relation_projector)

    @torch.no_grad()
    def build_targets(self, x, articulator_meta):
        tokens = self.teacher_encoder.forward_tokens(x)["tokens"]
        nodes, valid = self.region_pooler(tokens, articulator_meta)
        rel = self.teacher_relation_projector(nodes, valid)
        return self.temporal_target_builder(rel, valid)

    def forward(self, x, articulator_meta):
        mask = self.mask_generator.sample(...)
        with torch.no_grad():
            targets = self.build_targets(x, articulator_meta)

        context = self.student_encoder.forward_tokens(
            x, visible_mask=mask.visible_tokens
        )

        pred = self.relation_predictor(
            context["tokens"],
            queries=mask.relation_queries
        )

        target = gather_targets(targets, mask.relation_queries)
        return {
            "pred_relation": pred,
            "target_relation": target.detach(),
            "mask": mask,
        }
```

---

# 41. Research Failure Modes and Decision Rules

## Failure A — relation loss improves pretraining loss but not downstream

Action:

- inspect whether relation target is too easy;
- strengthen relation-breaking masks;
- remove direct access to both endpoints;
- test edge-specific metrics.

Do not simply increase model size.

## Failure B — articulator-only beats relation model

Interpretation:

- relation definition/projector may not encode useful additional information;
- relation graph may be poorly chosen;
- node targets may already contain relation context because backbone tokens are contextual.

Experiments:

1. create relation targets from earlier/intermediate backbone layers;
2. make relation target spatially localized before global contextual mixing;
3. compare deterministic relation feature vs learned projector;
4. test minimal-pair-specific performance.

## Failure C — signer leakage rises

Test:

- stronger appearance augmentation;
- relation-only pooling;
- optional SignRep-inspired variance/covariance;
- only then consider adversarial signer/style regularization.

Never use actual signer IDs in the core SSL loss if maintaining the strongest label-free claim.

## Failure D — landmark noise dominates

Test:

- confidence gating;
- larger boxes;
- temporal smoothing;
- learned saliency/random masks;
- robustness curve vs synthetic box jitter.

A strong paper result would show the method degrades gracefully when localization is noisy.

---

# 42. Suggested Paper-Level Experimental Table Structure

## Main table

Columns:

```text
Method
Pretraining labels? (none/gloss/text)
Input at inference
Pretraining data
Params
WLASL R@1/R@5/mAP
ASL Citizen R@1/R@5/mAP
Phonology Macro-F1
Signer probe accuracy
```

## Ablation table

```text
Generic latent target
+ articulator target
+ pairwise relation target
+ relation-breaking mask
+ multi-scale relation
```

## Target-type table

```text
pixel
pose prior
cluster/discrete pseudo-unit
continuous individual latent
continuous relation latent
```

Only implement pixel/pose/cluster alternatives if compute permits; otherwise compare to published/released baselines without pretending all targets were trained under identical code.

---

# 43. Claims the Final Paper May Make Only If Supported

Allowed if experiments confirm:

- “label-free/self-supervised pretraining on the target sign data”;
- “RGB-only downstream inference”;
- “relation prediction improves frozen transfer”;
- “representation better preserves selected phonological factors”;
- “reduced signer leakage at matched task quality”;
- “better cross-dataset/cross-language transfer.”

Do NOT claim without targeted evidence:

- “learns phonology”;
- “disentangles phonological factors”;
- “language universal”;
- “signer invariant”;
- “first relational SSL for sign language”;
- “state of the art.”

For “first” claims, rerun prior-art search immediately before submission.

---

# 44. Implementation Priority: What the Agent Should Build First

Strict order:

```text
1. provenance + environment
2. manifests + leakage audit
3. video decode
4. landmark cache
5. coordinate-consistent augmentation
6. token membership + visualization
7. generic student/EMA teacher baseline
8. deterministic region pooler
9. relation projector
10. relation-breaking mask
11. relation predictor
12. relation loss
13. frozen retrieval evaluation
14. phonological diagnostic
15. articulator-only and relation ablations
16. multi-scale temporal extension
17. scale-up
```

Do not implement multi-scale, adversarial invariance, graph transformers, or sophisticated geometry until the single-scale relation MVP is validated.

---

# 45. Definition of Done for MVP

The MVP is complete only if all conditions hold.

### Engineering

- all unit tests pass;
- DDP/mixed precision train runs without NaNs;
- checkpoint resume works;
- visualization verifies masks/boxes;
- teacher has no gradients;
- EMA is correct;
- no semantic labels enter pretraining;
- leakage audit report is clean.

### Scientific

At minimum compare:

```text
A. generic JEPA same data/backbone/steps
B. A + articulator target
C. A + relation target
D. A + articulator + relation
```

Evaluate with:

- frozen retrieval on at least one held-out sign dataset;
- frozen phonological probe;
- signer leakage probe.

### Go/no-go

Proceed to large-scale training only if relation prediction provides a reproducible benefit on representation-oriented evaluation, not merely lower SSL loss.

---

# 46. Optional Phase-2 Extensions

Only after the main hypothesis is supported.

## 46.1 Learned articulator discovery

Remove external landmark dependency.

Possible directions:

- learned spatial slots;
- attention-guided token grouping;
- motion-conditioned region proposal;
- teacher attention maps.

Goal:

\[
RGB\rightarrow learned\ articulator\ structure
\]

without pretrained pose estimator.

## 46.2 Higher-order relation hypergraph

Beyond pairwise:

\[
R(LH,RH,Face)
\]

for signs whose semantics depend on several simultaneous articulators.

Do not begin here because it weakens interpretability.

## 46.3 Deep relation supervision

Inspired by V-JEPA 2.1 deep self-supervision:

- build targets from several encoder layers;
- test whether earlier layers preserve local hand structure while later layers preserve semantic context.

## 46.4 Cross-lingual SSL

Pretrain on multiple sign languages without labels.

Test:

- unseen-language retrieval;
- language-probe leakage;
- shared vs language-specific relation structure.

---

# 47. Bibliographic / Code Reference Index

## Core sign representation works

### SignRep
- Paper: https://arxiv.org/html/2503.08529v1
- Code: https://github.com/ryanwongsa/SignRep

### SHuBERT
- ACL paper: https://aclanthology.org/2025.acl-long.1397/
- arXiv: https://arxiv.org/abs/2411.16765
- Code: https://github.com/ShesterG/SHuBERT

### SignSeek
- Paper: https://arxiv.org/html/2609.03695v1
- Status at verification: arXiv preprint; agent must re-check official code availability.

### SignDINO
- CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html

### SignMAE
- https://arxiv.org/abs/2605.02094

### Scaling up Multimodal Pre-training for Sign Language Understanding
- https://arxiv.org/abs/2408.08544

## Generic SSL / video representation

### V-JEPA 2 / V-JEPA 2.1
- https://github.com/facebookresearch/vjepa2

### V-JEPA
- https://github.com/facebookresearch/jepa

### VideoMAE
- https://github.com/MCG-NJU/VideoMAE

### VideoMAEv2
- https://github.com/OpenGVLab/VideoMAEv2

## Data / benchmarks

### YouTube-ASL
- https://github.com/google-research/google-research/blob/master/youtube_asl/README.md

### YouTube-SL-25
- https://arxiv.org/abs/2407.11144

### How2Sign
- https://how2sign.github.io/
- https://github.com/how2sign/how2sign-data

### WLASL
- https://github.com/adiacla/WLASL

### ASL Citizen
- https://www.microsoft.com/en-us/research/project/asl-citizen/
- https://github.com/microsoft/ASL-citizen-code

### Sem-Lex
- https://github.com/leekezar/SemLex

### ASL-LEX 2.0
- https://asl-lex.org/about.html
- https://asl-lex.org/download.html

### NMFs-CSL
- https://ustc-slr.github.io/datasets/2020_nmfs_csl/

---

# 48. Final Agent Instruction

Implement this project as a sequence of falsifiable experiments, not as a feature checklist.

The central experiment is:

\[
\boxed{
\text{same data + same backbone + same compute:}
\quad
\text{generic latent}
\;\text{vs}\;
\text{articulator latent}
\;\text{vs}\;
\text{relation latent}.
}
\]

The project succeeds scientifically only if the relation target produces a measurably stronger, more transferable sign representation under frozen evaluation.

Do not hide a negative result by stacking extra objectives. If relation prediction fails, diagnose the target definition and report the failure. If it succeeds, only then add multi-scale temporal structure and scale the model/data.

---

# Appendix A — Research Configuration Template

```yaml
experiment:
  name: signrel_relation_mvp
  seed: 42
  notes: "Single-scale relation target; no gloss/text supervision."

data:
  manifest: manifests/pretrain_clean.parquet
  num_frames: 16
  sampling_stride: 2
  crop_size: 384
  batch_size_per_gpu: 4
  num_workers: 8
  assert_no_semantic_labels: true

articulators:
  names: [left_hand, right_hand, face, body]
  confidence_threshold: 0.5
  left_right_swap_on_flip: true

model:
  backbone:
    provider: vjepa2
    name: vjepa2_1_vit_base_384
    pretrained: true
    freeze: false
  relation_dim: 384
  relation_edges:
    - [left_hand, right_hand]
    - [left_hand, face]
    - [right_hand, face]
    - [left_hand, body]
    - [right_hand, body]
  temporal_scales:
    - name: short
      kernel: 1
      weight: 1.0

masking:
  random_multiblock: 0.30
  articulator: 0.30
  relation_break: 0.30
  motion_salient: 0.10
  min_visible_ratio: 0.25

loss:
  relation:
    weight: 1.0
    type: l1
    normalize: layernorm
  articulator:
    weight: 0.25
  regularization:
    enabled: false

optimizer:
  name: adamw
  lr: 1.0e-4
  weight_decay: 0.04
  betas: [0.9, 0.95]

scheduler:
  type: cosine
  warmup_fraction: 0.05

ema:
  start: 0.996
  end: 1.0
  schedule: cosine

trainer:
  precision: bf16
  gradient_checkpointing: true
  grad_clip_norm: 1.0
  max_steps: 50000
  log_every: 20
  eval_every: 2000
  checkpoint_every: 2000

evaluation:
  frozen_retrieval: true
  linear_probe: true
  phonology_probe: true
  signer_probe: true
```

---

# Appendix B — Upstream Verification Checklist

Before using an upstream checkpoint:

```text
[ ] official source URL verified
[ ] download succeeds
[ ] SHA256 recorded
[ ] license checked
[ ] architecture name recorded
[ ] image resolution recorded
[ ] frame count / tubelet size recorded
[ ] patch size recorded
[ ] normalization recorded
[ ] checkpoint key mapping inspected
[ ] missing/unexpected keys printed and reviewed
[ ] one forward pass shape-tested
[ ] feature extraction numerically finite
```

Before vendoring code:

```text
[ ] upstream commit pinned
[ ] license compatible
[ ] source path documented
[ ] copied ranges/functions documented
[ ] upstream copyright preserved
[ ] local modifications documented
[ ] unit test added
```

---

# Appendix C — Novelty Guardrails

The agent must warn the researcher if implementation drifts into any of these already-covered directions:

```text
"MAE but mask hands more"                     -> too close to sign-aware MAE prior art
"predict one masked articulator"             -> close to SignSeek MAP
"cluster each articulator and predict IDs"   -> close to SHuBERT
"reconstruct pose/keypoint/angle targets"    -> close to SignRep / SignBERT family
"teacher global, student local hand/face"    -> close to SignDINO
"same-gloss positives in pretraining"        -> supervised representation learning, not pure SSL
```

The proposed identity must remain:

```text
label-free video
+ EMA latent teacher
+ structured context corruption
+ explicit pairwise inter-articulator relation targets
+ temporal relation prediction
+ RGB-only downstream encoder
```
