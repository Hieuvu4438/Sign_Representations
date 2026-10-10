# Implementation notes (proposal11 → code)

Status 2026-10-10: implemented and unit/smoke-tested; **no training run yet**, so nothing here is a result.

## Sources read for algorithmic correctness

| Component | Source | What was checked |
| --- | --- | --- |
| V-JEPA 2.1 encoder, predictor, multi-level targets | official code (facebookresearch/vjepa2 @204698b, vendored unchanged) + checkpoint | RoPE, `n_output_distillation` 1 vs 4 (blocks 3/6/9/12 via the checkpoint's `norms_block.0-3`), predictor `return_all_tokens` context head, per-level LN targets as in `app/vjepa_2_1/train.py::forward_target`; the released predictor projects to ViT-G (1664-d) and is therefore NOT loaded (new 4-block predictor, §7.2) |
| S-JEPA centred CE (B11) | ECCV 2024 paper PDF §3, §4.2 | centring rate β = 0.9, τp = 0.1, τt = 0.06, CE over the embedding dimension, EMA target encoder |
| SignRep baseline | `method/jepa_baseline` audit vs arXiv 2503.08529v1 §3–5/A1–A4 (see `research/protocols/EX16-v1.md`) | reproduction matches; used unchanged as baseline |
| Graph layers | ST-GCN normalised adjacency (A + I, degree-normalised) restricted to visible neighbours | — |

## Data contract (§6)

- Corpus: the SignRep reproduction's 3,000 unlabelled ASL Citizen train videos (`ASLC002/corpus/pretrain.jsonl`); gloss/signer fields never read by trainers.
- Landmarks: local **COCO-WholeBody** 2-D keypoints (not MediaPipe). J = 67: body 7 (COCO 0,5–10), LH/RH 21 (91–132), face 18 (iBUG brows 17–26, mouth 48/54/62/66, eye corners 36/39/42/45). Normalised xy → source pixels before any geometry (§6.4). Validity = part-mean and per-joint score thresholds; quality = uncalibrated heat-map score (not a probability). Chirality check: body wrist and hand root agree (< 40 px) in 90.8% of observed frames.
- Timing: effective fps = min(source, 25), frames chosen by timestamp, no upsampling/interpolation (§6.3). RGB T = 16 (0.64 s), B T = 64 (2.56 s).
- RGB geometry: full-frame letterbox (aspect kept) to S×S; padding tokens excluded from every loss and from the context; landmarks mapped with the same transform; region ownership hands > face > body > other is train-only and never given to the encoder.

## Proposal A (§7)

- Recipes A-R (final layer) and A-D (4 levels + context loss λC = 0.05), new predictor 4 × 384, 6 heads.
- Masks (hidden budget exactly 80% of valid tokens, C/M/G disjoint, verified per sample): `generic`, `generic_cov` (generic + the same ≥ 25% hand-support protection — coverage-matched control of §13.2), `partial` (§7.4 event policy, guards), `whole` (A14), `jitter` (A15: masks from translated regions, loss groups from true regions).
- Loss L_M: per-level target-LN L1, equal layer average; `balanced` = mean per present ownership group (pooled over the batch) then equal average; `uniform` = plain mean.
- Optimiser/EMA as §11.1–11.2; EMA updated once per successful optimizer step after micro-batch accumulation.
- Measured at 224 px: partial events give ≈ 19 structured target tokens and 13% fallback (hands occupy few patches); at 384 px ≈ 69 and 3%. 224 is the default for the SignRep-matched comparison (SignRep uses 224); 384 is the A5 factor.

## Proposal B (§8)

- Dependency-safe features (19 slots/frame/node; anchors c, s from VISIBLE shoulders, shared by the teacher), local articulation branch (same-bin palm/eye scale), masks expanded to 2-frame bins before the 50% budget, tied body-wrist/hand-root, events 50/30/20 + random fill.
- Encoder: tokenizer 38→256; local branch 2 graph blocks (hands share weights + side embedding, face separate); 6 main blocks (visible-only spatial graph → per-joint temporal attention → cross-part attention over 4 visible-pooled summaries → MLP); hidden/missing cells = learned missing token, excluded as keys/neighbours/pooling.
- Objectives: B10 raw (body-relative + root-relative xy of hidden cells), B11 centred-CE, B12 latent L1, B13 = B12 + 0.1 × observable anchors (shape/orientation from the local branch, location/motion/relation/face from part summaries; each family normalised, families averaged equally).
- Deviation: shape/face anchors check support per element (joint + root + ≥ 1 palm bone; bend triple; eye anchors + element) — requiring the whole hand visible made the family almost always empty under 50% masking.
- EMA schedule 0.996→0.9999 for all B variants (S-JEPA's own 0.9999→1.0 schedule is for 1,200 epochs; kept matched across variants instead).

## Proposal C (§9)

- C-observed only (λG,obs = 0.1, λG,mask = 0). Targets = frozen-B **local pre-fusion** articulation features of the same physical window (bin-local, so window length does not matter), pooled over observed part nodes, LN. Student = ROI mean of VISIBLE context tokens of the part in the bin (train-only ownership; privileged alignment, as §9.3 states).
- C-reserved masks: up to 2 complete visible bins per reliable hand are protected before masking, budget restored to 80%; loss only where the part is fully observed and every token of its region is context. All C controls must use `--c-reserve` too.

## Verification (tests/)

| Test | Checks |
| --- | --- |
| `test_rgb_masks` | all 5 RGB policies × 224/384: disjoint C/M/G, exact 80% budget, no padding token, ≥ 25% hand support where required |
| `test_rgb_model` | AR/AD shapes; **hidden-value perturbation**: overwriting hidden tubelets leaves student context features bit-identical; teacher gets no gradient |
| `test_pose_dependency` | **hidden-value perturbation** for B: student node/local features, visibility and all anchor targets invariant to hidden coordinates (3 policies × 12 windows); teacher side changes |
| `test_misc` | finite zero losses with no valid targets; letterbox round trip; EMA init = student; graph encoder visible states independent of hidden/missing cells |

Smoke runs: A (AR partial, AD generic_cov, AR jitter), C (AR partial + B teacher), B (all 4 objectives); extraction for rgb/graph/hiera/vjepa_fc; evaluation end-to-end. Throughput on the shared GPU (while another job ran): A AD 224, batch 64 ≈ 1.5 s/step; B batch 64 ≈ 2 s/step (CPU-bound feature construction).

## Not implemented yet (declared)

Metadata-only and context-swap controls (§6.7 items 1–2), predictability audit (§9.4), C-masked (§9.3 extension), lexical-disjoint phonology probes with ASL-LEX (§12.4), few-shot (§12.6), direct-descriptor baseline B00 as a feature extractor, distance-weighted context loss, 384-px profiling.
