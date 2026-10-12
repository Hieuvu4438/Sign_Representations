# Known limitations — final implementation handoff

- The implemented default is the proposed target-only, offline multimodal pipeline.
  It is an independent Path-inspired implementation, not an exact reproduction of
  Path-JEPA, SHuBERT or SignRep. No benchmark, SOTA or signature-superiority claim.
- The measured ASL Citizen development pool is label-curated: 8 classes, 32 train,
  16 validation clips, official signer-disjoint membership, seeds 17/29/43 and 100
  successful SSL updates per trial. No final test or full-corpus training/audit has run.
  The historical pilot and final locked-environment confirmation have separate roots
  and source snapshots; only matched trials within a root are compared.
- E03/E04 attribution keeps data, streams, masks, exposure, model/head budgets and
  32D outputs matched. Native information/rank differs. A small validation tie cannot
  prove either usefulness or uselessness at research scale. Seed SD and conditional
  grouped bootstrap intervals answer different uncertainty questions.
- Raw-video inference still needs actual IMAGE-mode pose/hand/face detectors and
  frozen DINOv2. Geometry target-only removes geometry inference, not static pose.
  Export excludes detector/DINO; its metadata binds their preprocessing and weights.
  Dynamic graph limits are B=1..16, T=1..512; full-clip inference uses registered tiling.
- Per-frame MediaPipe wrists may be ambiguous, especially crossing/occlusion. Ambiguous
  anatomical assignments are invalid. Category handedness scores are retained but are
  not joint localization confidence. Source mirror metadata is unknown for the local
  videos; diagnostic overlay review is limited and does not establish corpus quality.
- Missing hands can make geometry targets sparse. No missing path is manufactured
  as stationary zeros. Conservative grouped pose supports may exclude extra context.
- Anatomical masking hides pose measurements while holding RGB and ROI priors fixed.
  It is cross-modal prediction. It does not establish pixel/spatial information hiding.
  Strict synchronized temporal closure has separate RGB/detector counterfactual evidence.
- Resampled aliases are deduplicated in encoder context, semantic/cluster losses,
  target fitting, geometry paths and clip pooling. Dense outputs preserve selected slots
  and original PTS. Tiled context boundaries still depend on the registered window schedule.
- The decoder holds whole clips in RAM. Very long continuous videos need chunked
  decode/extraction. The current offline code is bidirectional, not causal streaming.
- CTC and fixed Transformer seq2seq heads have actual gradients, losses, decoding and
  metrics, including timestamp compaction for missing CTC inputs. Only toy continuous
  integration ran: no configured How2Sign/OpenASL/continuous gloss manifest. Beam/LM
  decoding and pretrained text decoders are optional unavailable backends.
- The verified-annotation phonology probe API exists, with unknown labels masked.
  No trusted phonological annotation manifest is configured, so no real phonology or
  minimal-pair score is reported. Static input-retention tests are correctness evidence.
- Quality buckets use fixed observed/coordinate-valid availability thresholds; they are
  uncalibrated proxies, not detector ground-truth accuracy or linguistic confidence.
- Two Gloo CPU ranks verify global loss/gradient/update equivalence including an empty
  local rank. A real cached two-process run also completed. Exact single-process 5→10
  resume and actual CUDA fp16 overflow/skip are tested. Multi-GPU NCCL, distributed AMP
  overflow consensus on multiple physical GPUs and interrupted DDP replay remain unmeasured.
- Cache workers are threads with main-owned sampler/mask RNG and stat-guarded bounded
  LRU. This preserves replay order. Multiprocess DataLoader scalability remains unprofiled.
- Stage ledgers retain immutable measured records for new preprocessing/cache stages.
  Complete original extraction GPU-hour accounting is unavailable and stays null with
  a reason. Cached timings include teacher/optimizer but exclude detector/DINO. Raw embed
  reports total decode-to-head latency. Concurrent pilot timings are not throughput claims.
- The clean main/extraction/continuous lock and editable installation were verified.
  The historical system-site-packages environment had unrelated dependency conflicts;
  it is preserved as historical evidence, not the recommended setup. Legacy Fairseq
  isolation remains conditional on implementing the original SHuBERT adapter.
- Optional M10 backbones/adaptation, E05 geometry inputs, E06 dense/deep, E07 paired/
  fusion, VC/SignCL/pose priors, lifts and higher degrees are explicitly unavailable.
  E04a/b descriptor ablations are not implemented. The pilot does not justify scaling
  signature complexity. No score or successful experiment is assigned to these branches.
- Local run pointers and asset registry paths are machine-specific. Move assets and
  update explicit paths when moving machines. Checkpoints are trusted local PyTorch
  artifacts; simple feature caches and embedding arrays use typed NumPy formats.
