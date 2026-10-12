# Decision memo — PILOT8

- Question: degree2 signature+anchors/timing có hơn ordinary motion+anchors/timing khi
  cùng inputs, projected32D targets, windows, raw masks, successful steps và probe budget?
- Competing explanations: temporal training/data scale, generic image features, detector
  quality, compression/native rank hoặc head calibration có thể quyết định kết quả.
- Protocol/data/availability: locked before model scores in `phase3_pilot_plan.json`;
  32 train/16 validation, 8 label-curated classes, official signer-disjoint metadata and
  actual source hashes. No test access. Sources and fingerprints verified before training.
- Correctness gates: source/cache hashes, train-only fitting/all phases, closure/raw
  counterfactual, padding, gradients/teacher, AMP/DDP/resume, fixed head trials and matched
  E03/E04 mask digests. Codebooks are proposed hidden units, not glosses or SHuBERT reproduction.
- Runs: 3seeds ×100 successful updates/SSL method. One earlier pre-LRU diagnostic was
  stopped after saving its checkpoint and excluded; no failed trial was converted to a score.
- Validation: E03 and E04 each27.08% mean PC Top1,3.61pp seedSD; differences0pp for all
  seeds. Frozen DINO retrieval reference is stronger on R@1 than these short temporal runs.
- Static phonology: raw static-token retention tests pass; no annotated phonology experiment
  has run. Do not infer semantic/static retention from those deterministic tests alone.
- Cost: cached update/load profile recorded; full preprocessing/GPU-hours are missing,
  not reported as zero. Same successful updates/frame masks do not mean identical wall time.
- Supported claim: the controlled code paths function, and this small pilot found no
  PC Top1/retrieval advantage for signature over ordinary motion.
- Unsupported claims: novelty, external reproduction, full ASL Citizen score, meaningful
  generalization improvement, collapse absence, or causal utility of level2.
- Decision: **no claim yet / revise experiment scale**. Keep E00 and E02 reference paths;
  keep geometry/control implementations for attribution, without selecting geometry by narrative.
- Next: complete deployment/protocol handoff, then a larger preregistered matched budget
  with source quality buckets, static/movement probes and uncertainty before investing in
  degree3, learned geometry teacher, geometry input or best-system expansion.
