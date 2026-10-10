# ASLC002 — fresh ASL-Citizen SignRep baseline

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: UNVERIFIED
- Execution Status: completed; one seed; no independent full training rerun
- Version Label: ASLC002_v1

## Protocol

300 fixed real classes; 3000 official train/gallery videos; 1500 signer-disjoint official validation queries. No test evaluation. Gloss metadata does not enter the self-supervised training loss.
Fresh official Kinetics MAE backbone plus random SignRep heads. No public SignRep or previous ASL pilot training checkpoint is loaded.
SignRep objectives train for 5000 updates, 4 views/2 distinct videos per batch,80% masking, 500 warmup updates,LR 0.0001. BF16 compute is explicit. Corpus bone medians use only selected training videos.
Same gallery/query membership at step0/final. Original full-frame inference preprocessing,16-frame windows,stride 2,inference batch 4,BF16 compute/float32 feature caches. Video mean or raw max(LH,RH) activity weighting,then L2/cosine. Distinct classes rank by best individual gallery video.

## Retrieval results

| Pooling | Step | Top1 | Top5 | MRR | DCG |
|---|---:|---:|---:|---:|---:|
| unweighted | 0 | 24/1500 (1.6000%) | 90/1500 (6.0000%) | 0.048479 | 0.191028 |
| unweighted | 5000 | 25/1500 (1.6667%) | 99/1500 (6.6000%) | 0.055724 | 0.200903 |
| activity_weighted | 0 | 22/1500 (1.4667%) | 86/1500 (5.7333%) | 0.047280 | 0.189825 |
| activity_weighted | 5000 | 27/1500 (1.8000%) | 92/1500 (6.1333%) | 0.055187 | 0.200367 |

Instance and macro Top1/Top5 are mathematically equal because each class has five queries. This is a reduced-data/compute single-seed baseline,not a500k-step YT-SL paper-score reproduction. ASLC001 scores use a different64-class dictionary and cannot be compared directly. Step0 activity head is random; supervised activity coverage below must qualify weighted pooling.

## Observed supervision coverage

Fraction of valid target elements across all actually sampled batches; repeated views count each time. Missing targets remain masked.

| Prior | Valid fraction |
|---|---:|
| body_pose | 47.3791% |
| right_pose | 44.9959% |
| left_pose | 35.3287% |
| full_body_angles | 100.0000% |
| full_right_angles | 44.9959% |
| full_left_angles | 35.3287% |
| right_dist | 44.9959% |
| left_dist | 35.3287% |
| body_dist | 27.6196% |
| activity | 0.1900% |

## Artifacts

`run_config.json`, `inventory.json`, official manifests, `corpus/pretrain.jsonl.audit.json`, corpus bone table, `train/metrics.jsonl`, complete `train/last.pt`, lightweight `final.model.pt`, per-phase feature manifests/metrics/predictions. Reviewable final summaries are archived under experiments/<experiment>.
