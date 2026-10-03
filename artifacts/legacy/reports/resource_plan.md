# Resource plan — current milestone

GPU 0 is authorized with one GPU worker. Dataset roots remain read-only.
New-download cap: 10 GiB; cache cap: 100 GiB; minimum free disk: 20 GiB;
pilot GPU allocation cap: 8 hours. No adaptation job is authorized by the scientific gates.

SignRep common extraction: 1.305 wall hours for 4,726 clips,
all successful. All eight common probe jobs completed serially.
Registered completed GPU allocation wall-time proxies total 1.445 hours;
failed startup attempts have incomplete resource telemetry and are excluded.
This total is not active GPU compute-hours.

Native SHuBERT train pilot: 8 clips, 772 frames, with
245 frames observing all four streams.
CPU preprocessing: 115.07 seconds.
GPU feature smoke including repeat checks: 3.25 seconds;
peak allocated memory: 626478592 bytes.

Provisional 4,726-clip forecast: 18.88 CPU hours
and 0.53 GPU feature wall hours.
Projected compressed all-layer feature cache: 16.96 GiB,
plus approximately 19 GiB of author preprocessing intermediates if retained.
Observed free disk at the original pilot: 79.75 GiB.
These are eight-train-clip extrapolations, not complete duration/landmark distributions.

The next native cohort job requires a locked native protocol, bounded duration/FPS
sampling, cache/resume integration, and a confirmed schedule that keeps GPU workers
serial. CPU preprocessing can proceed without reserving GPU allocation for its entire
wall duration. Measure preprocessing and GPU feature stages separately.

Completed smoke attempts are terminal. Full native CPU/GPU jobs remain live; do not infer liveness from tool-namespace PID absence. Never terminate unrelated processes.

Full native common CPU producer and soleGPU0consumer are now active; exact handles
are recorded in state/live_handles.json. Three real adapter clips and verified
resume passed;60fixed source headers were checked. Source FPS varies, so native
feature clocks retain decoded-frame PTS. The20GiBfree-space reserve is enforced
before each new artifact; stage wall-time includes decode/I/O and GPU waits and
is not labeled CUDA compute time.

The new fair_common_v1 runner is CPU-only and waits for full native terminal PASS. It does not reserve another GPU or evaluate partial test coverage. Its24planned controls use identical3LR grids and a single fixed temporal width64. Existing native worker locks remain authoritative under the isolated tool PID namespace.

At2026-10-01T00:26UTC, native lexical preparation had871/4726clips and
the soleGPUconsumer had870/4726; both exclusive locks were held.
This is advancing file/lock evidence, despite hidden host PIDs.

The NCSLGR extension runs222whole frontal-body videos on CPU only,
with22154source frames (median95.5,max189frames). Its validated two-clip
preparation pilot took46.183seconds and one-clip native CPU feature pilot
took24.520seconds including encoder loading/repeat. Completed pilot artifacts
are preserved in evidence/native_ncslgr_cpu_pilot_v1. At00:26UTC,27clips
were prepared and25featured. Each CPU stage has one worker and one Torch thread;
no extra GPU worker is allocated. An18head equal-budget CPU function suite waits
for full terminal extraction PASS and class-coverage checks. It includes the
predeclared train-only duration prior and cannot launch/retry extraction.

Latest observed free disk:68.64GiB. Workers enforce the20GiBreserve.
The inherited locked_utc metadata field in the native NCSLGR protocol is explained
in reports/native_ncslgr_cpu_pilot_audit.json; the hashed protocol is preserved.

## Latest reviewed state — 2026-10-01T06:36:13.646835+00:00

Native NCSLGR preprocessing and CPU features completed222/222;18strictfunction
heads and9author-imputed classification heads completed and audited. There is
no native NCSLGR worker still running. Original native lexical CPU/GPU workers
terminated at the disk reserve:1237prepared and1235featured of4726. Old logs,
terminal states and attempts are preserved in evidence/native_common_disk_stop_v1.

The user will free capacity. Current free disk is25.86GiB. One reviewed
capacity waiter, session89084, checks for60GiBfree before launching work. It then
finishes CPU preprocessing before the singleGPU0consumer; this avoids hidden host
PID assumptions in the original producer waiter. Each original worker preserves
its21GiBguard and unchanged cache identities. No per-clip failed audit was found;
1237source/stream audits and1235feature hashes passed recovery inspection.

CPU equal-grid lexical evaluation will use runs/fair_common_v1_recovery1 under
the same locked config. The old runs/fair_common_v1 prefittingFAIL remains intact.
A new recovery failure ends the wrapper; no automatic retry. Inspect
state/native_common_disk_recovery_v1/status.json and state/live_handles.json
before another launch.

Observation 2026-10-01T07:19:18.030604+00:00: reviewed recovery session 89084 verified live; CPU preprocessing 1433/4726; available filesystem space 73.70 GiB. GPU extraction resumes only after full preprocessing with one GPU 0 worker. Grounding and semantic numerical checks used CPU only.

Observation 2026-10-01T07:52:16.837399+00:00: live lexical recovery CPU preprocessing 1585/4726; independent CPU cue-intervention extraction 26/175 under pipeline session 5437. Both specific handles verified live. Free disk 72.61 GiB. No additional GPU worker. Full intervention extraction must precede24CPUheads.
