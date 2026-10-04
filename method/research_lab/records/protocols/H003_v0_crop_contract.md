# H003 v0 — CPU crop/body/color contract audit

Registered before CROP001, 2026-10-04. Diagnostic/software prerequisites only;
H003 scientific hypothesis remains unregistered/hold. Same four fixed train/dev
LANDMARK001 sources, no label fitting/test access or HPO. No new detector run.

Question: do author helper/crop/fill functions preserve a frame-aligned, finite
four-stream input and distinguish raw observation from reused pixels?
Reference: readonly SHuBERT crop_face.py, crop_hands.py, body_features.py, code
SHA bound in CROP001 config; main/source environment previously verified D016.

Predeclare CPU fixtures: analytic normalized body; missing body after observed;
initial whole-rowNone versus pose-fieldNone; collapsed shoulders; two separated
hands; one shared detection near both wrists; empty hand list; exact0.1 wrist
threshold; oversized box; valid/edge face cues; two-frame initialNone crop writer.
Dummy readers/writers are injected into isolated author modules for the last
fixture; native functions execute with own fixture paths, no upstream mains or
real default outputs. Expected exceptions/contract violations are observations,
not a failed model or improved method. Preserve every result/error explicitly.

Independent RGB oracle: generated colored frames encoded losslessly with CPU
FFmpeg, compare decord RGB to known bytes and OpenCV BGR reversed. Also register
author mp4v writer roundtrip as lossy diagnostic with no exact color/pixel-parity
claim. Helper crop uses channel permutation invariance; do not call the author's
RGB→BGR before OpenCV writing a color bug without checking the complete path.

On all396 actual train/dev frames: source/raw SHA, native PTS/count, selected
face/left/right detector indices, same-detection double assignment, distance,
crop validity/shape, finite normalized body. No corrected assignment or filling
policy is selected by this audit. Raw hand categories are model outputs, not
verified anatomical ground truth; pose-index names are API ontology only.

Success for audit completion: all fixtures and fixed raw frames have explicit
results, independent valid-body/color oracles pass, source/time identity fixed.
Gate for real feature extraction remains held if unsafe semantics/branches occur
until a versioned adapter/missingness policy and parity limitations are registered.
No exclusion/truncation/time-resampling/exception-to-zero substitution is allowed.

CPU deadline180s, no GPU inference/detector, no download/install. Fixture media and
small inputs ≤32MiB bounded /dev/shm; root lightweight metadata only. Raw crop
pixels are not bulk persisted in this audit. Recheck RAM/source assets first.
Metrics are counts/shapes/errors/color-byte differences; not task performance or
representative detector-failure rates. No frame-IID CI or linguistic interpretation.

Next decision from outcomes: define explicit one-to-one/ambiguous assignment,
body initialization, raw-observed versus filled masks and RGB/codec contract before
registering any adapted feature smoke. Keep native-reference deviations visible.
