# MSCLKTEST001 — manual temporal and framehash fixtures

Diagnostic/software only, H006 input validation. Four groups: rational half-open
boundaries with nonzero origin/backwards ticks and explicit ordinal membership;
clock changes adding/removing frames and native16 stride2 count without padding;
equal raw ticks preserved and invalid arrays/types rejected; manual SHA256
framehash parser fields/timebase and malformed/extra-stream rejection.
All expected arrays/counts are written independently by hand. No actual MS-ASL
samples/model calls, no GPU or quality. CPU60s,2MiB complete-artifact limit.
No HPO/reselection or tolerance changes. Entry/config/source/command bound before
execution; failures retained under a fresh version if corrected.
