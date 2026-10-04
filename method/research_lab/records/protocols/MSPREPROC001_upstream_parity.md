# MSPREPROC001 — Native crop normalization CPU parity

Locked before actual run/D046. Input four exact stored16×224×224×3uint8 windows
from MSNATIVE002, all identities/config/status/manifest/source hashes bound.
Upstream third_party/SignRep/augmentation/video/base_video_aug.py read-only,
gitcommit/hash recorded. Use existing .venv-signrep, no install or sharedenv edit.
No decoder, checkpoint/model loading or GPU; bodycrop remains adapted MS-ASL.

Three constants RGB[0,0,0]/[255,255,255]/[255,0,127] independent manualFP64 oracle
and all64nativeframe values vs exact upstream Transformation.aug_video, resize
224×224/Normalize(mean.485,.456,.406;std.229,.224,.225;scale255)/TCHW.
Manual oracle(x/255−mean)/std, fixed absolute1e-6 every scalar, finiteFP32 shape
(16,3,224,224), repeat upstreambitwise and native resized identity exact. No
change after outcomes. Full9,633,792scalars across four windows, all reported;
first normalized frame per sample saved float32NPY. No inferred encoder parity.

CPU120s/GPU0/HPO0, scratch planning256MiB+existingreserve,8MiB durable cap;
four(3,224,224)FP32 NPY exact602240B each plus all own dependencies/upstream
snapshot/config/env/log/status/error/report/manifest/replacementreserve bound.
No quality/testcohort/selection/benchmark scoring. Technicalfailure any guard,
no retries without diagnosed freshversion. Same-agent engineering, not scientific
review. Native failure/bounded caps/prior locked protocols remain immutable.
