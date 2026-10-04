# MSNATIVE001 — Native PTS/body-crop CPU feasibility

Khóa trước run, D044/H006 transfer feasibility; diagnostic, không model quality.
Nguồn official archived README trong MSMETA001: box[y0,x0,y1,x1], normalized
theo decoded width/height; annotation thời gian giây trên video gốc. README có
typo end_time/width, không coi nó là oracle pixel hoặc exact cutting code.

Chọn đúng bốn nguồn train/val: trong mỗi split, dòng đầu theo index có raw
video/<acceptedURLsourceID>.mp4 hiện hữu, không thuộc bound39197-ID publicpool,
duration(0,8]s; một start_time0 và một start_time>0. Hash source/annotation và IDs
trước launch. Đây là technical smoke có selection cho offset/cost, không cohort
khoa học/benchmark hoặc pretraining-disjoint claim. Không test hay reselection.
Các IDs thực phải được ghi trong config; thiếu/fail giữ nguyên, không thay mẫu.

Quy ước adapted: interval half-open[start_time,end_time), thời gian PTS rational
trừ stream.start_time*time_base; reject missing PTS/origin, corrupt/nonmonotonic,
multiplevideo hoặc dynamic dimensions. Seek tới keyframe trước start bằng stream
time_base rồi decode/lọc PTS. Không resample/FPS interpolation/pad. Metadata
start/end integer/FPS không quyết định frame range; report sai khác descriptive.
Box normalized yxyx, floor lower/ceil upper trên decoded width/height, cropbody
và OpenCV INTER_LINEAR resize224x224 RGB uint8. Không normalization/encoder.

Đối chiếu độc lập toàn PTS nguồn bằng ffprobe, framecount==header nb_frames;
half-open selection giống specification, PTS tolerance1e-6s (ffprobe decimal
rounding), không tự thay biên/epsilon. Independent OpenCV sourceframe decode
đúng reference frameindices, direct yxyx arithmetic; mọi resized RGB exact.
Strict native16/stride2 window plan; reject<16, không claim officialcut parity.
Hai previewpanel raw/crop chỉ kiểm engineering, không nhãn đúng/sai ngôn ngữ.

Limits CPU120s/GPU0/HPO0, per subprocess15/45s,4parents, sourcecompressed<=32MiB,
headerreference<=20000frames, stdoutPTS<=2MiB, selected<=240frames/clip,
seekdecode<=2048frames, resolution<=3840x2160, RAM plan256MiB+existingreserve.
Durable20MiB: four exact(16,224,224,3)uint8 firstwindow NPY headers+payload,
four<=240float64PTS NPY, four PNG<=256KiB, audit<=512KiB, all source/config/env/
status/error/log/manifest plus rewrite reserve preregistered. No fullframe/video
copy. Small bounded record exception to failed bulkroot20%reserve.

MSNATIVETEST001 prerequisite: three fixed oracle groups: nonsymmetric yxyx/
floor-ceil, rational half-open boundaries, lossless96frame in-memory FFV1 source
seek versus full-reference and short-native failure. Generated binary<=128KiB,
no external media/GPU/taskgain; CPU120s/durable2MiB. Test pass before smokeconfig
registration. Network0; archive/source/hash read-only. No installs/globalenv edits.

Success only engineering parity/all four nativecoverage; technicalfailure if
any registered guard fails, keep failedrun with missing artifacts and no silent
drop/retry/rescue. Unknown raw YouTube authenticity, full cohort coverage,
upstream normalization/feature parity, scientific evaluator/cost/contribution
remain open. Same-agent sequential roles, independent arithmetic not external
scientific review; originalP004/all prior artifacts unchanged.
