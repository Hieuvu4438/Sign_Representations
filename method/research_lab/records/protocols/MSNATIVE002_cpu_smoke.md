# MSNATIVE002 — Variable PTS artifact repair, same native smoke

Locked before new tests/run; D045. MSNATIVE001_cpu_smoke input/crop/PTS/reference/
pixel/native16/stride2/sample/CPU/GPU/memory/disk bounds unchanged. All four
samples copied exactly from original bound config; no reselection or quality.
Original failed run retained. No known parity test is relaxed or rescued.

Only serializer repair: fixed first16RGB frame arrays retain exact write_array;
PTS uses finite strictly increasing16..240element little-endianfloat64 NPYv2
payload written through existing bytebounded store. Actualheader+payload size
checked against exactshape spec and preregistered240element maximum. No array
padding or invented timestamps. Raw timestamps/inputs unchanged. New fixture
tests lengths16/84/240 readback exact and all rejection cases before smoke launch.

MSNATIVETEST002 runs original three nativecrop/time/losslessseek groups plus one
PTSserialization group, CPU120s/GPU0/durable2MiB; no network/taskgain. MSNATIVE002
same CPU120s/GPU0/durable20MiB, four complete source identities and input/control
hashes preserved. Registered plan includes all fresh dependencies/log/meta/
arrays/PNGs/status/manifest/replacementreserve. First technical retry only;
subsequent failed guard stops for diagnosis, no automatic reselection.
