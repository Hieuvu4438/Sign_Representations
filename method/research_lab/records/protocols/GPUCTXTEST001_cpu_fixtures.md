# GPUCTXTEST001 — Software allocation/accounting context prerequisite

Locked before test,D048. Fresh bounded_gpu_context copies fixed CPUcontext
semantics; original remains immutable/CPU-only. Positive integerGPUallowance
within jobtimeout, boundH_id; start_gpu validates matchingbudgetH/ceilings and
min(remaining,allocation,jobremaining)>1, only once in enteredcontext. Start
before configureCUDA/modelmove; runningstatus persisted GPU_allocated and budget.
Terminal completion/failure/interruption charge monotonic elapsedfromstart; even
artifact finalization failure cannot reset charge. Wholeallocationtime counted,
not kernel-only; exactCPUsetup beforestart excluded. Missing/oversize output fails
with retained partialstatus/error. Existing strict fileplan/log/snapshot/inputSHA
deadline and signalrestore preserved. Wholeprocess hardSIGKILL/IO failure can
still preventterminal persistence; unresolved runningGPU ledger stops newjob.

CPUsoftware3groups: pureallowance/rejection; simulatedcompleted unique-start/
signalrestore/fullstore; simulatedfailed byteoverrun retainscharge/missinglist/
partialcaps. No CUDAimport, no physicalGPU. Simulatedchildren under exclusive
/dev/shm fixture root retained<=1MiB; NEVER include artificialGPU_hours in hardware
ledger. Parent CPUcontext GPU0,120s,2MiB durable and explicitfullfileplan.
Sameagentsoftware, not realCUDAcoverage/performance. ActualGPUrequires fresh
effectivebudget allthree runtime roots+bounderratum and actualdevice/freeVRAM/
RAM/storage, oneownedjob/8GiBcap/visibilitypreservation. No policy relaxation.
