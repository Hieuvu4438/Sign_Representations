# ASLC002 pre-execution review

Authorization: user requested actual baseline training and retrieval statistics,using the proposed300-class scope and fresh initialization. The5000-step screening budget and Kinetics MAE interpretation were stated before execution. No new permission is needed for routine preparation,training or validation.

Isolation: new runtime/ASLC002 and experiments/ASLC002;ASLC001 artifacts,unrelated work and external datasets remain unchanged. Existing source modules preserve objectives/geometry/masks. New experiment runner must derive counts,step labels and reports from frozen configuration instead of ASLC001 hardcoded64/1000 fields. Independent verifier must check1500 ranks,5000 updates and model/data provenance.

Resources: RTX5880 Ada49GB shared;allocator capped.042;four pose workers. Free disk observed42GB;large decoded RGB caches are avoided. Inference batching is raised to4 with a real-video memory/finite-output calibration before launch. Precision,batch size and incomplete historical paper choices remain explicit.

Guardrails: audit all pretraining against complete official train and held-out manifests;fresh initialization only from Kinetics;no metric-driven sampling,budget expansion or test-based selection. Pose supervision coverage is an observed limitation,not silently imputed. Final numeric verification does not imply an independent full training rerun.
