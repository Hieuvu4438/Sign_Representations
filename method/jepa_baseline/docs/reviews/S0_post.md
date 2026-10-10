# S0 post-implementation review

Verified: 33 original source files match recorded SHA256; all nine adapted SignRep
Python modules differ only by import namespaces. All original modules and licenses
are retained. Public SignRep main has one branch; no hidden upstream training
entrypoint was found in the public snapshot. Teacher source is separately pinned.

Runtime audit: Python 3.13.5, torch 2.11.0+cu128; pytest/timm/OpenCV present. The pose
README asks for Python 3.10, torch 2.0 and MediaPipe 0.10.14; use an isolated pose
environment instead of changing the shared SignRep environment.

Evidence: `evidence/source_verification.json`, `evidence/sources.json`.
Status: source audit PASS; original environment equivalence UNVERIFIED.
