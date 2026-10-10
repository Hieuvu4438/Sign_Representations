# signrep_spectral — namespace instructions

- Implementation spec: `docs/proposal3/signrep_spectral_implementation_spec.md`.
- Lab rules (when this direction is run inside the research program):
  `docs/proposal2/AGENTS.md` and `docs/proposal2/research_lab/`.
- All code, configs, tests, records and runtime outputs of this direction live in
  this folder. Do not modify `third_party/`, other `method/*` folders, datasets or
  checkpoints. Do not import code from other methods.
- Runtime outputs go to `runtime/<new_run_id>/` (exclusive create, never overwrite).
- Results are "SignRep-initialized canonical continuation + spectral auxiliary";
  never call them a SignRep reproduction or an official benchmark.
