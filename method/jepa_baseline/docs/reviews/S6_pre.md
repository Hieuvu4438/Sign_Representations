# S6 pre-implementation review

Complete orchestration must expose extraction, corpus standardization, manifest
preparation, pretraining/resume, feature export, retrieval evaluation, class-distribution
generation, recognition/resume and translation adapters. Default pretrain config keeps
paper steps/batch/optimizer; runnable technical stops are explicitly marked.

Production pretraining cannot silently accept smoke assets: require WLASL exclusion
audit and a bone-statistics source-manifest pin consistent across all caches. Freeze
data/source/environment/config hashes in checkpoints. Scheduled retrieval requires
training gallery versus WLASL validation, and selection must occur only after a complete
step. CLI config relative paths resolve against config location; CLI argument paths
resolve against working directory. Errors remain explicit; never silently skip videos.

Translation launch must use actual pinned author trainers and retain recipes. Adapter
source may register a 768-dimensional input model rather than change immutable vendor
files. Test real original components in isolated compatible environments where possible;
record exact dependency constraints and remaining runtime limitations. Provide runnable
commands and separate module reconstruction from benchmark-score reproduction.

Final audit checks every contract row, pre/post reviews, source immutability, complete
tests and end-to-end CLI smoke. No achieved status until required code and verification
are complete; source-underdetermined historical equivalence remains explicitly unverified.
