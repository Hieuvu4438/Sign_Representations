# Sign_Representations

Pinned SignRep and SHuBERT feature adapters, frozen representation diagnostics,
and experiment provenance. This project does not yet reproduce all training and
downstream benchmarks from either paper.

- [Execution and environment instructions](REPRODUCE.md)
- [Reproduction/checkpoint audit (2026-10-02, Vietnamese)](reports/reproduction_audit_20261002_VI.md)
- [Registered sources](provenance/source_commits.json) and [assets](provenance/assets.json)

Weights, third-party checkouts, videos, feature caches and logs are local assets
excluded by `.gitignore`. A clone requires the pinned sources and registered
asset downloads before inference can run.

Audit available source revisions and registered weight hashes without loading
models or downloading assets:

```bash
python3 scripts/audit_reproduction.py --check-upstream --output reports/reproduction_inventory_fresh.json
```

A passing inventory check verifies available bytes and source revisions. Full
paper reproduction also requires the original data, task code and protocols.
