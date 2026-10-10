"""Paired fixed-protocol summaries; incomplete suites cannot give a verdict."""
import numpy as np


METRICS = ("Recall@1","Recall@5","mAP_instance","phonology_macro_f1","signer_accuracy")


def metrics_from_report(report):
    properties = report["phonology_signer_independent"]
    evaluated = [p for p in sorted(properties) if properties[p]["status"] == "evaluated"]
    if not evaluated:
        raise ValueError("No evaluated phonological properties")
    return {"Recall@1":report["retrieval"]["Recall@1"],"Recall@5":report["retrieval"]["Recall@5"],
            "mAP_instance":report["retrieval"]["mAP_instance"],
            "phonology_macro_f1":float(np.mean([properties[p]["macro_f1"] for p in evaluated])),
            "signer_accuracy":report["signer_probe"].get("top1")}


def coverage_contract(report):
    r = report["retrieval"]
    keys = ("queries_total","queries_evaluated","queries_without_positive","skipped_sample_ids", "gallery_size","gallery_classes","ranking","same_signer_excluded")
    out = {"retrieval":{k:r[k] for k in keys},"phonology_metadata_sha256":report.get("phonology_metadata_sha256"), "config":report["config"]}
    for family in ("phonology_signer_independent","phonology_gloss_disjoint"):
        out[family] = {p:{k:v.get(k) for k in ("status","train_samples","test_samples","train_classes","known_class_coverage","signer_independent","gloss_disjoint")}
                       for p,v in report[family].items()}
    s = report["signer_probe"]
    out["signer_probe"] = {k:s.get(k) for k in ("status","train_samples","test_samples","train_classes","retained_signers","gloss_disjoint")}
    return out


def summarize(grouped, expected_seeds, expected_variants, split="val", audit_verified=False):
    expected_seeds = sorted(expected_seeds)
    expected_variants = list(expected_variants)
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Invalid planned seeds")
    for name, seeds in grouped.items():
        if name not in expected_variants or set(seeds)-set(expected_seeds):
            raise ValueError("Unplanned variant or seed")
        for value in seeds.values():
            if any(value[k] is not None and not np.isfinite(value[k]) for k in METRICS):
                raise ValueError("Non-finite evaluation metric")
    stats = {}
    for name, seeds in grouped.items():
        stats[name] = {}
        for metric in METRICS:
            data = [seeds[s][metric] for s in sorted(seeds) if seeds[s][metric] is not None]
            stats[name][metric] = {"mean":float(np.mean(data)) if data else None,
                                  "std":float(np.std(data,ddof=1)) if len(data) > 1 else None,
                                  "seeds":len(data)}
    missing = [{"name":name,"seed":seed} for name in expected_variants for seed in expected_seeds if seed not in grouped.get(name,{})]
    decision = {"status":"insufficient_evidence", "phase2_allowed":False, "missing":missing,
                "audit_verified":audit_verified, "comparisons":{}}
    # D-B isolates adding relation above the articulator objective; C-A isolates
    # adding relation above generic JEPA. C-B guards superiority over node control.
    for candidate, control in (("C_relation","A_generic"),("D_combined","A_generic"),
                               ("C_relation","B_articulator"),("D_combined","B_articulator")):
        common = sorted(set(grouped.get(candidate,{})) & set(grouped.get(control,{})))
        if not common:
            continue
        deltas = {k:[grouped[candidate][s][k]-grouped[control][s][k] for s in common]
                  for k in METRICS if all(grouped[candidate][s][k] is not None and grouped[control][s][k] is not None for s in common)}
        positive = len(common) >= 3 and all(np.mean(deltas[k]) > 0 and sum(v > 0 for v in deltas[k]) >= len(common)-1
                                           for k in ("Recall@1","phonology_macro_f1"))
        decision["comparisons"][f"{candidate}_vs_{control}"] = {
            "seeds":common,"paired_seed_deltas":deltas,"mean_delta":{k:float(np.mean(v)) for k,v in deltas.items()},
            "std_delta":{k:float(np.std(v,ddof=1)) if len(v) > 1 else None for k,v in deltas.items()},
            "consistent_two_metric_pilot_benefit":bool(positive)}
    complete = not missing and all(len(grouped.get(name,{})) >= 3 for name in expected_variants)
    if complete and audit_verified:
        comp = decision["comparisons"]
        promising = any(all(comp.get(f"{name}_vs_{control}",{}).get("consistent_two_metric_pilot_benefit",False)
                             for control in ("A_generic","B_articulator")) for name in ("C_relation","D_combined"))
        decision["status"] = "promising_pilot_requires_broader_confirmation" if promising else "relation_benefit_not_supported_by_this_pilot"
        if split == "test":
            decision["use"] = "descriptive held-out check; test results cannot select recipes or authorize phase 2"
    return {"split":split,"summary":stats,"per_seed":grouped,"decision":decision,
            "scope":"40-gloss local pilot; descriptive training-seed variation; no significance, full benchmark or SOTA claim",
            "phase2_rule":"Both generic and articulator controls, complete paired seeds and matching coverage must be checked. Broader validation is required before multiscale/scale-up."}
