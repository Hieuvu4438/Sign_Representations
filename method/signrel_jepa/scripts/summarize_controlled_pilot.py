"""Audit artifacts and summarize fixed A/B/C/D runs with paired controls."""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p/"signrel/config.py").exists())
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from signrel.data.manifest import sha256
from signrel.engine.checkpoint import contract_hash
try:
    from signrel.eval.pilot_summary import summarize, metrics_from_report, coverage_contract, METRICS
except ImportError:
    spec = importlib.util.spec_from_file_location("pending_summary",ROOT/"pending_source/signrel/eval/pilot_summary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summarize, metrics_from_report, coverage_contract, METRICS = module.summarize,module.metrics_from_report,module.coverage_contract,module.METRICS

OBJECTIVES = {"A_generic":(1.,0.,0.), "B_articulator":(1.,.25,0.),
              "C_relation":(1.,0.,1.), "D_combined":(1.,.25,1.)}


def control_config(config):
    c = copy.deepcopy(config)
    c.pop("experiment",None)
    c.pop("seed",None)
    c["trainer"].pop("run_dir",None)
    for k in ("generic_weight","articulator_weight","relation_weight"):
        c["loss"].pop(k,None)
    return c


def audit_run(run, status, split):
    cfg = yaml.safe_load(Path(run["config"]).read_text())
    run_dir = Path(run["run_dir"])
    resolved = yaml.safe_load((run_dir/"config.resolved.yaml").read_text())
    assert cfg == resolved, "Run resolved config differs from suite config"
    assert cfg["seed"] == run["seed"]
    assert tuple(cfg["loss"][k] for k in ("generic_weight","articulator_weight","relation_weight")) == OBJECTIVES[run["name"]]
    assert cfg["trainer"]["max_steps"] == status["arguments"]["steps"] == run["steps"]
    assert sha256(cfg["data"]["manifest"]) == status["manifest_sha256"] == run["manifest_sha256"]
    checkpoint = run_dir/"checkpoints/latest.pt"
    assert sha256(checkpoint) == run["checkpoint_sha256"], "Checkpoint content changed"
    state = torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True)
    assert state["global_step"] == run["steps"], "Incomplete checkpoint"
    assert state["config_hash"] == contract_hash(cfg)
    contract = json.loads((run_dir/"input_contract.json").read_text())
    assert all(state.get(k) == v for k,v in contract.items())
    upstream = json.loads((run_dir/"upstream_checkpoints.json").read_text())
    assert state["upstream_init"] == upstream
    assert run["effective_batch"] == cfg["data"]["batch_size_per_gpu"]*cfg["trainer"]["accumulate_steps"]*contract["world_size"]
    for path,digest in run["artifact_sha256"].items():
        assert sha256(path) == digest, f"Artifact content changed: {path}"
    feature_contract = {}
    warnings = []
    feature_root = Path(status["arguments"].get("feature_root","features/controlled_pilot"))
    for part in ("train",split):
        features = Path(run.get("features",{}).get(part,feature_root/f"{run['name']}_seed{run['seed']}_{part}.npz"))
        meta = json.loads(features.with_suffix(".report.json").read_text())
        assert meta["checkpoint_sha256"] == run["checkpoint_sha256"]
        assert meta["manifest_sha256"] == contract["evaluation_manifest_hashes"][meta["arguments"]["manifest"]]
        assert meta["errors"] == [], "Decode failures must be handled before paired comparison"
        assert meta["rgb_only"] and meta["retained_rows"] == meta["total_rows"]
        if meta["upstream_init"] != upstream:
            warnings.append({"path":str(features.with_suffix('.report.json')),"issue":"constructor initialization metadata differs from restored trained provenance","authoritative_upstream_init":upstream})
        with np.load(features,allow_pickle=False) as data:
            assert np.isfinite(data["features"]).all()
            feature_contract[part] = {"rows_json":data["rows_json"].tolist(),"shape":list(data["features"].shape),
                                      "pooling":meta["pooling"],"num_frames":meta["num_frames"],"crop_size":meta["crop_size"],
                                      "windows":meta["arguments"]["windows"],"precision":meta["arguments"]["precision"]}
    metrics = [json.loads(line) for line in (run_dir/"metrics.jsonl").read_text().splitlines()]
    assert [m["global_step"] for m in metrics] == list(range(1,run["steps"]+1)), "Missing/duplicated optimizer steps"
    assert all(all(np.isfinite(v) for v in m.values()) for m in metrics)
    edge_keys = [k for k in metrics[0] if k.startswith("edge_") and k.endswith("_valid_queries")]
    health = {"steps":len(metrics),"max_collapse_alarm_count":max(m['collapse_alarm_count'] for m in metrics),
              "final_loss":metrics[-1]['loss_total'], "final_student_std":metrics[-1]['student_embedding_std'],
              "final_teacher_std":metrics[-1]['teacher_embedding_std'],
              "final_between_sample_cosine":metrics[-1].get('between_sample_cosine'),
              "edge_query_totals":{k:sum(m[k] for m in metrics) for k in edge_keys}}
    assert all(v > 0 for v in health['edge_query_totals'].values()), "A relation edge never had valid targets"
    report = json.loads(Path(run["evaluation"][split]).read_text())
    assert report["feature_sha256"] == sha256(Path(run.get("features",{}).get(split,feature_root/f"{run['name']}_seed{run['seed']}_{split}.npz")))
    assert report["gallery_sha256"] == sha256(Path(run.get("features",{}).get("train",feature_root/f"{run['name']}_seed{run['seed']}_train.npz")))
    return report,{"controls":control_config(cfg), "input_contract":contract, "upstream_init":upstream,
                   "feature_contract":feature_contract,"coverage_contract":coverage_contract(report)},health,warnings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--status",default="reports/controlled_pilot/status.json")
    p.add_argument("--split",choices=["val","test"],default="val")
    p.add_argument("--out",default="reports/controlled_pilot/summary.json")
    args = p.parse_args()
    status = json.loads(Path(args.status).read_text())
    assert sha256(status["arguments"]["eval_config"]) == status["eval_config_sha256"]
    assert sha256(Path(status["arguments"]["eval_root"])/"protocol.json") == status["eval_protocol_sha256"]
    grouped, health, warnings = {}, {}, []
    reference, seen = None, set()
    for run in status["runs"]:
        key = (run["name"],run["seed"])
        assert key not in seen, "Duplicated suite run"
        seen.add(key)
        if run["status"] != "complete" or args.split not in run["evaluation"]:
            continue
        report, contract, h, w = audit_run(run,status,args.split)
        if reference is None:
            reference = contract
        assert reference == contract, "Unequal controls, feature rows or evaluation coverage"
        grouped.setdefault(run["name"],{})[run["seed"]] = metrics_from_report(report)
        health[f"{run['name']}_seed{run['seed']}"] = h
        warnings.extend(w)
    report = summarize(grouped,status["arguments"]["seeds"],status["arguments"]["variants"],args.split,audit_verified=reference is not None)
    report.update(artifact_audit={"verified_runs":len(health),"contract_equal":reference is not None,
                                 "manifest_sha256":status["manifest_sha256"],"eval_protocol_sha256":status["eval_protocol_sha256"],
                                 "source_hash":reference['input_contract']['source_hash'] if reference else None,
                                 "checks":"checkpoint step/config/hash, artifact bytes, finite feature tensors, identical sample rows, normalization, coverage and controls"},
                  health=health,provenance_warnings=warnings,summary_source_sha256=sha256(__file__))
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    md = ["# Controlled frozen evaluation pilot","",f"Split: {args.split}. Same pretrained encoder, data, optimizer, batch, steps and random masks.","",
          "| Variant | R@1 | R@5 | Instance mAP | Phonology macro-F1 | Signer accuracy |","|---|---:|---:|---:|---:|---:|"]
    for name,row in report["summary"].items():
        def fmt(k):
            v = row[k]
            return "unavailable" if v['mean'] is None else f"{v['mean']:.4f}"+(f" ± {v['std']:.4f}" if v['std'] is not None else "")
        md.append("| "+name+" | "+" | ".join(fmt(k) for k in METRICS)+" |")
    md += ["",f"Decision: **{report['decision']['status']}**.","",report["scope"],"",report["phase2_rule"],"",
           "## Paired relation comparisons","","| Comparison | Δ R@1 | Δ phonology F1 | Δ signer accuracy | Seeds |","|---|---:|---:|---:|---:|"]
    for name,item in report['decision']['comparisons'].items():
        delta = item['mean_delta']
        md.append(f"| {name} | {delta['Recall@1']:+.4f} | {delta['phonology_macro_f1']:+.4f} | {delta.get('signer_accuracy',float('nan')):+.4f} | {len(item['seeds'])} |")
    md += ["","A smaller signer score is beneficial only when linguistic task quality is maintained.","",
           f"Verified runs: {len(health)}. Provenance reporting warnings: {len(warnings)}. Original feature/evaluation artifacts are preserved."]
    out.with_suffix(".md").write_text("\n".join(md)+"\n")
    print(json.dumps({"runs":len(health),"decision":report['decision']}))


if __name__ == "__main__":
    main()
