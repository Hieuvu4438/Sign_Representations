from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from signcoord.config import BASE_CONFIG,PROJECT, load_config
from signcoord.provenance import atomic_json, execution_stamp, object_hash, sha256_file


def doctor(c):
    import av
    import torch
    import numpy as np
    missing=[]
    for key in (() if c["arm"] in ("A0","A1_ref") else ("ssl_train_manifest","pose_cache_root")):
        if c["data"][key] is None or not Path(c["data"][key]).exists(): missing.append("data."+key)
    if c["model"]["initialization_mode"] != "random_init":
        checkpoint=c["model"]["initialization_checkpoint"]
        if checkpoint is None or not Path(checkpoint).is_file(): missing.append("model.initialization_checkpoint")
    available=c["model"]["backbone"] in ("tiny_fixture","signrep_hiera_adapter","hiera_base_video_adapter")
    return {**execution_stamp(),"status":"passed" if not missing and available else "inputs_missing",
            "missing_inputs":missing,"backbone_implemented":available,"config_hash":object_hash(c),
            "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"av":av.__version__,
                           "cuda_available":torch.cuda.is_available()},
            "spectral_windows":len(range(0,c["model"]["input_frames"]-c["loss"]["relation"]["window"],c["loss"]["relation"]["hop"]))}


def main(argv=None):
    p=argparse.ArgumentParser(prog="signcoord")
    sub=p.add_subparsers(dest="command",required=True)
    for command in ("doctor","audit-data","fit-target-stats","audit-targets","smoke","train",
                    "extract-features","eval-retrieval","export","profile","init-reference","extract-signrep-reference","manifest"):
        sp=sub.add_parser(command)
        sp.add_argument("--config",default=str(PROJECT/"configs/base.yaml") if (PROJECT/"configs/base.yaml").exists() else str(BASE_CONFIG))
        sp.add_argument("--set",action="append",default=[]); sp.add_argument("--arm"); sp.add_argument("--seed",type=int)
        sp.add_argument("--output")
        if command == "manifest":
            sp.add_argument("--adapter",required=True,choices=("asl_citizen","wlasl2000","how2sign","configured_source"))
            sp.add_argument("--root",required=True);sp.add_argument("--split",required=True)
            sp.add_argument("--annotation");sp.add_argument("--max-classes",type=int)
            sp.add_argument("--max-per-class",type=int);sp.add_argument("--limit",type=int)
            sp.add_argument("--workers",type=int,default=2)
            sp.add_argument("--exclude-groups-from")
            sp.add_argument("--sources",default=str(PROJECT/"configs/sources.yaml"));sp.add_argument("--source-name")
        if command == "fit-target-stats": sp.add_argument("--phase-output",required=True)
        if command == "audit-data":
            sp.add_argument("--perceptual",action="store_true")
            sp.add_argument("--perceptual-cache",default=str(PROJECT/"data/perceptual_cache"))
            sp.add_argument("--review-decisions")
        if command in ("train","smoke"):
            sp.add_argument("--resume"); sp.add_argument("--steps",type=int)
        if command in ("extract-features","export"):
            sp.add_argument("--checkpoint",required=True)
        if command=='extract-features':sp.add_argument('--clip-only',action='store_true')
        if command in ("extract-features","extract-signrep-reference"): sp.add_argument("--manifest",required=True)
        if command == "eval-retrieval":
            sp.add_argument("--gallery",required=True); sp.add_argument("--queries",required=True)
            sp.add_argument("--labels",required=True)
        if command == "profile":
            sp.add_argument("--warmup-steps",type=int,default=50); sp.add_argument("--measure-steps",type=int,default=500)
    a=p.parse_args(argv)
    try:
        c=load_config(a.config,a.set,a.arm,a.seed)
        if a.command == "doctor":
            report=doctor(c)
        elif a.command == "manifest":
            from signcoord.data.build_manifest import build_manifest
            adapter=a.adapter
            if adapter=="configured_source":
                import yaml
                if not a.source_name:raise ValueError("configured_source requires --source-name")
                source=yaml.safe_load(Path(a.sources).read_text())["datasets"][a.source_name]
                adapter=source["adapter"]
                if not a.annotation:a.annotation=source.get("annotation")
            if not a.output:raise ValueError("manifest requires --output")
            report=build_manifest(adapter,a.root,a.split,a.output,a.annotation,c["seed"],
                                  a.max_classes,a.max_per_class,a.limit,a.workers,a.exclude_groups_from)
        elif a.command == "init-reference":
            from signcoord.engine.reference import initialize_reference
            if not a.output: raise ValueError("init-reference requires --output")
            report=initialize_reference(c,a.output)
        elif a.command == "extract-signrep-reference":
            from signcoord.eval.signrep_reference import extract_official_reference
            if not a.output: raise ValueError("reference extraction requires --output")
            report=extract_official_reference(c,a.manifest,a.output,c["train"]["device"])
        elif a.command == "audit-data":
            from signcoord.data.manifest import audit_overlap,read_manifest
            train=read_manifest(c["data"]["ssl_train_manifest"],{"ssl_train"})
            eval_paths=c["data"]["downstream_manifests"]
            if c["data"]["ssl_dev_manifest"]: eval_paths=[c["data"]["ssl_dev_manifest"]]+eval_paths
            evaluation=[r for path in eval_paths for r in read_manifest(path)]
            report={**execution_stamp(),**audit_overlap(train,evaluation),"status":"passed_exact_audit_only",
                    "config_hash":object_hash(c),"manifest_hash":sha256_file(c["data"]["ssl_train_manifest"])}
            if a.perceptual:
                from signcoord.data.perceptual import audit_perceptual
                decisions=json.loads(Path(a.review_decisions).read_text()) if a.review_decisions else None
                screen=audit_perceptual(train,evaluation,a.perceptual_cache,decisions)
                report['perceptual']=screen
                report['near_duplicate_verification']=screen['near_duplicate_verification']
                report['passed']=report['passed'] and screen['passed']
                report['status']='passed_exact_and_perceptual_screen'
            report['evaluation_manifest_hashes']={path:sha256_file(path) for path in eval_paths}
            if not report["passed"]: report["status"]="failed"
        elif a.command == "fit-target-stats":
            from signcoord.engine.train import make_dataset
            from signcoord.targets.statistics import fit_target_statistics
            if not a.output: raise ValueError("fit-target-stats requires --output")
            rc=c["loss"]["relation"]
            stats,phase=fit_target_statistics(make_dataset(c),c["data"]["ssl_train_manifest"],a.output,a.phase_output,
                          rc["window"],rc["hop"],rc["min_confidence"],rc["calibration_percentile"],rc["energy_floor"])
            report={**execution_stamp(),"status":"passed","stats_path":a.output,"phase_path":a.phase_output,
                    "calibration_count":len(stats["sample_ids"]),"config_hash":object_hash(c)}
        elif a.command == "audit-targets":
            from signcoord.engine.train import make_dataset
            ds=make_dataset(c); rows=[]
            for i in range(len(ds)):
                sample=ds[i]; target=sample["targets"]
                rows.append({"sample_id":sample["audit"]["sample_id"],
                  "normalization_valid":bool(target["normalization_valid"]),
                  "geometry_coverage":target["joint_valid"].float().mean().item(),
                  "spectral_timing_coverage":target["spectral_frame_valid"].float().mean().item(),
                  "repeat_fraction":sample["audit"]["repeat_fraction"]})
            report={**execution_stamp(),"status":"passed","samples":rows,"config_hash":object_hash(c),
                    "manifest_hash":sha256_file(c["data"]["ssl_train_manifest"])}
        elif a.command == "profile":
            from signcoord.engine.profile import profile
            report=profile(c,a.warmup_steps,a.measure_steps)
        elif a.command == "extract-features":
            from signcoord.eval.features import extract_features
            if not a.output: raise ValueError("extract-features requires --output")
            report=extract_features(a.checkpoint,a.manifest,a.output,c["train"]["device"],not a.clip_only)
        elif a.command == "export":
            from signcoord.export.bundle import export_encoder
            if not a.output: raise ValueError("export requires --output")
            report=export_encoder(a.checkpoint,a.output)
        elif a.command == "eval-retrieval":
            import numpy as np
            from signcoord.eval.retrieval import evaluate_retrieval
            labels=json.loads(Path(a.labels).read_text())
            with np.load(a.gallery,allow_pickle=False) as g,np.load(a.queries,allow_pickle=False) as q:
                gids,qids=g["sample_ids"].tolist(),q["sample_ids"].tolist()
                if set(g["source_groups"].tolist())&set(q["source_groups"].tolist()):
                    raise ValueError("gallery/query source-group overlap")
                report={**execution_stamp(),"status":"passed",**evaluate_retrieval(g["z"],q["z"],
                        [labels[i] for i in gids],[labels[i] for i in qids],gids,qids,q["groups"].tolist())}
            report.update(gallery_hash=sha256_file(a.gallery),query_hash=sha256_file(a.queries),labels_hash=sha256_file(a.labels))
        else:
            from signcoord.engine.train import train
            if a.command == "smoke" and a.steps is not None:
                c["train"]["max_updates"]=a.steps
            report=train(c,a.resume,stop_after=a.steps if a.command == "train" else None)
        # fit outputs are the actual stats file; avoid replacing them with a report.
        if a.output and a.command not in ("manifest","fit-target-stats","extract-features","extract-signrep-reference","export","init-reference"): atomic_json(a.output,report)
        print(json.dumps(report,allow_nan=False))
        return 0 if report["status"].startswith("passed") else 1
    except (ValueError,FileNotFoundError,KeyError) as exc:
        print(json.dumps({"status":"failed","command":a.command,"error":str(exc)}),file=sys.stderr)
        return 2


if __name__ == "__main__": raise SystemExit(main())
