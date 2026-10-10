"""Run a fixed A/B/C/D continued-SSL pilot, with no metric-driven recipe changes."""
import _bootstrap
import argparse
import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path
import yaml
from signrel.config import load_config
from signrel.data.manifest import read_manifest,sha256
from signrel.engine.checkpoint import contract_hash

VARIANTS={"A_generic":"jepa_baseline.yaml","B_articulator":"articulator_mvp.yaml",
          "C_relation":"relation_controlled.yaml","D_combined":"combined_controlled.yaml"}


def feature_root_for(run_root,report_root,requested=None):
    if requested:
        return Path(requested)
    if str(run_root)=="runs/controlled_pilot" and str(report_root)=="reports/controlled_pilot":
        return Path("features/controlled_pilot")
    identity = str(Path(run_root).resolve())+"\n"+str(Path(report_root).resolve())
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return Path("features")/f"{Path(report_root).name}_{suffix}"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="manifests/how2sign_mvp_ready_v2.jsonl")
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    p.add_argument("--variants",nargs="+",choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument("--run-root",default="runs/controlled_pilot")
    p.add_argument("--report-root",default="reports/controlled_pilot")
    p.add_argument("--feature-root",help="Distinct feature destination; derived from custom run/report roots by default")
    p.add_argument("--eval-root",default="manifests/eval_pilot")
    p.add_argument("--eval-config",default="configs/eval/retrieval_asl_citizen.yaml")
    p.add_argument("--splits",nargs="+",choices=["val","test"],default=["val"])
    p.add_argument("--accumulate",type=int,default=2)
    args=p.parse_args()
    args.feature_root=str(feature_root_for(args.run_root,args.report_root,args.feature_root))
    reports=Path(args.report_root)
    reports.mkdir(parents=True,exist_ok=True)
    protocol={"arguments":vars(args),"manifest_sha256":sha256(args.manifest),
              "eval_protocol_sha256":sha256(Path(args.eval_root)/"protocol.json"),
              "eval_config_sha256":sha256(args.eval_config),"runs":[]}
    status_path=reports/"status.json"
    previous={}
    if status_path.exists():
        old=json.loads(status_path.read_text())
        # The first implementation used one fixed feature root. Preserve that
        # exact contract when verifying completed legacy suites.
        old["arguments"].setdefault("feature_root","features/controlled_pilot")
        for key in ("arguments","manifest_sha256","eval_protocol_sha256","eval_config_sha256"):
            if old.get(key)!=protocol[key]:
                raise AssertionError(f"Existing suite {key} mismatch; choose another report-root")
        previous={(r["name"],r["seed"]):r for r in old["runs"]}
    def update():
        tmp=status_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(protocol,indent=2))
        tmp.replace(status_path)
    def run(command,log):
        print(json.dumps({"command":command,"log":str(log)}),flush=True)
        with log.open("a") as f:
            subprocess.run([sys.executable,*command],check=True,stdout=f,stderr=subprocess.STDOUT,
                           env={**os.environ,"OMP_NUM_THREADS":"4"})
    for seed in args.seeds:
        for name,base in VARIANTS.items():
            if name not in args.variants:
                continue
            cfg=load_config(Path("configs/pretrain")/base)
            cfg["seed"]=seed
            cfg["experiment"]["scope"]="fixed pilot; not full benchmark or paper validation"
            cfg["data"]["manifest"]=args.manifest
            cfg["trainer"].update(max_steps=args.steps,accumulate_steps=args.accumulate,
                                  checkpoint_every=args.steps,log_every=20,
                                  run_dir=str(Path(args.run_root)/f"{name}_seed{seed}"))
            run_dir=Path(cfg["trainer"]["run_dir"])
            label=f"{name}_seed{seed}"
            config_path=reports/f"{label}.yaml"
            if config_path.exists():
                existing=yaml.safe_load(config_path.read_text())
                if contract_hash(existing)!=contract_hash(cfg):
                    raise AssertionError("Existing pilot config differs; choose another report-root")
            else:
                config_path.write_text(yaml.safe_dump(cfg))
            record={"name":name,"seed":seed,"config":str(config_path),"run_dir":str(run_dir),"status":"training",
                    "effective_batch":cfg["data"]["batch_size_per_gpu"]*args.accumulate,"steps":args.steps,
                    "manifest_sha256":protocol["manifest_sha256"],"evaluation":{}}
            prior=previous.get((name,seed))
            if prior and prior["status"]=="complete":
                if sha256(run_dir/"checkpoints/latest.pt")!=prior["checkpoint_sha256"]:
                    raise AssertionError("Completed suite checkpoint changed")
                for path,digest in prior["artifact_sha256"].items():
                    if sha256(path)!=digest:
                        raise AssertionError("Completed suite artifact changed")
                protocol["runs"].append(prior)
                update()
                continue
            protocol["runs"].append(record)
            update()
            ckpt=run_dir/"checkpoints/latest.pt"
            command=["scripts/train.py","--config",str(config_path)]
            if ckpt.exists():
                command += ["--resume",str(ckpt)]
            run(command,reports/f"{label}_train.log")
            record["status"]="extracting"
            update()
            paths={}
            for split in ["train",*args.splits]:
                feature_path=Path(args.feature_root)/f"{label}_{split}.npz"
                if feature_path.exists() and prior is None:
                    raise FileExistsError(f"Features already exist for another study: {feature_path}")
                run(["scripts/extract_features.py","--checkpoint",str(ckpt),"--manifest",
                     str(Path(args.eval_root)/f"asl_citizen_{split}.jsonl"),"--out",str(feature_path)],
                    reports/f"{label}_{split}_extract.log")
                paths[split]=feature_path
            record["features"]={split:str(path) for split,path in paths.items()}
            for split in args.splits:
                report_path=reports/f"{label}_{split}_eval.json"
                run(["scripts/evaluate_all.py","--features",str(paths[split]),"--gallery",str(paths["train"]),
                     "--config",args.eval_config,"--out",str(report_path)],reports/f"{label}_{split}_eval.log")
                record["evaluation"][split]=str(report_path)
            record["status"]="complete"
            record["checkpoint_sha256"]=sha256(ckpt)
            record["artifact_sha256"]={str(path):sha256(path) for path in paths.values()}
            record["artifact_sha256"].update({path:sha256(path) for path in record["evaluation"].values()})
            update()
    protocol["status"]="complete"
    update()


if __name__=="__main__":
    main()
