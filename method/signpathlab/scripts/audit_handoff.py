"""Read actual final artifacts; fail instead of inferring a completed experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import torch

from signpath.cache.hashing import canonical_hash, file_hash
from signpath.cli import write_json
from signpath.experiments import summarize
from signpath.inference.deployment import graph_inputs
from signpath.inference.embed import embed_features
from signpath.inference.export import ExportedSession
from signpath.inference.run import load_run
from signpath.evaluation.protocols import development_corpora
from signpath.training.runner import full_clip_batch, runtime_implementation_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", default="artifacts/verified_pilot")
    parser.add_argument("--run", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--embed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, implementation = Path(args.pilot_root), runtime_implementation_hash()
    checks = {}
    suite = ET.parse("artifacts/phase4_tests.xml").getroot().find("testsuite")
    assert int(suite.attrib["tests"]) >= 128 and all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
    checks["tests"] = dict(suite.attrib)
    pip = subprocess.run([str(Path(__import__("sys").executable)), "-m", "pip", "check"], capture_output=True, text=True, check=True)
    checks["pip_check"] = pip.stdout.strip()
    reports = [json.loads((root / f"runs_seed{seed}.json").read_text()) for seed in (17, 29, 43)]
    assert all(len(r) == 5 and all(x["status"] == "completed" for x in r) for r in reports)
    common = ("environment.json", "requirements-main-extraction.lock.txt", "resolved_config.yaml", "source_registry.json",
              "local_patch.diff", "checkpoint_load_report.json", "manifest_hashes.json", "split_audit.json",
              "cache_and_target_provenance.json", "train_log.jsonl", "mask_diagnostics.jsonl", "feature_spectrum.npz",
              "downstream_head_config.yaml", "decision_memo.md", "source_snapshot.tar.gz", "source_snapshot.json")
    completed = []
    for reports_by_seed in reports:
        for entry in reports_by_seed:
            head_run = Path(entry["head_run"])
            assert all((head_run / name).is_file() for name in (*common, "predictions.val.jsonl", "metrics.val.json", "protocol.lock.json", "head.pt"))
            head = torch.load(head_run / "head.pt", map_location="cpu", weights_only=False)
            assert head["identity"]["implementation_hash"] == implementation
            predictions = [json.loads(line) for line in (head_run / "predictions.val.jsonl").read_text().splitlines()]
            assert len(predictions) == 16 and len({p["sample_id"] for p in predictions}) == 16
            for prediction in predictions:
                assert all(key in prediction for key in ("reference_label_id", "prediction_label_id", "top5_label_ids", "source_group_id", "signer_id", "quality_bucket", "model_hash"))
                assert prediction["split"] == "val" and prediction["model_hash"] == file_hash(head_run / "head.pt")
            record = {"head_run": str(head_run), "head_sha256": file_hash(head_run / "head.pt"), "prediction_count": len(predictions)}
            if head["encoder_path"]:
                run = Path(head["encoder_path"]).parent.parent
                assert all((run / name).is_file() for name in common)
                report = json.loads((run / "summary.json").read_text())
                assert report["status"] == "completed" and report["successful_steps"] == 100 and not report["test_used_for_selection"]
                state = torch.load(head["encoder_path"], map_location="cpu", weights_only=False)
                assert state["identities"]["runtime_implementation_hash"] == implementation
                archive = json.loads((run / "source_snapshot.json").read_text())
                assert file_hash(run / "source_snapshot.tar.gz") == archive["archive_sha256"]
                logs = [json.loads(line) for line in (run / "logs.jsonl").read_text().splitlines()]
                assert all(r["distinct_source_frame_exposure"] > 0 for r in logs)
                record.update(training_run=str(run), successful_steps=report["successful_steps"],
                    distinct_source_frame_exposure=sum(r["distinct_source_frame_exposure"] for r in logs))
            completed.append(record)
    checks["completed_trials"] = completed
    summarize(root, root / "comparison.csv")
    comparison = json.loads((root / "comparison.json").read_text())
    pairs = comparison["primary_E03_E04_pairs"]
    assert len(pairs) == 3 and all(p["matched_successful_steps_and_masks"] and p["paired_sampling_intervals"]["status"] == "measured" for p in pairs)
    for pair in pairs:
        states, exposures, eligible = [], [], []
        for method in ("E03", "E04"):
            row = next(r for r in comparison["rows"] if r["experiment"] == method and r["seed"] == pair["seed"])
            head = torch.load(Path(row["run_id"]) / "head.pt", map_location="cpu", weights_only=False)
            state = torch.load(head["encoder_path"], map_location="cpu", weights_only=False)
            states.append(state["trainer"]["model"])
            logs = [json.loads(line) for line in (Path(head["encoder_path"]).parent.parent / "logs.jsonl").read_text().splitlines()]
            exposures.append(sum(r["distinct_source_frame_exposure"] for r in logs))
            eligible.append(sum(sum(loss["geometry"]["counts"]) for r in logs for loss in r["logs"]))
        distance = sum(float((states[0][k].float() - states[1][k].float()).square().sum()) for k in states[0] if k.startswith("online.")) ** .5
        assert exposures[0] == exposures[1] and all(n > 0 for n in eligible) and distance > 0
        pair.update(distinct_source_frame_exposure=exposures[0], eligible_geometry_targets=eligible, online_weight_l2_difference=distance)
    checks["paired_comparison"] = pairs
    checks["aggregate"] = comparison["aggregate"]
    run, session = load_run(args.run), ExportedSession(args.export_dir)
    assert session.meta["model_hash"] == run.model_hash
    train, val = development_corpora(run.resolved)
    bundle = val.load(val.available[0])
    inputs = graph_inputs(full_clip_batch([bundle], run.model_config))
    with torch.no_grad():
        eager = embed_features(run, bundle)
        exported = session.full_clip(inputs)
    for first, second in zip(eager, (exported[0][0], exported[2][0], exported[3][0], exported[4][0])):
        torch.testing.assert_close(first, second, atol=2e-5, rtol=1e-4)
    checks["real_full_clip_export_parity"] = {"sample_id": val.available[0], "frames": len(bundle.timestamps), "atol": 2e-5, "rtol": 1e-4}
    embedding = Path(args.embed)
    raw = json.loads(embedding.with_suffix(embedding.suffix + ".json").read_text())
    assert raw["artifact_sha256"] == file_hash(embedding) and raw["model_hash"] == run.model_hash
    with np.load(embedding, allow_pickle=False) as arrays:
        assert arrays["dense"].shape[0] == len(arrays["timestamps"]) == len(arrays["valid"])
        assert arrays["valid"].any() and np.isfinite(arrays["dense"]).all()
    checks["raw_video_embed"] = raw
    for kind in ("ctc", "slt"):
        report = json.loads(Path(f"artifacts/phase4_{kind}_interface.json").read_text())
        assert report["synthetic_correctness_only"] and report["losses"][-1] < report["losses"][0]
        assert report["real_experiment"]["status"] == "not_run"
    checks["continuous"] = "functional toy heads/metrics; real continuous corpus not configured"
    target = json.loads(Path("data/pilot8/targets/logsig_stats.npz.json").read_text())
    allowed = set(train.pretrain_sources())
    assert set(target["fitting_source_ids"]) <= allowed
    checks["geometry_train_only"] = {"source_count": len(target["fitting_source_ids"]), "source_ids_hash": canonical_hash(target["fitting_source_ids"])}
    write_json(args.output, {"status": "passed", "runtime_implementation_hash": implementation, "checks": checks,
        "scope": "mandatory implemented pipeline, final locked-environment pilot and artifact consistency; optional unrun backends remain documented"})
    print(json.dumps({"status": "passed", "trials": len(completed), "runtime_implementation_hash": implementation, "output": args.output}))


if __name__ == "__main__":
    main()
