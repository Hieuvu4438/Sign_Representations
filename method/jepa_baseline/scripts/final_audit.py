"""Bind final module verification and reviews; never imply reproduced paper scores."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from signrep_baseline.cache import implementation_sha256, sha256


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_sources.py")], check=True)
    evidence = ROOT / "evidence"
    requirements = {
        "S0": ["source_verification.json"],
        "S1": ["all_pytest.txt", "S1_kinetics_init.json"],
        "S2": ["all_pytest.txt", "S2_pose_pipeline.json", "checkpoint_target_diagnostic.json"],
        "S3": ["all_pytest.txt"],
        "S4": ["all_pytest.txt", "S4_training_smoke.json", "S6_cli_smoke.json"],
        "S5": ["all_pytest.txt", "S5_rgb_smoke.json", "S5_sign2gpt_smoke.json",
               "S5_how2sign_smoke.json", "S6_recognition_eval.json", "S6_sign2gpt_inspect.json"],
        "S6": ["all_pytest.txt", "S6_cli_smoke.json"],
    }
    reviews = ROOT / "docs/reviews"
    review_files = []
    for stage in requirements:
        for phase in ["pre", "post"]:
            file = reviews / f"{stage}_{phase}.md"
            if not file.is_file() or not file.read_text().strip():
                raise ValueError(f"Missing review: {file}")
            review_files.append(file)
    for name in ["S2_mask_amendment.md", "S2_root_amendment_pre.md", "S2_root_amendment_post.md",
                 "S2_distance_amendment_pre.md", "S2_distance_amendment_post.md"]:
        file = reviews / name
        if not file.is_file():
            raise ValueError(f"Missing target amendment: {file}")
        review_files.append(file)
    pytest_output = (evidence / "all_pytest.txt").read_text()
    match = re.search(r"\b(\d+) passed\b", pytest_output)
    if not match or int(match[1]) != 28 or re.search(r"\b\d+ (?:failed|error|errors)\b", pytest_output):
        raise ValueError("Final complete test suite must pass")
    for file in sorted({name for names in requirements.values() for name in names}):
        if file.endswith(".json"):
            content = json.loads((evidence / file).read_text())
            if "result" in content and content["result"] != "PASS":
                raise ValueError(f"Failed integration: {file}")
    sources = json.loads((evidence / "source_verification.json").read_text())
    if sources["status"] != "PASS" or sources["pinned_files_checked"] != 1595:
        raise ValueError("Pinned upstream source audit failed")
    kinetics = json.loads((evidence / "S1_kinetics_init.json").read_text())
    if len(kinetics["loaded"]) != 298 or len(kinetics["excluded_pretraining_heads"]) != 110:
        raise ValueError("Kinetics encoder initialization coverage changed")
    pose = json.loads((evidence / "S2_pose_pipeline.json").read_text())
    if pose["body_root_mode"] != "canonical" or max(pose["teacher_FK_max_abs_error"].values()) > 1e-6:
        raise ValueError("Canonical teacher FK verification failed")
    replay = json.loads((evidence / "S4_training_smoke.json").read_text())
    for key in ["resume_batch_bitwise_equal", "resume_encoder_discriminator_ema_bitwise_equal",
                "resume_metrics_equal", "test_checkpoint_selection_rejected", "validation_checkpoint_selection"]:
        if replay[key] is not True:
            raise ValueError(f"Training replay check failed: {key}")
    cli = json.loads((evidence / "S6_cli_smoke.json").read_text())
    for key in ["held_out_video_sources_distinct", "audit_excludes_blocked_held_out_id",
                "cli_checkpoint_resume", "scheduled_validation_and_best_model_export",
                "frozen_feature_retrieval_and_translation_export"]:
        if cli[key] is not True:
            raise ValueError(f"CLI integration check failed: {key}")
    contract_path = ROOT / "runtime/cli_smoke/train/run_contract.json"
    contract = json.loads(contract_path.read_text())
    if contract["config"].get("choices"):
        raise ValueError("Final technical smoke should exercise default fidelity choices")
    code = []
    for directory in ["src", "scripts", "tests", "integrations", "configs"]:
        code.extend(path for path in (ROOT / directory).rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts)
    code.append(ROOT / "pyproject.toml")
    artifacts = sorted({evidence / name for names in requirements.values() for name in names})
    artifacts.extend(review_files + [ROOT / "docs/REPRODUCTION_SPEC.md", ROOT / "docs/TRANSLATION.md",
                                    ROOT / "README.md", contract_path])
    report = {
        "status": "PASS",
        "scope": "Module reconstruction and technical verification only",
        "tests_passed": 28,
        "pinned_upstream_files_checked": 1595,
        "implementation_sha256": implementation_sha256(),
        "default_body_root": "canonical; native body angles, camera-oriented hand angles",
        "default_distance": "signed_xyz_source_minus_destination",
        "requirements": {stage: {"module_status": "PASS", "evidence": names,
                                  "pre_review": f"docs/reviews/{stage}_pre.md",
                                  "post_review": f"docs/reviews/{stage}_post.md"}
                         for stage, names in requirements.items()},
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(code)},
        "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(artifacts))},
        "full_500000_step_YT_SL_run": "NOT_RUN",
        "paper_benchmark_scores": "NOT_REPRODUCED",
        "historical_equivalence": "UNVERIFIED: U01-U10 remain explicitly documented",
        "technical_data": "Small real ASL-Citizen clips; synthetic evaluation labels and translation text",
    }
    (evidence / "FINAL_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["status", "tests_passed", "pinned_upstream_files_checked",
                                                 "implementation_sha256", "paper_benchmark_scores"]}))


if __name__ == "__main__":
    main()
