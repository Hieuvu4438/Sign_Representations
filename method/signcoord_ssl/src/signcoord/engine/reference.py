"""Materialize the frozen A0 baseline without targets or optimizer state."""
from pathlib import Path

from signcoord.engine.checkpoint import atomic_checkpoint
from signcoord.export.inference import encoder_from_config
from signcoord.provenance import execution_stamp, sha256_file


def initialize_reference(c, output):
    if c["arm"] != "A0":
        raise ValueError("init-reference currently supports generic frozen A0 only")
    path = Path(output)
    if path.exists():
        raise ValueError("reference output exists; use a new immutable path")
    encoder = encoder_from_config(c, initialize=True).eval().requires_grad_(False)
    metadata = {**execution_stamp(), "arm": "A0", "optimizer_updates": 0,
                "sign_ssl_exposure": 0, "pose_targets_used": False,
                "checkpoint_load_report": encoder.load_report,
                "readout": "final_stage_pre_MAE_fusion_and_norm_uniform_temporal_pool",
                "scientific_evidence": False}
    atomic_checkpoint(path, {"schema_version": "signcoord_checkpoint_v1", "microstep_boundary": 0,
                            "model_state": {"encoder."+k: v for k, v in encoder.state_dict().items()},
                            "optimizer_update": 0, "resolved_config": c,
                            "reference_metadata": metadata})
    return {"status": "passed", "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
            **metadata}
