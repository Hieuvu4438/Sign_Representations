"""Dynamic torch.export artifact with eager/serialized parity and preprocessing contracts."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import torch

from ..cache.hashing import canonical_hash, file_hash
from ..cli import write_json
from ..pose.topology import EDGES, JOINT_NAMES, TOPOLOGY_HASH
from .run import load_run


@contextmanager
def exportable_attention():
    enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(enabled)


def example_inputs(raw_dimensions, length=8, batch=2, *, missing=True):
    generator = torch.Generator().manual_seed(137)
    streams = tuple(torch.randn(batch, length, dimension, generator=generator) for dimension in raw_dimensions)
    valid = torch.ones(batch, length, 7, dtype=torch.bool)
    if missing:
        valid[:, :, 1] = False
        valid[:, length // 2:, 6] = False
    timestamps = torch.arange(length, dtype=torch.float64)[None].expand(batch, -1).clone() / 25 + 2.0
    sources = torch.arange(length, dtype=torch.int64)[None].expand(batch, -1).clone()
    return streams, valid, timestamps, sources


def export_graph(graph, raw_dimensions, *, max_frames=512, max_batch=16):
    graph = graph.cpu().eval()
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    frames = torch.export.Dim("frames", min=1, max=max_frames)
    dynamic = (tuple({0: batch, 1: frames} for _ in range(7)), {0: batch, 1: frames}, {0: batch, 1: frames}, {0: batch, 1: frames})
    with exportable_attention():
        return torch.export.export(graph, example_inputs(raw_dimensions), dynamic_shapes=dynamic, strict=False)


def verify_parity(graph, exported_module, raw_dimensions, *, lengths=(1, 5, 48, 71)):
    records = []
    with torch.no_grad(), exportable_attention():
        for length in lengths:
            for batch in (1, 2):
                inputs = example_inputs(raw_dimensions, length, batch)
                expected, actual = graph(*inputs), exported_module(*inputs)
                maximum = 0.0
                for first, second in zip(expected, actual):
                    torch.testing.assert_close(first, second, atol=2e-5, rtol=1e-4)
                    if first.numel() and first.dtype != torch.bool:
                        maximum = max(maximum, float((first - second).abs().max()))
                records.append({"frames": length, "batch": batch, "missing_streams": True, "max_absolute_error": maximum})
        inputs = list(example_inputs(raw_dimensions, 9, 2))
        inputs[1] = torch.zeros_like(inputs[1])
        actual = exported_module(*inputs)
        assert all(torch.isfinite(x).all() for x in actual if x.is_floating_point())
        assert not actual[0].any() and not actual[2].any() and not actual[3].any() and not actual[4].any()
        records.append({"frames": 9, "batch": 2, "all_invalid": True, "finite_zero_outputs": True})
    return records


def export_run(run_dir, output_dir, format="torch_export"):
    if format != "torch_export":
        raise ValueError("Only torch_export is implemented; ONNX/quantization are optional unavailable backends")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("Export directory exists; choose a new artifact path")
    run = load_run(run_dir, "cpu")
    graph = run.graph().cpu().eval()
    dimensions = run.model_config["model"]["raw_dimensions"]
    torch.set_num_threads(1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    try:
        program = export_graph(graph, dimensions)
        path = temporary / "encoder.pt2"
        torch.export.save(program, path)
        loaded = torch.export.load(path).module()
        parity = verify_parity(graph, loaded, dimensions)
        torch.save(graph.state_dict(), temporary / "online_and_head_weights.pt")
        metadata = {"schema_version": 1, "format": format, "torch_version": torch.__version__,
            "model_hash": run.model_hash, "encoder_pt2_sha256": file_hash(path), "input_identity": run.input_identity,
            "raw_dimensions": dimensions, "inputs": ["streams: tuple7 float32[B,T,F_stream]", "valid: bool[B,T,7]",
                "timestamps: float64[B,T] original seconds", "source_frame_index: int64[B,T], -1 only padding"],
            "outputs": ["dense[B,T,D]", "timestamps[B,T]", "valid[B,T]", "clip[B,D]", "logits[B,C] or [B,0] without head"],
            "dynamic_limits": {"batch": [1, 16], "frames": [1, 512]}, "class_ids": list(run.class_ids),
            "inference_mode": "offline_bidirectional_unmasked", "teacher_predictor_included": False,
            "head_kind": run.resolved.values["downstream"]["head"] if run.head is not None else None,
            "head_dimension": run.head.classifier.in_features if run.head is not None else None,
            "attentive_heads": run.resolved.values["downstream"]["attentive_heads"],
            "l2_normalize": run.resolved.values["downstream"]["l2_normalize"],
            "weights_sha256": file_hash(temporary / "online_and_head_weights.pt"),
            "topology_hash": TOPOLOGY_HASH, "pose_and_dino_outside_graph": True,
            "preprocessing_config": {"cache": run.resolved.values["cache"], "pose": run.resolved.values["pose"],
                "target_fps": run.resolved.values["data"]["target_fps"]}, "tiling": {key: run.resolved.values["downstream"][key]
                for key in ("window_samples", "overlap_samples")}, "parity": parity}
        metadata["artifact_contract_hash"] = canonical_hash(metadata)
        write_json(temporary / "metadata.json", metadata)
        write_json(temporary / "topology.json", {"joint_names": list(JOINT_NAMES), "edges": list(EDGES), "sha256": TOPOLOGY_HASH})
        (temporary / "README.md").write_text(
            "# Offline SignPathLab export\n\nLoad with torch.export.load('encoder.pt2').module(). "
            "Pass the four typed inputs documented in metadata.json. DINO/IMAGE landmarks, "
            "ROI normalization and asset verification remain external; their identities are recorded. "
            "This graph accepts one window or full sequence within dynamic limits. For the fitted "
            "long-clip protocol use timestamp-aligned window averaging from signpath.inference.embed. "
            "The model is bidirectional, with no streaming/quantization claim.\n")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "completed", "format": format, "output": str(destination), "model_hash": run.model_hash,
            "encoder_sha256": metadata["encoder_pt2_sha256"], "parity_cases": len(parity)}


class ExportedSession:
    """Checked artifact loader; tiling follows the fitted full-clip protocol."""
    def __init__(self, directory, device="cpu"):
        from ..models.heads import LinearHead, AttentiveHead
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        digest = meta.pop("artifact_contract_hash", None)
        if canonical_hash(meta) != digest or file_hash(directory / "encoder.pt2") != meta["encoder_pt2_sha256"]:
            raise ValueError("Export metadata/graph checksum mismatch")
        if file_hash(directory / "online_and_head_weights.pt") != meta["weights_sha256"]:
            raise ValueError("Export state checksum mismatch")
        self.meta, self.device = meta, torch.device(device)
        self.module = torch.export.load(directory / "encoder.pt2").module().to(device)
        self.head = None
        if meta["head_kind"] is not None:
            self.head = (AttentiveHead(meta["head_dimension"], len(meta["class_ids"]), meta["attentive_heads"])
                         if meta["head_kind"] == "attentive" else LinearHead(meta["head_dimension"], len(meta["class_ids"]))).to(device)
            state = torch.load(directory / "online_and_head_weights.pt", weights_only=True, map_location=device)
            self.head.load_state_dict({k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")}, strict=True)
            self.head.eval().requires_grad_(False)

    @torch.no_grad()
    def window(self, inputs):
        from .deployment import validate_inputs
        validate_inputs(inputs, self.meta["raw_dimensions"])
        return self.module(*inputs)

    @torch.no_grad()
    def full_clip(self, inputs):
        from .deployment import validate_inputs
        from .extract import window_starts
        validate_inputs(inputs, self.meta["raw_dimensions"])
        streams, valid, times, sources = inputs
        if len(valid) != 1:
            raise ValueError("full_clip takes one variable-length clip at a time")
        n = valid.shape[1]
        sums, counts = None, torch.zeros(1, n, 1, device=valid.device)
        real = torch.zeros(1, n, dtype=torch.bool, device=valid.device)
        schedule = self.meta["tiling"]
        for start in window_starts(n, schedule["window_samples"], schedule["overlap_samples"]):
            stop = min(n, start + schedule["window_samples"])
            output = self.window((tuple(x[:, start:stop] for x in streams), valid[:, start:stop], times[:, start:stop], sources[:, start:stop]))
            dense, _, eligible, _, _ = output
            if sums is None:
                sums = torch.zeros(1, n, dense.shape[-1], device=dense.device)
            sums[:, start:stop] += dense * eligible[..., None]
            counts[:, start:stop] += eligible[..., None]
            real[:, start:stop] |= eligible
        dense = sums / counts.clamp_min(1)
        earlier = torch.tril(torch.ones(n, n, dtype=torch.bool, device=valid.device), diagonal=-1)
        repeated = ((sources[:, :, None] == sources[:, None, :]) & earlier[None] & real[:, None, :]).any(-1)
        unique = real & ~repeated
        clip = (dense * unique[..., None]).sum(1) / unique.sum(1, keepdim=True).clamp_min(1)
        if self.head is None:
            logits = clip[:, :0]
        elif self.meta["head_kind"] == "attentive":
            if not real.any():
                logits = clip.new_zeros(1, len(self.meta["class_ids"]))
            else:
                logits = self.head(torch.nn.functional.normalize(dense, dim=-1) if self.meta["l2_normalize"] else dense, real)
        else:
            logits = self.head(torch.nn.functional.normalize(clip, dim=-1) if self.meta["l2_normalize"] else clip)
            logits = torch.where(real.any(1, keepdim=True), logits, 0)
        return dense, times, real, clip, logits
