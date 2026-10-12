from dataclasses import replace
import pytest
import torch
from signpath.inference.deployment import OfflineEncoderGraph, graph_inputs, validate_inputs
from signpath.models.heads import LinearHead, AttentiveHead
from signpath.models.teacher_student import ResearchModel
from model_helpers import batch_from, tiny_config


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("head_kind", ["linear", "attentive"])
def test_dense_deployment_parity_missing_padding_and_empty(bundle, head_kind):
    torch.manual_seed(7)
    config = tiny_config()
    model = ResearchModel(config).eval()
    head = LinearHead(24, 3) if head_kind == "linear" else AttentiveHead(24, 3, heads=4)
    head.eval()
    graph = OfflineEncoderGraph(model.online, head, head_kind=head_kind)
    assert not any("teacher" in key or "predictor" in key for key in graph.state_dict())
    batch = batch_from(bundle, geometry=False, pad_to=64)
    valid = batch.valid.clone()
    valid[:, :, 1], valid[:, 8:11, 5] = False, False
    batch = replace(batch, valid=valid)
    inputs = graph_inputs(batch)
    validate_inputs(inputs, config["model"]["raw_dimensions"])
    with torch.no_grad():
        expected = model.extract_embeddings(batch)
        dense, times, real, clip, logits = graph(*inputs)
        torch.testing.assert_close(dense, expected.dense, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(clip, expected.clip, atol=2e-6, rtol=1e-5)
        reference = (head(torch.nn.functional.normalize(clip, dim=-1)) if head_kind == "linear"
                     else head(torch.nn.functional.normalize(dense, dim=-1), real))
        torch.testing.assert_close(logits, reference, atol=2e-6, rtol=1e-5)
        empty = graph(inputs[0], torch.zeros_like(valid), inputs[2], inputs[3])
        assert not empty[0].any() and not empty[2].any() and not empty[3].any() and not empty[4].any()


def test_deployment_alias_clip_pooling_matches_encoder(bundle):
    model = ResearchModel(tiny_config()).eval()
    batch = batch_from(bundle, geometry=False)
    # Uneven upsampling must not change context weighting. Repeat actual values,
    # validity and original PTS together with their source observation IDs.
    ids = torch.arange(batch.valid.shape[1]).repeat_interleave(torch.arange(batch.valid.shape[1]) % 3 + 1)
    changes = {name: getattr(batch, name)[:, ids] for name in
               ("valid", "relative_seconds", "timestamps", "source_frame_index")}
    aliased = replace(batch, **changes, streams=tuple(x[:, ids] for x in batch.streams), lengths=torch.tensor([len(ids)]))
    with torch.no_grad():
        expected = model.extract_embeddings(batch)
        repeated = model.extract_embeddings(aliased)
        output = OfflineEncoderGraph(model.online)(*graph_inputs(aliased))
    torch.testing.assert_close(repeated.dense, expected.dense[:, ids], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(repeated.clip, expected.clip, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(output[0], repeated.dense, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(output[3], expected.clip, atol=2e-6, rtol=1e-5)


def test_left_padding_does_not_reset_time_origin_to_padding(bundle):
    model = ResearchModel(tiny_config()).eval()
    graph = OfflineEncoderGraph(model.online)
    inputs = graph_inputs(batch_from(bundle, geometry=False))
    streams, valid, times, sources = inputs
    padded = (tuple(torch.nn.functional.pad(x, (0, 0, 3, 0)) for x in streams),
              torch.nn.functional.pad(valid, (0, 0, 3, 0)), torch.nn.functional.pad(times, (3, 0)),
              torch.nn.functional.pad(sources, (3, 0), value=-1))
    with torch.no_grad():
        expected, actual = graph(*inputs), graph(*padded)
    torch.testing.assert_close(expected[0], actual[0][:, 3:], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(expected[3], actual[3], atol=2e-6, rtol=1e-5)
