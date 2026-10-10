import pytest
import torch

from signcoord.models.generic_hiera_adapter import encoder_state_from_mae


def test_generic_mae_boundary_and_temporal_only_adaptation():
    original = {"pos_embed_temporal": torch.arange(8*96).reshape(1,8,96).float(),
                "blocks.0.weight": torch.ones(2,2), "decoder_pred.weight": torch.ones(2,2)}
    expected = {"pos_embed_temporal": torch.zeros(1,16,96), "blocks.0.weight": torch.zeros(2,2)}
    state, excluded, adapted = encoder_state_from_mae(original, expected, 16)
    assert state["pos_embed_temporal"].shape == (1,16,96)
    assert state["blocks.0.weight"] is original["blocks.0.weight"]
    assert excluded == ["decoder_pred.weight"] and adapted == ["pos_embed_temporal"]
    for invalid in ({**original, "classifier.weight": torch.ones(1)},
                    {k:v for k,v in original.items() if k != "blocks.0.weight"}):
        with pytest.raises(ValueError): encoder_state_from_mae(invalid, expected, 16)


def test_non_official_temporal_geometry_rejected():
    with pytest.raises(ValueError, match="T16"):
        encoder_state_from_mae({"pos_embed_temporal":torch.zeros(1,4,96)},
                              {"pos_embed_temporal":torch.zeros(1,16,96)},16)
