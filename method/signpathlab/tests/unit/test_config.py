from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from signpath.config import ConfigError, load_config, validate

ROOT = Path(__file__).resolve().parents[2]


def test_configs_and_paths():
    base = load_config(ROOT / "configs/base.yaml")
    smoke = load_config(ROOT / "configs/smoke.yaml")
    assert smoke.root == ROOT
    assert smoke.values["model"]["encoder"]["layers"] == 2
    assert smoke.values["model"]["encoder"]["mlp_ratio"] == 4
    assert smoke.values["model"]["encoder"]["return_layers"] == [1, 2]
    assert base.hash != smoke.hash
    assert smoke.path("artifacts/smoke") == ROOT / "artifacts/smoke"
    for file in (ROOT / "configs/experiments").glob("*.yaml"):
        load_config(file)


@pytest.mark.parametrize("edit", [
    lambda c: c["model"].update(typo=True),
    lambda c: c["geometry"].update(degree=3),
    lambda c: c["data"].update(expose_labels_to_pretrain=True),
    lambda c: c["model"]["encoder"].update(heads=5),
    lambda c: c["cache"].update(detector_mode="VIDEO"),
    lambda c: c["runtime"].update(allow_random_pretrained_fallback=True),
    lambda c: c["model"]["predictor"].update(query_values="hidden_anchors"),
    lambda c: c["geometry"]["projection"].update(fit_statistics_split="val"),
    lambda c: c["training"].update(gradient_accumulation=0),
    lambda c: c["cache"].update(visual_dim=True),
    lambda c: c["training"].update(learning_rate=float("nan")),
])
def test_invalid_contracts(edit):
    config = deepcopy(load_config(ROOT / "configs/base.yaml").values)
    edit(config)
    with pytest.raises(ConfigError):
        validate(config)


def test_yaml_duplicate_and_cycles(tmp_path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("model:\n  heads: 4\n  heads: 5\n")
    with pytest.raises(ConfigError, match="Duplicate"):
        load_config(duplicate)
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\n")
    b.write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(a)


def test_packaged_schema_matches_base():
    assert yaml.safe_load((ROOT / "configs/base.yaml").read_text()) == yaml.safe_load((ROOT / "src/signpath/config_schema.yaml").read_text())


def test_original_spec_config_gets_optional_extension_defaults(tmp_path):
    config = deepcopy(load_config(ROOT / "configs/base.yaml").values)
    del config["downstream"]
    del config["cache"]["reader_lru_samples"]
    path = tmp_path / "proposal14.yaml"
    path.write_text(yaml.safe_dump(config))
    loaded = load_config(path, project_root=ROOT)
    assert loaded.values["downstream"]["head"] == "linear"
    assert loaded.values["cache"]["reader_lru_samples"] == 64
