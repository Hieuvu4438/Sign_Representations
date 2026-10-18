import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from asm_jepa.config import load_config  # noqa: E402
from asm_jepa.features.stats import identity_stats  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "pretrain"


def cfg_for(name: str, *sets):
    return load_config(CONFIGS / f"{name}.yaml", "pretrain", list(sets))


@pytest.fixture
def full_cfg():
    return cfg_for("full")


@pytest.fixture
def stats():
    return identity_stats(3)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
