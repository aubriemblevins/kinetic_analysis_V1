import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXAMPLE_DATA = ROOT / "examples" / "Example_data.csv"
EXAMPLE_LAYOUT = ROOT / "examples" / "example_layout.yaml"


@pytest.fixture(scope="session")
def example_data():
    from enzkin import read_kinetics
    return read_kinetics(EXAMPLE_DATA)


@pytest.fixture(scope="session")
def example_map():
    from enzkin import read_map
    return read_map(EXAMPLE_LAYOUT)


@pytest.fixture(scope="session")
def example_result(example_data, example_map):
    from enzkin import analyse
    return analyse(example_data, example_map)


@pytest.fixture
def straight_curve():
    """A clean line: 20 RFU/min with light noise, sampled every minute."""
    rng = np.random.default_rng(1)
    time = np.arange(0, 61) * 60.0
    signal = 500 + 20.0 * (time / 60) + rng.normal(0, 4, time.size)
    return time, signal, 20.0
