import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ethanol_ai.paths import MODELS  # noqa: E402
from ethanol_ai import pretreatment, hydrolysis  # noqa: E402
from ethanol_ai.surrogate import PretreatmentSurrogate, HydrolysisSurrogate  # noqa: E402


@pytest.fixture(scope="session")
def pre_kinetics():
    return pretreatment.load_kinetics(MODELS / "pretreatment_kinetics.json")


@pytest.fixture(scope="session")
def hyd_kinetics():
    return hydrolysis.load_kinetics(MODELS / "hydrolysis_kinetics.json")


@pytest.fixture(scope="session")
def pre_surrogate():
    return PretreatmentSurrogate.load(MODELS / "pretreatment_surrogate.npz")


@pytest.fixture(scope="session")
def hyd_surrogate():
    return HydrolysisSurrogate.load(MODELS / "hydrolysis_surrogate.npz")
