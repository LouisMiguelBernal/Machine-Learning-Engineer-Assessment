"""Shared fixtures.

Real-data fixtures are session-scoped: reading 6 MB of CSV once and fitting the
pipeline once keeps the suite fast enough to run on every change, which is the
only way a test suite actually gets run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from freight_rate import load_december_inputs, load_template, load_train, load_validation  # noqa: E402
from freight_rate.config import PATHS, RunConfig  # noqa: E402
from freight_rate.pipeline import FreightRatePipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutations  # noqa: E402


def pytest_configure(config) -> None:  # noqa: ARG001 - pytest hook signature
    """Inject a deliberate defect when FREIGHT_MUTATION is set.

    Used only by ``scripts/verify_guards.py`` to prove each regression test
    actually fails when the thing it guards is removed. A plain pytest run does
    not set the variable, so this is a no-op.
    """
    mutation = mutations.apply_if_requested()
    if mutation is not None:
        print(f"\n[MUTATION ACTIVE] {mutation.name}: removed {mutation.breaks}")


@pytest.fixture(scope="session")
def paths():
    return PATHS


@pytest.fixture(scope="session")
def train() -> pd.DataFrame:
    return load_train()


@pytest.fixture(scope="session")
def validation() -> pd.DataFrame:
    return load_validation()


@pytest.fixture(scope="session")
def template() -> pd.DataFrame:
    return load_template()


@pytest.fixture(scope="session")
def december() -> pd.DataFrame:
    return load_december_inputs()


@pytest.fixture(scope="session")
def small_train(train: pd.DataFrame) -> pd.DataFrame:
    """A fast stratified-by-date slice for pipeline tests that do not need scale."""
    return train.groupby(train["date"].dt.to_period("W"), observed=True).head(30).reset_index(
        drop=True
    )


@pytest.fixture(scope="session")
def fitted_pipeline(small_train, validation) -> FreightRatePipeline:
    return FreightRatePipeline(RunConfig()).fit(small_train, context_frames=(validation,))


@pytest.fixture
def synthetic_loads() -> pd.DataFrame:
    """A tiny hand-built frame with a known answer, for unit-level assertions."""
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "load_id": [f"TR-{i:06d}" for i in range(1, 13)],
            "pickup": ["Dallas", "Austin"] * 6,
            "delivery": ["Austin", "Dallas"] * 6,
            "pickup_lat": [31.83025, 30.0] * 6,
            "pickup_lon": [-94.38343, -95.0] * 6,
            "delivery_lat": [30.0, 31.83025] * 6,
            "delivery_lon": [-95.0, -94.38343] * 6,
            "distance": np.linspace(200.0, 1800.0, 12),
            "equipment": ["Dry Van", "Reefer", "Flatbed"] * 4,
            "weight": np.linspace(15_000.0, 45_000.0, 12),
            "date": dates,
            "market_index": np.linspace(0.9, 1.2, 12),
            "quote_signal": np.linspace(1.8, 2.4, 12),
            "posted_rate": np.linspace(200.0, 1800.0, 12) * 2.2,
        }
    )
