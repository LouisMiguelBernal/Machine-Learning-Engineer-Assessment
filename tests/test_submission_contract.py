"""The vendor submission contract, asserted independently of ``score.py``.

These re-derive every rule in the supplied scorer from the assessment text rather
than importing its functions, so a mistake in one is not silently mirrored by the
other. They are the tests that decide whether the submission can be sent.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = REPO_ROOT / "validation_predictions.csv"
DECEMBER = REPO_ROOT / "data" / "december_chart_inputs.csv"
VENDOR_SCORE = REPO_ROOT / ".assessment_original" / "score.py.txt"

EXPECTED_IDS = {f"TE-{i:06d}" for i in range(1, 12_001)}
DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")
DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]

pytestmark = pytest.mark.contract

REQUIRE = os.environ.get("FREIGHT_REQUIRE_SUBMISSION", "") not in ("", "0", "false")
"""Set FREIGHT_REQUIRE_SUBMISSION=1 (the release gate does) to turn every
"not built yet" skip in this module into a failure. Without it a fresh clone
reports a green pytest before the deliverables exist, which is not a result."""


def _not_built(reason: str) -> None:
    if REQUIRE:
        pytest.fail(reason + " (FREIGHT_REQUIRE_SUBMISSION is set)")
    pytest.skip(reason)


needs_submission = pytest.mark.skipif(
    not REQUIRE and not SUBMISSION.is_file(),
    reason="validation_predictions.csv not built yet — run scripts/predict.py",
)


@pytest.fixture(scope="module")
def submission() -> pd.DataFrame:
    if not SUBMISSION.is_file():
        _not_built("validation_predictions.csv not built yet — run scripts/predict.py")
    return pd.read_csv(SUBMISSION)


@pytest.fixture(scope="module")
def december_output() -> pd.DataFrame:
    return pd.read_csv(DECEMBER)


# -- validation_predictions.csv --------------------------------------------


@needs_submission
def test_submission_columns_exact_and_ordered(submission):
    assert list(submission.columns) == ["load_id", "predicted_rate"]


@needs_submission
def test_submission_row_count(submission):
    assert len(submission) == 12_000


@needs_submission
def test_submission_ids_match_the_scored_set_exactly(submission):
    ids = set(submission["load_id"].astype(str))
    assert not (EXPECTED_IDS - ids), "missing load_id values"
    assert not (ids - EXPECTED_IDS), "unexpected load_id values"


@needs_submission
def test_submission_ids_unique_and_present(submission):
    assert not submission["load_id"].isna().any()
    assert not submission["load_id"].duplicated().any()


@needs_submission
def test_submission_rates_are_finite_and_strictly_positive(submission):
    rates = pd.to_numeric(submission["predicted_rate"], errors="coerce")
    assert rates.notna().all(), "non-numeric predicted_rate"
    assert np.isfinite(rates).all()
    assert (rates > 0).all(), f"{(rates <= 0).sum()} non-positive rates"


@needs_submission
def test_submission_rates_are_economically_plausible(submission):
    """A guard against a broken back-transform that still satisfies the scorer.

    The training labels span $57 to $25,533. Predictions far outside that band
    would pass the vendor checks while being obvious nonsense.
    """
    rates = pd.to_numeric(submission["predicted_rate"])
    assert rates.min() > 50, f"implausibly low minimum {rates.min():,.2f}"
    assert rates.max() < 60_000, f"implausibly high maximum {rates.max():,.2f}"
    assert 500 < rates.median() < 10_000, f"implausible median {rates.median():,.2f}"


# -- december_chart_inputs.csv ---------------------------------------------


def test_december_keeps_original_columns_in_order(december_output):
    assert list(december_output.columns) == DECEMBER_COLUMNS


def test_december_has_one_row_per_day(december_output):
    dates = pd.to_datetime(december_output["date"])
    assert len(december_output) == 31
    assert not dates.duplicated().any()
    assert set(dates) == set(DECEMBER_DATES)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("pickup", "Lexington"),
        ("delivery", "Fort Wayne"),
        ("equipment", "Dry Van"),
    ],
)
def test_december_fixed_text_inputs_unchanged(december_output, column, expected):
    assert december_output[column].eq(expected).all()


@pytest.mark.parametrize(("column", "expected"), [("distance", 360.0), ("weight", 32_000.0)])
def test_december_fixed_numeric_inputs_unchanged(december_output, column, expected):
    assert np.isclose(pd.to_numeric(december_output[column]), expected).all()


def test_december_rates_are_positive_when_filled(december_output):
    rates = pd.to_numeric(december_output["predicted_rate"], errors="coerce")
    if rates.isna().all():
        _not_built("December predictions not built yet — run scripts/predict.py")
    assert rates.notna().all(), "December predictions partially filled"
    assert (rates > 0).all()


def test_december_curve_is_not_flat(december_output):
    """The chart exists to show the model responds to date. A flat line means the
    date features are not reaching the prediction."""
    rates = pd.to_numeric(december_output["predicted_rate"], errors="coerce")
    if rates.isna().all():
        _not_built("December predictions not built yet")
    assert rates.nunique() > 1, "every December day predicts the same rate"
    assert rates.std() > 0.0


def test_december_curve_has_no_implausible_jumps(december_output):
    """Day-to-day freight rates on one fixed lane do not double overnight."""
    rates = pd.to_numeric(december_output["predicted_rate"], errors="coerce")
    if rates.isna().all():
        _not_built("December predictions not built yet")
    ordered = rates.loc[pd.to_datetime(december_output["date"]).sort_values().index]
    relative_step = ordered.diff().abs() / ordered.shift()
    assert relative_step.max(skipna=True) < 0.35, (
        f"largest day-over-day move is {relative_step.max():.1%}"
    )


# -- the scorer itself ------------------------------------------------------

# Recorded from the files as supplied with the assessment. Pinned as constants
# rather than compared against .assessment_original/, because that directory is
# git-ignored and would be absent from a fresh clone — leaving the most important
# check in this file silently unrunnable exactly where it matters most.
VENDOR_SHA256 = {
    "score.py": "8da8b6bf3e39078f34c9998d14dcca4f7aa64ad5ddea1ce4449cd362168d2a28",
    "requirements.txt": "38523b081dfb40d36c12b1af29e6029023dbf0ceab7c55bd01ced7be3cb0f9ac",
}


@pytest.mark.parametrize("filename", sorted(VENDOR_SHA256))
def test_vendor_file_is_unmodified(filename):
    """A test that passes only because the scorer was edited is worthless."""
    digest = hashlib.sha256((REPO_ROOT / filename).read_bytes()).hexdigest()
    assert digest == VENDOR_SHA256[filename], (
        f"{filename} differs from the copy supplied with the assessment "
        f"(sha256 {digest})"
    )


@pytest.mark.skipif(
    not (REPO_ROOT / ".assessment_original").is_dir(),
    reason="raw email attachments not present (they are git-ignored)",
)
@pytest.mark.parametrize(
    ("filename", "original"),
    [("score.py", "score.py.txt"), ("requirements.txt", "requirements.txt.txt")],
)
def test_vendor_file_matches_the_raw_attachment(filename, original):
    """Belt and braces where the raw attachments are still on disk."""
    assert (REPO_ROOT / filename).read_bytes() == (
        REPO_ROOT / ".assessment_original" / original
    ).read_bytes()
