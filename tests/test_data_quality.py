"""Assertions about the supplied data and the repairs applied to it.

Each defect the cleaner handles gets a test that fails if the fix regresses, and
each *non*-defect gets a test pinning why it is left alone. The counts are the
observed ones; if a future data drop changes them the tests should be re-derived,
not loosened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.cleaning import DataCleaner, MarketIndexSeries
from freight_rate.geo import CityReference, InconsistentCityCoordinates, haversine_miles
from freight_rate.outliers import TargetOutlierDetector
from freight_rate.schema import (
    EXPECTED_DECEMBER_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALIDATION_ROWS,
)


# -- the supplied files are what we think they are --------------------------


def test_row_counts(train, validation, december):
    assert len(train) == EXPECTED_TRAIN_ROWS
    assert len(validation) == EXPECTED_VALIDATION_ROWS
    assert len(december) == EXPECTED_DECEMBER_ROWS


def test_development_and_scored_windows_do_not_overlap(train, validation):
    assert train["date"].max() < validation["date"].min(), (
        "the scored window must be strictly in the future for temporal validation to mean anything"
    )
    assert train["date"].min() == pd.Timestamp("2025-01-01")
    assert train["date"].max() == pd.Timestamp("2025-10-31")
    assert validation["date"].min() == pd.Timestamp("2025-11-01")
    assert validation["date"].max() == pd.Timestamp("2025-12-31")


def test_no_load_id_overlap(train, validation):
    assert not set(train["load_id"]) & set(validation["load_id"])


def test_december_lane_is_represented_in_the_training_data(train, december):
    """The chart lane must not be an unseen extrapolation."""
    lane = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")]
    assert len(lane) > 0, "Lexington -> Fort Wayne never appears in training"
    assert np.isclose(lane["distance"].median(), december["distance"].iloc[0], rtol=0.1), (
        "the chart distance is far from the observed distance for that lane"
    )


# -- known defects ----------------------------------------------------------


def test_negative_weights_exist_and_are_sign_errors(train):
    negative = train[train["weight"] < 0]
    assert len(negative) == 292
    # If these were sentinels rather than sign flips, |weight| would not sit in
    # the same range as the valid rows.
    positive = train[train["weight"] > 0]
    assert 0.9 < negative["weight"].abs().mean() / positive["weight"].mean() < 1.1


def test_missing_values_are_present_in_both_frames(train, validation):
    assert train["weight"].isna().sum() == 300
    assert train["market_index"].isna().sum() == 374
    assert validation["weight"].isna().sum() == 165
    assert validation["market_index"].isna().sum() == 249, (
        "missing values appear at inference too, so rows must be imputed, not dropped"
    )


def test_cleaner_removes_all_missing_and_negative_weights(train, validation):
    cleaned = DataCleaner().fit(train, coordinate_frames=(validation,)).transform(train)
    assert cleaned["weight"].notna().all()
    assert (cleaned["weight"] > 0).all()
    assert cleaned["weight_was_negative"].sum() == 292
    assert cleaned["weight_was_missing"].sum() == 300


def test_negative_weights_keep_their_magnitude(train, validation):
    """The 292 sign-flipped rows must be corrected by abs(), not discarded.

    This is the assertion the test above cannot make. Dropping ``abs()`` turns
    those weights into NaN, which the imputer then fills with an equipment
    median — leaving the column just as non-null and just as positive, and the
    flags untouched. Only comparing the *value* distinguishes "corrected the
    sign" from "threw the measurement away and guessed".

    The distinction is real: the true magnitudes run 20,003 / 36,559 / 41,397 lb
    while every equipment median sits near 31,500, so imputing would corrupt 292
    genuine measurements. Found by ``scripts/verify_guards.py``.
    """
    cleaner = DataCleaner().fit(train, coordinate_frames=(validation,))
    cleaned = cleaner.transform(train)
    negative = (train["weight"] < 0).to_numpy()
    assert negative.sum() == 292

    expected = train.loc[negative, "weight"].abs().clip(*cleaner.weight_bounds)
    np.testing.assert_allclose(cleaned.loc[negative, "weight"].to_numpy(), expected.to_numpy())

    # And confirm the imputer was not what produced them.
    medians = set(cleaner.weight_median_by_equipment.values())
    assert not medians & set(cleaned.loc[negative, "weight"].to_numpy().tolist())


def test_cleaner_fills_market_index_everywhere(train, validation):
    cleaner = DataCleaner().fit(train, coordinate_frames=(validation,), index_frames=(validation,))
    for frame in (train, validation):
        cleaned = cleaner.transform(frame)
        assert cleaned["market_index"].notna().all()
        assert cleaned["market_index_daily"].notna().all()


def test_cleaner_supplies_coordinates_for_december(train, validation, december):
    """December ships city names only; the reference table must complete them."""
    cleaner = DataCleaner().fit(train, coordinate_frames=(validation,), index_frames=(validation,))
    cleaned = cleaner.transform(december)
    for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        assert cleaned[column].notna().all(), f"{column} still missing for December"
    assert cleaned["market_index"].notna().all()


def test_cleaner_does_not_mutate_its_input(train, validation):
    before = train.copy(deep=True)
    DataCleaner().fit(train, coordinate_frames=(validation,)).transform(train)
    pd.testing.assert_frame_equal(train, before)


# -- non-defects, pinned so they are not "fixed" later ----------------------


def test_distance_is_never_repaired(train, validation):
    """Rows with a high distance/great-circle ratio are short hauls on a coarse
    coordinate grid, not corrupt mileage — repairing them roughly doubles their
    error. The cleaner must leave ``distance`` alone."""
    cleaner = DataCleaner().fit(train, coordinate_frames=(validation,))
    cleaned = cleaner.transform(train)
    original = pd.to_numeric(train["distance"])
    pd.testing.assert_series_equal(
        cleaned["distance"], original, check_names=False, check_dtype=False
    )


def test_high_circuity_rows_are_short_hauls(train):
    great_circle = haversine_miles(
        train["pickup_lat"], train["pickup_lon"], train["delivery_lat"], train["delivery_lon"]
    )
    ratio = train["distance"] / great_circle
    suspicious = train[ratio > 1.6]
    assert len(suspicious) == 111
    assert suspicious["distance"].max() < 200, (
        "the high-ratio rows are supposed to be short hauls; if long hauls appear "
        "the coarse-coordinate explanation no longer holds"
    )


def test_distance_floor_is_a_billing_rule(train, validation):
    assert train["distance"].min() == 70.0
    assert validation["distance"].min() == 70.0
    assert (train["distance"] == 70.0).sum() > 10, "a hard floor, not a coincidence"


def test_quote_signal_is_uninformative(train):
    """Documents why the feature is excluded by default."""
    correlation = np.corrcoef(train["quote_signal"], train["posted_rate"])[0, 1]
    assert abs(correlation) < 0.1, (
        f"quote_signal correlates {correlation:.3f} with the target; if that ever "
        "rises materially, revisit FeatureConfig.use_quote_signal"
    )


# -- target contamination ---------------------------------------------------


def test_target_contamination_is_detected(train):
    report = TargetOutlierDetector().fit(train).report_
    assert 0.01 < report.flagged_fraction < 0.02, "expected roughly 1.4% corrupted labels"
    assert report.n_inflated > 250 and report.n_deflated > 250, "corruption runs both ways"
    assert report.residual_std_after < 0.5 * report.residual_std_before, (
        "removing the flagged rows should more than halve the residual spread"
    )


def test_contamination_boundary_sits_in_an_empty_band(train):
    """The clean and corrupted populations are separated by a gap, which is what
    makes the threshold choice insensitive."""
    detector = TargetOutlierDetector().fit(train)
    residual = np.log(train["posted_rate"]) - (
        detector.intercept_ + detector.slope_ * np.log(train["distance"])
    )
    deviation = (residual - detector.centre_).abs()
    in_gap = ((deviation > 0.30) & (deviation < 0.38)).sum()
    assert in_gap == 0, f"{in_gap} rows sit inside the expected empty band"


def test_contamination_is_not_predictable_from_features(train):
    """If a feature predicted contamination we could model it instead of dropping it."""
    flagged = TargetOutlierDetector().fit(train).flag(train)
    share = flagged.groupby(train["equipment"]).mean()
    assert share.max() - share.min() < 0.01, (
        "contamination rate differs materially by equipment, so it may not be random"
    )


def test_detector_threshold_is_stable_across_reasonable_sigmas(train):
    from freight_rate.config import OutlierConfig

    counts = {
        sigma: TargetOutlierDetector(OutlierConfig(n_sigma=sigma)).fit(train).report_.n_flagged
        for sigma in (4.0, 5.0, 6.0)
    }
    assert max(counts.values()) - min(counts.values()) < 20, (
        f"flagged count is sensitive to the threshold: {counts}"
    )


# -- geography --------------------------------------------------------------


def test_every_city_has_exactly_one_coordinate(train, validation):
    reference = CityReference.from_frames(train, validation)
    assert len(reference) == 72
    assert len(CityReference.from_frames(train)) == 64
    assert "Lexington" in reference and "Fort Wayne" in reference


def test_the_scored_set_introduces_cities_never_seen_in_training(train, validation):
    """A cold-start case worth stating plainly: 8 cities appear only in the
    scored data, touching 12.1% of the rows that must be predicted. City
    categoricals cannot help there, which is why the coordinate features — which
    are available for every city — are kept rather than pruned."""
    trained = set(train["pickup"]) | set(train["delivery"])
    scored = set(validation["pickup"]) | set(validation["delivery"])
    unseen = scored - trained
    assert unseen == {
        "Allentown",
        "Charlotte",
        "Chicago",
        "Jackson",
        "Knoxville",
        "Laredo",
        "Norfolk",
        "San Diego",
    }
    touched = validation["pickup"].isin(unseen) | validation["delivery"].isin(unseen)
    assert 0.10 < touched.mean() < 0.15
    assert not (validation["pickup"].isin(unseen) & validation["delivery"].isin(unseen)).any(), (
        "no load has an unseen city at both ends, so one end always anchors the lane"
    )


def test_unseen_cities_still_receive_coordinates(train, validation):
    """The reference table is built from every frame that carries coordinates,
    so a training-only table would silently leave 12% of the scored rows without
    geography."""
    reference = CityReference.from_frames(train, validation)
    attached = reference.attach(validation.drop(columns=["pickup_lat", "pickup_lon"]))
    assert attached["pickup_lat"].notna().all()
    assert attached["pickup_lon"].notna().all()


def test_inconsistent_coordinates_are_rejected(train):
    corrupted = train.head(200).copy()
    corrupted.loc[corrupted.index[0], "pickup_lat"] = 0.0
    with pytest.raises(InconsistentCityCoordinates):
        CityReference.from_frames(corrupted)


def test_haversine_matches_reported_distance_closely(train):
    great_circle = haversine_miles(
        train["pickup_lat"], train["pickup_lon"], train["delivery_lat"], train["delivery_lon"]
    )
    assert np.corrcoef(great_circle, train["distance"])[0, 1] > 0.99


def test_haversine_is_zero_for_identical_points():
    assert haversine_miles([40.0], [-80.0], [40.0], [-80.0])[0] == pytest.approx(0.0)


def test_haversine_is_symmetric():
    forward = haversine_miles([40.0], [-80.0], [34.0], [-118.0])
    backward = haversine_miles([34.0], [-118.0], [40.0], [-80.0])
    assert forward[0] == pytest.approx(backward[0])


def test_market_index_series_covers_every_december_date(train, validation, december):
    series = MarketIndexSeries.from_frames(train, validation)
    values = series.value_for(december["date"])
    assert values.notna().all()
    assert (values > 0).all()
    assert values.nunique() > 1, "December must not collapse to a single index value"
