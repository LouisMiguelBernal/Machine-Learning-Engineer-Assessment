"""Feature construction, target transform, level model and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from freight_rate.cleaning import DataCleaner
from freight_rate.config import FeatureConfig, ModelConfig
from freight_rate.features import (
    CATEGORICAL_FEATURES,
    CITY_COLUMNS,
    FORBIDDEN_FEATURES,
    FeatureBuilder,
    apply_city_dropout,
)
from freight_rate.level import MarketLevelModel
from freight_rate.model import (
    EXCLUDED_FROM_ESTIMATOR,
    DistanceBaseline,
    LogTargetRegressor,
    RatePerMileBaseline,
    build_model,
)
from freight_rate import metrics


# -- features ---------------------------------------------------------------


@pytest.fixture
def cleaned(synthetic_loads) -> pd.DataFrame:
    return DataCleaner().fit(synthetic_loads).transform(synthetic_loads)


@pytest.fixture
def level(cleaned) -> MarketLevelModel:
    return MarketLevelModel().fit(
        cleaned["date"],
        cleaned["market_index_daily"],
        np.log(cleaned["posted_rate"]) - np.log(cleaned["distance"]),
    )


def test_feature_matrix_never_contains_the_target(cleaned, level):
    builder = FeatureBuilder(level_model=level)
    features = builder.fit_transform(cleaned)
    assert not FORBIDDEN_FEATURES & set(features.columns)


def test_feature_matrix_row_order_is_preserved(cleaned, level):
    features = FeatureBuilder(level_model=level).fit_transform(cleaned)
    assert features.index.equals(cleaned.index)
    assert len(features) == len(cleaned)


def test_transform_is_deterministic(cleaned, level):
    builder = FeatureBuilder(level_model=level).fit(cleaned)
    pd.testing.assert_frame_equal(builder.transform(cleaned), builder.transform(cleaned))


def test_columns_are_stable_between_fit_and_transform(cleaned, level):
    builder = FeatureBuilder(level_model=level).fit(cleaned)
    assert list(builder.transform(cleaned).columns) == builder.columns_


def test_quote_signal_is_excluded_by_default(cleaned, level):
    default = FeatureBuilder(level_model=level).fit_transform(cleaned)
    assert "quote_signal" not in default.columns
    enabled = FeatureBuilder(
        FeatureConfig(use_quote_signal=True), level_model=level
    ).fit_transform(cleaned)
    assert "quote_signal" in enabled.columns


def test_no_linear_time_index_by_default(cleaned, level):
    """A tree cannot extrapolate a trend; the level model carries it instead."""
    features = FeatureBuilder(level_model=level).fit_transform(cleaned)
    assert "days_since_epoch" not in features.columns


def test_seasonality_is_encoded_cyclically(cleaned, level):
    features = FeatureBuilder(level_model=level).fit_transform(cleaned)
    assert {"day_of_year_sin", "day_of_year_cos"} <= set(features.columns)
    radius = features["day_of_year_sin"] ** 2 + features["day_of_year_cos"] ** 2
    np.testing.assert_allclose(radius, 1.0, atol=1e-9)


def test_category_levels_are_frozen_at_fit(cleaned, level):
    builder = FeatureBuilder(level_model=level).fit(cleaned, cities=["Dallas", "Austin", "Tulsa"])
    unseen = cleaned.copy()
    unseen["pickup"] = "Nowhere"
    transformed = builder.transform(unseen)
    assert list(transformed["pickup"].cat.categories) == ["Austin", "Dallas", "Tulsa"], (
        "an unseen city must not reshuffle the codes of the known ones"
    )
    assert transformed["pickup"].isna().all(), "an unseen city becomes a missing category"


def test_categorical_columns_are_reported(cleaned, level):
    builder = FeatureBuilder(level_model=level).fit(cleaned)
    assert set(builder.categorical_columns) <= set(CATEGORICAL_FEATURES)
    assert "equipment" in builder.categorical_columns


def test_transform_before_fit_is_an_error(cleaned, level):
    with pytest.raises(RuntimeError):
        FeatureBuilder(level_model=level).transform(cleaned)


def test_city_dropout_blanks_roughly_the_requested_share(cleaned, level):
    features = FeatureBuilder(level_model=level).fit_transform(cleaned)
    large = pd.concat([features] * 400, ignore_index=True)
    dropped = apply_city_dropout(large, rate=0.1, seed=1)
    for column in CITY_COLUMNS:
        assert 0.07 < dropped[column].isna().mean() < 0.13


def test_city_dropout_is_deterministic_and_non_mutating(cleaned, level):
    features = FeatureBuilder(level_model=level).fit_transform(cleaned)
    before = features.copy(deep=True)
    first = apply_city_dropout(features, rate=0.5, seed=4)
    second = apply_city_dropout(features, rate=0.5, seed=4)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(features, before), "input must not be mutated"


def test_city_dropout_of_zero_is_a_no_op(cleaned, level):
    features = FeatureBuilder(level_model=level).fit_transform(cleaned)
    pd.testing.assert_frame_equal(apply_city_dropout(features, rate=0.0, seed=1), features)


def test_city_dropout_rejects_an_out_of_range_rate(cleaned, level):
    features = FeatureBuilder(level_model=level).fit_transform(cleaned)
    with pytest.raises(ValueError, match="city_dropout_rate"):
        apply_city_dropout(features, rate=1.5, seed=1)


def test_city_dropout_touches_only_city_columns(cleaned, level):
    features = FeatureBuilder(level_model=level).fit_transform(cleaned)
    dropped = apply_city_dropout(features, rate=0.5, seed=2)
    for column in features.columns:
        if column not in CITY_COLUMNS:
            pd.testing.assert_series_equal(dropped[column], features[column])


def test_market_level_requires_a_fitted_level_model(cleaned):
    with pytest.raises(RuntimeError, match="MarketLevelModel"):
        FeatureBuilder(FeatureConfig(use_market_level=True)).fit(cleaned)


def test_december_and_training_produce_the_same_feature_columns(
    train, validation, december
):
    """The core train/serve skew guard: one feature path, identical columns."""
    cleaner = DataCleaner().fit(
        train, coordinate_frames=(validation,), index_frames=(validation,)
    )
    cleaned_train = cleaner.transform(train)
    level_model = MarketLevelModel().fit(
        cleaned_train["date"],
        cleaned_train["market_index_daily"],
        np.log(cleaned_train["posted_rate"]) - np.log(cleaned_train["distance"]),
    )
    builder = FeatureBuilder(level_model=level_model).fit(
        cleaned_train, cities=cleaner.city_reference.cities
    )
    for frame in (validation, december):
        features = builder.transform(cleaner.transform(frame))
        assert list(features.columns) == builder.columns_
        assert len(features) == len(frame)


def test_december_features_have_no_missing_numeric_values(train, validation, december):
    cleaner = DataCleaner().fit(
        train, coordinate_frames=(validation,), index_frames=(validation,)
    )
    cleaned_train = cleaner.transform(train)
    level_model = MarketLevelModel().fit(
        cleaned_train["date"],
        cleaned_train["market_index_daily"],
        np.log(cleaned_train["posted_rate"]) - np.log(cleaned_train["distance"]),
    )
    builder = FeatureBuilder(level_model=level_model).fit(
        cleaned_train, cities=cleaner.city_reference.cities
    )
    features = builder.transform(cleaner.transform(december))
    numeric = features.select_dtypes(include=[np.number])
    assert numeric.notna().all().all(), (
        f"December has missing features: "
        f"{numeric.columns[numeric.isna().any()].tolist()}"
    )


# -- market level -----------------------------------------------------------


def test_level_model_recovers_a_known_trend():
    dates = pd.Series(pd.date_range("2025-01-01", periods=200, freq="D"))
    market = pd.Series(np.full(200, 1.0))
    truth = 0.5 + 0.001 * np.arange(200)
    model = MarketLevelModel(use_market_index=False).fit(dates, market, pd.Series(truth))
    assert model.fit_.trend_per_day == pytest.approx(0.001, rel=1e-6)
    np.testing.assert_allclose(model.predict(dates, market), truth, atol=1e-9)


def test_level_model_extrapolates_beyond_the_fitted_range():
    dates = pd.Series(pd.date_range("2025-01-01", periods=100, freq="D"))
    market = pd.Series(np.full(100, 1.0))
    model = MarketLevelModel(use_market_index=False).fit(
        dates, market, pd.Series(0.5 + 0.001 * np.arange(100))
    )
    future = pd.Series([pd.Timestamp("2025-06-01")])
    expected = 0.5 + 0.001 * (pd.Timestamp("2025-06-01") - pd.Timestamp("2025-01-01")).days
    assert model.predict(future, pd.Series([1.0]))[0] == pytest.approx(expected, rel=1e-6)


def test_trend_damping_only_affects_extrapolation():
    dates = pd.Series(pd.date_range("2025-01-01", periods=100, freq="D"))
    market = pd.Series(np.full(100, 1.0))
    values = pd.Series(0.5 + 0.001 * np.arange(100))
    full = MarketLevelModel(use_market_index=False, trend_damping=1.0).fit(dates, market, values)
    damped = MarketLevelModel(use_market_index=False, trend_damping=0.0).fit(dates, market, values)

    np.testing.assert_allclose(
        full.predict(dates, market), damped.predict(dates, market), atol=1e-9
    )
    future = pd.Series([pd.Timestamp("2025-07-01")])
    assert damped.predict(future, pd.Series([1.0]))[0] < full.predict(future, pd.Series([1.0]))[0]
    assert damped.predict(future, pd.Series([1.0]))[0] == pytest.approx(
        full.predict(pd.Series([dates.iloc[-1]]), pd.Series([1.0]))[0]
    )


def test_level_model_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        MarketLevelModel().predict(pd.Series([pd.Timestamp("2025-01-01")]), pd.Series([1.0]))


# -- target transform -------------------------------------------------------


def test_log_target_round_trips_a_perfect_signal():
    frame = pd.DataFrame({"log_distance": np.log(np.linspace(100, 2000, 60))})
    rate = np.exp(frame["log_distance"]) * 2.0
    model = LogTargetRegressor(DummyRegressor(strategy="mean"), target_mode="log_rpm")
    model.fit(frame, rate)
    np.testing.assert_allclose(model.predict(frame), rate, rtol=1e-9)


def test_log_target_predictions_are_always_positive():
    frame = pd.DataFrame({"log_distance": np.log(np.linspace(100, 2000, 60))})
    rate = np.linspace(200, 4000, 60)
    model = LogTargetRegressor(DummyRegressor(strategy="mean")).fit(frame, rate)
    assert (model.predict(frame) > 0).all()


def test_log_target_rejects_non_positive_rates():
    frame = pd.DataFrame({"log_distance": np.log(np.linspace(100, 2000, 10))})
    with pytest.raises(ValueError, match="strictly positive"):
        LogTargetRegressor(DummyRegressor()).fit(frame, np.zeros(10))


def test_unknown_target_mode_is_rejected():
    with pytest.raises(ValueError, match="target_mode"):
        LogTargetRegressor(DummyRegressor(), target_mode="nonsense")


def test_missing_offset_column_is_reported_clearly():
    model = LogTargetRegressor(DummyRegressor(), target_mode="log_rpm")
    with pytest.raises(ValueError, match="log_distance"):
        model.fit(pd.DataFrame({"other": [1.0, 2.0]}), np.array([1.0, 2.0]))


def test_market_level_is_hidden_from_the_estimator():
    frame = pd.DataFrame(
        {
            "log_distance": np.log(np.linspace(100, 2000, 40)),
            "market_level": np.linspace(0.7, 0.8, 40),
            "weight": np.linspace(10_000, 40_000, 40),
        }
    )
    rate = np.exp(frame["log_distance"] + frame["market_level"])
    model = LogTargetRegressor(DummyRegressor(strategy="mean"), target_mode="log_rpm_level")
    model.fit(frame, rate)
    assert "market_level" in EXCLUDED_FROM_ESTIMATOR
    assert "market_level" not in model._estimator_features(frame).columns
    assert "log_distance" in model._estimator_features(frame).columns


def test_smearing_can_be_disabled():
    frame = pd.DataFrame({"log_distance": np.log(np.linspace(100, 2000, 40))})
    rate = np.linspace(200, 4000, 40)
    off = LogTargetRegressor(DummyRegressor(), smearing=False).fit(frame, rate)
    assert off.smearing_factor_ == 1.0
    on = LogTargetRegressor(DummyRegressor(), smearing=True).fit(frame, rate)
    assert on.smearing_factor_ >= 1.0, "smearing corrects a downward bias, so it cannot shrink"


def test_build_model_honours_the_configured_mode():
    model = build_model(ModelConfig(target_mode="log_rate"), ["equipment"])
    assert model.target_mode == "log_rate"
    assert model.offset_columns == ()


# -- baselines --------------------------------------------------------------


def test_distance_baseline_recovers_a_power_law():
    distance = np.linspace(100, 3000, 200)
    frame = pd.DataFrame({"distance": distance})
    rate = 12.0 * distance**0.87
    model = DistanceBaseline().fit(frame, rate)
    assert model.slope_ == pytest.approx(0.87, rel=1e-6)
    np.testing.assert_allclose(model.predict(frame), rate, rtol=1e-6)


def test_rate_per_mile_baseline_uses_the_median():
    frame = pd.DataFrame({"distance": np.array([100.0, 200.0, 400.0])})
    model = RatePerMileBaseline().fit(frame, np.array([200.0, 400.0, 1200.0]))
    assert model.rate_per_mile_ == pytest.approx(2.0)


# -- metrics ----------------------------------------------------------------


def test_metrics_are_zero_for_a_perfect_prediction():
    truth = np.array([100.0, 250.0, 900.0])
    result = metrics.compute(truth, truth)
    assert result.mae == 0 and result.rmse == 0 and result.mape == 0
    assert result.r2 == pytest.approx(1.0)


def test_metrics_match_hand_computed_values():
    result = metrics.compute([100.0, 200.0], [110.0, 180.0])
    assert result.mae == pytest.approx(15.0)
    assert result.rmse == pytest.approx(np.sqrt((100 + 400) / 2))
    assert result.mape == pytest.approx((10 / 100 + 20 / 200) / 2 * 100)
    assert result.bias == pytest.approx(-5.0)


def test_metrics_reject_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.compute([1.0, 2.0], [1.0])


def test_metrics_ignore_non_finite_pairs():
    result = metrics.compute([100.0, 200.0, np.nan], [110.0, 180.0, 5.0])
    assert result.n == 2


def test_stratified_covers_every_group():
    truth = np.array([100.0, 200.0, 300.0, 400.0])
    groups = pd.Series(["a", "a", "b", "b"])
    table = metrics.stratified(truth, truth * 1.1, groups, "equipment")
    assert list(table.index) == ["a", "b"]
    assert (table["mape"] - 10.0).abs().max() < 1e-9


def test_summarise_folds_appends_a_mean_row():
    table = metrics.summarise_folds(
        {"a": metrics.compute([100.0], [110.0]), "b": metrics.compute([200.0], [180.0])}
    )
    assert "mean" in table.index
    assert table.loc["mean", "mae"] == pytest.approx(15.0)
