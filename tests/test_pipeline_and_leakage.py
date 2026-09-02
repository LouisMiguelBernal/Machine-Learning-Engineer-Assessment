"""End-to-end pipeline behaviour, leakage audit and determinism.

The leakage tests are the ones worth reading twice. On a problem where the scored
window is strictly in the future, a leak does not announce itself — it just
returns a validation number that the real submission will not reproduce.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from freight_rate.cleaning import MarketIndexCoverageError
from freight_rate.config import CleaningConfig, FeatureConfig, OutlierConfig, RunConfig
from freight_rate.data import write_submission
from freight_rate.features import FORBIDDEN_FEATURES
from freight_rate.pipeline import FreightRatePipeline
from freight_rate.schema import TARGET
from freight_rate.validation import holdout_split


# -- basic behaviour --------------------------------------------------------


def test_pipeline_predicts_one_positive_rate_per_row(fitted_pipeline, validation):
    predicted = fitted_pipeline.predict(validation)
    assert predicted.shape == (len(validation),)
    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()


def test_pipeline_handles_the_december_frame(fitted_pipeline, december):
    predicted = fitted_pipeline.predict(december)
    assert predicted.shape == (31,)
    assert (predicted > 0).all()


def test_predict_before_fit_is_an_error(validation):
    with pytest.raises(RuntimeError):
        FreightRatePipeline().predict(validation)


def test_fit_summary_records_what_happened(fitted_pipeline):
    summary = fitted_pipeline.summary
    assert summary.n_rows_used <= summary.n_rows_supplied
    assert summary.n_features > 10
    assert "equipment" in summary.categorical_features
    assert summary.outliers is not None
    assert "level =" in summary.level


def test_pipeline_rejects_context_frames_carrying_the_target(small_train):
    with pytest.raises(ValueError, match="target column"):
        FreightRatePipeline().fit(small_train, context_frames=(small_train,))


def test_pipeline_does_not_mutate_the_frame_it_predicts(fitted_pipeline, december):
    before = december.copy(deep=True)
    fitted_pipeline.predict(december)
    pd.testing.assert_frame_equal(december, before)


# -- leakage ----------------------------------------------------------------


def test_no_target_derived_column_reaches_the_model(fitted_pipeline, validation):
    matrix = fitted_pipeline.build_matrix(validation)
    assert not FORBIDDEN_FEATURES & set(matrix.columns)


def test_the_scored_frame_carries_no_target_column(validation, december):
    assert TARGET not in validation.columns
    assert TARGET not in december.columns


def test_prediction_ignores_a_target_column_if_one_is_present(fitted_pipeline, small_train):
    """Defence in depth: a stray label column must not change the prediction."""
    without = small_train.drop(columns=[TARGET])
    with_target = small_train.copy()
    np.testing.assert_allclose(
        fitted_pipeline.predict(without), fitted_pipeline.predict(with_target)
    )


def test_predictions_are_unchanged_when_the_target_is_scrambled(fitted_pipeline, small_train):
    scrambled = small_train.copy()
    scrambled[TARGET] = scrambled[TARGET].sample(frac=1.0, random_state=0).to_numpy()
    np.testing.assert_allclose(
        fitted_pipeline.predict(small_train.drop(columns=[TARGET])),
        fitted_pipeline.predict(scrambled),
    )


def test_a_fold_model_never_sees_future_rows(train, validation):
    """The end-to-end guarantee: fitting on a past window and predicting a future
    one must not reach across the boundary."""
    split = holdout_split(train["date"], 61)
    past = train.iloc[split.train_index].reset_index(drop=True)
    future = train.iloc[split.eval_index].reset_index(drop=True)

    pipeline = FreightRatePipeline().fit(past, context_frames=(validation,))
    assert pipeline.summary.train_end < future["date"].min()
    assert pipeline.level_model.fit_.fit_end < future["date"].min(), (
        "the level model must be fitted only on the training window"
    )


def test_evaluation_window_receives_its_own_market_index(train, validation):
    """The defect this pins: a holdout window whose dates are absent from the
    daily index used to receive one constant for every day, so the market
    level was evaluated as ``trend + constant`` while deployment runs
    ``trend + real index``. Measured cost of the mismatch: holdout MAE
    $38.94 reported against $48.15 actual for the same configuration.

    Fitting with the window's own inputs (never its label) as context must
    reproduce the window's real daily index in the feature matrix."""
    split = holdout_split(train["date"], 61)
    past = train.iloc[split.train_index].reset_index(drop=True)
    future = train.iloc[split.eval_index].reset_index(drop=True)
    inputs = future.drop(columns=[TARGET])

    pipeline = FreightRatePipeline().fit(past, context_frames=(validation, inputs))
    matrix = pipeline.build_matrix(future)
    expected = future.groupby("date")["market_index"].mean()

    assert matrix["market_index_daily"].nunique() > 1, "the window is priced at one constant index"
    np.testing.assert_allclose(
        matrix["market_index_daily"].to_numpy(),
        expected.reindex(future["date"]).to_numpy(),
    )
    assert pipeline.cleaner.market_index.last_fallback_share == 0.0


def test_pipeline_refuses_a_window_without_index_coverage(train, validation):
    """Without the window's inputs as context, prediction must raise rather
    than silently price 61 days at the global median."""
    split = holdout_split(train["date"], 61)
    past = train.iloc[split.train_index].reset_index(drop=True)
    future = train.iloc[split.eval_index].reset_index(drop=True)

    pipeline = FreightRatePipeline().fit(past, context_frames=(validation,))
    with pytest.raises(MarketIndexCoverageError, match="daily market_index"):
        pipeline.predict(future)


def test_index_fallback_median_comes_from_the_training_frame(train, validation):
    """The imputation constant must not move with whatever context was supplied."""
    split = holdout_split(train["date"], 61)
    past = train.iloc[split.train_index].reset_index(drop=True)
    future = train.iloc[split.eval_index].drop(columns=[TARGET])
    alone = FreightRatePipeline().fit(past, context_frames=())
    with_context = FreightRatePipeline().fit(past, context_frames=(validation, future))
    assert alone.cleaner.market_index.fallback == with_context.cleaner.market_index.fallback


def test_cleaner_statistics_come_only_from_the_training_rows(train, validation):
    """Imputation constants must not shift when the scored frame changes."""
    subset = train.head(5_000)
    first = FreightRatePipeline().fit(subset, context_frames=(validation,))
    second = FreightRatePipeline().fit(subset, context_frames=())
    assert first.cleaner.weight_median_global == second.cleaner.weight_median_global
    assert first.cleaner.weight_median_by_equipment == second.cleaner.weight_median_by_equipment


def test_outlier_detection_is_disabled_by_config(small_train, validation):
    config = RunConfig(outliers=OutlierConfig(enabled=False))
    pipeline = FreightRatePipeline(config).fit(small_train, context_frames=(validation,))
    assert pipeline.summary.n_rows_used == pipeline.summary.n_rows_supplied


# -- determinism ------------------------------------------------------------


def test_two_fits_give_identical_predictions(small_train, validation):
    first = FreightRatePipeline().fit(small_train, context_frames=(validation,))
    second = FreightRatePipeline().fit(small_train, context_frames=(validation,))
    np.testing.assert_array_equal(first.predict(validation), second.predict(validation))


def test_written_submission_is_byte_identical_across_runs(fitted_pipeline, validation, template, tmp_path):
    predicted = fitted_pipeline.predict(validation)
    digests = []
    for name in ("first.csv", "second.csv"):
        path = write_submission(predicted, template, tmp_path / name)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert digests[0] == digests[1]


def test_prediction_does_not_depend_on_row_order(fitted_pipeline, validation):
    shuffled = validation.sample(frac=1.0, random_state=3)
    reference = pd.Series(fitted_pipeline.predict(validation), index=validation.index)
    reordered = pd.Series(fitted_pipeline.predict(shuffled), index=shuffled.index)
    np.testing.assert_allclose(reference.to_numpy(), reordered.reindex(validation.index).to_numpy())


# -- persistence ------------------------------------------------------------


def test_saved_and_reloaded_pipeline_predicts_identically(fitted_pipeline, december, tmp_path):
    path = fitted_pipeline.save(tmp_path / "model.joblib")
    reloaded = FreightRatePipeline.load(path)
    np.testing.assert_allclose(fitted_pipeline.predict(december), reloaded.predict(december))


def test_loading_a_foreign_artefact_is_rejected(tmp_path):
    import joblib

    path = tmp_path / "not_a_pipeline.joblib"
    joblib.dump({"nope": True}, path)
    with pytest.raises(TypeError):
        FreightRatePipeline.load(path)


# -- submission writer ------------------------------------------------------


def test_writer_refuses_non_positive_rates(template, tmp_path):
    values = np.ones(len(template))
    values[0] = -1.0
    with pytest.raises(ValueError, match="non-positive"):
        write_submission(values, template, tmp_path / "bad.csv")


def test_writer_refuses_non_finite_rates(template, tmp_path):
    values = np.ones(len(template))
    values[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_submission(values, template, tmp_path / "bad.csv")


def test_writer_refuses_a_length_mismatch(template, tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        write_submission(np.ones(10), template, tmp_path / "bad.csv")


def test_writer_preserves_template_order(fitted_pipeline, validation, template, tmp_path):
    predicted = fitted_pipeline.predict(validation)
    path = write_submission(predicted, template, tmp_path / "out.csv")
    written = pd.read_csv(path)
    assert written["load_id"].tolist() == template["load_id"].tolist()


# -- sanity of the December curve -------------------------------------------


def test_december_responds_to_date_only(fitted_pipeline, december):
    """Every input except the date is constant, so any variation must come from
    the calendar and market-level features."""
    predicted = fitted_pipeline.predict(december)
    assert np.ptp(predicted) > 0, "the model is ignoring the date entirely"
    assert np.ptp(predicted) / predicted.mean() < 0.6, (
        "December swings more than 60% on a fixed lane, which is not credible"
    )


def test_december_rate_is_in_a_sane_band_for_a_360_mile_dry_van(fitted_pipeline, december, train):
    predicted = fitted_pipeline.predict(december)
    lane = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")]
    observed = lane["posted_rate"].median()
    assert 0.5 * observed < predicted.mean() < 2.0 * observed, (
        f"December mean {predicted.mean():,.0f} is far from the observed lane "
        f"median {observed:,.0f}"
    )


def test_unseen_cities_still_get_a_plausible_prediction(fitted_pipeline, train, validation):
    """12% of scored rows involve a city with no training history. Those rows
    must still price sensibly off distance, equipment and geography."""
    trained = set(train["pickup"]) | set(train["delivery"])
    unseen = (set(validation["pickup"]) | set(validation["delivery"])) - trained
    cold = validation[validation["pickup"].isin(unseen) | validation["delivery"].isin(unseen)]
    warm = validation[~validation.index.isin(cold.index)]
    assert len(cold) > 1_000

    cold_predicted = fitted_pipeline.predict(cold)
    warm_predicted = fitted_pipeline.predict(warm)
    assert np.isfinite(cold_predicted).all() and (cold_predicted > 0).all()

    cold_rpm = np.median(cold_predicted / cold["distance"].to_numpy())
    warm_rpm = np.median(warm_predicted / warm["distance"].to_numpy())
    assert 0.6 < cold_rpm / warm_rpm < 1.6, (
        f"cold-start rate-per-mile {cold_rpm:.3f} is far from the warm-start "
        f"{warm_rpm:.3f}; the unseen-category path is likely broken"
    )


def test_config_downgrades_the_target_mode_when_the_level_is_off():
    config = RunConfig(features=FeatureConfig(use_market_level=False))
    assert config.model.target_mode == "log_rpm"
    assert RunConfig().model.target_mode == "log_rpm_level"


def test_disabling_the_level_changes_december(small_train, validation, december):
    """Guards against the level silently not being wired in."""
    with_level = FreightRatePipeline().fit(small_train, context_frames=(validation,))
    without = FreightRatePipeline(
        RunConfig(features=FeatureConfig(use_market_level=False))
    ).fit(small_train, context_frames=(validation,))
    assert not np.allclose(with_level.predict(december), without.predict(december))
