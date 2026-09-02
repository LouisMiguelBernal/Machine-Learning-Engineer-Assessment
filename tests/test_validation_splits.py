"""Temporal split correctness.

The single most consequential thing that can silently go wrong in this project is
a split that lets a future row inform a past prediction. These tests make that
failure loud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.config import SplitConfig
from freight_rate.validation import (
    Split,
    TemporalLeakage,
    assert_no_temporal_leakage,
    build_splits,
    holdout_split,
    random_kfold_splits,
    rolling_origin_splits,
)


@pytest.fixture
def dates() -> pd.Series:
    return pd.Series(pd.date_range("2025-01-01", "2025-10-31", freq="D").repeat(4))


def test_holdout_is_the_final_window(dates):
    split = holdout_split(dates, holdout_days=61)
    assert split.eval_end == pd.Timestamp("2025-10-31")
    assert split.eval_start == pd.Timestamp("2025-09-01")
    assert split.train_end < split.eval_start


def test_holdout_length_matches_the_scored_window(train):
    """61 days, because the scored window is 2025-11-01..2025-12-31."""
    scored_days = (pd.Timestamp("2025-12-31") - pd.Timestamp("2025-11-01")).days + 1
    assert SplitConfig().holdout_days == scored_days
    split = holdout_split(train["date"], SplitConfig().holdout_days)
    span = (split.eval_end - split.eval_start).days + 1
    assert span == scored_days


def test_holdout_partitions_every_row(dates):
    split = holdout_split(dates, holdout_days=30)
    combined = np.concatenate([split.train_index, split.eval_index])
    assert sorted(combined.tolist()) == list(range(len(dates)))
    assert not set(split.train_index) & set(split.eval_index)


def test_holdout_rejects_a_degenerate_window(dates):
    with pytest.raises(ValueError):
        holdout_split(dates, holdout_days=10_000)


def test_rolling_folds_are_ordered_and_expanding(dates):
    folds = rolling_origin_splits(dates, n_splits=4, horizon_days=30, min_train_days=90)
    assert len(folds) == 4
    starts = [f.eval_start for f in folds]
    assert starts == sorted(starts), "folds must be returned oldest first"
    sizes = [len(f.train_index) for f in folds]
    assert sizes == sorted(sizes), "the training window must expand, not slide"


def test_every_rolling_fold_trains_strictly_in_the_past(dates):
    for fold in rolling_origin_splits(dates, n_splits=5, horizon_days=30, min_train_days=90):
        assert_no_temporal_leakage(dates, fold)
        train_dates = pd.to_datetime(dates).to_numpy()[fold.train_index]
        eval_dates = pd.to_datetime(dates).to_numpy()[fold.eval_index]
        assert train_dates.max() < eval_dates.min()


def test_rolling_folds_do_not_overlap_their_evaluation_windows(dates):
    folds = rolling_origin_splits(dates, n_splits=5, horizon_days=30, min_train_days=90)
    seen: set[int] = set()
    for fold in folds:
        current = set(fold.eval_index.tolist())
        assert not seen & current, "evaluation windows must be disjoint"
        seen |= current


def test_rolling_respects_the_minimum_training_span(dates):
    folds = rolling_origin_splits(dates, n_splits=20, horizon_days=30, min_train_days=200)
    for fold in folds:
        span = (fold.train_end - pd.to_datetime(dates).min()).days
        assert span >= 200 - 30


def test_rolling_raises_when_nothing_fits(dates):
    with pytest.raises(ValueError):
        rolling_origin_splits(dates, n_splits=3, horizon_days=30, min_train_days=10_000)


def test_leakage_guard_catches_a_bad_split(dates):
    positions = np.arange(len(dates))
    bad = Split(name="bad", train_index=positions, eval_index=positions[:10])
    with pytest.raises(TemporalLeakage):
        assert_no_temporal_leakage(dates, bad)


def test_leakage_guard_catches_an_empty_side(dates):
    with pytest.raises(TemporalLeakage):
        assert_no_temporal_leakage(
            dates, Split(name="empty", train_index=np.array([], dtype=int), eval_index=np.arange(5))
        )


def test_random_kfold_is_a_full_partition():
    folds = random_kfold_splits(100, n_splits=5, seed=0)
    assert len(folds) == 5
    covered = np.concatenate([f.eval_index for f in folds])
    assert sorted(covered.tolist()) == list(range(100))


def test_random_kfold_is_deterministic():
    first = random_kfold_splits(100, n_splits=5, seed=7)
    second = random_kfold_splits(100, n_splits=5, seed=7)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.eval_index, b.eval_index)


def test_build_splits_returns_holdout_and_folds(train):
    holdout, folds = build_splits(train["date"], SplitConfig())
    assert holdout.name.startswith("holdout")
    assert len(folds) == SplitConfig().n_rolling_splits
    for fold in folds:
        assert_no_temporal_leakage(train["date"], fold)
