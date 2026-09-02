"""Temporal validation splits.

The development data covers 2025-01-01 to 2025-10-31; the scored data covers
2025-11-01 to 2025-12-31. The scored window is therefore strictly in the future
relative to every training row, and the only honest way to estimate performance
is to reproduce that arrangement inside the development data.

Two split generators:

* :func:`holdout_split` — the last ``holdout_days`` of the development data,
  defaulting to 61 days to match the length of the scored window exactly. This
  produces the headline number.
* :func:`rolling_origin_splits` — forward-chained expanding windows. Each fold
  trains on everything before a cut date and evaluates on the next
  ``horizon_days``. This shows whether the headline number is a fluke of one
  particular month.

:func:`random_kfold_splits` is provided only to quantify how optimistic a
non-temporal split would be. It is never used for model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from .config import SplitConfig


@dataclass(frozen=True)
class Split:
    """One train/evaluate partition, described by position and by date."""

    name: str
    train_index: np.ndarray
    eval_index: np.ndarray
    train_end: pd.Timestamp | None = None
    eval_start: pd.Timestamp | None = None
    eval_end: pd.Timestamp | None = None

    def __len__(self) -> int:
        return len(self.eval_index)

    def describe(self) -> str:
        if self.eval_start is None:
            return f"{self.name}: train={len(self.train_index):,} eval={len(self.eval_index):,}"
        return (
            f"{self.name}: train<= {self.train_end:%Y-%m-%d} ({len(self.train_index):,} rows) "
            f"eval {self.eval_start:%Y-%m-%d}..{self.eval_end:%Y-%m-%d} "
            f"({len(self.eval_index):,} rows)"
        )


class TemporalLeakage(AssertionError):
    """Raised when a split would let a future row inform a past prediction."""


def assert_no_temporal_leakage(dates: pd.Series, split: Split) -> None:
    """Every training row must predate every evaluation row."""
    values = pd.to_datetime(dates).to_numpy()
    train_dates = values[split.train_index]
    eval_dates = values[split.eval_index]
    if len(train_dates) == 0 or len(eval_dates) == 0:
        raise TemporalLeakage(f"{split.name}: empty side of the split")
    if train_dates.max() >= eval_dates.min():
        raise TemporalLeakage(
            f"{split.name}: training reaches {pd.Timestamp(train_dates.max()):%Y-%m-%d}, "
            f"which is not before the evaluation start "
            f"{pd.Timestamp(eval_dates.min()):%Y-%m-%d}"
        )


def holdout_split(dates: pd.Series, holdout_days: int = 61) -> Split:
    """Final ``holdout_days`` calendar days as the evaluation set."""
    values = pd.to_datetime(dates)
    cutoff = values.max() - pd.Timedelta(days=holdout_days - 1)
    is_eval = values >= cutoff
    if not is_eval.any() or is_eval.all():
        raise ValueError(f"holdout of {holdout_days} days does not partition the data")

    positions = np.arange(len(values))
    split = Split(
        name=f"holdout_{holdout_days}d",
        train_index=positions[~is_eval.to_numpy()],
        eval_index=positions[is_eval.to_numpy()],
        train_end=values[~is_eval].max(),
        eval_start=values[is_eval].min(),
        eval_end=values[is_eval].max(),
    )
    assert_no_temporal_leakage(values, split)
    return split


def rolling_origin_splits(
    dates: pd.Series,
    n_splits: int = 5,
    horizon_days: int = 30,
    min_train_days: int = 120,
) -> list[Split]:
    """Expanding-window forward-chained splits, oldest fold first.

    Folds are laid out backwards from the end of the data so the last fold always
    ends on the final available date, then returned in chronological order.
    """
    values = pd.to_datetime(dates)
    positions = np.arange(len(values))
    start, end = values.min(), values.max()
    horizon = pd.Timedelta(days=horizon_days)

    splits: list[Split] = []
    eval_end = end
    for fold in range(n_splits):
        eval_start = eval_end - horizon + pd.Timedelta(days=1)
        if (eval_start - start).days < min_train_days:
            break
        train_mask = (values < eval_start).to_numpy()
        eval_mask = ((values >= eval_start) & (values <= eval_end)).to_numpy()
        if not train_mask.any() or not eval_mask.any():
            break
        split = Split(
            name=f"fold_{n_splits - fold}",
            train_index=positions[train_mask],
            eval_index=positions[eval_mask],
            train_end=values[train_mask].max(),
            eval_start=eval_start,
            eval_end=eval_end,
        )
        assert_no_temporal_leakage(values, split)
        splits.append(split)
        eval_end = eval_start - pd.Timedelta(days=1)

    if not splits:
        raise ValueError(
            f"no rolling splits fit in {(end - start).days + 1} days with "
            f"min_train_days={min_train_days}, horizon_days={horizon_days}"
        )
    ordered = sorted(splits, key=lambda s: s.eval_start)
    return [
        Split(
            name=f"fold_{i + 1}",
            train_index=s.train_index,
            eval_index=s.eval_index,
            train_end=s.train_end,
            eval_start=s.eval_start,
            eval_end=s.eval_end,
        )
        for i, s in enumerate(ordered)
    ]


def leave_cities_out_split(
    frame: pd.DataFrame, n_cities: int = 8, seed: int = 0
) -> tuple[Split, list[str]]:
    """Hold out every load touching a sample of cities, to simulate cold start.

    The scored data introduces 8 cities that never appear in training, and they
    touch 12.1% of the rows that must be predicted. No temporal split can measure
    how the model behaves there, because the held-out months contain no new
    cities. This split does: it removes a comparable number of cities from
    training entirely and evaluates only on the loads that touch them.

    Unlike the temporal splits this one is *not* leakage-free in time — it is a
    deliberately different question, and is reported separately.
    """
    cities = sorted(set(frame["pickup"].astype(str)) | set(frame["delivery"].astype(str)))
    if n_cities >= len(cities):
        raise ValueError(f"cannot hold out {n_cities} of {len(cities)} cities")

    rng = np.random.default_rng(seed)
    held_out = sorted(rng.choice(cities, size=n_cities, replace=False).tolist())
    touched = (
        frame["pickup"].astype(str).isin(held_out)
        | frame["delivery"].astype(str).isin(held_out)
    ).to_numpy()

    positions = np.arange(len(frame))
    return (
        Split(
            name=f"cold_start_{n_cities}_cities",
            train_index=positions[~touched],
            eval_index=positions[touched],
        ),
        held_out,
    )


def random_kfold_splits(n_rows: int, n_splits: int = 5, seed: int = 0) -> list[Split]:
    """Shuffled K-fold. Diagnostic contrast only — never a selection metric."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    folds = np.array_split(order, n_splits)
    return [
        Split(
            name=f"random_fold_{i + 1}",
            train_index=np.sort(np.concatenate([f for j, f in enumerate(folds) if j != i])),
            eval_index=np.sort(fold),
        )
        for i, fold in enumerate(folds)
    ]


def build_splits(dates: pd.Series, config: SplitConfig) -> tuple[Split, list[Split]]:
    """The pair used by ``scripts/train.py``: headline holdout plus rolling folds."""
    return (
        holdout_split(dates, config.holdout_days),
        rolling_origin_splits(
            dates,
            n_splits=config.n_rolling_splits,
            horizon_days=config.rolling_horizon_days,
            min_train_days=config.min_train_days,
        ),
    )


def iter_splits(splits: list[Split]) -> Iterator[Split]:
    yield from splits
