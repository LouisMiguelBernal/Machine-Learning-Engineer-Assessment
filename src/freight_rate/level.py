"""Market level model - the part a tree cannot extrapolate.

The scored window sits 61 days beyond the last training row, and the market level
does not stand still over that gap. Decomposing the daily mean of
``log(rate) - log(distance)`` across the 304 training days gives

    level(t, m) = intercept + trend * t + coefficient * m

where ``t`` is days since the start of the data and ``m`` is that day's mean
``market_index``. See ``reports/validation.md`` for the fitted values.

Why this cannot be left to the gradient-boosted tree. A tree is piecewise
constant, so beyond the last training bin it repeats its final value. Given a raw
time index it would freeze at the October level and lose the remaining drift.
So the level is fitted as an explicit linear model, subtracted from the target
before boosting, and added back at prediction time, where it extrapolates by
construction. Removing it costs about $9 of holdout MAE
(``reports/model_selection.md``, "target: log_rpm (no level)").

**What the two validation geometries say about the index term.** Once every
evaluation window is supplied its own ``market_index`` - which an earlier
version of the validation did not do, so the index term was evaluated as a
constant - the 61-day holdout and the rolling folds disagree:

* trend + index: holdout $48.15, fold mean $46.82
* trend only:    holdout $40.56, fold mean $51.19 (fold 2 at $80.61)

The index term helps where the index swings hard (June-August) and hurts in
September-October, where rates held up while the index fell. Neither variant
wins on both geometries, so the configured default keeps both terms and
``market_index_shrink`` exists as a measured hedge (``reports/level_selection.md``).
That disagreement is a stated limitation, not a solved problem.

``trend_damping`` shrinks the extrapolated portion of the trend for runs where a
secular drift is not trusted to continue; it does not affect fitted in-sample
values. ``market_index_shrink`` pulls the index term towards its training mean;
0.0 reproduces a trend-only level without refitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2025-01-01")


@dataclass(frozen=True)
class LevelFit:
    intercept: float
    trend_per_day: float
    market_coefficient: float
    r_squared: float
    residual_std: float
    n_days: int
    fit_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"level = {self.intercept:.4f} + {self.trend_per_day:.6f}*t "
            f"+ {self.market_coefficient:.4f}*market_index  "
            f"(R2={self.r_squared:.4f}, daily sigma={self.residual_std:.4f}, "
            f"{self.n_days} days, fit ends {self.fit_end:%Y-%m-%d})"
        )


class MarketLevelModel:
    """Weighted least squares on daily means of log rate-per-mile."""

    def __init__(
        self,
        use_trend: bool = True,
        use_market_index: bool = True,
        trend_damping: float = 1.0,
        market_index_shrink: float = 1.0,
    ) -> None:
        self.use_trend = use_trend
        self.use_market_index = use_market_index
        self.trend_damping = float(trend_damping)
        self.market_index_shrink = float(market_index_shrink)
        self.fit_: LevelFit | None = None
        self.fitted: bool = False

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _time_index(dates: pd.Series) -> np.ndarray:
        return (pd.to_datetime(dates) - EPOCH).dt.days.to_numpy(dtype=float)

    def _design(self, t: np.ndarray, market: np.ndarray) -> np.ndarray:
        columns = [np.ones_like(t)]
        if self.use_trend:
            columns.append(t)
        if self.use_market_index:
            columns.append(market)
        return np.column_stack(columns)

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        dates: pd.Series,
        market_index_daily: pd.Series,
        log_rate_per_mile: pd.Series,
    ) -> "MarketLevelModel":
        """Fit on daily means, weighted by how many loads each day carries.

        Aggregating first is deliberate: the level is a property of the day, and
        fitting on 48,000 rows would let a heavy day's cross-sectional spread
        masquerade as level information.
        """
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(dates).to_numpy(),
                "market": pd.to_numeric(market_index_daily, errors="coerce").to_numpy(float),
                "y": pd.to_numeric(log_rate_per_mile, errors="coerce").to_numpy(float),
            }
        ).dropna()
        if frame.empty:
            raise ValueError("no usable rows to fit the market level")

        daily = frame.groupby("date").agg(market=("market", "mean"), y=("y", "mean"), n=("y", "size"))
        t = self._time_index(pd.Series(daily.index))
        design = self._design(t, daily["market"].to_numpy(float))
        weights = np.sqrt(daily["n"].to_numpy(float))

        coefficients, *_ = np.linalg.lstsq(
            design * weights[:, None], daily["y"].to_numpy(float) * weights, rcond=None
        )
        self.coefficients_ = coefficients

        fitted_values = design @ coefficients
        residual = daily["y"].to_numpy(float) - fitted_values
        variance = float(np.var(daily["y"].to_numpy(float)))

        index = 1
        trend = float(coefficients[index]) if self.use_trend else 0.0
        index += int(self.use_trend)
        market_coefficient = float(coefficients[index]) if self.use_market_index else 0.0

        self.trend_origin_ = float(t.max())
        self.market_origin_ = float(daily["market"].mean())
        self.fit_ = LevelFit(
            intercept=float(coefficients[0]),
            trend_per_day=trend,
            market_coefficient=market_coefficient,
            r_squared=float(1.0 - np.var(residual) / variance) if variance > 0 else float("nan"),
            residual_std=float(np.std(residual)),
            n_days=int(len(daily)),
            fit_end=pd.Timestamp(daily.index.max()),
        )
        self.fitted = True
        return self

    # -- predict -----------------------------------------------------------

    def predict(self, dates: pd.Series, market_index_daily: pd.Series) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MarketLevelModel.predict called before fit")
        t = self._time_index(dates)
        if self.use_trend and self.trend_damping != 1.0:
            # Damp only the portion of the trend beyond the fitted range, so
            # in-sample predictions are unchanged.
            overshoot = np.clip(t - self.trend_origin_, 0.0, None)
            t = t - overshoot * (1.0 - self.trend_damping)
        market = pd.to_numeric(market_index_daily, errors="coerce").to_numpy(float)
        market = np.nan_to_num(market, nan=float(np.nanmedian(market)) if market.size else 0.0)
        if self.use_market_index and self.market_index_shrink != 1.0:
            # Pull the index towards its training mean; 0.0 makes the level
            # trend-only without refitting, 1.0 leaves the fitted term intact.
            market = self.market_origin_ + self.market_index_shrink * (market - self.market_origin_)
        return self._design(t, market) @ self.coefficients_

    def describe(self) -> str:
        return self.fit_.describe() if self.fit_ else "MarketLevelModel(unfitted)"
