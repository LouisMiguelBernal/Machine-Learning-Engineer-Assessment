"""Feature-side data-quality repair.

One stateful cleaner, fitted on training rows and applied unchanged everywhere
else. Every statistic it needs (medians, weight bounds, the daily market index)
is learned in :meth:`DataCleaner.fit` and frozen, so an inference frame can never
influence how it is itself cleaned.

Target-side contamination is a separate concern and lives in :mod:`outliers`,
because detecting it requires the label and must therefore never touch a frame
being predicted.

Defects addressed, all measured on the supplied data:

===========================  =========  ==========================================
Defect                       Rows       Handling
===========================  =========  ==========================================
Negative ``weight``          292        absolute value; flagged
Missing ``weight``           300 / 165  median by equipment; flagged
Missing ``market_index``     374 / 249  daily index, then global median; flagged
Absent coordinates           December   filled from the city reference table
===========================  =========  ==========================================

A note on what is deliberately *not* repaired. 111 training rows carry
``distance / great_circle > 1.6``, which looks like corrupt mileage. It is not.
Those rows are all short hauls (70-146 miles) whose reported distance sits within
1.16x the median distance of the same lane; the ratio is inflated because the
synthetic coordinates place nearby city pairs almost on top of each other
(New Orleans -> Shreveport great-circles to 7 miles). Rebuilding their distance
from the great circle raises the error on those rows from 15.4% to 31.2% MAPE.
The reported ``distance`` is trusted; ``circuity`` is kept only as a feature.
The observed 70-mile floor is likewise a billing rule, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CleaningConfig
from .geo import CityReference, haversine_miles


class MarketIndexCoverageError(ValueError):
    """Raised when a frame's dates are mostly absent from the fitted daily index.

    The silent alternative is worse than a crash. The first version of the
    temporal validation fitted the daily series from the training window plus
    the scored file, so the holdout window (September-October) had no entry and
    every one of its 61 days quietly received the global median. The level
    model was then evaluated as ``trend + constant`` while at deployment it
    runs as ``trend + real index`` - a 24% difference in holdout MAE that no
    test caught. See ``tests/test_pipeline_and_leakage.py``.
    """

FLAG_COLUMNS: tuple[str, ...] = (
    "weight_was_missing",
    "weight_was_negative",
    "market_index_was_missing",
    "distance_at_floor",
)

DERIVED_COLUMNS: tuple[str, ...] = (
    "great_circle_miles",
    "circuity",
    "market_index_daily",
)


@dataclass
class MarketIndexSeries:
    """Daily market index, with rolling context.

    ``market_index`` is a per-row observation of a market-wide daily state. At
    row level it is nearly uninformative (r = 0.034 against the rate), but its
    daily mean tracks the daily mean rate-per-mile at r = 0.577. Aggregating to a
    daily series both denoises it and supplies a value for the December chart
    rows, which ship no ``market_index`` column at all.
    """

    daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fallback: float = 1.0
    last_fallback_share: float = field(default=0.0, compare=False)
    """Share of the dates in the most recent lookup that were absent from the
    series and received ``fallback``. Read by ``DataCleaner.transform``."""

    @classmethod
    def from_frames(cls, *frames: pd.DataFrame) -> "MarketIndexSeries":
        """Daily mean index over every frame that carries the column.

        The fallback median is taken from the **first** frame only - the
        training frame - so an imputation constant never depends on which
        context frames happened to be supplied.
        """
        parts = [
            f.loc[:, ["date", "market_index"]]
            for f in frames
            if {"date", "market_index"}.issubset(f.columns)
        ]
        if not parts:
            return cls()
        stacked = pd.concat(parts, ignore_index=True).dropna(subset=["market_index"])
        if stacked.empty:
            return cls()
        stacked["date"] = pd.to_datetime(stacked["date"])
        daily = stacked.groupby("date")["market_index"].mean().sort_index()
        first = parts[0].dropna(subset=["market_index"])
        fallback = float(first["market_index"].median()) if not first.empty else float(daily.median())
        return cls(daily=daily, fallback=fallback)

    def rolling(self, window: int) -> pd.Series:
        if self.daily.empty:
            return self.daily
        return self.daily.rolling(window=window, min_periods=1).mean()

    def _reindex(self, dates: pd.Series, source: pd.Series) -> pd.Series:
        if source.empty:
            self.last_fallback_share = 1.0 if len(dates) else 0.0
            return pd.Series(self.fallback, index=dates.index, dtype=float)
        values = source.reindex(pd.to_datetime(dates).to_numpy()).to_numpy(dtype=float)
        missing = np.isnan(values)
        self.last_fallback_share = float(missing.mean()) if len(values) else 0.0
        return pd.Series(values, index=dates.index).fillna(self.fallback)

    def coverage(self, dates: pd.Series) -> float:
        """Share of ``dates`` that have an entry in the daily series."""
        if self.daily.empty:
            return 0.0
        present = pd.to_datetime(dates).isin(self.daily.index)
        return float(present.mean()) if len(present) else 1.0

    def value_for(self, dates: pd.Series) -> pd.Series:
        """Daily index for each date, falling back to the global median."""
        return self._reindex(dates, self.daily)

    def rolling_for(self, dates: pd.Series, window: int) -> pd.Series:
        """Trailing ``window``-day mean of the daily index, aligned to each date."""
        return self._reindex(dates, self.rolling(window))


class DataCleaner:
    """Fit repair statistics on training rows; apply them everywhere."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()
        self.city_reference: CityReference | None = None
        self.market_index: MarketIndexSeries = MarketIndexSeries()
        self.weight_median_by_equipment: dict[str, float] = {}
        self.weight_median_global: float = float("nan")
        self.weight_bounds: tuple[float, float] = (0.0, np.inf)
        self.distance_floor: float = 0.0
        self.fitted: bool = False

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        train: pd.DataFrame,
        *,
        coordinate_frames: tuple[pd.DataFrame, ...] = (),
        index_frames: tuple[pd.DataFrame, ...] = (),
    ) -> "DataCleaner":
        """Learn repair statistics from ``train``.

        ``coordinate_frames`` and ``index_frames`` may include the scored frame.
        That is deliberate and is not target leakage: city coordinates and
        ``market_index`` are *input features supplied for the scored rows*, so a
        deployed model would have them at prediction time too. No target-derived
        quantity is read from any frame other than ``train``.
        """
        self.city_reference = CityReference.from_frames(train, *coordinate_frames)
        self.market_index = MarketIndexSeries.from_frames(train, *index_frames)

        weight = self._normalise_weight(train["weight"])
        self.weight_median_global = float(weight.median())
        self.weight_median_by_equipment = {
            str(equipment): float(group.median())
            for equipment, group in weight.groupby(train["equipment"].astype(str))
            if np.isfinite(group.median())
        }
        # Weight is clipped at source: 1,191 rows sit exactly at 47,500 lb. Learn
        # the observed envelope rather than asserting an arbitrary legal maximum.
        self.weight_bounds = (
            float(np.nanquantile(weight, self.config.weight_clip_quantile)),
            float(np.nanmax(weight)),
        )
        self.distance_floor = float(pd.to_numeric(train["distance"]).min())
        self.fitted = True
        return self

    # -- transform ---------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.city_reference is None:
            raise RuntimeError("DataCleaner.transform called before fit")

        result = self.city_reference.attach(frame)
        result["date"] = pd.to_datetime(result["date"])
        result["distance"] = pd.to_numeric(result["distance"], errors="coerce")

        result["great_circle_miles"] = haversine_miles(
            result["pickup_lat"],
            result["pickup_lon"],
            result["delivery_lat"],
            result["delivery_lon"],
        )
        # Guarded against the near-coincident synthetic city pairs.
        result["circuity"] = result["distance"] / result["great_circle_miles"].replace(
            0.0, np.nan
        )
        result["distance_at_floor"] = (
            result["distance"] <= self.distance_floor
        ).astype("int8")

        result = self._repair_weight(result)
        result = self._repair_market_index(result)
        return result

    def check_index_coverage(self, frame: pd.DataFrame) -> float:
        """Raise unless the frame's dates are (almost all) in the daily index.

        A frame whose dates the series has never seen would be priced at a
        constant market level, which is not what the model does at deployment.
        Returns the coverage share on success.
        """
        covered = self.market_index.coverage(frame["date"])
        minimum = 1.0 - self.config.max_index_fallback_share
        if covered < minimum:
            raise MarketIndexCoverageError(
                f"only {covered:.1%} of the frame's dates have a daily market_index "
                f"(minimum {minimum:.0%}); pass a frame carrying market_index for "
                "these dates as a context frame when fitting, or the level is a constant"
            )
        return covered

    def fit_transform(self, train: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.fit(train, **kwargs).transform(train)

    # -- individual repairs ------------------------------------------------

    def _normalise_weight(self, weight: pd.Series) -> pd.Series:
        """Negative weights are sign-entry errors, not sentinels.

        The 292 negative rows have |weight| distributed like the positive rows
        and a matching rate-per-mile within each weight decile, so the magnitude
        is real and only the sign is wrong.
        """
        values = pd.to_numeric(weight, errors="coerce")
        if self.config.fix_negative_weight:
            values = values.abs()
        return values.where(values > 0)

    def _repair_weight(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = pd.to_numeric(frame["weight"], errors="coerce")
        frame["weight_was_missing"] = raw.isna().astype("int8")
        frame["weight_was_negative"] = (raw < 0).fillna(False).astype("int8")

        weight = self._normalise_weight(raw)
        by_equipment = (
            frame["equipment"].astype(str).map(self.weight_median_by_equipment).astype(float)
        )
        weight = weight.fillna(by_equipment).fillna(self.weight_median_global)
        frame["weight"] = weight.clip(*self.weight_bounds)
        return frame

    def _repair_market_index(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "market_index" in frame.columns:
            raw = pd.to_numeric(frame["market_index"], errors="coerce")
        else:
            # The December chart inputs carry no market_index column at all.
            raw = pd.Series(np.nan, index=frame.index, dtype=float)
        daily = self.market_index.value_for(frame["date"])
        frame["market_index_was_missing"] = raw.isna().astype("int8")
        frame["market_index"] = raw.fillna(daily)
        frame["market_index_daily"] = daily
        return frame
