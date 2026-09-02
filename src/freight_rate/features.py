"""Feature construction.

A pure transform: cleaned frame in, model matrix out. No I/O, no target, no
knowledge of where the rows came from. Training and inference call the same
:meth:`FeatureBuilder.transform`, which is what keeps train/serve skew out.

Design constraints that shaped this module:

1. **Every feature must be derivable from the December inputs.** Those rows ship
   only ``pickup, delivery, distance, equipment, weight, date``. Coordinates come
   from the fitted city reference; the market index from the fitted daily series.
2. **The scored window is two months beyond the training range.** A gradient-
   boosted tree cannot extrapolate a trend — it freezes at its last bin. So there
   is no linear time index by default. Seasonality is carried by cyclic
   day-of-year encodings, which wrap and therefore keep November and December
   inside the convex hull of the training support, and by ``market_index``, which
   is genuinely observed for the scored rows.
3. **No target-derived column.** No rate-per-mile, no target encoding, nothing
   that would have to be recomputed differently at inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import MarketIndexSeries
from .config import FeatureConfig
from .level import MarketLevelModel

DAYS_IN_YEAR = 365.25

CATEGORICAL_FEATURES: tuple[str, ...] = ("equipment", "pickup", "delivery")

# Columns a feature frame must never contain, asserted by the leakage tests.
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"posted_rate", "rate_per_mile", "log_rate", "log_rpm", "load_id", "predicted_rate"}
)


class FeatureBuilder:
    """Builds the model matrix; holds the fitted category levels.

    Category levels are frozen at fit time from the full known city list so an
    unseen city at inference becomes a missing category rather than shifting the
    integer codes of every other city.
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        *,
        market_index: MarketIndexSeries | None = None,
        level_model: MarketLevelModel | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.market_index = market_index or MarketIndexSeries()
        self.level_model = level_model
        self.categories_: dict[str, pd.CategoricalDtype] = {}
        self.columns_: list[str] = []
        self.fitted: bool = False

    # -- fit ---------------------------------------------------------------

    def fit(self, frame: pd.DataFrame, *, cities: list[str] | None = None) -> "FeatureBuilder":
        levels: dict[str, list[str]] = {}
        if cities is not None:
            levels["pickup"] = sorted(cities)
            levels["delivery"] = sorted(cities)
        for column in CATEGORICAL_FEATURES:
            if column in levels:
                continue
            if column in frame.columns:
                levels[column] = sorted(frame[column].astype(str).dropna().unique().tolist())

        self.categories_ = {
            column: pd.CategoricalDtype(categories=values, ordered=False)
            for column, values in levels.items()
        }
        self.columns_ = list(self._build(frame).columns)
        self.fitted = True
        return self

    def fit_transform(self, frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.fit(frame, **kwargs).transform(frame)

    # -- transform ---------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureBuilder.transform called before fit")
        features = self._build(frame)
        missing = [c for c in self.columns_ if c not in features.columns]
        if missing:
            raise ValueError(f"feature builder produced no {missing} for this frame")
        return features.loc[:, self.columns_]

    @property
    def categorical_columns(self) -> list[str]:
        return [c for c in self.columns_ if c in self.categories_]

    # -- internals ---------------------------------------------------------

    def _build(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        distance = pd.to_numeric(frame["distance"], errors="coerce")
        weight = pd.to_numeric(frame["weight"], errors="coerce")
        date = pd.to_datetime(frame["date"])

        # -- haul geometry. Rate is multiplicative in distance, so the model
        #    sees it in log space; raw distance is kept for threshold splits.
        out["log_distance"] = np.log(distance)
        out["distance"] = distance
        out["distance_at_floor"] = self._flag(frame, "distance_at_floor")

        great_circle = pd.to_numeric(
            frame.get("great_circle_miles", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        circuity = pd.to_numeric(
            frame.get("circuity", pd.Series(np.nan, index=frame.index)), errors="coerce"
        )
        out["log_great_circle"] = np.log(great_circle.clip(lower=1.0))
        # Capped: near-coincident synthetic city pairs produce ratios up to 78.
        out["circuity"] = circuity.clip(upper=5.0)

        # -- load characteristics
        out["weight"] = weight
        out["log_weight"] = np.log(weight.clip(lower=1.0))
        out["weight_per_mile"] = weight / distance.replace(0.0, np.nan)
        out["weight_was_missing"] = self._flag(frame, "weight_was_missing")
        out["weight_was_negative"] = self._flag(frame, "weight_was_negative")

        # -- calendar. Cyclic only: see the module docstring.
        if self.config.use_cyclic_calendar:
            day_of_year = date.dt.dayofyear.astype(float)
            angle = 2.0 * np.pi * day_of_year / DAYS_IN_YEAR
            out["day_of_year_sin"] = np.sin(angle)
            out["day_of_year_cos"] = np.cos(angle)
        out["day_of_week"] = date.dt.dayofweek.astype("int8")
        out["is_weekend"] = date.dt.dayofweek.ge(5).astype("int8")
        out["day_of_month"] = date.dt.day.astype("int8")
        if self.config.use_linear_time_trend:
            out["days_since_epoch"] = (date - pd.Timestamp("2025-01-01")).dt.days.astype(float)

        # -- market state
        market_index = pd.to_numeric(
            frame.get("market_index", pd.Series(np.nan, index=frame.index)), errors="coerce"
        )
        daily = pd.to_numeric(
            frame.get("market_index_daily", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        if daily.isna().all():
            daily = self.market_index.value_for(date)
        out["market_index"] = market_index.fillna(daily)
        out["market_index_daily"] = daily
        out["market_index_was_missing"] = self._flag(frame, "market_index_was_missing")
        for window in self.config.market_index_windows:
            rolled = self.market_index.rolling_for(date, window)
            out[f"market_index_{window}d"] = rolled
            # Deviation from the trailing mean separates "the market is high"
            # from "the market is rising", which price differently.
            out[f"market_index_dev_{window}d"] = daily - rolled

        # -- geography
        if self.config.use_geo_coordinates:
            for column in (
                "pickup_lat",
                "pickup_lon",
                "delivery_lat",
                "delivery_lon",
            ):
                out[column] = pd.to_numeric(frame[column], errors="coerce")
            out["delta_lat"] = out["delivery_lat"] - out["pickup_lat"]
            out["delta_lon"] = out["delivery_lon"] - out["pickup_lon"]

        # -- categoricals
        for column in CATEGORICAL_FEATURES:
            if column in ("pickup", "delivery") and not self.config.use_city_categoricals:
                continue
            if column not in frame.columns:
                continue
            dtype = self.categories_.get(column)
            values = frame[column].astype(str)
            out[column] = values.astype(dtype) if dtype is not None else values.astype("category")

        # -- market level. Smooth, extrapolating, and subtracted from the target
        #    by the estimator rather than split on. See level.py.
        if self.config.use_market_level:
            if self.level_model is None or not self.level_model.fitted:
                raise RuntimeError(
                    "use_market_level is set but no fitted MarketLevelModel was supplied"
                )
            out["market_level"] = self.level_model.predict(date, out["market_index_daily"])

        if self.config.use_quote_signal and "quote_signal" in frame.columns:
            out["quote_signal"] = pd.to_numeric(frame["quote_signal"], errors="coerce")

        leaked = FORBIDDEN_FEATURES & set(out.columns)
        if leaked:
            raise AssertionError(f"target-derived columns leaked into features: {sorted(leaked)}")
        return out

    @staticmethod
    def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int8")
        return pd.Series(0, index=frame.index, dtype="int8")


CITY_COLUMNS: tuple[str, ...] = ("pickup", "delivery")


def apply_city_dropout(
    features: pd.DataFrame, rate: float, seed: int
) -> pd.DataFrame:
    """Blank a random share of city values. **Training matrices only.**

    Eight cities appear in the scored data and nowhere in training, touching
    12.1% of the loads that must be predicted. A gradient-boosted tree sends an
    unknown category down its missing-value branch — and if no training row ever
    had a missing city, that branch was never fitted and the routing is
    arbitrary. Measured on a leave-cities-out split, that costs $11.44 MAE, more
    than wiping out the $1.15 the city features earn on seen cities.

    Masking a small share of cities during training turns that branch into a real
    fallback: the model learns to price an unknown city from distance, equipment
    and coordinates, which it has for every row. The alternative — dropping the
    city features outright — throws away genuine lane-level signal for the 88% of
    rows where it is available.

    Applied after the feature matrix is built and never on a prediction path, so
    inference stays deterministic.
    """
    if rate <= 0:
        return features
    if not 0 < rate < 1:
        raise ValueError(f"city_dropout_rate must be in (0, 1), got {rate}")

    out = features.copy()
    generator = np.random.default_rng(seed)
    for column in CITY_COLUMNS:
        if column not in out.columns:
            continue
        mask = generator.random(len(out)) < rate
        if mask.any():
            out.loc[mask, column] = np.nan
    return out
