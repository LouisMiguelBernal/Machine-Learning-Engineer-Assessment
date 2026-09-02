"""The composition root.

One class wires cleaning, outlier rejection, feature building and the estimator
together. Training and inference both go through it, so there is exactly one code
path that turns a raw frame into a feature matrix — which is what stops
train/serve skew before it starts.

Fit order, and why it matters:

1. ``DataCleaner.fit`` learns imputation statistics **from training rows only**.
   It is allowed to see the scored frame's *coordinates* and *market_index*,
   because those are input features supplied for the scored rows; a deployed
   model would have them. It never sees a label outside the training set.
2. ``TargetOutlierDetector.fit`` reads the label, so it runs on training rows
   only, and the rows it flags are dropped from the fit — never from anything
   being predicted.
3. ``FeatureBuilder.fit`` freezes category levels from the full city list so an
   unseen city degrades to a missing category instead of reshuffling codes.
4. The estimator fits on the clean rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .cleaning import DataCleaner
from .config import RunConfig
from .features import FeatureBuilder, apply_city_dropout
from .level import MarketLevelModel
from .model import LogTargetRegressor, build_model
from .outliers import OutlierReport, TargetOutlierDetector
from .schema import TARGET


@dataclass
class FitSummary:
    """What a fit did, for logging and for the report."""

    n_rows_supplied: int
    n_rows_used: int
    n_features: int
    categorical_features: list[str] = field(default_factory=list)
    outliers: OutlierReport | None = None
    smearing_factor: float = 1.0
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None
    level: str = ""

    def describe(self) -> str:
        window = ""
        if self.train_start is not None:
            window = f" [{self.train_start:%Y-%m-%d}..{self.train_end:%Y-%m-%d}]"
        lines = [
            f"fitted on {self.n_rows_used:,} of {self.n_rows_supplied:,} rows{window}",
            f"features: {self.n_features} "
            f"({len(self.categorical_features)} categorical: "
            f"{', '.join(self.categorical_features) or 'none'})",
            f"smearing factor: {self.smearing_factor:.5f}",
        ]
        if self.outliers is not None:
            lines.append(self.outliers.describe())
        if self.level:
            lines.append(self.level)
        return "\n".join(lines)


class FreightRatePipeline:
    """Fit on labelled loads, predict a dollar rate for any load frame."""

    def __init__(
        self,
        config: RunConfig | None = None,
        *,
        estimator_factory: Callable[[list[str]], Any] | None = None,
    ) -> None:
        """``estimator_factory`` swaps the underlying regressor - it receives the
        categorical column names and returns an unfitted estimator. Used only
        by ``scripts/compare_estimators.py``; the target transform, cleaning,
        outlier handling and features stay exactly as configured."""
        self.config = config or RunConfig()
        self.estimator_factory = estimator_factory
        self.cleaner = DataCleaner(self.config.cleaning)
        self.outlier_detector = TargetOutlierDetector(self.config.outliers)
        self.features = FeatureBuilder(self.config.features)
        self.level_model: MarketLevelModel | None = None
        self.model: Any = None
        self.summary: FitSummary | None = None
        self.fitted: bool = False

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        train: pd.DataFrame,
        *,
        context_frames: tuple[pd.DataFrame, ...] = (),
    ) -> "FreightRatePipeline":
        """Fit every stage.

        ``context_frames`` supply inference-time feature values — city
        coordinates and the daily ``market_index`` for the scored dates. They
        must never contain a target column; that is asserted, not trusted.
        """
        for frame in context_frames:
            if TARGET in frame.columns:
                raise ValueError(
                    "context_frames must not carry the target column; "
                    "they are for inference-time features only"
                )

        cleaned = self.cleaner.fit(
            train, coordinate_frames=context_frames, index_frames=context_frames
        ).transform(train)
        self.features.market_index = self.cleaner.market_index

        if self.config.outliers.enabled:
            contaminated = self.outlier_detector.fit_flag(cleaned)
            usable = cleaned.loc[~contaminated]
        else:
            usable = cleaned

        # The level is fitted on clean rows only: 1.4% of labels are scaled by a
        # random factor, and a handful of them landing on a thin day would tilt
        # the daily mean the trend is read from.
        if self.config.features.use_market_level:
            self.level_model = MarketLevelModel(
                use_trend=self.config.features.level_use_trend,
                use_market_index=self.config.features.level_use_market_index,
                trend_damping=self.config.features.level_trend_damping,
                market_index_shrink=self.config.features.level_market_index_shrink,
            ).fit(
                usable["date"],
                usable["market_index_daily"],
                np.log(usable[TARGET]) - np.log(usable["distance"]),
            )
            self.features.level_model = self.level_model

        cities = self.cleaner.city_reference.cities if self.cleaner.city_reference else None
        X = self.features.fit(usable, cities=cities).transform(usable)
        # Training-only augmentation, so the model has a fitted path for the 8
        # cities that appear solely in the scored data.
        X = apply_city_dropout(X, self.config.features.city_dropout_rate, self.config.seed)
        y = usable[TARGET].to_numpy(dtype=float)

        if self.estimator_factory is None:
            self.model = build_model(self.config.model, self.features.categorical_columns)
        else:
            self.model = LogTargetRegressor(
                estimator=self.estimator_factory(self.features.categorical_columns),
                target_mode=self.config.model.target_mode,
                smearing=self.config.model.smearing,
            )
        self.model.fit(X, y)

        self.summary = FitSummary(
            n_rows_supplied=len(cleaned),
            n_rows_used=len(usable),
            n_features=X.shape[1],
            categorical_features=self.features.categorical_columns,
            outliers=self.outlier_detector.report_ if self.config.outliers.enabled else None,
            smearing_factor=getattr(self.model, "smearing_factor_", 1.0),
            train_start=cleaned["date"].min(),
            train_end=cleaned["date"].max(),
            level=self.level_model.describe() if self.level_model else "",
        )
        self.fitted = True
        return self

    # -- predict -----------------------------------------------------------

    def build_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Cleaned frame -> model matrix. The only inference feature path.

        Refuses a frame whose dates the daily market index has never seen:
        pricing such a frame would silently use a constant level, which is
        not the configuration the model runs in at deployment.
        """
        if not self.fitted:
            raise RuntimeError("FreightRatePipeline.build_matrix called before fit")
        self.cleaner.check_index_coverage(frame)
        return self.features.transform(self.cleaner.transform(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predicted dollar rate, one per row, in the input row order."""
        if not self.fitted or self.model is None:
            raise RuntimeError("FreightRatePipeline.predict called before fit")
        prediction = self.model.predict(self.build_matrix(frame))
        if not np.isfinite(prediction).all():
            raise RuntimeError("pipeline produced non-finite predictions")
        if (prediction <= 0).any():
            # A log-space model cannot emit a non-positive rate. If one appears,
            # the fault is upstream and clipping would only hide it.
            raise RuntimeError(
                f"pipeline produced {(prediction <= 0).sum()} non-positive predictions"
            )
        return prediction

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> Path:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "FreightRatePipeline":
        import joblib

        pipeline = joblib.load(path)
        if not isinstance(pipeline, FreightRatePipeline):
            raise TypeError(f"{path} does not contain a FreightRatePipeline")
        return pipeline
