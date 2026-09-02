"""Estimator and target transform.

Model choice: :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.

* The signal is tabular with strong non-linear interactions (rate-per-mile falls
  with distance, and by a different amount per equipment type). Gradient boosting
  is the right family; a linear model gives up 30% of the achievable accuracy.
* It has native categorical support, so 64 pickup and 64 delivery cities need no
  one-hot expansion and no target encoding — the second of which would have been
  the most likely source of leakage in this project.
* It handles NaN natively, which matters because ``weight`` and ``market_index``
  are missing at inference too.
* It ships with scikit-learn. A reviewer cloning the repo does not have to build
  a LightGBM wheel to reproduce the numbers, and the accuracy difference on
  48,000 rows is not measurable.

Target transform: the estimator regresses ``log(rate) - log(distance)``, i.e. log
rate-per-mile. Distance alone explains R^2 = 0.936 of ``log(rate)``, and a tree
approximates that smooth log-linear spine with a staircase it cannot extend past
its last bin. Subtracting it analytically hands the tree only the residual
structure, which is what it is actually good at. :class:`LogTargetRegressor`
re-applies the offset on the way out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import ModelConfig

OFFSET_COLUMNS: dict[str, tuple[str, ...]] = {
    "log_rate": (),
    "log_rpm": ("log_distance",),
    "log_rpm_level": ("log_distance", "market_level"),
}
TARGET_MODES = tuple(OFFSET_COLUMNS)

EXCLUDED_FROM_ESTIMATOR: frozenset[str] = frozenset({"market_level"})
"""Offset columns hidden from the tree. ``market_level`` is a pure additive
level that has already been subtracted from the target; leaving it visible would
let the tree split on it and partially undo the subtraction, which reinstates the
extrapolation failure it exists to prevent. ``log_distance`` is deliberately
*not* excluded — the curvature of rate-per-mile against distance is real
cross-sectional structure the tree should still model."""


def make_estimator(config: ModelConfig, categorical_features: list[str] | None = None):
    """Construct the underlying regressor from config. No fitting here.

    Early stopping uses scikit-learn's internal split, which is *random*, not
    temporal: ``validation_fraction`` of the training rows are held back at
    random and the round count is chosen on them. That is acceptable for a
    stopping rule because it never touches the evaluation window, but it
    means the round count is tuned in-window, and ``max_iter`` caps it anyway.
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,
        early_stopping=config.early_stopping,
        validation_fraction=config.validation_fraction if config.early_stopping else None,
        n_iter_no_change=config.n_iter_no_change,
        categorical_features=categorical_features or [],
        random_state=config.seed,
    )


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Fit in log space with a distance offset; predict in dollars.

    Back-transforming with a bare ``exp`` returns the conditional *median*, which
    under-states the conditional mean by roughly ``exp(sigma^2 / 2)``. Duan's
    smearing estimator corrects that non-parametrically from the training
    residuals, so the predictions are unbiased in dollar space rather than
    systematically low. The factor is reported as :attr:`smearing_factor_` so the
    correction is visible rather than buried.
    """

    def __init__(
        self,
        estimator,
        target_mode: str = "log_rpm",
        smearing: bool = True,
    ) -> None:
        if target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
        self.estimator = estimator
        self.target_mode = target_mode
        self.smearing = smearing

    # -- offset ------------------------------------------------------------

    @property
    def offset_columns(self) -> tuple[str, ...]:
        return OFFSET_COLUMNS[self.target_mode]

    def _offset(self, X: pd.DataFrame) -> np.ndarray:
        """The analytic part of the target that the tree does not have to learn."""
        missing = [c for c in self.offset_columns if c not in X.columns]
        if missing:
            raise ValueError(f"target_mode={self.target_mode!r} requires features {missing}")
        if not self.offset_columns:
            return np.zeros(len(X), dtype=float)
        return X.loc[:, list(self.offset_columns)].to_numpy(dtype=float).sum(axis=1)

    def _estimator_features(self, X: pd.DataFrame) -> pd.DataFrame:
        hidden = [c for c in self.offset_columns if c in EXCLUDED_FROM_ESTIMATOR]
        return X.drop(columns=hidden) if hidden else X

    # -- sklearn API -------------------------------------------------------

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "LogTargetRegressor":
        rate = np.asarray(y, dtype=float)
        if (rate <= 0).any():
            raise ValueError("posted_rate must be strictly positive to fit in log space")

        offset = self._offset(X)
        features = self._estimator_features(X)
        target = np.log(rate) - offset
        if sample_weight is None:
            # A bare fit: wrapped sklearn Pipelines reject an explicit None.
            self.estimator.fit(features, target)
        else:
            self.estimator.fit(features, target, sample_weight=sample_weight)

        residual = target - self.estimator.predict(features)
        self.smearing_factor_ = float(np.mean(np.exp(residual))) if self.smearing else 1.0
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        offset = self._offset(X)
        raw = self.estimator.predict(self._estimator_features(X))
        prediction = np.exp(raw + offset) * self.smearing_factor_
        if not np.isfinite(prediction).all():
            raise RuntimeError("model produced non-finite predictions")
        return prediction

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.target_tags.required = True
        return tags


class DistanceBaseline(BaseEstimator, RegressorMixin):
    """``log(rate) ~ a + b * log(distance)``, fitted by least squares.

    The floor every other model must clear. Reported alongside the real model so
    a headline metric can be read as a gain over something, not in a vacuum.
    """

    def fit(self, X: pd.DataFrame, y) -> "DistanceBaseline":
        log_distance = np.log(pd.to_numeric(X["distance"], errors="coerce").to_numpy(float))
        log_rate = np.log(np.asarray(y, dtype=float))
        self.slope_, self.intercept_ = np.polyfit(log_distance, log_rate, 1)
        residual = log_rate - (self.intercept_ + self.slope_ * log_distance)
        self.smearing_factor_ = float(np.mean(np.exp(residual)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_distance = np.log(pd.to_numeric(X["distance"], errors="coerce").to_numpy(float))
        return np.exp(self.intercept_ + self.slope_ * log_distance) * self.smearing_factor_


class RatePerMileBaseline(BaseEstimator, RegressorMixin):
    """``rate = median_rate_per_mile * distance``. The naive dispatcher heuristic."""

    def fit(self, X: pd.DataFrame, y) -> "RatePerMileBaseline":
        distance = pd.to_numeric(X["distance"], errors="coerce").to_numpy(float)
        self.rate_per_mile_ = float(np.median(np.asarray(y, dtype=float) / distance))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        distance = pd.to_numeric(X["distance"], errors="coerce").to_numpy(float)
        return self.rate_per_mile_ * distance


def build_model(config: ModelConfig, categorical_features: list[str] | None = None):
    """The configured model, ready to fit."""
    return LogTargetRegressor(
        estimator=make_estimator(config, categorical_features),
        target_mode=config.target_mode,
        smearing=config.smearing,
    )
