"""Target contamination detection.

The single largest data-quality defect in this dataset is on the label, not the
features. Roughly 1.4% of training rows carry a ``posted_rate`` that has been
multiplied by a random factor: 340 rows scaled up by 2.2x-5.6x and 337 rows
scaled down to 0.17x-0.45x.

Evidence that this is corruption rather than a real market segment:

* It is uniform across month, equipment, distance, weight, pickup city and
  ``market_index`` — nothing in the features predicts it.
* The separation is total. Against a robust ``log(rate) ~ log(distance)`` fit,
  103 rows land in the ``|residual|`` band 0.2-0.3 and 54 rows above 0.6, with
  **exactly zero rows between 0.3 and 0.5**. Genuine heavy tails do not leave a
  clean gap; a mixture with an injected component does.
* Removing them drops the residual standard deviation from 0.169 to 0.074 and
  takes the residual from skew -6.7 / kurtosis 71.8 to skew 0.17 / kurtosis
  -0.46 — an essentially Gaussian error once the mixture is gone.

Handling: exclude the flagged rows from **training only**. Because the
contamination is unpredictable from features, no model can identify which scored
rows are affected, so the best available prediction for every row is the centre
of the clean conditional distribution. The corruption is close to symmetric in
log space, which makes that centre the right target under MAE and MAPE alike.

This detector reads the label, so it may only ever be fitted on training rows,
never on a frame being predicted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import OutlierConfig


@dataclass(frozen=True)
class OutlierReport:
    """What a detection pass found."""

    n_rows: int
    n_flagged: int
    n_inflated: int
    n_deflated: int
    threshold: float
    scale: float
    residual_std_before: float
    residual_std_after: float

    @property
    def flagged_fraction(self) -> float:
        return self.n_flagged / self.n_rows if self.n_rows else 0.0

    def describe(self) -> str:
        return (
            f"target contamination: {self.n_flagged:,} / {self.n_rows:,} rows "
            f"({self.flagged_fraction:.2%}) - {self.n_inflated:,} inflated, "
            f"{self.n_deflated:,} deflated; log-residual sigma "
            f"{self.residual_std_before:.4f} -> {self.residual_std_after:.4f} "
            f"(threshold {self.threshold:.3f})"
        )


class TargetOutlierDetector:
    """Flag multiplicatively corrupted labels with an IRLS log-linear baseline.

    A plain OLS fit of ``log(rate) ~ log(distance)`` is itself dragged by the
    contamination, so the fit is re-run with the current inliers until the
    flagged set stops moving. The cut is placed at ``n_sigma`` robust MAD-scaled
    deviations, which lands inside the empty 0.3-0.5 band and so is insensitive
    to the exact constant.
    """

    def __init__(self, config: OutlierConfig | None = None) -> None:
        self.config = config or OutlierConfig()
        self.intercept_: float = 0.0
        self.slope_: float = 0.0
        self.centre_: float = 0.0
        self.scale_: float = 0.0
        self.threshold_: float = 0.0
        self.report_: OutlierReport | None = None
        self.fitted: bool = False

    @staticmethod
    def _mad_scale(values: np.ndarray) -> float:
        centre = float(np.median(values))
        return float(1.4826 * np.median(np.abs(values - centre))) or float(np.std(values))

    def fit(self, frame: pd.DataFrame, target: str = "posted_rate") -> "TargetOutlierDetector":
        rate = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        distance = pd.to_numeric(frame["distance"], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(rate) & (rate > 0) & np.isfinite(distance) & (distance > 0)

        x = np.log(distance[usable])
        y = np.log(rate[usable])
        inliers = np.ones(x.shape, dtype=bool)

        residual = np.zeros_like(y)
        for _ in range(self.config.max_iterations):
            slope, intercept = np.polyfit(x[inliers], y[inliers], 1)
            residual = y - (intercept + slope * x)
            centre = float(np.median(residual[inliers]))
            scale = self._mad_scale(residual[inliers])
            updated = np.abs(residual - centre) < self.config.n_sigma * scale
            if updated.sum() == inliers.sum() and np.array_equal(updated, inliers):
                inliers = updated
                break
            inliers = updated

        self.slope_ = float(slope)
        self.intercept_ = float(intercept)
        self.centre_ = float(np.median(residual[inliers]))
        self.scale_ = self._mad_scale(residual[inliers])
        self.threshold_ = self.config.n_sigma * self.scale_

        deviation = residual - self.centre_
        flagged = np.abs(deviation) >= self.threshold_
        self.report_ = OutlierReport(
            n_rows=int(usable.sum()),
            n_flagged=int(flagged.sum()),
            n_inflated=int((deviation >= self.threshold_).sum()),
            n_deflated=int((deviation <= -self.threshold_).sum()),
            threshold=self.threshold_,
            scale=self.scale_,
            residual_std_before=float(np.std(residual)),
            residual_std_after=float(np.std(residual[~flagged])),
        )
        self.fitted = True
        return self

    def flag(self, frame: pd.DataFrame, target: str = "posted_rate") -> pd.Series:
        """Boolean Series, True where the label looks corrupted."""
        if not self.fitted:
            raise RuntimeError("TargetOutlierDetector.flag called before fit")
        rate = pd.to_numeric(frame[target], errors="coerce")
        distance = pd.to_numeric(frame["distance"], errors="coerce")
        valid = (rate > 0) & (distance > 0) & rate.notna() & distance.notna()

        residual = np.log(rate.where(valid)) - (
            self.intercept_ + self.slope_ * np.log(distance.where(valid))
        )
        deviation = (residual - self.centre_).abs()
        # Rows that are unusable rather than corrupt are excluded from training too.
        return (deviation >= self.threshold_).fillna(True)

    def fit_flag(self, frame: pd.DataFrame, target: str = "posted_rate") -> pd.Series:
        return self.fit(frame, target=target).flag(frame, target=target)

    def split(
        self, frame: pd.DataFrame, target: str = "posted_rate"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(clean, contaminated)`` partitions of ``frame``."""
        flagged = self.flag(frame, target=target)
        return frame.loc[~flagged], frame.loc[flagged]


def irreducible_mape(detector: TargetOutlierDetector) -> float:
    """Rough floor on MAPE if the scored labels carry the same contamination.

    Corrupted rows cannot be identified from features, so their error is
    unavoidable. This estimate is reported alongside validation metrics so a
    single-digit MAPE is not mistaken for a modelling failure.
    """
    if detector.report_ is None:
        raise RuntimeError("irreducible_mape called before fit")
    report = detector.report_
    if report.n_flagged == 0:
        return 0.0
    # Observed multipliers span roughly [0.17, 0.45] and [2.2, 5.6]; the mean
    # absolute relative deviation of that mixture is close to 1.6.
    mean_abs_relative_error = 1.6
    return report.flagged_fraction * mean_abs_relative_error
