"""Evaluation metrics.

Four headline numbers are always reported together, because each one alone is
misleading on this dataset:

* **MAE** — dollars. Interpretable, but dominated by the long hauls that cost
  thousands.
* **RMSE** — punishes the contaminated rows hardest, so it drifts with how much
  label corruption a given fold happens to contain.
* **MAPE** — scale-free, but over-weights the cheap 70-mile loads.
* **Median APE** — the same idea with the contaminated tail removed, which makes
  it the most stable read on typical accuracy.

Stratified breakdowns matter as much as the headline: a model that is excellent
on Dry Van and broken on Reefer averages out to looking fine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    n: int
    mae: float
    rmse: float
    mape: float
    median_ape: float
    r2: float
    bias: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def describe(self) -> str:
        return (
            f"n={self.n:,}  MAE={self.mae:,.2f}  RMSE={self.rmse:,.2f}  "
            f"MAPE={self.mape:.2f}%  MedAPE={self.median_ape:.2f}%  "
            f"R2={self.r2:.4f}  bias={self.bias:+.2f}"
        )


def _clean_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    usable = np.isfinite(true) & np.isfinite(pred)
    if not usable.any():
        raise ValueError("no finite (y_true, y_pred) pairs to score")
    return true[usable], pred[usable]


def compute(y_true, y_pred) -> Metrics:
    true, pred = _clean_pair(y_true, y_pred)
    error = pred - true
    absolute = np.abs(error)
    with np.errstate(divide="ignore", invalid="ignore"):
        percentage = np.where(true != 0, absolute / np.abs(true), np.nan) * 100.0
    variance = np.sum((true - true.mean()) ** 2)
    return Metrics(
        n=int(true.size),
        mae=float(absolute.mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        mape=float(np.nanmean(percentage)),
        median_ape=float(np.nanmedian(percentage)),
        r2=float(1.0 - np.sum(error**2) / variance) if variance > 0 else float("nan"),
        bias=float(error.mean()),
    )


def stratified(
    y_true,
    y_pred,
    groups: pd.Series,
    label: str = "group",
) -> pd.DataFrame:
    """Metrics per level of ``groups``, sorted by group label."""
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            label: np.asarray(groups),
        }
    )
    rows = [
        {label: key, **compute(part["y_true"], part["y_pred"]).as_dict()}
        for key, part in frame.groupby(label, observed=True, sort=True)
    ]
    return pd.DataFrame(rows).set_index(label)


def by_distance_decile(y_true, y_pred, distance: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Metrics per distance decile — the axis errors are most structured along."""
    values = pd.to_numeric(pd.Series(np.asarray(distance)), errors="coerce")
    decile = pd.qcut(values, bins, labels=False, duplicates="drop")
    table = stratified(y_true, y_pred, decile, label="distance_decile")
    table.insert(0, "distance_mean", values.groupby(decile).mean().round(1))
    return table


def summarise_folds(results: dict[str, Metrics]) -> pd.DataFrame:
    """One row per fold plus a mean row, for printing a CV table."""
    frame = pd.DataFrame({name: m.as_dict() for name, m in results.items()}).T
    frame.index.name = "split"
    if len(frame) > 1:
        mean = frame.mean(numeric_only=True)
        mean.name = "mean"
        frame = pd.concat([frame, mean.to_frame().T])
    frame["n"] = frame["n"].astype(int)
    return frame
