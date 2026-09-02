"""Typed loaders for the four supplied CSVs.

Loaders validate against ``schema`` and never mutate anything on disk. The one
writer here is :func:`write_submission`, which is the deliverable path and is
kept next to the readers so the round trip is obvious.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema
from .config import PATHS, Paths


def _read(path: Path, spec: schema.FrameSpec, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{spec.name}: not found at {path}")
    frame = pd.read_csv(path, parse_dates=parse_dates)
    spec.validate(frame)
    return frame


def load_train(paths: Paths = PATHS) -> pd.DataFrame:
    """Labelled development data, 2025-01-01 .. 2025-10-31."""
    return _read(paths.train, schema.TRAIN_SPEC, parse_dates=["date"])


def load_validation(paths: Paths = PATHS) -> pd.DataFrame:
    """The 12,000 scored loads, 2025-11-01 .. 2025-12-31. Unlabelled."""
    return _read(paths.validation, schema.VALIDATION_SPEC, parse_dates=["date"])


def load_template(paths: Paths = PATHS) -> pd.DataFrame:
    """Submission template — defines the required load_id order."""
    return _read(paths.prediction_template, schema.TEMPLATE_SPEC)


def load_december_inputs(paths: Paths = PATHS) -> pd.DataFrame:
    """Fixed-input December rows for the scorer chart."""
    return _read(paths.december_inputs, schema.DECEMBER_SPEC, parse_dates=["date"])


def write_submission(
    predictions: pd.Series | np.ndarray,
    template: pd.DataFrame,
    path: Path | None = None,
) -> Path:
    """Write ``load_id,predicted_rate`` in the template order.

    Rates are rounded to cents — a freight rate is a dollar amount, and rounding
    at the boundary keeps the written file identical across runs rather than
    differing in float formatting noise.
    """
    values = np.asarray(predictions, dtype=float)
    if len(values) != len(template):
        raise ValueError(
            f"prediction count {len(values)} does not match template rows {len(template)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("refusing to write non-finite predicted_rate values")
    if (values <= 0).any():
        raise ValueError("refusing to write non-positive predicted_rate values")

    destination = path or PATHS.submission
    out = pd.DataFrame(
        {"load_id": template["load_id"].astype(str), "predicted_rate": np.round(values, 2)}
    )
    out.to_csv(destination, index=False)
    return destination


def write_december(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Write back the December inputs with ``predicted_rate`` filled.

    Column order is asserted rather than assumed: the scorer rejects the file
    outright if the original seven columns are reordered.
    """
    destination = path or PATHS.december_inputs
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out[list(schema.DECEMBER_COLUMNS)]
    out["predicted_rate"] = np.round(pd.to_numeric(out["predicted_rate"]), 2)
    out.to_csv(destination, index=False)
    return destination
