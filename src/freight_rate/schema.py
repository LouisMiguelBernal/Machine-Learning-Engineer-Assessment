"""Declared column contracts for the four supplied CSVs.

Loaders validate against these, so a malformed or swapped input fails at read
time with a precise message instead of surfacing as a strange metric later.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "posted_rate"

TRAIN_COLUMNS: tuple[str, ...] = (
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "posted_rate",
)

VALIDATION_COLUMNS: tuple[str, ...] = tuple(c for c in TRAIN_COLUMNS if c != TARGET)

TEMPLATE_COLUMNS: tuple[str, ...] = ("load_id", "predicted_rate")

DECEMBER_COLUMNS: tuple[str, ...] = (
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
)
"""The December chart inputs carry no coordinates, no market_index and no
quote_signal. Any feature the model uses must be derivable from these six inputs
plus state fitted on training data."""

NUMERIC_COLUMNS: tuple[str, ...] = (
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
)

EXPECTED_TRAIN_ROWS = 48_000
EXPECTED_VALIDATION_ROWS = 12_000
EXPECTED_DECEMBER_ROWS = 31

TRAIN_DATE_RANGE = ("2025-01-01", "2025-10-31")
VALIDATION_DATE_RANGE = ("2025-11-01", "2025-12-31")


class SchemaError(ValueError):
    """Raised when a supplied file does not match its declared contract."""


@dataclass(frozen=True)
class FrameSpec:
    name: str
    columns: tuple[str, ...]
    expected_rows: int | None = None
    ordered: bool = True

    def validate(self, frame: pd.DataFrame) -> None:
        actual = tuple(frame.columns)
        if self.ordered:
            if actual != self.columns:
                raise SchemaError(
                    f"{self.name}: expected columns {list(self.columns)} in this order, "
                    f"got {list(actual)}"
                )
        else:
            missing = set(self.columns) - set(actual)
            if missing:
                raise SchemaError(f"{self.name}: missing columns {sorted(missing)}")
        if self.expected_rows is not None and len(frame) != self.expected_rows:
            raise SchemaError(
                f"{self.name}: expected {self.expected_rows:,} rows, got {len(frame):,}"
            )
        non_numeric = [
            c for c in NUMERIC_COLUMNS
            if c in frame.columns and not pd.api.types.is_numeric_dtype(frame[c])
        ]
        if non_numeric:
            raise SchemaError(
                f"{self.name}: columns {non_numeric} must be numeric; a text value has "
                "crept into a numeric column"
            )


TRAIN_SPEC = FrameSpec("train_test.csv", TRAIN_COLUMNS, EXPECTED_TRAIN_ROWS)
VALIDATION_SPEC = FrameSpec("validation.csv", VALIDATION_COLUMNS, EXPECTED_VALIDATION_ROWS)
TEMPLATE_SPEC = FrameSpec(
    "validation_predictions_template.csv", TEMPLATE_COLUMNS, EXPECTED_VALIDATION_ROWS
)
DECEMBER_SPEC = FrameSpec(
    "december_chart_inputs.csv", DECEMBER_COLUMNS, EXPECTED_DECEMBER_ROWS
)
