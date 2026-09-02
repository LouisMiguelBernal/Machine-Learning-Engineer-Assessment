"""Freight rate prediction — Spotter ML Engineer assessment.

Public surface:

    from freight_rate import FreightRatePipeline, load_train, load_validation

    pipeline = FreightRatePipeline().fit(load_train())
    rates = pipeline.predict(load_validation())
"""

from __future__ import annotations

from .config import DEFAULT_RUN, PATHS, SEED, RunConfig
from .data import (
    load_december_inputs,
    load_template,
    load_train,
    load_validation,
    write_december,
    write_submission,
)
from .pipeline import FitSummary, FreightRatePipeline

__all__ = [
    "DEFAULT_RUN",
    "PATHS",
    "SEED",
    "FitSummary",
    "FreightRatePipeline",
    "RunConfig",
    "load_december_inputs",
    "load_template",
    "load_train",
    "load_validation",
    "write_december",
    "write_submission",
]

__version__ = "1.0.0"
