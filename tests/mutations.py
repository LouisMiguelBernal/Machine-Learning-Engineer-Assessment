"""Deliberate defect injection, for proving the guards actually bite.

A regression test that passes whether or not the fix is present is worthless —
it looks like coverage and provides none. The only way to know is to break the
thing on purpose and confirm the test goes red.

Each mutation below removes one real safeguard from the codebase and names the
test that must fail as a result. ``scripts/verify_guards.py`` drives them:
it runs each named test twice, once mutated and once clean, and requires
**fail then pass**. A mutation that leaves its test green is reported as a hole
in the suite, not as a success.

Nothing here is imported unless ``FREIGHT_MUTATION`` is set in the environment,
so a normal ``pytest`` run never touches production behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

ENV_VAR = "FREIGHT_MUTATION"


@dataclass(frozen=True)
class Mutation:
    """One injected defect and the guard that is supposed to catch it."""

    name: str
    guard: str
    """The pytest ``-k`` selector for the test that must turn red."""
    breaks: str
    """Plain description of the safeguard being removed."""
    apply: Callable[[], None]


def _patch_dataclass_default(cls, field: str, value) -> None:
    """Override a frozen-dataclass field default, and register the undo.

    Assigning to the class attribute does **not** work: ``@dataclass`` bakes the
    defaults into ``__init__.__defaults__`` at class-creation time, so
    ``OutlierConfig.n_sigma = 1000`` leaves ``OutlierConfig().n_sigma`` at 5.0.
    A mutation that silently fails to apply makes its guard look verified when
    nothing was ever removed — the exact false confidence this harness exists to
    prevent. So patch where the default actually lives.
    """
    parameters = [f for f in cls.__dataclass_fields__ if f in cls.__init__.__code__.co_varnames]
    defaults = list(cls.__init__.__defaults__)
    offset = len(parameters) - len(defaults)
    index = parameters.index(field) - offset
    if not 0 <= index < len(defaults):
        raise AssertionError(f"{cls.__name__}.{field} has no default to patch")

    original = tuple(defaults)
    defaults[index] = value
    cls.__init__.__defaults__ = tuple(defaults)
    if cls.__init__.__defaults__[index] == original[index]:
        raise AssertionError(f"patching {cls.__name__}.{field} had no effect")
    _RESTORE.append(lambda: setattr(cls.__init__, "__defaults__", original))


# -- the injected defects ---------------------------------------------------


def _keep_negative_weights() -> None:
    """Stop treating negative weight as a sign-entry error."""
    from freight_rate import cleaning

    original = cleaning.DataCleaner._normalise_weight

    def broken(self, weight):
        values = pd.to_numeric(weight, errors="coerce")
        return values.where(values > 0)  # abs() removed

    cleaning.DataCleaner._normalise_weight = broken
    _RESTORE.append(lambda: setattr(cleaning.DataCleaner, "_normalise_weight", original))


def _blind_outlier_detector() -> None:
    """Widen the threshold so no corrupted label is ever flagged."""
    from freight_rate import config

    _patch_dataclass_default(config.OutlierConfig, "n_sigma", 1_000.0)


def _disable_leakage_guard() -> None:
    """Turn the temporal-leakage assertion into a no-op."""
    from freight_rate import validation

    original = validation.assert_no_temporal_leakage
    validation.assert_no_temporal_leakage = lambda dates, split: None
    _RESTORE.append(
        lambda: setattr(validation, "assert_no_temporal_leakage", original)
    )


def _readmit_the_decoy_feature() -> None:
    """Put ``quote_signal`` back into the feature matrix by default."""
    from freight_rate import config

    _patch_dataclass_default(config.FeatureConfig, "use_quote_signal", True)


def _drop_submission_validation() -> None:
    """Let the writer emit non-positive or non-finite rates."""
    from freight_rate import data

    original = data.write_submission

    def unchecked(predictions, template, path=None):
        from freight_rate.config import PATHS

        values = np.asarray(predictions, dtype=float)
        destination = path or PATHS.submission
        pd.DataFrame(
            {
                "load_id": template["load_id"].astype(str),
                "predicted_rate": np.round(values, 2),
            }
        ).to_csv(destination, index=False)
        return destination

    data.write_submission = unchecked
    _RESTORE.append(lambda: setattr(data, "write_submission", original))


def _forget_scored_frame_coordinates() -> None:
    """Build the city reference from training rows only.

    This is the subtle one: the model still fits and scores, but the 8 cities
    that appear solely in the scored data silently lose their coordinates.
    """
    from freight_rate import geo

    original = geo.CityReference.from_frames.__func__

    def train_only(cls, *frames):
        return original(cls, frames[0])

    geo.CityReference.from_frames = classmethod(train_only)
    _RESTORE.append(
        lambda: setattr(geo.CityReference, "from_frames", classmethod(original))
    )


def _ignore_index_coverage() -> None:
    """Let a frame whose dates the daily index has never seen be priced at a
    constant - the defect behind the mis-measured holdout."""
    from freight_rate import config

    _patch_dataclass_default(config.CleaningConfig, "max_index_fallback_share", 1.0)


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="negative_weight",
        guard="test_negative_weights_keep_their_magnitude",
        breaks="absolute value on the 292 sign-flipped weights",
        apply=_keep_negative_weights,
    ),
    Mutation(
        name="blind_outliers",
        guard="test_target_contamination_is_detected",
        breaks="detection of the 677 corrupted labels",
        apply=_blind_outlier_detector,
    ),
    Mutation(
        name="no_leakage_guard",
        guard="test_leakage_guard_catches_a_bad_split",
        breaks="the temporal-leakage assertion on every split",
        apply=_disable_leakage_guard,
    ),
    Mutation(
        name="decoy_feature",
        guard="test_quote_signal_is_excluded_by_default",
        breaks="exclusion of the uninformative quote_signal",
        apply=_readmit_the_decoy_feature,
    ),
    Mutation(
        name="unchecked_submission",
        guard="test_writer_refuses_non_positive_rates",
        breaks="positivity and finiteness checks on the submission writer",
        apply=_drop_submission_validation,
    ),
    Mutation(
        name="constant_index_window",
        guard="test_pipeline_refuses_a_window_without_index_coverage",
        breaks="the refusal to price a window whose dates have no daily market index",
        apply=_ignore_index_coverage,
    ),
    Mutation(
        name="train_only_cities",
        guard="test_unseen_cities_still_receive_coordinates",
        breaks="coordinates for the 8 cities seen only in the scored data",
        apply=_forget_scored_frame_coordinates,
    ),
)

BY_NAME = {mutation.name: mutation for mutation in MUTATIONS}

_RESTORE: list[Callable[[], None]] = []


def active() -> Mutation | None:
    """The mutation requested by the environment, if any."""
    name = os.environ.get(ENV_VAR)
    if not name:
        return None
    if name not in BY_NAME:
        raise SystemExit(
            f"unknown {ENV_VAR}={name!r}; expected one of {sorted(BY_NAME)}"
        )
    return BY_NAME[name]


def apply_if_requested() -> Mutation | None:
    """Called from conftest. A no-op unless FREIGHT_MUTATION is set."""
    mutation = active()
    if mutation is not None:
        mutation.apply()
    return mutation
