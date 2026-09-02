"""Holdout ablation harness.

Every design choice in this repo that could reasonably have gone the other way is
decided here, on the 61-day holdout that mirrors the scored window — not by
argument. Results are written to ``reports/model_selection.md``.

    python scripts/ablate.py                 # the standard set
    python scripts/ablate.py --only level    # one group
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight_rate import load_train, load_validation  # noqa: E402
from freight_rate import metrics  # noqa: E402
from freight_rate.config import RunConfig  # noqa: E402
from freight_rate.outliers import TargetOutlierDetector  # noqa: E402
from freight_rate.pipeline import FreightRatePipeline  # noqa: E402
from freight_rate.reporting import markdown_table  # noqa: E402
from freight_rate.validation import holdout_split, rolling_origin_splits  # noqa: E402


def variant(name: str, group: str, **overrides) -> tuple[str, str, RunConfig]:
    """A RunConfig with dotted overrides applied, e.g. features__use_quote_signal."""
    base = RunConfig()
    sections: dict[str, dict] = {}
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        sections.setdefault(section, {})[field] = value
    replacements = {
        section: dataclasses.replace(getattr(base, section), **values)
        for section, values in sections.items()
    }
    return name, group, dataclasses.replace(base, **replacements)


VARIANTS: list[tuple[str, str, RunConfig]] = [
    variant("baseline (full model)", "reference"),
    # -- target parameterisation
    variant("target: log_rate", "target", model__target_mode="log_rate"),
    variant("target: log_rpm (no level)", "target", model__target_mode="log_rpm"),
    # -- market level decomposition. Both index settings are listed so the
    #    table reads the same whichever one is the configured default.
    variant("level: trend + market_index", "level", features__level_use_market_index=True),
    variant("level: trend only", "level", features__level_use_market_index=False),
    variant("level: market_index only", "level", features__level_use_trend=False),
    variant("level: index shrunk 0.5", "level", features__level_market_index_shrink=0.5),
    variant("level: index shrunk 0.25", "level", features__level_market_index_shrink=0.25),
    variant("level: trend damped 0.5", "level", features__level_trend_damping=0.5),
    variant("level: trend frozen (0.0)", "level", features__level_trend_damping=0.0),
    # -- data quality
    variant("outliers: kept in training", "data-quality", outliers__enabled=False),
    # -- features
    variant("features: + quote_signal", "features", features__use_quote_signal=True),
    variant("features: - city categoricals", "features", features__use_city_categoricals=False),
    variant("features: - coordinates", "features", features__use_geo_coordinates=False),
    variant("features: - cyclic calendar", "features", features__use_cyclic_calendar=False),
    variant("features: + raw time trend", "features", features__use_linear_time_trend=True),
    variant("features: no city dropout", "features", features__city_dropout_rate=0.0),
]


def evaluate(
    config: RunConfig,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    context: tuple[pd.DataFrame, ...],
    contaminated: np.ndarray,
) -> tuple[metrics.Metrics, metrics.Metrics, float]:
    started = time.perf_counter()
    # The holdout's own inputs (never its label) are context, as at deployment.
    pipeline = FreightRatePipeline(config).fit(
        train, context_frames=(*context, holdout.drop(columns=["posted_rate"]))
    )
    predicted = pipeline.predict(holdout)
    actual = holdout["posted_rate"].to_numpy(dtype=float)
    return (
        metrics.compute(actual[~contaminated], predicted[~contaminated]),
        metrics.compute(actual, predicted),
        time.perf_counter() - started,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run a single group (target, level, features, ...)")
    parser.add_argument("--output", default="reports/model_selection.md")
    parser.add_argument(
        "--folds",
        action="store_true",
        help="also score every variant on the rolling-origin folds (6x the compute)",
    )
    args = parser.parse_args()

    train_all = load_train()
    validation = load_validation()
    split = holdout_split(train_all["date"], RunConfig().split.holdout_days)
    train = train_all.iloc[split.train_index].reset_index(drop=True)
    holdout = train_all.iloc[split.eval_index].reset_index(drop=True)

    # The contamination mask is for reporting only. It never touches a fit; it
    # lets the clean-row column separate model error from the irreducible error
    # of corrupted labels. The detector is fitted on the training window so the
    # threshold never reads a holdout label.
    detector = TargetOutlierDetector().fit(train)
    contaminated = detector.flag(holdout).to_numpy(dtype=bool)

    fold_context: list[tuple] = []
    if args.folds:
        cfg = RunConfig().split
        mask_all = detector.flag(train_all).to_numpy(dtype=bool)
        for fold in rolling_origin_splits(
            train_all["date"], cfg.n_rolling_splits, cfg.rolling_horizon_days, cfg.min_train_days
        ):
            fold_context.append(
                (
                    fold.name,
                    train_all.iloc[fold.train_index].reset_index(drop=True),
                    train_all.iloc[fold.eval_index].reset_index(drop=True),
                    mask_all[fold.eval_index],
                )
            )

    selected = [v for v in VARIANTS if args.only is None or v[1] == args.only]
    if not selected:
        parser.error(f"no variants in group {args.only!r}")

    print(f"holdout: {split.describe()}")
    print(f"contaminated rows in holdout: {contaminated.sum():,} of {len(holdout):,}\n")

    rows = []
    for name, group, config in selected:
        clean, full, seconds = evaluate(config, train, holdout, (validation,), contaminated)
        row = {
            "variant": name,
            "group": group,
            "MAE_clean": clean.mae,
            "MAPE_clean": clean.mape,
            "MedAPE_clean": clean.median_ape,
            "bias_clean": clean.bias,
            "R2_clean": clean.r2,
            "MAE_all": full.mae,
            "MAPE_all": full.mape,
            "fit_s": seconds,
        }
        if fold_context:
            fold_maes = []
            for fold_name, fold_train, fold_eval, fold_mask in fold_context:
                fold_clean, _, _ = evaluate(config, fold_train, fold_eval, (validation,), fold_mask)
                row[f"MAE_{fold_name}"] = fold_clean.mae
                fold_maes.append(fold_clean.mae)
            row["MAE_fold_mean"] = float(np.mean(fold_maes))
            row["MAE_6window_mean"] = float(np.mean([clean.mae, *fold_maes]))
        rows.append(row)
        print(f"{name:34s} {clean.describe()}  [{seconds:.0f}s]", flush=True)

    table = pd.DataFrame(rows).set_index("variant")
    if "baseline (full model)" in table.index:
        reference = table.loc["baseline (full model)", "MAE_clean"]
        table.insert(1, "dMAE_vs_baseline", table["MAE_clean"] - reference)
    if fold_context:
        # Keep the decision columns together at the left.
        front = ["group", "MAE_clean", "MAE_fold_mean", "MAE_6window_mean"]
        table = table[front + [c for c in table.columns if c not in front]]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(
            "# Model selection — holdout ablation\n\n"
            if not fold_context
            else "# Level selection — holdout and rolling folds\n\n"
        )
        handle.write(
            f"Holdout: {split.eval_start:%Y-%m-%d}..{split.eval_end:%Y-%m-%d} "
            f"({len(holdout):,} loads), trained on {len(train):,} loads up to "
            f"{split.train_end:%Y-%m-%d}.\n\n"
            f"`_clean` columns exclude the {contaminated.sum():,} holdout rows whose label is "
            "multiplicatively corrupted; their error is irreducible and would otherwise mask "
            "differences between variants. `_all` columns include them.\n\n"
            "Every evaluation window is supplied its own inputs (coordinates and "
            "`market_index`, never the label) as context, exactly as the scored file "
            "supplies them for November-December.\n\n"
        )
        if fold_context:
            handle.write(
                "`MAE_fold_mean` is the mean over the five rolling-origin folds; "
                "`MAE_6window_mean` averages the holdout with them. A variant is adopted "
                "only if it does not lose on either geometry.\n\n"
            )
        handle.write(markdown_table(table.round(4), index_label="variant"))
        handle.write("\n")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
