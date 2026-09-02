"""Estimator comparison on the corrected temporal splits.

    python scripts/compare_estimators.py

Everything except the estimator is held fixed: the same cleaning, the same
outlier exclusion, the same ``log(rate) - log(distance) - market_level`` target,
the same features, the same city dropout, and the same deployment-faithful
splits (the evaluation window's own inputs supplied as context). Only the
regressor changes. Results go to ``reports/estimator_comparison.md``.

Candidates:

* scikit-learn ``HistGradientBoostingRegressor`` - the shipped model.
* LightGBM and XGBoost with matched capacity, if installed. Both are optional;
  a missing package is reported, not treated as a failure.
* A ridge regression on the same matrix (one-hot categoricals). The linear
  floor: it says how much of the accuracy the trees actually earn.

The decision rule is the repo's usual one: mean MAE across the 61-day holdout
and the five rolling folds, and a "win" smaller than the fold-to-fold spread of
the incumbent is not a win.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.compose import ColumnTransformer, make_column_selector  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from freight_rate import load_train, load_validation  # noqa: E402
from freight_rate import metrics  # noqa: E402
from freight_rate.config import RunConfig  # noqa: E402
from freight_rate.model import make_estimator  # noqa: E402
from freight_rate.outliers import TargetOutlierDetector  # noqa: E402
from freight_rate.pipeline import FreightRatePipeline  # noqa: E402
from freight_rate.reporting import markdown_table  # noqa: E402
from freight_rate.schema import TARGET  # noqa: E402
from freight_rate.validation import holdout_split, rolling_origin_splits  # noqa: E402


def hgb_factory(config: RunConfig):
    return lambda categorical: make_estimator(config.model, categorical)


def lightgbm_factory(config: RunConfig):
    try:
        import lightgbm
    except ImportError:
        return None
    m = config.model
    return lambda categorical: lightgbm.LGBMRegressor(
        learning_rate=m.learning_rate,
        n_estimators=m.max_iter,
        num_leaves=m.max_leaf_nodes,
        min_child_samples=m.min_samples_leaf,
        reg_lambda=m.l2_regularization,
        max_bin=m.max_bins,
        random_state=m.seed,
        verbose=-1,
    )


def xgboost_factory(config: RunConfig):
    try:
        import xgboost
    except ImportError:
        return None
    m = config.model
    return lambda categorical: xgboost.XGBRegressor(
        learning_rate=m.learning_rate,
        n_estimators=m.max_iter,
        max_leaves=m.max_leaf_nodes,
        grow_policy="lossguide",
        min_child_weight=m.min_samples_leaf,
        reg_lambda=m.l2_regularization,
        max_bin=m.max_bins,
        tree_method="hist",
        enable_categorical=True,
        random_state=m.seed,
        n_jobs=0,
    )


def ridge_factory(config: RunConfig):  # noqa: ARG001 - same signature as the others
    def build(categorical: list[str]):  # noqa: ARG001 - selected by dtype instead
        return Pipeline(
            [
                (
                    "columns",
                    ColumnTransformer(
                        [
                            (
                                "cat",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                make_column_selector(dtype_include="category"),
                            ),
                            (
                                "num",
                                Pipeline(
                                    [
                                        ("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                make_column_selector(dtype_exclude="category"),
                            ),
                        ]
                    ),
                ),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

    return build


CANDIDATES = {
    "sklearn HistGradientBoosting (shipped)": hgb_factory,
    "LightGBM, matched params": lightgbm_factory,
    "XGBoost, matched params": xgboost_factory,
    "Ridge on the same matrix (linear floor)": ridge_factory,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/estimator_comparison.md")
    args = parser.parse_args()

    config = RunConfig()
    train_all = load_train()
    validation = load_validation()
    holdout = holdout_split(train_all["date"], config.split.holdout_days)
    folds = rolling_origin_splits(
        train_all["date"],
        n_splits=config.split.n_rolling_splits,
        horizon_days=config.split.rolling_horizon_days,
        min_train_days=config.split.min_train_days,
    )
    contaminated = (
        TargetOutlierDetector(config.outliers)
        .fit(train_all.iloc[holdout.train_index])
        .flag(train_all)
        .to_numpy(dtype=bool)
    )
    splits = [("holdout 61d", holdout)] + [
        (f"{f.name} ({f.eval_start:%b %d}-{f.eval_end:%b %d})", f) for f in folds
    ]

    rows, skipped = [], []
    for name, make_factory in CANDIDATES.items():
        factory = make_factory(config)
        if factory is None:
            skipped.append(name)
            print(f"{name}: package not installed, skipped")
            continue
        row: dict[str, float | str] = {"estimator": name}
        seconds = 0.0
        for label, split in splits:
            train = train_all.iloc[split.train_index].reset_index(drop=True)
            evaluate = train_all.iloc[split.eval_index].reset_index(drop=True)
            mask = contaminated[split.eval_index]
            started = time.perf_counter()
            pipeline = FreightRatePipeline(config, estimator_factory=factory).fit(
                train, context_frames=(validation, evaluate.drop(columns=[TARGET]))
            )
            seconds += time.perf_counter() - started
            predicted = pipeline.predict(evaluate)
            actual = evaluate[TARGET].to_numpy(dtype=float)
            row[label] = metrics.compute(actual[~mask], predicted[~mask]).mae
        fold_values = [row[label] for label, _ in splits]
        row["mean"] = float(np.mean(fold_values))
        row["fit_s_per_split"] = seconds / len(splits)
        rows.append(row)
        print(
            f"{name:42s} "
            + "  ".join(f"{v:.2f}" for v in fold_values)
            + f"  mean={row['mean']:.2f}",
            flush=True,
        )

    table = pd.DataFrame(rows).set_index("estimator")
    incumbent = table.index[0]
    reference = table.loc[incumbent]
    spread = float(np.std([reference[label] for label, _ in splits[1:]]))
    table.insert(0, "dMean_vs_shipped", table["mean"] - reference["mean"])

    lines = [
        "# Estimator comparison - corrected temporal splits\n",
        "Generated by `python scripts/compare_estimators.py`. Everything except the "
        "estimator is held fixed: cleaning, outlier exclusion, target transform, "
        "features, city dropout, and the deployment-faithful splits (each evaluation "
        "window's own inputs supplied as context). MAE on clean rows, in dollars.\n",
        markdown_table(table.round(3), index_label="estimator"),
        "",
        f"Fold-to-fold standard deviation of the shipped model across the five rolling "
        f"folds: **${spread:.2f}**. A mean difference smaller than that is noise, not a win.",
    ]
    best = table["mean"].idxmin()
    if best == incumbent:
        lines.append(f"\n**Decision: keep {incumbent}.** Nothing beats it on the mean.")
    else:
        gain = float(reference["mean"] - table.loc[best, "mean"])
        verdict = "inside" if gain < spread else "outside"
        lines.append(
            f"\n**{best}** has the lowest mean, by ${gain:.2f}, which is {verdict} the "
            f"fold spread. "
            + (
                "That is not enough to justify a compiled dependency; the shipped model stays."
                if gain < spread
                else "That is a real difference and should be acted on."
            )
        )
    if skipped:
        lines.append("\nNot run (package not installed): " + ", ".join(skipped) + ".")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
