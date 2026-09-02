"""Train and validate the freight rate model.

    python scripts/train.py            # holdout evaluation, fit final model, save it
    python scripts/train.py --cv       # add rolling-origin folds and baselines
    python scripts/train.py --no-save  # evaluate only

Writes ``reports/validation.md`` and ``artifacts/model.joblib``.

The final model is refit on **all** development data, including the holdout.
The holdout exists to estimate error honestly, not to be permanently withheld:
once the estimate is taken, discarding the two most recent months of a
trending market would be strictly worse for a model that must predict the
months after them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight_rate import load_train, load_validation  # noqa: E402
from freight_rate import metrics  # noqa: E402
from freight_rate.config import PATHS, RunConfig  # noqa: E402
from freight_rate.model import DistanceBaseline, RatePerMileBaseline  # noqa: E402
from freight_rate.outliers import TargetOutlierDetector, irreducible_mape  # noqa: E402
from freight_rate.pipeline import FreightRatePipeline  # noqa: E402
from freight_rate.reporting import markdown_table  # noqa: E402
from freight_rate.schema import TARGET  # noqa: E402
from freight_rate.validation import (  # noqa: E402
    Split,
    holdout_split,
    random_kfold_splits,
    rolling_origin_splits,
)


def _evaluate_split(
    config: RunConfig,
    frame: pd.DataFrame,
    split: Split,
    context: tuple[pd.DataFrame, ...],
    contaminated: np.ndarray,
) -> dict[str, metrics.Metrics]:
    """Fit on the split's training rows, score its evaluation rows."""
    train = frame.iloc[split.train_index].reset_index(drop=True)
    evaluate = frame.iloc[split.eval_index].reset_index(drop=True)
    actual = evaluate[TARGET].to_numpy(dtype=float)
    mask = contaminated[split.eval_index]

    # The evaluation window's own inputs - coordinates and market_index, never
    # the label - are supplied exactly as the scored file supplies them for
    # November-December. Without this the window's dates are absent from the
    # daily index and every one of them silently receives the global median,
    # so the level is evaluated as a constant it never is at deployment.
    pipeline = FreightRatePipeline(config).fit(
        train, context_frames=(*context, evaluate.drop(columns=[TARGET]))
    )
    predicted = pipeline.predict(evaluate)

    results = {
        "model": metrics.compute(actual[~mask], predicted[~mask]),
        "model_all_rows": metrics.compute(actual, predicted),
    }
    for name, estimator in (
        ("baseline_log_linear", DistanceBaseline()),
        ("baseline_rate_per_mile", RatePerMileBaseline()),
    ):
        estimator.fit(train[["distance"]], train[TARGET])
        results[name] = metrics.compute(
            actual[~mask], estimator.predict(evaluate[["distance"]])[~mask]
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", action="store_true", help="also run rolling-origin folds")
    parser.add_argument(
        "--random-kfold",
        action="store_true",
        help="also run a shuffled K-fold, to show how optimistic a non-temporal split is",
    )
    parser.add_argument("--no-save", action="store_true", help="do not write the model artefact")
    parser.add_argument("--output", default="reports/validation.md")
    args = parser.parse_args()

    config = RunConfig()
    PATHS.ensure_dirs()
    train_all = load_train()
    validation = load_validation()
    context = (validation,)

    # Reporting-only mask: separates model error from the irreducible error of
    # corrupted labels. Never used to fit anything. The detector is fitted on
    # the holdout's *training* window so the rows excluded from a holdout
    # metric are chosen without reading holdout labels for the threshold.
    split = holdout_split(train_all["date"], config.split.holdout_days)
    detector = TargetOutlierDetector(config.outliers).fit(
        train_all.iloc[split.train_index]
    )
    contaminated = detector.flag(train_all).to_numpy(dtype=bool)

    print(f"development data: {len(train_all):,} loads, "
          f"{train_all['date'].min():%Y-%m-%d}..{train_all['date'].max():%Y-%m-%d}")
    print(f"scored data:      {len(validation):,} loads, "
          f"{validation['date'].min():%Y-%m-%d}..{validation['date'].max():%Y-%m-%d}")
    print(f"\n{detector.report_.describe()}")
    print(f"estimated irreducible MAPE if the scored labels carry the same "
          f"contamination: ~{irreducible_mape(detector) * 100:.1f}%\n")

    sections: list[tuple[str, pd.DataFrame]] = []

    # -- headline: 61-day holdout, the same horizon as the scored window
    print(f"HOLDOUT  {split.describe()}")
    holdout_results = _evaluate_split(config, train_all, split, context, contaminated)
    for name, result in holdout_results.items():
        print(f"  {name:24s} {result.describe()}")
    sections.append(
        ("Headline holdout (61 days, mirrors the scored window)",
         metrics.summarise_folds(holdout_results))
    )

    # -- stability: rolling origin
    if args.cv:
        folds = rolling_origin_splits(
            train_all["date"],
            n_splits=config.split.n_rolling_splits,
            horizon_days=config.split.rolling_horizon_days,
            min_train_days=config.split.min_train_days,
        )
        print(f"\nROLLING ORIGIN  {len(folds)} expanding-window folds")
        fold_results: dict[str, metrics.Metrics] = {}
        for fold in folds:
            result = _evaluate_split(config, train_all, fold, context, contaminated)["model"]
            fold_results[f"{fold.name} ({fold.eval_start:%b %d}-{fold.eval_end:%b %d})"] = result
            print(f"  {fold.describe()}\n    {result.describe()}")
        sections.append(("Rolling-origin folds (expanding window)",
                         metrics.summarise_folds(fold_results)))

    # -- contrast: a shuffled split, to quantify its optimism
    if args.random_kfold:
        print("\nRANDOM K-FOLD (diagnostic contrast, never a selection metric)")
        random_results: dict[str, metrics.Metrics] = {}
        for fold in random_kfold_splits(len(train_all), n_splits=3, seed=config.seed):
            result = _evaluate_split(config, train_all, fold, context, contaminated)["model"]
            random_results[fold.name] = result
            print(f"  {fold.name}: {result.describe()}")
        sections.append(("Random K-fold (diagnostic only)",
                         metrics.summarise_folds(random_results)))

    # -- final fit on everything
    print("\nFINAL FIT on all development data")
    started = time.perf_counter()
    final = FreightRatePipeline(config).fit(train_all, context_frames=context)
    print(final.summary.describe())
    print(f"fit in {time.perf_counter() - started:.0f}s")

    # -- error structure on the holdout, for the report
    holdout_frame = train_all.iloc[split.eval_index].reset_index(drop=True)
    holdout_mask = contaminated[split.eval_index]
    holdout_train = train_all.iloc[split.train_index].reset_index(drop=True)
    holdout_pipeline = FreightRatePipeline(config).fit(
        holdout_train, context_frames=(*context, holdout_frame.drop(columns=[TARGET]))
    )
    predicted = holdout_pipeline.predict(holdout_frame)
    clean = holdout_frame.loc[~holdout_mask]
    clean_predicted = predicted[~holdout_mask]

    # -- generalisation gap: the same fitted model scored in-sample and out of
    #    sample, plus the boosting curve, so over/under-fitting is a number.
    train_mask = contaminated[split.train_index]
    in_sample = holdout_pipeline.predict(holdout_train)
    gap = {
        "train (in-sample, clean rows)": metrics.compute(
            holdout_train[TARGET].to_numpy(float)[~train_mask], in_sample[~train_mask]
        ),
        "holdout (out-of-sample, clean rows)": holdout_results["model"],
    }
    # scikit-learn stores -0.5 * mean squared error per round; convert to the
    # RMSE of the log-space residual so it reads against the 0.074 sigma of the
    # clean labels.
    estimator = holdout_pipeline.model.estimator
    to_rmse = lambda score: float(np.sqrt(2.0 * -score))  # noqa: E731
    curve = pd.DataFrame(
        {
            "rounds": [int(estimator.n_iter_)],
            "max_iter": [config.model.max_iter],
            "train_rmse_log": [to_rmse(estimator.train_score_[-1])],
            "valid_rmse_log": [to_rmse(estimator.validation_score_[-1])],
            "valid_rmse_log_best": [to_rmse(max(estimator.validation_score_))],
        },
        index=["boosting curve"],
    )
    sections.append(("Generalisation gap (holdout model)", metrics.summarise_folds(gap)))
    sections.append(("Early-stopping curve (log-space RMSE)", curve))
    sections.append(
        ("Holdout error by equipment",
         metrics.stratified(clean[TARGET], clean_predicted, clean["equipment"], "equipment"))
    )
    sections.append(
        ("Holdout error by distance decile",
         metrics.by_distance_decile(clean[TARGET], clean_predicted, clean["distance"]))
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Validation results\n\n")
        handle.write(
            "Development data spans 2025-01-01 to 2025-10-31; the scored data spans "
            "2025-11-01 to 2025-12-31. Every split below is temporal — training rows "
            "always predate evaluation rows.\n\n"
            f"{detector.report_.describe()}\n\n"
            "Rows with a corrupted label are excluded from the metrics unless a row is "
            "marked `model_all_rows`; their error is irreducible because nothing in the "
            "features predicts which loads are affected.\n\n"
        )
        handle.write("## Final model\n\n```\n" + final.summary.describe() + "\n```\n\n")
        for title, table in sections:
            handle.write(f"## {title}\n\n{markdown_table(table)}\n\n")
    print(f"\nwrote {output}")

    if not args.no_save:
        path = final.save(PATHS.model)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
