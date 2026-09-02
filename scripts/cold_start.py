"""Cold-start ablation: does the model survive a city it has never seen?

12.1% of the scored loads touch one of 8 cities that never appear in the training
data. No temporal split can measure that — the held-out months contain no new
cities — so this script removes 8 cities from training outright and scores only
the loads that touch them.

The question it settles: are the coordinate features earning their place? On the
temporal holdout, dropping them costs nothing, because every city there is
already known. If they matter anywhere, it is here.

    python scripts/cold_start.py
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight_rate import load_train, load_validation  # noqa: E402
from freight_rate import metrics  # noqa: E402
from freight_rate.config import FeatureConfig, RunConfig  # noqa: E402
from freight_rate.outliers import TargetOutlierDetector  # noqa: E402
from freight_rate.pipeline import FreightRatePipeline  # noqa: E402
from freight_rate.reporting import markdown_table  # noqa: E402
from freight_rate.schema import TARGET  # noqa: E402
from freight_rate.validation import leave_cities_out_split  # noqa: E402

VARIANTS: dict[str, FeatureConfig] = {
    "full model (dropout 0.10)": FeatureConfig(),
    "no city dropout": FeatureConfig(city_dropout_rate=0.0),
    "no coordinates": FeatureConfig(use_geo_coordinates=False),
    "no city categoricals": FeatureConfig(use_city_categoricals=False),
    "neither (distance only geography)": FeatureConfig(
        use_geo_coordinates=False, use_city_categoricals=False
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cities", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3, help="city samples to average over")
    parser.add_argument("--output", default="reports/cold_start.md")
    args = parser.parse_args()

    train_all = load_train()
    validation = load_validation()
    contaminated = TargetOutlierDetector().fit(train_all).flag(train_all).to_numpy(dtype=bool)

    rows: list[dict] = []
    for seed in range(args.seeds):
        split, held_out = leave_cities_out_split(train_all, args.n_cities, seed=seed)
        train = train_all.iloc[split.train_index].reset_index(drop=True)
        evaluate = train_all.iloc[split.eval_index].reset_index(drop=True)
        mask = contaminated[split.eval_index]
        actual = evaluate[TARGET].to_numpy(dtype=float)

        print(
            f"\nseed {seed}: held out {', '.join(held_out)}\n"
            f"  train {len(train):,} rows / evaluate {len(evaluate):,} rows "
            f"({len(evaluate) / len(train_all):.1%} of the data)"
        )
        for name, features in VARIANTS.items():
            config = dataclasses.replace(RunConfig(), features=features)
            pipeline = FreightRatePipeline(config).fit(
                train, context_frames=(validation, evaluate.drop(columns=[TARGET]))
            )
            predicted = pipeline.predict(evaluate)
            result = metrics.compute(actual[~mask], predicted[~mask])
            rows.append({"seed": seed, "variant": name, **result.as_dict()})
            print(f"    {name:34s} {result.describe()}")

    frame = pd.DataFrame(rows)
    summary = frame.groupby("variant")[["mae", "rmse", "mape", "median_ape", "bias"]].mean()
    summary = summary.reindex(list(VARIANTS))
    reference = summary.iloc[0]["mae"]
    summary.insert(0, "dMAE_vs_full", summary["mae"] - reference)

    print("\nmean across seeds:")
    print(summary.round(3).to_string())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Cold-start ablation — unseen cities\n\n")
        handle.write(
            f"{args.n_cities} cities removed from training entirely, evaluated only on the loads "
            f"that touch them, averaged over {args.seeds} different city samples. This mirrors the "
            "8 cities that appear only in the scored data and touch 12.1% of the loads that must "
            "be predicted.\n\nRows with corrupted labels are excluded.\n\n"
        )
        handle.write(markdown_table(summary.round(4), index_label="variant"))
        handle.write("\n\n## Per-seed detail\n\n")
        handle.write(markdown_table(frame.round(4).set_index(["seed", "variant"])))
        handle.write("\n")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
