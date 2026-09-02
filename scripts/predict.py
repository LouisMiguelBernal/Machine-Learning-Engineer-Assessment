"""Produce both deliverable files.

    python scripts/predict.py                 # fit, then write both outputs
    python scripts/predict.py --model artifacts/model.joblib   # reuse a saved fit

Writes:
  * ``validation_predictions.csv``          — load_id, predicted_rate for 12,000 loads
  * ``data/december_chart_inputs.csv``      — the same seven columns, predicted_rate filled

The December rows carry only ``pickup, delivery, distance, equipment, weight,
date``. Everything else the model needs is reconstructed from state fitted during
training: coordinates from the city reference table, and ``market_index`` from the
daily index series. Those are inference-time inputs, not labels — the same
information a deployed model would have on the day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight_rate import (  # noqa: E402
    load_december_inputs,
    load_template,
    load_train,
    load_validation,
    write_december,
    write_submission,
)
from freight_rate.config import PATHS, RunConfig  # noqa: E402
from freight_rate.pipeline import FreightRatePipeline  # noqa: E402


def _align_to_template(
    validation: pd.DataFrame, predicted: np.ndarray, template: pd.DataFrame
) -> np.ndarray:
    """Reorder predictions into template order, by load_id.

    The two files happen to share a row order, but the scorer checks set
    equality on ``load_id``, so the join is done explicitly rather than trusted.
    """
    lookup = pd.Series(predicted, index=validation["load_id"].astype(str))
    aligned = lookup.reindex(template["load_id"].astype(str))
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()[:5]
        raise ValueError(f"no prediction for {aligned.isna().sum()} template ids, e.g. {missing}")
    return aligned.to_numpy(dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="path to a saved pipeline; refits if omitted")
    parser.add_argument("--submission", default=None, help="override the submission path")
    args = parser.parse_args()

    PATHS.ensure_dirs()
    validation = load_validation()
    template = load_template()
    december = load_december_inputs()

    if args.model:
        pipeline = FreightRatePipeline.load(Path(args.model))
        print(f"loaded model from {args.model}")
    else:
        train = load_train()
        pipeline = FreightRatePipeline(RunConfig()).fit(train, context_frames=(validation,))
        print(pipeline.summary.describe())

    predicted = pipeline.predict(validation)
    aligned = _align_to_template(validation, predicted, template)
    submission = write_submission(
        aligned, template, Path(args.submission) if args.submission else None
    )
    print(
        f"\nwrote {submission}: {len(aligned):,} rows, "
        f"min={aligned.min():,.2f} median={np.median(aligned):,.2f} max={aligned.max():,.2f}"
    )

    december = december.copy()
    december["predicted_rate"] = pipeline.predict(december)
    path = write_december(december)
    rates = december["predicted_rate"].to_numpy(dtype=float)
    print(
        f"wrote {path}: 31 December rows, "
        f"min={rates.min():,.2f} max={rates.max():,.2f} "
        f"spread={100 * (rates.max() / rates.min() - 1):.1f}%"
    )
    print("\nNext:")
    print("  python score.py --predictions validation_predictions.csv "
          "--december-predictions data/december_chart_inputs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
