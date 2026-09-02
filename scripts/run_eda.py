"""Reproduce every empirical claim the design rests on.

Each section prints a number that is quoted somewhere in the README, a docstring
or the report. If a claim cannot be reproduced here, it should not be in the
write-up.

    python scripts/run_eda.py

Writes ``reports/eda_summary.md`` and ``reports/eda_figures.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight_rate import load_december_inputs, load_train, load_validation  # noqa: E402
from freight_rate.config import PATHS  # noqa: E402
from freight_rate.geo import CityReference, haversine_miles  # noqa: E402
from freight_rate.outliers import TargetOutlierDetector  # noqa: E402
from freight_rate.reporting import markdown_table  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/eda_summary.md")
    parser.add_argument("--figures", default="reports/eda_figures.png")
    args = parser.parse_args()
    PATHS.ensure_dirs()

    train = load_train()
    validation = load_validation()
    december = load_december_inputs()
    blocks: list[tuple[str, str]] = []

    # ---------------------------------------------------------------- shape
    section("1. Shape and coverage")
    print(f"train      {train.shape}  {train['date'].min():%Y-%m-%d}..{train['date'].max():%Y-%m-%d}")
    print(f"validation {validation.shape}  "
          f"{validation['date'].min():%Y-%m-%d}..{validation['date'].max():%Y-%m-%d}")
    print(f"december   {december.shape}")
    print(f"\nthe scored window is strictly later than every training row: "
          f"{train['date'].max() < validation['date'].min()}")
    scored_days = (validation["date"].max() - validation["date"].min()).days + 1
    print(f"scored horizon: {scored_days} days -> the holdout is sized to match")

    coverage = pd.DataFrame(
        {
            "rows": [len(train), len(validation), len(december)],
            "first_date": [train["date"].min(), validation["date"].min(), december["date"].min()],
            "last_date": [train["date"].max(), validation["date"].max(), december["date"].max()],
            "has_target": [True, False, False],
        },
        index=["train_test", "validation", "december_chart"],
    )
    blocks.append(("Coverage", markdown_table(coverage, index_label="file")))

    # ------------------------------------------------------- missing values
    section("2. Missing and invalid values")
    quality = pd.DataFrame(
        {
            "train_missing": train.isna().sum(),
            "validation_missing": validation.reindex(columns=train.columns).isna().sum(),
        }
    )
    quality = quality[quality.sum(axis=1) > 0]
    print(quality.to_string())
    print(f"\nnegative weight rows: {(train['weight'] < 0).sum()}")
    print(f"weight clipped at the top: {(train['weight'] == train['weight'].max()).sum()} rows "
          f"sit exactly at {train['weight'].max():,.0f} lb")
    negative = train[train["weight"] < 0]
    positive = train[train["weight"] > 0]
    print(f"|negative weight| mean {negative['weight'].abs().mean():,.0f} vs "
          f"positive mean {positive['weight'].mean():,.0f} "
          "-> a sign-entry error, so the magnitude is kept")
    blocks.append(("Missing values", markdown_table(quality, index_label="column")))

    # ------------------------------------------------------------ geography
    section("3. Geography")
    reference = CityReference.from_frames(train, validation)
    trained_cities = set(train["pickup"]) | set(train["delivery"])
    unseen = sorted((set(validation["pickup"]) | set(validation["delivery"])) - trained_cities)
    touched = validation["pickup"].isin(unseen) | validation["delivery"].isin(unseen)
    print(f"cities: {len(reference)} across both files, {len(trained_cities)} in training")
    print(f"every city has exactly one coordinate pair (asserted by CityReference)")
    print(f"\ncities appearing only in the scored data: {len(unseen)} -> {', '.join(unseen)}")
    print(f"scored rows touching one: {touched.sum():,} ({touched.mean():.1%})")

    great_circle = haversine_miles(
        train["pickup_lat"], train["pickup_lon"], train["delivery_lat"], train["delivery_lon"]
    )
    ratio = train["distance"] / great_circle
    print(f"\ncorr(great circle, reported distance) = "
          f"{np.corrcoef(great_circle, train['distance'])[0, 1]:.5f}")
    print(f"median circuity = {ratio.median():.3f}")
    suspicious = train[ratio > 1.6]
    print(f"rows with circuity > 1.6: {len(suspicious)}; their distance runs "
          f"{suspicious['distance'].min():.0f}-{suspicious['distance'].max():.0f} miles "
          "-> short hauls on a coarse coordinate grid, not corrupt mileage")
    print(f"distance floor: {train['distance'].min():.0f} miles, "
          f"{(train['distance'] == train['distance'].min()).sum()} rows exactly on it")

    geography = pd.DataFrame(
        {
            "value": [
                len(reference),
                len(trained_cities),
                len(unseen),
                f"{touched.mean():.1%}",
                round(float(np.corrcoef(great_circle, train["distance"])[0, 1]), 5),
                round(float(ratio.median()), 3),
                len(suspicious),
                float(train["distance"].min()),
            ]
        },
        index=[
            "cities (both files)",
            "cities in training",
            "cities only in scored data",
            "scored rows touching an unseen city",
            "corr(great circle, distance)",
            "median circuity",
            "rows with circuity > 1.6",
            "distance floor (miles)",
        ],
    )
    blocks.append(("Geography", markdown_table(geography, index_label="measure")))

    # -------------------------------------------------------- target shape
    section("4. Target structure")
    log_distance = np.log(train["distance"])
    log_rate = np.log(train["posted_rate"])
    slope, intercept = np.polyfit(log_distance, log_rate, 1)
    r_squared = float(np.corrcoef(log_distance, log_rate)[0, 1] ** 2)
    residual = log_rate - (intercept + slope * log_distance)
    print(f"log(rate) = {intercept:.4f} + {slope:.4f} * log(distance)   R2 = {r_squared:.4f}")
    print(f"residual sigma = {residual.std():.4f}, skew = {residual.skew():.2f}, "
          f"kurtosis = {residual.kurt():.1f}")

    train = train.assign(rate_per_mile=train["posted_rate"] / train["distance"])
    decile = train.assign(
        decile=pd.qcut(train["distance"], 10, labels=False)
    ).groupby("decile").agg(
        distance=("distance", "mean"),
        rate=("posted_rate", "mean"),
        rate_per_mile=("rate_per_mile", "mean"),
    )
    print("\nrate per mile falls with distance (the long-haul discount):")
    print(decile.round(3).to_string())
    blocks.append(("Rate per mile by distance decile", markdown_table(decile.round(3))))

    equipment = train.groupby("equipment").agg(
        loads=("posted_rate", "size"),
        rate_per_mile=("rate_per_mile", "mean"),
        log_residual=("posted_rate", lambda s: residual.loc[s.index].mean()),
    )
    print("\nequipment premium:")
    print(equipment.round(4).to_string())
    blocks.append(("Equipment premium", markdown_table(equipment.round(4))))

    # ------------------------------------------------------- contamination
    section("5. Target contamination")
    detector = TargetOutlierDetector().fit(train)
    print(detector.report_.describe())
    deviation = (residual - np.median(residual)).abs()
    bands = pd.DataFrame(
        {
            "rows": [
                int(((deviation >= lo) & (deviation < hi)).sum())
                for lo, hi in [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 3.0)]
            ]
        },
        index=["0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.7", "0.7+"],
    )
    print("\n|log residual| distribution — note the empty band that separates the two populations:")
    print(bands.to_string())
    flagged = detector.flag(train)
    by_equipment = flagged.groupby(train["equipment"]).mean()
    print(f"\ncontamination rate by equipment: "
          f"{ {k: round(v, 4) for k, v in by_equipment.items()} }")
    print("-> flat across equipment, so nothing in the features predicts it")
    blocks.append(("Log-residual bands", markdown_table(bands, index_label="abs log-residual band")))

    # -------------------------------------------------------- market signal
    section("6. Market signals")
    print(f"corr(quote_signal, posted_rate)  = "
          f"{np.corrcoef(train['quote_signal'], train['posted_rate'])[0, 1]:+.4f}")
    indexed = train.dropna(subset=["market_index"])
    print(f"corr(market_index, posted_rate)  = "
          f"{np.corrcoef(indexed['market_index'], indexed['posted_rate'])[0, 1]:+.4f}")

    clean = train.loc[~flagged].copy()
    clean["log_rpm"] = np.log(clean["posted_rate"]) - np.log(clean["distance"])
    daily = clean.groupby("date").agg(log_rpm=("log_rpm", "mean"), market=("market_index", "mean"))
    daily["t"] = (daily.index - daily.index.min()).days
    print(f"corr(daily mean rate-per-mile, daily market_index) = "
          f"{daily['log_rpm'].corr(daily['market']):+.4f}")
    print("-> weak per row, real in aggregate; use the daily series, not the raw value")

    design = np.column_stack([np.ones(len(daily)), daily["t"], daily["market"]])
    coefficients, *_ = np.linalg.lstsq(design, daily["log_rpm"], rcond=None)
    fitted = design @ coefficients
    total = float(np.var(daily["log_rpm"]))
    print(f"\ndaily level = {coefficients[0]:.4f} + {coefficients[1]:.6f}*t "
          f"+ {coefficients[2]:.4f}*market_index")
    print(f"  R2 = {1 - np.var(daily['log_rpm'] - fitted) / total:.4f}   "
          f"trend over the training window = "
          f"{100 * (np.exp(coefficients[1] * daily['t'].max()) - 1):.1f}%")
    for name, columns in {"trend only": [0, 1], "market_index only": [0, 2], "both": [0, 1, 2]}.items():
        subset = design[:, columns]
        beta, *_ = np.linalg.lstsq(subset, daily["log_rpm"], rcond=None)
        print(f"  {name:18s} R2 = {1 - np.var(daily['log_rpm'] - subset @ beta) / total:.4f}")
    print("-> both terms are load-bearing; neither substitutes for the other")

    signals = pd.DataFrame(
        {
            "value": [
                round(float(np.corrcoef(train["quote_signal"], train["posted_rate"])[0, 1]), 4),
                round(float(np.corrcoef(indexed["market_index"], indexed["posted_rate"])[0, 1]), 4),
                round(float(daily["log_rpm"].corr(daily["market"])), 4),
                round(float(coefficients[1]), 6),
                round(float(coefficients[2]), 4),
                round(float(100 * (np.exp(coefficients[1] * daily["t"].max()) - 1)), 2),
            ]
        },
        index=[
            "corr(quote_signal, rate)",
            "corr(market_index, rate) per row",
            "corr(market_index, rate-per-mile) daily",
            "level trend per day (log)",
            "level market_index coefficient",
            "trend across the training window (%)",
        ],
    )
    blocks.append(("Market signals", markdown_table(signals, index_label="measure")))

    # -------------------------------------------------------------- figures
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0][0].scatter(train["distance"], train["posted_rate"], s=1, alpha=0.05, color="#064A56")
    grid = np.linspace(train["distance"].min(), train["distance"].max(), 200)
    axes[0][0].plot(grid, np.exp(intercept + slope * np.log(grid)), color="#C1440E", linewidth=2)
    axes[0][0].set(xlabel="distance (miles)", ylabel="posted rate ($)",
                   title=f"Rate vs distance (log-linear fit, R2={r_squared:.3f})")

    axes[0][1].hist(residual, bins=120, color="#064A56")
    axes[0][1].set_yscale("log")
    axes[0][1].axvline(detector.threshold_, color="#C1440E", linestyle="--", linewidth=1)
    axes[0][1].axvline(-detector.threshold_, color="#C1440E", linestyle="--", linewidth=1)
    axes[0][1].set(xlabel="log residual", ylabel="rows (log scale)",
                   title="Contaminated labels sit beyond an empty band")

    axes[1][0].plot(daily.index, daily["log_rpm"], color="#064A56", linewidth=1, label="observed")
    axes[1][0].plot(daily.index, fitted, color="#C1440E", linewidth=2, label="trend + market index")
    axes[1][0].legend(frameon=False)
    axes[1][0].tick_params(axis="x", rotation=30)
    axes[1][0].set(ylabel="mean log rate-per-mile", title="Daily market level")

    axes[1][1].plot(decile["distance"], decile["rate_per_mile"], marker="o", color="#064A56")
    axes[1][1].set(xlabel="mean distance (miles)", ylabel="mean rate per mile ($)",
                   title="Long-haul discount")

    for row in axes:
        for axis in row:
            axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.figures, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {args.figures}")

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Exploratory data analysis\n\n")
        handle.write(
            "Generated by `python scripts/run_eda.py`. Every number quoted in the README, "
            "the report or a module docstring is reproduced here.\n\n"
            f"![EDA figures]({Path(args.figures).name})\n\n"
        )
        for title, body in blocks:
            handle.write(f"## {title}\n\n{body}\n\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
