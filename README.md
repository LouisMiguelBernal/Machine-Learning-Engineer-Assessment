# Freight Rate Prediction

Solution to the Spotter Machine Learning Engineer assessment: predict `posted_rate`
for 12,000 freight loads in a window that sits entirely in the future relative to
the training data.

**Headline result — 61-day temporal holdout, the same horizon as the scored window,
with the holdout supplied its own inputs exactly as the scored file supplies them:**

| | MAE | RMSE | MAPE | Median APE | R² |
|---|---:|---:|---:|---:|---:|
| **This model, clean rows** | **$48.15** | **$73.47** | **2.01%** | **1.62%** | **0.9971** |
| This model, all rows (what a scorer sees) | $104.08 | $633.30 | 4.51% | 1.65% | 0.8278 |
| Log-linear distance baseline | $135.76 | $183.23 | 5.87% | 5.49% | 0.9822 |
| Median rate-per-mile baseline | $203.69 | $295.93 | 9.27% | 7.91% | 0.9537 |

"Clean rows" exclude the 1.5% of holdout loads whose label is multiplicatively
corrupted; that error is irreducible and is reported separately. The all-rows line
is the honest number for a scorer that cannot exclude them: its median APE is
essentially the clean one, so the gap between mean and median is the contamination,
not the model.

> **An earlier version of this README reported $38.94.** That number was measured
> with the holdout window's daily market index silently replaced by a constant, a
> configuration the deployed model never runs in. See *Validation* below and
> `docs/PROCESS.md`, Reversal 3. The submitted files did not change; the claims did.

---

## Quick start

```bash
python -m pip install -r requirements-solution.txt
```

```bash
python scripts/train.py --cv --random-kfold
```

```bash
python scripts/predict.py && python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

```bash
FREIGHT_REQUIRE_SUBMISSION=1 python -m pytest
```

```bash
python scripts/verify_guards.py
```

`score.py` is the vendor scorer, unmodified; it validates both outputs and writes
`scorer_results/candidate_december.png`. `requirements.txt` is the vendor file the
scorer ships with and is left byte-identical (it does not install the solution;
`requirements-solution.txt` does). The four commands are the release gate, in
that order: the deliverables are produced before pytest runs so the contract tests
have something to test, and the environment variable turns their "not built yet"
skips into failures.

Python 3.10+. Dependencies are numpy, pandas, scikit-learn, matplotlib, joblib and
python-docx for the report generator — no compiled extras to build.

## Repository layout

```
src/freight_rate/     the library — see "Architecture" below
scripts/              thin CLI entry points
tests/                129 tests: contract, data quality, leakage, evaluation fidelity, determinism
notebooks/            walkthrough.ipynb — narrative over the package, outputs stored
data/                 the four supplied CSVs (canonical names)
reports/              generated: EDA, validation, every ablation
docs/                 the written report and the process record
score.py              vendor scorer, byte-identical to the supplied copy
```

| Command | What it does |
|---|---|
| `python scripts/run_eda.py` | reproduces every empirical claim in this README |
| `python scripts/train.py --cv --random-kfold` | holdout, rolling folds, random-fold contrast, generalisation gap; saves the model |
| `python scripts/predict.py` | writes both deliverable files |
| `python scripts/ablate.py` | holdout ablation for every design decision |
| `python scripts/ablate.py --only level --folds --output reports/level_selection.md` | the level decision on both split geometries |
| `python scripts/cold_start.py` | leave-cities-out ablation |
| `python scripts/compare_estimators.py` | LightGBM, XGBoost and a linear floor against the shipped estimator |
| `python scripts/make_report.py` | builds `docs/freight_rate_report.docx` from the reports above; nothing is retyped |
| `python -m pytest` | the full test suite |
| `python scripts/verify_guards.py` | red-green: proves each guard fails when its safeguard is removed |

---

## What the data turned out to be

Development data covers 2025-01-01 to 2025-10-31 (48,000 loads); the scored data
covers 2025-11-01 to 2025-12-31 (12,000 loads). Reproduce all of this with
`python scripts/run_eda.py`.

**Rate is a power law in distance.** `log(rate) = 1.651 + 0.871·log(distance)`
gives R² = 0.936 on its own. Rate-per-mile falls monotonically from $2.72 at 226
miles to $1.92 at 2,606 miles — the long-haul discount. Everything else in the
model prices the residual around that spine.

**Equipment and weight are real but secondary.** Reefer carries a +0.074 premium
in log space, Flatbed +0.035, Dry Van −0.044. Rate-per-mile rises monotonically
across weight deciles ($2.12 → $2.29).

**1.4% of the training labels are corrupted.** 340 rows are scaled up by
2.2×–5.6× and 337 rows down to 0.17×–0.45×. The evidence that it is injected
corruption rather than a real market segment:

- Nothing predicts it — the rate is flat across equipment, month, distance,
  weight and city.
- The separation is total. Against a robust distance fit, 129 rows land in the
  `|log residual|` band 0.2–0.3 and 676 rows above 0.7, with exactly one row
  between 0.3 and 0.7. Genuine heavy tails do not leave a clean gap.
- Removing them takes the residual from σ = 0.169, kurtosis 51 to σ = 0.074,
  kurtosis −0.46.

Excluding them from training is worth **$10.41 MAE** on the corrected holdout.

**`quote_signal` correlates −0.04 with the target and is excluded — with a caveat.**
On the corrected splits it improves the 61-day holdout by $3.95 and worsens the
rolling-fold mean by $2.04, so it fails the adoption rule (below). It is kept
behind a flag; see *Known limitations*.

**`market_index` is weak per row and strong in aggregate.** Correlation with the
rate is +0.034 row-by-row, but +0.66 between the daily mean index and the daily
mean rate-per-mile. It is used as a daily series with 7- and 28-day rolling
context, never as a bare row value.

**Eight cities appear only in the scored data** — Allentown, Charlotte, Chicago,
Jackson, Knoxville, Laredo, Norfolk, San Diego — touching 12.1% of the loads that
must be predicted. No load has an unseen city at both ends. See *Cold start*.

### Data-quality handling

| Defect | Rows | Handling |
|---|---:|---|
| Corrupted `posted_rate` | 677 | excluded from training only, never from scoring |
| Negative `weight` | 292 | absolute value — \|weight\| matches the positive distribution and rate-per-mile by decile, so it is a sign-entry error |
| Missing `weight` | 300 / 165 | median by equipment + missingness flag |
| Missing `market_index` | 374 / 249 | daily index, then training-set median + flag |
| Absent coordinates | December | filled from the 72-city reference table |

**What is deliberately not "fixed":** 111 training rows have
`distance / great-circle > 1.6`, which looks like corrupt mileage. It is not.
They are all short hauls (70–146 miles) whose reported distance sits within 1.16×
the median distance of the same lane; the ratio is inflated because the synthetic
coordinates place nearby city pairs almost on top of each other. Rebuilding their
distance from the great circle raises their error from 15.4% to 31.2% MAPE. The
70-mile floor is likewise a billing rule, not a defect. Both are pinned by tests.

---

## Validation and split approach

**The split is temporal, because the problem is.** Every scored load is dated
after every training load, so a shuffled split answers the wrong question. Three
split types are implemented, with different jobs:

1. **61-day holdout (headline).** Train on 2025-01-01 → 2025-08-31, evaluate on
   2025-09-01 → 2025-10-31. The 61 days match the scored window exactly.
2. **Rolling origin (stability).** Five expanding-window folds, each training on
   everything before a cut date and evaluating the next 30 days.
3. **Leave-cities-out (cold start).** Eight cities removed from training entirely,
   scored only on the loads that touch them.

**Every evaluation window receives its own inputs.** The scored file carries
`market_index` and coordinates for November–December, so a deployed model has them
on the day. Each evaluation window is therefore fitted with its own rows — minus
the label, which `fit` refuses to accept — as a context frame. An earlier version
did not do this: the holdout's dates were absent from the daily index, every one
of its 61 days silently received the global median, and the market level was
evaluated as `trend + constant` while at deployment it runs as `trend + real
index`. The headline read $38.94; the same configuration measures $48.15. The
pipeline now raises rather than pricing a window it has no index for
(`DataCleaner.check_index_coverage`), and
`test_pipeline_refuses_a_window_without_index_coverage` pins it with a mutation.

`assert_no_temporal_leakage` runs on every generated split, and the test suite
asserts that fold evaluation windows are disjoint, that training windows expand
rather than slide, and that the fitted level model never reaches past its
training window.

**Rolling-origin results** (clean rows):

| Fold | Evaluation window | Train rows | MAE | MAPE | Bias |
|---|---|---:|---:|---:|---:|
| 1 | Jun 04 – Jul 03 | 24,521 | $51.30 | 2.06% | −$31 |
| 2 | Jul 04 – Aug 02 | 29,284 | $58.69 | 2.47% | +$48 |
| 3 | Aug 03 – Sep 01 | 34,027 | $30.75 | 1.35% | −$3 |
| 4 | Sep 02 – Oct 01 | 38,634 | $56.44 | 2.35% | −$47 |
| 5 | Oct 02 – Oct 31 | 43,309 | $36.90 | 1.58% | +$16 |

Mean $46.82, standard deviation $11.05. That spread is the honest uncertainty on
any single number in this document, and the bias column is where the error lives:
the model is rarely noisy, it is periodically off by a level.

**How optimistic would a random split have been?** Shuffled 3-fold on the same
data gives MAE $31.77 against $48.15 on the temporal holdout — 34% optimistic on
MAE. Part of that is shuffling; part is that a random fold's evaluation rows share
dates with training rows, so the market level is interpolated rather than
extrapolated. It is reported as a diagnostic and never used for selection.

**Does it overfit?** The holdout model scored on the rows it was fitted on gives
MAE $22.56 / MAPE 0.97% with zero bias; on the holdout, $48.15 / 2.01% with bias
−$36. Out-of-sample error is 2.1× in-sample, and almost all of the difference is
the level shift, not variance: the same model has near-zero bias in-sample and
−$36 out. Boosting runs to the 900-round cap; the internal validation curve is
flat well before it. The remedy for this kind of gap is a better level model, not
more regularisation — which is what the ablations below are about.

**Final fit uses all 48,000 loads.** The holdout exists to produce an honest
estimate, not to be permanently withheld.

---

## Model

`HistGradientBoostingRegressor` on `log(rate) − log(distance) − market_level`.

**Why gradient boosting.** The signal is tabular with genuine interactions
(rate-per-mile falls with distance, by a different amount per equipment type).
Native categorical support means 72 pickup and 72 delivery cities need no one-hot
expansion and no target encoding — the latter being the most likely source of
leakage in a project like this. It handles NaN natively, and it ships with
scikit-learn. A ridge regression on the very same feature matrix is $14.65 worse
on the six-window mean; that is what the trees earn (*Alternative estimators*).

**Why the target is decomposed.** Two components of the target are smooth,
dominant, and extend past the training window — exactly what a piecewise-constant
tree cannot represent beyond its last bin:

- `log(distance)`, which alone explains R² = 0.936. Putting it back into the
  tree costs $14.04.
- The daily market level, `0.5855 + 0.000222·t + 0.1440·market_index`
  (R² = 0.785 on the daily means). Dropping it costs $9.44.

Both are subtracted analytically and added back at prediction time, where the
linear part extrapolates by construction. Predictions are back-transformed with
Duan's smearing estimator rather than a bare `exp`.

### The level's index term: where the two geometries disagree

Once every window gets its real index, the holdout and the folds no longer agree
about the market-index term. From `reports/level_selection.md`:

| Level variant | Holdout MAE | Fold mean | 6-window mean |
|---|---:|---:|---:|
| **Trend + market_index (configured)** | $48.15 | **$46.82** | **$47.04** |
| Trend only | **$40.56** | $51.19 | $49.42 |
| Index term shrunk to 0.25 | $45.54 | $48.25 | $47.80 |
| Trend damped 0.5 | $51.71 | $46.51 | $47.38 |
| Trend frozen at the training edge | $55.97 | $46.67 | $48.22 |
| Market_index only, no trend | $81.15 | $54.59 | $59.01 |

The index term helps where the index swings hard (June–August) and hurts in
September–October, where rates held up while the index fell — fold 2 goes to
$80.61 without it. The adoption rule in this repo is that a change must improve
the holdout **and** not regress the folds; nothing passes both, so the configured
level keeps both terms and the disagreement is listed under *Known limitations*
rather than resolved by picking the window that flatters the model.
`level_market_index_shrink` exists as a measured hedge.

### Cold start

Twelve percent of scored loads touch a city with no training history. A gradient
boosted tree routes an unknown category down its missing-value branch — and if no
training row ever had a missing city, that branch was never fitted. The fix is
to train the branch: `city_dropout_rate = 0.10` blanks a random 10% of city values
in the training matrix only. Verified exactly: an unseen city and a blanked city
produce identical predictions on all 1,447 cold-start rows.

From `reports/cold_start.md` (three city samples) and `reports/feature_selection.md`:

| Variant | Cold-start MAE | Temporal holdout MAE |
|---|---:|---:|
| **Full model (dropout 0.10)** | **$40.84** | $48.15 |
| No city dropout | $51.26 | $47.69 |
| No coordinates | $49.21 | $47.88 |
| No city categoricals | $39.62 | $48.37 |
| Neither city feature | $56.32 | — |

On warm rows the city features are noise-level either way; on cold rows dropout
is worth $10.41 and coordinates $8.36, and coordinates only become load-bearing
once dropout exists — they are the fallback the model learns to use.

### Every decision, measured

From `python scripts/ablate.py` (holdout) and `--folds` runs, MAE against the
$48.15 / $46.82 configured model:

| Change | Δ holdout | Δ fold mean | Verdict |
|---|---:|---:|---|
| Level from `market_index` only, no trend | +$33.00 | +$7.77 | the trend is load-bearing |
| Target `log(rate)` instead of rate-per-mile | +$14.04 | — | distance belongs in the offset |
| Keep contaminated labels in training | +$10.41 | — | |
| Drop the level entirely | +$9.44 | — | |
| Freeze the trend at the training edge | +$7.82 | −$0.15 | carry the drift forward |
| Drop cyclic calendar features | −$0.87 | +$5.58 | fails the rule; kept |
| Add `quote_signal` | −$3.95 | +$2.04 | fails the rule; excluded |
| Trend only, no index term | −$7.59 | +$4.37 | fails the rule; see above |
| Drop city categoricals | +$0.22 | −$0.21 | noise; kept for lane signal |
| Disable city dropout | −$0.46 | −$0.71 | noise on warm rows, +$10.41 on cold |
| Drop coordinates | −$0.27 | −$0.28 | noise on warm rows, +$8.36 on cold |
| Add a raw time index | +$0.13 | −$0.78 | noise; the level carries time |

### Alternative estimators

From `python scripts/compare_estimators.py`, everything held fixed except the
regressor, MAE on the holdout plus five folds:

| Estimator | Holdout | 6-window mean | Δ mean | Fit per split |
|---|---:|---:|---:|---:|
| **scikit-learn HistGradientBoosting (shipped)** | $48.15 | $47.04 | — | 10s |
| LightGBM, matched params | $47.36 | $46.07 | −$0.96 | 5s |
| XGBoost, matched params | $47.45 | $47.63 | +$0.59 | 7s |
| Ridge on the same matrix (linear floor) | $57.69 | $61.69 | +$14.65 | 1s |

LightGBM's $0.96 sits inside the $11.05 fold spread and would cost a compiled
dependency. XGBoost is no better. The linear floor shows the trees earn about a
quarter of the error. What would actually move the number is a better level model,
not a different tree library: the alternatives worth trying next are a quantile
or conformal wrapper for intervals (a dispatcher wants "$840 ± 60"), and a
regime-aware level — a rolling re-estimate of the index coefficient — which is
the direct answer to the disagreement above.

---

## Architecture

```
config.py      frozen dataclasses; every path, seed and hyperparameter
schema.py      column contracts for all four CSVs, validated on load (dtypes too)
data.py        typed loaders and the two deliverable writers
geo.py         city -> (lat, lon) reference, haversine, circuity
cleaning.py    stateful cleaner: fit on train, apply everywhere; index-coverage guard
outliers.py    target contamination detection (training only — it reads the label)
level.py       the linear market level a tree cannot extrapolate
features.py    pure transform: cleaned frame -> model matrix
validation.py  temporal, rolling-origin and leave-cities-out splits
model.py       estimator factory, log-target wrapper, two baselines
metrics.py     MAE / RMSE / MAPE / median APE plus stratified breakdowns
pipeline.py    the one composition root
```

Two rules hold the design together. **One composition root**: training and
inference build the feature matrix through the same `FreightRatePipeline`, so
train/serve skew has no way in. **Fitted state is training only**: imputation
statistics, category levels, the contamination threshold and the level model are
all learned inside `fit` and frozen.

`context_frames` is the one deliberate exception, and it is narrow. The cleaner
may read *coordinates* and *`market_index`* from the scored frame and from an
evaluation window, because those are input features supplied for those rows. It
never reads a label from anywhere but the training set, and `fit` raises if a
context frame carries a target column. `build_matrix` raises if a frame's dates
are not covered by the daily index, so the silent-constant failure cannot recur.

## Testing

```bash
FREIGHT_REQUIRE_SUBMISSION=1 python -m pytest
```

129 tests. Five groups:

- **Submission contract** — every rule in the vendor `score.py`, re-derived from
  the assessment text rather than by importing the scorer. Plus a hash check that
  `score.py` and `requirements.txt` are byte-identical to the supplied copies.
  Without the environment variable these tests skip when the deliverables have
  not been built; with it, they fail.
- **Data quality** — one test per known defect, and one per *non*-defect pinning
  why it is left alone.
- **Leakage and evaluation fidelity** — no target-derived column reaches the
  model; predictions are unchanged when the target is scrambled or removed; an
  evaluation window receives its own daily index and the pipeline refuses one
  that has none; the imputation median never depends on which context was passed.
- **Determinism** — row order does not matter; two runs write byte-identical CSVs.
- **Units** — features, splits, metrics, the level model and the target transform.

### Proving the tests actually bite

```bash
python scripts/verify_guards.py
```

`tests/mutations.py` removes one real safeguard at a time — the `abs()` on
negative weights, contamination detection, the leakage assertion, the
`quote_signal` exclusion, the submission writer's positivity check, coordinates
for scored-only cities, and the index-coverage refusal — and requires the
matching test to go **red, then green**. Currently 7/7; see
`reports/guard_verification.md`. The script sets its own console to UTF-8; an
earlier version crashed on a stock Windows console before verifying anything.

This is mutation testing aimed at the safeguards rather than at the code as a
whole. The question it answers is not "does the test run" but "would the test
notice if the thing it protects disappeared". It earned its place on the first
run by exposing a weight-cleaning test that stayed green with the fix removed.

## Deliverables

| Item | Path |
|---|---|
| Predictions | `validation_predictions.csv` |
| December chart inputs | `data/december_chart_inputs.csv` |
| December chart | `scorer_results/candidate_december.png` |
| Written report | `docs/freight_rate_report.docx` (built by `scripts/make_report.py`) |
| Validation detail | `reports/validation.md` |
| Ablations | `reports/model_selection.md`, `reports/level_selection.md`, `reports/feature_selection.md`, `reports/cold_start.md`, `reports/estimator_comparison.md` |
| EDA | `reports/eda_summary.md`, `reports/eda_figures.png` |
| Process record | `docs/PROCESS.md` |

## Known limitations

- **The level's index term is contested.** The most recent 61 days favour a
  trend-only level by $7.59; the five rolling folds favour keeping the index by
  $4.37. The configured model keeps it because nothing wins on both geometries,
  but November–December's index is low (0.92, 0.93), much like the window where
  the index term hurt. If the scored labels behave like September–October, the
  trend-only variant would have been about $7 better.
- **The market-level trend is extrapolated linearly 61 days past its fitted
  range.** A genuine trend break would not be caught.
- **`quote_signal` and the cyclic calendar features** each split the two
  geometries the same way as the index term. Neither is a settled decision.
- **The headline excludes label-corrupted rows** chosen by a detector fitted on
  the training window. A scorer that includes them sees the all-rows line.
- **Cold-start performance** is measured on cities held out from *this* data.
- **No hyperparameter search was run.** The estimator comparison says the tree
  library does not matter; the level model does.
