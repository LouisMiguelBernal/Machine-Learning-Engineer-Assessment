# How this was actually built

An honest record of the working process, including the two things I got wrong and
had to reverse. Written because "here is the finished model" hides the part that
matters — the loop that produced it.

## Short answer: no, the experiments were not run in a notebook

The exploration was done **inline in a Python shell**, one throwaway script per
question. Findings that survived were then written into `scripts/run_eda.py` so
they are reproducible, and into module docstrings so the reasoning sits next to
the code it justifies.

`notebooks/walkthrough.ipynb` was added **last**, and it is a *presentation*
layer — it imports from `src/freight_rate/` and re-runs the story end to end. It
is not where the work happened, and nothing depends on it.

### Why not notebook-first

Both are legitimate. The split is about what the artefact has to survive:

| | Notebook | Package + scripts |
|---|---|---|
| Iteration speed | fastest | slower |
| Can be unit-tested | no | yes |
| Enforces execution order | no | yes |
| Reviewer can reproduce | maybe | guaranteed |
| Survives being run twice | often not | yes |

A notebook is for **finding out**. A package is for **being sure**. The
deliverable here is "an accessible GitHub repository containing your solution
code" that a reviewer will run — so the durable form had to be a package. A
notebook can't hold a leakage guard that fails a build, and out-of-order cell
execution is exactly the failure mode that produces a number you can't reproduce.

For a research spike with no deliverable, I would have stayed in a notebook.

---

## The five phases

### Phase 1 — Read the brief, verify the inputs

Before any code: extracted the PDF text, read `readme.md` and `score.py`, and
diffed what the two documents ask for. They disagree — the PDF's Submit section
never mentions the December step, only the readme does. Implemented it, because
the scorer enforces it.

Also normalised filenames. The email delivered `train-test.csv` and
`score.py.txt`; the readme references `data/train_test.csv`. Originals preserved
in `.assessment_original/` and the working copies placed under `data/`.

**Rule applied:** read the scorer before writing the model. It is the only
unambiguous statement of what "correct output" means.

### Phase 2 — Exploration, one question at a time

Roughly a dozen throwaway scripts, each answering a single question and printing
numbers. The sequence, and what each one changed:

| Question | Answer | Consequence |
|---|---|---|
| What are the date ranges? | train Jan–Oct, scored Nov–Dec, zero overlap | split must be temporal |
| Does distance explain rate? | R² 0.936 log-log | subtract `log(distance)` from the target |
| Is `quote_signal` useful? | r = −0.04 | excluded |
| Is `market_index` useful? | r = 0.034 per row, 0.66 daily | aggregate to a daily series |
| Why is kurtosis 51? | 677 corrupted labels | drop from training — biggest single win |
| Are the odd distances corrupt? | **no** | reversed a planned "fix" |
| Do train and scored share cities? | **no, 8 are new** | added city dropout |

The last two are the ones worth noting, because I had already decided wrongly on
both.

### Phase 3 — Two reversals

**Reversal 1: the distance "repair" I nearly shipped.**
111 rows have `distance / great-circle > 1.6`. I wrote a cleaner that rebuilt
their distance from the great circle, and had it in `cleaning.py` before testing
whether it helped. It did not: those rows are short hauls of 70–146 miles whose
reported distance sits within 1.16× the median for the same lane. Repairing them
took their error from **15.4% to 31.1% MAPE**. The coordinates are coarse, not
the mileage — New Orleans to Shreveport great-circles to 7 miles.

The repair was deleted. A test now pins the non-repair so nobody re-adds it.

**Reversal 2: the −$56 bias.**
The first end-to-end model scored MAE $61.44 with **bias −$56.47** — 92% of the
error was a systematic level shift, not noise. Rather than tune, I decomposed the
daily mean of `log(rate) − log(distance)` and found trend and `market_index` are
confounded inside Jan–Aug but separate afterwards. A model given only the index
reads November's low index as "the low-rate start of the year".

That produced `level.py` and took the holdout from **$61.44 to $38.94**.

**Rule applied:** when the error is mostly bias, the model is missing a term.
Tuning a biased model just fits the bias more confidently.

**Reversal 3: the holdout that was not measuring the deployed model.**
Caught late, on a re-read of the validation path after everything above was
written. The temporal validation fitted the daily market-index series from the
training window plus the scored file, and the holdout window
(September-October) sat in neither. Every one of its 61 days silently received the global-median index, so
the level model was evaluated as `trend + constant` while at deployment it runs
as `trend + real index`. The headline read **$38.94**; measured with the
window's own index supplied, as the scored file supplies it for
November-December, the same configuration scores **$48.15**.

Three things changed. Every evaluation script now passes the window's own inputs
(never its label) as a context frame. `DataCleaner.check_index_coverage` raises
instead of falling back, and a test plus a mutation pin it. And selection now
uses both temporal geometries, because once the index was real the 61-day
holdout and the rolling folds disagreed on the index term (trend-only wins the
holdout by $7.59 and loses the fold mean by $4.37). The rule applied - a change
is adopted only if it improves the holdout *and* does not regress the folds -
kept the trend-plus-index level, and the disagreement is listed as a limitation.

**Rule applied:** the evaluation window must receive exactly the inputs the
scored window will receive. A silent fallback in an imputer is a validation bug
waiting to happen; make it raise.

### Phase 4 — Build the package, then measure every decision

Modules written in dependency order — `config` → `schema` → `geo` → `cleaning` →
`features` → `model` → `pipeline` — with tests alongside, not after.

Then `scripts/ablate.py`: every choice that could have gone the other way is a
config flag, and each is re-run on the holdout. Twelve variants, ~15 minutes of
compute, and the results are committed to `reports/model_selection.md`. Nothing
in the README is asserted from intuition.

The cold-start finding needed a split that did not exist — no temporal split can
measure unseen cities, because the held-out months contain no new ones. So
`leave_cities_out_split` was added and `scripts/cold_start.py` swept city dropout
over {0, 0.02, 0.05, 0.10, 0.20} against both axes. 0.10 won on the blend
weighted by the real 12.1% unseen-city share.

### Phase 5 — Verification, then the write-up

The four-command gate, run in a **clean copy of the repo**, not the working
directory:

```bash
python scripts/train.py --cv --random-kfold
python scripts/predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
FREIGHT_REQUIRE_SUBMISSION=1 python -m pytest
python scripts/verify_guards.py
```

The deliverables are produced before pytest runs, and the environment variable
turns the contract tests' "not built yet" skips into failures. Run in the other
order, pytest reports green with nine contract tests skipped, which is not a
result.

The clean copy caught a real bug the working directory hid: the vendor-integrity
tests read `score.py` from `.assessment_original/`, which is git-ignored. On a
fresh clone those tests would have been **silently unrunnable** — the single most
important check in the suite, quietly absent. Now pinned to SHA-256 constants,
with a negative control proving the check fails when `score.py` is edited.

---

## The loop, in one line

> State the hypothesis → make it a flag → run it on the holdout → keep it only if
> the holdout improves and rolling-origin does not regress → write the number down.

Six things were tried and rejected by that loop: `quote_signal`, a raw time
index, dropping coordinates, dropping city categoricals, damping the trend, and
freezing the trend. Each rejection is a row in `reports/model_selection.md` with
its cost.

## What each tool was for

| Stage | Tool | Output |
|---|---|---|
| Exploration | throwaway shell scripts | numbers, discarded |
| Codifying findings | `scripts/run_eda.py` | `reports/eda_summary.md`, figures |
| Deciding | `scripts/ablate.py` (`--folds` for both geometries), `scripts/cold_start.py`, `scripts/compare_estimators.py` | `reports/model_selection.md`, `reports/level_selection.md`, `reports/feature_selection.md`, `reports/cold_start.md`, `reports/estimator_comparison.md` |
| Training | `scripts/train.py --cv` | `reports/validation.md`, `artifacts/model.joblib` |
| Producing deliverables | `scripts/predict.py` | `validation_predictions.csv`, December CSV |
| Validating output | vendor `score.py` | `candidate_december.png` |
| Guarding all of it | `pytest`, `scripts/verify_guards.py` | the test suite and its red-green proof |
| Explaining it | `notebooks/walkthrough.ipynb`, `scripts/make_report.py` | notebook, DOCX report |

## Where the accuracy came from

About $204 → $48 MAE on the corrected holdout. Almost none of it is the algorithm:

| Step | MAE |
|---|---|
| Median rate-per-mile | ~$204 |
| Log-linear on distance | ~$136 |
| + gradient boosting, all features | ~$68 |
| + drop corrupted labels | ~$58 |
| + market level model | **~$48** |

Figures re-measured after Reversal 3; the exact current values are in
`reports/validation.md` and `reports/model_selection.md`.

Swapping the estimator for LightGBM would move this by cents. The gap is data
quality and target decomposition — which is why the exploration phase got more
time than the modelling phase.

## What I would do with more time

- Quantile regression for a prediction interval. A dispatcher wants "$840 ± 60",
  not a point estimate.
- Lane-level effects with proper shrinkage, rather than relying on the tree's
  categorical splits.
- A drift monitor on the market level. The trend is extrapolated 61 days; in
  production that assumption needs a tripwire, not a comment.
