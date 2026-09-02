# Validation results

Development data spans 2025-01-01 to 2025-10-31; the scored data spans 2025-11-01 to 2025-12-31. Every split below is temporal — training rows always predate evaluation rows.

target contamination: 533 / 38,477 rows (1.39%) - 267 inflated, 266 deflated; log-residual sigma 0.1681 -> 0.0747 (threshold 0.392)

Rows with a corrupted label are excluded from the metrics unless a row is marked `model_all_rows`; their error is irreducible because nothing in the features predicts which loads are affected.

## Final model

```
fitted on 47,323 of 48,000 rows [2025-01-01..2025-10-31]
features: 32 (3 categorical: equipment, pickup, delivery)
smearing factor: 1.00011
target contamination: 677 / 48,000 rows (1.41%) - 340 inflated, 337 deflated; log-residual sigma 0.1689 -> 0.0738 (threshold 0.391)
level = 0.5855 + 0.000222*t + 0.1440*market_index  (R2=0.7849, daily sigma=0.0153, 304 days, fit ends 2025-10-31)
```

## Headline holdout (61 days, mirrors the scored window)

| index                  |     n |     mae |    rmse |  mape | median_ape |     r2 |    bias |
|:-----------------------|------:|--------:|--------:|------:|-----------:|-------:|--------:|
| model                  | 9,379 |  48.147 |  73.465 | 2.011 |      1.618 | 0.9971 | -36.436 |
| model_all_rows         | 9,523 | 104.080 | 633.299 | 4.509 |      1.647 | 0.8278 | -70.035 |
| baseline_log_linear    | 9,379 | 135.756 | 183.230 | 5.875 |      5.492 | 0.9822 |   7.117 |
| baseline_rate_per_mile | 9,379 | 203.687 | 295.927 | 9.267 |      7.905 | 0.9537 |  71.972 |
| mean                   | 9,415 | 122.918 | 296.480 | 5.415 |      4.166 | 0.9402 |  -6.846 |

## Rolling-origin folds (expanding window)

| index                  |     n |    mae |   rmse |  mape | median_ape |     r2 |    bias |
|:-----------------------|------:|-------:|-------:|------:|-----------:|-------:|--------:|
| fold_1 (Jun 04-Jul 03) | 4,700 | 51.298 | 77.697 | 2.059 |      1.674 | 0.9970 | -30.842 |
| fold_2 (Jul 04-Aug 02) | 4,687 | 58.692 | 85.361 | 2.467 |      2.062 | 0.9960 |  47.997 |
| fold_3 (Aug 03-Sep 01) | 4,539 | 30.752 | 44.224 | 1.347 |      1.139 | 0.9989 |  -3.014 |
| fold_4 (Sep 02-Oct 01) | 4,610 | 56.440 | 85.414 | 2.353 |      1.926 | 0.9962 | -46.723 |
| fold_5 (Oct 02-Oct 31) | 4,613 | 36.895 | 54.018 | 1.581 |      1.309 | 0.9984 |  15.944 |
| mean                   | 4,629 | 46.815 | 69.343 | 1.961 |      1.622 | 0.9973 |  -3.328 |

## Random K-fold (diagnostic only)

| index         |      n |    mae |   rmse |  mape | median_ape |     r2 |    bias |
|:--------------|-------:|-------:|-------:|------:|-----------:|-------:|--------:|
| random_fold_1 | 15,776 | 31.779 | 46.386 | 1.355 |      1.141 | 0.9988 | -0.6683 |
| random_fold_2 | 15,778 | 31.849 | 46.150 | 1.361 |      1.134 | 0.9988 |  0.5731 |
| random_fold_3 | 15,769 | 31.666 | 45.993 | 1.366 |      1.138 | 0.9988 |  0.2254 |
| mean          | 15,774 | 31.765 | 46.176 | 1.361 |      1.138 | 0.9988 |  0.0434 |

## Generalisation gap (holdout model)

| index                               |      n |    mae |   rmse |   mape | median_ape |     r2 |    bias |
|:------------------------------------|-------:|-------:|-------:|-------:|-----------:|-------:|--------:|
| train (in-sample, clean rows)       | 37,944 | 22.556 | 33.110 | 0.9700 |     0.7934 | 0.9994 |  0.0007 |
| holdout (out-of-sample, clean rows) |  9,379 | 48.147 | 73.465 |  2.011 |      1.618 | 0.9971 | -36.436 |
| mean                                | 23,661 | 35.352 | 53.288 |  1.491 |      1.206 | 0.9983 | -18.218 |

## Early-stopping curve (log-space RMSE)

| index          | rounds | max_iter | train_rmse_log | valid_rmse_log | valid_rmse_log_best |
|:---------------|-------:|---------:|---------------:|---------------:|--------------------:|
| boosting curve |    900 |      900 |         0.0112 |         0.0173 |              0.0173 |

## Holdout error by equipment

| equipment |     n |    mae |    rmse |  mape | median_ape |     r2 |    bias |
|:----------|------:|-------:|--------:|------:|-----------:|-------:|--------:|
| Dry Van   | 5,275 | 37.695 |  54.312 | 1.671 |      1.417 | 0.9983 | -26.278 |
| Flatbed   | 1,746 | 69.423 | 106.194 | 2.814 |      2.104 | 0.9945 | -56.611 |
| Reefer    | 2,358 | 55.777 |  80.734 | 2.178 |      1.820 | 0.9970 | -44.221 |

## Holdout error by distance decile

| distance_decile | distance_mean |   n |     mae |    rmse |  mape | median_ape |     r2 |    bias |
|----------------:|--------------:|----:|--------:|--------:|------:|-----------:|-------:|--------:|
|               0 |       223.900 | 938 |  12.727 |  17.606 | 2.111 |      1.731 | 0.9899 |  -9.361 |
|               1 |       402.300 | 940 |  19.927 |  26.872 | 2.001 |      1.558 | 0.9446 | -14.589 |
|               2 |       545.100 | 936 |  25.559 |  34.391 | 1.987 |      1.529 | 0.9223 | -19.471 |
|               3 |       707.200 | 939 |  32.476 |  43.348 | 2.024 |      1.613 | 0.9004 | -23.714 |
|               4 |       862.800 | 938 |  39.549 |  51.938 | 2.046 |      1.709 | 0.8965 | -29.358 |
|               5 |       1,041.4 | 936 |  45.972 |  61.399 | 2.033 |      1.649 | 0.8900 | -33.475 |
|               6 |       1,276.2 | 938 |  52.049 |  67.085 | 1.925 |      1.541 | 0.9177 | -36.611 |
|               7 |       1,633.2 | 938 |  65.839 |  88.053 | 1.937 |      1.555 | 0.9149 | -48.233 |
|               8 |       2,036.6 | 938 |  85.897 | 113.479 | 2.081 |      1.674 | 0.8720 | -68.723 |
|               9 |       2,631.0 | 938 | 101.501 | 135.205 | 1.970 |      1.592 | 0.9481 | -80.840 |

