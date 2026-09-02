# Level selection — holdout and rolling folds

Holdout: 2025-09-01..2025-10-31 (9,523 loads), trained on 38,477 loads up to 2025-08-31.

`_clean` columns exclude the 144 holdout rows whose label is multiplicatively corrupted; their error is irreducible and would otherwise mask differences between variants. `_all` columns include them.

Every evaluation window is supplied its own inputs (coordinates and `market_index`, never the label) as context, exactly as the scored file supplies them for November-December.

`MAE_fold_mean` is the mean over the five rolling-origin folds; `MAE_6window_mean` averages the holdout with them. A variant is adopted only if it does not lose on either geometry.

| variant                       | group    | MAE_clean | MAE_fold_mean | MAE_6window_mean | MAPE_clean | MedAPE_clean | bias_clean | R2_clean | MAE_all | MAPE_all |  fit_s | MAE_fold_1 | MAE_fold_2 | MAE_fold_3 | MAE_fold_4 | MAE_fold_5 |
|:------------------------------|:---------|----------:|--------------:|-----------------:|-----------:|-------------:|-----------:|---------:|--------:|---------:|-------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| features: + quote_signal      | features |    44.198 |        48.856 |           48.080 |      1.826 |        1.533 |    -24.742 |   0.9976 | 100.183 |    4.331 | 12.494 |     41.540 |     53.270 |     47.526 |     48.602 |     53.342 |
| features: - city categoricals | features |    48.366 |        46.607 |           46.900 |      2.019 |        1.613 |    -36.002 |   0.9971 | 104.284 |    4.518 |  9.859 |     50.851 |     56.065 |     31.154 |     56.986 |     37.978 |
| features: - coordinates       | features |    47.882 |        46.531 |           46.756 |      2.007 |        1.611 |    -34.609 |   0.9972 | 103.818 |    4.507 | 10.741 |     51.065 |     57.539 |     32.025 |     55.494 |     36.528 |
| features: - cyclic calendar   | features |    47.277 |        52.394 |           51.541 |      1.971 |        1.579 |    -35.602 |   0.9972 | 103.221 |    4.469 | 15.121 |     53.122 |     84.180 |     34.450 |     57.689 |     32.528 |
| features: + raw time trend    | features |    48.272 |        46.032 |           46.405 |      2.017 |        1.621 |    -37.017 |   0.9971 | 104.191 |    4.513 | 15.453 |     51.305 |     58.486 |     30.881 |     56.587 |     32.899 |
| features: no city dropout     | features |    47.692 |        46.110 |           46.374 |      1.994 |        1.585 |    -35.548 |   0.9972 | 103.621 |    4.492 | 15.469 |     51.713 |     56.926 |     30.548 |     55.858 |     35.506 |
