# Level selection — holdout and rolling folds

Holdout: 2025-09-01..2025-10-31 (9,523 loads), trained on 38,477 loads up to 2025-08-31.

`_clean` columns exclude the 144 holdout rows whose label is multiplicatively corrupted; their error is irreducible and would otherwise mask differences between variants. `_all` columns include them.

Every evaluation window is supplied its own inputs (coordinates and `market_index`, never the label) as context, exactly as the scored file supplies them for November-December.

`MAE_fold_mean` is the mean over the five rolling-origin folds; `MAE_6window_mean` averages the holdout with them. A variant is adopted only if it does not lose on either geometry.

| variant                     | group | MAE_clean | MAE_fold_mean | MAE_6window_mean | MAPE_clean | MedAPE_clean | bias_clean | R2_clean | MAE_all | MAPE_all |  fit_s | MAE_fold_1 | MAE_fold_2 | MAE_fold_3 | MAE_fold_4 | MAE_fold_5 |
|:----------------------------|:------|----------:|--------------:|-----------------:|-----------:|-------------:|-----------:|---------:|--------:|---------:|-------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| level: trend + market_index | level |    48.147 |        46.815 |           47.037 |      2.011 |        1.618 |    -36.436 |   0.9971 | 104.080 |    4.509 | 37.325 |     51.298 |     58.692 |     30.752 |     56.440 |     36.895 |
| level: trend only           | level |    40.557 |        51.189 |           49.417 |      1.705 |        1.362 |    -13.692 |   0.9979 |  96.593 |    4.238 | 16.195 |     46.628 |     80.609 |     45.870 |     48.474 |     34.362 |
| level: market_index only    | level |    81.150 |        54.586 |           59.013 |      3.408 |        3.294 |    -79.810 |   0.9939 | 136.609 |    5.835 | 15.476 |     61.930 |     36.261 |     51.583 |     76.841 |     46.313 |
| level: index shrunk 0.5     | level |    46.238 |        48.544 |           48.160 |      1.917 |        1.540 |    -34.303 |   0.9973 | 102.183 |    4.421 |  8.724 |     51.210 |     69.997 |     30.602 |     55.698 |     35.215 |
| level: index shrunk 0.25    | level |    45.537 |        48.253 |           47.800 |      1.891 |        1.521 |    -33.056 |   0.9974 | 101.505 |    4.394 |  5.291 |     51.390 |     70.794 |     31.153 |     53.952 |     33.975 |
| level: trend damped 0.5     | level |    51.714 |        46.513 |           47.380 |      2.153 |        1.768 |    -42.865 |   0.9968 | 107.596 |    4.641 |  5.325 |     54.051 |     52.050 |     31.277 |     58.801 |     36.385 |
| level: trend frozen (0.0)   | level |    55.968 |        46.665 |           48.215 |      2.327 |        1.973 |    -49.270 |   0.9964 | 111.790 |    4.804 |  5.089 |     57.378 |     46.258 |     32.182 |     61.252 |     36.254 |
