# Model selection — holdout ablation

Holdout: 2025-09-01..2025-10-31 (9,523 loads), trained on 38,477 loads up to 2025-08-31.

`_clean` columns exclude the 144 holdout rows whose label is multiplicatively corrupted; their error is irreducible and would otherwise mask differences between variants. `_all` columns include them.

Every evaluation window is supplied its own inputs (coordinates and `market_index`, never the label) as context, exactly as the scored file supplies them for November-December.

| variant                       | group        | dMAE_vs_baseline | MAE_clean | MAPE_clean | MedAPE_clean | bias_clean | R2_clean | MAE_all | MAPE_all |  fit_s |
|:------------------------------|:-------------|-----------------:|----------:|-----------:|-------------:|-----------:|---------:|--------:|---------:|-------:|
| baseline (full model)         | reference    |            0.000 |    48.147 |      2.011 |        1.618 |    -36.436 |   0.9971 | 104.080 |    4.509 | 19.156 |
| target: log_rate              | target       |           14.043 |    62.191 |      2.567 |        2.257 |    -56.654 |   0.9956 | 117.916 |    5.037 | 17.890 |
| target: log_rpm (no level)    | target       |            9.442 |    57.589 |      2.400 |        2.099 |    -51.391 |   0.9963 | 113.414 |    4.875 | 17.389 |
| level: trend + market_index   | level        |            0.000 |    48.147 |      2.011 |        1.618 |    -36.436 |   0.9971 | 104.080 |    4.509 | 17.298 |
| level: trend only             | level        |           -7.590 |    40.557 |      1.705 |        1.362 |    -13.692 |   0.9979 |  96.593 |    4.238 | 16.564 |
| level: market_index only      | level        |           33.002 |    81.150 |      3.408 |        3.294 |    -79.810 |   0.9939 | 136.609 |    5.835 | 18.066 |
| level: index shrunk 0.5       | level        |           -1.909 |    46.238 |      1.917 |        1.540 |    -34.303 |   0.9973 | 102.183 |    4.421 | 17.300 |
| level: index shrunk 0.25      | level        |           -2.610 |    45.537 |      1.891 |        1.521 |    -33.056 |   0.9974 | 101.505 |    4.394 | 12.760 |
| level: trend damped 0.5       | level        |            3.567 |    51.714 |      2.153 |        1.768 |    -42.865 |   0.9968 | 107.596 |    4.641 | 12.899 |
| level: trend frozen (0.0)     | level        |            7.821 |    55.968 |      2.327 |        1.973 |    -49.270 |   0.9964 | 111.790 |    4.804 | 10.070 |
| outliers: kept in training    | data-quality |           10.408 |    58.555 |      2.549 |        2.015 |     -4.071 |   0.9960 | 114.318 |    5.080 |  1.812 |
| features: + quote_signal      | features     |           -3.949 |    44.198 |      1.826 |        1.533 |    -24.742 |   0.9976 | 100.183 |    4.331 |  9.730 |
| features: - city categoricals | features     |           0.2186 |    48.366 |      2.019 |        1.613 |    -36.002 |   0.9971 | 104.284 |    4.518 |  8.831 |
| features: - coordinates       | features     |          -0.2648 |    47.882 |      2.007 |        1.611 |    -34.609 |   0.9972 | 103.818 |    4.507 |  7.798 |
| features: - cyclic calendar   | features     |          -0.8707 |    47.277 |      1.971 |        1.579 |    -35.602 |   0.9972 | 103.221 |    4.469 |  9.790 |
| features: + raw time trend    | features     |           0.1249 |    48.272 |      2.017 |        1.621 |    -37.017 |   0.9971 | 104.191 |    4.513 | 10.060 |
| features: no city dropout     | features     |          -0.4551 |    47.692 |      1.994 |        1.585 |    -35.548 |   0.9972 | 103.621 |    4.492 |  9.800 |
