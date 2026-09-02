# Cold-start ablation — unseen cities

8 cities removed from training entirely, evaluated only on the loads that touch them, averaged over 3 different city samples. This mirrors the 8 cities that appear only in the scored data and touch 12.1% of the loads that must be predicted.

Rows with corrupted labels are excluded.

| variant                           | dMAE_vs_full |    mae |   rmse |  mape | median_ape |   bias |
|:----------------------------------|-------------:|-------:|-------:|------:|-----------:|-------:|
| full model (dropout 0.10)         |        0.000 | 40.841 | 58.687 | 1.610 |      1.350 | 0.5439 |
| no city dropout                   |       10.414 | 51.255 | 70.112 | 2.131 |      1.792 | 15.679 |
| no coordinates                    |        8.364 | 49.206 | 67.657 | 2.026 |      1.707 |  1.346 |
| no city categoricals              |       -1.225 | 39.617 | 56.940 | 1.554 |      1.299 | 0.6364 |
| neither (distance only geography) |       15.482 | 56.324 | 74.236 | 2.531 |      2.136 | 0.0830 |

## Per-seed detail

| seed | variant                           |      n |    mae |   rmse |  mape | median_ape |     r2 |    bias |
|-----:|:----------------------------------|-------:|-------:|-------:|------:|-----------:|-------:|--------:|
|    0 | full model (dropout 0.10)         |  8,831 | 38.081 | 52.779 | 1.590 |      1.341 | 0.9978 |  -8.091 |
|    0 | no city dropout                   |  8,831 | 45.600 | 61.599 | 1.994 |      1.649 | 0.9970 | -10.752 |
|    0 | no coordinates                    |  8,831 | 48.647 | 65.480 | 2.065 |      1.737 | 0.9966 | -20.463 |
|    0 | no city categoricals              |  8,831 | 36.567 | 50.210 | 1.532 |      1.299 | 0.9980 |  -1.408 |
|    0 | neither (distance only geography) |  8,831 | 57.576 | 73.639 | 2.660 |      2.324 | 0.9957 | -21.372 |
|    1 | full model (dropout 0.10)         | 11,281 | 36.450 | 53.287 | 1.585 |      1.340 | 0.9985 |  -3.003 |
|    1 | no city dropout                   | 11,281 | 49.829 | 66.334 | 2.327 |      2.003 | 0.9976 |  22.829 |
|    1 | no coordinates                    | 11,281 | 45.758 | 62.427 | 2.110 |      1.802 | 0.9979 |   9.417 |
|    1 | no city categoricals              | 11,281 | 35.427 | 52.139 | 1.530 |      1.278 | 0.9985 |  -5.366 |
|    1 | neither (distance only geography) | 11,281 | 51.719 | 67.684 | 2.635 |      2.199 | 0.9975 |   9.384 |
|    2 | full model (dropout 0.10)         | 10,705 | 47.993 | 69.996 | 1.655 |      1.369 | 0.9979 |  12.726 |
|    2 | no city dropout                   | 10,705 | 58.337 | 82.401 | 2.071 |      1.723 | 0.9970 |  34.959 |
|    2 | no coordinates                    | 10,705 | 53.212 | 75.063 | 1.902 |      1.582 | 0.9975 |  15.084 |
|    2 | no city categoricals              | 10,705 | 46.856 | 68.471 | 1.599 |      1.320 | 0.9980 |   8.684 |
|    2 | neither (distance only geography) | 10,705 | 59.676 | 81.384 | 2.299 |      1.887 | 0.9971 |  12.237 |
