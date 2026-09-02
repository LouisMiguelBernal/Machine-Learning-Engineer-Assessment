"""Central configuration.

Every path, seed and hyperparameter lives here. Module code reads these objects
rather than embedding literals, so a run is fully described by this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

SEED = 20251201
"""Single source of randomness for the whole project."""

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem layout, resolved against the repository root."""

    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    artifacts: Path = REPO_ROOT / "artifacts"
    reports: Path = REPO_ROOT / "reports"
    scorer_results: Path = REPO_ROOT / "scorer_results"

    @property
    def train(self) -> Path:
        return self.data / "train_test.csv"

    @property
    def validation(self) -> Path:
        return self.data / "validation.csv"

    @property
    def prediction_template(self) -> Path:
        return self.data / "validation_predictions_template.csv"

    @property
    def december_inputs(self) -> Path:
        return self.data / "december_chart_inputs.csv"

    @property
    def submission(self) -> Path:
        return self.root / "validation_predictions.csv"

    @property
    def model(self) -> Path:
        return self.artifacts / "model.joblib"

    def ensure_dirs(self) -> None:
        for directory in (self.artifacts, self.reports, self.scorer_results):
            directory.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds for the known feature-side data-quality defects.

    Note what is absent: there is no distance repair. Rows whose reported
    ``distance`` far exceeds the great-circle figure are short hauls sitting on a
    coarse synthetic coordinate grid, not corrupt mileage — see the module
    docstring in ``cleaning.py`` for the evidence.
    """

    fix_negative_weight: bool = True
    weight_clip_quantile: float = 0.0
    """Lower clip for weight, as a quantile of the fitted training weights.
    Zero keeps the full observed range; raise it only with evidence."""
    max_index_fallback_share: float = 0.05
    """Largest share of a predicted frame's dates allowed to fall back to the
    global-median market index. Above this the pipeline raises rather than
    silently pricing the frame at a constant level. 5% tolerates the odd day
    with no observed index; a whole evaluation window (100%) is the defect
    this guards against."""


@dataclass(frozen=True)
class OutlierConfig:
    """Target-contamination detection.

    ``n_sigma`` is measured in robust MAD-scaled units of the log-residual around
    a ``log(rate) ~ log(distance)`` baseline. The contaminated and clean
    populations are separated by an empty band, so any cut between roughly 4 and
    6 sigma selects the same rows; 5.0 sits in the middle of that plateau.
    """

    enabled: bool = True
    n_sigma: float = 5.0
    max_iterations: int = 10


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering switches.

    ``use_quote_signal`` defaults to False: EDA puts its correlation with the
    target at -0.04, and replacing it with a constant moves naive MAPE by
    0.33pp. It is retained as a flag so the exclusion stays falsifiable.

    ``use_linear_time_trend`` defaults to False because the scored window is two
    months beyond the training range. A tree cannot extrapolate a trend; it
    would freeze at the last training bin while pretending to model time.
    Seasonality is carried by cyclic encodings and by ``market_index``, which is
    observed for the scored window.
    """

    use_quote_signal: bool = False
    use_linear_time_trend: bool = False
    use_city_categoricals: bool = True
    use_geo_coordinates: bool = True
    use_cyclic_calendar: bool = True
    """Day-of-year sine/cosine. November-December land in a (sin < 0, cos > 0.5)
    corner no training row occupies, so the encoding is ablated rather than
    assumed; see ``reports/model_selection.md``."""
    city_dropout_rate: float = 0.10
    """Share of training rows whose city is blanked so the model learns a real
    unknown-city path. Set to 0.0 to disable.

    Swept over {0, 0.02, 0.05, 0.10, 0.20} against both a leave-cities-out split
    and the temporal holdout. 0.10 minimises the blend of the two weighted by the
    scored set's actual 12.1% unseen-city share: cold-start MAE falls from $50.86
    to $40.61 while the temporal holdout is unchanged at $38.94 (from $39.18).
    See ``features.apply_city_dropout`` and ``reports/cold_start.md``."""
    market_index_windows: tuple[int, ...] = (7, 28)

    # -- market level component (see level.py)
    use_market_level: bool = True
    level_use_trend: bool = True
    level_use_market_index: bool = True
    level_trend_damping: float = 1.0
    """Shrinks the trend only where it extrapolates past the fitted range.
    1.0 carries the fitted drift forward in full; 0.0 freezes at the last
    fitted day. Chosen by holdout ablation, not by taste."""
    level_market_index_shrink: float = 1.0
    """Shrinks the market-index term of the level towards its training mean:
    1.0 uses the fitted coefficient in full, 0.0 is equivalent to a trend-only
    level. A hedge for when the two validation geometries disagree on the
    index term - see ``reports/level_selection.md``."""


@dataclass(frozen=True)
class ModelConfig:
    """Estimator settings.

    ``target_mode`` selects what the estimator actually regresses:
      - ``log_rate``      : log(rate) directly.
      - ``log_rpm``       : log(rate) - log(distance), i.e. log rate-per-mile.
      - ``log_rpm_level`` : log rate-per-mile with the daily market level also
        removed, so the tree sees only cross-sectional structure.

    Distance explains R^2 = 0.936 of log(rate) on its own, and the daily market
    level a further 0.785 of what remains. Both are smooth and both extend past
    the training window; a piecewise-constant tree can represent neither outside
    its last bin. Removing them analytically leaves the tree the part it is
    actually good at. All three modes are implemented so the choice is measured.
    """

    target_mode: str = "log_rpm_level"
    learning_rate: float = 0.05
    max_iter: int = 900
    """Note: this cap binds. Left to run freely, early stopping halts at 941
    rounds — so 900 truncates 41 rounds short. Measured cost: $0.02 holdout MAE
    (38.94 vs 38.92), and the curve is flat from there to max_iter=4000. That is
    noise, and raising it would change every prediction and force a full
    regeneration of the deliverables for no real gain, so it stays. Recorded
    here so nobody reads `early_stopping=True` and assumes 900 was chosen by it."""
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    max_bins: int = 255
    early_stopping: bool = True
    validation_fraction: float = 0.12
    n_iter_no_change: int = 40
    smearing: bool = True
    seed: int = SEED


@dataclass(frozen=True)
class SplitConfig:
    """Temporal validation geometry.

    ``holdout_days`` matches the 61-day scored window (2025-11-01..2025-12-31),
    so the headline metric is measured over the same horizon it will be judged on.
    """

    holdout_days: int = 61
    n_rolling_splits: int = 5
    rolling_horizon_days: int = 30
    min_train_days: int = 120


@dataclass(frozen=True)
class RunConfig:
    """Everything a single run needs."""

    paths: Paths = field(default_factory=Paths)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = SEED

    def __post_init__(self) -> None:
        # ``log_rpm_level`` subtracts a feature the feature builder only emits
        # when the level is enabled. Turning the level off — which ablations do —
        # should degrade the target parameterisation, not crash the fit.
        if not self.features.use_market_level and self.model.target_mode == "log_rpm_level":
            object.__setattr__(self, "model", replace(self.model, target_mode="log_rpm"))


DEFAULT_RUN = RunConfig()
