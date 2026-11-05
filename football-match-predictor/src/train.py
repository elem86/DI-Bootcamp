"""Train and evaluate the baseline match-outcome model (multinomial Logistic Regression).

Usage (from the project folder, after ``python -m src.feature_engineering``)::

    python -m src.train

Chronological evaluation
------------------------
1. **Tune**: fit on seasons 1993-94 to 2024-25 and score every candidate
   (feature set x regularisation strength ``C``) on 2025-26. The candidate
   with the lowest validation log loss is selected.
2. **Final fit**: refit the selected candidate on 1993-94 to 2025-26.
3. **Test**: predict the completed matches of 2026-27, which were never used
   for training, tuning, imputation or feature selection.

Every model is a scikit-learn ``Pipeline`` (median imputation with missing
indicators -> standardisation -> Logistic Regression), so the imputer and
scaler are fitted on the training rows only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, PREDICTIONS_DIR, RANDOM_STATE
from src.evaluate import evaluate, metrics_table, predict_probabilities, results_table
from src.feature_engineering import FEATURE_COLUMNS, SIDES, TARGET_COLUMN, TARGET_COLUMNS, load_features

TRAIN_LAST_SEASON = 2024  # 1993-94 .. 2024-25
VALIDATION_SEASON = 2025  # 2025-26
TEST_SEASON = 2026  # 2026-27 (in progress: completed matches only)

# Inverse regularisation strength: smaller C = stronger L2 penalty, smaller coefficients.
C_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

# The compact set keeps one window per feature family (the stronger 10-match
# windows), drops goals-for/against (``gd`` already combines them, and
# ``gd = gf - ga`` exactly) and drops the 5-match windows, which overlap
# heavily with the 10-match ones (see notebook 02).
COMPACT_TEAM_FEATURES = [
    "history_matches",
    "form10_ppg",
    "form10_gd",
    "venue_form10_ppg",
    "venue_form10_gd",
    "season_played",
    "season_ppg",
    "season_gd",
    "prev_season_ppg",
    "prev_season_gd",
    "promoted",
]
FEATURE_SETS = {
    "all": list(FEATURE_COLUMNS),
    "compact": [f"{side}_{name}" for side in SIDES for name in COMPACT_TEAM_FEATURES],
}

MODEL_FILE = MODELS_DIR / "logistic_regression.joblib"
METRICS_FILE = MODELS_DIR / "logistic_regression_metrics.json"
TEST_PREDICTIONS_FILE = PREDICTIONS_DIR / "logistic_regression_test_2026-27.csv"
VALIDATION_PREDICTIONS_FILE = PREDICTIONS_DIR / "logistic_regression_validation_2025-26.csv"


# ------------------------------------------------------------------ data


def chronological_split(
    features: pd.DataFrame, train_last_season: int, eval_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows up to ``train_last_season`` and evaluation rows of ``eval_season``.

    Only completed matches (with a known result) are returned.
    """
    if eval_season <= train_last_season:
        raise ValueError(
            f"Evaluation season {eval_season} must come after the last training season "
            f"{train_last_season}."
        )
    played = features[features[TARGET_COLUMN].notna()]
    train = played[played["season_start"] <= train_last_season]
    evaluation = played[played["season_start"] == eval_season]
    if train.empty or evaluation.empty:
        raise ValueError(f"Empty split: {len(train)} training and {len(evaluation)} evaluation rows.")
    return train, evaluation


def check_feature_columns(columns: list[str]) -> None:
    """Refuse anything other than the engineered pre-match features."""
    leaking = set(columns) & set(TARGET_COLUMNS)
    if leaking:
        raise ValueError(f"Target columns cannot be used as features: {sorted(leaking)}")
    unknown = set(columns) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Not pre-match features from feature_engineering: {sorted(unknown)}")


def feature_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Selected features as float64, with missing values as NaN.

    The nullable ``promoted`` flag becomes 1.0 / 0.0 / NaN (unknown).
    """
    check_feature_columns(columns)
    return frame[columns].astype("float64")


def target(frame: pd.DataFrame) -> pd.Series:
    return frame[TARGET_COLUMN].astype(str)


# ------------------------------------------------------------------ models


def build_model(C: float = 1.0) -> Pipeline:
    """Median imputation (+ missing indicators) -> scaling -> Logistic Regression.

    ``add_indicator=True`` adds a 0/1 column for every feature that had missing
    values in training, so the model knows *that* history was missing (e.g. a
    promoted team without last-season stats), not just the filled-in value.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def build_baseline() -> DummyClassifier:
    """Always predicts the most frequent training outcome.

    Its probabilities are the training outcome frequencies, so log loss is
    meaningful as well.
    """
    return DummyClassifier(strategy="prior")


# ------------------------------------------------------------------ workflow


@dataclass
class Experiment:
    """Everything produced by one tuning + final evaluation run."""

    tuning: pd.DataFrame
    feature_set: str
    C: float
    validation: dict[str, dict]
    validation_results: pd.DataFrame
    model: Pipeline
    baseline: DummyClassifier
    test: dict[str, dict]
    test_results: pd.DataFrame
    seasons: dict = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        return FEATURE_SETS[self.feature_set]


def tune(train: pd.DataFrame, validation: pd.DataFrame, feature_sets=None, c_grid=C_GRID) -> pd.DataFrame:
    """Score every feature set x C candidate on the validation season."""
    feature_sets = feature_sets or FEATURE_SETS
    rows = []
    for name, columns in feature_sets.items():
        X_train, X_val = feature_matrix(train, columns), feature_matrix(validation, columns)
        for C in c_grid:
            model = build_model(C).fit(X_train, target(train))
            metrics = evaluate(target(validation), predict_probabilities(model, X_val))
            rows.append(
                {
                    "feature_set": name,
                    "features": len(columns),
                    "C": C,
                    "log_loss": metrics["log_loss"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
    return pd.DataFrame(rows).sort_values("log_loss", kind="stable").reset_index(drop=True)


def run_experiment(
    features: pd.DataFrame,
    train_last_season: int = TRAIN_LAST_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
    feature_sets=None,
    c_grid=C_GRID,
) -> Experiment:
    """Tune on the validation season, refit on train + validation, test on the test season."""
    feature_sets = feature_sets or FEATURE_SETS

    # 1. Tune: train <= train_last_season, validate on validation_season
    train, validation = chronological_split(features, train_last_season, validation_season)
    tuning = tune(train, validation, feature_sets, c_grid)
    best = tuning.iloc[0]
    feature_set, C = str(best["feature_set"]), float(best["C"])
    columns = feature_sets[feature_set]

    X_train, X_val = feature_matrix(train, columns), feature_matrix(validation, columns)
    val_model = build_model(C).fit(X_train, target(train))
    val_baseline = build_baseline().fit(X_train, target(train))
    val_probs = predict_probabilities(val_model, X_val)
    validation_metrics = {
        "Logistic Regression": evaluate(target(validation), val_probs),
        "Baseline (always most frequent)": evaluate(
            target(validation), predict_probabilities(val_baseline, X_val)
        ),
    }

    # 2. Final fit on everything before the test season; 3. test
    final_train, test = chronological_split(features, validation_season, test_season)
    X_final, X_test = feature_matrix(final_train, columns), feature_matrix(test, columns)
    model = build_model(C).fit(X_final, target(final_train))
    baseline = build_baseline().fit(X_final, target(final_train))
    test_probs = predict_probabilities(model, X_test)
    test_metrics = {
        "Logistic Regression": evaluate(target(test), test_probs),
        "Baseline (always most frequent)": evaluate(target(test), predict_probabilities(baseline, X_test)),
    }

    return Experiment(
        tuning=tuning,
        feature_set=feature_set,
        C=C,
        validation=validation_metrics,
        validation_results=results_table(validation, val_probs),
        model=model,
        baseline=baseline,
        test=test_metrics,
        test_results=results_table(test, test_probs),
        seasons={
            "train": [int(final_train["season_start"].min()), train_last_season],
            "validation": validation_season,
            "final_train": [int(final_train["season_start"].min()), validation_season],
            "test": test_season,
        },
    )


def save_artifacts(experiment: Experiment) -> dict[str, Path]:
    """Save the final model, metrics and predictions to the (git-ignored) output folders."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": experiment.model, "feature_columns": experiment.feature_columns},
        MODEL_FILE,
    )

    def serialisable(results):
        return {
            name: {k: v for k, v in m.items() if k != "confusion_matrix"}
            | {"confusion_matrix": m["confusion_matrix"].to_dict(orient="index")}
            for name, m in results.items()
        }

    METRICS_FILE.write_text(
        json.dumps(
            {
                "selected": {"feature_set": experiment.feature_set, "C": experiment.C},
                "seasons": experiment.seasons,
                "validation": serialisable(experiment.validation),
                "test": serialisable(experiment.test),
                "tuning": experiment.tuning.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    experiment.test_results.to_csv(TEST_PREDICTIONS_FILE, index=False, date_format="%Y-%m-%d")
    experiment.validation_results.to_csv(VALIDATION_PREDICTIONS_FILE, index=False, date_format="%Y-%m-%d")
    return {
        "model": MODEL_FILE,
        "metrics": METRICS_FILE,
        "test_predictions": TEST_PREDICTIONS_FILE,
        "validation_predictions": VALIDATION_PREDICTIONS_FILE,
    }


def main() -> None:
    features = load_features()
    latest = int(features["season_start"].max())
    if latest != TEST_SEASON:
        raise SystemExit(
            f"Latest season in the data starts in {latest}, but TEST_SEASON is {TEST_SEASON}. "
            "Update the season constants in src/train.py."
        )

    experiment = run_experiment(features)
    paths = save_artifacts(experiment)

    pd.set_option("display.width", 120)
    print(f"Selected: feature set '{experiment.feature_set}', C = {experiment.C}\n")
    print("Validation (2025-26):")
    print(metrics_table(experiment.validation).round(3).to_string(), "\n")
    print("Test (2026-27, completed matches):")
    print(metrics_table(experiment.test).round(3).to_string(), "\n")
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
