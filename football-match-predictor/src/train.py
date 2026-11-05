"""Train and evaluate match-outcome models with a strictly chronological split.

Usage (from the project folder, after ``python -m src.feature_engineering``)::

    python -m src.train                                  # all models
    python -m src.train --models logistic_regression     # one or more models

Models (defined in ``src/models.py``): multinomial Logistic Regression, Random
Forest and XGBoost, each compared with a baseline that always predicts the
most frequent training outcome.

Chronological evaluation
------------------------
1. **Tune**: fit on seasons 1993-94 to 2024-25 and score every candidate
   (feature set x the model's hyperparameter grid) on 2025-26. The candidate
   with the lowest validation log loss is selected.
2. **Final fit**: refit the selected candidate on 1993-94 to 2025-26.
3. **Test**: predict the completed matches of 2026-27, which were never used
   for training, tuning, imputation or feature selection.

Every model is a scikit-learn ``Pipeline``, so any imputation or scaling is
fitted on the training rows only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, PREDICTIONS_DIR
from src.evaluate import evaluate, metrics_table, predict_probabilities, results_table
from src.feature_engineering import FEATURE_COLUMNS, SIDES, TARGET_COLUMN, TARGET_COLUMNS, load_features
from src.models import C_GRID, MODEL_SPECS, build_logistic_regression

TRAIN_LAST_SEASON = 2024  # 1993-94 .. 2024-25
VALIDATION_SEASON = 2025  # 2025-26
TEST_SEASON = 2026  # 2026-27 (in progress: completed matches only)

BASELINE_LABEL = "Baseline (always most frequent)"

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

COMPARISON_FILE = MODELS_DIR / "model_comparison.csv"

# Kept for backwards compatibility (notebook 03 and earlier code use this name).
build_model = build_logistic_regression


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


def build_baseline() -> DummyClassifier:
    """Always predicts the most frequent training outcome.

    Its probabilities are the training outcome frequencies, so log loss is
    meaningful as well.
    """
    return DummyClassifier(strategy="prior")


def season_name(season_start: int) -> str:
    return f"{season_start}-{str(season_start + 1)[-2:]}"


# ------------------------------------------------------------------ workflow


@dataclass
class Experiment:
    """Everything produced by one tuning + final evaluation run of one model."""

    model_name: str
    tuning: pd.DataFrame
    feature_set: str
    params: dict
    validation: dict[str, dict]
    validation_results: pd.DataFrame
    model: Pipeline
    baseline: DummyClassifier
    test: dict[str, dict]
    test_results: pd.DataFrame
    seasons: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return MODEL_SPECS[self.model_name].label

    @property
    def feature_columns(self) -> list[str]:
        return FEATURE_SETS[self.feature_set]

    @property
    def C(self) -> float | None:
        """Regularisation strength of a Logistic Regression experiment."""
        return self.params.get("C")


def _resolve_grid(model_name: str, param_grid, c_grid) -> list[dict]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_SPECS)}")
    if c_grid is not None:
        if model_name != "logistic_regression":
            raise ValueError("c_grid only applies to logistic_regression; use param_grid instead.")
        return [{"C": C} for C in c_grid]
    return list(param_grid) if param_grid is not None else MODEL_SPECS[model_name].param_grid


def tune(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_sets=None,
    c_grid=None,
    model_name: str = "logistic_regression",
    param_grid=None,
) -> pd.DataFrame:
    """Score every feature set x parameter combination on the validation season."""
    feature_sets = feature_sets or FEATURE_SETS
    candidates = _resolve_grid(model_name, param_grid, c_grid)
    build = MODEL_SPECS[model_name].build
    rows = []
    for name, columns in feature_sets.items():
        X_train, X_val = feature_matrix(train, columns), feature_matrix(validation, columns)
        for index, params in enumerate(candidates):
            model = build(**params).fit(X_train, target(train))
            metrics = evaluate(target(validation), predict_probabilities(model, X_val))
            rows.append(
                {"feature_set": name, "features": len(columns)}
                | params
                | {
                    "log_loss": metrics["log_loss"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "candidate": index,  # position in the parameter grid
                }
            )
    return pd.DataFrame(rows).sort_values("log_loss", kind="stable").reset_index(drop=True)


def run_experiment(
    features: pd.DataFrame,
    train_last_season: int = TRAIN_LAST_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
    feature_sets=None,
    c_grid=None,
    model_name: str = "logistic_regression",
    param_grid=None,
) -> Experiment:
    """Tune on the validation season, refit on train + validation, test on the test season."""
    feature_sets = feature_sets or FEATURE_SETS
    candidates = _resolve_grid(model_name, param_grid, c_grid)
    spec = MODEL_SPECS[model_name]

    # 1. Tune: train <= train_last_season, validate on validation_season
    train, validation = chronological_split(features, train_last_season, validation_season)
    tuning = tune(train, validation, feature_sets, model_name=model_name, param_grid=candidates)
    best = tuning.iloc[0]
    feature_set = str(best["feature_set"])
    # Take the parameters from the grid itself: the tuning table turns None into
    # NaN and integers into floats when a column mixes them (e.g. max_depth).
    params = dict(candidates[int(best["candidate"])])
    columns = feature_sets[feature_set]

    X_train, X_val = feature_matrix(train, columns), feature_matrix(validation, columns)
    val_model = spec.build(**params).fit(X_train, target(train))
    val_baseline = build_baseline().fit(X_train, target(train))
    val_probs = predict_probabilities(val_model, X_val)
    validation_metrics = {
        spec.label: evaluate(target(validation), val_probs),
        BASELINE_LABEL: evaluate(target(validation), predict_probabilities(val_baseline, X_val)),
    }

    # 2. Final fit on everything before the test season; 3. test
    final_train, test = chronological_split(features, validation_season, test_season)
    X_final, X_test = feature_matrix(final_train, columns), feature_matrix(test, columns)
    model = spec.build(**params).fit(X_final, target(final_train))
    baseline = build_baseline().fit(X_final, target(final_train))
    test_probs = predict_probabilities(model, X_test)
    test_metrics = {
        spec.label: evaluate(target(test), test_probs),
        BASELINE_LABEL: evaluate(target(test), predict_probabilities(baseline, X_test)),
    }

    return Experiment(
        model_name=model_name,
        tuning=tuning,
        feature_set=feature_set,
        params=params,
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


def comparison_table(experiments: list[Experiment]) -> pd.DataFrame:
    """Validation and test metrics for every model plus the shared baseline."""
    frames = []
    for split in ("validation", "test"):
        results = {}
        for experiment in experiments:
            results[experiment.label] = getattr(experiment, split)[experiment.label]
        results[BASELINE_LABEL] = getattr(experiments[0], split)[BASELINE_LABEL]
        table = metrics_table(results)
        table.index = pd.MultiIndex.from_product([[split], table.index], names=["split", "model"])
        frames.append(table)
    return pd.concat(frames)


# ------------------------------------------------------------------ artifacts


SPLIT_SEASONS = {"validation": VALIDATION_SEASON, "test": TEST_SEASON}


def predictions_path(model_name: str, split: str, season_start: int | None = None) -> Path:
    """Where the per-match predictions of one model on one split are saved."""
    if split not in SPLIT_SEASONS:
        raise ValueError(f"split must be one of {sorted(SPLIT_SEASONS)}")
    season_start = SPLIT_SEASONS[split] if season_start is None else season_start
    return PREDICTIONS_DIR / f"{model_name}_{split}_{season_name(season_start)}.csv"


def artifact_paths(experiment: Experiment) -> dict[str, Path]:
    name = experiment.model_name
    return {
        "model": MODELS_DIR / f"{name}.joblib",
        "metrics": MODELS_DIR / f"{name}_metrics.json",
        "test_predictions": predictions_path(name, "test", experiment.seasons["test"]),
        "validation_predictions": predictions_path(name, "validation", experiment.seasons["validation"]),
    }


def load_predictions(model_name: str, split: str, season_start: int | None = None) -> pd.DataFrame:
    """Read saved per-match predictions (written by ``save_artifacts``)."""
    path = predictions_path(model_name, split, season_start)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.train` first.")
    return pd.read_csv(path, parse_dates=["date"]).astype({"date": "datetime64[ns]"})


def save_artifacts(experiment: Experiment) -> dict[str, Path]:
    """Save the final model, metrics and predictions to the (git-ignored) output folders."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(experiment)

    joblib.dump(
        {"model": experiment.model, "feature_columns": experiment.feature_columns},
        paths["model"],
    )

    def serialisable(results):
        return {
            name: {k: v for k, v in m.items() if k != "confusion_matrix"}
            | {"confusion_matrix": m["confusion_matrix"].to_dict(orient="index")}
            for name, m in results.items()
        }

    tuning = experiment.tuning.astype(object).where(experiment.tuning.notna(), None)
    paths["metrics"].write_text(
        json.dumps(
            {
                "model": experiment.label,
                "selected": {"feature_set": experiment.feature_set} | experiment.params,
                "seasons": experiment.seasons,
                "validation": serialisable(experiment.validation),
                "test": serialisable(experiment.test),
                "tuning": tuning.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    experiment.test_results.to_csv(paths["test_predictions"], index=False, date_format="%Y-%m-%d")
    experiment.validation_results.to_csv(paths["validation_predictions"], index=False, date_format="%Y-%m-%d")
    return paths


def save_comparison(experiments: list[Experiment], path: Path = COMPARISON_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_table(experiments).to_csv(path)
    return path


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate match-outcome models.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="Models to train (default: all).",
    )
    args = parser.parse_args(argv)

    features = load_features()
    latest = int(features["season_start"].max())
    if latest != TEST_SEASON:
        raise SystemExit(
            f"Latest season in the data starts in {latest}, but TEST_SEASON is {TEST_SEASON}. "
            "Update the season constants in src/train.py."
        )

    pd.set_option("display.width", 140)
    experiments = []
    for model_name in args.models:
        experiment = run_experiment(features, model_name=model_name)
        paths = save_artifacts(experiment)
        experiments.append(experiment)
        params = ", ".join(f"{k} = {v}" for k, v in experiment.params.items())
        print(f"{experiment.label}: feature set '{experiment.feature_set}', {params}")
        for name, path in paths.items():
            print(f"  saved {name}: {path.relative_to(path.parents[2])}")

    path = save_comparison(experiments)
    print(f"\nValidation ({season_name(VALIDATION_SEASON)}) and test ({season_name(TEST_SEASON)}):")
    print(comparison_table(experiments).round(3).to_string())
    print(f"\nSaved comparison: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
