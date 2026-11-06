"""Do Elo ratings add information? A walk-forward comparison with and without Elo features.

Usage (from the project folder, after ``python -m src.feature_engineering``)::

    python -m src.elo_experiment                 # all models, seasons 2018-19 to 2024-25

Results are saved to ``outputs/elo/`` (git-ignored). Production models,
``outputs/backtest/`` and ``outputs/recency/`` are not touched.

Design
------
* **Elo settings are fixed in advance** (``src.elo.DEFAULT_SETTINGS``: start
  1500, K = 20, home advantage 60, 30% regression each summer). They are
  experimental defaults, not tuned on any season, so no evaluation season can
  influence them.
* **Two arms per model, everything else identical.** Both arms run the
  commit-10 fold procedure (``src.backtest.run_backtest``) on the *same* feature
  table, the same seasons and therefore the same fixtures:

  - *Control:* the existing feature sets, ``all`` (50 features) and ``compact`` (22).
  - *Elo:* the same two sets plus ``home_elo``, ``away_elo`` and
    ``elo_difference`` (``all_elo``: 53, ``compact_elo``: 25).

  In every fold the feature set and the hyperparameters are chosen on S-1
  after fitting on seasons up to S-2, then the choice is refitted on seasons
  up to S-1 and scored on S. Season S never takes part in a choice.
* **No leakage from the ratings.** Each match's Elo uses only results from
  strictly earlier dates (``src.elo``). ``check_elo_is_past_only`` recomputes
  the ratings with every later season removed and requires identical values
  for every evaluation season before any model is fitted.
* 2025-26 (validation season of the production models) and 2026-27 (holdout)
  are excluded from this experiment.
* Bookmaker probabilities are a benchmark only, never a feature.

Pre-specified decision rule
---------------------------
Written before the experiment was run. Elo is recommended for a production
model only if, for that model, **all** of these hold:

1. pooled log loss improves, with a paired bootstrap interval that excludes 0
   after a Bonferroni correction for the three model comparisons;
2. the improvement appears in at least 5 of the 7 seasons;
3. the pooled improvement survives leaving out any one season.

Otherwise the evidence is reported as "not demonstrated" (which is not the
same as "Elo is useless").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (
    BACKTEST_SEASONS,
    BASELINE,
    Backtest,
    common_fixtures,
    load_backtest,
    pooled_bootstrap,
    pooled_metrics,
    run_backtest,
    save_backtest,
)
from src.bookmaker import BOOKMAKER_LABEL, FIXTURE_KEYS
from src.config import OUTPUTS_DIR, RANDOM_STATE
from src.elo import DEFAULT_SETTINGS, ELO_COLUMNS, ELO_FEATURE_COLUMNS, EloSettings, add_elo_features, elo_features
from src.evaluate import OUTCOMES, predicted_outcomes
from src.models import MODEL_SPECS
from src.recency import season_level_summary
from src.train import FEATURE_SETS, VALIDATION_SEASON, season_name

ELO_DIR = OUTPUTS_DIR / "elo"
ELO_SUFFIX = " + Elo"
CONTROL_FEATURE_SETS = dict(FEATURE_SETS)  # "all" (50) and "compact" (22): unchanged
ELO_FEATURE_SETS = {f"{name}_elo": columns + ELO_FEATURE_COLUMNS for name, columns in FEATURE_SETS.items()}
FAMILY_WISE_ALPHA = 0.05
MIN_SEASONS_BETTER = 5  # decision rule: improvement in at least 5 of 7 seasons


def elo_label(model_label: str) -> str:
    return f"{model_label}{ELO_SUFFIX}"


def model_pairs(models=None) -> list[tuple[str, str]]:
    """(Elo arm, control) forecast labels for each model."""
    labels = [MODEL_SPECS[m].label for m in (models or MODEL_SPECS)]
    return [(elo_label(label), label) for label in labels]


# ====================================================================== leakage check


def check_elo_is_past_only(features: pd.DataFrame, seasons, settings: EloSettings = DEFAULT_SETTINGS) -> None:
    """Elo computed on the full table must equal Elo computed with every later season removed.

    For each evaluation season S the ratings are recomputed from seasons up to S
    only. Any difference would mean that a later result leaked into an earlier
    rating, so it raises instead of continuing.
    """
    full = elo_features(features, settings)
    for season in seasons:
        visible = features[features["season_start"] <= season]
        if not elo_features(visible, settings).equals(full.loc[visible.index]):
            raise AssertionError(f"Elo ratings up to {season_name(season)} depend on later seasons")


def check_experiment_seasons(seasons) -> None:
    """Only seasons before the production validation season (2025-26) may be used."""
    late = [s for s in seasons if s >= VALIDATION_SEASON]
    if late:
        raise ValueError(f"The Elo experiment must not use {season_name(VALIDATION_SEASON)} or later "
                         f"(validation and holdout seasons); got {[season_name(s) for s in late]}")


# ====================================================================== experiment


def run_elo_backtest(
    features: pd.DataFrame,
    matches: pd.DataFrame,
    seasons=BACKTEST_SEASONS,
    models=None,
    param_grids: dict[str, list[dict]] | None = None,
    settings: EloSettings = DEFAULT_SETTINGS,
    progress=None,
) -> Backtest:
    """Walk-forward backtest of every model with and without Elo, on identical fixtures.

    ``predictions`` holds the control rows under the model's label, the Elo
    rows under ``"<model> + Elo"``, and the baseline and bookmaker rows once.
    ``folds`` has one row per (arm, model, season) with the settings chosen.
    """
    seasons = sorted(seasons)
    check_experiment_seasons(seasons)
    models = list(models or MODEL_SPECS)
    table = add_elo_features(features, settings)  # 50 original columns untouched + Elo columns
    check_elo_is_past_only(table, seasons, settings)

    arms = {}
    for arm, feature_sets in (("control", CONTROL_FEATURE_SETS), ("elo", ELO_FEATURE_SETS)):
        report = (lambda message, a=arm: progress(f"[{a}] {message}")) if progress else None
        arms[arm] = run_backtest(table, matches, seasons=seasons, models=models, param_grids=param_grids,
                                 feature_sets=feature_sets, progress=report)

    control, elo = arms["control"].predictions, arms["elo"].predictions
    keys = ["season", "forecast", *FIXTURE_KEYS, "actual", "has_odds"]
    if not control[keys].equals(elo[keys]):
        raise ValueError("Control and Elo arms were evaluated on different fixtures")
    for forecast in (BASELINE, BOOKMAKER_LABEL):  # do not depend on the features
        a, b = control[control["forecast"] == forecast], elo[elo["forecast"] == forecast]
        if not np.allclose(a[OUTCOMES].to_numpy(), b[OUTCOMES].to_numpy()):
            raise ValueError(f"{forecast} differs between the arms")

    model_labels = [MODEL_SPECS[m].label for m in models]
    elo_rows = elo[elo["forecast"].isin(model_labels)].copy()
    elo_rows["forecast"] = elo_rows["forecast"] + ELO_SUFFIX
    predictions = pd.concat([control, elo_rows], ignore_index=True)

    folds = pd.concat([arms["control"].folds.assign(arm="control"),
                       arms["elo"].folds.assign(arm="elo")], ignore_index=True)
    return Backtest(predictions=predictions, folds=folds)


# ====================================================================== analysis


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _paired_interval(differences: np.ndarray, confidence: float, n_resamples: int, random_state: int):
    rng = np.random.default_rng(random_state)
    n = len(differences)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 1000):
        stop = min(start + 1000, n_resamples)
        means[start:stop] = differences[rng.integers(0, n, size=(stop - start, n))].mean(axis=1)
    tail = (1 - confidence) / 2
    return np.quantile(means, [tail, 1 - tail])


def accuracy_bootstrap(predictions: pd.DataFrame, pairs, confidence: float = 0.95,
                       n_resamples: int = 10_000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Paired bootstrap of the accuracy difference ``a - b`` on the common fixtures (positive = a better)."""
    rows = common_fixtures(predictions)
    order = ["season", *FIXTURE_KEYS]
    result = []
    for a, b in pairs:
        ra = rows[rows["forecast"] == a].sort_values(order).reset_index(drop=True)
        rb = rows[rows["forecast"] == b].sort_values(order).reset_index(drop=True)
        if not ra[order].equals(rb[order]):
            raise ValueError(f"{a} and {b} cover different fixtures")
        correct_a = (predicted_outcomes(ra[OUTCOMES]) == ra["actual"]).to_numpy(float)
        correct_b = (predicted_outcomes(rb[OUTCOMES]) == rb["actual"]).to_numpy(float)
        diff = correct_a - correct_b
        low, high = _paired_interval(diff, confidence, n_resamples, random_state)
        result.append({"a": a, "b": b, "metric": "accuracy", "matches": len(diff), "difference": float(diff.mean()),
                       "ci_low": float(low), "ci_high": float(high), "confidence": confidence,
                       "ci_includes_zero": bool(low <= 0 <= high)})
    return pd.DataFrame(result)


def paired_intervals(predictions: pd.DataFrame, pairs, metric: str = "log_loss", n_comparisons: int | None = None,
                     n_resamples: int = 10_000) -> pd.DataFrame:
    """Pooled paired bootstrap of ``a - b``: a 95% interval and a Bonferroni-adjusted one.

    The adjusted interval uses confidence ``1 - 0.05 / n_comparisons`` (default:
    one comparison per pair), so the family of comparisons keeps a 5% error rate.
    """
    n_comparisons = n_comparisons or len(pairs)
    adjusted = 1 - FAMILY_WISE_ALPHA / n_comparisons
    tables = {}
    for confidence in (0.95, adjusted):
        if metric == "accuracy":
            tables[confidence] = accuracy_bootstrap(predictions, pairs, confidence, n_resamples)
        else:
            tables[confidence] = pooled_bootstrap(predictions, pairs, metric=metric, confidence=confidence,
                                                  n_resamples=n_resamples)
    table = tables[0.95][["a", "b", "matches", "difference", "ci_low", "ci_high"]].copy()
    table["adjusted_confidence"] = adjusted
    table["adj_ci_low"] = tables[adjusted]["ci_low"].to_numpy()
    table["adj_ci_high"] = tables[adjusted]["ci_high"].to_numpy()
    table.insert(2, "metric", metric)
    return table


def comparison_table(predictions: pd.DataFrame, models=None, metric: str = "log_loss",
                     n_resamples: int = 10_000) -> pd.DataFrame:
    """Elo arm vs control for every model on one metric: pooled scores, intervals and season-level checks.

    ``difference`` is Elo minus control (negative = Elo better for log loss and
    Brier, positive = Elo better for accuracy and macro F1). Sign-test p-values
    are Holm-adjusted across the models.
    """
    pairs = model_pairs(models)
    pooled = pooled_metrics(predictions)[metric]
    rows = []
    for elo, control in pairs:
        subset = predictions[predictions["forecast"].isin([elo, control])]
        season = season_level_summary(subset, control=control, metric=metric).loc[elo]
        rows.append({"model": control, "control": pooled[control], "elo": pooled[elo],
                     "difference": pooled[elo] - pooled[control]}
                    | {k: season[k] for k in ["seasons", "seasons_better", "seasons_worse", "sign_test_p",
                                              "mean_difference", "ci_low", "ci_high",
                                              "leave_one_out_min", "leave_one_out_max"]})
    table = pd.DataFrame(rows).set_index("model").rename(
        columns={"ci_low": "season_t_low", "ci_high": "season_t_high"})
    table["sign_test_p_holm"] = holm_adjust(table["sign_test_p"])
    if metric != "macro_f1":  # macro F1 is not an average over matches, so no per-match bootstrap
        intervals = paired_intervals(predictions, pairs, metric, n_resamples=n_resamples).set_index("b")
        for column in ["ci_low", "ci_high", "adj_ci_low", "adj_ci_high"]:
            table[f"bootstrap_{column}"] = intervals[column]
    table.insert(0, "metric", metric)
    return table


def decision(predictions: pd.DataFrame, models=None, n_resamples: int = 10_000) -> pd.DataFrame:
    """Apply the pre-specified decision rule (module docstring) to every model."""
    table = comparison_table(predictions, models, "log_loss", n_resamples)
    out = pd.DataFrame(index=table.index)
    out["log_loss_difference"] = table["difference"]
    out["adjusted_interval_below_zero"] = table["bootstrap_adj_ci_high"] < 0
    out["seasons_better"] = table["seasons_better"].astype(int)
    out["enough_seasons"] = out["seasons_better"] >= MIN_SEASONS_BETTER
    out["survives_leave_one_out"] = table["leave_one_out_max"] < 0
    out["recommend_elo"] = out[["adjusted_interval_below_zero", "enough_seasons", "survives_leave_one_out"]].all(axis=1)
    return out


def promoted_flags(features: pd.DataFrame) -> pd.DataFrame:
    """Per fixture: does a newly promoted (or returning) club play in it?"""
    flags = features[FIXTURE_KEYS].copy()
    home = features["home_promoted"].fillna(False).astype(bool)
    away = features["away_promoted"].fillna(False).astype(bool)
    flags["promoted_club"] = (home | away).to_numpy()
    return flags


def subgroup_metrics(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics for fixtures with and without a promoted club (common fixtures only)."""
    rows = common_fixtures(predictions).merge(promoted_flags(features), on=FIXTURE_KEYS, how="left",
                                              validate="many_to_one")
    if rows["promoted_club"].isna().any():
        raise ValueError("Some predicted fixtures are not in the feature table")
    tables = []
    for flag, group in rows.groupby("promoted_club"):
        table = pooled_metrics(group.drop(columns="promoted_club"))
        table.insert(0, "promoted_club", bool(flag))
        tables.append(table)
    return pd.concat(tables).rename_axis("forecast").reset_index()


def subgroup_intervals(predictions: pd.DataFrame, features: pd.DataFrame, models=None,
                       metric: str = "log_loss", n_resamples: int = 10_000) -> pd.DataFrame:
    """Paired bootstrap (Elo - control) within fixtures with / without a promoted club."""
    rows = predictions.merge(promoted_flags(features), on=FIXTURE_KEYS, how="left", validate="many_to_one")
    tables = []
    for flag, group in rows.groupby("promoted_club"):
        table = paired_intervals(group.drop(columns="promoted_club"), model_pairs(models), metric,
                                 n_resamples=n_resamples)
        table.insert(0, "promoted_club", bool(flag))
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# ====================================================================== artifacts


def save_elo_experiment(backtest: Backtest, features: pd.DataFrame, settings: EloSettings = DEFAULT_SETTINGS,
                        directory: Path = ELO_DIR) -> dict[str, Path]:
    paths = save_backtest(backtest, directory)
    elo = pd.concat([features[["season_start", *FIXTURE_KEYS]], elo_features(features, settings)], axis=1)
    paths["elo_features"] = directory / "elo_features.csv"
    elo.to_csv(paths["elo_features"], index=False, date_format="%Y-%m-%d")
    paths["settings"] = directory / "settings.json"
    paths["settings"].write_text(json.dumps({"elo": settings.as_dict(), "model_features": ELO_FEATURE_COLUMNS,
                                             "descriptive_columns": [c for c in ELO_COLUMNS
                                                                     if c not in ELO_FEATURE_COLUMNS],
                                             "feature_sets": {k: len(v) for k, v in ELO_FEATURE_SETS.items()},
                                             "seasons": sorted(int(s) for s in backtest.predictions["season"].unique())},
                                            indent=2))
    return paths


def load_elo_experiment(directory: Path = ELO_DIR) -> Backtest:
    try:
        return load_backtest(directory)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{error} Run `python -m src.elo_experiment` first.") from None


def main(argv: list[str] | None = None) -> None:
    from src.feature_engineering import load_features
    from src.preprocessing import load_matches

    parser = argparse.ArgumentParser(description="Walk-forward comparison of models with and without Elo features.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--seasons", nargs="+", type=int, default=list(BACKTEST_SEASONS))
    args = parser.parse_args(argv)

    features = load_features()
    backtest = run_elo_backtest(features, load_matches(), seasons=args.seasons, models=args.models, progress=print)
    paths = save_elo_experiment(backtest, features)
    pd.set_option("display.width", 200)
    print("\nPooled (identical fixtures):")
    print(pooled_metrics(backtest.predictions).round(4).to_string())
    for metric in ("log_loss", "brier", "accuracy", "macro_f1"):
        print(f"\nElo minus control, {metric}:")
        print(comparison_table(backtest.predictions, args.models, metric).round(4).to_string())
    print("\nPre-specified decision rule:")
    print(decision(backtest.predictions, args.models).to_string())
    for name, path in paths.items():
        print(f"saved {name}: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
