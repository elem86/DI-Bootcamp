"""Walk-forward (expanding-window) backtest over past Serie A seasons.

Usage (from the project folder, after ``python -m src.feature_engineering``)::

    python -m src.backtest                       # all models, seasons 2018-19 to 2024-25
    python -m src.backtest --models logistic_regression

Results are saved to ``outputs/backtest/`` (git-ignored).

How one fold works (evaluation season S)
----------------------------------------
The fold only ever sees seasons up to S: later seasons are removed before
anything else happens. Then, exactly as in ``src.train.run_experiment``:

1. **Tune (past only):** fit every candidate (feature set x the model's
   hyperparameter grid) on seasons up to S-2 and score it on season S-1.
   The candidate with the lowest log loss on S-1 is selected.
2. **Refit:** fit the selected candidate on all seasons up to S-1.
3. **Evaluate:** predict every match of season S.

Season S's results are only used for scoring in step 3: they never enter
fitting, imputation, feature-set choice or hyperparameter choice.

Why past-only tuning, not the commit-7 settings
-----------------------------------------------
The settings chosen in commit 7 (e.g. Random Forest depth 12, 10 matches per
leaf) were selected on 2025-26. Reusing them for 2018-19 would use a later
season to configure an earlier forecast, so each fold selects its own settings
from the same predefined grids (``src/models.py``) using only earlier seasons.
One caveat remains: the *grid ranges* were set in commit 7, partly after
looking at 2025-26 results. The grids are broad, so this second-order effect
is small, but the backtest is not a perfectly blind experiment.

The 2025-26 validation season and the 2026-27 holdout are not touched:
backtest seasons must come before the 2026-27 test season, and the default
seasons end with 2024-25.

Benchmarks
----------
* **Baseline:** the outcome frequencies of the fold's training seasons.
* **Bookmaker:** Bet365 odds with the margin removed (``src/bookmaker.py``),
  used only for comparison, on fixtures with complete odds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.bookmaker import BOOKMAKER_LABEL, FIXTURE_KEYS, fixture_odds
from src.config import OUTPUTS_DIR
from src.evaluate import OUTCOMES, bootstrap_table, evaluate, predict_probabilities, probabilities_from_results
from src.feature_engineering import load_features
from src.models import MODEL_SPECS
from src.preprocessing import load_matches
from src.train import TEST_SEASON, chronological_split, feature_matrix, run_experiment, season_name

BACKTEST_SEASONS = tuple(range(2018, 2025))  # 2018-19 .. 2024-25
BACKTEST_DIR = OUTPUTS_DIR / "backtest"
BASELINE = "Baseline"
MODELS = [spec.label for spec in MODEL_SPECS.values()]
FORECASTS = MODELS + [BOOKMAKER_LABEL, BASELINE]
METRICS = ["accuracy", "macro_f1", "log_loss", "brier"]
PREDICTION_COLUMNS = ["season", "forecast", *FIXTURE_KEYS, "actual", *OUTCOMES, "has_odds"]


# ====================================================================== folds


@dataclass
class FoldResult:
    """One model evaluated on one season, with the settings chosen from earlier seasons."""

    season: int
    model_name: str
    first_train_season: int
    feature_set: str
    params: dict
    tuning_log_loss: float  # best log loss on the inner validation season (S-1)
    train_matches: int
    predictions: pd.DataFrame  # results_table of season S (src.evaluate.results_table)
    baseline: pd.DataFrame  # baseline probabilities H/D/A, aligned with predictions

    @property
    def label(self) -> str:
        return MODEL_SPECS[self.model_name].label

    @property
    def last_train_season(self) -> int:
        return self.season - 1

    @property
    def tuning_season(self) -> int:
        return self.season - 1


def check_season(season: int, features: pd.DataFrame) -> None:
    """Refuse the holdout season and seasons without two earlier seasons to learn from."""
    if season >= TEST_SEASON:
        raise ValueError(
            f"{season_name(season)} is not allowed: {season_name(TEST_SEASON)} is the untouched holdout season."
        )
    first = int(features["season_start"].min())
    if season - 2 < first:
        raise ValueError(
            f"{season_name(season)} needs at least two earlier seasons (tune on S-2, validate on S-1); "
            f"the data starts in {season_name(first)}."
        )


def run_fold(
    features: pd.DataFrame,
    season: int,
    model_name: str,
    param_grid: list[dict] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    strategy=None,
) -> FoldResult:
    """Tune on S-2 -> S-1, refit on everything before S, predict season S.

    ``strategy`` (optional, e.g. a ``src.recency.RecencyStrategy``) is applied to
    the training rows of the tuning fits and of the final fit alike.
    """
    check_season(season, features)
    # Nothing after the evaluation season exists for this fold.
    visible = features[features["season_start"] <= season]
    experiment = run_experiment(
        visible,
        train_last_season=season - 2,
        validation_season=season - 1,
        test_season=season,
        feature_sets=feature_sets,
        model_name=model_name,
        param_grid=param_grid,
        strategy=strategy,
    )
    final_train, evaluation = chronological_split(visible, season - 1, season)
    baseline = predict_probabilities(experiment.baseline, feature_matrix(evaluation, experiment.feature_columns))
    return FoldResult(
        season=season,
        model_name=model_name,
        first_train_season=int(final_train["season_start"].min()),
        feature_set=experiment.feature_set,
        params=experiment.params,
        tuning_log_loss=float(experiment.tuning.iloc[0]["log_loss"]),
        train_matches=len(final_train),
        predictions=experiment.test_results,
        baseline=baseline.reset_index(drop=True),
    )


# ====================================================================== backtest


@dataclass
class Backtest:
    """Per-match predictions of every forecast plus the settings each fold selected.

    ``predictions`` is long format: one row per (season, forecast, fixture)
    with probabilities ``H``, ``D``, ``A`` and ``has_odds`` (complete Bet365
    odds for that fixture). Bookmaker rows exist only where ``has_odds``.
    """

    predictions: pd.DataFrame
    folds: pd.DataFrame


def _long(season, forecast, fixtures, probs, has_odds):
    frame = fixtures[FIXTURE_KEYS + ["actual"]].reset_index(drop=True).copy()
    frame.insert(0, "forecast", forecast)
    frame.insert(0, "season", season)
    frame[OUTCOMES] = probs[OUTCOMES].to_numpy()
    frame["has_odds"] = has_odds
    return frame


def run_backtest(
    features: pd.DataFrame,
    matches: pd.DataFrame,
    seasons=BACKTEST_SEASONS,
    models=None,
    param_grids: dict[str, list[dict]] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    progress=None,
    strategy=None,
) -> Backtest:
    """Walk-forward evaluation of every model on every season, plus baseline and bookmaker."""
    models = list(models or MODEL_SPECS)
    param_grids = param_grids or {}
    frames, fold_rows = [], []
    for season in sorted(seasons):
        reference, baseline = None, None
        for model_name in models:
            fold = run_fold(features, season, model_name, param_grids.get(model_name), feature_sets, strategy)
            table = fold.predictions.sort_values(FIXTURE_KEYS, kind="stable")
            if reference is None:
                reference = table.reset_index(drop=True)
                baseline = fold.baseline.loc[table.index].reset_index(drop=True)
                odds = fixture_odds(reference, matches)  # checks fixtures and results against the match data
            elif not table[FIXTURE_KEYS + ["actual"]].reset_index(drop=True).equals(reference[FIXTURE_KEYS + ["actual"]]):
                raise ValueError(f"{fold.label} was evaluated on different {season_name(season)} fixtures")
            has_odds = odds["has_odds"].to_numpy()
            frames.append(_long(season, fold.label, reference,
                                probabilities_from_results(table.reset_index(drop=True)), has_odds))
            fold_rows.append({
                "season": season,
                "model": fold.label,
                "first_train_season": fold.first_train_season,
                "last_train_season": fold.last_train_season,
                "tuning_season": fold.tuning_season,
                "train_matches": fold.train_matches,
                "feature_set": fold.feature_set,
                "params": json.dumps(fold.params),
                "tuning_log_loss": fold.tuning_log_loss,
            })
            if progress:
                progress(f"{season_name(season)} {fold.label}: {fold.feature_set}, {fold.params}")
        frames.append(_long(season, BASELINE, reference, baseline, has_odds))
        book = odds[has_odds]
        frames.append(_long(season, BOOKMAKER_LABEL, reference[has_odds], book, True))
    predictions = pd.concat(frames, ignore_index=True)
    return Backtest(predictions=predictions[PREDICTION_COLUMNS], folds=pd.DataFrame(fold_rows))


# ====================================================================== metrics


def common_fixtures(predictions: pd.DataFrame, forecasts=None) -> pd.DataFrame:
    """Only fixtures with complete odds, checked to be identical for every forecast in each season."""
    forecasts = list(forecasts or predictions["forecast"].unique())
    common = predictions[predictions["has_odds"].astype(bool) & predictions["forecast"].isin(forecasts)]
    for season, rows in common.groupby("season"):
        keys = {f: set(map(tuple, g[FIXTURE_KEYS].astype(str).to_numpy())) for f, g in rows.groupby("forecast")}
        if set(keys) != set(forecasts):
            raise ValueError(f"{season_name(season)}: forecasts {sorted(set(forecasts) - set(keys))} are missing")
        if len({frozenset(k) for k in keys.values()}) != 1:
            raise ValueError(f"{season_name(season)}: forecasts cover different fixtures")
    return common.sort_values(["season", "forecast", *FIXTURE_KEYS], kind="stable").reset_index(drop=True)


def forecast_order(names) -> list[str]:
    """The standard forecasts in their usual order, then any others (e.g. recency strategies) as they appear."""
    names = list(dict.fromkeys(names))
    return [f for f in FORECASTS if f in names] + [n for n in names if n not in FORECASTS]


def _scores(rows: pd.DataFrame) -> dict:
    m = evaluate(rows["actual"].reset_index(drop=True), rows[OUTCOMES].reset_index(drop=True))
    return {"matches": m["matches"]} | {k: m[k] for k in METRICS}


def season_metrics(predictions: pd.DataFrame, common: bool = True) -> pd.DataFrame:
    """Accuracy, macro F1, log loss and Brier score per (season, forecast).

    With ``common=True`` (default) every forecast is scored on the same
    fixtures: those with complete bookmaker odds.
    """
    rows = common_fixtures(predictions) if common else predictions
    table = {key: _scores(group) for key, group in rows.groupby(["season", "forecast"], sort=True)}
    result = pd.DataFrame(table).T
    result.index.names = ["season", "forecast"]
    result["matches"] = result["matches"].astype(int)
    order = forecast_order(result.index.get_level_values("forecast"))
    return result.reindex(pd.MultiIndex.from_product([sorted(result.index.unique("season")), order],
                                                     names=["season", "forecast"]))


def pooled_metrics(predictions: pd.DataFrame, common: bool = True) -> pd.DataFrame:
    """Metrics over all seasons' per-match predictions pooled together (one row per forecast).

    Pooling weights every *match* equally: pooled log loss and Brier are the
    match-weighted means of the seasonal values, while pooled accuracy and
    macro F1 are computed from the pooled predictions themselves.
    """
    rows = common_fixtures(predictions) if common else predictions
    table = pd.DataFrame({forecast: _scores(group) for forecast, group in rows.groupby("forecast")}).T
    table["matches"] = table["matches"].astype(int)
    table["seasons"] = rows.groupby("forecast")["season"].nunique()
    return table.reindex(forecast_order(table.index))


def summary_table(predictions: pd.DataFrame, common: bool = True) -> pd.DataFrame:
    """Per forecast: mean of the seasonal scores (each season counts once) next to the pooled scores."""
    seasonal = season_metrics(predictions, common)
    mean = seasonal.groupby(level="forecast", sort=False)[METRICS].mean()
    worst = seasonal.groupby(level="forecast", sort=False)["log_loss"].max()
    best = seasonal.groupby(level="forecast", sort=False)["log_loss"].min()
    pooled = pooled_metrics(predictions, common)
    table = pd.DataFrame(index=pooled.index)
    table["seasons"] = pooled["seasons"]
    table["matches"] = pooled["matches"]
    for metric in METRICS:
        table[f"{metric}_season_mean"] = mean[metric]
        table[f"{metric}_pooled"] = pooled[metric]
    table["log_loss_best_season"] = best
    table["log_loss_worst_season"] = worst
    return table


def season_ranks(predictions: pd.DataFrame, metric: str = "log_loss", common: bool = True) -> pd.DataFrame:
    """Rank of every forecast within each season (1 = best); lower is better for losses."""
    seasonal = season_metrics(predictions, common)[metric].unstack("forecast")
    ascending = metric in ("log_loss", "brier")
    return seasonal.rank(axis=1, ascending=ascending, method="min").astype(int)


def coverage_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per season: fixtures predicted, fixtures with complete odds, coverage."""
    # The baseline has a row for every predicted fixture, whichever models were run
    reference = predictions[predictions["forecast"] == BASELINE]
    table = reference.groupby("season").agg(fixtures=("has_odds", "size"), with_odds=("has_odds", "sum"))
    table["coverage"] = table["with_odds"] / table["fixtures"]
    return table


def _bootstrap(rows: pd.DataFrame, pairs, metric: str, **kwargs) -> pd.DataFrame:
    order = ["season", *FIXTURE_KEYS]
    forecasts = {
        name: group.sort_values(order)[OUTCOMES].reset_index(drop=True) for name, group in rows.groupby("forecast")
    }
    reference = rows[rows["forecast"] == pairs[0][0]].sort_values(order)
    return bootstrap_table(reference["actual"].reset_index(drop=True), forecasts, pairs, metric=metric, **kwargs)


def pooled_bootstrap(predictions: pd.DataFrame, pairs, metric: str = "log_loss", **kwargs) -> pd.DataFrame:
    """Paired bootstrap of pooled per-match differences (all seasons, common fixtures)."""
    return _bootstrap(common_fixtures(predictions), pairs, metric, **kwargs)


def seasonal_bootstrap(predictions: pd.DataFrame, pairs, metric: str = "log_loss", **kwargs) -> pd.DataFrame:
    """The same paired bootstrap, separately for every season (one row per season and pair)."""
    rows = common_fixtures(predictions)
    tables = []
    for season, group in rows.groupby("season"):
        table = _bootstrap(group, pairs, metric, **kwargs)
        table.insert(0, "season", season)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# ====================================================================== artifacts


def save_backtest(backtest: Backtest, directory: Path = BACKTEST_DIR) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"predictions": directory / "predictions.csv", "folds": directory / "folds.csv"}
    backtest.predictions.to_csv(paths["predictions"], index=False, date_format="%Y-%m-%d")
    backtest.folds.to_csv(paths["folds"], index=False)
    return paths


def load_backtest(directory: Path = BACKTEST_DIR) -> Backtest:
    paths = {"predictions": directory / "predictions.csv", "folds": directory / "folds.csv"}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m src.backtest` first.")
    predictions = pd.read_csv(paths["predictions"], parse_dates=["date"]).astype({"date": "datetime64[ns]"})
    predictions["has_odds"] = predictions["has_odds"].astype(bool)
    return Backtest(predictions=predictions, folds=pd.read_csv(paths["folds"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest over past Serie A seasons.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--seasons", nargs="+", type=int, default=list(BACKTEST_SEASONS),
                        help="Evaluation seasons by starting year (default: 2018 ... 2024).")
    args = parser.parse_args(argv)

    backtest = run_backtest(load_features(), load_matches(), seasons=args.seasons, models=args.models,
                            progress=print)
    paths = save_backtest(backtest)
    pd.set_option("display.width", 160)
    print("\nCoverage (fixtures with complete Bet365 odds):")
    print(coverage_table(backtest.predictions).to_string())
    print("\nSeason mean vs pooled, on identical fixtures:")
    print(summary_table(backtest.predictions).round(4).to_string())
    for name, path in paths.items():
        print(f"saved {name}: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
