"""Walk-forward comparison of direct Over/Under and BTTS models with the Poisson-derived probabilities.

Usage (from the project folder, after ``python -m src.feature_engineering``)::

    python -m src.event_backtest                 # seasons 2018-19 to 2024-25

Results are saved to ``outputs/events/`` (git-ignored). The app, production
models, saved 1X2 predictions, the Elo experiment and the commit-15 results in
``outputs/goals/`` are not touched: the Poisson forecasts are recomputed here
with the same commit-15 fold code.

Methods (all scored on identical fixtures)
------------------------------------------
* **Event frequency**: Jeffreys-smoothed share of the event in the training seasons.
* **League average**, **Team attack/defence**, **Poisson regression**: the
  commit-15 Poisson forecasts (``src.goals_backtest.run_goals_fold``).
* **Direct logistic**: one Logistic Regression per event (``src.event_models``),
  Over lines made non-increasing by ``monotone_over``.
* **Poisson regression, recalibrated** (a separate experiment): the Poisson
  regression's probabilities passed through a logistic calibrator.

One fold (evaluation season S)
------------------------------
Seasons after S are removed first. Then, as in every earlier backtest:

1. **Tune:** each event's classifier is fitted on seasons up to S-2 for every
   feature set (``goals``, ``compact``, ``all``) x ``C`` in ``src.models.C_GRID``
   and scored by log loss on S-1. The Poisson methods choose their own settings
   the same way (commit 15).
2. **Refit** the chosen settings on all seasons up to S-1; the event
   frequencies use the same seasons.
3. **Evaluate** every completed match of S.

How refitting and calibration interact
--------------------------------------
A calibrator must learn from predictions that were genuinely out of sample.
For season S it is fitted on the Poisson regression's walk-forward predictions
for the ``CALIBRATION_SEASONS`` (3) seasons S-3, S-2 and S-1. The prediction for
each of those seasons s came from its own fold (trained on seasons up to s-1,
tuned on s-1 after fitting up to s-2), so no calibration season was ever
used to fit the predictions it calibrates. The fitted calibrator is then
applied to season S's Poisson predictions, from the fold that refits on
seasons up to S-1. Season S is never used to fit the calibrator.

The base model is refitted every season, so the calibrator learns the
*typical* miscalibration of recent folds and assumes the next fold behaves
similarly. That is a real assumption: if the base model changes character
from one fold to the next (for example a different penalty), the correction
can be off. The window length (3 seasons) is fixed in advance, not tuned.
Each line and BTTS has its own calibrator; the calibrated Over lines are then
made non-increasing with the same ``monotone_over`` correction.

2025-26 and 2026-27 are refused for fitting, tuning, calibration and evaluation.
Bookmaker odds are never a feature.

Pre-specified decision rule
---------------------------
Written before the experiment was run. Primary metrics: **Over 2.5 log loss**
and **BTTS log loss**, judged separately. The candidates are the direct
logistic models and the recalibrated Poisson regression. The comparator
"best existing Poisson" is, for each event, whichever of the three commit-15
Poisson methods has the lowest pooled log loss on that event (the hardest
existing benchmark). The family of comparisons for the rule is 2 candidates x
2 events x 2 comparators = 8, with Bonferroni-adjusted bootstrap intervals.

A candidate is **ready for the app for an event** only if all of these hold:

1. **Beats the event frequency:** the Bonferroni-adjusted interval of the
   pooled log-loss difference is entirely below 0.
2. **Competitive with the best Poisson:** its pooled log loss is not higher.
3. **Consistent:** it beats the event frequency in at least 5 of the 7
   seasons, and the pooled difference stays below 0 when any one season is
   left out.
4. **Calibrated:** mean probability within 2 percentage points of the observed
   share, calibration slope between 0.8 and 1.2, and at most 1 of 10
   equal-count bins with a 95% Wilson interval that excludes the bin's mean
   prediction.

Accuracy is not used to choose anything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.backtest import BACKTEST_SEASONS
from src.config import OUTPUTS_DIR
from src.elo_experiment import holm_adjust
from src.event_models import (
    EVENT_NAMES,
    OVER_EVENTS,
    PRIMARY_EVENTS,
    DirectEventClassifier,
    EventChoice,
    EventFrequency,
    LogisticCalibrator,
    binary_log_loss,
    check_valid,
    crossing_fixtures,
    event_targets,
    monotone_over,
    tune_event,
)
from src.goals_backtest import (
    FIXTURE_KEYS,
    LEAGUE,
    MODEL,
    TEAM,
    _paired,
    bootstrap_means,
    calibration_slope,
    check_goals_season,
    check_identical_fixtures,
    reliability_table,
    run_goals_fold,
    with_probabilities,
)
from src.goals_model import EVENTS, GOALS_FEATURE_SETS, played
from src.models import C_GRID
from src.train import season_name

EVENTS_DIR = OUTPUTS_DIR / "events"
FREQUENCY = EventFrequency.label
DIRECT = DirectEventClassifier.label
RECALIBRATED = "Poisson regression, recalibrated"
POISSON_METHODS = [LEAGUE, TEAM, MODEL]
MAIN_METHODS = [FREQUENCY, *POISSON_METHODS, DIRECT]
METHODS = [*MAIN_METHODS, RECALIBRATED]
CANDIDATES = [DIRECT, RECALIBRATED]
CALIBRATION_SEASONS = 3

PRIMARY_METRICS = [f"log_loss_{e}" for e in PRIMARY_EVENTS]
LOSS_METRICS = [f"{kind}_{e}" for e in EVENT_NAMES for kind in ("log_loss", "brier")]
FAMILY_WISE_ALPHA = 0.05
RULE_COMPARISONS = len(CANDIDATES) * len(PRIMARY_EVENTS) * 2
MIN_SEASONS_BETTER = 5
CALIBRATION_TOLERANCE = 0.02
SLOPE_RANGE = (0.8, 1.2)
CALIBRATION_BINS = 10
MAX_BINS_OUTSIDE = 1

ID_COLUMNS = ["season", "method", *FIXTURE_KEYS, "home_goals", "away_goals", "promoted_club"]
PREDICTION_COLUMNS = ID_COLUMNS + EVENT_NAMES


def metric_label(metric: str) -> str:
    for prefix, name in (("log_loss_", "Log loss"), ("brier_", "Brier")):
        if metric.startswith(prefix):
            event = metric[len(prefix):]
            return f"{name}, {EVENTS.get(event, event)}"
    return metric


# ====================================================================== folds


@dataclass
class EventBacktest:
    """``predictions``: one row per (season, method, fixture) with the five event probabilities.

    ``uncorrected`` holds the direct and recalibrated forecasts *before* the
    monotone correction, for the consistency analysis only.
    """

    predictions: pd.DataFrame
    uncorrected: pd.DataFrame
    folds: pd.DataFrame
    tuning: pd.DataFrame
    calibrators: pd.DataFrame


def _frame(season: int, method: str, base: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    frame.insert(0, "method", method)
    frame.insert(0, "season", season)
    frame[EVENT_NAMES] = probabilities[EVENT_NAMES].to_numpy(dtype="float64")
    return frame[PREDICTION_COLUMNS]


def _aligned(source: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``source`` in the order of ``base``'s fixtures (every fixture exactly once)."""
    merged = base[FIXTURE_KEYS].merge(source, on=FIXTURE_KEYS, how="left", validate="one_to_one")
    if merged[EVENT_NAMES].isna().any().any():
        raise ValueError("A forecast is missing some of the evaluated fixtures")
    merged.index = base.index
    return merged


def calibration_window(season: int, length: int = CALIBRATION_SEASONS) -> list[int]:
    """The seasons whose out-of-sample predictions calibrate season ``season``: S-length .. S-1."""
    if length < 1:
        raise ValueError("The calibration window needs at least one season")
    return list(range(season - length, season))


def fit_calibrators(history: dict[int, pd.DataFrame], season: int,
                    length: int = CALIBRATION_SEASONS) -> dict[str, LogisticCalibrator]:
    """One calibrator per event from the Poisson regression's predictions of the window seasons.

    ``history[s]`` must hold season s's walk-forward predictions (with probabilities).
    Seasons at or after ``season`` are refused.
    """
    window = calibration_window(season, length)
    if any(s >= season for s in window):
        raise ValueError("A calibrator cannot use the season it calibrates")
    missing = [s for s in window if s not in history]
    if missing:
        raise ValueError(f"Out-of-sample predictions missing for calibration seasons {missing}")
    rows = pd.concat([history[s][history[s]["method"] == MODEL] for s in window], ignore_index=True)
    if (rows["season"] >= season).any():
        raise ValueError("Calibration rows include the evaluated season or later")
    targets = event_targets(rows)
    return {e: LogisticCalibrator().fit(rows[e], targets[e]) for e in EVENT_NAMES}


def run_event_backtest(features: pd.DataFrame, seasons=BACKTEST_SEASONS, calibration_seasons: int = CALIBRATION_SEASONS,
                       feature_sets=None, c_grid=C_GRID, goal_grids: dict | None = None, progress=None) -> EventBacktest:
    """Walk-forward evaluation of every method (see the module docstring)."""
    seasons = sorted(seasons)
    feature_sets = feature_sets or GOALS_FEATURE_SETS
    goal_grids = goal_grids or {}
    needed = sorted(set(range(min(seasons) - calibration_seasons, max(seasons) + 1)))
    for season in needed:
        check_goals_season(season, features)  # refuses 2025-26 and later, and seasons without two earlier ones

    # Commit-15 Poisson forecasts, out of sample for every season they cover
    history = {}
    for season in needed:
        history[season] = with_probabilities(run_goals_fold(features, season, **goal_grids).predictions)
        if progress:
            progress(f"Poisson fold {season_name(season)} done")

    frames, raw_frames, folds, tunings, calibrators = [], [], [], [], []
    for season in seasons:
        visible = played(features[features["season_start"] <= season])
        inner_train = visible[visible["season_start"] <= season - 2]
        inner_validation = visible[visible["season_start"] == season - 1]
        train = visible[visible["season_start"] <= season - 1]
        evaluation = visible[visible["season_start"] == season]

        choices = {}
        for event in EVENT_NAMES:
            table = tune_event(inner_train, inner_validation, event, feature_sets, c_grid)
            choices[event] = EventChoice(str(table.iloc[0]["feature_set"]), float(table.iloc[0]["C"]))
            tunings.append(table.assign(season=season))
        direct = DirectEventClassifier(choices, dict(feature_sets)).fit(train)
        frequency = EventFrequency().fit(train)

        base = evaluation[FIXTURE_KEYS + ["home_goals", "away_goals"]].astype({"home_goals": "int64",
                                                                               "away_goals": "int64"})
        promoted = evaluation[["home_promoted", "away_promoted"]].astype("boolean").fillna(False).astype(bool)
        base = base.assign(promoted_club=promoted.any(axis=1))

        raw_direct = direct.predict(evaluation, monotone=False)
        corrected = raw_direct.copy()
        corrected[OVER_EVENTS] = monotone_over(raw_direct)
        frames.append(_frame(season, FREQUENCY, base, frequency.predict(evaluation)))
        poisson = history[season]
        for method in POISSON_METHODS:
            frames.append(_frame(season, method, base, _aligned(poisson[poisson["method"] == method], base)))
        frames.append(_frame(season, DIRECT, base, corrected))

        fitted = fit_calibrators(history, season, calibration_seasons)
        model_rows = _aligned(poisson[poisson["method"] == MODEL], base)
        raw_calibrated = pd.DataFrame({e: fitted[e].transform(model_rows[e]) for e in EVENT_NAMES}, index=base.index)
        calibrated = raw_calibrated.copy()
        calibrated[OVER_EVENTS] = monotone_over(raw_calibrated)
        frames.append(_frame(season, RECALIBRATED, base, calibrated))

        raw_frames += [_frame(season, DIRECT, base, raw_direct), _frame(season, RECALIBRATED, base, raw_calibrated)]
        window = calibration_window(season, calibration_seasons)
        for event, calibrator in fitted.items():
            calibrators.append({"season": season, "event": event, "intercept": calibrator.intercept,
                                "slope": calibrator.slope, "fitted": calibrator.fitted,
                                "calibration_matches": calibrator.matches,
                                "first_calibration_season": window[0], "last_calibration_season": window[-1]})
        folds.append({"season": season, "last_train_season": season - 1, "tuning_season": season - 1,
                      "train_matches": len(train), "evaluation_matches": len(evaluation),
                      "crossing_fixtures": int(crossing_fixtures(raw_direct).sum())}
                     | direct.params | {f"frequency_{e}": frequency.rates_[e] for e in EVENT_NAMES})
        if progress:
            progress(f"{season_name(season)}: direct choices {direct.params}")

    predictions = pd.concat(frames, ignore_index=True)
    check_valid(predictions)
    check_identical_fixtures(predictions, METHODS)
    return EventBacktest(predictions, pd.concat(raw_frames, ignore_index=True), pd.DataFrame(folds),
                         pd.concat(tunings, ignore_index=True), pd.DataFrame(calibrators))


# ====================================================================== scores


def event_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-match log loss and Brier score of every event (lower is better)."""
    targets = event_targets(predictions)
    table = predictions[ID_COLUMNS].copy()
    for event in EVENT_NAMES:
        table[f"log_loss_{event}"] = binary_log_loss(predictions[event], targets[event])
        table[f"brier_{event}"] = (predictions[event] - targets[event]) ** 2
    return table


def _order(predictions: pd.DataFrame) -> list[str]:
    names = list(dict.fromkeys(predictions["method"]))
    return [m for m in METHODS if m in names] + [m for m in names if m not in METHODS]


def _summary(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    losses = event_losses(predictions)
    targets = event_targets(predictions)
    frame = losses[LOSS_METRICS].copy()
    for event in EVENT_NAMES:
        frame[f"mean_{event}"] = predictions[event]
        frame[f"observed_{event}"] = targets[event]
    frame["matches"] = 1
    keys = [predictions[k] for k in by]
    return frame.groupby(keys, sort=False).agg({c: ("sum" if c == "matches" else "mean") for c in frame.columns})


def season_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean scores, mean probability and observed share of every event per (season, method)."""
    table = _summary(predictions, ["season", "method"])
    index = pd.MultiIndex.from_product([sorted(predictions["season"].unique()), _order(predictions)],
                                       names=["season", "method"])
    return table.reindex(index)


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """The same over all seasons pooled (every match counts once)."""
    table = _summary(predictions, ["method"])
    table["seasons"] = predictions.groupby("method")["season"].nunique()
    return table.reindex(_order(predictions))


def best_poisson(predictions: pd.DataFrame, event: str) -> str:
    """The commit-15 Poisson method with the lowest pooled log loss on ``event``."""
    pooled = pooled_metrics(predictions[predictions["method"].isin(POISSON_METHODS)])
    return str(pooled[f"log_loss_{event}"].idxmin())


def compare(predictions: pd.DataFrame, triples, n_comparisons: int | None = None,
            n_resamples: int = 10_000) -> pd.DataFrame:
    """Pooled paired bootstrap of ``a - b`` for each ``(a, b, metric)`` (negative = a better).

    Gives a 95% interval, a Bonferroni-adjusted interval for a family of
    ``n_comparisons`` (default: the number of triples), a two-sided bootstrap
    p-value and its Holm adjustment within the table.
    """
    losses = event_losses(predictions)
    triples = list(triples)
    n_comparisons = n_comparisons or len(triples)
    adjusted = 1 - FAMILY_WISE_ALPHA / n_comparisons
    rows = []
    for a, b, metric in triples:
        diff, _ = _paired(losses, a, b, metric)
        means = bootstrap_means(diff, n_resamples)
        tail = min((means <= 0).mean(), (means >= 0).mean())
        low, high = np.quantile(means, [0.025, 0.975])
        adj_low, adj_high = np.quantile(means, [(1 - adjusted) / 2, 1 - (1 - adjusted) / 2])
        rows.append({"a": a, "b": b, "metric": metric, "label": metric_label(metric), "matches": len(diff),
                     "difference": float(diff.mean()), "ci_low": low, "ci_high": high,
                     "adjusted_confidence": adjusted, "adj_ci_low": adj_low, "adj_ci_high": adj_high,
                     "p_value": float(min(1.0, max(2 * tail, 1 / n_resamples)))})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_value"])
    table["adjusted_interval_excludes_zero"] = (table["adj_ci_high"] < 0) | (table["adj_ci_low"] > 0)
    return table


def paired_comparisons(predictions: pd.DataFrame, pairs, metrics=PRIMARY_METRICS, n_comparisons: int | None = None,
                       n_resamples: int = 10_000) -> pd.DataFrame:
    """``compare`` for every pair x metric."""
    return compare(predictions, [(a, b, m) for a, b in pairs for m in metrics], n_comparisons, n_resamples)


def rule_comparisons(predictions: pd.DataFrame, n_resamples: int = 10_000) -> pd.DataFrame:
    """The decision rule's family: each candidate vs the event frequency and vs the best Poisson,
    on Over 2.5 and BTTS log loss (8 comparisons, Bonferroni-adjusted for 8)."""
    triples = []
    for candidate in CANDIDATES:
        for event in PRIMARY_EVENTS:
            triples.append((candidate, FREQUENCY, f"log_loss_{event}"))
            triples.append((candidate, best_poisson(predictions, event), f"log_loss_{event}"))
    return compare(predictions, triples, RULE_COMPARISONS, n_resamples)


def season_sensitivity(predictions: pd.DataFrame, pairs, metrics=PRIMARY_METRICS) -> pd.DataFrame:
    """Seasons won, a sign test and leave-one-season-out range of ``a - b`` for every pair and metric."""
    losses = event_losses(predictions)
    rows = []
    for a, b in pairs:
        for metric in metrics:
            diff, keys = _paired(losses, a, b, metric)
            seasons = keys["season"].to_numpy()
            per_season = pd.Series(diff).groupby(seasons).mean()
            better = int((per_season < 0).sum())
            loo = [diff[seasons != s].mean() for s in per_season.index] if len(per_season) > 1 else [np.nan]
            rows.append({"a": a, "b": b, "metric": metric, "label": metric_label(metric),
                         "seasons": len(per_season), "seasons_better": better,
                         "sign_test_p": binomtest(better, len(per_season), 0.5).pvalue,
                         "pooled_difference": float(diff.mean()),
                         "worst_season_difference": float(per_season.max()),
                         "leave_one_out_min": float(np.min(loo)), "leave_one_out_max": float(np.max(loo))})
    return pd.DataFrame(rows)


def season_differences(predictions: pd.DataFrame, metric: str, a: str, b: str = FREQUENCY) -> pd.Series:
    diff, keys = _paired(event_losses(predictions), a, b, metric)
    return pd.Series(diff).groupby(keys["season"].to_numpy()).mean().rename(a)


def calibration_table(predictions: pd.DataFrame, n_bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Per method and event: mean vs observed, calibration intercept and slope, ECE and bins outside."""
    targets = event_targets(predictions)
    rows = []
    for method in _order(predictions):
        mask = (predictions["method"] == method).to_numpy()
        for event in EVENT_NAMES:
            p, y = predictions.loc[mask, event], targets.loc[mask, event]
            table = reliability_table(p, y, n_bins)
            gap = (table["observed"] - table["mean_predicted"]).abs()
            varies = p.round(6).nunique() >= 20  # a near-constant forecast has no slope worth estimating
            intercept, slope = calibration_slope(p, y) if varies else (np.nan, np.nan)
            rows.append({"method": method, "event": event, "label": EVENTS[event], "matches": int(mask.sum()),
                         "mean_predicted": float(p.mean()), "observed": float(y.mean()),
                         "difference": float(p.mean() - y.mean()),
                         "calibration_intercept": intercept, "calibration_slope": slope,
                         "ece": float((gap * table["matches"]).sum() / table["matches"].sum()),
                         "bins_outside": int(table["outside"].sum()), "bins": len(table)})
    return pd.DataFrame(rows)


# ====================================================================== probability consistency


def crossing_table(backtest: EventBacktest) -> pd.DataFrame:
    """How often the Over lines had to be repaired, per method and season, and by how much."""
    raw = backtest.uncorrected
    rows = []
    for (season, method), group in raw.groupby(["season", "method"], sort=True):
        fixed = monotone_over(group)
        change = (fixed[OVER_EVENTS] - group[OVER_EVENTS]).abs().max(axis=1)
        crossing = crossing_fixtures(group)
        rows.append({"season": season, "method": method, "fixtures": len(group), "crossing": int(crossing.sum()),
                     "share": float(crossing.mean()),
                     "mean_change_when_crossing": float(change[crossing].mean()) if crossing.any() else 0.0,
                     "max_change": float(change.max())})
    return pd.DataFrame(rows)


def correction_effect(backtest: EventBacktest) -> pd.DataFrame:
    """Pooled Over-line log loss before and after the monotone correction (identical fixtures)."""
    keys = ["season", "method", *FIXTURE_KEYS]
    raw = backtest.uncorrected.sort_values(keys).reset_index(drop=True)
    fixed = backtest.predictions[backtest.predictions["method"].isin(raw["method"].unique())]
    fixed = fixed.sort_values(keys).reset_index(drop=True)
    if not raw[keys].astype(str).equals(fixed[keys].astype(str)):
        raise ValueError("Corrected and uncorrected forecasts cover different fixtures")
    targets = event_targets(raw)
    rows = []
    for method in raw["method"].unique():
        mask = (raw["method"] == method).to_numpy()
        for event in OVER_EVENTS:
            before = binary_log_loss(raw.loc[mask, event], targets.loc[mask, event]).mean()
            after = binary_log_loss(fixed.loc[mask, event], targets.loc[mask, event]).mean()
            rows.append({"method": method, "event": event, "label": EVENTS[event], "log_loss_uncorrected": before,
                         "log_loss_corrected": after, "difference": after - before,
                         "fixtures_changed": int((raw.loc[mask, event] != fixed.loc[mask, event]).sum())})
    return pd.DataFrame(rows)


# ====================================================================== decision rule


def readiness(predictions: pd.DataFrame, n_resamples: int = 10_000) -> pd.DataFrame:
    """Apply the pre-specified rule (module docstring) to every candidate and primary event."""
    family = rule_comparisons(predictions, n_resamples).set_index(["a", "b", "metric"])
    calibration = calibration_table(predictions).set_index(["method", "event"])
    sensitivity = season_sensitivity(predictions, [(c, FREQUENCY) for c in CANDIDATES]).set_index(["a", "metric"])
    rows = []
    for candidate in CANDIDATES:
        for event in PRIMARY_EVENTS:
            metric = f"log_loss_{event}"
            comparator = best_poisson(predictions, event)
            vs_frequency = family.loc[(candidate, FREQUENCY, metric)]
            vs_poisson = family.loc[(candidate, comparator, metric)]
            season = sensitivity.loc[(candidate, metric)]
            cal = calibration.loc[(candidate, event)]
            slope_ok = SLOPE_RANGE[0] <= cal["calibration_slope"] <= SLOPE_RANGE[1]
            checks = {
                "beats_frequency": bool(vs_frequency["adj_ci_high"] < 0),
                "competitive_with_poisson": bool(vs_poisson["difference"] <= 0),
                "consistent": bool(season["seasons_better"] >= MIN_SEASONS_BETTER and season["leave_one_out_max"] < 0),
                "calibrated": bool(abs(cal["difference"]) <= CALIBRATION_TOLERANCE and slope_ok
                                   and cal["bins_outside"] <= MAX_BINS_OUTSIDE),
            }
            rows.append({"candidate": candidate, "event": EVENTS[event],
                         "vs_frequency": float(vs_frequency["difference"]),
                         "vs_frequency_adj_ci_high": float(vs_frequency["adj_ci_high"]),
                         "best_poisson": comparator, "vs_best_poisson": float(vs_poisson["difference"]),
                         "seasons_better": int(season["seasons_better"]),
                         "leave_one_out_max": float(season["leave_one_out_max"]),
                         "mean_minus_observed": float(cal["difference"]),
                         "calibration_slope": float(cal["calibration_slope"]),
                         "bins_outside": int(cal["bins_outside"])} | checks | {"ready": all(checks.values())})
    return pd.DataFrame(rows)


# ====================================================================== artifacts


FILES = ("predictions", "uncorrected", "folds", "tuning", "calibrators")


def save_event_backtest(backtest: EventBacktest, directory: Path = EVENTS_DIR) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.csv" for name in FILES}
    for name in FILES:
        getattr(backtest, name).to_csv(paths[name], index=False, date_format="%Y-%m-%d")
    paths["settings"] = directory / "settings.json"
    paths["settings"].write_text(json.dumps({
        "seasons": sorted(int(s) for s in backtest.predictions["season"].unique()),
        "methods": METHODS, "calibration_seasons": CALIBRATION_SEASONS, "c_grid": list(C_GRID),
        "feature_sets": {k: len(v) for k, v in GOALS_FEATURE_SETS.items()},
        "primary_metrics": PRIMARY_METRICS, "rule_comparisons": RULE_COMPARISONS,
        "slope_range": list(SLOPE_RANGE), "calibration_tolerance": CALIBRATION_TOLERANCE,
    }, indent=2))
    return paths


def load_event_backtest(directory: Path = EVENTS_DIR) -> EventBacktest:
    paths = {name: directory / f"{name}.csv" for name in FILES}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m src.event_backtest` first.")
    frames = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, parse_dates=["date"] if name in ("predictions", "uncorrected") else None)
        if "date" in frame.columns and name in ("predictions", "uncorrected"):
            frame["date"] = frame["date"].astype("datetime64[ns]")
            for column in ("home_team", "away_team", "method"):
                frame[column] = frame[column].astype("string")
        frames[name] = frame
    return EventBacktest(**frames)


def main(argv: list[str] | None = None) -> None:
    from src.feature_engineering import load_features

    parser = argparse.ArgumentParser(description="Direct Over/Under and BTTS models vs Poisson-derived probabilities.")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(BACKTEST_SEASONS))
    args = parser.parse_args(argv)

    backtest = run_event_backtest(load_features(), seasons=args.seasons, progress=print)
    paths = save_event_backtest(backtest)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pooled = pooled_metrics(backtest.predictions)
    print("\nPooled log loss:")
    print(pooled[[f"log_loss_{e}" for e in EVENT_NAMES]].round(4).to_string())
    print("\nCrossing Over probabilities (direct models, before correction):")
    print(crossing_table(backtest).round(4).to_string())
    print("\nPre-specified decision rule:")
    print(readiness(backtest.predictions).round(4).to_string())
    for name, path in paths.items():
        print(f"saved {name}: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
