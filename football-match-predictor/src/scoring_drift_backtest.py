"""Walk-forward test of adaptive league scoring rates against the existing goals-market forecasts.

Usage (from the project folder, after ``python -m src.goals_backtest`` and
``python -m src.event_backtest``)::

    python -m src.scoring_drift_backtest         # seasons 2018-19 to 2024-25

Hypothesis (not assumed to hold)
--------------------------------
Scoring in Serie A changes from season to season and within seasons. Methods
that follow the most recent completed league results should forecast Over/Under
and BTTS better than methods fixed at the level of whole earlier seasons.

Reuse of earlier results
------------------------
Nothing is refitted that already exists. The six existing methods are read
from the commit-16 predictions in ``outputs/events/``. The Poisson regression's
expected goals come from the commit-15 predictions in ``outputs/goals/``. Both
are checked against each other and against the training seasons before use,
and neither folder is written to. Results go to ``outputs/scoring_drift/``
(git-ignored). The app and production models are not touched.

Methods (identical fixtures, 2018-19 to 2024-25)
------------------------------------------------
Existing: **Event frequency** (static, Jeffreys-smoothed share of the training
seasons), **League average**, **Team attack/defence**, **Poisson regression**
(commit 15), **Direct logistic** and **Poisson regression, recalibrated**
(commit 16).

New (``src.scoring_drift``; window 200 matches and 200 pseudo-matches of
shrinkage, fixed in advance):

* **Adaptive event frequency** (candidate A): the window's event share shrunk
  toward the static event frequency.
* **Adaptive Poisson regression** (candidate B): the Poisson regression's
  expected goals rescaled by the shrunk recent home and away scoring levels.

Chronology for evaluation season S
----------------------------------
* **Fitted on seasons up to S-1 only:** the static frequencies, the training
  mean home and away goals, and the Poisson regression (its own commit-15 fold).
* **Updated online during S:** the 200-match window, from matches dated
  strictly before each fixture. Earlier fixtures of S may enter it, as they
  would in real use; the fixture itself, other fixtures on the same date and
  later results never do.
* 2025-26 and 2026-27 are refused and are removed from the history before any
  window is built. Bookmaker odds are never an input.

Pre-specified decision rule
---------------------------
Written before the experiment was run, and judged separately for **Over 2.5
log loss** and **BTTS log loss** (the primary metrics). For each event the
"strongest existing model" is whichever existing method other than the static
event frequency has the lowest pooled log loss on that event. The rule's family
is 2 candidates x 2 events x 2 comparators (static frequency, strongest
existing model) = 8 comparisons, with Bonferroni-adjusted paired bootstrap
intervals (and Holm-adjusted p-values reported).

A candidate is **ready for an event** only if all of these hold:

1. **Beats the static frequency:** the Bonferroni-adjusted interval of the
   pooled log-loss difference lies entirely below 0.
2. **Competitive:** its pooled log loss is not higher than the strongest
   existing model's.
3. **Season consistency:** it has lower log loss than the static frequency
   in at least 5 of the 7 seasons.
4. **No single-season dependence:** the pooled difference against the static
   frequency stays below 0 when any one season is left out.
5. **Calibrated:** calibration slope between 0.8 and 1.2, mean probability
   within 2 percentage points of the observed share, and at most 1 of 10
   equal-count bins whose 95% Wilson interval excludes the bin's mean
   prediction.

The window and shrinkage are not changed after seeing the results.

**These results are exploratory.** The 2018-19 to 2024-25 seasons have already
shaped earlier modelling decisions (and the idea for this experiment came from
them), so even a pass here would not be independent confirmation. That would
need seasons not yet used, such as 2025-26 once it is released for evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import BACKTEST_SEASONS
from src.config import OUTPUTS_DIR
from src.event_backtest import (
    FREQUENCY,
    PREDICTION_COLUMNS,
    calibration_table,
    compare,
    event_losses,
    pooled_metrics,
    season_sensitivity,
)
from src.event_backtest import METHODS as EXISTING_METHODS
from src.event_models import EVENT_NAMES, PRIMARY_EVENTS, binary_log_loss, check_valid, event_targets
from src.goals_backtest import LEAGUE, MODEL, check_goals_season, check_identical_fixtures
from src.goals_model import EVENTS, FIXTURE_KEYS, LAMBDA_COLUMNS
from src.scoring_drift import (
    PRIOR_MATCHES,
    WINDOW_MATCHES,
    AdaptiveEventFrequency,
    AdaptiveScoringLevel,
    completed_history,
    poisson_events,
)
from src.train import season_name

DRIFT_DIR = OUTPUTS_DIR / "scoring_drift"
ADAPTIVE_FREQUENCY = AdaptiveEventFrequency.label
ADAPTIVE_POISSON = AdaptiveScoringLevel.label
CANDIDATES = [ADAPTIVE_FREQUENCY, ADAPTIVE_POISSON]
METHODS = [*EXISTING_METHODS, *CANDIDATES]
MODELS = [m for m in EXISTING_METHODS if m != FREQUENCY]  # pool for the "strongest existing model"

PRIMARY_METRICS = [f"log_loss_{e}" for e in PRIMARY_EVENTS]
SECONDARY_METRICS = ([f"log_loss_{e}" for e in EVENT_NAMES if e not in PRIMARY_EVENTS]
                     + [f"brier_{e}" for e in EVENT_NAMES])
RULE_COMPARISONS = len(CANDIDATES) * len(PRIMARY_EVENTS) * 2
MIN_SEASONS_BETTER = 5
CALIBRATION_TOLERANCE = 0.02
SLOPE_RANGE = (0.8, 1.2)
MAX_BINS_OUTSIDE = 1
CONSISTENCY_TOLERANCE = 1e-9

DRIFT_COLUMNS = ["season", *FIXTURE_KEYS, "home_goals", "away_goals", "window_matches", "window_first_date",
                 "window_last_date", "recent_home", "recent_away", "train_home", "train_away", "level_home",
                 "level_away", "home_factor", "away_factor", "lambda_home", "lambda_away", "lambda_home_adjusted",
                 "lambda_away_adjusted"]


# ====================================================================== folds


@dataclass
class DriftBacktest:
    """``predictions``: one row per (season, method, fixture) with the five event probabilities.

    ``drift`` has one row per fixture with the window behind the adaptive
    forecasts and the Poisson rates before and after adjustment; ``folds``
    has one row per season.
    """

    predictions: pd.DataFrame
    drift: pd.DataFrame
    folds: pd.DataFrame


def _aligned(source: pd.DataFrame, base: pd.DataFrame, columns: list[str], what: str) -> pd.DataFrame:
    merged = base[FIXTURE_KEYS].merge(source[FIXTURE_KEYS + columns], on=FIXTURE_KEYS, how="left",
                                      validate="one_to_one")
    if merged[columns].isna().any().any():
        raise ValueError(f"{what} is missing some of the evaluated fixtures")
    merged.index = base.index
    return merged


def _frame(season: int, method: str, base: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    frame["season"] = season
    frame["method"] = method
    frame[EVENT_NAMES] = probabilities[EVENT_NAMES].to_numpy(dtype="float64")
    return frame[PREDICTION_COLUMNS]


def _check_close(a, b, message: str) -> None:
    gap = np.max(np.abs(np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64")), initial=0.0)
    if not gap <= CONSISTENCY_TOLERANCE:
        raise ValueError(f"{message} (largest difference {gap:.2e})")


def _first_window_season(history: pd.DataFrame, date, window: int) -> int | None:
    """Earliest season in the window of the season's first fixture (None if the window is empty)."""
    earlier = history.loc[history["date"] < date, "season_start"].iloc[-window:]
    return int(earlier.min()) if len(earlier) else None


def run_scoring_drift_backtest(matches: pd.DataFrame, goals_predictions: pd.DataFrame,
                               event_predictions: pd.DataFrame, seasons=BACKTEST_SEASONS,
                               window: int = WINDOW_MATCHES, prior_matches: float = PRIOR_MATCHES) -> DriftBacktest:
    """Add the two adaptive candidates to the existing forecasts (see the module docstring).

    ``matches``: the match or feature table (completed results give the
    training means and the online window). ``goals_predictions``: commit-15
    rows with ``lambda_home`` / ``lambda_away``. ``event_predictions``:
    commit-16 rows of the six existing methods.
    """
    seasons = sorted(int(s) for s in seasons)
    for season in seasons:
        check_goals_season(season, matches)  # refuses 2025-26 and later
    history = completed_history(matches)
    existing = event_predictions[event_predictions["season"].isin(seasons)]
    missing = [m for m in EXISTING_METHODS if m not in set(existing["method"])]
    if missing:
        raise ValueError(f"The existing predictions lack {missing}")
    existing = existing[existing["method"].isin(EXISTING_METHODS)][PREDICTION_COLUMNS]

    frames, drift_frames, folds = [existing], [], []
    for season in seasons:
        visible = history[history["season_start"] <= season]  # later seasons cannot enter any window
        train = visible[visible["season_start"] <= season - 1]
        rows = existing[existing["season"] == season]
        base = rows[rows["method"] == FREQUENCY].drop(columns=["season", "method", *EVENT_NAMES])
        base = base.sort_values(FIXTURE_KEYS, kind="stable").reset_index(drop=True)
        if base.empty:
            raise ValueError(f"No existing predictions for {season_name(season)}")
        played_in_season = _aligned(visible[visible["season_start"] == season], base,
                                    ["home_goals", "away_goals"], "The match history")
        if not (played_in_season[["home_goals", "away_goals"]].to_numpy()
                == base[["home_goals", "away_goals"]].to_numpy()).all():
            raise ValueError(f"Scores in the saved predictions differ from the match history in {season_name(season)}")

        # Candidate A: static prior from the training seasons, online window from earlier dates
        frequency = AdaptiveEventFrequency(window, prior_matches).fit(train)
        static = _aligned(rows[rows["method"] == FREQUENCY], base, EVENT_NAMES, "The static frequency")
        for event in EVENT_NAMES:
            _check_close(static[event], frequency.prior_[event],
                         f"The static {event} frequency does not match the training seasons of {season_name(season)}")
        adaptive_frequency = frequency.predict(base, visible)

        # Candidate B: the commit-15 Poisson regression rates, rescaled
        level = AdaptiveScoringLevel(window, prior_matches).fit(train)
        goals = goals_predictions[goals_predictions["season"] == season]
        league = _aligned(goals[goals["method"] == LEAGUE], base, LAMBDA_COLUMNS, "The league-average rates")
        _check_close(league["lambda_home"], level.mu_home_, "Training mean home goals differ from commit 15")
        _check_close(league["lambda_away"], level.mu_away_, "Training mean away goals differ from commit 15")
        rates = _aligned(goals[goals["method"] == MODEL], base, LAMBDA_COLUMNS, "The Poisson regression rates")
        poisson_saved = _aligned(rows[rows["method"] == MODEL], base, EVENT_NAMES, "The Poisson regression")
        _check_close(poisson_events(rates)[EVENT_NAMES], poisson_saved[EVENT_NAMES],
                     "The commit-15 rates do not reproduce the commit-16 Poisson probabilities")
        factors = level.factors(base, visible)
        adjusted = level.adjust(rates, base, visible)
        adaptive_poisson = poisson_events(adjusted)

        frames += [_frame(season, ADAPTIVE_FREQUENCY, base, adaptive_frequency),
                   _frame(season, ADAPTIVE_POISSON, base, adaptive_poisson)]
        drift = pd.concat([base[FIXTURE_KEYS + ["home_goals", "away_goals"]], factors], axis=1)
        drift["season"] = season
        drift[LAMBDA_COLUMNS] = rates[LAMBDA_COLUMNS].to_numpy()
        drift["lambda_home_adjusted"] = adjusted["lambda_home"].to_numpy()
        drift["lambda_away_adjusted"] = adjusted["lambda_away"].to_numpy()
        drift_frames.append(drift[DRIFT_COLUMNS])
        folds.append({"season": season, "last_train_season": season - 1, "train_matches": len(train),
                      "evaluation_matches": len(base), "train_home_goals": level.mu_home_,
                      "train_away_goals": level.mu_away_, "window": window, "prior_matches": prior_matches,
                      "first_fixture_window_matches": int(factors["window_matches"].iloc[0]),
                      "window_from_season": _first_window_season(visible, base["date"].min(), window),
                      "mean_home_factor": float(factors["home_factor"].mean()),
                      "mean_away_factor": float(factors["away_factor"].mean())}
                     | {f"static_{e}": frequency.prior_[e] for e in EVENT_NAMES})

    predictions = pd.concat(frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"]).astype("datetime64[ns]")
    check_valid(predictions)
    check_identical_fixtures(predictions, METHODS)
    return DriftBacktest(predictions, pd.concat(drift_frames, ignore_index=True), pd.DataFrame(folds))


# ====================================================================== comparisons


def strongest_existing(predictions: pd.DataFrame, event: str) -> str:
    """The existing model (not the static frequency) with the lowest pooled log loss on ``event``."""
    pooled = pooled_metrics(predictions[predictions["method"].isin(MODELS)])
    return str(pooled[f"log_loss_{event}"].idxmin())


def rule_comparisons(predictions: pd.DataFrame, n_resamples: int = 10_000) -> pd.DataFrame:
    """The rule's family: each candidate vs the static frequency and vs the strongest existing model,
    on Over 2.5 and BTTS log loss (8 comparisons, Bonferroni-adjusted for 8)."""
    triples = []
    for candidate in CANDIDATES:
        for event in PRIMARY_EVENTS:
            triples.append((candidate, FREQUENCY, f"log_loss_{event}"))
            triples.append((candidate, strongest_existing(predictions, event), f"log_loss_{event}"))
    return compare(predictions, triples, RULE_COMPARISONS, n_resamples)


def primary_comparisons(predictions: pd.DataFrame, n_resamples: int = 10_000) -> pd.DataFrame:
    """Each candidate against every existing method on the two primary metrics (24, adjusted for 24)."""
    triples = [(c, m, f"log_loss_{e}") for c in CANDIDATES for e in PRIMARY_EVENTS for m in EXISTING_METHODS]
    return compare(predictions, triples, None, n_resamples)


def secondary_comparisons(predictions: pd.DataFrame, n_resamples: int = 10_000) -> pd.DataFrame:
    """Each candidate vs the static frequency and the Poisson regression on the other lines and the
    Brier scores (32 comparisons, adjusted for 32)."""
    triples = [(c, b, m) for c in CANDIDATES for b in (FREQUENCY, MODEL) for m in SECONDARY_METRICS]
    return compare(predictions, triples, None, n_resamples)


def readiness(predictions: pd.DataFrame, n_resamples: int = 10_000) -> pd.DataFrame:
    """Apply the pre-specified rule (module docstring) to each candidate and primary event."""
    family = rule_comparisons(predictions, n_resamples).set_index(["a", "b", "metric"])
    calibration = calibration_table(predictions).set_index(["method", "event"])
    sensitivity = season_sensitivity(predictions, [(c, FREQUENCY) for c in CANDIDATES]).set_index(["a", "metric"])
    rows = []
    for candidate in CANDIDATES:
        for event in PRIMARY_EVENTS:
            metric = f"log_loss_{event}"
            strongest = strongest_existing(predictions, event)
            vs_static = family.loc[(candidate, FREQUENCY, metric)]
            vs_strongest = family.loc[(candidate, strongest, metric)]
            season = sensitivity.loc[(candidate, metric)]
            cal = calibration.loc[(candidate, event)]
            slope = cal["calibration_slope"]
            checks = {
                "beats_static": bool(vs_static["adj_ci_high"] < 0),
                "competitive": bool(vs_strongest["difference"] <= 0),
                "five_of_seven": bool(season["seasons_better"] >= MIN_SEASONS_BETTER),
                "leave_one_out": bool(season["leave_one_out_max"] < 0),
                "calibrated": bool(SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]
                                   and abs(cal["difference"]) <= CALIBRATION_TOLERANCE
                                   and cal["bins_outside"] <= MAX_BINS_OUTSIDE),
            }
            rows.append({"candidate": candidate, "event": EVENTS[event],
                         "vs_static": float(vs_static["difference"]),
                         "vs_static_adj_ci_low": float(vs_static["adj_ci_low"]),
                         "vs_static_adj_ci_high": float(vs_static["adj_ci_high"]),
                         "strongest_existing": strongest, "vs_strongest": float(vs_strongest["difference"]),
                         "seasons_better": int(season["seasons_better"]),
                         "leave_one_out_max": float(season["leave_one_out_max"]),
                         "mean_minus_observed": float(cal["difference"]), "calibration_slope": float(slope),
                         "bins_outside": int(cal["bins_outside"])} | checks | {"ready": all(checks.values())})
    return pd.DataFrame(rows)


# ====================================================================== drift diagnostics


def season_goal_levels(matches: pd.DataFrame) -> pd.DataFrame:
    """Home, away and total goals per match and the event shares of every completed season."""
    history = completed_history(matches)
    targets = event_targets(history)
    table = pd.DataFrame({"home_goals": history["home_goals"], "away_goals": history["away_goals"],
                          "total_goals": history["home_goals"] + history["away_goals"]})
    table = pd.concat([table, targets], axis=1).groupby(history["season_start"]).mean()
    table["matches"] = history.groupby("season_start").size()
    return table


def rolling_levels(matches: pd.DataFrame, window: int = WINDOW_MATCHES, prior_matches: float = PRIOR_MATCHES,
                   seasons=None) -> pd.DataFrame:
    """For every match date: the window's home and away goals per match (strictly earlier matches)
    and, for comparison, the shrunk level the adaptive Poisson would use with that season's training means."""
    history = completed_history(matches)
    seasons = sorted(seasons) if seasons is not None else sorted(history["season_start"].unique())
    rows = []
    for season in seasons:
        train = history[history["season_start"] <= season - 1]
        if train.empty:
            continue
        level = AdaptiveScoringLevel(window, prior_matches).fit(train)
        dates = pd.DataFrame({"date": sorted(history.loc[history["season_start"] == season, "date"].unique())})
        factors = level.factors(dates, history[history["season_start"] <= season])
        rows.append(pd.concat([dates, factors.drop(columns=["window_first_date", "window_last_date"])], axis=1)
                    .assign(season=season))
    return pd.concat(rows, ignore_index=True)


def season_phase(predictions: pd.DataFrame, phases: int = 2) -> pd.Series:
    """Chronological phase of every row within its season and method: 0 = earliest fixtures.

    Fixtures are ranked by date (ties by club names) and cut into ``phases``
    equal-count parts.
    """
    order = predictions.sort_values(["season", "method", *FIXTURE_KEYS], kind="stable")
    rank = order.groupby(["season", "method"]).cumcount()
    size = order.groupby(["season", "method"])["date"].transform("size")
    return (rank * phases // size).reindex(predictions.index)


def phase_table(predictions: pd.DataFrame, phases: int = 2, metrics=PRIMARY_METRICS) -> pd.DataFrame:
    """Mean log loss per method in each season phase, pooled over seasons."""
    losses = event_losses(predictions)
    losses["phase"] = season_phase(predictions, phases).to_numpy()
    return losses.groupby(["phase", "method"], sort=False)[list(metrics)].mean().unstack("phase")


def season_bias(predictions: pd.DataFrame, methods=None, events=PRIMARY_EVENTS) -> pd.DataFrame:
    """Mean predicted minus observed share per season, method and event (percentage points as fractions)."""
    methods = methods or METHODS
    targets = event_targets(predictions)
    rows = []
    for (season, method), group in predictions[predictions["method"].isin(methods)].groupby(["season", "method"]):
        for event in events:
            rows.append({"season": season, "method": method, "event": event,
                         "bias": float(group[event].mean() - targets.loc[group.index, event].mean())})
    return pd.DataFrame(rows).pivot_table(index=["event", "season"], columns="method", values="bias")[
        [m for m in methods if m in set(predictions["method"])]]


def block_errors(drift: pd.DataFrame, blocks_per_season: int = 10) -> pd.DataFrame:
    """Chronological blocks of fixtures within each season, with the Poisson regression's mean error.

    ``error`` is observed minus predicted total goals of the (unadjusted)
    Poisson regression; ``adjustment`` is the mean change in expected total
    goals the adaptive level applied, which was known before each fixture.
    """
    table = drift.sort_values(["season", *FIXTURE_KEYS], kind="stable").copy()
    rank = table.groupby("season").cumcount()
    size = table.groupby("season")["date"].transform("size")
    table["block"] = rank * blocks_per_season // size
    table["error"] = table["home_goals"] + table["away_goals"] - table["lambda_home"] - table["lambda_away"]
    table["adjustment"] = (table["lambda_home_adjusted"] + table["lambda_away_adjusted"]
                           - table["lambda_home"] - table["lambda_away"])
    table["adjusted_error"] = table["error"] - table["adjustment"]
    out = table.groupby(["season", "block"]).agg(first_date=("date", "min"), matches=("date", "size"),
                                                  error=("error", "mean"), adjustment=("adjustment", "mean"),
                                                  adjusted_error=("adjusted_error", "mean"))
    return out.reset_index()


def error_persistence(blocks: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Do model errors persist, and does the adjustment lead or lag them?

    For lags k = 0..max_lag (blocks, consecutive across season boundaries):
    the correlation of a block's Poisson error with the error ``k`` blocks
    earlier (k >= 1: do past errors predict the next ones?), and with the
    adjustment applied ``k`` blocks earlier and ``k`` blocks later. Only the
    "earlier" columns describe information available at prediction time; the
    "later" column is a description of how the statistic follows the errors.
    """
    ordered = blocks.sort_values(["season", "block"]).reset_index(drop=True)
    error, adjustment = ordered["error"], ordered["adjustment"]
    rows = []
    for lag in range(max_lag + 1):
        rows.append({"lag_blocks": lag,
                     "error_vs_earlier_error": float(error.corr(error.shift(lag))) if lag else np.nan,
                     "error_vs_earlier_adjustment": float(error.corr(adjustment.shift(lag))),
                     "error_vs_later_adjustment": float(error.corr(adjustment.shift(-lag)))})
    return pd.DataFrame(rows)


def per_match_gain(predictions: pd.DataFrame, a: str, b: str, event: str) -> pd.DataFrame:
    """Per-fixture log loss of ``a`` minus ``b`` with the keys (negative = a better)."""
    keys = ["season", *FIXTURE_KEYS]
    targets = event_targets(predictions)
    table = predictions[keys + ["method", event]].assign(loss=binary_log_loss(predictions[event], targets[event]))
    pivot = table.pivot_table(index=keys, columns="method", values="loss")
    return (pivot[a] - pivot[b]).rename("difference").reset_index()


# ====================================================================== artifacts


FILES = ("predictions", "drift", "folds")


def save_drift_backtest(backtest: DriftBacktest, directory: Path = DRIFT_DIR) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.csv" for name in FILES}
    for name in FILES:
        getattr(backtest, name).to_csv(paths[name], index=False, date_format="%Y-%m-%d")
    paths["settings"] = directory / "settings.json"
    paths["settings"].write_text(json.dumps({
        "seasons": sorted(int(s) for s in backtest.predictions["season"].unique()),
        "methods": METHODS, "candidates": CANDIDATES, "window_matches": WINDOW_MATCHES,
        "prior_matches": PRIOR_MATCHES, "primary_metrics": PRIMARY_METRICS,
        "rule_comparisons": RULE_COMPARISONS, "min_seasons_better": MIN_SEASONS_BETTER,
        "slope_range": list(SLOPE_RANGE), "calibration_tolerance": CALIBRATION_TOLERANCE,
        "max_bins_outside": MAX_BINS_OUTSIDE,
    }, indent=2))
    return paths


def load_drift_backtest(directory: Path = DRIFT_DIR) -> DriftBacktest:
    paths = {name: directory / f"{name}.csv" for name in FILES}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m src.scoring_drift_backtest` first.")
    frames = {}
    for name, path in paths.items():
        frame = pd.read_csv(path)
        for column in ("date", "window_first_date", "window_last_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
        for column in ("home_team", "away_team", "method"):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        frames[name] = frame
    return DriftBacktest(**frames)


def main(argv: list[str] | None = None) -> None:
    from src.event_backtest import load_event_backtest
    from src.feature_engineering import load_features
    from src.goals_backtest import load_goals_backtest

    parser = argparse.ArgumentParser(description="Adaptive league scoring rates vs the existing goals forecasts.")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(BACKTEST_SEASONS))
    args = parser.parse_args(argv)

    backtest = run_scoring_drift_backtest(load_features(), load_goals_backtest().predictions,
                                          load_event_backtest().predictions, seasons=args.seasons)
    paths = save_drift_backtest(backtest)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pooled = pooled_metrics(backtest.predictions)
    print("Pooled log loss:")
    print(pooled[[f"log_loss_{e}" for e in EVENT_NAMES]].round(4).to_string())
    print("\nPre-specified decision rule:")
    print(readiness(backtest.predictions).round(4).to_string())
    for name, path in paths.items():
        print(f"saved {name}: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
