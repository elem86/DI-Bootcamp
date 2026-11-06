"""Walk-forward evaluation of expected-goals forecasts (Poisson regression vs baselines).

Usage (from the project folder, after ``python -m src.feature_engineering``)::

    python -m src.goals_backtest                 # seasons 2018-19 to 2024-25

Results are saved to ``outputs/goals/`` (git-ignored). Production models,
saved predictions and every earlier experiment folder are not touched.

How one fold works (evaluation season S)
----------------------------------------
The same scheme as the 1X2 backtest (``src.backtest``):

1. Seasons after S are removed before anything else happens.
2. **Tune (past only):** fit every candidate on seasons up to S-2 and score it
   on S-1 by mean Poisson deviance. The home and away regressions each choose
   their feature set and penalty; the team baseline chooses its window and
   prior. The league average has nothing to choose.
3. **Refit** the chosen candidates on all seasons up to S-1.
4. **Evaluate** every completed match of season S. All three methods predict
   exactly the same fixtures.

Season S is only used for scoring. Club histories for the team baseline and
all engineered features use strictly earlier dates, so a match never sees its
own result or any result from the same day or later.

Only seasons 2018-19 to 2024-25 are allowed: 2025-26 (validation season of the
production models) and 2026-27 (holdout) are excluded from this experiment and
from every choice made in it. Bookmaker odds are never a feature: the market
average Over/Under 2.5 odds already present in the raw Football-Data files are
used only as a comparison, on the fixtures where they exist.

Pre-specified decision rule
---------------------------
Written before the experiment was run. The primary metrics are the combined
(home + away) Poisson deviance and the log loss of Over/Under 2.5 and of BTTS.
The family of comparisons is the Poisson model against each of the two
baselines on the 11 headline metrics (deviance, plus log loss and Brier score
for Over 0.5 / 1.5 / 2.5 / 3.5 and BTTS): 22 comparisons, with
Bonferroni-adjusted bootstrap intervals and Holm-adjusted p-values.

The goals model is called **ready for the app** only if all of these hold:

1. **Skill:** against the league average, the adjusted interval of the pooled
   difference is below 0 for all three primary metrics.
2. **Stability:** it beats the league average on each primary metric in at
   least 5 of the 7 seasons.
3. **Worth its complexity:** its pooled score is at least as good as the
   simpler team attack/defence baseline on all three primary metrics.
4. **Calibration:** for Over 0.5 / 1.5 / 2.5 / 3.5 and BTTS, the pooled mean
   probability is within 2 percentage points of the observed frequency, and at
   most 1 of 10 equal-count probability bins has a 95% Wilson interval that
   excludes the bin's mean prediction.

Otherwise the result is reported as not ready, with the criteria that failed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.backtest import BACKTEST_SEASONS, check_season
from src.config import OUTPUTS_DIR, RANDOM_STATE, RAW_DATA_DIR
from src.elo_experiment import holm_adjust
from src.evaluate import OUTCOMES, _wilson_interval, per_match_brier, per_match_log_loss
from src.goal_stats import GOAL_LINES, line_key
from src.goals_model import (
    ALPHA_GRID,
    EVENTS,
    FIXTURE_KEYS,
    GOALS_FEATURE_SETS,
    PRIOR_MATCHES_GRID,
    STRENGTH_WINDOWS,
    LeagueAverageGoals,
    PoissonGoalsModel,
    SideChoice,
    TeamStrengthGoals,
    exact_score_probability,
    goal_probabilities,
    played,
    poisson_deviance,
    strength_inputs,
    tune_poisson_side,
    tune_team_strength,
)
from src.preprocessing import parse_dates, read_raw_season, season_files
from src.data_loader import season_start_year
from src.train import VALIDATION_SEASON, season_name

GOALS_DIR = OUTPUTS_DIR / "goals"
LEAGUE = LeagueAverageGoals.label
TEAM = TeamStrengthGoals.label
MODEL = PoissonGoalsModel.label
METHODS = [LEAGUE, TEAM, MODEL]
BASELINES = [LEAGUE, TEAM]
MARKET = "Market average (O/U 2.5)"

PRIMARY_METRICS = ["deviance", "log_loss_over_2_5", "log_loss_btts_yes"]
HEADLINE_METRICS = ["deviance"] + [f"log_loss_{e}" for e in EVENTS] + [f"brier_{e}" for e in EVENTS]
FAMILY_WISE_ALPHA = 0.05
MIN_SEASONS_BETTER = 5
CALIBRATION_TOLERANCE = 0.02
CALIBRATION_BINS = 10
MAX_BINS_OUTSIDE = 1
_EPS = 1e-15

ID_COLUMNS = ["season", "method", *FIXTURE_KEYS, "home_goals", "away_goals", "promoted_club"]


def metric_label(metric: str) -> str:
    """Readable name of a loss column, e.g. 'Log loss, Over 2.5'."""
    if metric == "deviance":
        return "Poisson deviance (home + away)"
    for prefix, name in (("log_loss_", "Log loss"), ("brier_", "Brier")):
        if metric.startswith(prefix):
            event = metric[len(prefix):]
            return f"{name}, {EVENTS.get(event, event)}"
    return metric


# ====================================================================== folds


def check_goals_season(season: int, features: pd.DataFrame) -> None:
    """Refuse 2025-26 and later, and seasons without two earlier seasons (tune on S-2 -> S-1)."""
    if season >= VALIDATION_SEASON:
        raise ValueError(f"{season_name(season)} is not allowed: {season_name(VALIDATION_SEASON)} and later "
                         "are the validation and holdout seasons and stay out of this experiment.")
    check_season(season, features)


@dataclass
class GoalsFold:
    season: int
    model: PoissonGoalsModel
    team: TeamStrengthGoals
    league: LeagueAverageGoals
    tuning: pd.DataFrame  # every candidate's deviance on S-1
    train_matches: int
    predictions: pd.DataFrame  # one row per method and fixture of season S


def run_goals_fold(features: pd.DataFrame, season: int, alpha_grid=ALPHA_GRID, feature_sets=None,
                   windows=STRENGTH_WINDOWS, priors=PRIOR_MATCHES_GRID) -> GoalsFold:
    """Tune on S-2 -> S-1, refit on everything before S, predict season S with all three methods."""
    check_goals_season(season, features)
    feature_sets = feature_sets or GOALS_FEATURE_SETS
    visible = played(features[features["season_start"] <= season])  # nothing after S exists here
    inputs = strength_inputs(visible, windows)
    inner_train = visible[visible["season_start"] <= season - 2]
    inner_validation = visible[visible["season_start"] == season - 1]
    train = visible[visible["season_start"] <= season - 1]
    evaluation = visible[visible["season_start"] == season]
    if evaluation.empty or inner_validation.empty:
        raise ValueError(f"{season_name(season)}: no matches to evaluate or to tune on")

    tables, choices = [], {}
    for side in ("home", "away"):
        table = tune_poisson_side(inner_train, inner_validation, side, feature_sets, alpha_grid)
        best = table.iloc[0]
        choices[side] = SideChoice(str(best["feature_set"]), float(best["alpha"]))
        tables.append(table.assign(method=MODEL))
    team_table = tune_team_strength(inner_train, inner_validation, inputs, windows, priors)
    best = team_table.iloc[0]
    tables.append(team_table.assign(method=TEAM, side="both"))

    model = PoissonGoalsModel(choices["home"], choices["away"], dict(feature_sets)).fit(train)
    team = TeamStrengthGoals(int(best["window"]), float(best["prior_matches"])).fit(train)
    league = LeagueAverageGoals().fit(train)

    base = evaluation[FIXTURE_KEYS + ["home_goals", "away_goals"]].copy()
    base["home_goals"] = base["home_goals"].astype("int64")
    base["away_goals"] = base["away_goals"].astype("int64")
    promoted = evaluation[["home_promoted", "away_promoted"]].astype("boolean").fillna(False).astype(bool)
    base["promoted_club"] = promoted.any(axis=1)
    frames = []
    for method, fitted in ((LEAGUE, league), (TEAM, team), (MODEL, model)):
        rates = fitted.predict(evaluation, inputs)
        frame = base.copy()
        frame.insert(0, "method", method)
        frame.insert(0, "season", season)
        frame[["lambda_home", "lambda_away"]] = rates[["lambda_home", "lambda_away"]].to_numpy()
        frames.append(frame)
    tuning = pd.concat(tables, ignore_index=True).assign(season=season)
    return GoalsFold(season, model, team, league, tuning, len(train), pd.concat(frames, ignore_index=True))


# ====================================================================== backtest


@dataclass
class GoalsBacktest:
    """``predictions``: one row per (season, method, fixture) with the rates and every derived probability."""

    predictions: pd.DataFrame
    folds: pd.DataFrame
    tuning: pd.DataFrame


def with_probabilities(rates: pd.DataFrame) -> pd.DataFrame:
    """Add Over/Under, BTTS, 1X2, most likely score and truncated mass to a table of rates."""
    probabilities = goal_probabilities(rates["lambda_home"], rates["lambda_away"])
    probabilities.index = rates.index
    return pd.concat([rates, probabilities.drop(columns=["lambda_home", "lambda_away"])], axis=1)


def check_identical_fixtures(predictions: pd.DataFrame, methods=None) -> None:
    """Every method must cover the same fixtures (with the same scores) in every season."""
    methods = list(methods or predictions["method"].unique())
    keys = ["season", *FIXTURE_KEYS, "home_goals", "away_goals"]
    reference = None
    for method in methods:
        rows = predictions[predictions["method"] == method]
        if rows.empty:
            raise ValueError(f"{method} has no predictions")
        table = rows[keys].sort_values(keys[:4], kind="stable").reset_index(drop=True)
        if table.duplicated(keys[:4]).any():
            raise ValueError(f"{method} predicts some fixtures twice")
        if reference is None:
            reference = table
        elif not table.astype(str).equals(reference.astype(str)):
            raise ValueError(f"{method} was evaluated on different fixtures than {methods[0]}")


def run_goals_backtest(features: pd.DataFrame, seasons=BACKTEST_SEASONS, progress=None, **grids) -> GoalsBacktest:
    """Walk-forward evaluation of the three methods on every season (see the module docstring)."""
    seasons = sorted(seasons)
    for season in seasons:
        check_goals_season(season, features)
    frames, folds, tuning = [], [], []
    for season in seasons:
        fold = run_goals_fold(features, season, **grids)
        frames.append(fold.predictions)
        tuning.append(fold.tuning)
        folds.append({"season": season, "first_train_season": int(features["season_start"].min()),
                      "last_train_season": season - 1, "tuning_season": season - 1,
                      "train_matches": fold.train_matches,
                      "evaluation_matches": int((fold.predictions["method"] == MODEL).sum())}
                     | fold.model.params | {f"team_{k}": v for k, v in fold.team.params.items()}
                     | {"league_mu_home": fold.league.mu_home_, "league_mu_away": fold.league.mu_away_})
        if progress:
            progress(f"{season_name(season)}: Poisson {fold.model.params}; team {fold.team.params}")
    rates = pd.concat(frames, ignore_index=True)
    predictions = with_probabilities(rates)
    check_identical_fixtures(predictions, METHODS)
    return GoalsBacktest(predictions, pd.DataFrame(folds), pd.concat(tuning, ignore_index=True))


# ====================================================================== per-match scores


def event_outcomes(predictions: pd.DataFrame) -> pd.DataFrame:
    """What happened: 1/0 for each goal line and BTTS, plus the 1X2 result."""
    home, away = predictions["home_goals"].to_numpy(), predictions["away_goals"].to_numpy()
    total = home + away
    table = pd.DataFrame({line_key(line): (total > line).astype(float) for line in GOAL_LINES},
                         index=predictions.index)
    table["btts_yes"] = ((home > 0) & (away > 0)).astype(float)
    table["result"] = np.select([home > away, home < away], ["H", "A"], default="D")
    table["total_goals"] = total
    return table


def binary_log_loss(probability, outcome) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype="float64"), _EPS, 1 - _EPS)
    y = np.asarray(outcome, dtype="float64")
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def binary_brier(probability, outcome) -> np.ndarray:
    return (np.asarray(probability, dtype="float64") - np.asarray(outcome, dtype="float64")) ** 2


def match_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-match scores of every prediction row (lower is better for every loss)."""
    outcomes = event_outcomes(predictions)
    table = predictions[ID_COLUMNS].copy()
    table["deviance_home"] = poisson_deviance(predictions["home_goals"], predictions["lambda_home"])
    table["deviance_away"] = poisson_deviance(predictions["away_goals"], predictions["lambda_away"])
    table["deviance"] = table["deviance_home"] + table["deviance_away"]
    table["abs_error_home"] = (predictions["home_goals"] - predictions["lambda_home"]).abs()
    table["abs_error_away"] = (predictions["away_goals"] - predictions["lambda_away"]).abs()
    for event in EVENTS:
        table[f"log_loss_{event}"] = binary_log_loss(predictions[event], outcomes[event])
        table[f"brier_{event}"] = binary_brier(predictions[event], outcomes[event])
    probs = predictions[OUTCOMES]
    table["log_loss_1x2"] = per_match_log_loss(outcomes["result"], probs).to_numpy()
    table["brier_1x2"] = per_match_brier(outcomes["result"], probs).to_numpy()
    score_p = exact_score_probability(predictions["lambda_home"], predictions["lambda_away"],
                                      predictions["home_goals"], predictions["away_goals"])
    table["log_loss_exact_score"] = -np.log(np.clip(score_p, _EPS, 1.0))
    return table


LOSS_COLUMNS = (["deviance_home", "deviance_away", "deviance", "abs_error_home", "abs_error_away"]
                + [f"{kind}_{e}" for e in EVENTS for kind in ("log_loss", "brier")]
                + ["log_loss_1x2", "brier_1x2", "log_loss_exact_score"])


def _summarise(predictions: pd.DataFrame, losses: pd.DataFrame, by) -> pd.DataFrame:
    frame = losses[LOSS_COLUMNS].copy()
    frame["matches"] = 1
    for column in ("lambda_home", "lambda_away", "lambda_total"):
        frame[f"mean_{column}"] = predictions[column]
    frame["mean_home_goals"] = predictions["home_goals"]
    frame["mean_away_goals"] = predictions["away_goals"]
    frame["mean_total_goals"] = predictions["home_goals"] + predictions["away_goals"]
    keys = [predictions[k] for k in by]
    table = frame.groupby(keys, sort=False).agg({c: ("sum" if c == "matches" else "mean") for c in frame.columns})
    table = table.rename(columns={"abs_error_home": "mae_home", "abs_error_away": "mae_away"})
    return table


def season_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean of every per-match score, and mean predicted vs actual goals, per (season, method)."""
    table = _summarise(predictions, match_losses(predictions), ["season", "method"])
    order = pd.MultiIndex.from_product([sorted(predictions["season"].unique()), _method_order(predictions)],
                                       names=["season", "method"])
    return table.reindex(order)


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """The same scores over all seasons pooled (every match counts once)."""
    table = _summarise(predictions, match_losses(predictions), ["method"])
    table["seasons"] = predictions.groupby("method")["season"].nunique()
    return table.reindex(_method_order(predictions))


def _method_order(predictions: pd.DataFrame) -> list[str]:
    names = list(dict.fromkeys(predictions["method"]))
    return [m for m in METHODS if m in names] + [m for m in names if m not in METHODS]


# ====================================================================== uncertainty


def _paired(losses: pd.DataFrame, a: str, b: str, metric: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-match ``metric(a) - metric(b)`` on identical fixtures, with the fixture keys."""
    keys = ["season", *FIXTURE_KEYS]
    rows_a = losses[losses["method"] == a].sort_values(keys, kind="stable").reset_index(drop=True)
    rows_b = losses[losses["method"] == b].sort_values(keys, kind="stable").reset_index(drop=True)
    if rows_a.empty or not rows_a[keys].astype(str).equals(rows_b[keys].astype(str)):
        raise ValueError(f"{a} and {b} do not cover the same fixtures")
    return (rows_a[metric] - rows_b[metric]).to_numpy(dtype="float64"), rows_a[keys]


def week_clusters(fixtures: pd.DataFrame) -> np.ndarray:
    """A cluster id per fixture: its season and Wednesday-to-Tuesday week (roughly one matchday)."""
    weeks = pd.to_datetime(fixtures["date"]).dt.to_period("W-TUE").astype(str)
    return pd.factorize(fixtures["season"].astype(str) + "|" + weeks)[0]


def bootstrap_means(differences: np.ndarray, n_resamples: int = 10_000, clusters=None,
                    random_state: int = RANDOM_STATE) -> np.ndarray:
    """Bootstrap distribution of the mean difference (matches, or whole clusters, resampled)."""
    rng = np.random.default_rng(random_state)
    if clusters is None:
        values, counts = differences, np.ones(len(differences))
    else:
        clusters = np.asarray(clusters)
        _, ids = np.unique(clusters, return_inverse=True)
        values = np.bincount(ids, weights=differences)
        counts = np.bincount(ids).astype(float)
    n = len(values)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 500):
        stop = min(start + 500, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return means


def paired_comparisons(predictions: pd.DataFrame, metrics=HEADLINE_METRICS, model: str = MODEL,
                       baselines=BASELINES, n_resamples: int = 10_000, cluster: str | None = None,
                       n_comparisons: int | None = None) -> pd.DataFrame:
    """Pooled paired bootstrap of ``model - baseline`` for every metric (negative = model better).

    Gives a 95% interval and a Bonferroni-adjusted one for the whole family
    (``n_comparisons``, default: every metric x baseline here), a two-sided
    bootstrap p-value and its Holm adjustment. ``cluster="week"`` resamples
    whole weeks instead of single matches, which allows for matches on the
    same weekend being related.
    """
    losses = match_losses(predictions)
    n_comparisons = n_comparisons or len(metrics) * len(baselines)
    adjusted = 1 - FAMILY_WISE_ALPHA / n_comparisons
    rows = []
    for baseline in baselines:
        for metric in metrics:
            diff, keys = _paired(losses, model, baseline, metric)
            clusters = week_clusters(keys) if cluster == "week" else None
            means = bootstrap_means(diff, n_resamples, clusters)
            low, high = np.quantile(means, [0.025, 0.975])
            adj_low, adj_high = np.quantile(means, [(1 - adjusted) / 2, 1 - (1 - adjusted) / 2])
            tail = min((means <= 0).mean(), (means >= 0).mean())
            rows.append({"model": model, "baseline": baseline, "metric": metric, "label": metric_label(metric),
                         "primary": metric in PRIMARY_METRICS, "matches": len(diff),
                         "difference": float(diff.mean()), "relative": float(diff.mean() / _mean(losses, baseline, metric)),
                         "ci_low": low, "ci_high": high, "adjusted_confidence": adjusted,
                         "adj_ci_low": adj_low, "adj_ci_high": adj_high,
                         "p_value": float(min(1.0, max(2 * tail, 1 / n_resamples)))})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_value"])
    table["adjusted_interval_excludes_zero"] = (table["adj_ci_high"] < 0) | (table["adj_ci_low"] > 0)
    return table


def _mean(losses: pd.DataFrame, method: str, metric: str) -> float:
    return float(losses.loc[losses["method"] == method, metric].mean())


def season_sensitivity(predictions: pd.DataFrame, metrics=PRIMARY_METRICS, model: str = MODEL,
                       baselines=BASELINES) -> pd.DataFrame:
    """Season-level view of ``model - baseline``: seasons won, sign test and leave-one-season-out.

    ``leave_one_out_min``/``max`` is the range of the pooled difference after
    dropping each season in turn; a conclusion that flips sign there rests on a
    single season.
    """
    losses = match_losses(predictions)
    rows = []
    for baseline in baselines:
        for metric in metrics:
            diff, keys = _paired(losses, model, baseline, metric)
            seasons = keys["season"].to_numpy()
            per_season = pd.Series(diff).groupby(seasons).mean()
            better = int((per_season < 0).sum())
            loo = [diff[seasons != s].mean() for s in per_season.index]
            rows.append({"baseline": baseline, "metric": metric, "label": metric_label(metric),
                         "seasons": len(per_season), "seasons_better": better,
                         "sign_test_p": binomtest(better, len(per_season), 0.5).pvalue,
                         "pooled_difference": float(diff.mean()),
                         "mean_season_difference": float(per_season.mean()),
                         "best_season_difference": float(per_season.min()),
                         "worst_season_difference": float(per_season.max()),
                         "leave_one_out_min": float(min(loo)), "leave_one_out_max": float(max(loo))})
    return pd.DataFrame(rows)


def season_differences(predictions: pd.DataFrame, metric: str, model: str = MODEL, baseline: str = LEAGUE) -> pd.Series:
    """Mean ``model - baseline`` of one metric in every season."""
    diff, keys = _paired(match_losses(predictions), model, baseline, metric)
    return pd.Series(diff).groupby(keys["season"].to_numpy()).mean().rename(metric)


# ====================================================================== calibration and bias


def reliability_table(probability, outcome, n_bins: int = CALIBRATION_BINS, strategy: str = "quantile") -> pd.DataFrame:
    """Mean predicted probability vs observed frequency of a yes/no event, in bins.

    ``quantile`` bins hold (almost) equal numbers of matches; ``uniform`` bins
    are 0-0.1, 0.1-0.2, ... with empty ones dropped. ``ci_low``/``ci_high`` is
    a 95% Wilson interval for the observed frequency; ``outside`` marks bins
    whose interval does not contain the mean prediction.
    """
    p = pd.Series(np.asarray(probability, dtype="float64"))
    y = pd.Series(np.asarray(outcome, dtype="float64"))
    if strategy == "quantile":
        bins = pd.qcut(p.rank(method="first"), q=min(n_bins, len(p)), labels=False)
    elif strategy == "uniform":
        bins = pd.Series(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]))
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    table = (pd.DataFrame({"bin": bins, "p": p, "hit": y}).groupby("bin")
             .agg(matches=("hit", "size"), mean_predicted=("p", "mean"), observed=("hit", "mean"),
                  lowest=("p", "min"), highest=("p", "max")).reset_index(drop=True))
    table["ci_low"], table["ci_high"] = _wilson_interval(table["observed"] * table["matches"], table["matches"])
    table["outside"] = (table["mean_predicted"] < table["ci_low"]) | (table["mean_predicted"] > table["ci_high"])
    return table


def calibration_slope(probability, outcome) -> tuple[float, float]:
    """Intercept and slope of a logistic regression of the outcome on logit(p).

    A well-calibrated forecast has slope 1 and intercept 0; a slope below 1
    means the probabilities are too extreme, above 1 too timid.
    """
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fitted = LogisticRegression(C=np.inf, max_iter=1000).fit(logit, np.asarray(outcome).astype(int))
    return float(fitted.intercept_[0]), float(fitted.coef_[0, 0])


def calibration_table(predictions: pd.DataFrame, n_bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Per method and event: calibration in the large, ECE, bins outside their interval, slope."""
    outcomes = event_outcomes(predictions)
    rows = []
    for method in _method_order(predictions):
        mask = (predictions["method"] == method).to_numpy()
        for event, label in EVENTS.items():
            p, y = predictions.loc[mask, event], outcomes.loc[mask, event]
            table = reliability_table(p, y, n_bins)
            gap = (table["observed"] - table["mean_predicted"]).abs()
            # a near-constant forecast (the league average) has no slope worth estimating
            intercept, slope = calibration_slope(p, y) if p.round(6).nunique() >= 20 else (np.nan, np.nan)
            rows.append({"method": method, "event": event, "label": label, "matches": int(mask.sum()),
                         "mean_predicted": float(p.mean()), "observed": float(y.mean()),
                         "difference": float(p.mean() - y.mean()),
                         "ece": float((gap * table["matches"]).sum() / table["matches"].sum()),
                         "bins_outside": int(table["outside"].sum()), "bins": len(table),
                         "calibration_intercept": intercept, "calibration_slope": slope})
    return pd.DataFrame(rows)


def bias_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Expected vs observed frequency of high-scoring games, draws, 0-0 and BTTS, per method.

    ``expected`` is the mean model probability over the evaluated fixtures, so
    a positive ``difference`` means the method predicts the event too often.
    """
    outcomes = event_outcomes(predictions)
    total = outcomes["total_goals"]
    lam_h, lam_a = predictions["lambda_home"], predictions["lambda_away"]
    # P(total >= 5) and P(0-0) from the same independent-Poisson distribution
    from scipy.stats import poisson

    expected = {
        "4+ goals (Over 3.5)": (predictions["over_3_5"], total >= 4),
        "5+ goals (Over 4.5)": (pd.Series(poisson.sf(4, lam_h + lam_a), index=predictions.index), total >= 5),
        "0-1 goals (Under 1.5)": (predictions["under_1_5"], total <= 1),
        "Draw": (predictions["D"], outcomes["result"] == "D"),
        "0-0": (pd.Series(np.exp(-(lam_h + lam_a)), index=predictions.index), total == 0),
        "BTTS Yes": (predictions["btts_yes"], outcomes["btts_yes"] == 1),
        "Home win": (predictions["H"], outcomes["result"] == "H"),
        "Away win": (predictions["A"], outcomes["result"] == "A"),
    }
    rows = []
    for method in _method_order(predictions):
        mask = predictions["method"] == method
        for name, (p, hit) in expected.items():
            n, hits = int(mask.sum()), int(hit[mask].sum())
            low, high = _wilson_interval(hits, n)
            rows.append({"method": method, "event": name, "matches": n, "expected": float(p[mask].mean()),
                         "observed": hits / n, "difference": float(p[mask].mean() - hits / n),
                         "ci_low": float(low), "ci_high": float(high)})
    table = pd.DataFrame(rows)
    table["outside_interval"] = (table["expected"] < table["ci_low"]) | (table["expected"] > table["ci_high"])
    return table


def score_frequency_table(predictions: pd.DataFrame, method: str = MODEL, max_goals: int = 4) -> pd.DataFrame:
    """Expected vs observed count of every exact score up to ``max_goals`` per side (rest pooled)."""
    from scipy.stats import poisson

    rows = predictions[predictions["method"] == method]
    lam_h, lam_a = rows["lambda_home"].to_numpy(), rows["lambda_away"].to_numpy()
    records = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            expected = (poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)).sum()
            observed = int(((rows["home_goals"] == h) & (rows["away_goals"] == a)).sum())
            records.append({"home_goals": h, "away_goals": a, "score": f"{h}-{a}",
                            "expected": expected, "observed": observed})
    table = pd.DataFrame(records)
    table.loc[len(table)] = {"home_goals": -1, "away_goals": -1, "score": "other",
                             "expected": len(rows) - table["expected"].sum(),
                             "observed": len(rows) - table["observed"].sum()}
    table["difference"] = table["expected"] - table["observed"]
    return table


def dispersion_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Do the goal counts vary more than the Poisson assumption allows? Are home and away goals related?

    Under the model a side's goals have variance ``mean(lambda) + var(lambda)``
    across fixtures (the law of total variance). ``variance_ratio`` > 1 means
    more spread than the model implies (over-dispersion).
    ``residual_correlation`` is the correlation of ``goals - lambda`` between
    the home and away side; independence implies about 0.
    """
    rows = []
    for method in _method_order(predictions):
        p = predictions[predictions["method"] == method]
        record = {"method": method, "matches": len(p)}
        for side in ("home", "away", "total"):
            goals = p["home_goals"] + p["away_goals"] if side == "total" else p[f"{side}_goals"]
            lam = p["lambda_total"] if side == "total" else p[f"lambda_{side}"]
            implied = lam.mean() + lam.var(ddof=0)
            record |= {f"{side}_mean_goals": goals.mean(), f"{side}_variance": goals.var(ddof=0),
                       f"{side}_implied_variance": implied, f"{side}_variance_ratio": goals.var(ddof=0) / implied}
        record["goals_correlation"] = float(np.corrcoef(p["home_goals"], p["away_goals"])[0, 1])
        record["residual_correlation"] = float(np.corrcoef(p["home_goals"] - p["lambda_home"],
                                                           p["away_goals"] - p["lambda_away"])[0, 1])
        rows.append(record)
    return pd.DataFrame(rows).set_index("method")


def goals_by_band(predictions: pd.DataFrame, method: str = MODEL, bands: int = 10) -> pd.DataFrame:
    """Mean predicted vs actual total goals in equal-count bands of the predicted total."""
    rows = predictions[predictions["method"] == method]
    band = pd.qcut(rows["lambda_total"].rank(method="first"), bands, labels=False)
    total = rows["home_goals"] + rows["away_goals"]
    table = pd.DataFrame({"band": band, "predicted": rows["lambda_total"], "actual": total})
    out = table.groupby("band").agg(matches=("actual", "size"), predicted=("predicted", "mean"),
                                    actual=("actual", "mean"), actual_sd=("actual", "std"))
    out["se"] = out["actual_sd"] / np.sqrt(out["matches"])
    return out


# ====================================================================== bookmaker Over/Under 2.5

MARKET_SOURCES = {"over": ["Avg>2.5", "BbAv>2.5"], "under": ["Avg<2.5", "BbAv<2.5"]}


def market_over_under(raw_dir: Path = RAW_DATA_DIR, seasons=None) -> pd.DataFrame:
    """Pre-match market-average Over/Under 2.5 odds from the raw Football-Data files already on disk.

    ``Avg`` (2019-20 onwards) and ``BbAv`` (BetBrain average, earlier seasons)
    are the average decimal odds across bookmakers. ``market_over_2_5`` is the
    implied probability with the margin removed proportionally. Nothing is
    downloaded: seasons without a raw file or without these columns are absent.
    """
    frames = []
    for path in season_files(raw_dir):
        start = season_start_year(path.stem.split("_")[1])
        if seasons is not None and start not in seasons:
            continue
        raw = read_raw_season(path)
        sides = {}
        for side, names in MARKET_SOURCES.items():
            present = [n for n in names if n in raw.columns]
            values = pd.Series(np.nan, index=raw.index)
            source = pd.Series(pd.NA, index=raw.index, dtype="string")
            for name in present:
                fill = values.isna() & raw[name].notna()
                values[fill] = pd.to_numeric(raw.loc[fill, name])
                source[fill] = name.split(">")[0].split("<")[0]
            sides[side], sides[f"{side}_source"] = values, source
        frames.append(pd.DataFrame({"season_start": start, "date": parse_dates(raw["Date"]),
                                    "home_team": raw["HomeTeam"].astype("string"),
                                    "away_team": raw["AwayTeam"].astype("string"),
                                    "over_odds": sides["over"], "under_odds": sides["under"],
                                    "source": sides["over_source"]}))
    if not frames:
        return pd.DataFrame(columns=["season_start", *FIXTURE_KEYS, "over_odds", "under_odds", "source",
                                     "overround", "market_over_2_5"])
    odds = pd.concat(frames, ignore_index=True)
    valid = (odds["over_odds"] > 1) & (odds["under_odds"] > 1)
    odds.loc[~valid, ["over_odds", "under_odds"]] = np.nan
    inverse_over, inverse_under = 1 / odds["over_odds"], 1 / odds["under_odds"]
    odds["overround"] = inverse_over + inverse_under - 1
    odds["market_over_2_5"] = inverse_over / (inverse_over + inverse_under)
    return odds


def market_coverage(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Per season: evaluated fixtures, fixtures with market Over/Under 2.5 odds, source and margin."""
    fixtures = predictions[predictions["method"] == MODEL][["season", *FIXTURE_KEYS]]
    merged = fixtures.merge(odds.dropna(subset=["market_over_2_5"]), on=FIXTURE_KEYS, how="left",
                            validate="one_to_one")
    table = merged.groupby("season").agg(fixtures=("home_team", "size"), with_odds=("market_over_2_5", "count"),
                                         source=("source", lambda s: ", ".join(sorted(s.dropna().unique()))),
                                         mean_overround=("overround", "mean"))
    table["coverage"] = table["with_odds"] / table["fixtures"]
    return table


def market_predictions(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Over 2.5 forecasts of every method plus the market, on the fixtures that have odds (long format)."""
    usable = odds.dropna(subset=["market_over_2_5"])[[*FIXTURE_KEYS, "market_over_2_5"]]
    rows = predictions[[*ID_COLUMNS, "over_2_5"]].merge(usable, on=FIXTURE_KEYS, how="inner",
                                                        validate="many_to_one")
    market = rows[rows["method"] == MODEL].copy()
    market["method"] = MARKET
    market["over_2_5"] = market["market_over_2_5"]
    long = pd.concat([rows, market], ignore_index=True).drop(columns="market_over_2_5")
    long["outcome"] = (long["home_goals"] + long["away_goals"] > 2.5).astype(float)
    long["log_loss_over_2_5"] = binary_log_loss(long["over_2_5"], long["outcome"])
    long["brier_over_2_5"] = binary_brier(long["over_2_5"], long["outcome"])
    return long


def market_comparison(predictions: pd.DataFrame, odds: pd.DataFrame, n_resamples: int = 10_000) -> pd.DataFrame:
    """Each method vs the market on Over/Under 2.5 (identical fixtures): scores and paired intervals."""
    long = market_predictions(predictions, odds)
    keys = ["season", *FIXTURE_KEYS]
    market = long[long["method"] == MARKET].sort_values(keys, kind="stable").reset_index(drop=True)
    rows = []
    for method in [*METHODS, MARKET]:
        mine = long[long["method"] == method].sort_values(keys, kind="stable").reset_index(drop=True)
        if not mine[keys].astype(str).equals(market[keys].astype(str)):
            raise ValueError(f"{method} and the market cover different fixtures")
        record = {"method": method, "matches": len(mine),
                  "log_loss": mine["log_loss_over_2_5"].mean(), "brier": mine["brier_over_2_5"].mean(),
                  "mean_predicted": mine["over_2_5"].mean(), "observed": mine["outcome"].mean()}
        if method != MARKET:
            diff = (mine["log_loss_over_2_5"] - market["log_loss_over_2_5"]).to_numpy()
            low, high = np.quantile(bootstrap_means(diff, n_resamples), [0.025, 0.975])
            record |= {"log_loss_vs_market": diff.mean(), "ci_low": low, "ci_high": high}
        rows.append(record)
    return pd.DataFrame(rows).set_index("method")


# ====================================================================== decision rule


def readiness(predictions: pd.DataFrame, n_resamples: int = 10_000) -> pd.DataFrame:
    """Apply the pre-specified decision rule (module docstring). One row per criterion."""
    comparisons = paired_comparisons(predictions, n_resamples=n_resamples)
    seasons = season_sensitivity(predictions)
    calibration = calibration_table(predictions)
    primary = comparisons[comparisons["primary"]]
    vs_league = primary[primary["baseline"] == LEAGUE].set_index("metric")
    vs_team = primary[primary["baseline"] == TEAM].set_index("metric")
    stable = seasons[seasons["baseline"] == LEAGUE].set_index("metric")
    mine = calibration[calibration["method"] == MODEL].set_index("event")

    def describe(values, fmt):
        return "; ".join(f"{metric_label(k) if k in PRIMARY_METRICS else EVENTS.get(k, k)}: {fmt(v)}"
                         for k, v in values.items())

    rows = [
        {"criterion": "1. Skill vs league average (adjusted interval below 0)",
         "passed": bool((vs_league["adj_ci_high"] < 0).all()),
         "evidence": describe(vs_league["adj_ci_high"], lambda v: f"upper bound {v:+.4f}")},
        {"criterion": f"2. Better than league average in >= {MIN_SEASONS_BETTER} of 7 seasons",
         "passed": bool((stable["seasons_better"] >= MIN_SEASONS_BETTER).all()),
         "evidence": describe(stable["seasons_better"], lambda v: f"{int(v)} of 7")},
        {"criterion": "3. At least as good as the team attack/defence baseline",
         "passed": bool((vs_team["difference"] <= 0).all()),
         "evidence": describe(vs_team["difference"], lambda v: f"{v:+.4f}")},
        {"criterion": f"4a. Mean probability within {CALIBRATION_TOLERANCE:.0%} of observed",
         "passed": bool((mine["difference"].abs() <= CALIBRATION_TOLERANCE).all()),
         "evidence": describe(mine["difference"], lambda v: f"{v:+.1%}")},
        {"criterion": f"4b. At most {MAX_BINS_OUTSIDE} of {CALIBRATION_BINS} bins outside their interval",
         "passed": bool((mine["bins_outside"] <= MAX_BINS_OUTSIDE).all()),
         "evidence": describe(mine["bins_outside"], lambda v: f"{int(v)}")},
    ]
    table = pd.DataFrame(rows)
    table.loc[len(table)] = {"criterion": "Ready for the app (all criteria)", "passed": bool(table["passed"].all()),
                             "evidence": ""}
    return table


# ====================================================================== artifacts


def save_goals_backtest(backtest: GoalsBacktest, directory: Path = GOALS_DIR, odds: pd.DataFrame | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.csv" for name in ("predictions", "folds", "tuning")}
    backtest.predictions.to_csv(paths["predictions"], index=False, date_format="%Y-%m-%d")
    backtest.folds.to_csv(paths["folds"], index=False)
    backtest.tuning.to_csv(paths["tuning"], index=False)
    if odds is not None:
        paths["market_odds"] = directory / "market_odds.csv"
        odds.to_csv(paths["market_odds"], index=False, date_format="%Y-%m-%d")
    paths["settings"] = directory / "settings.json"
    paths["settings"].write_text(json.dumps({
        "seasons": sorted(int(s) for s in backtest.predictions["season"].unique()),
        "alpha_grid": list(ALPHA_GRID), "feature_sets": {k: len(v) for k, v in GOALS_FEATURE_SETS.items()},
        "strength_windows": list(STRENGTH_WINDOWS), "prior_matches_grid": list(PRIOR_MATCHES_GRID),
        "primary_metrics": PRIMARY_METRICS, "headline_metrics": HEADLINE_METRICS,
    }, indent=2))
    return paths


def load_goals_backtest(directory: Path = GOALS_DIR) -> GoalsBacktest:
    paths = {name: directory / f"{name}.csv" for name in ("predictions", "folds", "tuning")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m src.goals_backtest` first.")
    predictions = pd.read_csv(paths["predictions"], parse_dates=["date"]).astype({"date": "datetime64[ns]"})
    for column in ("home_team", "away_team", "method"):
        predictions[column] = predictions[column].astype("string")
    return GoalsBacktest(predictions, pd.read_csv(paths["folds"]), pd.read_csv(paths["tuning"]))


def main(argv: list[str] | None = None) -> None:
    from src.feature_engineering import load_features

    parser = argparse.ArgumentParser(description="Walk-forward evaluation of the Poisson goals model.")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(BACKTEST_SEASONS))
    args = parser.parse_args(argv)

    backtest = run_goals_backtest(load_features(), seasons=args.seasons, progress=print)
    odds = market_over_under(seasons=set(args.seasons))
    paths = save_goals_backtest(backtest, odds=odds)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pooled = pooled_metrics(backtest.predictions)
    print("\nPooled:")
    print(pooled[["matches", "deviance", "mae_home", "mae_away", "log_loss_over_2_5", "log_loss_btts_yes",
                  "mean_lambda_total", "mean_total_goals"]].round(4).to_string())
    print("\nPoisson regression minus baselines (negative = Poisson better):")
    print(paired_comparisons(backtest.predictions)[["baseline", "metric", "difference", "adj_ci_low",
                                                    "adj_ci_high", "p_holm"]].round(4).to_string())
    print("\nOver/Under 2.5 vs market:")
    print(market_comparison(backtest.predictions, odds).round(4).to_string())
    print("\nDecision rule:")
    print(readiness(backtest.predictions).to_string())
    for name, path in paths.items():
        print(f"saved {name}: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
