"""Adaptive league-wide scoring rates for Over/Under and BTTS probabilities.

Commits 15 and 16 found that the largest source of error in the goals
forecasts is season-to-season drift in how many goals Serie A produces. Every
method there learns its level from whole earlier seasons, so it cannot react
when scoring changes during a season. This module adds two adaptive methods
that follow the most recent completed league results. The walk-forward
comparison lives in ``src.scoring_drift_backtest``.

Two different kinds of "learning"
---------------------------------
* **Model fitting** uses whole training seasons only (seasons up to S-1 for
  evaluation season S): the historical event frequencies, the historical mean
  home and away goals, and the Poisson regression of commit 15. Nothing is
  fitted on season S.
* **The online scoring statistic** is updated during the evaluated season,
  exactly as it could be in real use: for a fixture on date D it counts the
  last ``WINDOW_MATCHES`` completed Serie A matches **on dates strictly before
  D**. That window may include earlier fixtures of season S. It never includes
  the fixture itself, any other fixture on the same calendar date, or anything
  later. Nothing is fitted to it; it is a running average that enters through
  the fixed formulas below.

Both settings are fixed in advance and are not tuned on any season:
``WINDOW_MATCHES = 200`` (a little over half a season) and
``PRIOR_MATCHES = 200`` pseudo-matches of shrinkage.

Candidate A: adaptive event frequency
-------------------------------------
For each event (Over 0.5, 1.5, 2.5, 3.5 and BTTS Yes) the prior rate ``p0`` is
the static event frequency of commit 16, the Jeffreys-smoothed share of the
training seasons, ``(k + 0.5) / (n + 1)``. With ``k_recent`` events in the
``n_recent`` matches of the window::

    p = (k_recent + PRIOR_MATCHES * p0) / (n_recent + PRIOR_MATCHES)

``p0`` is never 0 or 1, so neither is ``p``. With a full window the forecast is
half recent, half historical. With no earlier match it is ``p0``. The Over
events are nested (a 4-goal match is also Over 2.5), so the counts and the
priors both fall with the line and the forecasts are already non-increasing;
``monotone_over`` is applied anyway as a safeguard.

Candidate B: adaptive Poisson scoring level
-------------------------------------------
Starting from the existing Poisson regression's expected goals
``lambda_home`` and ``lambda_away`` for the fixture:

1. ``mu_home``, ``mu_away``: mean home and away goals of the training seasons;
2. ``recent_home``, ``recent_away``: mean home and away goals in the window;
3. shrink: ``level_home = (n_recent * recent_home + PRIOR_MATCHES * mu_home) / (n_recent + PRIOR_MATCHES)``
   (likewise away);
4. adjust: ``lambda_home' = lambda_home * level_home / mu_home`` (likewise away);
5. recompute Over/Under and BTTS with the commit-15 Poisson formulas
   (``src.goals_model.total_goals_probabilities`` and ``btts_probabilities``).

If the window is empty, or a training mean is not positive, the factor is 1
and the prediction is the unadjusted one. Invalid expected goals (missing,
negative or infinite) are refused rather than adjusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_models import EVENT_NAMES, OVER_EVENTS, EventFrequency, event_targets, monotone_over
from src.goals_model import (
    FIXTURE_KEYS,
    btts_probabilities,
    check_rates,
    goals_target,
    played,
    total_goals_probabilities,
)
from src.train import VALIDATION_SEASON

WINDOW_MATCHES = 200
PRIOR_MATCHES = 200.0

HISTORY_COLUMNS = ["season_start", *FIXTURE_KEYS, "home_goals", "away_goals"]
WINDOW_COLUMNS = ["window_matches", "window_home_goals", "window_away_goals",
                  *[f"window_{e}" for e in EVENT_NAMES], "window_first_date", "window_last_date"]


# ====================================================================== the online statistic


def completed_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Completed Serie A matches in a fixed chronological order, without 2025-26 and later.

    Ties on a date are ordered by home and away club so that the window is
    deterministic. Rows from the validation season onward are dropped, so they
    can never enter a window.
    """
    history = played(matches)
    history = history[history["season_start"] < VALIDATION_SEASON][HISTORY_COLUMNS].copy()
    history["date"] = pd.to_datetime(history["date"]).astype("datetime64[ns]")
    history = history.astype({"home_goals": "int64", "away_goals": "int64"})
    return history.sort_values(FIXTURE_KEYS, kind="stable").reset_index(drop=True)


def recent_window(history: pd.DataFrame, dates, window: int = WINDOW_MATCHES) -> pd.DataFrame:
    """For each date D: totals over the last ``window`` completed matches dated strictly before D.

    ``history`` comes from ``completed_history``. Returns one row per date
    (in the order given) with the number of matches in the window, their home
    and away goals, how many of them had each event, and the dates the window
    spans. Matches on D itself are excluded, as is everything later.
    """
    if window < 1:
        raise ValueError("The window needs at least one match")
    if not history["date"].is_monotonic_increasing:
        raise ValueError("history must be in chronological order (use completed_history)")
    dates = pd.to_datetime(pd.Series(dates)).astype("datetime64[ns]").to_numpy()
    played_dates = history["date"].to_numpy()
    end = np.searchsorted(played_dates, dates, side="left")  # matches strictly before D
    start = np.maximum(end - window, 0)

    def window_sum(values) -> np.ndarray:
        cumulative = np.concatenate([[0], np.cumsum(np.asarray(values, dtype="int64"))])
        return cumulative[end] - cumulative[start]

    targets = event_targets(history)
    table = {"window_matches": end - start,
             "window_home_goals": window_sum(history["home_goals"]),
             "window_away_goals": window_sum(history["away_goals"])}
    for event in EVENT_NAMES:
        table[f"window_{event}"] = window_sum(targets[event])
    empty = end == start
    first = played_dates[np.minimum(start, max(len(played_dates) - 1, 0))] if len(played_dates) else dates
    last = played_dates[np.maximum(end - 1, 0)] if len(played_dates) else dates
    table["window_first_date"] = np.where(empty, np.datetime64("NaT", "ns"), first)
    table["window_last_date"] = np.where(empty, np.datetime64("NaT", "ns"), last)
    return pd.DataFrame(table)[WINDOW_COLUMNS]


def shrunk_mean(total, matches, prior_mean, prior_matches: float = PRIOR_MATCHES) -> np.ndarray:
    """``(total + prior_matches * prior_mean) / (matches + prior_matches)``: a recent mean pulled toward history."""
    if prior_matches <= 0:
        raise ValueError("prior_matches must be positive")
    total = np.asarray(total, dtype="float64")
    matches = np.asarray(matches, dtype="float64")
    return (total + prior_matches * prior_mean) / (matches + prior_matches)


def level_factor(total, matches, prior_mean: float, prior_matches: float = PRIOR_MATCHES) -> np.ndarray:
    """Scoring-level multiplier ``shrunk recent mean / historical mean``; exactly 1 when it cannot be formed.

    The factor is 1 for an empty window and when ``prior_mean`` is not a
    positive finite number (no division by zero).
    """
    matches = np.asarray(matches, dtype="float64")
    if not np.isfinite(prior_mean) or prior_mean <= 0:
        return np.ones_like(matches)
    factor = shrunk_mean(total, matches, prior_mean, prior_matches) / prior_mean
    return np.where(matches > 0, factor, 1.0)


def _check_fixtures(fixtures: pd.DataFrame) -> None:
    if fixtures["date"].isna().any():
        raise ValueError("Every fixture needs a date")


# ====================================================================== Candidate A


class AdaptiveEventFrequency:
    """Event frequency of the recent window, shrunk toward the static training frequency.

    ``fit`` uses only the training rows (the static Jeffreys frequency);
    ``predict`` reads the online window from ``history`` for each fixture's date.
    """

    label = "Adaptive event frequency"

    def __init__(self, window: int = WINDOW_MATCHES, prior_matches: float = PRIOR_MATCHES):
        if window < 1 or prior_matches <= 0:
            raise ValueError("window must be at least 1 and prior_matches positive")
        self.window = int(window)
        self.prior_matches = float(prior_matches)

    @property
    def params(self) -> dict:
        return {"window": self.window, "prior_matches": self.prior_matches}

    def fit(self, train: pd.DataFrame) -> "AdaptiveEventFrequency":
        static = EventFrequency().fit(train)
        self.prior_ = dict(static.rates_)
        self.training_matches_ = static.training_matches_
        return self

    def components(self, fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """The window totals behind each fixture's forecast (indexed like ``fixtures``)."""
        _check_fixtures(fixtures)
        table = recent_window(history, fixtures["date"], self.window)
        table.index = fixtures.index
        return table

    def predict(self, fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        window = self.components(fixtures, history)
        out = pd.DataFrame(index=fixtures.index)
        for event in EVENT_NAMES:
            out[event] = shrunk_mean(window[f"window_{event}"], window["window_matches"], self.prior_[event],
                                     self.prior_matches)
        out[OVER_EVENTS] = monotone_over(out)
        return out


# ====================================================================== Candidate B


class AdaptiveScoringLevel:
    """Rescales Poisson expected goals by the recent league scoring level (see the module docstring)."""

    label = "Adaptive Poisson regression"

    def __init__(self, window: int = WINDOW_MATCHES, prior_matches: float = PRIOR_MATCHES):
        if window < 1 or prior_matches <= 0:
            raise ValueError("window must be at least 1 and prior_matches positive")
        self.window = int(window)
        self.prior_matches = float(prior_matches)

    @property
    def params(self) -> dict:
        return {"window": self.window, "prior_matches": self.prior_matches}

    def fit(self, train: pd.DataFrame) -> "AdaptiveScoringLevel":
        train = played(train)
        if train.empty:
            raise ValueError("No training matches")
        self.mu_home_ = float(goals_target(train, "home").mean())
        self.mu_away_ = float(goals_target(train, "away").mean())
        self.training_matches_ = len(train)
        return self

    def factors(self, fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Window totals, recent and shrunk means, and the home and away multipliers per fixture."""
        _check_fixtures(fixtures)
        window = recent_window(history, fixtures["date"], self.window)
        window.index = fixtures.index
        n = window["window_matches"].to_numpy(dtype="float64")
        out = window[["window_matches", "window_first_date", "window_last_date"]].copy()
        for side, mu in (("home", self.mu_home_), ("away", self.mu_away_)):
            total = window[f"window_{side}_goals"].to_numpy(dtype="float64")
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"recent_{side}"] = np.where(n > 0, total / np.where(n > 0, n, 1), np.nan)
            out[f"train_{side}"] = mu
            out[f"level_{side}"] = np.where(n > 0, shrunk_mean(total, n, mu, self.prior_matches), mu)
            out[f"{side}_factor"] = level_factor(total, n, mu, self.prior_matches)
        return out

    def adjust(self, rates: pd.DataFrame, fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Adjusted ``lambda_home`` and ``lambda_away`` (rows aligned with ``fixtures``)."""
        if len(rates) != len(fixtures):
            raise ValueError("rates and fixtures must have the same rows")
        home = check_rates(rates["lambda_home"], "lambda_home")
        away = check_rates(rates["lambda_away"], "lambda_away")
        factors = self.factors(fixtures, history)
        adjusted = pd.DataFrame({"lambda_home": home * factors["home_factor"].to_numpy(),
                                 "lambda_away": away * factors["away_factor"].to_numpy()}, index=fixtures.index)
        check_rates(adjusted["lambda_home"], "adjusted lambda_home")
        check_rates(adjusted["lambda_away"], "adjusted lambda_away")
        return adjusted

    def predict(self, rates: pd.DataFrame, fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Over/Under and BTTS probabilities from the adjusted rates (event columns only)."""
        return poisson_events(self.adjust(rates, fixtures, history))


def poisson_events(rates: pd.DataFrame) -> pd.DataFrame:
    """The five event probabilities from expected goals, with the commit-15 formulas."""
    over = total_goals_probabilities(rates["lambda_home"], rates["lambda_away"])
    btts = btts_probabilities(rates["lambda_home"], rates["lambda_away"])
    out = pd.concat([over, btts], axis=1)[EVENT_NAMES]
    out.index = rates.index
    return out


def response_share(matches_since_change, window: int = WINDOW_MATCHES, prior_matches: float = PRIOR_MATCHES):
    """Share of a permanent league-wide level shift the shrunk estimate has absorbed after
    ``matches_since_change`` matches, starting from a full window at the old level.

    ``min(j, window) / (window + prior_matches)``: with 200 and 200, a quarter
    after 100 matches and at most half after 200.
    """
    j = np.minimum(np.asarray(matches_since_change, dtype="float64"), window)
    return j / (window + prior_matches)
