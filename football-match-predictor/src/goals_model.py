"""Expected goals and goal-market probabilities from independent Poisson scoring.

Each fixture gets two expected-goals rates, ``lambda_home`` and
``lambda_away``. Assuming the two goal counts are independent Poisson
variables, every goals probability follows exactly:

* **Total goals** ``T = home + away`` is Poisson with mean
  ``lambda_total = lambda_home + lambda_away``, so
  ``P(Under x.5) = P(T <= x)`` is the Poisson CDF at ``x`` and
  ``P(Over x.5) = 1 - P(Under x.5)``.
* **Both teams to score:** ``P(BTTS Yes) = (1 - exp(-lambda_home)) * (1 - exp(-lambda_away))``.
* **Exact scores:** ``P(h, a) = Poisson(h; lambda_home) * Poisson(a; lambda_away)`` on a
  grid of scores, whose size adapts to the rates so that the probability left
  outside it is below ``GRID_TOLERANCE``; that truncated mass is reported,
  never silently dropped.
* **1X2** (evaluation only): home win, draw and away win are sums over the grid.

Every pair of complementary probabilities (Over / Under, BTTS Yes / No) is
computed as ``p`` and ``1 - p``, so they sum to one.

Three ways of producing the rates
---------------------------------
All three use pre-match information only and are fitted on training rows only.

* ``LeagueAverageGoals``: the mean home and away goals of the training seasons,
  the same for every fixture.
* ``TeamStrengthGoals``: a smoothed attack / defence baseline with home
  advantage. A club's attack is its goals scored per match over its last
  ``window`` Serie A matches (strictly earlier dates, current spell only, as
  in ``src.feature_engineering``), shrunk toward the league average by
  ``prior_matches`` pseudo-matches and expressed relative to that average;
  defence likewise with goals conceded. Then
  ``lambda_home = mu_home * attack_home * defence_away`` and
  ``lambda_away = mu_away * attack_away * defence_home``. A club with no
  history (promoted, returning or in the first season) is league average.
* ``PoissonGoalsModel``: two L2-regularised Poisson regressions
  (``sklearn.linear_model.PoissonRegressor``), one for home goals and one for
  away goals, on the existing engineered features (``src.feature_engineering``).
  Missing history is median-imputed with missing indicators inside the
  pipeline, exactly like the 1X2 Logistic Regression. Elo is not used.

The feature set and penalty of each regression, and the window and prior of
the team baseline, are chosen by the walk-forward procedure in
``src.goals_backtest`` using earlier seasons only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import SIDES, _before_date
from src.goal_stats import GOAL_LINES, line_key, team_goal_rows
from src.train import FEATURE_SETS, feature_matrix

# ====================================================================== probabilities

GRID_TOLERANCE = 1e-12  # largest marginal tail probability left outside the score grid
MIN_GRID_GOALS = 10
MAX_GRID_GOALS = 250  # beyond this the grid is not extended; the truncated mass says how much is lost
MAX_TRUNCATED_MASS = 1e-9  # goal_probabilities refuses fixtures that lose more than this outside the grid
LAMBDA_COLUMNS = ["lambda_home", "lambda_away"]

OVER_COLUMNS = [line_key(line, "over") for line in GOAL_LINES]
UNDER_COLUMNS = [line_key(line, "under") for line in GOAL_LINES]
BTTS_COLUMNS = ["btts_yes", "btts_no"]
OUTCOME_COLUMNS = ["H", "D", "A"]
# The binary events scored in the backtest: probability column -> label
EVENTS = {line_key(line): f"Over {line}" for line in GOAL_LINES} | {"btts_yes": "BTTS Yes"}


def check_rates(rates, name: str = "rate") -> np.ndarray:
    """Expected goals as a float array; they must be finite and not negative."""
    values = np.atleast_1d(np.asarray(rates, dtype="float64"))
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite (got NaN or infinity)")
    if (values < 0).any():
        raise ValueError(f"{name} cannot be negative")
    return values


def _pair(lambda_home, lambda_away) -> tuple[np.ndarray, np.ndarray]:
    home, away = check_rates(lambda_home, "lambda_home"), check_rates(lambda_away, "lambda_away")
    if home.shape != away.shape:
        raise ValueError(f"lambda_home and lambda_away have different shapes {home.shape} and {away.shape}")
    return home, away


def _check_line(line: float) -> int:
    """The largest total that is *under* a goal line: 2 for 2.5. Only x.5 lines are allowed."""
    if line < 0 or line % 1 != 0.5:
        raise ValueError(f"Goal lines must be non-negative half-goal lines such as 2.5 (got {line})")
    return int(line)


def total_goals_probabilities(lambda_home, lambda_away, lines=GOAL_LINES) -> pd.DataFrame:
    """Exact Over/Under probabilities from ``Total ~ Poisson(lambda_home + lambda_away)``.

    One row per fixture; columns ``over_0_5``, ``under_0_5``, ... for each line.
    """
    home, away = _pair(lambda_home, lambda_away)
    total = home + away
    columns = {}
    for line in lines:
        under = poisson.cdf(_check_line(line), total)
        columns[line_key(line, "over")] = 1.0 - under
        columns[line_key(line, "under")] = under
    return pd.DataFrame(columns)


def btts_probabilities(lambda_home, lambda_away) -> pd.DataFrame:
    """``P(BTTS Yes) = (1 - exp(-lambda_home)) * (1 - exp(-lambda_away))`` and its complement."""
    home, away = _pair(lambda_home, lambda_away)
    yes = -np.expm1(-home) * -np.expm1(-away)  # expm1 keeps precision for small rates
    return pd.DataFrame({"btts_yes": yes, "btts_no": 1.0 - yes})


def grid_goals(lambda_max: float, tolerance: float = GRID_TOLERANCE) -> int:
    """Largest number of goals per side on an adaptive grid for rates up to ``lambda_max``.

    Chosen so that ``P(goals > max_goals) <= tolerance`` for the largest rate,
    with at least ``MIN_GRID_GOALS`` and at most ``MAX_GRID_GOALS``.
    """
    needed = int(poisson.isf(tolerance, float(lambda_max))) if lambda_max > 0 else 0
    return int(min(max(needed, MIN_GRID_GOALS), MAX_GRID_GOALS))


@dataclass
class ScoreGrid:
    """Exact-score probabilities ``probabilities[i, h, a]`` for scores 0..max_goals per side.

    ``truncated_mass[i]`` is the probability of fixture ``i`` that falls outside
    the grid (some side scores more than ``max_goals``).
    """

    probabilities: np.ndarray
    max_goals: int
    truncated_mass: np.ndarray

    @property
    def goals(self) -> np.ndarray:
        return np.arange(self.max_goals + 1)


def score_grid(lambda_home, lambda_away, max_goals: int | None = None,
               tolerance: float = GRID_TOLERANCE) -> ScoreGrid:
    """Independent-Poisson score probabilities on a grid (adaptive by default)."""
    home, away = _pair(lambda_home, lambda_away)
    if max_goals is None:
        max_goals = grid_goals(max(home.max(initial=0.0), away.max(initial=0.0)), tolerance)
    if max_goals < 0:
        raise ValueError("max_goals cannot be negative")
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals[None, :], home[:, None])
    p_away = poisson.pmf(goals[None, :], away[:, None])
    grid = p_home[:, :, None] * p_away[:, None, :]
    # P(outside) = 1 - P(home <= m) * P(away <= m), from the CDFs rather than 1 - grid.sum()
    inside = poisson.cdf(max_goals, home) * poisson.cdf(max_goals, away)
    return ScoreGrid(grid, int(max_goals), np.clip(1.0 - inside, 0.0, 1.0))


def outcome_probabilities(grid: ScoreGrid, renormalise: bool = True) -> pd.DataFrame:
    """Home win, draw and away win from a score grid, plus the ``truncated_mass`` it lost.

    With ``renormalise=True`` (default) H + D + A = 1 by dividing by the
    probability inside the grid; the truncated mass is kept as a column either way.
    """
    p = grid.probabilities
    home = np.tril(np.ones(p.shape[1:], dtype=bool), k=-1)  # home goals > away goals
    draw = np.eye(p.shape[1], dtype=bool)
    away = np.triu(np.ones(p.shape[1:], dtype=bool), k=1)
    table = pd.DataFrame({"H": p[:, home].sum(axis=1), "D": p[:, draw].sum(axis=1), "A": p[:, away].sum(axis=1)})
    if renormalise:
        table = table.div(table.sum(axis=1), axis=0)
    table["truncated_mass"] = grid.truncated_mass
    return table


def most_likely_scores(lambda_home: float, lambda_away: float, top: int = 5) -> pd.DataFrame:
    """The ``top`` most likely exact scores of one fixture, most likely first."""
    grid = score_grid(lambda_home, lambda_away)
    if check_rates(lambda_home).size != 1:
        raise ValueError("most_likely_scores takes the rates of one fixture")
    p = grid.probabilities[0]
    h, a = np.unravel_index(np.argsort(-p, axis=None, kind="stable")[:top], p.shape)
    return pd.DataFrame({"home_goals": h, "away_goals": a, "probability": p[h, a],
                         "score": [f"{x}-{y}" for x, y in zip(h, a)]})


def goal_probabilities(lambda_home, lambda_away) -> pd.DataFrame:
    """Every derived probability for each fixture, from its two expected-goals rates.

    Columns: the rates and ``lambda_total``; Over/Under for each goal line; BTTS
    Yes/No; 1X2 from the score grid (evaluation only); the most likely exact
    score and its probability; the grid's ``truncated_mass``. Raises if more
    than ``MAX_TRUNCATED_MASS`` of any fixture's probability falls outside the
    grid, instead of quietly dropping it.
    """
    home, away = _pair(lambda_home, lambda_away)
    table = pd.DataFrame({"lambda_home": home, "lambda_away": away, "lambda_total": home + away})
    grid = score_grid(home, away)
    if (grid.truncated_mass > MAX_TRUNCATED_MASS).any():
        worst = float(grid.truncated_mass.max())
        raise ValueError(f"Expected goals too large for the score grid: {worst:.2e} of the probability "
                         f"would fall outside 0-{grid.max_goals} goals per side")
    flat = grid.probabilities.reshape(len(home), -1)
    best = flat.argmax(axis=1)
    h, a = np.divmod(best, grid.max_goals + 1)
    parts = [
        table,
        total_goals_probabilities(home, away),
        btts_probabilities(home, away),
        outcome_probabilities(grid),
        pd.DataFrame({"likely_home_goals": h, "likely_away_goals": a,
                      "likely_score_probability": flat[np.arange(len(home)), best]}),
    ]
    out = pd.concat(parts, axis=1)
    return out[[c for c in out.columns if c != "truncated_mass"] + ["truncated_mass"]]


def exact_score_probability(lambda_home, lambda_away, home_goals, away_goals) -> np.ndarray:
    """Probability each fixture gave to the score that actually happened."""
    home, away = _pair(lambda_home, lambda_away)
    return poisson.pmf(np.asarray(home_goals), home) * poisson.pmf(np.asarray(away_goals), away)


# ====================================================================== targets and features

GOAL_TARGETS = {"home": "home_goals", "away": "away_goals"}

# Goal-focused feature set: the goals-scored and goals-conceded families of the
# engineered features (form, venue form, season to date), previous-season
# strength and the history / promoted indicators.
GOALS_TEAM_FEATURES = [
    "history_matches",
    "form5_gf", "form5_ga", "form10_gf", "form10_ga",
    "venue_form5_gf", "venue_form5_ga", "venue_form10_gf", "venue_form10_ga",
    "season_played", "season_gf", "season_ga",
    "prev_season_ppg", "prev_season_gd",
    "promoted",
]
GOALS_FEATURE_SETS = {
    "goals": [f"{side}_{name}" for side in SIDES for name in GOALS_TEAM_FEATURES],
    "compact": list(FEATURE_SETS["compact"]),
    "all": list(FEATURE_SETS["all"]),
}
# L2 penalty of PoissonRegressor (larger = smaller coefficients); features are standardised.
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def played(frame: pd.DataFrame) -> pd.DataFrame:
    """Only fixtures with a known score."""
    return frame[frame["home_goals"].notna() & frame["away_goals"].notna()]


def goals_target(frame: pd.DataFrame, side: str) -> np.ndarray:
    if side not in GOAL_TARGETS:
        raise ValueError(f"side must be one of {sorted(GOAL_TARGETS)}")
    values = frame[GOAL_TARGETS[side]]
    if values.isna().any():
        raise ValueError(f"{GOAL_TARGETS[side]} is missing for some rows; use played() first")
    return values.to_numpy(dtype="float64")


def poisson_deviance(goals, rates) -> np.ndarray:
    """Per-match Poisson deviance ``2 * (y * log(y / lambda) - (y - lambda))`` (0 * log 0 = 0)."""
    y = np.asarray(goals, dtype="float64")
    lam = check_rates(rates)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(y > 0, y * np.log(y / lam), 0.0)
    return 2.0 * (log_term - (y - lam))


# ====================================================================== models


def build_poisson_regression(alpha: float = 0.01) -> Pipeline:
    """Median imputation (+ missing indicators) -> scaling -> L2 Poisson regression (log link)."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=alpha, max_iter=2000)),
        ]
    )


def _rates_frame(home, away, index) -> pd.DataFrame:
    return pd.DataFrame({"lambda_home": np.asarray(home, float), "lambda_away": np.asarray(away, float)},
                        index=index)


class LeagueAverageGoals:
    """The same rates for every fixture: mean home and away goals of the training rows."""

    label = "League average"

    def fit(self, train: pd.DataFrame, inputs: pd.DataFrame | None = None) -> "LeagueAverageGoals":
        train = played(train)
        if train.empty:
            raise ValueError("No training matches")
        self.mu_home_ = float(goals_target(train, "home").mean())
        self.mu_away_ = float(goals_target(train, "away").mean())
        self.training_matches_ = len(train)
        return self

    def predict(self, frame: pd.DataFrame, inputs: pd.DataFrame | None = None) -> pd.DataFrame:
        n = len(frame)
        return _rates_frame(np.full(n, self.mu_home_), np.full(n, self.mu_away_), frame.index)

    @property
    def params(self) -> dict:
        return {}


# ------------------------------------------------------------------ team strength baseline

FIXTURE_KEYS = ["date", "home_team", "away_team"]
STRENGTH_WINDOWS = (10, 20, 38)  # last N Serie A matches of the current spell (38 = one season)
PRIOR_MATCHES_GRID = (3.0, 10.0, 30.0)  # pseudo-matches at league average
STRENGTH_COLUMNS = [f"{side}_{name}" for side in SIDES for name in ("matches", "scored", "conceded")]


def strength_inputs(history: pd.DataFrame, windows=STRENGTH_WINDOWS) -> dict[int, pd.DataFrame]:
    """Per fixture and window: each club's matches, goals scored and conceded before kick-off.

    ``history`` is a match or feature table with completed results. For every
    played fixture and each window N, the home and away clubs' totals over their
    last N Serie A matches **on strictly earlier dates** in their current spell
    (``home_matches``, ``home_scored``, ``home_conceded``, then ``away_...``).
    Matches on the same date are never seen, and later results cannot change
    the values (the same rule as the engineered features). A club with no earlier
    match in its spell has 0 matches.

    Returned frames are indexed like ``played(history)`` and include the fixture keys.
    """
    games = played(history)
    keep = ["season", "season_start", "date", "home_team", "away_team", "home_goals", "away_goals"]
    # one row per club and match, with spells (team_goal_rows only needs a non-missing result)
    rows = team_goal_rows(games[keep].assign(time=pd.NA, result="played"))
    by = ["team", "spell"]
    grouped = rows.groupby(by, sort=False)
    fixtures = games[FIXTURE_KEYS].copy()
    fixtures["_order"] = np.arange(len(games))
    result = {}
    for window in windows:
        after = pd.DataFrame(
            {
                "matches": grouped["goals_for"].transform(lambda s: s.rolling(window, min_periods=1).count()),
                "scored": grouped["goals_for"].transform(lambda s: s.rolling(window, min_periods=1).sum()),
                "conceded": grouped["goals_against"].transform(lambda s: s.rolling(window, min_periods=1).sum()),
            },
            index=rows.index,
        ).astype("float64")
        before = _before_date(rows, by, after).fillna(0.0)
        per_team = pd.concat([rows[["date", "team", "venue", "opponent"]], before], axis=1)
        table = fixtures
        for side in SIDES:
            side_rows = per_team[per_team["venue"] == side].rename(
                columns={"team": f"{side}_team", "opponent": "home_team" if side == "away" else "away_team"}
                | {c: f"{side}_{c}" for c in ("matches", "scored", "conceded")}
            ).drop(columns="venue")
            table = table.merge(side_rows, on=FIXTURE_KEYS, how="left", validate="one_to_one")
        table = table.sort_values("_order").drop(columns="_order")
        table.index = games.index
        if table[STRENGTH_COLUMNS].isna().any().any():
            raise ValueError("Some fixtures could not be matched to their clubs' history")
        result[window] = table[FIXTURE_KEYS + STRENGTH_COLUMNS]
    return result


class TeamStrengthGoals:
    """Smoothed attack / defence ratings with home advantage (see the module docstring).

    ``fit`` learns only league averages from the training rows; the club
    histories come from ``inputs`` (``strength_inputs``), which use strictly
    earlier dates.
    """

    label = "Team attack/defence"

    def __init__(self, window: int = 38, prior_matches: float = 10.0):
        if window not in STRENGTH_WINDOWS:
            raise ValueError(f"window must be one of {STRENGTH_WINDOWS}")
        if prior_matches <= 0:
            raise ValueError("prior_matches must be positive")
        self.window = window
        self.prior_matches = float(prior_matches)

    @property
    def params(self) -> dict:
        return {"window": self.window, "prior_matches": self.prior_matches}

    def fit(self, train: pd.DataFrame, inputs=None) -> "TeamStrengthGoals":
        train = played(train)
        if train.empty:
            raise ValueError("No training matches")
        self.mu_home_ = float(goals_target(train, "home").mean())
        self.mu_away_ = float(goals_target(train, "away").mean())
        self.mu_ = (self.mu_home_ + self.mu_away_) / 2  # goals per club per match
        return self

    def ratings(self, frame: pd.DataFrame, inputs: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Attack and defence of both clubs (1 = league average; defence > 1 concedes more)."""
        table = inputs[self.window].loc[frame.index]
        if not table[FIXTURE_KEYS].reset_index(drop=True).equals(frame[FIXTURE_KEYS].reset_index(drop=True)):
            raise ValueError("Strength inputs are not aligned with the fixtures")
        k, mu = self.prior_matches, self.mu_
        out = pd.DataFrame(index=frame.index)
        for side in SIDES:
            n = table[f"{side}_matches"]
            out[f"{side}_attack"] = (table[f"{side}_scored"] + k * mu) / (n + k) / mu
            out[f"{side}_defence"] = (table[f"{side}_conceded"] + k * mu) / (n + k) / mu
        return out

    def predict(self, frame: pd.DataFrame, inputs: dict[int, pd.DataFrame]) -> pd.DataFrame:
        r = self.ratings(frame, inputs)
        return _rates_frame(self.mu_home_ * r["home_attack"] * r["away_defence"],
                            self.mu_away_ * r["away_attack"] * r["home_defence"], frame.index)


# ------------------------------------------------------------------ Poisson regression


@dataclass
class SideChoice:
    """The settings of one side's regression: feature set and penalty."""

    feature_set: str
    alpha: float


@dataclass
class PoissonGoalsModel:
    """Two Poisson regressions, one per side, each with its own feature set and penalty."""

    home: SideChoice = field(default_factory=lambda: SideChoice("goals", 0.01))
    away: SideChoice = field(default_factory=lambda: SideChoice("goals", 0.01))
    feature_sets: dict = field(default_factory=lambda: dict(GOALS_FEATURE_SETS))

    label = "Poisson regression"

    @property
    def params(self) -> dict:
        return {f"{side}_{k}": v for side in SIDES for k, v in vars(getattr(self, side)).items()}

    def columns(self, side: str) -> list[str]:
        return self.feature_sets[getattr(self, side).feature_set]

    def fit(self, train: pd.DataFrame, inputs=None) -> "PoissonGoalsModel":
        train = played(train)
        if train.empty:
            raise ValueError("No training matches")
        self.models_ = {}
        for side in SIDES:
            columns = self.columns(side)
            model = build_poisson_regression(getattr(self, side).alpha)
            self.models_[side] = model.fit(feature_matrix(train, columns), goals_target(train, side))
        return self

    def predict(self, frame: pd.DataFrame, inputs=None) -> pd.DataFrame:
        rates = {side: self.models_[side].predict(feature_matrix(frame, self.columns(side))) for side in SIDES}
        return _rates_frame(rates["home"], rates["away"], frame.index)

    def coefficients(self, side: str) -> pd.Series:
        """Coefficients on the standardised inputs (missing-indicator columns included)."""
        model = self.models_[side]
        names = model[:-1].get_feature_names_out(self.columns(side))
        return pd.Series(model[-1].coef_, index=names)


def tune_poisson_side(train: pd.DataFrame, validation: pd.DataFrame, side: str,
                      feature_sets=None, alpha_grid=ALPHA_GRID) -> pd.DataFrame:
    """Mean Poisson deviance on ``validation`` of every feature set x alpha for one side (best first)."""
    feature_sets = feature_sets or GOALS_FEATURE_SETS
    train, validation = played(train), played(validation)
    y_train, y_val = goals_target(train, side), goals_target(validation, side)
    rows = []
    for name, columns in feature_sets.items():
        X_train, X_val = feature_matrix(train, columns), feature_matrix(validation, columns)
        for alpha in alpha_grid:
            model = build_poisson_regression(alpha).fit(X_train, y_train)
            rows.append({"side": side, "feature_set": name, "features": len(columns), "alpha": alpha,
                         "deviance": float(poisson_deviance(y_val, model.predict(X_val)).mean())})
    return pd.DataFrame(rows).sort_values("deviance", kind="stable").reset_index(drop=True)


def tune_team_strength(train: pd.DataFrame, validation: pd.DataFrame, inputs: dict[int, pd.DataFrame],
                       windows=STRENGTH_WINDOWS, priors=PRIOR_MATCHES_GRID) -> pd.DataFrame:
    """Combined home + away mean deviance on ``validation`` for every window x prior (best first)."""
    validation = played(validation)
    rows = []
    for window in windows:
        for prior in priors:
            model = TeamStrengthGoals(window, prior).fit(train)
            rates = model.predict(validation, inputs)
            deviance = (poisson_deviance(goals_target(validation, "home"), rates["lambda_home"])
                        + poisson_deviance(goals_target(validation, "away"), rates["lambda_away"]))
            rows.append({"window": window, "prior_matches": prior, "deviance": float(deviance.mean())})
    return pd.DataFrame(rows).sort_values("deviance", kind="stable").reset_index(drop=True)
