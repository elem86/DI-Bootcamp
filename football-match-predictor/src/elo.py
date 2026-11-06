"""Chronological, leakage-safe Elo ratings for Serie A clubs.

Usage::

    from src.elo import EloSettings, add_elo_features
    features_with_elo = add_elo_features(load_features())   # the 50 features + Elo columns

The Elo features are *extra* columns: the original 50 engineered features, the
saved production models and the Streamlit app do not use them.

Rating update
-------------
For a match between ``home`` and ``away``::

    E_home = 1 / (1 + 10 ** ((R_away - R_home - home_advantage) / 400))
    E_away = 1 - E_home
    S_home = 1 (home win), 0.5 (draw), 0 (away win);  S_away = 1 - S_home
    R_home_new = R_home + K * (S_home - E_home)
    R_away_new = R_away + K * (S_away - E_away) = R_away - K * (S_home - E_home)

The two changes cancel out, so a match never creates or destroys rating points.

No leakage: one calendar date at a time
---------------------------------------
Matches are processed date by date. Every match on a date first receives
both clubs' *pre-match* ratings, computed from results strictly before that
date. Only then are the updates of all that date's matches applied (each one
computed from the pre-date ratings), so:

* a match never sees its own result;
* matches on the same date never see each other's results, whatever their
  order in the input (if a club played twice on one date, both changes are
  computed from its pre-date rating and added);
* later results can never change an earlier rating.

Matches without a result (e.g. a future fixture) get pre-match ratings but do
not update anything.

Season transitions
------------------
Before the first date of every season after the first one in the data:

1. **Regression toward the mean:** every club's rating moves
   ``season_regression`` (30%) of the way back to ``initial_rating`` (1500):
   ``R = 1500 + (1 - 0.30) * (R - 1500)``. Squads change over the summer, so
   last season's gaps are only partly carried over.
2. **Promoted clubs:** a club that was not in Serie A the previous season
   (newly promoted, or returning after one or more seasons away) enters at
   the **mean rating of the clubs that left the league** at the end of the
   previous season (after the same regression). It is the rating of the clubs
   it replaces, and it keeps the league average at 1500.

Policies for clubs without usable history (documented choices, fixed before
any evaluation and never tuned):

* **First season in the data (1993-94):** every club starts at 1500; nothing
  earlier is known.
* **Promoted clubs:** we only have Serie A results, so nothing is known about a
  promoted club's Serie B season. No Serie B results are invented; the club is
  assumed to be as strong as the clubs it replaces.
* **Clubs returning after relegation:** their old rating is **discarded**, not
  carried over, because we cannot assume a club kept its strength while away.
  They enter exactly like a newly promoted club. This mirrors the "spells" in
  ``src.feature_engineering``.
* If no club left the league (only possible if the league grows), entrants
  start at ``initial_rating``.

When the league grows, more clubs enter than leave and the average drops a
little: the 2004-05 expansion (18 -> 20 clubs, 5 promoted for 3 relegated)
moved it to about 1493. The summer regression toward 1500 removes most of the
gap within a few seasons (1499.95 by 2018-19).

Which clubs play in a season is known before the season starts, so the
entry rating uses no results from the season being rated.

The Elo expected score
----------------------
``home_elo_expected`` is ``E_home`` above: the home side's expected *score*,
where a draw counts as half a win. It is **not** a probability of a home win:
football has three outcomes, and ``E_home`` blends the win and draw chances.
It is kept for analysis only and is not used as a model feature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ELO_FEATURE_COLUMNS = ["home_elo", "away_elo", "elo_difference"]  # model features (difference = home - away)
ELO_COLUMNS = ELO_FEATURE_COLUMNS + ["home_elo_expected"]  # + the descriptive expected score
FIXTURE_KEYS = ["date", "home_team", "away_team"]
RESULT_SCORES = {"H": 1.0, "D": 0.5, "A": 0.0}  # home team's actual score


@dataclass(frozen=True)
class EloSettings:
    """Experimental defaults, fixed in advance; they were not tuned on any evaluation season."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 60.0
    season_regression: float = 0.30  # share of the gap to initial_rating removed each summer

    def __post_init__(self):
        if self.k_factor <= 0:
            raise ValueError("k_factor must be positive")
        if not 0 <= self.season_regression <= 1:
            raise ValueError("season_regression must be between 0 and 1")

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_SETTINGS = EloSettings()


# ====================================================================== formulas


def expected_score(rating: float, opponent: float, home_advantage: float = 0.0):
    """Expected score (win 1, draw 0.5, loss 0) of a side rated ``rating`` against ``opponent``.

    ``home_advantage`` is added to ``rating``; pass 0 for a neutral venue or for the away side.
    Works element-wise on arrays.
    """
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(opponent) - np.asarray(rating) - home_advantage) / 400.0))


def rating_change(home_rating: float, away_rating: float, home_score: float,
                  settings: EloSettings = DEFAULT_SETTINGS) -> float:
    """``K * (S_home - E_home)``: added to the home rating and subtracted from the away rating."""
    expected = expected_score(home_rating, away_rating, settings.home_advantage)
    return float(settings.k_factor * (home_score - expected))


def update_ratings(home_rating: float, away_rating: float, home_score: float,
                   settings: EloSettings = DEFAULT_SETTINGS) -> tuple[float, float]:
    """New (home, away) ratings after one match."""
    change = rating_change(home_rating, away_rating, home_score, settings)
    return home_rating + change, away_rating - change


def regress(rating: float, settings: EloSettings = DEFAULT_SETTINGS) -> float:
    """Off-season regression toward ``initial_rating``."""
    return settings.initial_rating + (1 - settings.season_regression) * (rating - settings.initial_rating)


# ====================================================================== the rating engine


def _validate(matches: pd.DataFrame) -> None:
    missing = {"season_start", "result", *FIXTURE_KEYS} - set(matches.columns)
    if missing:
        raise ValueError(f"Matches are missing columns: {sorted(missing)}")
    if not matches.index.is_unique:
        raise ValueError("Matches need a unique index (the Elo columns are aligned on it)")
    if matches[["season_start", *FIXTURE_KEYS]].isna().any().any():
        raise ValueError("Every match needs a season, date, home team and away team")
    if (matches["home_team"] == matches["away_team"]).any():
        raise ValueError("A club cannot play itself")
    results = matches["result"].dropna().astype(str)
    if not results.isin(list(RESULT_SCORES)).all():
        raise ValueError(f"Results must be one of {list(RESULT_SCORES)} (or missing for unplayed fixtures)")
    if matches.duplicated(FIXTURE_KEYS).any():
        raise ValueError("Duplicate fixtures (same date, home team and away team)")
    dates = matches.groupby("season_start")["date"].agg(["min", "max"]).sort_index()
    if (dates["min"].iloc[1:].to_numpy() <= dates["max"].iloc[:-1].to_numpy()).any():
        raise ValueError("Seasons overlap in time; every season must end before the next one starts")


def _season_clubs(matches: pd.DataFrame) -> dict[int, set[str]]:
    clubs = pd.concat([matches[["season_start", "home_team"]].set_axis(["season_start", "team"], axis=1),
                       matches[["season_start", "away_team"]].set_axis(["season_start", "team"], axis=1)])
    return {int(season): set(group["team"]) for season, group in clubs.groupby("season_start")}


def _start_season(ratings: dict[str, float], previous: set[str], current: set[str],
                  settings: EloSettings) -> tuple[dict[str, float], dict]:
    """Ratings at the start of a season: regression for every club, entry ratings for newcomers."""
    regressed = {team: regress(rating, settings) for team, rating in ratings.items()}
    leavers = sorted(previous - current)
    entrants = sorted(current - previous)
    entry = float(np.mean([regressed[t] for t in leavers])) if leavers else settings.initial_rating
    # Only clubs of the new season are kept: anyone away from Serie A loses their old rating
    start = {team: regressed[team] for team in current & previous}
    start |= {team: entry for team in entrants}
    return start, {"leavers": leavers, "entrants": entrants, "entry_rating": entry}


def run_elo(matches: pd.DataFrame, settings: EloSettings = DEFAULT_SETTINGS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the rating engine over every match.

    Returns three tables:

    * ``pre_match``: one row per input match (same index as ``matches``) with
      ``home_elo``, ``away_elo``, ``elo_difference`` and ``home_elo_expected``
      before kick-off;
    * ``history``: one row per club per match with its rating before and after
      the match (``rating_after`` equals ``rating_before`` for unplayed fixtures);
    * ``transitions``: one row per season with the clubs that left / entered
      and the entry rating of promoted clubs.
    """
    _validate(matches)
    clubs = _season_clubs(matches)
    order = matches.sort_values(["season_start", "date", "home_team"], kind="stable")
    seasons = order["season_start"].astype(int).to_numpy()
    dates = order["date"].to_numpy()
    homes, aways = order["home_team"].astype(str).to_numpy(), order["away_team"].astype(str).to_numpy()
    scores = order["result"].astype("object").map(RESULT_SCORES).astype(float).to_numpy()  # NaN = unplayed
    n = len(order)
    home_before, away_before, changes = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    transitions: list[dict] = []
    ratings: dict[str, float] = {}

    start = 0
    while start < n:
        season, date = seasons[start], dates[start]
        if start == 0 or season != seasons[start - 1]:
            if start == 0:
                ratings = {team: settings.initial_rating for team in clubs[season]}
                info = {"leavers": [], "entrants": sorted(clubs[season]), "entry_rating": settings.initial_rating}
            else:
                ratings, info = _start_season(ratings, clubs[seasons[start - 1]], clubs[season], settings)
            transitions.append({"season_start": int(season), "first_season": start == 0,
                                "clubs": len(clubs[season]), "entrants": info["entrants"],
                                "leavers": info["leavers"], "entry_rating": info["entry_rating"],
                                "mean_rating": float(np.mean(list(ratings.values())))})
        stop = start
        while stop < n and seasons[stop] == season and dates[stop] == date:
            stop += 1
        # 1. Pre-match ratings for every match of the day, from results strictly before it
        for i in range(start, stop):
            home_before[i], away_before[i] = ratings[homes[i]], ratings[aways[i]]
        # 2. Then all of the day's updates, each computed from the pre-date ratings
        for i in range(start, stop):
            if not np.isnan(scores[i]):
                changes[i] = rating_change(home_before[i], away_before[i], scores[i], settings)
        for i in range(start, stop):
            ratings[homes[i]] += changes[i]
            ratings[aways[i]] -= changes[i]
        start = stop

    history = pd.concat(
        [pd.DataFrame({"match_index": order.index, "season_start": seasons, "date": dates, "team": team,
                       "opponent": opponent, "venue": venue, "played": ~np.isnan(scores),
                       "rating_before": before, "rating_change": sign * changes})
         for team, opponent, venue, before, sign in ((homes, aways, "home", home_before, 1),
                                                    (aways, homes, "away", away_before, -1))],
        ignore_index=True,
    ).sort_values(["date", "match_index", "venue"], ascending=[True, True, False], kind="stable")
    history["rating_after"] = history["rating_before"] + history["rating_change"]
    history = history.reset_index(drop=True)
    pre_match = pd.DataFrame({"home_elo": home_before, "away_elo": away_before}, index=order.index).reindex(matches.index)
    pre_match["elo_difference"] = pre_match["home_elo"] - pre_match["away_elo"]
    pre_match["home_elo_expected"] = expected_score(pre_match["home_elo"], pre_match["away_elo"],
                                                    settings.home_advantage)
    return pre_match[ELO_COLUMNS].astype("float64"), history, pd.DataFrame(transitions)


def elo_features(matches: pd.DataFrame, settings: EloSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """Pre-match Elo columns for every match, aligned with ``matches`` (same index)."""
    return run_elo(matches, settings)[0]


def add_elo_features(features: pd.DataFrame, settings: EloSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """A copy of ``features`` with the Elo columns appended; existing columns are untouched.

    ``features`` needs ``season_start``, ``date``, ``home_team``, ``away_team``
    and ``result`` (``features.csv`` and ``matches.csv`` both have them).
    """
    if set(ELO_COLUMNS) & set(features.columns):
        raise ValueError("features already contain Elo columns")
    elo = elo_features(features, settings)
    return pd.concat([features, elo], axis=1)


# ====================================================================== analysis helpers


def ratings_before(matches: pd.DataFrame, cutoff, settings: EloSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """Every club's latest rating using only matches played strictly before ``cutoff``.

    Only clubs of the season that the cutoff falls in (or the last season
    started before it) are listed, with their matches played that season.
    """
    cutoff = pd.Timestamp(cutoff)
    played = matches[matches["date"] < cutoff]
    if played.empty:
        raise ValueError(f"No matches before {cutoff.date()}")
    _, history, _ = run_elo(played, settings)
    season = int(played["season_start"].max())
    current = history[history["season_start"] == season]
    latest = current.sort_values("date", kind="stable").groupby("team").tail(1)
    table = latest[["team", "rating_after"]].rename(columns={"rating_after": "rating"})
    table["matches_this_season"] = table["team"].map(current[current["played"]].groupby("team").size()).fillna(0)
    table["matches_this_season"] = table["matches_this_season"].astype(int)
    table["season_start"] = season
    table["as_of"] = cutoff
    return table.sort_values("rating", ascending=False).reset_index(drop=True)


def end_of_season_ratings(history: pd.DataFrame) -> pd.DataFrame:
    """Every club's rating after its last match of each season (rows = seasons, columns = clubs)."""
    played = history[history["played"]]
    last = played.sort_values("date", kind="stable").groupby(["season_start", "team"]).tail(1)
    return last.pivot(index="season_start", columns="team", values="rating_after")
