"""Historical goal statistics for a fixture's two clubs: goals, Over/Under and BTTS.

These are **descriptive frequencies** from each club's own past Serie A
matches, for example "7 of its last 10 matches had over 2.5 goals". They are
not predictions for the upcoming fixture and have not been validated as
probabilities. The 1X2 models, features and predictions are not affected.

Usage::

    from src.goal_stats import fixture_goal_stats, team_goal_rows
    rows = team_goal_rows(matches)  # once; cheap to filter afterwards
    stats = fixture_goal_stats(rows, "Inter", "Milan", "2026-11-22", "last10", season_start=2026)
    stats.home.stats["over_2_5"], stats.home.size

How the match rows are built
----------------------------
The match processing of ``src.feature_engineering`` is reused, not
re-implemented. ``order_matches`` and ``team_match_rows`` turn every
completed match into two team rows (goals for, goals against, venue, season
and Serie A *spell*), and this module only adds the opponent and the
goal-line indicators.

Time safety
-----------
A sample for a fixture on date ``D`` contains only matches completed
**strictly before** ``D``: matches on the same day, the fixture itself and
anything later are never used. Historical demo mode therefore shows exactly
what was known on the morning of the match. The season of the fixture comes
from the calendar (``season_start``), never from results.

Windows, seasons, promoted and returning clubs
----------------------------------------------
* **Spells.** As in the feature engineering, a club's history is its current
  Serie A *spell* (consecutive seasons). Rolling windows (last 5/10/20) run
  across the summer break within a spell, so early in a season they include
  the end of the previous one.
* **Promoted and returning clubs** start with no history: their Serie B
  matches are not in the data, and matches from an earlier Serie A spell
  (often many years and a different squad ago) are deliberately not used.
* **Current season to date** restarts for every club each season.
* **Venue windows** use home matches for the fixture's home club and away
  matches for its away club.
* **Short and empty samples.** A window uses the matches that exist, up to
  its size, and reports the exact count. With no matches every statistic is
  missing (NaN), never 0%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.feature_engineering import order_matches, team_match_rows
from src.train import season_name

GOAL_LINES = (0.5, 1.5, 2.5, 3.5)
VENUES = ("home", "away")
ROW_COLUMNS = ["match_id", "season", "season_start", "date", "team", "opponent", "venue", "spell",
               "goals_for", "goals_against", "total_goals", "scored", "clean_sheet", "btts"]


def line_key(line: float, side: str = "over") -> str:
    """Column/statistic name for a goal line, e.g. ``line_key(2.5)`` -> ``"over_2_5"``."""
    if side not in ("over", "under"):
        raise ValueError(f"side must be 'over' or 'under', not {side!r}")
    return f"{side}_{line:g}".replace(".", "_")


# Counted events: statistic name -> label. Rates are count / matches.
EVENTS = {"scored": "Scored", "clean_sheet": "Clean sheet", "btts_yes": "BTTS Yes", "btts_no": "BTTS No"}
for _line in GOAL_LINES:
    EVENTS[line_key(_line, "over")] = f"Over {_line:g} goals"
    EVENTS[line_key(_line, "under")] = f"Under {_line:g} goals"
AVERAGES = {"goals_for": "Goals scored per match", "goals_against": "Goals conceded per match",
            "total_goals": "Total goals per match"}


# ====================================================================== windows


@dataclass(frozen=True)
class HistoryWindow:
    key: str
    label: str
    size: int | None  # number of matches; None = the whole current season so far
    by_venue: bool = False  # home matches for the home club, away matches for the away club

    @property
    def current_season(self) -> bool:
        return self.size is None

    def team_label(self, venue: str) -> str:
        """The label for one club, e.g. 'Last 5 home matches'."""
        return f"Last {self.size} {venue} matches" if self.by_venue else self.label


HISTORY_WINDOWS = {
    window.key: window
    for window in (
        HistoryWindow("last5", "Last 5 matches", 5),
        HistoryWindow("last10", "Last 10 matches", 10),
        HistoryWindow("last20", "Last 20 matches", 20),
        HistoryWindow("season", "Current season to date", None),
        HistoryWindow("venue5", "Last 5 home / away matches", 5, by_venue=True),
        HistoryWindow("venue10", "Last 10 home / away matches", 10, by_venue=True),
    )
}


def get_window(window: str | HistoryWindow) -> HistoryWindow:
    if isinstance(window, HistoryWindow):
        return window
    if window not in HISTORY_WINDOWS:
        raise ValueError(f"Unknown window {window!r}; choose from {list(HISTORY_WINDOWS)}")
    return HISTORY_WINDOWS[window]


# ====================================================================== team match rows


def team_goal_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per club per completed match, with goal-line and BTTS indicators.

    Unplayed rows (no result) are dropped. Built with
    ``feature_engineering.order_matches`` and ``team_match_rows``, so venues,
    seasons and spells are defined exactly as for the model features. Spells
    depend only on earlier seasons, so computing them on the full data gives
    the same values as computing them on any earlier cut-off.
    """
    played = matches[matches["result"].notna()]
    if "time" not in played.columns:
        played = played.assign(time=pd.NA)
    ordered = order_matches(played)
    rows = team_match_rows(ordered)

    teams = ordered.set_index("match_id")[["home_team", "away_team"]]
    home_rows = rows["venue"] == "home"
    rows["opponent"] = np.where(home_rows, teams["away_team"].reindex(rows["match_id"]).to_numpy(),
                                teams["home_team"].reindex(rows["match_id"]).to_numpy())
    rows["opponent"] = rows["opponent"].astype("string")
    rows["venue"] = rows["venue"].astype("string")
    rows["goals_for"] = rows["goals_for"].astype("int64")
    rows["goals_against"] = rows["goals_against"].astype("int64")
    rows["total_goals"] = rows["goals_for"] + rows["goals_against"]
    rows["scored"] = rows["goals_for"] > 0
    rows["clean_sheet"] = rows["goals_against"] == 0
    rows["btts"] = rows["scored"] & (rows["goals_against"] > 0)
    for line in GOAL_LINES:
        rows[line_key(line)] = rows["total_goals"] > line
    columns = ROW_COLUMNS + [line_key(line) for line in GOAL_LINES]
    return rows.sort_values(["team", "date", "match_id"], kind="stable")[columns].reset_index(drop=True)


# ====================================================================== statistics


def goal_statistics(sample: pd.DataFrame) -> dict:
    """Counts and averages for a sample of team rows (from ``team_goal_rows``).

    ``matches`` is the denominator; ``rate(stats, key)`` turns a count into a
    share. Over and Under use the total goals of both clubs; BTTS Yes needs
    both clubs to score at least once. With an empty sample the averages are
    NaN (and so are the rates), never 0.
    """
    n = len(sample)
    stats: dict = {"matches": n}
    for column in AVERAGES:
        stats[column] = float(sample[column].mean()) if n else float("nan")
    stats["scored"] = int(sample["scored"].sum())
    stats["clean_sheet"] = int(sample["clean_sheet"].sum())
    stats["btts_yes"] = int(sample["btts"].sum())
    stats["btts_no"] = n - stats["btts_yes"]
    for line in GOAL_LINES:
        over = int(sample[line_key(line)].sum())
        stats[line_key(line, "over")] = over
        stats[line_key(line, "under")] = n - over
    return stats


def rate(stats: dict, key: str) -> float:
    """Share of the sample's matches with event ``key``; NaN when the sample is empty."""
    if key not in EVENTS:
        raise ValueError(f"{key!r} is not a counted event; choose from {list(EVENTS)}")
    return stats[key] / stats["matches"] if stats["matches"] else float("nan")


def total_goals_distribution(sample: pd.DataFrame, max_goals: int = 5) -> pd.Series:
    """Number of matches with 0, 1, ..., ``max_goals - 1`` and ``max_goals``+ total goals."""
    labels = [str(goals) for goals in range(max_goals)] + [f"{max_goals}+"]
    capped = sample["total_goals"].clip(upper=max_goals)
    counts = capped.value_counts().reindex(range(max_goals + 1), fill_value=0)
    return pd.Series(counts.to_numpy(), index=labels, name="matches")


# ====================================================================== samples


@dataclass(frozen=True)
class TeamSample:
    """One club's matches in a window, as known before the fixture date."""

    team: str
    venue: str  # the club's venue in the fixture: "home" or "away"
    window: HistoryWindow
    date: pd.Timestamp
    season_start: int
    matches: pd.DataFrame = field(repr=False)
    spell_matches: int  # Serie A matches in the club's current spell before the date
    earlier_spell_end: int | None  # last season of an earlier, excluded Serie A spell

    @property
    def size(self) -> int:
        return len(self.matches)

    @property
    def target(self) -> int | None:
        return self.window.size

    @property
    def complete(self) -> bool:
        """True when the window holds its full number of matches (always for season to date)."""
        return self.window.current_season or self.size == self.window.size

    @property
    def label(self) -> str:
        return self.window.team_label(self.venue)

    @property
    def stats(self) -> dict:
        return goal_statistics(self.matches)

    @property
    def seasons(self) -> list[int]:
        return sorted(int(s) for s in self.matches["season_start"].unique())

    def sample_text(self) -> str:
        """E.g. '10 of 10 matches', '3 of 10 matches' or '6 matches'."""
        unit = "match" if self.size == 1 else "matches"
        return f"{self.size} {unit}" if self.window.current_season else f"{self.size} of {self.target} {unit}"

    def empty_reason(self) -> str | None:
        """Why the sample is empty, in words; None if it isn't."""
        if self.size:
            return None
        if self.spell_matches:
            if self.window.current_season:
                return f"no {season_name(self.season_start)} matches yet before this date"
            return f"no {self.venue} matches yet in its current Serie A spell"
        if self.earlier_spell_end is not None:
            return (f"back in Serie A after a gap (last Serie A season {season_name(self.earlier_spell_end)}); "
                    "earlier spells and Serie B matches are not used")
        return "no earlier Serie A matches in the data"


def current_spell(team_rows: pd.DataFrame, season_start: int) -> tuple[pd.DataFrame, int | None]:
    """The club's rows from its current Serie A spell, and the end of an excluded earlier spell.

    ``team_rows`` are one club's rows, all before the fixture date. The
    current spell continues only if the club played Serie A this season or
    last season; otherwise it is promoted/returning and has no current history.
    """
    if team_rows.empty:
        return team_rows, None
    last_season = int(team_rows["season_start"].max())
    if last_season > season_start:
        raise ValueError(f"History from {season_name(last_season)} is later than the fixture's season "
                         f"{season_name(season_start)}")
    if last_season < season_start - 1:  # a gap: promoted again after relegation
        return team_rows.iloc[0:0], last_season
    spell = team_rows[team_rows["spell"] == team_rows["spell"].max()]
    earlier = team_rows.loc[team_rows["spell"] < team_rows["spell"].max(), "season_start"]
    return spell, (int(earlier.max()) if len(earlier) else None)


def _as_date(date) -> pd.Timestamp:
    stamp = pd.Timestamp(date)
    if pd.isna(stamp):
        raise ValueError("A fixture date is required")
    return stamp.normalize()


def team_sample(rows: pd.DataFrame, team: str, date, window, venue: str, season_start: int) -> TeamSample:
    """``team``'s matches in ``window`` completed strictly before ``date``.

    ``venue`` is the club's venue in the fixture (used by venue windows) and
    ``season_start`` the fixture's season (used by spells and season to date).
    """
    window = get_window(window)
    if venue not in VENUES:
        raise ValueError(f"venue must be 'home' or 'away', not {venue!r}")
    date = _as_date(date)
    season_start = int(season_start)

    before = rows[(rows["team"] == team) & (rows["date"] < date)]
    spell, earlier_end = current_spell(before, season_start)
    if window.current_season:
        sample = spell[spell["season_start"] == season_start]
    else:
        sample = spell[spell["venue"] == venue] if window.by_venue else spell
        sample = sample.tail(window.size)
    return TeamSample(team, venue, window, date, season_start, sample.reset_index(drop=True),
                      spell_matches=len(spell), earlier_spell_end=earlier_end)


@dataclass(frozen=True)
class FixtureGoalStats:
    home: TeamSample
    away: TeamSample

    @property
    def window(self) -> HistoryWindow:
        return self.home.window

    def table(self) -> pd.DataFrame:
        """One row per statistic with each club's value, count and sample size (for display)."""
        records = []
        for sample in (self.home, self.away):
            stats = sample.stats
            for key, label in AVERAGES.items():
                records.append({"team": sample.team, "venue": sample.venue, "statistic": key, "label": label,
                                "kind": "average", "count": np.nan, "matches": stats["matches"],
                                "value": stats[key]})
            for key, label in EVENTS.items():
                records.append({"team": sample.team, "venue": sample.venue, "statistic": key, "label": label,
                                "kind": "rate", "count": stats[key], "matches": stats["matches"],
                                "value": rate(stats, key)})
        return pd.DataFrame(records)


def fixture_goal_stats(rows: pd.DataFrame, home_team: str, away_team: str, date, window,
                       season_start: int) -> FixtureGoalStats:
    """Goal statistics for both clubs of a fixture, from matches strictly before ``date``.

    ``rows`` come from ``team_goal_rows``. Venue windows use the home club's
    home matches and the away club's away matches.
    """
    if home_team == away_team:
        raise ValueError("The home and away teams must be different")
    return FixtureGoalStats(
        home=team_sample(rows, home_team, date, window, "home", season_start),
        away=team_sample(rows, away_team, date, window, "away", season_start),
    )


def goal_stats_for_fixture(matches: pd.DataFrame, home_team: str, away_team: str, date,
                           window="last10") -> FixtureGoalStats:
    """Convenience wrapper: builds the rows and finds the fixture's season from the calendar."""
    from src.predict import resolve_season  # the season boundaries used by the app

    season = resolve_season(matches, date).season_start
    return fixture_goal_stats(team_goal_rows(matches), home_team, away_team, date, window, season)
