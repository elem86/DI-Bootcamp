"""Pre-match team performance features for every Serie A match.

Reads ``data/processed/matches.csv`` and writes ``data/processed/features.csv``:
one row per match with identifiers, the prediction targets (the original
result and score) and features describing the home and away teams **as they
stood before kick-off**.

Usage (from the project folder)::

    python -m src.feature_engineering

How leakage is prevented
------------------------
Every feature is first computed as a running "state after each match"
(e.g. points per game over a team's last 5 matches *including* that match).
Each match then looks up the latest state from a **strictly earlier date**
(``pd.merge_asof`` with ``allow_exact_matches=False``). As a result:

* a match never sees its own score or statistics;
* matches played on the same date never see each other's results, even
  when they are processed in the same run;
* row order in the input file does not matter.

Season transitions and promoted teams
-------------------------------------
A team's *spell* is a run of consecutive Serie A seasons. Form windows
(last 5 / 10 matches) run across the summer break within a spell, so a team
starts a new season carrying its form from the end of the previous one.
A team that arrives after a gap, which almost always means promotion from
Serie B, starts a new spell with no history, because its Serie B matches are
not in the data.

Season-to-date features always restart at zero for every team each season.
Previous-season features are missing for promoted teams; the ``promoted``
flag makes that explicit.

Missing history
---------------
A form window is only filled once the team has played the full number of
matches in its current spell (e.g. 5 for ``form5``); before that the
feature is missing (NaN) instead of being averaged over fewer matches.
``history_matches`` and ``season_played`` record how much history exists.
In the first season in the data (1993-94) nothing earlier is known, so
history-based features are missing and ``promoted`` is unknown (<NA>).
"""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.preprocessing import load_matches

FEATURES_FILE = PROCESSED_DATA_DIR / "features.csv"

FORM_WINDOWS = (5, 10)
SIDES = ("home", "away")

# Per-game averages calculated over each window: feature suffix -> per-match value
FORM_STATS = {
    "ppg": "points",  # points per game (3 win, 1 draw, 0 loss)
    "gf": "goals_for",  # goals scored per game
    "ga": "goals_against",  # goals conceded per game
    "gd": "goal_diff",  # goal difference per game
}

ID_COLUMNS = ["match_id", "season", "season_start", "date", "home_team", "away_team"]
TARGET_COLUMN = "result"
TARGET_COLUMNS = [TARGET_COLUMN, "home_goals", "away_goals"]


def _team_feature_names() -> list[str]:
    """Feature names for one team, without the home_/away_ prefix."""
    names = ["history_matches"]
    for window in FORM_WINDOWS:
        names += [f"form{window}_{stat}" for stat in FORM_STATS]
    for window in FORM_WINDOWS:
        names += [f"venue_form{window}_{stat}" for stat in FORM_STATS]
    names += ["season_played", "season_ppg", "season_gf", "season_ga", "season_gd"]
    names += ["prev_season_ppg", "prev_season_gd", "promoted"]
    return names


TEAM_FEATURES = _team_feature_names()
FEATURE_COLUMNS = [f"{side}_{name}" for side in SIDES for name in TEAM_FEATURES]
OUTPUT_COLUMNS = ID_COLUMNS + TARGET_COLUMNS + FEATURE_COLUMNS


def order_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Sort matches chronologically and give each one a stable ``match_id``."""
    ordered = matches.sort_values(
        ["date", "time", "home_team"], na_position="first", kind="stable"
    ).reset_index(drop=True)
    ordered.insert(0, "match_id", range(len(ordered)))
    return ordered


def team_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape to one row per team per match (each match appears twice).

    Adds the team's ``spell``: a counter that increases whenever the team
    returns to Serie A after missing at least one season.
    """
    sides = []
    for venue, other in (("home", "away"), ("away", "home")):
        sides.append(
            pd.DataFrame(
                {
                    "match_id": matches["match_id"],
                    "season": matches["season"],
                    "season_start": matches["season_start"],
                    "date": matches["date"],
                    "team": matches[f"{venue}_team"],
                    "venue": venue,
                    "goals_for": matches[f"{venue}_goals"].astype("float64"),
                    "goals_against": matches[f"{other}_goals"].astype("float64"),
                }
            )
        )
    rows = pd.concat(sides, ignore_index=True)
    rows["goal_diff"] = rows["goals_for"] - rows["goals_against"]
    rows["points"] = 3.0 * (rows["goal_diff"] > 0) + 1.0 * (rows["goal_diff"] == 0)

    team_seasons = rows[["team", "season_start"]].drop_duplicates().sort_values(["team", "season_start"])
    gap = team_seasons.groupby("team")["season_start"].diff() != 1
    team_seasons["spell"] = gap.astype(int).groupby(team_seasons["team"]).cumsum()
    rows = rows.merge(team_seasons, on=["team", "season_start"], how="left")

    return rows.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)


def _before_date(rows: pd.DataFrame, by: list[str], after_match: pd.DataFrame) -> pd.DataFrame:
    """For each row, the latest values of ``after_match`` from an earlier date.

    ``after_match`` holds a running state *after* each row's match (aligned
    with ``rows``). Groups are defined by the ``by`` columns. Rows with no
    earlier match in their group get NaN.
    """
    state = pd.concat([rows[by + ["date"]], after_match], axis=1)
    # One state per group and date: the one after the last match that day.
    state = state.drop_duplicates(by + ["date"], keep="last").sort_values("date", kind="stable")

    left = rows[by + ["date"]].reset_index().sort_values("date", kind="stable")
    merged = pd.merge_asof(left, state, on="date", by=by, allow_exact_matches=False)
    return merged.set_index("index").reindex(rows.index)[list(after_match.columns)]


def _rolling_form(rows: pd.DataFrame, by: list[str], window: int, prefix: str) -> pd.DataFrame:
    """Per-game averages over the last ``window`` matches of each group."""
    grouped = rows.groupby(by, sort=False)
    after = pd.DataFrame(
        {
            f"{prefix}{window}_{name}": grouped[column].transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            )
            for name, column in FORM_STATS.items()
        },
        index=rows.index,
    )
    return _before_date(rows, by, after)


def _season_to_date(rows: pd.DataFrame) -> pd.DataFrame:
    """Matches played and per-game averages so far this season."""
    grouped = rows.groupby(["team", "season"], sort=False)
    played = grouped.cumcount() + 1
    after = pd.DataFrame(
        {
            "season_played": played,
            "season_ppg": grouped["points"].cumsum() / played,
            "season_gf": grouped["goals_for"].cumsum() / played,
            "season_ga": grouped["goals_against"].cumsum() / played,
            "season_gd": grouped["goal_diff"].cumsum() / played,
        },
        index=rows.index,
    )
    before = _before_date(rows, ["team", "season"], after)
    before["season_played"] = before["season_played"].fillna(0).astype("int64")
    return before


def _history_count(rows: pd.DataFrame) -> pd.Series:
    """Number of earlier Serie A matches in the team's current spell."""
    after = pd.DataFrame(
        {"history_matches": rows.groupby(["team", "spell"], sort=False).cumcount() + 1},
        index=rows.index,
    )
    before = _before_date(rows, ["team", "spell"], after)
    return before["history_matches"].fillna(0).astype("int64")


def _previous_season(rows: pd.DataFrame) -> pd.DataFrame:
    """Last season's points and goal difference per game, plus the promoted flag."""
    table = (
        rows.groupby(["team", "season_start"])
        .agg(prev_season_ppg=("points", "mean"), prev_season_gd=("goal_diff", "mean"))
        .reset_index()
    )
    table["season_start"] += 1  # last season's numbers become this season's features
    prev = rows[["team", "season_start"]].merge(table, on=["team", "season_start"], how="left")
    prev.index = rows.index

    first_season = rows["season_start"].min()
    promoted = prev["prev_season_ppg"].isna().astype("boolean")
    promoted[rows["season_start"] == first_season] = pd.NA  # no earlier data to tell
    prev["promoted"] = promoted
    return prev[["prev_season_ppg", "prev_season_gd", "promoted"]]


def team_features(rows: pd.DataFrame) -> pd.DataFrame:
    """All pre-match features for each team-match row (see module docstring)."""
    parts = [_history_count(rows).to_frame()]
    for window in FORM_WINDOWS:
        parts.append(_rolling_form(rows, ["team", "spell"], window, "form"))
    for window in FORM_WINDOWS:
        # venue form: home matches only for the home side, away matches only for the away side
        parts.append(_rolling_form(rows, ["team", "spell", "venue"], window, "venue_form"))
    parts += [_season_to_date(rows), _previous_season(rows)]

    features = pd.concat(parts, axis=1)[TEAM_FEATURES]
    return pd.concat([rows[["match_id", "venue"]], features], axis=1)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match: identifiers, targets and home/away team features."""
    ordered = order_matches(matches)
    per_team = team_features(team_match_rows(ordered))

    result = ordered[ID_COLUMNS + TARGET_COLUMNS]
    for side in SIDES:
        side_features = (
            per_team[per_team["venue"] == side]
            .drop(columns="venue")
            .rename(columns={name: f"{side}_{name}" for name in TEAM_FEATURES})
        )
        result = result.merge(side_features, on="match_id", how="left", validate="one_to_one")

    return result[OUTPUT_COLUMNS]


def save_features(features: pd.DataFrame, path=FEATURES_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def load_features(path=FEATURES_FILE) -> pd.DataFrame:
    """Load the feature table with the same dtypes it was saved with."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.feature_engineering` first.")
    dtypes = {col: "string" for col in ["season", "home_team", "away_team", TARGET_COLUMN]}
    dtypes |= {"home_goals": "Int64", "away_goals": "Int64"}
    dtypes |= {f"{side}_promoted": "boolean" for side in SIDES}
    features = pd.read_csv(path, dtype=dtypes, parse_dates=["date"])
    features["date"] = features["date"].astype("datetime64[ns]")
    return features


def main() -> None:
    features = build_features(load_matches())
    path = save_features(features)
    complete = features[FEATURE_COLUMNS].notna().all(axis=1)
    print(
        f"{len(features):,} matches x {len(FEATURE_COLUMNS)} features "
        f"({complete.sum():,} matches have every feature available)."
    )
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
