"""Import the raw Football-Data CSVs and standardise them into one dataset.

Each raw season file (``data/raw/I1_<season>.csv``) is read, mapped onto a
single schema with snake_case column names and proper types, and all seasons
are combined into ``data/processed/matches.csv``.

Quirks in the raw files that this module handles:

* **Ragged rows** - in some older seasons rows have more or fewer fields than
  the header (the extra fields are always empty). pandas' CSV reader would
  silently drop those rows, so files are read with the ``csv`` module and
  each row is padded or trimmed to the header length.
* **Unnamed columns** - several 1990s files have trailing blank headers.
* **Byte-order mark** - files from 2024-25 onwards start with a UTF-8 BOM.
* **Two date formats** - ``dd/mm/yy`` up to 2017-18, ``dd/mm/yyyy`` after.
* **Renamed columns** - the market-average odds are ``BbAvH``/``BbAvD``/
  ``BbAvA`` up to 2018-19 and ``AvgH``/``AvgD``/``AvgA`` afterwards;
  Football-Data also documents ``HG``/``AG``/``Res`` as alternatives to
  ``FTHG``/``FTAG``/``FTR``.

Usage (from the project folder)::

    python -m src.preprocessing
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_loader import SERIE_A_CODE, season_label, season_start_year

MATCHES_FILE = PROCESSED_DATA_DIR / "matches.csv"

# Standard column name -> raw Football-Data column names, in priority order.
# When several raw names exist, the first non-missing value is used.
MATCH_COLUMNS = {
    "date": ["Date"],
    "time": ["Time"],
    "home_team": ["HomeTeam"],
    "away_team": ["AwayTeam"],
    "home_goals": ["FTHG", "HG"],
    "away_goals": ["FTAG", "AG"],
    "result": ["FTR", "Res"],
    "ht_home_goals": ["HTHG"],
    "ht_away_goals": ["HTAG"],
    "ht_result": ["HTR"],
    "referee": ["Referee"],
}

STAT_COLUMNS = {
    "home_shots": ["HS"],
    "away_shots": ["AS"],
    "home_shots_on_target": ["HST"],
    "away_shots_on_target": ["AST"],
    "home_fouls": ["HF"],
    "away_fouls": ["AF"],
    "home_corners": ["HC"],
    "away_corners": ["AC"],
    "home_yellow_cards": ["HY"],
    "away_yellow_cards": ["AY"],
    "home_red_cards": ["HR"],
    "away_red_cards": ["AR"],
}

# Decimal (European) pre-match odds. Bet365 is the longest-running single
# bookmaker in the Serie A files; the market average smooths out any one
# bookmaker's margin.
ODDS_COLUMNS = {
    "b365_home": ["B365H"],
    "b365_draw": ["B365D"],
    "b365_away": ["B365A"],
    "avg_odds_home": ["AvgH", "BbAvH"],
    "avg_odds_draw": ["AvgD", "BbAvD"],
    "avg_odds_away": ["AvgA", "BbAvA"],
}

COLUMN_SOURCES = {**MATCH_COLUMNS, **STAT_COLUMNS, **ODDS_COLUMNS}

INTEGER_COLUMNS = ["home_goals", "away_goals", "ht_home_goals", "ht_away_goals", *STAT_COLUMNS]
FLOAT_COLUMNS = list(ODDS_COLUMNS)
OUTPUT_COLUMNS = ["season", "season_start", *COLUMN_SOURCES]

# Columns every match must have.
REQUIRED_COLUMNS = ["season", "date", "home_team", "away_team", "home_goals", "away_goals", "result"]


def read_raw_season(path: Path) -> pd.DataFrame:
    """Read one raw Football-Data CSV into a DataFrame of strings.

    Rows are padded/trimmed to the header length, unnamed columns and fully
    blank rows are dropped, and empty cells become missing values (NA).
    """
    text = path.read_bytes().decode("utf-8-sig")  # utf-8-sig strips the BOM
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{path.name} is empty")

    header = [name.strip() for name in rows[0]]
    width = len(header)
    body = []
    for line_no, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue  # blank line
        extra = row[width:]
        if any(cell.strip() for cell in extra):
            raise ValueError(f"{path.name} line {line_no}: data beyond the last header column")
        body.append((row + [""] * width)[:width])

    keep = [i for i, name in enumerate(header) if name]
    names = [header[i] for i in keep]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: duplicate column names {duplicates}")

    df = pd.DataFrame([[row[i].strip() for i in keep] for row in body], columns=names)
    return df.replace("", pd.NA)


def parse_dates(dates: pd.Series) -> pd.Series:
    """Parse ``dd/mm/yy`` and ``dd/mm/yyyy`` dates (both occur in the files)."""
    long_format = dates.str.len() == 10
    parsed = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    parsed[long_format] = pd.to_datetime(dates[long_format], format="%d/%m/%Y")
    parsed[~long_format] = pd.to_datetime(dates[~long_format], format="%d/%m/%y")
    return parsed


def _coalesce(raw: pd.DataFrame, sources: list[str]) -> pd.Series:
    """First non-missing value across the raw columns that exist in this file."""
    result = pd.Series(pd.NA, index=raw.index, dtype="object")
    for name in sources:
        if name in raw.columns:
            result = result.fillna(raw[name])
    return result


def standardise_season(raw: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """Map one raw season onto the standard schema with proper dtypes.

    Columns that a season does not have (e.g. shots before 2005-06) are kept
    as missing values, so every season has identical columns.
    """
    df = pd.DataFrame({name: _coalesce(raw, sources) for name, sources in COLUMN_SOURCES.items()})

    df.insert(0, "season", pd.Series(season_label(season_code), index=df.index, dtype="string"))
    df.insert(1, "season_start", season_start_year(season_code))
    df["date"] = parse_dates(df["date"])

    # errors="raise" (the default) so an unexpected value such as "abc" in a
    # goals column fails loudly instead of quietly becoming missing.
    for col in INTEGER_COLUMNS:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col]).astype("Float64")
    for col in ["time", "home_team", "away_team", "result", "ht_result", "referee"]:
        df[col] = df[col].astype("string")

    return df[OUTPUT_COLUMNS]


def season_files(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """All raw Serie A season files, oldest season first."""
    files = raw_dir.glob(f"{SERIE_A_CODE}_*.csv")
    return sorted(files, key=lambda p: season_start_year(p.stem.split("_")[1]))


def build_matches(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Read, standardise and combine every raw season into one DataFrame."""
    paths = season_files(raw_dir)
    if not paths:
        raise FileNotFoundError(
            f"No {SERIE_A_CODE}_*.csv files in {raw_dir}. Run `python -m src.data_loader` first."
        )
    seasons = [
        standardise_season(read_raw_season(path), path.stem.split("_")[1]) for path in paths
    ]
    matches = pd.concat(seasons, ignore_index=True)
    return matches.sort_values(
        ["date", "time", "home_team"], na_position="first", kind="stable"
    ).reset_index(drop=True)


def _expected_result(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    diff = (home_goals - away_goals).astype("Float64")
    return diff.map(lambda d: pd.NA if pd.isna(d) else "H" if d > 0 else "A" if d < 0 else "D")


def validate_matches(matches: pd.DataFrame) -> list[str]:
    """Check the combined dataset for problems. Returns a list of issues (empty = OK)."""
    issues = []

    for col in REQUIRED_COLUMNS:
        n = matches[col].isna().sum()
        if n:
            issues.append(f"{n} matches missing {col}")

    wrong = matches["result"] != _expected_result(matches["home_goals"], matches["away_goals"])
    if wrong.fillna(False).any():
        issues.append(f"{wrong.sum()} matches where result disagrees with the score")

    ht_known = matches["ht_result"].notna()
    ht_expected = _expected_result(matches["ht_home_goals"], matches["ht_away_goals"])
    wrong_ht = ht_known & (matches["ht_result"] != ht_expected)
    if wrong_ht.fillna(False).any():
        issues.append(f"{wrong_ht.sum()} matches where half-time result disagrees with the score")

    if (matches["home_team"] == matches["away_team"]).any():
        issues.append("matches where a team plays itself")

    dupes = matches.duplicated(["season", "home_team", "away_team"])
    if dupes.any():
        issues.append(f"{dupes.sum()} duplicate fixtures (same season, home and away team)")

    year = matches["date"].dt.year
    outside = ~year.between(matches["season_start"], matches["season_start"] + 1)
    if outside.any():
        issues.append(f"{outside.sum()} matches dated outside their season")

    counts = matches[INTEGER_COLUMNS]
    if (counts < 0).fillna(False).any().any():
        issues.append("negative values in goal/stat columns")

    odds = matches[FLOAT_COLUMNS]
    if (odds <= 1).fillna(False).any().any():
        issues.append("decimal odds of 1.0 or lower")

    # Every completed season is a double round robin: each team hosts every
    # other team once. The latest season is still being played, so skip it.
    latest = matches["season_start"].max()
    for season, games in matches[matches["season_start"] < latest].groupby("season"):
        n_teams = len(set(games["home_team"]) | set(games["away_team"]))
        if len(games) != n_teams * (n_teams - 1):
            issues.append(
                f"{season}: {len(games)} matches, expected {n_teams * (n_teams - 1)} "
                f"for {n_teams} teams"
            )

    return issues


def season_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per season: number of teams, matches and date range."""
    return matches.groupby("season").agg(
        teams=("home_team", "nunique"),
        matches=("result", "size"),
        first_match=("date", "min"),
        last_match=("date", "max"),
    )


def save_matches(matches: pd.DataFrame, path: Path = MATCHES_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def load_matches(path: Path = MATCHES_FILE) -> pd.DataFrame:
    """Load the processed dataset with the same dtypes it was saved with."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.preprocessing` first.")
    dtypes = {col: "Int64" for col in INTEGER_COLUMNS}
    dtypes |= {col: "Float64" for col in FLOAT_COLUMNS}
    dtypes |= {
        col: "string"
        for col in ["season", "time", "home_team", "away_team", "result", "ht_result", "referee"]
    }
    matches = pd.read_csv(path, dtype=dtypes, parse_dates=["date"])
    matches["date"] = matches["date"].astype("datetime64[ns]")
    return matches


def main() -> int:
    matches = build_matches()
    issues = validate_matches(matches)
    if issues:
        print("Validation failed; processed file not written:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    path = save_matches(matches)
    summary = season_summary(matches)
    print(
        f"{len(matches):,} matches from {len(summary)} seasons "
        f"({summary.index[0]} to {summary.index[-1]}) passed validation."
    )
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
