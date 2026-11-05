"""Tests for src.preprocessing, using small hand-written raw files."""

import pandas as pd
import pytest

from src.preprocessing import (
    OUTPUT_COLUMNS,
    build_matches,
    load_matches,
    parse_dates,
    read_raw_season,
    save_matches,
    standardise_season,
    validate_matches,
)

# 1990s-style file: two-digit years, trailing blank header columns, and a
# ragged row that is shorter than the header.
OLD_SEASON = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,,\n"
    "I1,29/08/93,Inter,Milan,2,1,H,,\n"
    "I1,05/09/93,Milan,Inter,0,0,D\n"
    "\n"
)

# Modern file: BOM, four-digit years, renamed market-average odds (AvgH) and a
# row longer than the header whose extra fields are empty.
NEW_SEASON = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
    "I1,17/08/2024,18:30,Inter,Roma,1,3,A,12,9,1.50,4.20,6.00,1.52,4.10,5.80,,\n"
    "I1,24/08/2024,20:45,Roma,Inter,2,2,D,10,11,2.90,3.30,2.50,2.95,3.25,2.45\n"
)

# Mid-era file using the older BetBrain average column names.
BETBRAIN_SEASON = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,BbAvH,BbAvD,BbAvA\n"
    "I1,10/09/05,Lazio,Roma,1,0,H,2.40,3.10,2.90\n"
    "I1,20/02/06,Roma,Lazio,0,2,A,1.90,3.30,4.00\n"
)


def write(path, text):
    path.write_bytes(text.encode("utf-8"))
    return path


def test_read_raw_season_handles_ragged_rows_blank_columns_and_bom(tmp_path):
    old = read_raw_season(write(tmp_path / "I1_9394.csv", OLD_SEASON))
    new = read_raw_season(write(tmp_path / "I1_2425.csv", NEW_SEASON))

    assert list(old.columns) == ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    assert len(old) == 2  # short row kept, blank line dropped
    assert new.columns[0] == "Div"  # BOM stripped
    assert len(new) == 2  # long row kept


def test_read_raw_season_rejects_data_beyond_header(tmp_path):
    path = write(tmp_path / "I1_9394.csv", "Div,Date\nI1,29/08/93,surprise\n")
    with pytest.raises(ValueError, match="beyond the last header column"):
        read_raw_season(path)


def test_parse_dates_accepts_both_formats():
    parsed = parse_dates(pd.Series(["29/08/93", "17/08/2024", "01/01/00"]))
    assert list(parsed.dt.strftime("%Y-%m-%d")) == ["1993-08-29", "2024-08-17", "2000-01-01"]


def test_standardise_season_maps_columns_and_types(tmp_path):
    raw = read_raw_season(write(tmp_path / "I1_2425.csv", NEW_SEASON))

    df = standardise_season(raw, "2425")

    assert list(df.columns) == OUTPUT_COLUMNS
    first = df.iloc[0]
    assert first["season"] == "2024-25"
    assert first["season_start"] == 2024
    assert first["home_team"] == "Inter"
    assert first["away_goals"] == 3
    assert first["home_shots"] == 12
    assert first["avg_odds_home"] == pytest.approx(1.52)
    assert df["home_goals"].dtype == "Int64"
    assert df["b365_home"].dtype == "Float64"
    assert pd.isna(first["home_corners"])  # not in this file -> missing, not dropped


def test_standardise_season_uses_older_odds_names(tmp_path):
    raw = read_raw_season(write(tmp_path / "I1_0506.csv", BETBRAIN_SEASON))

    df = standardise_season(raw, "0506")

    assert list(df["avg_odds_away"]) == pytest.approx([2.90, 4.00])


def test_standardise_season_rejects_non_numeric_goals(tmp_path):
    raw = read_raw_season(write(tmp_path / "I1_9394.csv", OLD_SEASON.replace(",2,1,H", ",two,1,H")))
    with pytest.raises(ValueError):
        standardise_season(raw, "9394")


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "I1_9394.csv", OLD_SEASON)
    write(raw / "I1_0506.csv", BETBRAIN_SEASON)
    write(raw / "I1_2425.csv", NEW_SEASON)
    return raw


def test_build_matches_combines_seasons_in_date_order(raw_dir):
    matches = build_matches(raw_dir)

    assert len(matches) == 6
    assert matches["date"].is_monotonic_increasing
    assert list(matches["season"].unique()) == ["1993-94", "2005-06", "2024-25"]
    assert validate_matches(matches) == []


def test_build_matches_without_files_explains_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_loader"):
        build_matches(tmp_path)


def test_save_and_load_round_trip(raw_dir, tmp_path):
    matches = build_matches(raw_dir)
    path = save_matches(matches, tmp_path / "processed" / "matches.csv")

    assert load_matches(path).equals(matches)


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("result", "A", "result disagrees with the score"),
        ("home_team", None, "missing home_team"),
        ("away_team", "Inter", "plays itself"),
        ("b365_home", 0.9, "odds of 1.0 or lower"),
        ("date", pd.Timestamp("1990-01-01"), "outside their season"),
    ],
)
def test_validate_matches_flags_problems(raw_dir, column, value, message):
    matches = build_matches(raw_dir)
    matches.loc[0, column] = value  # first match: Inter v Milan, 1993-94

    issues = validate_matches(matches)

    assert any(message in issue for issue in issues), issues


def test_validate_matches_flags_incomplete_past_season(raw_dir):
    matches = build_matches(raw_dir)
    matches = matches.drop(index=0)  # 1993-94 now has 1 of its 2 fixtures

    issues = validate_matches(matches)

    assert any("1993-94: 1 matches, expected 2" in issue for issue in issues), issues
