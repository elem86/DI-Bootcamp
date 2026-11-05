"""Tests for src.feature_engineering, using small synthetic leagues."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMNS,
    build_features,
    load_features,
    save_features,
)


def make_matches(rows):
    """Build a matches table from (date, home, away, home_goals, away_goals) tuples."""
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")  # as load_matches()
    # Seasons run August to May, as in Serie A
    df["season_start"] = np.where(df["date"].dt.month >= 7, df["date"].dt.year, df["date"].dt.year - 1)
    df["season_start"] = df["season_start"].astype("int64")
    df["season"] = df["season_start"].map(lambda y: f"{y}-{str(y + 1)[-2:]}").astype("string")
    df["time"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    diff = df["home_goals"] - df["away_goals"]
    df["result"] = np.select([diff > 0, diff < 0], ["H", "A"], default="D")
    df["result"] = df["result"].astype("string")
    for col in ["home_team", "away_team"]:
        df[col] = df[col].astype("string")
    return df


def make_league(teams_by_season, seed=0):
    """Double round robin for each season with random scores.

    Two matches share each date, so same-date leakage would show up.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season_start, teams in teams_by_season.items():
        a, b, c, d = teams
        rounds = [[(a, b), (c, d)], [(a, c), (b, d)], [(a, d), (b, c)]]
        rounds += [[(away, home) for home, away in r] for r in rounds]  # return fixtures
        date = pd.Timestamp(f"{season_start}-09-01")
        for fixtures in rounds:
            for home, away in fixtures:
                rows.append((date, home, away, *rng.integers(0, 4, size=2)))
            date += pd.Timedelta(days=7)
    return make_matches(rows)


LEAGUE = {
    2000: ["A", "B", "C", "D"],
    2001: ["A", "B", "C", "E"],  # D relegated, E promoted
    2002: ["A", "B", "C", "D"],  # D returns after a season away
}


def match(features, date, home):
    rows = features[(features["date"] == pd.Timestamp(date)) & (features["home_team"] == home)]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------------------------------------------------------------- calculations

TEAM_A_RUN = [
    # date,        home, away, hg, ag    -> A's points, goals for, goals against
    ("2000-09-01", "A", "B", 2, 0),  # W 3, 2-0
    ("2000-09-08", "C", "A", 1, 1),  # D 1, 1-1
    ("2000-09-15", "A", "D", 0, 3),  # L 0, 0-3
    ("2000-09-22", "E", "A", 0, 4),  # W 3, 4-0
    ("2000-09-29", "A", "F", 2, 2),  # D 1, 2-2
    ("2000-10-06", "G", "A", 1, 0),  # L 0, 0-1
    ("2000-10-13", "A", "B", 1, 0),  # target match for the checks below
]


def test_form_features_match_hand_calculation():
    features = build_features(make_matches(TEAM_A_RUN))

    sixth = match(features, "2000-10-06", "G")  # A is the away side, 5 matches played
    assert sixth["away_history_matches"] == 5
    assert sixth["away_form5_ppg"] == pytest.approx((3 + 1 + 0 + 3 + 1) / 5)
    assert sixth["away_form5_gf"] == pytest.approx((2 + 1 + 0 + 4 + 2) / 5)
    assert sixth["away_form5_ga"] == pytest.approx((0 + 1 + 3 + 0 + 2) / 5)
    assert sixth["away_form5_gd"] == pytest.approx((2 + 0 - 3 + 4 + 0) / 5)

    seventh = match(features, "2000-10-13", "A")  # window has moved on by one match
    assert seventh["home_form5_ppg"] == pytest.approx((1 + 0 + 3 + 1 + 0) / 5)


def test_venue_form_uses_only_matches_at_that_venue():
    rows = TEAM_A_RUN + [
        ("2000-10-20", "H", "A", 0, 0),
        ("2000-10-27", "A", "C", 3, 1),
        ("2000-11-03", "A", "D", 0, 0),
    ]
    features = build_features(make_matches(rows))

    target = match(features, "2000-11-03", "A")
    # A's previous five home matches: 2-0 W, 0-3 L, 2-2 D, 1-0 W, 3-1 W
    assert target["home_venue_form5_ppg"] == pytest.approx((3 + 0 + 1 + 3 + 3) / 5)
    assert target["home_venue_form5_gf"] == pytest.approx((2 + 0 + 2 + 1 + 3) / 5)


def test_season_to_date_features():
    features = build_features(make_matches(TEAM_A_RUN))

    third = match(features, "2000-09-15", "A")
    assert third["home_season_played"] == 2
    assert third["home_season_ppg"] == pytest.approx((3 + 1) / 2)
    assert third["home_season_gd"] == pytest.approx((2 + 0) / 2)


def test_missing_history_is_explicit():
    features = build_features(make_matches(TEAM_A_RUN))

    first = match(features, "2000-09-01", "A")
    assert first["home_history_matches"] == 0
    assert first["home_season_played"] == 0
    assert pd.isna(first["home_form5_ppg"])
    assert pd.isna(first["home_season_ppg"])  # no matches yet: unknown, not 0

    fifth = match(features, "2000-09-29", "A")  # only 4 earlier matches
    assert fifth["home_history_matches"] == 4
    assert pd.isna(fifth["home_form5_ppg"])  # window not full: missing, not a 4-match average
    assert pd.isna(fifth["home_form10_ppg"])


# ---------------------------------------------------------------- leakage

def test_match_never_uses_its_own_score():
    original = make_matches(TEAM_A_RUN)
    changed = original.copy()
    last = changed.index[-1]
    changed.loc[last, ["home_goals", "away_goals", "result"]] = [9, 0, "H"]

    before = build_features(original).iloc[-1]
    after = build_features(changed).iloc[-1]

    pd.testing.assert_series_equal(before[FEATURE_COLUMNS], after[FEATURE_COLUMNS])
    assert after["home_goals"] == 9  # only the target changed


@pytest.mark.parametrize("cutoff", ["2000-09-22", "2001-09-01", "2002-09-22"])
def test_features_ignore_results_from_the_same_date_or_later(cutoff):
    original = make_league(LEAGUE, seed=1)
    cutoff = pd.Timestamp(cutoff)
    # Replace every result from the cutoff date onwards with different scores
    changed = original.copy()
    later = changed["date"] >= cutoff
    changed.loc[later, "home_goals"] = (changed.loc[later, "home_goals"] + 3).astype("Int64")
    changed.loc[later, "result"] = "H"

    before = build_features(original)
    after = build_features(changed)

    up_to_cutoff = before["date"] <= cutoff  # includes the other match on the cutoff date
    pd.testing.assert_frame_equal(
        before.loc[up_to_cutoff, FEATURE_COLUMNS], after.loc[up_to_cutoff, FEATURE_COLUMNS]
    )
    # Sanity check: later matches really are affected, so the test can fail
    assert not before.loc[~up_to_cutoff, FEATURE_COLUMNS].equals(after.loc[~up_to_cutoff, FEATURE_COLUMNS])


def test_two_matches_for_one_team_on_the_same_date_do_not_see_each_other():
    rows = TEAM_A_RUN[:5] + [
        ("2000-10-06", "A", "X", 5, 0),
        ("2000-10-06", "Y", "A", 0, 5),  # same date: must not see the 5-0 above
    ]
    features = build_features(make_matches(rows))

    first, second = match(features, "2000-10-06", "A"), match(features, "2000-10-06", "Y")
    assert first["home_history_matches"] == second["away_history_matches"] == 5
    assert first["home_form5_ppg"] == second["away_form5_ppg"] == pytest.approx(8 / 5)


def test_input_order_does_not_matter():
    league = make_league(LEAGUE, seed=2)
    shuffled = league.sample(frac=1, random_state=0)

    pd.testing.assert_frame_equal(build_features(league), build_features(shuffled))


def test_matches_are_in_chronological_order():
    features = build_features(make_league(LEAGUE, seed=3).sample(frac=1, random_state=1))

    assert features["date"].is_monotonic_increasing
    assert list(features["match_id"]) == list(range(len(features)))


# ---------------------------------------------------------------- seasons

def test_continuing_team_carries_form_into_the_new_season():
    features = build_features(make_league(LEAGUE, seed=4))
    a_rows = features[(features["home_team"] == "A") | (features["away_team"] == "A")]
    opener = a_rows[a_rows["season_start"] == 2001].iloc[0]
    side = "home" if opener["home_team"] == "A" else "away"

    last_season = a_rows[a_rows["season_start"] == 2000]
    a_points = np.where(
        last_season["home_team"] == "A",
        last_season["result"].map({"H": 3, "D": 1, "A": 0}),
        last_season["result"].map({"A": 3, "D": 1, "H": 0}),
    ).astype(float)

    assert opener[f"{side}_history_matches"] == 6
    assert opener[f"{side}_form5_ppg"] == pytest.approx(a_points[-5:].mean())  # end of last season
    assert opener[f"{side}_season_played"] == 0  # season stats restart
    assert pd.isna(opener[f"{side}_season_ppg"])
    assert opener[f"{side}_prev_season_ppg"] == pytest.approx(a_points.mean())
    assert opener[f"{side}_promoted"] == False  # noqa: E712 (nullable boolean)


@pytest.mark.parametrize("team, season_start", [("E", 2001), ("D", 2002)])
def test_promoted_or_returning_team_starts_without_history(team, season_start):
    features = build_features(make_league(LEAGUE, seed=5))
    rows = features[(features["home_team"] == team) | (features["away_team"] == team)]
    opener = rows[rows["season_start"] == season_start].iloc[0]
    side = "home" if opener["home_team"] == team else "away"

    assert opener[f"{side}_promoted"] == True  # noqa: E712
    assert opener[f"{side}_history_matches"] == 0  # D's 2000 matches are not carried over
    assert pd.isna(opener[f"{side}_form5_ppg"])
    assert pd.isna(opener[f"{side}_prev_season_ppg"])


def test_promoted_is_unknown_in_the_first_season():
    features = build_features(make_league(LEAGUE, seed=6))
    first_season = features[features["season_start"] == 2000]

    assert first_season["home_promoted"].isna().all()
    assert first_season["away_promoted"].isna().all()
    assert features.loc[features["season_start"] > 2000, "home_promoted"].notna().all()


# ---------------------------------------------------------------- output

def test_targets_are_the_original_results():
    league = make_league(LEAGUE, seed=7)
    features = build_features(league)

    keys = ["date", "home_team", "away_team"]
    merged = features.merge(league, on=keys, suffixes=("", "_original"))
    assert len(merged) == len(league)
    for col in TARGET_COLUMNS:
        assert (merged[col] == merged[f"{col}_original"]).all()


def test_no_target_columns_among_features():
    assert not set(TARGET_COLUMNS) & set(FEATURE_COLUMNS)
    assert all(col.startswith(("home_", "away_")) for col in FEATURE_COLUMNS)


def test_save_and_load_round_trip(tmp_path):
    features = build_features(make_league(LEAGUE, seed=8))
    path = save_features(features, tmp_path / "features.csv")

    loaded = load_features(path)
    assert list(loaded.columns) == OUTPUT_COLUMNS
    pd.testing.assert_frame_equal(loaded, features)
