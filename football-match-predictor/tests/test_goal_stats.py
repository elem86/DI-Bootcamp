"""Tests for the historical goals, Over/Under and BTTS statistics (src.goal_stats) and their app section."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.feature_engineering import build_features
from src.goal_stats import (
    AVERAGES,
    EVENTS,
    GOAL_LINES,
    HISTORY_WINDOWS,
    current_spell,
    fixture_goal_stats,
    get_window,
    goal_stats_for_fixture,
    line_key,
    rate,
    team_goal_rows,
    team_sample,
    total_goals_distribution,
)
from tests.test_predict import make_matches as make_league

PROJECT = Path(__file__).resolve().parents[1]
HAS_DATA = (PROCESSED_DATA_DIR / "matches.csv").exists() and (PROCESSED_DATA_DIR / "features.csv").exists()
HAS_MODELS = all((MODELS_DIR / f"{name}.joblib").exists() for name in ("logistic_regression", "random_forest",
                                                                        "xgboost"))

# ---------------------------------------------------------------- hand-made league
# Club X plays eight 2010-11 matches, alternating home and away (all scores from X's side below):
#   k  date        venue  score (X-opponent)  total  BTTS
#   0  1 Sep 2010  home   0-0 A                0     no
#   1  8 Sep       away   0-1 B                1     no
#   2  15 Sep      home   1-1 C                2     yes
#   3  22 Sep      away   1-2 D                3     yes
#   4  29 Sep      home   2-2 A                4     yes
#   5  6 Oct       away   1-3 B                4     yes
#   6  13 Oct      home   4-0 C                4     no
#   7  20 Oct      away   3-0 D                3     no
# Then 2011-12 opens on 20 Aug 2011 with X 1-0 A.

START = pd.Timestamp("2010-09-01")
X_MATCHES = [  # (home, away, home_goals, away_goals)
    ("X", "A", 0, 0), ("B", "X", 1, 0), ("X", "C", 1, 1), ("D", "X", 2, 1),
    ("X", "A", 2, 2), ("B", "X", 3, 1), ("X", "C", 4, 0), ("D", "X", 0, 3),
]
AFTER_2010 = pd.Timestamp("2010-10-27")  # a fixture date after all eight matches
OPENING_2011 = pd.Timestamp("2011-08-20")


def match_row(season_start, date, home, away, home_goals, away_goals):
    if pd.isna(home_goals):
        result = pd.NA
    else:
        result = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")
    return {"season": f"{season_start}-{str(season_start + 1)[-2:]}", "season_start": season_start,
            "date": pd.Timestamp(date), "time": pd.NA, "home_team": home, "away_team": away,
            "home_goals": home_goals, "away_goals": away_goals, "result": result}


def as_matches(rows):
    return pd.DataFrame(rows).astype({"season": "string", "season_start": "int64", "date": "datetime64[ns]",
                                      "time": "string", "home_team": "string", "away_team": "string",
                                      "home_goals": "Int64", "away_goals": "Int64", "result": "string"})


def hand_rows(extra=()):
    rows = [match_row(2010, START + pd.Timedelta(days=7 * k), *match) for k, match in enumerate(X_MATCHES)]
    rows.append(match_row(2011, OPENING_2011, "X", "A", 1, 0))
    return rows + list(extra)


@pytest.fixture(scope="module")
def hand():
    return team_goal_rows(as_matches(hand_rows()))


def same(a: dict, b: dict) -> bool:
    """Statistics are equal (a missing average equals a missing average)."""
    return pd.Series(a).equals(pd.Series(b))


def x_sample(rows, window, date=AFTER_2010, venue="home", season=2010):
    return team_sample(rows, "X", date, window, venue, season)


# ---------------------------------------------------------------- goal lines, BTTS, averages


def test_goal_lines_count_the_total_goals_of_both_clubs(hand):
    stats = x_sample(hand, "last10").stats  # all eight matches; totals 0, 1, 2, 3, 4, 4, 4, 3
    assert stats["matches"] == 8
    expected_over = {0.5: 7, 1.5: 6, 2.5: 5, 3.5: 3}
    for line, over in expected_over.items():
        assert stats[line_key(line, "over")] == over
        assert stats[line_key(line, "under")] == 8 - over
    assert rate(stats, "over_2_5") == 5 / 8 and rate(stats, "under_3_5") == 5 / 8


def test_exactly_two_goals_is_over_1_5_and_under_2_5():
    rows = team_goal_rows(as_matches([match_row(2010, START, "X", "A", 1, 1), match_row(2010, START, "B", "C", 2, 0)]))
    for team, venue in (("X", "home"), ("A", "away"), ("B", "home"), ("C", "away")):
        stats = team_sample(rows, team, START + pd.Timedelta(days=1), "last5", venue, 2010).stats
        assert (stats["over_1_5"], stats["under_2_5"], stats["over_2_5"]) == (1, 1, 0)


def test_btts_needs_both_clubs_to_score(hand):
    stats = x_sample(hand, "last10").stats
    assert (stats["btts_yes"], stats["btts_no"]) == (4, 4)
    assert stats["scored"] == 6 and stats["clean_sheet"] == 3  # 0-0 is a clean sheet but no goal
    # BTTS is a property of the match: both clubs' rows agree, and 0-0 / 1-0 / 0-3 are all "No".
    both = hand.groupby("match_id")["btts"].nunique()
    assert (both == 1).all()
    scores = hand.set_index(["goals_for", "goals_against"])["btts"]
    assert not scores.loc[[(0, 0), (1, 0), (0, 1), (3, 0), (0, 3)]].any()
    assert scores.loc[[(1, 1), (2, 1), (1, 3)]].all()


def test_averages_and_exact_denominators(hand):
    stats = x_sample(hand, "last10").stats
    assert stats["goals_for"] == pytest.approx(12 / 8)
    assert stats["goals_against"] == pytest.approx(9 / 8)
    assert stats["total_goals"] == pytest.approx(21 / 8)
    for key in EVENTS:
        assert rate(stats, key) == stats[key] / 8
    for line in GOAL_LINES:
        assert stats[line_key(line, "over")] + stats[line_key(line, "under")] == stats["matches"]
    assert stats["btts_yes"] + stats["btts_no"] == stats["matches"]


def test_total_goals_distribution(hand):
    distribution = total_goals_distribution(x_sample(hand, "last10").matches, max_goals=4)
    assert distribution.to_dict() == {"0": 1, "1": 1, "2": 1, "3": 2, "4+": 3}
    assert distribution.sum() == 8


def test_line_keys_and_events():
    assert line_key(2.5) == "over_2_5" and line_key(0.5, "under") == "under_0_5"
    with pytest.raises(ValueError):
        line_key(2.5, "exactly")
    assert {line_key(line, side) for line in GOAL_LINES for side in ("over", "under")} <= set(EVENTS)
    with pytest.raises(ValueError, match="not a counted event"):
        rate({"matches": 3, "goals_for": 1.0}, "goals_for")


# ---------------------------------------------------------------- windows, venues and sample sizes


def test_rolling_windows_take_the_latest_matches_in_order(hand):
    last5 = x_sample(hand, "last5")
    assert last5.size == 5 and last5.complete
    assert list(last5.matches["date"]) == [START + pd.Timedelta(days=7 * k) for k in range(3, 8)]
    assert last5.stats["over_2_5"] == 5 and last5.stats["over_3_5"] == 3  # totals 3, 4, 4, 4, 3
    assert x_sample(hand, "last10").matches["date"].is_monotonic_increasing


def test_short_windows_use_the_matches_that_exist_and_say_so(hand):
    last20 = x_sample(hand, "last20")
    assert last20.size == 8 and not last20.complete
    assert last20.sample_text() == "8 of 20 matches"
    assert same(last20.stats, x_sample(hand, "last10").stats)
    early = x_sample(hand, "last10", date=START + pd.Timedelta(days=15))  # after k = 0, 1, 2
    assert early.size == 3 and early.sample_text() == "3 of 10 matches"


def test_venue_windows_use_home_matches_for_the_home_club_and_away_matches_for_the_away_club(hand):
    stats = fixture_goal_stats(hand, "X", "B", AFTER_2010, "venue5", season_start=2010)
    assert set(stats.home.matches["venue"]) == {"home"} and stats.home.size == 4  # 0-0, 1-1, 2-2, 4-0
    assert stats.home.stats["goals_for"] == pytest.approx(7 / 4)
    assert stats.home.stats["goals_against"] == pytest.approx(3 / 4)
    assert stats.home.label == "Last 5 home matches" and stats.home.sample_text() == "4 of 5 matches"
    # B never played away: an empty away sample, even though it has two home matches
    assert stats.away.size == 0 and stats.away.spell_matches == 2
    assert stats.away.empty_reason() == "no away matches yet in its current Serie A spell"
    away_x = team_sample(hand, "X", AFTER_2010, "venue10", "away", 2010)
    assert set(away_x.matches["venue"]) == {"away"} and away_x.stats["goals_for"] == pytest.approx(5 / 4)
    # Non-venue windows mix home and away matches
    assert set(x_sample(hand, "last10").matches["venue"]) == {"home", "away"}


def test_sample_sizes_are_reported_separately_for_each_club(hand):
    stats = fixture_goal_stats(hand, "X", "A", AFTER_2010, "last10", season_start=2010)
    assert (stats.home.size, stats.away.size) == (8, 2)
    table = stats.table()
    sizes = table.groupby("team")["matches"].unique()
    assert list(sizes["X"]) == [8] and list(sizes["A"]) == [2]
    over = table[(table["team"] == "A") & (table["statistic"] == "over_2_5")].iloc[0]
    assert (over["count"], over["matches"], over["value"]) == (1, 2, 0.5)  # 0-0 and 2-2
    assert set(table["statistic"]) == set(EVENTS) | set(AVERAGES)


def test_windows_are_defined_as_requested():
    assert [w.size for w in HISTORY_WINDOWS.values()] == [5, 10, 20, None, 5, 10]
    assert [w.by_venue for w in HISTORY_WINDOWS.values()] == [False] * 4 + [True] * 2
    assert get_window("season").current_season and get_window(HISTORY_WINDOWS["last5"]).size == 5
    with pytest.raises(ValueError, match="Unknown window"):
        get_window("last7")


# ---------------------------------------------------------------- time safety


def test_same_day_matches_and_the_fixture_result_are_never_used(hand):
    on_the_day = [match_row(2010, AFTER_2010, "X", "A", 5, 5),  # the fixture itself, already played
                  match_row(2010, AFTER_2010, "D", "X", 7, 0)]  # another same-day match (stress test)
    with_day = team_goal_rows(as_matches(hand_rows(on_the_day)))
    for window in HISTORY_WINDOWS:
        for team, venue in (("X", "home"), ("A", "away")):
            got = team_sample(with_day, team, AFTER_2010, window, venue, 2010)
            expected = team_sample(hand, team, AFTER_2010, window, venue, 2010)
            assert same(got.stats, expected.stats)
            assert (got.matches["date"] < AFTER_2010).all()
    # The next day they are included
    assert x_sample(with_day, "last10", date=AFTER_2010 + pd.Timedelta(days=1)).size == 10


def test_the_previous_day_is_used(hand):
    day_before = START + pd.Timedelta(days=7 * 7 - 1)  # the day before k = 7
    assert x_sample(hand, "last10", date=day_before).size == 7
    assert x_sample(hand, "last10", date=day_before + pd.Timedelta(days=1)).size == 7  # k = 7 is that day
    assert x_sample(hand, "last10", date=day_before + pd.Timedelta(days=2)).size == 8


def scramble_from(matches, date, seed=0):
    """Random scores (and results) for every match on or after ``date``."""
    rng = np.random.default_rng(seed)
    changed = matches.copy()
    later = changed["date"] >= date
    changed.loc[later, "home_goals"] = rng.integers(0, 6, later.sum())
    changed.loc[later, "away_goals"] = rng.integers(0, 6, later.sum())
    hg, ag = changed["home_goals"], changed["away_goals"]
    changed.loc[later, "result"] = np.where(hg > ag, "H", np.where(ag > hg, "A", "D"))[later.to_numpy()]
    return changed


def league_fixtures(league):
    """Every played fixture of the synthetic league, with its season."""
    return list(league[["home_team", "away_team", "date", "season_start"]].itertuples(index=False))


@pytest.fixture(scope="module")
def league():
    return make_league(realistic=True)


@pytest.fixture(scope="module")
def league_rows(league):
    return team_goal_rows(league)


@pytest.mark.parametrize("window", list(HISTORY_WINDOWS))
def test_future_and_same_day_results_cannot_change_the_statistics(league, league_rows, window):
    for home, away, date, season in league_fixtures(league)[::7]:
        changed = team_goal_rows(scramble_from(league, date, seed=int(date.day_of_year)))
        got = fixture_goal_stats(changed, home, away, date, window, season)
        expected = fixture_goal_stats(league_rows, home, away, date, window, season)
        assert same(got.home.stats, expected.home.stats) and same(got.away.stats, expected.away.stats)
        assert got.table().equals(expected.table())


@pytest.mark.parametrize("window", list(HISTORY_WINDOWS))
def test_historical_fixtures_reproduce_what_was_known_before_the_day(league, league_rows, window):
    """Demo mode: stats from the full data equal stats from only the matches before the fixture date."""
    for home, away, date, season in league_fixtures(league)[::5]:
        known = team_goal_rows(league[league["date"] < date])
        full = fixture_goal_stats(league_rows, home, away, date, window, season)
        morning = fixture_goal_stats(known, home, away, date, window, season)
        for got, expected in ((full.home, morning.home), (full.away, morning.away)):
            pd.testing.assert_frame_equal(got.matches.drop(columns="match_id"),
                                          expected.matches.drop(columns="match_id"))
            assert same(got.stats, expected.stats)


def test_unplayed_fixtures_are_ignored(hand):
    unplayed = match_row(2010, START + pd.Timedelta(days=3), "X", "B", pd.NA, pd.NA)
    rows = team_goal_rows(as_matches(hand_rows([unplayed])))
    assert len(rows) == len(hand)
    assert same(x_sample(rows, "last10").stats, x_sample(hand, "last10").stats)


# ---------------------------------------------------------------- missing history, promotion, seasons


def test_missing_history_is_missing_not_zero(hand):
    first_day = x_sample(hand, "last10", date=START)
    assert first_day.size == 0 and first_day.spell_matches == 0
    stats = first_day.stats
    assert all(math.isnan(stats[key]) for key in AVERAGES)
    assert all(stats[key] == 0 for key in EVENTS)  # no matches to count
    assert all(math.isnan(rate(stats, key)) for key in EVENTS)  # but the share is unknown, not 0%
    assert first_day.empty_reason() == "no earlier Serie A matches in the data"
    unknown_club = team_sample(hand, "Nobody", AFTER_2010, "last5", "away", 2010)
    assert unknown_club.size == 0 and math.isnan(unknown_club.stats["total_goals"])
    assert fixture_goal_stats(hand, "X", "A", AFTER_2010, "last5", 2010).table()["value"].notna().all()


def test_a_club_new_to_serie_a_starts_with_no_history(league_rows):
    # G plays Serie A for the first time in 2002-03 (see tests/test_predict.py)
    first = league_rows[league_rows["team"] == "G"]["date"].min()
    sample = team_sample(league_rows, "G", first, "last10", "home", 2002)
    assert sample.size == 0 and sample.earlier_spell_end is None
    assert team_sample(league_rows, "G", first + pd.Timedelta(days=1), "last10", "home", 2002).size == 1


def test_a_returning_club_does_not_reuse_its_earlier_spell(league_rows):
    # F played 2000-01 and 2001-02, missed 2002-03 and returns in 2003-04
    f_rows = league_rows[league_rows["team"] == "F"]
    first_back = f_rows.loc[f_rows["season_start"] == 2003, "date"].min()
    returning = team_sample(league_rows, "F", first_back, "last20", "home", 2003)
    assert returning.size == 0 and returning.earlier_spell_end == 2001
    assert "back in Serie A after a gap" in returning.empty_reason()
    later = team_sample(league_rows, "F", f_rows["date"].max() + pd.Timedelta(days=1), "last20", "home", 2003)
    assert later.size > 0 and set(later.matches["season_start"]) == {2003}
    # The same rule decides the spells as the feature engineering: no earlier Serie A spell is mixed in
    assert later.spell_matches == len(f_rows[f_rows["season_start"] == 2003])


def test_season_to_date_restarts_but_rolling_windows_carry_across_the_summer(hand):
    opening = x_sample(hand, "season", date=OPENING_2011, season=2011)
    assert opening.size == 0 and opening.empty_reason() == "no 2011-12 matches yet before this date"
    carried = x_sample(hand, "last5", date=OPENING_2011, season=2011)
    assert carried.size == 5 and carried.seasons == [2010]
    after = x_sample(hand, "last5", date=OPENING_2011 + pd.Timedelta(days=7), season=2011)
    assert after.seasons == [2010, 2011] and after.matches["date"].iloc[-1] == OPENING_2011
    season = x_sample(hand, "season", date=OPENING_2011 + pd.Timedelta(days=7), season=2011)
    assert season.size == 1 and season.stats["clean_sheet"] == 1 and season.complete
    assert season.sample_text() == "1 match"


def test_a_continuing_club_keeps_its_spell_across_seasons(league_rows):
    # A plays every season (10 matches each): on 2003-04's opening day its last 20 span two seasons
    first_2003 = league_rows.loc[league_rows["season_start"] == 2003, "date"].min()
    sample = team_sample(league_rows, "A", first_2003, "last20", "home", 2003)
    assert sample.size == 20 and sample.seasons == [2001, 2002]


def test_history_later_than_the_fixture_season_is_rejected(hand):
    x_rows = hand[hand["team"] == "X"]
    with pytest.raises(ValueError, match="later than the fixture's season"):
        current_spell(x_rows, 2009)


def test_matches_the_saved_form_features(league, league_rows):
    """Complete windows reproduce form5_gf, form10_ga, venue_form5_gf, season_gf and history_matches."""
    built = build_features(league)
    checks = [("last5", "form5_gf", "goals_for"), ("last10", "form10_ga", "goals_against"),
              ("venue5", "venue_form5_gf", "goals_for"), ("venue10", "venue_form10_ga", "goals_against"),
              ("season", "season_gf", "goals_for")]
    for row in built.itertuples(index=False):
        for venue, team in (("home", row.home_team), ("away", row.away_team)):
            for window, feature, stat in checks:
                sample = team_sample(league_rows, team, row.date, window, venue, row.season_start)
                expected = getattr(row, f"{venue}_{feature}")
                if pd.isna(expected):
                    assert not sample.complete or (window == "season" and sample.size == 0)
                else:
                    assert sample.complete and sample.stats[stat] == pytest.approx(expected, abs=1e-12)
            assert sample.spell_matches == getattr(row, f"{venue}_history_matches")


# ---------------------------------------------------------------- reproducibility and inputs


def test_reproducible_and_independent_of_row_order(league, league_rows):
    shuffled = team_goal_rows(league.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(shuffled, league_rows)
    pd.testing.assert_frame_equal(team_goal_rows(league), league_rows)
    for home, away, date, season in league_fixtures(league)[::9]:
        a = fixture_goal_stats(league_rows, home, away, date, "last10", season)
        b = fixture_goal_stats(shuffled, home, away, date, "last10", season)
        assert a.table().equals(b.table())


def test_inputs_are_not_modified(league):
    before = league.copy()
    rows = team_goal_rows(league)
    rows_before = rows.copy()
    fixture_goal_stats(rows, "A", "B", league["date"].max(), "venue10", 2003)
    pd.testing.assert_frame_equal(league, before)
    pd.testing.assert_frame_equal(rows, rows_before)


def test_invalid_inputs(hand):
    with pytest.raises(ValueError, match="different"):
        fixture_goal_stats(hand, "X", "X", AFTER_2010, "last5", 2010)
    with pytest.raises(ValueError, match="venue"):
        team_sample(hand, "X", AFTER_2010, "last5", "neutral", 2010)
    with pytest.raises(ValueError, match="date"):
        team_sample(hand, "X", None, "last5", "home", 2010)
    with pytest.raises(ValueError, match="Unknown window"):
        fixture_goal_stats(hand, "X", "A", AFTER_2010, "all_time", 2010)


def test_convenience_wrapper_finds_the_season_from_the_calendar(league, league_rows):
    date = league.loc[league["season_start"] == 2003, "date"].max()
    wrapped = goal_stats_for_fixture(league, "A", "B", date, "season")
    direct = fixture_goal_stats(league_rows, "A", "B", date, "season", 2003)
    assert same(wrapped.home.stats, direct.home.stats) and same(wrapped.away.stats, direct.away.stats)


# ---------------------------------------------------------------- real data and the app


@pytest.mark.skipif(not HAS_DATA, reason="processed data not generated")
def test_real_data_matches_the_saved_features():
    from src.feature_engineering import load_features
    from src.preprocessing import load_matches

    rows = team_goal_rows(load_matches())
    features = load_features()
    assert len(rows) == 2 * len(features)
    sample = features.sample(150, random_state=7)
    sample = pd.concat([sample, features.groupby("season_start").head(1)])  # every opening day
    for row in sample.itertuples(index=False):
        for venue, team in (("home", row.home_team), ("away", row.away_team)):
            for window, feature in (("last10", "form10_gf"), ("venue5", "venue_form5_ga"), ("season", "season_ga")):
                got = team_sample(rows, team, row.date, window, venue, row.season_start)
                expected = getattr(row, f"{venue}_{feature}")
                stat = "goals_for" if feature.endswith("gf") else "goals_against"
                if pd.isna(expected):
                    assert not got.complete or got.size == 0
                else:
                    assert got.stats[stat] == pytest.approx(expected, abs=1e-12)
            assert got.spell_matches == getattr(row, f"{venue}_history_matches")


@pytest.mark.skipif(not (HAS_DATA and HAS_MODELS), reason="processed data or saved models not generated")
def test_app_goals_section_is_separate_and_leaves_predictions_unchanged():
    from streamlit.testing.v1 import AppTest

    from src.predict import fixture_features, load_saved_models, predict_fixture
    from src.preprocessing import load_matches

    at = AppTest.from_file(str(PROJECT / "app.py"), default_timeout=120).run()
    assert not at.exception
    headers = [header.value for header in at.header]
    assert "Goals & BTTS Analysis" in headers
    assert headers.index("Goals & BTTS Analysis") > 0  # beneath the match-outcome predictions

    # The model predictions shown are exactly predict_fixture's
    home, away = at.selectbox(key="home").value, at.selectbox(key="away").value
    date = [d for d in at.date_input if d.key.startswith("date_")][0].value
    expected = predict_fixture(load_saved_models(), fixture_features(load_matches(), home, away, date))
    shown = at.dataframe[0].value
    assert np.allclose(shown["Probability"].to_numpy() / 100, expected.iloc[0].to_numpy())

    selector = at.segmented_control(key="goal_window")
    assert list(selector.options) == [w.label for w in HISTORY_WINDOWS.values()]
    for label in selector.options:
        at.segmented_control(key="goal_window").set_value(label).run()
        assert not at.exception
        text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        assert "not" in text and "probabilit" in text  # the frequencies-are-not-probabilities note

    # Historical demo on the first day in the data: no history, shown as missing, not 0%
    at.segmented_control(key="mode").set_value("Historical demo").run()
    matches = load_matches()
    first = int(matches["season_start"].min())
    at.selectbox(key="season").set_value(first).run()
    first_date = pd.Timestamp(matches["date"].min()).date()
    at.date_input(key=f"date_Historical demo_{first}").set_value(first_date).run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "no earlier Serie A matches in the data" in warnings
