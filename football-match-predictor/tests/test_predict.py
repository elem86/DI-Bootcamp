"""Tests for future-fixture inference (src.predict) and the Streamlit app."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.evaluate import OUTCOMES
from src.feature_engineering import FEATURE_COLUMNS, build_features
from src.predict import (
    FIXTURE_COLUMNS,
    FixtureError,
    ModelFilesMissing,
    SavedModel,
    check_probabilities,
    current_season,
    fair_odds,
    fixture_features,
    history_summary,
    latest_completed_date,
    load_saved_model,
    most_likely,
    outcome_table,
    played_fixture,
    plays_twice_on_one_day,
    predict_fixture,
    resolve_season,
    season_windows,
    simple_average,
)

PROJECT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- synthetic league
# 2000-01 and 2001-02: clubs A-F. 2002-03: F is relegated, G arrives for the first time.
# 2003-04 (the latest, incomplete season): F returns after a gap (promoted), G stays.
# Every matchday has three matches on the same date.
# realistic=True: a proper round-robin, every club plays once per matchday (as in real leagues).
# realistic=False: shuffled pairs, so a club can play twice on one day (a stress test).

SEASON_CLUBS = {2000: "ABCDEF", 2001: "ABCDEF", 2002: "ABCDEG", 2003: "ABCDFG"}
PLAYED_IN_LATEST = 12


def round_robin(clubs):
    """Double round-robin by the circle method: one list of (home, away) pairs per matchday."""
    clubs = list(clubs)
    rounds = []
    for _ in range(len(clubs) - 1):
        rounds.append([(clubs[i], clubs[-1 - i]) for i in range(len(clubs) // 2)])
        clubs = [clubs[0], clubs[-1], *clubs[1:-1]]
    return rounds + [[(a, h) for h, a in matchday] for matchday in rounds]


def make_matches(seed=0, realistic=False):
    rng = np.random.default_rng(seed)
    rows = []
    for season, clubs in SEASON_CLUBS.items():
        if realistic:
            pairs = [pair for matchday in round_robin(clubs) for pair in matchday]
        else:
            pairs = [(h, a) for h in clubs for a in clubs if h != a]
            rng.shuffle(pairs)
        if season == max(SEASON_CLUBS):
            pairs = pairs[:PLAYED_IN_LATEST]
        start = pd.Timestamp(f"{season}-08-26")
        for k, (home, away) in enumerate(pairs):
            date = start + pd.Timedelta(days=7 * (k // 3))  # three matches per matchday
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 3))
            result = "H" if hg > ag else ("A" if ag > hg else "D")
            rows.append({"season": f"{season}-{str(season + 1)[-2:]}", "season_start": season, "date": date,
                         "time": pd.NA, "home_team": home, "away_team": away, "home_goals": hg, "away_goals": ag,
                         "result": result})
    matches = pd.DataFrame(rows)
    return matches.astype({"season": "string", "season_start": "int64", "date": "datetime64[ns]",
                           "time": "string", "home_team": "string", "away_team": "string",
                           "home_goals": "Int64", "away_goals": "Int64", "result": "string"})


@pytest.fixture(scope="module")
def matches():
    return make_matches(realistic=True)


@pytest.fixture(scope="module")
def built(matches):
    return build_features(matches)


def saved_row(built, home, away, date):
    rows = built[(built["home_team"] == home) & (built["away_team"] == away) & (built["date"] == date)]
    assert len(rows) == 1
    return rows[FEATURE_COLUMNS].reset_index(drop=True)


def features_of(matches, home, away, date, **kwargs):
    return fixture_features(matches, home, away, date, **kwargs)[FEATURE_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------- parity with build_features


@pytest.mark.parametrize("realistic", [True, False])
@pytest.mark.parametrize("restrict_history", [True, False])
def test_every_played_match_is_reproduced_exactly(realistic, restrict_history):
    """Rebuilding any played match as a 'fixture' gives its saved features, including NaN and <NA>.

    The default (fast) path is checked on every match; the slower full-history path on every third.
    """
    league = make_matches(realistic=realistic)
    saved = build_features(league)
    step = 1 if restrict_history else 3
    for row in list(saved.itertuples(index=False))[::step]:
        got = features_of(league, row.home_team, row.away_team, row.date, restrict_history=restrict_history)
        pd.testing.assert_frame_equal(got, saved_row(saved, row.home_team, row.away_team, row.date))


def test_fast_path_is_used_for_real_schedules_and_skipped_for_double_matchdays():
    """With one match per club per day the fast path applies; otherwise the full history is used."""
    realistic, shuffled = make_matches(realistic=True), make_matches(realistic=False)
    assert not any(plays_twice_on_one_day(realistic, [club]) for club in "ABCDEFG")
    assert plays_twice_on_one_day(shuffled, ["E"])  # E plays three times on 23 Sep 2000

    date = pd.Timestamp("2000-10-28")
    pd.testing.assert_frame_equal(fixture_features(shuffled, "E", "F", date),
                                  fixture_features(shuffled, "E", "F", date, restrict_history=False))


def test_output_columns_match_the_saved_feature_table(matches, built):
    result = fixture_features(matches, "A", "B", "2003-12-30")
    assert list(result.columns) == FIXTURE_COLUMNS + FEATURE_COLUMNS
    assert dict(result[FEATURE_COLUMNS].dtypes) == dict(built[FEATURE_COLUMNS].dtypes)
    assert len(result) == 1


def test_input_matches_are_not_modified(matches):
    before = matches.copy()
    fixture_features(matches, "A", "G", "2003-12-30")
    pd.testing.assert_frame_equal(matches, before)


# ---------------------------------------------------------------- leakage


def test_same_day_matches_and_the_fixture_result_are_never_used(matches):
    date = matches.loc[matches["season_start"] == 2001, "date"].iloc[9]  # a matchday with three matches
    day = matches[matches["date"] == date]
    target = day.iloc[0]
    original = features_of(matches, target.home_team, target.away_team, date)

    changed = matches.copy()
    same_day = changed["date"] == date
    changed.loc[same_day, ["home_goals", "away_goals"]] = 9
    changed.loc[same_day, "result"] = "D"
    later = changed["date"] > date
    changed.loc[later, ["home_goals", "away_goals"]] = 0
    changed.loc[later, "result"] = "D"

    pd.testing.assert_frame_equal(features_of(changed, target.home_team, target.away_team, date), original)


def test_the_previous_matchday_is_used(matches):
    date = matches.loc[matches["season_start"] == 2001, "date"].iloc[9]
    target = matches[matches["date"] == date].iloc[0]
    earlier = matches[(matches["date"] < date) & (matches["home_team"] == target.home_team)].index[-1]
    changed = matches.copy()
    changed.loc[earlier, ["home_goals", "away_goals"]] = [9, 0]
    changed.loc[earlier, "result"] = "H"
    assert not features_of(changed, target.home_team, target.away_team, date).equals(
        features_of(matches, target.home_team, target.away_team, date))


# ---------------------------------------------------------------- seasons, promotion, missing history


def test_season_transition_restarts_season_stats_but_carries_form(matches):
    first_day = matches.loc[matches["season_start"] == 2001, "date"].min()
    row = fixture_features(matches, "A", "B", first_day).iloc[0]
    assert row["season_start"] == 2001
    assert row["home_season_played"] == 0
    assert pd.isna(row["home_season_ppg"])

    last_season = matches[(matches["season_start"] == 2000)]
    a_rows = last_season[(last_season["home_team"] == "A") | (last_season["away_team"] == "A")].sort_values("date")
    points = [3 if (r.home_team == "A") == (r.result == "H") and r.result != "D" else (1 if r.result == "D" else 0)
              for r in a_rows.itertuples()]
    assert row["home_form5_ppg"] == pytest.approx(np.mean(points[-5:]))  # form carried over the summer
    assert row["home_prev_season_ppg"] == pytest.approx(np.mean(points))
    assert row["home_promoted"] is False or row["home_promoted"] == False  # noqa: E712


def test_a_club_playing_its_first_serie_a_match(matches):
    first_day = matches.loc[matches["season_start"] == 2002, "date"].min()
    row = fixture_features(matches, "G", "A", first_day).iloc[0]
    assert row["home_history_matches"] == 0
    assert row["home_promoted"] == True  # noqa: E712
    assert pd.isna(row["home_prev_season_ppg"]) and pd.isna(row["home_form5_ppg"])
    assert row["away_history_matches"] > 0


def test_a_promoted_club_returning_after_a_gap_starts_a_new_spell(matches):
    first_day = matches.loc[matches["season_start"] == 2003, "date"].min()
    row = fixture_features(matches, "F", "A", first_day).iloc[0]
    assert row["home_history_matches"] == 0  # its 2000-02 matches belong to an earlier spell
    assert row["home_promoted"] == True  # noqa: E712
    assert pd.isna(row["home_prev_season_gd"])
    assert pd.isna(row["home_form10_ppg"]) and pd.isna(row["home_venue_form5_ppg"])


def test_first_season_in_the_data_has_unknown_promotion(matches):
    first_day = matches["date"].min()
    row = fixture_features(matches, "A", "B", first_day).iloc[0]
    assert pd.isna(row["home_promoted"]) and pd.isna(row["away_promoted"])
    assert row["home_history_matches"] == 0


# ---------------------------------------------------------------- future fixtures


def test_future_fixture_absent_from_the_data(matches):
    last = latest_completed_date(matches)
    date = last + pd.Timedelta(days=10)
    assert played_fixture(matches, "A", "G", date) is None
    row = fixture_features(matches, "A", "G", date).iloc[0]

    season = matches[matches["season_start"] == 2003]
    a_season = season[(season["home_team"] == "A") | (season["away_team"] == "A")]
    assert row["season_start"] == 2003
    assert row["home_season_played"] == len(a_season)  # every 2003-04 match so far is used
    a_all = matches[(matches["home_team"] == "A") | (matches["away_team"] == "A")]
    assert row["home_history_matches"] == len(a_all)  # A has never left the league
    goals = [(r.home_goals if r.home_team == "A" else r.away_goals) for r in a_all.sort_values("date").itertuples()]
    assert row["home_form5_gf"] == pytest.approx(np.mean(goals[-5:]))


def test_future_fixture_gives_the_same_result_with_full_history(matches):
    date = latest_completed_date(matches) + pd.Timedelta(days=30)
    pd.testing.assert_frame_equal(fixture_features(matches, "F", "B", date),
                                  fixture_features(matches, "F", "B", date, restrict_history=False))


def test_seasons_and_teams_come_from_the_data(matches):
    windows = season_windows(matches)
    assert sorted(windows) == [2000, 2001, 2002, 2003]
    latest = current_season(matches)
    assert latest.season_start == 2003 and not latest.complete
    assert latest.teams == tuple("ABCDFG")
    assert latest.last_date == pd.Timestamp("2004-06-30")
    assert windows[2001].complete and windows[2001].last_date == matches.loc[
        matches["season_start"] == 2001, "date"].max()
    assert resolve_season(matches, "2003-12-30").season_start == 2003
    summary = history_summary(matches, "2003-12-30")
    assert summary["matches"] == len(matches) and summary["last_date"] == latest_completed_date(matches)


# ---------------------------------------------------------------- invalid input


@pytest.mark.parametrize(
    "home, away, date, message",
    [
        ("A", "A", "2003-12-30", "must be different"),
        ("A", "", "2003-12-30", "away team"),
        (None, "B", "2003-12-30", "home team"),
        ("A", "E", "2003-12-30", "E did not play"),  # relegated after 2002-03
        ("A", "Z", "2003-12-30", "Z did not play"),
        ("A", "B", "2004-08-20", "after the 2003-04 season"),
        ("A", "B", "2001-07-01", "between two seasons"),
        ("A", "B", "1999-01-01", "before the first match"),
        ("A", "B", "not a date", "not a valid date"),
        ("A", "B", None, "date is required"),
    ],
)
def test_invalid_fixtures_raise_a_readable_error(matches, home, away, date, message):
    with pytest.raises(FixtureError, match=message):
        fixture_features(matches, home, away, date)


# ---------------------------------------------------------------- models and probabilities


class RecordingModel:
    """Stands in for a fitted pipeline: records the columns it receives."""

    classes_ = np.array(["A", "D", "H"])

    def __init__(self, probs=(0.2, 0.3, 0.5)):
        self.probs = np.array([probs])
        self.columns = None

    def predict_proba(self, X):
        self.columns = list(X.columns)
        return self.probs


def test_predictions_use_each_models_own_column_order(matches):
    row = fixture_features(matches, "A", "B", "2003-12-30")
    order = ["away_form10_ppg", "home_form10_ppg", "home_promoted"]
    model = RecordingModel()
    probs = predict_fixture({"Stub": SavedModel("stub", "Stub", model, order, {})}, row)

    assert model.columns == order
    assert list(probs.columns) == OUTCOMES
    assert probs.loc["Stub"].tolist() == pytest.approx([0.5, 0.3, 0.2])  # reordered from classes A, D, H


def test_invalid_model_probabilities_are_rejected(matches):
    row = fixture_features(matches, "A", "B", "2003-12-30")
    bad = SavedModel("stub", "Stub", RecordingModel((0.2, 0.3, 0.7)), ["home_form10_ppg"], {})
    with pytest.raises(ValueError, match="sum to"):
        predict_fixture({"Stub": bad}, row)
    with pytest.raises(ValueError, match="invalid"):
        check_probabilities(pd.Series({"H": 1.2, "D": -0.1, "A": -0.1}))


def test_missing_model_files_give_instructions(tmp_path):
    with pytest.raises(ModelFilesMissing, match="python -m src.train"):
        load_saved_model("xgboost", tmp_path)
    with pytest.raises(ValueError, match="Unknown model"):
        load_saved_model("neural_net", tmp_path)


def test_fair_odds_and_outcome_table():
    assert fair_odds(0.5) == pytest.approx(2.0)
    assert fair_odds(0.25) == pytest.approx(4.0)
    assert math.isinf(fair_odds(0.0))
    with pytest.raises(ValueError):
        fair_odds(1.5)

    probs = pd.Series({"H": 0.2, "D": 0.45, "A": 0.35})
    table = outcome_table(probs)
    assert table["Outcome"].tolist() == ["Home win", "Draw", "Away win"]
    assert table["Fair odds"].tolist() == pytest.approx([5.0, 1 / 0.45, 1 / 0.35])
    assert table["Most likely"].tolist() == [False, True, False]
    assert most_likely(probs) == "D"


def test_simple_average_is_the_unweighted_mean():
    probs = pd.DataFrame({"H": [0.5, 0.3], "D": [0.3, 0.3], "A": [0.2, 0.4]}, index=["m1", "m2"])
    assert simple_average(probs).tolist() == pytest.approx([0.4, 0.3, 0.3])


# ---------------------------------------------------------------- real data and saved models (if generated)

HAS_DATA = (PROCESSED_DATA_DIR / "features.csv").exists() and (PROCESSED_DATA_DIR / "matches.csv").exists()
HAS_MODELS = all((MODELS_DIR / f"{name}.joblib").exists() for name in ("logistic_regression", "random_forest",
                                                                       "xgboost"))


@pytest.mark.skipif(not HAS_DATA, reason="processed data not generated")
def test_real_historical_feature_parity():
    """Sampled real matches, plus each season's opening day and promoted clubs' first matches."""
    from src.feature_engineering import load_features
    from src.preprocessing import load_matches

    real_matches, saved = load_matches(), load_features()
    picks = set(np.random.default_rng(7).choice(len(saved), 40, replace=False))
    picks |= set(saved.groupby("season_start")["date"].idxmin())
    picks |= set(saved.index[(saved["home_promoted"] == True) & (saved["home_history_matches"] == 0)][:5])  # noqa: E712
    picks |= set(saved.index[(saved["away_promoted"] == True) & (saved["away_history_matches"] == 0)][:5])  # noqa: E712
    for i in sorted(picks):
        row = saved.loc[i]
        got = features_of(real_matches, row.home_team, row.away_team, row.date)
        pd.testing.assert_frame_equal(got, saved.loc[[i], FEATURE_COLUMNS].reset_index(drop=True))


@pytest.mark.skipif(not (HAS_DATA and HAS_MODELS), reason="processed data or saved models not generated")
def test_saved_models_predict_a_future_fixture():
    from src.predict import load_saved_models
    from src.preprocessing import load_matches

    real_matches = load_matches()
    season = current_season(real_matches)
    date = latest_completed_date(real_matches) + pd.Timedelta(days=7)
    home, away = season.teams[0], season.teams[-1]
    probs = predict_fixture(load_saved_models(), fixture_features(real_matches, home, away, date))

    assert list(probs.index) == ["Logistic Regression", "Random Forest", "XGBoost"]
    assert ((probs >= 0) & (probs <= 1)).all().all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.skipif(not (HAS_DATA and HAS_MODELS), reason="processed data or saved models not generated")
def test_streamlit_app_starts_and_switches_modes():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Serie A Match Predictor"
    home = at.selectbox(key="home").value
    assert home not in at.selectbox(key="away").options  # a club cannot play itself
    assert len(at.dataframe) >= 3  # one table per model

    at.segmented_control(key="mode").set_value("Historical demo").run()
    assert not at.exception
    season = at.selectbox(key="season")
    assert season.index == 0 and season.options[0] == max(season.options)  # newest season first
