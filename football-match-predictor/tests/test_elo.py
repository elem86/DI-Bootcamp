"""Tests for the Elo ratings (src.elo) and the Elo walk-forward experiment (src.elo_experiment)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.elo_experiment as experiment_module
from src.backtest import PREDICTION_COLUMNS, common_fixtures
from src.bookmaker import BOOKMAKER_LABEL
from src.elo import (
    DEFAULT_SETTINGS,
    ELO_COLUMNS,
    ELO_FEATURE_COLUMNS,
    EloSettings,
    add_elo_features,
    elo_features,
    end_of_season_ratings,
    expected_score,
    rating_change,
    ratings_before,
    regress,
    run_elo,
    update_ratings,
)
from src.elo_experiment import (
    CONTROL_FEATURE_SETS,
    ELO_FEATURE_SETS,
    accuracy_bootstrap,
    check_elo_is_past_only,
    check_experiment_seasons,
    comparison_table,
    decision,
    holm_adjust,
    model_pairs,
    promoted_flags,
    run_elo_backtest,
    subgroup_metrics,
)
from src.evaluate import OUTCOMES
from src.feature_engineering import FEATURE_COLUMNS, build_features
from src.train import FEATURE_SETS, check_feature_columns
from tests.test_backtest import make_matches as make_odds
from tests.test_predict import make_matches as make_league
from tests.test_train import make_features

PROJECT = Path(__file__).resolve().parents[1]
FEATURES_FILE = PROJECT / "data" / "processed" / "features.csv"


def manual_expected(home, away, home_advantage=60.0):
    """E_home written out independently of src.elo."""
    return 1 / (1 + 10 ** ((away - home - home_advantage) / 400))


def fixtures(rows):
    """Minimal match table from (season, date, home, away, result) tuples."""
    frame = pd.DataFrame(rows, columns=["season_start", "date", "home_team", "away_team", "result"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["result"] = frame["result"].astype("string")
    return frame


# ---------------------------------------------------------------- formulas


def test_expected_score_formula():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1500, 1500, home_advantage=60) == pytest.approx(1 / (1 + 10 ** (-60 / 400)))
    assert expected_score(1500, 1500, home_advantage=60) == pytest.approx(0.585499, abs=1e-6)
    assert expected_score(1900, 1500) == pytest.approx(1 / 1.1)  # 400 points = 10:1 odds
    # the two sides' expectations add up to 1
    assert expected_score(1620, 1480, 60) + expected_score(1480, 1620 + 60) == pytest.approx(1)


@pytest.mark.parametrize("score, change", [(1.0, 8.290026), (0.5, -1.709974), (0.0, -11.709974)])
def test_win_draw_and_loss_updates(score, change):
    """Equal ratings, home advantage 60: E_home = 0.585499, so K * (S - E) with K = 20."""
    home, away = update_ratings(1500.0, 1500.0, score)
    assert home - 1500 == pytest.approx(change, abs=1e-6)
    assert away - 1500 == pytest.approx(-change, abs=1e-6)


def test_rating_points_are_conserved_within_a_match():
    for home, away, score in [(1612.3, 1433.1, 1.0), (1388.0, 1702.5, 0.5), (1500.0, 1499.0, 0.0)]:
        new_home, new_away = update_ratings(home, away, score)
        assert new_home + new_away == pytest.approx(home + away)


def test_home_advantage():
    neutral = EloSettings(home_advantage=0)
    # at a neutral venue a win is worth the same to either side
    assert rating_change(1500, 1500, 1.0, neutral) == pytest.approx(-rating_change(1500, 1500, 0.0, neutral))
    # with home advantage a home win gains less than an away win, and a draw costs the home side
    assert rating_change(1500, 1500, 1.0) < -rating_change(1500, 1500, 0.0)
    assert rating_change(1500, 1500, 0.5) < 0
    # 60 points of home advantage are exactly worth 60 rating points
    assert expected_score(1500, 1560, 60) == pytest.approx(0.5)


def test_regression_toward_the_mean():
    assert regress(1600) == pytest.approx(1570)  # 30% of the 100-point gap removed
    assert regress(1400) == pytest.approx(1430)
    assert regress(1500) == 1500
    assert regress(1600, EloSettings(season_regression=0)) == 1600
    assert regress(1600, EloSettings(season_regression=1)) == 1500


def test_default_settings_are_the_predetermined_values():
    assert DEFAULT_SETTINGS.as_dict() == {"initial_rating": 1500.0, "k_factor": 20.0, "home_advantage": 60.0,
                                          "season_regression": 0.30}
    with pytest.raises(ValueError):
        EloSettings(k_factor=0)
    with pytest.raises(ValueError):
        EloSettings(season_regression=1.5)


# ---------------------------------------------------------------- a hand-calculated league


def hand_league():
    """2000-01: A B C D. 2001-02: D relegated, E promoted. 2002-03: C relegated, D returns.

    Matches are listed out of date order on purpose.
    """
    return fixtures([
        (2000, "2000-09-10", "A", "C", "A"),
        (2000, "2000-09-03", "A", "B", "H"),
        (2000, "2000-09-03", "C", "D", "D"),
        (2001, "2001-09-02", "E", "A", "H"),
        (2001, "2001-09-02", "B", "C", "A"),
        (2002, "2002-09-01", "D", "E", "H"),
        (2002, "2002-09-01", "A", "B", "D"),
    ])


def hand_calculation():
    """Every rating of hand_league worked out step by step without src.elo."""
    K, mean, keep = 20, 1500.0, 0.7
    r = {team: 1500.0 for team in "ABCD"}
    pre = {}
    # 2000-09-03: both matches rated before either is applied
    pre[1] = (r["A"], r["B"])
    pre[2] = (r["C"], r["D"])
    d1 = K * (1 - manual_expected(r["A"], r["B"]))
    d2 = K * (0.5 - manual_expected(r["C"], r["D"]))
    r["A"] += d1
    r["B"] -= d1
    r["C"] += d2
    r["D"] -= d2
    # 2000-09-10: A v C, away win
    pre[0] = (r["A"], r["C"])
    d0 = K * (0 - manual_expected(r["A"], r["C"]))
    r["A"] += d0
    r["C"] -= d0
    final_2000 = dict(r)
    # summer 2001: everyone regresses; D leaves, E enters at D's regressed rating
    r = {t: mean + keep * (v - mean) for t, v in r.items()}
    entry_2001 = r.pop("D")
    r["E"] = entry_2001
    pre[3] = (r["E"], r["A"])
    pre[4] = (r["B"], r["C"])
    d3 = K * (1 - manual_expected(r["E"], r["A"]))
    d4 = K * (0 - manual_expected(r["B"], r["C"]))
    r["E"] += d3
    r["A"] -= d3
    r["B"] += d4
    r["C"] -= d4
    # summer 2002: C leaves; D returns at C's regressed rating, not its own old rating
    r = {t: mean + keep * (v - mean) for t, v in r.items()}
    entry_2002 = r.pop("C")
    r["D"] = entry_2002
    pre[5] = (r["D"], r["E"])
    pre[6] = (r["A"], r["B"])
    return pre, final_2000, entry_2001, entry_2002


def test_pre_match_ratings_match_a_hand_calculation():
    pre_match = elo_features(hand_league())
    expected, *_ = hand_calculation()
    for index, (home, away) in expected.items():
        assert pre_match.loc[index, "home_elo"] == pytest.approx(home, abs=1e-9), index
        assert pre_match.loc[index, "away_elo"] == pytest.approx(away, abs=1e-9), index
        assert pre_match.loc[index, "elo_difference"] == pytest.approx(home - away, abs=1e-9)
        assert pre_match.loc[index, "home_elo_expected"] == pytest.approx(manual_expected(home, away))
    # spot checks with literal numbers: the first matchday
    assert pre_match.loc[1, ["home_elo", "away_elo"]].tolist() == [1500.0, 1500.0]
    assert pre_match.loc[0, "home_elo"] == pytest.approx(1508.290026, abs=1e-6)
    assert pre_match.loc[0, "away_elo"] == pytest.approx(1498.290026, abs=1e-6)


def test_season_transitions_promoted_and_returning_clubs():
    _, history, transitions = run_elo(hand_league())
    pre, final_2000, entry_2001, entry_2002 = hand_calculation()

    t = transitions.set_index("season_start")
    assert t.loc[2000, "first_season"] and t.loc[2000, "entry_rating"] == 1500
    assert t.loc[2001, "leavers"] == ["D"] and t.loc[2001, "entrants"] == ["E"]
    assert t.loc[2001, "entry_rating"] == pytest.approx(1500 + 0.7 * (final_2000["D"] - 1500))
    assert t.loc[2001, "entry_rating"] == pytest.approx(entry_2001)
    # D returns in 2002-03 at the rating of the club it replaces (C), not its own 2000-01 rating
    assert t.loc[2002, "leavers"] == ["C"] and t.loc[2002, "entrants"] == ["D"]
    d_return = history[(history["team"] == "D") & (history["season_start"] == 2002)]["rating_before"].iloc[0]
    assert d_return == pytest.approx(entry_2002)
    assert d_return != pytest.approx(1500 + 0.7 * (final_2000["D"] - 1500))
    # continuing clubs carry 70% of their gap to 1500 into the new season
    a_2001 = history[(history["team"] == "A") & (history["season_start"] == 2001)]["rating_before"].iloc[0]
    assert a_2001 == pytest.approx(1500 + 0.7 * (final_2000["A"] - 1500))


def test_replacing_relegated_clubs_keeps_the_league_average():
    league = make_league(realistic=True)
    _, _, transitions = run_elo(league)
    # equal numbers of clubs leave and enter every summer, so the mean stays at 1500
    assert transitions["mean_rating"].to_numpy() == pytest.approx(1500.0)


def test_first_season_clubs_start_at_the_initial_rating():
    league = make_league(realistic=True)
    pre_match = elo_features(league, EloSettings(initial_rating=1000))
    first_day = league["date"] == league["date"].min()
    assert (pre_match.loc[first_day, ["home_elo", "away_elo"]] == 1000).all().all()


# ---------------------------------------------------------------- leakage


def test_matches_on_the_same_date_never_see_each_others_results():
    base = hand_league()
    flipped = base.copy()
    flipped.loc[2, "result"] = "H"  # C v D on 2000-09-03 now a home win
    a, b = elo_features(base), elo_features(flipped)
    # A v B on the same date is unaffected; the later A v C is affected through C
    pd.testing.assert_series_equal(a.loc[1], b.loc[1])
    assert a.loc[0, "away_elo"] != b.loc[0, "away_elo"]


def test_a_club_playing_twice_on_one_date_uses_its_pre_date_rating_both_times():
    league = fixtures([
        (2000, "2000-09-01", "A", "B", "H"),
        (2000, "2000-09-01", "C", "A", "H"),
        (2000, "2000-09-08", "A", "C", "D"),
    ])
    pre_match = elo_features(league)
    assert pre_match.loc[0, "home_elo"] == 1500 and pre_match.loc[1, "away_elo"] == 1500
    win = 20 * (1 - manual_expected(1500, 1500))
    assert pre_match.loc[2, "home_elo"] == pytest.approx(1500 + win - win)  # both changes from 1500, then added
    assert pre_match.loc[2, "away_elo"] == pytest.approx(1500 + win)


@pytest.mark.parametrize("realistic", [True, False])
def test_row_order_does_not_matter(realistic):
    league = make_league(realistic=realistic)
    shuffled = league.sample(frac=1, random_state=7)
    pd.testing.assert_frame_equal(elo_features(shuffled).loc[league.index], elo_features(league))


def test_future_results_cannot_change_past_ratings():
    league = make_league(realistic=True)
    cutoff = pd.Timestamp("2001-12-01")
    changed = league.copy()
    later = changed["date"] >= cutoff
    changed.loc[later, "result"] = changed.loc[later, "result"].map({"H": "A", "A": "D", "D": "H"})
    before = league["date"] < cutoff
    # rows up to and including the cutoff date are unaffected (results on the cutoff date only change later rows)
    upto = league["date"] <= cutoff
    pd.testing.assert_frame_equal(elo_features(league)[upto], elo_features(changed)[upto])
    # and removing every later match changes nothing either
    pd.testing.assert_frame_equal(elo_features(league[before]), elo_features(league)[before])


def test_whole_later_seasons_can_be_removed_without_changing_earlier_ratings():
    league = make_league(realistic=True)
    full = elo_features(league)
    for season in sorted(league["season_start"].unique()):
        visible = league[league["season_start"] <= season]
        pd.testing.assert_frame_equal(elo_features(visible), full.loc[visible.index])
    check_elo_is_past_only(league, sorted(league["season_start"].unique()))  # the workflow's own guard


def test_the_leakage_guard_catches_a_leaky_rating(monkeypatch):
    league = make_league(realistic=True)

    def leaky(matches, settings=DEFAULT_SETTINGS):
        out = elo_features(matches, settings)
        return out + (matches["result"] == "H").mean()  # depends on every result, including later ones

    monkeypatch.setattr(experiment_module, "elo_features", leaky)
    with pytest.raises(AssertionError, match="depend on later seasons"):
        experiment_module.check_elo_is_past_only(league, [2000, 2001, 2002])


def test_unplayed_fixtures_get_ratings_but_do_not_update():
    league = make_league(realistic=True)
    last = league["date"].max()
    future = pd.DataFrame([{"season_start": 2003, "date": last + pd.Timedelta(days=7), "home_team": "A",
                            "away_team": "B", "result": pd.NA},
                           {"season_start": 2003, "date": last + pd.Timedelta(days=14), "home_team": "A",
                            "away_team": "C", "result": pd.NA}])
    with_future = pd.concat([league[future.columns], future], ignore_index=True).astype({"result": "string"})
    pre_match = elo_features(with_future)
    latest = ratings_before(league, last + pd.Timedelta(days=1)).set_index("team")["rating"]
    future_rows = pre_match.tail(2)
    assert future_rows["home_elo"].tolist() == pytest.approx([latest["A"], latest["A"]])  # no update in between
    assert future_rows["away_elo"].tolist() == pytest.approx([latest["B"], latest["C"]])
    pd.testing.assert_frame_equal(pre_match.iloc[: len(league)], elo_features(league[future.columns]))


def test_ratings_before_a_cutoff_use_only_earlier_matches():
    league = make_league(realistic=True)
    cutoff = pd.Timestamp("2001-10-01")
    table = ratings_before(league, cutoff)
    played = league[league["date"] < cutoff]
    _, history, _ = run_elo(played)
    assert set(table["team"]) == set("ABCDEF")
    assert table["rating"].is_monotonic_decreasing
    assert (table["as_of"] == cutoff).all() and (table["season_start"] == 2001).all()
    last = history.sort_values("date").groupby("team").tail(1).set_index("team")["rating_after"]
    assert table.set_index("team")["rating"].to_dict() == pytest.approx(last.to_dict())
    with pytest.raises(ValueError, match="No matches"):
        ratings_before(league, "1990-01-01")


def test_end_of_season_ratings_table():
    _, history, _ = run_elo(hand_league())
    table = end_of_season_ratings(history)
    _, final_2000, _, _ = hand_calculation()
    assert table.loc[2000, "A"] == pytest.approx(final_2000["A"])
    assert np.isnan(table.loc[2001, "D"])  # relegated that season


def test_reproducible_and_sensitive_to_settings():
    league = make_league(realistic=False)
    pd.testing.assert_frame_equal(elo_features(league), elo_features(league.copy()))
    assert not elo_features(league).equals(elo_features(league, EloSettings(k_factor=30)))


@pytest.mark.parametrize("change, message", [
    (lambda m: m.assign(result=m["result"].replace("H", "X")), "Results must be"),
    (lambda m: m.assign(away_team=m["home_team"]), "cannot play itself"),
    (lambda m: pd.concat([m, m.head(1)]).reset_index(drop=True), "Duplicate fixtures"),
    (lambda m: m.drop(columns="season_start"), "missing columns"),
    (lambda m: pd.concat([m, m.head(1).assign(home_team="Z")]), "unique index"),
    (lambda m: m.assign(season_start=np.where(m.index % 2 == 0, 2000, 2001)), "overlap"),
])
def test_invalid_match_tables_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        elo_features(change(make_league(realistic=True)))


# ---------------------------------------------------------------- the original features are untouched


def test_adding_elo_does_not_change_the_original_features():
    built = build_features(make_league(realistic=True))
    snapshot = built.copy()
    with_elo = add_elo_features(built)

    pd.testing.assert_frame_equal(built, snapshot)  # input not modified
    pd.testing.assert_frame_equal(with_elo[list(built.columns)], built)  # same 50 features, values and dtypes
    assert list(with_elo.columns) == list(built.columns) + ELO_COLUMNS
    with pytest.raises(ValueError, match="already contain"):
        add_elo_features(with_elo)


def test_production_feature_sets_do_not_include_elo():
    assert len(FEATURE_SETS["all"]) == 50 and len(FEATURE_SETS["compact"]) == 22
    assert not set(ELO_COLUMNS) & set(FEATURE_COLUMNS)
    assert not any(set(ELO_COLUMNS) & set(columns) for columns in FEATURE_SETS.values())
    assert CONTROL_FEATURE_SETS == FEATURE_SETS
    for name, columns in FEATURE_SETS.items():
        assert ELO_FEATURE_SETS[f"{name}_elo"] == columns + ELO_FEATURE_COLUMNS
    check_feature_columns(ELO_FEATURE_SETS["all_elo"])  # Elo ratings are allowed pre-match features
    with pytest.raises(ValueError):
        check_feature_columns(["home_elo", "home_elo_expected"])  # the expected score is descriptive only


@pytest.mark.skipif(not FEATURES_FILE.exists(), reason="data/processed/features.csv not generated")
def test_real_data_elo_is_past_only_and_leaves_the_50_features_alone():
    from src.feature_engineering import load_features

    features = load_features()
    with_elo = add_elo_features(features)
    pd.testing.assert_frame_equal(with_elo[list(features.columns)], features)
    assert with_elo[ELO_COLUMNS].notna().all().all()
    first = features["date"] == features["date"].min()
    assert (with_elo.loc[first, ["home_elo", "away_elo"]] == 1500).all().all()
    check_elo_is_past_only(features, range(2018, 2027))
    # The league average stays at 1500, except that the 2004-05 expansion (18 -> 20 clubs: 5 promoted
    # clubs replaced 3 relegated ones) lowered it by ~7 points, which the summer regression then removes.
    mean = run_elo(features)[2].set_index("season_start")["mean_rating"]
    assert mean.loc[:2003].to_numpy() == pytest.approx(1500)
    assert 1490 < mean.loc[2004] < 1495
    assert mean.loc[2004:].is_monotonic_increasing
    assert mean.loc[2018:].to_numpy() == pytest.approx(1500, abs=0.1)


# ---------------------------------------------------------------- the experiment

FAST_GRIDS = {"logistic_regression": [{"C": 0.01}, {"C": 1.0}],
              "random_forest": [{"max_depth": 3, "min_samples_leaf": 20}],
              "xgboost": [{"max_depth": 2, "n_estimators": 20}]}
EXPERIMENT_SEASONS = (2004, 2005)


@pytest.fixture(scope="module")
def synthetic():
    features = make_features(seasons=range(2000, 2008))
    return features, make_odds(features)


@pytest.fixture(scope="module")
def elo_backtest(synthetic):
    features, matches = synthetic
    return run_elo_backtest(features, matches, seasons=EXPERIMENT_SEASONS, param_grids=FAST_GRIDS)


def test_control_and_elo_arms_use_identical_fixtures(elo_backtest):
    preds = elo_backtest.predictions
    assert list(preds.columns) == PREDICTION_COLUMNS
    keys = ["season", "date", "home_team", "away_team", "actual", "has_odds"]
    for elo, control in model_pairs():
        a = preds[preds["forecast"] == elo][keys].reset_index(drop=True)
        b = preds[preds["forecast"] == control][keys].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    common_fixtures(preds)  # raises if any forecast covers different fixtures
    assert (preds["forecast"] == BOOKMAKER_LABEL).sum() == preds["has_odds"][preds["forecast"] == "Baseline"].sum()


def test_each_arm_selects_only_its_own_feature_sets(elo_backtest):
    folds = elo_backtest.folds
    assert set(folds.loc[folds["arm"] == "control", "feature_set"]) <= set(CONTROL_FEATURE_SETS)
    assert set(folds.loc[folds["arm"] == "elo", "feature_set"]) <= set(ELO_FEATURE_SETS)
    assert len(folds) == 2 * 3 * len(EXPERIMENT_SEASONS)
    # past-only: tuned on S-1 after training on <= S-2, refitted on <= S-1
    assert (folds["last_train_season"] == folds["season"] - 1).all()
    assert (folds["tuning_season"] == folds["season"] - 1).all()


def test_evaluation_season_results_never_influence_model_selection(synthetic):
    """Scrambling season S's results changes no choice and no fitted model for that fold."""
    features, matches = synthetic
    scrambled = features.copy()
    season = scrambled["season_start"] == 2005
    scrambled.loc[season, "result"] = scrambled.loc[season, "result"].map({"H": "A", "A": "D", "D": "H"})
    scrambled_matches = matches.copy()
    scrambled_matches["result"] = scrambled["result"]
    a = run_elo_backtest(features, matches, seasons=[2005], models=["logistic_regression"],
                         param_grids=FAST_GRIDS)
    b = run_elo_backtest(scrambled, scrambled_matches, seasons=[2005], models=["logistic_regression"],
                         param_grids=FAST_GRIDS)
    cols = ["arm", "feature_set", "params", "tuning_log_loss", "train_matches"]
    pd.testing.assert_frame_equal(a.folds[cols], b.folds[cols])
    # the first matchday of S has the same inputs, so the same predictions
    first = a.predictions["date"] == a.predictions.loc[a.predictions["season"] == 2005, "date"].min()
    pd.testing.assert_frame_equal(a.predictions.loc[first, OUTCOMES], b.predictions.loc[first, OUTCOMES])


@pytest.mark.parametrize("seasons", [[2025], [2024, 2026], [2026]])
def test_validation_and_holdout_seasons_are_refused(synthetic, seasons):
    with pytest.raises(ValueError, match="must not use 2025-26"):
        check_experiment_seasons(seasons)
    features, matches = synthetic
    with pytest.raises(ValueError, match="must not use"):
        run_elo_backtest(features, matches, seasons=seasons, param_grids=FAST_GRIDS)


def test_comparison_and_subgroup_tables(elo_backtest, synthetic):
    features, _ = synthetic
    preds = elo_backtest.predictions
    for metric in ("log_loss", "brier", "accuracy", "macro_f1"):
        table = comparison_table(preds, metric=metric, n_resamples=200)
        assert list(table.index) == ["Logistic Regression", "Random Forest", "XGBoost"]
        assert np.allclose(table["difference"], table["elo"] - table["control"])
        assert (table["seasons_better"] + table["seasons_worse"] <= len(EXPERIMENT_SEASONS)).all()
        if metric != "macro_f1":
            # the Bonferroni-adjusted interval is wider than the 95% one
            assert (table["bootstrap_adj_ci_low"] <= table["bootstrap_ci_low"] + 1e-12).all()
            assert (table["bootstrap_adj_ci_high"] >= table["bootstrap_ci_high"] - 1e-12).all()
    groups = subgroup_metrics(preds, features)
    assert set(groups["promoted_club"]) == {True, False}
    flags = promoted_flags(features)
    common = common_fixtures(preds)
    n_promoted = common[common["forecast"] == "Baseline"].merge(flags, on=["date", "home_team", "away_team"])
    assert groups.loc[groups["promoted_club"] & (groups["forecast"] == "Baseline"), "matches"].item() == \
        n_promoted["promoted_club"].sum()


def test_holm_adjustment():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def designed_predictions(elo_better: bool, seasons=range(2010, 2017), per_season=80, seed=0):
    """Seven seasons where one forecast knows the true probabilities and the other only the base rates."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        truth = rng.dirichlet([3, 2, 2], size=per_season)
        actual = [rng.choice(OUTCOMES, p=p) for p in truth]
        flat = np.tile([0.45, 0.27, 0.28], (per_season, 1))
        good, bad = ("Logistic Regression + Elo", "Logistic Regression") if elo_better else \
            ("Logistic Regression", "Logistic Regression + Elo")
        for forecast, probs in ((good, truth), (bad, flat), ("Baseline", flat)):
            for k in range(per_season):
                rows.append({"season": season, "forecast": forecast, "date": pd.Timestamp(f"{season}-09-01")
                             + pd.Timedelta(days=k), "home_team": f"T{k}", "away_team": f"U{k}",
                             "actual": actual[k], "H": probs[k, 0], "D": probs[k, 1], "A": probs[k, 2],
                             "has_odds": True})
    return pd.DataFrame(rows)[PREDICTION_COLUMNS]


def test_decision_rule():
    yes = decision(designed_predictions(elo_better=True), models=["logistic_regression"], n_resamples=500)
    no = decision(designed_predictions(elo_better=False), models=["logistic_regression"], n_resamples=500)
    assert yes.loc["Logistic Regression", "recommend_elo"]
    assert yes.loc["Logistic Regression", "seasons_better"] >= 5
    assert not no.loc["Logistic Regression", "recommend_elo"]
    assert no.loc["Logistic Regression", "log_loss_difference"] > 0


def test_accuracy_bootstrap_is_paired_and_signed():
    preds = designed_predictions(elo_better=True)
    table = accuracy_bootstrap(preds, [("Logistic Regression + Elo", "Logistic Regression")], n_resamples=500)
    row = table.iloc[0]
    assert row["matches"] == 7 * 80
    assert row["ci_low"] <= row["difference"] <= row["ci_high"]
