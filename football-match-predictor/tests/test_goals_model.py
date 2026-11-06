"""Tests for the Poisson goals model (src.goals_model) and its walk-forward evaluation (src.goals_backtest)."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson, skellam
from sklearn.metrics import mean_poisson_deviance

from src.backtest import BACKTEST_SEASONS
from src.feature_engineering import build_features
from src.goals_backtest import (
    BASELINES,
    GOALS_DIR,
    HEADLINE_METRICS,
    LEAGUE,
    MARKET,
    METHODS,
    MODEL,
    PRIMARY_METRICS,
    TEAM,
    bootstrap_means,
    calibration_table,
    check_identical_fixtures,
    event_outcomes,
    load_goals_backtest,
    market_comparison,
    market_coverage,
    market_over_under,
    match_losses,
    paired_comparisons,
    pooled_metrics,
    readiness,
    reliability_table,
    run_goals_backtest,
    run_goals_fold,
    save_goals_backtest,
    score_frequency_table,
    season_metrics,
    season_sensitivity,
    week_clusters,
)
from src.goals_model import (
    BTTS_COLUMNS,
    EVENTS,
    GOALS_FEATURE_SETS,
    MAX_GRID_GOALS,
    OVER_COLUMNS,
    UNDER_COLUMNS,
    LeagueAverageGoals,
    PoissonGoalsModel,
    SideChoice,
    TeamStrengthGoals,
    btts_probabilities,
    exact_score_probability,
    goal_probabilities,
    grid_goals,
    most_likely_scores,
    outcome_probabilities,
    poisson_deviance,
    score_grid,
    strength_inputs,
    total_goals_probabilities,
)
from tests.test_predict import round_robin

PROJECT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- synthetic league
# Eight clubs per season, one matchday per week, four matches on every matchday date.
# 2002-03: H is relegated and I is promoted for the first time.
# 2004-05: H returns after a two-season gap; G goes down.
# Scores are Poisson with rates driven by fixed club strengths, so there is signal to learn.

SEASON_CLUBS = {2000: "ABCDEFGH", 2001: "ABCDEFGH", 2002: "ABCDEFGI", 2003: "ABCDEFGI"}
SEASON_CLUBS |= {season: "ABCDEFHI" for season in range(2004, 2010)}
ATTACK = dict(zip("ABCDEFGHI", [0.35, 0.25, 0.1, 0.0, -0.05, -0.15, -0.2, -0.25, -0.3]))
DEFENCE = dict(zip("ABCDEFGHI", [-0.3, -0.2, -0.05, 0.0, 0.05, 0.1, 0.2, 0.25, 0.3]))
FOLDS = (2006, 2007)
FAST = dict(alpha_grid=(0.01, 1.0), feature_sets={k: GOALS_FEATURE_SETS[k] for k in ("goals", "compact")},
            windows=(10, 38), priors=(3.0, 10.0))


def make_goal_league(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season, clubs in SEASON_CLUBS.items():
        start = pd.Timestamp(f"{season}-08-26")
        for k, matchday in enumerate(round_robin(clubs)):
            date = start + pd.Timedelta(days=7 * k)
            for home, away in matchday:
                hg = int(rng.poisson(np.exp(0.35 + ATTACK[home] + DEFENCE[away])))
                ag = int(rng.poisson(np.exp(0.1 + ATTACK[away] + DEFENCE[home])))
                rows.append({"season": f"{season}-{str(season + 1)[-2:]}", "season_start": season, "date": date,
                             "time": pd.NA, "home_team": home, "away_team": away, "home_goals": hg,
                             "away_goals": ag, "result": "H" if hg > ag else ("A" if ag > hg else "D")})
    return pd.DataFrame(rows).astype({"season": "string", "date": "datetime64[ns]", "time": "string",
                                      "home_team": "string", "away_team": "string", "home_goals": "Int64",
                                      "away_goals": "Int64", "result": "string"})


def scramble(matches, mask, seed=99):
    """New random scores (and results) for the masked matches."""
    rng = np.random.default_rng(seed)
    out = matches.copy()
    n = int(mask.sum())
    out.loc[mask, "home_goals"] = rng.integers(0, 6, n)
    out.loc[mask, "away_goals"] = rng.integers(0, 6, n)
    hg, ag = out["home_goals"].astype(int), out["away_goals"].astype(int)
    out["result"] = pd.Series(np.select([hg > ag, hg < ag], ["H", "A"], "D"), index=out.index).astype("string")
    return out


@pytest.fixture(scope="module")
def league():
    return make_goal_league()


@pytest.fixture(scope="module")
def features(league):
    return build_features(league)


@pytest.fixture(scope="module")
def backtest(features):
    return run_goals_backtest(features, seasons=FOLDS, **FAST)


def rates_of(predictions, method):
    rows = predictions[predictions["method"] == method]
    return rows.set_index(["date", "home_team", "away_team"])[["lambda_home", "lambda_away"]].sort_index()


# ---------------------------------------------------------------- exact probabilities


def test_over_under_probabilities_are_exact_poisson_sums():
    lam_h, lam_a = 1.2, 0.8  # total 2.0
    table = total_goals_probabilities([lam_h], [lam_a]).iloc[0]
    for line in (0.5, 1.5, 2.5, 3.5):
        under = sum(math.exp(-2.0) * 2.0**k / math.factorial(k) for k in range(int(line) + 1))
        key = str(line).replace(".", "_")
        assert table[f"under_{key}"] == pytest.approx(under, abs=1e-14)
        assert table[f"over_{key}"] == pytest.approx(1 - under, abs=1e-14)
    assert table["under_2_5"] == pytest.approx(5 * math.exp(-2.0), abs=1e-14)  # e^-2 (1 + 2 + 2)


def test_over_under_from_total_agrees_with_the_score_grid():
    lam_h, lam_a = np.array([0.3, 1.4, 2.2]), np.array([0.9, 1.1, 2.7])
    grid = score_grid(lam_h, lam_a)
    totals = grid.goals[:, None] + grid.goals[None, :]
    table = total_goals_probabilities(lam_h, lam_a)
    for line in (0.5, 1.5, 2.5, 3.5):
        over = (grid.probabilities * (totals > line)).sum(axis=(1, 2))
        assert np.allclose(table[f"over_{str(line).replace('.', '_')}"], over, atol=1e-11)


def test_btts_formula():
    table = btts_probabilities([1.5, 0.0, 2.0], [1.0, 1.3, 0.0])
    expected = (1 - math.exp(-1.5)) * (1 - math.exp(-1.0))
    assert table["btts_yes"].iloc[0] == pytest.approx(expected, abs=1e-15)
    assert table["btts_yes"].iloc[1] == 0.0 and table["btts_yes"].iloc[2] == 0.0  # a side that cannot score
    grid = score_grid([1.5], [1.0])
    assert grid.probabilities[0, 1:, 1:].sum() == pytest.approx(expected, abs=1e-11)


def test_complementary_probabilities_sum_to_one():
    rng = np.random.default_rng(1)
    lam_h = np.concatenate([[0.0, 0.0, 1e-9, 12.0, 40.0], rng.uniform(0, 5, 300)])
    lam_a = np.concatenate([[0.0, 2.0, 1e-9, 0.0, 35.0], rng.uniform(0, 5, 300)])
    table = goal_probabilities(lam_h, lam_a)
    for over, under in zip(OVER_COLUMNS, UNDER_COLUMNS):
        assert np.allclose(table[over] + table[under], 1.0, rtol=0, atol=1e-15)
    assert np.allclose(table[BTTS_COLUMNS].sum(axis=1), 1.0, rtol=0, atol=1e-15)
    assert np.allclose(table[["H", "D", "A"]].sum(axis=1), 1.0, rtol=0, atol=1e-12)
    columns = OVER_COLUMNS + UNDER_COLUMNS + BTTS_COLUMNS + ["H", "D", "A", "likely_score_probability"]
    assert ((table[columns] >= 0) & (table[columns] <= 1)).all().all()


def test_1x2_from_the_grid_matches_the_skellam_distribution():
    lam_h, lam_a = np.array([1.6, 0.7, 3.1]), np.array([1.1, 1.9, 0.4])
    table = outcome_probabilities(score_grid(lam_h, lam_a))
    assert np.allclose(table["D"], skellam.pmf(0, lam_h, lam_a), atol=1e-11)
    assert np.allclose(table["H"], skellam.sf(0, lam_h, lam_a), atol=1e-11)
    assert np.allclose(table["A"], skellam.cdf(-1, lam_h, lam_a), atol=1e-11)


def test_score_grid_reports_its_truncated_mass():
    lam_h, lam_a = np.array([1.5, 2.5]), np.array([1.2, 0.5])
    small = score_grid(lam_h, lam_a, max_goals=3)
    expected = 1 - poisson.cdf(3, lam_h) * poisson.cdf(3, lam_a)
    assert np.allclose(small.truncated_mass, expected)
    assert np.allclose(small.probabilities.sum(axis=(1, 2)) + small.truncated_mass, 1.0)
    assert (small.truncated_mass > 0.01).all()
    table = outcome_probabilities(small)  # renormalised, but the lost mass is still reported
    assert np.allclose(table[["H", "D", "A"]].sum(axis=1), 1.0)
    assert np.allclose(table["truncated_mass"], expected)
    raw = outcome_probabilities(small, renormalise=False)
    assert np.allclose(raw[["H", "D", "A"]].sum(axis=1) + raw["truncated_mass"], 1.0)
    adaptive = score_grid(lam_h, lam_a)
    assert (adaptive.truncated_mass < 1e-11).all() and adaptive.max_goals >= 10


def test_zero_expected_goals():
    table = goal_probabilities([0.0], [0.0]).iloc[0]
    assert table[UNDER_COLUMNS].tolist() == [1.0] * 4 and table[OVER_COLUMNS].tolist() == [0.0] * 4
    assert table["btts_yes"] == 0.0 and table["btts_no"] == 1.0
    assert (table["H"], table["D"], table["A"]) == (0.0, 1.0, 0.0)
    assert (table["likely_home_goals"], table["likely_away_goals"], table["likely_score_probability"]) == (0, 0, 1.0)
    assert table["truncated_mass"] == 0.0
    one_side = goal_probabilities([0.0], [1.4]).iloc[0]
    assert one_side["H"] == 0.0 and one_side["btts_yes"] == 0.0
    assert one_side["D"] == pytest.approx(math.exp(-1.4))


def test_very_high_expected_goals_extend_the_grid():
    table = goal_probabilities([25.0, 60.0], [20.0, 55.0])
    assert (table["truncated_mass"] < 1e-9).all()
    assert np.allclose(table[["H", "D", "A"]].sum(axis=1), 1.0)
    assert table["over_3_5"].min() > 1 - 1e-12 and (table["btts_yes"] > 1 - 1e-8).all()
    assert grid_goals(25.0) > 50 and grid_goals(0.01) == 10 and grid_goals(0.0) == 10
    assert grid_goals(10_000.0) == MAX_GRID_GOALS
    with pytest.raises(ValueError, match="too large for the score grid"):
        goal_probabilities([400.0], [1.0])  # would lose almost everything: refused, not dropped


def test_invalid_rates_and_lines():
    for bad in ([-0.1], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            goal_probabilities(bad, [1.0])
    with pytest.raises(ValueError, match="different shapes"):
        goal_probabilities([1.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="half-goal"):
        total_goals_probabilities([1.0], [1.0], lines=[2.0])


def test_most_likely_scores():
    scores = most_likely_scores(1.5, 1.1, top=6)
    assert scores["score"].iloc[0] == "1-1"
    assert scores["probability"].is_monotonic_decreasing
    for row in scores.itertuples():
        assert row.probability == pytest.approx(poisson.pmf(row.home_goals, 1.5) * poisson.pmf(row.away_goals, 1.1))
    table = goal_probabilities([1.5], [1.1]).iloc[0]
    assert (table["likely_home_goals"], table["likely_away_goals"]) == (1, 1)
    assert table["likely_score_probability"] == pytest.approx(scores["probability"].iloc[0])
    assert most_likely_scores(0.0, 0.0, top=1)["score"].tolist() == ["0-0"]


def test_historical_score_probabilities():
    lam_h, lam_a = np.array([1.4, 0.9, 2.0]), np.array([1.0, 1.6, 0.3])
    home, away = np.array([2, 0, 5]), np.array([1, 0, 0])
    p = exact_score_probability(lam_h, lam_a, home, away)
    grid = score_grid(lam_h, lam_a)
    assert np.allclose(p, grid.probabilities[np.arange(3), home, away])
    predictions = pd.DataFrame({"season": 2020, "method": MODEL, "date": pd.Timestamp("2020-09-01"),
                                "home_team": ["A", "B", "C"], "away_team": ["D", "E", "F"], "home_goals": home,
                                "away_goals": away, "promoted_club": False, "lambda_home": lam_h,
                                "lambda_away": lam_a})
    from src.goals_backtest import with_probabilities

    losses = match_losses(with_probabilities(predictions))
    assert np.allclose(losses["log_loss_exact_score"], -np.log(p))
    outcomes = event_outcomes(predictions)
    assert outcomes["over_2_5"].tolist() == [1.0, 0.0, 1.0] and outcomes["btts_yes"].tolist() == [1.0, 0.0, 0.0]
    assert outcomes["result"].tolist() == ["H", "D", "H"]


def test_poisson_deviance_matches_sklearn():
    y = np.array([0, 1, 2, 5, 0, 3])
    lam = np.array([0.4, 1.3, 1.9, 2.2, 1.1, 3.0])
    assert poisson_deviance(y, lam).mean() == pytest.approx(mean_poisson_deviance(y, lam))
    assert poisson_deviance([0], [0.7])[0] == pytest.approx(1.4)  # 2 * lambda when no goals
    assert poisson_deviance([2], [2.0])[0] == pytest.approx(0.0)


# ---------------------------------------------------------------- club histories (team baseline inputs)


def test_strength_inputs_count_earlier_matches_of_the_current_spell(league):
    inputs = strength_inputs(league, windows=(10, 38))[38]
    rows = league.join(inputs[["home_matches", "home_scored", "away_matches"]])
    first = rows[rows["date"] == rows["date"].min()]
    assert (first[["home_matches", "away_matches"]] == 0).all().all()  # nothing earlier in the data

    # A's scored total before its 3rd match of 2000-01 = its goals in its first two matches
    a_rows = league[(league["home_team"] == "A") | (league["away_team"] == "A")].sort_values("date")
    goals = np.where(a_rows["home_team"] == "A", a_rows["home_goals"], a_rows["away_goals"]).astype(int)
    third = a_rows.index[2]
    side = "home" if league.loc[third, "home_team"] == "A" else "away"
    assert inputs.loc[third, f"{side}_matches"] == 2
    assert inputs.loc[third, f"{side}_scored"] == goals[:2].sum()

    # Window 10 holds at most 10 matches; window 38 crosses the summer within a spell
    ten = strength_inputs(league, windows=(10,))[10]
    assert ten[["home_matches", "away_matches"]].max().max() == 10
    opening_2001 = league[(league["season_start"] == 2001)].index[0]
    assert inputs.loc[opening_2001, "home_matches"] == 14  # all of 2000-01


def test_promoted_and_returning_clubs_start_without_history(league):
    inputs = strength_inputs(league, windows=(38,))[38]
    for club, season in (("I", 2002), ("H", 2004)):  # new club; club back after a gap
        games = league[((league["home_team"] == club) | (league["away_team"] == club))
                       & (league["season_start"] == season)].sort_values("date")
        first = games.index[0]
        side = "home" if league.loc[first, "home_team"] == club else "away"
        assert inputs.loc[first, f"{side}_matches"] == 0
        second = games.index[1]
        side = "home" if league.loc[second, "home_team"] == club else "away"
        assert inputs.loc[second, f"{side}_matches"] == 1  # only the current spell counts

    # No history = league average attack and defence
    model = TeamStrengthGoals(38, 10.0).fit(league[league["season_start"] < 2004])
    games = league[(league["season_start"] == 2004) & (league["date"] == league.loc[league["season_start"] == 2004, "date"].min())]
    ratings = model.ratings(games, {38: inputs})
    h_rows = games.index[(games["home_team"] == "H") | (games["away_team"] == "H")]
    side = "home" if games.loc[h_rows[0], "home_team"] == "H" else "away"
    assert ratings.loc[h_rows[0], f"{side}_attack"] == pytest.approx(1.0)
    assert ratings.loc[h_rows[0], f"{side}_defence"] == pytest.approx(1.0)


def test_strength_inputs_ignore_same_day_and_later_results(league):
    day = league.loc[league["season_start"] == 2005, "date"].unique()[5]
    original = strength_inputs(league)
    changed = strength_inputs(scramble(league, (league["date"] >= day).to_numpy()))
    earlier = league.index[league["date"] <= day]
    for window in original:
        pd.testing.assert_frame_equal(original[window].loc[earlier], changed[window].loc[earlier])
        later = league.index[league["date"] > day]
        assert not original[window].loc[later].equals(changed[window].loc[later])  # the check has teeth

    cut = league[league["date"] <= day]  # later matches removed entirely
    for window, table in strength_inputs(cut).items():
        pd.testing.assert_frame_equal(table, original[window].loc[cut.index])


def test_team_strength_formula(league):
    train = league[league["season_start"] < 2003]
    model = TeamStrengthGoals(10, 5.0).fit(train)
    mu_h, mu_a = train["home_goals"].astype(float).mean(), train["away_goals"].astype(float).mean()
    assert model.mu_home_ == pytest.approx(mu_h) and model.mu_ == pytest.approx((mu_h + mu_a) / 2)
    inputs = strength_inputs(league, windows=(10,))
    fixture = league[league["season_start"] == 2003].iloc[[20]]
    row = inputs[10].loc[fixture.index[0]]
    mu, k = model.mu_, 5.0
    attack_home = (row["home_scored"] + k * mu) / (row["home_matches"] + k) / mu
    defence_away = (row["away_conceded"] + k * mu) / (row["away_matches"] + k) / mu
    rates = model.predict(fixture, inputs)
    assert rates["lambda_home"].iloc[0] == pytest.approx(mu_h * attack_home * defence_away)
    with pytest.raises(ValueError):
        TeamStrengthGoals(window=7)
    with pytest.raises(ValueError):
        TeamStrengthGoals(prior_matches=0)


# ---------------------------------------------------------------- models


def test_poisson_model_handles_missing_history(features):
    train = features[features["season_start"] <= 2003]
    model = PoissonGoalsModel(SideChoice("goals", 0.01), SideChoice("all", 0.1)).fit(train)
    first_season = features[features["season_start"] == 2000]  # every history feature is missing here
    assert first_season[GOALS_FEATURE_SETS["goals"]].isna().any(axis=1).all()
    rates = model.predict(first_season)
    assert np.isfinite(rates.to_numpy()).all() and (rates.to_numpy() > 0).all()
    assert len(model.coefficients("home")) > len(GOALS_FEATURE_SETS["goals"])  # missing indicators included
    # Canonical log link with an intercept: mean prediction on the training rows = mean goals
    fitted = model.predict(train)
    assert fitted["lambda_home"].mean() == pytest.approx(train["home_goals"].astype(float).mean(), rel=1e-3)


def test_league_average_uses_training_rows_only(features):
    train = features[features["season_start"] <= 2003]
    model = LeagueAverageGoals().fit(train)
    rates = model.predict(features[features["season_start"] == 2005])
    assert (rates["lambda_home"] == train["home_goals"].astype(float).mean()).all()
    assert (rates["lambda_away"] == train["away_goals"].astype(float).mean()).all()


# ---------------------------------------------------------------- walk-forward folds


def test_folds_train_only_on_earlier_seasons(backtest, features):
    folds = backtest.folds.set_index("season")
    for season in FOLDS:
        assert folds.loc[season, "last_train_season"] == season - 1
        assert folds.loc[season, "tuning_season"] == season - 1
        assert folds.loc[season, "train_matches"] == int((features["season_start"] < season).sum())
        assert folds.loc[season, "evaluation_matches"] == int((features["season_start"] == season).sum())
    predictions = backtest.predictions
    assert sorted(predictions["season"].unique()) == list(FOLDS)
    for season, rows in predictions.groupby("season"):
        assert (pd.to_datetime(rows["date"]).dt.year >= season).all()
    assert set(backtest.tuning["season"]) == set(FOLDS)


def test_every_method_predicts_identical_fixtures(backtest):
    predictions = backtest.predictions
    assert list(dict.fromkeys(predictions["method"])) == METHODS
    check_identical_fixtures(predictions, METHODS)
    sizes = predictions.groupby(["season", "method"]).size().unstack()
    assert (sizes.nunique(axis=1) == 1).all()
    broken = predictions.drop(predictions.index[(predictions["method"] == TEAM)][:1])
    with pytest.raises(ValueError, match="different fixtures"):
        check_identical_fixtures(broken, METHODS)


def test_fold_ignores_later_seasons_and_holdout(features, league):
    fold = run_goals_fold(features, 2006, **FAST)
    changed = build_features(scramble(league, (league["season_start"] > 2006).to_numpy()))
    other = run_goals_fold(changed, 2006, **FAST)
    pd.testing.assert_frame_equal(fold.predictions, other.predictions)
    pd.testing.assert_frame_equal(fold.tuning, other.tuning)


def test_same_day_and_future_results_do_not_change_predictions(features, league):
    season = 2007
    day = league.loc[league["season_start"] == season, "date"].unique()[6]
    fold = run_goals_fold(features, season, **FAST)
    changed = build_features(scramble(league, (league["date"] >= day).to_numpy()))
    other = run_goals_fold(changed, season, **FAST)
    pd.testing.assert_frame_equal(fold.tuning, other.tuning)  # choices use S-2 -> S-1 only
    assert fold.model.params == other.model.params and fold.team.params == other.team.params
    for method in METHODS:
        a, b = rates_of(fold.predictions, method), rates_of(other.predictions, method)
        upto = a.index.get_level_values("date") <= day
        pd.testing.assert_frame_equal(a[upto], b[upto])  # matches on the scrambled day itself included
    later = rates_of(fold.predictions, TEAM).index.get_level_values("date") > day
    assert not rates_of(fold.predictions, TEAM)[later].equals(rates_of(other.predictions, TEAM)[later])


def test_holdout_and_validation_seasons_are_refused(features):
    for season in (2025, 2026):
        with pytest.raises(ValueError, match="not allowed"):
            run_goals_fold(features, season)
    with pytest.raises(ValueError, match="not allowed"):
        run_goals_backtest(features, seasons=(2007, 2025))
    with pytest.raises(ValueError, match="two earlier seasons"):
        run_goals_fold(features, 2001)
    assert max(BACKTEST_SEASONS) == 2024 and min(BACKTEST_SEASONS) == 2018


def test_reproducible_and_independent_of_row_order(features, backtest):
    again = run_goals_backtest(features, seasons=FOLDS, **FAST)
    pd.testing.assert_frame_equal(again.predictions, backtest.predictions)
    shuffled = features.sample(frac=1.0, random_state=5)
    other = run_goals_backtest(shuffled, seasons=FOLDS, **FAST)
    keys = ["season", "method", "date", "home_team", "away_team"]
    a = backtest.predictions.sort_values(keys).reset_index(drop=True)
    b = other.predictions.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


# ---------------------------------------------------------------- metrics


def test_pooled_metrics_are_match_weighted_season_means(backtest):
    predictions = backtest.predictions
    seasonal, pooled = season_metrics(predictions), pooled_metrics(predictions)
    losses = match_losses(predictions)
    for method in METHODS:
        rows = seasonal.xs(method, level="method")
        for metric in ["deviance", "log_loss_over_2_5", "brier_btts_yes", "mae_home"]:
            weighted = (rows[metric] * rows["matches"]).sum() / rows["matches"].sum()
            assert pooled.loc[method, metric] == pytest.approx(weighted)
        mine = losses[losses["method"] == method]
        assert pooled.loc[method, "deviance"] == pytest.approx(mine["deviance"].mean())
        assert pooled.loc[method, "matches"] == len(mine)
        p = predictions[predictions["method"] == method]
        assert pooled.loc[method, "mae_home"] == pytest.approx((p["home_goals"] - p["lambda_home"]).abs().mean())
        assert pooled.loc[method, "mean_total_goals"] == pytest.approx((p["home_goals"] + p["away_goals"]).mean())
        assert pooled.loc[method, "deviance"] == pytest.approx(
            mean_poisson_deviance(p["home_goals"], p["lambda_home"]) + mean_poisson_deviance(p["away_goals"], p["lambda_away"]))
    assert pooled.loc[MODEL, "seasons"] == len(FOLDS)


def test_paired_comparisons(backtest):
    predictions = backtest.predictions
    table = paired_comparisons(predictions, n_resamples=2000)
    assert len(table) == len(HEADLINE_METRICS) * len(BASELINES)
    assert set(table.loc[table["primary"], "metric"]) == set(PRIMARY_METRICS)
    pooled = pooled_metrics(predictions)
    for row in table.itertuples():
        assert row.difference == pytest.approx(pooled.loc[MODEL, row.metric] - pooled.loc[row.baseline, row.metric])
        assert row.adj_ci_low <= row.ci_low <= row.difference <= row.ci_high <= row.adj_ci_high
    assert (table["p_holm"] >= table["p_value"]).all()
    clustered = paired_comparisons(predictions, metrics=["deviance"], n_resamples=2000, cluster="week")
    assert clustered["difference"].tolist() == pytest.approx(
        table.loc[table["metric"] == "deviance", "difference"].tolist())
    assert len(np.unique(week_clusters(predictions[predictions["method"] == MODEL]))) == 2 * 14

    sensitivity = season_sensitivity(predictions)
    assert (sensitivity["seasons"] == len(FOLDS)).all()
    assert (sensitivity["leave_one_out_min"] <= sensitivity["leave_one_out_max"]).all()


def test_bootstrap_means():
    diff = np.array([1.0, -1.0, 2.0, 0.0])
    means = bootstrap_means(diff, 500)
    assert np.isclose(means.mean(), diff.mean(), atol=0.1)
    assert np.array_equal(means, bootstrap_means(diff, 500))  # fixed seed
    clustered = bootstrap_means(diff, 500, clusters=[0, 0, 1, 1])  # cluster means 0 and 1
    assert set(np.round(clustered, 6)) <= {0.0, 0.5, 1.0}


def test_reliability_and_calibration():
    p = np.repeat([0.2, 0.8], 500)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(400)], np.r_[np.ones(400), np.zeros(100)]])
    table = reliability_table(p, y, n_bins=2)
    assert table["observed"].tolist() == pytest.approx([0.2, 0.8])
    assert table["matches"].tolist() == [500, 500] and not table["outside"].any()
    off = reliability_table(p, 1 - y, n_bins=2)
    assert off["outside"].all()


def test_calibration_and_bias_tables(backtest):
    predictions = backtest.predictions
    calibration = calibration_table(predictions)
    assert len(calibration) == len(METHODS) * len(EVENTS)
    assert calibration.loc[calibration["method"] == LEAGUE, "calibration_slope"].isna().all()  # constant forecast
    outcomes = event_outcomes(predictions)
    mine = calibration[(calibration["method"] == MODEL) & (calibration["event"] == "btts_yes")].iloc[0]
    mask = predictions["method"] == MODEL
    assert mine["observed"] == pytest.approx(outcomes.loc[mask, "btts_yes"].mean())
    scores = score_frequency_table(predictions)
    n = int(mask.sum())
    assert scores["expected"].sum() == pytest.approx(n) and scores["observed"].sum() == n


def test_readiness_rule(backtest):
    table = readiness(backtest.predictions, n_resamples=1000)
    assert len(table) == 6 and table["passed"].map(type).eq(bool).all()
    assert table["passed"].iloc[-1] == table["passed"].iloc[:-1].all()


# ---------------------------------------------------------------- market odds (comparison only)


def write_raw(directory, name, rows, columns):
    lines = [",".join(columns)] + [",".join(map(str, row)) for row in rows]
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_market_over_under_odds(tmp_path, backtest):
    fixtures = backtest.predictions[backtest.predictions["method"] == MODEL]
    first = fixtures[fixtures["season"] == 2006]
    rows = [(d.strftime("%d/%m/%Y"), h, a, 1.9, 2.0) for d, h, a in first[["date", "home_team", "away_team"]].to_numpy()]
    rows[0] = rows[0][:3] + ("", "")  # one fixture without odds
    write_raw(tmp_path, "I1_0607.csv", rows, ["Date", "HomeTeam", "AwayTeam", "BbAv>2.5", "BbAv<2.5"])
    odds = market_over_under(tmp_path)
    assert len(odds) == len(first) and odds["source"].dropna().unique().tolist() == ["BbAv"]
    expected = (1 / 1.9) / (1 / 1.9 + 1 / 2.0)
    assert odds["market_over_2_5"].dropna().iloc[0] == pytest.approx(expected)
    assert odds["overround"].dropna().iloc[0] == pytest.approx(1 / 1.9 + 1 / 2.0 - 1)
    coverage = market_coverage(backtest.predictions, odds)
    assert coverage.loc[2006, "with_odds"] == len(first) - 1 and coverage.loc[2007, "with_odds"] == 0
    table = market_comparison(backtest.predictions, odds, n_resamples=500)
    assert list(table.index) == [*METHODS, MARKET] and (table["matches"] == len(first) - 1).all()
    assert market_over_under(tmp_path / "missing").empty


# ---------------------------------------------------------------- artifacts and real results


def test_save_and_load_round_trip(tmp_path, backtest):
    save_goals_backtest(backtest, tmp_path)
    loaded = load_goals_backtest(tmp_path)
    pd.testing.assert_frame_equal(loaded.predictions, backtest.predictions, check_dtype=False)
    assert (tmp_path / "settings.json").exists()
    with pytest.raises(FileNotFoundError):
        load_goals_backtest(tmp_path / "nothing")


@pytest.mark.skipif(not (GOALS_DIR / "predictions.csv").exists(), reason="run python -m src.goals_backtest first")
def test_saved_experiment_respects_the_holdout():
    predictions = load_goals_backtest().predictions
    assert set(predictions["season"]) == set(BACKTEST_SEASONS)
    assert (pd.to_datetime(predictions["date"]) < pd.Timestamp("2025-07-01")).all()
    check_identical_fixtures(predictions, METHODS)
    for over, under in zip(OVER_COLUMNS, UNDER_COLUMNS):
        assert np.allclose(predictions[over] + predictions[under], 1.0)
    assert predictions["truncated_mass"].max() < 1e-9
